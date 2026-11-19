"""Authored-only admission and continuation checks. No optimizer steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from scripts.train import autocast_context, move_batch, save_checkpoint, scheduled_lr, set_seed
from toricgt.config import ModelConfig, TrainConfig
from toricgt.metrics import masked_mse
from toricgt.model import ToricTokenGT
from toricgt.prepared_stream import SCHEMA, PreparedBatchStream, open_prepared_stream
from toricgt.synthetic_resume import (
    capture_synthetic_recovery,
    configure_prepared_cuda,
    prepared_resume_plan,
    restore_synthetic_recovery,
)
from toricgt.toricmora_symbolic_splits import MATH_500, SymbolicRecord, audit_symbolic_splits


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def pin(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec(path):
    return {"file": path.name, "sha256": pin(path)}


def build_fixture(root, *, leaked_split=None, duplicate=False, mismatched_payload=False):
    examples = [
        SymbolicRecord(f"example-{i}", "authored-arithmetic", f"arith-{i}", retrieval_allowed=True)
        for i in range(32)
    ] + [
        SymbolicRecord("validation-a", "authored-arithmetic", declared_holdout="validation"),
        SymbolicRecord("validation-b", "authored-graph-rewrite", declared_holdout="validation"),
        SymbolicRecord("held-out", MATH_500),
    ]
    audit = audit_symbolic_splits(examples)
    training = [row.record_id for row in audit.assignments if row.split == "train"][:5]
    validation = [row.record_id for row in audit.assignments if row.split == "validation"][:2]
    retrieval = [training[0]]
    assert len(training) == 5 and len(validation) == 2
    if leaked_split == "train":
        training[-1] = "held-out"
    if leaked_split == "validation":
        validation[-1] = training[0]
    if leaked_split == "retrieval":
        retrieval.append("held-out")
    if duplicate:
        training[-1] = training[0]

    def make_shard(name, ids):
        records = []
        for i, record_id in enumerate(ids):
            graph = {
                "record_id": record_id if not mismatched_payload or i != 4 else "unadmitted",
                "nodes": [{"id": "q", "text": f"{i}+2", "type": "question"},
                          {"id": "a", "text": str(i + 2), "type": "answer"}],
                "edges": [{"source": "q", "target": "a", "type": "derived"}],
            }
            records.append({"record_id": record_id, "graph_json": json.dumps(graph)})
        path = root / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(records), path, row_group_size=2)
        return spec(path)

    review_path = root / "review.json"
    write_json(review_path, {"records": [asdict(record) for record in examples], "similarity_links": []})
    index_path = root / "retrieval.json"
    write_json(index_path, {"record_ids": retrieval})
    manifest_path = root / "manifest.json"
    write_json(manifest_path, {
        "schema": SCHEMA, "train": make_shard("train", training),
        "validation": make_shard("validation", validation),
        "review": spec(review_path), "retrieval": spec(index_path),
    })
    return manifest_path, pin(manifest_path)


def config():
    return ModelConfig(
        d_model=16, num_heads=2, num_layers=1, max_nodes=6, max_edges=8, attention="hybrid",
        ring_block_size=8, gflownet_hidden_dim=16,
    )


def recipe(**change):
    settings = {
        "resume": None, "steps": 1, "schedule_total_steps": 5,
        "grad_accum_steps": 2, "prepared_stream_manifest": "old-directory/manifest.json",
        "prepared_stream_sha256": None, "check_prepared_stream": False, "seed": 29,
    }
    return argparse.Namespace(**(settings | change))


def test_prepared_admission_cli_exits_before_model_or_checkpoint(tmp_path):
    manifest, digest = build_fixture(tmp_path)
    argv = [
        sys.executable, "-B", "scripts/train.py", "--check-prepared-stream", "--device", "cpu",
        "--resume", str(tmp_path / "unopened.pt"), "--checkpoint-dir", str(tmp_path / "no-output"),
        "--prepared-stream-manifest", str(manifest), "--prepared-stream-sha256", digest,
    ]
    result = subprocess.run(argv, check=True, text=True, capture_output=True)
    assert json.loads(result.stdout) == {
        "manifest_sha256": digest, "train_records": 5, "validation_records": 2, "retrieval_records": 1,
    }
    assert not (tmp_path / "no-output").exists()


@pytest.mark.parametrize("split", ["train", "validation", "retrieval"])
def test_prepared_admission_rejects_leaked_membership(tmp_path, split):
    manifest, digest = build_fixture(tmp_path, leaked_split=split)
    with pytest.raises(ValueError, match="violates reviewed split assignments"):
        with open_prepared_stream(manifest, digest):
            pytest.fail("leaked data admitted")


def test_prepared_admission_rejects_duplicate_records_and_changed_bytes(tmp_path):
    manifest, digest = build_fixture(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="unique nonempty record IDs"):
        with open_prepared_stream(manifest, digest):
            pytest.fail("duplicate data admitted")
    manifest, digest = build_fixture(tmp_path)
    with (tmp_path / "train.parquet").open("ab") as dst:
        dst.write(b"changed")
    with pytest.raises(ValueError, match="Parquet SHA-256"):
        with open_prepared_stream(manifest, digest):
            pytest.fail("tampered data admitted")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        with open_prepared_stream(manifest, "0" * 64):
            pytest.fail("untrusted manifest admitted")


def test_prepared_reader_seeks_short_batch_and_wraps_without_payload_replay(tmp_path):
    manifest, digest = build_fixture(tmp_path)
    with open_prepared_stream(manifest, digest) as (shards, _):
        uninterrupted = PreparedBatchStream(shards["train"], config(), 2)
        sequence = [uninterrupted.next_batch()[1] for _ in range(6)]
        assert [len(target) for target in sequence] == [2, 2, 1, 2, 2, 1]
        assert torch.equal(sequence[0], sequence[3])
        assert not torch.equal(sequence[0], sequence[1])
    with open_prepared_stream(manifest, digest) as (shards, _):
        shard = shards["train"]
        resumed = PreparedBatchStream(shard, config(), 2)
        resumed.restore(2, step=1, grad_accum_steps=2)
        parquet = shard._file
        reads = []

        class TracedParquet:
            def read_row_group(self, index, **kwargs):
                reads.append(index)
                return parquet.read_row_group(index, **kwargs)

        shard._file = TracedParquet()
        assert torch.equal(resumed.next_batch()[1], sequence[2])
        assert reads == [2]
        assert torch.equal(resumed.next_batch()[1], sequence[3])
        assert resumed.batches_consumed == 4
        with pytest.raises(ValueError, match="cursor"):
            resumed.restore(3, step=2, grad_accum_steps=2)
        assert resumed.batches_consumed == 4


def test_failed_graph_fetch_leaves_cursor_unchanged(tmp_path):
    manifest, digest = build_fixture(tmp_path, mismatched_payload=True)
    with open_prepared_stream(manifest, digest) as (shards, _):
        reader = PreparedBatchStream(shards["train"], config(), 2)
        reader.restore(2, step=1, grad_accum_steps=2)
        with pytest.raises(ValueError, match="graph ID differs"):
            reader.next_batch()
        assert reader.batches_consumed == 2


def test_resume_plan_pins_runtime_and_stream_content_but_allows_relocation():
    args = recipe()
    runtime = {"torch_cuda": "authored"}
    h, fingerprint = prepared_resume_plan(args, None, "a" * 64, runtime)
    ckpt = {"step": 1, "optimizer": {}, "prepared_recovery": {
        "version": 1, "schedule_total_steps": h, "plan_sha256": fingerprint,
    }}
    resumed = recipe(resume="authored.pt", steps=4, schedule_total_steps=None, prepared_stream_manifest="elsewhere")
    assert prepared_resume_plan(resumed, ckpt, "a" * 64, runtime) == (h, fingerprint)
    for digest, rt in (("b" * 64, runtime), ("a" * 64, {})):
        with pytest.raises(ValueError, match="configuration differs"):
            prepared_resume_plan(resumed, ckpt, digest, rt)
    with pytest.raises(ValueError, match="fixed prepared schedule"):
        prepared_resume_plan(recipe(steps=5), ckpt, "a" * 64, runtime)


def cuda_continuation(root, resumed):
    """Worker entry used by two independent subprocesses."""
    device = "cuda:0"
    runtime = configure_prepared_cuda(device)
    manifest, digest = root / "manifest.json", pin(root / "manifest.json")
    set_seed(71 if resumed else 29)
    cfg = config()
    train_cfg = TrainConfig(device=device, precision="fp16", use_wandb=False, batch_size=2, grad_accum_steps=2)
    model = ToricTokenGT(cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0 if resumed else 16.0)
    checkpoint_path = root / "authored.pt"
    with open_prepared_stream(manifest, digest) as (shards, _):
        reader = PreparedBatchStream(shards["train"], cfg, 2)
        if resumed:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            h, _ = prepared_resume_plan(
                recipe(resume="authored.pt", steps=4, schedule_total_steps=None), checkpoint, digest, runtime,
            )
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            state = checkpoint["prepared_recovery"]
            reader.restore(state["batches_consumed"], checkpoint["step"], train_cfg.grad_accum_steps)
            restore_synthetic_recovery(state, scaler, device)
        else:
            for _ in range(2):
                reader.next_batch()
            parameter = next(model.parameters())
            optimizer.state[parameter] = {
                "step": torch.tensor(1.0),
                "exp_avg": torch.full_like(parameter, 0.03125),
                "exp_avg_sq": torch.full_like(parameter, 0.0625),
            }
            scaler.scale(torch.ones((), device=device))
            h, fingerprint = prepared_resume_plan(recipe(), None, digest, runtime)
            state = capture_synthetic_recovery(scaler, device, h, fingerprint)
            state["batches_consumed"] = reader.batches_consumed
            save_checkpoint(root, checkpoint_path.name, model, optimizer, cfg, train_cfg, 1, prepared_recovery=state)

        before = [p.detach().clone() for p in model.parameters()]
        observations = {
            "python_rng": random.random(), "numpy_rng": np.random.random(),
            "torch_cpu_rng": torch.rand(8), "torch_cuda_rng": torch.rand(8, device=device).cpu(),
            "lr": scheduled_lr(0.001, 1, h, 2, 0.1, "cosine"),
            "scaler": scaler.state_dict(), "microbatches": [],
            "weights": [value.cpu() for value in before],
        }
        candidates = []
        for _ in range(2):
            batch, target = move_batch(*reader.next_batch(), device=device)
            with autocast_context(device, train_cfg.precision):
                logits = model(batch)["node"]
                loss = masked_mse(logits.float(), target, batch.node_mask) / train_cfg.grad_accum_steps
            observations["microbatches"].append({
                "target": target.cpu(), "node_features": batch.node_features.cpu(),
                "edge_features": batch.edge_features.cpu(), "edge_index": batch.edge_index.cpu(),
                "logits": logits.detach().cpu(),
            })
            candidates.append(loss)
        # Probe the gradients of the next two microbatches without an optimizer step.
        gradient = torch.autograd.grad(scaler.scale(sum(candidates)), tuple(model.parameters()), allow_unused=True)
        observations["gradients"] = [g.detach().cpu() if g is not None else None for g in gradient]
        observations["cursor"] = reader.batches_consumed
        optim_state = optimizer.state[next(model.parameters())]
        observations["optimizer"] = {name: tensor.cpu() for name, tensor in optim_state.items()}
        assert any(g is not None and bool(torch.count_nonzero(g)) for g in gradient)
        assert all(g is None or bool(torch.isfinite(g).all()) for g in gradient)
        assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    torch.save(observations, root / ("resumed-results.pt" if resumed else "continuous-results.pt"))


@pytest.mark.skipif(os.environ.get("TORICMORA_CUDA_PARITY") != "1", reason="opt-in authored CUDA parity")
def test_cuda_prepared_interruption_parity_without_optimizer_updates(tmp_path):
    build_fixture(tmp_path)
    for resumed in (False, True):
        subprocess.run(
            [
                sys.executable, "-B", "-c",
                "import runpy,sys; from pathlib import Path;"
                "runpy.run_path(sys.argv[1])['cuda_continuation'](Path(sys.argv[2]),sys.argv[3]=='True')",
                str(Path(__file__).resolve()), str(tmp_path), str(resumed),
            ],
            check=True,
        )
    uninterrupted = torch.load(tmp_path / "continuous-results.pt", weights_only=True)
    restored = torch.load(tmp_path / "resumed-results.pt", weights_only=True)
    torch.testing.assert_close(uninterrupted, restored, atol=0, rtol=0, equal_nan=False)
    assert restored["cursor"] == 4
    assert [len(batch["target"]) for batch in restored["microbatches"]] == [1, 2]


def test_two_epoch_plan_covers_partial_batches_and_preserves_output(tmp_path):
    manifest, digest = build_fixture(tmp_path)
    output = tmp_path / "future-run"
    command = [
        sys.executable, "-B", "scripts/plan_prepared_training.py",
        "--manifest", str(manifest), "--sha256", digest, "--output", str(output),
        "--batch-size", "2", "--max-accum", "2",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    plan = json.loads(result.stdout)
    assert not output.exists()
    assert plan["batches_per_epoch"] == 3
    assert plan["grad_accum_steps"] == 1
    assert plan["schedule_total_steps"] == 6
    first, second = plan["segments"]
    assert first["expected_batches_consumed"] == 3 and second["expected_batches_consumed"] == 6
    assert first["checkpoint"] != second["checkpoint"]
    assert second["argv"][-2:] == ["--resume", first["checkpoint"]]
    assert first["argv"][first["argv"].index("--device") + 1] == "cuda:0"
    assert first["argv"][first["argv"].index("--eval-batches") + 1] == "1"
    output.mkdir()
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    assert result.returncode == 2 and "choose a new experiment path" in result.stderr
    assert not list(output.iterdir())


def test_reviewed_shell_check_and_plan_leave_training_unstarted(tmp_path):
    import yaml

    environment = os.environ | {"TORICMORA_PYTHON": sys.executable}
    default = subprocess.run(
        ["scripts/run_toricmora_reviewed.sh"], env=environment, check=True, capture_output=True, text=True,
    )
    assert json.loads(default.stdout)["configuration_checked"]
    manifest, digest = build_fixture(tmp_path)
    config_path = tmp_path / "recipe.yaml"
    payload = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    payload["metadata"]["prepared_launch"].update(
        manifest=str(manifest), manifest_sha256=digest, output=str(tmp_path / "future"),
    )
    payload["training"].update(batch_size=2, grad_accum_steps=2, precision="bf16")
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    command = ["scripts/run_toricmora_reviewed.sh", "plan", "--config", str(config_path)]
    planned = subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
    receipt = json.loads(planned.stdout)
    assert receipt["batches_per_epoch"] == 3
    assert receipt["grad_accum_steps"] == 1
    for segment in receipt["segments"]:
        argv = segment["argv"]
        assert argv[argv.index("--precision") + 1] == "bf16"
        assert argv[argv.index("--config") + 1] == str(config_path)
    assert not (tmp_path / "future").exists()
    unapproved = command.copy()
    unapproved[1] = "train"
    stopped = subprocess.run(unapproved, env=environment, check=False, capture_output=True, text=True)
    assert stopped.returncode == 2 and "explicit approval" in stopped.stderr
    assert not (tmp_path / "future").exists()

    payload["model"]["d_model"] = 191
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    rejected = subprocess.run(command, env=environment, check=False, capture_output=True, text=True)
    assert rejected.returncode == 2 and "divisible by num_heads" in rejected.stderr


def test_prepared_reader_never_truncates_to_a_smaller_runtime_capacity(tmp_path):
    from dataclasses import replace

    manifest, digest = build_fixture(tmp_path)
    with open_prepared_stream(manifest, digest) as (shards, _):
        stream = PreparedBatchStream(shards["train"], replace(config(), max_nodes=1), 2)
        with pytest.raises(ValueError, match="node capacity"):
            stream.next_batch()
        assert stream.batches_consumed == 0
