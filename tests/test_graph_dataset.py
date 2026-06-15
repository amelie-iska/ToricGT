import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from toricgt.config import ModelConfig
from toricgt.got_trajectory import graph_json_has_branch_merge, got_dag_summary_np
from toricgt.graph_dataset import (
    CuratedGraphIterableDataset,
    collate_graph_items,
    fineweb_tokens_to_graph_json,
    graph_json_to_item,
    interleave_dataset_paths,
    load_competition_token_memmap,
    record_to_graph_json,
    text_to_graph_json,
)
from toricgt.metrics import masked_mse
from toricgt.model import ToricTokenGT


def test_graph_json_item_with_padding_trains_finite():
    payload = {
        "task_family": "unit",
        "nodes": [
            {"id": "problem", "type": "problem", "text": "show a short proof"},
            {"id": "step_000", "type": "reasoning_step", "text": "use the definition"},
            {"id": "answer", "type": "answer", "text": "done"},
        ],
        "edges": [
            {"source": "problem", "target": "step_000", "type": "depends_on"},
            {"source": "step_000", "target": "answer", "type": "supports_answer"},
        ],
    }
    cfg = ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_nodes=16,
        max_edges=32,
        attention="tropical_ring",
        ring_block_size=8,
        soft_moe_start_layer=0,
    )
    item = graph_json_to_item(json.dumps(payload), cfg)
    batch, target = collate_graph_items([item, item])
    model = ToricTokenGT(cfg)
    out = model(batch)
    loss = masked_mse(out["node"], target, batch.node_mask)
    loss.backward()
    assert torch.isfinite(loss)


def test_text_fallback_builds_branching_merging_graph():
    text = "Start with the hypothesis. Try a direct proof. Also try contradiction. Merge the cases. Finish the answer."
    graph_json = text_to_graph_json(text, dataset="unit", task_family="got", record_id="r0", max_spans=8)
    payload = json.loads(graph_json)

    assert payload["trajectory_kind"] == "branch_merge_dag"
    assert graph_json_has_branch_merge(payload)
    assert "context_order" not in {edge["type"] for edge in payload["edges"]}
    node_ids = {node["id"]: idx for idx, node in enumerate(payload["nodes"])}
    edges = np.asarray([[node_ids[edge["source"]], node_ids[edge["target"]]] for edge in payload["edges"]], dtype=np.int64)
    summary = got_dag_summary_np(len(payload["nodes"]), edges)
    assert summary["branch_merge_edge_fraction"] > summary["linear_chain_fraction"]


def test_record_to_graph_json_upgrades_linear_graph():
    payload = {
        "nodes": [
            {"id": "n0", "text": "a"},
            {"id": "n1", "text": "b"},
            {"id": "n2", "text": "c"},
            {"id": "n3", "text": "d"},
        ],
        "edges": [
            {"source": "n0", "target": "n1", "type": "next"},
            {"source": "n1", "target": "n2", "type": "next"},
            {"source": "n2", "target": "n3", "type": "next"},
        ],
    }
    cfg = ModelConfig(max_nodes=16, max_edges=32)
    upgraded = json.loads(record_to_graph_json({"graph_json": json.dumps(payload)}, cfg))

    assert graph_json_has_branch_merge(upgraded)
    assert "next" not in {edge["type"] for edge in upgraded["edges"]}
    assert upgraded["suppressed_linear_chain_edges"] == 3


def write_fake_challenge_bin(path, tokens):
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    with path.open("wb") as handle:
        header.tofile(handle)
        np.asarray(tokens, dtype="<u2").tofile(handle)


def test_fineweb_bin_tokens_stream_as_sequential_graph(tmp_path):
    token_path = tmp_path / "fineweb_train_000000.bin"
    write_fake_challenge_bin(token_path, list(range(1, 33)))
    tokens = load_competition_token_memmap(token_path)
    assert int(tokens.shape[0]) == 32
    graph_payload = json.loads(fineweb_tokens_to_graph_json(tokens[:16], max_nodes=16))

    assert graph_payload["trajectory_kind"] == "sequential_token_chain"
    assert len(graph_payload["nodes"]) == 15
    assert len(graph_payload["edges"]) == 14
    assert {edge["type"] for edge in graph_payload["edges"]} == {"next_token"}

    cfg = ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_nodes=16,
        max_edges=64,
        attention="tropical_ring",
        ring_block_size=8,
    )
    dataset = CuratedGraphIterableDataset(
        [token_path],
        cfg,
        fineweb_tokenizer_path=tmp_path / "missing.model",
        fineweb_tokens_per_graph=16,
        fineweb_stride_tokens=8,
    )
    item = next(iter(dataset))
    batch, target = collate_graph_items([item, item])
    model = ToricTokenGT(cfg)
    out = model(batch)
    loss = masked_mse(out["node"], target, batch.node_mask)

    assert batch.node_mask.any()
    assert batch.edge_mask.any()
    assert batch.lm_input_ids is not None
    assert batch.lm_target_ids is not None
    assert batch.lm_mask is not None
    assert int(batch.edge_mask.sum().item()) == 28
    assert int(batch.lm_mask.sum().item()) == 30
    assert int(batch.node_mask.sum().item()) >= 30
    assert int(batch.lm_input_ids[0, 0].item()) == 1
    assert int(batch.lm_target_ids[0, 0].item()) == 2
    assert torch.isfinite(loss)


def test_fineweb_lm_targets_drive_tokengt_lm_head(tmp_path):
    token_path = tmp_path / "fineweb_train_000000.bin"
    write_fake_challenge_bin(token_path, list(range(1, 33)))
    cfg = ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_nodes=16,
        max_edges=64,
        attention="tropical_ring",
        ring_block_size=8,
        use_lm_head=True,
        lm_vocab_size=1024,
    )
    dataset = CuratedGraphIterableDataset(
        [token_path],
        cfg,
        fineweb_tokenizer_path=tmp_path / "missing.model",
        fineweb_tokens_per_graph=16,
        fineweb_stride_tokens=8,
    )
    item = next(iter(dataset))
    batch, _ = collate_graph_items([item, item])
    model = ToricTokenGT(cfg)
    out = model(batch)

    assert out["lm_logits"].shape == (2, cfg.max_nodes, cfg.lm_vocab_size)
    assert batch.lm_mask.any()


def test_interleaved_paths_emit_jsonl_and_fineweb_bin(tmp_path):
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(json.dumps({"text": "A. B. C. D.", "record_id": "jsonl"}) + "\n", encoding="utf-8")
    token_path = tmp_path / "fineweb_train_000001.bin"
    write_fake_challenge_bin(token_path, list(range(1, 65)))
    cfg = ModelConfig(max_nodes=16, max_edges=64)
    dataset = CuratedGraphIterableDataset(
        [jsonl_path, token_path],
        cfg,
        interleave_paths=True,
        fineweb_tokenizer_path=tmp_path / "missing.model",
        fineweb_tokens_per_graph=16,
        fineweb_stride_tokens=16,
    )
    iterator = iter(dataset)
    first = next(iterator)
    second = next(iterator)

    assert first.graph.node_mask.any()
    assert second.graph.node_mask.any()
    assert first.graph.edge_mask.any()
    assert second.graph.edge_mask.any()
    metadata = [first.metadata, second.metadata]
    assert any(item["dataset"] == "fineweb10B_sp1024" for item in metadata)
    assert any(item["task_family"] == "fineweb_language_modeling_graph" for item in metadata)


def test_path_type_interleaving_places_fineweb_bins_early():
    paths = [
        *[Path(f"train-{idx:05d}.parquet") for idx in range(4)],
        *[Path(f"fineweb_train_{idx:06d}.bin") for idx in range(2)],
    ]
    ordered = interleave_dataset_paths(paths)

    assert ordered[0].suffix == ".parquet"
    assert ordered[1].suffix == ".bin"
    assert ordered[2].suffix == ".parquet"
    assert ordered[3].suffix == ".bin"


def test_validate_tokengt_graph_data_report(tmp_path):
    jsonl_path = tmp_path / "records.jsonl"
    graph_payload = {
        "dataset": "unit_graphs",
        "task_family": "proof_graph",
        "record_id": "dag_0",
        "nodes": [
            {"id": "root", "type": "problem", "text": "prove it"},
            {"id": "left", "type": "reasoning_step", "text": "case left"},
            {"id": "right", "type": "reasoning_step", "text": "case right"},
            {"id": "join", "type": "reasoning_step", "text": "join"},
            {"id": "answer", "type": "answer", "text": "done"},
        ],
        "edges": [
            {"source": "root", "target": "left", "type": "branch_left"},
            {"source": "root", "target": "right", "type": "branch_right"},
            {"source": "left", "target": "join", "type": "merge_candidate"},
            {"source": "right", "target": "join", "type": "merge_candidate"},
            {"source": "join", "target": "answer", "type": "supports_answer"},
        ],
    }
    rows = [
        {"record_id": "dag_0", "graph_json": json.dumps(graph_payload)},
        {"record_id": "text_0", "dataset": "unit_text", "task_family": "got", "text": "Start. Branch left. Branch right. Merge. Finish."},
    ]
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output_dir = tmp_path / "graph_validation"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src{os.pathsep}."
    subprocess.run(
        [
            sys.executable,
            "scripts/validate_tokengt_graph_data.py",
            "--input",
            str(jsonl_path),
            "--output-dir",
            str(output_dir),
            "--max-records",
            "8",
            "--max-nodes",
            "16",
            "--max-edges",
            "32",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/validate_analysis_exactness.py",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    records = json.loads((output_dir / "records.json").read_text(encoding="utf-8"))
    html = (output_dir / "index.html").read_text(encoding="utf-8")

    assert summary["schema"] == "toricgt.tokengt_graph_data_validation.v1"
    assert summary["records"] == 2
    assert summary["all_masks_and_endpoints_valid"] is True
    assert summary["wandb_metrics"]["tokengt_graph_data/node_mask_valid_fraction"] == 1.0
    assert summary["wandb_metrics"]["tokengt_graph_data/edge_endpoint_valid_fraction"] == 1.0
    assert summary["wandb_metrics"]["tokengt_graph_data/graph_token_utilization"] > 0.0
    assert any(record["causal_rank_kind"] == "topological_dag" for record in records)
    assert "TokenGT Graph Data Validation" in html
