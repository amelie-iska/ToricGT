import json
import math
from dataclasses import replace
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.train import evaluate_loader, synthetic_batch
from toricgt.config import ModelConfig, TrainConfig
from toricgt.model import ToricTokenGT


def run_tokengt_fineweb_eval(tmp_path, cfg, tokenizer_path=None):
    checkpoint = tmp_path / "toricgt_step_00000001.pt"
    model = ToricTokenGT(cfg)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "train_config": TrainConfig(device="cpu", precision="fp32").__dict__,
            "step": 1,
        },
        checkpoint,
    )
    token_path = tmp_path / "fineweb_val_000000.bin"
    write_fake_challenge_bin(token_path, list(range(1, 65)))
    output_json = tmp_path / "analysis" / "oai_competition" / "summary.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_tokengt_fineweb_bpb.py",
            "--checkpoint",
            str(checkpoint),
            "--token-glob",
            str(token_path),
            "--tokenizer-path",
            str(tokenizer_path or tmp_path / "missing.model"),
            "--fineweb-tokens-per-graph",
            "16",
            "--fineweb-stride-tokens",
            "16",
            "--batch-size",
            "2",
            "--batches",
            "1",
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--output-json",
            str(output_json),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return json.loads(output_json.read_text(encoding="utf-8")), output_json


def write_fake_challenge_bin(path, tokens):
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    with path.open("wb") as handle:
        header.tofile(handle)
        np.asarray(tokens, dtype="<u2").tofile(handle)


def test_tokengt_fineweb_bpb_evaluator_scores_lm_head_checkpoint(tmp_path):
    cfg = ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=1,
        max_nodes=16,
        max_edges=64,
        attention="tropical_ring",
        ring_block_size=8,
        use_lm_head=True,
        lm_vocab_size=1024,
    )
    summary, output_json = run_tokengt_fineweb_eval(tmp_path, cfg)
    assert summary["source"] == "tokengt_graph_node_sp1024_lm"
    assert summary["oai_competition/available"] == 0.0
    assert summary["oai_competition/bpb"] is None
    assert summary["oai_competition/source_sp1024_graph_nodes"] == 1.0
    assert summary["oai_competition/official_byte_level_bpb_available"] == 0.0
    assert np.isfinite(summary["tokengt/fineweb_graph_node_sp1024_bpt"])
    assert np.isfinite(summary["tokengt/fineweb_graph_node_sp1024_estimated_bpb"])
    assert summary["tokengt/fineweb_graph_node_sp1024_tokens"] == 30.0
    assert (output_json.parent / "REPORT.md").exists()


@pytest.mark.parametrize("use_tokenizer", [False, True])
def test_tokengt_fineweb_bpb_evaluator_requires_tokenizer_for_exact_bytes(tmp_path, use_tokenizer):
    cfg = ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=1,
        max_nodes=16,
        max_edges=64,
        attention="tropical_ring",
        ring_block_size=8,
        use_lm_head=True,
        lm_vocab_size=1024,
        use_lm_token_embeddings=True,
        use_causal_graph_attention=True,
        tie_lm_head_to_token_embeddings=True,
        use_lm_position_embeddings=True,
        use_lm_toric_position_features=True,
        use_lm_context_hash_embeddings=True,
        lm_context_hash_buckets=64,
        use_lm_caseops_features=True,
        use_lm_smear_gate=True,
    )
    path = None
    if use_tokenizer:
        model_pb = pytest.importorskip("sentencepiece.sentencepiece_model_pb2")

        tokenizer = model_pb.ModelProto()
        tokenizer.trainer_spec.vocab_size = 65
        tokenizer.trainer_spec.bos_id = -1
        tokenizer.trainer_spec.eos_id = -1
        tokenizer.pieces.add(piece="<unk>", type=model_pb.ModelProto.SentencePiece.UNKNOWN)
        for token_id in range(1, 65):
            tokenizer.pieces.add(piece=f"x{token_id}")
        path = tmp_path / "authored.model"
        path.write_bytes(tokenizer.SerializeToString())

    summary, output_json = run_tokengt_fineweb_eval(tmp_path, cfg, path)
    assert summary["source"] == "tokengt_graph_node_sp1024_lm"
    assert summary["oai_competition/available"] == 1.0
    assert summary["oai_competition/source_sp1024_graph_nodes"] == 1.0
    assert summary["oai_competition/official_byte_level_bpb_available"] == float(use_tokenizer)
    assert summary["oai_competition/bpb_available"] == float(use_tokenizer)
    if use_tokenizer:
        nll = summary["tokengt/fineweb_graph_node_sp1024_loss"] * 30
        assert summary["tokengt/fineweb_graph_node_sp1024_estimated_bytes"] == 82
        assert summary["oai_competition/bpb"] == pytest.approx(nll / (math.log(2) * 82))
    else:
        assert summary["oai_competition/bpb"] is None
        assert np.isfinite(summary["tokengt/fineweb_graph_node_sp1024_estimated_bpb"])
    assert summary["tokengt/fineweb_graph_node_sp1024_tokens"] == 30.0
    assert (output_json.parent / "REPORT.md").exists()


def test_validation_omits_bpb_if_any_scored_batch_lacks_valid_byte_lengths():
    cfg = ModelConfig(d_model=16, num_heads=2, num_layers=1, max_nodes=3, max_edges=4)
    batch, target = synthetic_batch(cfg, 1, "cpu")
    batch = replace(
        batch, lm_target_ids=torch.tensor([[0, 1, 2]]), lm_mask=batch.node_mask,
        lm_target_byte_lengths=torch.tensor([[0.0, 2.0, 3.0]]), lm_byte_accounting_exact=True,
    )

    class UniformGraphModel(torch.nn.Module):
        def forward(self, graph):
            return {"node": target, "lm_logits": torch.zeros(1, 3, 4)}

    model = UniformGraphModel()
    covered = evaluate_loader(model, [(batch, target)], "cpu", "fp32", 2)
    assert covered["fineweb_lm_exact_byte_accounting"] == 1.0
    assert covered["fineweb_lm_estimated_bpb"] == pytest.approx(6 / 5)
    incomplete = replace(batch, lm_target_byte_lengths=torch.tensor([[0.0, -1.0, 3.0]]))
    mixed = evaluate_loader(model, [(batch, target), (incomplete, target)], "cpu", "fp32", 2)
    assert mixed["fineweb_lm_tokens"] == 6
    assert mixed["fineweb_lm_bits_per_token"] == pytest.approx(2.0)
    assert "fineweb_lm_estimated_bpb" not in mixed
