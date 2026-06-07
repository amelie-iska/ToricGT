import json
import subprocess
import sys

import numpy as np
import torch

from toricgt.config import ModelConfig, TrainConfig
from toricgt.model import ToricTokenGT


def run_tokengt_fineweb_eval(tmp_path, cfg):
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
            str(tmp_path / "missing.model"),
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


def test_tokengt_fineweb_bpb_evaluator_marks_causal_token_route_available(tmp_path):
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
    summary, output_json = run_tokengt_fineweb_eval(tmp_path, cfg)
    assert summary["source"] == "tokengt_graph_node_sp1024_lm"
    assert summary["oai_competition/available"] == 1.0
    assert summary["oai_competition/source_sp1024_graph_nodes"] == 1.0
    assert summary["oai_competition/official_byte_level_bpb_available"] == 1.0
    assert np.isfinite(summary["oai_competition/bpb"])
    assert np.isfinite(summary["03_validation/oai_parameter_golf_restricted_fineweb_causal_tokengt_bpb"])
    assert summary["tokengt/fineweb_graph_node_sp1024_tokens"] == 30.0
    assert (output_json.parent / "REPORT.md").exists()
