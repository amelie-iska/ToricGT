"""Checkpoint-backed search without learned or existing model weights."""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

from toricgt.config import ModelConfig
from toricgt.model import ToricTokenGT
from toricgt.toricmora_symbolic_search import METHODS, PROBLEMS


def test_authored_checkpoint_search_uses_one_scoring_call_per_expansion(tmp_path):
    model_pb = pytest.importorskip("sentencepiece.sentencepiece_model_pb2")
    tokenizer = model_pb.ModelProto()
    tokenizer.normalizer_spec.name = "identity"
    tokenizer.normalizer_spec.add_dummy_prefix = True
    tokenizer.normalizer_spec.remove_extra_whitespaces = True
    tokenizer.normalizer_spec.escape_whitespaces = True
    tokenizer.trainer_spec.vocab_size = 1 + 94 + 4
    tokenizer.trainer_spec.bos_id = -1
    tokenizer.trainer_spec.eos_id = -1
    tokenizer.pieces.add(piece="<unk>", type=model_pb.ModelProto.SentencePiece.UNKNOWN)
    for piece in map(chr, range(33, 127)):
        tokenizer.pieces.add(piece=piece, score=-2)
    for piece in ("▁", "▁+", "▁-", "▁*"):
        tokenizer.pieces.add(piece=piece)
    tokenizer_path = tmp_path / "authored.model"
    tokenizer_path.write_bytes(tokenizer.SerializeToString())
    tokenizer_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()

    torch.manual_seed(29)
    cfg = ModelConfig(
        d_model=8, num_heads=2, num_layers=1, max_nodes=192, max_edges=192, attention="softmax",
        use_soft_moe=False, use_gflownet_head=False, use_lm_head=True, use_lm_token_embeddings=True,
        use_causal_graph_attention=True, lm_vocab_size=tokenizer.trainer_spec.vocab_size,
    )
    weights = tmp_path / "authored.pt"
    torch.save({"config": asdict(cfg), "model": ToricTokenGT(cfg).state_dict()}, weights)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    command = [
        sys.executable, "-B", "-m", "toricgt.toricmora_symbolic_search", "--expansions", "1",
        "--trusted-checkpoint", str(weights), "--checkpoint-sha256", digest,
        "--tokenizer", str(tokenizer_path), "--tokenizer-sha256", tokenizer_sha,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["proposer"] == "native-checkpoint-v1"
    assert report["checkpoint_provenance"]["checkpoint_sha256"] == digest
    assert report["checkpoint_provenance"]["model_input_token_cap_per_case"] == 192
    assert set(report["summary"]) == set(METHODS)
    assert len(report["runs"]) == len(METHODS) * len(PROBLEMS)
    assert all(run["model_calls"] == run["used"]["expansions"] == 1 for run in report["runs"])
    assert all(0 < run["model_input_tokens"] <= 192 for run in report["runs"])

    command[command.index("--checkpoint-sha256") + 1] = "0" * 64
    rejected = subprocess.run(command, check=False, capture_output=True, text=True)
    assert rejected.returncode == 2
    assert "checkpoint SHA-256 mismatch" in rejected.stderr
