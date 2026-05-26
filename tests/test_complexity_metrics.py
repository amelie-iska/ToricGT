import json
import importlib.util
from pathlib import Path

import torch

from toricgt.complexity import (
    canonical_graph_bytes,
    compress_len,
    conditional_compress_len,
    graph_mdl_metrics,
    normalized_compression_distance,
    random_order_complexity_metrics,
)
from toricgt.random_order_lm import random_order_batch

_TRAINER_SPEC = importlib.util.spec_from_file_location(
    "train_parameter_golf_random_order",
    Path(__file__).resolve().parents[1] / "scripts" / "train_parameter_golf_random_order.py",
)
assert _TRAINER_SPEC is not None and _TRAINER_SPEC.loader is not None
_TRAINER = importlib.util.module_from_spec(_TRAINER_SPEC)
_TRAINER_SPEC.loader.exec_module(_TRAINER)
checkpoint_publish_decision = _TRAINER.checkpoint_publish_decision
publish_quality_score = _TRAINER.publish_quality_score


def test_compressor_and_conditional_complexity_are_finite():
    context = b"problem: prove a+b=b+a"
    payload = b"reasoning: use commutativity of addition"
    assert compress_len(payload, "zlib") > 0
    assert compress_len(payload, "lzma") > 0
    assert conditional_compress_len(context, payload, "zlib") >= 0
    assert normalized_compression_distance(payload, payload, "zlib") >= 0


def test_canonical_graph_bytes_and_mdl_metrics_are_stable():
    graph_a = {
        "edges": [{"target": "b", "source": "a", "type": "depends"}],
        "nodes": [{"id": "b", "type": "answer"}, {"id": "a", "type": "problem"}],
        "targets": {"answer": "42"},
    }
    graph_b = {
        "targets": {"answer": "42"},
        "nodes": [{"type": "problem", "id": "a"}, {"type": "answer", "id": "b"}],
        "edges": [{"type": "depends", "source": "a", "target": "b"}],
    }
    assert canonical_graph_bytes(graph_a) == canonical_graph_bytes(json.dumps(graph_b))
    metrics = graph_mdl_metrics(graph_a)
    assert metrics["complexity/graph/nodes"] == 2
    assert metrics["complexity/graph/edges"] == 1
    assert metrics["complexity/graph/mdl_total_zlib"] > 0


def test_random_order_complexity_metrics_include_bpb_adjacent_proxies():
    tokens = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19]])
    batch = random_order_batch(tokens, seed=3, bos_token_id=32)
    logits = torch.randn(tokens.shape[0], tokens.shape[1], 32)
    action_ids = torch.randint(0, 4, tokens.shape)
    metrics = random_order_complexity_metrics(
        tokens=tokens,
        previous_tokens=batch.previous_tokens,
        target_tokens=batch.target_tokens,
        permutation=batch.permutation,
        logits=logits,
        action_ids=action_ids,
        byte_offset=4,
        max_samples=2,
    )
    assert metrics["complexity/train/samples"] == 2
    assert metrics["complexity/train/target_cond_k_zlib_mean"] >= 0
    assert metrics["complexity/train/order_program_k_lzma_mean"] > 0
    assert "complexity/train/gflownet_action_trace_k_zlib_mean" in metrics


def test_checkpoint_publish_decision_uses_bpb_and_complexity():
    metrics = {"complexity/val/prediction_target_ncd_lzma_mean": 0.8}
    score, complexity_value = publish_quality_score(
        val_bpb=1.15,
        metrics=metrics,
        complexity_metric="complexity/val/prediction_target_ncd_lzma_mean",
        complexity_weight=0.05,
    )
    assert complexity_value == 0.8
    assert score > 1.15
    current = {"quality_score": score, "val_bpb": 1.15, "complexity_value": complexity_value}
    previous = {"quality_score": 1.25, "val_bpb": 1.20, "complexity_value": 0.79}
    assert checkpoint_publish_decision(current, previous)
    worse = {"quality_score": 1.30, "val_bpb": 1.10, "complexity_value": 5.0}
    assert not checkpoint_publish_decision(worse, current)
