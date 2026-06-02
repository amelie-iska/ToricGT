import numpy as np
import torch

from toricgt.random_order_lm import DenseRandomOrderToricLM, RandomOrderLMConfig
from toricgt.trajectory_memory import (
    TrajectoryMemoryIndex,
    TrajectoryMemoryRecord,
    TrajectoryRetrievalHead,
    summarize_trajectory_np,
)


def test_trajectory_memory_index_roundtrip_and_search(tmp_path):
    hidden = np.random.default_rng(17).normal(size=(12, 16)).astype(np.float32)
    summary = summarize_trajectory_np(hidden, positions=np.arange(12), losses=np.linspace(2.0, 1.0, 12))
    index = TrajectoryMemoryIndex()
    index.add(
        TrajectoryMemoryRecord(
            record_id="r0",
            key=summary["key"],
            value={"text": "proof trajectory"},
            dataset="unit",
            task_family="math",
            quality=summary["quality"],
            topology=summary["topology"],
            toric=summary["toric"],
        )
    )
    path = tmp_path / "memory.jsonl"
    index.save_jsonl(path)
    loaded = TrajectoryMemoryIndex.load_jsonl(path)
    results = loaded.search(summary["key"], top_k=1, dataset="unit", task_family="math")
    assert len(results) == 1
    assert results[0][0].record_id == "r0"
    assert results[0][1] > 0.99


def test_trajectory_retrieval_head_metrics_are_finite():
    hidden = torch.randn(4, 10, 32)
    positions = torch.arange(10).repeat(4, 1)
    nll = torch.rand(4, 10) + 1.0
    graphcg_basis = torch.randn(8, 32)
    head = TrajectoryRetrievalHead(32)
    out = head(hidden, positions, nll, graphcg_basis=graphcg_basis)
    assert out["trajectory_memory_loss"].isfinite()
    assert out["trajectory_memory_ce"].isfinite()
    assert out["trajectory_memory_distill_loss"].isfinite()
    assert out["trajectory_memory_quality_loss"].isfinite()
    assert 0.0 <= float(out["trajectory_memory_recall1"]) <= 1.0
    assert out["trajectory_memory_entropy"].isfinite()


def test_random_order_lm_trajectory_memory_loss_path_is_finite():
    cfg = RandomOrderLMConfig(
        vocab_size=48,
        max_seq_len=16,
        d_model=32,
        num_heads=4,
        num_layers=2,
        recurrent_passes=1,
        ffn_multiplier=2,
        dropout=0.0,
        attention="hybrid",
        ring_block_size=8,
        seed=11,
        byte_offset=4,
        bigram_hash_buckets=64,
        aux_mtp_offsets=1,
        use_graphcg=True,
        graphcg_num_directions=8,
        use_trajectory_memory_head=True,
    )
    model = DenseRandomOrderToricLM(cfg)
    tokens = torch.randint(4, cfg.vocab_size, (3, 12))
    out = model(tokens, sample_ids=torch.arange(3))
    assert out["trajectory_memory_loss"].isfinite()
    assert out["trajectory_memory_recall1"].isfinite()
