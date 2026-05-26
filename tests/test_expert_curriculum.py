import json

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from toricgt.config import ModelConfig
from toricgt.expert_curriculum import CyclicExpertCurriculum
from toricgt.graph_dataset import CuratedGraphIterableDataset, stable_partition_id
from toricgt.soft_moe import GraphTokenSoftMoE


def test_braided_curriculum_covers_every_subset_per_expert():
    curriculum = CyclicExpertCurriculum(num_experts=4, num_subsets=4, phase_steps=10, order="braid")

    assert curriculum.expert_order(0) == [0, 3, 2, 1]
    for expert_idx in range(4):
        assert sorted(curriculum.expert_order(expert_idx)) == [0, 1, 2, 3]

    first = curriculum.assignment(0)
    after_coverage = curriculum.assignment(4 * 4 * 10)
    assert first.active_expert == 0
    assert first.subset_id == 0
    assert not first.full_coverage_complete
    assert after_coverage.full_coverage_complete
    assert after_coverage.teacher_expert is not None


def test_curated_dataset_subset_filtering(tmp_path):
    graph = json.dumps(
        {
            "nodes": [{"id": "problem", "type": "problem", "text": "x"}],
            "edges": [],
        }
    )
    rows = []
    expected = {0: 0, 1: 0, 2: 0}
    for idx in range(12):
        group_hash = f"group-{idx}"
        subset = stable_partition_id([group_hash, f"content-{idx}", f"record-{idx}"], 3, 23)
        expected[subset] += 1
        rows.append(
            {
                "graph_json": graph,
                "group_hash": group_hash,
                "content_hash": f"content-{idx}",
                "record_id": f"record-{idx}",
            }
        )
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    cfg = ModelConfig(d_model=16, num_heads=4, num_layers=1, max_nodes=4, max_edges=4)

    counts = []
    for subset_id in range(3):
        dataset = CuratedGraphIterableDataset(
            [path],
            cfg,
            subset_id=subset_id,
            num_subsets=3,
            subset_salt=23,
        )
        counts.append(sum(1 for _ in dataset))
    assert counts == [expected[0], expected[1], expected[2]]
    assert sum(counts) == len(rows)


def test_active_soft_moe_expert_limits_gradients():
    torch.manual_seed(0)
    layer = GraphTokenSoftMoE(8, 16, num_experts=4, slots_per_expert=1, dropout=0.0)
    layer.set_active_experts([2])
    x = torch.randn(2, 5, 8)
    loss = layer(x).sum()
    loss.backward()

    for idx, expert in enumerate(layer.experts):
        grad_sum = sum(
            0.0 if param.grad is None else float(param.grad.abs().sum())
            for param in expert.parameters()
        )
        if idx == 2:
            assert grad_sum > 0
        else:
            assert grad_sum == 0.0
