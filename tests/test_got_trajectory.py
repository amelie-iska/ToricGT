import torch

from toricgt.got_trajectory import default_branch_merge_edges, got_dag_metrics, got_dag_summary_np


def test_default_graph_of_thought_template_branches_and_merges():
    edges = default_branch_merge_edges(8)
    summary = got_dag_summary_np(8, edges.numpy())

    assert edges.shape[1] == 2
    assert summary["branch_count"] >= 2.0
    assert summary["merge_count"] >= 2.0
    assert summary["back_edge_fraction"] == 0.0


def test_got_dag_metrics_are_finite_for_branching_hidden_states():
    hidden = torch.randn(3, 8, 16)
    node_mask = torch.ones(3, 8, dtype=torch.bool)
    edge_index = default_branch_merge_edges(8).unsqueeze(0).repeat(3, 1, 1)
    edge_mask = torch.ones(edge_index.shape[:2], dtype=torch.bool)

    out = got_dag_metrics(hidden, node_mask=node_mask, edge_index=edge_index, edge_mask=edge_mask)

    assert out["got_dag_loss"].isfinite()
    assert out["got_dag_branch_count"] > 0
    assert out["got_dag_merge_count"] > 0
    assert out["got_dag_branch_count_batch"].shape == (3,)
