import json

from toricgt.config import ModelConfig
from toricgt.graph_tokenizer import GraphBatch
from toricgt.model import ToricTokenGT


def synthetic_batch(cfg: ModelConfig, batch_size: int, device: str):
    import torch

    n = min(16, cfg.max_nodes)
    e = min(48, cfg.max_edges)
    node = torch.randn(batch_size, n, cfg.node_feature_dim, device=device)
    edge = torch.randn(batch_size, e, cfg.edge_feature_dim, device=device)
    edge_index = torch.randint(0, n, (batch_size, e, 2), device=device)
    node_mask = torch.ones(batch_size, n, dtype=torch.bool, device=device)
    edge_mask = torch.ones(batch_size, e, dtype=torch.bool, device=device)
    target = torch.zeros(batch_size, n, cfg.output_dim, device=device)
    return GraphBatch(node, edge, edge_index, node_mask, edge_mask), target


def test_model_shapes_cpu():
    cfg = ModelConfig(d_model=64, num_heads=4, num_layers=2, max_nodes=32, max_edges=64)
    model = ToricTokenGT(cfg)
    batch, _ = synthetic_batch(cfg, batch_size=2, device="cpu")
    out = model(batch)
    assert out["node"].shape == (2, 16, cfg.output_dim)
    assert out["edge"].shape == (2, 48, cfg.output_dim)
    assert out["graph"].shape == (2, cfg.output_dim)
    assert out["node_embeddings"].shape == (2, 16, cfg.d_model)
    assert out["edge_embeddings"].shape == (2, 48, cfg.d_model)
    assert out["gflownet_forward_logits"].shape == (2, cfg.gflownet_num_actions)
    assert out["gflownet_backward_logits"].shape == (2, cfg.gflownet_num_actions)


def test_tokengt_optional_trajectory_memory_head_cpu():
    cfg = ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_nodes=16,
        max_edges=32,
        use_trajectory_memory_head=True,
        trajectory_memory_projection_dim=16,
    )
    model = ToricTokenGT(cfg)
    batch, _ = synthetic_batch(cfg, batch_size=3, device="cpu")
    out = model(batch)
    assert model.trajectory_memory_head is not None
    assert out["node_embeddings"].shape == (3, 16, cfg.d_model)


def test_tokengt_optional_lm_head_cpu():
    cfg = ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_nodes=16,
        max_edges=32,
        use_lm_head=True,
        lm_vocab_size=1024,
    )
    model = ToricTokenGT(cfg)
    batch, _ = synthetic_batch(cfg, batch_size=2, device="cpu")
    out = model(batch)

    assert out["lm_logits"].shape == (2, 16, 1024)


def test_tokengt_revealed_neighbor_hash_is_rank_safe_cpu():
    import torch

    cfg = ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=1,
        max_nodes=4,
        max_edges=4,
        use_lm_head=True,
        lm_vocab_size=32,
        use_lm_token_embeddings=True,
        use_causal_graph_attention=True,
        use_lm_revealed_neighbor_hash=True,
        lm_revealed_neighbor_hash_buckets=16,
    )
    model = ToricTokenGT(cfg)
    batch = GraphBatch(
        node_features=torch.randn(1, 4, cfg.node_feature_dim),
        edge_features=torch.randn(1, 4, cfg.edge_feature_dim),
        edge_index=torch.tensor([[[0, 1], [1, 2], [2, 3], [0, 0]]], dtype=torch.long),
        node_mask=torch.ones(1, 4, dtype=torch.bool),
        edge_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
        lm_input_ids=torch.tensor([[4, 5, 6, 7]], dtype=torch.long),
        lm_target_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
        lm_mask=torch.ones(1, 4, dtype=torch.bool),
        lm_target_positions=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        node_causal_rank=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
    )
    out = model(batch)

    assert out["lm_logits"].shape == (1, 4, cfg.lm_vocab_size)
    assert torch.isfinite(out["lm_logits"]).all()
    assert torch.isclose(out["lm_revealed_neighbor_known_fraction"], torch.tensor(0.75))


def test_tokengt_can_emit_memory_trace_and_derived_certificates_cpu():
    cfg = ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_nodes=16,
        max_edges=32,
        use_trajectory_memory_head=True,
        trajectory_memory_projection_dim=16,
        derived_category_max_vertices=8,
    )
    model = ToricTokenGT(cfg)
    batch, _ = synthetic_batch(cfg, batch_size=3, device="cpu")
    out = model(batch, include_derived_category=True, include_memory_trace=True, memory_trace_top_k=2)

    json.dumps(out["memory_trace"])
    assert "derived_category" in out
    assert out["memory_trace"]["enabled"]
    assert len(out["memory_trace"]["queries"]) == 3
    assert len(out["memory_trace"]["queries"][0]["top_candidates"]) == 2
    assert "derived_category_similarity" in out["memory_trace"]["queries"][0]["top_candidates"][0]["components"]


def test_tokengt_can_emit_derived_category_certificates_cpu():
    cfg = ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_nodes=16,
        max_edges=32,
        output_derived_category_certificates=False,
        derived_category_max_vertices=8,
    )
    model = ToricTokenGT(cfg)
    batch, _ = synthetic_batch(cfg, batch_size=2, device="cpu")
    out = model(batch, include_derived_category=True)

    assert "derived_category" in out
    assert len(out["derived_category"]) == 2
    assert out["derived_category"][0]["chain_complex"]["kind"] == "got_simplicial_chain_complex"
    assert out["derived_category"][0]["projective_resolution"]["ambient_category"] == "D^b(grmod-S)"
