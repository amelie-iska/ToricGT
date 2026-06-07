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
