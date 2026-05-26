import torch

from toricgt.config import ModelConfig
from toricgt.gflownet import EmbeddingPolicy, TrajectoryBatch, trajectory_balance_loss
from toricgt.graph_tokenizer import GraphBatch
from toricgt.model import ToricTokenGT
from toricgt.soft_moe import GraphTokenSoftMoE, PrefixCausalSoftMoE


def synthetic_batch(cfg: ModelConfig, batch_size: int, device: str):
    n = min(8, cfg.max_nodes)
    e = min(16, cfg.max_edges)
    node = torch.randn(batch_size, n, cfg.node_feature_dim, device=device)
    edge = torch.randn(batch_size, e, cfg.edge_feature_dim, device=device)
    edge_index = torch.randint(0, n, (batch_size, e, 2), device=device)
    node_mask = torch.ones(batch_size, n, dtype=torch.bool, device=device)
    edge_mask = torch.ones(batch_size, e, dtype=torch.bool, device=device)
    return GraphBatch(node, edge, edge_index, node_mask, edge_mask), None


def test_graph_token_soft_moe_permutation_equivariance():
    torch.manual_seed(0)
    layer = GraphTokenSoftMoE(
        d_model=16,
        hidden_dim=32,
        num_experts=4,
        slots_per_expert=2,
        dropout=0.0,
    )
    layer.eval()
    x = torch.randn(2, 10, 16)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True, True, False, False, False],
            [True, True, True, True, False, False, False, False, False, False],
        ]
    )
    perm = torch.tensor([3, 0, 1, 9, 4, 2, 7, 5, 6, 8])
    y = layer(x, mask)
    y_perm = layer(x[:, perm], mask[:, perm])
    assert torch.allclose(y[:, perm], y_perm, atol=1e-5, rtol=1e-5)


def test_model_soft_moe_default_four_experts():
    cfg = ModelConfig(d_model=64, num_heads=4, num_layers=4, max_nodes=32, max_edges=64)
    model = ToricTokenGT(cfg)
    soft_moe_blocks = [block for block in model.blocks if getattr(block, "uses_soft_moe", False)]
    assert cfg.use_soft_moe
    assert soft_moe_blocks
    assert soft_moe_blocks[0].ffn.num_experts == 4


def test_tropical_ring_soft_moe_gflownet_backward_cpu():
    torch.manual_seed(1)
    cfg = ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_nodes=16,
        max_edges=32,
        attention="tropical_ring",
        ring_block_size=8,
        use_soft_moe=True,
        soft_moe_num_experts=4,
        soft_moe_slots_per_expert=1,
        soft_moe_start_layer=0,
    )
    model = ToricTokenGT(cfg)
    batch, _ = synthetic_batch(cfg, batch_size=2, device="cpu")
    out = model(batch)
    token_mask = out["token_mask"]
    pooled = (out["token_embeddings"] * token_mask[..., None]).sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp_min(1)
    states = torch.stack([pooled, pooled + 0.01, pooled + 0.02], dim=1)
    policy = EmbeddingPolicy(embedding_dim=cfg.d_model, num_actions=5, hidden_dim=32)
    trajectories = TrajectoryBatch(
        states=states,
        actions=torch.tensor([[0, 1], [2, 3]]),
        terminal_rewards=torch.tensor([1.0, 1.5]),
        lengths=torch.tensor([3, 3]),
    )
    loss = trajectory_balance_loss(policy, trajectories)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(param.grad is not None for param in model.parameters())
    assert policy.log_z.grad is not None


def test_prefix_causal_soft_moe_shape_and_prefix_property():
    torch.manual_seed(2)
    layer = PrefixCausalSoftMoE(d_model=12, hidden_dim=24, num_experts=4, slots_per_expert=1, dropout=0.0)
    layer.eval()
    x = torch.randn(1, 6, 12)
    y = layer(x)
    changed = x.clone()
    changed[:, 4:, :] = torch.randn_like(changed[:, 4:, :])
    y_changed = layer(changed)
    assert y.shape == x.shape
    assert torch.allclose(y[:, :4], y_changed[:, :4], atol=1e-5, rtol=1e-5)
