#!/usr/bin/env python3
"""CPU-only implementation validation that avoids meaningful VRAM use."""

from __future__ import annotations

from toricgt.config import ModelConfig
from toricgt.gflownet import EmbeddingPolicy, TrajectoryBatch, trajectory_balance_loss
from toricgt.graph_tokenizer import GraphBatch
from toricgt.model import ToricTokenGT
from toricgt.toric_algebra import RotationAlgebraApproximant


def synthetic_batch(cfg: ModelConfig, batch_size: int, device: str) -> tuple[GraphBatch, object]:
    import torch

    n = min(16, cfg.max_nodes)
    e = min(48, cfg.max_edges)
    node = torch.randn(batch_size, n, cfg.node_feature_dim, device=device)
    edge = torch.randn(batch_size, e, cfg.edge_feature_dim, device=device)
    edge_index = torch.randint(0, n, (batch_size, e, 2), device=device)
    node_mask = torch.ones(batch_size, n, dtype=torch.bool, device=device)
    edge_mask = torch.ones(batch_size, e, dtype=torch.bool, device=device)
    return GraphBatch(node, edge, edge_index, node_mask, edge_mask), None


def main() -> None:
    cfg = ModelConfig(d_model=64, num_heads=4, num_layers=2, max_nodes=32, max_edges=64)
    model = ToricTokenGT(cfg)
    batch, _ = synthetic_batch(cfg, batch_size=2, device="cpu")
    out = model(batch)
    import torch

    pooled = (out["token_embeddings"] * out["token_mask"][..., None]).sum(dim=1)
    pooled = pooled / out["token_mask"].sum(dim=1, keepdim=True).clamp_min(1)
    policy = EmbeddingPolicy(embedding_dim=cfg.d_model, num_actions=5, hidden_dim=64)
    trajectories = TrajectoryBatch(
        states=torch.stack([pooled, pooled + 0.01, pooled + 0.02], dim=1),
        actions=torch.tensor([[0, 1], [2, 3]]),
        terminal_rewards=torch.tensor([1.0, 1.25]),
        lengths=torch.tensor([3, 3]),
    )
    gflownet_loss = trajectory_balance_loss(policy, trajectories)
    gflownet_loss.backward()
    approx = RotationAlgebraApproximant(theta=cfg.theta, max_denominator=32)
    print(
        {
            "params": model.parameter_count(),
            "soft_moe_blocks": sum(getattr(block, "uses_soft_moe", False) for block in model.blocks),
            "soft_moe_experts": cfg.soft_moe_num_experts,
            "node_shape": tuple(out["node"].shape),
            "edge_shape": tuple(out["edge"].shape),
            "gflownet_loss": float(gflownet_loss.detach()),
            "commutator_error": approx.commutator_error(),
        }
    )


if __name__ == "__main__":
    main()
