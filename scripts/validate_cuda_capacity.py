#!/usr/bin/env python3
"""One-step CUDA capacity validation for ToricGT.

This intentionally uses random graph batches. It verifies model construction,
full forward/backward, optimizer update, default Soft-MoE routing, and optional
embedding-space GFlowNet loss at a requested graph-token size.
"""

from __future__ import annotations

import argparse

import torch

from toricgt.config import ModelConfig
from toricgt.gflownet import TrajectoryBatch, trajectory_balance_loss
from toricgt.graph_tokenizer import GraphBatch
from toricgt.metrics import masked_mse
from toricgt.model import ToricTokenGT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=["softmax", "tropical", "tropical_ring", "hybrid"], default="hybrid")
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--max-edges", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--gflownet-loss-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-soft-moe", action="store_true")
    return parser.parse_args()


def autocast_context(device: str, precision: str):
    if not device.startswith("cuda") or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_batch(cfg: ModelConfig, batch_size: int, device: str) -> GraphBatch:
    return GraphBatch(
        node_features=torch.randn(batch_size, cfg.max_nodes, cfg.node_feature_dim, device=device),
        edge_features=torch.randn(batch_size, cfg.max_edges, cfg.edge_feature_dim, device=device),
        edge_index=torch.randint(0, cfg.max_nodes, (batch_size, cfg.max_edges, 2), device=device),
        node_mask=torch.ones(batch_size, cfg.max_nodes, dtype=torch.bool, device=device),
        edge_mask=torch.ones(batch_size, cfg.max_edges, dtype=torch.bool, device=device),
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    cfg = ModelConfig(
        attention=args.attention,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        use_soft_moe=not args.no_soft_moe,
    )
    model = ToricTokenGT(cfg).to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = make_batch(cfg, args.batch_size, args.device)
    target = torch.zeros(args.batch_size, cfg.max_nodes, cfg.output_dim, device=args.device)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    with autocast_context(args.device, args.precision):
        out = model(batch)
        supervised = masked_mse(out["node"], target, batch.node_mask)
        loss = supervised
        if args.gflownet_loss_weight > 0:
            token_mask = out["token_mask"]
            pooled = (out["token_embeddings"] * token_mask[..., None].to(out["token_embeddings"].dtype)).sum(dim=1)
            pooled = pooled / token_mask.sum(dim=1, keepdim=True).clamp_min(1).to(out["token_embeddings"].dtype)
            trajectories = TrajectoryBatch(
                states=torch.stack([pooled, pooled + 0.01 * torch.tanh(pooled)], dim=1),
                actions=torch.zeros(args.batch_size, 1, dtype=torch.long, device=args.device),
                terminal_rewards=torch.exp(-supervised.detach()).expand(args.batch_size).clamp_min(1e-6),
                lengths=torch.full((args.batch_size,), 2, dtype=torch.long, device=args.device),
            )
            loss = loss + args.gflownet_loss_weight * trajectory_balance_loss(model.gflownet_policy, trajectories)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    peak_vram_gb = None
    if args.device.startswith("cuda"):
        peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    print(
        {
            "params": model.parameter_count(),
            "tokens": cfg.max_nodes + cfg.max_edges,
            "loss": float(loss.detach().cpu()),
            "peak_vram_gb": peak_vram_gb,
            "soft_moe_blocks": sum(1 for block in model.blocks if getattr(block, "uses_soft_moe", False)),
            "gflownet_loss_weight": args.gflownet_loss_weight,
        }
    )


if __name__ == "__main__":
    main()
