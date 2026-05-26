#!/usr/bin/env python3
"""Small evaluation harness for checkpoints."""

from __future__ import annotations

import argparse

import torch
from tqdm.auto import tqdm

from toricgt.config import ModelConfig
from toricgt.graph_tokenizer import GraphBatch
from toricgt.metrics import masked_mse
from toricgt.model import ToricTokenGT


def synthetic_batch(cfg: ModelConfig, batch_size: int, device: str) -> tuple[GraphBatch, torch.Tensor]:
    n = min(16, cfg.max_nodes)
    e = min(48, cfg.max_edges)
    node = torch.randn(batch_size, n, cfg.node_feature_dim, device=device)
    edge = torch.randn(batch_size, e, cfg.edge_feature_dim, device=device)
    edge_index = torch.randint(0, n, (batch_size, e, 2), device=device)
    node_mask = torch.ones(batch_size, n, dtype=torch.bool, device=device)
    edge_mask = torch.ones(batch_size, e, dtype=torch.bool, device=device)
    target = torch.zeros(batch_size, n, cfg.output_dim, device=device)
    target[..., : min(cfg.output_dim, cfg.node_feature_dim)] = node[..., : min(cfg.output_dim, cfg.node_feature_dim)]
    return GraphBatch(node, edge, edge_index, node_mask, edge_mask), target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/toricgt_final.pt")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location=args.device)
    cfg = ModelConfig(**payload["config"])
    model = ToricTokenGT(cfg).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in tqdm(range(args.batches), desc="eval"):
            batch, target = synthetic_batch(cfg, 4, args.device)
            out = model(batch)
            losses.append(masked_mse(out["node"], target, batch.node_mask).item())
    print({"eval/masked_mse": sum(losses) / len(losses), "batches": args.batches})


if __name__ == "__main__":
    main()
