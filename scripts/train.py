#!/usr/bin/env python3
"""Training entrypoint for ToricGT."""

from __future__ import annotations

import argparse
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from toricgt.config import ModelConfig, TrainConfig
from toricgt.gflownet import TrajectoryBatch, trajectory_balance_loss
from toricgt.graph_dataset import CuratedGraphIterableDataset, collate_graph_items
from toricgt.graph_tokenizer import GraphBatch
from toricgt.metrics import masked_mse
from toricgt.model import ToricTokenGT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: str, precision: str):
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    if precision == "fp32" or device_type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype)


def save_checkpoint(
    checkpoint_dir: Path,
    name: str,
    model: ToricTokenGT,
    optimizer: AdamW,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    step: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": model_cfg.__dict__,
            "train_config": train_cfg.__dict__,
            "step": step,
        },
        checkpoint_dir / name,
    )


def scheduled_lr(
    base_lr: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    schedule: str,
) -> float:
    if schedule == "constant":
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def set_optimizer_lr(optimizer: AdamW, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def auxiliary_gflownet_loss(
    model: ToricTokenGT,
    out: dict[str, torch.Tensor],
    reward: torch.Tensor,
    space: str,
) -> torch.Tensor:
    if model.gflownet_policy is None:
        raise ValueError("model has no GFlowNet policy head")
    token_mask = out["token_mask"]
    tokens = out["token_embeddings"]
    if space == "embedding":
        pooled = (tokens * token_mask[..., None].to(tokens.dtype)).sum(dim=1)
        pooled = pooled / token_mask.sum(dim=1, keepdim=True).clamp_min(1).to(tokens.dtype)
        states = torch.stack([pooled, pooled + 0.01 * torch.tanh(pooled)], dim=1)
    elif space == "token":
        first = tokens[:, 0, :]
        last_index = token_mask.long().sum(dim=1).clamp_min(1) - 1
        last = tokens[torch.arange(tokens.shape[0], device=tokens.device), last_index, :]
        states = torch.stack([first, last], dim=1)
    else:
        raise ValueError(f"unknown GFlowNet space: {space}")
    actions = torch.zeros(tokens.shape[0], 1, dtype=torch.long, device=tokens.device)
    trajectories = TrajectoryBatch(
        states=states,
        actions=actions,
        terminal_rewards=reward.detach().clamp_min(1e-6),
        lengths=torch.full((tokens.shape[0],), 2, dtype=torch.long, device=tokens.device),
    )
    return trajectory_balance_loss(model.gflownet_policy, trajectories)


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


def move_batch(batch: GraphBatch, target: torch.Tensor, device: str) -> tuple[GraphBatch, torch.Tensor]:
    return (
        GraphBatch(
            node_features=batch.node_features.to(device),
            edge_features=batch.edge_features.to(device),
            edge_index=batch.edge_index.to(device),
            node_mask=batch.node_mask.to(device),
            edge_mask=batch.edge_mask.to(device),
        ),
        target.to(device),
    )


@torch.no_grad()
def evaluate_loader(
    model: ToricTokenGT,
    loader: DataLoader,
    device: str,
    precision: str,
    max_batches: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    data_iter = iter(loader)
    for _ in range(max_batches):
        try:
            batch, target = next(data_iter)
        except StopIteration:
            break
        batch, target = move_batch(batch, target, device)
        with autocast_context(device, precision):
            out = model(batch)
            loss = masked_mse(out["node"], target, batch.node_mask)
        total += float(loss.detach())
        count += 1
    model.train()
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--attention", choices=["softmax", "tropical", "tropical_ring", "hybrid"], default="tropical_ring")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no-soft-moe", action="store_true", help="Disable the default Soft-MoE feed-forward blocks.")
    parser.add_argument("--soft-moe-all-layers", action="store_true", help="Use Soft-MoE in every encoder layer.")
    parser.add_argument("--soft-moe-experts", type=int, default=4)
    parser.add_argument("--soft-moe-slots", type=int, default=2)
    parser.add_argument("--data-path", action="append", default=[], help="Curated Parquet path. Can be repeated.")
    parser.add_argument("--parquet-batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--max-edges", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--gflownet-loss-weight", type=float, default=0.0)
    parser.add_argument("--gflownet-space", choices=["embedding", "token"], default="embedding")
    parser.add_argument("--val-data-path", action="append", default=[], help="Validation Parquet/JSONL path. Can be repeated.")
    parser.add_argument("--eval-every", type=int, default=0, help="Run validation every N optimizer steps; disabled when 0.")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--resume", default=None, help="Resume model and optimizer state from a checkpoint.")
    parser.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    args = parser.parse_args()
    set_seed(args.seed)

    model_cfg = ModelConfig(
        attention=args.attention,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        use_soft_moe=not args.no_soft_moe,
        soft_moe_num_experts=args.soft_moe_experts,
        soft_moe_slots_per_expert=args.soft_moe_slots,
        soft_moe_start_layer=0 if args.soft_moe_all_layers else None,
    )
    train_cfg = TrainConfig(
        device=args.device,
        use_wandb=args.wandb,
        batch_size=args.batch_size if args.batch_size is not None else TrainConfig.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.steps,
        precision=args.precision,
        log_interval=args.log_interval,
    )
    model = ToricTokenGT(model_cfg).to(args.device)
    optimizer = AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda") and train_cfg.precision == "fp16")
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
    data_iter = None
    loader = None
    if args.data_path:
        dataset = CuratedGraphIterableDataset(args.data_path, model_cfg, parquet_batch_size=args.parquet_batch_size)
        loader = DataLoader(dataset, batch_size=train_cfg.batch_size, collate_fn=collate_graph_items, num_workers=0)
        data_iter = iter(loader)
    val_loader = None
    if args.val_data_path:
        val_dataset = CuratedGraphIterableDataset(args.val_data_path, model_cfg, parquet_batch_size=args.parquet_batch_size)
        val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, collate_fn=collate_graph_items, num_workers=0)

    run = None
    if train_cfg.use_wandb:
        import wandb

        run = wandb.init(project=train_cfg.wandb_project, config={**model_cfg.__dict__, **train_cfg.__dict__})
        run.summary["parameter_count"] = model.parameter_count()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    ckpt_dir = Path(args.checkpoint_dir)
    final_step = start_step + args.steps
    pbar = tqdm(range(start_step, final_step), desc="train")
    for step in pbar:
        lr = scheduled_lr(
            train_cfg.lr,
            step,
            final_step,
            train_cfg.warmup_steps,
            args.min_lr_ratio,
            args.lr_schedule,
        )
        set_optimizer_lr(optimizer, lr)
        raw_loss_value = 0.0
        supervised_loss_value = 0.0
        gflownet_loss_value = 0.0
        graph_tokens_value = 0
        for accum_idx in range(train_cfg.grad_accum_steps):
            if data_iter is None:
                batch, target = synthetic_batch(model_cfg, train_cfg.batch_size, args.device)
            else:
                try:
                    batch, target = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch, target = next(data_iter)
                batch, target = move_batch(batch, target, args.device)
            with autocast_context(args.device, train_cfg.precision):
                out = model(batch)
                supervised_loss = masked_mse(out["node"], target, batch.node_mask)
                raw_loss = supervised_loss
                gflownet_loss = None
                if args.gflownet_loss_weight > 0:
                    reward = torch.exp(-supervised_loss.detach()).expand(batch.node_features.shape[0])
                    gflownet_loss = auxiliary_gflownet_loss(
                        model,
                        out,
                        reward,
                        args.gflownet_space,
                    )
                    raw_loss = raw_loss + args.gflownet_loss_weight * gflownet_loss
                loss = raw_loss / train_cfg.grad_accum_steps
            raw_loss_value += float(raw_loss.detach())
            supervised_loss_value += float(supervised_loss.detach())
            if gflownet_loss is not None:
                gflownet_loss_value += float(gflownet_loss.detach())
            graph_tokens_value += int(out["token_mask"].sum().detach().cpu())
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        mean_loss = raw_loss_value / train_cfg.grad_accum_steps
        mean_supervised_loss = supervised_loss_value / train_cfg.grad_accum_steps
        mean_gflownet_loss = gflownet_loss_value / max(train_cfg.grad_accum_steps, 1)
        mean_graph_tokens = graph_tokens_value / max(train_cfg.grad_accum_steps, 1)
        pbar.set_postfix(loss=f"{mean_loss:.4f}", params=model.parameter_count())
        if run is not None and step % train_cfg.log_interval == 0:
            metrics = {
                "train/loss": mean_loss,
                "train/supervised_loss": mean_supervised_loss,
                "train/gflownet_loss": mean_gflownet_loss,
                "train/gflownet_loss_weight": args.gflownet_loss_weight,
                "train/graph_tokens_per_microbatch": mean_graph_tokens,
                "train/lr": lr,
                "train/grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                "train/step": step,
            }
            if args.device.startswith("cuda"):
                metrics["system/vram_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            moe_diags = model.soft_moe_diagnostics()
            if moe_diags:
                metrics.update(
                    {
                        "moe/expert_mass_min": min(item["expert_mass_min"] for item in moe_diags),
                        "moe/expert_mass_max": max(item["expert_mass_max"] for item in moe_diags),
                        "moe/dispatch_entropy": sum(item["dispatch_entropy_mean"] for item in moe_diags) / len(moe_diags),
                        "moe/combine_entropy": sum(item["combine_entropy"] for item in moe_diags) / len(moe_diags),
                    }
                )
            run.log(metrics)
        if val_loader is not None and args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            val_loss = evaluate_loader(model, val_loader, args.device, train_cfg.precision, args.eval_batches)
            pbar.write(f"validation step={step + 1} masked_mse={val_loss:.6f}")
            if run is not None:
                run.log({"val/masked_mse": val_loss, "train/step": step + 1})
        if args.checkpoint_every > 0 and (step + 1) % args.checkpoint_every == 0:
            save_checkpoint(ckpt_dir, f"toricgt_step_{step + 1:08d}.pt", model, optimizer, model_cfg, train_cfg, step + 1)

    save_checkpoint(ckpt_dir, "toricgt_final.pt", model, optimizer, model_cfg, train_cfg, final_step)


if __name__ == "__main__":
    main()
