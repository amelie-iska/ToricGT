#!/usr/bin/env python3
"""Training entrypoint for ToricGT."""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from toricgt.cli_config import apply_yaml_defaults, parse_config_path
from toricgt.config import ModelConfig, TrainConfig
from toricgt.derived_category_metrics import (
    DerivedCategoryConfig,
    analogical_derived_category_loss,
    derived_category_objects_from_batch,
)
from toricgt.expert_curriculum import CyclicExpertCurriculum, ExpertCurriculumAssignment
from toricgt.gflownet import TrajectoryBatch, trajectory_balance_loss
from toricgt.got_trajectory import got_dag_metrics
from toricgt.graph_dataset import CuratedGraphIterableDataset, collate_graph_items
from toricgt.graph_tokenizer import GraphBatch
from toricgt.metrics import masked_mse
from toricgt.model import ToricTokenGT
from toricgt.wandb_organization import configure_wandb_metrics, organize_wandb_payload


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
    extra_metadata: dict | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model_cfg.__dict__,
        "train_config": train_cfg.__dict__,
        "step": step,
    }
    if extra_metadata:
        payload.update(extra_metadata)
    torch.save(payload, checkpoint_dir / name)


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


def per_node_prediction_mse(out_node: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    err = (out_node.float() - target.float()).pow(2).mean(dim=-1)
    return torch.where(node_mask, err, torch.zeros_like(err))


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


def next_loader_batch(
    loader: DataLoader,
    data_iter,
    device: str,
) -> tuple[GraphBatch, torch.Tensor, object]:
    try:
        batch, target = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch, target = next(data_iter)
    batch, target = move_batch(batch, target, device)
    return batch, target, data_iter


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
    parser.add_argument("--config", default=None, help="YAML file whose keys become CLI defaults.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--attention", choices=["softmax", "tropical", "tropical_ring", "hybrid"], default="tropical_ring")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-soft-moe", action="store_true", help="Disable the default Soft-MoE feed-forward blocks.")
    parser.add_argument("--soft-moe-all-layers", action="store_true", help="Use Soft-MoE in every encoder layer.")
    parser.add_argument("--soft-moe-experts", type=int, default=4)
    parser.add_argument("--soft-moe-slots", type=int, default=2)
    parser.add_argument("--data-path", action="append", default=[], help="Curated Parquet path. Can be repeated.")
    parser.add_argument("--parquet-batch-size", type=int, default=512)
    parser.add_argument("--interleave-data-paths", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fineweb-tokenizer-path",
        default="amelie-iska/parameter-golf/data/tokenizers/fineweb_1024_bpe.model",
    )
    parser.add_argument("--fineweb-tokens-per-graph", type=int, default=1024)
    parser.add_argument("--fineweb-stride-tokens", type=int, default=1024)
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
    parser.add_argument("--got-dag-loss-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-memory-loss-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-memory-projection-dim", type=int, default=128)
    parser.add_argument("--trajectory-memory-dag-weight", type=float, default=0.20)
    parser.add_argument("--trajectory-memory-derived-weight", type=float, default=0.20)
    parser.add_argument("--derived-category-loss-weight", type=float, default=0.0)
    parser.add_argument("--derived-category-max-vertices", type=int, default=8)
    parser.add_argument("--output-derived-category-certificates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-derived-category-examples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--derived-category-example-every", type=int, default=1000)
    parser.add_argument("--derived-category-example-max-vertices", type=int, default=6)
    parser.add_argument("--derived-category-example-samples", type=int, default=1)
    parser.add_argument("--write-memory-trace-examples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--memory-trace-example-every", type=int, default=1000)
    parser.add_argument("--memory-trace-top-k", type=int, default=3)
    parser.add_argument("--memory-trace-example-samples", type=int, default=2)
    parser.add_argument("--full-dataset-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fineweb-mix-ratio", type=float, default=0.0)
    parser.add_argument("--target-artifact-bytes", type=int, default=16_000_000)
    parser.add_argument("--val-data-path", action="append", default=[], help="Validation Parquet/JSONL path. Can be repeated.")
    parser.add_argument("--eval-every", type=int, default=0, help="Run validation every N optimizer steps; disabled when 0.")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--resume", default=None, help="Resume model and optimizer state from a checkpoint.")
    parser.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--expert-cyclic-curriculum", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expert-curriculum-subsets", type=int, default=0)
    parser.add_argument("--expert-curriculum-phase-steps", type=int, default=500)
    parser.add_argument("--expert-curriculum-order", choices=["cyclic", "braid"], default="braid")
    parser.add_argument("--expert-curriculum-start-step", type=int, default=0)
    parser.add_argument("--expert-curriculum-salt", type=int, default=17)
    parser.add_argument("--expert-curriculum-distill-weight", type=float, default=0.0)
    apply_yaml_defaults(parser, parse_config_path())
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
        use_trajectory_memory_head=args.trajectory_memory_loss_weight > 0 or args.write_memory_trace_examples,
        trajectory_memory_projection_dim=args.trajectory_memory_projection_dim,
        trajectory_memory_dag_weight=args.trajectory_memory_dag_weight,
        trajectory_memory_derived_weight=args.trajectory_memory_derived_weight,
        output_derived_category_certificates=args.output_derived_category_certificates,
        derived_category_max_vertices=args.derived_category_max_vertices,
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

    curriculum = None
    curriculum_config = None
    if args.expert_cyclic_curriculum:
        if model_cfg.soft_moe_num_experts < 2:
            raise ValueError("expert cyclic curriculum requires at least two Soft-MoE experts")
        num_subsets = args.expert_curriculum_subsets or model_cfg.soft_moe_num_experts
        curriculum = CyclicExpertCurriculum(
            num_experts=model_cfg.soft_moe_num_experts,
            num_subsets=num_subsets,
            phase_steps=args.expert_curriculum_phase_steps,
            order=args.expert_curriculum_order,
            start_step=args.expert_curriculum_start_step,
        )
        curriculum_config = {
            "enabled": True,
            "num_experts": model_cfg.soft_moe_num_experts,
            "num_subsets": num_subsets,
            "phase_steps": args.expert_curriculum_phase_steps,
            "order": args.expert_curriculum_order,
            "start_step": args.expert_curriculum_start_step,
            "subset_salt": args.expert_curriculum_salt,
            "distill_weight": args.expert_curriculum_distill_weight,
            "expert_orders": {
                str(expert_idx): curriculum.expert_order(expert_idx)
                for expert_idx in range(model_cfg.soft_moe_num_experts)
            },
        }

    data_iter = None
    loader = None
    subset_loaders: list[DataLoader] = []
    subset_iters: list[object] = []
    if args.data_path:
        if curriculum is None:
            dataset = CuratedGraphIterableDataset(
                args.data_path,
                model_cfg,
                parquet_batch_size=args.parquet_batch_size,
                interleave_paths=args.interleave_data_paths,
                fineweb_tokenizer_path=args.fineweb_tokenizer_path,
                fineweb_tokens_per_graph=args.fineweb_tokens_per_graph,
                fineweb_stride_tokens=args.fineweb_stride_tokens,
            )
            loader = DataLoader(dataset, batch_size=train_cfg.batch_size, collate_fn=collate_graph_items, num_workers=0)
            data_iter = iter(loader)
        else:
            for subset_id in range(curriculum.num_subsets):
                dataset = CuratedGraphIterableDataset(
                    args.data_path,
                    model_cfg,
                    parquet_batch_size=args.parquet_batch_size,
                    subset_id=subset_id,
                    num_subsets=curriculum.num_subsets,
                    subset_salt=args.expert_curriculum_salt,
                    interleave_paths=args.interleave_data_paths,
                    fineweb_tokenizer_path=args.fineweb_tokenizer_path,
                    fineweb_tokens_per_graph=args.fineweb_tokens_per_graph,
                    fineweb_stride_tokens=args.fineweb_stride_tokens,
                )
                subset_loader = DataLoader(
                    dataset,
                    batch_size=train_cfg.batch_size,
                    collate_fn=collate_graph_items,
                    num_workers=0,
                )
                subset_loaders.append(subset_loader)
                subset_iters.append(iter(subset_loader))
    val_loader = None
    if args.val_data_path:
        val_dataset = CuratedGraphIterableDataset(
            args.val_data_path,
            model_cfg,
            parquet_batch_size=args.parquet_batch_size,
            interleave_paths=args.interleave_data_paths,
            fineweb_tokenizer_path=args.fineweb_tokenizer_path,
            fineweb_tokens_per_graph=args.fineweb_tokens_per_graph,
            fineweb_stride_tokens=args.fineweb_stride_tokens,
        )
        val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, collate_fn=collate_graph_items, num_workers=0)

    run = None
    if train_cfg.use_wandb:
        import wandb

        run = wandb.init(
            project=train_cfg.wandb_project,
            config={
                **model_cfg.__dict__,
                **train_cfg.__dict__,
                "expert_curriculum": curriculum_config or {"enabled": False},
                "got_dag_loss_weight": args.got_dag_loss_weight,
                "trajectory_memory_loss_weight": args.trajectory_memory_loss_weight,
                "trajectory_memory_dag_weight": args.trajectory_memory_dag_weight,
                "trajectory_memory_derived_weight": args.trajectory_memory_derived_weight,
                "derived_category_loss_weight": args.derived_category_loss_weight,
                "derived_category_max_vertices": args.derived_category_max_vertices,
                "output_derived_category_certificates": args.output_derived_category_certificates,
                "write_derived_category_examples": args.write_derived_category_examples,
                "derived_category_example_max_vertices": args.derived_category_example_max_vertices,
                "derived_category_example_samples": args.derived_category_example_samples,
                "write_memory_trace_examples": args.write_memory_trace_examples,
                "memory_trace_example_every": args.memory_trace_example_every,
                "memory_trace_top_k": args.memory_trace_top_k,
                "memory_trace_example_samples": args.memory_trace_example_samples,
                "full_dataset_run": args.full_dataset_run,
                "fineweb_mix_ratio": args.fineweb_mix_ratio,
                "interleave_data_paths": args.interleave_data_paths,
                "fineweb_tokenizer_path": args.fineweb_tokenizer_path,
                "fineweb_tokens_per_graph": args.fineweb_tokens_per_graph,
                "fineweb_stride_tokens": args.fineweb_stride_tokens,
                "target_artifact_bytes": args.target_artifact_bytes,
            },
        )
        configure_wandb_metrics(wandb, step_metric="train/step")
        run.summary["parameter_count"] = model.parameter_count()
        if curriculum_config is not None:
            run.summary["expert_curriculum_order"] = str(curriculum_config["expert_orders"])

    model.train()
    optimizer.zero_grad(set_to_none=True)
    ckpt_dir = Path(args.checkpoint_dir)
    final_step = start_step + args.steps
    pbar = tqdm(range(start_step, final_step), desc="train")
    for step in pbar:
        assignment: ExpertCurriculumAssignment | None = curriculum.assignment(step) if curriculum is not None else None
        if assignment is not None:
            model.set_active_soft_moe_experts([assignment.active_expert])
        else:
            model.set_active_soft_moe_experts(None)
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
        got_dag_loss_value = 0.0
        trajectory_memory_loss_value = 0.0
        trajectory_memory_recall1_value = 0.0
        trajectory_memory_entropy_value = 0.0
        trajectory_memory_dag_similarity_value = 0.0
        derived_category_loss_value = 0.0
        got_metric_sums: dict[str, float] = {}
        trajectory_metric_sums: dict[str, float] = {}
        derived_metric_sums: dict[str, float] = {}
        distill_loss_value = 0.0
        teacher_loss_value = 0.0
        graph_tokens_value = 0
        last_batch_for_examples: GraphBatch | None = None
        last_node_embeddings_for_examples: torch.Tensor | None = None
        last_per_node_mse_for_examples: torch.Tensor | None = None
        for accum_idx in range(train_cfg.grad_accum_steps):
            if data_iter is None and not subset_loaders:
                batch, target = synthetic_batch(model_cfg, train_cfg.batch_size, args.device)
            elif assignment is not None:
                subset_id = assignment.subset_id
                batch, target, subset_iters[subset_id] = next_loader_batch(
                    subset_loaders[subset_id],
                    subset_iters[subset_id],
                    args.device,
                )
            else:
                batch, target, data_iter = next_loader_batch(loader, data_iter, args.device)

            teacher_out = None
            teacher_supervised_loss = None
            if (
                assignment is not None
                and assignment.teacher_expert is not None
                and args.expert_curriculum_distill_weight > 0
            ):
                model.set_active_soft_moe_experts([assignment.teacher_expert])
                with torch.no_grad(), autocast_context(args.device, train_cfg.precision):
                    teacher_out = model(batch)
                    teacher_supervised_loss = masked_mse(teacher_out["node"], target, batch.node_mask)
                model.set_active_soft_moe_experts([assignment.active_expert])

            with autocast_context(args.device, train_cfg.precision):
                out = model(batch)
                supervised_loss = masked_mse(out["node"], target, batch.node_mask)
                raw_loss = supervised_loss
                distill_loss = None
                if teacher_out is not None and args.expert_curriculum_distill_weight > 0:
                    distill_loss = masked_mse(out["node"], teacher_out["node"].detach(), batch.node_mask)
                    raw_loss = raw_loss + args.expert_curriculum_distill_weight * distill_loss
                gflownet_loss = None
                if args.gflownet_loss_weight > 0:
                    reward_scalar = torch.exp(-supervised_loss.detach())
                    if teacher_supervised_loss is not None:
                        advantage = (teacher_supervised_loss - supervised_loss).detach().clamp(-5.0, 5.0)
                        reward_scalar = reward_scalar * torch.exp(advantage)
                    reward = reward_scalar.expand(batch.node_features.shape[0])
                    gflownet_loss = auxiliary_gflownet_loss(
                        model,
                        out,
                        reward,
                        args.gflownet_space,
                    )
                    raw_loss = raw_loss + args.gflownet_loss_weight * gflownet_loss
                got_metrics = None
                if args.got_dag_loss_weight > 0 or args.trajectory_memory_loss_weight > 0:
                    got_metrics = got_dag_metrics(
                        out["node_embeddings"],
                        node_mask=batch.node_mask,
                        edge_index=batch.edge_index,
                        edge_mask=batch.edge_mask,
                    )
                    if args.got_dag_loss_weight > 0:
                        raw_loss = raw_loss + args.got_dag_loss_weight * got_metrics["got_dag_loss"]
                derived_category_metrics = None
                if args.derived_category_loss_weight > 0:
                    derived_category_metrics = analogical_derived_category_loss(
                        out["node_embeddings"],
                        node_mask=batch.node_mask,
                        edge_index=batch.edge_index,
                        edge_mask=batch.edge_mask,
                        config=DerivedCategoryConfig(max_vertices=args.derived_category_max_vertices),
                    )
                    raw_loss = raw_loss + args.derived_category_loss_weight * derived_category_metrics["derived_category_loss"]
                trajectory_memory_metrics = None
                if model.trajectory_memory_head is not None and args.trajectory_memory_loss_weight > 0:
                    positions = torch.arange(out["node_embeddings"].shape[1], device=args.device).unsqueeze(0)
                    positions = positions.expand(out["node_embeddings"].shape[0], -1)
                    per_node_mse = per_node_prediction_mse(out["node"], target, batch.node_mask)
                    trajectory_memory_metrics = model.trajectory_memory_head(
                        out["node_embeddings"],
                        positions,
                        per_node_mse,
                        trajectory_node_mask=batch.node_mask,
                        trajectory_edge_index=batch.edge_index,
                        trajectory_edge_mask=batch.edge_mask,
                    )
                    raw_loss = raw_loss + args.trajectory_memory_loss_weight * trajectory_memory_metrics["trajectory_memory_loss"]
                elif args.write_memory_trace_examples and model.trajectory_memory_head is not None:
                    per_node_mse = per_node_prediction_mse(out["node"], target, batch.node_mask)
                loss = raw_loss / train_cfg.grad_accum_steps
            last_batch_for_examples = batch
            if args.write_memory_trace_examples and model.trajectory_memory_head is not None:
                last_node_embeddings_for_examples = out["node_embeddings"].detach()
                last_per_node_mse_for_examples = per_node_mse.detach()
            raw_loss_value += float(raw_loss.detach())
            supervised_loss_value += float(supervised_loss.detach())
            if distill_loss is not None:
                distill_loss_value += float(distill_loss.detach())
            if teacher_supervised_loss is not None:
                teacher_loss_value += float(teacher_supervised_loss.detach())
            if gflownet_loss is not None:
                gflownet_loss_value += float(gflownet_loss.detach())
            if got_metrics is not None:
                got_dag_loss_value += float(got_metrics["got_dag_loss"].detach())
                for key, value in got_metrics.items():
                    if key.endswith("_batch"):
                        continue
                    if torch.is_tensor(value) and value.ndim == 0:
                        got_metric_sums[key] = got_metric_sums.get(key, 0.0) + float(value.detach())
            if derived_category_metrics is not None:
                derived_category_loss_value += float(derived_category_metrics["derived_category_loss"].detach())
                for key, value in derived_category_metrics.items():
                    if torch.is_tensor(value) and value.ndim == 0:
                        derived_metric_sums[key] = derived_metric_sums.get(key, 0.0) + float(value.detach())
            if trajectory_memory_metrics is not None:
                trajectory_memory_loss_value += float(trajectory_memory_metrics["trajectory_memory_loss"].detach())
                trajectory_memory_recall1_value += float(trajectory_memory_metrics["trajectory_memory_recall1"].detach())
                trajectory_memory_entropy_value += float(trajectory_memory_metrics["trajectory_memory_entropy"].detach())
                trajectory_memory_dag_similarity_value += float(
                    trajectory_memory_metrics["trajectory_memory_dag_similarity"].detach()
                )
                for key, value in trajectory_memory_metrics.items():
                    if torch.is_tensor(value) and value.ndim == 0:
                        trajectory_metric_sums[key] = trajectory_metric_sums.get(key, 0.0) + float(value.detach())
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
        mean_got_dag_loss = got_dag_loss_value / max(train_cfg.grad_accum_steps, 1)
        mean_trajectory_memory_loss = trajectory_memory_loss_value / max(train_cfg.grad_accum_steps, 1)
        mean_trajectory_memory_recall1 = trajectory_memory_recall1_value / max(train_cfg.grad_accum_steps, 1)
        mean_trajectory_memory_entropy = trajectory_memory_entropy_value / max(train_cfg.grad_accum_steps, 1)
        mean_trajectory_memory_dag_similarity = trajectory_memory_dag_similarity_value / max(train_cfg.grad_accum_steps, 1)
        mean_derived_category_loss = derived_category_loss_value / max(train_cfg.grad_accum_steps, 1)
        mean_distill_loss = distill_loss_value / max(train_cfg.grad_accum_steps, 1)
        mean_teacher_loss = teacher_loss_value / max(train_cfg.grad_accum_steps, 1)
        mean_graph_tokens = graph_tokens_value / max(train_cfg.grad_accum_steps, 1)
        pbar.set_postfix(loss=f"{mean_loss:.4f}", params=model.parameter_count())
        if run is not None and step % train_cfg.log_interval == 0:
            metrics = {
                "train/loss": mean_loss,
                "train/supervised_loss": mean_supervised_loss,
                "train/gflownet_loss": mean_gflownet_loss,
                "train/gflownet_loss_weight": args.gflownet_loss_weight,
                "train/got_dag_loss": mean_got_dag_loss,
                "train/got_dag_loss_weight": args.got_dag_loss_weight,
                "train/trajectory_memory_loss": mean_trajectory_memory_loss,
                "train/trajectory_memory_loss_weight": args.trajectory_memory_loss_weight,
                "train/trajectory_memory_recall1": mean_trajectory_memory_recall1,
                "train/trajectory_memory_entropy": mean_trajectory_memory_entropy,
                "train/trajectory_memory_dag_similarity": mean_trajectory_memory_dag_similarity,
                "train/derived_category_loss": mean_derived_category_loss,
                "train/derived_category_loss_weight": args.derived_category_loss_weight,
                "train/derived_category_max_vertices": args.derived_category_max_vertices,
                "train/expert_distill_loss": mean_distill_loss,
                "train/expert_teacher_supervised_loss": mean_teacher_loss,
                "train/graph_tokens_per_microbatch": mean_graph_tokens,
                "train/lr": lr,
                "train/grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                "train/step": step,
                "data/full_curated_train_split_active": float(bool(args.full_dataset_run)),
                "data/fineweb_mix_ratio": args.fineweb_mix_ratio,
                "data/interleave_data_paths": float(bool(args.interleave_data_paths)),
                "data/fineweb_tokens_per_graph": float(args.fineweb_tokens_per_graph),
                "data/fineweb_stride_tokens": float(args.fineweb_stride_tokens),
                "artifact/target_size_limit_bytes": args.target_artifact_bytes,
            }
            for key, value in got_metric_sums.items():
                metrics[f"train/{key}"] = value / max(train_cfg.grad_accum_steps, 1)
            for key, value in trajectory_metric_sums.items():
                metrics[f"train/{key}"] = value / max(train_cfg.grad_accum_steps, 1)
            for key, value in derived_metric_sums.items():
                metrics[f"train/{key}"] = value / max(train_cfg.grad_accum_steps, 1)
            if assignment is not None:
                metrics.update(
                    {
                        "expert_curriculum/phase": assignment.phase,
                        "expert_curriculum/round": assignment.round_index,
                        "expert_curriculum/active_expert": assignment.active_expert,
                        "expert_curriculum/subset_id": assignment.subset_id,
                        "expert_curriculum/full_coverage_cycles": assignment.full_coverage_cycles,
                        "expert_curriculum/full_coverage_complete": float(assignment.full_coverage_complete),
                        "expert_curriculum/teacher_expert": -1
                        if assignment.teacher_expert is None
                        else assignment.teacher_expert,
                    }
                )
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
            run.log(organize_wandb_payload(metrics))
        if (
            args.write_derived_category_examples
            and last_batch_for_examples is not None
            and args.derived_category_example_every > 0
            and (step + 1) % args.derived_category_example_every == 0
        ):
            example_dir = ckpt_dir / "derived_category_examples"
            example_dir.mkdir(parents=True, exist_ok=True)
            example_samples = max(1, min(int(args.derived_category_example_samples), last_batch_for_examples.edge_index.shape[0]))
            objects = derived_category_objects_from_batch(
                last_batch_for_examples.edge_index[:example_samples],
                node_mask=last_batch_for_examples.node_mask[:example_samples],
                edge_mask=last_batch_for_examples.edge_mask[:example_samples],
                max_vertices=args.derived_category_example_max_vertices,
            )
            with (example_dir / f"step_{step + 1:08d}.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "step": int(step + 1),
                        "derived_category_loss_weight": float(args.derived_category_loss_weight),
                        "objects": objects,
                    },
                    handle,
                    indent=2,
                )
        if (
            args.write_memory_trace_examples
            and model.trajectory_memory_head is not None
            and last_batch_for_examples is not None
            and last_node_embeddings_for_examples is not None
            and last_per_node_mse_for_examples is not None
            and args.memory_trace_example_every > 0
            and (step + 1) % args.memory_trace_example_every == 0
        ):
            memory_dir = ckpt_dir / "memory_trace_examples"
            memory_dir.mkdir(parents=True, exist_ok=True)
            trace_samples = max(1, min(int(args.memory_trace_example_samples), last_node_embeddings_for_examples.shape[0]))
            trace_positions = torch.arange(
                last_node_embeddings_for_examples.shape[1],
                device=last_node_embeddings_for_examples.device,
            ).unsqueeze(0)
            trace_positions = trace_positions.expand(trace_samples, -1)
            trace = model.trajectory_memory_head.trace(
                last_node_embeddings_for_examples[:trace_samples],
                trace_positions,
                last_per_node_mse_for_examples[:trace_samples],
                trajectory_node_mask=last_batch_for_examples.node_mask[:trace_samples],
                trajectory_edge_index=last_batch_for_examples.edge_index[:trace_samples],
                trajectory_edge_mask=last_batch_for_examples.edge_mask[:trace_samples],
                top_k=args.memory_trace_top_k,
            )
            derived_objects = derived_category_objects_from_batch(
                last_batch_for_examples.edge_index[:trace_samples],
                node_mask=last_batch_for_examples.node_mask[:trace_samples],
                edge_mask=last_batch_for_examples.edge_mask[:trace_samples],
                max_vertices=args.derived_category_example_max_vertices,
            )
            with (memory_dir / f"step_{step + 1:08d}.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "step": int(step + 1),
                        "trajectory_memory_loss_weight": float(args.trajectory_memory_loss_weight),
                        "trajectory_memory_dag_weight": float(args.trajectory_memory_dag_weight),
                        "trajectory_memory_derived_weight": float(args.trajectory_memory_derived_weight),
                        "memory_trace": trace,
                        "derived_category_objects": derived_objects,
                    },
                    handle,
                    indent=2,
                )
        if val_loader is not None and args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            if assignment is not None:
                model.set_active_soft_moe_experts(None)
            val_loss = evaluate_loader(model, val_loader, args.device, train_cfg.precision, args.eval_batches)
            if assignment is not None:
                model.set_active_soft_moe_experts([assignment.active_expert])
            pbar.write(f"validation step={step + 1} masked_mse={val_loss:.6f}")
            if run is not None:
                run.log(organize_wandb_payload({"val/masked_mse": val_loss, "train/step": step + 1}))
        if args.checkpoint_every > 0 and (step + 1) % args.checkpoint_every == 0:
            save_checkpoint(
                ckpt_dir,
                f"toricgt_step_{step + 1:08d}.pt",
                model,
                optimizer,
                model_cfg,
                train_cfg,
                step + 1,
                {"expert_curriculum": curriculum_config} if curriculum_config is not None else None,
            )

    model.set_active_soft_moe_experts(None)
    save_checkpoint(
        ckpt_dir,
        "toricgt_final.pt",
        model,
        optimizer,
        model_cfg,
        train_cfg,
        final_step,
        {"expert_curriculum": curriculum_config} if curriculum_config is not None else None,
    )


if __name__ == "__main__":
    main()
