#!/usr/bin/env python3
"""Train the dense random-order ToricGT Parameter-Golf adapter."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm.auto import tqdm

from toricgt.parameter_golf_export import PARAMETER_GOLF_BYTE_LIMIT, write_artifact
from toricgt.random_order_lm import (
    DenseRandomOrderToricLM,
    RandomOrderLMConfig,
    byte_encode,
    estimate_uncompressed_quantized_bytes,
)


TEXT_COLUMNS = ("text", "question", "reasoning", "solution", "answer", "graph_json")


def read_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def config_get(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return config.get(section, {}).get(key, default)


class ParquetByteChunkDataset(IterableDataset):
    """Stream byte-level chunks from curated Parquet rows."""

    def __init__(
        self,
        parquet_glob: str,
        seq_len: int,
        byte_offset: int,
        rows_per_batch: int = 128,
        seed: int = 17,
        repeat: bool = True,
        shuffle_files: bool = True,
        include_graph_projection: bool = True,
        graph_projection_max_chars: int = 4096,
    ) -> None:
        super().__init__()
        self.files = sorted(glob.glob(parquet_glob))
        self.seq_len = seq_len
        self.byte_offset = byte_offset
        self.rows_per_batch = rows_per_batch
        self.seed = seed
        self.repeat = repeat
        self.shuffle_files = shuffle_files
        self.include_graph_projection = include_graph_projection
        self.graph_projection_max_chars = graph_projection_max_chars

    def _graph_projection(self, graph_json: Any) -> str:
        if not self.include_graph_projection or not isinstance(graph_json, str) or not graph_json.strip():
            return ""
        try:
            payload = json.loads(graph_json)
        except json.JSONDecodeError:
            return graph_json[: self.graph_projection_max_chars]
        if not isinstance(payload, dict):
            return str(payload)[: self.graph_projection_max_chars]
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])
        targets = payload.get("targets", {})
        parts = ["<graph>"]
        if isinstance(nodes, list):
            for node in nodes[:96]:
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("id", ""))[:48]
                node_type = str(node.get("type", ""))[:48]
                node_text = str(node.get("text", node.get("label", node.get("payload", ""))))[:220]
                parts.append(f"node id={node_id} type={node_type} text={node_text}")
        if isinstance(edges, list):
            for edge in edges[:160]:
                if not isinstance(edge, dict):
                    continue
                src = str(edge.get("source", edge.get("src", "")))[:48]
                dst = str(edge.get("target", edge.get("dst", "")))[:48]
                edge_type = str(edge.get("type", edge.get("label", "")))[:48]
                parts.append(f"edge {src}->{dst} type={edge_type}")
        if isinstance(targets, dict) and targets:
            parts.append("targets " + json.dumps(targets, ensure_ascii=False, sort_keys=True)[:512])
        elif isinstance(targets, list) and targets:
            parts.append("targets " + json.dumps(targets[:16], ensure_ascii=False)[:512])
        return "\n".join(parts)[: self.graph_projection_max_chars]

    def _row_text(self, row: dict[str, Any]) -> str:
        primary = row.get("text")
        graph_text = self._graph_projection(row.get("graph_json"))
        if isinstance(primary, str) and primary.strip():
            return f"{primary}\n\n{graph_text}" if graph_text else primary
        parts = []
        for column in ("question", "reasoning", "solution", "answer"):
            value = row.get(column)
            if isinstance(value, str) and value.strip():
                parts.append(value)
        if graph_text:
            parts.append(graph_text)
        return "\n\n".join(parts)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        files = [path for index, path in enumerate(self.files) if index % num_workers == worker_id]
        rng = random.Random(self.seed + worker_id)
        counter = worker_id * 10**12
        epoch = 0
        while True:
            if self.shuffle_files:
                rng.shuffle(files)
            for file_path in files:
                parquet = pq.ParquetFile(file_path)
                available = [name for name in TEXT_COLUMNS if name in parquet.schema.names]
                for batch in parquet.iter_batches(batch_size=self.rows_per_batch, columns=available):
                    table = batch.to_pydict()
                    rows = [dict(zip(table, values)) for values in zip(*table.values())]
                    buffer: list[int] = []
                    for row in rows:
                        text = self._row_text(row)
                        if not text:
                            continue
                        buffer.extend(byte_encode(text, byte_offset=self.byte_offset))
                        while len(buffer) >= self.seq_len:
                            chunk = buffer[: self.seq_len]
                            del buffer[: self.seq_len]
                            yield {
                                "tokens": torch.tensor(chunk, dtype=torch.long),
                                "sample_id": torch.tensor(counter, dtype=torch.long),
                            }
                            counter += 1
            epoch += 1
            if not self.repeat:
                break
            counter += 10**9 + epoch


class SyntheticByteChunkDataset(IterableDataset):
    """Fallback dataset for command and CUDA validation checks."""

    def __init__(self, seq_len: int, vocab_size: int, seed: int = 17) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + worker_id)
        counter = worker_id * 10**12
        while True:
            yield {
                "tokens": torch.randint(4, self.vocab_size, (self.seq_len,), generator=generator, dtype=torch.long),
                "sample_id": torch.tensor(counter, dtype=torch.long),
            }
            counter += 1


def collate_chunks(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.stack([item["tokens"] for item in items], dim=0),
        "sample_ids": torch.stack([item["sample_id"] for item in items], dim=0),
    }


def build_loader(
    parquet_glob: str,
    batch_size: int,
    seq_len: int,
    byte_offset: int,
    rows_per_batch: int,
    seed: int,
    workers: int,
    repeat: bool,
    synthetic: bool,
    vocab_size: int,
    include_graph_projection: bool,
    graph_projection_max_chars: int,
) -> DataLoader:
    if synthetic or not glob.glob(parquet_glob):
        dataset = SyntheticByteChunkDataset(seq_len=seq_len, vocab_size=vocab_size, seed=seed)
    else:
        dataset = ParquetByteChunkDataset(
            parquet_glob=parquet_glob,
            seq_len=seq_len,
            byte_offset=byte_offset,
            rows_per_batch=rows_per_batch,
            seed=seed,
            repeat=repeat,
            shuffle_files=repeat,
            include_graph_projection=include_graph_projection,
            graph_projection_max_chars=graph_projection_max_chars,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_chunks,
    )


def parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def cosine_lr(step: int, base_lr: float, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    path: Path,
    model: DenseRandomOrderToricLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: RandomOrderLMConfig,
    args: argparse.Namespace,
    metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "random_order_dense_lm",
            "step": step,
            "config": asdict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics or {},
        },
        path,
    )


@torch.no_grad()
def evaluate(
    model: DenseRandomOrderToricLM,
    loader: DataLoader,
    device: torch.device,
    batches: int,
    precision: str,
    pass_id: int,
    order_samples: int,
    gflownet_samples: int,
) -> dict[str, float]:
    model.eval()
    losses = []
    iterator = iter(loader)
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = device.type == "cuda" and precision in {"bf16", "fp16"}
    for index in range(batches):
        batch = next(iterator)
        tokens = batch["tokens"].to(device, non_blocking=True)
        sample_ids = batch["sample_ids"].to(device, non_blocking=True)
        sample_losses = []
        for sample in range(max(1, order_samples)):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(
                    tokens,
                    sample_ids=sample_ids,
                    pass_id=pass_id + index * 997 + sample,
                    sample_gflownet=gflownet_samples > 1,
                    gflownet_samples=max(1, gflownet_samples),
                )
            sample_losses.append(out["loss"].float())
        losses.append(torch.stack(sample_losses).mean())
    loss = torch.stack(losses).mean().item()
    return {"loss": loss, "bpb": loss / math.log(2)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--train-parquet-glob")
    parser.add_argument("--val-parquet-glob")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--resume")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--recurrent-passes", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--eval-order-samples", type=int)
    parser.add_argument("--eval-gflownet-samples", type=int)
    parser.add_argument("--ckpt-interval", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--rows-per-batch", type=int)
    parser.add_argument("--include-graph-projection", action="store_true")
    parser.add_argument("--no-graph-projection", action="store_true")
    parser.add_argument("--graph-projection-max-chars", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--attention", choices=["softmax", "tropical", "tropical_ring", "hybrid"])
    parser.add_argument("--ring-block-size", type=int)
    parser.add_argument("--polarquant-kv-bits", type=int)
    parser.add_argument("--polarquant-train", action="store_true")
    parser.add_argument("--use-gflownet-policy", action="store_true")
    parser.add_argument("--no-gflownet-policy", action="store_true")
    parser.add_argument("--gflownet-num-actions", type=int)
    parser.add_argument("--gflownet-hidden-dim", type=int)
    parser.add_argument("--gflownet-action-scale", type=float)
    parser.add_argument("--gflownet-loss-weight", type=float)
    parser.add_argument("--gflownet-entropy-weight", type=float)
    parser.add_argument("--export-bits", type=int, choices=[4, 6, 8])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_config = read_yaml(args.config)

    seed = args.seed if args.seed is not None else config_get(file_config, "training", "seed", 17)
    torch.manual_seed(seed)
    random.seed(seed)

    model_config = RandomOrderLMConfig(
        vocab_size=config_get(file_config, "model", "vocab_size", 260),
        max_seq_len=args.seq_len if args.seq_len is not None else config_get(file_config, "model", "max_seq_len", 1024),
        d_model=args.d_model if args.d_model is not None else config_get(file_config, "model", "d_model", 384),
        num_heads=args.num_heads if args.num_heads is not None else config_get(file_config, "model", "num_heads", 6),
        num_layers=args.num_layers if args.num_layers is not None else config_get(file_config, "model", "num_layers", 7),
        recurrent_passes=args.recurrent_passes
        if args.recurrent_passes is not None
        else config_get(file_config, "model", "recurrent_passes", 2),
        ffn_multiplier=config_get(file_config, "model", "ffn_multiplier", 4),
        dropout=config_get(file_config, "model", "dropout", 0.1),
        attention=args.attention if args.attention is not None else config_get(file_config, "model", "attention", "hybrid"),
        ring_block_size=args.ring_block_size
        if args.ring_block_size is not None
        else config_get(file_config, "model", "ring_block_size", 256),
        seed=seed,
        polarquant_kv_bits=args.polarquant_kv_bits
        if args.polarquant_kv_bits is not None
        else config_get(file_config, "model", "polarquant_kv_bits", 8),
        polarquant_train=bool(args.polarquant_train or config_get(file_config, "model", "polarquant_train", False)),
        use_gflownet_policy=(
            False
            if args.no_gflownet_policy
            else bool(args.use_gflownet_policy or config_get(file_config, "model", "use_gflownet_policy", True))
        ),
        gflownet_num_actions=args.gflownet_num_actions
        if args.gflownet_num_actions is not None
        else config_get(file_config, "model", "gflownet_num_actions", 8),
        gflownet_hidden_dim=args.gflownet_hidden_dim
        if args.gflownet_hidden_dim is not None
        else config_get(file_config, "model", "gflownet_hidden_dim", 128),
        gflownet_action_scale=args.gflownet_action_scale
        if args.gflownet_action_scale is not None
        else config_get(file_config, "model", "gflownet_action_scale", 0.05),
        target_artifact_bytes=config_get(file_config, "model", "target_artifact_bytes", 15_600_000),
        byte_offset=config_get(file_config, "model", "byte_offset", 4),
        use_soft_moe=False,
    )

    device_name = args.device or config_get(file_config, "training", "device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    precision = args.precision or config_get(file_config, "training", "precision", "bf16")
    steps = args.steps if args.steps is not None else config_get(file_config, "training", "max_steps", 50_000)
    batch_size = args.batch_size if args.batch_size is not None else config_get(file_config, "training", "batch_size", 4)
    grad_accum = (
        args.grad_accum_steps
        if args.grad_accum_steps is not None
        else config_get(file_config, "training", "grad_accum_steps", 8)
    )
    lr = args.lr if args.lr is not None else config_get(file_config, "training", "lr", 3e-4)
    weight_decay = (
        args.weight_decay if args.weight_decay is not None else config_get(file_config, "training", "weight_decay", 0.05)
    )
    warmup_steps = (
        args.warmup_steps if args.warmup_steps is not None else config_get(file_config, "training", "warmup_steps", 1_000)
    )
    log_interval = (
        args.log_interval if args.log_interval is not None else config_get(file_config, "training", "log_interval", 10)
    )
    eval_interval = (
        args.eval_interval if args.eval_interval is not None else config_get(file_config, "training", "eval_interval", 500)
    )
    eval_batches = (
        args.eval_batches if args.eval_batches is not None else config_get(file_config, "training", "eval_batches", 20)
    )
    eval_order_samples = (
        args.eval_order_samples
        if args.eval_order_samples is not None
        else config_get(file_config, "training", "eval_order_samples", 2)
    )
    eval_gflownet_samples = (
        args.eval_gflownet_samples
        if args.eval_gflownet_samples is not None
        else config_get(file_config, "training", "eval_gflownet_samples", 2)
    )
    gflownet_loss_weight = (
        args.gflownet_loss_weight
        if args.gflownet_loss_weight is not None
        else config_get(file_config, "training", "gflownet_loss_weight", 0.01)
    )
    gflownet_entropy_weight = (
        args.gflownet_entropy_weight
        if args.gflownet_entropy_weight is not None
        else config_get(file_config, "training", "gflownet_entropy_weight", 0.001)
    )
    ckpt_interval = (
        args.ckpt_interval if args.ckpt_interval is not None else config_get(file_config, "training", "ckpt_interval", 1_000)
    )
    workers = args.num_workers if args.num_workers is not None else config_get(file_config, "data", "num_workers", 2)
    rows_per_batch = (
        args.rows_per_batch if args.rows_per_batch is not None else config_get(file_config, "data", "rows_per_batch", 128)
    )
    include_graph_projection = (
        False
        if args.no_graph_projection
        else bool(args.include_graph_projection or config_get(file_config, "data", "include_graph_projection", True))
    )
    graph_projection_max_chars = (
        args.graph_projection_max_chars
        if args.graph_projection_max_chars is not None
        else config_get(file_config, "data", "graph_projection_max_chars", 4096)
    )
    train_glob = args.train_parquet_glob or config_get(
        file_config, "data", "train_parquet_glob", "data/curated_hf_shards/train/*.parquet"
    )
    val_glob = args.val_parquet_glob or config_get(
        file_config, "data", "val_parquet_glob", "data/curated_hf_shards/validation/*.parquet"
    )
    checkpoint_dir = Path(args.checkpoint_dir or config_get(file_config, "training", "checkpoint_dir", "checkpoints/parameter_golf_random_order"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = DenseRandomOrderToricLM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_step = 0
    best_val = float("inf")
    if args.resume:
        payload = torch.load(args.resume, map_location=device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0))
        best_val = float(payload.get("metrics", {}).get("val_bpb", best_val))

    params = parameter_count(model)
    export_bits = args.export_bits or config_get(file_config, "export", "bits", 8)
    estimated_tensor_bytes = estimate_uncompressed_quantized_bytes(model, bits=export_bits)
    report = write_artifact(
        model,
        checkpoint_dir / "initial_parameter_golf_artifact.zip",
        config=model.config_dict(),
        bits=export_bits,
    )
    if report.bytes_total > PARAMETER_GOLF_BYTE_LIMIT:
        raise RuntimeError(f"initial artifact exceeds Parameter-Golf cap: {report.bytes_total} bytes")

    train_loader = build_loader(
        parquet_glob=train_glob,
        batch_size=batch_size,
        seq_len=model_config.max_seq_len,
        byte_offset=model_config.byte_offset,
        rows_per_batch=rows_per_batch,
        seed=seed,
        workers=workers,
        repeat=True,
        synthetic=args.synthetic,
        vocab_size=model_config.vocab_size,
        include_graph_projection=include_graph_projection,
        graph_projection_max_chars=graph_projection_max_chars,
    )
    val_loader = build_loader(
        parquet_glob=val_glob,
        batch_size=batch_size,
        seq_len=model_config.max_seq_len,
        byte_offset=model_config.byte_offset,
        rows_per_batch=rows_per_batch,
        seed=seed + 10_000,
        workers=max(0, min(workers, 2)),
        repeat=True,
        synthetic=args.synthetic,
        vocab_size=model_config.vocab_size,
        include_graph_projection=include_graph_projection,
        graph_projection_max_chars=graph_projection_max_chars,
    )

    use_wandb = bool(args.wandb or config_get(file_config, "logging", "wandb", False)) and not args.no_wandb
    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project or config_get(file_config, "logging", "project", "toricgt-parameter-golf"),
            name=args.wandb_run_name or config_get(file_config, "logging", "run_name", None),
            config={
                "model": model.config_dict(),
                "training": {
                    "batch_size": batch_size,
                    "grad_accum_steps": grad_accum,
                    "max_steps": steps,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "warmup_steps": warmup_steps,
                    "eval_order_samples": eval_order_samples,
                    "eval_gflownet_samples": eval_gflownet_samples,
                    "gflownet_loss_weight": gflownet_loss_weight,
                    "gflownet_entropy_weight": gflownet_entropy_weight,
                },
                "data": {
                    "include_graph_projection": include_graph_projection,
                    "graph_projection_max_chars": graph_projection_max_chars,
                },
                "artifact": {
                    "initial_bytes": report.bytes_total,
                    "estimated_tensor_bytes": estimated_tensor_bytes,
                    "limit": PARAMETER_GOLF_BYTE_LIMIT,
                    "bits": export_bits,
                },
            },
            tags=["parameter-golf", "random-order-ar", "dense", "tropical-ring", "toricgt"],
        )

    print(json.dumps(
        {
            "model_type": "random_order_dense_lm",
            "parameters": params,
            "artifact_bytes": report.bytes_total,
            "estimated_tensor_bytes": estimated_tensor_bytes,
            "artifact_limit": PARAMETER_GOLF_BYTE_LIMIT,
            "device": str(device),
            "precision": precision,
            "train_glob": train_glob,
            "val_glob": val_glob,
            "checkpoint_dir": str(checkpoint_dir),
            "include_graph_projection": include_graph_projection,
            "graph_projection_max_chars": graph_projection_max_chars,
        },
        indent=2,
    ))

    iterator = iter(train_loader)
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = device.type == "cuda" and precision in {"bf16", "fp16"}
    progress = tqdm(range(start_step + 1, steps + 1), initial=start_step, total=steps, desc="parameter-golf-random-order")
    running_loss = 0.0
    for step in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_total_loss = 0.0
        step_gflownet_loss = 0.0
        step_gflownet_entropy = 0.0
        step_gflownet_diversity = 0.0
        lr_step = cosine_lr(step, lr, warmup_steps, steps)
        for group in optimizer.param_groups:
            group["lr"] = lr_step
        for accum_idx in range(grad_accum):
            batch = next(iterator)
            tokens = batch["tokens"].to(device, non_blocking=True)
            sample_ids = batch["sample_ids"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(
                    tokens,
                    sample_ids=sample_ids,
                    pass_id=step * grad_accum + accum_idx,
                    sample_gflownet=model_config.use_gflownet_policy and gflownet_loss_weight > 0,
                    gflownet_samples=1,
                )
                micro_loss = out["loss"]
                gflownet_loss = out.get("gflownet_loss", torch.zeros((), device=device))
                gflownet_entropy = out.get("gflownet_entropy", torch.zeros((), device=device))
                total_micro_loss = micro_loss + gflownet_loss_weight * gflownet_loss - gflownet_entropy_weight * gflownet_entropy
                loss = total_micro_loss / grad_accum
            loss.backward()
            step_loss += float(micro_loss.detach().cpu())
            step_total_loss += float(total_micro_loss.detach().cpu())
            step_gflownet_loss += float(gflownet_loss.detach().cpu())
            step_gflownet_entropy += float(gflownet_entropy.detach().cpu())
            step_gflownet_diversity += float(out.get("gflownet_action_diversity", torch.zeros(())).detach().cpu())
        step_loss /= grad_accum
        step_total_loss /= grad_accum
        step_gflownet_loss /= grad_accum
        step_gflownet_entropy /= grad_accum
        step_gflownet_diversity /= grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss = 0.97 * running_loss + 0.03 * step_loss if running_loss else step_loss
        bpb = step_loss / math.log(2)
        progress.set_postfix(loss=f"{step_loss:.4f}", bpb=f"{bpb:.3f}", gfn=f"{step_gflownet_loss:.3f}", lr=f"{lr_step:.2e}")

        if step % log_interval == 0:
            metrics = {
                "train/loss": step_loss,
                "train/total_loss": step_total_loss,
                "train/loss_ema": running_loss,
                "train/bpb": bpb,
                "train/lr": lr_step,
                "train/grad_norm": float(grad_norm.detach().cpu()),
                "train/gflownet_loss": step_gflownet_loss,
                "train/gflownet_entropy": step_gflownet_entropy,
                "train/gflownet_action_diversity": step_gflownet_diversity,
                "train/gflownet_loss_weight": gflownet_loss_weight,
                "train/gflownet_entropy_weight": gflownet_entropy_weight,
                "artifact/initial_bytes": report.bytes_total,
                "artifact/estimated_tensor_bytes": estimated_tensor_bytes,
                "model/parameters": params,
            }
            if device.type == "cuda":
                metrics["system/vram_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
                metrics["system/vram_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)

        if step % eval_interval == 0 or step == steps:
            val = evaluate(
                model,
                val_loader,
                device=device,
                batches=eval_batches,
                precision=precision,
                pass_id=step * 100_000,
                order_samples=eval_order_samples,
                gflownet_samples=eval_gflownet_samples,
            )
            metrics = {
                "val/loss": val["loss"],
                "val/bpb": val["bpb"],
                "val/order_samples": eval_order_samples,
                "val/gflownet_samples": eval_gflownet_samples,
            }
            print(json.dumps({"step": step, **metrics}, indent=2))
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            if val["bpb"] < best_val:
                best_val = val["bpb"]
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=model_config,
                    args=args,
                    metrics={"val_bpb": best_val, "val_loss": val["loss"]},
                )
            model.train()

        if step % ckpt_interval == 0 or step == steps:
            save_checkpoint(
                checkpoint_dir / f"random_order_step_{step:08d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                config=model_config,
                args=args,
                metrics={"train_loss": step_loss, "train_bpb": bpb, "best_val_bpb": best_val},
            )

    final_artifact = write_artifact(
        model,
        checkpoint_dir / "final_parameter_golf_artifact.zip",
        config=model.config_dict(),
        bits=export_bits,
    )
    print(json.dumps(asdict(final_artifact), indent=2))
    if wandb_run is not None:
        wandb_run.log(
            {
                "artifact/final_bytes": final_artifact.bytes_total,
                "artifact/within_limit": float(final_artifact.within_limit),
                "val/best_bpb": best_val,
            },
            step=steps,
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
