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
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm.auto import tqdm

from toricgt.complexity import random_order_complexity_metrics
from toricgt.parameter_golf_export import PARAMETER_GOLF_BYTE_LIMIT, write_artifact
from toricgt.random_order_lm import (
    DenseRandomOrderToricLM,
    RandomOrderLMConfig,
    byte_encode,
    estimate_uncompressed_quantized_bytes,
)


TEXT_COLUMNS = (
    "text",
    "question",
    "reasoning",
    "solution",
    "answer",
    "graph_json",
    "dataset",
    "task_family",
    "language",
    "role",
    "metadata_json",
    "quality_flags_json",
)


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


def load_optional_training_tokens(path: str | Path = "keys.txt") -> None:
    """Load local auth tokens without printing or checkpointing them.

    The repository ignores ``keys.txt``.  This helper only populates standard
    environment variables when they are absent, which lets tmux-launched runs
    publish best checkpoints and log to W&B without placing secrets in config
    files, logs, or checkpoints.
    """

    key_path = Path(path)
    if not key_path.exists():
        return
    aliases = {
        "hf_token": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"),
        "huggingface_token": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"),
        "hugging_face": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"),
        "hugging_face_token": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"),
        "wandb": ("WANDB_API_KEY",),
        "wandb_token": ("WANDB_API_KEY",),
        "wandb_api_key": ("WANDB_API_KEY",),
    }
    try:
        lines = key_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            raw_key, raw_value = line.split("=", 1)
        elif ":" in line:
            raw_key, raw_value = line.split(":", 1)
        else:
            continue
        key = raw_key.strip().lower().replace("-", "_").replace(" ", "_")
        value = raw_value.strip().strip('"').strip("'")
        if not value:
            continue
        for env_name in aliases.get(key, ()):
            os.environ.setdefault(env_name, value)


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
        coprime_row_stride: bool = True,
        document_separator: str = "\n\n",
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
        self.coprime_row_stride = coprime_row_stride
        self.document_separator = document_separator
        self._separator_bytes = byte_encode(document_separator, byte_offset=byte_offset) if document_separator else []

    def _stride_rows(self, rows: list[dict[str, Any]], epoch: int, worker_id: int) -> list[dict[str, Any]]:
        if not self.coprime_row_stride or len(rows) <= 2:
            return rows
        size = len(rows)
        stride = (self.seed + 2 * epoch + 2_003 * worker_id + 7_919) % size
        stride = max(1, stride)
        while math.gcd(stride, size) != 1:
            stride = (stride + 1) % size or 1
        start = (self.seed * 1_315_423_911 + epoch * 1_000_003 + worker_id * 65_537) % size
        return [rows[(start + index * stride) % size] for index in range(size)]

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
        domain_prefix = self._domain_prefix(row)
        primary = row.get("text")
        graph_text = self._graph_projection(row.get("graph_json"))
        if isinstance(primary, str) and primary.strip():
            body = f"{primary}\n\n{graph_text}" if graph_text else primary
            return f"{domain_prefix}\n{body}" if domain_prefix else body
        parts = []
        for column in ("question", "reasoning", "solution", "answer"):
            value = row.get(column)
            if isinstance(value, str) and value.strip():
                parts.append(value)
        if graph_text:
            parts.append(graph_text)
        if domain_prefix:
            parts.insert(0, domain_prefix)
        return "\n\n".join(parts)

    def _domain_prefix(self, row: dict[str, Any]) -> str:
        fields = []
        for key in ("dataset", "task_family", "language", "role"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                fields.append(value.lower())
        for key in ("metadata_json", "quality_flags_json"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                fields.append(value[:512].lower())
        haystack = " ".join(fields)
        tags = []
        rules = [
            ("<domain:math>", ("math", "algebra", "geometry", "proof", "theorem", "gsm8k", "hendrycks")),
            ("<domain:code>", ("code", "program", "python", "algorithm", "codeforces")),
            ("<domain:graph>", ("graph", "walk", "dag", "node", "edge", "got", "tree_of_thought")),
            ("<domain:hebrew>", ("hebrew", "rabbinic", "sefaria", "unimorph", "morphology")),
            ("<domain:biomed>", ("health", "medical", "medicine", "biomed", "clinical", "doctor")),
            ("<domain:biochem>", ("protein", "enzyme", "ligand", "molecule", "drug", "chem", "structure")),
            ("<domain:physics>", ("physics", "biophysics", "spatial", "mechanics", "cfd", "fluid")),
            ("<domain:toric>", ("toric", "tropical", "coxeter", "braid", "noncommutative")),
        ]
        for tag, needles in rules:
            if any(needle in haystack for needle in needles):
                tags.append(tag)
        return " ".join(dict.fromkeys(tags))

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
                    rows = self._stride_rows(rows, epoch=epoch, worker_id=worker_id)
                    buffer: list[int] = []
                    for row in rows:
                        text = self._row_text(row)
                        if not text:
                            continue
                        if buffer and self._separator_bytes:
                            buffer.extend(self._separator_bytes)
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
    coprime_row_stride: bool,
    document_separator: str,
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
            coprime_row_stride=coprime_row_stride,
            document_separator=document_separator,
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


def quantization_grid_loss(named_params: list[tuple[str, torch.nn.Parameter]], bits: int) -> torch.Tensor:
    """Lightweight QAT pull toward the current symmetric quantization grid."""

    if not named_params:
        raise ValueError("named_params must be non-empty when QAT is enabled")
    qmax = 2 ** (bits - 1) - 1
    losses = []
    for _, param in named_params:
        scale = param.detach().float().abs().amax().clamp_min(1e-8) / qmax
        target = torch.round(param.detach().float() / scale).clamp(-qmax, qmax) * scale
        losses.append(F.mse_loss(param.float(), target))
    return torch.stack(losses).mean()


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


def read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def publish_quality_score(
    val_bpb: float,
    metrics: dict[str, float],
    complexity_metric: str,
    complexity_weight: float,
) -> tuple[float, float | None]:
    """Return lower-is-better checkpoint score including a complexity proxy."""

    complexity_value = metrics.get(complexity_metric)
    if complexity_value is None and complexity_metric != "complexity/val/prediction_target_ncd_lzma_mean":
        complexity_value = metrics.get("complexity/val/prediction_target_ncd_lzma_mean")
    if complexity_value is None:
        complexity_value = metrics.get("complexity/val/target_cond_k_lzma_mean")
    score = float(val_bpb)
    if complexity_value is not None and math.isfinite(float(complexity_value)):
        score += float(complexity_weight) * float(complexity_value)
    return score, None if complexity_value is None else float(complexity_value)


def checkpoint_publish_decision(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    min_score_delta: float = 0.0,
    max_complexity_regression: float = 0.10,
) -> bool:
    """Decide whether the current checkpoint is better than the published one."""

    if previous is None:
        return True
    current_score = current.get("quality_score")
    previous_score = previous.get("quality_score")
    if current_score is None:
        return False
    if previous_score is None:
        return True
    if float(current_score) >= float(previous_score) - float(min_score_delta):
        return False
    current_k = current.get("complexity_value")
    previous_k = previous.get("complexity_value")
    if current_k is not None and previous_k is not None:
        allowed = float(previous_k) * (1.0 + max(0.0, float(max_complexity_regression)))
        if float(current_k) > allowed:
            previous_bpb = previous.get("val_bpb", float("inf"))
            current_bpb = current.get("val_bpb", float("inf"))
            if float(current_bpb) >= float(previous_bpb):
                return False
    return True


def better_publish_state(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any] | None:
    if left is None:
        return right
    if right is None:
        return left
    left_score = left.get("quality_score")
    right_score = right.get("quality_score")
    if left_score is None:
        return right
    if right_score is None:
        return left
    return left if float(left_score) <= float(right_score) else right


def download_remote_publish_manifest(
    repo_id: str,
    repo_type: str,
    manifest_filename: str,
    cache_dir: Path,
) -> dict[str, Any] | None:
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=manifest_filename,
            local_dir=cache_dir,
            force_download=True,
        )
    except Exception:
        return None
    return read_json_file(Path(path))


def publish_best_checkpoint_to_hf(
    checkpoint_path: Path,
    step: int,
    metrics: dict[str, float],
    repo_id: str,
    repo_type: str,
    checkpoint_filename: str,
    manifest_filename: str,
    state_path: Path,
    complexity_metric: str,
    complexity_weight: float,
    min_score_delta: float,
    max_complexity_regression: float,
    private: bool,
    wandb_url: str | None,
) -> dict[str, float]:
    """Upload the best checkpoint to HF only when the quality score improves."""

    val_bpb = float(metrics["val/bpb"])
    score, complexity_value = publish_quality_score(
        val_bpb=val_bpb,
        metrics=metrics,
        complexity_metric=complexity_metric,
        complexity_weight=complexity_weight,
    )
    state_path = state_path.expanduser()
    local_previous = read_json_file(state_path)
    remote_previous = download_remote_publish_manifest(
        repo_id=repo_id,
        repo_type=repo_type,
        manifest_filename=manifest_filename,
        cache_dir=state_path.parent / ".hf_manifest_cache",
    )
    previous = better_publish_state(local_previous, remote_previous)
    manifest = {
        "model_type": "random_order_dense_lm",
        "step": int(step),
        "val_bpb": val_bpb,
        "val_loss": float(metrics["val/loss"]),
        "quality_score": float(score),
        "complexity_metric": complexity_metric,
        "complexity_value": complexity_value,
        "complexity_weight": float(complexity_weight),
        "checkpoint_filename": checkpoint_filename,
        "wandb_url": wandb_url,
        "published_at_unix": int(time.time()),
    }
    publish_metrics = {
        "hf_publish/score": float(score),
        "hf_publish/val_bpb": val_bpb,
        "hf_publish/published": 0.0,
    }
    if complexity_value is not None:
        publish_metrics["hf_publish/complexity_value"] = float(complexity_value)
    if previous and previous.get("quality_score") is not None:
        publish_metrics["hf_publish/previous_score"] = float(previous["quality_score"])
    if not checkpoint_publish_decision(
        manifest,
        previous,
        min_score_delta=min_score_delta,
        max_complexity_regression=max_complexity_regression,
    ):
        write_json_atomic(state_path, previous or manifest)
        publish_metrics["hf_publish/skipped_not_better"] = 1.0
        return publish_metrics

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
        api.upload_file(
            repo_id=repo_id,
            repo_type=repo_type,
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=checkpoint_filename,
            commit_message=f"Update ToricGT best Parameter-Golf checkpoint at step {step}",
        )
        manifest_path = state_path.parent / manifest_filename
        write_json_atomic(manifest_path, manifest)
        api.upload_file(
            repo_id=repo_id,
            repo_type=repo_type,
            path_or_fileobj=str(manifest_path),
            path_in_repo=manifest_filename,
            commit_message=f"Update ToricGT best checkpoint manifest at step {step}",
        )
        write_json_atomic(state_path, manifest)
        publish_metrics["hf_publish/published"] = 1.0
    except Exception as exc:
        publish_metrics["hf_publish/error"] = 1.0
        error_path = state_path.with_suffix(".error.txt")
        try:
            error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        except OSError:
            pass
    return publish_metrics


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
    score_first_bias_lr: float,
    score_first_bias_decay: float,
    score_first_bias_clip: float,
) -> dict[str, float]:
    model.eval()
    losses = []
    bias_norms = []
    iterator = iter(loader)
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = device.type == "cuda" and precision in {"bf16", "fp16"}
    for index in range(batches):
        batch = next(iterator)
        tokens = batch["tokens"].to(device, non_blocking=True)
        sample_ids = batch["sample_ids"].to(device, non_blocking=True)
        sample_losses = []
        for sample in range(max(1, order_samples)):
            if score_first_bias_lr > 0:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model.score_with_bias_adaptation(
                        tokens,
                        sample_ids=sample_ids,
                        pass_id=pass_id + index * 997 + sample,
                        lr=score_first_bias_lr,
                        decay=score_first_bias_decay,
                        clip=score_first_bias_clip,
                        gflownet_samples=max(1, gflownet_samples),
                    )
                bias_norms.append(out["bias_norm"].float())
            else:
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
    metrics = {"loss": loss, "bpb": loss / math.log(2)}
    if bias_norms:
        metrics["bias_norm"] = torch.stack(bias_norms).mean().item()
    return metrics


@torch.no_grad()
def causal_audit(
    model: DenseRandomOrderToricLM,
    tokens: torch.Tensor,
    max_len: int = 32,
) -> float:
    was_training = model.training
    model.eval()
    clipped = tokens[:1, : min(max_len, tokens.shape[1])].detach()
    error = float(model.causal_future_permutation_error(clipped).detach().cpu())
    if was_training:
        model.train()
    return error


@torch.no_grad()
def compute_batch_complexity_metrics(
    model: DenseRandomOrderToricLM,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    precision: str,
    pass_id: int,
    max_samples: int,
    compressors: tuple[str, ...],
    prefix: str,
) -> dict[str, float]:
    """Compute small-sample Kolmogorov-style diagnostics for one byte batch."""

    if max_samples <= 0:
        return {}
    was_training = model.training
    model.eval()
    tokens = batch["tokens"][:max_samples].to(device, non_blocking=True)
    sample_ids = batch["sample_ids"][:max_samples].to(device, non_blocking=True)
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = device.type == "cuda" and precision in {"bf16", "fp16"}
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        out = model(
            tokens,
            sample_ids=sample_ids,
            pass_id=pass_id,
            sample_gflownet=model.config.use_gflownet_policy,
            return_order=True,
        )
    metrics = random_order_complexity_metrics(
        tokens=tokens,
        previous_tokens=out["previous_tokens"],
        target_tokens=out["target_tokens"],
        permutation=out["permutation"],
        logits=out["logits"],
        action_ids=out.get("gflownet_action_ids"),
        byte_offset=model.config.byte_offset,
        prefix=prefix,
        compressors=compressors,
        max_samples=max_samples,
    )
    metrics[f"{prefix}/loss"] = float(out["loss"].detach().cpu())
    metrics[f"{prefix}/bpb"] = float(out["bpb"].detach().cpu())
    if was_training:
        model.train()
    return metrics


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
    parser.add_argument("--coprime-row-stride", action="store_true")
    parser.add_argument("--no-coprime-row-stride", action="store_true")
    parser.add_argument("--document-separator")
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
    parser.add_argument("--use-bigram-hash", action="store_true")
    parser.add_argument("--no-bigram-hash", action="store_true")
    parser.add_argument("--bigram-hash-buckets", type=int)
    parser.add_argument("--bigram-hash-weight", type=float)
    parser.add_argument("--use-caseops-features", action="store_true")
    parser.add_argument("--no-caseops-features", action="store_true")
    parser.add_argument("--caseops-weight", type=float)
    parser.add_argument("--use-smear-gate", action="store_true")
    parser.add_argument("--no-smear-gate", action="store_true")
    parser.add_argument("--smear-temperature-min", type=float)
    parser.add_argument("--smear-temperature-max", type=float)
    parser.add_argument("--aux-mtp-offsets", type=int)
    parser.add_argument("--mtp-loss-weight", type=float)
    parser.add_argument("--qat-loss-weight", type=float)
    parser.add_argument("--qat-bits", type=int, choices=[4, 6, 8])
    parser.add_argument("--qat-max-tensors", type=int)
    parser.add_argument("--contrastive-loss-weight", type=float)
    parser.add_argument("--trajectory-flow-loss-weight", type=float)
    parser.add_argument("--eval-score-first-bias-lr", type=float)
    parser.add_argument("--eval-score-first-bias-decay", type=float)
    parser.add_argument("--eval-score-first-bias-clip", type=float)
    parser.add_argument("--causal-audit-interval", type=int)
    parser.add_argument("--complexity", action="store_true")
    parser.add_argument("--no-complexity", action="store_true")
    parser.add_argument("--complexity-eval-every", type=int)
    parser.add_argument("--complexity-eval-samples", type=int)
    parser.add_argument("--complexity-compressors", nargs="+")
    parser.add_argument("--hf-publish-best", action="store_true")
    parser.add_argument("--no-hf-publish-best", action="store_true")
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--hf-repo-type", default=None)
    parser.add_argument("--hf-checkpoint-filename")
    parser.add_argument("--hf-manifest-filename")
    parser.add_argument("--hf-publish-state")
    parser.add_argument("--hf-publish-complexity-metric")
    parser.add_argument("--hf-publish-complexity-weight", type=float)
    parser.add_argument("--hf-publish-min-score-delta", type=float)
    parser.add_argument("--hf-publish-max-complexity-regression", type=float)
    parser.add_argument("--hf-publish-private", action="store_true")
    parser.add_argument("--export-bits", type=int, choices=[4, 6, 8])
    parser.add_argument("--quantization-mode", choices=["tensor", "row"])
    parser.add_argument("--artifact-compression", choices=["deflated", "bzip2", "lzma"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_config = read_yaml(args.config)
    load_optional_training_tokens()

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
        use_bigram_hash=(
            False
            if args.no_bigram_hash
            else bool(args.use_bigram_hash or config_get(file_config, "model", "use_bigram_hash", True))
        ),
        bigram_hash_buckets=args.bigram_hash_buckets
        if args.bigram_hash_buckets is not None
        else config_get(file_config, "model", "bigram_hash_buckets", 4096),
        bigram_hash_weight=args.bigram_hash_weight
        if args.bigram_hash_weight is not None
        else config_get(file_config, "model", "bigram_hash_weight", 0.35),
        use_caseops_features=(
            False
            if args.no_caseops_features
            else bool(args.use_caseops_features or config_get(file_config, "model", "use_caseops_features", True))
        ),
        caseops_weight=args.caseops_weight
        if args.caseops_weight is not None
        else config_get(file_config, "model", "caseops_weight", 0.35),
        use_smear_gate=(
            False
            if args.no_smear_gate
            else bool(args.use_smear_gate or config_get(file_config, "model", "use_smear_gate", True))
        ),
        smear_temperature_min=args.smear_temperature_min
        if args.smear_temperature_min is not None
        else config_get(file_config, "model", "smear_temperature_min", 0.55),
        smear_temperature_max=args.smear_temperature_max
        if args.smear_temperature_max is not None
        else config_get(file_config, "model", "smear_temperature_max", 1.75),
        use_toric_memory=config_get(file_config, "model", "use_toric_memory", True),
        toric_memory_slots=config_get(file_config, "model", "toric_memory_slots", 32),
        toric_memory_weight=config_get(file_config, "model", "toric_memory_weight", 0.08),
        contrastive_temperature=config_get(file_config, "model", "contrastive_temperature", 0.2),
        trajectory_flow_viscosity=config_get(file_config, "model", "trajectory_flow_viscosity", 0.05),
        aux_mtp_offsets=args.aux_mtp_offsets
        if args.aux_mtp_offsets is not None
        else config_get(file_config, "model", "aux_mtp_offsets", 2),
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
    mtp_loss_weight = (
        args.mtp_loss_weight if args.mtp_loss_weight is not None else config_get(file_config, "training", "mtp_loss_weight", 0.05)
    )
    qat_loss_weight = (
        args.qat_loss_weight if args.qat_loss_weight is not None else config_get(file_config, "training", "qat_loss_weight", 1e-6)
    )
    qat_bits = args.qat_bits if args.qat_bits is not None else config_get(file_config, "training", "qat_bits", 6)
    qat_max_tensors = (
        args.qat_max_tensors if args.qat_max_tensors is not None else config_get(file_config, "training", "qat_max_tensors", 16)
    )
    contrastive_loss_weight = (
        args.contrastive_loss_weight
        if args.contrastive_loss_weight is not None
        else config_get(file_config, "training", "contrastive_loss_weight", 0.01)
    )
    trajectory_flow_loss_weight = (
        args.trajectory_flow_loss_weight
        if args.trajectory_flow_loss_weight is not None
        else config_get(file_config, "training", "trajectory_flow_loss_weight", 0.002)
    )
    eval_score_first_bias_lr = (
        args.eval_score_first_bias_lr
        if args.eval_score_first_bias_lr is not None
        else config_get(file_config, "training", "eval_score_first_bias_lr", 0.025)
    )
    eval_score_first_bias_decay = (
        args.eval_score_first_bias_decay
        if args.eval_score_first_bias_decay is not None
        else config_get(file_config, "training", "eval_score_first_bias_decay", 0.98)
    )
    eval_score_first_bias_clip = (
        args.eval_score_first_bias_clip
        if args.eval_score_first_bias_clip is not None
        else config_get(file_config, "training", "eval_score_first_bias_clip", 3.0)
    )
    causal_audit_interval = (
        args.causal_audit_interval
        if args.causal_audit_interval is not None
        else config_get(file_config, "training", "causal_audit_interval", 1000)
    )
    complexity_enabled = (
        False
        if args.no_complexity
        else bool(args.complexity or config_get(file_config, "complexity", "enabled", True))
    )
    complexity_eval_every = (
        args.complexity_eval_every
        if args.complexity_eval_every is not None
        else config_get(file_config, "complexity", "eval_every", 50)
    )
    complexity_eval_samples = (
        args.complexity_eval_samples
        if args.complexity_eval_samples is not None
        else config_get(file_config, "complexity", "eval_samples", 2)
    )
    complexity_compressors = tuple(
        args.complexity_compressors
        if args.complexity_compressors is not None
        else config_get(file_config, "complexity", "compressors", ["zlib", "lzma"])
    )
    publish_best_to_hf = (
        False
        if args.no_hf_publish_best
        else bool(args.hf_publish_best or config_get(file_config, "checkpoint_publishing", "enabled", False))
    )
    hf_repo_id = args.hf_repo_id or config_get(
        file_config, "checkpoint_publishing", "repo_id", "AmelieSchreiber/toricgt-checkpoints"
    )
    hf_repo_type = args.hf_repo_type or config_get(file_config, "checkpoint_publishing", "repo_type", "model")
    hf_checkpoint_filename = args.hf_checkpoint_filename or config_get(
        file_config, "checkpoint_publishing", "checkpoint_filename", "parameter_golf_oai_best.pt"
    )
    hf_manifest_filename = args.hf_manifest_filename or config_get(
        file_config, "checkpoint_publishing", "manifest_filename", "parameter_golf_oai_best.json"
    )
    hf_publish_state = Path(
        args.hf_publish_state
        or config_get(
            file_config,
            "checkpoint_publishing",
            "state_path",
            "checkpoints/parameter_golf_oai_dense/hf_best_publish_state.json",
        )
    )
    hf_complexity_metric = args.hf_publish_complexity_metric or config_get(
        file_config,
        "checkpoint_publishing",
        "complexity_metric",
        "complexity/val/prediction_target_ncd_lzma_mean",
    )
    hf_complexity_weight = (
        args.hf_publish_complexity_weight
        if args.hf_publish_complexity_weight is not None
        else config_get(file_config, "checkpoint_publishing", "complexity_weight", 0.05)
    )
    hf_min_score_delta = (
        args.hf_publish_min_score_delta
        if args.hf_publish_min_score_delta is not None
        else config_get(file_config, "checkpoint_publishing", "min_score_delta", 0.0)
    )
    hf_max_complexity_regression = (
        args.hf_publish_max_complexity_regression
        if args.hf_publish_max_complexity_regression is not None
        else config_get(file_config, "checkpoint_publishing", "max_complexity_regression", 0.10)
    )
    hf_private = bool(
        args.hf_publish_private or config_get(file_config, "checkpoint_publishing", "private", False)
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
    coprime_row_stride = (
        False
        if args.no_coprime_row_stride
        else bool(args.coprime_row_stride or config_get(file_config, "data", "coprime_row_stride", True))
    )
    document_separator = (
        args.document_separator if args.document_separator is not None else config_get(file_config, "data", "document_separator", "\n\n")
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
        resume_metrics = payload.get("metrics", {})
        best_val = float(resume_metrics.get("val_bpb", resume_metrics.get("best_val_bpb", best_val)))

    params = parameter_count(model)
    export_bits = args.export_bits or config_get(file_config, "export", "bits", 8)
    quantization_mode = args.quantization_mode or config_get(file_config, "export", "quantization_mode", "row")
    artifact_compression = args.artifact_compression or config_get(file_config, "export", "compression", "lzma")
    estimated_tensor_bytes = estimate_uncompressed_quantized_bytes(model, bits=export_bits)
    report = write_artifact(
        model,
        checkpoint_dir / "initial_parameter_golf_artifact.zip",
        config=model.config_dict(),
        bits=export_bits,
        quantization_mode=quantization_mode,
        compression=artifact_compression,
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
        coprime_row_stride=coprime_row_stride,
        document_separator=document_separator,
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
        coprime_row_stride=coprime_row_stride,
        document_separator=document_separator,
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
                    "mtp_loss_weight": mtp_loss_weight,
                    "qat_loss_weight": qat_loss_weight,
                    "qat_bits": qat_bits,
                    "qat_max_tensors": qat_max_tensors,
                    "contrastive_loss_weight": contrastive_loss_weight,
                    "trajectory_flow_loss_weight": trajectory_flow_loss_weight,
                    "eval_score_first_bias_lr": eval_score_first_bias_lr,
                    "eval_score_first_bias_decay": eval_score_first_bias_decay,
                    "eval_score_first_bias_clip": eval_score_first_bias_clip,
                    "causal_audit_interval": causal_audit_interval,
                },
                "complexity": {
                    "enabled": complexity_enabled,
                    "eval_every": complexity_eval_every,
                    "eval_samples": complexity_eval_samples,
                    "compressors": list(complexity_compressors),
                    "training_regularizer_weight": 0.0,
                },
                "checkpoint_publishing": {
                    "enabled": publish_best_to_hf,
                    "repo_id": hf_repo_id,
                    "repo_type": hf_repo_type,
                    "checkpoint_filename": hf_checkpoint_filename,
                    "manifest_filename": hf_manifest_filename,
                    "complexity_metric": hf_complexity_metric,
                    "complexity_weight": hf_complexity_weight,
                    "min_score_delta": hf_min_score_delta,
                    "max_complexity_regression": hf_max_complexity_regression,
                },
                "data": {
                    "include_graph_projection": include_graph_projection,
                    "graph_projection_max_chars": graph_projection_max_chars,
                    "coprime_row_stride": coprime_row_stride,
                    "document_separator": document_separator,
                },
                "artifact": {
                    "initial_bytes": report.bytes_total,
                    "estimated_tensor_bytes": estimated_tensor_bytes,
                    "limit": PARAMETER_GOLF_BYTE_LIMIT,
                    "bits": export_bits,
                    "quantization_mode": quantization_mode,
                    "compression": artifact_compression,
                    "deployment_parameters": report.deployment_parameters,
                    "excluded_tensors": report.excluded_tensors,
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
            "coprime_row_stride": coprime_row_stride,
            "document_separator": document_separator,
            "quantization_mode": quantization_mode,
            "artifact_compression": artifact_compression,
            "deployment_parameters": report.deployment_parameters,
            "excluded_tensors": report.excluded_tensors,
            "complexity_enabled": complexity_enabled,
            "complexity_eval_every": complexity_eval_every,
            "complexity_eval_samples": complexity_eval_samples,
            "complexity_compressors": list(complexity_compressors),
            "hf_publish_best": publish_best_to_hf,
            "hf_repo_id": hf_repo_id,
            "hf_checkpoint_filename": hf_checkpoint_filename,
            "hf_manifest_filename": hf_manifest_filename,
            "hf_publish_complexity_metric": hf_complexity_metric,
        },
        indent=2,
    ))

    iterator = iter(train_loader)
    qat_named_params = [
        item
        for item in sorted(model.named_parameters(), key=lambda pair: pair[1].numel(), reverse=True)
        if item[1].requires_grad and not item[0].startswith("aux_") and item[1].ndim >= 2
    ][: max(0, int(qat_max_tensors))]
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
        step_mtp_loss = 0.0
        step_qat_loss = 0.0
        step_contrastive_loss = 0.0
        step_trajectory_flow_loss = 0.0
        step_trajectory_kinetic = 0.0
        step_trajectory_viscous = 0.0
        step_smear_temperature = 0.0
        step_toric_memory_entropy = 0.0
        last_train_batch: dict[str, torch.Tensor] | None = None
        lr_step = cosine_lr(step, lr, warmup_steps, steps)
        for group in optimizer.param_groups:
            group["lr"] = lr_step
        for accum_idx in range(grad_accum):
            batch = next(iterator)
            tokens = batch["tokens"].to(device, non_blocking=True)
            sample_ids = batch["sample_ids"].to(device, non_blocking=True)
            last_train_batch = {"tokens": tokens.detach(), "sample_ids": sample_ids.detach()}
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
                mtp_loss = out.get("mtp_loss", torch.zeros((), device=device))
                qat_loss = (
                    quantization_grid_loss(qat_named_params, bits=qat_bits)
                    if qat_loss_weight > 0 and qat_named_params
                    else torch.zeros((), device=device)
                )
                contrastive_loss = out.get("contrastive_loss", torch.zeros((), device=device))
                trajectory_flow_loss = out.get("trajectory_flow_loss", torch.zeros((), device=device))
                total_micro_loss = (
                    micro_loss
                    + gflownet_loss_weight * gflownet_loss
                    - gflownet_entropy_weight * gflownet_entropy
                    + mtp_loss_weight * mtp_loss
                    + qat_loss_weight * qat_loss
                    + contrastive_loss_weight * contrastive_loss
                    + trajectory_flow_loss_weight * trajectory_flow_loss
                )
                loss = total_micro_loss / grad_accum
            loss.backward()
            step_loss += float(micro_loss.detach().cpu())
            step_total_loss += float(total_micro_loss.detach().cpu())
            step_gflownet_loss += float(gflownet_loss.detach().cpu())
            step_gflownet_entropy += float(gflownet_entropy.detach().cpu())
            step_gflownet_diversity += float(out.get("gflownet_action_diversity", torch.zeros(())).detach().cpu())
            step_mtp_loss += float(mtp_loss.detach().cpu())
            step_qat_loss += float(qat_loss.detach().cpu())
            step_contrastive_loss += float(contrastive_loss.detach().cpu())
            step_trajectory_flow_loss += float(trajectory_flow_loss.detach().cpu())
            step_trajectory_kinetic += float(out.get("trajectory_kinetic_energy", torch.zeros(())).detach().cpu())
            step_trajectory_viscous += float(out.get("trajectory_viscous_dissipation", torch.zeros(())).detach().cpu())
            step_smear_temperature += float(out.get("smear_temperature", torch.zeros(())).detach().cpu())
            step_toric_memory_entropy += float(out.get("toric_memory_entropy", torch.zeros(())).detach().cpu())
        step_loss /= grad_accum
        step_total_loss /= grad_accum
        step_gflownet_loss /= grad_accum
        step_gflownet_entropy /= grad_accum
        step_gflownet_diversity /= grad_accum
        step_mtp_loss /= grad_accum
        step_qat_loss /= grad_accum
        step_contrastive_loss /= grad_accum
        step_trajectory_flow_loss /= grad_accum
        step_trajectory_kinetic /= grad_accum
        step_trajectory_viscous /= grad_accum
        step_smear_temperature /= grad_accum
        step_toric_memory_entropy /= grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss = 0.97 * running_loss + 0.03 * step_loss if running_loss else step_loss
        bpb = step_loss / math.log(2)
        progress.set_postfix(loss=f"{step_loss:.4f}", bpb=f"{bpb:.3f}", gfn=f"{step_gflownet_loss:.3f}", lr=f"{lr_step:.2e}")
        audit_error = None
        if causal_audit_interval > 0 and step % causal_audit_interval == 0:
            audit_error = causal_audit(model, tokens)
        complexity_due = complexity_enabled and last_train_batch is not None and (
            step == start_step + 1 or (complexity_eval_every > 0 and step % complexity_eval_every == 0)
        )
        complexity_metrics: dict[str, float] = {}
        if complexity_due:
            complexity_metrics.update(
                compute_batch_complexity_metrics(
                    model=model,
                    batch=last_train_batch,
                    device=device,
                    precision=precision,
                    pass_id=step * 1_000_003,
                    max_samples=int(complexity_eval_samples),
                    compressors=complexity_compressors,
                    prefix="complexity/train",
                )
            )

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
                "train/mtp_loss": step_mtp_loss,
                "train/mtp_loss_weight": mtp_loss_weight,
                "train/qat_loss": step_qat_loss,
                "train/qat_loss_weight": qat_loss_weight,
                "train/qat_bits": qat_bits,
                "train/contrastive_loss": step_contrastive_loss,
                "train/contrastive_loss_weight": contrastive_loss_weight,
                "train/trajectory_flow_loss": step_trajectory_flow_loss,
                "train/trajectory_flow_loss_weight": trajectory_flow_loss_weight,
                "train/trajectory_kinetic_energy": step_trajectory_kinetic,
                "train/trajectory_viscous_dissipation": step_trajectory_viscous,
                "train/smear_temperature": step_smear_temperature,
                "train/toric_memory_entropy": step_toric_memory_entropy,
                "artifact/initial_bytes": report.bytes_total,
                "artifact/estimated_tensor_bytes": estimated_tensor_bytes,
                "artifact/deployment_parameters": report.deployment_parameters,
                "artifact/excluded_tensors": report.excluded_tensors,
                "model/parameters": params,
                "data/coprime_row_stride": float(coprime_row_stride),
                "eval/score_first_bias_lr": eval_score_first_bias_lr,
            }
            if audit_error is not None:
                metrics["audit/future_permutation_logit_error"] = audit_error
            metrics.update(complexity_metrics)
            if device.type == "cuda":
                metrics["system/vram_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
                metrics["system/vram_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
        elif complexity_metrics and wandb_run is not None:
            wandb_run.log(complexity_metrics, step=step)

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
                score_first_bias_lr=eval_score_first_bias_lr,
                score_first_bias_decay=eval_score_first_bias_decay,
                score_first_bias_clip=eval_score_first_bias_clip,
            )
            metrics = {
                "val/loss": val["loss"],
                "val/bpb": val["bpb"],
                "val/order_samples": eval_order_samples,
                "val/gflownet_samples": eval_gflownet_samples,
                "val/score_first_bias_lr": eval_score_first_bias_lr,
            }
            if "bias_norm" in val:
                metrics["val/score_first_bias_norm"] = val["bias_norm"]
            if complexity_enabled and complexity_eval_samples > 0:
                try:
                    val_batch = next(iter(val_loader))
                    metrics.update(
                        compute_batch_complexity_metrics(
                            model=model,
                            batch=val_batch,
                            device=device,
                            precision=precision,
                            pass_id=step * 1_000_003 + 17,
                            max_samples=int(complexity_eval_samples),
                            compressors=complexity_compressors,
                            prefix="complexity/val",
                        )
                    )
                except StopIteration:
                    pass
            print(json.dumps({"step": step, **metrics}, indent=2))
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            if val["bpb"] < best_val:
                best_val = val["bpb"]
                best_checkpoint_path = checkpoint_dir / "best.pt"
                checkpoint_metrics = {"val_bpb": best_val, "val_loss": val["loss"], **metrics}
                save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=model_config,
                    args=args,
                    metrics=checkpoint_metrics,
                )
                if publish_best_to_hf:
                    wandb_url = None
                    if wandb_run is not None:
                        try:
                            wandb_url = wandb_run.get_url()
                        except Exception:
                            wandb_url = None
                    publish_metrics = publish_best_checkpoint_to_hf(
                        checkpoint_path=best_checkpoint_path,
                        step=step,
                        metrics=metrics,
                        repo_id=hf_repo_id,
                        repo_type=hf_repo_type,
                        checkpoint_filename=hf_checkpoint_filename,
                        manifest_filename=hf_manifest_filename,
                        state_path=hf_publish_state,
                        complexity_metric=hf_complexity_metric,
                        complexity_weight=float(hf_complexity_weight),
                        min_score_delta=float(hf_min_score_delta),
                        max_complexity_regression=float(hf_max_complexity_regression),
                        private=hf_private,
                        wandb_url=wandb_url,
                    )
                    print(json.dumps({"step": step, **publish_metrics}, indent=2))
                    if wandb_run is not None:
                        wandb_run.log(publish_metrics, step=step)
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
        quantization_mode=quantization_mode,
        compression=artifact_compression,
    )
    print(json.dumps(asdict(final_artifact), indent=2))
    if wandb_run is not None:
        wandb_run.log(
            {
                "artifact/final_bytes": final_artifact.bytes_total,
                "artifact/within_limit": float(final_artifact.within_limit),
                "artifact/final_deployment_parameters": final_artifact.deployment_parameters,
                "artifact/final_excluded_tensors": final_artifact.excluded_tensors,
                "val/best_bpb": best_val,
            },
            step=steps,
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
