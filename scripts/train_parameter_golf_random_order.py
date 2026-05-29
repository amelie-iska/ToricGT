#!/usr/bin/env python3
"""Train the dense random-order ToricGT Parameter-Golf adapter."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
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

FILTER_COLUMNS = (
    "estimated_tokens",
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
        min_estimated_tokens: int = 0,
        max_estimated_tokens: int = 0,
        task_family_keywords: tuple[str, ...] = (),
        dataset_keywords: tuple[str, ...] = (),
        interleave_row_groups: bool = True,
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
        self.min_estimated_tokens = int(min_estimated_tokens or 0)
        self.max_estimated_tokens = int(max_estimated_tokens or 0)
        self.task_family_keywords = tuple(keyword.lower() for keyword in task_family_keywords if keyword)
        self.dataset_keywords = tuple(keyword.lower() for keyword in dataset_keywords if keyword)
        self.interleave_row_groups = bool(interleave_row_groups)
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

    def _row_allowed(self, row: dict[str, Any]) -> bool:
        if self.min_estimated_tokens or self.max_estimated_tokens:
            try:
                estimated = int(row.get("estimated_tokens") or 0)
            except (TypeError, ValueError):
                estimated = 0
            if self.min_estimated_tokens and estimated and estimated < self.min_estimated_tokens:
                return False
            if self.max_estimated_tokens and estimated and estimated > self.max_estimated_tokens:
                return False
        if self.task_family_keywords:
            task_family = str(row.get("task_family", "")).lower()
            if not any(keyword in task_family for keyword in self.task_family_keywords):
                return False
        if self.dataset_keywords:
            dataset = str(row.get("dataset", "")).lower()
            if not any(keyword in dataset for keyword in self.dataset_keywords):
                return False
        return True

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
            if self.interleave_row_groups:
                units: list[tuple[str, int | None]] = []
                for file_path in files:
                    try:
                        parquet = pq.ParquetFile(file_path)
                        units.extend((file_path, row_group) for row_group in range(parquet.num_row_groups))
                    except Exception:
                        units.append((file_path, None))
                if self.shuffle_files:
                    rng.shuffle(units)
            else:
                if self.shuffle_files:
                    rng.shuffle(files)
                units = [(file_path, None) for file_path in files]
            for file_path, row_group in units:
                parquet = pq.ParquetFile(file_path)
                available = [name for name in (*TEXT_COLUMNS, *FILTER_COLUMNS) if name in parquet.schema.names]
                row_groups = None if row_group is None else [row_group]
                for batch in parquet.iter_batches(batch_size=self.rows_per_batch, columns=available, row_groups=row_groups):
                    table = batch.to_pydict()
                    rows = [dict(zip(table, values)) for values in zip(*table.values())]
                    rows = self._stride_rows(rows, epoch=epoch, worker_id=worker_id)
                    buffer: list[int] = []
                    for row in rows:
                        if not self._row_allowed(row):
                            continue
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
    min_estimated_tokens: int = 0,
    max_estimated_tokens: int = 0,
    task_family_keywords: tuple[str, ...] = (),
    dataset_keywords: tuple[str, ...] = (),
    interleave_row_groups: bool = True,
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
            min_estimated_tokens=min_estimated_tokens,
            max_estimated_tokens=max_estimated_tokens,
            task_family_keywords=task_family_keywords,
            dataset_keywords=dataset_keywords,
            interleave_row_groups=interleave_row_groups,
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


def linear_ramp(step: int, start_step: int, ramp_steps: int) -> float:
    """Return a stable [0, 1] ramp for delayed regularizers."""

    if step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return max(0.0, min(1.0, float(step - start_step) / float(ramp_steps)))


def deterministic_ratio_choice(step: int, accum_idx: int, ratio: float) -> bool:
    """Deterministically mix ordinary and complex curricula."""

    if ratio <= 0:
        return False
    if ratio >= 1:
        return True
    value = (step * 1_103_515_245 + accum_idx * 12_345 + 97_531) & 0xFFFF_FFFF
    return (value / 0x1_0000_0000) < ratio


PHASE_CONTROL_KEYS = {
    "complex_mix_ratio",
    "gflownet_loss_weight",
    "gflownet_entropy_weight",
    "gflownet_entropy_target",
    "trajectory_flow_loss_weight",
    "contrastive_loss_weight",
    "mtp_loss_weight",
    "toric_entropy_loss_weight",
    "qat_loss_weight",
    "qat_start_step",
    "qat_warmup_steps",
    "lr_multiplier",
}


def active_phase_controls(config: dict[str, Any], step: int) -> tuple[dict[str, Any], str, int]:
    """Return scheduled training controls for an absolute step.

    The phase curriculum is intentionally a light wrapper over scalar controls:
    it does not change architecture or optimizer state, and it can be inspected
    directly in W&B through the active phase metrics.
    """

    section = config.get("phase_curriculum", {}) or {}
    if not bool(section.get("enabled", False)):
        return {}, "disabled", -1
    phases = section.get("phases", [])
    if not isinstance(phases, list):
        return {}, "malformed", -1
    fallback: tuple[dict[str, Any], str, int] = ({}, "none", -1)
    for index, raw_phase in enumerate(phases):
        if not isinstance(raw_phase, dict):
            continue
        start = int(raw_phase.get("start_step", 0) or 0)
        end_value = raw_phase.get("end_step")
        end = None if end_value is None else int(end_value)
        if int(step) < start:
            continue
        if end is not None and int(step) >= end:
            continue
        controls = {key: raw_phase[key] for key in PHASE_CONTROL_KEYS if key in raw_phase}
        name = str(raw_phase.get("name", f"phase_{index}"))
        return controls, name, index
    return fallback


def control_float(controls: dict[str, Any], key: str, default: float) -> float:
    value = controls.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def control_int(controls: dict[str, Any], key: str, default: int) -> int:
    value = controls.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def burn_in_training_iterators(
    iterator: Any,
    complex_iterator: Any | None,
    *,
    config: dict[str, Any],
    start_step: int,
    burnin_steps: int,
    grad_accum: int,
    complex_start_step: int,
    complex_mix_ratio: float,
) -> dict[str, float]:
    """Advance dataloaders to match an earlier stream origin.

    Rollback checkpoints can intentionally reuse a previous data-order stream
    while loading a later weight checkpoint.  This burn-in consumes the same
    ordinary/complex microbatch choices that the training loop would have seen
    between the stream origin and the loaded checkpoint step, without running
    forward or backward passes.
    """

    burnin_steps = max(0, int(burnin_steps))
    grad_accum = max(1, int(grad_accum))
    if burnin_steps <= 0:
        return {"steps": 0.0, "microbatches": 0.0, "complex_microbatches": 0.0}
    burn_start = max(1, int(start_step) - burnin_steps + 1)
    burn_end = int(start_step)
    complex_microbatches = 0
    total_microbatches = 0
    for burn_step in range(burn_start, burn_end + 1):
        phase_controls, _, _ = active_phase_controls(config, burn_step)
        effective_complex_mix_ratio = max(
            0.0,
            min(1.0, control_float(phase_controls, "complex_mix_ratio", complex_mix_ratio)),
        )
        for accum_idx in range(grad_accum):
            complex_active = (
                complex_iterator is not None
                and burn_step >= int(complex_start_step or 0)
                and deterministic_ratio_choice(burn_step, accum_idx, effective_complex_mix_ratio)
            )
            next(complex_iterator if complex_active else iterator)
            complex_microbatches += int(complex_active)
            total_microbatches += 1
    return {
        "steps": float(burnin_steps),
        "microbatches": float(total_microbatches),
        "complex_microbatches": float(complex_microbatches),
    }


@dataclass
class AdaptiveTrainingController:
    """Bounded checkpoint-level updates for fragile training-control scalars."""

    enabled: bool
    state_path: Path
    start_step: int
    initial_gflownet_loss_weight: float
    initial_entropy_target: float
    initial_complex_mix_ratio: float
    entropy_target_min: float = 1.95
    entropy_target_tau_steps: float = 8_000.0
    gflownet_loss_min: float = 0.008
    gflownet_loss_max: float = 0.025
    gflownet_gap_step: float = 0.001
    gflownet_relax_step: float = 0.00025
    complex_mix_min: float = 0.12
    complex_mix_max: float = 0.30
    complex_down_step: float = 0.02
    complex_up_step: float = 0.005
    train_bpb_ema_beta: float = 0.7
    train_drift_threshold: float = 0.02
    val_improvement_delta: float = 0.002
    warmup_updates: int = 1
    reset_on_start_step_mismatch: bool = True
    gflownet_increase_drift_guard: float = 0.03
    recovery_mode_enabled: bool = False
    recovery_train_drift_threshold: float = 0.08
    recovery_complex_down_step: float = 0.04
    recovery_gflownet_down_step: float = 0.001
    recovery_exit_val_improvements: int = 2

    def __post_init__(self) -> None:
        self.state_path = Path(self.state_path)
        self.update_count = 0
        self.train_bpb_ema: float | None = None
        self.previous_train_bpb_ema: float | None = None
        self.best_val_bpb = float("inf")
        self.gflownet_loss_weight = float(self.initial_gflownet_loss_weight)
        self.gflownet_entropy_target = float(self.initial_entropy_target)
        self.complex_mix_ratio = float(self.initial_complex_mix_ratio)
        self.consecutive_val_improvements = 0
        self.last_metrics: dict[str, float] = {}
        self.load()

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        checkpoint_dir: Path,
        start_step: int,
        gflownet_loss_weight: float,
        gflownet_entropy_target: float,
        complex_mix_ratio: float,
    ) -> "AdaptiveTrainingController":
        section = config.get("adaptive_training", {}) or {}
        default_state_path = checkpoint_dir / "adaptive_controller_state.json"
        return cls(
            enabled=bool(section.get("enabled", False)),
            state_path=Path(section.get("state_path", default_state_path)),
            start_step=int(start_step),
            initial_gflownet_loss_weight=float(gflownet_loss_weight),
            initial_entropy_target=float(gflownet_entropy_target),
            initial_complex_mix_ratio=float(complex_mix_ratio),
            entropy_target_min=float(section.get("entropy_target_min", 1.95)),
            entropy_target_tau_steps=float(section.get("entropy_target_tau_steps", 8_000.0)),
            gflownet_loss_min=float(section.get("gflownet_loss_min", 0.008)),
            gflownet_loss_max=float(section.get("gflownet_loss_max", 0.025)),
            gflownet_gap_step=float(section.get("gflownet_gap_step", 0.001)),
            gflownet_relax_step=float(section.get("gflownet_relax_step", 0.00025)),
            complex_mix_min=float(section.get("complex_mix_min", 0.12)),
            complex_mix_max=float(section.get("complex_mix_max", 0.30)),
            complex_down_step=float(section.get("complex_down_step", 0.02)),
            complex_up_step=float(section.get("complex_up_step", 0.005)),
            train_bpb_ema_beta=float(section.get("train_bpb_ema_beta", 0.7)),
            train_drift_threshold=float(section.get("train_drift_threshold", 0.02)),
            val_improvement_delta=float(section.get("val_improvement_delta", 0.002)),
            warmup_updates=int(section.get("warmup_updates", 1)),
            reset_on_start_step_mismatch=bool(section.get("reset_on_start_step_mismatch", True)),
            gflownet_increase_drift_guard=float(section.get("gflownet_increase_drift_guard", 0.03)),
            recovery_mode_enabled=bool(section.get("recovery_mode_enabled", False)),
            recovery_train_drift_threshold=float(section.get("recovery_train_drift_threshold", 0.08)),
            recovery_complex_down_step=float(section.get("recovery_complex_down_step", 0.04)),
            recovery_gflownet_down_step=float(section.get("recovery_gflownet_down_step", 0.001)),
            recovery_exit_val_improvements=int(section.get("recovery_exit_val_improvements", 2)),
        )

    def load(self) -> None:
        payload = read_json_file(self.state_path)
        if not payload:
            self._clamp()
            return
        if self.reset_on_start_step_mismatch and int(payload.get("start_step", self.start_step)) != int(self.start_step):
            print(
                json.dumps(
                    {
                        "adaptive_controller_notice": "state_reset_start_step_mismatch",
                        "state_path": str(self.state_path),
                        "state_start_step": payload.get("start_step"),
                        "resume_start_step": self.start_step,
                    }
                )
            )
            self._clamp()
            return
        self.update_count = int(payload.get("update_count", self.update_count))
        train_bpb_ema = payload.get("train_bpb_ema")
        previous_train_bpb_ema = payload.get("previous_train_bpb_ema")
        best_val_bpb = payload.get("best_val_bpb")
        if train_bpb_ema is not None:
            self.train_bpb_ema = float(train_bpb_ema)
        if previous_train_bpb_ema is not None:
            self.previous_train_bpb_ema = float(previous_train_bpb_ema)
        if best_val_bpb is not None:
            self.best_val_bpb = float(best_val_bpb)
        self.gflownet_loss_weight = float(payload.get("gflownet_loss_weight", self.gflownet_loss_weight))
        self.gflownet_entropy_target = float(payload.get("gflownet_entropy_target", self.gflownet_entropy_target))
        self.complex_mix_ratio = float(payload.get("complex_mix_ratio", self.complex_mix_ratio))
        self.consecutive_val_improvements = int(
            payload.get("consecutive_val_improvements", self.consecutive_val_improvements)
        )
        last_metrics = payload.get("last_metrics")
        if isinstance(last_metrics, dict):
            self.last_metrics = {str(key): float(value) for key, value in last_metrics.items()}
        self._clamp()

    def _clamp(self) -> None:
        self.gflownet_loss_weight = max(
            self.gflownet_loss_min,
            min(self.gflownet_loss_max, float(self.gflownet_loss_weight)),
        )
        self.gflownet_entropy_target = max(
            self.entropy_target_min,
            min(float(self.initial_entropy_target), float(self.gflownet_entropy_target)),
        )
        self.complex_mix_ratio = max(
            self.complex_mix_min,
            min(self.complex_mix_max, float(self.complex_mix_ratio)),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "update_count": self.update_count,
            "start_step": self.start_step,
            "train_bpb_ema": self.train_bpb_ema,
            "previous_train_bpb_ema": self.previous_train_bpb_ema,
            "best_val_bpb": self.best_val_bpb,
            "gflownet_loss_weight": self.gflownet_loss_weight,
            "gflownet_entropy_target": self.gflownet_entropy_target,
            "complex_mix_ratio": self.complex_mix_ratio,
            "consecutive_val_improvements": self.consecutive_val_improvements,
            "last_metrics": self.last_metrics,
            "bounds": {
                "gflownet_loss": [self.gflownet_loss_min, self.gflownet_loss_max],
                "entropy_target": [self.entropy_target_min, self.initial_entropy_target],
                "complex_mix": [self.complex_mix_min, self.complex_mix_max],
            },
        }

    def save(self) -> None:
        write_json_atomic(self.state_path, self.state_dict())

    def log_metrics(self) -> dict[str, float]:
        metrics = {
            "controller/enabled": float(self.enabled),
            "controller/update_count": float(self.update_count),
            "controller/gflownet_loss_weight": float(self.gflownet_loss_weight),
            "controller/gflownet_entropy_target": float(self.gflownet_entropy_target),
            "controller/complex_mix_ratio": float(self.complex_mix_ratio),
            "controller/recovery_mode_enabled": float(self.recovery_mode_enabled),
            "controller/consecutive_val_improvements": float(self.consecutive_val_improvements),
        }
        metrics.update(self.last_metrics)
        return metrics

    def update(
        self,
        *,
        step: int,
        train_bpb: float,
        val_bpb: float,
        val_gflownet_bpb: float,
    ) -> dict[str, float]:
        if not self.enabled:
            self.save()
            return self.log_metrics()

        previous_train_ema = self.train_bpb_ema
        if self.train_bpb_ema is None:
            self.train_bpb_ema = float(train_bpb)
        else:
            beta = max(0.0, min(0.99, float(self.train_bpb_ema_beta)))
            self.train_bpb_ema = beta * self.train_bpb_ema + (1.0 - beta) * float(train_bpb)
        self.previous_train_bpb_ema = previous_train_ema

        train_drift = 0.0 if previous_train_ema is None else self.train_bpb_ema - previous_train_ema
        previous_best = self.best_val_bpb
        val_improved = float(val_bpb) < previous_best - float(self.val_improvement_delta)
        self.best_val_bpb = min(self.best_val_bpb, float(val_bpb))
        gfn_eval_gap = float(val_gflownet_bpb) - float(val_bpb)
        self.consecutive_val_improvements = self.consecutive_val_improvements + 1 if val_improved else 0

        elapsed = max(0, int(step) - int(self.start_step))
        tau = max(1.0, float(self.entropy_target_tau_steps))
        scheduled_target = self.entropy_target_min + (
            float(self.initial_entropy_target) - self.entropy_target_min
        ) * math.exp(-elapsed / tau)
        self.gflownet_entropy_target = min(self.gflownet_entropy_target, scheduled_target)

        recovery_active = False
        if self.update_count >= int(self.warmup_updates):
            recovery_active = bool(
                self.recovery_mode_enabled
                and (
                    train_drift > float(self.recovery_train_drift_threshold)
                    or (train_drift > float(self.train_drift_threshold) and not val_improved)
                )
            )
            if recovery_active:
                self.complex_mix_ratio -= self.recovery_complex_down_step
                self.gflownet_loss_weight -= self.recovery_gflownet_down_step

            if (
                gfn_eval_gap > 0.0
                and train_drift <= float(self.gflownet_increase_drift_guard)
                and not recovery_active
            ):
                fraction = min(1.0, gfn_eval_gap / 0.02)
                self.gflownet_loss_weight += self.gflownet_gap_step * fraction
            elif gfn_eval_gap < -0.005:
                self.gflownet_loss_weight -= self.gflownet_relax_step

            if recovery_active:
                pass
            elif train_drift > float(self.train_drift_threshold) and not val_improved:
                self.complex_mix_ratio -= self.complex_down_step
            elif (
                val_improved
                and train_drift <= float(self.train_drift_threshold)
                and self.consecutive_val_improvements >= int(self.recovery_exit_val_improvements)
            ):
                self.complex_mix_ratio += self.complex_up_step

        self.update_count += 1
        self._clamp()
        self.last_metrics = {
            "controller/train_bpb_ema": float(self.train_bpb_ema),
            "controller/train_bpb_drift": float(train_drift),
            "controller/val_bpb": float(val_bpb),
            "controller/best_val_bpb": float(self.best_val_bpb),
            "controller/val_gflownet_gap": float(gfn_eval_gap),
            "controller/val_improved": float(val_improved),
            "controller/scheduled_entropy_target": float(scheduled_target),
            "controller/recovery_active": float(recovery_active),
        }
        self.save()
        return self.log_metrics()


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


def _flattened_vector_norm(vectors: list[torch.Tensor]) -> torch.Tensor:
    if not vectors:
        return torch.zeros(())
    total = None
    for vector in vectors:
        value = vector.float().pow(2).sum()
        total = value if total is None else total + value
    if total is None:
        return torch.zeros(())
    return total.sqrt()


def _normalize_parameter_vector(vectors: list[torch.Tensor], eps: float = 1e-12) -> list[torch.Tensor]:
    norm = _flattened_vector_norm(vectors).clamp_min(eps)
    return [vector / norm.to(device=vector.device, dtype=vector.dtype) for vector in vectors]


def _rademacher_like(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    vectors = []
    for param in parameters:
        sample = torch.randint(
            low=0,
            high=2,
            size=param.shape,
            device=param.device,
            dtype=torch.int8,
        )
        vectors.append(sample.to(dtype=param.dtype).mul_(2).sub_(1))
    return vectors


def select_hessian_probe_parameters(
    model: DenseRandomOrderToricLM,
    max_tensors: int,
    max_parameters: int,
) -> list[torch.nn.Parameter]:
    """Select a bounded high-impact parameter subset for Hessian probes.

    Full Hessians are intractable here and full-model HVPs are too expensive
    for an interruptible competition run.  The probe therefore targets the
    largest trainable matrices, which capture the sharpness of the main
    embedding/projection/backbone surfaces while keeping memory bounded.
    """

    candidates = [
        param
        for _, param in sorted(
            ((name, param) for name, param in model.named_parameters() if param.requires_grad and param.ndim >= 2),
            key=lambda item: item[1].numel(),
            reverse=True,
        )
    ]
    selected: list[torch.nn.Parameter] = []
    total = 0
    for param in candidates:
        if max_tensors > 0 and len(selected) >= max_tensors:
            break
        if max_parameters > 0 and total + param.numel() > max_parameters:
            if selected:
                break
            continue
        selected.append(param)
        total += param.numel()
        if max_parameters > 0 and total >= max_parameters:
            break
    if not selected and candidates:
        selected = [min(candidates, key=lambda param: param.numel())]
    return selected


def hessian_probe_metrics(
    model: DenseRandomOrderToricLM,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    precision: str,
    pass_id: int,
    *,
    max_tokens: int,
    trace_samples: int,
    power_iters: int,
    max_tensors: int,
    max_parameters: int,
    prefix: str = "hessian",
) -> dict[str, float]:
    """Estimate local Hessian sharpness with stochastic HVP probes.

    The reported values are diagnostics, not optimizer inputs.  `trace_per_param`
    is a Hutchinson estimate of average curvature on the probed parameter
    subspace.  `dominant_curvature` is a small power-iteration Rayleigh estimate
    and can be negative for nonconvex local geometry; its absolute value is the
    sharpness proxy used in the metric review.
    """

    if trace_samples <= 0 and power_iters <= 0:
        return {}
    parameters = select_hessian_probe_parameters(
        model,
        max_tensors=max(0, int(max_tensors)),
        max_parameters=max(0, int(max_parameters)),
    )
    if not parameters:
        return {}
    was_training = model.training
    model.eval()
    tokens = batch["tokens"][:1, : max(8, min(int(max_tokens), batch["tokens"].shape[1]))].to(device, non_blocking=True)
    sample_ids = batch["sample_ids"][:1].to(device, non_blocking=True)
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = device.type == "cuda" and precision in {"bf16", "fp16"}

    def loss_closure() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(
                tokens,
                sample_ids=sample_ids,
                pass_id=pass_id,
                sample_gflownet=False,
                gflownet_samples=1,
            )
            loss = out["loss"]
        return loss.float()

    def hvp(vectors: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        loss = loss_closure()
        grads = torch.autograd.grad(
            loss,
            parameters,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        grad_norm_terms = [grad.detach().float().pow(2).sum() for grad in grads if grad is not None]
        grad_norm = torch.stack(grad_norm_terms).sum().sqrt() if grad_norm_terms else torch.zeros((), device=device)
        dot = None
        for grad, vector in zip(grads, vectors):
            if grad is None:
                continue
            term = (grad.float() * vector.float()).sum()
            dot = term if dot is None else dot + term
        if dot is None:
            zeros = [torch.zeros_like(param) for param in parameters]
            return zeros, loss.detach(), grad_norm.detach()
        hessian_vectors = torch.autograd.grad(
            dot,
            parameters,
            retain_graph=False,
            allow_unused=True,
        )
        out = [
            hv.detach() if hv is not None else torch.zeros_like(param)
            for hv, param in zip(hessian_vectors, parameters)
        ]
        return out, loss.detach(), grad_norm.detach()

    metrics: dict[str, float] = {}
    parameter_count = int(sum(param.numel() for param in parameters))
    metrics[f"{prefix}/probed_parameter_count"] = float(parameter_count)
    metrics[f"{prefix}/probed_tensor_count"] = float(len(parameters))
    trace_values = []
    last_loss = None
    last_grad_norm = None
    for _ in range(max(0, int(trace_samples))):
        vector = _rademacher_like(parameters)
        hessian_vector, loss_value, grad_norm = hvp(vector)
        trace_values.append(
            float(
                sum((v.float() * hv.float()).sum().detach().cpu() for v, hv in zip(vector, hessian_vector))
            )
        )
        last_loss = loss_value
        last_grad_norm = grad_norm
    if trace_values:
        trace = float(sum(trace_values) / len(trace_values))
        metrics[f"{prefix}/trace_estimate"] = trace
        metrics[f"{prefix}/trace_per_param"] = trace / max(1, parameter_count)
        metrics[f"{prefix}/trace_abs_per_param"] = abs(trace) / max(1, parameter_count)

    if int(power_iters) > 0:
        vector = _normalize_parameter_vector(_rademacher_like(parameters))
        rayleigh = 0.0
        hv_norm = 0.0
        for _ in range(int(power_iters)):
            hessian_vector, loss_value, grad_norm = hvp(vector)
            rayleigh = float(
                sum((v.float() * hv.float()).sum().detach().cpu() for v, hv in zip(vector, hessian_vector))
            )
            hv_norm = float(_flattened_vector_norm(hessian_vector).detach().cpu())
            if hv_norm <= 1e-12:
                break
            vector = [hv / hv_norm for hv in hessian_vector]
            last_loss = loss_value
            last_grad_norm = grad_norm
        metrics[f"{prefix}/dominant_curvature"] = rayleigh
        metrics[f"{prefix}/dominant_abs_curvature"] = abs(rayleigh)
        metrics[f"{prefix}/hvp_norm"] = hv_norm
    if last_loss is not None:
        metrics[f"{prefix}/probe_loss"] = float(last_loss.detach().cpu())
    if last_grad_norm is not None:
        metrics[f"{prefix}/probe_grad_norm"] = float(last_grad_norm.detach().cpu())
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


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
    if path.exists() and path.name.startswith("random_order_step_"):
        archive_dir = path.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        archived_path = archive_dir / f"{path.stem}__archived_{timestamp}_{path.stat().st_mtime_ns}{path.suffix}"
        shutil.copy2(path, archived_path)
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


def resize_position_embedding_weight(old_weight: torch.Tensor, new_shape: torch.Size) -> torch.Tensor:
    """Resize learned position embeddings for longer packed-context resumes."""

    if old_weight.ndim != 2 or len(new_shape) != 2 or old_weight.shape[1] != new_shape[1]:
        raise ValueError("position embedding resize requires matching embedding dimensions")
    if old_weight.shape[0] == new_shape[0]:
        return old_weight
    source = old_weight.detach().float().transpose(0, 1).unsqueeze(0)
    resized = F.interpolate(source, size=int(new_shape[0]), mode="linear", align_corners=True)
    return resized.squeeze(0).transpose(0, 1).to(dtype=old_weight.dtype)


def load_state_dict_with_optional_position_resize(
    model: DenseRandomOrderToricLM,
    state_dict: dict[str, torch.Tensor],
    allow_position_resize: bool,
) -> bool:
    model_state = model.state_dict()
    key = "position_embedding.weight"
    resized = False
    if (
        allow_position_resize
        and key in state_dict
        and key in model_state
        and state_dict[key].shape != model_state[key].shape
    ):
        state_dict = dict(state_dict)
        state_dict[key] = resize_position_embedding_weight(state_dict[key], model_state[key].shape)
        resized = True
    model.load_state_dict(state_dict)
    return resized


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
    parser.add_argument("--reset-optimizer", action="store_true")
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
    parser.add_argument("--min-estimated-tokens", type=int)
    parser.add_argument("--max-estimated-tokens", type=int)
    parser.add_argument("--task-family-keywords", nargs="+")
    parser.add_argument("--dataset-keywords", nargs="+")
    parser.add_argument("--complex-start-step", type=int)
    parser.add_argument("--complex-min-estimated-tokens", type=int)
    parser.add_argument("--complex-max-estimated-tokens", type=int)
    parser.add_argument("--complex-task-family-keywords", nargs="+")
    parser.add_argument("--complex-dataset-keywords", nargs="+")
    parser.add_argument("--complex-mix-ratio", type=float)
    parser.add_argument("--stream-origin-step", type=int)
    parser.add_argument("--stream-burnin-steps", type=int)
    parser.add_argument("--no-stream-burnin", action="store_true")
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
    parser.add_argument("--gflownet-entropy-target", type=float)
    parser.add_argument("--toric-entropy-floor", type=float)
    parser.add_argument("--toric-entropy-loss-weight", type=float)
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
    parser.add_argument("--qat-start-step", type=int)
    parser.add_argument("--qat-warmup-steps", type=int)
    parser.add_argument("--contrastive-loss-weight", type=float)
    parser.add_argument("--trajectory-flow-loss-weight", type=float)
    parser.add_argument("--trajectory-flow-target", type=float)
    parser.add_argument("--eval-score-first-bias-lr", type=float)
    parser.add_argument("--eval-score-first-bias-decay", type=float)
    parser.add_argument("--eval-score-first-bias-clip", type=float)
    parser.add_argument("--causal-audit-interval", type=int)
    parser.add_argument("--resize-position-embedding", action="store_true")
    parser.add_argument("--no-resize-position-embedding", action="store_true")
    parser.add_argument("--complexity", action="store_true")
    parser.add_argument("--no-complexity", action="store_true")
    parser.add_argument("--complexity-eval-every", type=int)
    parser.add_argument("--complexity-eval-samples", type=int)
    parser.add_argument("--complexity-compressors", nargs="+")
    parser.add_argument("--hessian-probes", action="store_true")
    parser.add_argument("--no-hessian-probes", action="store_true")
    parser.add_argument("--hessian-eval-every", type=int)
    parser.add_argument("--hessian-max-tokens", type=int)
    parser.add_argument("--hessian-trace-samples", type=int)
    parser.add_argument("--hessian-power-iters", type=int)
    parser.add_argument("--hessian-max-tensors", type=int)
    parser.add_argument("--hessian-max-parameters", type=int)
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
    gflownet_entropy_target = (
        args.gflownet_entropy_target
        if args.gflownet_entropy_target is not None
        else config_get(file_config, "training", "gflownet_entropy_target", -1.0)
    )
    toric_entropy_floor = (
        args.toric_entropy_floor
        if args.toric_entropy_floor is not None
        else config_get(file_config, "training", "toric_entropy_floor", 0.0)
    )
    toric_entropy_loss_weight = (
        args.toric_entropy_loss_weight
        if args.toric_entropy_loss_weight is not None
        else config_get(file_config, "training", "toric_entropy_loss_weight", 0.0)
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
    qat_start_step = (
        args.qat_start_step if args.qat_start_step is not None else config_get(file_config, "training", "qat_start_step", 0)
    )
    qat_warmup_steps = (
        args.qat_warmup_steps
        if args.qat_warmup_steps is not None
        else config_get(file_config, "training", "qat_warmup_steps", 0)
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
    trajectory_flow_target = (
        args.trajectory_flow_target
        if args.trajectory_flow_target is not None
        else config_get(file_config, "training", "trajectory_flow_target", 0.0)
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
    resize_position_embedding = (
        False
        if args.no_resize_position_embedding
        else bool(args.resize_position_embedding or config_get(file_config, "training", "resize_position_embedding", True))
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
    hessian_enabled = (
        False
        if args.no_hessian_probes
        else bool(args.hessian_probes or config_get(file_config, "hessian", "enabled", False))
    )
    hessian_eval_every = (
        args.hessian_eval_every
        if args.hessian_eval_every is not None
        else config_get(file_config, "hessian", "eval_every", 250)
    )
    hessian_max_tokens = (
        args.hessian_max_tokens
        if args.hessian_max_tokens is not None
        else config_get(file_config, "hessian", "max_tokens", 128)
    )
    hessian_trace_samples = (
        args.hessian_trace_samples
        if args.hessian_trace_samples is not None
        else config_get(file_config, "hessian", "trace_samples", 1)
    )
    hessian_power_iters = (
        args.hessian_power_iters
        if args.hessian_power_iters is not None
        else config_get(file_config, "hessian", "power_iters", 1)
    )
    hessian_max_tensors = (
        args.hessian_max_tensors
        if args.hessian_max_tensors is not None
        else config_get(file_config, "hessian", "max_tensors", 4)
    )
    hessian_max_parameters = (
        args.hessian_max_parameters
        if args.hessian_max_parameters is not None
        else config_get(file_config, "hessian", "max_parameters", 1_500_000)
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
    interleave_row_groups = bool(config_get(file_config, "data", "interleave_row_groups", True))
    resume_aware_stream_seed = bool(config_get(file_config, "data", "resume_aware_stream_seed", True))
    configured_stream_origin_step = (
        args.stream_origin_step
        if args.stream_origin_step is not None
        else config_get(file_config, "data", "stream_origin_step", None)
    )
    configured_stream_burnin_steps = (
        args.stream_burnin_steps
        if args.stream_burnin_steps is not None
        else config_get(file_config, "data", "stream_burnin_steps", None)
    )
    document_separator = (
        args.document_separator if args.document_separator is not None else config_get(file_config, "data", "document_separator", "\n\n")
    )
    min_estimated_tokens = (
        args.min_estimated_tokens
        if args.min_estimated_tokens is not None
        else config_get(file_config, "data", "min_estimated_tokens", 0)
    )
    max_estimated_tokens = (
        args.max_estimated_tokens
        if args.max_estimated_tokens is not None
        else config_get(file_config, "data", "max_estimated_tokens", 0)
    )
    task_family_keywords = tuple(
        args.task_family_keywords
        if args.task_family_keywords is not None
        else config_get(file_config, "data", "task_family_keywords", [])
    )
    dataset_keywords = tuple(
        args.dataset_keywords
        if args.dataset_keywords is not None
        else config_get(file_config, "data", "dataset_keywords", [])
    )
    complex_start_step = (
        args.complex_start_step
        if args.complex_start_step is not None
        else config_get(file_config, "data", "complex_start_step", 0)
    )
    complex_min_estimated_tokens = (
        args.complex_min_estimated_tokens
        if args.complex_min_estimated_tokens is not None
        else config_get(file_config, "data", "complex_min_estimated_tokens", min_estimated_tokens)
    )
    complex_max_estimated_tokens = (
        args.complex_max_estimated_tokens
        if args.complex_max_estimated_tokens is not None
        else config_get(file_config, "data", "complex_max_estimated_tokens", max_estimated_tokens)
    )
    complex_task_family_keywords = tuple(
        args.complex_task_family_keywords
        if args.complex_task_family_keywords is not None
        else config_get(file_config, "data", "complex_task_family_keywords", task_family_keywords)
    )
    complex_dataset_keywords = tuple(
        args.complex_dataset_keywords
        if args.complex_dataset_keywords is not None
        else config_get(file_config, "data", "complex_dataset_keywords", dataset_keywords)
    )
    complex_include_graph_projection = bool(
        config_get(file_config, "data", "complex_include_graph_projection", include_graph_projection)
    )
    complex_mix_ratio = (
        args.complex_mix_ratio
        if args.complex_mix_ratio is not None
        else config_get(file_config, "data", "complex_mix_ratio", 1.0)
    )
    complex_mix_ratio = max(0.0, min(1.0, float(complex_mix_ratio)))
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
    optimizer_state_loaded = False
    if args.resume:
        payload = torch.load(args.resume, map_location=device)
        resized_position_embedding = load_state_dict_with_optional_position_resize(
            model,
            payload["model"],
            allow_position_resize=resize_position_embedding,
        )
        if resized_position_embedding:
            print(
                json.dumps(
                    {
                        "resume_notice": "position_embedding_resized_optimizer_state_reset",
                        "checkpoint": args.resume,
                        "new_max_seq_len": model_config.max_seq_len,
                    }
                )
            )
        elif args.reset_optimizer:
            print(
                json.dumps(
                    {
                        "resume_notice": "optimizer_state_reset_requested",
                        "checkpoint": args.resume,
                    }
                )
            )
        else:
            optimizer.load_state_dict(payload["optimizer"])
            optimizer_state_loaded = True
        start_step = int(payload.get("step", 0))
        resume_metrics = payload.get("metrics", {})
        best_val = float(resume_metrics.get("val_bpb", resume_metrics.get("best_val_bpb", best_val)))
    if resume_aware_stream_seed:
        stream_origin_step = int(
            configured_stream_origin_step
            if configured_stream_origin_step is not None
            else start_step
        )
        stream_seed_offset = stream_origin_step
    else:
        stream_origin_step = 0
        stream_seed_offset = 0
    if args.no_stream_burnin or not resume_aware_stream_seed:
        stream_burnin_steps = 0
    elif configured_stream_burnin_steps is not None:
        stream_burnin_steps = max(0, int(configured_stream_burnin_steps))
    else:
        stream_burnin_steps = max(0, int(start_step) - int(stream_origin_step))
    stream_burnin_microbatches = int(stream_burnin_steps) * int(max(1, grad_accum))

    adaptive_controller = AdaptiveTrainingController.from_config(
        file_config,
        checkpoint_dir=checkpoint_dir,
        start_step=start_step,
        gflownet_loss_weight=gflownet_loss_weight,
        gflownet_entropy_target=gflownet_entropy_target,
        complex_mix_ratio=complex_mix_ratio,
    )
    if adaptive_controller.enabled:
        gflownet_loss_weight = adaptive_controller.gflownet_loss_weight
        gflownet_entropy_target = adaptive_controller.gflownet_entropy_target
        complex_mix_ratio = adaptive_controller.complex_mix_ratio

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
        seed=seed + stream_seed_offset,
        workers=workers,
        repeat=True,
        synthetic=args.synthetic,
        vocab_size=model_config.vocab_size,
        include_graph_projection=include_graph_projection,
        graph_projection_max_chars=graph_projection_max_chars,
        coprime_row_stride=coprime_row_stride,
        document_separator=document_separator,
        min_estimated_tokens=int(min_estimated_tokens or 0),
        max_estimated_tokens=int(max_estimated_tokens or 0),
        task_family_keywords=task_family_keywords,
        dataset_keywords=dataset_keywords,
        interleave_row_groups=interleave_row_groups,
    )
    complex_train_loader = None
    if int(complex_start_step or 0) > 0:
        complex_train_loader = build_loader(
            parquet_glob=train_glob,
            batch_size=batch_size,
            seq_len=model_config.max_seq_len,
            byte_offset=model_config.byte_offset,
            rows_per_batch=rows_per_batch,
            seed=seed + 500_000 + stream_seed_offset,
            workers=workers,
            repeat=True,
            synthetic=args.synthetic,
            vocab_size=model_config.vocab_size,
            include_graph_projection=complex_include_graph_projection,
            graph_projection_max_chars=graph_projection_max_chars,
            coprime_row_stride=coprime_row_stride,
            document_separator=document_separator,
            min_estimated_tokens=int(complex_min_estimated_tokens or 0),
            max_estimated_tokens=int(complex_max_estimated_tokens or 0),
            task_family_keywords=complex_task_family_keywords,
            dataset_keywords=complex_dataset_keywords,
            interleave_row_groups=interleave_row_groups,
        )
    val_loader = build_loader(
        parquet_glob=val_glob,
        batch_size=batch_size,
        seq_len=model_config.max_seq_len,
        byte_offset=model_config.byte_offset,
        rows_per_batch=rows_per_batch,
        seed=seed + 10_000,
        workers=max(0, min(workers, 2)),
        repeat=False,
        synthetic=args.synthetic,
        vocab_size=model_config.vocab_size,
        include_graph_projection=include_graph_projection,
        graph_projection_max_chars=graph_projection_max_chars,
        coprime_row_stride=coprime_row_stride,
        document_separator=document_separator,
        min_estimated_tokens=int(min_estimated_tokens or 0),
        max_estimated_tokens=int(max_estimated_tokens or 0),
        task_family_keywords=task_family_keywords,
        dataset_keywords=dataset_keywords,
        interleave_row_groups=interleave_row_groups,
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
                    "gflownet_entropy_target": gflownet_entropy_target,
                    "toric_entropy_floor": toric_entropy_floor,
                    "toric_entropy_loss_weight": toric_entropy_loss_weight,
                    "mtp_loss_weight": mtp_loss_weight,
                    "qat_loss_weight": qat_loss_weight,
                    "qat_bits": qat_bits,
                    "qat_max_tensors": qat_max_tensors,
                    "qat_start_step": qat_start_step,
                    "qat_warmup_steps": qat_warmup_steps,
                    "contrastive_loss_weight": contrastive_loss_weight,
                    "trajectory_flow_loss_weight": trajectory_flow_loss_weight,
                    "trajectory_flow_target": trajectory_flow_target,
                    "eval_score_first_bias_lr": eval_score_first_bias_lr,
                    "eval_score_first_bias_decay": eval_score_first_bias_decay,
                    "eval_score_first_bias_clip": eval_score_first_bias_clip,
                    "causal_audit_interval": causal_audit_interval,
                    "resize_position_embedding": resize_position_embedding,
                },
                "complexity": {
                    "enabled": complexity_enabled,
                    "eval_every": complexity_eval_every,
                    "eval_samples": complexity_eval_samples,
                    "compressors": list(complexity_compressors),
                    "training_regularizer_weight": 0.0,
                },
                "hessian": {
                    "enabled": hessian_enabled,
                    "eval_every": hessian_eval_every,
                    "max_tokens": hessian_max_tokens,
                    "trace_samples": hessian_trace_samples,
                    "power_iters": hessian_power_iters,
                    "max_tensors": hessian_max_tensors,
                    "max_parameters": hessian_max_parameters,
                    "role": "diagnostic_only",
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
                    "interleave_row_groups": interleave_row_groups,
                    "resume_aware_stream_seed": resume_aware_stream_seed,
                    "stream_origin_step": stream_origin_step,
                    "stream_seed_offset": stream_seed_offset,
                    "stream_burnin_steps": stream_burnin_steps,
                    "stream_burnin_microbatches": stream_burnin_microbatches,
                    "document_separator": document_separator,
                    "min_estimated_tokens": min_estimated_tokens,
                    "max_estimated_tokens": max_estimated_tokens,
                    "task_family_keywords": list(task_family_keywords),
                    "dataset_keywords": list(dataset_keywords),
                    "complex_start_step": complex_start_step,
                    "complex_mix_ratio": complex_mix_ratio,
                    "complex_include_graph_projection": complex_include_graph_projection,
                    "complex_min_estimated_tokens": complex_min_estimated_tokens,
                    "complex_max_estimated_tokens": complex_max_estimated_tokens,
                    "complex_task_family_keywords": list(complex_task_family_keywords),
                    "complex_dataset_keywords": list(complex_dataset_keywords),
                },
                "adaptive_training": adaptive_controller.state_dict(),
                "phase_curriculum": file_config.get("phase_curriculum", {}),
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
            id=os.environ.get("WANDB_RUN_ID") or None,
            resume=os.environ.get("WANDB_RESUME") or None,
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
            "interleave_row_groups": interleave_row_groups,
            "resume_aware_stream_seed": resume_aware_stream_seed,
            "stream_origin_step": stream_origin_step,
            "stream_seed_offset": stream_seed_offset,
            "stream_burnin_steps": stream_burnin_steps,
            "stream_burnin_microbatches": stream_burnin_microbatches,
            "document_separator": document_separator,
            "min_estimated_tokens": min_estimated_tokens,
            "max_estimated_tokens": max_estimated_tokens,
            "task_family_keywords": list(task_family_keywords),
            "dataset_keywords": list(dataset_keywords),
            "complex_start_step": complex_start_step,
            "complex_include_graph_projection": complex_include_graph_projection,
            "complex_min_estimated_tokens": complex_min_estimated_tokens,
            "complex_max_estimated_tokens": complex_max_estimated_tokens,
            "complex_task_family_keywords": list(complex_task_family_keywords),
            "complex_dataset_keywords": list(complex_dataset_keywords),
            "quantization_mode": quantization_mode,
            "artifact_compression": artifact_compression,
            "deployment_parameters": report.deployment_parameters,
            "excluded_tensors": report.excluded_tensors,
            "complexity_enabled": complexity_enabled,
            "complexity_eval_every": complexity_eval_every,
            "complexity_eval_samples": complexity_eval_samples,
            "complexity_compressors": list(complexity_compressors),
            "hessian_enabled": hessian_enabled,
            "hessian_eval_every": hessian_eval_every,
            "hessian_max_tokens": hessian_max_tokens,
            "hessian_trace_samples": hessian_trace_samples,
            "hessian_power_iters": hessian_power_iters,
            "hessian_max_tensors": hessian_max_tensors,
            "hessian_max_parameters": hessian_max_parameters,
            "gflownet_entropy_target": gflownet_entropy_target,
            "toric_entropy_floor": toric_entropy_floor,
            "toric_entropy_loss_weight": toric_entropy_loss_weight,
            "trajectory_flow_target": trajectory_flow_target,
            "qat_start_step": qat_start_step,
            "qat_warmup_steps": qat_warmup_steps,
            "complex_mix_ratio": complex_mix_ratio,
            "adaptive_training": adaptive_controller.state_dict(),
            "phase_curriculum": file_config.get("phase_curriculum", {}),
            "resize_position_embedding": resize_position_embedding,
            "optimizer_state_loaded": optimizer_state_loaded,
            "hf_publish_best": publish_best_to_hf,
            "hf_repo_id": hf_repo_id,
            "hf_checkpoint_filename": hf_checkpoint_filename,
            "hf_manifest_filename": hf_manifest_filename,
            "hf_publish_complexity_metric": hf_complexity_metric,
        },
        indent=2,
    ))

    iterator = iter(train_loader)
    complex_iterator = iter(complex_train_loader) if complex_train_loader is not None else None
    burnin_result = burn_in_training_iterators(
        iterator,
        complex_iterator,
        config=file_config,
        start_step=start_step,
        burnin_steps=stream_burnin_steps,
        grad_accum=grad_accum,
        complex_start_step=int(complex_start_step or 0),
        complex_mix_ratio=complex_mix_ratio,
    )
    if burnin_result["microbatches"] > 0:
        print(
            json.dumps(
                {
                    "stream_burnin": {
                        "stream_origin_step": stream_origin_step,
                        "start_step": start_step,
                        "stream_seed_offset": stream_seed_offset,
                        "requested_steps": stream_burnin_steps,
                        **burnin_result,
                    }
                },
                indent=2,
            )
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "data/stream_origin_step": float(stream_origin_step),
                    "data/stream_burnin_steps": float(stream_burnin_steps),
                    "data/stream_burnin_microbatches": float(burnin_result["microbatches"]),
                    "data/stream_burnin_complex_microbatches": float(burnin_result["complex_microbatches"]),
                },
                step=start_step,
            )
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
        step_gflownet_entropy_objective = 0.0
        step_gflownet_diversity = 0.0
        step_mtp_loss = 0.0
        step_qat_loss = 0.0
        step_qat_weight = 0.0
        step_contrastive_loss = 0.0
        step_trajectory_flow_loss = 0.0
        step_trajectory_flow_penalty = 0.0
        step_trajectory_kinetic = 0.0
        step_trajectory_viscous = 0.0
        step_smear_temperature = 0.0
        step_toric_memory_entropy = 0.0
        step_toric_entropy_loss = 0.0
        step_complex_microbatches = 0.0
        last_train_batch: dict[str, torch.Tensor] | None = None
        phase_controls, phase_name, phase_index = active_phase_controls(file_config, step)
        effective_complex_mix_ratio = max(0.0, min(1.0, control_float(phase_controls, "complex_mix_ratio", complex_mix_ratio)))
        effective_gflownet_loss_weight = max(0.0, control_float(phase_controls, "gflownet_loss_weight", gflownet_loss_weight))
        effective_gflownet_entropy_weight = max(
            0.0,
            control_float(phase_controls, "gflownet_entropy_weight", gflownet_entropy_weight),
        )
        effective_gflownet_entropy_target = control_float(
            phase_controls,
            "gflownet_entropy_target",
            gflownet_entropy_target,
        )
        effective_mtp_loss_weight = max(0.0, control_float(phase_controls, "mtp_loss_weight", mtp_loss_weight))
        effective_contrastive_loss_weight = max(
            0.0,
            control_float(phase_controls, "contrastive_loss_weight", contrastive_loss_weight),
        )
        effective_trajectory_flow_loss_weight = max(
            0.0,
            control_float(phase_controls, "trajectory_flow_loss_weight", trajectory_flow_loss_weight),
        )
        effective_toric_entropy_loss_weight = max(
            0.0,
            control_float(phase_controls, "toric_entropy_loss_weight", toric_entropy_loss_weight),
        )
        effective_qat_base_weight = max(0.0, control_float(phase_controls, "qat_loss_weight", qat_loss_weight))
        effective_qat_start_step = control_int(phase_controls, "qat_start_step", int(qat_start_step or 0))
        effective_qat_warmup_steps = control_int(phase_controls, "qat_warmup_steps", int(qat_warmup_steps or 0))
        effective_lr_multiplier = max(0.0, control_float(phase_controls, "lr_multiplier", 1.0))
        lr_step = cosine_lr(step, lr * effective_lr_multiplier, warmup_steps, steps)
        effective_qat_loss_weight = effective_qat_base_weight * linear_ramp(
            step,
            effective_qat_start_step,
            effective_qat_warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_step
        for accum_idx in range(grad_accum):
            complex_active = (
                complex_iterator is not None
                and step >= int(complex_start_step or 0)
                and deterministic_ratio_choice(step, accum_idx, effective_complex_mix_ratio)
            )
            batch = next(complex_iterator if complex_active else iterator)
            step_complex_microbatches += float(complex_active)
            tokens = batch["tokens"].to(device, non_blocking=True)
            sample_ids = batch["sample_ids"].to(device, non_blocking=True)
            last_train_batch = {"tokens": tokens.detach(), "sample_ids": sample_ids.detach()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(
                    tokens,
                    sample_ids=sample_ids,
                    pass_id=step * grad_accum + accum_idx,
                    sample_gflownet=model_config.use_gflownet_policy and effective_gflownet_loss_weight > 0,
                    gflownet_samples=1,
                )
                micro_loss = out["loss"]
                gflownet_loss = out.get("gflownet_loss", torch.zeros((), device=device))
                gflownet_entropy = out.get("gflownet_entropy", torch.zeros((), device=device))
                if float(effective_gflownet_entropy_target) >= 0:
                    gflownet_entropy_objective = (gflownet_entropy - float(effective_gflownet_entropy_target)).pow(2)
                else:
                    gflownet_entropy_objective = -gflownet_entropy
                mtp_loss = out.get("mtp_loss", torch.zeros((), device=device))
                qat_loss = (
                    quantization_grid_loss(qat_named_params, bits=qat_bits)
                    if effective_qat_loss_weight > 0 and qat_named_params
                    else torch.zeros((), device=device)
                )
                contrastive_loss = out.get("contrastive_loss", torch.zeros((), device=device))
                trajectory_flow_loss = out.get("trajectory_flow_loss", torch.zeros((), device=device))
                if float(trajectory_flow_target) > 0:
                    trajectory_flow_penalty = torch.relu(trajectory_flow_loss - float(trajectory_flow_target)).pow(2)
                else:
                    trajectory_flow_penalty = trajectory_flow_loss
                toric_memory_entropy = out.get("toric_memory_entropy", torch.zeros((), device=device))
                toric_entropy_loss = (
                    torch.relu(torch.as_tensor(float(toric_entropy_floor), device=device) - toric_memory_entropy).pow(2)
                    if effective_toric_entropy_loss_weight > 0 and float(toric_entropy_floor) > 0
                    else torch.zeros((), device=device)
                )
                total_micro_loss = (
                    micro_loss
                    + effective_gflownet_loss_weight * gflownet_loss
                    + effective_gflownet_entropy_weight * gflownet_entropy_objective
                    + effective_mtp_loss_weight * mtp_loss
                    + effective_qat_loss_weight * qat_loss
                    + effective_contrastive_loss_weight * contrastive_loss
                    + effective_trajectory_flow_loss_weight * trajectory_flow_penalty
                    + effective_toric_entropy_loss_weight * toric_entropy_loss
                )
                loss = total_micro_loss / grad_accum
            loss.backward()
            step_loss += float(micro_loss.detach().cpu())
            step_total_loss += float(total_micro_loss.detach().cpu())
            step_gflownet_loss += float(gflownet_loss.detach().cpu())
            step_gflownet_entropy += float(gflownet_entropy.detach().cpu())
            step_gflownet_entropy_objective += float(gflownet_entropy_objective.detach().cpu())
            step_gflownet_diversity += float(out.get("gflownet_action_diversity", torch.zeros(())).detach().cpu())
            step_mtp_loss += float(mtp_loss.detach().cpu())
            step_qat_loss += float(qat_loss.detach().cpu())
            step_qat_weight += float(effective_qat_loss_weight)
            step_contrastive_loss += float(contrastive_loss.detach().cpu())
            step_trajectory_flow_loss += float(trajectory_flow_loss.detach().cpu())
            step_trajectory_flow_penalty += float(trajectory_flow_penalty.detach().cpu())
            step_trajectory_kinetic += float(out.get("trajectory_kinetic_energy", torch.zeros(())).detach().cpu())
            step_trajectory_viscous += float(out.get("trajectory_viscous_dissipation", torch.zeros(())).detach().cpu())
            step_smear_temperature += float(out.get("smear_temperature", torch.zeros(())).detach().cpu())
            step_toric_memory_entropy += float(out.get("toric_memory_entropy", torch.zeros(())).detach().cpu())
            step_toric_entropy_loss += float(toric_entropy_loss.detach().cpu())
        step_loss /= grad_accum
        step_total_loss /= grad_accum
        step_gflownet_loss /= grad_accum
        step_gflownet_entropy /= grad_accum
        step_gflownet_entropy_objective /= grad_accum
        step_gflownet_diversity /= grad_accum
        step_mtp_loss /= grad_accum
        step_qat_loss /= grad_accum
        step_qat_weight /= grad_accum
        step_contrastive_loss /= grad_accum
        step_trajectory_flow_loss /= grad_accum
        step_trajectory_flow_penalty /= grad_accum
        step_trajectory_kinetic /= grad_accum
        step_trajectory_viscous /= grad_accum
        step_smear_temperature /= grad_accum
        step_toric_memory_entropy /= grad_accum
        step_toric_entropy_loss /= grad_accum
        step_complex_microbatches /= grad_accum
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
                "train/gflownet_entropy_objective": step_gflownet_entropy_objective,
                "train/gflownet_entropy_target": float(effective_gflownet_entropy_target),
                "train/gflownet_action_diversity": step_gflownet_diversity,
                "train/gflownet_loss_weight": effective_gflownet_loss_weight,
                "train/gflownet_entropy_weight": effective_gflownet_entropy_weight,
                "train/mtp_loss": step_mtp_loss,
                "train/mtp_loss_weight": effective_mtp_loss_weight,
                "train/qat_loss": step_qat_loss,
                "train/qat_loss_weight": effective_qat_base_weight,
                "train/qat_effective_loss_weight": step_qat_weight,
                "train/qat_start_step": float(effective_qat_start_step),
                "train/qat_bits": qat_bits,
                "train/contrastive_loss": step_contrastive_loss,
                "train/contrastive_loss_weight": effective_contrastive_loss_weight,
                "train/trajectory_flow_loss": step_trajectory_flow_loss,
                "train/trajectory_flow_penalty": step_trajectory_flow_penalty,
                "train/trajectory_flow_target": float(trajectory_flow_target),
                "train/trajectory_flow_loss_weight": effective_trajectory_flow_loss_weight,
                "train/trajectory_kinetic_energy": step_trajectory_kinetic,
                "train/trajectory_viscous_dissipation": step_trajectory_viscous,
                "train/smear_temperature": step_smear_temperature,
                "train/toric_memory_entropy": step_toric_memory_entropy,
                "train/toric_entropy_floor": float(toric_entropy_floor),
                "train/toric_entropy_loss": step_toric_entropy_loss,
                "train/toric_entropy_loss_weight": float(effective_toric_entropy_loss_weight),
                "artifact/initial_bytes": report.bytes_total,
                "artifact/estimated_tensor_bytes": estimated_tensor_bytes,
                "artifact/deployment_parameters": report.deployment_parameters,
                "artifact/excluded_tensors": report.excluded_tensors,
                "model/parameters": params,
                "data/coprime_row_stride": float(coprime_row_stride),
                "data/stream_origin_step": float(stream_origin_step),
                "data/stream_burnin_steps": float(stream_burnin_steps),
                "data/complex_curriculum_active": float(
                    complex_iterator is not None and step >= int(complex_start_step or 0)
                ),
                "data/complex_microbatch_fraction": step_complex_microbatches,
                "data/complex_mix_ratio": effective_complex_mix_ratio,
                "data/complex_start_step": float(complex_start_step or 0),
                "eval/score_first_bias_lr": eval_score_first_bias_lr,
                "phase/index": float(phase_index),
                "phase/lr_multiplier": float(effective_lr_multiplier),
                "phase/base_gflownet_loss_weight": float(gflownet_loss_weight),
                "phase/base_complex_mix_ratio": float(complex_mix_ratio),
            }
            metrics.update(adaptive_controller.log_metrics())
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
            val_deterministic = evaluate(
                model,
                val_loader,
                device=device,
                batches=eval_batches,
                precision=precision,
                pass_id=seed + 10_000,
                order_samples=1,
                gflownet_samples=1,
                score_first_bias_lr=0.0,
                score_first_bias_decay=eval_score_first_bias_decay,
                score_first_bias_clip=eval_score_first_bias_clip,
            )
            val_gflownet = evaluate(
                model,
                val_loader,
                device=device,
                batches=eval_batches,
                precision=precision,
                pass_id=seed + 20_000 + step,
                order_samples=eval_order_samples,
                gflownet_samples=eval_gflownet_samples,
                score_first_bias_lr=0.0,
                score_first_bias_decay=eval_score_first_bias_decay,
                score_first_bias_clip=eval_score_first_bias_clip,
            )
            val_score_first = evaluate(
                model,
                val_loader,
                device=device,
                batches=eval_batches,
                precision=precision,
                pass_id=seed + 30_000 + step,
                order_samples=eval_order_samples,
                gflownet_samples=eval_gflownet_samples,
                score_first_bias_lr=eval_score_first_bias_lr,
                score_first_bias_decay=eval_score_first_bias_decay,
                score_first_bias_clip=eval_score_first_bias_clip,
            )
            metrics = {
                "val/loss": val_deterministic["loss"],
                "val/bpb": val_deterministic["bpb"],
                "val/deterministic_loss": val_deterministic["loss"],
                "val/deterministic_bpb": val_deterministic["bpb"],
                "val/gflownet_loss": val_gflownet["loss"],
                "val/gflownet_bpb": val_gflownet["bpb"],
                "val/score_first_loss": val_score_first["loss"],
                "val/score_first_bpb": val_score_first["bpb"],
                "val/order_samples": eval_order_samples,
                "val/gflownet_samples": eval_gflownet_samples,
                "val/score_first_bias_lr": eval_score_first_bias_lr,
            }
            if "bias_norm" in val_score_first:
                metrics["val/score_first_bias_norm"] = val_score_first["bias_norm"]
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
            hessian_due = (
                hessian_enabled
                and last_train_batch is not None
                and int(hessian_eval_every or 0) > 0
                and step % int(hessian_eval_every) == 0
            )
            if hessian_due:
                try:
                    metrics.update(
                        hessian_probe_metrics(
                            model=model,
                            batch=last_train_batch,
                            device=device,
                            precision=precision,
                            pass_id=step * 1_000_003 + 29,
                            max_tokens=int(hessian_max_tokens),
                            trace_samples=int(hessian_trace_samples),
                            power_iters=int(hessian_power_iters),
                            max_tensors=int(hessian_max_tensors),
                            max_parameters=int(hessian_max_parameters),
                            prefix="hessian/train",
                        )
                    )
                except RuntimeError as exc:
                    model.train()
                    model.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    metrics["hessian/train/error"] = 1.0
                    print(json.dumps({"step": step, "hessian_probe_error": str(exc)}))
            controller_metrics = adaptive_controller.update(
                step=step,
                train_bpb=bpb,
                val_bpb=float(val_deterministic["bpb"]),
                val_gflownet_bpb=float(val_gflownet["bpb"]),
            )
            if adaptive_controller.enabled:
                gflownet_loss_weight = adaptive_controller.gflownet_loss_weight
                gflownet_entropy_target = adaptive_controller.gflownet_entropy_target
                complex_mix_ratio = adaptive_controller.complex_mix_ratio
            metrics.update(controller_metrics)
            print(json.dumps({"step": step, **metrics}, indent=2))
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            if val_deterministic["bpb"] < best_val:
                best_val = val_deterministic["bpb"]
                best_checkpoint_path = checkpoint_dir / "best.pt"
                checkpoint_metrics = {"val_bpb": best_val, "val_loss": val_deterministic["loss"], **metrics}
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
