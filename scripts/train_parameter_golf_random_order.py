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

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm.auto import tqdm

from toricgt.complexity import random_order_complexity_metrics
from toricgt.parameter_golf_export import PARAMETER_GOLF_BYTE_LIMIT, write_artifact
from toricgt.polar_cache import polarquant_memory_estimate
from toricgt.random_order_lm import (
    DenseRandomOrderToricLM,
    RandomOrderLMConfig,
    advanced_byte_offset,
    advanced_vocab_size,
    byte_encode,
    byte_encode_with_special_tokens,
    estimate_uncompressed_quantized_bytes,
    special_token_map_for_mode,
)
from toricgt.wandb_organization import configure_wandb_metrics, organize_wandb_payload


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


def config_get_float(config: dict[str, Any], section: str, key: str, default: float) -> float:
    """Return a float config value while preserving explicit zero values."""

    value = config_get(config, section, key, default)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def config_get_int(config: dict[str, Any], section: str, key: str, default: int) -> int:
    """Return an int config value while preserving explicit zero values."""

    value = config_get(config, section, key, default)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def add_weighted_aux_loss(total: torch.Tensor, weight: float, term: torch.Tensor) -> torch.Tensor:
    """Add a finite auxiliary scalar only when its effective weight is active.

    Several geometry/topology diagnostics are computed even during phases where
    their loss weight is zero.  PyTorch still propagates ``0 * NaN`` as NaN, so
    disabled diagnostics must be skipped explicitly.  Active auxiliaries are
    finite-clamped before they are allowed to affect the optimizer.
    """

    weight = float(weight)
    if weight <= 0.0:
        return total
    if term.ndim > 0:
        term = term.mean()
    finite = torch.nan_to_num(term.float(), nan=0.0, posinf=1.0e4, neginf=-1.0e4).to(dtype=total.dtype)
    return total + weight * finite


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
        special_token_mode: str = "none",
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
        self.special_token_mode = str(special_token_mode or "none")
        self.special_token_map = special_token_map_for_mode(self.special_token_mode)
        if self.special_token_map and int(byte_offset) <= max(self.special_token_map.values()):
            raise ValueError("byte_offset must be greater than reserved reasoning/memory special token IDs")
        self._separator_bytes = self._encode_text(document_separator) if document_separator else []

    def _encode_text(self, text: str) -> list[int]:
        if self.special_token_map:
            return byte_encode_with_special_tokens(
                text,
                byte_offset=self.byte_offset,
                special_token_map=self.special_token_map,
            )
        return byte_encode(text, byte_offset=self.byte_offset)

    def _advanced_tokens_enabled(self) -> bool:
        return bool(self.special_token_map)

    @staticmethod
    def _has_any(haystack: str, needles: tuple[str, ...]) -> bool:
        return any(needle in haystack for needle in needles)

    def _is_reasoning_text(self, text: str) -> bool:
        return self._has_any(
            text.lower(),
            ("reason", "step", "thought", "chain", "tree", "trajectory", "simplex", "proof", "inference"),
        )

    def _is_memory_text(self, text: str) -> bool:
        return self._has_any(
            text.lower(),
            ("memory", "retriev", "recall", "read", "write", "store", "lookup", "prior", "consolidat"),
        )

    def _is_analogy_text(self, text: str) -> bool:
        return self._has_any(text.lower(), ("analog", "analogy", "analogue", "analoga", "::"))

    def _memory_marker(self, text: str) -> str:
        lowered = text.lower()
        if self._has_any(lowered, ("write", "store", "save", "update")):
            return "<|memory_write|>"
        if self._has_any(lowered, ("consolidat", "compress", "summar")):
            return "<|memory_consolidate|>"
        if self._has_any(lowered, ("link", "connect", "associat", "edge")):
            return "<|memory_link|>"
        return "<|memory_read|>"

    def _wrap_structured_span(self, text: str, *, force_reasoning: bool = False) -> str:
        if not self._advanced_tokens_enabled() or not text:
            return text
        lowered = text.lower()
        if self._is_memory_text(lowered):
            return f"<|memory_begin|>{self._memory_marker(lowered)} {text}<|memory_end|>"
        if self._is_analogy_text(lowered):
            return f"<|analogy_begin|>{text}<|analogy_end|>"
        if force_reasoning or self._is_reasoning_text(lowered):
            return f"<|reason_step_begin|>{text}<|reason_step_end|>"
        return text

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
        advanced = self._advanced_tokens_enabled()
        parts = ["<|got_begin|>", "<|simplex_begin|>"] if advanced else ["<graph>"]
        if isinstance(nodes, list):
            for node in nodes[:96]:
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("id", ""))[:48]
                node_type = str(node.get("type", ""))[:48]
                node_text = str(node.get("text", node.get("label", node.get("payload", ""))))[:220]
                line = f"node id={node_id} type={node_type} text={node_text}"
                if advanced:
                    line = self._wrap_structured_span(
                        line,
                        force_reasoning=self._is_reasoning_text(f"{node_id} {node_type}"),
                    )
                parts.append(line)
        if isinstance(edges, list):
            for edge in edges[:160]:
                if not isinstance(edge, dict):
                    continue
                src = str(edge.get("source", edge.get("src", "")))[:48]
                dst = str(edge.get("target", edge.get("dst", "")))[:48]
                edge_type = str(edge.get("type", edge.get("label", "")))[:48]
                line = f"edge {src}->{dst} type={edge_type}"
                if advanced:
                    edge_context = f"{src} {dst} {edge_type}"
                    prefixes = ["<|reason_edge|>"]
                    if self._is_memory_text(edge_context) or "link" in edge_context.lower():
                        prefixes.append(self._memory_marker(edge_context))
                    line = " ".join((*prefixes, line))
                    if self._is_memory_text(edge_context):
                        line = f"<|memory_begin|>{line}<|memory_end|>"
                    if self._is_analogy_text(edge_context):
                        line = f"<|analogy_begin|>{line}<|analogy_end|>"
                parts.append(line)
        if isinstance(targets, dict) and targets:
            parts.append("targets " + json.dumps(targets, ensure_ascii=False, sort_keys=True)[:512])
        elif isinstance(targets, list) and targets:
            parts.append("targets " + json.dumps(targets[:16], ensure_ascii=False)[:512])
        if not advanced:
            return "\n".join(parts)[: self.graph_projection_max_chars]
        suffix = "\n<|simplex_end|>\n<|got_end|>"
        projected = "\n".join(parts) + suffix
        if len(projected) <= self.graph_projection_max_chars:
            return projected
        keep = max(0, self.graph_projection_max_chars - len(suffix))
        return projected[:keep] + suffix

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
                if column in ("reasoning", "solution"):
                    parts.append(self._wrap_structured_span(value, force_reasoning=True))
                else:
                    parts.append(self._wrap_structured_span(value))
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
                        buffer.extend(self._encode_text(text))
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

    def __init__(self, seq_len: int, vocab_size: int, seed: int = 17, min_token_id: int = 4) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed
        self.min_token_id = max(1, min(int(min_token_id), max(1, int(vocab_size) - 1)))

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + worker_id)
        counter = worker_id * 10**12
        while True:
            yield {
                "tokens": torch.randint(
                    self.min_token_id,
                    self.vocab_size,
                    (self.seq_len,),
                    generator=generator,
                    dtype=torch.long,
                ),
                "sample_id": torch.tensor(counter, dtype=torch.long),
            }
            counter += 1


def load_competition_token_shard(path: str | Path) -> np.ndarray:
    """Read a Parameter-Golf challenge-format uint16 token shard."""

    file_path = Path(path)
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file_path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"unexpected challenge shard header for {file_path}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file_path.stat().st_size != expected_size:
        raise ValueError(f"challenge shard size mismatch for {file_path}: expected {expected_size} bytes")
    tokens = np.fromfile(file_path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if int(tokens.size) != num_tokens:
        raise ValueError(f"short read for challenge shard {file_path}")
    return tokens.astype(np.int32, copy=False)


class SentencePieceShardByteDataset(IterableDataset):
    """Decode local OAI competition SentencePiece shards and stream byte chunks.

    The native ToricGT Parameter-Golf model is byte-level, while the locally
    cached competition validation shard is the ``sp1024`` SentencePiece export.
    This dataset decodes that exact validation source back to text and scores
    the byte model on the resulting UTF-8 bytes.  Metrics are intentionally
    named ``oai_competition/*`` rather than ``fineweb/*`` to avoid confusing
    this native checkpoint evaluation with the separate FineWeb scaffold run.
    """

    def __init__(
        self,
        token_glob: str,
        tokenizer_path: str,
        seq_len: int,
        byte_offset: int,
        sp_tokens_per_decode: int = 4096,
        seed: int = 17,
        repeat: bool = False,
        shuffle_files: bool = False,
    ) -> None:
        super().__init__()
        self.files = sorted(glob.glob(token_glob))
        self.tokenizer_path = str(tokenizer_path)
        self.seq_len = int(seq_len)
        self.byte_offset = int(byte_offset)
        self.sp_tokens_per_decode = max(1, int(sp_tokens_per_decode))
        self.seed = int(seed)
        self.repeat = bool(repeat)
        self.shuffle_files = bool(shuffle_files)
        if not self.files:
            raise FileNotFoundError(f"no OAI competition token shards matched: {token_glob}")
        if not Path(self.tokenizer_path).is_file():
            raise FileNotFoundError(f"OAI competition tokenizer not found: {self.tokenizer_path}")

    def __iter__(self):
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise RuntimeError("sentencepiece is required for OAI competition shard decoding") from exc

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        files = [path for index, path in enumerate(self.files) if index % num_workers == worker_id]
        rng = random.Random(self.seed + worker_id)
        processor = spm.SentencePieceProcessor(model_file=self.tokenizer_path)
        counter = worker_id * 10**12
        epoch = 0
        while True:
            if self.shuffle_files:
                rng.shuffle(files)
            buffer: list[int] = []
            for file_path in files:
                shard_tokens = load_competition_token_shard(file_path)
                for start in range(0, int(shard_tokens.size), self.sp_tokens_per_decode):
                    piece_ids = [int(x) for x in shard_tokens[start : start + self.sp_tokens_per_decode]]
                    text = processor.decode(piece_ids)
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
    special_token_mode: str = "none",
) -> DataLoader:
    if synthetic or not glob.glob(parquet_glob):
        dataset = SyntheticByteChunkDataset(seq_len=seq_len, vocab_size=vocab_size, seed=seed, min_token_id=byte_offset)
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
            special_token_mode=special_token_mode,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_chunks,
    )


def build_oai_competition_loader(
    token_glob: str,
    tokenizer_path: str,
    batch_size: int,
    seq_len: int,
    byte_offset: int,
    sp_tokens_per_decode: int,
    seed: int,
    workers: int,
    repeat: bool = False,
) -> DataLoader:
    dataset = SentencePieceShardByteDataset(
        token_glob=token_glob,
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
        byte_offset=byte_offset,
        sp_tokens_per_decode=sp_tokens_per_decode,
        seed=seed,
        repeat=repeat,
        shuffle_files=repeat,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=max(0, int(workers)),
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


def graphcg_aux_activation_bytes(max_codes: int, directions: int, d_model: int) -> int:
    """Approximate fp32 activation bytes for one GraphCG auxiliary pass."""

    n = max(2, int(max_codes))
    d = max(2, int(directions))
    width = max(1, int(d_model))
    floats = (2 * n * d * width) + (n * d * d) + (d * d) + (n * d)
    return int(4 * floats)


def resolve_graphcg_num_directions(raw_value: Any, config: dict[str, Any], d_model: int, max_codes: int) -> int:
    """Resolve an integer or ``auto`` GraphCG basis width.

    ``auto`` chooses the largest multiple of eight under a small auxiliary
    memory budget with a ten percent safety margin, then caps the result for
    throughput.  On a 24 GB 4090 this resolves to the practical cap rather than
    the memory ceiling.
    """

    if not isinstance(raw_value, str) or raw_value.lower() != "auto":
        return int(raw_value)
    model_section = config.get("model", {}) or {}
    cap = int(model_section.get("graphcg_max_directions", 96))
    floor = int(model_section.get("graphcg_min_directions", 16))
    memory_fraction = float(model_section.get("graphcg_memory_fraction", 0.09))
    safety = float(model_section.get("graphcg_direction_safety", 0.90))
    total_memory = 0
    if torch.cuda.is_available():
        try:
            total_memory = int(torch.cuda.get_device_properties(0).total_memory)
        except RuntimeError:
            total_memory = 0
    if total_memory <= 0:
        return max(2, floor)
    budget = int(total_memory * max(0.0, memory_fraction) * max(0.1, min(1.0, safety)))
    best = max(2, floor)
    for directions in range(max(8, floor), max(8, cap) + 1, 8):
        if graphcg_aux_activation_bytes(max_codes, directions, d_model) <= budget:
            best = directions
    return int(best)


def deterministic_ratio_choice(step: int, accum_idx: int, ratio: float) -> bool:
    """Deterministically mix ordinary and complex curricula."""

    if ratio <= 0:
        return False
    if ratio >= 1:
        return True
    value = (step * 1_103_515_245 + accum_idx * 12_345 + 97_531) & 0xFFFF_FFFF
    return (value / 0x1_0000_0000) < ratio


def deterministic_unit_interval(step: int, accum_idx: int, salt: int = 0) -> float:
    value = (
        step * 1_103_515_245
        + accum_idx * 12_345
        + int(salt) * 2_654_435_761
        + 97_531
    ) & 0xFFFF_FFFF
    return value / 0x1_0000_0000


def choose_difficulty_stream(step: int, accum_idx: int, medium_ratio: float, hard_ratio: float) -> str:
    """Choose easy/medium/hard stream with deterministic Bernoulli mixtures.

    Hard receives the first slice of probability mass so rare hard examples are
    not accidentally starved when medium+hard is clipped.  The remaining mass
    is assigned to medium, and the complement is easy.
    """

    hard = max(0.0, min(1.0, float(hard_ratio)))
    medium = max(0.0, min(1.0 - hard, float(medium_ratio)))
    value = deterministic_unit_interval(step, accum_idx, salt=13)
    if value < hard:
        return "hard"
    if value < hard + medium:
        return "medium"
    return "easy"


PHASE_CONTROL_KEYS = {
    "complex_mix_ratio",
    "fineweb_mix_ratio",
    "hard_mix_ratio",
    "medium_mix_ratio",
    "gflownet_loss_weight",
    "gflownet_entropy_weight",
    "gflownet_entropy_target",
    "graphcg_loss_weight",
    "analogy_lattice_loss_weight",
    "koszul_persistence_loss_weight",
    "slepian_pollak_loss_weight",
    "tokengt_graph_loss_weight",
    "trajectory_flow_loss_weight",
    "trajectory_memory_loss_weight",
    "contrastive_loss_weight",
    "mtp_loss_weight",
    "toric_geometry_loss_weight",
    "toric_vector_bundle_loss_weight",
    "toric_bgg_loss_weight",
    "toric_entropy_loss_weight",
    "qat_loss_weight",
    "qat_start_step",
    "qat_warmup_steps",
    "lr_multiplier",
    "grad_clip_norm",
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
    medium_iterator: Any | None,
    complex_iterator: Any | None,
    *,
    config: dict[str, Any],
    start_step: int,
    burnin_steps: int,
    grad_accum: int,
    medium_start_step: int,
    medium_mix_ratio: float,
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
        return {"steps": 0.0, "microbatches": 0.0, "medium_microbatches": 0.0, "complex_microbatches": 0.0}
    burn_start = max(1, int(start_step) - burnin_steps + 1)
    burn_end = int(start_step)
    medium_microbatches = 0
    complex_microbatches = 0
    total_microbatches = 0
    for burn_step in range(burn_start, burn_end + 1):
        phase_controls, _, _ = active_phase_controls(config, burn_step)
        effective_medium_mix_ratio = max(
            0.0,
            min(1.0, control_float(phase_controls, "medium_mix_ratio", medium_mix_ratio)),
        )
        effective_complex_mix_ratio = max(
            0.0,
            min(
                1.0,
                control_float(
                    phase_controls,
                    "hard_mix_ratio",
                    control_float(phase_controls, "complex_mix_ratio", complex_mix_ratio),
                ),
            ),
        )
        for accum_idx in range(grad_accum):
            medium_ratio = (
                effective_medium_mix_ratio
                if medium_iterator is not None and burn_step >= int(medium_start_step or 0)
                else 0.0
            )
            hard_ratio = (
                effective_complex_mix_ratio
                if complex_iterator is not None and burn_step >= int(complex_start_step or 0)
                else 0.0
            )
            stream = choose_difficulty_stream(burn_step, accum_idx, medium_ratio, hard_ratio)
            if stream == "hard" and complex_iterator is not None:
                next(complex_iterator)
                complex_microbatches += 1
            elif stream == "medium" and medium_iterator is not None:
                next(medium_iterator)
                medium_microbatches += 1
            else:
                next(iterator)
            total_microbatches += 1
    return {
        "steps": float(burnin_steps),
        "microbatches": float(total_microbatches),
        "medium_microbatches": float(medium_microbatches),
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
    structural_loss_min: float = 0.70
    structural_loss_max: float = 1.45
    structural_loss_up_step: float = 0.04
    structural_loss_down_step: float = 0.08
    structural_signal_threshold: float = 0.12
    structural_train_drift_guard: float = 0.025

    def __post_init__(self) -> None:
        self.state_path = Path(self.state_path)
        self.update_count = 0
        self.train_bpb_ema: float | None = None
        self.previous_train_bpb_ema: float | None = None
        self.best_val_bpb = float("inf")
        self.gflownet_loss_weight = float(self.initial_gflownet_loss_weight)
        self.gflownet_entropy_target = float(self.initial_entropy_target)
        self.complex_mix_ratio = float(self.initial_complex_mix_ratio)
        self.structural_loss_multiplier = 1.0
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
            structural_loss_min=float(section.get("structural_loss_min", 0.70)),
            structural_loss_max=float(section.get("structural_loss_max", 1.45)),
            structural_loss_up_step=float(section.get("structural_loss_up_step", 0.04)),
            structural_loss_down_step=float(section.get("structural_loss_down_step", 0.08)),
            structural_signal_threshold=float(section.get("structural_signal_threshold", 0.12)),
            structural_train_drift_guard=float(section.get("structural_train_drift_guard", 0.025)),
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
        self.structural_loss_multiplier = float(
            payload.get("structural_loss_multiplier", self.structural_loss_multiplier)
        )
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
        self.structural_loss_multiplier = max(
            self.structural_loss_min,
            min(self.structural_loss_max, float(self.structural_loss_multiplier)),
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
            "structural_loss_multiplier": self.structural_loss_multiplier,
            "consecutive_val_improvements": self.consecutive_val_improvements,
            "last_metrics": self.last_metrics,
            "bounds": {
                "gflownet_loss": [self.gflownet_loss_min, self.gflownet_loss_max],
                "entropy_target": [self.entropy_target_min, self.initial_entropy_target],
                "complex_mix": [self.complex_mix_min, self.complex_mix_max],
                "structural_loss_multiplier": [self.structural_loss_min, self.structural_loss_max],
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
            "controller/structural_loss_multiplier": float(self.structural_loss_multiplier),
            "controller/recovery_mode_enabled": float(self.recovery_mode_enabled),
            "controller/consecutive_val_improvements": float(self.consecutive_val_improvements),
        }
        metrics.update(self.last_metrics)
        return metrics

    @staticmethod
    def _metric_value(metrics: dict[str, float], *keys: str) -> float:
        for key in keys:
            value = metrics.get(key)
            if value is None:
                continue
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value_float):
                return value_float
        return 0.0

    @classmethod
    def structural_pressure(cls, metrics: dict[str, float]) -> float:
        """Compress advanced-geometry diagnostics into a bounded control signal."""

        slepian_leakage = cls._metric_value(metrics, "slepian_pollak/leakage", "toric/slepian_leakage")
        graphcg_loss = cls._metric_value(metrics, "graphcg/basis_loss", "train/analogy_basis_loss")
        topology_loss = cls._metric_value(metrics, "topology/step_directed_loss", "topology/directed_loss")
        koszul_loss = cls._metric_value(metrics, "koszul_persistence/loss", "train/koszul_persistence_loss")
        bgg_loss = cls._metric_value(metrics, "bgg_category_o/loss", "train/toric_bgg_loss")
        memory_loss = cls._metric_value(metrics, "train/trajectory_memory_loss")
        pressure = (
            0.35 * max(0.0, min(1.0, slepian_leakage))
            + 0.15 * max(0.0, min(1.0, graphcg_loss))
            + 0.15 * max(0.0, min(1.0, topology_loss))
            + 0.12 * max(0.0, min(1.0, koszul_loss))
            + 0.10 * max(0.0, min(1.0, bgg_loss))
            + 0.13 * max(0.0, min(1.0, memory_loss))
        )
        return max(0.0, min(1.0, pressure))

    def update(
        self,
        *,
        step: int,
        train_bpb: float,
        val_bpb: float,
        val_gflownet_bpb: float,
        train_metrics: dict[str, float] | None = None,
    ) -> dict[str, float]:
        train_metrics = train_metrics or {}
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
        structural_pressure = self.structural_pressure(train_metrics)

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

            if recovery_active or (train_drift > float(self.train_drift_threshold) and not val_improved):
                self.structural_loss_multiplier -= self.structural_loss_down_step
            elif (
                structural_pressure > float(self.structural_signal_threshold)
                and val_improved
                and train_drift <= float(self.structural_train_drift_guard)
            ):
                fraction = min(1.0, structural_pressure / max(1e-6, float(self.structural_signal_threshold)))
                self.structural_loss_multiplier += self.structural_loss_up_step * fraction
            elif structural_pressure < 0.5 * float(self.structural_signal_threshold) and not val_improved:
                self.structural_loss_multiplier -= 0.5 * self.structural_loss_down_step

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
            "controller/structural_pressure": float(structural_pressure),
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
) -> list[str]:
    model_state = model.state_dict()
    adjusted_keys: list[str] = []
    key = "position_embedding.weight"
    if (
        allow_position_resize
        and key in state_dict
        and key in model_state
        and state_dict[key].shape != model_state[key].shape
    ):
        state_dict = dict(state_dict)
        state_dict[key] = resize_position_embedding_weight(state_dict[key], model_state[key].shape)
        adjusted_keys.append(key)
    key = "graphcg_direction_basis"
    if key in state_dict and key in model_state and state_dict[key].shape != model_state[key].shape:
        state_dict = dict(state_dict)
        if state_dict[key].ndim == 2 and model_state[key].ndim == 2 and state_dict[key].shape[1] == model_state[key].shape[1]:
            resized_basis = model_state[key].detach().clone()
            rows = min(int(resized_basis.shape[0]), int(state_dict[key].shape[0]))
            resized_basis[:rows] = state_dict[key].detach()[:rows].to(dtype=resized_basis.dtype, device=resized_basis.device)
            state_dict[key] = resized_basis
            adjusted_keys.append(key)
        else:
            del state_dict[key]
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = (
        "graphcg_",
        "toric_geometry_probe.",
        "toric_vector_bundle_probe.",
        "toric_bgg_probe.",
        "trajectory_memory_head.",
        "revealed_left_logits.",
        "revealed_right_logits.",
    )
    bad_missing = [key for key in incompatible.missing_keys if not key.startswith(allowed_missing_prefixes)]
    bad_unexpected = list(incompatible.unexpected_keys)
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}"
        )
    if incompatible.missing_keys:
        print(
            json.dumps(
                {
                    "resume_notice": "new_parameters_initialized",
                    "missing_keys": incompatible.missing_keys,
                }
            )
        )
    return adjusted_keys


def sanitize_optimizer_state_shapes(optimizer: torch.optim.Optimizer) -> int:
    """Drop stale per-parameter moments whose tensor shape no longer matches."""

    resets = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state:
                continue
            bad_shape = False
            for value in state.values():
                if torch.is_tensor(value) and value.ndim > 0 and value.shape != parameter.shape:
                    bad_shape = True
                    break
            if bad_shape:
                optimizer.state[parameter] = {}
                resets += 1
    return resets


def pad_optimizer_state_dict_for_new_parameters(
    optimizer: torch.optim.Optimizer,
    optimizer_state: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Preserve optimizer moments when the current model has appended params.

    PyTorch refuses to load an optimizer state if a parameter group length
    changed. For compatible resumes where new parameters were appended, such as
    introducing ``graphcg_direction_basis`` after an older checkpoint, the old
    moments are still valid for every existing parameter. This pads serialized
    parameter-id lists with fresh ids that have empty state, so AdamW initializes
    only the new tensors on their first update.
    """

    groups = optimizer_state.get("param_groups")
    if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        return optimizer_state, 0
    patched = dict(optimizer_state)
    patched_groups: list[dict[str, Any]] = []
    state = dict(patched.get("state", {}))
    next_id = max((int(key) for key in state.keys()), default=-1) + 1
    added = 0
    for saved_group, current_group in zip(groups, optimizer.param_groups):
        saved_params = list(saved_group.get("params", []))
        current_params = list(current_group.get("params", []))
        if len(saved_params) > len(current_params):
            return optimizer_state, 0
        group_copy = dict(saved_group)
        if len(saved_params) < len(current_params):
            for _ in range(len(current_params) - len(saved_params)):
                saved_params.append(next_id)
                state[next_id] = {}
                next_id += 1
                added += 1
        group_copy["params"] = saved_params
        patched_groups.append(group_copy)
    patched["param_groups"] = patched_groups
    patched["state"] = state
    return patched, added


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
    parser.add_argument("--oai-competition-eval", action="store_true")
    parser.add_argument("--no-oai-competition-eval", action="store_true")
    parser.add_argument("--oai-competition-token-glob")
    parser.add_argument("--oai-competition-tokenizer-path")
    parser.add_argument("--oai-competition-eval-interval", type=int)
    parser.add_argument("--oai-competition-eval-batches", type=int)
    parser.add_argument("--oai-competition-sp-tokens-per-decode", type=int)
    parser.add_argument("--oai-competition-workers", type=int)
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
    parser.add_argument("--polarquant-train-sample-tokens", type=int)
    parser.add_argument("--polarquant-eval-sample-tokens", type=int)
    parser.add_argument("--polarquant-precondition", action="store_true")
    parser.add_argument("--no-polarquant-precondition", action="store_true")
    parser.add_argument("--polarquant-radius-bits", type=int)
    parser.add_argument("--polarquant-seed", type=int)
    parser.add_argument("--special-token-mode", choices=["none", "reasoning_memory"])
    parser.add_argument("--advanced-reasoning-tokens", action="store_true")
    parser.add_argument("--no-advanced-reasoning-tokens", action="store_true")
    parser.add_argument("--use-gflownet-policy", action="store_true")
    parser.add_argument("--no-gflownet-policy", action="store_true")
    parser.add_argument("--gflownet-num-actions", type=int)
    parser.add_argument("--gflownet-hidden-dim", type=int)
    parser.add_argument("--gflownet-action-scale", type=float)
    parser.add_argument("--gflownet-loss-weight", type=float)
    parser.add_argument("--gflownet-entropy-weight", type=float)
    parser.add_argument("--gflownet-entropy-target", type=float)
    parser.add_argument("--use-graphcg", action="store_true")
    parser.add_argument("--no-graphcg", action="store_true")
    parser.add_argument("--graphcg-loss-weight", type=float)
    parser.add_argument("--use-analogy-lattice", action="store_true")
    parser.add_argument("--no-analogy-lattice", action="store_true")
    parser.add_argument("--analogy-lattice-loss-weight", type=float)
    parser.add_argument("--toric-geometry-loss-weight", type=float)
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
    parser.add_argument("--use-trajectory-memory-head", action="store_true")
    parser.add_argument("--no-trajectory-memory-head", action="store_true")
    parser.add_argument("--trajectory-memory-loss-weight", type=float)
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

    configured_d_model = args.d_model if args.d_model is not None else config_get(file_config, "model", "d_model", 384)
    if args.no_advanced_reasoning_tokens:
        special_token_mode = "none"
    elif args.advanced_reasoning_tokens:
        special_token_mode = "reasoning_memory"
    elif args.special_token_mode is not None:
        special_token_mode = args.special_token_mode
    else:
        configured_special_token_mode = config_get(file_config, "model", "special_token_mode", None)
        use_advanced_reasoning_tokens = bool(
            config_get(file_config, "model", "use_advanced_reasoning_tokens", False)
        )
        special_token_mode = (
            configured_special_token_mode
            if configured_special_token_mode is not None
            else ("reasoning_memory" if use_advanced_reasoning_tokens else "none")
        )
    special_token_ids = special_token_map_for_mode(special_token_mode)
    advanced_structural_mode = bool(special_token_ids)
    default_byte_offset = advanced_byte_offset() if advanced_structural_mode else 4
    default_vocab_size = advanced_vocab_size() if advanced_structural_mode else 260
    configured_graphcg_max_codes = config_get(file_config, "model", "graphcg_max_codes", 256)
    configured_graphcg_directions = resolve_graphcg_num_directions(
        config_get(file_config, "model", "graphcg_num_directions", 12),
        file_config,
        d_model=int(configured_d_model),
        max_codes=int(configured_graphcg_max_codes),
    )
    model_config = RandomOrderLMConfig(
        vocab_size=config_get(file_config, "model", "vocab_size", default_vocab_size),
        max_seq_len=args.seq_len if args.seq_len is not None else config_get(file_config, "model", "max_seq_len", 1024),
        d_model=configured_d_model,
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
        polarquant_train_sample_tokens=(
            args.polarquant_train_sample_tokens
            if args.polarquant_train_sample_tokens is not None
            else config_get(file_config, "model", "polarquant_train_sample_tokens", 0)
        ),
        polarquant_eval_sample_tokens=(
            args.polarquant_eval_sample_tokens
            if args.polarquant_eval_sample_tokens is not None
            else config_get(file_config, "model", "polarquant_eval_sample_tokens", 0)
        ),
        polarquant_precondition=(
            False
            if args.no_polarquant_precondition
            else bool(
                args.polarquant_precondition
                or config_get(file_config, "model", "polarquant_precondition", False)
            )
        ),
        polarquant_radius_bits=(
            args.polarquant_radius_bits
            if args.polarquant_radius_bits is not None
            else config_get(file_config, "model", "polarquant_radius_bits", 16)
        ),
        polarquant_seed=(
            args.polarquant_seed
            if args.polarquant_seed is not None
            else config_get(file_config, "model", "polarquant_seed", 271828)
        ),
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
        use_revealed_neighbor_context=config_get(file_config, "model", "use_revealed_neighbor_context", False),
        revealed_neighbor_radius=config_get(file_config, "model", "revealed_neighbor_radius", 2),
        revealed_neighbor_context_weight=config_get(file_config, "model", "revealed_neighbor_context_weight", 0.75),
        use_revealed_context_prior=config_get(file_config, "model", "use_revealed_context_prior", False),
        revealed_context_prior_alpha=config_get(file_config, "model", "revealed_context_prior_alpha", 0.25),
        revealed_context_prior_weight=config_get(file_config, "model", "revealed_context_prior_weight", 0.35),
        revealed_context_prior_mode=config_get(file_config, "model", "revealed_context_prior_mode", "additive"),
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
        use_toric_geometry_tasks=config_get(file_config, "model", "use_toric_geometry_tasks", True),
        toric_geometry_num_exponents=config_get(file_config, "model", "toric_geometry_num_exponents", 16),
        toric_geometry_exponent_dim=config_get(file_config, "model", "toric_geometry_exponent_dim", 4),
        toric_geometry_probe_rank=config_get(file_config, "model", "toric_geometry_probe_rank", 12),
        toric_geometry_quant_bits=config_get(file_config, "model", "toric_geometry_quant_bits", 6),
        toric_geometry_teacher_temperature=config_get(
            file_config,
            "model",
            "toric_geometry_teacher_temperature",
            0.55,
        ),
        toric_geometry_margin=config_get(file_config, "model", "toric_geometry_margin", 0.18),
        toric_geometry_fan_weight=config_get(file_config, "model", "toric_geometry_fan_weight", 1.0),
        toric_geometry_bend_weight=config_get(file_config, "model", "toric_geometry_bend_weight", 0.3),
        toric_geometry_binom_weight=config_get(file_config, "model", "toric_geometry_binom_weight", 0.3),
        toric_geometry_moment_weight=config_get(file_config, "model", "toric_geometry_moment_weight", 0.5),
        toric_geometry_coxeter_weight=config_get(file_config, "model", "toric_geometry_coxeter_weight", 0.2),
        toric_geometry_braid_weight=config_get(file_config, "model", "toric_geometry_braid_weight", 0.1),
        toric_geometry_leaf_weight=config_get(file_config, "model", "toric_geometry_leaf_weight", 0.25),
        toric_geometry_max_positions=config_get(file_config, "model", "toric_geometry_max_positions", 256),
        toric_geometry_cas_toric_ideal_certificate_path=config_get(
            file_config,
            "model",
            "toric_geometry_cas_toric_ideal_certificate_path",
            os.environ.get("TORICGT_CAS_TORIC_IDEAL_CERT", ""),
        ),
        use_toric_vector_bundle=config_get(file_config, "model", "use_toric_vector_bundle", True),
        toric_vector_bundle_rank=config_get(file_config, "model", "toric_vector_bundle_rank", 8),
        toric_vector_bundle_num_rays=config_get(file_config, "model", "toric_vector_bundle_num_rays", 8),
        toric_vector_bundle_num_cones=config_get(file_config, "model", "toric_vector_bundle_num_cones", 8),
        toric_vector_bundle_filtration_levels=config_get(
            file_config,
            "model",
            "toric_vector_bundle_filtration_levels",
            3,
        ),
        toric_vector_bundle_max_positions=config_get(file_config, "model", "toric_vector_bundle_max_positions", 128),
        toric_vector_bundle_temperature=config_get(file_config, "model", "toric_vector_bundle_temperature", 0.25),
        toric_vector_bundle_ray_weight=config_get(file_config, "model", "toric_vector_bundle_ray_weight", 0.20),
        toric_vector_bundle_filtration_weight=config_get(
            file_config,
            "model",
            "toric_vector_bundle_filtration_weight",
            1.00,
        ),
        toric_vector_bundle_splitting_weight=config_get(
            file_config,
            "model",
            "toric_vector_bundle_splitting_weight",
            0.35,
        ),
        toric_vector_bundle_cech_weight=config_get(file_config, "model", "toric_vector_bundle_cech_weight", 0.25),
        toric_vector_bundle_cocycle_weight=config_get(
            file_config,
            "model",
            "toric_vector_bundle_cocycle_weight",
            0.05,
        ),
        use_graphcg=(
            False
            if args.no_graphcg
            else bool(args.use_graphcg or config_get(file_config, "model", "use_graphcg", advanced_structural_mode))
        ),
        graphcg_num_directions=configured_graphcg_directions,
        graphcg_alpha=config_get(file_config, "model", "graphcg_alpha", 0.12),
        graphcg_temperature=config_get(file_config, "model", "graphcg_temperature", 0.2),
        graphcg_max_codes=configured_graphcg_max_codes,
        graphcg_orthogonal_weight=config_get(file_config, "model", "graphcg_orthogonal_weight", 0.2),
        graphcg_covariance_weight=config_get(file_config, "model", "graphcg_covariance_weight", 0.05),
        graphcg_sparsity_weight=config_get(file_config, "model", "graphcg_sparsity_weight", 0.0001),
        use_analogy_lattice=(
            False
            if args.no_analogy_lattice
            else bool(
                args.use_analogy_lattice
                or config_get(file_config, "model", "use_analogy_lattice", advanced_structural_mode)
            )
        ),
        analogy_lattice_max_pairs=config_get(file_config, "model", "analogy_lattice_max_pairs", 256),
        analogy_lattice_stride=config_get(file_config, "model", "analogy_lattice_stride", 1),
        analogy_lattice_temperature=config_get(file_config, "model", "analogy_lattice_temperature", 0.2),
        analogy_lattice_basis_weight=config_get(file_config, "model", "analogy_lattice_basis_weight", 0.5),
        analogy_lattice_parallelogram_weight=config_get(
            file_config,
            "model",
            "analogy_lattice_parallelogram_weight",
            0.5,
        ),
        analogy_lattice_topology_weight=config_get(file_config, "model", "analogy_lattice_topology_weight", 0.25),
        analogy_topology_max_points_per_group=config_get(
            file_config,
            "model",
            "analogy_topology_max_points_per_group",
            16,
        ),
        analogy_topology_max_groups=config_get(file_config, "model", "analogy_topology_max_groups", 24),
        analogy_topology_k=config_get(file_config, "model", "analogy_topology_k", 4),
        analogy_topology_filtration_levels=config_get(file_config, "model", "analogy_topology_filtration_levels", 4),
        analogy_topology_radius_min=config_get(file_config, "model", "analogy_topology_radius_min", 0.55),
        analogy_topology_radius_max=config_get(file_config, "model", "analogy_topology_radius_max", 1.65),
        analogy_topology_chain_weight=config_get(file_config, "model", "analogy_topology_chain_weight", 0.25),
        analogy_topology_inclusion_weight=config_get(file_config, "model", "analogy_topology_inclusion_weight", 0.1),
        analogy_topology_directed=config_get(file_config, "model", "analogy_topology_directed", True),
        analogy_topology_directed_weight=config_get(file_config, "model", "analogy_topology_directed_weight", 0.35),
        analogy_topology_skew_scale=config_get(file_config, "model", "analogy_topology_skew_scale", 0.35),
        analogy_topology_cycle_weight=config_get(file_config, "model", "analogy_topology_cycle_weight", 0.1),
        analogy_hdbscan_enabled=config_get(file_config, "model", "analogy_hdbscan_enabled", True),
        analogy_hdbscan_weight=config_get(file_config, "model", "analogy_hdbscan_weight", 0.2),
        analogy_hdbscan_min_cluster_size=config_get(file_config, "model", "analogy_hdbscan_min_cluster_size", 4),
        analogy_hdbscan_min_samples=config_get(file_config, "model", "analogy_hdbscan_min_samples", 4),
        analogy_hdbscan_stability_threshold=config_get(
            file_config,
            "model",
            "analogy_hdbscan_stability_threshold",
            0.18,
        ),
        analogy_step_topology_weight=config_get(file_config, "model", "analogy_step_topology_weight", 0.25),
        analogy_step_topology_max_points=config_get(file_config, "model", "analogy_step_topology_max_points", 24),
        analogy_step_topology_max_windows=config_get(file_config, "model", "analogy_step_topology_max_windows", 6),
        analogy_step_topology_window_size=config_get(file_config, "model", "analogy_step_topology_window_size", 32),
        analogy_step_topology_step_stride=config_get(file_config, "model", "analogy_step_topology_step_stride", 8),
        analogy_step_topology_time_bias=config_get(file_config, "model", "analogy_step_topology_time_bias", 0.18),
        use_koszul_persistence=config_get(file_config, "model", "use_koszul_persistence", True),
        koszul_max_points=config_get(file_config, "model", "koszul_max_points", 24),
        koszul_max_windows=config_get(file_config, "model", "koszul_max_windows", 4),
        koszul_window_size=config_get(file_config, "model", "koszul_window_size", 32),
        koszul_step_stride=config_get(file_config, "model", "koszul_step_stride", 8),
        koszul_num_parameters=config_get(file_config, "model", "koszul_num_parameters", 3),
        koszul_temperature=config_get(file_config, "model", "koszul_temperature", 0.12),
        koszul_chart_exponents=config_get(file_config, "model", "koszul_chart_exponents", 12),
        koszul_rank_temperature=config_get(file_config, "model", "koszul_rank_temperature", 0.05),
        use_slepian_pollak=config_get(file_config, "model", "use_slepian_pollak", False),
        slepian_pollak_max_positions=config_get(file_config, "model", "slepian_pollak_max_positions", 256),
        slepian_pollak_modes=config_get(file_config, "model", "slepian_pollak_modes", 8),
        slepian_pollak_bandwidth=config_get(file_config, "model", "slepian_pollak_bandwidth", 0.075),
        slepian_pollak_target_concentration=config_get(
            file_config,
            "model",
            "slepian_pollak_target_concentration",
            0.72,
        ),
        use_tokengt_causal_graph=config_get(file_config, "model", "use_tokengt_causal_graph", False),
        tokengt_graph_max_nodes=config_get(file_config, "model", "tokengt_graph_max_nodes", 128),
        tokengt_graph_neighbor_radius=config_get(file_config, "model", "tokengt_graph_neighbor_radius", 2),
        tokengt_graph_temperature=config_get(file_config, "model", "tokengt_graph_temperature", 0.25),
        tokengt_graph_edge_weight=config_get(file_config, "model", "tokengt_graph_edge_weight", 0.50),
        tokengt_graph_direction_weight=config_get(file_config, "model", "tokengt_graph_direction_weight", 0.20),
        tokengt_graph_position_weight=config_get(file_config, "model", "tokengt_graph_position_weight", 0.20),
        tokengt_graph_byte_class_weight=config_get(file_config, "model", "tokengt_graph_byte_class_weight", 0.10),
        tokengt_graph_cycle_weight=config_get(file_config, "model", "tokengt_graph_cycle_weight", 0.05),
        tokengt_graph_noncausal_policy=config_get(
            file_config,
            "model",
            "tokengt_graph_noncausal_policy",
            "causal_when_possible",
        ),
        use_tokengt_graph_fusion=config_get(file_config, "model", "use_tokengt_graph_fusion", False),
        tokengt_graph_fusion_weight=config_get(file_config, "model", "tokengt_graph_fusion_weight", 0.08),
        tokengt_graph_fusion_layers=config_get(file_config, "model", "tokengt_graph_fusion_layers", 1),
        tokengt_graph_fusion_max_edges=config_get(file_config, "model", "tokengt_graph_fusion_max_edges", 0),
        contrastive_temperature=config_get(file_config, "model", "contrastive_temperature", 0.2),
        trajectory_flow_viscosity=config_get(file_config, "model", "trajectory_flow_viscosity", 0.05),
        use_trajectory_memory_head=(
            False
            if args.no_trajectory_memory_head
            else bool(
                args.use_trajectory_memory_head
                or config_get(file_config, "model", "use_trajectory_memory_head", advanced_structural_mode)
            )
        ),
        trajectory_memory_projection_dim=config_get(file_config, "model", "trajectory_memory_projection_dim", 128),
        trajectory_memory_teacher_temperature=config_get(file_config, "model", "trajectory_memory_teacher_temperature", 0.20),
        trajectory_memory_retrieval_temperature=config_get(
            file_config,
            "model",
            "trajectory_memory_retrieval_temperature",
            0.20,
        ),
        trajectory_memory_distill_weight=config_get(file_config, "model", "trajectory_memory_distill_weight", 0.25),
        trajectory_memory_quality_weight=config_get(file_config, "model", "trajectory_memory_quality_weight", 0.10),
        trajectory_memory_topology_weight=config_get(file_config, "model", "trajectory_memory_topology_weight", 0.20),
        trajectory_memory_graphcg_weight=config_get(file_config, "model", "trajectory_memory_graphcg_weight", 0.30),
        trajectory_memory_toric_weight=config_get(file_config, "model", "trajectory_memory_toric_weight", 0.20),
        trajectory_memory_persistence_weight=config_get(
            file_config,
            "model",
            "trajectory_memory_persistence_weight",
            0.20,
        ),
        trajectory_memory_persistence_max_points=config_get(
            file_config,
            "model",
            "trajectory_memory_persistence_max_points",
            48,
        ),
        trajectory_memory_persistence_landscape_layers=config_get(
            file_config,
            "model",
            "trajectory_memory_persistence_landscape_layers",
            3,
        ),
        trajectory_memory_persistence_landscape_resolution=config_get(
            file_config,
            "model",
            "trajectory_memory_persistence_landscape_resolution",
            24,
        ),
        trajectory_memory_persistence_image_resolution=config_get(
            file_config,
            "model",
            "trajectory_memory_persistence_image_resolution",
            12,
        ),
        use_toric_bgg=config_get(file_config, "model", "use_toric_bgg", False),
        toric_bgg_num_standard_tokens=config_get(file_config, "model", "toric_bgg_num_standard_tokens", 8),
        toric_bgg_probe_rank=config_get(file_config, "model", "toric_bgg_probe_rank", 8),
        toric_bgg_signature_dim=config_get(file_config, "model", "toric_bgg_signature_dim", 16),
        toric_bgg_max_positions=config_get(file_config, "model", "toric_bgg_max_positions", 64),
        toric_bgg_d2_weight=config_get(file_config, "model", "toric_bgg_d2_weight", 1.0),
        toric_bgg_standard_weight=config_get(file_config, "model", "toric_bgg_standard_weight", 0.25),
        toric_bgg_koszul_weight=config_get(file_config, "model", "toric_bgg_koszul_weight", 0.15),
        toric_bgg_gale_weight=config_get(file_config, "model", "toric_bgg_gale_weight", 0.10),
        toric_bgg_signature_weight=config_get(file_config, "model", "toric_bgg_signature_weight", 0.10),
        aux_mtp_offsets=args.aux_mtp_offsets
        if args.aux_mtp_offsets is not None
        else config_get(file_config, "model", "aux_mtp_offsets", 2),
        target_artifact_bytes=config_get(file_config, "model", "target_artifact_bytes", 15_600_000),
        byte_offset=config_get(file_config, "model", "byte_offset", default_byte_offset),
        special_token_mode=special_token_mode,
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
    grad_clip_norm = config_get_float(file_config, "training", "grad_clip_norm", 1.0)
    shock_guard_enabled = bool(config_get(file_config, "training", "shock_guard_enabled", False))
    shock_guard_start_step = config_get_int(file_config, "training", "shock_guard_start_step", 0)
    shock_guard_end_step = config_get_int(file_config, "training", "shock_guard_end_step", 0)
    shock_guard_loss_ratio = config_get_float(file_config, "training", "shock_guard_loss_ratio", 1.12)
    shock_guard_loss_delta = config_get_float(file_config, "training", "shock_guard_loss_delta", 0.35)
    shock_guard_grad_norm = config_get_float(file_config, "training", "shock_guard_grad_norm", 0.75)
    shock_guard_update_scale = config_get_float(file_config, "training", "shock_guard_update_scale", 0.35)
    robust_micro_loss_guard_enabled = bool(config_get(file_config, "training", "robust_micro_loss_guard_enabled", False))
    robust_micro_loss_guard_start_step = config_get_int(file_config, "training", "robust_micro_loss_guard_start_step", 0)
    robust_micro_loss_guard_end_step = config_get_int(file_config, "training", "robust_micro_loss_guard_end_step", 0)
    robust_micro_loss_guard_ratio = config_get_float(file_config, "training", "robust_micro_loss_guard_ratio", 1.08)
    robust_micro_loss_guard_delta = config_get_float(file_config, "training", "robust_micro_loss_guard_delta", 0.18)
    robust_micro_loss_guard_min_scale = config_get_float(file_config, "training", "robust_micro_loss_guard_min_scale", 0.10)
    robust_micro_loss_guard_min_scale = max(0.0, min(1.0, robust_micro_loss_guard_min_scale))
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
    oai_competition_eval_enabled = (
        False
        if args.no_oai_competition_eval
        else bool(args.oai_competition_eval or config_get(file_config, "oai_competition", "enabled", False))
    )
    oai_competition_token_glob = args.oai_competition_token_glob or config_get(
        file_config,
        "oai_competition",
        "val_token_glob",
        "amelie-iska/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin",
    )
    oai_competition_tokenizer_path = args.oai_competition_tokenizer_path or config_get(
        file_config,
        "oai_competition",
        "tokenizer_path",
        "amelie-iska/parameter-golf/data/tokenizers/fineweb_1024_bpe.model",
    )
    oai_competition_eval_interval = (
        args.oai_competition_eval_interval
        if args.oai_competition_eval_interval is not None
        else config_get(file_config, "oai_competition", "eval_interval", eval_interval)
    )
    oai_competition_eval_batches = (
        args.oai_competition_eval_batches
        if args.oai_competition_eval_batches is not None
        else config_get(file_config, "oai_competition", "eval_batches", max(1, min(8, int(eval_batches))))
    )
    oai_competition_order_samples = int(
        config_get(file_config, "oai_competition", "order_samples", max(1, int(eval_order_samples)))
    )
    oai_competition_gflownet_samples = int(
        config_get(file_config, "oai_competition", "gflownet_samples", max(1, int(eval_gflownet_samples)))
    )
    oai_competition_score_first_enabled = bool(
        config_get(file_config, "oai_competition", "score_first_enabled", True)
    )
    oai_competition_sp_tokens_per_decode = (
        args.oai_competition_sp_tokens_per_decode
        if args.oai_competition_sp_tokens_per_decode is not None
        else config_get(file_config, "oai_competition", "sp_tokens_per_decode", 4096)
    )
    oai_competition_workers = (
        args.oai_competition_workers
        if args.oai_competition_workers is not None
        else config_get(file_config, "oai_competition", "workers", 0)
    )
    fineweb_calibration_enabled = bool(config_get(file_config, "fineweb_calibration", "enabled", False))
    fineweb_train_token_glob = config_get(
        file_config,
        "fineweb_calibration",
        "train_token_glob",
        "amelie-iska/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin",
    )
    fineweb_tokenizer_path = config_get(
        file_config,
        "fineweb_calibration",
        "tokenizer_path",
        oai_competition_tokenizer_path,
    )
    fineweb_mix_ratio = max(
        0.0,
        min(1.0, config_get_float(file_config, "fineweb_calibration", "mix_ratio", 0.0)),
    )
    fineweb_sp_tokens_per_decode = config_get_int(
        file_config,
        "fineweb_calibration",
        "sp_tokens_per_decode",
        int(oai_competition_sp_tokens_per_decode),
    )
    fineweb_workers = config_get_int(file_config, "fineweb_calibration", "workers", int(oai_competition_workers))
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
    graphcg_loss_weight = (
        args.graphcg_loss_weight
        if args.graphcg_loss_weight is not None
        else config_get(file_config, "training", "graphcg_loss_weight", 6e-5 if advanced_structural_mode else 0.0)
    )
    analogy_lattice_loss_weight = (
        args.analogy_lattice_loss_weight
        if args.analogy_lattice_loss_weight is not None
        else config_get(
            file_config,
            "training",
            "analogy_lattice_loss_weight",
            1e-4 if advanced_structural_mode else 0.0,
        )
    )
    toric_geometry_loss_weight = (
        args.toric_geometry_loss_weight
        if args.toric_geometry_loss_weight is not None
        else config_get(file_config, "training", "toric_geometry_loss_weight", 0.0)
    )
    toric_vector_bundle_loss_weight = config_get_float(file_config, "training", "toric_vector_bundle_loss_weight", 0.0)
    toric_bgg_loss_weight = config_get_float(file_config, "training", "toric_bgg_loss_weight", 0.0)
    koszul_persistence_loss_weight = config_get(file_config, "training", "koszul_persistence_loss_weight", 0.0)
    slepian_pollak_loss_weight = config_get_float(file_config, "training", "slepian_pollak_loss_weight", 0.0)
    tokengt_graph_loss_weight = config_get_float(file_config, "training", "tokengt_graph_loss_weight", 0.0)
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
    trajectory_memory_loss_weight = (
        args.trajectory_memory_loss_weight
        if args.trajectory_memory_loss_weight is not None
        else config_get(
            file_config,
            "training",
            "trajectory_memory_loss_weight",
            2e-5 if advanced_structural_mode else 0.0,
        )
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
    if args.synthetic and publish_best_to_hf:
        publish_best_to_hf = False
        print(
            json.dumps(
                {
                    "checkpoint_publishing_notice": "disabled_for_synthetic_smoke",
                    "reason": "Synthetic smoke runs must not replace real best checkpoints.",
                },
                indent=2,
            )
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
    difficulty_curriculum_enabled = bool(config_get(file_config, "data", "difficulty_curriculum_enabled", False))
    easy_min_estimated_tokens = int(config_get(file_config, "data", "easy_min_estimated_tokens", min_estimated_tokens) or 0)
    easy_max_estimated_tokens = int(config_get(file_config, "data", "easy_max_estimated_tokens", max_estimated_tokens) or 0)
    if difficulty_curriculum_enabled:
        min_estimated_tokens = easy_min_estimated_tokens
        max_estimated_tokens = easy_max_estimated_tokens
    medium_start_step = int(config_get(file_config, "data", "medium_start_step", 0) or 0)
    medium_mix_ratio = float(config_get(file_config, "data", "medium_mix_ratio", 0.0) or 0.0)
    medium_min_estimated_tokens = int(
        config_get(file_config, "data", "medium_min_estimated_tokens", max(0, int(easy_max_estimated_tokens or 128) + 1)) or 0
    )
    medium_max_estimated_tokens = int(config_get(file_config, "data", "medium_max_estimated_tokens", 384) or 0)
    medium_task_family_keywords = tuple(config_get(file_config, "data", "medium_task_family_keywords", task_family_keywords) or [])
    medium_dataset_keywords = tuple(config_get(file_config, "data", "medium_dataset_keywords", dataset_keywords) or [])
    medium_include_graph_projection = bool(
        config_get(file_config, "data", "medium_include_graph_projection", include_graph_projection)
    )
    complex_start_step = (
        args.complex_start_step
        if args.complex_start_step is not None
        else config_get(file_config, "data", "hard_start_step", config_get(file_config, "data", "complex_start_step", 0))
    )
    complex_min_estimated_tokens = (
        args.complex_min_estimated_tokens
        if args.complex_min_estimated_tokens is not None
        else config_get(file_config, "data", "hard_min_estimated_tokens", config_get(file_config, "data", "complex_min_estimated_tokens", min_estimated_tokens))
    )
    complex_max_estimated_tokens = (
        args.complex_max_estimated_tokens
        if args.complex_max_estimated_tokens is not None
        else config_get(file_config, "data", "hard_max_estimated_tokens", config_get(file_config, "data", "complex_max_estimated_tokens", 0))
    )
    complex_task_family_keywords = tuple(
        args.complex_task_family_keywords
        if args.complex_task_family_keywords is not None
        else config_get(file_config, "data", "hard_task_family_keywords", config_get(file_config, "data", "complex_task_family_keywords", task_family_keywords))
    )
    complex_dataset_keywords = tuple(
        args.complex_dataset_keywords
        if args.complex_dataset_keywords is not None
        else config_get(file_config, "data", "hard_dataset_keywords", config_get(file_config, "data", "complex_dataset_keywords", dataset_keywords))
    )
    complex_include_graph_projection = bool(
        config_get(file_config, "data", "hard_include_graph_projection", config_get(file_config, "data", "complex_include_graph_projection", include_graph_projection))
    )
    complex_mix_ratio = (
        args.complex_mix_ratio
        if args.complex_mix_ratio is not None
        else config_get(file_config, "data", "hard_mix_ratio", config_get(file_config, "data", "complex_mix_ratio", 1.0))
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
    best_oai_competition_bpb = float("inf")
    best_oai_competition_tts_bpb = float("inf")
    optimizer_state_loaded = False
    if args.resume:
        payload = torch.load(args.resume, map_location=device)
        state_adjustments = load_state_dict_with_optional_position_resize(
            model,
            payload["model"],
            allow_position_resize=resize_position_embedding,
        )
        if state_adjustments:
            print(
                json.dumps(
                    {
                        "resume_notice": "checkpoint_tensors_adjusted",
                        "checkpoint": args.resume,
                        "new_max_seq_len": model_config.max_seq_len,
                        "adjusted_keys": state_adjustments,
                    }
                )
            )
        if args.reset_optimizer:
            print(
                json.dumps(
                    {
                        "resume_notice": "optimizer_state_reset_requested",
                        "checkpoint": args.resume,
                    }
                )
            )
        else:
            try:
                optimizer_state, padded_optimizer_params = pad_optimizer_state_dict_for_new_parameters(
                    optimizer,
                    payload["optimizer"],
                )
                if padded_optimizer_params:
                    print(
                        json.dumps(
                            {
                                "resume_notice": "optimizer_state_padded_for_new_parameters",
                                "checkpoint": args.resume,
                                "new_parameter_states": padded_optimizer_params,
                            }
                        )
                    )
                optimizer.load_state_dict(optimizer_state)
                stale_states = sanitize_optimizer_state_shapes(optimizer)
                if stale_states:
                    print(
                        json.dumps(
                            {
                                "resume_notice": "optimizer_state_partially_reset_after_shape_change",
                                "checkpoint": args.resume,
                                "stale_parameter_states": stale_states,
                            }
                        )
                    )
                optimizer_state_loaded = True
            except ValueError as exc:
                print(
                    json.dumps(
                        {
                            "resume_notice": "optimizer_state_reset_after_parameter_change",
                            "checkpoint": args.resume,
                            "reason": str(exc),
                        }
                    )
                )
        start_step = int(payload.get("step", 0))
        resume_metrics = payload.get("metrics", {})
        best_val = float(resume_metrics.get("val_bpb", resume_metrics.get("best_val_bpb", best_val)))
        best_oai_competition_bpb = float(
            resume_metrics.get(
                "oai_competition/best_bpb",
                resume_metrics.get("best_oai_competition_bpb", best_oai_competition_bpb),
            )
        )
        best_oai_competition_tts_bpb = float(
            resume_metrics.get(
                "oai_competition/best_test_time_scaled_bpb",
                resume_metrics.get("best_oai_competition_tts_bpb", best_oai_competition_tts_bpb),
            )
        )
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
    dtype_bits = 32 if precision == "fp32" else 16
    polarquant_head_dim = int(model_config.d_model) // max(int(model_config.num_heads), 1)
    polarquant_metrics: dict[str, float] = {
        "polarquant/enabled": float(model_config.polarquant_kv_bits > 0),
        "polarquant/kv_bits": float(model_config.polarquant_kv_bits),
        "polarquant/radius_bits": float(model_config.polarquant_radius_bits),
        "polarquant/precondition_enabled": float(model_config.polarquant_precondition),
        "polarquant/train_sample_tokens": float(model_config.polarquant_train_sample_tokens),
        "polarquant/eval_sample_tokens": float(model_config.polarquant_eval_sample_tokens),
        "polarquant/seed": float(model_config.polarquant_seed),
        "polarquant/head_dim": float(polarquant_head_dim),
        "polarquant/dtype_bits": float(dtype_bits),
    }
    if (
        model_config.polarquant_kv_bits > 0
        and polarquant_head_dim >= 2
        and not (polarquant_head_dim & (polarquant_head_dim - 1))
    ):
        kv_estimate = polarquant_memory_estimate(
            (
                max(int(batch_size), 1),
                max(int(model_config.num_layers), 1) * max(int(model_config.recurrent_passes), 1) * 2,
                max(int(model_config.num_heads), 1),
                max(int(model_config.max_seq_len), 1),
                polarquant_head_dim,
            ),
            angle_bits=int(model_config.polarquant_kv_bits),
            radius_bits=int(model_config.polarquant_radius_bits),
            dtype_bits=dtype_bits,
            include_sign_bits=True,
        )
        polarquant_metrics.update(
            {
                "polarquant/estimated_fp_kv_cache_bytes": float(kv_estimate.fp_bytes),
                "polarquant/estimated_polarquant_kv_cache_bytes": float(kv_estimate.polarquant_bytes),
                "polarquant/estimated_fp_kv_cache_mb": float(kv_estimate.fp_bytes) / 1_000_000.0,
                "polarquant/estimated_polarquant_kv_cache_mb": float(kv_estimate.polarquant_bytes) / 1_000_000.0,
                "polarquant/estimated_saved_mb": float(kv_estimate.saved_bytes) / 1_000_000.0,
                "polarquant/estimated_compression_ratio": float(kv_estimate.compression_ratio),
                "polarquant/estimated_vectors": float(kv_estimate.vectors),
                "polarquant/estimated_values_per_vector": float(kv_estimate.values_per_vector),
            }
        )

    curated_train_files = sorted(glob.glob(train_glob))
    curated_val_files = sorted(glob.glob(val_glob))
    fineweb_train_files = sorted(glob.glob(fineweb_train_token_glob))
    oai_competition_val_files = sorted(glob.glob(oai_competition_token_glob))

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
        special_token_mode=model_config.special_token_mode,
    )
    medium_train_loader = None
    if difficulty_curriculum_enabled and (int(medium_start_step or 0) > 0 or float(medium_mix_ratio or 0.0) > 0.0):
        medium_train_loader = build_loader(
            parquet_glob=train_glob,
            batch_size=batch_size,
            seq_len=model_config.max_seq_len,
            byte_offset=model_config.byte_offset,
            rows_per_batch=rows_per_batch,
            seed=seed + 250_000 + stream_seed_offset,
            workers=workers,
            repeat=True,
            synthetic=args.synthetic,
            vocab_size=model_config.vocab_size,
            include_graph_projection=medium_include_graph_projection,
            graph_projection_max_chars=graph_projection_max_chars,
            coprime_row_stride=coprime_row_stride,
            document_separator=document_separator,
            min_estimated_tokens=int(medium_min_estimated_tokens or 0),
            max_estimated_tokens=int(medium_max_estimated_tokens or 0),
            task_family_keywords=medium_task_family_keywords,
            dataset_keywords=medium_dataset_keywords,
            interleave_row_groups=interleave_row_groups,
            special_token_mode=model_config.special_token_mode,
        )
    complex_train_loader = None
    if int(complex_start_step or 0) > 0 or float(complex_mix_ratio or 0.0) > 0.0:
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
            special_token_mode=model_config.special_token_mode,
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
        special_token_mode=model_config.special_token_mode,
    )
    oai_competition_loader = None
    oai_competition_available = False
    oai_competition_error = ""
    if oai_competition_eval_enabled:
        try:
            oai_competition_loader = build_oai_competition_loader(
                token_glob=oai_competition_token_glob,
                tokenizer_path=oai_competition_tokenizer_path,
                batch_size=batch_size,
                seq_len=model_config.max_seq_len,
                byte_offset=model_config.byte_offset,
                sp_tokens_per_decode=int(oai_competition_sp_tokens_per_decode),
                seed=seed + 40_000,
                workers=int(oai_competition_workers),
                repeat=False,
            )
            oai_competition_available = True
        except Exception as exc:
            oai_competition_error = f"{type(exc).__name__}: {exc}"
            print(json.dumps({"oai_competition_eval_error": oai_competition_error}))
    fineweb_train_loader = None
    fineweb_calibration_available = False
    fineweb_calibration_error = ""
    if fineweb_calibration_enabled and fineweb_mix_ratio > 0.0:
        try:
            fineweb_train_loader = build_oai_competition_loader(
                token_glob=fineweb_train_token_glob,
                tokenizer_path=fineweb_tokenizer_path,
                batch_size=batch_size,
                seq_len=model_config.max_seq_len,
                byte_offset=model_config.byte_offset,
                sp_tokens_per_decode=int(fineweb_sp_tokens_per_decode),
                seed=seed + 750_000 + stream_seed_offset,
                workers=int(fineweb_workers),
                repeat=True,
            )
            fineweb_calibration_available = True
        except Exception as exc:
            fineweb_calibration_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"fineweb calibration loader unavailable: {fineweb_calibration_error}") from exc

    use_wandb = bool(args.wandb or config_get(file_config, "logging", "wandb", False)) and not args.no_wandb
    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project or config_get(file_config, "logging", "project", "toricgt-parameter-golf"),
            name=args.wandb_run_name or config_get(file_config, "logging", "run_name", None),
            config={
                "model": model.config_dict(),
                "special_token_mode": model_config.special_token_mode,
                "special_token_ids": model_config.special_token_ids,
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
                    "graphcg_loss_weight": graphcg_loss_weight,
                    "analogy_lattice_loss_weight": analogy_lattice_loss_weight,
                    "toric_geometry_loss_weight": toric_geometry_loss_weight,
                    "toric_vector_bundle_loss_weight": toric_vector_bundle_loss_weight,
                    "toric_bgg_loss_weight": toric_bgg_loss_weight,
                    "koszul_persistence_loss_weight": koszul_persistence_loss_weight,
                    "slepian_pollak_loss_weight": slepian_pollak_loss_weight,
                    "tokengt_graph_loss_weight": tokengt_graph_loss_weight,
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
                    "trajectory_memory_loss_weight": trajectory_memory_loss_weight,
                    "use_trajectory_memory_head": model_config.use_trajectory_memory_head,
                    "eval_score_first_bias_lr": eval_score_first_bias_lr,
                    "eval_score_first_bias_decay": eval_score_first_bias_decay,
                    "eval_score_first_bias_clip": eval_score_first_bias_clip,
                    "causal_audit_interval": causal_audit_interval,
                    "resize_position_embedding": resize_position_embedding,
                    "robust_micro_loss_guard_enabled": robust_micro_loss_guard_enabled,
                    "robust_micro_loss_guard_start_step": robust_micro_loss_guard_start_step,
                    "robust_micro_loss_guard_end_step": robust_micro_loss_guard_end_step,
                    "robust_micro_loss_guard_ratio": robust_micro_loss_guard_ratio,
                    "robust_micro_loss_guard_delta": robust_micro_loss_guard_delta,
                    "robust_micro_loss_guard_min_scale": robust_micro_loss_guard_min_scale,
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
                "polarquant": {
                    "enabled": bool(model_config.polarquant_kv_bits > 0),
                    "kv_bits": model_config.polarquant_kv_bits,
                    "radius_bits": model_config.polarquant_radius_bits,
                    "precondition_enabled": bool(model_config.polarquant_precondition),
                    "train_sample_tokens": model_config.polarquant_train_sample_tokens,
                    "eval_sample_tokens": model_config.polarquant_eval_sample_tokens,
                    "seed": model_config.polarquant_seed,
                    "head_dim": polarquant_head_dim,
                    "dtype_bits": dtype_bits,
                    "estimated_fp_kv_cache_mb": polarquant_metrics.get("polarquant/estimated_fp_kv_cache_mb"),
                    "estimated_polarquant_kv_cache_mb": polarquant_metrics.get(
                        "polarquant/estimated_polarquant_kv_cache_mb"
                    ),
                    "estimated_compression_ratio": polarquant_metrics.get(
                        "polarquant/estimated_compression_ratio"
                    ),
                },
                "oai_competition": {
                    "enabled": oai_competition_eval_enabled,
                    "available": oai_competition_available,
                    "val_token_glob": oai_competition_token_glob,
                    "tokenizer_path": oai_competition_tokenizer_path,
                    "eval_interval": oai_competition_eval_interval,
                    "eval_batches": oai_competition_eval_batches,
                    "order_samples": oai_competition_order_samples,
                    "gflownet_samples": oai_competition_gflownet_samples,
                    "score_first_enabled": oai_competition_score_first_enabled,
                    "sp_tokens_per_decode": oai_competition_sp_tokens_per_decode,
                    "workers": oai_competition_workers,
                    "metric_namespace": "oai_competition",
                    "source": "local_fineweb10B_sp1024_validation_decoded_to_utf8_bytes",
                    "error": oai_competition_error,
                },
                "fineweb_calibration": {
                    "enabled": fineweb_calibration_enabled,
                    "available": fineweb_calibration_available,
                    "train_token_glob": fineweb_train_token_glob,
                    "train_shards": len(fineweb_train_files),
                    "tokenizer_path": fineweb_tokenizer_path,
                    "mix_ratio": fineweb_mix_ratio,
                    "sp_tokens_per_decode": fineweb_sp_tokens_per_decode,
                    "workers": fineweb_workers,
                    "metric_namespace": "fineweb_calibration",
                    "source": "local_fineweb10B_sp1024_train_decoded_to_utf8_bytes",
                    "error": fineweb_calibration_error,
                },
                "metric_policy": {
                    "all_metric_namespaces_always_on": True,
                    "losses_follow_phase_curriculum": True,
                    "aliases": [
                        "topology",
                        "toric",
                        "tropical",
                        "complexity",
                        "bgg_category_o",
                        "category_o",
                        "slepian_pollak",
                        "polarquant",
                        "oai_competition",
                        "hessian",
                    ],
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
                    "train_parquet_glob": train_glob,
                    "val_parquet_glob": val_glob,
                    "full_curated_train_split_active": bool(curated_train_files and not task_family_keywords and not dataset_keywords),
                    "curated_train_shards": len(curated_train_files),
                    "curated_val_shards": len(curated_val_files),
                    "fineweb_train_shards": len(fineweb_train_files),
                    "oai_competition_val_shards": len(oai_competition_val_files),
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
                    "difficulty_curriculum_enabled": difficulty_curriculum_enabled,
                    "easy_min_estimated_tokens": easy_min_estimated_tokens,
                    "easy_max_estimated_tokens": easy_max_estimated_tokens,
                    "medium_start_step": medium_start_step,
                    "medium_mix_ratio": medium_mix_ratio,
                    "medium_include_graph_projection": medium_include_graph_projection,
                    "medium_min_estimated_tokens": medium_min_estimated_tokens,
                    "medium_max_estimated_tokens": medium_max_estimated_tokens,
                    "medium_task_family_keywords": list(medium_task_family_keywords),
                    "medium_dataset_keywords": list(medium_dataset_keywords),
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
        configure_wandb_metrics(wandb)
        wandb_run.log(
            organize_wandb_payload({
                "trainer/step": float(start_step),
                "metrics_status/all_metric_namespaces_always_on": 1.0,
                "metrics_status/losses_follow_phase_curriculum": 1.0,
                "metrics_status/topology_probe_instantiated": float(model_config.use_analogy_lattice),
                "metrics_status/toric_probe_instantiated": float(model_config.use_toric_geometry_tasks),
                "metrics_status/tropical_metric_aliases_instantiated": float(model_config.use_toric_geometry_tasks),
                "metrics_status/toric_vector_bundle_probe_instantiated": float(model_config.use_toric_vector_bundle),
                "metrics_status/bgg_category_o_probe_instantiated": float(model_config.use_toric_bgg),
                "metrics_status/koszul_persistence_probe_instantiated": float(model_config.use_koszul_persistence),
                "metrics_status/slepian_pollak_probe_instantiated": float(model_config.use_slepian_pollak),
                "metrics_status/tokengt_causal_graph_probe_instantiated": float(model_config.use_tokengt_causal_graph),
                "metrics_status/complexity_enabled": float(complexity_enabled),
                "metrics_status/hessian_enabled": float(hessian_enabled),
                "oai_competition/enabled": float(oai_competition_eval_enabled),
                "oai_competition/available": float(oai_competition_available),
                "oai_competition/source_sp1024_decoded_bytes": float(oai_competition_available),
                "fineweb_calibration/enabled": float(fineweb_calibration_enabled),
                "fineweb_calibration/available": float(fineweb_calibration_available),
                "fineweb_calibration/mix_ratio": float(fineweb_mix_ratio),
                **polarquant_metrics,
            }),
            step=start_step,
        )

    print(json.dumps(
        {
            "model_type": "random_order_dense_lm",
            "parameters": params,
            "artifact_bytes": report.bytes_total,
            "estimated_tensor_bytes": estimated_tensor_bytes,
            "artifact_limit": PARAMETER_GOLF_BYTE_LIMIT,
            "polarquant": {
                key.split("/", 1)[1]: value
                for key, value in polarquant_metrics.items()
                if key.startswith("polarquant/")
            },
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
            "difficulty_curriculum_enabled": difficulty_curriculum_enabled,
            "easy_min_estimated_tokens": easy_min_estimated_tokens,
            "easy_max_estimated_tokens": easy_max_estimated_tokens,
            "medium_start_step": medium_start_step,
            "medium_mix_ratio": medium_mix_ratio,
            "medium_include_graph_projection": medium_include_graph_projection,
            "medium_min_estimated_tokens": medium_min_estimated_tokens,
            "medium_max_estimated_tokens": medium_max_estimated_tokens,
            "medium_task_family_keywords": list(medium_task_family_keywords),
            "medium_dataset_keywords": list(medium_dataset_keywords),
            "complex_start_step": complex_start_step,
            "complex_mix_ratio": complex_mix_ratio,
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
            "oai_competition_eval_enabled": oai_competition_eval_enabled,
            "oai_competition_available": oai_competition_available,
            "oai_competition_token_glob": oai_competition_token_glob,
            "oai_competition_tokenizer_path": oai_competition_tokenizer_path,
            "oai_competition_eval_interval": oai_competition_eval_interval,
            "oai_competition_eval_batches": oai_competition_eval_batches,
            "oai_competition_order_samples": oai_competition_order_samples,
            "oai_competition_gflownet_samples": oai_competition_gflownet_samples,
            "oai_competition_score_first_enabled": oai_competition_score_first_enabled,
            "oai_competition_sp_tokens_per_decode": oai_competition_sp_tokens_per_decode,
            "oai_competition_error": oai_competition_error,
            "fineweb_calibration_enabled": fineweb_calibration_enabled,
            "fineweb_calibration_available": fineweb_calibration_available,
            "fineweb_train_token_glob": fineweb_train_token_glob,
            "fineweb_tokenizer_path": fineweb_tokenizer_path,
            "fineweb_mix_ratio": fineweb_mix_ratio,
            "fineweb_sp_tokens_per_decode": fineweb_sp_tokens_per_decode,
            "fineweb_workers": fineweb_workers,
            "fineweb_calibration_error": fineweb_calibration_error,
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
            "toric_geometry_loss_weight": toric_geometry_loss_weight,
            "toric_vector_bundle_loss_weight": toric_vector_bundle_loss_weight,
            "toric_bgg_loss_weight": toric_bgg_loss_weight,
            "koszul_persistence_loss_weight": koszul_persistence_loss_weight,
            "slepian_pollak_loss_weight": slepian_pollak_loss_weight,
            "tokengt_graph_loss_weight": tokengt_graph_loss_weight,
            "trajectory_flow_target": trajectory_flow_target,
            "trajectory_memory_loss_weight": trajectory_memory_loss_weight,
            "use_trajectory_memory_head": model_config.use_trajectory_memory_head,
            "qat_start_step": qat_start_step,
            "qat_warmup_steps": qat_warmup_steps,
            "complex_mix_ratio": complex_mix_ratio,
            "adaptive_training": adaptive_controller.state_dict(),
            "phase_curriculum": file_config.get("phase_curriculum", {}),
            "resize_position_embedding": resize_position_embedding,
            "robust_micro_loss_guard_enabled": robust_micro_loss_guard_enabled,
            "robust_micro_loss_guard_start_step": robust_micro_loss_guard_start_step,
            "robust_micro_loss_guard_end_step": robust_micro_loss_guard_end_step,
            "robust_micro_loss_guard_ratio": robust_micro_loss_guard_ratio,
            "robust_micro_loss_guard_delta": robust_micro_loss_guard_delta,
            "robust_micro_loss_guard_min_scale": robust_micro_loss_guard_min_scale,
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
    medium_iterator = iter(medium_train_loader) if medium_train_loader is not None else None
    complex_iterator = iter(complex_train_loader) if complex_train_loader is not None else None
    fineweb_iterator = iter(fineweb_train_loader) if fineweb_train_loader is not None else None
    burnin_result = burn_in_training_iterators(
        iterator,
        medium_iterator,
        complex_iterator,
        config=file_config,
        start_step=start_step,
        burnin_steps=stream_burnin_steps,
        grad_accum=grad_accum,
        medium_start_step=int(medium_start_step or 0),
        medium_mix_ratio=medium_mix_ratio,
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
                organize_wandb_payload({
                    "data/stream_origin_step": float(stream_origin_step),
                    "data/stream_burnin_steps": float(stream_burnin_steps),
                    "data/stream_burnin_microbatches": float(burnin_result["microbatches"]),
                    "data/stream_burnin_medium_microbatches": float(burnin_result["medium_microbatches"]),
                    "data/stream_burnin_complex_microbatches": float(burnin_result["complex_microbatches"]),
                }),
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
    last_train_metrics: dict[str, float] = {}
    for step in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_neural_loss = 0.0
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
        step_trajectory_memory_loss = 0.0
        step_trajectory_memory_ce = 0.0
        step_trajectory_memory_distill_loss = 0.0
        step_trajectory_memory_quality_loss = 0.0
        step_trajectory_memory_recall1 = 0.0
        step_trajectory_memory_entropy = 0.0
        step_trajectory_memory_score_gap = 0.0
        step_trajectory_memory_teacher_diag_prob = 0.0
        step_trajectory_memory_persistence_similarity = 0.0
        step_trajectory_memory_persistence_norm = 0.0
        step_trajectory_memory_persistence_entropy = 0.0
        step_trajectory_memory_persistence_total = 0.0
        step_trajectory_memory_persistence_weight = 0.0
        step_smear_temperature = 0.0
        step_revealed_neighbor_known_fraction = 0.0
        step_revealed_neighbor_context_norm = 0.0
        step_revealed_context_prior_weight = 0.0
        step_revealed_context_prior_mixture_weight = 0.0
        step_toric_memory_entropy = 0.0
        step_toric_entropy_loss = 0.0
        step_toric_geometry_loss = 0.0
        step_toric_fan_loss = 0.0
        step_toric_active_face_ce = 0.0
        step_toric_active_face_margin = 0.0
        step_toric_active_face_entropy = 0.0
        step_toric_bend_loss = 0.0
        step_toric_bend_magnitude = 0.0
        step_toric_binomial_loss = 0.0
        step_toric_binomial_residual = 0.0
        step_toric_binomial_relation_source_exact = 0.0
        step_toric_moment_loss = 0.0
        step_toric_coxeter_loss = 0.0
        step_toric_affine_wall_distance = 0.0
        step_toric_braid_loss = 0.0
        step_toric_leaf_residual = 0.0
        step_toric_probe_rank = 0.0
        step_toric_probe_quant_bits = 0.0
        step_toric_vector_bundle_loss = 0.0
        step_toric_vector_bundle_ray_ce = 0.0
        step_toric_vector_bundle_level_ce = 0.0
        step_toric_vector_bundle_filtration_residual = 0.0
        step_toric_vector_bundle_nesting_residual = 0.0
        step_toric_vector_bundle_splitting_residual = 0.0
        step_toric_vector_bundle_cech_gluing_residual = 0.0
        step_toric_sheaf_cocycle_residual = 0.0
        step_toric_vector_bundle_ray_entropy = 0.0
        step_toric_vector_bundle_active_ray_mass = 0.0
        step_toric_vector_bundle_rank = 0.0
        step_toric_vector_bundle_num_rays = 0.0
        step_toric_vector_bundle_filtration_levels = 0.0
        step_toric_bgg_loss = 0.0
        step_toric_bgg_d2_residual = 0.0
        step_toric_bgg_resolution_consistency = 0.0
        step_toric_bgg_standard_leakage = 0.0
        step_toric_bgg_standard_allowed_mass = 0.0
        step_toric_bgg_koszul_linearity_residual = 0.0
        step_toric_bgg_gale_dual_consistency = 0.0
        step_toric_bgg_signature_smoothness = 0.0
        step_toric_bgg_standard_entropy = 0.0
        step_koszul_persistence_loss = 0.0
        step_koszul_exactness_residual = 0.0
        step_koszul_syzygy_residual = 0.0
        step_koszul_fitting_rank_residual = 0.0
        step_koszul_be_rank_residual = 0.0
        step_koszul_be_multiplier_residual = 0.0
        step_koszul_multigraded_betti_mass = 0.0
        step_koszul_chart_entropy = 0.0
        step_koszul_chart_coverage = 0.0
        step_koszul_chart_transition_shift = 0.0
        step_koszul_windows = 0.0
        step_slepian_pollak_loss = 0.0
        step_slepian_pollak_concentration = 0.0
        step_slepian_pollak_leakage = 0.0
        step_slepian_pollak_mode_entropy = 0.0
        step_slepian_pollak_effective_modes = 0.0
        step_slepian_pollak_modes = 0.0
        step_slepian_pollak_bandwidth = 0.0
        step_tokengt_graph_loss = 0.0
        step_tokengt_graph_edge_bce = 0.0
        step_tokengt_graph_direction_loss = 0.0
        step_tokengt_graph_position_loss = 0.0
        step_tokengt_graph_byte_class_loss = 0.0
        step_tokengt_graph_cycle_loss = 0.0
        step_tokengt_graph_edge_density = 0.0
        step_tokengt_graph_causal_edge_fraction = 0.0
        step_tokengt_graph_policy_causal = 0.0
        step_tokengt_graph_fusion_context_norm = 0.0
        step_tokengt_graph_fusion_gate = 0.0
        step_tokengt_graph_fusion_nodes = 0.0
        step_tokengt_graph_fusion_edges = 0.0
        step_tokengt_graph_fusion_token_count = 0.0
        step_tokengt_graph_fusion_edge_density = 0.0
        step_robust_micro_loss_guard_fraction = 0.0
        step_robust_micro_loss_guard_scale = 0.0
        step_robust_micro_loss_guard_cap = 0.0
        step_graphcg_loss = 0.0
        step_graphcg_code_loss = 0.0
        step_graphcg_orthogonal_loss = 0.0
        step_graphcg_covariance_loss = 0.0
        step_graphcg_sparsity_loss = 0.0
        step_graphcg_basis_coherence = 0.0
        step_graphcg_axis_variance = 0.0
        step_analogy_lattice_loss = 0.0
        step_analogy_functor_loss = 0.0
        step_analogy_parallelogram_loss = 0.0
        step_analogy_topology_loss = 0.0
        step_analogy_barcode_loss = 0.0
        step_analogy_simplex_closure_loss = 0.0
        step_analogy_filtration_inclusion_loss = 0.0
        step_analogy_chain_map_loss = 0.0
        step_analogy_directed_topology_loss = 0.0
        step_analogy_directed_transitive_loss = 0.0
        step_analogy_directed_cycle_loss = 0.0
        step_analogy_directed_chain_map_loss = 0.0
        step_analogy_directed_asymmetry = 0.0
        step_analogy_directed_skew_norm = 0.0
        step_analogy_hdbscan_loss = 0.0
        step_analogy_hdbscan_stability = 0.0
        step_analogy_hdbscan_persistent_edge_density = 0.0
        step_analogy_hdbscan_outlier_score = 0.0
        step_analogy_hdbscan_core_radius = 0.0
        step_analogy_filtration_edge_density = 0.0
        step_analogy_filtration_triangle_density = 0.0
        step_analogy_step_topology_loss = 0.0
        step_analogy_step_barcode_loss = 0.0
        step_analogy_step_simplex_closure_loss = 0.0
        step_analogy_step_filtration_inclusion_loss = 0.0
        step_analogy_step_boundary_residual = 0.0
        step_analogy_step_dirichlet_energy = 0.0
        step_analogy_step_dec_conservation_loss = 0.0
        step_analogy_step_dec_mass_residual = 0.0
        step_analogy_step_dec_vorticity_drift = 0.0
        step_analogy_step_dec_kinetic_energy = 0.0
        step_analogy_step_dec_kinetic_energy_drift = 0.0
        step_analogy_step_dec_hodge_balance = 0.0
        step_analogy_step_dec_wedge_interior_residual = 0.0
        step_analogy_step_directed_topology_loss = 0.0
        step_analogy_step_directed_transitive_loss = 0.0
        step_analogy_step_directed_cycle_flux = 0.0
        step_analogy_step_directed_chain_commutator = 0.0
        step_analogy_step_directed_asymmetry = 0.0
        step_analogy_step_analogical_map_loss = 0.0
        step_analogy_step_directed_map_loss = 0.0
        step_analogy_step_transport_entropy = 0.0
        step_analogy_step_hdbscan_loss = 0.0
        step_analogy_step_hdbscan_stability = 0.0
        step_analogy_step_hdbscan_noise_fraction = 0.0
        step_analogy_step_hdbscan_persistent_edge_density = 0.0
        step_analogy_step_hdbscan_core_radius = 0.0
        step_analogy_step_ph_landscape_loss = 0.0
        step_analogy_step_ph_landscape_norm = 0.0
        step_analogy_step_ph_image_energy = 0.0
        step_analogy_step_edge_density = 0.0
        step_analogy_step_triangle_density = 0.0
        step_analogy_step_cycle_rank = 0.0
        step_analogy_step_betti0 = 0.0
        step_analogy_step_windows = 0.0
        step_analogy_basis_loss = 0.0
        step_analogy_axis_entropy = 0.0
        step_analogy_lattice_margin = 0.0
        step_analogy_relation_groups = 0.0
        step_analogy_topology_groups = 0.0
        step_analogy_graphcg_chart_dim = 0.0
        step_analogy_graphcg_chart_energy = 0.0
        step_medium_microbatches = 0.0
        step_complex_microbatches = 0.0
        step_fineweb_microbatches = 0.0
        last_train_batch: dict[str, torch.Tensor] | None = None
        phase_controls, phase_name, phase_index = active_phase_controls(file_config, step)
        effective_fineweb_mix_ratio = max(
            0.0,
            min(1.0, control_float(phase_controls, "fineweb_mix_ratio", fineweb_mix_ratio)),
        )
        effective_medium_mix_ratio = max(0.0, min(1.0, control_float(phase_controls, "medium_mix_ratio", medium_mix_ratio)))
        effective_complex_mix_ratio = max(
            0.0,
            min(
                1.0,
                control_float(
                    phase_controls,
                    "hard_mix_ratio",
                    control_float(phase_controls, "complex_mix_ratio", complex_mix_ratio),
                ),
            ),
        )
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
        effective_graphcg_loss_weight = max(0.0, control_float(phase_controls, "graphcg_loss_weight", graphcg_loss_weight))
        effective_analogy_lattice_loss_weight = max(
            0.0,
            control_float(phase_controls, "analogy_lattice_loss_weight", analogy_lattice_loss_weight),
        )
        effective_toric_geometry_loss_weight = max(
            0.0,
            control_float(phase_controls, "toric_geometry_loss_weight", toric_geometry_loss_weight),
        )
        effective_toric_vector_bundle_loss_weight = max(
            0.0,
            control_float(phase_controls, "toric_vector_bundle_loss_weight", toric_vector_bundle_loss_weight),
        )
        effective_toric_bgg_loss_weight = max(
            0.0,
            control_float(phase_controls, "toric_bgg_loss_weight", toric_bgg_loss_weight),
        )
        effective_koszul_persistence_loss_weight = max(
            0.0,
            control_float(phase_controls, "koszul_persistence_loss_weight", koszul_persistence_loss_weight),
        )
        effective_slepian_pollak_loss_weight = max(
            0.0,
            control_float(phase_controls, "slepian_pollak_loss_weight", slepian_pollak_loss_weight),
        )
        effective_tokengt_graph_loss_weight = max(
            0.0,
            control_float(phase_controls, "tokengt_graph_loss_weight", tokengt_graph_loss_weight),
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
        effective_trajectory_memory_loss_weight = max(
            0.0,
            control_float(phase_controls, "trajectory_memory_loss_weight", trajectory_memory_loss_weight),
        )
        structural_loss_multiplier = (
            adaptive_controller.structural_loss_multiplier if adaptive_controller.enabled else 1.0
        )
        effective_graphcg_loss_weight *= structural_loss_multiplier
        effective_analogy_lattice_loss_weight *= structural_loss_multiplier
        effective_toric_geometry_loss_weight *= structural_loss_multiplier
        effective_toric_vector_bundle_loss_weight *= structural_loss_multiplier
        effective_toric_bgg_loss_weight *= structural_loss_multiplier
        effective_koszul_persistence_loss_weight *= structural_loss_multiplier
        effective_slepian_pollak_loss_weight *= structural_loss_multiplier
        effective_tokengt_graph_loss_weight *= structural_loss_multiplier
        effective_trajectory_memory_loss_weight *= structural_loss_multiplier
        effective_toric_entropy_loss_weight = max(
            0.0,
            control_float(phase_controls, "toric_entropy_loss_weight", toric_entropy_loss_weight),
        )
        effective_qat_base_weight = max(0.0, control_float(phase_controls, "qat_loss_weight", qat_loss_weight))
        effective_qat_start_step = control_int(phase_controls, "qat_start_step", int(qat_start_step or 0))
        effective_qat_warmup_steps = control_int(phase_controls, "qat_warmup_steps", int(qat_warmup_steps or 0))
        effective_lr_multiplier = max(0.0, control_float(phase_controls, "lr_multiplier", 1.0))
        effective_grad_clip_norm = max(0.0, control_float(phase_controls, "grad_clip_norm", grad_clip_norm))
        lr_step = cosine_lr(step, lr * effective_lr_multiplier, warmup_steps, steps)
        effective_qat_loss_weight = effective_qat_base_weight * linear_ramp(
            step,
            effective_qat_start_step,
            effective_qat_warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_step
        for accum_idx in range(grad_accum):
            use_fineweb = (
                fineweb_iterator is not None
                and effective_fineweb_mix_ratio > 0.0
                and deterministic_unit_interval(step, accum_idx, salt=29) < effective_fineweb_mix_ratio
            )
            if use_fineweb:
                batch = next(fineweb_iterator)
                step_fineweb_microbatches += 1.0
            else:
                medium_ratio = (
                    effective_medium_mix_ratio
                    if medium_iterator is not None and step >= int(medium_start_step or 0)
                    else 0.0
                )
                hard_ratio = (
                    effective_complex_mix_ratio
                    if complex_iterator is not None and step >= int(complex_start_step or 0)
                    else 0.0
                )
                stream_name = choose_difficulty_stream(step, accum_idx, medium_ratio, hard_ratio)
                if stream_name == "hard" and complex_iterator is not None:
                    batch = next(complex_iterator)
                    step_complex_microbatches += 1.0
                elif stream_name == "medium" and medium_iterator is not None:
                    batch = next(medium_iterator)
                    step_medium_microbatches += 1.0
                else:
                    batch = next(iterator)
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
                graphcg_loss = out.get("graphcg_loss", torch.zeros((), device=device))
                analogy_lattice_loss = out.get("analogy_lattice_loss", torch.zeros((), device=device))
                toric_geometry_loss = out.get("toric_geometry_loss", torch.zeros((), device=device))
                toric_vector_bundle_loss = out.get("toric_vector_bundle_loss", torch.zeros((), device=device))
                toric_bgg_loss = out.get("toric_bgg_loss", torch.zeros((), device=device))
                koszul_persistence_loss = out.get("koszul_persistence_loss", torch.zeros((), device=device))
                slepian_pollak_loss = out.get("slepian_pollak_loss", torch.zeros((), device=device))
                tokengt_graph_loss = out.get("tokengt_graph_loss", torch.zeros((), device=device))
                qat_loss = (
                    quantization_grid_loss(qat_named_params, bits=qat_bits)
                    if effective_qat_loss_weight > 0 and qat_named_params
                    else torch.zeros((), device=device)
                )
                contrastive_loss = out.get("contrastive_loss", torch.zeros((), device=device))
                trajectory_flow_loss = out.get("trajectory_flow_loss", torch.zeros((), device=device))
                trajectory_memory_loss = out.get("trajectory_memory_loss", torch.zeros((), device=device))
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
                total_micro_loss = torch.nan_to_num(
                    micro_loss.float(),
                    nan=0.0,
                    posinf=1.0e4,
                    neginf=1.0e4,
                ).to(dtype=micro_loss.dtype)
                total_micro_loss = add_weighted_aux_loss(
                    total_micro_loss,
                    effective_gflownet_loss_weight,
                    gflownet_loss,
                )
                total_micro_loss = add_weighted_aux_loss(
                    total_micro_loss,
                    effective_gflownet_entropy_weight,
                    gflownet_entropy_objective,
                )
                total_micro_loss = add_weighted_aux_loss(total_micro_loss, effective_mtp_loss_weight, mtp_loss)
                total_micro_loss = add_weighted_aux_loss(
                    total_micro_loss,
                    effective_graphcg_loss_weight,
                    graphcg_loss,
                )
                total_micro_loss = add_weighted_aux_loss(
                    total_micro_loss,
                    effective_analogy_lattice_loss_weight,
                    analogy_lattice_loss,
                )
                total_micro_loss = add_weighted_aux_loss(
                    total_micro_loss,
                    effective_toric_geometry_loss_weight,
                    toric_geometry_loss,
                )
                total_micro_loss = add_weighted_aux_loss(
                    total_micro_loss,
                    effective_toric_vector_bundle_loss_weight,
                    toric_vector_bundle_loss,
                )
                total_micro_loss = add_weighted_aux_loss(
                    total_micro_loss,
                    effective_toric_bgg_loss_weight,
                    toric_bgg_loss,
                )
                total_micro_loss = add_weighted_aux_loss(
                    total_micro_loss,
                    effective_koszul_persistence_loss_weight,
                    koszul_persistence_loss,
                )
                total_micro_loss = add_weighted_aux_loss(
                    total_micro_loss,
                    effective_slepian_pollak_loss_weight,
                    slepian_pollak_loss,
                )
                total_micro_loss = add_weighted_aux_loss(
                    total_micro_loss,
                    effective_tokengt_graph_loss_weight,
                    tokengt_graph_loss,
                )
                total_micro_loss = add_weighted_aux_loss(total_micro_loss, effective_qat_loss_weight, qat_loss)
                total_micro_loss = add_weighted_aux_loss(
                    total_micro_loss,
                    effective_contrastive_loss_weight,
                    contrastive_loss,
                )
                total_micro_loss = add_weighted_aux_loss(
                    total_micro_loss,
                    effective_trajectory_flow_loss_weight,
                    trajectory_flow_penalty,
                )
                total_micro_loss = add_weighted_aux_loss(
                    total_micro_loss,
                    effective_trajectory_memory_loss_weight,
                    trajectory_memory_loss,
                )
                total_micro_loss = add_weighted_aux_loss(
                    total_micro_loss,
                    effective_toric_entropy_loss_weight,
                    toric_entropy_loss,
                )
                micro_guard_scale = 1.0
                micro_guard_cap = 0.0
                if robust_micro_loss_guard_enabled and running_loss > 0.0:
                    in_micro_guard_window = step >= robust_micro_loss_guard_start_step and (
                        robust_micro_loss_guard_end_step <= 0 or step < robust_micro_loss_guard_end_step
                    )
                    if in_micro_guard_window:
                        micro_guard_cap = min(
                            running_loss + robust_micro_loss_guard_delta,
                            running_loss * robust_micro_loss_guard_ratio,
                        )
                        micro_loss_value = float(micro_loss.detach().cpu())
                        if micro_loss_value > micro_guard_cap > 0.0:
                            micro_guard_scale = max(
                                robust_micro_loss_guard_min_scale,
                                min(1.0, micro_guard_cap / max(micro_loss_value, 1e-9)),
                            )
                loss = (total_micro_loss * micro_guard_scale) / grad_accum
            loss.backward()
            step_robust_micro_loss_guard_fraction += 1.0 if micro_guard_scale < 0.999 else 0.0
            step_robust_micro_loss_guard_scale += float(micro_guard_scale)
            step_robust_micro_loss_guard_cap += float(micro_guard_cap)
            step_loss += float(micro_loss.detach().cpu())
            step_neural_loss += float(out.get("neural_loss", micro_loss).detach().cpu())
            step_total_loss += float(total_micro_loss.detach().cpu())
            step_gflownet_loss += float(gflownet_loss.detach().cpu())
            step_gflownet_entropy += float(gflownet_entropy.detach().cpu())
            step_gflownet_entropy_objective += float(gflownet_entropy_objective.detach().cpu())
            step_gflownet_diversity += float(out.get("gflownet_action_diversity", torch.zeros(())).detach().cpu())
            step_mtp_loss += float(mtp_loss.detach().cpu())
            step_graphcg_loss += float(graphcg_loss.detach().cpu())
            step_graphcg_code_loss += float(out.get("graphcg_code_loss", torch.zeros(())).detach().cpu())
            step_graphcg_orthogonal_loss += float(out.get("graphcg_orthogonal_loss", torch.zeros(())).detach().cpu())
            step_graphcg_covariance_loss += float(out.get("graphcg_covariance_loss", torch.zeros(())).detach().cpu())
            step_graphcg_sparsity_loss += float(out.get("graphcg_sparsity_loss", torch.zeros(())).detach().cpu())
            step_graphcg_basis_coherence += float(out.get("graphcg_basis_coherence", torch.zeros(())).detach().cpu())
            step_graphcg_axis_variance += float(out.get("graphcg_axis_variance", torch.zeros(())).detach().cpu())
            step_analogy_lattice_loss += float(analogy_lattice_loss.detach().cpu())
            step_analogy_functor_loss += float(out.get("analogy_functor_loss", torch.zeros(())).detach().cpu())
            step_analogy_parallelogram_loss += float(out.get("analogy_parallelogram_loss", torch.zeros(())).detach().cpu())
            step_analogy_topology_loss += float(out.get("analogy_topology_loss", torch.zeros(())).detach().cpu())
            step_analogy_barcode_loss += float(out.get("analogy_barcode_loss", torch.zeros(())).detach().cpu())
            step_analogy_simplex_closure_loss += float(
                out.get("analogy_simplex_closure_loss", torch.zeros(())).detach().cpu()
            )
            step_analogy_filtration_inclusion_loss += float(
                out.get("analogy_filtration_inclusion_loss", torch.zeros(())).detach().cpu()
            )
            step_analogy_chain_map_loss += float(out.get("analogy_chain_map_loss", torch.zeros(())).detach().cpu())
            step_analogy_directed_topology_loss += float(
                out.get("analogy_directed_topology_loss", torch.zeros(())).detach().cpu()
            )
            step_analogy_directed_transitive_loss += float(
                out.get("analogy_directed_transitive_loss", torch.zeros(())).detach().cpu()
            )
            step_analogy_directed_cycle_loss += float(
                out.get("analogy_directed_cycle_loss", torch.zeros(())).detach().cpu()
            )
            step_analogy_directed_chain_map_loss += float(
                out.get("analogy_directed_chain_map_loss", torch.zeros(())).detach().cpu()
            )
            step_analogy_directed_asymmetry += float(
                out.get("analogy_directed_asymmetry", torch.zeros(())).detach().cpu()
            )
            step_analogy_directed_skew_norm += float(
                out.get("analogy_directed_skew_norm", torch.zeros(())).detach().cpu()
            )
            step_analogy_hdbscan_loss += float(out.get("analogy_hdbscan_loss", torch.zeros(())).detach().cpu())
            step_analogy_hdbscan_stability += float(
                out.get("analogy_hdbscan_stability", torch.zeros(())).detach().cpu()
            )
            step_analogy_hdbscan_persistent_edge_density += float(
                out.get("analogy_hdbscan_persistent_edge_density", torch.zeros(())).detach().cpu()
            )
            step_analogy_hdbscan_outlier_score += float(
                out.get("analogy_hdbscan_outlier_score", torch.zeros(())).detach().cpu()
            )
            step_analogy_hdbscan_core_radius += float(
                out.get("analogy_hdbscan_core_radius", torch.zeros(())).detach().cpu()
            )
            step_analogy_filtration_edge_density += float(
                out.get("analogy_filtration_edge_density", torch.zeros(())).detach().cpu()
            )
            step_analogy_filtration_triangle_density += float(
                out.get("analogy_filtration_triangle_density", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_topology_loss += float(out.get("analogy_step_topology_loss", torch.zeros(())).detach().cpu())
            step_analogy_step_barcode_loss += float(out.get("analogy_step_barcode_loss", torch.zeros(())).detach().cpu())
            step_analogy_step_simplex_closure_loss += float(
                out.get("analogy_step_simplex_closure_loss", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_filtration_inclusion_loss += float(
                out.get("analogy_step_filtration_inclusion_loss", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_boundary_residual += float(
                out.get("analogy_step_boundary_residual", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_dirichlet_energy += float(
                out.get("analogy_step_dirichlet_energy", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_dec_conservation_loss += float(
                out.get("analogy_step_dec_conservation_loss", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_dec_mass_residual += float(
                out.get("analogy_step_dec_mass_residual", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_dec_vorticity_drift += float(
                out.get("analogy_step_dec_vorticity_drift", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_dec_kinetic_energy += float(
                out.get("analogy_step_dec_kinetic_energy", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_dec_kinetic_energy_drift += float(
                out.get("analogy_step_dec_kinetic_energy_drift", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_dec_hodge_balance += float(
                out.get("analogy_step_dec_hodge_balance", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_dec_wedge_interior_residual += float(
                out.get("analogy_step_dec_wedge_interior_residual", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_directed_topology_loss += float(
                out.get("analogy_step_directed_topology_loss", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_directed_transitive_loss += float(
                out.get("analogy_step_directed_transitive_loss", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_directed_cycle_flux += float(
                out.get("analogy_step_directed_cycle_flux", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_directed_chain_commutator += float(
                out.get("analogy_step_directed_chain_commutator", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_directed_asymmetry += float(
                out.get("analogy_step_directed_asymmetry", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_analogical_map_loss += float(
                out.get("analogy_step_analogical_map_loss", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_directed_map_loss += float(
                out.get("analogy_step_directed_map_loss", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_transport_entropy += float(
                out.get("analogy_step_transport_entropy", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_hdbscan_loss += float(out.get("analogy_step_hdbscan_loss", torch.zeros(())).detach().cpu())
            step_analogy_step_hdbscan_stability += float(
                out.get("analogy_step_hdbscan_stability", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_hdbscan_noise_fraction += float(
                out.get("analogy_step_hdbscan_noise_fraction", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_hdbscan_persistent_edge_density += float(
                out.get("analogy_step_hdbscan_persistent_edge_density", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_hdbscan_core_radius += float(
                out.get("analogy_step_hdbscan_core_radius", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_ph_landscape_loss += float(
                out.get("analogy_step_ph_landscape_loss", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_ph_landscape_norm += float(
                out.get("analogy_step_ph_landscape_norm", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_ph_image_energy += float(
                out.get("analogy_step_ph_image_energy", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_edge_density += float(
                out.get("analogy_step_edge_density", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_triangle_density += float(
                out.get("analogy_step_triangle_density", torch.zeros(())).detach().cpu()
            )
            step_analogy_step_cycle_rank += float(out.get("analogy_step_cycle_rank", torch.zeros(())).detach().cpu())
            step_analogy_step_betti0 += float(out.get("analogy_step_betti0", torch.zeros(())).detach().cpu())
            step_analogy_step_windows += float(out.get("analogy_step_windows", torch.zeros(())).detach().cpu())
            step_analogy_basis_loss += float(out.get("analogy_basis_loss", torch.zeros(())).detach().cpu())
            step_analogy_axis_entropy += float(out.get("analogy_axis_entropy", torch.zeros(())).detach().cpu())
            step_analogy_lattice_margin += float(out.get("analogy_lattice_margin", torch.zeros(())).detach().cpu())
            step_analogy_relation_groups += float(out.get("analogy_relation_groups", torch.zeros(())).detach().cpu())
            step_analogy_topology_groups += float(out.get("analogy_topology_groups", torch.zeros(())).detach().cpu())
            step_analogy_graphcg_chart_dim += float(out.get("analogy_graphcg_chart_dim", torch.zeros(())).detach().cpu())
            step_analogy_graphcg_chart_energy += float(
                out.get("analogy_graphcg_chart_energy", torch.zeros(())).detach().cpu()
            )
            step_toric_geometry_loss += float(toric_geometry_loss.detach().cpu())
            step_toric_fan_loss += float(out.get("toric_fan_loss", torch.zeros(())).detach().cpu())
            step_toric_active_face_ce += float(out.get("toric_active_face_ce", torch.zeros(())).detach().cpu())
            step_toric_active_face_margin += float(
                out.get("toric_active_face_margin", torch.zeros(())).detach().cpu()
            )
            step_toric_active_face_entropy += float(
                out.get("toric_active_face_entropy", torch.zeros(())).detach().cpu()
            )
            step_toric_bend_loss += float(out.get("toric_bend_loss", torch.zeros(())).detach().cpu())
            step_toric_bend_magnitude += float(out.get("toric_bend_magnitude", torch.zeros(())).detach().cpu())
            step_toric_binomial_loss += float(out.get("toric_binomial_loss", torch.zeros(())).detach().cpu())
            step_toric_binomial_residual += float(
                out.get("toric_binomial_residual", torch.zeros(())).detach().cpu()
            )
            step_toric_binomial_relation_source_exact += float(
                out.get("toric_binomial_relation_source_exact", torch.zeros(())).detach().cpu()
            )
            step_toric_moment_loss += float(out.get("toric_moment_loss", torch.zeros(())).detach().cpu())
            step_toric_coxeter_loss += float(out.get("toric_coxeter_loss", torch.zeros(())).detach().cpu())
            step_toric_affine_wall_distance += float(
                out.get("toric_affine_wall_distance", torch.zeros(())).detach().cpu()
            )
            step_toric_braid_loss += float(out.get("toric_braid_loss", torch.zeros(())).detach().cpu())
            step_toric_leaf_residual += float(out.get("toric_leaf_residual", torch.zeros(())).detach().cpu())
            step_toric_probe_rank += float(out.get("toric_probe_rank", torch.zeros(())).detach().cpu())
            step_toric_probe_quant_bits += float(
                out.get("toric_probe_quant_bits", torch.zeros(())).detach().cpu()
            )
            step_toric_vector_bundle_loss += float(toric_vector_bundle_loss.detach().cpu())
            step_toric_vector_bundle_ray_ce += float(
                out.get("toric_vector_bundle_ray_ce", torch.zeros(())).detach().cpu()
            )
            step_toric_vector_bundle_level_ce += float(
                out.get("toric_vector_bundle_level_ce", torch.zeros(())).detach().cpu()
            )
            step_toric_vector_bundle_filtration_residual += float(
                out.get("toric_vector_bundle_filtration_residual", torch.zeros(())).detach().cpu()
            )
            step_toric_vector_bundle_nesting_residual += float(
                out.get("toric_vector_bundle_klyachko_nesting_residual", torch.zeros(())).detach().cpu()
            )
            step_toric_vector_bundle_splitting_residual += float(
                out.get("toric_vector_bundle_cone_splitting_residual", torch.zeros(())).detach().cpu()
            )
            step_toric_vector_bundle_cech_gluing_residual += float(
                out.get("toric_vector_bundle_cech_gluing_residual", torch.zeros(())).detach().cpu()
            )
            step_toric_sheaf_cocycle_residual += float(
                out.get("toric_sheaf_cocycle_residual", torch.zeros(())).detach().cpu()
            )
            step_toric_vector_bundle_ray_entropy += float(
                out.get("toric_vector_bundle_ray_entropy", torch.zeros(())).detach().cpu()
            )
            step_toric_vector_bundle_active_ray_mass += float(
                out.get("toric_vector_bundle_active_ray_mass", torch.zeros(())).detach().cpu()
            )
            step_toric_vector_bundle_rank += float(
                out.get("toric_vector_bundle_rank", torch.zeros(())).detach().cpu()
            )
            step_toric_vector_bundle_num_rays += float(
                out.get("toric_vector_bundle_num_rays", torch.zeros(())).detach().cpu()
            )
            step_toric_vector_bundle_filtration_levels += float(
                out.get("toric_vector_bundle_filtration_levels", torch.zeros(())).detach().cpu()
            )
            step_toric_bgg_loss += float(toric_bgg_loss.detach().cpu())
            step_toric_bgg_d2_residual += float(out.get("toric_bgg_d2_residual", torch.zeros(())).detach().cpu())
            step_toric_bgg_resolution_consistency += float(
                out.get("toric_bgg_resolution_consistency", torch.zeros(())).detach().cpu()
            )
            step_toric_bgg_standard_leakage += float(
                out.get("toric_bgg_standard_leakage", torch.zeros(())).detach().cpu()
            )
            step_toric_bgg_standard_allowed_mass += float(
                out.get("toric_bgg_standard_allowed_mass", torch.zeros(())).detach().cpu()
            )
            step_toric_bgg_koszul_linearity_residual += float(
                out.get("toric_bgg_koszul_linearity_residual", torch.zeros(())).detach().cpu()
            )
            step_toric_bgg_gale_dual_consistency += float(
                out.get("toric_bgg_gale_dual_consistency", torch.zeros(())).detach().cpu()
            )
            step_toric_bgg_signature_smoothness += float(
                out.get("toric_bgg_signature_smoothness", torch.zeros(())).detach().cpu()
            )
            step_toric_bgg_standard_entropy += float(
                out.get("toric_bgg_standard_entropy", torch.zeros(())).detach().cpu()
            )
            step_koszul_persistence_loss += float(koszul_persistence_loss.detach().cpu())
            step_koszul_exactness_residual += float(
                out.get("koszul_exactness_residual", torch.zeros(())).detach().cpu()
            )
            step_koszul_syzygy_residual += float(
                out.get("koszul_syzygy_residual", torch.zeros(())).detach().cpu()
            )
            step_koszul_fitting_rank_residual += float(
                out.get("koszul_fitting_rank_residual", torch.zeros(())).detach().cpu()
            )
            step_koszul_be_rank_residual += float(
                out.get("koszul_buchsbaum_eisenbud_rank_residual", torch.zeros(())).detach().cpu()
            )
            step_koszul_be_multiplier_residual += float(
                out.get("koszul_buchsbaum_eisenbud_multiplier_residual", torch.zeros(())).detach().cpu()
            )
            step_koszul_multigraded_betti_mass += float(
                out.get("koszul_multigraded_betti_mass", torch.zeros(())).detach().cpu()
            )
            step_koszul_chart_entropy += float(
                out.get("koszul_toric_affine_chart_entropy", torch.zeros(())).detach().cpu()
            )
            step_koszul_chart_coverage += float(
                out.get("koszul_toric_affine_chart_coverage", torch.zeros(())).detach().cpu()
            )
            step_koszul_chart_transition_shift += float(
                out.get("koszul_chart_transition_resolution_shift", torch.zeros(())).detach().cpu()
            )
            step_koszul_windows += float(out.get("koszul_windows", torch.zeros(())).detach().cpu())
            step_slepian_pollak_loss += float(slepian_pollak_loss.detach().cpu())
            step_slepian_pollak_concentration += float(
                out.get("slepian_pollak_concentration", torch.zeros(())).detach().cpu()
            )
            step_slepian_pollak_leakage += float(out.get("slepian_pollak_leakage", torch.zeros(())).detach().cpu())
            step_slepian_pollak_mode_entropy += float(
                out.get("slepian_pollak_mode_entropy", torch.zeros(())).detach().cpu()
            )
            step_slepian_pollak_effective_modes += float(
                out.get("slepian_pollak_effective_modes", torch.zeros(())).detach().cpu()
            )
            step_slepian_pollak_modes += float(out.get("slepian_pollak_modes", torch.zeros(())).detach().cpu())
            step_slepian_pollak_bandwidth += float(
                out.get("slepian_pollak_bandwidth", torch.zeros(())).detach().cpu()
            )
            step_tokengt_graph_loss += float(tokengt_graph_loss.detach().cpu())
            step_tokengt_graph_edge_bce += float(out.get("tokengt_graph_edge_bce", torch.zeros(())).detach().cpu())
            step_tokengt_graph_direction_loss += float(
                out.get("tokengt_graph_direction_loss", torch.zeros(())).detach().cpu()
            )
            step_tokengt_graph_position_loss += float(
                out.get("tokengt_graph_position_loss", torch.zeros(())).detach().cpu()
            )
            step_tokengt_graph_byte_class_loss += float(
                out.get("tokengt_graph_byte_class_loss", torch.zeros(())).detach().cpu()
            )
            step_tokengt_graph_cycle_loss += float(
                out.get("tokengt_graph_cycle_loss", torch.zeros(())).detach().cpu()
            )
            step_tokengt_graph_edge_density += float(
                out.get("tokengt_graph_edge_density", torch.zeros(())).detach().cpu()
            )
            step_tokengt_graph_causal_edge_fraction += float(
                out.get("tokengt_graph_causal_edge_fraction", torch.zeros(())).detach().cpu()
            )
            step_tokengt_graph_policy_causal += float(
                out.get("tokengt_graph_policy_causal", torch.zeros(())).detach().cpu()
            )
            step_tokengt_graph_fusion_context_norm += float(
                out.get("tokengt_graph_fusion_context_norm", torch.zeros(())).detach().cpu()
            )
            step_tokengt_graph_fusion_gate += float(out.get("tokengt_graph_fusion_gate", torch.zeros(())).detach().cpu())
            step_tokengt_graph_fusion_nodes += float(out.get("tokengt_graph_fusion_nodes", torch.zeros(())).detach().cpu())
            step_tokengt_graph_fusion_edges += float(out.get("tokengt_graph_fusion_edges", torch.zeros(())).detach().cpu())
            step_tokengt_graph_fusion_token_count += float(
                out.get("tokengt_graph_fusion_token_count", torch.zeros(())).detach().cpu()
            )
            step_tokengt_graph_fusion_edge_density += float(
                out.get("tokengt_graph_fusion_edge_density", torch.zeros(())).detach().cpu()
            )
            step_qat_loss += float(qat_loss.detach().cpu())
            step_qat_weight += float(effective_qat_loss_weight)
            step_contrastive_loss += float(contrastive_loss.detach().cpu())
            step_trajectory_flow_loss += float(trajectory_flow_loss.detach().cpu())
            step_trajectory_flow_penalty += float(trajectory_flow_penalty.detach().cpu())
            step_trajectory_kinetic += float(out.get("trajectory_kinetic_energy", torch.zeros(())).detach().cpu())
            step_trajectory_viscous += float(out.get("trajectory_viscous_dissipation", torch.zeros(())).detach().cpu())
            step_trajectory_memory_loss += float(trajectory_memory_loss.detach().cpu())
            step_trajectory_memory_ce += float(out.get("trajectory_memory_ce", torch.zeros(())).detach().cpu())
            step_trajectory_memory_distill_loss += float(
                out.get("trajectory_memory_distill_loss", torch.zeros(())).detach().cpu()
            )
            step_trajectory_memory_quality_loss += float(
                out.get("trajectory_memory_quality_loss", torch.zeros(())).detach().cpu()
            )
            step_trajectory_memory_recall1 += float(out.get("trajectory_memory_recall1", torch.zeros(())).detach().cpu())
            step_trajectory_memory_entropy += float(out.get("trajectory_memory_entropy", torch.zeros(())).detach().cpu())
            step_trajectory_memory_score_gap += float(out.get("trajectory_memory_score_gap", torch.zeros(())).detach().cpu())
            step_trajectory_memory_teacher_diag_prob += float(
                out.get("trajectory_memory_teacher_diag_prob", torch.zeros(())).detach().cpu()
            )
            step_trajectory_memory_persistence_similarity += float(
                out.get("trajectory_memory_persistence_similarity", torch.zeros(())).detach().cpu()
            )
            step_trajectory_memory_persistence_norm += float(
                out.get("trajectory_memory_persistence_norm", torch.zeros(())).detach().cpu()
            )
            step_trajectory_memory_persistence_entropy += float(
                out.get("trajectory_memory_persistence_entropy", torch.zeros(())).detach().cpu()
            )
            step_trajectory_memory_persistence_total += float(
                out.get("trajectory_memory_persistence_total", torch.zeros(())).detach().cpu()
            )
            step_trajectory_memory_persistence_weight += float(
                out.get("trajectory_memory_persistence_weight", torch.zeros(())).detach().cpu()
            )
            step_smear_temperature += float(out.get("smear_temperature", torch.zeros(())).detach().cpu())
            step_revealed_neighbor_known_fraction += float(
                out.get("revealed_neighbor_known_fraction", torch.zeros(())).detach().cpu()
            )
            step_revealed_neighbor_context_norm += float(
                out.get("revealed_neighbor_context_norm", torch.zeros(())).detach().cpu()
            )
            step_revealed_context_prior_weight += float(
                out.get("revealed_context_prior_weight", torch.zeros(())).detach().cpu()
            )
            step_revealed_context_prior_mixture_weight += float(
                out.get("revealed_context_prior_mixture_weight", torch.zeros(())).detach().cpu()
            )
            step_toric_memory_entropy += float(out.get("toric_memory_entropy", torch.zeros(())).detach().cpu())
            step_toric_entropy_loss += float(toric_entropy_loss.detach().cpu())
        step_loss /= grad_accum
        step_neural_loss /= grad_accum
        step_total_loss /= grad_accum
        step_gflownet_loss /= grad_accum
        step_gflownet_entropy /= grad_accum
        step_gflownet_entropy_objective /= grad_accum
        step_gflownet_diversity /= grad_accum
        step_mtp_loss /= grad_accum
        step_graphcg_loss /= grad_accum
        step_graphcg_code_loss /= grad_accum
        step_graphcg_orthogonal_loss /= grad_accum
        step_graphcg_covariance_loss /= grad_accum
        step_graphcg_sparsity_loss /= grad_accum
        step_graphcg_basis_coherence /= grad_accum
        step_graphcg_axis_variance /= grad_accum
        step_analogy_lattice_loss /= grad_accum
        step_analogy_functor_loss /= grad_accum
        step_analogy_parallelogram_loss /= grad_accum
        step_analogy_topology_loss /= grad_accum
        step_analogy_barcode_loss /= grad_accum
        step_analogy_simplex_closure_loss /= grad_accum
        step_analogy_filtration_inclusion_loss /= grad_accum
        step_analogy_chain_map_loss /= grad_accum
        step_analogy_directed_topology_loss /= grad_accum
        step_analogy_directed_transitive_loss /= grad_accum
        step_analogy_directed_cycle_loss /= grad_accum
        step_analogy_directed_chain_map_loss /= grad_accum
        step_analogy_directed_asymmetry /= grad_accum
        step_analogy_directed_skew_norm /= grad_accum
        step_analogy_hdbscan_loss /= grad_accum
        step_analogy_hdbscan_stability /= grad_accum
        step_analogy_hdbscan_persistent_edge_density /= grad_accum
        step_analogy_hdbscan_outlier_score /= grad_accum
        step_analogy_hdbscan_core_radius /= grad_accum
        step_analogy_filtration_edge_density /= grad_accum
        step_analogy_filtration_triangle_density /= grad_accum
        step_analogy_step_topology_loss /= grad_accum
        step_analogy_step_barcode_loss /= grad_accum
        step_analogy_step_simplex_closure_loss /= grad_accum
        step_analogy_step_filtration_inclusion_loss /= grad_accum
        step_analogy_step_boundary_residual /= grad_accum
        step_analogy_step_dirichlet_energy /= grad_accum
        step_analogy_step_dec_conservation_loss /= grad_accum
        step_analogy_step_dec_mass_residual /= grad_accum
        step_analogy_step_dec_vorticity_drift /= grad_accum
        step_analogy_step_dec_kinetic_energy /= grad_accum
        step_analogy_step_dec_kinetic_energy_drift /= grad_accum
        step_analogy_step_dec_hodge_balance /= grad_accum
        step_analogy_step_dec_wedge_interior_residual /= grad_accum
        step_analogy_step_directed_topology_loss /= grad_accum
        step_analogy_step_directed_transitive_loss /= grad_accum
        step_analogy_step_directed_cycle_flux /= grad_accum
        step_analogy_step_directed_chain_commutator /= grad_accum
        step_analogy_step_directed_asymmetry /= grad_accum
        step_analogy_step_analogical_map_loss /= grad_accum
        step_analogy_step_directed_map_loss /= grad_accum
        step_analogy_step_transport_entropy /= grad_accum
        step_analogy_step_hdbscan_loss /= grad_accum
        step_analogy_step_hdbscan_stability /= grad_accum
        step_analogy_step_hdbscan_noise_fraction /= grad_accum
        step_analogy_step_hdbscan_persistent_edge_density /= grad_accum
        step_analogy_step_hdbscan_core_radius /= grad_accum
        step_analogy_step_ph_landscape_loss /= grad_accum
        step_analogy_step_ph_landscape_norm /= grad_accum
        step_analogy_step_ph_image_energy /= grad_accum
        step_analogy_step_edge_density /= grad_accum
        step_analogy_step_triangle_density /= grad_accum
        step_analogy_step_cycle_rank /= grad_accum
        step_analogy_step_betti0 /= grad_accum
        step_analogy_step_windows /= grad_accum
        step_analogy_basis_loss /= grad_accum
        step_analogy_axis_entropy /= grad_accum
        step_analogy_lattice_margin /= grad_accum
        step_analogy_relation_groups /= grad_accum
        step_analogy_topology_groups /= grad_accum
        step_analogy_graphcg_chart_dim /= grad_accum
        step_analogy_graphcg_chart_energy /= grad_accum
        step_qat_loss /= grad_accum
        step_qat_weight /= grad_accum
        step_contrastive_loss /= grad_accum
        step_trajectory_flow_loss /= grad_accum
        step_trajectory_flow_penalty /= grad_accum
        step_trajectory_kinetic /= grad_accum
        step_trajectory_viscous /= grad_accum
        step_trajectory_memory_loss /= grad_accum
        step_trajectory_memory_ce /= grad_accum
        step_trajectory_memory_distill_loss /= grad_accum
        step_trajectory_memory_quality_loss /= grad_accum
        step_trajectory_memory_recall1 /= grad_accum
        step_trajectory_memory_entropy /= grad_accum
        step_trajectory_memory_score_gap /= grad_accum
        step_trajectory_memory_teacher_diag_prob /= grad_accum
        step_trajectory_memory_persistence_similarity /= grad_accum
        step_trajectory_memory_persistence_norm /= grad_accum
        step_trajectory_memory_persistence_entropy /= grad_accum
        step_trajectory_memory_persistence_total /= grad_accum
        step_trajectory_memory_persistence_weight /= grad_accum
        step_smear_temperature /= grad_accum
        step_revealed_neighbor_known_fraction /= grad_accum
        step_revealed_neighbor_context_norm /= grad_accum
        step_revealed_context_prior_weight /= grad_accum
        step_revealed_context_prior_mixture_weight /= grad_accum
        step_toric_memory_entropy /= grad_accum
        step_toric_entropy_loss /= grad_accum
        step_toric_geometry_loss /= grad_accum
        step_toric_fan_loss /= grad_accum
        step_toric_active_face_ce /= grad_accum
        step_toric_active_face_margin /= grad_accum
        step_toric_active_face_entropy /= grad_accum
        step_toric_bend_loss /= grad_accum
        step_toric_bend_magnitude /= grad_accum
        step_toric_binomial_loss /= grad_accum
        step_toric_binomial_residual /= grad_accum
        step_toric_binomial_relation_source_exact /= grad_accum
        step_toric_moment_loss /= grad_accum
        step_toric_coxeter_loss /= grad_accum
        step_toric_affine_wall_distance /= grad_accum
        step_toric_braid_loss /= grad_accum
        step_toric_leaf_residual /= grad_accum
        step_toric_probe_rank /= grad_accum
        step_toric_probe_quant_bits /= grad_accum
        step_toric_vector_bundle_loss /= grad_accum
        step_toric_vector_bundle_ray_ce /= grad_accum
        step_toric_vector_bundle_level_ce /= grad_accum
        step_toric_vector_bundle_filtration_residual /= grad_accum
        step_toric_vector_bundle_nesting_residual /= grad_accum
        step_toric_vector_bundle_splitting_residual /= grad_accum
        step_toric_vector_bundle_cech_gluing_residual /= grad_accum
        step_toric_sheaf_cocycle_residual /= grad_accum
        step_toric_vector_bundle_ray_entropy /= grad_accum
        step_toric_vector_bundle_active_ray_mass /= grad_accum
        step_toric_vector_bundle_rank /= grad_accum
        step_toric_vector_bundle_num_rays /= grad_accum
        step_toric_vector_bundle_filtration_levels /= grad_accum
        step_toric_bgg_loss /= grad_accum
        step_toric_bgg_d2_residual /= grad_accum
        step_toric_bgg_resolution_consistency /= grad_accum
        step_toric_bgg_standard_leakage /= grad_accum
        step_toric_bgg_standard_allowed_mass /= grad_accum
        step_toric_bgg_koszul_linearity_residual /= grad_accum
        step_toric_bgg_gale_dual_consistency /= grad_accum
        step_toric_bgg_signature_smoothness /= grad_accum
        step_toric_bgg_standard_entropy /= grad_accum
        step_koszul_persistence_loss /= grad_accum
        step_koszul_exactness_residual /= grad_accum
        step_koszul_syzygy_residual /= grad_accum
        step_koszul_fitting_rank_residual /= grad_accum
        step_koszul_be_rank_residual /= grad_accum
        step_koszul_be_multiplier_residual /= grad_accum
        step_koszul_multigraded_betti_mass /= grad_accum
        step_koszul_chart_entropy /= grad_accum
        step_koszul_chart_coverage /= grad_accum
        step_koszul_chart_transition_shift /= grad_accum
        step_koszul_windows /= grad_accum
        step_slepian_pollak_loss /= grad_accum
        step_slepian_pollak_concentration /= grad_accum
        step_slepian_pollak_leakage /= grad_accum
        step_slepian_pollak_mode_entropy /= grad_accum
        step_slepian_pollak_effective_modes /= grad_accum
        step_slepian_pollak_modes /= grad_accum
        step_slepian_pollak_bandwidth /= grad_accum
        step_tokengt_graph_loss /= grad_accum
        step_tokengt_graph_edge_bce /= grad_accum
        step_tokengt_graph_direction_loss /= grad_accum
        step_tokengt_graph_position_loss /= grad_accum
        step_tokengt_graph_byte_class_loss /= grad_accum
        step_tokengt_graph_cycle_loss /= grad_accum
        step_tokengt_graph_edge_density /= grad_accum
        step_tokengt_graph_causal_edge_fraction /= grad_accum
        step_tokengt_graph_policy_causal /= grad_accum
        step_tokengt_graph_fusion_context_norm /= grad_accum
        step_tokengt_graph_fusion_gate /= grad_accum
        step_tokengt_graph_fusion_nodes /= grad_accum
        step_tokengt_graph_fusion_edges /= grad_accum
        step_tokengt_graph_fusion_token_count /= grad_accum
        step_tokengt_graph_fusion_edge_density /= grad_accum
        step_robust_micro_loss_guard_fraction /= grad_accum
        step_robust_micro_loss_guard_scale /= grad_accum
        step_robust_micro_loss_guard_cap /= grad_accum
        step_medium_microbatches /= grad_accum
        step_complex_microbatches /= grad_accum
        step_fineweb_microbatches /= grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=effective_grad_clip_norm if effective_grad_clip_norm > 0 else float("inf"),
        )
        grad_norm_is_finite = bool(torch.isfinite(grad_norm).detach().cpu())
        step_loss_is_finite = math.isfinite(float(step_loss))
        nonfinite_update_skip = 0.0
        shock_guard_active = 0.0
        shock_guard_scale = 1.0
        shock_guard_loss_delta_value = 0.0
        shock_guard_loss_ratio_value = 1.0
        if not grad_norm_is_finite or not step_loss_is_finite:
            nonfinite_update_skip = 1.0
            optimizer.zero_grad(set_to_none=True)
        elif shock_guard_enabled and running_loss > 0.0:
            in_shock_window = step >= shock_guard_start_step and (
                shock_guard_end_step <= 0 or step < shock_guard_end_step
            )
            shock_guard_loss_delta_value = float(step_loss - running_loss)
            shock_guard_loss_ratio_value = float(step_loss / max(running_loss, 1e-9))
            grad_norm_value = float(grad_norm.detach().cpu())
            loss_is_shock = (
                shock_guard_loss_delta_value >= shock_guard_loss_delta
                or shock_guard_loss_ratio_value >= shock_guard_loss_ratio
            )
            if in_shock_window and loss_is_shock and grad_norm_value >= shock_guard_grad_norm:
                shock_guard_active = 1.0
                shock_guard_scale = max(0.0, min(1.0, shock_guard_update_scale))
                if shock_guard_scale < 1.0:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(shock_guard_scale)
        if nonfinite_update_skip < 0.5:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        finite_step_loss = float(step_loss) if step_loss_is_finite else float(running_loss or 0.0)
        running_loss = 0.97 * running_loss + 0.03 * finite_step_loss if running_loss else finite_step_loss
        bpb = finite_step_loss / math.log(2) if math.isfinite(finite_step_loss) else float("nan")
        neural_bpb = step_neural_loss / math.log(2)
        progress.set_postfix(loss=f"{finite_step_loss:.4f}", bpb=f"{bpb:.3f}", gfn=f"{step_gflownet_loss:.3f}", lr=f"{lr_step:.2e}")
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
            complexity_metrics["trainer/step"] = float(step)

        if step % log_interval == 0:
            metrics = {
                "trainer/step": float(step),
                "train/loss": finite_step_loss,
                "train/neural_loss": step_neural_loss,
                "train/total_loss": step_total_loss,
                "train/loss_ema": running_loss,
                "train/bpb": bpb,
                "train/neural_bpb": neural_bpb,
                "train/lr": lr_step,
                "train/grad_norm": float(torch.nan_to_num(grad_norm.detach().cpu(), nan=0.0, posinf=1.0e9, neginf=0.0)),
                "train/grad_norm_finite": float(grad_norm_is_finite),
                "train/nonfinite_update_skip": nonfinite_update_skip,
                "train/shock_guard_active": shock_guard_active,
                "train/shock_guard_update_scale": shock_guard_scale,
                "train/shock_guard_loss_delta": shock_guard_loss_delta_value,
                "train/shock_guard_loss_ratio": shock_guard_loss_ratio_value,
                "train/robust_micro_loss_guard_fraction": step_robust_micro_loss_guard_fraction,
                "train/robust_micro_loss_guard_scale": step_robust_micro_loss_guard_scale,
                "train/robust_micro_loss_guard_cap": step_robust_micro_loss_guard_cap,
                "train/gflownet_loss": step_gflownet_loss,
                "train/gflownet_entropy": step_gflownet_entropy,
                "train/gflownet_entropy_objective": step_gflownet_entropy_objective,
                "train/gflownet_entropy_target": float(effective_gflownet_entropy_target),
                "train/gflownet_action_diversity": step_gflownet_diversity,
                "train/gflownet_loss_weight": effective_gflownet_loss_weight,
                "train/gflownet_entropy_weight": effective_gflownet_entropy_weight,
                "train/mtp_loss": step_mtp_loss,
                "train/mtp_loss_weight": effective_mtp_loss_weight,
                "train/graphcg_loss": step_graphcg_loss,
                "train/graphcg_code_loss": step_graphcg_code_loss,
                "train/graphcg_orthogonal_loss": step_graphcg_orthogonal_loss,
                "train/graphcg_covariance_loss": step_graphcg_covariance_loss,
                "train/graphcg_sparsity_loss": step_graphcg_sparsity_loss,
                "train/graphcg_basis_coherence": step_graphcg_basis_coherence,
                "train/graphcg_axis_variance": step_graphcg_axis_variance,
                "train/graphcg_loss_weight": effective_graphcg_loss_weight,
                "train/analogy_lattice_loss": step_analogy_lattice_loss,
                "train/analogy_functor_loss": step_analogy_functor_loss,
                "train/analogy_parallelogram_loss": step_analogy_parallelogram_loss,
                "train/analogy_topology_loss": step_analogy_topology_loss,
                "train/analogy_barcode_loss": step_analogy_barcode_loss,
                "train/analogy_simplex_closure_loss": step_analogy_simplex_closure_loss,
                "train/analogy_filtration_inclusion_loss": step_analogy_filtration_inclusion_loss,
                "train/analogy_chain_map_loss": step_analogy_chain_map_loss,
                "train/analogy_directed_topology_loss": step_analogy_directed_topology_loss,
                "train/analogy_directed_transitive_loss": step_analogy_directed_transitive_loss,
                "train/analogy_directed_cycle_loss": step_analogy_directed_cycle_loss,
                "train/analogy_directed_chain_map_loss": step_analogy_directed_chain_map_loss,
                "train/analogy_directed_asymmetry": step_analogy_directed_asymmetry,
                "train/analogy_directed_skew_norm": step_analogy_directed_skew_norm,
                "train/analogy_hdbscan_loss": step_analogy_hdbscan_loss,
                "train/analogy_hdbscan_stability": step_analogy_hdbscan_stability,
                "train/analogy_hdbscan_persistent_edge_density": step_analogy_hdbscan_persistent_edge_density,
                "train/analogy_hdbscan_outlier_score": step_analogy_hdbscan_outlier_score,
                "train/analogy_hdbscan_core_radius": step_analogy_hdbscan_core_radius,
                "train/analogy_filtration_edge_density": step_analogy_filtration_edge_density,
                "train/analogy_filtration_triangle_density": step_analogy_filtration_triangle_density,
                "train/analogy_step_topology_loss": step_analogy_step_topology_loss,
                "train/analogy_step_barcode_loss": step_analogy_step_barcode_loss,
                "train/analogy_step_simplex_closure_loss": step_analogy_step_simplex_closure_loss,
                "train/analogy_step_filtration_inclusion_loss": step_analogy_step_filtration_inclusion_loss,
                "train/analogy_step_boundary_residual": step_analogy_step_boundary_residual,
                "train/analogy_step_dirichlet_energy": step_analogy_step_dirichlet_energy,
                "train/analogy_step_dec_conservation_loss": step_analogy_step_dec_conservation_loss,
                "train/analogy_step_dec_mass_residual": step_analogy_step_dec_mass_residual,
                "train/analogy_step_dec_vorticity_drift": step_analogy_step_dec_vorticity_drift,
                "train/analogy_step_dec_kinetic_energy": step_analogy_step_dec_kinetic_energy,
                "train/analogy_step_dec_kinetic_energy_drift": step_analogy_step_dec_kinetic_energy_drift,
                "train/analogy_step_dec_hodge_balance": step_analogy_step_dec_hodge_balance,
                "train/analogy_step_dec_wedge_interior_residual": step_analogy_step_dec_wedge_interior_residual,
                "train/analogy_step_directed_topology_loss": step_analogy_step_directed_topology_loss,
                "train/analogy_step_directed_transitive_loss": step_analogy_step_directed_transitive_loss,
                "train/analogy_step_directed_cycle_flux": step_analogy_step_directed_cycle_flux,
                "train/analogy_step_directed_chain_commutator": step_analogy_step_directed_chain_commutator,
                "train/analogy_step_directed_asymmetry": step_analogy_step_directed_asymmetry,
                "train/analogy_step_analogical_map_loss": step_analogy_step_analogical_map_loss,
                "train/analogy_step_directed_map_loss": step_analogy_step_directed_map_loss,
                "train/analogy_step_transport_entropy": step_analogy_step_transport_entropy,
                "train/analogy_step_hdbscan_loss": step_analogy_step_hdbscan_loss,
                "train/analogy_step_hdbscan_stability": step_analogy_step_hdbscan_stability,
                "train/analogy_step_hdbscan_noise_fraction": step_analogy_step_hdbscan_noise_fraction,
                "train/analogy_step_hdbscan_persistent_edge_density": step_analogy_step_hdbscan_persistent_edge_density,
                "train/analogy_step_hdbscan_core_radius": step_analogy_step_hdbscan_core_radius,
                "train/analogy_step_ph_landscape_loss": step_analogy_step_ph_landscape_loss,
                "train/analogy_step_ph_landscape_norm": step_analogy_step_ph_landscape_norm,
                "train/analogy_step_ph_image_energy": step_analogy_step_ph_image_energy,
                "train/analogy_step_edge_density": step_analogy_step_edge_density,
                "train/analogy_step_triangle_density": step_analogy_step_triangle_density,
                "train/analogy_step_cycle_rank": step_analogy_step_cycle_rank,
                "train/analogy_step_betti0": step_analogy_step_betti0,
                "train/analogy_step_windows": step_analogy_step_windows,
                "train/analogy_basis_loss": step_analogy_basis_loss,
                "train/analogy_axis_entropy": step_analogy_axis_entropy,
                "train/analogy_lattice_margin": step_analogy_lattice_margin,
                "train/analogy_relation_groups": step_analogy_relation_groups,
                "train/analogy_topology_groups": step_analogy_topology_groups,
                "train/analogy_graphcg_chart_dim": step_analogy_graphcg_chart_dim,
                "train/analogy_graphcg_chart_energy": step_analogy_graphcg_chart_energy,
                "topology/lattice_loss": step_analogy_lattice_loss,
                "topology/functor_loss": step_analogy_functor_loss,
                "topology/parallelogram_loss": step_analogy_parallelogram_loss,
                "topology/persistence_loss": step_analogy_topology_loss,
                "topology/barcode_loss": step_analogy_barcode_loss,
                "topology/simplex_closure_loss": step_analogy_simplex_closure_loss,
                "topology/filtration_inclusion_loss": step_analogy_filtration_inclusion_loss,
                "topology/chain_map_loss": step_analogy_chain_map_loss,
                "topology/directed_loss": step_analogy_directed_topology_loss,
                "topology/directed_transitive_loss": step_analogy_directed_transitive_loss,
                "topology/directed_cycle_loss": step_analogy_directed_cycle_loss,
                "topology/directed_chain_map_loss": step_analogy_directed_chain_map_loss,
                "topology/directed_asymmetry": step_analogy_directed_asymmetry,
                "topology/directed_skew_norm": step_analogy_directed_skew_norm,
                "topology/hdbscan_loss": step_analogy_hdbscan_loss,
                "topology/hdbscan_stability": step_analogy_hdbscan_stability,
                "topology/hdbscan_persistent_edge_density": step_analogy_hdbscan_persistent_edge_density,
                "topology/hdbscan_outlier_score": step_analogy_hdbscan_outlier_score,
                "topology/hdbscan_core_radius": step_analogy_hdbscan_core_radius,
                "topology/filtration_edge_density": step_analogy_filtration_edge_density,
                "topology/filtration_triangle_density": step_analogy_filtration_triangle_density,
                "topology/step_loss": step_analogy_step_topology_loss,
                "topology/step_barcode_loss": step_analogy_step_barcode_loss,
                "topology/step_boundary_residual": step_analogy_step_boundary_residual,
                "topology/step_dirichlet_energy": step_analogy_step_dirichlet_energy,
                "topology/dec_conservation_loss": step_analogy_step_dec_conservation_loss,
                "topology/dec_mass_residual": step_analogy_step_dec_mass_residual,
                "topology/dec_vorticity_drift": step_analogy_step_dec_vorticity_drift,
                "topology/dec_kinetic_energy": step_analogy_step_dec_kinetic_energy,
                "topology/dec_kinetic_energy_drift": step_analogy_step_dec_kinetic_energy_drift,
                "topology/dec_hodge_balance": step_analogy_step_dec_hodge_balance,
                "topology/dec_wedge_interior_residual": step_analogy_step_dec_wedge_interior_residual,
                "topology/step_directed_loss": step_analogy_step_directed_topology_loss,
                "topology/step_directed_transitive_loss": step_analogy_step_directed_transitive_loss,
                "topology/step_directed_cycle_flux": step_analogy_step_directed_cycle_flux,
                "topology/step_directed_chain_commutator": step_analogy_step_directed_chain_commutator,
                "topology/step_directed_asymmetry": step_analogy_step_directed_asymmetry,
                "topology/step_analogical_map_loss": step_analogy_step_analogical_map_loss,
                "topology/step_directed_map_loss": step_analogy_step_directed_map_loss,
                "topology/step_transport_entropy": step_analogy_step_transport_entropy,
                "topology/step_hdbscan_loss": step_analogy_step_hdbscan_loss,
                "topology/step_hdbscan_stability": step_analogy_step_hdbscan_stability,
                "topology/step_hdbscan_noise_fraction": step_analogy_step_hdbscan_noise_fraction,
                "topology/step_hdbscan_persistent_edge_density": step_analogy_step_hdbscan_persistent_edge_density,
                "topology/step_hdbscan_core_radius": step_analogy_step_hdbscan_core_radius,
                "topology/ph_landscape_loss": step_analogy_step_ph_landscape_loss,
                "topology/ph_landscape_norm": step_analogy_step_ph_landscape_norm,
                "topology/ph_image_energy": step_analogy_step_ph_image_energy,
                "topology/step_edge_density": step_analogy_step_edge_density,
                "topology/step_triangle_density": step_analogy_step_triangle_density,
                "topology/step_cycle_rank": step_analogy_step_cycle_rank,
                "topology/step_betti0": step_analogy_step_betti0,
                "topology/step_windows": step_analogy_step_windows,
                "graphcg/basis_loss": step_analogy_basis_loss,
                "graphcg/axis_entropy": step_analogy_axis_entropy,
                "graphcg/lattice_margin": step_analogy_lattice_margin,
                "graphcg/relation_groups": step_analogy_relation_groups,
                "graphcg/topology_groups": step_analogy_topology_groups,
                "graphcg/chart_dim": step_analogy_graphcg_chart_dim,
                "graphcg/chart_energy": step_analogy_graphcg_chart_energy,
                "train/analogy_lattice_loss_weight": effective_analogy_lattice_loss_weight,
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
                "train/trajectory_memory_loss": step_trajectory_memory_loss,
                "train/trajectory_memory_ce": step_trajectory_memory_ce,
                "train/trajectory_memory_distill_loss": step_trajectory_memory_distill_loss,
                "train/trajectory_memory_quality_loss": step_trajectory_memory_quality_loss,
                "train/trajectory_memory_recall1": step_trajectory_memory_recall1,
                "train/trajectory_memory_entropy": step_trajectory_memory_entropy,
                "train/trajectory_memory_score_gap": step_trajectory_memory_score_gap,
                "train/trajectory_memory_teacher_diag_prob": step_trajectory_memory_teacher_diag_prob,
                "train/trajectory_memory_persistence_similarity": step_trajectory_memory_persistence_similarity,
                "train/trajectory_memory_persistence_norm": step_trajectory_memory_persistence_norm,
                "train/trajectory_memory_persistence_entropy": step_trajectory_memory_persistence_entropy,
                "train/trajectory_memory_persistence_total": step_trajectory_memory_persistence_total,
                "train/trajectory_memory_persistence_weight": step_trajectory_memory_persistence_weight,
                "trajectory_memory/persistence_similarity": step_trajectory_memory_persistence_similarity,
                "trajectory_memory/persistence_norm": step_trajectory_memory_persistence_norm,
                "trajectory_memory/persistence_entropy": step_trajectory_memory_persistence_entropy,
                "trajectory_memory/persistence_total": step_trajectory_memory_persistence_total,
                "trajectory_memory/persistence_weight": step_trajectory_memory_persistence_weight,
                "train/trajectory_memory_loss_weight": effective_trajectory_memory_loss_weight,
                "train/smear_temperature": step_smear_temperature,
                "train/revealed_neighbor_known_fraction": step_revealed_neighbor_known_fraction,
                "train/revealed_neighbor_context_norm": step_revealed_neighbor_context_norm,
                "train/revealed_context_prior_weight": step_revealed_context_prior_weight,
                "train/revealed_context_prior_mixture_weight": step_revealed_context_prior_mixture_weight,
                "train/toric_memory_entropy": step_toric_memory_entropy,
                "train/toric_entropy_floor": float(toric_entropy_floor),
                "train/toric_entropy_loss": step_toric_entropy_loss,
                "train/toric_entropy_loss_weight": float(effective_toric_entropy_loss_weight),
                "train/toric_geometry_loss": step_toric_geometry_loss,
                "train/toric_geometry_loss_weight": float(effective_toric_geometry_loss_weight),
                "train/toric_fan_loss": step_toric_fan_loss,
                "train/toric_active_face_ce": step_toric_active_face_ce,
                "train/toric_active_face_margin": step_toric_active_face_margin,
                "train/toric_active_face_entropy": step_toric_active_face_entropy,
                "train/toric_bend_loss": step_toric_bend_loss,
                "train/toric_bend_magnitude": step_toric_bend_magnitude,
                "train/toric_binomial_loss": step_toric_binomial_loss,
                "train/toric_binomial_residual": step_toric_binomial_residual,
                "train/toric_binomial_relation_source_exact": step_toric_binomial_relation_source_exact,
                "train/toric_moment_loss": step_toric_moment_loss,
                "train/toric_coxeter_loss": step_toric_coxeter_loss,
                "train/toric_affine_wall_distance": step_toric_affine_wall_distance,
                "train/toric_braid_loss": step_toric_braid_loss,
                "train/toric_leaf_residual": step_toric_leaf_residual,
                "train/toric_probe_rank": step_toric_probe_rank,
                "train/toric_probe_quant_bits": step_toric_probe_quant_bits,
                "toric/memory_entropy": step_toric_memory_entropy,
                "toric/entropy_floor": float(toric_entropy_floor),
                "toric/entropy_loss": step_toric_entropy_loss,
                "toric/entropy_loss_weight": float(effective_toric_entropy_loss_weight),
                "toric/geometry_loss": step_toric_geometry_loss,
                "toric/geometry_loss_weight": float(effective_toric_geometry_loss_weight),
                "toric/fan_loss": step_toric_fan_loss,
                "toric/active_face_ce": step_toric_active_face_ce,
                "toric/active_face_margin": step_toric_active_face_margin,
                "toric/active_face_entropy": step_toric_active_face_entropy,
                "toric/bend_loss": step_toric_bend_loss,
                "toric/bend_magnitude": step_toric_bend_magnitude,
                "toric/binomial_loss": step_toric_binomial_loss,
                "toric/binomial_residual": step_toric_binomial_residual,
                "toric/binomial_relation_source_exact": step_toric_binomial_relation_source_exact,
                "toric/moment_loss": step_toric_moment_loss,
                "toric/coxeter_loss": step_toric_coxeter_loss,
                "toric/affine_wall_distance": step_toric_affine_wall_distance,
                "toric/braid_loss": step_toric_braid_loss,
                "toric/leaf_residual": step_toric_leaf_residual,
                "toric/probe_rank": step_toric_probe_rank,
                "toric/probe_quant_bits": step_toric_probe_quant_bits,
                "tropical/active_face_ce": step_toric_active_face_ce,
                "tropical/active_face_margin": step_toric_active_face_margin,
                "tropical/active_face_entropy": step_toric_active_face_entropy,
                "tropical/fan_loss": step_toric_fan_loss,
                "tropical/bend_magnitude": step_toric_bend_magnitude,
                "train/toric_vector_bundle_loss": step_toric_vector_bundle_loss,
                "train/toric_vector_bundle_loss_weight": float(effective_toric_vector_bundle_loss_weight),
                "train/toric_vector_bundle_ray_ce": step_toric_vector_bundle_ray_ce,
                "train/toric_vector_bundle_level_ce": step_toric_vector_bundle_level_ce,
                "train/toric_vector_bundle_filtration_residual": step_toric_vector_bundle_filtration_residual,
                "train/toric_vector_bundle_klyachko_nesting_residual": step_toric_vector_bundle_nesting_residual,
                "train/toric_vector_bundle_cone_splitting_residual": step_toric_vector_bundle_splitting_residual,
                "train/toric_vector_bundle_cech_gluing_residual": step_toric_vector_bundle_cech_gluing_residual,
                "train/toric_sheaf_cocycle_residual": step_toric_sheaf_cocycle_residual,
                "train/toric_vector_bundle_ray_entropy": step_toric_vector_bundle_ray_entropy,
                "train/toric_vector_bundle_active_ray_mass": step_toric_vector_bundle_active_ray_mass,
                "toric_vector_bundle/loss": step_toric_vector_bundle_loss,
                "toric_vector_bundle/loss_weight": float(effective_toric_vector_bundle_loss_weight),
                "toric_vector_bundle/ray_ce": step_toric_vector_bundle_ray_ce,
                "toric_vector_bundle/level_ce": step_toric_vector_bundle_level_ce,
                "toric_vector_bundle/filtration_residual": step_toric_vector_bundle_filtration_residual,
                "toric_vector_bundle/klyachko_nesting_residual": step_toric_vector_bundle_nesting_residual,
                "toric_vector_bundle/cone_splitting_residual": step_toric_vector_bundle_splitting_residual,
                "toric_vector_bundle/cech_gluing_residual": step_toric_vector_bundle_cech_gluing_residual,
                "toric_vector_bundle/ray_entropy": step_toric_vector_bundle_ray_entropy,
                "toric_vector_bundle/active_ray_mass": step_toric_vector_bundle_active_ray_mass,
                "toric_vector_bundle/rank": step_toric_vector_bundle_rank,
                "toric_vector_bundle/num_rays": step_toric_vector_bundle_num_rays,
                "toric_vector_bundle/filtration_levels": step_toric_vector_bundle_filtration_levels,
                "toric_sheaf/chart_gluing_residual": step_toric_vector_bundle_cech_gluing_residual,
                "toric_sheaf/cocycle_residual": step_toric_sheaf_cocycle_residual,
                "train/toric_bgg_loss": step_toric_bgg_loss,
                "train/toric_bgg_loss_weight": float(effective_toric_bgg_loss_weight),
                "train/toric_bgg_d2_residual": step_toric_bgg_d2_residual,
                "train/toric_bgg_resolution_consistency": step_toric_bgg_resolution_consistency,
                "train/toric_bgg_standard_leakage": step_toric_bgg_standard_leakage,
                "train/toric_bgg_standard_allowed_mass": step_toric_bgg_standard_allowed_mass,
                "train/toric_bgg_koszul_linearity_residual": step_toric_bgg_koszul_linearity_residual,
                "train/toric_bgg_gale_dual_consistency": step_toric_bgg_gale_dual_consistency,
                "train/toric_bgg_signature_smoothness": step_toric_bgg_signature_smoothness,
                "train/toric_bgg_standard_entropy": step_toric_bgg_standard_entropy,
                "bgg_category_o/loss": step_toric_bgg_loss,
                "bgg_category_o/loss_weight": float(effective_toric_bgg_loss_weight),
                "bgg_category_o/d2_residual": step_toric_bgg_d2_residual,
                "bgg_category_o/resolution_consistency": step_toric_bgg_resolution_consistency,
                "bgg_category_o/standard_leakage": step_toric_bgg_standard_leakage,
                "bgg_category_o/standard_allowed_mass": step_toric_bgg_standard_allowed_mass,
                "bgg_category_o/koszul_linearity_residual": step_toric_bgg_koszul_linearity_residual,
                "bgg_category_o/gale_dual_consistency": step_toric_bgg_gale_dual_consistency,
                "bgg_category_o/signature_smoothness": step_toric_bgg_signature_smoothness,
                "bgg_category_o/standard_entropy": step_toric_bgg_standard_entropy,
                "category_o/standard_leakage": step_toric_bgg_standard_leakage,
                "category_o/standard_allowed_mass": step_toric_bgg_standard_allowed_mass,
                "category_o/resolution_consistency": step_toric_bgg_resolution_consistency,
                "category_o/d2_residual": step_toric_bgg_d2_residual,
                "category_o/koszul_linearity_residual": step_toric_bgg_koszul_linearity_residual,
                "category_o/gale_dual_consistency": step_toric_bgg_gale_dual_consistency,
                "train/koszul_persistence_loss": step_koszul_persistence_loss,
                "train/koszul_persistence_loss_weight": float(effective_koszul_persistence_loss_weight),
                "train/koszul_exactness_residual": step_koszul_exactness_residual,
                "train/koszul_syzygy_residual": step_koszul_syzygy_residual,
                "train/koszul_fitting_rank_residual": step_koszul_fitting_rank_residual,
                "train/koszul_buchsbaum_eisenbud_rank_residual": step_koszul_be_rank_residual,
                "train/koszul_buchsbaum_eisenbud_multiplier_residual": step_koszul_be_multiplier_residual,
                "train/koszul_multigraded_betti_mass": step_koszul_multigraded_betti_mass,
                "train/koszul_toric_affine_chart_entropy": step_koszul_chart_entropy,
                "train/koszul_toric_affine_chart_coverage": step_koszul_chart_coverage,
                "train/koszul_chart_transition_resolution_shift": step_koszul_chart_transition_shift,
                "train/koszul_windows": step_koszul_windows,
                "koszul_persistence/loss": step_koszul_persistence_loss,
                "koszul_persistence/loss_weight": float(effective_koszul_persistence_loss_weight),
                "koszul_persistence/exactness_residual": step_koszul_exactness_residual,
                "koszul_persistence/syzygy_residual": step_koszul_syzygy_residual,
                "koszul_persistence/fitting_rank_residual": step_koszul_fitting_rank_residual,
                "koszul_persistence/buchsbaum_eisenbud_rank_residual": step_koszul_be_rank_residual,
                "koszul_persistence/buchsbaum_eisenbud_multiplier_residual": step_koszul_be_multiplier_residual,
                "koszul_persistence/multigraded_betti_mass": step_koszul_multigraded_betti_mass,
                "koszul_persistence/toric_affine_chart_entropy": step_koszul_chart_entropy,
                "koszul_persistence/toric_affine_chart_coverage": step_koszul_chart_coverage,
                "koszul_persistence/chart_transition_resolution_shift": step_koszul_chart_transition_shift,
                "koszul_persistence/windows": step_koszul_windows,
                "train/slepian_pollak_loss": step_slepian_pollak_loss,
                "train/slepian_pollak_loss_weight": float(effective_slepian_pollak_loss_weight),
                "slepian_pollak/loss": step_slepian_pollak_loss,
                "slepian_pollak/loss_weight": float(effective_slepian_pollak_loss_weight),
                "slepian_pollak/concentration": step_slepian_pollak_concentration,
                "slepian_pollak/leakage": step_slepian_pollak_leakage,
                "slepian_pollak/mode_entropy": step_slepian_pollak_mode_entropy,
                "slepian_pollak/effective_modes": step_slepian_pollak_effective_modes,
                "slepian_pollak/modes": step_slepian_pollak_modes,
                "slepian_pollak/bandwidth": step_slepian_pollak_bandwidth,
                "toric/slepian_concentration": step_slepian_pollak_concentration,
                "toric/slepian_leakage": step_slepian_pollak_leakage,
                "toric/slepian_mode_entropy": step_slepian_pollak_mode_entropy,
                "toric/slepian_effective_modes": step_slepian_pollak_effective_modes,
                "diagnostics/latest/pollak_prolate_slepian_concentration": step_slepian_pollak_concentration,
                "diagnostics/latest/pollak_prolate_slepian_leakage": step_slepian_pollak_leakage,
                "train/tokengt_graph_loss": step_tokengt_graph_loss,
                "train/tokengt_graph_loss_weight": float(effective_tokengt_graph_loss_weight),
                "train/tokengt_graph_edge_bce": step_tokengt_graph_edge_bce,
                "train/tokengt_graph_direction_loss": step_tokengt_graph_direction_loss,
                "train/tokengt_graph_position_loss": step_tokengt_graph_position_loss,
                "train/tokengt_graph_byte_class_loss": step_tokengt_graph_byte_class_loss,
                "train/tokengt_graph_cycle_loss": step_tokengt_graph_cycle_loss,
                "tokengt_graph/loss": step_tokengt_graph_loss,
                "tokengt_graph/loss_weight": float(effective_tokengt_graph_loss_weight),
                "tokengt_graph/edge_bce": step_tokengt_graph_edge_bce,
                "tokengt_graph/direction_loss": step_tokengt_graph_direction_loss,
                "tokengt_graph/position_loss": step_tokengt_graph_position_loss,
                "tokengt_graph/byte_class_loss": step_tokengt_graph_byte_class_loss,
                "tokengt_graph/cycle_loss": step_tokengt_graph_cycle_loss,
                "tokengt_graph/edge_density": step_tokengt_graph_edge_density,
                "tokengt_graph/causal_edge_fraction": step_tokengt_graph_causal_edge_fraction,
                "tokengt_graph/policy_causal": step_tokengt_graph_policy_causal,
                "tokengt_graph/fusion_context_norm": step_tokengt_graph_fusion_context_norm,
                "tokengt_graph/fusion_gate": step_tokengt_graph_fusion_gate,
                "tokengt_graph/fusion_nodes": step_tokengt_graph_fusion_nodes,
                "tokengt_graph/fusion_edges": step_tokengt_graph_fusion_edges,
                "tokengt_graph/fusion_token_count": step_tokengt_graph_fusion_token_count,
                "tokengt_graph/fusion_edge_density": step_tokengt_graph_fusion_edge_density,
                "train/tokengt_graph_fusion_context_norm": step_tokengt_graph_fusion_context_norm,
                "train/tokengt_graph_fusion_gate": step_tokengt_graph_fusion_gate,
                "artifact/initial_bytes": report.bytes_total,
                "artifact/estimated_tensor_bytes": estimated_tensor_bytes,
                "artifact/deployment_parameters": report.deployment_parameters,
                "artifact/excluded_tensors": report.excluded_tensors,
                "model/parameters": params,
                "data/coprime_row_stride": float(coprime_row_stride),
                "data/stream_origin_step": float(stream_origin_step),
                "data/stream_burnin_steps": float(stream_burnin_steps),
                "data/difficulty_curriculum_active": float(difficulty_curriculum_enabled),
                "data/easy_microbatch_fraction": max(
                    0.0,
                    1.0
                    - float(step_fineweb_microbatches)
                    - float(step_medium_microbatches)
                    - float(step_complex_microbatches),
                ),
                "data/fineweb_calibration_active": float(fineweb_iterator is not None),
                "data/full_curated_train_split_active": float(
                    bool(curated_train_files and not task_family_keywords and not dataset_keywords)
                ),
                "data/curated_train_shards": float(len(curated_train_files)),
                "data/curated_val_shards": float(len(curated_val_files)),
                "data/fineweb_train_shards": float(len(fineweb_train_files)),
                "data/oai_competition_val_shards": float(len(oai_competition_val_files)),
                "data/fineweb_microbatch_fraction": step_fineweb_microbatches,
                "data/fineweb_mix_ratio": effective_fineweb_mix_ratio,
                "data/medium_curriculum_active": float(
                    medium_iterator is not None and step >= int(medium_start_step or 0)
                ),
                "data/medium_microbatch_fraction": step_medium_microbatches,
                "data/medium_mix_ratio": effective_medium_mix_ratio,
                "data/medium_start_step": float(medium_start_step or 0),
                "data/complex_curriculum_active": float(
                    complex_iterator is not None and step >= int(complex_start_step or 0)
                ),
                "data/complex_microbatch_fraction": step_complex_microbatches,
                "data/complex_mix_ratio": effective_complex_mix_ratio,
                "data/hard_microbatch_fraction": step_complex_microbatches,
                "data/hard_mix_ratio": effective_complex_mix_ratio,
                "data/complex_start_step": float(complex_start_step or 0),
                "eval/score_first_bias_lr": eval_score_first_bias_lr,
                "phase/index": float(phase_index),
                "phase/lr_multiplier": float(effective_lr_multiplier),
                "phase/grad_clip_norm": float(effective_grad_clip_norm),
                "phase/base_gflownet_loss_weight": float(gflownet_loss_weight),
                "phase/base_complex_mix_ratio": float(complex_mix_ratio),
                "phase/structural_loss_multiplier": float(structural_loss_multiplier),
                "phase/structural_controller_enabled": float(adaptive_controller.enabled),
                "metrics_status/all_metric_namespaces_always_on": 1.0,
                "metrics_status/losses_follow_phase_curriculum": 1.0,
                "metrics_status/topology_probe_instantiated": float(model_config.use_analogy_lattice),
                "metrics_status/toric_probe_instantiated": float(model_config.use_toric_geometry_tasks),
                "metrics_status/tropical_metric_aliases_instantiated": float(model_config.use_toric_geometry_tasks),
                "metrics_status/bgg_category_o_probe_instantiated": float(model_config.use_toric_bgg),
                "metrics_status/koszul_persistence_probe_instantiated": float(model_config.use_koszul_persistence),
                "metrics_status/slepian_pollak_probe_instantiated": float(model_config.use_slepian_pollak),
                "metrics_status/tokengt_causal_graph_probe_instantiated": float(model_config.use_tokengt_causal_graph),
                "metrics_status/complexity_enabled": float(complexity_enabled),
                "metrics_status/hessian_enabled": float(hessian_enabled),
                "metrics_status/oai_competition_enabled": float(oai_competition_eval_enabled),
                "metrics_status/oai_competition_available": float(oai_competition_available),
                "metrics_status/fineweb_calibration_enabled": float(fineweb_calibration_enabled),
                "metrics_status/fineweb_calibration_available": float(fineweb_calibration_available),
            }
            metrics.update(polarquant_metrics)
            metrics.update(adaptive_controller.log_metrics())
            if audit_error is not None:
                metrics["audit/future_permutation_logit_error"] = audit_error
            metrics.update(complexity_metrics)
            if device.type == "cuda":
                metrics["system/vram_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
                metrics["system/vram_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
            last_train_metrics = dict(metrics)
            if wandb_run is not None:
                wandb_run.log(organize_wandb_payload(metrics), step=step)
        elif complexity_metrics:
            last_train_metrics.update(complexity_metrics)
            if wandb_run is not None:
                wandb_run.log(organize_wandb_payload(complexity_metrics), step=step)

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
                "trainer/step": float(step),
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
            oai_competition_due = (
                oai_competition_loader is not None
                and oai_competition_available
                and int(oai_competition_eval_interval or 0) > 0
                and (step % int(oai_competition_eval_interval) == 0 or step == steps)
            )
            if oai_competition_due:
                try:
                    oai_deterministic = evaluate(
                        model,
                        oai_competition_loader,
                        device=device,
                        batches=int(oai_competition_eval_batches),
                        precision=precision,
                        pass_id=seed + 40_000 + step,
                        order_samples=1,
                        gflownet_samples=1,
                        score_first_bias_lr=0.0,
                        score_first_bias_decay=eval_score_first_bias_decay,
                        score_first_bias_clip=eval_score_first_bias_clip,
                    )
                    oai_gflownet = evaluate(
                        model,
                        oai_competition_loader,
                        device=device,
                        batches=int(oai_competition_eval_batches),
                        precision=precision,
                        pass_id=seed + 50_000 + step,
                        order_samples=max(1, int(oai_competition_order_samples)),
                        gflownet_samples=max(1, int(oai_competition_gflownet_samples)),
                        score_first_bias_lr=0.0,
                        score_first_bias_decay=eval_score_first_bias_decay,
                        score_first_bias_clip=eval_score_first_bias_clip,
                    )
                    if oai_competition_score_first_enabled:
                        oai_score_first = evaluate(
                            model,
                            oai_competition_loader,
                            device=device,
                            batches=int(oai_competition_eval_batches),
                            precision=precision,
                            pass_id=seed + 60_000 + step,
                            order_samples=max(1, int(oai_competition_order_samples)),
                            gflownet_samples=max(1, int(oai_competition_gflownet_samples)),
                            score_first_bias_lr=eval_score_first_bias_lr,
                            score_first_bias_decay=eval_score_first_bias_decay,
                            score_first_bias_clip=eval_score_first_bias_clip,
                        )
                    else:
                        oai_score_first = oai_deterministic
                    oai_test_time_scaled_bpb = min(
                        float(oai_deterministic["bpb"]),
                        float(oai_gflownet["bpb"]),
                        float(oai_score_first["bpb"]),
                    )
                    best_oai_competition_bpb = min(
                        best_oai_competition_bpb,
                        float(oai_deterministic["bpb"]),
                    )
                    best_oai_competition_tts_bpb = min(best_oai_competition_tts_bpb, oai_test_time_scaled_bpb)
                    metrics.update(
                        {
                            "oai_competition/loss": oai_deterministic["loss"],
                            "oai_competition/bpb": oai_deterministic["bpb"],
                            "oai_competition/deterministic_loss": oai_deterministic["loss"],
                            "oai_competition/deterministic_bpb": oai_deterministic["bpb"],
                            "oai_competition/gflownet_loss": oai_gflownet["loss"],
                            "oai_competition/gflownet_bpb": oai_gflownet["bpb"],
                            "oai_competition/score_first_loss": oai_score_first["loss"],
                            "oai_competition/score_first_bpb": oai_score_first["bpb"],
                            "oai_competition/test_time_scaled_bpb": oai_test_time_scaled_bpb,
                            "oai_competition/best_test_time_scaled_bpb": best_oai_competition_tts_bpb,
                            "oai_competition/test_time_scaling_delta_bpb": float(oai_deterministic["bpb"])
                            - oai_test_time_scaled_bpb,
                            "oai_competition/order_samples": float(oai_competition_order_samples),
                            "oai_competition/gflownet_samples": float(oai_competition_gflownet_samples),
                            "oai_competition/score_first_enabled": float(oai_competition_score_first_enabled),
                            "oai_competition/best_bpb": best_oai_competition_bpb,
                            "oai_competition/eval_batches": float(oai_competition_eval_batches),
                            "oai_competition/available": 1.0,
                            "oai_competition/source_sp1024_decoded_bytes": 1.0,
                            "competition/oai_bpb": oai_deterministic["bpb"],
                            "competition/oai_test_time_scaled_bpb": oai_test_time_scaled_bpb,
                            "bpb/oai_competition": oai_deterministic["bpb"],
                            "bpb/oai_competition_test_time_scaled": oai_test_time_scaled_bpb,
                        }
                    )
                except Exception as exc:
                    metrics["oai_competition/error"] = 1.0
                    metrics["oai_competition/available"] = 0.0
                    print(json.dumps({"step": step, "oai_competition_eval_error": f"{type(exc).__name__}: {exc}"}))
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
                train_metrics=last_train_metrics,
            )
            if adaptive_controller.enabled:
                gflownet_loss_weight = adaptive_controller.gflownet_loss_weight
                gflownet_entropy_target = adaptive_controller.gflownet_entropy_target
                complex_mix_ratio = adaptive_controller.complex_mix_ratio
            metrics.update(controller_metrics)
            print(json.dumps({"step": step, **metrics}, indent=2))
            if wandb_run is not None:
                wandb_run.log(organize_wandb_payload(metrics), step=step)
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
                        wandb_run.log(organize_wandb_payload(publish_metrics), step=step)
            model.train()

        if step % ckpt_interval == 0 or step == steps:
            save_checkpoint(
                checkpoint_dir / f"random_order_step_{step:08d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                config=model_config,
                args=args,
                metrics={
                    "train_loss": step_loss,
                    "train_bpb": bpb,
                    "best_val_bpb": best_val,
                    "best_oai_competition_bpb": best_oai_competition_bpb,
                    "best_oai_competition_tts_bpb": best_oai_competition_tts_bpb,
                    "oai_competition/best_bpb": best_oai_competition_bpb,
                    "oai_competition/best_test_time_scaled_bpb": best_oai_competition_tts_bpb,
                },
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
            organize_wandb_payload({
                "trainer/step": float(steps),
                "artifact/final_bytes": final_artifact.bytes_total,
                "artifact/within_limit": float(final_artifact.within_limit),
                "artifact/final_deployment_parameters": final_artifact.deployment_parameters,
                "artifact/final_excluded_tensors": final_artifact.excluded_tensors,
                "val/best_bpb": best_val,
                **polarquant_metrics,
            }),
            step=steps,
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
