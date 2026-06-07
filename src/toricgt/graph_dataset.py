"""Parquet-backed graph dataset utilities for curated ToricGT records."""

from __future__ import annotations

import hashlib
import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import numpy as np
import torch
from torch.utils.data import IterableDataset

from .config import ModelConfig
from .got_trajectory import graph_json_has_branch_merge
from .graph_tokenizer import GraphBatch


@dataclass
class GraphTrainingItem:
    graph: GraphBatch
    target: torch.Tensor
    metadata: dict[str, object] | None = None


TEXT_COLUMNS = (
    "graph_json",
    "text",
    "content",
    "document",
    "prompt",
    "question",
    "problem",
    "reasoning",
    "solution",
    "answer",
    "completion",
    "messages",
)


def expand_dataset_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Expand explicit files, directories, and shell-style parquet/jsonl/bin globs."""

    expanded: list[Path] = []
    for raw_path in paths:
        raw = str(raw_path)
        if any(token in raw for token in ("*", "?", "[")):
            expanded.extend(Path(match) for match in sorted(glob.glob(raw)))
            continue
        path = Path(raw_path)
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.parquet")))
            expanded.extend(sorted(path.glob("*.jsonl")))
            expanded.extend(sorted(path.glob("*.bin")))
        else:
            expanded.append(path)
    return expanded


def interleave_dataset_paths(paths: Iterable[Path]) -> list[Path]:
    """Round-robin expanded paths by source type so FineWeb appears early."""

    groups: dict[str, list[Path]] = {".parquet": [], ".bin": [], ".jsonl": [], "other": []}
    for path in paths:
        groups[path.suffix if path.suffix in groups else "other"].append(path)
    ordered_suffixes = [".parquet", ".bin", ".jsonl", "other"]
    out: list[Path] = []
    max_len = max((len(groups[suffix]) for suffix in ordered_suffixes), default=0)
    for index in range(max_len):
        for suffix in ordered_suffixes:
            values = groups[suffix]
            if index < len(values):
                out.append(values[index])
    return out


def load_competition_token_memmap(path: str | Path) -> np.memmap:
    """Read a Parameter-Golf challenge-format uint16 token shard as a memmap."""

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
    return np.memmap(file_path, dtype="<u2", mode="r", offset=header_bytes, shape=(num_tokens,))


def _load_sentencepiece_tokenizer(tokenizer_path: str | Path):
    try:
        import sentencepiece as spm
    except Exception:
        return None
    path = Path(tokenizer_path)
    if not path.exists():
        return None
    processor = spm.SentencePieceProcessor()
    try:
        processor.Load(str(path))
    except Exception:
        return None
    return processor


def decode_sp1024_tokens(tokens: np.ndarray, tokenizer=None) -> str:
    """Decode FineWeb sp1024 token ids to text, with a deterministic fallback."""

    ids = [int(token) for token in np.asarray(tokens, dtype=np.int64).tolist()]
    if tokenizer is not None:
        try:
            return str(tokenizer.DecodeIds(ids))
        except Exception:
            try:
                return str(tokenizer.decode(ids))
            except Exception:
                pass
    return " ".join(f"<sp{token}>" for token in ids)


def sentencepiece_target_byte_lengths(
    input_ids: np.ndarray,
    target_ids: np.ndarray,
    tokenizer=None,
) -> np.ndarray:
    """Return challenge-compatible byte counts for SP1024 next-token targets."""

    prev = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    tgt = np.asarray(target_ids, dtype=np.int64).reshape(-1)
    if tokenizer is None:
        decoded = decode_sp1024_tokens(tgt, tokenizer)
        mean_bytes = max(1, len(decoded.encode("utf-8", errors="replace"))) / max(1, int(tgt.size))
        return np.full((int(tgt.size),), float(mean_bytes), dtype=np.float32)

    out = np.zeros((int(tgt.size),), dtype=np.float32)
    for idx, token in enumerate(tgt.tolist()):
        token_id = int(token)
        prev_id = int(prev[idx]) if idx < int(prev.size) else -1
        try:
            if tokenizer.is_control(token_id) or tokenizer.is_unknown(token_id) or tokenizer.is_unused(token_id):
                out[idx] = 0.0
                continue
            if tokenizer.is_byte(token_id):
                out[idx] = 1.0
                continue
            piece = str(tokenizer.id_to_piece(token_id))
            has_leading_space = piece.startswith("▁")
            if has_leading_space:
                piece = piece[1:]
            byte_count = len(piece.encode("utf-8"))
            previous_is_boundary = True
            if prev_id >= 0:
                previous_is_boundary = bool(
                    tokenizer.is_control(prev_id) or tokenizer.is_unknown(prev_id) or tokenizer.is_unused(prev_id)
                )
            if has_leading_space and not previous_is_boundary:
                byte_count += 1
            out[idx] = float(byte_count)
        except Exception:
            out[idx] = 0.0
    return out


def deterministic_random_ranks(node_ids: list[str], record_id: str) -> dict[str, int]:
    keyed = []
    for node_id in node_ids:
        digest = hashlib.blake2b(f"{record_id}:{node_id}".encode("utf-8", errors="replace"), digest_size=8).digest()
        keyed.append((int.from_bytes(digest, "big"), node_id))
    return {node_id: rank for rank, (_key, node_id) in enumerate(sorted(keyed))}


def graph_causal_ranks(
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
    *,
    record_id: str,
    force_linear: bool = False,
) -> tuple[torch.Tensor, str]:
    """Compute node reveal ranks for causal TokenGT attention.

    FineWeb chains use their natural order.  Directed acyclic graphs use a
    topological rank.  Cyclic or noncausal graphs get deterministic random
    reveal ranks so training remains autoregressive without inventing a false
    causal direction.
    """

    node_ids = [str(node.get("id", idx)) for idx, node in enumerate(nodes)]
    if not node_ids:
        return torch.zeros(0, dtype=torch.long), "empty"
    if force_linear:
        return torch.arange(len(node_ids), dtype=torch.long), "linear_causal"

    node_set = set(node_ids)
    incoming: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    outgoing: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    edge_count = 0
    for edge in edges:
        src = str(edge.get("source"))
        dst = str(edge.get("target"))
        if src not in node_set or dst not in node_set or src == dst:
            continue
        outgoing[src].add(dst)
        incoming[dst].add(src)
        edge_count += 1
    if edge_count <= 0:
        ranks = deterministic_random_ranks(node_ids, record_id)
        return torch.tensor([ranks[node_id] for node_id in node_ids], dtype=torch.long), "random_no_edges"

    ready = sorted(node_id for node_id in node_ids if not incoming[node_id])
    topo: list[str] = []
    incoming_work = {node_id: set(values) for node_id, values in incoming.items()}
    while ready:
        node_id = ready.pop(0)
        topo.append(node_id)
        for dst in sorted(outgoing[node_id]):
            incoming_work[dst].discard(node_id)
            if not incoming_work[dst] and dst not in topo and dst not in ready:
                ready.append(dst)
        ready.sort()
    if len(topo) != len(node_ids):
        ranks = deterministic_random_ranks(node_ids, record_id)
        return torch.tensor([ranks[node_id] for node_id in node_ids], dtype=torch.long), "random_cycle"
    ranks = {node_id: rank for rank, node_id in enumerate(topo)}
    return torch.tensor([ranks[node_id] for node_id in node_ids], dtype=torch.long), "topological_dag"


def stable_partition_id(
    values: Iterable[object],
    num_subsets: int,
    salt: int = 17,
) -> int:
    """Map row identity fields to a deterministic subset id."""

    if num_subsets < 1:
        raise ValueError("num_subsets must be positive")
    key = "||".join("" if value is None else str(value) for value in values)
    digest = hashlib.blake2b(f"{salt}:{key}".encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % num_subsets


def _hash_unit(text: str, salt: int) -> float:
    digest = hashlib.blake2b(f"{salt}:{text}".encode("utf-8", errors="replace"), digest_size=4).digest()
    value = int.from_bytes(digest, "big")
    return (value / 0xFFFFFFFF) * 2.0 - 1.0


def _text_features(text: str, type_name: str, dim: int) -> list[float]:
    words = text.split()
    chars = len(text)
    vals = [
        math.tanh(chars / 512.0),
        math.tanh(len(words) / 128.0),
        math.tanh(len(type_name) / 32.0),
        _hash_unit(type_name, 0),
        _hash_unit(text[:256], 1),
        _hash_unit(text[-256:], 2),
    ]
    for idx in range(max(0, dim - len(vals))):
        vals.append(_hash_unit(f"{type_name}:{text[:64]}", idx + 10))
    return vals[:dim]


def _split_reasoning_spans(text: str, *, max_spans: int) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ["empty document"]
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    spans: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        words = piece.split()
        if len(words) <= 28:
            spans.append(piece)
        else:
            for start in range(0, len(words), 28):
                spans.append(" ".join(words[start : start + 28]))
        if len(spans) >= max_spans:
            break
    return spans[:max_spans] or [text[:512]]


def text_to_graph_json(
    text: str,
    *,
    dataset: str = "",
    task_family: str = "",
    record_id: str = "",
    max_spans: int = 64,
) -> str:
    """Convert raw text into a branch-and-merge graph-of-thought record.

    The root branches into alternative early spans.  The trajectory is then a
    branch/merge DAG of local reasoning cells, not a full adjacent-span chain:
    ``source -> {alternative_a, alternative_b} -> merge``.  This keeps
    full-dataset text rows compatible with TokenGT while preserving the
    graph-of-thought assumption that reasoning trajectories split and rejoin.
    """

    spans = _split_reasoning_spans(text, max_spans=max(2, max_spans))
    nodes: list[dict[str, str]] = [
        {
            "id": "root",
            "type": "reasoning_root",
            "text": f"{dataset or 'dataset'} {task_family or 'reasoning'} {record_id}".strip() or "document root",
        }
    ]
    for idx, span in enumerate(spans):
        nodes.append({"id": f"step_{idx:03d}", "type": "reasoning_step", "text": span})
    edges: list[dict[str, str]] = []
    source = "root"
    idx = 0
    while idx < len(spans):
        left = f"step_{idx:03d}"
        if idx + 1 >= len(spans):
            edges.append({"source": source, "target": left, "type": "tail_continuation"})
            break
        right = f"step_{idx + 1:03d}"
        edges.append({"source": source, "target": left, "type": "branch_left"})
        edges.append({"source": source, "target": right, "type": "branch_right"})
        if idx + 2 < len(spans):
            merge_target = f"step_{idx + 2:03d}"
            edges.append({"source": left, "target": merge_target, "type": "merge_candidate"})
            edges.append({"source": right, "target": merge_target, "type": "merge_candidate"})
            source = merge_target
            idx += 3
        else:
            edges.append({"source": left, "target": right, "type": "merge_candidate"})
            source = right
            idx += 2
    payload = {
        "task_family": task_family or "text_branch_merge_got",
        "dataset": dataset,
        "record_id": record_id,
        "trajectory_kind": "branch_merge_dag",
        "nodes": nodes,
        "edges": edges,
    }
    return json.dumps(payload, ensure_ascii=False)


def fineweb_tokens_to_graph_json(
    tokens: np.ndarray,
    *,
    dataset: str = "fineweb10B_sp1024",
    record_id: str = "",
    max_nodes: int = 256,
) -> str:
    """Represent a FineWeb SP1024 slice as a left-to-right token chain.

    FineWeb is the OAI Parameter-Golf language-modeling stream, so its
    structural graph should preserve next-token order. Branch/merge graph
    structure is reserved for curated reasoning, memory, and graph-of-thought
    records where multiple reasoning alternatives are actually present.
    """

    token_ids = [int(token) for token in np.asarray(tokens, dtype=np.int64).reshape(-1).tolist()]
    node_count = min(max(1, int(max_nodes)), max(1, len(token_ids) - 1))
    nodes = [
        {
            "id": f"tok_{idx:04d}",
            "type": "sp1024_token_position",
            "text": f"<sp{token_ids[idx] if idx < len(token_ids) else 0}>",
        }
        for idx in range(node_count)
    ]
    edges = [
        {
            "source": f"tok_{idx:04d}",
            "target": f"tok_{idx + 1:04d}",
            "type": "next_token",
        }
        for idx in range(max(0, node_count - 1))
    ]
    payload = {
        "task_family": "fineweb_language_modeling_graph",
        "dataset": dataset,
        "record_id": record_id,
        "trajectory_kind": "sequential_token_chain",
        "node_basis": "sp1024_token_positions",
        "sequence_order": "left_to_right_next_token",
        "nodes": nodes,
        "edges": edges,
    }
    return json.dumps(payload, ensure_ascii=False)


def ensure_branch_merge_graph_json(graph_json: str) -> str:
    """Add conservative branch/merge edges to older linear graph records."""

    payload = json.loads(graph_json or "{}")
    if graph_json_has_branch_merge(payload):
        return graph_json
    nodes = list(payload.get("nodes") or [])
    if len(nodes) < 4:
        return graph_json
    suppressed_types = {"context_order", "next"}
    original_edges = list(payload.get("edges") or [])
    edges = [edge for edge in original_edges if str(edge.get("type") or "") not in suppressed_types]
    suppressed_linear_chain_edges = len(original_edges) - len(edges)
    node_ids = [str(node.get("id", idx)) for idx, node in enumerate(nodes)]
    existing = {(str(edge.get("source")), str(edge.get("target")), str(edge.get("type") or "")) for edge in edges}
    for idx in range(0, len(node_ids) - 3, 3):
        source = node_ids[idx]
        left = node_ids[idx + 1]
        right = node_ids[idx + 2]
        join = node_ids[idx + 3]
        candidates = [
            (source, left, "branch_left"),
            (source, right, "branch_right"),
            (left, join, "merge_candidate"),
            (right, join, "merge_candidate"),
        ]
        for src, dst, typ in candidates:
            if (src, dst, typ) not in existing:
                edges.append({"source": src, "target": dst, "type": typ})
                existing.add((src, dst, typ))
    payload["edges"] = edges
    payload["trajectory_kind"] = "branch_merge_dag"
    payload["suppressed_linear_chain_edges"] = int(suppressed_linear_chain_edges)
    return json.dumps(payload, ensure_ascii=False)


def _stringify_text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify_text_value(item) for item in value)
    if isinstance(value, dict):
        if "text" in value:
            return _stringify_text_value(value.get("text"))
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def record_to_graph_json(record: dict[str, object], cfg: ModelConfig) -> str:
    graph_json = record.get("graph_json")
    if graph_json:
        try:
            return ensure_branch_merge_graph_json(str(graph_json))
        except Exception:
            return str(graph_json)
    text_parts: list[str] = []
    for column in TEXT_COLUMNS:
        if column == "graph_json" or column not in record:
            continue
        value = _stringify_text_value(record.get(column)).strip()
        if value:
            text_parts.append(value)
    if not text_parts:
        text_parts.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return text_to_graph_json(
        "\n".join(text_parts),
        dataset=str(record.get("dataset") or record.get("source") or ""),
        task_family=str(record.get("task_family") or record.get("family") or ""),
        record_id=str(record.get("record_id") or record.get("content_hash") or record.get("id") or ""),
        max_spans=max(2, cfg.max_nodes - 1),
    )


def _lm_tensors_from_tokens(
    tokens: np.ndarray | None,
    cfg: ModelConfig,
    tokenizer=None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if tokens is None:
        return None, None, None, None
    token_ids = np.asarray(tokens, dtype=np.int64).reshape(-1)
    if token_ids.size < 2:
        return None, None, None, None
    length = min(int(cfg.max_nodes), int(token_ids.size) - 1)
    if length <= 0:
        return None, None, None, None
    vocab_size = max(1, int(getattr(cfg, "lm_vocab_size", 1024)))
    input_ids = torch.zeros(cfg.max_nodes, dtype=torch.long)
    target_ids = torch.zeros(cfg.max_nodes, dtype=torch.long)
    lm_mask = torch.zeros(cfg.max_nodes, dtype=torch.bool)
    target_byte_lengths = torch.zeros(cfg.max_nodes, dtype=torch.float32)
    input_ids[:length] = torch.from_numpy(np.clip(token_ids[:length], 0, vocab_size - 1)).long()
    target_slice = np.clip(token_ids[1 : length + 1], 0, vocab_size - 1)
    target_ids[:length] = torch.from_numpy(target_slice).long()
    lm_mask[:length] = True
    byte_lengths = sentencepiece_target_byte_lengths(token_ids[:length], target_slice, tokenizer)
    target_byte_lengths[:length] = torch.from_numpy(byte_lengths).to(dtype=torch.float32)
    return input_ids, target_ids, lm_mask, target_byte_lengths


def graph_json_to_item(
    graph_json: str,
    cfg: ModelConfig,
    *,
    lm_tokens: np.ndarray | None = None,
    tokenizer=None,
) -> GraphTrainingItem:
    raw_payload = json.loads(graph_json or "{}")
    is_fineweb_lm_graph = (
        str(raw_payload.get("dataset") or "") == "fineweb10B_sp1024"
        or str(raw_payload.get("task_family") or "") == "fineweb_language_modeling_graph"
        or str(raw_payload.get("trajectory_kind") or "") == "sequential_token_chain"
    )
    payload = raw_payload if is_fineweb_lm_graph else json.loads(ensure_branch_merge_graph_json(graph_json or "{}"))
    raw_nodes = list(payload.get("nodes") or [])
    raw_edges = list(payload.get("edges") or [])
    nodes = raw_nodes[: cfg.max_nodes]
    record_id = str(payload.get("record_id") or raw_payload.get("record_id") or "")
    node_causal_rank, causal_rank_kind = graph_causal_ranks(
        nodes,
        raw_edges,
        record_id=record_id,
        force_linear=is_fineweb_lm_graph,
    )
    padded_node_causal_rank = torch.arange(cfg.max_nodes, dtype=torch.long)
    if node_causal_rank.numel() > 0:
        padded_node_causal_rank[: node_causal_rank.numel()] = node_causal_rank[: cfg.max_nodes]
    node_ids = {str(node.get("id", idx)): idx for idx, node in enumerate(nodes)}
    node_count = max(1, len(nodes))

    node_features = torch.zeros(cfg.max_nodes, cfg.node_feature_dim, dtype=torch.float32)
    node_mask = torch.zeros(cfg.max_nodes, dtype=torch.bool)
    for idx, node in enumerate(nodes):
        text = str(node.get("text") or node.get("id") or "")
        type_name = str(node.get("type") or "node")
        node_features[idx] = torch.tensor(_text_features(text, type_name, cfg.node_feature_dim), dtype=torch.float32)
        node_mask[idx] = True

    edge_features = torch.zeros(cfg.max_edges, cfg.edge_feature_dim, dtype=torch.float32)
    edge_index = torch.zeros(cfg.max_edges, 2, dtype=torch.long)
    edge_mask = torch.zeros(cfg.max_edges, dtype=torch.bool)
    edge_out = 0
    for edge in raw_edges:
        if edge_out >= cfg.max_edges:
            break
        src = node_ids.get(str(edge.get("source")))
        dst = node_ids.get(str(edge.get("target")))
        if src is None or dst is None:
            continue
        type_name = str(edge.get("type") or "edge")
        edge_index[edge_out] = torch.tensor([src, dst], dtype=torch.long)
        edge_features[edge_out] = torch.tensor(
            _text_features(f"{src}->{dst}", type_name, cfg.edge_feature_dim),
            dtype=torch.float32,
        )
        edge_mask[edge_out] = True
        edge_out += 1

    if not node_mask.any():
        node_mask[0] = True
    if not edge_mask.any():
        edge_index[0] = torch.tensor([0, min(node_count - 1, 0)], dtype=torch.long)
        edge_mask[0] = True

    target = torch.zeros(cfg.max_nodes, cfg.output_dim, dtype=torch.float32)
    copy_dim = min(cfg.output_dim, cfg.node_feature_dim)
    target[:, :copy_dim] = node_features[:, :copy_dim]
    lm_input_ids, lm_target_ids, lm_mask, lm_target_byte_lengths = _lm_tensors_from_tokens(lm_tokens, cfg, tokenizer)
    graph = GraphBatch(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
        node_mask=node_mask,
        edge_mask=edge_mask,
        lm_input_ids=lm_input_ids,
        lm_target_ids=lm_target_ids,
        lm_mask=lm_mask,
        lm_target_byte_lengths=lm_target_byte_lengths,
        node_causal_rank=padded_node_causal_rank,
    )
    metadata = {
        "dataset": payload.get("dataset", ""),
        "task_family": payload.get("task_family", ""),
        "record_id": record_id,
        "trajectory_kind": payload.get("trajectory_kind", ""),
        "causal_rank_kind": causal_rank_kind,
    }
    return GraphTrainingItem(graph=graph, target=target, metadata=metadata)


def collate_graph_items(items: list[GraphTrainingItem]) -> tuple[GraphBatch, torch.Tensor]:
    if not items:
        raise ValueError("cannot collate an empty graph batch")
    any_lm = any(item.graph.lm_target_ids is not None and item.graph.lm_mask is not None for item in items)
    lm_input_ids = None
    lm_target_ids = None
    lm_mask = None
    lm_target_byte_lengths = None
    if any_lm:
        lm_input_ids = torch.stack(
            [
                item.graph.lm_input_ids
                if item.graph.lm_input_ids is not None
                else torch.zeros_like(items[0].graph.node_mask, dtype=torch.long)
                for item in items
            ]
        )
        lm_target_ids = torch.stack(
            [
                item.graph.lm_target_ids
                if item.graph.lm_target_ids is not None
                else torch.zeros_like(items[0].graph.node_mask, dtype=torch.long)
                for item in items
            ]
        )
        lm_mask = torch.stack(
            [
                item.graph.lm_mask
                if item.graph.lm_mask is not None
                else torch.zeros_like(items[0].graph.node_mask, dtype=torch.bool)
                for item in items
            ]
        )
        lm_target_byte_lengths = torch.stack(
            [
                item.graph.lm_target_byte_lengths
                if item.graph.lm_target_byte_lengths is not None
                else torch.zeros_like(items[0].graph.node_mask, dtype=torch.float32)
                for item in items
            ]
        )
    node_causal_rank = torch.stack(
        [
            item.graph.node_causal_rank
            if item.graph.node_causal_rank is not None
            else torch.arange(items[0].graph.node_mask.numel(), dtype=torch.long)
            for item in items
        ]
    )
    graph = GraphBatch(
        node_features=torch.stack([item.graph.node_features for item in items]),
        edge_features=torch.stack([item.graph.edge_features for item in items]),
        edge_index=torch.stack([item.graph.edge_index for item in items]),
        node_mask=torch.stack([item.graph.node_mask for item in items]),
        edge_mask=torch.stack([item.graph.edge_mask for item in items]),
        lm_input_ids=lm_input_ids,
        lm_target_ids=lm_target_ids,
        lm_mask=lm_mask,
        lm_target_byte_lengths=lm_target_byte_lengths,
        node_causal_rank=node_causal_rank,
    )
    target = torch.stack([item.target for item in items])
    return graph, target


class CuratedGraphIterableDataset(IterableDataset[GraphTrainingItem]):
    """Stream graph records from curated Parquet, JSONL, or FineWeb token shards."""

    def __init__(
        self,
        paths: Iterable[str | Path],
        cfg: ModelConfig,
        parquet_batch_size: int = 512,
        subset_id: int | None = None,
        num_subsets: int = 1,
        subset_salt: int = 17,
        subset_columns: tuple[str, ...] = ("group_hash", "content_hash", "record_id"),
        interleave_paths: bool = False,
        fineweb_tokenizer_path: str | Path = "amelie-iska/parameter-golf/data/tokenizers/fineweb_1024_bpe.model",
        fineweb_tokens_per_graph: int = 1024,
        fineweb_stride_tokens: int = 1024,
        fineweb_mix_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        if num_subsets < 1:
            raise ValueError("num_subsets must be positive")
        if subset_id is not None and not (0 <= subset_id < num_subsets):
            raise ValueError("subset_id must be in [0, num_subsets)")
        expanded_paths = expand_dataset_paths(paths)
        self.paths = interleave_dataset_paths(expanded_paths) if interleave_paths else expanded_paths
        self.cfg = cfg
        self.parquet_batch_size = parquet_batch_size
        self.subset_id = subset_id
        self.num_subsets = num_subsets
        self.subset_salt = subset_salt
        self.subset_columns = subset_columns
        self.interleave_paths = bool(interleave_paths)
        self.fineweb_tokenizer_path = Path(fineweb_tokenizer_path)
        self.fineweb_tokens_per_graph = max(1, int(fineweb_tokens_per_graph))
        self.fineweb_stride_tokens = max(1, int(fineweb_stride_tokens))
        self.fineweb_mix_ratio = float(fineweb_mix_ratio)

    def _keep_record(self, record: dict[str, object]) -> bool:
        if self.subset_id is None:
            return True
        values = [record.get(column) for column in self.subset_columns]
        if not any(value not in (None, "") for value in values):
            values = [record.get("graph_json", "")]
        return stable_partition_id(values, self.num_subsets, self.subset_salt) == self.subset_id

    def _iter_jsonl_path(self, path: Path):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if self._keep_record(record):
                    yield graph_json_to_item(record_to_graph_json(record, self.cfg), self.cfg)

    def _iter_parquet_path(self, path: Path):
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        columns = [column for column in TEXT_COLUMNS if column in available]
        if "graph_json" not in columns and not columns:
            columns = list(parquet_file.schema_arrow.names)
        if self.subset_id is not None:
            columns.extend(column for column in self.subset_columns if column in available)
        columns = sorted(set(columns))
        for batch in parquet_file.iter_batches(batch_size=self.parquet_batch_size, columns=columns):
            batch_dict = batch.to_pydict()
            row_count = len(next(iter(batch_dict.values()))) if batch_dict else 0
            for row_idx in range(row_count):
                record = {column: batch_dict[column][row_idx] for column in batch_dict}
                if self._keep_record(record):
                    yield graph_json_to_item(record_to_graph_json(record, self.cfg), self.cfg)

    def _iter_fineweb_bin_path(self, path: Path):
        tokens = load_competition_token_memmap(path)
        tokenizer = _load_sentencepiece_tokenizer(self.fineweb_tokenizer_path)
        chunk = self.fineweb_tokens_per_graph
        stride = self.fineweb_stride_tokens
        if int(tokens.shape[0]) < chunk:
            starts = [0]
        else:
            starts = range(0, int(tokens.shape[0]) - chunk + 1, stride)
        for start in starts:
            end = min(int(start) + chunk, int(tokens.shape[0]))
            token_slice = np.asarray(tokens[int(start) : end], dtype=np.int32)
            record_id = f"{path.stem}:{int(start)}:{int(end)}"
            record = {
                "dataset": "fineweb10B_sp1024",
                "source": str(path),
                "task_family": "fineweb_language_modeling_graph",
                "record_id": record_id,
                "content_hash": hashlib.blake2b(token_slice.tobytes(), digest_size=12).hexdigest(),
                "token_count": int(token_slice.size),
            }
            if self._keep_record(record):
                yield graph_json_to_item(
                    fineweb_tokens_to_graph_json(
                        token_slice,
                        dataset="fineweb10B_sp1024",
                        record_id=record_id,
                        max_nodes=self.cfg.max_nodes,
                    ),
                    self.cfg,
                    lm_tokens=token_slice,
                    tokenizer=tokenizer,
                )

    def _iter_path(self, path: Path):
        if path.suffix == ".jsonl":
            yield from self._iter_jsonl_path(path)
        elif path.suffix == ".bin":
            yield from self._iter_fineweb_bin_path(path)
        else:
            yield from self._iter_parquet_path(path)

    def _iter_paths_sequential(self, paths: Iterable[Path]):
        for path in paths:
            yield from self._iter_path(path)

    def __iter__(self):
        if not self.interleave_paths:
            for path in self.paths:
                yield from self._iter_path(path)
            return

        fineweb_paths = [path for path in self.paths if path.suffix == ".bin"]
        curated_paths = [path for path in self.paths if path.suffix != ".bin"]
        if not fineweb_paths or not curated_paths:
            for path in self.paths:
                yield from self._iter_path(path)
            return

        active: dict[str, object] = {
            "curated": iter(self._iter_paths_sequential(curated_paths)),
            "fineweb": iter(self._iter_paths_sequential(fineweb_paths)),
        }
        ratio = min(1.0, max(0.0, self.fineweb_mix_ratio))
        if ratio <= 0.0:
            ratio = 0.5
        step = 0
        while active:
            fineweb_due = int((step + 1) * ratio) > int(step * ratio)
            preferred = "fineweb" if fineweb_due else "curated"
            if preferred not in active:
                preferred = "curated" if "curated" in active else "fineweb"
            iterator = active[preferred]
            try:
                yield next(iterator)
                step += 1
            except StopIteration:
                active.pop(preferred, None)
