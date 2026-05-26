"""Parquet-backed graph dataset utilities for curated ToricGT records."""

from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from .config import ModelConfig
from .graph_tokenizer import GraphBatch


@dataclass
class GraphTrainingItem:
    graph: GraphBatch
    target: torch.Tensor


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


def graph_json_to_item(graph_json: str, cfg: ModelConfig) -> GraphTrainingItem:
    payload = json.loads(graph_json or "{}")
    raw_nodes = list(payload.get("nodes") or [])
    raw_edges = list(payload.get("edges") or [])
    nodes = raw_nodes[: cfg.max_nodes]
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
    graph = GraphBatch(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
        node_mask=node_mask,
        edge_mask=edge_mask,
    )
    return GraphTrainingItem(graph=graph, target=target)


def collate_graph_items(items: list[GraphTrainingItem]) -> tuple[GraphBatch, torch.Tensor]:
    if not items:
        raise ValueError("cannot collate an empty graph batch")
    graph = GraphBatch(
        node_features=torch.stack([item.graph.node_features for item in items]),
        edge_features=torch.stack([item.graph.edge_features for item in items]),
        edge_index=torch.stack([item.graph.edge_index for item in items]),
        node_mask=torch.stack([item.graph.node_mask for item in items]),
        edge_mask=torch.stack([item.graph.edge_mask for item in items]),
    )
    target = torch.stack([item.target for item in items])
    return graph, target


class CuratedGraphIterableDataset(IterableDataset[GraphTrainingItem]):
    """Stream graph records from curated Parquet or normalized JSONL files."""

    def __init__(
        self,
        paths: Iterable[str | Path],
        cfg: ModelConfig,
        parquet_batch_size: int = 512,
        subset_id: int | None = None,
        num_subsets: int = 1,
        subset_salt: int = 17,
        subset_columns: tuple[str, ...] = ("group_hash", "content_hash", "record_id"),
    ) -> None:
        super().__init__()
        if num_subsets < 1:
            raise ValueError("num_subsets must be positive")
        if subset_id is not None and not (0 <= subset_id < num_subsets):
            raise ValueError("subset_id must be in [0, num_subsets)")
        self.paths = [Path(path) for path in paths]
        self.cfg = cfg
        self.parquet_batch_size = parquet_batch_size
        self.subset_id = subset_id
        self.num_subsets = num_subsets
        self.subset_salt = subset_salt
        self.subset_columns = subset_columns

    def _keep_record(self, record: dict[str, object]) -> bool:
        if self.subset_id is None:
            return True
        values = [record.get(column) for column in self.subset_columns]
        if not any(value not in (None, "") for value in values):
            values = [record.get("graph_json", "")]
        return stable_partition_id(values, self.num_subsets, self.subset_salt) == self.subset_id

    def __iter__(self):
        for path in self.paths:
            if path.suffix == ".jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if self._keep_record(record):
                            yield graph_json_to_item(str(record.get("graph_json") or "{}"), self.cfg)
            else:
                parquet_file = pq.ParquetFile(path)
                available = set(parquet_file.schema_arrow.names)
                columns = ["graph_json"]
                if self.subset_id is not None:
                    columns.extend(column for column in self.subset_columns if column in available)
                for batch in parquet_file.iter_batches(batch_size=self.parquet_batch_size, columns=columns):
                    batch_dict = batch.to_pydict()
                    graph_values = batch_dict["graph_json"]
                    for row_idx, graph_json in enumerate(graph_values):
                        record = {column: batch_dict[column][row_idx] for column in batch_dict}
                        if self._keep_record(record):
                            yield graph_json_to_item(graph_json, self.cfg)
