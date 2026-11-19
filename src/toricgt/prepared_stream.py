"""Admission and seekable batching for reviewed symbolic Parquet streams."""

from __future__ import annotations

import hashlib
import json
import os
import re
from bisect import bisect_right
from contextlib import ExitStack, contextmanager
from itertools import accumulate
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .graph_dataset import collate_graph_items, ensure_branch_merge_graph_json, graph_json_to_item
from .toricmora_symbolic_splits import ReviewedSimilarityLink, SymbolicRecord, audit_symbolic_splits


SCHEMA = "toricgt-prepared-symbolic-v1"


def _json_object(raw, label):
    def unique_keys(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError(f"{label} contains a duplicate key")
        return result

    value = json.loads(raw, object_pairs_hook=unique_keys)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a lowercase SHA-256 digest")
    return value


def _artifact(root, spec, suffix):
    if not isinstance(spec, dict) or spec.keys() != {"file", "sha256"}:
        raise ValueError("each prepared artifact must identify a file and SHA-256")
    name = spec["file"]
    if not isinstance(name, str):
        raise ValueError("prepared artifact paths must be relative")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != suffix:
        raise ValueError("invalid prepared artifact path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("prepared artifact must be a file inside the manifest directory")
    return path, _sha256(spec["sha256"])


def _stamp(stream):
    stat = os.fstat(stream.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class PreparedShard:
    """A verified Parquet descriptor with a one-row-group payload cache."""

    def __init__(self, stream, digest):
        self._stream = stream
        self._stamp = _stamp(stream)
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
        if checksum.hexdigest() != digest:
            raise ValueError("prepared Parquet SHA-256 mismatch")
        stream.seek(0)
        self._file = pq.ParquetFile(stream)
        schema = self._file.schema_arrow
        if schema.names != ["record_id", "graph_json"] or any(
            not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)) for field in schema
        ):
            raise ValueError("prepared Parquet requires string record_id and graph_json columns in that order")
        sizes = [self._file.metadata.row_group(i).num_rows for i in range(self._file.num_row_groups)]
        self._ends = tuple(accumulate(sizes))
        if not self._ends or not all(sizes):
            raise ValueError("prepared Parquet must have nonempty row groups")
        self._cached_group = -1
        self._cached_rows = []
        self._check_unchanged()

    def _check_unchanged(self):
        if _stamp(self._stream) != self._stamp:
            raise ValueError("prepared Parquet changed after verification")

    def __len__(self):
        return self._ends[-1]

    def inventory(self):
        self._check_unchanged()
        ids = set()
        for group in range(len(self._ends)):
            table = self._file.read_row_group(group, columns=["record_id"], use_threads=False)
            for record_id in table["record_id"].to_pylist():
                if not isinstance(record_id, str) or not record_id.strip() or record_id in ids:
                    raise ValueError("prepared Parquet requires unique nonempty record IDs")
                ids.add(record_id)
        self._check_unchanged()
        return ids

    def records(self, start, stop):
        if not 0 <= start < stop <= len(self):
            raise ValueError("prepared stream range is outside the split")
        self._check_unchanged()
        rows = []
        while start < stop:
            group = bisect_right(self._ends, start)
            if group != self._cached_group:
                table = self._file.read_row_group(group, use_threads=False)
                self._cached_rows = table.to_pylist()
                self._cached_group = group
            group_start = self._ends[group - 1] if group else 0
            end = min(stop, self._ends[group])
            rows.extend(self._cached_rows[start - group_start : end - group_start])
            start = end
        self._check_unchanged()
        return rows


class PreparedBatchStream:
    """Cycle only after the short final batch and resume by absolute batch count."""

    def __init__(self, shard, config, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("prepared stream batch size must be positive")
        self.shard = shard
        self.config = config
        self.batch_size = batch_size
        self.batches_per_epoch = (len(shard) + batch_size - 1) // batch_size
        self.batches_consumed = 0

    def restore(self, batches_consumed, step, grad_accum_steps):
        if type(batches_consumed) is not int or batches_consumed < 0 or batches_consumed != step * grad_accum_steps:
            raise ValueError("prepared stream cursor is inconsistent with the checkpoint step and accumulation")
        self.batches_consumed = batches_consumed

    def _batch(self, index):
        start = index * self.batch_size
        rows = self.shard.records(start, min(start + self.batch_size, len(self.shard)))
        items = []
        for row in rows:
            payload = _json_object(row["graph_json"], "prepared graph")
            if payload.get("record_id") != row["record_id"]:
                raise ValueError("prepared graph ID differs from its admitted Parquet record ID")
            if len(payload["nodes"]) > self.config.max_nodes:
                raise ValueError("prepared graph exceeds configured node capacity")
            effective = json.loads(ensure_branch_merge_graph_json(row["graph_json"]))
            if len(effective["edges"]) > self.config.max_edges:
                raise ValueError("prepared graph exceeds configured edge capacity")
            items.append(graph_json_to_item(row["graph_json"], self.config))
        return collate_graph_items(items)

    def next_batch(self):
        result = self._batch(self.batches_consumed % self.batches_per_epoch)
        self.batches_consumed += 1
        return result

    def __iter__(self):
        for index in range(self.batches_per_epoch):
            yield self._batch(index)


@contextmanager
def open_prepared_stream(manifest_path, digest):
    """Bind the actual training/validation IDs and an independent retrieval inventory."""
    path = Path(manifest_path).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha256(digest):
        raise ValueError("prepared stream manifest SHA-256 mismatch")
    manifest = _json_object(raw, "prepared stream manifest")
    if manifest.keys() != {"schema", "review", "train", "validation", "retrieval"} or manifest["schema"] != SCHEMA:
        raise ValueError("unsupported prepared symbolic manifest")

    def checked_json(spec, label):
        artifact, expected = _artifact(path.parent, spec, ".json")
        data = artifact.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"{label} SHA-256 mismatch")
        return _json_object(data, label)

    review = checked_json(manifest["review"], "symbolic review")
    retrieval = checked_json(manifest["retrieval"], "retrieval inventory")
    if review.keys() != {"records", "similarity_links"} or any(not isinstance(v, list) for v in review.values()):
        raise ValueError("symbolic review requires records and reviewed similarity links")
    if retrieval.keys() != {"record_ids"} or not isinstance(retrieval["record_ids"], list):
        raise ValueError("retrieval inventory requires an explicit record ID list")
    retrieval_ids = retrieval["record_ids"]
    if any(not isinstance(i, str) or not i.strip() for i in retrieval_ids) or len(set(retrieval_ids)) != len(retrieval_ids):
        raise ValueError("retrieval inventory requires unique nonempty record IDs")

    with ExitStack() as files:
        shards = {}
        for split in ("train", "validation"):
            spec = manifest[split]
            if spec is not None or split == "train":
                artifact, expected = _artifact(path.parent, spec, ".parquet")
                stream = files.enter_context(artifact.open("rb"))
                shards[split] = PreparedShard(stream, expected)
        train_ids = shards["train"].inventory()
        validation_ids = shards["validation"].inventory() if "validation" in shards else set()
        audit = audit_symbolic_splits(
            (SymbolicRecord(**row) for row in review["records"]),
            (ReviewedSimilarityLink(**link) for link in review["similarity_links"]),
            seen_training_record_ids=train_ids,
            indexed_retrieval_record_ids=retrieval_ids,
        )
        if audit.training_exposure.violations or audit.retrieval_exposure.violations:
            raise ValueError("prepared training or retrieval inventory violates reviewed split assignments")
        by_id = {row.record_id: row.split for row in audit.assignments}
        if any(by_id.get(i) != "validation" for i in validation_ids):
            raise ValueError("prepared validation inventory violates reviewed split assignments")
        yield shards, {
            "manifest_sha256": digest,
            "train_records": len(train_ids),
            "validation_records": len(validation_ids),
            "retrieval_records": len(retrieval_ids),
        }
