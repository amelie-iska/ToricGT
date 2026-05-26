#!/usr/bin/env python3
"""Download, normalize, and segment ToricGT datasets into Parquet files."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from toricgt.config import DataConfig
from toricgt.datasets import DATASET_SPECS, DatasetSpec, normalize_record, safe_name, write_manifest


CURATED_COLUMNS = [
    ("record_id", pa.string()),
    ("dataset", pa.string()),
    ("config", pa.string()),
    ("source_split", pa.string()),
    ("source_index", pa.int64()),
    ("task_family", pa.string()),
    ("language", pa.string()),
    ("license", pa.string()),
    ("role", pa.string()),
    ("question", pa.string()),
    ("answer", pa.string()),
    ("solution", pa.string()),
    ("reasoning", pa.string()),
    ("metadata_json", pa.string()),
    ("text", pa.string()),
    ("graph_json", pa.string()),
    ("content_hash", pa.string()),
    ("group_hash", pa.string()),
    ("split", pa.string()),
    ("estimated_tokens", pa.int64()),
    ("quality_flags_json", pa.string()),
]
CURATED_SCHEMA = pa.schema(CURATED_COLUMNS)


class ParquetSink:
    def __init__(self, path: Path, schema: pa.Schema, compression: str = "zstd") -> None:
        self.path = path
        self.schema = schema
        self.compression = compression
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0

    def write(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, self.schema, compression=self.compression)
        self.writer.write_table(table)
        self.rows += len(rows)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class SinkSet:
    def __init__(self, output_dir: Path, chunk_size: int) -> None:
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.sinks: dict[str, ParquetSink] = {
            "all": ParquetSink(output_dir / "all.parquet", CURATED_SCHEMA),
            "train": ParquetSink(output_dir / "train.parquet", CURATED_SCHEMA),
            "validation": ParquetSink(output_dir / "validation.parquet", CURATED_SCHEMA),
            "test": ParquetSink(output_dir / "test.parquet", CURATED_SCHEMA),
        }
        self.buffers: dict[str, list[dict]] = defaultdict(list)
        self.dataset_sinks: dict[str, ParquetSink] = {}

    def add(self, row: dict) -> None:
        keys = ["all", row["split"], f"dataset::{safe_name(row['dataset'] + '__' + row['config'])}"]
        for key in keys:
            if key.startswith("dataset::") and key not in self.dataset_sinks:
                name = key.split("::", 1)[1]
                self.dataset_sinks[key] = ParquetSink(self.output_dir / "by_dataset" / f"{name}.parquet", CURATED_SCHEMA)
            self.buffers[key].append(row)
            if len(self.buffers[key]) >= self.chunk_size:
                self.flush_key(key)

    def flush_key(self, key: str) -> None:
        rows = self.buffers.get(key, [])
        if not rows:
            return
        sink = self.dataset_sinks[key] if key.startswith("dataset::") else self.sinks[key]
        sink.write(rows)
        self.buffers[key] = []

    def close(self) -> None:
        for key in list(self.buffers):
            self.flush_key(key)
        for sink in [*self.sinks.values(), *self.dataset_sinks.values()]:
            sink.close()


def normalize_batch(spec: DatasetSpec, batch: list[tuple[str, int, dict]], seed: int) -> list[dict]:
    return [
        normalize_record(spec, raw, source_split, source_index, seed=seed)
        for source_split, source_index, raw in batch
    ]


def iter_datasets_rows(spec: DatasetSpec) -> Iterable[tuple[str, int, dict]]:
    for split in spec.splits:
        ds = load_dataset(spec.name, spec.config, split=split, streaming=True)
        for idx, row in enumerate(ds):
            yield split, idx, dict(row)


def copy_raw_file(src: str | Path, raw_dir: Path, spec: DatasetSpec) -> Path:
    target_dir = raw_dir / safe_name(spec.name)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(src).name
    if not target.exists():
        shutil.copy2(src, target)
    return target


def iter_hf_jsonl_rows(spec: DatasetSpec, raw_dir: Path) -> Iterable[tuple[str, int, dict]]:
    if spec.file_name is None:
        raise ValueError(f"{spec.key} requires file_name")
    cached = hf_hub_download(spec.name, spec.file_name, repo_type="dataset")
    path = copy_raw_file(cached, raw_dir, spec)
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A few HF-hosted "jsonl" files are mixed streams: ordinary
                # JSONL for a long prefix followed by pretty-printed JSON
                # arrays/objects.  Recover the remainder with a streaming
                # raw decoder instead of dropping the tail of the dataset.
                remainder = line + "\n" + handle.read()
                decoder = json.JSONDecoder()
                pos = 0
                recovered_idx = idx
                while pos < len(remainder):
                    while pos < len(remainder) and remainder[pos].isspace():
                        pos += 1
                    if pos >= len(remainder):
                        break
                    value, end = decoder.raw_decode(remainder, pos)
                    values = value if isinstance(value, list) else [value]
                    for item in values:
                        if isinstance(item, dict):
                            yield "train", recovered_idx, item
                            recovered_idx += 1
                    pos = end
                return
            if isinstance(row, dict):
                yield "train", idx, row


def iter_hf_json_rows(spec: DatasetSpec, raw_dir: Path) -> Iterable[tuple[str, int, dict]]:
    if spec.file_name is None:
        raise ValueError(f"{spec.key} requires file_name")
    cached = hf_hub_download(spec.name, spec.file_name, repo_type="dataset")
    path = copy_raw_file(cached, raw_dir, spec)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("data", data.get("rows", []))
    for idx, row in enumerate(data):
        yield "train", idx, dict(row)


def iter_raw_tsv_url_rows(spec: DatasetSpec, raw_dir: Path) -> Iterable[tuple[str, int, dict]]:
    if spec.raw_url is None:
        raise ValueError(f"{spec.key} requires raw_url")
    target_dir = raw_dir / safe_name(spec.name)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{spec.config}.tsv"
    if not path.exists():
        with urllib.request.urlopen(spec.raw_url, timeout=120) as response, path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                yield "train", idx, {"lemma": parts[0], "form": parts[1], "features": parts[2]}


def iter_spec_rows(spec: DatasetSpec, raw_dir: Path) -> Iterable[tuple[str, int, dict]]:
    if spec.loader == "datasets":
        yield from iter_datasets_rows(spec)
    elif spec.loader == "hf_jsonl":
        yield from iter_hf_jsonl_rows(spec, raw_dir)
    elif spec.loader == "hf_json":
        yield from iter_hf_json_rows(spec, raw_dir)
    elif spec.loader == "raw_tsv_url":
        yield from iter_raw_tsv_url_rows(spec, raw_dir)
    else:
        raise ValueError(f"unknown loader {spec.loader}")


def write_reports(output_dir: Path, stats: dict, errors: list[dict], cfg: DataConfig) -> None:
    report = {
        "config": cfg.__dict__,
        "stats": stats,
        "errors": errors,
    }
    (output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ToricGT Split Report",
        "",
        f"Total records: {stats['total_records']}",
        f"Total estimated tokens: {stats['total_estimated_tokens']}",
        "",
        "## Split Counts",
        "",
        "| Split | Records | Estimated Tokens |",
        "|---|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        lines.append(
            f"| {split} | {stats['split_counts'].get(split, 0)} | {stats['split_tokens'].get(split, 0)} |"
        )
    lines.extend(["", "## Dataset Counts", "", "| Dataset / Config | Records |", "|---|---:|"])
    for key, count in sorted(stats["dataset_counts"].items()):
        lines.append(f"| `{key}` | {count} |")
    if errors:
        lines.extend(["", "## Errors", ""])
        for err in errors:
            lines.append(f"- `{err['dataset']}` `{err['config']}`: {err['error']}")
    (output_dir / "split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_specs(names: list[str] | None) -> tuple[DatasetSpec, ...]:
    if not names:
        return DATASET_SPECS
    wanted = set(names)
    specs = tuple(spec for spec in DATASET_SPECS if spec.name in wanted or spec.key in wanted)
    missing = wanted - {spec.name for spec in specs} - {spec.key for spec in specs}
    if missing:
        raise ValueError(f"unknown dataset selectors: {sorted(missing)}")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/curated")
    parser.add_argument("--raw-dir", default="data/raw/hf")
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel CPU workers for normalization")
    parser.add_argument("--normalize-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dataset", action="append", default=None, help="Dataset name or name::config to include")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cfg = DataConfig(output_dir=str(output_dir), seed=args.seed)
    write_manifest(output_dir / "manifest.json", cfg)
    specs = selected_specs(args.dataset)

    sinks = SinkSet(output_dir, args.chunk_size)
    split_counts: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    errors: list[dict] = []
    total_records = 0
    total_tokens = 0
    executor = ProcessPoolExecutor(max_workers=args.num_workers) if args.num_workers > 1 else None

    def consume_rows(rows: list[dict], pbar: tqdm) -> None:
        nonlocal total_records, total_tokens
        for row in rows:
            sinks.add(row)
            split_counts[row["split"]] += 1
            split_tokens[row["split"]] += row["estimated_tokens"]
            dataset_counts[f"{row['dataset']}::{row['config']}"] += 1
            family_counts[row["task_family"]] += 1
            total_records += 1
            total_tokens += row["estimated_tokens"]
        pbar.update(len(rows))

    def drain_one(pending: list[Future], pbar: tqdm) -> None:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            pending.remove(future)
            consume_rows(future.result(), pbar)

    def drain_all(pending: list[Future], pbar: tqdm) -> None:
        while pending:
            drain_one(pending, pbar)

    try:
        for spec in tqdm(specs, desc="datasets"):
            source_count = 0
            expected = spec.expected_rows if args.max_records_per_source is None else args.max_records_per_source
            pbar = tqdm(total=expected, desc=safe_name(spec.key), leave=False)
            pending: list[Future] = []
            batch: list[tuple[str, int, dict]] = []
            try:
                for source_split, source_index, raw in iter_spec_rows(spec, raw_dir):
                    source_count += 1
                    batch.append((source_split, source_index, raw))
                    if len(batch) >= args.normalize_batch_size:
                        if executor is None:
                            consume_rows(normalize_batch(spec, batch, args.seed), pbar)
                        else:
                            pending.append(executor.submit(normalize_batch, spec, batch, args.seed))
                            if len(pending) >= args.num_workers * 4:
                                drain_one(pending, pbar)
                        batch = []
                    if args.max_records_per_source is not None and source_count >= args.max_records_per_source:
                        break
                if batch:
                    if executor is None:
                        consume_rows(normalize_batch(spec, batch, args.seed), pbar)
                    else:
                        pending.append(executor.submit(normalize_batch, spec, batch, args.seed))
                drain_all(pending, pbar)
            except Exception as exc:
                errors.append({"dataset": spec.name, "config": spec.config, "error": repr(exc)})
            finally:
                pbar.close()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        sinks.close()

    stats = {
        "total_records": total_records,
        "total_estimated_tokens": total_tokens,
        "split_counts": dict(split_counts),
        "split_tokens": dict(split_tokens),
        "dataset_counts": dict(dataset_counts),
        "family_counts": dict(family_counts),
        "output_rows": {key: sink.rows for key, sink in sinks.sinks.items()},
        "by_dataset_files": {key: sink.rows for key, sink in sinks.dataset_sinks.items()},
    }
    write_reports(output_dir, stats, errors, cfg)
    if errors:
        print(json.dumps({"status": "completed_with_errors", "errors": errors}, indent=2))
    else:
        print(json.dumps({"status": "ok", "total_records": total_records, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
