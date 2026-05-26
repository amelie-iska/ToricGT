#!/usr/bin/env python3
"""Rebuild split Parquets from by-dataset shards.

This is useful after repairing one dataset shard without rerunning every raw
download.  It trusts each row's existing deterministic split assignment and
rewrites all.parquet plus train/validation/test.parquet.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from tqdm.auto import tqdm

from curate_datasets import CURATED_SCHEMA, ParquetSink, write_reports
from toricgt.config import DataConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--drop-existing", action="store_true")
    args = parser.parse_args()

    curated_dir = Path(args.curated_dir)
    by_dataset = curated_dir / "by_dataset"
    if not by_dataset.exists():
        raise SystemExit(f"missing by_dataset directory: {by_dataset}")

    outputs = {
        "all": curated_dir / "all.parquet",
        "train": curated_dir / "train.parquet",
        "validation": curated_dir / "validation.parquet",
        "test": curated_dir / "test.parquet",
    }
    if args.drop_existing:
        for path in outputs.values():
            path.unlink(missing_ok=True)

    sinks = {key: ParquetSink(path, CURATED_SCHEMA) for key, path in outputs.items()}
    split_counts: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    by_dataset_counts: dict[str, int] = {}
    total_records = 0
    total_tokens = 0

    try:
        for path in tqdm(sorted(by_dataset.glob("*.parquet")), desc="dataset shards"):
            shard_rows = 0
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=args.chunk_size):
                table = batch.to_pydict()
                rows = [dict(zip(table, values)) for values in zip(*table.values(), strict=True)]
                if not rows:
                    continue
                sinks["all"].write(rows)
                by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
                for row in rows:
                    split = row["split"]
                    if split not in by_split:
                        raise ValueError(f"unexpected split {split!r} in {path}")
                    by_split[split].append(row)
                    split_counts[split] += 1
                    split_tokens[split] += int(row["estimated_tokens"] or 0)
                    dataset_counts[f"{row['dataset']}::{row['config']}"] += 1
                    family_counts[row["task_family"]] += 1
                    total_records += 1
                    total_tokens += int(row["estimated_tokens"] or 0)
                    shard_rows += 1
                for split, split_rows in by_split.items():
                    sinks[split].write(split_rows)
            by_dataset_counts[f"dataset::{path.stem}"] = shard_rows
    finally:
        for sink in sinks.values():
            sink.close()

    cfg = DataConfig(output_dir=str(curated_dir))
    stats = {
        "total_records": total_records,
        "total_estimated_tokens": total_tokens,
        "split_counts": dict(split_counts),
        "split_tokens": dict(split_tokens),
        "dataset_counts": dict(dataset_counts),
        "family_counts": dict(family_counts),
        "output_rows": {key: sink.rows for key, sink in sinks.items()},
        "by_dataset_files": by_dataset_counts,
    }
    write_reports(curated_dir, stats, [], cfg)
    print(json.dumps({"status": "ok", "total_records": total_records, "output_dir": str(curated_dir)}, indent=2))


if __name__ == "__main__":
    main()
