#!/usr/bin/env python3
"""Inspect curated ToricGT parquet outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return pq.ParquetFile(path).metadata.num_rows


def file_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def sum_numeric_column(path: Path, column: str, batch_size: int = 65_536) -> int:
    if not path.exists():
        return 0
    parquet = pq.ParquetFile(path)
    if column not in parquet.schema.names:
        return 0
    total = 0
    for batch in parquet.iter_batches(columns=[column], batch_size=batch_size):
        array = batch.column(0)
        if pa.types.is_integer(array.type) or pa.types.is_floating(array.type):
            total += int(array.to_pandas().fillna(0).sum())
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--with-token-counts", action="store_true")
    args = parser.parse_args()

    root = Path(args.curated_dir)
    report_path = root / "split_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    split_rows = {
        "train": parquet_rows(root / "train.parquet"),
        "validation": parquet_rows(root / "validation.parquet"),
        "test": parquet_rows(root / "test.parquet"),
    }
    total_rows = sum(split_rows.values())
    summary = {
        "curated_dir": str(root),
        "all_rows": parquet_rows(root / "all.parquet"),
        "train_rows": split_rows["train"],
        "validation_rows": split_rows["validation"],
        "test_rows": split_rows["test"],
        "split_percent": {key: (value / total_rows if total_rows else 0.0) for key, value in split_rows.items()},
        "file_bytes": {
            "all": file_bytes(root / "all.parquet"),
            "train": file_bytes(root / "train.parquet"),
            "validation": file_bytes(root / "validation.parquet"),
            "test": file_bytes(root / "test.parquet"),
        },
        "report_total": report.get("stats", {}).get("total_records"),
        "datasets": report.get("stats", {}).get("dataset_counts", {}),
        "families": report.get("stats", {}).get("family_counts", {}),
        "errors": report.get("errors", []),
    }
    if args.with_token_counts:
        token_counts = {
            "train": sum_numeric_column(root / "train.parquet", "estimated_tokens"),
            "validation": sum_numeric_column(root / "validation.parquet", "estimated_tokens"),
            "test": sum_numeric_column(root / "test.parquet", "estimated_tokens"),
        }
        token_counts["total"] = sum(token_counts.values())
        summary["estimated_tokens"] = token_counts
        summary["tokens_per_parameter"] = {
            "8m": token_counts["total"] / 8_000_000,
            "15m": token_counts["total"] / 15_000_000,
            "35m": token_counts["total"] / 35_000_000,
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
