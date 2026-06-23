#!/usr/bin/env python3
"""Split leakage-aware ToricBLM FoT Parquet records into train/val/test files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="toricblm_fot")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    pf = pq.ParquetFile(input_path)
    for batch in pf.iter_batches(batch_size=256):
        for row in pa.Table.from_batches([batch]).to_pylist():
            split = str(row.get("split") or "train")
            if split not in rows_by_split:
                split = "train"
            rows_by_split[split].append(row)
    outputs = {}
    for split, rows in rows_by_split.items():
        if not rows:
            continue
        path = output_dir / f"{args.prefix}_{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd", use_dictionary=True)
        outputs[split] = str(path)
    counts = {split: len(rows) for split, rows in rows_by_split.items()}
    cluster_counts = Counter()
    for rows in rows_by_split.values():
        for row in rows:
            cluster_counts[str(row.get("split_cluster"))] += 1
    report = {
        "schema": "toricblm.fot_split_shard_report.v1",
        "input": str(input_path),
        "outputs": outputs,
        "split_counts": counts,
        "cluster_count": len(cluster_counts),
        "max_cluster_size": max(cluster_counts.values()) if cluster_counts else 0,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
