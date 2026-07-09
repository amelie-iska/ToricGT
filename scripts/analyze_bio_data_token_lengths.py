#!/usr/bin/env python3
"""Report ConvexTok token-length statistics for ToricBLM bio/FoT data."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "amelie-iska" / "parameter-golf"))

from convextok import ConvexTokTokenizer  # type: ignore  # noqa: E402


TEXT_COLUMNS = (
    "text",
    "annotation_text",
    "sequence",
    "graph_json",
    "forest_json",
    "thought_forest_json",
    "convextok_dag_json",
    "training_views_json",
    "metadata_json",
)


def row_text(row: dict[str, Any], *, max_chars: int) -> str:
    parts: list[str] = []
    for key in TEXT_COLUMNS:
        value = row.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, str):
            parts.append(value[:max_chars])
        else:
            parts.append(json.dumps(value, ensure_ascii=True, sort_keys=True)[:max_chars])
    return "\n".join(parts)


def stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    vals = sorted(values)
    def pct(q: float) -> int:
        idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
        return int(vals[idx])
    return {
        "count": len(vals),
        "min": int(vals[0]),
        "p05": pct(0.05),
        "p25": pct(0.25),
        "median": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": int(vals[-1]),
        "mean": float(statistics.fmean(vals)),
        "stdev": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
    }


def analyze_pattern(pattern: str, tokenizer: ConvexTokTokenizer, *, max_rows: int, max_chars_per_field: int) -> dict[str, Any]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    token_lengths: list[int] = []
    byte_lengths: list[int] = []
    datasets: dict[str, int] = {}
    columns_seen: set[str] = set()
    for path in paths:
        pf = pq.ParquetFile(path)
        schema_names = set(pf.schema_arrow.names)
        columns = [name for name in TEXT_COLUMNS + ("dataset",) if name in schema_names]
        columns_seen.update(columns)
        for batch in pf.iter_batches(batch_size=128, columns=columns):
            for row in pa.Table.from_batches([batch]).to_pylist():
                text = row_text(row, max_chars=max_chars_per_field)
                ids = tokenizer.encode(text)
                token_lengths.append(len(ids))
                byte_lengths.append(len(text.encode("utf-8", errors="replace")))
                dataset = str(row.get("dataset") or path.parent.name)
                datasets[dataset] = datasets.get(dataset, 0) + 1
                if max_rows > 0 and len(token_lengths) >= max_rows:
                    return {
                        "pattern": pattern,
                        "files": len(paths),
                        "sampled_rows": len(token_lengths),
                        "columns_seen": sorted(columns_seen),
                        "datasets": datasets,
                        "tokens": stats(token_lengths),
                        "bytes": stats(byte_lengths),
                        "tokens_per_byte_mean": float(sum(token_lengths) / max(1, sum(byte_lengths))),
                    }
    return {
        "pattern": pattern,
        "files": len(paths),
        "sampled_rows": len(token_lengths),
        "columns_seen": sorted(columns_seen),
        "datasets": datasets,
        "tokens": stats(token_lengths),
        "bytes": stats(byte_lengths),
        "tokens_per_byte_mean": float(sum(token_lengths) / max(1, sum(byte_lengths))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--pattern", action="append", required=True)
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--max-chars-per-field", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    tokenizer = ConvexTokTokenizer.load_json(args.tokenizer)
    report = {
        "schema": "toricblm.bio_token_length_report.v1",
        "tokenizer": args.tokenizer,
        "max_rows_per_pattern": int(args.max_rows),
        "max_chars_per_field": int(args.max_chars_per_field),
        "patterns": [
            analyze_pattern(pattern, tokenizer, max_rows=args.max_rows, max_chars_per_field=args.max_chars_per_field)
            for pattern in args.pattern
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
