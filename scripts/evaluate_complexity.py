#!/usr/bin/env python3
"""Evaluate Kolmogorov-style complexity metrics on curated Parquet rows.

This script is diagnostic-only.  It does not train or update a model.  It can
optionally log aggregate metrics to an existing W&B project/run namespace.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from toricgt.complexity import (
    canonical_graph_bytes,
    compress_len,
    conditional_compress_len,
    graph_mdl_metrics,
    normalized_compression_distance,
    relative_complexity,
    relative_complexity_reward,
    shortest_known_program_len,
)


TEXT_FIELDS = ("text", "question", "reasoning", "solution", "answer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-glob", default="data/curated_hf_shards/validation/*.parquet")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", default="outputs/complexity")
    parser.add_argument("--compressors", nargs="+", default=["zlib", "lzma"])
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name", default="complexity-eval")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def iter_rows(data_glob: str, samples: int, seed: int):
    files = sorted(glob.glob(data_glob))
    rng = random.Random(seed)
    rng.shuffle(files)
    emitted = 0
    for file_path in files:
        parquet = pq.ParquetFile(file_path)
        columns = [name for name in (*TEXT_FIELDS, "graph_json", "dataset", "task_family") if name in parquet.schema.names]
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            table = batch.to_pydict()
            rows = [dict(zip(table, values)) for values in zip(*table.values())]
            rng.shuffle(rows)
            for row in rows:
                yield row
                emitted += 1
                if emitted >= samples:
                    return


def row_text(row: dict[str, Any]) -> bytes:
    parts = []
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts).encode("utf-8", errors="replace")


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def aggregate(rows: list[dict[str, Any]], compressors: list[str]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}

    def add(name: str, value: float) -> None:
        buckets.setdefault(name, []).append(float(value))

    for row in rows:
        text = row_text(row)
        graph_raw = row.get("graph_json")
        graph_bytes = canonical_graph_bytes(graph_raw) if isinstance(graph_raw, str) and graph_raw.strip() else b""
        for compressor in compressors:
            if text:
                text_k = compress_len(text, compressor)
                add(f"complexity/text_k_{compressor}", text_k)
                add(f"complexity/text_k_{compressor}_per_byte", text_k / max(1, len(text)))
            if graph_bytes:
                graph_k = compress_len(graph_bytes, compressor)
                add(f"complexity/graph_k_{compressor}", graph_k)
                add(f"complexity/graph_k_{compressor}_per_byte", graph_k / max(1, len(graph_bytes)))
            if text and graph_bytes:
                graph_cond = conditional_compress_len(text, graph_bytes, compressor)
                text_cond = conditional_compress_len(graph_bytes, text, compressor)
                graph_known = shortest_known_program_len(graph_bytes, compressor=compressor, references=(text,))
                text_known = shortest_known_program_len(text, compressor=compressor, references=(graph_bytes,))
                add(f"complexity/graph_cond_k_{compressor}", graph_cond)
                add(f"complexity/text_cond_k_{compressor}", text_cond)
                add(f"complexity/graph_relative_to_text_helper_k_{compressor}", relative_complexity(graph_cond, graph_k))
                add(f"complexity/text_relative_to_graph_helper_k_{compressor}", relative_complexity(text_cond, text_k))
                add(f"complexity/graph_shortest_known_program_k_{compressor}", graph_known)
                add(f"complexity/text_shortest_known_program_k_{compressor}", text_known)
                add(f"complexity/graph_relative_k_reward_{compressor}", relative_complexity_reward(graph_known, graph_k))
                add(f"complexity/text_relative_k_reward_{compressor}", relative_complexity_reward(text_known, text_k))
                add(
                    f"complexity/text_graph_information_symmetry_gap_k_{compressor}",
                    abs(float(text_k - text_cond) - float(graph_k - graph_cond)),
                )
                add(f"complexity/text_graph_ncd_{compressor}", normalized_compression_distance(text, graph_bytes, compressor))
        if graph_bytes and isinstance(graph_raw, str):
            for key, value in graph_mdl_metrics(graph_raw, prefix="complexity/graph_mdl", compressors=tuple(compressors)).items():
                add(key, value)

    summary = {f"{key}_mean": mean(values) for key, values in sorted(buckets.items())}
    summary.update({f"{key}_min": float(min(values)) for key, values in sorted(buckets.items()) if values})
    summary.update({f"{key}_max": float(max(values)) for key, values in sorted(buckets.items()) if values})
    summary["complexity/samples"] = float(len(rows))
    return summary


def main() -> None:
    args = parse_args()
    rows = list(iter_rows(args.data_glob, samples=args.samples, seed=args.seed))
    summary = aggregate(rows, compressors=args.compressors)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "complexity_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        run.log(summary)
        run.finish()


if __name__ == "__main__":
    main()
