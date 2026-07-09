#!/usr/bin/env python3
"""Write a combined manifest for parallel non-protein FoT curation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


REQUIRED_COLUMNS = (
    "graph_json",
    "forest_json",
    "thought_forest_json",
    "convextok_dag_json",
    "training_views_json",
)


def parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def validate_sample(path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    missing = [column for column in REQUIRED_COLUMNS if column not in names]
    if missing:
        return {"path": str(path), "ok": False, "missing_columns": missing}
    row = pq.read_table(path, columns=list(REQUIRED_COLUMNS)).slice(0, 1).to_pylist()
    if not row:
        return {"path": str(path), "ok": False, "error": "empty_file"}
    empty = [column for column in REQUIRED_COLUMNS if row[0].get(column) in (None, "")]
    if empty:
        return {"path": str(path), "ok": False, "empty_columns": empty}
    try:
        thought = json.loads(row[0]["thought_forest_json"])
        convextok = json.loads(row[0]["convextok_dag_json"])
        graph = json.loads(row[0]["graph_json"])
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "ok": False, "error": f"json_error:{type(exc).__name__}:{exc}"}
    errors = []
    if thought.get("schema") != "toricgt.biomed_source_grounded_forest_of_thought.v1":
        errors.append(f"thought_schema={thought.get('schema')}")
    if convextok.get("schema") != "toricgt.convextok_tokenization_dag.v1":
        errors.append(f"convextok_schema={convextok.get('schema')}")
    if not graph.get("nodes") or not graph.get("edges"):
        errors.append("empty_graph")
    return {"path": str(path), "ok": not errors, "errors": errors}


def build_manifest(out_dir: Path, *, min_train_rows: int, validate_all: bool) -> dict[str, Any]:
    writer_counts: dict[str, dict[str, int]] = {}
    writer_paths: dict[str, dict[str, list[str]]] = {}
    validation: list[dict[str, Any]] = []
    for modality in ("dna", "rna", "small_molecule"):
        writer_counts[modality] = {}
        writer_paths[modality] = {}
        for split in ("train", "validation", "test"):
            paths = sorted((out_dir / modality / split).glob("*.parquet"))
            writer_paths[modality][split] = [str(path) for path in paths]
            writer_counts[modality][split] = sum(parquet_rows(path) for path in paths)
            if paths:
                sample_paths = paths if validate_all else [paths[0], paths[-1]]
                seen = set()
                for path in sample_paths:
                    if path in seen:
                        continue
                    seen.add(path)
                    validation.append(validate_sample(path))
    ready = all(writer_counts.get(modality, {}).get("train", 0) >= int(min_train_rows) for modality in ("dna", "rna", "small_molecule"))
    valid = all(item.get("ok") for item in validation)
    return {
        "schema": "toricblm.nonprotein_similarity_fot_split_manifest.v1",
        "out_dir": str(out_dir),
        "parallel_finalizer": True,
        "targets": {
            "minimum_train_rows_per_modality": int(min_train_rows),
            "modalities": ["dna", "rna", "small_molecule"],
        },
        "writer_counts": writer_counts,
        "writer_paths": writer_paths,
        "validation": validation,
        "ready_for_training": bool(ready and valid),
        "notes": [
            "This manifest is generated from completed Parquet shards after parallel per-modality FoT curation.",
            "Every sampled shard must expose graph_json, forest_json, thought_forest_json, convextok_dag_json, and training_views_json.",
            "Raw upstream raw_hf_bio_scale files remain curation inputs only and are not direct training inputs.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest-name", default="toricblm_nonprotein_similarity_fot_split_manifest.json")
    parser.add_argument("--min-train-rows", type=int, default=450_000)
    parser.add_argument("--validate-all", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(args.out_dir, min_train_rows=int(args.min_train_rows), validate_all=bool(args.validate_all))
    manifest_path = args.out_dir / "manifests" / args.manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True))
    if not manifest["ready_for_training"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
