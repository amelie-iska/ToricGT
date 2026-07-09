#!/usr/bin/env python3
"""Metadata-only audit for ToricBLM graph/FoT training corpora.

The large biomedical run intentionally mixes several evidence classes:

* graph/FoT records with sequence, function, and structure coordinates;
* graph/FoT records with sequence and function only;
* non-protein graph/FoT records;
* raw train-only sequence/text shards that are graphified on the fly.

This script expands the configured glob groups, reads Parquet metadata and
schemas, and writes a compact report before training starts.  It deliberately
does not scan row contents except for optional schema inspection, so it is safe
to run on multi-million-row corpora.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


STRUCTURE_COLUMNS = {
    "structure_coordinates",
    "ca_coordinates",
    "backbone_coordinates",
    "coordinate_mask",
    "atom_symbols",
    "atom_modalities",
}
GRAPH_COLUMNS = {"graph_json", "forest_json", "thought_forest_json", "convextok_dag_json", "training_views_json"}
SEQUENCE_COLUMNS = {"sequence", "protein_sequence", "nucleotide_sequence", "SELFIES", "selfies"}
FUNCTION_COLUMNS = {
    "function",
    "annotation_text",
    "protein_name",
    "go_mf",
    "go_bp",
    "go_cc",
    "ec_numbers",
    "metadata_json",
    "text",
    "description",
}


def expand_patterns(patterns: str) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for piece in [part.strip() for part in patterns.split(",") if part.strip()]:
        for item in sorted(glob.glob(piece)):
            if item not in seen:
                files.append(Path(item))
                seen.add(item)
    return files


def classify_columns(columns: set[str]) -> str:
    has_structure = bool(columns & STRUCTURE_COLUMNS)
    has_graph = bool(columns & GRAPH_COLUMNS)
    has_sequence = bool(columns & SEQUENCE_COLUMNS)
    has_function = bool(columns & FUNCTION_COLUMNS)
    if has_structure and has_sequence and has_function:
        return "trimodal_structure_candidate"
    if has_structure and has_graph:
        return "structure_graph"
    if has_sequence and has_function and not has_structure:
        return "sequence_function_without_structure_trainable"
    if has_sequence and not has_structure:
        return "sequence_only_trainable"
    if has_graph:
        return "graph_fot_trainable"
    return "text_or_other_trainable"


def audit_group(label: str, patterns: str, *, sample_files: int) -> dict[str, Any]:
    files = expand_patterns(patterns)
    class_counts: Counter[str] = Counter()
    column_counts: Counter[str] = Counter()
    inspected: list[dict[str, Any]] = []
    rows_total = 0
    readable_files = 0
    unreadable: list[dict[str, str]] = []
    for index, path in enumerate(files):
        try:
            parquet = pq.ParquetFile(path)
            rows = int(parquet.metadata.num_rows)
            rows_total += rows
            readable_files += 1
            columns = set(parquet.schema_arrow.names)
            klass = classify_columns(columns)
            class_counts[klass] += rows
            for column in columns:
                column_counts[column] += 1
            if len(inspected) < sample_files:
                inspected.append(
                    {
                        "path": str(path),
                        "rows": rows,
                        "class": klass,
                        "columns": sorted(columns),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            unreadable.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            if index < sample_files:
                inspected.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return {
        "label": label,
        "patterns": patterns,
        "matched_files": len(files),
        "readable_files": readable_files,
        "rows_total": rows_total,
        "row_class_counts": dict(sorted(class_counts.items())),
        "file_column_presence": dict(sorted(column_counts.items())),
        "inspected": inspected,
        "unreadable_count": len(unreadable),
        "unreadable_examples": unreadable[:16],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", default=[], help="label=comma,separated,glob,patterns")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sample-files", type=int, default=4)
    parser.add_argument("--require-sequence-function-trainable", action="store_true")
    args = parser.parse_args()

    groups: list[dict[str, Any]] = []
    for item in args.group:
        if "=" not in item:
            raise SystemExit(f"--group must be label=patterns, got: {item}")
        label, patterns = item.split("=", 1)
        groups.append(audit_group(label.strip(), patterns.strip(), sample_files=max(0, args.sample_files)))

    total_class_counts: Counter[str] = Counter()
    total_rows = 0
    for group in groups:
        total_rows += int(group["rows_total"])
        total_class_counts.update({key: int(value) for key, value in group["row_class_counts"].items()})

    report = {
        "schema": "toricblm.training_corpus_audit.v1",
        "groups": groups,
        "totals": {
            "rows_total": total_rows,
            "row_class_counts": dict(sorted(total_class_counts.items())),
        },
        "sequence_function_without_structure_trainable": int(
            total_class_counts.get("sequence_function_without_structure_trainable", 0)
        ),
        "trimodal_structure_candidate": int(total_class_counts.get("trimodal_structure_candidate", 0)),
    }
    if args.require_sequence_function_trainable and report["sequence_function_without_structure_trainable"] <= 0:
        raise SystemExit(
            "No sequence/function-only trainable rows were found. "
            "Rows without structure should remain in graph/FoT training streams."
        )

    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "toricblm_training_corpus_audit:"
        f" rows={total_rows}"
        f" seq_function_no_structure={report['sequence_function_without_structure_trainable']}"
        f" trimodal_structure={report['trimodal_structure_candidate']}"
        f" output={path}"
    )


if __name__ == "__main__":
    main()
