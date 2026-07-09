#!/usr/bin/env python3
"""Build strict per-modality ToricBLM structure readiness manifests.

The output is a compact JSON contract for training and reports.  It counts
actual coordinate-bearing Parquet rows and marks missing modalities unavailable
instead of replacing them with synthetic examples or inferred stand-ins.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


COORD_COLUMNS = {
    "structure_coordinates",
    "ca_coordinates",
    "backbone_coordinates",
    "coordinate_mask",
}
IDENTITY_COLUMNS = {
    "record_id",
    "structure_id",
    "compound_id",
    "entry_id",
}
MODALITY_COLUMNS = {
    "structure_modality",
    "task_family",
    "dataset",
    "structure_source",
    "atom_modalities",
}


def expand(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in sorted(glob.glob(pattern)):
            path = Path(match)
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                out.append(path)
    return out


def has_coordinates(row: dict[str, Any]) -> bool:
    for key in COORD_COLUMNS:
        value = row.get(key)
        if isinstance(value, list) and len(value) > 0:
            return True
    return False


def row_modality(row: dict[str, Any]) -> str:
    explicit = " ".join(
        str(row.get(key) or "")
        for key in ("structure_modality", "task_family", "dataset", "structure_source")
    ).lower()
    if "pubchem" in explicit or "small_molecule" in explicit or "ligand" in explicit:
        return "pubchem_3d_or_ligand"
    if "protein_nucleic" in explicit or "complex" in explicit:
        return "pdb_complex"
    if "rna" in explicit and "protein" not in explicit:
        return "pdb_rna"
    if "dna" in explicit and "protein" not in explicit:
        return "pdb_dna"
    if "afdb" in explicit:
        return "protein_afdb"
    if "protein" in explicit:
        return "pdb_protein"
    modalities = row.get("atom_modalities")
    if isinstance(modalities, list):
        values = {str(item).lower() for item in modalities}
        if "protein" in values and ({"rna", "dna"} & values):
            return "pdb_complex"
        if "rna" in values:
            return "pdb_rna"
        if "dna" in values:
            return "pdb_dna"
        if "protein" in values:
            return "pdb_protein"
        if "ligand" in values:
            return "pubchem_3d_or_ligand"
    return "unknown_coordinate_modality"


def file_modality_hint(path: Path) -> str | None:
    text = str(path).lower()
    if "pubchem3d" in text or "pubchem" in text:
        return "pubchem_3d_or_ligand"
    if "afdb" in text:
        return "protein_afdb"
    return None


def first_examples(path: Path, columns: list[str], *, limit: int = 5) -> list[str]:
    available = [column for column in columns if column]
    if not available:
        return []
    try:
        table = pq.read_table(path, columns=available).slice(0, limit)
    except Exception:
        return []
    examples: list[str] = []
    for row in table.to_pylist():
        value = row.get("record_id") or row.get("structure_id") or row.get("compound_id") or row.get("entry_id")
        if value is not None:
            examples.append(str(value))
    return examples


def scan_file(path: Path, *, max_rows: int) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    names = list(pf.schema_arrow.names)
    name_set = set(names)
    coord_columns = sorted(COORD_COLUMNS & name_set)
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    hinted_modality = file_modality_hint(path)
    if hinted_modality and coord_columns:
        rows = int(pf.metadata.num_rows)
        sample_columns = sorted((IDENTITY_COLUMNS & name_set) | {coord_columns[0]})
        sample = pq.read_table(path, columns=sample_columns).slice(0, min(max_rows or 32, 32)).to_pylist()
        coordinate_sample_rows = sum(1 for row in sample if has_coordinates(row))
        if sample and coordinate_sample_rows <= 0:
            counts["rows_without_coordinates"] += rows
            coordinate_rows = 0
        else:
            counts[hinted_modality] += rows
            coordinate_rows = rows
            examples[hinted_modality] = first_examples(path, sorted(IDENTITY_COLUMNS & name_set), limit=5)
        return {
            "path": str(path),
            "rows": rows,
            "coordinate_rows": coordinate_rows,
            "counts": dict(sorted(counts.items())),
            "examples": examples,
            "columns": names,
            "scan_mode": "parquet_metadata_with_coordinate_sample",
            "modality_hint": hinted_modality,
            "coordinate_columns": coord_columns,
            "coordinate_sample_rows": coordinate_sample_rows,
        }

    rows = 0
    coordinate_rows = 0
    selected_columns = sorted((COORD_COLUMNS | IDENTITY_COLUMNS | MODALITY_COLUMNS) & name_set)
    for batch in pf.iter_batches(batch_size=256, columns=selected_columns or None):
        for row in pa.Table.from_batches([batch]).to_pylist():
            if max_rows > 0 and rows >= max_rows:
                break
            rows += 1
            if not has_coordinates(row):
                counts["rows_without_coordinates"] += 1
                continue
            coordinate_rows += 1
            modality = row_modality(row)
            counts[modality] += 1
            examples.setdefault(modality, [])
            if len(examples[modality]) < 5:
                examples[modality].append(str(row.get("record_id") or row.get("structure_id") or row.get("compound_id") or "unknown"))
        if max_rows > 0 and rows >= max_rows:
            break
    return {
        "path": str(path),
        "rows": rows,
        "coordinate_rows": coordinate_rows,
        "counts": dict(sorted(counts.items())),
        "examples": examples,
        "columns": names,
        "scan_mode": "row_scan_selected_columns",
        "coordinate_columns": coord_columns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Parquet glob; may repeat.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--require", action="append", default=[], help="Required modality key, e.g. protein_afdb,pdb_rna,pdb_dna,pdb_complex,pubchem_3d_or_ligand")
    parser.add_argument("--min-coordinate-rows", type=int, default=0, help="Minimum total coordinate-bearing rows required.")
    parser.add_argument("--min-count", action="append", default=[], help="Minimum modality count as modality=count. May repeat.")
    parser.add_argument("--allow-shortfall", action="store_true", help="Record target shortfalls without failing the preflight.")
    args = parser.parse_args()
    paths = expand(args.input)
    if not paths:
        raise SystemExit(f"no input parquet files matched {args.input}")
    files = [scan_file(path, max_rows=args.max_rows_per_file) for path in paths]
    total: Counter[str] = Counter()
    total_rows = 0
    total_coordinate_rows = 0
    examples: dict[str, list[str]] = {}
    for item in files:
        total_rows += int(item["rows"])
        total_coordinate_rows += int(item["coordinate_rows"])
        for key, value in item["counts"].items():
            total[key] += int(value)
        for key, values in item["examples"].items():
            examples.setdefault(key, [])
            for value in values:
                if len(examples[key]) < 8:
                    examples[key].append(value)
    required = list(args.require)
    missing_required = [key for key in required if total.get(key, 0) <= 0]
    minimum_counts: dict[str, int] = {}
    for raw in args.min_count:
        if "=" not in raw:
            raise SystemExit(f"--min-count must be modality=count, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--min-count has empty modality key: {raw!r}")
        try:
            minimum_counts[key] = int(value)
        except ValueError as exc:
            raise SystemExit(f"invalid --min-count value: {raw!r}") from exc
    target_shortfalls: dict[str, dict[str, int]] = {}
    if args.min_coordinate_rows > 0 and total_coordinate_rows < args.min_coordinate_rows:
        target_shortfalls["total_coordinate_rows"] = {
            "available": int(total_coordinate_rows),
            "required": int(args.min_coordinate_rows),
            "missing": int(args.min_coordinate_rows - total_coordinate_rows),
        }
    for key, required_count in minimum_counts.items():
        available = int(total.get(key, 0))
        if available < required_count:
            target_shortfalls[key] = {
                "available": available,
                "required": int(required_count),
                "missing": int(required_count - available),
            }
    report = {
        "schema": "toricblm.modality_readiness_manifest.v1",
        "inputs": args.input,
        "files": files,
        "file_count": len(files),
        "rows": total_rows,
        "coordinate_rows": total_coordinate_rows,
        "modality_counts": dict(sorted(total.items())),
        "examples": examples,
        "required_modalities": required,
        "missing_required_modalities": missing_required,
        "minimum_coordinate_rows": int(args.min_coordinate_rows),
        "minimum_modality_counts": minimum_counts,
        "target_shortfalls": target_shortfalls,
        "target_shortfall_policy": "allowed_use_maximum_available" if args.allow_shortfall else "strict_fail",
        "ready": not missing_required and (args.allow_shortfall or not target_shortfalls),
        "policy": "No modality is inferred. Missing required coordinate modalities are reported, not replaced.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if missing_required:
        raise SystemExit(f"missing required modalities: {', '.join(missing_required)}")
    if target_shortfalls and not args.allow_shortfall:
        raise SystemExit("structure target shortfall: " + json.dumps(target_shortfalls, sort_keys=True))


if __name__ == "__main__":
    main()
