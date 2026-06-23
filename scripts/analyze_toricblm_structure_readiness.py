#!/usr/bin/env python3
"""Audit ToricBLM FoT records for structure-training readiness.

The audit distinguishes structure *association* records, which contain PDB,
AlphaFold, site, function, or annotation hooks, from coordinate-bearing records
that can drive native structure flow-matching, contact, and distogram losses.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


COORD_KEYS = {
    "coordinates",
    "coords",
    "atom_positions",
    "atom37",
    "ca_coordinates",
    "backbone_coordinates",
    "structure_coordinates",
}


def load_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def contains_coordinate_key(obj: Any) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in COORD_KEYS:
                return True
            if contains_coordinate_key(value):
                return True
    elif isinstance(obj, list):
        return any(contains_coordinate_key(value) for value in obj[:64])
    return False


def text_has_structure_hook(text: str) -> bool:
    lower = text.lower()
    return any(
        needle in lower
        for needle in (
            "pdb:",
            "pdb_",
            "alphafold",
            "afdb",
            "esmfold",
            "binding site",
            "active site",
            "catalytic activity",
            "ec:",
            "go:",
        )
    )


def summarize(path: Path, max_rows: int | None = None) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_dataset: Counter[str] = Counter()
    coordinate_examples: list[str] = []
    structure_hook_examples: list[str] = []
    pf = pq.ParquetFile(path)
    seen = 0
    for batch in pf.iter_batches(batch_size=256):
        for row in pa.Table.from_batches([batch]).to_pylist():
            if max_rows is not None and seen >= max_rows:
                break
            seen += 1
            dataset = str(row.get("dataset") or row.get("dataset_name") or row.get("source_dataset") or "unknown")
            by_dataset[dataset] += 1
            graph = load_json(row.get("graph_json"))
            forest = load_json(row.get("thought_forest_json"))
            enrichment = load_json(row.get("enrichment_status_json")) or {}
            present = enrichment.get("present", {}) if isinstance(enrichment, dict) else {}
            row_text = " ".join(
                str(row.get(key) or "")
                for key in ("text", "annotation_text", "sequence", "entry_id", "record_id")
            )[:20000]
            has_hook = bool(
                present.get("structure_lookup_hooks")
                or present.get("sites_or_domains_in_text")
                or present.get("kinetic_parameters_in_text")
                or present.get("ec_numbers")
                or present.get("go_terms")
                or present.get("function_text")
                or text_has_structure_hook(row_text)
            )
            has_coords = contains_coordinate_key(graph) or contains_coordinate_key(forest) or contains_coordinate_key(row)
            if has_hook:
                counts["structure_association_records"] += 1
                if len(structure_hook_examples) < 5:
                    structure_hook_examples.append(str(row.get("record_id") or "unknown"))
            if has_coords:
                counts["coordinate_bearing_records"] += 1
                if len(coordinate_examples) < 5:
                    coordinate_examples.append(str(row.get("record_id") or "unknown"))
            if "uniprot" in dataset:
                counts["uniprot_records"] += 1
            if "selfies" in dataset or "pubchem" in dataset:
                counts["chemistry_records"] += 1
            if graph:
                nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
                edges = graph.get("edges", []) if isinstance(graph, dict) else []
                counts["graph_nodes_total"] += len(nodes)
                counts["graph_edges_total"] += len(edges)
            if forest:
                nodes = forest.get("nodes", []) if isinstance(forest, dict) else []
                edges = forest.get("edges", []) if isinstance(forest, dict) else []
                counts["fot_nodes_total"] += len(nodes)
                counts["fot_edges_total"] += len(edges)
        if max_rows is not None and seen >= max_rows:
            break
    records = max(1, seen)
    return {
        "schema": "toricblm.structure_readiness_report.v1",
        "input": str(path),
        "records": seen,
        "counts_by_dataset": dict(sorted(by_dataset.items())),
        "structure_association_records": int(counts["structure_association_records"]),
        "coordinate_bearing_records": int(counts["coordinate_bearing_records"]),
        "uniprot_records": int(counts["uniprot_records"]),
        "chemistry_records": int(counts["chemistry_records"]),
        "avg_graph_nodes": counts["graph_nodes_total"] / records,
        "avg_graph_edges": counts["graph_edges_total"] / records,
        "avg_fot_nodes": counts["fot_nodes_total"] / records,
        "avg_fot_edges": counts["fot_edges_total"] / records,
        "structure_flow_status": (
            "ready_for_coordinate_losses"
            if counts["coordinate_bearing_records"] > 0
            else "association_only_until_coordinate_records_are_added"
        ),
        "recommended_training_mode": (
            "enable flow/contact/distogram losses on coordinate-bearing batches"
            if counts["coordinate_bearing_records"] > 0
            else "train graph/FoT structure association now; keep coordinate losses dormant"
        ),
        "structure_hook_examples": structure_hook_examples,
        "coordinate_examples": coordinate_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    report = summarize(Path(args.input), max_rows=args.max_rows or None)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
