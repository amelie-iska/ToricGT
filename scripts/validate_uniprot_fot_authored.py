#!/usr/bin/env python3
"""Validate and package handwritten UniProt FoT reasoning records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


AUTHORED_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("task_family", pa.string()),
        ("domain", pa.string()),
        ("topic", pa.string()),
        ("source_json", pa.large_string()),
        ("nodes_json", pa.large_string()),
        ("edges_json", pa.large_string()),
        ("targets_json", pa.large_string()),
        ("metadata_json", pa.large_string()),
        ("split_cluster_json", pa.large_string()),
        ("content_hash", pa.string()),
        ("split", pa.string()),
    ]
)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def validate_record(record: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    required = ["id", "source", "task_family", "nodes", "edges", "targets", "metadata", "split_cluster"]
    for key in required:
        if key not in record:
            errors.append(f"missing top-level key {key}")

    source = record.get("source", {})
    if source.get("reasoning_tree_author") != "Codex":
        errors.append("source.reasoning_tree_author must be Codex")
    if source.get("content_creation_mode") != "handwritten":
        errors.append("source.content_creation_mode must be handwritten")
    if source.get("contains_imported_reasoning_trace") is not False:
        errors.append("source.contains_imported_reasoning_trace must be false")

    nodes = record.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        errors.append("nodes must be a nonempty list")
        nodes = []
    node_ids = set()
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"node {idx} is not an object")
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            errors.append(f"node {idx} missing id")
        elif node_id in node_ids:
            errors.append(f"duplicate node id {node_id}")
        else:
            node_ids.add(node_id)
        for key in ("type", "content", "symbolic_payload"):
            if key not in node:
                errors.append(f"node {node_id or idx} missing {key}")
        if "latent_coordinates" not in node and "trajectory_notes" not in node:
            errors.append(f"node {node_id or idx} missing latent coordinates or trajectory notes")

    edges = record.get("edges", [])
    if not isinstance(edges, list):
        errors.append("edges must be a list")
        edges = []
    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"edge {idx} is not an object")
            continue
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in node_ids:
            errors.append(f"edge {edge.get('id', idx)} source {src} missing")
        if tgt not in node_ids:
            errors.append(f"edge {edge.get('id', idx)} target {tgt} missing")
        if edge.get("directed") is not True:
            errors.append(f"edge {edge.get('id', idx)} must be directed")

    targets = record.get("targets", {})
    for key in ("answer", "verifier_metadata", "gflownet_reward_metadata", "active_support_nodes"):
        if key not in targets:
            errors.append(f"targets missing {key}")
    for node_id in targets.get("active_support_nodes", []) if isinstance(targets, dict) else []:
        if node_id not in node_ids:
            errors.append(f"active support node {node_id} missing")

    metadata = record.get("metadata", {})
    for key in (
        "domain",
        "topic",
        "merge_status",
        "multiple_viable_solution_status",
        "continuous_embedding_fields",
        "tokengt_tropical_toric_fields",
        "quality_flags",
    ):
        if key not in metadata:
            errors.append(f"metadata missing {key}")

    split_cluster = record.get("split_cluster", {})
    if "split" not in split_cluster:
        errors.append("split_cluster missing split")
    elif split_cluster["split"] not in {"train", "validation", "test"}:
        errors.append("split_cluster.split must be train, validation, or test")

    return [f"{path}: {error}" for error in errors]


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    split_cluster = record.get("split_cluster", {})
    return {
        "id": record["id"],
        "task_family": record.get("task_family", ""),
        "domain": metadata.get("domain", ""),
        "topic": metadata.get("topic", ""),
        "source_json": json.dumps(record.get("source", {}), ensure_ascii=True),
        "nodes_json": json.dumps(record.get("nodes", []), ensure_ascii=True),
        "edges_json": json.dumps(record.get("edges", []), ensure_ascii=True),
        "targets_json": json.dumps(record.get("targets", {}), ensure_ascii=True),
        "metadata_json": json.dumps(metadata, ensure_ascii=True),
        "split_cluster_json": json.dumps(split_cluster, ensure_ascii=True),
        "content_hash": sha256_value(record),
        "split": split_cluster.get("split", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", default="data/uniprot_fot/authored/records")
    parser.add_argument("--output-dir", default="data/uniprot_fot/authored/accepted")
    parser.add_argument("--write-parquet", action="store_true")
    args = parser.parse_args()

    records_dir = Path(args.records_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted = []
    errors = []
    for path in sorted(records_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - validation report should record all parse failures.
            errors.append(f"{path}: JSON parse error {type(exc).__name__}: {exc}")
            continue
        record_errors = validate_record(record, path)
        if record_errors:
            errors.extend(record_errors)
            continue
        accepted.append(flatten_record(record))

    jsonl_path = output_dir / "accepted_records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    parquet_path = output_dir / "accepted_records.parquet"
    if args.write_parquet and accepted:
        table = pa.Table.from_pylist(accepted, schema=AUTHORED_SCHEMA)
        pq.write_table(table, parquet_path, compression="zstd", use_dictionary=True)

    split_counts = Counter(row["split"] for row in accepted)
    domain_counts = Counter(row["domain"] for row in accepted)
    report = {
        "records_dir": str(records_dir),
        "accepted_jsonl": str(jsonl_path),
        "accepted_parquet": str(parquet_path) if parquet_path.exists() else None,
        "accepted_count": len(accepted),
        "split_counts": dict(split_counts),
        "domain_counts": dict(domain_counts),
        "errors": errors,
        "ok": not errors,
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
