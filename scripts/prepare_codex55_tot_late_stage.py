#!/usr/bin/env python3
"""Prepare AmelieSchreiber/codex5.5_ToT as a train-only ToricBLM late stage.

The source dataset is already graph structured.  This script preserves that
structure in ``graph_json`` rows that the OAI Parameter-Golf adaptation can
consume through ``GraphParquetTokenStream``.  It intentionally writes no
validation or test split; FineWeb remains the BPB validation source.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi


DEFAULT_REPO = "AmelieSchreiber/codex5.5_ToT"
DEFAULT_OUTPUT = (
    "/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/"
    "toricgt/codex55_tot_late_stage"
)


def parse_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def normalize_node(node: Any, index: int) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {"id": f"n{index}", "type": "thought", "text": str(node)}
    return {
        "id": str(node.get("id") or node.get("node_id") or f"n{index}"),
        "type": str(node.get("type") or node.get("kind") or "thought"),
        "text": str(node.get("content") or node.get("text") or node.get("label") or ""),
        "metadata": {
            key: value
            for key, value in node.items()
            if key not in {"id", "node_id", "type", "kind", "content", "text", "label"}
        },
    }


def normalize_edge(edge: Any, index: int) -> dict[str, Any]:
    if not isinstance(edge, dict):
        return {"id": f"e{index}", "source": "", "target": "", "type": "edge", "text": str(edge)}
    return {
        "id": str(edge.get("id") or edge.get("edge_id") or f"e{index}"),
        "source": str(edge.get("source") or edge.get("src") or edge.get("from") or ""),
        "target": str(edge.get("target") or edge.get("dst") or edge.get("to") or ""),
        "type": str(edge.get("type") or edge.get("kind") or "thought_transition"),
        "text": str(edge.get("causal_note") or edge.get("content") or edge.get("text") or edge.get("label") or ""),
        "metadata": {
            key: value
            for key, value in edge.items()
            if key
            not in {
                "id",
                "edge_id",
                "source",
                "src",
                "from",
                "target",
                "dst",
                "to",
                "type",
                "kind",
                "causal_note",
                "content",
                "text",
                "label",
            }
        },
    }


def build_record(row: dict[str, Any], row_index: int, repo_id: str) -> dict[str, Any]:
    nodes = [normalize_node(node, i) for i, node in enumerate(parse_json(row.get("nodes_json"), []))]
    edges = [normalize_edge(edge, i) for i, edge in enumerate(parse_json(row.get("edges_json"), []))]
    targets = parse_json(row.get("targets_json"), {})
    metadata = parse_json(row.get("metadata_json"), {})
    split_cluster = parse_json(row.get("split_cluster_json"), {})
    record_id = str(row.get("id") or f"codex55_tot_{row_index:06d}")
    graph_payload = {
        "record_id": record_id,
        "dataset": repo_id,
        "split": "train",
        "task_family": row.get("task_family"),
        "domain": row.get("domain"),
        "topic": row.get("topic"),
        "trajectory_kind": "tree_of_thought_forest_of_thought",
        "train_only": True,
        "upsample_policy": "late_stage_three_pass_train_only",
        "nodes": nodes,
        "edges": [edge for edge in edges if edge["source"] and edge["target"]],
        "targets": targets,
        "metadata": metadata,
        "split_cluster": split_cluster,
        "safety": {
            "medical": bool(row.get("medical")),
            "educational_only": bool(row.get("educational_only")),
            "mechanism_only": bool(row.get("mechanism_only")),
            "bio_design_safety": row.get("bio_design_safety"),
            "no_sequence_design": bool(row.get("no_sequence_design")),
            "not_patient_specific": bool(row.get("not_patient_specific")),
        },
        "reward": {
            "gflownet_reward": row.get("gflownet_reward"),
            "tropical_margin": row.get("tropical_margin"),
            "active_support_nodes": parse_json(row.get("active_support_nodes_json"), []),
        },
    }
    text_parts = [
        f"dataset: {repo_id}",
        f"task_family: {row.get('task_family')}",
        f"domain: {row.get('domain')}",
        f"topic: {row.get('topic')}",
        "nodes:",
    ]
    text_parts.extend(f"- {node['id']} [{node['type']}]: {node['text']}" for node in nodes)
    if graph_payload["edges"]:
        text_parts.append("edges:")
        text_parts.extend(
            f"- {edge['source']} -> {edge['target']} [{edge['type']}]: {edge['text']}"
            for edge in graph_payload["edges"]
        )
    return {
        "id": record_id,
        "source": repo_id,
        "dataset": "codex5.5_ToT",
        "task_family": str(row.get("task_family") or ""),
        "domain": str(row.get("domain") or ""),
        "topic": str(row.get("topic") or ""),
        "split": "train",
        "train_only": True,
        "text": "\n".join(text_parts),
        "graph_json": json.dumps(graph_payload, ensure_ascii=True, sort_keys=True),
        "metadata_json": json.dumps(metadata, ensure_ascii=True, sort_keys=True),
        "targets_json": json.dumps(targets, ensure_ascii=True, sort_keys=True),
        "source_row_json": json.dumps(row, ensure_ascii=True, sort_keys=True),
    }


def write_shards(records: list[dict[str, Any]], shard_dir: Path, shard_size: int) -> list[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob("*.parquet"):
        old.unlink()
    paths: list[Path] = []
    for start in range(0, len(records), shard_size):
        shard_records = records[start : start + shard_size]
        path = shard_dir / f"codex55_tot_train_{start // shard_size:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(shard_records), path, compression="zstd")
        paths.append(path)
    return paths


def make_repeated_symlinks(shards: list[Path], repeated_dir: Path, passes: int) -> list[Path]:
    repeated_dir.mkdir(parents=True, exist_ok=True)
    for old in repeated_dir.glob("*.parquet"):
        old.unlink()
    repeated: list[Path] = []
    for pass_idx in range(max(1, passes)):
        for shard in shards:
            link = repeated_dir / f"pass{pass_idx:02d}_{shard.name}"
            try:
                link.symlink_to(shard.resolve())
            except OSError:
                import shutil

                shutil.copy2(shard, link)
            repeated.append(link)
    return repeated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    info = HfApi().dataset_info(args.repo_id, files_metadata=True)
    dataset = load_dataset(args.repo_id, split="train")
    if args.max_records > 0:
        dataset = dataset.select(range(min(args.max_records, len(dataset))))
    records = [build_record(dict(row), idx, args.repo_id) for idx, row in enumerate(dataset)]
    train_dir = output_dir / "train"
    repeated_dir = output_dir / "train_3pass"
    shards = write_shards(records, train_dir, max(1, int(args.shard_size)))
    repeated = make_repeated_symlinks(shards, repeated_dir, int(args.passes))

    domain_counts = Counter(record["domain"] for record in records)
    family_counts = Counter(record["task_family"] for record in records)
    manifest = {
        "repo_id": args.repo_id,
        "source_sha": getattr(info, "sha", None),
        "source_files": [s.rfilename for s in getattr(info, "siblings", [])],
        "row_count": len(records),
        "split": "train_only",
        "passes": int(args.passes),
        "train_glob": str(train_dir / "*.parquet"),
        "train_3pass_glob": str(repeated_dir / "*.parquet"),
        "shards": [str(path) for path in shards],
        "repeated_shards": [str(path) for path in repeated],
        "domain_counts": dict(domain_counts),
        "task_family_counts": dict(family_counts),
        "notes": (
            "Prepared for ToricBLM late-stage graph LM, sidecar, FoT, and GFlowNet training. "
            "No validation/test records are written; FineWeb remains the BPB validation stream."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
