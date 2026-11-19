#!/usr/bin/env python3
"""Package explicitly supplied symbolic math/graph JSONL for native admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from toricgt.config import ModelConfig
from toricgt.graph_dataset import ensure_branch_merge_graph_json
from toricgt.prepared_stream import SCHEMA, _json_object, open_prepared_stream
from toricgt.toricmora_symbolic_splits import (
    ReviewedSimilarityLink,
    SymbolicRecord,
    audit_symbolic_splits,
)

ROW_SCHEMA = pa.schema([("record_id", pa.string()), ("graph_json", pa.string())])


def read_records(paths):
    records = []
    graphs = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = _json_object(line, "symbolic JSONL row")
                    if "record" not in row or row.keys() - {"record", "graph"}:
                        raise ValueError("a row requires record metadata and may contain a graph")
                    record = SymbolicRecord(**row["record"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if record.record_id in graphs:
                    raise ValueError("duplicate record_id in symbolic input")
                records.append(record)
                graphs[record.record_id] = row.get("graph")
    return sorted(records, key=lambda record: record.record_id), graphs


def read_array(path, field):
    if path is None:
        return []
    if not path.is_file():
        raise ValueError(f"{field} must be supplied as an explicit JSON file")
    value = _json_object(path.read_bytes(), field)
    if value.keys() != {field} or not isinstance(value[field], list):
        raise ValueError(f"{field} requires a JSON object with a single {field} array")
    return value[field]


def read_links(path):
    links = []
    for value in read_array(path, "similarity_links"):
        link = ReviewedSimilarityLink(**value)
        left, right = sorted((link.left_record_id, link.right_record_id))
        links.append((left, right, link.review_ref))
    return [ReviewedSimilarityLink(*link) for link in sorted(set(links))]


def stable_json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def graph_json(record, graph, config):
    if not isinstance(graph, dict) or graph.keys() != {"nodes", "edges"}:
        raise ValueError("a train or validation row requires a graph with nodes and edges")
    nodes, edges = graph["nodes"], graph["edges"]
    if not isinstance(nodes, list) or not nodes or not isinstance(edges, list):
        raise ValueError("a symbolic graph needs a nonempty nodes array and an edges array")
    node_ids = set()
    for node in nodes:
        if (
            not isinstance(node, dict) or node.keys() != {"id", "type", "text"}
            or any(not isinstance(node[key], str) for key in node)
            or not (node["id"].strip() and node["type"].strip())
        ):
            raise ValueError("nodes require string id, type, and text fields")
        if node["id"] in node_ids:
            raise ValueError("node IDs must be unique within a symbolic graph")
        node_ids.add(node["id"])
    for edge in edges:
        if (
            not isinstance(edge, dict) or edge.keys() != {"source", "target", "type"}
            or any(not isinstance(edge[key], str) for key in edge)
            or not edge["type"].strip()
        ):
            raise ValueError("edges require string source, target, and type fields")
        if edge["source"] not in node_ids or edge["target"] not in node_ids:
            raise ValueError("edge endpoints must identify declared nodes")

    payload = stable_json({
        "record_id": record.record_id,
        "dataset": record.source,
        "nodes": nodes,
        "edges": edges,
    })
    if len(nodes) > config.max_nodes:
        raise ValueError("symbolic graph exceeds the native recipe's node capacity")
    effective = json.loads(ensure_branch_merge_graph_json(payload))
    if len(effective["edges"]) > config.max_edges:
        raise ValueError("symbolic graph exceeds the native recipe's edge capacity after graph conversion")
    return payload


def artifact(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file": path.name, "sha256": digest.hexdigest()}


def write_json(path, value):
    path.write_text(stable_json(value) + "\n", encoding="utf-8")
    return artifact(path)


def prepare_symbolic_stream(paths, output, *, similarity_links=None, retrieval_inventory=None):
    if os.path.lexists(output):
        raise FileExistsError("choose a fresh prepared output directory")
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("provide one or more explicit symbolic JSONL files")

    records, graphs = read_records(paths)
    links = read_links(similarity_links)
    retrieval_ids = read_array(retrieval_inventory, "record_ids")
    if any(not isinstance(record_id, str) or not record_id.strip() for record_id in retrieval_ids):
        raise ValueError("retrieval requires nonempty record IDs")
    if len(set(retrieval_ids)) != len(retrieval_ids):
        raise ValueError("retrieval requires unique record IDs")

    audit = audit_symbolic_splits(records, links, indexed_retrieval_record_ids=retrieval_ids)
    if audit.retrieval_exposure.violations:
        counts = stable_json(audit.retrieval_exposure.violations)
        raise ValueError(f"retrieval inventory violates reviewed split assignments: {counts}")

    by_id = {record.record_id: record for record in records}
    rows = {"train": [], "validation": []}
    config = ModelConfig()
    for assignment in audit.assignments:
        if assignment.split in rows:
            record_id = assignment.record_id
            encoded = graph_json(by_id[record_id], graphs[record_id], config)
            rows[assignment.split].append({"record_id": record_id, "graph_json": encoded})
    if not rows["train"]:
        raise ValueError("symbolic curation requires at least one admitted training record")

    with TemporaryDirectory(dir=output.parent, prefix=".prepared-symbolic-") as temporary:
        stage = Path(temporary)
        manifest = {
            "schema": SCHEMA,
            "review": write_json(stage / "review.json", {
                "records": [asdict(record) for record in records],
                "similarity_links": [asdict(link) for link in links],
            }),
            "retrieval": write_json(stage / "retrieval.json", {"record_ids": sorted(retrieval_ids)}),
        }
        for split, data in rows.items():
            manifest[split] = None
            if data:
                shard = stage / f"{split}.parquet"
                pq.write_table(pa.Table.from_pylist(data, schema=ROW_SCHEMA), shard, row_group_size=1024)
                manifest[split] = artifact(shard)
        receipt = write_json(stage / "manifest.json", manifest)
        with open_prepared_stream(stage / receipt["file"], receipt["sha256"]) as (_, admission):
            report = dict(admission)

        # Publish the verified manifest last and preserve any existing destination.
        output.mkdir(mode=0o700)
        for spec in (manifest["review"], manifest["retrieval"], manifest["train"], manifest["validation"], receipt):
            if spec is not None:
                name = spec["file"]
                os.link(stage / name, output / name)

    return {
        **report,
        "manifest": str((output / receipt["file"]).resolve()),
        "records_by_split": audit.records_by_split,
        "groups_by_split": audit.groups_by_split,
        "quarantine_groups_by_reason": audit.quarantine_groups_by_reason,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", action="append", type=Path, required=True, help="Explicit symbolic JSONL input")
    parser.add_argument("--output", type=Path, required=True, help="New directory under an existing parent")
    parser.add_argument("--similarity-links", type=Path, help="JSON object with a similarity_links array")
    parser.add_argument("--retrieval-inventory", type=Path, help="JSON object with actual indexed record_ids")
    args = parser.parse_args()
    try:
        result = prepare_symbolic_stream(
            args.records, args.output,
            similarity_links=args.similarity_links, retrieval_inventory=args.retrieval_inventory,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
