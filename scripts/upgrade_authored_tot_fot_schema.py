#!/usr/bin/env python3
"""Upgrade authored graph/Tree-of-Thought shards to canonical FoT schema."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_uniprot_fot_dataset import (  # noqa: E402
    DEFAULT_CONVEXTOK,
    convextok_dag,
    deterministic_vector,
    index_convextok_tokens,
    load_convextok_tokens,
    short_hash,
    stable_json,
)


REQUIRED_COLUMNS = ("forest_json", "thought_forest_json", "convextok_dag_json", "training_views_json")


def expand_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for raw in sorted(glob.glob(pattern)):
            path = Path(raw)
            # Keep distinct symlink paths.  Late-stage three-pass datasets often
            # intentionally point multiple pass files at different shared-cache
            # targets; every visible training path must satisfy the FoT schema.
            key = str(path)
            if key not in seen:
                files.append(path)
                seen.add(key)
    return files


def parse_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def htext(value: Any, n: int = 16) -> str:
    text = value if isinstance(value, str) else stable_json(value)
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=16).hexdigest()[:n]


def normalize_graph(row: dict[str, Any]) -> dict[str, Any]:
    graph = parse_json(row.get("graph_json"), {})
    if not isinstance(graph, dict):
        graph = {}
    source = parse_json(row.get("source_row_json"), {})
    if not graph.get("nodes") and source:
        graph["nodes"] = parse_json(source.get("nodes_json"), [])
    if not graph.get("edges") and source:
        graph["edges"] = parse_json(source.get("edges_json"), [])
    graph.setdefault("nodes", [])
    graph.setdefault("edges", [])
    graph.setdefault("record_id", row.get("record_id") or row.get("id") or "authored_tot_" + htext(row.get("text") or row))
    graph.setdefault("dataset", row.get("dataset") or row.get("source") or "authored_graph_tot")
    graph.setdefault("task_family", row.get("task_family") or graph.get("task_family") or "")
    graph.setdefault("domain", row.get("domain") or graph.get("domain") or "")
    graph.setdefault("topic", row.get("topic") or graph.get("topic") or "")
    graph.setdefault("split", row.get("split") or "train")
    graph.setdefault("train_only", bool(row.get("train_only", True)))
    graph.setdefault("targets", parse_json(row.get("targets_json"), {}))
    graph.setdefault("metadata", parse_json(row.get("metadata_json"), {}))
    graph.setdefault("split_cluster", row.get("split_cluster") or "authored_tot_" + htext(graph.get("record_id")))
    clean_nodes = []
    for idx, node in enumerate(graph.get("nodes") or []):
        if not isinstance(node, dict):
            node = {"id": f"n{idx}", "type": "thought", "text": str(node)}
        node = dict(node)
        node.setdefault("id", f"n{idx}")
        node.setdefault("type", "thought")
        node.setdefault("text", str(node.get("label") or node.get("content") or ""))
        clean_nodes.append(node)
    graph["nodes"] = clean_nodes
    node_ids = {str(node.get("id")) for node in clean_nodes}
    clean_edges = []
    for idx, edge in enumerate(graph.get("edges") or []):
        if not isinstance(edge, dict):
            continue
        edge = dict(edge)
        edge.setdefault("id", f"e{idx}")
        edge.setdefault("type", "thought_transition")
        edge.setdefault("directed", True)
        if edge.get("source") in node_ids and edge.get("target") in node_ids:
            clean_edges.append(edge)
    graph["edges"] = clean_edges
    metadata = graph.setdefault("metadata", {})
    metadata.setdefault(
        "tokengt_tropical_toric_fields",
        {
            "node_token_order": [str(node.get("id")) for node in clean_nodes],
            "edge_token_order": [str(edge.get("id")) for edge in clean_edges],
            "structural_channels": ["problem", "branch", "candidate", "merge", "verifier", "answer"],
            "tropical_active_support_nodes": active_support_nodes(graph),
            "toric_phase_basis": "authored_tot_deterministic_phase",
            "convextok_alignment": "authored ToT graph text is byte-packed with ConvexTok DAG sidecar",
        },
    )
    metadata.setdefault("quality_flags", {"authored_reasoning_graph": True, "contains_imported_reasoning_trace": True})
    return graph


def active_support_nodes(graph: dict[str, Any]) -> list[str]:
    targets = graph.get("targets") if isinstance(graph.get("targets"), dict) else {}
    support = targets.get("active_support_nodes")
    if not support and isinstance(graph.get("reward"), dict):
        support = graph["reward"].get("active_support_nodes")
    if isinstance(support, str):
        support = parse_json(support, [])
    if not isinstance(support, list) or not support:
        support = [str(node.get("id")) for node in graph.get("nodes", []) if str(node.get("type", "")).lower() in {"answer", "verifier", "merge", "consensus"}]
    if not support:
        support = [str(node.get("id")) for node in graph.get("nodes", [])[-3:]]
    return [str(item) for item in support]


def build_forest(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    types = sorted({str(node.get("type") or "thought") for node in nodes})
    return {
        "forest_type": "authored_graph_tree_of_thought_forest",
        "tree_count": max(1, len(types)),
        "sparse_activation_policy": "activate authored reasoning branches by graph node type and edge topology",
        "trees": [
            {
                "tree_id": f"authored_{node_type}_tree",
                "root": str(nodes[0].get("id")) if nodes else "root",
                "active": True,
                "candidate_nodes": [str(node.get("id")) for node in nodes if str(node.get("type") or "thought") == node_type],
                "purpose": f"authored {node_type} reasoning evidence",
            }
            for node_type in types
        ],
        "consensus": {"leaf_selection_rule": "active support nodes and verifier/answer leaves", "consensus_nodes": active_support_nodes(graph)},
        "trajectory_balance_metadata": {
            "forward_policy_basis": "authored graph expansion",
            "backward_policy_basis": "contract authored branches toward problem root",
            "reward_key": "graph_json.targets.gflownet_reward.reward",
            "edge_count": len(edges),
        },
    }


def build_thought_forest(graph: dict[str, Any]) -> dict[str, Any]:
    record_id = str(graph.get("record_id"))
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    support = set(active_support_nodes(graph))
    thought_nodes = [
        {
            "id": "ft_root",
            "label": "authored reasoning source graph",
            "tree_id": "root",
            "graph_node_id": str(nodes[0].get("id")) if nodes else "",
            "budget_level": 0,
            "active": True,
            "purpose": "start from the authored Tree-of-Thought graph",
            "latent_coordinates": deterministic_vector(record_id + ":ft_root", dims=8),
        }
    ]
    thought_edges: list[dict[str, Any]] = []

    def add_edge(source: str, target: str, edge_type: str, tree_id: str, rationale: str, weight: float = 1.0) -> None:
        thought_edges.append(
            {
                "id": "fte_" + short_hash([record_id, source, target, edge_type], 12),
                "source": source,
                "target": target,
                "type": edge_type,
                "tree_id": tree_id,
                "directed": True,
                "rationale": rationale,
                "weight": round(float(weight), 6),
            }
        )

    graph_to_ft: dict[str, str] = {}
    for idx, node in enumerate(nodes):
        gid = str(node.get("id"))
        tree_id = f"authored_{str(node.get('type') or 'thought')}_tree"
        ft_id = f"ft_node_{idx:04d}"
        graph_to_ft[gid] = ft_id
        thought_nodes.append(
            {
                "id": ft_id,
                "label": str(node.get("text") or gid)[:160],
                "tree_id": tree_id,
                "graph_node_id": gid,
                "budget_level": 1 + idx,
                "active": True,
                "purpose": f"authored {node.get('type', 'thought')} node; support={gid in support}",
                "latent_coordinates": deterministic_vector(record_id + ":" + ft_id, dims=8),
            }
        )
        add_edge("ft_root", ft_id, "tree_activation" if idx == 0 else "expansion", tree_id, f"Expose authored graph node {gid}.")
    for edge in edges:
        source = graph_to_ft.get(str(edge.get("source")))
        target = graph_to_ft.get(str(edge.get("target")))
        if source and target:
            add_edge(source, target, "authored_transition", f"authored_{edge.get('type', 'transition')}_tree", str(edge.get("text") or edge.get("type") or "authored edge"), 1.0)
    consensus_id = "ft_consensus"
    thought_nodes.append(
        {
            "id": consensus_id,
            "label": "authored reasoning consensus",
            "tree_id": "consensus",
            "graph_node_id": next(iter(support), ""),
            "budget_level": len(thought_nodes) + 1,
            "active": True,
            "purpose": "merge active support, verifier, and answer nodes",
            "latent_coordinates": deterministic_vector(record_id + ":" + consensus_id, dims=8),
        }
    )
    for gid in support:
        ft_id = graph_to_ft.get(str(gid))
        if ft_id:
            add_edge(ft_id, consensus_id, "consensus", "consensus", "Active support node contributes to the authored consensus target.", 0.9)
    return {
        "schema": "toricgt.biomed_source_grounded_forest_of_thought.v1",
        "forest_type": "authored_graph_tree_of_thought_fot",
        "source_record_id": record_id,
        "trees": [{"tree_id": item["tree_id"], "label": item["tree_id"].replace("_", " ")} for item in thought_nodes if item["tree_id"] != "root"],
        "nodes": thought_nodes,
        "edges": thought_edges,
        "sparse_activation_policy": "activate authored branches, alternatives, merges, verifier, and consensus leaves from the graph topology",
        "dynamic_self_correction": {
            "enabled": True,
            "policy": "preserve authored rejected branches and verifier nodes as explicit correction evidence when present",
        },
        "consensus": {"node": consensus_id, "leaf_selection_rule": "active support plus verifier/answer leaves"},
        "gflownet_training": {
            "state_space": "authored graph-of-thought nodes plus FoT nodes in embedding space",
            "forward_actions": ["tree_activation", "expansion", "authored_transition", "consensus"],
            "backward_actions": ["contract_to_parent", "remove_unverified_leaf", "return_to_problem_root"],
            "reward_key": "graph_json.targets.gflownet_reward.reward",
            "trajectory_balance_compatible": True,
        },
    }


def needs_upgrade(row: dict[str, Any]) -> bool:
    if any(row.get(column) in (None, "") for column in REQUIRED_COLUMNS):
        return True
    thought = parse_json(row.get("thought_forest_json"), {})
    convextok = parse_json(row.get("convextok_dag_json"), {})
    return thought.get("schema") != "toricgt.biomed_source_grounded_forest_of_thought.v1" or convextok.get("schema") != "toricgt.convextok_tokenization_dag.v1"


def upgrade_row(row: dict[str, Any], *, convextok_tokens: dict[int, list[dict[str, Any]]], convextok_max_bytes: int) -> tuple[dict[str, Any], bool]:
    if not needs_upgrade(row):
        return row, False
    graph = normalize_graph(row)
    forest = build_forest(graph)
    thought_forest = build_thought_forest(graph)
    text = str(row.get("text") or "")
    if not text:
        text = "\n".join(str(node.get("text") or node.get("id")) for node in graph.get("nodes", []))
    convextok_view = convextok_dag(text, convextok_tokens, max_bytes=convextok_max_bytes)
    training_views = {
        "schema": "toricgt.authored_tot_training_views.v1",
        "views": {
            "graph_in_graph_out": {
                "input_nodes": graph["metadata"]["tokengt_tropical_toric_fields"]["node_token_order"],
                "input_edges": graph["metadata"]["tokengt_tropical_toric_fields"]["edge_token_order"],
                "target_kind": "authored_reasoning_graph",
            },
            "forest_of_thought": {
                "thought_forest_column": "thought_forest_json",
                "tree_count": len(thought_forest.get("trees", [])),
                "edge_count": len(thought_forest.get("edges", [])),
                "node_count": len(thought_forest.get("nodes", [])),
            },
            "convextok_flattening": {
                "convextok_dag_column": "convextok_dag_json",
                "rounded_path_edges": len(convextok_view.get("rounded_shortest_path", [])),
                "tokenizer": convextok_view.get("tokenizer"),
            },
            "graphcg_axes": ["problem", "branch", "alternative", "verifier", "consensus", "domain"],
            "tropical_toric": graph["metadata"]["tokengt_tropical_toric_fields"],
        },
    }
    out = dict(row)
    out.update(
        {
            "record_id": out.get("record_id") or graph.get("record_id"),
            "entry_id": out.get("entry_id") or graph.get("record_id"),
            "annotation_text": out.get("annotation_text") or text,
            "graph_json": json.dumps(graph, ensure_ascii=True, sort_keys=True),
            "forest_json": json.dumps(forest, ensure_ascii=True, sort_keys=True),
            "thought_forest_json": json.dumps(thought_forest, ensure_ascii=True, sort_keys=True),
            "convextok_dag_json": json.dumps(convextok_view, ensure_ascii=True, sort_keys=True),
            "training_views_json": json.dumps(training_views, ensure_ascii=True, sort_keys=True),
            "enrichment_status_json": out.get("enrichment_status_json")
            or json.dumps({"schema": "toricgt.authored_tot_enrichment_status.v1", "authored_graph": True}, ensure_ascii=True, sort_keys=True),
            "leakage_signature_json": out.get("leakage_signature_json")
            or json.dumps(
                {
                    "schema": "toricgt.authored_tot_leakage_signature.v1",
                    "split_cluster": graph.get("split_cluster"),
                    "source": graph.get("dataset"),
                    "record_hash": htext(graph.get("record_id"), 32),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            "metadata_json": json.dumps(graph.get("metadata", {}), ensure_ascii=True, sort_keys=True),
            "content_hash": out.get("content_hash")
            or hashlib.sha256(stable_json({"graph": graph, "forest": forest, "thought_forest": thought_forest, "convextok": convextok_view}).encode("utf-8")).hexdigest(),
            "group_hash": out.get("group_hash") or htext(graph.get("split_cluster") or graph.get("record_id"), 64),
            "quality_flags_json": out.get("quality_flags_json") or json.dumps(graph.get("metadata", {}).get("quality_flags", {}), ensure_ascii=True, sort_keys=True),
        }
    )
    return out, True


def upgrade_file(path: Path, *, convextok_tokens: dict[int, list[dict[str, Any]]], convextok_max_bytes: int, dry_run: bool) -> dict[str, Any]:
    rows = pq.read_table(path).to_pylist()
    out_rows = []
    upgraded = 0
    for row in rows:
        new_row, changed = upgrade_row(row, convextok_tokens=convextok_tokens, convextok_max_bytes=convextok_max_bytes)
        out_rows.append(new_row)
        upgraded += int(changed)
    if upgraded and not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(pa.Table.from_pylist(out_rows), tmp, compression="zstd")
        tmp.replace(path)
    return {"path": str(path), "rows": len(rows), "upgraded": upgraded, "dry_run": dry_run}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--convextok-tokenizer", type=Path, default=DEFAULT_CONVEXTOK)
    parser.add_argument("--convextok-max-bytes", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-files", type=int, default=0)
    args = parser.parse_args()
    files = expand_inputs(args.input)
    if args.limit_files:
        files = files[: int(args.limit_files)]
    if not files:
        raise SystemExit("No input Parquet files matched.")
    convextok_tokens = index_convextok_tokens(load_convextok_tokens(args.convextok_tokenizer, max_token_bytes=96))
    reports = [
        upgrade_file(path, convextok_tokens=convextok_tokens, convextok_max_bytes=int(args.convextok_max_bytes), dry_run=bool(args.dry_run))
        for path in files
    ]
    print(json.dumps({"files": len(files), "reports": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
