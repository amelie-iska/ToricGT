#!/usr/bin/env python3
"""Upgrade coordinate-bearing ToricBLM shards to graph/FoT schema.

Older AFDB coordinate shards were structure-native but did not include the full
Forest-of-Thought and ConvexTok DAG sidecars used by the graph-in/graph-out
training path.  This script is an idempotent in-place upgrader: it preserves all
coordinate arrays and provenance fields, adds the missing FoT columns, writes a
temporary Parquet file, and atomically replaces the original shard.
"""

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

from build_afdb_structure_fot_dataset import (  # noqa: E402
    enrich_structure_graph,
    forest_for_structure,
    safe_text,
    split_for_accession,
    structure_graph,
    thought_forest_for_structure,
)
from build_uniprot_fot_dataset import (  # noqa: E402
    DEFAULT_CONVEXTOK,
    convextok_dag,
    index_convextok_tokens,
    load_convextok_tokens,
    stable_json,
)


REQUIRED_COLUMNS = (
    "graph_json",
    "forest_json",
    "thought_forest_json",
    "convextok_dag_json",
    "training_views_json",
    "enrichment_status_json",
    "leakage_signature_json",
)


def expand_inputs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for raw in sorted(glob.glob(pattern)):
            path = Path(raw)
            key = str(path.resolve())
            if key not in seen:
                out.append(path)
                seen.add(key)
    return out


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def coords_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    ca = row.get("ca_coordinates") or row.get("structure_coordinates")
    if not isinstance(ca, list) or len(ca) < 8:
        return None
    backbone = row.get("backbone_coordinates")
    if not isinstance(backbone, list) or len(backbone) != len(ca):
        backbone = [[[0.0, 0.0, 0.0] for _ in range(4)] for _ in ca]
    plddt = row.get("plddt")
    if not isinstance(plddt, list) or len(plddt) != len(ca):
        mean = float(row.get("mean_plddt") or 50.0)
        plddt = [mean for _ in ca]
    residue_names = row.get("residue_names")
    if not isinstance(residue_names, list) or len(residue_names) != len(ca):
        residue_names = ["UNK" for _ in ca]
    residue_indices = row.get("residue_indices")
    if not isinstance(residue_indices, list) or len(residue_indices) != len(ca):
        residue_indices = list(range(1, len(ca) + 1))
    chain_ids = row.get("chain_ids")
    if not isinstance(chain_ids, list) or len(chain_ids) != len(ca):
        chain_ids = ["A" for _ in ca]
    return {
        "ca_coordinates": ca,
        "backbone_coordinates": backbone,
        "residue_names": residue_names,
        "residue_indices": residue_indices,
        "chain_ids": chain_ids,
        "plddt": plddt,
        "residue_count": len(ca),
        "coordinate_residue_count": int(row.get("coordinate_residue_count") or len(ca)),
        "source_residue_count": int(row.get("source_residue_count") or len(ca)),
        "coordinate_truncated": bool(row.get("coordinate_truncated") or False),
        "mean_plddt": float(row.get("mean_plddt") or (sum(float(x) for x in plddt) / max(1, len(plddt)))),
    }


def needs_upgrade(row: dict[str, Any]) -> bool:
    for column in REQUIRED_COLUMNS:
        if row.get(column) in (None, ""):
            return True
    graph = parse_json(row.get("graph_json"))
    thought = parse_json(row.get("thought_forest_json"))
    convextok = parse_json(row.get("convextok_dag_json"))
    if graph.get("targets", {}).get("structure_flow") is None:
        return True
    if thought.get("schema") != "toricgt.biomed_source_grounded_forest_of_thought.v1":
        return True
    if convextok.get("schema") != "toricgt.convextok_tokenization_dag.v1":
        return True
    return False


def upgrade_row(row: dict[str, Any], *, convextok_tokens: dict[int, list[dict[str, Any]]], convextok_max_bytes: int) -> tuple[dict[str, Any], bool]:
    if not needs_upgrade(row):
        return row, False
    coords = coords_from_row(row)
    if coords is None:
        return row, False
    accession = safe_text(row.get("uniprot_accession") or row.get("entry_id") or row.get("record_id"), 64)
    entry_name = safe_text(row.get("entry_name") or accession, 256)
    protein_name = safe_text(row.get("protein_name") or "protein", 1024)
    sequence = safe_text(row.get("sequence"), 100000)
    function_text = safe_text(row.get("function") or row.get("annotation_text"), 2400)
    record_id = safe_text(row.get("record_id") or f"afdb_structure_fot_{accession}", 256)
    metadata = parse_json(row.get("metadata_json"))
    prediction = {
        "modelEntityId": metadata.get("model_entity_id") or metadata.get("modelEntityId") or f"AF-{accession}-F1",
        "latestVersion": metadata.get("latest_version") or metadata.get("latestVersion") or "",
        "cifUrl": row.get("structure_cif_url") or metadata.get("cif_url") or metadata.get("cifUrl") or "",
        "bcifUrl": metadata.get("bcif_url") or metadata.get("bcifUrl") or "",
    }
    graph = parse_json(row.get("graph_json"))
    if not graph.get("nodes") or not graph.get("edges"):
        graph = structure_graph(
            accession=accession,
            entry_name=entry_name,
            protein_name=protein_name,
            sequence=sequence,
            function=function_text,
            prediction=prediction,
            coords=coords,
        )
    foldseek_3di_status = safe_text(row.get("foldseek_3di_status") or "not_requested", 128)
    split = safe_text(row.get("split") or split_for_accession(accession), 32)
    graph = enrich_structure_graph(
        graph=graph,
        record_id=record_id,
        accession=accession,
        entry_name=entry_name,
        protein_name=protein_name,
        sequence=sequence,
        function_text=function_text,
        coords=coords,
        prediction=prediction,
        foldseek_3di_status=foldseek_3di_status,
        split=split,
    )
    forest = parse_json(row.get("forest_json")) or forest_for_structure(record_id, coords)
    thought_forest = thought_forest_for_structure(
        record_id=record_id,
        accession=accession,
        graph=graph,
        coords=coords,
        foldseek_3di_status=foldseek_3di_status,
    )
    text = safe_text(row.get("text"), 10000)
    if not text:
        text = (
            f"UniProt {accession} structure-function record. Protein: {protein_name}. "
            f"Function: {function_text}. Train on real coordinates for graph/FoT structure-flow training."
        )
    convextok_view = convextok_dag(text + "\n" + sequence[:512], convextok_tokens, max_bytes=convextok_max_bytes)
    training_views = {
        "schema": "toricgt.biomed_training_views.v1",
        "views": {
            "graph_in_graph_out": {
                "input_nodes": graph["metadata"]["tokengt_tropical_toric_fields"]["node_token_order"],
                "input_edges": graph["metadata"]["tokengt_tropical_toric_fields"]["edge_token_order"],
                "target_kind": graph["targets"]["answer"]["kind"],
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
            "structure_flow": graph["targets"]["structure_flow"],
            "graphcg_axes": ["sequence", "function", "coordinate_geometry", "confidence", "design_constraints"],
            "tropical_toric": graph["metadata"]["tokengt_tropical_toric_fields"],
        },
    }
    enrichment_status = {
        "schema": "toricblm.structure_enrichment_status.v1",
        "coordinate_source": row.get("structure_source") or "AFDB",
        "coordinate_available": True,
        "foldseek_3di_status": foldseek_3di_status,
        "mean_plddt": round(float(coords["mean_plddt"]), 4),
        "coordinate_truncated": bool(coords["coordinate_truncated"]),
    }
    leakage_signature = {
        "schema": "toricblm.structure_leakage_signature.v1",
        "split": split,
        "cluster_id": graph["split_cluster"]["cluster_id"],
        "basis": graph["split_cluster"]["leakage_resistant_basis"],
        "accession_hash": hashlib.blake2b(accession.encode("utf-8"), digest_size=16).hexdigest(),
    }
    out = dict(row)
    out.update(
        {
            "source_file": out.get("source_file") or "",
            "source_row_index": int(out.get("source_row_index") or 0),
            "entry_id": out.get("entry_id") or accession,
            "annotation_text": out.get("annotation_text") or function_text,
            "graph_json": json.dumps(graph, ensure_ascii=True, sort_keys=True),
            "forest_json": json.dumps(forest, ensure_ascii=True, sort_keys=True),
            "thought_forest_json": json.dumps(thought_forest, ensure_ascii=True, sort_keys=True),
            "convextok_dag_json": json.dumps(convextok_view, ensure_ascii=True, sort_keys=True),
            "training_views_json": json.dumps(training_views, ensure_ascii=True, sort_keys=True),
            "enrichment_status_json": json.dumps(enrichment_status, ensure_ascii=True, sort_keys=True),
            "leakage_signature_json": json.dumps(leakage_signature, ensure_ascii=True, sort_keys=True),
            "metadata_json": json.dumps(graph["metadata"], ensure_ascii=True, sort_keys=True),
            "content_hash": hashlib.sha256(
                stable_json({"graph": graph, "forest": forest, "thought_forest": thought_forest, "convextok": convextok_view}).encode("utf-8")
            ).hexdigest(),
            "group_hash": graph["split_cluster"]["group_hash"],
            "split_cluster": json.dumps(graph["split_cluster"], ensure_ascii=True, sort_keys=True),
            "quality_flags_json": json.dumps(graph["metadata"]["quality_flags"], ensure_ascii=True, sort_keys=True),
        }
    )
    return out, True


def upgrade_file(path: Path, *, convextok_tokens: dict[int, list[dict[str, Any]]], convextok_max_bytes: int, dry_run: bool) -> dict[str, Any]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    upgraded = 0
    out_rows = []
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
    parser.add_argument("--input", action="append", required=True, help="Parquet glob to upgrade; repeatable.")
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
