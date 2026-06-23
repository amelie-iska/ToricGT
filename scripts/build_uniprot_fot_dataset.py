#!/usr/bin/env python3
"""Build graph/FoT training records from local bio-scale Parquet shards.

The converter is deliberately deterministic. It preserves upstream row text and
adds graph structure, hashes, split metadata, TokenGT/ToricGT sidecars, and
Forest-of-Thought training fields without inventing biological claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


RAW_ROOT = Path("/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale")
OUTPUT_DIR = Path("/home/iska/Documents/amelie/bio/ToricGT/data/uniprot_fot")

EC_RE = re.compile(r"\bEC\s+([0-9-]+(?:\.[0-9-]+){1,3})\b")
GO_RE = re.compile(r"\bGO:[0-9]{7}\b")

RECORD_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("dataset", pa.string()),
        ("source_file", pa.string()),
        ("source_row_index", pa.int64()),
        ("entry_id", pa.string()),
        ("task_family", pa.string()),
        ("sequence", pa.large_string()),
        ("annotation_text", pa.large_string()),
        ("text", pa.large_string()),
        ("graph_json", pa.large_string()),
        ("forest_json", pa.large_string()),
        ("metadata_json", pa.large_string()),
        ("content_hash", pa.string()),
        ("group_hash", pa.string()),
        ("split", pa.string()),
        ("split_cluster", pa.string()),
        ("quality_flags_json", pa.string()),
    ]
)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(value: Any, length: int = 16) -> str:
    return sha256_text(stable_json(value))[:length]


def compact_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "...<truncated>"
    if isinstance(value, list):
        out = []
        used = 0
        for item in value:
            c = compact_value(item, max_chars)
            out.append(c)
            used += len(repr(c))
            if used >= max_chars:
                out.append("<truncated-list>")
                break
        return out
    if isinstance(value, dict):
        out = {}
        used = 0
        for key, item in value.items():
            c = compact_value(item, max_chars)
            out[str(key)] = c
            used += len(str(key)) + len(repr(c))
            if used >= max_chars:
                out["<truncated>"] = True
                break
        return out
    return value


def deterministic_vector(key: str, dims: int = 8) -> list[float]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    coords = []
    for i in range(dims):
        chunk = digest[2 * i : 2 * i + 2]
        val = int.from_bytes(chunk, "big") / 65535.0
        coords.append(round(2.0 * val - 1.0, 6))
    return coords


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def string_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


def first_nonempty(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
        if value is not None and not isinstance(value, (list, dict)):
            return str(value)
    return ""


def split_for_group(group_hash: str) -> str:
    bucket = int(group_hash[:8], 16) % 1000
    if bucket < 970:
        return "train"
    if bucket < 985:
        return "validation"
    return "test"


def raw_quality_flags(dataset: str, row: dict[str, Any]) -> dict[str, Any]:
    sequence = string_or_empty(row.get("sequence"))
    annotation_bits = 0
    for key, value in row.items():
        if key == "sequence":
            continue
        if value not in (None, "", [], {}):
            annotation_bits += 1
    return {
        "deterministic_graphification": True,
        "copied_reasoning_trace": False,
        "contains_patient_specific_advice": False,
        "requires_license_review_before_redistribution": True,
        "sequence_present": bool(sequence),
        "sequence_length": len(sequence),
        "annotation_field_count": annotation_bits,
        "dataset": dataset,
    }


def afdb_links(accessions: list[str]) -> list[dict[str, str]]:
    links = []
    for acc in accessions[:8]:
        if re.fullmatch(r"[A-NR-Z0-9][A-Z0-9]{5,9}", acc or ""):
            links.append(
                {
                    "accession": acc,
                    "afdb_url": f"https://alphafold.ebi.ac.uk/entry/{acc}",
                    "kind": "predicted_structure_lookup",
                }
            )
    return links


def make_node(
    node_id: str,
    node_type: str,
    content: Any,
    symbolic_payload: dict[str, Any] | None = None,
    max_chars: int = 4096,
) -> dict[str, Any]:
    payload = symbolic_payload or {}
    compact = compact_value(content, max_chars=max_chars)
    return {
        "id": node_id,
        "type": node_type,
        "content": compact,
        "latent_coordinates": deterministic_vector(node_id + ":" + stable_json(compact), dims=8),
        "trajectory_notes": {
            "space": "deterministic_source_field_embedding_proxy",
            "continuous_or_hybrid": True,
            "note": "Coordinates are stable hash features for data curation, not learned embeddings.",
        },
        "symbolic_payload": payload,
    }


def make_edge(source: str, target: str, edge_type: str, rationale: str) -> dict[str, Any]:
    edge_id = f"e_{short_hash([source, target, edge_type], 12)}"
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "type": edge_type,
        "directed": True,
        "causal_semantics": rationale,
        "token_gt_payload": {
            "edge_token": f"{source}->{target}",
            "direction_sensitive": True,
            "edge_hash": edge_id,
        },
    }


def infer_dataset_family(dataset: str) -> str:
    if dataset.startswith("uniprot"):
        return "uniprot_annotation_graphification"
    if dataset.startswith("rfam") or dataset.startswith("rnacentral"):
        return "rna_annotation_graphification"
    if dataset.startswith("dna"):
        return "coding_region_graphification"
    if dataset.startswith("pubchem"):
        return "medicinal_chemistry_selfies_graphification"
    return "bio_scale_graphification"


def graphify_row(
    dataset: str,
    row: dict[str, Any],
    source_file: str,
    source_row_index: int,
    max_sequence_chars: int,
    max_field_chars: int,
) -> dict[str, Any]:
    task_family = infer_dataset_family(dataset)
    entry_id = first_nonempty(row, "entry", "id", "upi", "accession", "seed_id", "SELFIES")
    if not entry_id:
        entry_id = f"row_{source_row_index}"

    sequence = string_or_empty(row.get("sequence"))
    sequence_for_hash = sequence if len(sequence) <= 2048 else sequence[:2048] + f":len={len(sequence)}"
    group_basis = {
        "dataset": dataset,
        "entry_id": entry_id,
        "sequence_crc64": row.get("sequence_crc64"),
        "sequence_xxh128": row.get("sequence_xxh128"),
        "sequence_prefix": sequence_for_hash,
    }
    group_hash = sha256_text(stable_json(group_basis))
    split = split_for_group(group_hash)
    record_id = f"uniprot_fot_{dataset}_{short_hash([dataset, entry_id, source_file, source_row_index], 20)}"

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    root = make_node(
        "n_problem",
        "source_record",
        {
            "dataset": dataset,
            "entry_id": entry_id,
            "source_file": source_file,
            "source_row_index": source_row_index,
        },
        {"role": "root", "task_family": task_family},
        max_chars=max_field_chars,
    )
    nodes.append(root)

    if sequence:
        nodes.append(
            make_node(
                "n_sequence",
                "biomolecular_sequence",
                sequence,
                {
                    "alphabet_hint": "protein_or_nucleotide",
                    "length": len(sequence),
                    "truncated_for_text_field": len(sequence) > max_sequence_chars,
                },
                max_chars=max_sequence_chars,
            )
        )
        edges.append(make_edge("n_problem", "n_sequence", "has_sequence", "The source record supplies this molecular sequence."))

    annotation_fields = []
    for key in sorted(row):
        if key == "sequence":
            continue
        value = row[key]
        if value in (None, "", [], {}):
            continue
        node_id = "n_" + re.sub(r"[^a-zA-Z0-9]+", "_", key).strip("_").lower()[:48]
        node_type = "source_annotation_field"
        if key in {"function", "protein_name", "rep_protein_name", "name"}:
            node_type = "functional_annotation"
        elif key.startswith("go_"):
            node_type = "go_annotation_set"
        elif key in {"exons", "introns", "proteins"}:
            node_type = "genomic_feature_list"
        elif key == "SELFIES":
            node_type = "molecular_string"
        nodes.append(make_node(node_id, node_type, value, {"source_field": key}, max_chars=max_field_chars))
        edges.append(make_edge("n_problem", node_id, "has_annotation", f"The source row includes field {key}."))
        if sequence:
            edges.append(
                make_edge(
                    "n_sequence",
                    node_id,
                    "sequence_context_for_annotation",
                    f"The molecular sequence is the primary context for interpreting source field {key}.",
                )
            )
        annotation_fields.append(key)

    accessions = []
    for key in ("entry", "seed_id", "rep_member_id", "rep_accessions"):
        value = row.get(key)
        if isinstance(value, str):
            accessions.append(value)
        elif isinstance(value, list):
            accessions.extend(str(v) for v in value if v)
    structure_links = afdb_links(accessions)
    if structure_links:
        nodes.append(
            make_node(
                "n_structure_links",
                "structure_lookup",
                structure_links,
                {"source": "AlphaFold DB URL pattern", "requires_external_fetch": True},
                max_chars=max_field_chars,
            )
        )
        edges.append(make_edge("n_problem", "n_structure_links", "has_structure_lookup", "Accessions can be used to query linked predicted structures."))
        if sequence:
            edges.append(make_edge("n_sequence", "n_structure_links", "sequence_to_structure_lookup", "Sequence accession anchors structure lookup."))

    free_text = " ".join(
        str(row.get(k, ""))
        for k in ("protein_name", "rep_protein_name", "name", "function", "description", "family", "clan", "type")
        if row.get(k)
    )
    ec_numbers = sorted(set(EC_RE.findall(free_text)))
    go_terms = sorted(
        set(
            term
            for key in ("go_mf", "go_bp", "go_cc")
            for term in as_list(row.get(key))
            if isinstance(term, str)
        )
        | set(GO_RE.findall(free_text))
    )
    if ec_numbers:
        nodes.append(make_node("n_ec_numbers", "enzyme_commission_annotations", ec_numbers, {"count": len(ec_numbers)}, max_chars=max_field_chars))
        edges.append(make_edge("n_problem", "n_ec_numbers", "has_ec_annotation", "Enzyme Commission labels were present in source annotation text."))
    if go_terms:
        nodes.append(make_node("n_go_terms", "go_annotation_union", go_terms, {"count": len(go_terms)}, max_chars=max_field_chars))
        edges.append(make_edge("n_problem", "n_go_terms", "has_go_annotation", "GO labels were present in the source row."))

    info_weights = []
    for node in nodes:
        content_len = len(stable_json(node.get("content", "")))
        info_weights.append((node["id"], math.log1p(content_len)))
    info_weights.sort(key=lambda item: item[1], reverse=True)
    active_support_nodes = [node_id for node_id, _ in info_weights[: min(6, len(info_weights))]]
    margin = 0.0
    if len(info_weights) >= 2:
        margin = round(info_weights[0][1] - info_weights[1][1], 6)

    graph_json = {
        "id": record_id,
        "source": {
            "origin": "raw_hf_bio_scale",
            "dataset": dataset,
            "source_file": source_file,
            "source_row_index": source_row_index,
            "content_creation_mode": "deterministic_graphification_of_source_data",
            "contains_imported_reasoning_trace": False,
        },
        "task_family": task_family,
        "nodes": nodes,
        "edges": edges,
        "targets": {
            "answer": {
                "kind": "source_grounded_annotation_bundle",
                "entry_id": entry_id,
                "available_annotation_fields": annotation_fields,
                "sequence_length": len(sequence),
                "ec_numbers": ec_numbers,
                "go_terms": go_terms,
                "structure_links": structure_links,
            },
            "verifier_metadata": {
                "json_parse_required": True,
                "directed_edge_integrity_required": True,
                "source_fields_preserved": sorted(row.keys()),
                "no_patient_specific_advice": True,
            },
            "gflownet_reward_metadata": {
                "reward_type": "posterior_like_source_density_proxy",
                "reward_components": {
                    "annotation_density": min(1.0, len(annotation_fields) / 12.0),
                    "has_sequence": 1.0 if sequence else 0.0,
                    "has_go": 1.0 if go_terms else 0.0,
                    "has_ec": 1.0 if ec_numbers else 0.0,
                    "has_structure_lookup": 1.0 if structure_links else 0.0,
                    "directed_graph_connected_from_root": 1.0 if edges else 0.0,
                },
                "terminal_reward_proxy": None,
            },
            "active_support_nodes": active_support_nodes,
        },
        "metadata": {
            "domain": "biological_sequence_annotation",
            "topic_flags": {
                "biochemistry": dataset.startswith("uniprot"),
                "biomedicine": dataset.startswith("uniprot") or dataset.startswith("dna"),
                "rna_medicine_relevant": dataset.startswith("rfam") or dataset.startswith("rnacentral"),
                "medicinal_chemistry": dataset.startswith("pubchem"),
                "gene_regulatory_context_possible": dataset.startswith("dna") or dataset.startswith("uniprot"),
            },
            "merge_status": "graphified_source_fields_merge_at_record_root",
            "multiple_viable_solution_status": "not_a_reasoning_solution_record",
            "continuous_embedding_fields": {
                "node_latent_coordinates": "stable_hash_proxy",
                "trajectory_space": "continuous_or_hybrid_proxy",
                "graphcg_axis_hints": ["sequence", "function", "structure", "taxonomy", "chemistry"],
            },
            "tokengt_tropical_toric_fields": {
                "node_token_order": [node["id"] for node in nodes],
                "edge_token_order": [edge["id"] for edge in edges],
                "tropical_active_support_nodes": active_support_nodes,
                "tropical_margin_proxy": margin,
                "toric_phase_basis": "deterministic_hash_orthonormal_proxy",
                "convextok_alignment": "source text and sequence can be byte-packed with graph_json sidecar",
            },
            "quality_flags": raw_quality_flags(dataset, row),
        },
        "split_cluster": {
            "cluster_id": "cluster_" + group_hash[:20],
            "group_hash": group_hash,
            "split": split,
            "leakage_resistant_basis": "dataset + entry/accession + sequence hash or prefix",
        },
    }
    reward_components = graph_json["targets"]["gflownet_reward_metadata"]["reward_components"]
    graph_json["targets"]["gflownet_reward_metadata"]["terminal_reward_proxy"] = round(
        sum(float(v) for v in reward_components.values()) / max(1, len(reward_components)),
        6,
    )

    forest_json = {
        "forest_type": "embedding_space_source_field_forest",
        "tree_count": 4,
        "sparse_activation_policy": "activate trees whose source fields are present",
        "trees": [
            {
                "tree_id": "sequence_tree",
                "root": "n_problem",
                "active": bool(sequence),
                "candidate_nodes": [node["id"] for node in nodes if node["id"] in {"n_sequence", "n_proteins", "n_exons", "n_introns"}],
                "purpose": "sequence and feature context",
            },
            {
                "tree_id": "annotation_tree",
                "root": "n_problem",
                "active": bool(annotation_fields),
                "candidate_nodes": [node["id"] for node in nodes if node["type"] in {"functional_annotation", "go_annotation_set", "source_annotation_field"}],
                "purpose": "function, GO, EC, taxonomy, and textual labels",
            },
            {
                "tree_id": "structure_tree",
                "root": "n_problem",
                "active": bool(structure_links),
                "candidate_nodes": ["n_structure_links"] if structure_links else [],
                "purpose": "PDB/AFDB style structural association when accession permits lookup",
            },
            {
                "tree_id": "design_condition_tree",
                "root": "n_problem",
                "active": bool(sequence and annotation_fields),
                "candidate_nodes": active_support_nodes,
                "purpose": "future conditional design constraints grounded in source fields",
            },
        ],
        "dynamic_self_correction": {
            "enabled_for_authored_reasoning_records": True,
            "status_for_this_record": "not_applicable_deterministic_source_graph",
        },
        "consensus": {
            "leaf_selection_rule": "highest source-field density with sequence support",
            "consensus_nodes": active_support_nodes,
        },
        "trajectory_balance_metadata": {
            "forward_policy_basis": "source-field expansion",
            "backward_policy_basis": "rooted field contraction",
            "reward_key": "targets.gflownet_reward_metadata.terminal_reward_proxy",
        },
    }

    annotation_text = free_text
    text = "\n".join(
        part
        for part in [
            f"Dataset: {dataset}",
            f"Entry: {entry_id}",
            f"Task family: {task_family}",
            f"Annotation fields: {', '.join(annotation_fields)}",
            f"Sequence length: {len(sequence)}" if sequence else "Sequence: <absent>",
            annotation_text[:max_field_chars] if annotation_text else "",
        ]
        if part
    )
    content_hash = sha256_text(stable_json({"graph": graph_json, "forest": forest_json}))
    return {
        "record_id": record_id,
        "dataset": dataset,
        "source_file": source_file,
        "source_row_index": source_row_index,
        "entry_id": entry_id,
        "task_family": task_family,
        "sequence": sequence if len(sequence) <= max_sequence_chars else sequence[:max_sequence_chars] + "...<truncated>",
        "annotation_text": annotation_text[:max_field_chars],
        "text": text,
        "graph_json": json.dumps(graph_json, ensure_ascii=True),
        "forest_json": json.dumps(forest_json, ensure_ascii=True),
        "metadata_json": json.dumps(graph_json["metadata"], ensure_ascii=True),
        "content_hash": content_hash,
        "group_hash": group_hash,
        "split": split,
        "split_cluster": json.dumps(graph_json["split_cluster"], ensure_ascii=True),
        "quality_flags_json": json.dumps(graph_json["metadata"]["quality_flags"], ensure_ascii=True),
    }


def dataset_dirs(raw_root: Path) -> list[Path]:
    return sorted(path for path in raw_root.iterdir() if path.is_dir())


def parquet_files(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.glob("**/*.parquet"))


def scan_manifest(raw_root: Path) -> dict[str, Any]:
    manifest = {
        "raw_root": str(raw_root),
        "datasets": [],
        "totals": {"datasets": 0, "parquet_files": 0, "readable_parquet_files": 0, "bytes": 0, "rows": 0},
    }
    for dpath in dataset_dirs(raw_root):
        files = parquet_files(dpath)
        rows = 0
        schema = []
        unreadable = []
        readable_files = 0
        for idx, path in enumerate(files):
            try:
                pf = pq.ParquetFile(path)
            except Exception as exc:  # noqa: BLE001 - curation manifest records source defects.
                unreadable.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            readable_files += 1
            rows += pf.metadata.num_rows
            if not schema:
                schema = [{"name": field.name, "type": str(field.type)} for field in pf.schema_arrow]
        size = sum(path.stat().st_size for path in files)
        manifest["datasets"].append(
            {
                "name": dpath.name,
                "path": str(dpath),
                "parquet_files": len(files),
                "readable_parquet_files": readable_files,
                "unreadable_files": unreadable,
                "bytes": size,
                "rows": rows,
                "schema": schema,
            }
        )
        manifest["totals"]["datasets"] += 1
        manifest["totals"]["parquet_files"] += len(files)
        manifest["totals"]["readable_parquet_files"] += readable_files
        manifest["totals"]["bytes"] += size
        manifest["totals"]["rows"] += rows
    return manifest


def ensure_raw_links(raw_root: Path, output_dir: Path, mode: str) -> list[dict[str, str]]:
    raw_dir = output_dir / "raw_sources"
    raw_dir.mkdir(parents=True, exist_ok=True)
    links = []
    for dpath in dataset_dirs(raw_root):
        target = raw_dir / dpath.name
        if target.exists() or target.is_symlink():
            links.append({"dataset": dpath.name, "path": str(target), "mode": "existing"})
            continue
        if mode == "none":
            continue
        if mode == "symlink":
            target.symlink_to(dpath, target_is_directory=True)
        elif mode == "copy":
            shutil.copytree(dpath, target)
        elif mode == "move":
            shutil.move(str(dpath), str(target))
        else:
            raise ValueError(f"unknown raw link mode {mode}")
        links.append({"dataset": dpath.name, "path": str(target), "mode": mode, "source": str(dpath)})
    return links


def iter_sample_rows(raw_root: Path, sample_per_dataset: int) -> tuple[str, Path, int, dict[str, Any]]:
    for dpath in dataset_dirs(raw_root):
        produced = 0
        for path in parquet_files(dpath):
            if produced >= sample_per_dataset:
                break
            try:
                pf = pq.ParquetFile(path)
            except Exception:
                continue
            for batch in pf.iter_batches(batch_size=min(512, sample_per_dataset - produced)):
                for row in pa.Table.from_batches([batch]).to_pylist():
                    yield dpath.name, path, produced, row
                    produced += 1
                    if produced >= sample_per_dataset:
                        break
                if produced >= sample_per_dataset:
                    break


def build_records(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (output_dir / "derived").mkdir(parents=True, exist_ok=True)

    manifest = scan_manifest(raw_root)
    raw_links = ensure_raw_links(raw_root, output_dir, args.raw_link_mode)
    manifest["raw_links"] = raw_links

    records = []
    counts = Counter()
    for dataset, source_file, source_row_index, row in iter_sample_rows(raw_root, args.sample_per_dataset):
        rec = graphify_row(
            dataset=dataset,
            row=row,
            source_file=str(source_file),
            source_row_index=source_row_index,
            max_sequence_chars=args.max_sequence_chars,
            max_field_chars=args.max_field_chars,
        )
        records.append(rec)
        counts[dataset] += 1

    jsonl_path = output_dir / "derived" / "uniprot_fot_graphified_sample.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=True) + "\n")

    parquet_path = output_dir / "derived" / "uniprot_fot_graphified_sample.parquet"
    table = pa.Table.from_pylist(records, schema=RECORD_SCHEMA)
    pq.write_table(table, parquet_path, compression="zstd", use_dictionary=True)

    build_manifest = {
        "format_version": "uniprot_fot_graphified_v0",
        "raw_manifest": manifest,
        "derived": {
            "jsonl": str(jsonl_path),
            "parquet": str(parquet_path),
            "records": len(records),
            "sample_per_dataset": args.sample_per_dataset,
            "counts_by_dataset": dict(counts),
            "schema": [{"name": field.name, "type": str(field.type)} for field in RECORD_SCHEMA],
        },
        "notes": [
            "This is a deterministic graphification sample, not the final authored reasoning corpus.",
            "Raw sources are symlinked by default to avoid duplicating 26GB on a nearly full filesystem.",
            "Use --raw-link-mode move only after confirming no other workflow depends on the old raw path.",
        ],
    }
    manifest_path = output_dir / "manifests" / "uniprot_fot_build_manifest.json"
    manifest_path.write_text(json.dumps(build_manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return build_manifest


def validate_records(path: Path) -> dict[str, Any]:
    errors = []
    counts = Counter()
    split_counts = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            row = json.loads(line)
            graph = json.loads(row["graph_json"])
            node_ids = {node["id"] for node in graph.get("nodes", [])}
            if not node_ids:
                errors.append({"line": line_no, "error": "missing_nodes"})
            for edge in graph.get("edges", []):
                if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
                    errors.append({"line": line_no, "error": "edge_endpoint_missing", "edge": edge})
                if edge.get("directed") is not True:
                    errors.append({"line": line_no, "error": "edge_not_directed", "edge": edge.get("id")})
            required = ["id", "source", "task_family", "nodes", "edges", "targets", "metadata", "split_cluster"]
            for key in required:
                if key not in graph:
                    errors.append({"line": line_no, "error": f"missing_graph_key:{key}"})
            if graph.get("source", {}).get("contains_imported_reasoning_trace") is not False:
                errors.append({"line": line_no, "error": "imported_reasoning_trace_flag_not_false"})
            metadata = graph.get("metadata", {})
            if "tokengt_tropical_toric_fields" not in metadata:
                errors.append({"line": line_no, "error": "missing_toric_fields"})
            if "gflownet_reward_metadata" not in graph.get("targets", {}):
                errors.append({"line": line_no, "error": "missing_gflownet_reward_metadata"})
            counts[row["dataset"]] += 1
            split_counts[row["split"]] += 1
    return {
        "path": str(path),
        "records": sum(counts.values()),
        "counts_by_dataset": dict(counts),
        "split_counts": dict(split_counts),
        "errors": errors,
        "ok": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--raw-root", default=str(RAW_ROOT))
    build.add_argument("--output-dir", default=str(OUTPUT_DIR))
    build.add_argument("--sample-per-dataset", type=int, default=8)
    build.add_argument("--max-sequence-chars", type=int, default=4096)
    build.add_argument("--max-field-chars", type=int, default=4096)
    build.add_argument("--raw-link-mode", choices=["none", "symlink", "copy", "move"], default="symlink")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--jsonl", default=str(OUTPUT_DIR / "derived" / "uniprot_fot_graphified_sample.jsonl"))

    args = parser.parse_args()
    if args.command == "build":
        result = build_records(args)
    elif args.command == "validate":
        result = validate_records(Path(args.jsonl))
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
