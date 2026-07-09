#!/usr/bin/env python3
"""Build graph/FoT training records from local bio-scale Parquet shards.

The converter is deliberately deterministic. It preserves upstream row text and
adds graph structure, hashes, split metadata, TokenGT/ToricGT sidecars, and
Forest-of-Thought training fields without inventing biological claims.
"""

from __future__ import annotations

import argparse
import base64
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
DEFAULT_CONVEXTOK = Path(
    "/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/"
    "parameter_golf_convextok8192_biomed_det_full/tokenizers/"
    "fineweb_convextok_8192_biomed_det.convextok.json"
)

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
        ("thought_forest_json", pa.large_string()),
        ("convextok_dag_json", pa.large_string()),
        ("training_views_json", pa.large_string()),
        ("enrichment_status_json", pa.large_string()),
        ("leakage_signature_json", pa.large_string()),
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


def sequence_kind(sequence: str, dataset: str) -> str:
    if not sequence:
        return "absent"
    alphabet = set(sequence.upper())
    if dataset.startswith("pubchem"):
        return "selfies"
    if alphabet <= set("ACGTNURYKMSWBDHV.-"):
        return "nucleotide_or_rna"
    if alphabet <= set("ACDEFGHIKLMNPQRSTVWYXBZUOJ*-"):
        return "protein"
    return "mixed_or_unknown"


def sequence_analytics(sequence: str, dataset: str) -> dict[str, Any]:
    kind = sequence_kind(sequence, dataset)
    counts = Counter(sequence.upper())
    length = len(sequence)
    alphabet = sorted(counts)
    analytics: dict[str, Any] = {
        "kind": kind,
        "length": length,
        "alphabet": alphabet,
        "composition": {key: counts[key] for key in alphabet[:64]},
        "composition_total": sum(counts.values()),
        "sha256_prefix": sha256_text(sequence[:4096] + f":len={length}")[:32],
    }
    if length and kind == "nucleotide_or_rna":
        gc = counts.get("G", 0) + counts.get("C", 0)
        analytics["gc_fraction"] = round(gc / length, 6)
        analytics["u_fraction"] = round(counts.get("U", 0) / length, 6)
    if length and kind == "protein":
        hydrophobic = sum(counts.get(aa, 0) for aa in "AILMFWVY")
        charged = sum(counts.get(aa, 0) for aa in "DEKRH")
        cysteine = counts.get("C", 0)
        analytics["hydrophobic_fraction"] = round(hydrophobic / length, 6)
        analytics["charged_fraction"] = round(charged / length, 6)
        analytics["cysteine_fraction"] = round(cysteine / length, 6)
    return analytics


def sequence_chunks(sequence: str, chunk_size: int = 256, max_chunks: int = 64) -> list[dict[str, Any]]:
    chunks = []
    if not sequence:
        return chunks
    for idx, start in enumerate(range(0, len(sequence), chunk_size)):
        if idx >= max_chunks:
            chunks.append({"chunk_index": idx, "truncated_remaining": True, "remaining_length": max(0, len(sequence) - start)})
            break
        end = min(len(sequence), start + chunk_size)
        part = sequence[start:end]
        chunks.append(
            {
                "chunk_index": idx,
                "start": start,
                "end": end,
                "length": end - start,
                "prefix": part[:80],
                "suffix": part[-32:],
                "content_hash": sha256_text(part)[:24],
            }
        )
    return chunks


def selfies_tokens(value: str, max_tokens: int = 96) -> list[str]:
    if not value:
        return []
    tokens = re.findall(r"\[[^\]]+\]", value)
    return tokens[:max_tokens]


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


def pdb_lookup_hooks(accessions: list[str]) -> list[dict[str, str]]:
    hooks = []
    for acc in accessions[:8]:
        if re.fullmatch(r"[0-9][A-Za-z0-9]{3}", acc or ""):
            hooks.append({"accession": acc.upper(), "kind": "experimental_structure_lookup", "pdb_url": f"https://www.rcsb.org/structure/{acc.upper()}"})
    return hooks


def enrichment_status(dataset: str, row: dict[str, Any], ec_numbers: list[str], go_terms: list[str], structure_links: list[dict[str, str]]) -> dict[str, Any]:
    text_blob = " ".join(str(v) for v in row.values() if isinstance(v, str))
    has_sites = bool(re.search(r"\b(active site|binding site|site|motif|domain)\b", text_blob, flags=re.IGNORECASE))
    has_kinetics = bool(re.search(r"\b(Km|Kcat|kcat|Vmax|turnover|kinetic)\b", text_blob))
    has_structure_hook = bool(structure_links)
    coordinate_columns_present = any(
        key in row and row.get(key) not in (None, "", [], {})
        for key in (
            "structure_coordinates",
            "ca_coordinates",
            "backbone_coordinates",
            "atom_coordinates",
            "coordinates",
        )
    )
    return {
        "dataset": dataset,
        "present": {
            "sequence": bool(row.get("sequence") or row.get("SELFIES")),
            "uniprot_accession_like_id": bool(first_nonempty(row, "entry", "seed_id", "rep_member_id", "accession")),
            "function_text": bool(row.get("function") or row.get("protein_name") or row.get("rep_protein_name")),
            "go_terms": bool(go_terms),
            "ec_numbers": bool(ec_numbers),
            "taxonomy": bool(row.get("common_taxon") or row.get("organism") or row.get("rep_organism")),
            "sites_or_domains_in_text": has_sites,
            "kinetic_parameters_in_text": has_kinetics,
            "genomic_features": bool(row.get("exons") or row.get("introns") or row.get("proteins")),
            "structure_lookup_hooks": has_structure_hook,
            "structure_available": coordinate_columns_present,
            "coordinate_training_available": coordinate_columns_present,
            "selfies_molecule": bool(row.get("SELFIES")),
        },
        "missing_or_external": {
            "full_uniprotkb_xml_json": "not_present_in_local_raw_hf_bio_scale",
            "pdb_cross_reference_table": "not_present_unless_accession_like_pdb_id_detected",
            "afdb_coordinates": "not_fetched_in_graph_stream; URL hook only when accession permits",
            "alphafold_confidence": "not_present",
            "binding_sites": "only text-detected if present upstream",
            "catalytic_activity": "only text-detected if present upstream",
            "kinetic_constants": "only regex-detected if present upstream",
            "variants_pathways_interactions": "not_present_unless upstream field contains text",
            "coordinate_structure_training_target": (
                "present_in_this_row"
                if coordinate_columns_present
                else "absent_from_this_graphified_row; keep graph/text/FoT training active and route coordinate losses only to coordinate-bearing shards"
            ),
            "structure_missing_policy": (
                "train_all_available_noncoordinate_modalities; do_not_drop_entry; do_not_impute_coordinates"
                if not coordinate_columns_present
                else "coordinate-bearing row may be used by structure-flow training"
            ),
        },
        "structure_policy": {
            "graph_fot_training_requires_coordinates": False,
            "coordinate_loss_requires_coordinates": True,
            "missing_structure_action": "train_without_structure_using_all_available_fields",
            "coordinate_imputation_allowed": False,
            "structure_lookup_count": len(structure_links),
        },
    }


def load_convextok_tokens(path: Path | None, max_token_bytes: int = 96) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    tokens = []
    for item in payload.get("priced_tokens", []):
        try:
            raw = base64.b64decode(item["bytes_b64"])
        except Exception:
            continue
        if not raw or len(raw) > max_token_bytes:
            continue
        tokens.append(
            {
                "id": int(item["id"]),
                "bytes": raw,
                "byte_length": int(item.get("byte_length", len(raw))),
                "lp_score": float(item.get("lp_score", 0.0)),
                "rank_score": float(item.get("rank_score", 0.0)),
                "text": raw.decode("utf-8", errors="replace"),
            }
        )
    tokens.sort(key=lambda item: (-item["byte_length"], item["id"]))
    return tokens


def index_convextok_tokens(tokens: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Index ConvexTok priced tokens by first byte for bounded DAG construction."""
    indexed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for tok in tokens:
        raw = tok.get("bytes", b"")
        if raw:
            indexed[int(raw[0])].append(tok)
    for bucket in indexed.values():
        bucket.sort(key=lambda item: (-item["byte_length"], item["id"]))
    return indexed


def convextok_dag(
    text: str,
    token_index: dict[int, list[dict[str, Any]]],
    max_bytes: int = 768,
    max_priced_edges: int = 192,
) -> dict[str, Any]:
    raw = text.encode("utf-8", errors="replace")[:max_bytes]
    n = len(raw)
    priced_edges: list[dict[str, Any]] = []
    if n and token_index:
        for start in range(n):
            local_matches = 0
            for tok in token_index.get(int(raw[start]), []):
                b = tok["bytes"]
                end = start + len(b)
                if end > n:
                    continue
                if raw[start:end] == b:
                    priced_edges.append(
                        {
                            "source": start,
                            "target": end,
                            "token_id": tok["id"],
                            "kind": "priced_vocabulary_token",
                            "byte_length": len(b),
                            "lp_score": round(tok["lp_score"], 6),
                            "rank_score": round(tok["rank_score"], 6),
                            "text": tok["text"][:96],
                        }
                    )
                    local_matches += 1
                    if local_matches >= 4 or len(priced_edges) >= max_priced_edges:
                        break
            if len(priced_edges) >= max_priced_edges:
                break
    free_edges = [
        {
            "source": i,
            "target": i + 1,
            "token_id": int(raw[i]) + 4,
            "kind": "free_byte_fallback",
            "byte_length": 1,
        }
        for i in range(n)
    ]
    all_edges = free_edges + priced_edges
    outgoing: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in all_edges:
        outgoing[int(edge["source"])].append(edge)
    # Exact shortest-path dynamic program on the bounded DAG prefix.
    best = [math.inf] * (n + 1)
    back: list[dict[str, Any] | None] = [None] * (n + 1)
    best[0] = 0.0
    for i in range(n):
        if not math.isfinite(best[i]):
            continue
        for edge in outgoing.get(i, []):
            cost = 1.0
            if edge["kind"] == "priced_vocabulary_token":
                cost = max(0.05, 1.0 - 0.15 * float(edge.get("lp_score", 0.0)))
            cand = best[i] + cost
            j = int(edge["target"])
            if cand < best[j]:
                best[j] = cand
                back[j] = edge
    path = []
    cursor = n
    while cursor > 0 and back[cursor] is not None:
        edge = back[cursor]
        path.append({k: v for k, v in edge.items() if k != "text" or edge["kind"] == "priced_vocabulary_token"})
        cursor = int(edge["source"])
    path.reverse()
    return {
        "schema": "toricgt.convextok_tokenization_dag.v1",
        "tokenizer": "ConvexTok-8192-biomed-det" if token_index else "unavailable",
        "byte_prefix_length": n,
        "truncated": len(text.encode("utf-8", errors="replace")) > n,
        "vertices": [{"id": i, "byte_offset": i} for i in range(n + 1)],
        "free_edge_count": len(free_edges),
        "priced_edge_count": len(priced_edges),
        "priced_edges": priced_edges,
        "rounded_shortest_path": path,
        "dp_cost": None if not math.isfinite(best[n]) else round(float(best[n]), 6),
        "tropical_attention_relevance": {
            "dynamic_programming_semiring": "min-plus shortest path over byte-boundary DAG",
            "active_path_edge_count": len(path),
            "priced_edge_support_count": sum(1 for e in path if e.get("kind") == "priced_vocabulary_token"),
        },
    }


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
    convextok_tokens: dict[int, list[dict[str, Any]]] | None = None,
    convextok_max_bytes: int = 768,
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
        nodes.append(
            make_node(
                "n_sequence_analytics",
                "sequence_analytics",
                sequence_analytics(sequence, dataset),
                {"derived_from": "n_sequence", "claim_status": "computed_from_source_sequence"},
                max_chars=max_field_chars,
            )
        )
        edges.append(make_edge("n_sequence", "n_sequence_analytics", "has_sequence_analytics", "Sequence analytics are computed deterministically from the source sequence."))
        for chunk in sequence_chunks(sequence):
            node_id = f"n_sequence_chunk_{chunk['chunk_index']:03d}"
            nodes.append(
                make_node(
                    node_id,
                    "sequence_chunk",
                    chunk,
                    {"derived_from": "n_sequence", "chunk_index": chunk["chunk_index"]},
                    max_chars=max_field_chars,
                )
            )
            edges.append(make_edge("n_sequence", node_id, "sequence_has_chunk", "Long sequence is exposed as local TokenGT-compatible chunks."))

    selfies = string_or_empty(row.get("SELFIES"))
    if selfies:
        nodes.append(
            make_node(
                "n_selfies_tokens",
                "selfies_token_sequence",
                selfies_tokens(selfies),
                {"derived_from": "SELFIES", "tokenization": "SELFIES bracket tokens"},
                max_chars=max_field_chars,
            )
        )
        edges.append(make_edge("n_problem", "n_selfies_tokens", "has_selfies_tokens", "SELFIES molecular string is exposed as bracket-token sequence."))

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
    structure_links.extend(pdb_lookup_hooks(accessions))
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

    enrich = enrichment_status(dataset, row, ec_numbers, go_terms, structure_links)
    missing_fields = [key for key, value in enrich["present"].items() if not value]
    nodes.append(
        make_node(
            "n_enrichment_status",
            "enrichment_status",
            enrich,
            {"missing_fields": missing_fields, "no_hallucination_policy": True},
            max_chars=max_field_chars,
        )
    )
    edges.append(make_edge("n_problem", "n_enrichment_status", "has_enrichment_status", "This node records which biological annotations are present locally and which require external enrichment."))

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
            "enrichment_status": enrich,
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

    def forest_node(node_id: str, label: str, tree_id: str, graph_node_id: str, budget_level: int, purpose: str, active: bool = True) -> dict[str, Any]:
        return {
            "id": node_id,
            "label": label,
            "tree_id": tree_id,
            "graph_node_id": graph_node_id,
            "budget_level": budget_level,
            "active": active,
            "purpose": purpose,
            "latent_coordinates": deterministic_vector(record_id + ":" + node_id, dims=8),
        }

    thought_nodes = [
        forest_node("ft_root", "source record", "root", "n_problem", 0, "start from source-grounded biological record"),
    ]
    thought_edges: list[dict[str, Any]] = []

    def add_fot_edge(source: str, target: str, edge_type: str, tree_id: str, rationale: str, weight: float = 1.0) -> None:
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

    tree_specs = [
        ("sequence_tree", "sequence evidence", ["n_sequence", "n_sequence_analytics"] + [node["id"] for node in nodes if node["id"].startswith("n_sequence_chunk_")][:4]),
        ("annotation_tree", "annotation evidence", [node["id"] for node in nodes if node["type"] in {"functional_annotation", "go_annotation_set", "go_annotation_union", "enzyme_commission_annotations", "source_annotation_field"}][:10]),
        ("structure_tree", "structure evidence", ["n_structure_links"] if structure_links else ["n_enrichment_status"]),
        ("design_condition_tree", "design condition evidence", active_support_nodes[:8]),
    ]
    if selfies:
        tree_specs.append(("chemistry_tree", "molecular chemistry evidence", ["n_selfies_tokens"]))
    for tree_id, label, graph_node_ids in tree_specs:
        root_id = f"ft_{tree_id}_root"
        active = any(gid in {node["id"] for node in nodes} for gid in graph_node_ids)
        thought_nodes.append(forest_node(root_id, label, tree_id, "n_problem", 1, f"activate {tree_id} when evidence is present", active=active))
        add_fot_edge("ft_root", root_id, "tree_activation", tree_id, f"Sparse activation opens {tree_id}.", 1.0 if active else 0.2)
        previous = root_id
        for pos, graph_node_id in enumerate(graph_node_ids):
            if graph_node_id not in {node["id"] for node in nodes}:
                continue
            nid = f"ft_{tree_id}_{pos:02d}"
            thought_nodes.append(forest_node(nid, graph_node_id, tree_id, graph_node_id, 2 + pos, f"source-grounded thought for {graph_node_id}", active=True))
            add_fot_edge(previous, nid, "expansion", tree_id, f"Expand {tree_id} to graph node {graph_node_id}.", 1.0)
            previous = nid
        if not active:
            correction_id = f"ft_{tree_id}_correction"
            thought_nodes.append(forest_node(correction_id, f"{tree_id} absent locally", tree_id, "n_enrichment_status", 2, "self-correction: do not hallucinate missing evidence", active=True))
            add_fot_edge(root_id, correction_id, "self_correction", tree_id, "Missing evidence is routed to enrichment-status rather than hallucinated.", 1.0)

    critical_missing = [
        key
        for key in (
            "go_terms",
            "ec_numbers",
            "sites_or_domains_in_text",
            "kinetic_parameters_in_text",
            "genomic_features",
            "structure_lookup_hooks",
        )
        if not enrich["present"].get(key, False)
    ]
    for pos, field_name in enumerate(critical_missing[:8]):
        correction_id = f"ft_missing_{field_name}"
        thought_nodes.append(
            forest_node(
                correction_id,
                f"missing {field_name}",
                "missing_evidence_corrections",
                "n_enrichment_status",
                2 + pos,
                f"self-correction: {field_name} is absent from the local source row",
                active=True,
            )
        )
        add_fot_edge(
            "ft_root",
            correction_id,
            "self_correction",
            "missing_evidence_corrections",
            f"The local row does not contain {field_name}; training target must not hallucinate it.",
            1.0,
        )

    consensus_id = "ft_consensus"
    thought_nodes.append(
        forest_node(
            consensus_id,
            "consensus source-grounded bio design context",
            "consensus",
            "n_problem",
            6,
            "merge active forest leaves into a training target",
            active=True,
        )
    )
    for node in thought_nodes:
        if node["id"].startswith("ft_") and node["id"] != consensus_id and node["budget_level"] >= 2:
            add_fot_edge(node["id"], consensus_id, "consensus", node["tree_id"], "Active leaf contributes to consensus target.", 0.75)

    thought_forest_json = {
        "schema": "toricgt.biomed_source_grounded_forest_of_thought.v1",
        "forest_type": "source_grounded_biological_fot",
        "source_record_id": record_id,
        "trees": [{"tree_id": tree_id, "label": label} for tree_id, label, _ in tree_specs],
        "nodes": thought_nodes,
        "edges": thought_edges,
        "sparse_activation_policy": "activate evidence trees whose graph nodes are present; route missing evidence to self-correction/enrichment-status nodes",
        "dynamic_self_correction": {
            "enabled": True,
            "policy": "when GO/EC/site/structure evidence is absent, train the model to say absent/external rather than invent details",
        },
        "consensus": {
            "node": consensus_id,
            "leaf_selection_rule": "source-field density, sequence support, and explicit enrichment availability",
        },
        "gflownet_training": {
            "state_space": "typed graph nodes plus forest nodes in embedding space",
            "forward_actions": ["tree_activation", "expansion", "self_correction", "consensus"],
            "backward_actions": ["contract_to_parent", "remove_unverified_leaf", "return_to_source_record"],
            "reward_key": "graph_json.targets.gflownet_reward_metadata.terminal_reward_proxy",
            "trajectory_balance_compatible": True,
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
    convextok_view = convextok_dag(text + "\n" + sequence[: min(len(sequence), 512)], convextok_tokens or [], max_bytes=convextok_max_bytes)
    training_views_json = {
        "schema": "toricgt.biomed_training_views.v1",
        "views": {
            "graph_in_graph_out": {
                "input_nodes": graph_json["metadata"]["tokengt_tropical_toric_fields"]["node_token_order"],
                "input_edges": graph_json["metadata"]["tokengt_tropical_toric_fields"]["edge_token_order"],
                "target_kind": graph_json["targets"]["answer"]["kind"],
            },
            "forest_of_thought": {
                "thought_forest_column": "thought_forest_json",
                "tree_count": len(tree_specs),
                "edge_count": len(thought_edges),
                "node_count": len(thought_nodes),
            },
            "convextok_flattening": {
                "convextok_dag_column": "convextok_dag_json",
                "rounded_path_edges": len(convextok_view.get("rounded_shortest_path", [])),
                "tokenizer": convextok_view.get("tokenizer"),
            },
            "graphcg_axes": ["sequence", "function", "structure", "taxonomy", "chemistry", "design_constraints"],
            "tropical_toric": graph_json["metadata"]["tokengt_tropical_toric_fields"],
            "structure_training_policy": enrich["structure_policy"],
        },
    }
    content_hash = sha256_text(stable_json({"graph": graph_json, "forest": forest_json, "thought_forest": thought_forest_json, "convextok": convextok_view}))
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
        "thought_forest_json": json.dumps(thought_forest_json, ensure_ascii=True),
        "convextok_dag_json": json.dumps(convextok_view, ensure_ascii=True),
        "training_views_json": json.dumps(training_views_json, ensure_ascii=True),
        "enrichment_status_json": json.dumps(enrich, ensure_ascii=True),
        "leakage_signature_json": json.dumps(
            {
                "schema": "toricblm.leakage_aware_split_signature.v0",
                "status": "basic_hash_split_from_builder",
                "upgrade_path": "run scripts/apply_toricblm_fot_leakage_splits.py for sequence/function/structure/FoT graph clustering",
            },
            ensure_ascii=True,
        ),
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


def iter_sample_rows(raw_root: Path, records_per_dataset: int | None) -> tuple[str, Path, int, dict[str, Any]]:
    for dpath in dataset_dirs(raw_root):
        produced = 0
        for path in parquet_files(dpath):
            if records_per_dataset is not None and produced >= records_per_dataset:
                break
            try:
                pf = pq.ParquetFile(path)
            except Exception:
                continue
            batch_size = 512
            if records_per_dataset is not None:
                batch_size = min(batch_size, records_per_dataset - produced)
            for batch in pf.iter_batches(batch_size=batch_size):
                for row in pa.Table.from_batches([batch]).to_pylist():
                    yield dpath.name, path, produced, row
                    produced += 1
                    if records_per_dataset is not None and produced >= records_per_dataset:
                        break
                if records_per_dataset is not None and produced >= records_per_dataset:
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
    convextok_path = Path(args.convextok_tokenizer) if args.convextok_tokenizer else None
    convextok_tokens = load_convextok_tokens(convextok_path, max_token_bytes=args.convextok_max_token_bytes)
    convextok_index = index_convextok_tokens(convextok_tokens)

    counts = Counter()
    output_prefix = args.output_prefix
    jsonl_path = output_dir / "derived" / f"{output_prefix}.jsonl"
    parquet_path = output_dir / "derived" / f"{output_prefix}.parquet"
    buffer: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    written = 0

    def flush_buffer() -> None:
        nonlocal buffer, writer
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=RECORD_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, RECORD_SCHEMA, compression="zstd", use_dictionary=True)
        writer.write_table(table)
        buffer = []

    handle = None if args.no_jsonl else jsonl_path.open("w", encoding="utf-8")
    try:
        for dataset, source_file, source_row_index, row in iter_sample_rows(raw_root, args.records_per_dataset):
            if args.max_total_records is not None and written >= args.max_total_records:
                break
            rec = graphify_row(
                dataset=dataset,
                row=row,
                source_file=str(source_file),
                source_row_index=source_row_index,
                max_sequence_chars=args.max_sequence_chars,
                max_field_chars=args.max_field_chars,
                convextok_tokens=convextok_index,
                convextok_max_bytes=args.convextok_max_bytes,
            )
            if handle is not None:
                handle.write(json.dumps(rec, ensure_ascii=True) + "\n")
            buffer.append(rec)
            counts[dataset] += 1
            written += 1
            if len(buffer) >= args.parquet_buffer_size:
                flush_buffer()
    finally:
        if handle is not None:
            handle.close()
    flush_buffer()
    if writer is not None:
        writer.close()
    elif parquet_path.exists():
        parquet_path.unlink()

    build_manifest = {
        "format_version": "uniprot_fot_graphified_v0",
        "raw_manifest": manifest,
        "derived": {
            "jsonl": str(jsonl_path),
            "jsonl_written": not args.no_jsonl,
            "parquet": str(parquet_path),
            "records": written,
            "records_per_dataset": args.records_per_dataset,
            "max_total_records": args.max_total_records,
            "counts_by_dataset": dict(counts),
            "schema": [{"name": field.name, "type": str(field.type)} for field in RECORD_SCHEMA],
        },
        "convextok": {
            "tokenizer_path": str(convextok_path) if convextok_path else None,
            "priced_tokens_loaded": len(convextok_tokens),
            "indexed_first_byte_buckets": len(convextok_index),
            "dag_max_bytes": args.convextok_max_bytes,
            "max_token_bytes": args.convextok_max_token_bytes,
        },
        "notes": [
            "This is a deterministic graphification sample, not the final authored reasoning corpus.",
            "Raw sources are symlinked by default to avoid duplicating 26GB on a nearly full filesystem.",
            "Use --raw-link-mode move only after confirming no other workflow depends on the old raw path.",
            "Full raw-derived builds should use --no-jsonl and compressed Parquet to avoid disk pressure.",
        ],
    }
    manifest_path = output_dir / "manifests" / "uniprot_fot_build_manifest.json"
    manifest_path.write_text(json.dumps(build_manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return build_manifest


def iter_validation_rows(path: Path):
    if path.suffix == ".parquet":
        pf = pq.ParquetFile(path)
        row_no = 0
        for batch in pf.iter_batches(batch_size=256):
            for row in pa.Table.from_batches([batch]).to_pylist():
                row_no += 1
                yield row_no, row
        return
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            yield line_no, json.loads(line)


def validate_records(path: Path) -> dict[str, Any]:
    errors = []
    counts = Counter()
    split_counts = Counter()
    for line_no, row in iter_validation_rows(path):
        graph = json.loads(row["graph_json"])
        thought_forest = json.loads(row["thought_forest_json"])
        convextok_view = json.loads(row["convextok_dag_json"])
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
        if thought_forest.get("schema") != "toricgt.biomed_source_grounded_forest_of_thought.v1":
            errors.append({"line": line_no, "error": "invalid_thought_forest_schema"})
        if not thought_forest.get("nodes") or not thought_forest.get("edges"):
            errors.append({"line": line_no, "error": "empty_thought_forest"})
        if convextok_view.get("schema") != "toricgt.convextok_tokenization_dag.v1":
            errors.append({"line": line_no, "error": "invalid_convextok_dag_schema"})
        if "training_views_json" not in row or "enrichment_status_json" not in row or "leakage_signature_json" not in row:
            errors.append({"line": line_no, "error": "missing_training_enrichment_or_leakage_json"})
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
    build.add_argument("--records-per-dataset", type=int, default=8, help="Rows to graphify from each source dataset. Use 0 for all rows.")
    build.add_argument("--sample-per-dataset", type=int, help="Deprecated alias for --records-per-dataset.")
    build.add_argument("--max-total-records", type=int)
    build.add_argument("--output-prefix", default="uniprot_fot_graphified_sample")
    build.add_argument("--parquet-buffer-size", type=int, default=256)
    build.add_argument("--max-sequence-chars", type=int, default=4096)
    build.add_argument("--max-field-chars", type=int, default=4096)
    build.add_argument("--raw-link-mode", choices=["none", "symlink", "copy", "move"], default="symlink")
    build.add_argument("--convextok-tokenizer", default=str(DEFAULT_CONVEXTOK))
    build.add_argument("--convextok-max-bytes", type=int, default=768)
    build.add_argument("--convextok-max-token-bytes", type=int, default=96)
    build.add_argument("--no-jsonl", action="store_true", help="Write only compressed Parquet for derived records.")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--jsonl", default=str(OUTPUT_DIR / "derived" / "uniprot_fot_graphified_sample.jsonl"))

    args = parser.parse_args()
    if getattr(args, "sample_per_dataset", None) is not None:
        args.records_per_dataset = args.sample_per_dataset
    if getattr(args, "records_per_dataset", None) == 0:
        args.records_per_dataset = None
    if args.command == "build":
        result = build_records(args)
    elif args.command == "validate":
        result = validate_records(Path(args.jsonl))
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
