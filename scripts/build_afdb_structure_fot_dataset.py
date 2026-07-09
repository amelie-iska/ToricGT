#!/usr/bin/env python3
"""Build coordinate-bearing ToricBLM FoT records from AlphaFold DB mmCIFs.

The existing biological FoT records may contain only structure identifiers or
links.  This script fetches current AFDB mmCIF files for UniProt accessions,
extracts real residue coordinates, and emits graph/FoT training records with
explicit coordinate arrays.  These are intentionally real coordinate targets:
if AFDB has no model or parsing fails, the accession is skipped rather than
filled with a synthetic stand-in.

The default mode is now a full-corpus, sharded, resumable build.  It does not
keep raw mmCIF files unless explicitly requested, because hundreds of thousands
of AFDB files can exhaust local disk quickly.  The emitted Parquet shards keep
the trainable coordinate tensors and provenance URLs.
"""

from __future__ import annotations

import argparse
import glob as globlib
import hashlib
import json
import math
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from Bio.PDB import MMCIFParser

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


AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
AFDB_CIF = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v{version}.cif"


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.blake2b(value.encode("utf-8", errors="replace"), digest_size=16).hexdigest()[:length]


def safe_text(value: Any, limit: int = 2400) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def fetch_json(url: str, *, timeout: int = 30) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "ToricGT-AFDB-structure-curator/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, path: Path, *, timeout: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "ToricGT-AFDB-structure-curator/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        tmp.write_bytes(response.read())
    tmp.replace(path)


def download_file_status(url: str, path: Path, *, timeout: int = 60) -> bool:
    try:
        download_file(url, path, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 403, 404}:
            return False
        raise
    return True


def afdb_prediction(accession: str) -> dict[str, Any] | None:
    try:
        payload = fetch_json(AFDB_API.format(accession=accession))
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 400}:
            return None
        raise
    if not isinstance(payload, list) or not payload:
        return None
    payload = sorted(payload, key=lambda item: int(item.get("latestVersion") or 0), reverse=True)
    return payload[0]


def parse_cif_coordinates(cif_path: Path, *, max_residues: int) -> dict[str, Any] | None:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    model = next(structure.get_models())
    ca_coords: list[list[float]] = []
    backbone_coords: list[list[list[float]]] = []
    residue_names: list[str] = []
    residue_indices: list[int] = []
    plddt: list[float] = []
    chain_ids: list[str] = []
    atom_order = ("N", "CA", "C", "O")
    total_ca_residues = 0
    for chain in model:
        for residue in chain:
            if "CA" not in residue:
                continue
            total_ca_residues += 1
            if len(ca_coords) >= int(max_residues):
                continue
            atoms = []
            complete = True
            for atom_name in atom_order:
                if atom_name not in residue:
                    complete = False
                    atoms.append([math.nan, math.nan, math.nan])
                else:
                    coord = residue[atom_name].get_coord()
                    atoms.append([float(coord[0]), float(coord[1]), float(coord[2])])
            ca = residue["CA"]
            coord = ca.get_coord()
            ca_coords.append([float(coord[0]), float(coord[1]), float(coord[2])])
            backbone_coords.append(atoms if complete else atoms)
            residue_names.append(str(residue.get_resname()))
            residue_indices.append(int(residue.id[1]))
            plddt.append(float(ca.get_bfactor()))
            chain_ids.append(str(chain.id))
    if len(ca_coords) < 8:
        return None
    return {
        "ca_coordinates": ca_coords,
        "backbone_coordinates": backbone_coords,
        "residue_names": residue_names,
        "residue_indices": residue_indices,
        "chain_ids": chain_ids,
        "plddt": plddt,
        "residue_count": len(ca_coords),
        "coordinate_residue_count": len(ca_coords),
        "source_residue_count": total_ca_residues,
        "coordinate_truncated": total_ca_residues > len(ca_coords),
        "mean_plddt": float(sum(plddt) / max(1, len(plddt))),
    }


def extract_foldseek_3di_sequence(
    cif_path: Path,
    *,
    protrek_root: Path,
    foldseek_bin: Path,
    require: bool,
) -> tuple[str, str]:
    """Extract a real Foldseek/3Di structure sequence before mmCIF cleanup.

    This is the disk-safe bridge between AFDB curation and ProTrek trimodal
    splitting.  If the caller requests 3Di extraction, the value comes from the
    ProTrek Foldseek utility and the real mmCIF file; no hash or metadata
    substitute is emitted.
    """
    if not foldseek_bin.exists() or not foldseek_bin.is_file():
        if require:
            raise RuntimeError(f"Foldseek binary not found: {foldseek_bin}")
        return "", "foldseek_binary_missing"
    if not protrek_root.exists():
        if require:
            raise RuntimeError(f"ProTrek root not found: {protrek_root}")
        return "", "protrek_root_missing"
    sys.path.insert(0, str(protrek_root))
    try:
        from utils.foldseek_util import get_struc_seq  # type: ignore
    except Exception as exc:  # noqa: BLE001 - strict provenance message.
        if require:
            raise RuntimeError(f"failed to import ProTrek Foldseek utility: {type(exc).__name__}: {exc}") from exc
        return "", f"foldseek_util_import_error:{type(exc).__name__}"
    try:
        seqs = get_struc_seq(str(foldseek_bin), str(cif_path), ["A"])
        if not seqs:
            return "", "foldseek_no_chains"
        if "A" in seqs:
            seq = str(seqs["A"][1]).lower()
        else:
            seq = "".join(str(value[1]).lower() for _, value in sorted(seqs.items()))
    except Exception as exc:  # noqa: BLE001 - per-row extraction status.
        if require:
            raise RuntimeError(f"Foldseek 3Di extraction failed for {cif_path}: {type(exc).__name__}: {exc}") from exc
        return "", f"foldseek_error:{type(exc).__name__}"
    if len(seq) < 8:
        if require:
            raise RuntimeError(f"Foldseek 3Di sequence too short for {cif_path}: {len(seq)}")
        return "", "foldseek_sequence_too_short"
    return seq, "foldseek_3di_extracted"


def contact_summary(coords: list[list[float]], cutoff: float = 8.0, max_pairs: int = 64) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    n = len(coords)
    for i in range(n):
        xi, yi, zi = coords[i]
        for j in range(i + 3, n):
            xj, yj, zj = coords[j]
            dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
            if dist <= cutoff:
                out.append({"i": i, "j": j, "distance": round(dist, 3)})
                if len(out) >= max_pairs:
                    return out
    return out


def structure_graph(
    *,
    accession: str,
    entry_name: str,
    protein_name: str,
    sequence: str,
    function: str,
    prediction: dict[str, Any],
    coords: dict[str, Any],
) -> dict[str, Any]:
    seq_preview = sequence[:256]
    function_preview = safe_text(function, 700)
    nodes: list[dict[str, Any]] = [
        {
            "id": "protein",
            "type": "protein",
            "text": f"{accession} {entry_name} {protein_name}",
            "uniprot_accession": accession,
        },
        {"id": "sequence", "type": "amino_acid_sequence", "text": seq_preview, "length": len(sequence)},
        {"id": "function", "type": "uniprot_function", "text": function_preview},
        {
            "id": "afdb_structure",
            "type": "afdb_coordinate_structure",
            "text": (
                f"AlphaFold DB {prediction.get('modelEntityId')} version {prediction.get('latestVersion')} "
                f"mean_plddt {coords['mean_plddt']:.2f} coordinate_residues {coords['coordinate_residue_count']} "
                f"source_residues {coords['source_residue_count']}"
            ),
            "cif_url": prediction.get("cifUrl", ""),
            "bcif_url": prediction.get("bcifUrl", ""),
            "mean_plddt": coords["mean_plddt"],
        },
        {
            "id": "coordinate_target",
            "type": "structure_flow_target",
            "text": "CA and backbone coordinates are stored in coordinate columns for flow/contact/distogram training.",
            "residue_count": coords["coordinate_residue_count"],
            "source_residue_count": coords["source_residue_count"],
            "coordinate_truncated": coords["coordinate_truncated"],
        },
    ]
    segment = 32
    for start in range(0, coords["coordinate_residue_count"], segment):
        end = min(coords["coordinate_residue_count"], start + segment)
        mean_plddt = sum(coords["plddt"][start:end]) / max(1, end - start)
        nodes.append(
            {
                "id": f"residue_block_{start:04d}_{end:04d}",
                "type": "residue_coordinate_block",
                "text": (
                    f"residues {start + 1}-{end}; CA coordinate block; "
                    f"mean_plddt {mean_plddt:.2f}; names {''.join(name[:1] for name in coords['residue_names'][start:end])}"
                ),
                "start": start,
                "end": end,
                "mean_plddt": mean_plddt,
            }
        )
    edges: list[dict[str, Any]] = [
        {"id": "protein_has_sequence", "source": "protein", "target": "sequence", "type": "has_sequence"},
        {"id": "protein_has_function", "source": "protein", "target": "function", "type": "has_function_annotation"},
        {"id": "protein_has_structure", "source": "protein", "target": "afdb_structure", "type": "has_afdb_structure"},
        {"id": "structure_has_target", "source": "afdb_structure", "target": "coordinate_target", "type": "has_coordinate_target"},
        {"id": "sequence_to_structure", "source": "sequence", "target": "afdb_structure", "type": "sequence_folds_to_structure"},
    ]
    block_ids = [node["id"] for node in nodes if node["type"] == "residue_coordinate_block"]
    for idx, block_id in enumerate(block_ids):
        edges.append({"id": f"structure_contains_{idx}", "source": "afdb_structure", "target": block_id, "type": "contains_residue_block"})
        if idx:
            edges.append({"id": f"residue_next_{idx-1}_{idx}", "source": block_ids[idx - 1], "target": block_id, "type": "next_residue_block"})
    for pair_index, pair in enumerate(contact_summary(coords["ca_coordinates"])):
        src = f"residue_block_{(pair['i'] // segment) * segment:04d}_{min(coords['coordinate_residue_count'], (pair['i'] // segment + 1) * segment):04d}"
        dst = f"residue_block_{(pair['j'] // segment) * segment:04d}_{min(coords['coordinate_residue_count'], (pair['j'] // segment + 1) * segment):04d}"
        if src != dst:
            edges.append(
                {
                    "id": f"contact_block_{pair_index}",
                    "source": src,
                    "target": dst,
                    "type": "spatial_contact_block",
                    "text": f"CA contact distance {pair['distance']}",
                    "distance": pair["distance"],
                }
            )
    return {"nodes": nodes, "edges": edges}


def forest_for_structure(record_id: str, coords: dict[str, Any]) -> dict[str, Any]:
    nodes = [
        {"id": "root", "type": "structure_reasoning_root", "text": "Infer structure-function constraints from sequence, annotations, and AFDB coordinates."},
        {"id": "geometry", "type": "geometry_branch", "text": "Reason over CA distances, contacts, and residue blocks."},
        {"id": "confidence", "type": "confidence_branch", "text": f"Use pLDDT mean {coords['mean_plddt']:.2f} as uncertainty control."},
        {"id": "function", "type": "function_branch", "text": "Connect residue geometry to UniProt function annotation."},
        {"id": "design", "type": "design_branch", "text": "Prepare conditional de novo design constraints for later flow-matching phases."},
    ]
    edges = [
        {"source": "root", "target": "geometry", "type": "branch"},
        {"source": "root", "target": "confidence", "type": "branch"},
        {"source": "root", "target": "function", "type": "branch"},
        {"source": "geometry", "target": "design", "type": "geometry_conditioned_design"},
        {"source": "function", "target": "design", "type": "function_conditioned_design"},
        {"source": "confidence", "target": "design", "type": "uncertainty_gate"},
    ]
    return {"record_id": record_id, "nodes": nodes, "edges": edges}


def thought_forest_for_structure(
    *,
    record_id: str,
    accession: str,
    graph: dict[str, Any],
    coords: dict[str, Any],
    foldseek_3di_status: str,
) -> dict[str, Any]:
    """Build the source-grounded FoT object for coordinate-native rows."""

    graph_node_ids = {str(node.get("id")) for node in graph.get("nodes", [])}
    residue_blocks = [node_id for node_id in graph_node_ids if node_id.startswith("residue_block_")]

    def node(node_id: str, label: str, tree_id: str, graph_node_id: str, level: int, purpose: str) -> dict[str, Any]:
        return {
            "id": node_id,
            "label": label,
            "tree_id": tree_id,
            "graph_node_id": graph_node_id,
            "budget_level": level,
            "active": graph_node_id in graph_node_ids or graph_node_id == "coordinate_target",
            "purpose": purpose,
            "latent_coordinates": deterministic_vector(record_id + ":" + node_id, dims=8),
        }

    nodes = [
        node("ft_root", f"AFDB structure-function record {accession}", "root", "protein", 0, "start from the source-grounded coordinate graph"),
        node("ft_sequence_root", "sequence evidence tree", "sequence_tree", "sequence", 1, "connect primary sequence to the coordinate model"),
        node("ft_geometry_root", "coordinate geometry tree", "geometry_tree", "afdb_structure", 1, "reason over residue blocks, contacts, and distances"),
        node("ft_confidence_root", "uncertainty tree", "confidence_tree", "afdb_structure", 1, "gate structure claims by pLDDT and truncation status"),
        node("ft_function_root", "function annotation tree", "function_tree", "function", 1, "bind function text to structural evidence"),
        node("ft_design_root", "design constraint tree", "design_tree", "coordinate_target", 1, "prepare conditional structure generation targets"),
    ]
    edges: list[dict[str, Any]] = []

    def edge(source: str, target: str, edge_type: str, tree_id: str, rationale: str, weight: float = 1.0) -> None:
        edges.append(
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

    for target, tree_id in [
        ("ft_sequence_root", "sequence_tree"),
        ("ft_geometry_root", "geometry_tree"),
        ("ft_confidence_root", "confidence_tree"),
        ("ft_function_root", "function_tree"),
        ("ft_design_root", "design_tree"),
    ]:
        edge("ft_root", target, "tree_activation", tree_id, f"Activate {tree_id} from the coordinate-bearing graph.")

    previous = "ft_geometry_root"
    for index, block_id in enumerate(sorted(residue_blocks)[:12]):
        block_node = node(
            f"ft_geometry_block_{index:02d}",
            block_id,
            "geometry_tree",
            block_id,
            2 + index,
            "local residue-block geometry evidence",
        )
        nodes.append(block_node)
        edge(previous, block_node["id"], "expansion", "geometry_tree", f"Expand geometry tree to {block_id}.")
        previous = block_node["id"]

    correction_items = []
    if coords.get("coordinate_truncated"):
        correction_items.append(("coordinate_truncated", "coordinates are windowed/truncated; do not treat absent residues as nonexistent"))
    if float(coords.get("mean_plddt") or 0.0) < 70.0:
        correction_items.append(("low_mean_plddt", "low-confidence regions should downweight precise structure claims"))
    if foldseek_3di_status != "foldseek_3di_extracted":
        correction_items.append(("missing_3di", f"Foldseek/3Di evidence status: {foldseek_3di_status}"))
    for index, (name, purpose) in enumerate(correction_items):
        correction_id = f"ft_correction_{name}"
        nodes.append(node(correction_id, name.replace("_", " "), "self_correction_tree", "afdb_structure", 2 + index, purpose))
        edge("ft_confidence_root", correction_id, "self_correction", "self_correction_tree", purpose)

    consensus_id = "ft_consensus"
    nodes.append(
        node(
            consensus_id,
            "coordinate-grounded structure-function consensus",
            "consensus",
            "coordinate_target",
            8,
            "merge sequence, structure, confidence, function, and design evidence",
        )
    )
    for item in nodes:
        if item["id"] != consensus_id and item["id"].startswith("ft_") and int(item.get("budget_level", 0)) >= 1:
            edge(item["id"], consensus_id, "consensus", str(item.get("tree_id", "consensus")), "Active structure FoT node contributes to the consensus target.", 0.75)

    return {
        "schema": "toricgt.biomed_source_grounded_forest_of_thought.v1",
        "forest_type": "coordinate_grounded_structure_fot",
        "source_record_id": record_id,
        "trees": [
            {"tree_id": "sequence_tree", "label": "sequence evidence"},
            {"tree_id": "geometry_tree", "label": "coordinate geometry evidence"},
            {"tree_id": "confidence_tree", "label": "structure confidence and uncertainty"},
            {"tree_id": "function_tree", "label": "function annotation evidence"},
            {"tree_id": "design_tree", "label": "conditional design constraints"},
            {"tree_id": "self_correction_tree", "label": "missing or uncertain evidence correction"},
        ],
        "nodes": nodes,
        "edges": edges,
        "sparse_activation_policy": "activate sequence, geometry, confidence, function, and design trees from source fields; route low-confidence or unavailable evidence to self-correction nodes",
        "dynamic_self_correction": {
            "enabled": True,
            "policy": "structure claims must be conditioned on real coordinate availability, coordinate truncation, pLDDT, and Foldseek/3Di status",
        },
        "consensus": {
            "node": consensus_id,
            "leaf_selection_rule": "coordinate evidence with uncertainty gating and source function support",
        },
        "gflownet_training": {
            "state_space": "typed coordinate graph plus FoT nodes in embedding space",
            "forward_actions": ["tree_activation", "expansion", "self_correction", "consensus"],
            "backward_actions": ["contract_to_parent", "remove_unverified_leaf", "return_to_coordinate_source"],
            "reward_key": "graph_json.targets.gflownet_reward_metadata.terminal_reward_proxy",
            "trajectory_balance_compatible": True,
        },
    }


def enrich_structure_graph(
    *,
    graph: dict[str, Any],
    record_id: str,
    accession: str,
    entry_name: str,
    protein_name: str,
    sequence: str,
    function_text: str,
    coords: dict[str, Any],
    prediction: dict[str, Any],
    foldseek_3di_status: str,
    split: str,
) -> dict[str, Any]:
    node_order = [str(node.get("id")) for node in graph.get("nodes", [])]
    edge_order = [str(edge.get("id", f"edge_{idx}")) for idx, edge in enumerate(graph.get("edges", []))]
    quality_flags = {
        "coordinate_available": True,
        "coordinate_truncated": bool(coords.get("coordinate_truncated")),
        "mean_plddt": round(float(coords.get("mean_plddt") or 0.0), 4),
        "foldseek_3di_status": foldseek_3di_status,
        "source_residue_count": int(coords.get("source_residue_count") or 0),
        "coordinate_residue_count": int(coords.get("coordinate_residue_count") or 0),
    }
    reward_components = {
        "coordinate_availability": 1.0,
        "confidence": max(0.0, min(1.0, float(coords.get("mean_plddt") or 0.0) / 100.0)),
        "coverage": min(1.0, float(coords.get("coordinate_residue_count") or 0) / max(1.0, float(coords.get("source_residue_count") or 1))),
        "function_text": 1.0 if safe_text(function_text, 32) else 0.3,
        "foldseek_3di": 1.0 if foldseek_3di_status == "foldseek_3di_extracted" else 0.4,
    }
    graph.update(
        {
            "id": record_id,
            "source": {
                "name": "AlphaFold DB mmCIF via EBI API",
                "contains_imported_reasoning_trace": False,
                "model_entity_id": prediction.get("modelEntityId"),
                "cif_url": prediction.get("cifUrl", ""),
            },
            "task_family": "structure_function_fot_coordinate_training",
            "targets": {
                "answer": {
                    "kind": "coordinate_native_graph_out",
                    "summary": "predict and reason over real CA/backbone coordinate targets with graph/FoT context",
                },
                "structure_flow": {
                    "ca_coordinate_column": "ca_coordinates",
                    "backbone_coordinate_column": "backbone_coordinates",
                    "coordinate_mask_column": "coordinate_mask",
                    "plddt_column": "plddt",
                },
                "gflownet_reward_metadata": {
                    "reward_components": reward_components,
                    "terminal_reward_proxy": round(sum(reward_components.values()) / max(1, len(reward_components)), 6),
                },
            },
            "metadata": {
                "entry_name": entry_name,
                "protein_name": protein_name,
                "sequence_length": len(sequence),
                "tokengt_tropical_toric_fields": {
                    "node_token_order": node_order,
                    "edge_token_order": edge_order,
                    "structural_channels": ["typed_node", "typed_edge", "residue_block", "contact_block", "coordinate_target"],
                    "tropical_active_support_nodes": node_order[: min(16, len(node_order))],
                    "toric_phase_basis": "deterministic_coordinate_graph_phase",
                    "convextok_alignment": "structure-function text and sequence are byte-packed with ConvexTok DAG sidecar",
                },
                "quality_flags": quality_flags,
            },
            "split_cluster": {
                "cluster_id": "afdb_" + stable_hash(accession, 20),
                "group_hash": stable_hash(accession, 64),
                "split": split,
                "leakage_resistant_basis": "accession hash; upgrade path is ProTrek trimodal split for mixed structure-function records",
            },
        }
    )
    return graph


def iter_uniprot_rows(paths: list[Path], *, max_scan_rows: int, batch_size: int) -> Any:
    remaining = int(max_scan_rows) if int(max_scan_rows) > 0 else None
    columns = [
        "entry",
        "entry_name",
        "protein_name",
        "sequence",
        "function",
        "id",
        "name",
        "seed_id",
        "rep_member_id",
        "rep_protein_name",
        "rep_accessions",
        "rep_organism",
        "rep_organism_tax_id",
        "member_count",
        "common_taxon",
        "common_taxon_id",
        "go_mf",
        "go_bp",
        "go_cc",
    ]
    for path in paths:
        pf = pq.ParquetFile(path)
        available = set(pf.schema.names)
        selected = [column for column in columns if column in available]
        if "sequence" not in selected or not ({"entry", "rep_accessions", "rep_member_id", "seed_id"} & set(selected)):
            continue
        for batch in pf.iter_batches(batch_size=batch_size, columns=selected):
            for row in pa.Table.from_batches([batch]).to_pylist():
                yield row
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return


def normalize_uniprot_accession(value: Any) -> str:
    text = safe_text(value, 64)
    if not text:
        return ""
    text = text.split()[0].strip()
    if "|" in text:
        parts = [part for part in text.split("|") if part]
        if len(parts) >= 2 and parts[0].lower() in {"sp", "tr", "uniprotkb"}:
            text = parts[1]
        else:
            text = parts[0]
    if "_" in text and re.fullmatch(r"[A-Z0-9]+_[A-Z0-9]+", text):
        text = text.split("_", 1)[0]
    text = re.sub(r"[^A-Za-z0-9-]", "", text)
    return text[:64]


def row_accessions(row: dict[str, Any], *, max_accessions_per_row: int) -> list[str]:
    candidates: list[Any] = []
    for key in ("entry", "accession", "rep_member_id", "seed_id"):
        value = row.get(key)
        if value not in (None, ""):
            candidates.append(value)
    rep_accessions = row.get("rep_accessions")
    if isinstance(rep_accessions, list):
        candidates.extend(rep_accessions)
    elif rep_accessions not in (None, ""):
        candidates.append(rep_accessions)
    out: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        accession = normalize_uniprot_accession(value)
        if not accession or accession in seen:
            continue
        seen.add(accession)
        out.append(accession)
        if len(out) >= max(1, int(max_accessions_per_row)):
            break
    return out


def row_entry_name(row: dict[str, Any], accession: str) -> str:
    return safe_text(row.get("entry_name") or row.get("id") or row.get("seed_id") or accession, 256)


def row_protein_name(row: dict[str, Any]) -> str:
    return safe_text(row.get("protein_name") or row.get("rep_protein_name") or row.get("name") or "protein", 1024)


def row_function_text(row: dict[str, Any]) -> str:
    parts = []
    for key in ("function", "rep_protein_name", "name", "common_taxon", "rep_organism", "go_mf", "go_bp", "go_cc"):
        value = row.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            parts.append(f"{key}: " + ", ".join(map(str, value[:32])))
        else:
            parts.append(f"{key}: {value}")
    return safe_text("; ".join(parts), 2400)


def split_for_accession(accession: str) -> str:
    bucket = int(stable_hash(str(accession), length=8), 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


class SplitShardWriter:
    def __init__(self, out_dir: Path, *, shard_size: int, prefix: str) -> None:
        self.out_dir = out_dir
        self.shard_size = int(shard_size)
        self.prefix = prefix
        self.buffers: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
        self.counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
        self.shard_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
        self.paths: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
        for split in self.buffers:
            (self.out_dir / split).mkdir(parents=True, exist_ok=True)

    def add(self, record: dict[str, Any]) -> None:
        split = split_for_accession(str(record["uniprot_accession"]))
        record = dict(record)
        record["split"] = split
        self.buffers[split].append(record)
        self.counts[split] += 1
        if len(self.buffers[split]) >= self.shard_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        shard_idx = self.shard_counts[split]
        path = self.out_dir / split / f"{self.prefix}_{split}_{shard_idx:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        self.paths[split].append(str(path))
        self.shard_counts[split] += 1
        self.buffers[split] = []

    def close(self) -> dict[str, Any]:
        for split in list(self.buffers):
            self.flush(split)
        return {
            "counts": dict(self.counts),
            "shards": dict(self.shard_counts),
            "paths": dict(self.paths),
        }


def load_processed(progress_path: Path) -> set[str]:
    processed: set[str] = set()
    if not progress_path.exists():
        return processed
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            accession = item.get("accession")
            if accession:
                processed.add(str(accession))
    return processed


def append_progress(progress_path: Path, payload: dict[str, Any]) -> None:
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return float(usage.free) / (1024.0 ** 3)


def resolve_cif(
    accession: str,
    *,
    out_dir: Path,
    cache_dir: Path,
    tmp_dir: Path,
    direct_version: int,
    keep_mmcif_cache: bool,
    timeout: int,
) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    direct_url = AFDB_CIF.format(accession=accession, version=int(direct_version))
    target_dir = cache_dir if keep_mmcif_cache else tmp_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    direct_path = target_dir / f"AF-{accession}-F1-model_v{int(direct_version)}.cif"
    if not direct_path.exists():
        try:
            ok = download_file_status(direct_url, direct_path, timeout=timeout)
        except Exception as exc:
            return None, None, f"direct_download_error:{type(exc).__name__}"
        if not ok:
            try:
                prediction = afdb_prediction(accession)
            except Exception as exc:
                return None, None, f"api_error:{type(exc).__name__}"
            if prediction is None or not prediction.get("cifUrl"):
                return None, None, "no_afdb_prediction"
            version = int(prediction.get("latestVersion") or direct_version)
            fallback_path = target_dir / f"AF-{accession}-F1-model_v{version}.cif"
            if not fallback_path.exists():
                try:
                    download_file(str(prediction["cifUrl"]), fallback_path, timeout=timeout)
                except Exception as exc:
                    return None, None, f"download_error:{type(exc).__name__}"
            return prediction, fallback_path, None
    prediction = {
        "modelEntityId": f"AF-{accession}-F1",
        "latestVersion": int(direct_version),
        "cifUrl": direct_url,
        "bcifUrl": "",
    }
    return prediction, direct_path, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniprot-function-parquet", type=Path, default=None)
    parser.add_argument("--uniprot-function-glob", type=str, default="/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale/uniprot_function_text_train/default/train/*.parquet")
    parser.add_argument("--out-dir", type=Path, default=Path("data/uniprot_fot/structures/afdb_v6"))
    parser.add_argument("--max-records", type=int, default=0, help="0 means no accepted-record cap")
    parser.add_argument("--max-scan-rows", type=int, default=0, help="0 means scan all available rows")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--max-residues", type=int, default=512, help="maximum coordinate residues stored per row; long proteins are windowed/truncated, not skipped")
    parser.add_argument("--min-residues", type=int, default=16)
    parser.add_argument("--direct-version", type=int, default=6)
    parser.add_argument("--download-timeout", type=int, default=90)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-accessions-per-row", type=int, default=1, help="Maximum UniProt-like accessions to attempt per source row.")
    parser.add_argument("--convextok-tokenizer", type=Path, default=DEFAULT_CONVEXTOK)
    parser.add_argument("--convextok-max-bytes", type=int, default=1024)
    parser.add_argument("--keep-mmcif-cache", action="store_true")
    parser.add_argument("--emit-foldseek-3di", action="store_true", help="extract and store real Foldseek/3Di structure sequences before deleting mmCIF files")
    parser.add_argument("--require-foldseek-3di", action="store_true", help="fail rows when requested Foldseek/3Di extraction is unavailable")
    parser.add_argument("--protrek-root", type=Path, default=Path("external/ProTrek"))
    parser.add_argument("--foldseek-bin", type=Path, default=Path("external/ProTrek/bin/foldseek"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prefix", type=str, default="toricblm_afdb_structure_fot")
    parser.add_argument("--min-free-gb", type=float, default=25.0, help="stop cleanly before downloads if output filesystem drops below this many GiB")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "mmcif_cache"
    tmp_dir = args.out_dir / "_tmp_mmcif"
    progress_path = args.out_dir / "build_progress.jsonl"
    input_paths: list[Path]
    if args.uniprot_function_parquet is not None:
        input_paths = [args.uniprot_function_parquet]
    else:
        input_paths = [Path(path) for path in sorted(globlib.glob(args.uniprot_function_glob))]
    if not input_paths:
        raise SystemExit(f"No input Parquet files matched: {args.uniprot_function_parquet or args.uniprot_function_glob}")
    convextok_tokens = index_convextok_tokens(load_convextok_tokens(args.convextok_tokenizer, max_token_bytes=96))
    processed = load_processed(progress_path) if args.resume else set()
    writer = SplitShardWriter(args.out_dir, shard_size=args.shard_size, prefix=args.prefix)
    skipped: dict[str, int] = {}
    accepted = 0
    scanned = 0
    stopped_reason: str | None = None
    for row in iter_uniprot_rows(input_paths, max_scan_rows=args.max_scan_rows, batch_size=args.batch_size):
        if args.max_records > 0 and accepted >= args.max_records:
            break
        if free_gb(args.out_dir) < float(args.min_free_gb):
            stopped_reason = f"disk_guard_free_gb_below_{float(args.min_free_gb):.1f}"
            append_progress(
                progress_path,
                {
                    "accession": "",
                    "status": "stopped",
                    "reason": stopped_reason,
                    "accepted": accepted,
                    "scanned": scanned,
                    "free_gb": round(free_gb(args.out_dir), 3),
                },
            )
            break
        scanned += 1
        sequence = safe_text(row.get("sequence"), 100000)
        accessions = row_accessions(row, max_accessions_per_row=args.max_accessions_per_row)
        if not accessions or len(sequence) < args.min_residues:
            skipped["length_or_accession"] = skipped.get("length_or_accession", 0) + 1
            append_progress(progress_path, {"accession": "", "status": "skipped", "reason": "length_or_accession"})
            continue
        for accession in accessions:
            if args.max_records > 0 and accepted >= args.max_records:
                break
            if accession in processed:
                continue
            prediction, cif_path, error = resolve_cif(
                accession,
                out_dir=args.out_dir,
                cache_dir=cache_dir,
                tmp_dir=tmp_dir,
                direct_version=args.direct_version,
                keep_mmcif_cache=args.keep_mmcif_cache,
                timeout=args.download_timeout,
            )
            if error is not None or prediction is None or cif_path is None:
                reason = error or "no_afdb_prediction"
                skipped[reason] = skipped.get(reason, 0) + 1
                append_progress(progress_path, {"accession": accession, "status": "skipped", "reason": reason})
                continue
            version = int(prediction.get("latestVersion") or args.direct_version)
            try:
                coords = parse_cif_coordinates(cif_path, max_residues=args.max_residues)
            except Exception as exc:
                reason = f"parse_error:{type(exc).__name__}"
                skipped[reason] = skipped.get(reason, 0) + 1
                append_progress(progress_path, {"accession": accession, "status": "skipped", "reason": reason})
                if not args.keep_mmcif_cache:
                    cif_path.unlink(missing_ok=True)
                continue
            if coords is None:
                skipped["too_few_coordinates"] = skipped.get("too_few_coordinates", 0) + 1
                append_progress(progress_path, {"accession": accession, "status": "skipped", "reason": "too_few_coordinates"})
                if not args.keep_mmcif_cache:
                    cif_path.unlink(missing_ok=True)
                continue
            foldseek_3di_sequence = ""
            foldseek_3di_status = "not_requested"
            if args.emit_foldseek_3di or args.require_foldseek_3di:
                try:
                    foldseek_3di_sequence, foldseek_3di_status = extract_foldseek_3di_sequence(
                        cif_path,
                        protrek_root=args.protrek_root,
                        foldseek_bin=args.foldseek_bin,
                        require=bool(args.require_foldseek_3di),
                    )
                except Exception as exc:  # noqa: BLE001 - strict row status, no substitute.
                    reason = f"foldseek_3di_error:{type(exc).__name__}"
                    skipped[reason] = skipped.get(reason, 0) + 1
                    append_progress(progress_path, {"accession": accession, "status": "skipped", "reason": f"{reason}: {exc}"})
                    if not args.keep_mmcif_cache:
                        cif_path.unlink(missing_ok=True)
                    continue
            if not args.keep_mmcif_cache:
                cif_path.unlink(missing_ok=True)
            record_id = f"afdb_structure_fot_{accession}_{stable_hash(accession)}"
            entry_name = row_entry_name(row, accession)
            protein_name = row_protein_name(row)
            function_text = row_function_text(row)
            graph = structure_graph(
                accession=accession,
                entry_name=entry_name,
                protein_name=protein_name,
                sequence=sequence,
                function=function_text,
                prediction=prediction,
                coords=coords,
            )
            split = split_for_accession(accession)
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
            forest = forest_for_structure(record_id, coords)
            thought_forest = thought_forest_for_structure(
                record_id=record_id,
                accession=accession,
                graph=graph,
                coords=coords,
                foldseek_3di_status=foldseek_3di_status,
            )
            text = (
                f"UniProt {accession} structure-function record. "
                f"Protein: {protein_name}. "
                f"Function: {safe_text(function_text, 1600)}. "
                f"AFDB model {prediction.get('modelEntityId')} v{version}; "
                f"coordinate residues {coords['coordinate_residue_count']} of source residues {coords['source_residue_count']}; "
                f"mean pLDDT {coords['mean_plddt']:.2f}. "
                "Train on real CA/backbone coordinates for flow matching, contact prediction, and distogram geometry."
            )
            convextok_view = convextok_dag(text + "\n" + sequence[:512], convextok_tokens, max_bytes=int(args.convextok_max_bytes))
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
                "coordinate_source": "AFDB",
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
                "accession_hash": stable_hash(accession, 32),
            }
            content_hash = hashlib.sha256(
                stable_json(
                    {
                        "graph": graph,
                        "forest": forest,
                        "thought_forest": thought_forest,
                        "convextok": convextok_view,
                        "accession": accession,
                    }
                ).encode("utf-8")
            ).hexdigest()
            record = {
                "record_id": record_id,
                "dataset": "toricblm_afdb_structure_fot",
                "source_file": ",".join(str(path) for path in input_paths[:4]),
                "source_row_index": scanned,
                "entry_id": accession,
                "task_family": "structure_function_fot_coordinate_training",
                "uniprot_accession": accession,
                "entry_name": entry_name,
                "protein_name": protein_name,
                "sequence": sequence,
                "annotation_text": function_text,
                "function": function_text,
                "text": text,
                "graph_json": json.dumps(graph, ensure_ascii=True, sort_keys=True),
                "forest_json": json.dumps(forest, ensure_ascii=True, sort_keys=True),
                "thought_forest_json": json.dumps(thought_forest, ensure_ascii=True, sort_keys=True),
                "convextok_dag_json": json.dumps(convextok_view, ensure_ascii=True, sort_keys=True),
                "training_views_json": json.dumps(training_views, ensure_ascii=True, sort_keys=True),
                "enrichment_status_json": json.dumps(enrichment_status, ensure_ascii=True, sort_keys=True),
                "leakage_signature_json": json.dumps(leakage_signature, ensure_ascii=True, sort_keys=True),
                "metadata_json": json.dumps(graph["metadata"], ensure_ascii=True, sort_keys=True),
                "content_hash": content_hash,
                "group_hash": graph["split_cluster"]["group_hash"],
                "quality_flags_json": json.dumps(graph["metadata"]["quality_flags"], ensure_ascii=True, sort_keys=True),
                "structure_coordinates": coords["ca_coordinates"],
                "ca_coordinates": coords["ca_coordinates"],
                "backbone_coordinates": coords["backbone_coordinates"],
                "coordinate_mask": [True] * coords["residue_count"],
                "plddt": coords["plddt"],
                "residue_names": coords["residue_names"],
                "residue_indices": coords["residue_indices"],
                "chain_ids": coords["chain_ids"],
                "structure_source": "AFDB",
                "structure_file": str(cif_path) if args.keep_mmcif_cache else "",
                "structure_file_cached": bool(args.keep_mmcif_cache),
                "structure_cif_url": str(prediction.get("cifUrl")),
                "foldseek_3di_sequence": foldseek_3di_sequence,
                "foldseek_3di_available": bool(foldseek_3di_sequence),
                "foldseek_3di_status": foldseek_3di_status,
                "mean_plddt": coords["mean_plddt"],
                "residue_count": coords["coordinate_residue_count"],
                "coordinate_residue_count": coords["coordinate_residue_count"],
                "source_residue_count": coords["source_residue_count"],
                "coordinate_truncated": coords["coordinate_truncated"],
                "split_cluster": json.dumps(graph["split_cluster"], ensure_ascii=True, sort_keys=True),
            }
            writer.add(record)
            accepted += 1
            append_progress(
                progress_path,
                {
                    "accession": accession,
                    "status": "accepted",
                    "split": split_for_accession(accession),
                    "coordinate_residue_count": coords["coordinate_residue_count"],
                    "source_residue_count": coords["source_residue_count"],
                    "mean_plddt": round(coords["mean_plddt"], 4),
                    "foldseek_3di_available": bool(foldseek_3di_sequence),
                    "foldseek_3di_status": foldseek_3di_status,
                },
            )
            if args.sleep > 0:
                time.sleep(max(0.0, float(args.sleep)))
            cap = "unlimited" if args.max_records <= 0 else str(args.max_records)
            print(
                f"accepted {accepted:07d}/{cap}: {accession} "
                f"coord_residues={coords['coordinate_residue_count']} source_residues={coords['source_residue_count']} "
                f"mean_plddt={coords['mean_plddt']:.2f}",
                flush=True,
            )

    split_report = writer.close()
    if accepted <= 0:
        raise SystemExit(f"No coordinate-bearing AFDB records built. skipped={skipped}")
    if tmp_dir.exists() and not args.keep_mmcif_cache:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    manifest = {
        "records": accepted,
        "scanned_rows": scanned,
        "split_report": split_report,
        "skipped": skipped,
        "coordinate_bearing_records": accepted,
        "coordinate_source": "AlphaFold DB current API mmCIF",
        "max_residues": args.max_residues,
        "direct_version": args.direct_version,
        "keep_mmcif_cache": bool(args.keep_mmcif_cache),
        "emit_foldseek_3di": bool(args.emit_foldseek_3di or args.require_foldseek_3di),
        "require_foldseek_3di": bool(args.require_foldseek_3di),
        "protrek_root": str(args.protrek_root),
        "foldseek_bin": str(args.foldseek_bin),
        "input_paths": [str(path) for path in input_paths],
        "progress_path": str(progress_path),
        "stopped_reason": stopped_reason,
        "free_gb_after": round(free_gb(args.out_dir), 3),
        "min_free_gb": float(args.min_free_gb),
    }
    manifest_path = args.out_dir / "toricblm_afdb_structure_fot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
