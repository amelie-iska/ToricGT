#!/usr/bin/env python3
"""Build coordinate-bearing ToricBLM FoT records from AlphaFold DB mmCIFs.

The existing biological FoT records may contain only structure identifiers or
links.  This script fetches current AFDB mmCIF files for UniProt accessions,
extracts real residue coordinates, and emits graph/FoT training records with
explicit coordinate arrays.  These are intentionally real coordinate targets:
if AFDB has no model or parsing fails, the accession is skipped rather than
filled with a synthetic stand-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from Bio.PDB import MMCIFParser


AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


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
    for chain in model:
        for residue in chain:
            if "CA" not in residue:
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
            if len(ca_coords) >= int(max_residues):
                break
        if len(ca_coords) >= int(max_residues):
            break
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
        "mean_plddt": float(sum(plddt) / max(1, len(plddt))),
    }


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
                f"mean_plddt {coords['mean_plddt']:.2f} residues {coords['residue_count']}"
            ),
            "cif_url": prediction.get("cifUrl", ""),
            "bcif_url": prediction.get("bcifUrl", ""),
            "mean_plddt": coords["mean_plddt"],
        },
        {
            "id": "coordinate_target",
            "type": "structure_flow_target",
            "text": "CA and backbone coordinates are stored in coordinate columns for flow/contact/distogram training.",
            "residue_count": coords["residue_count"],
        },
    ]
    segment = 32
    for start in range(0, coords["residue_count"], segment):
        end = min(coords["residue_count"], start + segment)
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
        src = f"residue_block_{(pair['i'] // segment) * segment:04d}_{min(coords['residue_count'], (pair['i'] // segment + 1) * segment):04d}"
        dst = f"residue_block_{(pair['j'] // segment) * segment:04d}_{min(coords['residue_count'], (pair['j'] // segment + 1) * segment):04d}"
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


def iter_uniprot_rows(path: Path, *, max_scan_rows: int) -> Any:
    pf = pq.ParquetFile(path)
    remaining = int(max_scan_rows)
    columns = ["entry", "entry_name", "protein_name", "sequence", "function"]
    for batch in pf.iter_batches(batch_size=512, columns=columns):
        for row in pa.Table.from_batches([batch]).to_pylist():
            yield row
            remaining -= 1
            if remaining <= 0:
                return


def write_splits(records: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    splits = {"train": [], "validation": [], "test": []}
    for record in records:
        bucket = int(stable_hash(str(record["uniprot_accession"]), length=8), 16) % 100
        if bucket < 90:
            splits["train"].append(record)
        elif bucket < 95:
            splits["validation"].append(record)
        else:
            splits["test"].append(record)
    split_paths: dict[str, str] = {}
    for split, rows in splits.items():
        path = out_dir / f"toricblm_afdb_structure_fot_{split}.parquet"
        if rows:
            pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        else:
            pq.write_table(pa.Table.from_pylist(records[:0]), path, compression="zstd")
        split_paths[split] = str(path)
    return {
        "counts": {split: len(rows) for split, rows in splits.items()},
        "paths": split_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniprot-function-parquet", type=Path, default=Path("/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale/uniprot_function_text_train/default/train/0000.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/uniprot_fot/structures/afdb_v6"))
    parser.add_argument("--max-records", type=int, default=256)
    parser.add_argument("--max-scan-rows", type=int, default=5000)
    parser.add_argument("--max-residues", type=int, default=512)
    parser.add_argument("--min-residues", type=int, default=16)
    parser.add_argument("--sleep", type=float, default=0.02)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "mmcif_cache"
    records: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for row in iter_uniprot_rows(args.uniprot_function_parquet, max_scan_rows=args.max_scan_rows):
        if len(records) >= args.max_records:
            break
        accession = safe_text(row.get("entry"), 64)
        sequence = safe_text(row.get("sequence"), 100000)
        if not accession or len(sequence) < args.min_residues or len(sequence) > args.max_residues:
            skipped["length_or_accession"] = skipped.get("length_or_accession", 0) + 1
            continue
        try:
            prediction = afdb_prediction(accession)
        except Exception as exc:
            skipped[f"api_error:{type(exc).__name__}"] = skipped.get(f"api_error:{type(exc).__name__}", 0) + 1
            continue
        if prediction is None or not prediction.get("cifUrl"):
            skipped["no_afdb_prediction"] = skipped.get("no_afdb_prediction", 0) + 1
            continue
        version = int(prediction.get("latestVersion") or 0)
        cif_path = cache_dir / f"AF-{accession}-F1-model_v{version}.cif"
        if not cif_path.exists():
            try:
                download_file(str(prediction["cifUrl"]), cif_path)
                time.sleep(max(0.0, float(args.sleep)))
            except Exception as exc:
                skipped[f"download_error:{type(exc).__name__}"] = skipped.get(f"download_error:{type(exc).__name__}", 0) + 1
                continue
        try:
            coords = parse_cif_coordinates(cif_path, max_residues=args.max_residues)
        except Exception as exc:
            skipped[f"parse_error:{type(exc).__name__}"] = skipped.get(f"parse_error:{type(exc).__name__}", 0) + 1
            continue
        if coords is None:
            skipped["too_few_coordinates"] = skipped.get("too_few_coordinates", 0) + 1
            continue
        record_id = f"afdb_structure_fot_{accession}_{stable_hash(accession)}"
        graph = structure_graph(
            accession=accession,
            entry_name=safe_text(row.get("entry_name"), 160),
            protein_name=safe_text(row.get("protein_name"), 500),
            sequence=sequence,
            function=safe_text(row.get("function"), 3000),
            prediction=prediction,
            coords=coords,
        )
        forest = forest_for_structure(record_id, coords)
        text = (
            f"UniProt {accession} structure-function record. "
            f"Protein: {safe_text(row.get('protein_name'), 500)}. "
            f"Function: {safe_text(row.get('function'), 1600)}. "
            f"AFDB model {prediction.get('modelEntityId')} v{version}; "
            f"residues {coords['residue_count']}; mean pLDDT {coords['mean_plddt']:.2f}. "
            "Train on real CA/backbone coordinates for flow matching, contact prediction, and distogram geometry."
        )
        records.append(
            {
                "record_id": record_id,
                "dataset": "toricblm_afdb_structure_fot",
                "task_family": "structure_function_fot_coordinate_training",
                "uniprot_accession": accession,
                "entry_name": safe_text(row.get("entry_name"), 160),
                "protein_name": safe_text(row.get("protein_name"), 500),
                "sequence": sequence,
                "function": safe_text(row.get("function"), 3000),
                "text": text,
                "graph_json": json.dumps(graph, ensure_ascii=True, sort_keys=True),
                "forest_json": json.dumps(forest, ensure_ascii=True, sort_keys=True),
                "metadata_json": json.dumps(
                    {
                        "source": "AlphaFold DB mmCIF via EBI API",
                        "model_entity_id": prediction.get("modelEntityId"),
                        "latest_version": version,
                        "cif_url": prediction.get("cifUrl"),
                        "bcif_url": prediction.get("bcifUrl"),
                        "mean_plddt": coords["mean_plddt"],
                        "residue_count": coords["residue_count"],
                        "coordinate_columns": ["ca_coordinates", "backbone_coordinates"],
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "structure_coordinates": coords["ca_coordinates"],
                "ca_coordinates": coords["ca_coordinates"],
                "backbone_coordinates": coords["backbone_coordinates"],
                "coordinate_mask": [True] * coords["residue_count"],
                "plddt": coords["plddt"],
                "residue_names": coords["residue_names"],
                "residue_indices": coords["residue_indices"],
                "chain_ids": coords["chain_ids"],
                "structure_source": "AFDB",
                "structure_file": str(cif_path),
                "structure_cif_url": str(prediction.get("cifUrl")),
                "mean_plddt": coords["mean_plddt"],
                "residue_count": coords["residue_count"],
                "split_cluster": stable_hash(accession, 12),
            }
        )
        print(f"accepted {len(records):04d}/{args.max_records}: {accession} residues={coords['residue_count']} mean_plddt={coords['mean_plddt']:.2f}", flush=True)

    if not records:
        raise SystemExit(f"No coordinate-bearing AFDB records built. skipped={skipped}")
    all_path = args.out_dir / "toricblm_afdb_structure_fot_all.parquet"
    pq.write_table(pa.Table.from_pylist(records), all_path, compression="zstd")
    split_report = write_splits(records, args.out_dir)
    manifest = {
        "records": len(records),
        "all_path": str(all_path),
        "split_report": split_report,
        "skipped": skipped,
        "coordinate_bearing_records": len(records),
        "coordinate_source": "AlphaFold DB current API mmCIF",
        "max_residues": args.max_residues,
        "uniprot_function_parquet": str(args.uniprot_function_parquet),
    }
    manifest_path = args.out_dir / "toricblm_afdb_structure_fot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
