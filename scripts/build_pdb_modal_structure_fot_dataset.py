#!/usr/bin/env python3
"""Build coordinate-bearing ToricBLM records from real PDB/mmCIF structures.

This script is intentionally conservative: it emits coordinate-native training
records only from real structure files or RCSB-downloaded mmCIFs.  It does not
invent molecule conformers, secondary structures, or coordinates.  Protein,
RNA, DNA, ligand, and mixed-complex rows share the same Parquet coordinate
schema consumed by the ToricBLM structure-flow trainer.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from Bio.PDB import MMCIFParser, PDBParser


RCSB_CIF = "https://files.rcsb.org/download/{pdb_id}.cif"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"

AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL", "MSE",
}
RNA3 = {"A", "C", "G", "U", "I", "PSU", "1MA", "2MG", "5MC", "7MG", "OMG", "OMC", "YG"}
DNA3 = {"DA", "DC", "DG", "DT", "DI", "DU"}
WATER = {"HOH", "WAT", "DOD"}


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.blake2b(value.encode("utf-8", errors="replace"), digest_size=16).hexdigest()[:length]


def safe_text(value: Any, limit: int = 2400) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def split_for_id(value: str) -> str:
    bucket = int(stable_hash(value, length=8), 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def download_rcsb_cif(pdb_id: str, path: Path, *, timeout: int = 60) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        RCSB_CIF.format(pdb_id=pdb_id.upper()),
        headers={"User-Agent": "ToricGT-RCSB-structure-curator/1.0"},
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            tmp.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 403, 404}:
            tmp.unlink(missing_ok=True)
            return False
        raise
    tmp.replace(path)
    return True


def rcsb_search_payload(modality: str, *, start: int, rows: int) -> dict[str, Any]:
    """Build a conservative RCSB search query for real coordinate entries."""
    modality = modality.lower().strip()

    def node(attribute: str) -> dict[str, Any]:
        return {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": attribute,
                "operator": "greater_or_equal",
                "value": 1,
            },
        }

    protein = node("rcsb_entry_info.polymer_entity_count_protein")
    rna = node("rcsb_entry_info.polymer_entity_count_RNA")
    dna = node("rcsb_entry_info.polymer_entity_count_DNA")
    ligand = node("rcsb_entry_info.nonpolymer_entity_count")
    if modality in {"protein", "pdb-protein"}:
        query = protein
    elif modality in {"rna", "pdb-rna", "ndb-rna"}:
        query = rna
    elif modality in {"dna", "pdb-dna", "ndb-dna"}:
        query = dna
    elif modality in {"ligand", "small_molecule"}:
        query = ligand
    elif modality in {"protein_rna", "protein-rna"}:
        query = {"type": "group", "logical_operator": "and", "nodes": [protein, rna]}
    elif modality in {"protein_dna", "protein-dna"}:
        query = {"type": "group", "logical_operator": "and", "nodes": [protein, dna]}
    elif modality in {"nucleic_acid", "rna_dna", "rna-dna"}:
        query = {"type": "group", "logical_operator": "or", "nodes": [rna, dna]}
    elif modality in {"complex", "biomolecular_complex", "mixed"}:
        query = {"type": "group", "logical_operator": "and", "nodes": [protein, {"type": "group", "logical_operator": "or", "nodes": [rna, dna, ligand]}]}
    else:
        raise ValueError(f"unsupported RCSB query modality: {modality}")
    return {
        "query": query,
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": int(start), "rows": int(rows)},
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}],
            "results_content_type": ["experimental"],
        },
    }


def query_rcsb_ids(
    modalities: list[str],
    *,
    limit_per_modality: int,
    page_size: int,
    timeout: int,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    page_size = max(1, int(page_size))
    for modality in modalities:
        start = 0
        remaining = int(limit_per_modality)
        while True:
            if limit_per_modality > 0 and remaining <= 0:
                break
            rows = min(page_size, remaining) if limit_per_modality > 0 else page_size
            payload = json.dumps(rcsb_search_payload(modality, start=start, rows=rows)).encode("utf-8")
            request = urllib.request.Request(
                RCSB_SEARCH,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "ToricGT-RCSB-structure-curator/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            result_set = data.get("result_set", [])
            if not result_set:
                break
            for item in result_set:
                identifier = str(item.get("identifier") or "").strip().lower()
                if identifier and identifier not in seen:
                    seen.add(identifier)
                    ids.append(identifier)
            print(
                f"rcsb_query modality={modality} start={start} rows={len(result_set)} unique_ids={len(ids)}",
                flush=True,
            )
            if len(result_set) < rows:
                break
            start += rows
            if limit_per_modality > 0:
                remaining -= rows
    return ids


def residue_modality(resname: str) -> str:
    name = resname.strip().upper()
    if name in AA3:
        return "protein"
    if name in RNA3:
        return "rna"
    if name in DNA3:
        return "dna"
    if name in WATER:
        return "water"
    return "ligand"


def representative_atom(residue: Any, modality: str) -> Any | None:
    preferences = {
        "protein": ("CA", "C", "N"),
        "rna": ("P", "C4'", "C4*", "C1'", "C1*"),
        "dna": ("P", "C4'", "C4*", "C1'", "C1*"),
        "ligand": tuple(atom.get_name() for atom in residue),
    }
    for atom_name in preferences.get(modality, ()):
        if atom_name in residue:
            return residue[atom_name]
    for atom in residue:
        return atom
    return None


def parse_structure(path: Path, *, max_atoms: int, include_ligands: bool) -> dict[str, Any] | None:
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    elif suffix in {".pdb", ".ent"}:
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported structure suffix: {path}")
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())
    coords: list[list[float]] = []
    residue_names: list[str] = []
    residue_indices: list[int] = []
    chain_ids: list[str] = []
    modalities: list[str] = []
    b_factors: list[float] = []
    counts: dict[str, int] = {"protein": 0, "rna": 0, "dna": 0, "ligand": 0}
    source_residue_count = 0
    for chain in model:
        for residue in chain:
            resname = str(residue.get_resname()).strip().upper()
            modality = residue_modality(resname)
            if modality == "water":
                continue
            if modality == "ligand" and not include_ligands:
                continue
            source_residue_count += 1
            counts[modality] = counts.get(modality, 0) + 1
            if len(coords) >= max_atoms:
                continue
            atom = representative_atom(residue, modality)
            if atom is None:
                continue
            xyz = atom.get_coord()
            coords.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
            residue_names.append(resname)
            residue_indices.append(int(residue.id[1]) if isinstance(residue.id[1], int) else len(coords))
            chain_ids.append(str(chain.id))
            modalities.append(modality)
            b_factors.append(float(atom.get_bfactor()))
    if len(coords) < 8:
        return None
    active = [key for key, value in counts.items() if value > 0]
    if {"rna", "dna"} & set(active) and "protein" in active:
        structure_modality = "protein_nucleic_acid_complex"
    elif len([x for x in active if x != "ligand"]) > 1 or (len(active) > 1 and "ligand" in active):
        structure_modality = "biomolecular_complex"
    elif "rna" in active:
        structure_modality = "rna"
    elif "dna" in active:
        structure_modality = "dna"
    elif "protein" in active:
        structure_modality = "protein"
    else:
        structure_modality = "ligand"
    return {
        "structure_coordinates": coords,
        "coordinate_mask": [True] * len(coords),
        "residue_names": residue_names,
        "residue_indices": residue_indices,
        "chain_ids": chain_ids,
        "atom_modalities": modalities,
        "b_factors": b_factors,
        "coordinate_residue_count": len(coords),
        "source_residue_count": source_residue_count,
        "coordinate_truncated": source_residue_count > len(coords),
        "modality_counts": counts,
        "structure_modality": structure_modality,
    }


def contact_summary(coords: list[list[float]], cutoff: float = 8.0, max_pairs: int = 96) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, a in enumerate(coords):
        for j in range(i + 3, len(coords)):
            b = coords[j]
            dist = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
            if dist <= cutoff:
                out.append({"i": i, "j": j, "distance": round(dist, 3)})
                if len(out) >= max_pairs:
                    return out
    return out


def graph_for_record(record_id: str, structure_id: str, parsed: dict[str, Any], source_path: str) -> dict[str, Any]:
    nodes = [
        {"id": "structure", "type": parsed["structure_modality"], "text": f"{structure_id} {parsed['structure_modality']} coordinate structure"},
        {"id": "coordinate_target", "type": "structure_flow_target", "text": "Real representative atom coordinates are stored for flow/contact/distogram training."},
    ]
    for modality, count in sorted(parsed["modality_counts"].items()):
        if count:
            nodes.append({"id": f"modality_{modality}", "type": f"{modality}_component", "text": f"{modality} component residues/atoms {count}"})
    edges = [
        {"source": "structure", "target": "coordinate_target", "type": "has_coordinate_target"},
    ]
    for node in nodes[2:]:
        edges.append({"source": "structure", "target": node["id"], "type": "has_component"})
    for idx, pair in enumerate(contact_summary(parsed["structure_coordinates"])):
        edges.append(
            {
                "source": "coordinate_target",
                "target": "coordinate_target",
                "type": "representative_atom_contact",
                "id": f"contact_{idx}",
                "distance": pair["distance"],
            }
        )
    return {
        "record_id": record_id,
        "nodes": nodes,
        "edges": edges,
        "source_path": source_path,
    }


def forest_for_record(record_id: str, parsed: dict[str, Any]) -> dict[str, Any]:
    nodes = [
        {"id": "root", "type": "multimodal_structure_root", "text": "Reason over real structure coordinates and component modalities."},
        {"id": "geometry", "type": "coordinate_geometry_branch", "text": "Use distances, residue/atom modality, and contact geometry."},
        {"id": "modality", "type": "modality_branch", "text": json.dumps(parsed["modality_counts"], sort_keys=True)},
        {"id": "design", "type": "design_branch", "text": "Expose constraints for conditional biomolecular design."},
    ]
    edges = [
        {"source": "root", "target": "geometry", "type": "branch"},
        {"source": "root", "target": "modality", "type": "branch"},
        {"source": "geometry", "target": "design", "type": "geometry_conditioned_design"},
        {"source": "modality", "target": "design", "type": "modality_conditioned_design"},
    ]
    return {"record_id": record_id, "nodes": nodes, "edges": edges}


class SplitShardWriter:
    def __init__(self, out_dir: Path, *, shard_size: int, prefix: str, resume: bool = False) -> None:
        self.out_dir = out_dir
        self.shard_size = int(shard_size)
        self.prefix = prefix
        self.buffers = {"train": [], "validation": [], "test": []}
        self.counts = {"train": 0, "validation": 0, "test": 0}
        self.shards = {"train": 0, "validation": 0, "test": 0}
        self.paths = {"train": [], "validation": [], "test": []}
        for split in self.buffers:
            (out_dir / split).mkdir(parents=True, exist_ok=True)
            if resume:
                existing = sorted((out_dir / split).glob(f"{prefix}_{split}_*.parquet"))
                self.paths[split] = [str(path) for path in existing]
                self.shards[split] = len(existing)

    def add(self, row: dict[str, Any], split: str) -> None:
        row = dict(row)
        row["split"] = split
        self.buffers[split].append(row)
        self.counts[split] += 1
        if len(self.buffers[split]) >= self.shard_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        if not self.buffers[split]:
            return
        path = self.out_dir / split / f"{self.prefix}_{split}_{self.shards[split]:05d}.parquet"
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(pa.Table.from_pylist(self.buffers[split]), tmp, compression="zstd")
        tmp.replace(path)
        self.paths[split].append(str(path))
        self.shards[split] += 1
        self.buffers[split] = []

    def close(self) -> dict[str, Any]:
        for split in list(self.buffers):
            self.flush(split)
        return {"counts": self.counts, "shards": self.shards, "paths": self.paths}


def iter_input_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for pattern in args.local_structure_glob:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    for list_path in args.local_structure_list:
        with list_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                value = raw.strip()
                if value:
                    paths.append(Path(value))
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def iter_pdb_ids(path: Path) -> Any:
    for raw in re.split(r"[\s,]+", path.read_text(encoding="utf-8", errors="replace")):
        pdb_id = raw.strip().lower()
        if pdb_id:
            yield pdb_id


def existing_structure_ids(out_dir: Path, prefix: str) -> set[str]:
    ids: set[str] = set()
    for split in ("train", "validation", "test"):
        for path in sorted((out_dir / split).glob(f"{prefix}_{split}_*.parquet")):
            try:
                table = pq.read_table(path, columns=["structure_id"])
            except Exception:
                continue
            for value in table.column("structure_id").to_pylist():
                if value:
                    ids.add(str(value).lower())
    return ids


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def process_structure_path(
    path: Path,
    *,
    writer: SplitShardWriter,
    max_atoms: int,
    include_ligands: bool,
    accepted: int,
    skipped: dict[str, int],
    modality_counts: dict[str, int],
    max_records: int,
) -> int:
    if max_records > 0 and accepted >= max_records:
        return accepted
    try:
        parsed = parse_structure(path, max_atoms=max_atoms, include_ligands=include_ligands)
    except Exception as exc:
        key = f"parse_error:{type(exc).__name__}"
        skipped[key] = skipped.get(key, 0) + 1
        return accepted
    if parsed is None:
        skipped["no_coordinate_rows"] = skipped.get("no_coordinate_rows", 0) + 1
        return accepted
    structure_id = path.stem
    record_id = f"pdb_modal_structure_{structure_id}_{stable_hash(str(path))}"
    graph = graph_for_record(record_id, structure_id, parsed, str(path))
    forest = forest_for_record(record_id, parsed)
    metadata = {
        "source": "PDB/mmCIF",
        "source_path": str(path),
        "structure_modality": parsed["structure_modality"],
        "modality_counts": parsed["modality_counts"],
        "coordinate_residue_count": parsed["coordinate_residue_count"],
        "source_residue_count": parsed["source_residue_count"],
        "coordinate_truncated": parsed["coordinate_truncated"],
    }
    row = {
        "record_id": record_id,
        "dataset": "toricblm_pdb_modal_structure_fot",
        "task_family": f"{parsed['structure_modality']}_coordinate_training",
        "structure_id": structure_id,
        "structure_modality": parsed["structure_modality"],
        "text": (
            f"{structure_id} {parsed['structure_modality']} real coordinate structure; "
            f"components {json.dumps(parsed['modality_counts'], sort_keys=True)}; "
            f"coordinate residues/atoms {parsed['coordinate_residue_count']}."
        ),
        "graph_json": json.dumps(graph, ensure_ascii=True, sort_keys=True),
        "forest_json": json.dumps(forest, ensure_ascii=True, sort_keys=True),
        "metadata_json": json.dumps(metadata, ensure_ascii=True, sort_keys=True),
        "structure_coordinates": parsed["structure_coordinates"],
        "coordinate_mask": parsed["coordinate_mask"],
        "residue_names": parsed["residue_names"],
        "residue_indices": parsed["residue_indices"],
        "chain_ids": parsed["chain_ids"],
        "atom_modalities": parsed["atom_modalities"],
        "b_factors": parsed["b_factors"],
        "structure_source": "PDB",
        "structure_file": str(path),
        "coordinate_residue_count": parsed["coordinate_residue_count"],
        "source_residue_count": parsed["source_residue_count"],
        "coordinate_truncated": parsed["coordinate_truncated"],
        "split_cluster": stable_hash(structure_id, 12),
    }
    split = split_for_id(structure_id)
    writer.add(row, split)
    accepted += 1
    modality_counts[parsed["structure_modality"]] = modality_counts.get(parsed["structure_modality"], 0) + 1
    print(f"accepted {accepted:07d}: {structure_id} modality={parsed['structure_modality']}", flush=True)
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-structure-glob", action="append", default=[])
    parser.add_argument("--local-structure-list", action="append", type=Path, default=[], help="text file with one local mmCIF/CIF path per line")
    parser.add_argument("--rcsb-ids", type=Path, default=None, help="optional file with PDB IDs to download as mmCIF")
    parser.add_argument("--rcsb-query-modality", action="append", default=[], help="query RCSB by modality: protein, rna, dna, protein_rna, protein_dna, nucleic_acid, complex, ligand")
    parser.add_argument("--rcsb-query-limit", type=int, default=256, help="maximum RCSB IDs to request per modality; <=0 means all available")
    parser.add_argument("--rcsb-query-page-size", type=int, default=1000, help="RCSB search pagination size")
    parser.add_argument("--out-dir", type=Path, default=Path("data/uniprot_fot/structures/pdb_modal"))
    parser.add_argument("--prefix", default="toricblm_pdb_modal_structure_fot")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-atoms", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--include-ligands", action="store_true")
    parser.add_argument("--download-timeout", type=int, default=60)
    parser.add_argument("--min-free-gb", type=float, default=0.0)
    parser.add_argument("--remove-downloaded-cache", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip PDB IDs already present in output shards and continue shard numbering")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = iter_input_paths(args)
    if not paths and not args.rcsb_ids and not args.rcsb_query_modality:
        raise SystemExit("No input structures found. Provide --local-structure-glob, --rcsb-ids, or --rcsb-query-modality.")
    seen_structures = existing_structure_ids(args.out_dir, args.prefix) if args.resume else set()
    writer = SplitShardWriter(args.out_dir, shard_size=args.shard_size, prefix=args.prefix, resume=args.resume)
    skipped: dict[str, int] = {"already_curated": len(seen_structures)} if seen_structures else {}
    accepted = 0
    modality_counts: dict[str, int] = {}
    for path in paths:
        if args.max_records > 0 and accepted >= args.max_records:
            break
        if path.stem.lower() in seen_structures:
            continue
        if args.min_free_gb > 0 and free_gb(args.out_dir) < args.min_free_gb:
            skipped["disk_guard_free_gb_below_minimum"] = skipped.get("disk_guard_free_gb_below_minimum", 0) + 1
            break
        accepted = process_structure_path(
            path,
            writer=writer,
            max_atoms=args.max_atoms,
            include_ligands=args.include_ligands,
            accepted=accepted,
            skipped=skipped,
            modality_counts=modality_counts,
            max_records=args.max_records,
        )
    cache = args.out_dir / "rcsb_tmp_mmcif"
    cache.mkdir(parents=True, exist_ok=True)
    id_sources: list[str] = []
    if args.rcsb_ids:
        id_sources.extend(iter_pdb_ids(args.rcsb_ids))
    if args.rcsb_query_modality:
        queried_ids = query_rcsb_ids(
            args.rcsb_query_modality,
            limit_per_modality=args.rcsb_query_limit,
            page_size=args.rcsb_query_page_size,
            timeout=args.download_timeout,
        )
        ids_path = args.out_dir / "rcsb_query_ids.json"
        ids_path.write_text(
            json.dumps({"modalities": args.rcsb_query_modality, "ids": queried_ids}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        id_sources.extend(queried_ids)
    seen_ids: set[str] = set()
    for pdb_id in id_sources:
        if args.max_records > 0 and len(seen_structures) >= args.max_records:
            break
        pdb_id = pdb_id.strip().lower()
        if not pdb_id or pdb_id in seen_ids:
            continue
        seen_ids.add(pdb_id)
        if pdb_id in seen_structures:
            continue
        if args.min_free_gb > 0 and free_gb(args.out_dir) < args.min_free_gb:
            skipped["disk_guard_free_gb_below_minimum"] = skipped.get("disk_guard_free_gb_below_minimum", 0) + 1
            break
        path = cache / f"{pdb_id}.cif"
        if not path.exists() and not download_rcsb_cif(pdb_id, path, timeout=args.download_timeout):
            skipped["download_unavailable"] = skipped.get("download_unavailable", 0) + 1
            continue
        accepted = process_structure_path(
            path,
            writer=writer,
            max_atoms=args.max_atoms,
            include_ligands=args.include_ligands,
            accepted=accepted,
            skipped=skipped,
            modality_counts=modality_counts,
            max_records=args.max_records,
        )
        seen_structures.add(pdb_id)
        if args.remove_downloaded_cache:
            path.unlink(missing_ok=True)
    report = writer.close()
    manifest = {
        "schema": "toricblm.pdb_modal_structure_fot.v1",
        "records": accepted,
        "input_count": len(paths),
        "split_report": report,
        "modality_counts": modality_counts,
        "skipped": skipped,
        "max_atoms": args.max_atoms,
        "include_ligands": bool(args.include_ligands),
        "rcsb_query_modalities": list(args.rcsb_query_modality),
        "rcsb_query_limit": int(args.rcsb_query_limit),
    }
    (args.out_dir / f"{args.prefix}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if args.remove_downloaded_cache:
        shutil.rmtree(args.out_dir / "rcsb_tmp_mmcif", ignore_errors=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
