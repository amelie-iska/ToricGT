#!/usr/bin/env python3
"""Build ToricBLM coordinate records from real PubChem3D SDF conformers.

This script only emits molecules that already contain 3D coordinates in an SDF
record.  It does not generate conformers or treat SELFIES/SMILES strings as
coordinate-native structure data.
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


PUBCHEM_SDF = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d"


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.blake2b(value.encode("utf-8", errors="replace"), digest_size=16).hexdigest()[:length]


def split_for_id(value: str) -> str:
    bucket = int(stable_hash(value, length=8), 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def download_pubchem_sdf(cid: str, path: Path, *, timeout: int = 60) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        PUBCHEM_SDF.format(cid=cid),
        headers={"User-Agent": "ToricGT-PubChem3D-curator/1.0"},
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            tmp.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 403, 404, 405}:
            tmp.unlink(missing_ok=True)
            return False
        raise
    tmp.replace(path)
    return True


def split_sdf_records(text: str) -> list[str]:
    return [record for record in text.split("$$$$") if record.strip()]


def parse_sdf_record(record: str, *, max_atoms: int) -> dict[str, Any] | None:
    lines = record.splitlines()
    if len(lines) < 5:
        return None
    title = lines[0].strip() or "pubchem3d_molecule"
    counts_line = lines[3] if len(lines) > 3 else ""
    match = re.match(r"\s*(\d+)\s+(\d+)", counts_line)
    if not match:
        return None
    atom_count = int(match.group(1))
    if atom_count < 2:
        return None
    coords: list[list[float]] = []
    atoms: list[str] = []
    for line in lines[4 : 4 + atom_count]:
        parts = line.split()
        if len(parts) < 4:
            return None
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return None
        atom = re.sub(r"[^A-Za-z]", "", parts[3])[:2] or "C"
        if len(coords) < max_atoms:
            coords.append([x, y, z])
            atoms.append(atom)
    if len(coords) < 2:
        return None
    # Reject flat 2D records; PubChem3D should have nontrivial z spread.
    z_values = [xyz[2] for xyz in coords]
    if max(z_values) - min(z_values) < 1e-4:
        return None
    props: dict[str, str] = {}
    current_key: str | None = None
    current_value: list[str] = []
    for line in lines[4 + atom_count :]:
        prop_match = re.match(r">\s*<([^>]+)>", line)
        if prop_match:
            if current_key is not None:
                props[current_key] = "\n".join(current_value).strip()
            current_key = prop_match.group(1)
            current_value = []
            continue
        if current_key is not None:
            current_value.append(line)
    if current_key is not None:
        props[current_key] = "\n".join(current_value).strip()
    cid = props.get("PUBCHEM_COMPOUND_CID") or props.get("PUBCHEM_CID") or title
    return {
        "compound_id": str(cid).strip(),
        "title": title,
        "structure_coordinates": coords,
        "atom_symbols": atoms,
        "coordinate_mask": [True] * len(coords),
        "coordinate_atom_count": len(coords),
        "source_atom_count": atom_count,
        "coordinate_truncated": atom_count > len(coords),
        "properties": props,
    }


def graph_for_record(record_id: str, parsed: dict[str, Any], source_path: str) -> dict[str, Any]:
    nodes = [
        {"id": "molecule", "type": "small_molecule", "text": f"PubChem3D {parsed['compound_id']} {parsed['title']}"},
        {"id": "coordinate_target", "type": "small_molecule_3d_conformer", "text": "Real PubChem3D SDF atom coordinates."},
        {"id": "atom_composition", "type": "atom_symbol_multiset", "text": json.dumps(parsed["atom_symbols"][:128])},
    ]
    edges = [
        {"source": "molecule", "target": "coordinate_target", "type": "has_3d_conformer"},
        {"source": "molecule", "target": "atom_composition", "type": "has_atom_composition"},
    ]
    return {"record_id": record_id, "nodes": nodes, "edges": edges, "source_path": source_path}


class SplitShardWriter:
    def __init__(self, out_dir: Path, *, shard_size: int, prefix: str) -> None:
        self.out_dir = out_dir
        self.shard_size = int(shard_size)
        self.prefix = prefix
        self.buffers = {"train": [], "validation": [], "test": []}
        self.counts = {"train": 0, "validation": 0, "test": 0}
        self.shards = {"train": 0, "validation": 0, "test": 0}
        self.paths = {"train": [], "validation": [], "test": []}
        for split in self.buffers:
            (out_dir / split).mkdir(parents=True, exist_ok=True)

    def add(self, row: dict[str, Any], split: str) -> None:
        row = dict(row)
        row["split"] = split
        self.buffers[split].append(row)
        self.counts[split] += 1
        if len(self.buffers[split]) >= self.shard_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        path = self.out_dir / split / f"{self.prefix}_{split}_{self.shards[split]:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        self.paths[split].append(str(path))
        self.shards[split] += 1
        self.buffers[split] = []

    def close(self) -> dict[str, Any]:
        for split in list(self.buffers):
            self.flush(split)
        return {"counts": self.counts, "shards": self.shards, "paths": self.paths}


def input_sdf_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for pattern in args.sdf_glob:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    if args.cid_file is not None:
        cache = args.out_dir / "pubchem3d_tmp_sdf"
        cache.mkdir(parents=True, exist_ok=True)
        for raw in re.split(r"[\s,]+", args.cid_file.read_text(encoding="utf-8")):
            cid = raw.strip()
            if not cid:
                continue
            path = cache / f"CID_{cid}.sdf"
            if not path.exists() and not download_pubchem_sdf(cid, path, timeout=args.download_timeout):
                continue
            paths.append(path)
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf-glob", action="append", default=[])
    parser.add_argument("--cid-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/uniprot_fot/structures/pubchem3d"))
    parser.add_argument("--prefix", default="toricblm_pubchem3d_structure_fot")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-atoms", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--download-timeout", type=int, default=60)
    parser.add_argument("--remove-downloaded-cache", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = input_sdf_paths(args)
    if not paths:
        raise SystemExit("No SDF inputs found. Provide --sdf-glob or --cid-file.")
    writer = SplitShardWriter(args.out_dir, shard_size=args.shard_size, prefix=args.prefix)
    accepted = 0
    skipped: dict[str, int] = {}
    for path in paths:
        if args.max_records > 0 and accepted >= args.max_records:
            break
        try:
            records = split_sdf_records(path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            key = f"read_error:{type(exc).__name__}"
            skipped[key] = skipped.get(key, 0) + 1
            continue
        for raw_record in records:
            if args.max_records > 0 and accepted >= args.max_records:
                break
            parsed = parse_sdf_record(raw_record, max_atoms=args.max_atoms)
            if parsed is None:
                skipped["missing_or_non3d_coordinates"] = skipped.get("missing_or_non3d_coordinates", 0) + 1
                continue
            compound_id = parsed["compound_id"]
            record_id = f"pubchem3d_structure_{compound_id}_{stable_hash(str(path) + ':' + compound_id)}"
            graph = graph_for_record(record_id, parsed, str(path))
            metadata = {
                "source": "PubChem3D SDF",
                "source_path": str(path),
                "compound_id": compound_id,
                "coordinate_atom_count": parsed["coordinate_atom_count"],
                "source_atom_count": parsed["source_atom_count"],
                "coordinate_truncated": parsed["coordinate_truncated"],
                "sdf_properties": parsed["properties"],
            }
            row = {
                "record_id": record_id,
                "dataset": "toricblm_pubchem3d_structure_fot",
                "task_family": "small_molecule_3d_coordinate_training",
                "compound_id": compound_id,
                "text": f"PubChem3D compound {compound_id}; real SDF conformer with {parsed['coordinate_atom_count']} atom coordinates.",
                "graph_json": json.dumps(graph, ensure_ascii=True, sort_keys=True),
                "forest_json": json.dumps(
                    {
                        "record_id": record_id,
                        "nodes": [
                            {"id": "root", "type": "molecule_structure_root", "text": "Reason over real small-molecule 3D conformer coordinates."},
                            {"id": "geometry", "type": "conformer_geometry_branch", "text": "Use atom distances and composition."},
                            {"id": "design", "type": "ligand_design_branch", "text": "Expose later conditional ligand design constraints."},
                        ],
                        "edges": [
                            {"source": "root", "target": "geometry", "type": "branch"},
                            {"source": "geometry", "target": "design", "type": "geometry_conditioned_design"},
                        ],
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "metadata_json": json.dumps(metadata, ensure_ascii=True, sort_keys=True),
                "structure_coordinates": parsed["structure_coordinates"],
                "coordinate_mask": parsed["coordinate_mask"],
                "atom_symbols": parsed["atom_symbols"],
                "atom_modalities": ["ligand"] * parsed["coordinate_atom_count"],
                "residue_names": parsed["atom_symbols"],
                "residue_indices": list(range(1, parsed["coordinate_atom_count"] + 1)),
                "chain_ids": ["L"] * parsed["coordinate_atom_count"],
                "structure_source": "PubChem3D",
                "structure_file": str(path),
                "coordinate_residue_count": parsed["coordinate_atom_count"],
                "source_residue_count": parsed["source_atom_count"],
                "coordinate_truncated": parsed["coordinate_truncated"],
                "split_cluster": stable_hash(compound_id, 12),
            }
            writer.add(row, split_for_id(compound_id))
            accepted += 1
            print(f"accepted {accepted:07d}: CID={compound_id} atoms={parsed['coordinate_atom_count']}", flush=True)
    report = writer.close()
    manifest = {
        "schema": "toricblm.pubchem3d_structure_fot.v1",
        "records": accepted,
        "input_count": len(paths),
        "split_report": report,
        "skipped": skipped,
        "max_atoms": args.max_atoms,
        "coordinate_source": "real SDF conformer coordinates only",
    }
    (args.out_dir / f"{args.prefix}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if args.remove_downloaded_cache:
        shutil.rmtree(args.out_dir / "pubchem3d_tmp_sdf", ignore_errors=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
