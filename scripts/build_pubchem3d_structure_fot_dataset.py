#!/usr/bin/env python3
"""Build ToricBLM coordinate records from real PubChem3D SDF conformers.

This script only emits molecules that already contain 3D coordinates in an SDF
record.  It does not generate conformers or treat SELFIES/SMILES strings as
coordinate-native structure data.
"""

from __future__ import annotations

import argparse
import ftplib
import gzip
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
PUBCHEM3D_FTP_HOST = "ftp.ncbi.nlm.nih.gov"
PUBCHEM3D_FTP_DIR = "pubchem/Compound_3D/01_conf_per_cmpd/SDF"


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


def download_pubchem_sdf_batch(cids: list[str], path: Path, *, timeout: int = 120) -> bool:
    cids = [str(cid).strip() for cid in cids if str(cid).strip()]
    if not cids:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        PUBCHEM_SDF.format(cid=",".join(cids)),
        headers={"User-Agent": "ToricGT-PubChem3D-curator/1.0"},
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            tmp.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 403, 404, 405, 414}:
            tmp.unlink(missing_ok=True)
            return False
        raise
    tmp.replace(path)
    return True


def split_sdf_records(text: str) -> list[str]:
    return [record for record in text.split("$$$$") if record.strip()]


def read_sdf_text(path: Path) -> str:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def parse_sdf_record(record: str, *, max_atoms: int) -> dict[str, Any] | None:
    lines = record.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
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
        rows = self.buffers[split]
        if not rows:
            return
        path = self.out_dir / split / f"{self.prefix}_{split}_{self.shards[split]:05d}.parquet"
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(pa.Table.from_pylist(rows), tmp, compression="zstd")
        tmp.replace(path)
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
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def pubchem3d_ftp_filenames(host: str, directory: str) -> list[str]:
    with ftplib.FTP(host, timeout=120) as ftp:
        ftp.login()
        ftp.cwd(directory)
        names = ftp.nlst()
    return sorted({name for name in names if name.endswith(".sdf.gz")})


def download_pubchem3d_ftp_file(host: str, directory: str, filename: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with ftplib.FTP(host, timeout=240) as ftp:
        ftp.login()
        ftp.cwd(directory)
        with tmp.open("wb") as handle:
            ftp.retrbinary(f"RETR {filename}", handle.write)
    tmp.replace(path)


def ftp_file_index_from_name(filename: str) -> int:
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else -1


def ftp_file_range_from_name(filename: str) -> tuple[int, int] | None:
    matches = re.findall(r"(\d+)", filename)
    if len(matches) < 2:
        return None
    return int(matches[0]), int(matches[1])


def max_numeric_compound_id(compound_ids: set[str]) -> int:
    max_id = 0
    for value in compound_ids:
        if str(value).isdigit():
            max_id = max(max_id, int(value))
    return max_id


def iter_cids(path: Path) -> Any:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            cid = raw.strip().split()[0] if raw.strip() else ""
            if cid:
                yield cid


def batched_iterable(values: Any, batch_size: int) -> Any:
    batch: list[str] = []
    for value in values:
        batch.append(str(value))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def existing_compound_ids(out_dir: Path, prefix: str) -> set[str]:
    ids: set[str] = set()
    for split in ("train", "validation", "test"):
        for path in sorted((out_dir / split).glob(f"{prefix}_{split}_*.parquet")):
            try:
                table = pq.read_table(path, columns=["compound_id"])
            except Exception:
                continue
            for value in table.column("compound_id").to_pylist():
                if value:
                    ids.add(str(value))
    return ids


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def process_sdf_path(
    path: Path,
    *,
    writer: SplitShardWriter,
    max_atoms: int,
    accepted: int,
    skipped: dict[str, int],
    max_records: int,
    seen_compounds: set[str] | None = None,
) -> int:
    if max_records > 0 and accepted >= max_records:
        return accepted
    try:
        records = split_sdf_records(read_sdf_text(path))
    except Exception as exc:
        key = f"read_error:{type(exc).__name__}"
        skipped[key] = skipped.get(key, 0) + 1
        return accepted
    for raw_record in records:
        if max_records > 0 and accepted >= max_records:
            break
        parsed = parse_sdf_record(raw_record, max_atoms=max_atoms)
        if parsed is None:
            skipped["missing_or_non3d_coordinates"] = skipped.get("missing_or_non3d_coordinates", 0) + 1
            continue
        compound_id = parsed["compound_id"]
        if seen_compounds is not None and compound_id in seen_compounds:
            skipped["already_curated"] = skipped.get("already_curated", 0) + 1
            continue
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
        if seen_compounds is not None:
            seen_compounds.add(compound_id)
        accepted += 1
        print(f"accepted {accepted:07d}: CID={compound_id} atoms={parsed['coordinate_atom_count']}", flush=True)
    return accepted


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
    parser.add_argument("--cid-batch-size", type=int, default=64, help="number of CIDs per PubChem3D request; failed batches fall back to per-CID requests")
    parser.add_argument("--pubchem3d-ftp-bulk", action="store_true", help="download/process PubChem3D bulk SDF gzip shards from the public FTP mirror")
    parser.add_argument("--ftp-host", default=PUBCHEM3D_FTP_HOST)
    parser.add_argument("--ftp-dir", default=PUBCHEM3D_FTP_DIR)
    parser.add_argument("--ftp-max-files", type=int, default=0, help="maximum number of bulk SDF gzip files to process; 0 means all available")
    parser.add_argument("--ftp-start-after", default="", help="skip bulk files lexicographically <= this basename")
    parser.add_argument(
        "--ftp-resume-after-existing-cid",
        action="store_true",
        help=(
            "advanced: with --resume, skip bulk files whose CID range ends before the maximum "
            "already-curated numeric CID; only safe when existing rows were created by a contiguous "
            "FTP bulk pass, not by arbitrary CID streams"
        ),
    )
    parser.add_argument("--ftp-cache-dir", type=Path, default=None)
    parser.add_argument("--min-free-gb", type=float, default=0.0)
    parser.add_argument("--remove-downloaded-cache", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip compound IDs already present in output shards and continue shard numbering")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = input_sdf_paths(args)
    if not paths and args.cid_file is None and not args.pubchem3d_ftp_bulk:
        raise SystemExit("No SDF inputs found. Provide --sdf-glob, --cid-file, or --pubchem3d-ftp-bulk.")
    seen_compounds = existing_compound_ids(args.out_dir, args.prefix) if args.resume else set()
    existing_record_count = len(seen_compounds)
    writer = SplitShardWriter(args.out_dir, shard_size=args.shard_size, prefix=args.prefix, resume=args.resume)
    accepted = 0
    skipped: dict[str, int] = {"already_curated": len(seen_compounds)} if seen_compounds else {}
    if args.pubchem3d_ftp_bulk and (args.max_records <= 0 or existing_record_count + accepted < args.max_records):
        cache = args.ftp_cache_dir or (args.out_dir / "pubchem3d_ftp_sdf")
        cache.mkdir(parents=True, exist_ok=True)
        names = pubchem3d_ftp_filenames(args.ftp_host, args.ftp_dir)
        if args.ftp_start_after:
            names = [name for name in names if name > args.ftp_start_after]
        if args.resume and args.ftp_resume_after_existing_cid and seen_compounds:
            max_seen_cid = max_numeric_compound_id(seen_compounds)
            retained: list[str] = []
            skipped_files = 0
            for name in names:
                file_range = ftp_file_range_from_name(name)
                if file_range is not None and file_range[1] <= max_seen_cid:
                    skipped_files += 1
                    continue
                retained.append(name)
            names = retained
            print(
                f"pubchem3d_ftp_resume_range_skip max_existing_cid={max_seen_cid} "
                f"skipped_files={skipped_files}",
                flush=True,
            )
        names = sorted(names, key=lambda name: (ftp_file_index_from_name(name), name))
        if args.ftp_max_files > 0:
            names = names[: args.ftp_max_files]
        print(
            f"pubchem3d_ftp_bulk files={len(names)} host={args.ftp_host} dir={args.ftp_dir} "
            f"existing_total={existing_record_count}",
            flush=True,
        )
        for file_index, name in enumerate(names, start=1):
            if args.max_records > 0 and existing_record_count + accepted >= args.max_records:
                break
            if args.min_free_gb > 0 and free_gb(args.out_dir) < args.min_free_gb:
                skipped["disk_guard_free_gb_below_minimum"] = skipped.get("disk_guard_free_gb_below_minimum", 0) + 1
                break
            local_path = cache / name
            if not local_path.exists():
                print(f"pubchem3d_ftp_download file={file_index}/{len(names)} name={name}", flush=True)
                try:
                    download_pubchem3d_ftp_file(args.ftp_host, args.ftp_dir, name, local_path)
                except Exception as exc:
                    skipped[f"ftp_download_error:{type(exc).__name__}"] = skipped.get(f"ftp_download_error:{type(exc).__name__}", 0) + 1
                    local_path.unlink(missing_ok=True)
                    continue
            before = accepted
            accepted = process_sdf_path(
                local_path,
                writer=writer,
                max_atoms=args.max_atoms,
                accepted=accepted,
                skipped=skipped,
                max_records=max(0, args.max_records - existing_record_count) if args.max_records > 0 else 0,
                seen_compounds=seen_compounds,
            )
            print(
                f"pubchem3d_ftp_progress file={file_index}/{len(names)} name={name} "
                f"accepted_new={accepted} accepted_file={accepted-before} existing_total={existing_record_count}",
                flush=True,
            )
            if args.remove_downloaded_cache:
                local_path.unlink(missing_ok=True)
    for path in paths:
        if args.max_records > 0 and accepted >= args.max_records:
            break
        if args.min_free_gb > 0 and free_gb(args.out_dir) < args.min_free_gb:
            skipped["disk_guard_free_gb_below_minimum"] = skipped.get("disk_guard_free_gb_below_minimum", 0) + 1
            break
        accepted = process_sdf_path(
            path,
            writer=writer,
            max_atoms=args.max_atoms,
            accepted=accepted,
            skipped=skipped,
            max_records=max(0, args.max_records - existing_record_count) if args.max_records > 0 else 0,
            seen_compounds=seen_compounds,
        )
    if args.cid_file is not None and (args.max_records <= 0 or accepted < args.max_records):
        cache = args.out_dir / "pubchem3d_tmp_sdf"
        cache.mkdir(parents=True, exist_ok=True)
        scanned_cids = 0
        pending_cids = (cid for cid in iter_cids(args.cid_file) if cid not in seen_compounds)
        for batch in batched_iterable(pending_cids, max(1, int(args.cid_batch_size))):
            scanned_cids += len(batch)
            if scanned_cids == len(batch) or scanned_cids % max(1000, int(args.cid_batch_size)) < len(batch):
                print(
                    f"pubchem3d_stream scanned_cids={scanned_cids} accepted_new={accepted} existing_total={len(seen_compounds)}",
                    flush=True,
                )
            if args.max_records > 0 and existing_record_count + accepted >= args.max_records:
                break
            if args.min_free_gb > 0 and free_gb(args.out_dir) < args.min_free_gb:
                skipped["disk_guard_free_gb_below_minimum"] = skipped.get("disk_guard_free_gb_below_minimum", 0) + 1
                break
            if args.max_records > 0:
                batch = batch[: max(0, args.max_records - existing_record_count - accepted)]
            path = cache / f"CID_batch_{batch[0]}_{batch[-1]}_{stable_hash(','.join(batch), 8)}.sdf"
            if not path.exists() and not download_pubchem_sdf_batch(batch, path, timeout=max(args.download_timeout, 120)):
                skipped["batch_download_unavailable"] = skipped.get("batch_download_unavailable", 0) + 1
                for cid in batch:
                    single_path = cache / f"CID_{cid}.sdf"
                    if not single_path.exists() and not download_pubchem_sdf(cid, single_path, timeout=args.download_timeout):
                        skipped["download_unavailable"] = skipped.get("download_unavailable", 0) + 1
                        continue
                    accepted = process_sdf_path(
                        single_path,
                        writer=writer,
                        max_atoms=args.max_atoms,
                        accepted=accepted,
                        skipped=skipped,
                        max_records=max(0, args.max_records - existing_record_count) if args.max_records > 0 else 0,
                        seen_compounds=seen_compounds,
                    )
                    if args.remove_downloaded_cache:
                        single_path.unlink(missing_ok=True)
                continue
            before = accepted
            accepted = process_sdf_path(
                path,
                writer=writer,
                max_atoms=args.max_atoms,
                accepted=accepted,
                skipped=skipped,
                max_records=max(0, args.max_records - existing_record_count) if args.max_records > 0 else 0,
                seen_compounds=seen_compounds,
            )
            if accepted == before:
                skipped["batch_no_accepted_3d_records"] = skipped.get("batch_no_accepted_3d_records", 0) + 1
            if args.remove_downloaded_cache:
                path.unlink(missing_ok=True)
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
