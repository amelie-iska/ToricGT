#!/usr/bin/env python3
"""Stream AFDB/EBI proteome tar archives into coordinate-native FoT records.

This is the disk-safe path for large AlphaFold DB ingestion.  It downloads one
official EBI tar archive at a time, extracts real mmCIF members into a temporary
file, parses real coordinates, optionally emits a real Foldseek/3Di structure
sequence for ProTrek splitting, writes Parquet shards, and deletes the archive.

It deliberately does not expand millions of files onto disk and it does not
fabricate missing structure annotations.  If a tar member cannot be parsed, that
member is skipped with an explicit progress reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_afdb_structure_fot_dataset import (  # noqa: E402
    SplitShardWriter,
    enrich_structure_graph,
    extract_foldseek_3di_sequence,
    forest_for_structure,
    parse_cif_coordinates,
    safe_text,
    split_for_accession,
    stable_hash,
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


EBI_AFDB_BASE = "https://ftp.ebi.ac.uk/pub/databases/alphafold/v6"
THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024.0**3)


def accession_from_member(name: str) -> tuple[str, int] | None:
    match = re.search(r"AF-([A-Za-z0-9]+)-F\d+-model_v(\d+)\.cif(?:\.gz)?$", name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def read_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            accession = item.get("accession")
            if accession:
                out.add(str(accession))
    return out


def append_progress(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def list_ebi_tar_urls(*, index_url: str, max_archives: int) -> list[str]:
    with urllib.request.urlopen(index_url, timeout=60) as response:
        html = response.read().decode("utf-8", errors="replace")
    names = sorted(set(re.findall(r'href="([^"]+\.tar)"', html)))
    urls = [f"{index_url.rstrip('/')}/{name}" for name in names if not name.startswith("swissprot_pdb")]
    if max_archives > 0:
        urls = urls[:max_archives]
    return urls


def load_tar_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = []
    for list_path in args.tar_url_list:
        for raw in Path(list_path).read_text(encoding="utf-8", errors="replace").splitlines():
            raw = raw.strip()
            if raw and not raw.startswith("#"):
                urls.append(raw)
    urls.extend(args.tar_url)
    if args.use_ebi_index:
        urls.extend(list_ebi_tar_urls(index_url=args.ebi_index_url, max_archives=args.max_archives))
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        key = url.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    if not deduped:
        raise SystemExit("No AFDB tar URLs supplied. Use --use-ebi-index or --tar-url.")
    return deduped


def download(url: str, path: Path, *, timeout: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    aria2_bin = shutil.which("aria2c")
    env_aria2 = Path(sys.executable).resolve().parent / "aria2c"
    if aria2_bin is None and env_aria2.exists():
        aria2_bin = str(env_aria2)
    if aria2_bin:
        connections = os.environ.get("AFDB_ARIA2_CONNECTIONS", "8")
        split = os.environ.get("AFDB_ARIA2_SPLIT", connections)
        cmd = [
            aria2_bin,
            "--continue=true",
            f"--max-connection-per-server={connections}",
            f"--split={split}",
            "--min-split-size=16M",
            "--file-allocation=none",
            "--max-tries=8",
            "--retry-wait=8",
            "--connect-timeout=30",
            "--timeout=120",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            f"--dir={tmp.parent}",
            f"--out={tmp.name}",
            url,
        ]
        subprocess.run(cmd, check=True, timeout=max(60, timeout))
        tmp.replace(path)
        return
    cmd = ["curl", "-L", "--fail", "--retry", "4", "--retry-delay", "5", "--connect-timeout", "30"]
    if tmp.exists() and tmp.stat().st_size > 0:
        cmd.extend(["-C", "-"])
    cmd.extend(["--max-time", str(timeout), "-o", str(tmp), url])
    subprocess.run(cmd, check=True)
    tmp.replace(path)


def sequence_from_coords(coords: dict[str, Any]) -> str:
    letters = []
    for name in coords.get("residue_names") or []:
        letters.append(THREE_TO_ONE.get(str(name).upper(), "X"))
    return "".join(letters)


def graph_record(
    *,
    accession: str,
    version: int,
    archive_url: str,
    archive_name: str,
    member_name: str,
    cif_path: Path,
    coords: dict[str, Any],
    convextok_tokens: dict[bytes, int],
    convextok_max_bytes: int,
    foldseek_3di_sequence: str,
    foldseek_3di_status: str,
    keep_mmcif_cache: bool,
) -> dict[str, Any]:
    sequence = sequence_from_coords(coords)
    entry_name = accession
    protein_name = f"AFDB v{version} proteome-tar protein {accession}"
    function_text = (
        f"AlphaFold DB v{version} EBI proteome archive member. "
        "No UniProt functional annotation was bundled in this tar member; train structure coordinates, "
        "confidence, sequence, and source provenance, and merge richer UniProt annotations when available."
    )
    prediction = {
        "modelEntityId": f"AF-{accession}-F1",
        "latestVersion": version,
        "cifUrl": f"{archive_url}#{member_name}",
        "bcifUrl": "",
        "archive_url": archive_url,
        "archive_member": member_name,
    }
    record_id = f"afdb_ebi_tar_structure_fot_{accession}_{stable_hash(member_name)}"
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
    graph["source"] = {
        "name": "AlphaFold DB v6 EBI proteome tar",
        "contains_imported_reasoning_trace": False,
        "archive_url": archive_url,
        "archive_name": archive_name,
        "archive_member": member_name,
    }
    graph["split_cluster"]["leakage_resistant_basis"] = (
        "accession hash; ProTrek trimodal split uses saved real Foldseek/3Di sequence when enabled"
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
        f"AFDB EBI tar structure record {accession}. "
        f"Archive: {archive_name}. Member: {member_name}. "
        f"Coordinate residues {coords['coordinate_residue_count']} of source residues {coords['source_residue_count']}; "
        f"mean pLDDT {coords['mean_plddt']:.2f}. "
        "Train on real CA/backbone coordinates for flow matching, contacts, distograms, and graph/FoT structure reasoning."
    )
    convextok_view = convextok_dag(text + "\n" + sequence[:512], convextok_tokens, max_bytes=int(convextok_max_bytes))
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
            "graphcg_axes": ["sequence", "coordinate_geometry", "confidence", "design_constraints"],
            "tropical_toric": graph["metadata"]["tokengt_tropical_toric_fields"],
        },
    }
    enrichment_status = {
        "schema": "toricblm.structure_enrichment_status.v1",
        "coordinate_source": "AFDB_EBI_TAR",
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
                "archive_member": member_name,
            }
        ).encode("utf-8")
    ).hexdigest()
    return {
        "record_id": record_id,
        "dataset": "toricblm_afdb_ebi_tar_structure_fot",
        "source_file": archive_url,
        "source_row_index": 0,
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
        "structure_source": "AFDB_EBI_TAR",
        "structure_file": str(cif_path) if keep_mmcif_cache else "",
        "structure_file_cached": bool(keep_mmcif_cache),
        "structure_cif_url": f"{archive_url}#{member_name}",
        "foldseek_3di_sequence": foldseek_3di_sequence,
        "foldseek_3di_available": bool(foldseek_3di_sequence),
        "foldseek_3di_status": foldseek_3di_status,
        "mean_plddt": coords["mean_plddt"],
        "residue_count": coords["coordinate_residue_count"],
        "coordinate_residue_count": coords["coordinate_residue_count"],
        "source_residue_count": coords["source_residue_count"],
        "coordinate_truncated": coords["coordinate_truncated"],
        "split": split,
        "split_cluster": json.dumps(graph["split_cluster"], ensure_ascii=True, sort_keys=True),
    }


def process_tar(
    *,
    tar_path: Path,
    archive_url: str,
    writer: SplitShardWriter,
    processed: set[str],
    progress_path: Path,
    convextok_tokens: dict[bytes, int],
    args: argparse.Namespace,
    accepted: int,
) -> int:
    with tarfile.open(tar_path, mode="r:*") as archive:
        for member in archive:
            if args.max_records > 0 and accepted >= args.max_records:
                break
            parsed = accession_from_member(member.name)
            if parsed is None:
                continue
            accession, version = parsed
            if accession in processed:
                continue
            if free_gb(args.out_dir) < args.min_free_gb:
                append_progress(progress_path, {"status": "stopped", "reason": "disk_guard", "accepted": accepted})
                return accepted
            handle = archive.extractfile(member)
            if handle is None:
                append_progress(progress_path, {"accession": accession, "status": "skipped", "reason": "member_extractfile_none"})
                continue
            suffix = ".cif.gz" if member.name.endswith(".gz") else ".cif"
            with tempfile.NamedTemporaryFile(dir=args.tmp_dir, suffix=suffix, delete=False) as tmp:
                shutil.copyfileobj(handle, tmp)
                tmp_path = Path(tmp.name)
            if suffix.endswith(".gz"):
                import gzip

                cif_path = tmp_path.with_suffix("")
                with gzip.open(tmp_path, "rb") as src, cif_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                tmp_path.unlink(missing_ok=True)
            else:
                cif_path = tmp_path
            try:
                coords = parse_cif_coordinates(cif_path, max_residues=args.max_residues)
            except Exception as exc:  # noqa: BLE001
                append_progress(progress_path, {"accession": accession, "status": "skipped", "reason": f"parse_error:{type(exc).__name__}"})
                cif_path.unlink(missing_ok=True)
                continue
            if coords is None:
                append_progress(progress_path, {"accession": accession, "status": "skipped", "reason": "too_few_coordinates"})
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
                except Exception as exc:  # noqa: BLE001
                    append_progress(progress_path, {"accession": accession, "status": "skipped", "reason": f"foldseek_3di_error:{type(exc).__name__}: {exc}"})
                    cif_path.unlink(missing_ok=True)
                    continue
            record = graph_record(
                accession=accession,
                version=version,
                archive_url=archive_url,
                archive_name=tar_path.name,
                member_name=member.name,
                cif_path=cif_path,
                coords=coords,
                convextok_tokens=convextok_tokens,
                convextok_max_bytes=args.convextok_max_bytes,
                foldseek_3di_sequence=foldseek_3di_sequence,
                foldseek_3di_status=foldseek_3di_status,
                keep_mmcif_cache=args.keep_mmcif_cache,
            )
            writer.add(record)
            processed.add(accession)
            accepted += 1
            append_progress(
                progress_path,
                {
                    "accession": accession,
                    "status": "accepted",
                    "source": "ebi_tar",
                    "archive": tar_path.name,
                    "member": member.name,
                    "split": record["split"],
                    "coordinate_residue_count": coords["coordinate_residue_count"],
                    "source_residue_count": coords["source_residue_count"],
                    "mean_plddt": round(float(coords["mean_plddt"]), 4),
                    "foldseek_3di_available": bool(foldseek_3di_sequence),
                    "foldseek_3di_status": foldseek_3di_status,
                },
            )
            if not args.keep_mmcif_cache:
                cif_path.unlink(missing_ok=True)
            print(
                f"accepted {accepted:07d}: {accession} archive={tar_path.name} "
                f"coord_residues={coords['coordinate_residue_count']} mean_plddt={coords['mean_plddt']:.2f}",
                flush=True,
            )
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/uniprot_fot/structures/afdb_ebi_tar_v6"))
    parser.add_argument("--tar-url", action="append", default=[])
    parser.add_argument("--tar-url-list", action="append", default=[])
    parser.add_argument("--use-ebi-index", action="store_true")
    parser.add_argument("--ebi-index-url", default=EBI_AFDB_BASE)
    parser.add_argument("--max-archives", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-residues", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--prefix", default="toricblm_afdb_ebi_tar_structure_fot")
    parser.add_argument("--download-timeout", type=int, default=7200)
    parser.add_argument("--min-free-gb", type=float, default=100.0)
    parser.add_argument("--convextok-tokenizer", type=Path, default=DEFAULT_CONVEXTOK)
    parser.add_argument("--convextok-max-bytes", type=int, default=1024)
    parser.add_argument("--emit-foldseek-3di", action="store_true")
    parser.add_argument("--require-foldseek-3di", action="store_true")
    parser.add_argument("--protrek-root", type=Path, default=Path("external/ProTrek"))
    parser.add_argument("--foldseek-bin", type=Path, default=Path("external/ProTrek/bin/foldseek"))
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--keep-mmcif-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir = args.out_dir / "_tmp"
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = args.out_dir / "_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / "build_progress.jsonl"
    processed = read_processed(progress_path) if args.resume else set()
    convextok_tokens = index_convextok_tokens(load_convextok_tokens(args.convextok_tokenizer, max_token_bytes=96))
    writer = SplitShardWriter(args.out_dir, shard_size=args.shard_size, prefix=args.prefix)
    urls = load_tar_urls(args)
    accepted = 0
    processed_archives = []
    stopped_reason = None
    try:
        for archive_index, url in enumerate(urls):
            if args.max_records > 0 and accepted >= args.max_records:
                break
            if free_gb(args.out_dir) < args.min_free_gb:
                stopped_reason = "disk_guard_before_archive"
                break
            name = url.rstrip("/").rsplit("/", 1)[-1]
            tar_path = archive_dir / name
            if not tar_path.exists():
                print(f"download_archive {archive_index + 1}/{len(urls)} {name} url={url}", flush=True)
                download(url, tar_path, timeout=args.download_timeout)
            print(f"process_archive {archive_index + 1}/{len(urls)} {name}", flush=True)
            accepted = process_tar(
                tar_path=tar_path,
                archive_url=url,
                writer=writer,
                processed=processed,
                progress_path=progress_path,
                convextok_tokens=convextok_tokens,
                args=args,
                accepted=accepted,
            )
            processed_archives.append(url)
            append_progress(progress_path, {"status": "archive_done", "archive": name, "accepted_total": accepted})
            if not args.keep_archives:
                tar_path.unlink(missing_ok=True)
    finally:
        if not args.keep_mmcif_cache:
            shutil.rmtree(args.tmp_dir, ignore_errors=True)
    split_report = writer.close()
    manifest = {
        "schema": "toricblm.afdb_ebi_tar_structure_manifest.v1",
        "records": accepted,
        "coordinate_bearing_records": accepted,
        "coordinate_source": "AlphaFold DB EBI v6 proteome tar archives",
        "processed_archives": processed_archives,
        "archive_count": len(processed_archives),
        "max_records": int(args.max_records),
        "max_residues": int(args.max_residues),
        "emit_foldseek_3di": bool(args.emit_foldseek_3di or args.require_foldseek_3di),
        "require_foldseek_3di": bool(args.require_foldseek_3di),
        "split_report": split_report,
        "progress_path": str(progress_path),
        "stopped_reason": stopped_reason,
        "free_gb_after": round(free_gb(args.out_dir), 3),
    }
    manifest_path = args.out_dir / "toricblm_afdb_ebi_tar_structure_fot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
