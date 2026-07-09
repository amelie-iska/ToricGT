#!/usr/bin/env python3
"""Build AFDB v4 coordinate graph/FoT records from Google Cloud Storage.

Input is a JSONL plan produced by ``plan_afdb_gcs_diverse_accessions.py``.
Each plan row carries UniProt accession, sequence, function text, and selection
metadata.  This script downloads only those selected AFDB mmCIF files from the
public DeepMind GCS bucket, parses real coordinates, optionally emits real
Foldseek/3Di strings, writes Parquet shards, and deletes raw mmCIF cache files
after each batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
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


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024.0**3)


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
                out.add(str(accession).upper())
    return out


def append_progress(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def iter_plan(path: Path) -> Any:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            item["_plan_line_no"] = line_no
            yield item


def gcs_uri(bucket: str, accession: str, version: int) -> str:
    return f"{bucket.rstrip('/')}/AF-{accession}-F1-model_v{int(version)}.cif"


def local_cif(cache_dir: Path, accession: str, version: int) -> Path:
    return cache_dir / f"AF-{accession}-F1-model_v{int(version)}.cif"


def copy_batch_with_gsutil(batch: list[dict[str, Any]], *, bucket: str, version: int, cache_dir: Path, timeout: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    uris = [gcs_uri(bucket, str(item["accession"]), version) for item in batch]
    gsutil = shutil.which("gsutil") or "/snap/bin/gsutil"
    if not Path(gsutil).exists():
        raise RuntimeError("gsutil not found; install google-cloud-cli and authenticate before AFDB GCS ingestion")
    cmd = [gsutil, "-m", "cp", "-c", "-I", str(cache_dir)]
    proc = subprocess.run(
        cmd,
        input="\n".join(uris) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(60, int(timeout)),
    )
    if proc.returncode == 0:
        return
    copied = sum(1 for item in batch if local_cif(cache_dir, str(item["accession"]), version).exists())
    if copied > 0:
        return
    fatal_markers = (
        "AccessDeniedException",
        "Unauthorized",
        "status=401",
        "status=403",
        "ServiceException: 401",
        "ServiceException: 403",
        "BucketNotFoundException",
    )
    stderr = proc.stderr or ""
    if any(marker in stderr for marker in fatal_markers):
        raise RuntimeError(f"gsutil batch copy failed before any object was copied: {stderr[-1200:]}")
    # If none of this batch exists in AFDB, downstream missing-file checks will
    # record row-level provenance without forcing a slow serial retry path.


def build_record(
    *,
    plan: dict[str, Any],
    version: int,
    bucket: str,
    cif_path: Path,
    coords: dict[str, Any],
    convextok_tokens: dict[bytes, int],
    convextok_max_bytes: int,
    foldseek_3di_sequence: str,
    foldseek_3di_status: str,
    keep_mmcif_cache: bool,
) -> dict[str, Any]:
    accession = str(plan["accession"]).upper()
    entry_name = safe_text(plan.get("entry_name") or accession, 256)
    protein_name = safe_text(plan.get("protein_name") or "protein", 1024)
    sequence = safe_text(plan.get("sequence") or "", 100000)
    function_text = safe_text(plan.get("function_text") or "", 2400)
    source_uri = gcs_uri(bucket, accession, version)
    prediction = {
        "modelEntityId": f"AF-{accession}-F1",
        "latestVersion": int(version),
        "cifUrl": source_uri,
        "bcifUrl": "",
        "gcs_uri": source_uri,
    }
    record_id = f"afdb_gcs_structure_fot_{accession}_{stable_hash(source_uri)}"
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
        "name": "AlphaFold DB v4 public Google Cloud Storage",
        "contains_imported_reasoning_trace": False,
        "gcs_uri": source_uri,
        "selection": plan.get("selection", {}),
    }
    graph["split_cluster"]["leakage_resistant_basis"] = (
        "accession hash until strict ProTrek trimodal full-row split rewrites structure rows"
    )
    forest = forest_for_structure(record_id, coords)
    thought_forest = thought_forest_for_structure(
        record_id=record_id,
        accession=accession,
        graph=graph,
        coords=coords,
        foldseek_3di_status=foldseek_3di_status,
    )
    selection = plan.get("selection") or {}
    text = (
        f"AFDB GCS structure-function record {accession}. "
        f"Protein: {protein_name}. Function: {safe_text(function_text, 1600)}. "
        f"Selection tier: {selection.get('tier', 'unknown')}; "
        f"enzyme evidence score: {(selection.get('enzyme_profile') or {}).get('enzyme_evidence_score', 'n/a')}. "
        f"AFDB v{version}; coordinate residues {coords['coordinate_residue_count']} of source residues {coords['source_residue_count']}; "
        f"mean pLDDT {coords['mean_plddt']:.2f}. "
        "Train on real CA/backbone coordinates for graph-in/graph-out structure reasoning, flow matching, contacts, and distograms."
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
            "graphcg_axes": ["sequence", "function", "coordinate_geometry", "confidence", "design_constraints", "enzyme_activity"],
            "tropical_toric": graph["metadata"]["tokengt_tropical_toric_fields"],
        },
    }
    enrichment_status = {
        "schema": "toricblm.structure_enrichment_status.v1",
        "coordinate_source": "AFDB_GCS_V4",
        "coordinate_available": True,
        "foldseek_3di_status": foldseek_3di_status,
        "mean_plddt": round(float(coords["mean_plddt"]), 4),
        "coordinate_truncated": bool(coords["coordinate_truncated"]),
        "selection": selection,
    }
    leakage_signature = {
        "schema": "toricblm.structure_leakage_signature.v1",
        "split": split,
        "cluster_id": graph["split_cluster"]["cluster_id"],
        "basis": graph["split_cluster"]["leakage_resistant_basis"],
        "accession_hash": stable_hash(accession, 32),
        "protrek_required_for_final_structure_split": True,
    }
    content_hash = hashlib.sha256(
        stable_json(
            {
                "graph": graph,
                "forest": forest,
                "thought_forest": thought_forest,
                "convextok": convextok_view,
                "accession": accession,
                "selection": selection,
            }
        ).encode("utf-8")
    ).hexdigest()
    return {
        "record_id": record_id,
        "dataset": "toricblm_afdb_gcs_structure_fot",
        "source_file": source_uri,
        "source_row_index": int(plan.get("_plan_line_no") or plan.get("source_row_index") or 0),
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
        "structure_source": "AFDB_GCS_V4",
        "structure_file": str(cif_path) if keep_mmcif_cache else "",
        "structure_file_cached": bool(keep_mmcif_cache),
        "structure_cif_url": source_uri,
        "foldseek_3di_sequence": foldseek_3di_sequence,
        "foldseek_3di_available": bool(foldseek_3di_sequence),
        "foldseek_3di_status": foldseek_3di_status,
        "mean_plddt": coords["mean_plddt"],
        "residue_count": coords["coordinate_residue_count"],
        "coordinate_residue_count": coords["coordinate_residue_count"],
        "source_residue_count": coords["source_residue_count"],
        "coordinate_truncated": coords["coordinate_truncated"],
        "selection_json": json.dumps(selection, ensure_ascii=True, sort_keys=True),
        "enzyme_tier": str(selection.get("tier") or "unknown"),
        "enzyme_evidence_score": float((selection.get("enzyme_profile") or {}).get("enzyme_evidence_score") or 0.0),
        "split": split,
        "split_cluster": json.dumps(graph["split_cluster"], ensure_ascii=True, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/uniprot_fot/structures/afdb_gcs_v4"))
    parser.add_argument("--bucket", default="gs://public-datasets-deepmind-alphafold-v4")
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-residues", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--prefix", default="toricblm_afdb_gcs_structure_fot")
    parser.add_argument("--min-free-gb", type=float, default=100.0)
    parser.add_argument("--convextok-tokenizer", type=Path, default=DEFAULT_CONVEXTOK)
    parser.add_argument("--convextok-max-bytes", type=int, default=1024)
    parser.add_argument("--gsutil-timeout", type=int, default=1800)
    parser.add_argument("--emit-foldseek-3di", action="store_true")
    parser.add_argument("--require-foldseek-3di", action="store_true")
    parser.add_argument("--protrek-root", type=Path, default=Path("external/ProTrek"))
    parser.add_argument("--foldseek-bin", type=Path, default=Path("external/ProTrek/bin/foldseek"))
    parser.add_argument("--keep-mmcif-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "_gcs_mmcif_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / "build_progress.jsonl"
    processed = read_processed(progress_path) if args.resume else set()
    convextok_tokens = index_convextok_tokens(load_convextok_tokens(args.convextok_tokenizer, max_token_bytes=96))
    writer = SplitShardWriter(args.out_dir, shard_size=args.shard_size, prefix=args.prefix)
    accepted = 0
    scanned = 0
    skipped: dict[str, int] = {}
    batch: list[dict[str, Any]] = []

    def process_batch(items: list[dict[str, Any]]) -> None:
        nonlocal accepted
        if not items:
            return
        copy_batch_with_gsutil(items, bucket=args.bucket, version=args.version, cache_dir=cache_dir, timeout=args.gsutil_timeout)
        for plan in items:
            accession = str(plan["accession"]).upper()
            if args.max_records > 0 and accepted >= args.max_records:
                return
            cif_path = local_cif(cache_dir, accession, args.version)
            if not cif_path.exists():
                skipped["gcs_object_missing"] = skipped.get("gcs_object_missing", 0) + 1
                append_progress(progress_path, {"accession": accession, "status": "skipped", "reason": "gcs_object_missing"})
                continue
            try:
                coords = parse_cif_coordinates(cif_path, max_residues=args.max_residues)
            except Exception as exc:  # noqa: BLE001
                skipped[f"parse_error:{type(exc).__name__}"] = skipped.get(f"parse_error:{type(exc).__name__}", 0) + 1
                append_progress(progress_path, {"accession": accession, "status": "skipped", "reason": f"parse_error:{type(exc).__name__}"})
                cif_path.unlink(missing_ok=True)
                continue
            if coords is None:
                skipped["too_few_coordinates"] = skipped.get("too_few_coordinates", 0) + 1
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
                    skipped[f"foldseek_3di_error:{type(exc).__name__}"] = skipped.get(f"foldseek_3di_error:{type(exc).__name__}", 0) + 1
                    append_progress(progress_path, {"accession": accession, "status": "skipped", "reason": f"foldseek_3di_error:{type(exc).__name__}: {exc}"})
                    cif_path.unlink(missing_ok=True)
                    continue
            record = build_record(
                plan=plan,
                version=args.version,
                bucket=args.bucket,
                cif_path=cif_path,
                coords=coords,
                convextok_tokens=convextok_tokens,
                convextok_max_bytes=args.convextok_max_bytes,
                foldseek_3di_sequence=foldseek_3di_sequence,
                foldseek_3di_status=foldseek_3di_status,
                keep_mmcif_cache=bool(args.keep_mmcif_cache),
            )
            writer.add(record)
            accepted += 1
            append_progress(
                progress_path,
                {
                    "accession": accession,
                    "status": "accepted",
                    "source": "gcs_afdb_v4",
                    "split": record["split"],
                    "enzyme_tier": record["enzyme_tier"],
                    "enzyme_evidence_score": record["enzyme_evidence_score"],
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
                f"accepted {accepted:07d}: {accession} tier={record['enzyme_tier']} "
                f"coord_residues={coords['coordinate_residue_count']} mean_plddt={coords['mean_plddt']:.2f} "
                f"3di={foldseek_3di_status}",
                flush=True,
            )

    for plan in iter_plan(args.plan_jsonl):
        if args.max_records > 0 and accepted >= args.max_records:
            break
        scanned += 1
        accession = str(plan.get("accession") or "").upper()
        if not accession or accession in processed:
            continue
        if free_gb(args.out_dir) < float(args.min_free_gb):
            append_progress(progress_path, {"accession": "", "status": "stopped", "reason": "disk_guard", "accepted": accepted})
            break
        processed.add(accession)
        batch.append(plan)
        if len(batch) >= int(args.batch_size):
            process_batch(batch)
            batch = []
    process_batch(batch)
    split_report = writer.close()
    if cache_dir.exists() and not args.keep_mmcif_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)
    if accepted <= 0:
        raise SystemExit(f"No AFDB GCS coordinate records built. skipped={skipped}")
    manifest = {
        "schema": "toricblm.afdb_gcs_structure_manifest.v1",
        "records": accepted,
        "scanned_plan_rows": scanned,
        "coordinate_bearing_records": accepted,
        "coordinate_source": "AlphaFold DB v4 public Google Cloud Storage",
        "bucket": args.bucket,
        "version": int(args.version),
        "emit_foldseek_3di": bool(args.emit_foldseek_3di or args.require_foldseek_3di),
        "require_foldseek_3di": bool(args.require_foldseek_3di),
        "split_report": split_report,
        "skipped": skipped,
        "progress_path": str(progress_path),
        "free_gb_after": round(free_gb(args.out_dir), 3),
    }
    manifest_path = args.out_dir / "toricblm_afdb_gcs_structure_fot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
