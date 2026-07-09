#!/usr/bin/env python3
"""Streaming ProTrek trimodal similarity split for large ToricBLM corpora.

This script is the strict large-corpus counterpart to
``split_with_protrek_trimodal.py``.  It uses actual ProTrek sequence, text, and
Foldseek/3Di structure embeddings.  Rows that do not have all three modalities
are reported as non-trimodal and are not used to fit or satisfy trimodal split
targets.

The output is a compact split-map Parquet by default, avoiding a second full
copy of millions of coordinate-bearing rows.  Set ``--write-full-rows`` only
when a duplicated split dataset is explicitly needed and disk capacity has been
checked.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from split_with_protrek_trimodal import (  # noqa: E402
    embed_or_empty,
    load_protrek,
    protrek_readiness,
    row_sequence,
    row_structure_3di_sequence,
    row_structure_path,
    row_text,
    split_for_cluster,
)


def stable_hash(value: Any, n: int = 16) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=True) if not isinstance(value, str) else value
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=16).hexdigest()[:n]


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for raw in sorted(glob.glob(pattern)):
            path = Path(raw)
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(path)
    if not paths:
        raise FileNotFoundError(f"no input parquet files matched {patterns}")
    return paths


def selected_columns(path: Path) -> list[str] | None:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    wanted = {
        "record_id",
        "entry_id",
        "uniprot_accession",
        "sequence",
        "annotation_text",
        "function",
        "protein_name",
        "text",
        "metadata_json",
        "foldseek_3di_sequence",
        "structure_3di_sequence",
        "protrek_structure_sequence",
        "structure_file",
        "pdb_file",
        "mmcif_file",
        "cif_path",
        "graph_json",
        "forest_json",
        "thought_forest_json",
        "convextok_dag_json",
        "training_views_json",
        "enrichment_status_json",
        "leakage_signature_json",
        "quality_flags_json",
        "structure_source",
        "structure_cif_url",
        "foldseek_3di_status",
        "mean_plddt",
        "residue_count",
        "coordinate_residue_count",
        "source_residue_count",
        "split",
        "split_cluster",
        "content_hash",
        "group_hash",
    }
    columns = [name for name in sorted(wanted) if name in names]
    return columns or None


def iter_rows(paths: list[Path], *, batch_rows: int) -> Iterable[tuple[Path, list[dict[str, Any]]]]:
    for path in paths:
        columns = selected_columns(path)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
            yield path, pa.Table.from_batches([batch]).to_pylist()


def row_id(row: dict[str, Any]) -> str:
    for key in ("record_id", "uniprot_accession", "entry_id", "content_hash"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "row_" + stable_hash(row, 24)


def trimodal_payload(
    row: dict[str, Any],
    *,
    structure_sequence_fn: Any,
    max_protein_length: int,
    max_structure_length: int,
    max_text_chars: int,
) -> tuple[str, str, str, dict[str, Any]]:
    seq = row_sequence(row, max_length=max_protein_length)
    text = row_text(row, max_chars=max_text_chars)
    struct_seq = row_structure_3di_sequence(row, max_length=max_structure_length)
    struct_source = "saved_foldseek_3di" if struct_seq else ""
    struct_path = row_structure_path(row)
    structure_error = ""
    if not struct_seq and struct_path:
        try:
            struct_seq = structure_sequence_fn(struct_path)[: max(8, int(max_structure_length))]
            struct_source = "foldseek_file" if struct_seq else ""
        except Exception as exc:  # noqa: BLE001 - per-row strict provenance.
            structure_error = f"{type(exc).__name__}: {exc}"
            struct_seq = ""
    modality = {
        "sequence_present": bool(seq),
        "text_present": bool(text),
        "structure_present": bool(struct_seq),
        "structure_sequence_source": struct_source,
        "structure_path": struct_path,
        "structure_error": structure_error,
    }
    return seq, text, struct_seq, modality


def feature_matrix(
    model: Any,
    seq_values: list[str],
    text_values: list[str],
    struct_values: list[str],
    *,
    batch_size: int,
) -> np.ndarray:
    seq_emb = embed_or_empty(model, "get_protein_repr", seq_values, batch_size=batch_size)
    text_emb = embed_or_empty(model, "get_text_repr", text_values, batch_size=batch_size)
    struct_emb = embed_or_empty(model, "get_structure_repr", struct_values, batch_size=batch_size)
    return np.concatenate([seq_emb, text_emb, struct_emb], axis=1).astype(np.float32)


def trimodal_batch(
    rows: list[dict[str, Any]],
    *,
    structure_sequence_fn: Any,
    max_protein_length: int,
    max_structure_length: int,
    max_text_chars: int,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str], list[dict[str, Any]], dict[str, int]]:
    trimodal_rows: list[dict[str, Any]] = []
    seq_values: list[str] = []
    text_values: list[str] = []
    struct_values: list[str] = []
    modalities: list[dict[str, Any]] = []
    missing_counts = {
        "missing_sequence": 0,
        "missing_text": 0,
        "missing_structure": 0,
        "nontrimodal": 0,
    }
    for row in rows:
        seq, text, struct_seq, modality = trimodal_payload(
            row,
            structure_sequence_fn=structure_sequence_fn,
            max_protein_length=max_protein_length,
            max_structure_length=max_structure_length,
            max_text_chars=max_text_chars,
        )
        if not seq:
            missing_counts["missing_sequence"] += 1
        if not text:
            missing_counts["missing_text"] += 1
        if not struct_seq:
            missing_counts["missing_structure"] += 1
        if not (seq and text and struct_seq):
            missing_counts["nontrimodal"] += 1
            continue
        trimodal_rows.append(row)
        seq_values.append(seq)
        text_values.append(text)
        struct_values.append(struct_seq)
        modalities.append(modality)
    return trimodal_rows, seq_values, text_values, struct_values, modalities, missing_counts


def count_trimodal(
    paths: list[Path],
    *,
    structure_sequence_fn: Any,
    batch_rows: int,
    max_records: int,
    max_protein_length: int,
    max_structure_length: int,
    max_text_chars: int,
) -> dict[str, int]:
    counts = {
        "rows_seen": 0,
        "trimodal_rows": 0,
        "missing_sequence": 0,
        "missing_text": 0,
        "missing_structure": 0,
        "nontrimodal": 0,
    }
    for _, rows in iter_rows(paths, batch_rows=batch_rows):
        if max_records > 0 and counts["trimodal_rows"] >= max_records:
            break
        _, seqs, texts, structs, _, missing = trimodal_batch(
            rows,
            structure_sequence_fn=structure_sequence_fn,
            max_protein_length=max_protein_length,
            max_structure_length=max_structure_length,
            max_text_chars=max_text_chars,
        )
        counts["rows_seen"] += len(rows)
        counts["trimodal_rows"] += min(len(seqs), max(0, max_records - counts["trimodal_rows"]) if max_records > 0 else len(seqs))
        for key, value in missing.items():
            counts[key] += int(value)
    return counts


class SplitMapWriter:
    def __init__(self, output_dir: Path, *, prefix: str, shard_size: int, write_full_rows: bool) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_size = int(shard_size)
        self.write_full_rows = bool(write_full_rows)
        self.map_rows: list[dict[str, Any]] = []
        self.full_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
        self.map_shards = 0
        self.full_shards = {"train": 0, "validation": 0, "test": 0}
        self.counts = {"train": 0, "validation": 0, "test": 0}
        (self.output_dir / "split_map").mkdir(parents=True, exist_ok=True)
        if self.write_full_rows:
            for split in self.full_rows:
                (self.output_dir / split).mkdir(parents=True, exist_ok=True)

    def add(self, split_map_row: dict[str, Any], source_row: dict[str, Any] | None = None) -> None:
        split = str(split_map_row["split"])
        self.counts[split] += 1
        self.map_rows.append(split_map_row)
        if self.write_full_rows and source_row is not None:
            row = dict(source_row)
            row["split"] = split
            row["split_cluster"] = split_map_row["cluster_id"]
            row["protrek_trimodal_signature_json"] = split_map_row["protrek_trimodal_signature_json"]
            self.full_rows[split].append(row)
            if len(self.full_rows[split]) >= self.shard_size:
                self.flush_full(split)
        if len(self.map_rows) >= self.shard_size:
            self.flush_map()

    def flush_map(self) -> None:
        if not self.map_rows:
            return
        path = self.output_dir / "split_map" / f"{self.prefix}_split_map_{self.map_shards:05d}.parquet"
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(pa.Table.from_pylist(self.map_rows), tmp, compression="zstd")
        tmp.replace(path)
        self.map_rows = []
        self.map_shards += 1

    def flush_full(self, split: str) -> None:
        rows = self.full_rows[split]
        if not rows:
            return
        path = self.output_dir / split / f"{self.prefix}_{split}_{self.full_shards[split]:05d}.parquet"
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(pa.Table.from_pylist(rows), tmp, compression="zstd")
        tmp.replace(path)
        self.full_rows[split] = []
        self.full_shards[split] += 1

    def close(self) -> dict[str, Any]:
        self.flush_map()
        if self.write_full_rows:
            for split in list(self.full_rows):
                self.flush_full(split)
        return {
            "split_counts": dict(self.counts),
            "split_map_shards": self.map_shards,
            "full_row_shards": dict(self.full_shards),
            "write_full_rows": self.write_full_rows,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prefix", default="toricblm_protrek_trimodal_streaming")
    parser.add_argument("--protrek-root", type=Path, default=Path("external/ProTrek"))
    parser.add_argument("--weights-dir", type=Path, default=Path("external/ProTrek/weights/ProTrek_35M"))
    parser.add_argument("--foldseek-bin", type=Path, default=Path("external/ProTrek/bin/foldseek"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--parquet-batch-rows", type=int, default=128)
    parser.add_argument("--max-clusters", type=int, default=8192)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--min-trimodal-records", type=int, default=1)
    parser.add_argument("--max-protein-length", type=int, default=1022)
    parser.add_argument("--max-structure-length", type=int, default=1022)
    parser.add_argument("--max-text-chars", type=int, default=1024)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--write-full-rows", action="store_true")
    parser.add_argument("--require-protrek", action="store_true")
    parser.add_argument("--require-trimodal", action="store_true")
    args = parser.parse_args()

    paths = expand_inputs(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    readiness = protrek_readiness(args.protrek_root, args.weights_dir, args.foldseek_bin)
    if not readiness["ready"]:
        report = {
            "schema": "toricblm.protrek_trimodal_streaming_split_report.v1",
            "ready": False,
            "protrek_inference_ran": False,
            "readiness": readiness,
            "reason": "ProTrek weights, configs, checkpoint, or Foldseek executable unavailable",
            "policy": "No trimodal split is emitted without actual ProTrek/Foldseek availability.",
        }
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.require_protrek or args.require_trimodal:
            raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    model, structure_sequence_fn = load_protrek(args.protrek_root, args.weights_dir, args.foldseek_bin, args.device)
    pre_counts = count_trimodal(
        paths,
        structure_sequence_fn=structure_sequence_fn,
        batch_rows=args.parquet_batch_rows,
        max_records=args.max_records,
        max_protein_length=args.max_protein_length,
        max_structure_length=args.max_structure_length,
        max_text_chars=args.max_text_chars,
    )
    if pre_counts["trimodal_rows"] < int(args.min_trimodal_records):
        report = {
            "schema": "toricblm.protrek_trimodal_streaming_split_report.v1",
            "ready": True,
            "protrek_inference_ran": False,
            "readiness": readiness,
            "pre_counts": pre_counts,
            "required_min_trimodal_records": int(args.min_trimodal_records),
            "reason": "insufficient rows with sequence, function text, and real Foldseek/3Di structure evidence",
            "policy": "Rows missing a modality are trainable, but cannot satisfy trimodal leakage splitting.",
        }
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.require_trimodal:
            raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    from sklearn.cluster import MiniBatchKMeans

    trimodal_total = int(pre_counts["trimodal_rows"])
    n_clusters = min(max(2, int(math.sqrt(trimodal_total))), int(args.max_clusters), trimodal_total)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=17,
        batch_size=max(n_clusters, int(args.batch_size)),
        n_init="auto",
    )
    fit_seen = 0
    initial_features: list[np.ndarray] = []
    for _, rows in iter_rows(paths, batch_rows=args.parquet_batch_rows):
        if args.max_records > 0 and fit_seen >= args.max_records:
            break
        trimodal_rows, seqs, texts, structs, _, _ = trimodal_batch(
            rows,
            structure_sequence_fn=structure_sequence_fn,
            max_protein_length=args.max_protein_length,
            max_structure_length=args.max_structure_length,
            max_text_chars=args.max_text_chars,
        )
        if not trimodal_rows:
            continue
        if args.max_records > 0:
            keep = max(0, min(len(trimodal_rows), int(args.max_records) - fit_seen))
            trimodal_rows, seqs, texts, structs = trimodal_rows[:keep], seqs[:keep], texts[:keep], structs[:keep]
        features = feature_matrix(model, seqs, texts, structs, batch_size=args.batch_size)
        fit_seen += len(trimodal_rows)
        if initial_features is not None:
            initial_features.append(features)
            stacked = np.concatenate(initial_features, axis=0)
            if len(stacked) < n_clusters:
                continue
            kmeans.partial_fit(stacked)
            initial_features = None  # type: ignore[assignment]
        else:
            kmeans.partial_fit(features)
    if initial_features:
        stacked = np.concatenate(initial_features, axis=0)
        kmeans.partial_fit(stacked)

    writer = SplitMapWriter(args.output_dir, prefix=args.prefix, shard_size=args.shard_size, write_full_rows=args.write_full_rows)
    assigned = 0
    modality_counts = {"sequence_present": 0, "text_present": 0, "structure_present": 0, "trimodal_present": 0}
    for source_path, rows in iter_rows(paths, batch_rows=args.parquet_batch_rows):
        if args.max_records > 0 and assigned >= args.max_records:
            break
        trimodal_rows, seqs, texts, structs, modalities, _ = trimodal_batch(
            rows,
            structure_sequence_fn=structure_sequence_fn,
            max_protein_length=args.max_protein_length,
            max_structure_length=args.max_structure_length,
            max_text_chars=args.max_text_chars,
        )
        if not trimodal_rows:
            continue
        if args.max_records > 0:
            keep = max(0, min(len(trimodal_rows), int(args.max_records) - assigned))
            trimodal_rows, seqs, texts, structs, modalities = trimodal_rows[:keep], seqs[:keep], texts[:keep], structs[:keep], modalities[:keep]
        features = feature_matrix(model, seqs, texts, structs, batch_size=args.batch_size)
        labels = kmeans.predict(features)
        for row, modality, label in zip(trimodal_rows, modalities, labels, strict=True):
            cluster_id = f"protrek_trimodal_cluster_{int(label):06d}"
            split = split_for_cluster(cluster_id)
            signature = {
                "schema": "toricblm.protrek_trimodal_signature.v2",
                "cluster_id": cluster_id,
                "split": split,
                "modality_presence": modality,
                "embedding_family": Path(args.weights_dir).name,
                "foldseek_bin": str(args.foldseek_bin),
                "protrek_root": str(args.protrek_root),
                "similarity_basis": "actual ProTrek sequence, text, and Foldseek/3Di structure embeddings",
                "source_parquet": str(source_path),
            }
            split_map_row = {
                "record_id": row_id(row),
                "entry_id": str(row.get("entry_id") or row.get("uniprot_accession") or row_id(row)),
                "uniprot_accession": str(row.get("uniprot_accession") or row.get("entry_id") or ""),
                "source_parquet": str(source_path),
                "cluster_id": cluster_id,
                "split": split,
                "structure_sequence_source": str(modality.get("structure_sequence_source") or ""),
                "protrek_trimodal_signature_json": json.dumps(signature, sort_keys=True, ensure_ascii=True),
            }
            writer.add(split_map_row, row)
            assigned += 1
            modality_counts["sequence_present"] += 1
            modality_counts["text_present"] += 1
            modality_counts["structure_present"] += 1
            modality_counts["trimodal_present"] += 1

    write_report = writer.close()
    report = {
        "schema": "toricblm.protrek_trimodal_streaming_split_report.v1",
        "ready": True,
        "protrek_inference_ran": True,
        "actual_trimodal_similarity_split": True,
        "readiness": readiness,
        "input_files": [str(path) for path in paths],
        "pre_counts": pre_counts,
        "records_assigned": assigned,
        "required_min_trimodal_records": int(args.min_trimodal_records),
        "cluster_count": n_clusters,
        "modality_counts": modality_counts,
        "output": write_report,
        "policy": (
            "Only rows with sequence, function text, and real Foldseek/3Di structure evidence are clustered. "
            "Rows missing structure but containing useful sequence/function evidence remain trainable through the ordinary "
            "protein graph/FoT streams; they are simply not counted as trimodal and do not satisfy the structure-split target."
        ),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.require_trimodal and assigned < int(args.min_trimodal_records):
        raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
