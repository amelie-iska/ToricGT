#!/usr/bin/env python3
"""Split ToricBLM protein records with real ProTrek trimodal embeddings.

This script is intentionally strict.  It does not approximate ProTrek with
hashes or metadata.  If `--require-protrek` is set and the ProTrek weights or
Foldseek binary are missing, it exits nonzero and records the readiness failure.

Rows without a coordinate structure remain trainable in the main graph/FoT
stream, but they cannot be claimed as trimodally split.  For those rows the
report records which modalities were available; no structure embedding is
fabricated.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def stable_hash(value: Any, n: int = 16) -> str:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True) if not isinstance(value, str) else value
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=16).hexdigest()[:n]


def split_for_cluster(cluster_key: str) -> str:
    value = int(hashlib.blake2b(cluster_key.encode("utf-8"), digest_size=8).hexdigest(), 16) % 1000
    if value < 900:
        return "train"
    if value < 950:
        return "validation"
    return "test"


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in sorted(glob.glob(pattern)):
            path = Path(match)
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(path)
    if not paths:
        raise FileNotFoundError(f"no input parquet files matched {patterns}")
    return paths


def protrek_readiness(protrek_root: Path, weights_dir: Path, foldseek_bin: Path) -> dict[str, Any]:
    checkpoint_files = sorted(weights_dir.glob("*.pt")) if weights_dir.exists() else []
    protein_dirs = sorted(weights_dir.glob("esm2_*")) if weights_dir.exists() else []
    structure_dirs = sorted(weights_dir.glob("foldseek_*")) if weights_dir.exists() else []
    text_dirs = [weights_dir / "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"]
    ready = (
        protrek_root.exists()
        and weights_dir.exists()
        and bool(checkpoint_files)
        and bool(protein_dirs)
        and bool(structure_dirs)
        and all(path.exists() for path in text_dirs)
        and foldseek_bin.exists()
        and os.access(foldseek_bin, os.X_OK)
    )
    return {
        "ready": ready,
        "protrek_root": str(protrek_root),
        "weights_dir": str(weights_dir),
        "foldseek_bin": str(foldseek_bin),
        "checkpoint_files": [str(path) for path in checkpoint_files],
        "protein_config_dirs": [str(path) for path in protein_dirs],
        "structure_config_dirs": [str(path) for path in structure_dirs],
        "text_config_dirs": [str(path) for path in text_dirs if path.exists()],
        "missing": {
            "protrek_root": not protrek_root.exists(),
            "weights_dir": not weights_dir.exists(),
            "checkpoint_pt": not bool(checkpoint_files),
            "protein_config": not bool(protein_dirs),
            "structure_config": not bool(structure_dirs),
            "text_config": not all(path.exists() for path in text_dirs),
            "foldseek_executable": not (foldseek_bin.exists() and os.access(foldseek_bin, os.X_OK)),
        },
    }


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256):
            for row in pa.Table.from_batches([batch]).to_pylist():
                row = dict(row)
                row["_source_parquet"] = str(path)
                rows.append(row)
    return rows


def limit_rows(rows: list[dict[str, Any]], max_records: int) -> list[dict[str, Any]]:
    if int(max_records) <= 0:
        return rows
    return rows[: int(max_records)]


def row_sequence(row: dict[str, Any], *, max_length: int) -> str:
    seq = str(row.get("sequence") or "")
    seq = re.sub(r"[^A-Za-z]", "", seq).upper()
    if set(seq) <= set("ACDEFGHIKLMNPQRSTVWYXBZUOJ") and len(seq) >= 8:
        return seq[: max(8, int(max_length))]
    return ""


def row_text(row: dict[str, Any], *, max_chars: int) -> str:
    parts = [
        row.get("annotation_text"),
        row.get("function"),
        row.get("protein_name"),
        row.get("text"),
        row.get("metadata_json"),
    ]
    text = " ".join(str(part) for part in parts if part not in (None, ""))
    return re.sub(r"\s+", " ", text).strip()[: max(64, int(max_chars))]


def row_structure_path(row: dict[str, Any]) -> str:
    for key in ("structure_file", "pdb_file", "mmcif_file", "cif_path"):
        raw = row.get(key)
        if raw and Path(str(raw)).exists():
            return str(raw)
    metadata_raw = row.get("metadata_json")
    if metadata_raw:
        try:
            metadata = json.loads(metadata_raw)
        except Exception:
            metadata = {}
        for key in ("structure_file", "source_path"):
            raw = metadata.get(key)
            if raw and Path(str(raw)).exists():
                return str(raw)
    return ""


def row_structure_3di_sequence(row: dict[str, Any], *, max_length: int) -> str:
    """Return an exact saved Foldseek/3Di sequence when curation emitted one.

    This lets disk-safe AFDB curation delete raw mmCIF files while preserving
    the real structure modality needed by ProTrek.  The value must come from a
    Foldseek/3Di field; no structure string is inferred from coordinates or IDs.
    """
    for key in ("foldseek_3di_sequence", "structure_3di_sequence", "protrek_structure_sequence"):
        raw = row.get(key)
        if raw:
            seq = re.sub(r"\s+", "", str(raw)).lower()
            if len(seq) >= 8:
                return seq[: max(8, int(max_length))]
    metadata_raw = row.get("metadata_json")
    if metadata_raw:
        try:
            metadata = json.loads(metadata_raw)
        except Exception:
            metadata = {}
        for key in ("foldseek_3di_sequence", "structure_3di_sequence", "protrek_structure_sequence"):
            raw = metadata.get(key)
            if raw:
                seq = re.sub(r"\s+", "", str(raw)).lower()
                if len(seq) >= 8:
                    return seq[: max(8, int(max_length))]
    return ""


def load_protrek(protrek_root: Path, weights_dir: Path, foldseek_bin: Path, device: str) -> tuple[Any, Any]:
    sys.path.insert(0, str(protrek_root))
    from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel  # type: ignore
    from utils.foldseek_util import get_struc_seq  # type: ignore

    protein_config = sorted(weights_dir.glob("esm2_*"))[0]
    structure_config = sorted(weights_dir.glob("foldseek_*"))[0]
    text_config = weights_dir / "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    checkpoint = sorted(weights_dir.glob("*.pt"))[0]
    model = ProTrekTrimodalModel(
        protein_config=str(protein_config),
        text_config=str(text_config),
        structure_config=str(structure_config),
        load_protein_pretrained=False,
        load_text_pretrained=False,
        from_checkpoint=str(checkpoint),
    )
    model.eval().to(device)

    def structure_sequence(path: str) -> str:
        seqs = get_struc_seq(str(foldseek_bin), path, ["A"])
        if "A" not in seqs:
            first = next(iter(seqs.values()))
            return str(first[1]).lower()
        return str(seqs["A"][1]).lower()

    return model, structure_sequence


def batched(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def embed_or_empty(model: Any, method: str, values: list[str], *, batch_size: int) -> np.ndarray:
    if not values:
        return np.zeros((0, 1024), dtype=np.float32)
    import torch

    chunks = []
    with torch.no_grad():
        for batch in batched(values, batch_size):
            tensor = getattr(model, method)(batch, batch_size=batch_size, verbose=False)
            tensor = torch.nn.functional.normalize(tensor.detach().float().cpu(), dim=-1)
            chunks.append(tensor.numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def make_feature_matrix(
    rows: list[dict[str, Any]],
    *,
    protrek_root: Path,
    weights_dir: Path,
    foldseek_bin: Path,
    device: str,
    batch_size: int,
    max_protein_length: int,
    max_structure_length: int,
    max_text_chars: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    model, structure_sequence = load_protrek(protrek_root, weights_dir, foldseek_bin, device)
    seq_values: list[str] = []
    text_values: list[str] = []
    struct_values: list[str] = []
    modality_rows: list[dict[str, Any]] = []
    for row in rows:
        seq = row_sequence(row, max_length=max_protein_length)
        text = row_text(row, max_chars=max_text_chars)
        struct_path = row_structure_path(row)
        struct_seq = row_structure_3di_sequence(row, max_length=max_structure_length)
        struct_source = "saved_foldseek_3di" if struct_seq else ""
        if not struct_seq and struct_path:
            try:
                struct_seq = structure_sequence(struct_path)[: max(8, int(max_structure_length))]
                struct_source = "foldseek_file" if struct_seq else ""
            except Exception as exc:  # noqa: BLE001 - report per-row structure extraction failure.
                row["_protrek_structure_error"] = f"{type(exc).__name__}: {exc}"
                struct_seq = ""
        modality_rows.append(
            {
                "sequence_present": bool(seq),
                "text_present": bool(text),
                "structure_present": bool(struct_seq),
                "structure_path": struct_path,
                "structure_sequence_source": struct_source,
                "structure_error": row.get("_protrek_structure_error", ""),
            }
        )
        seq_values.append(seq or "X")
        text_values.append(text or "missing protein function annotation")
        struct_values.append(struct_seq or "x")
    seq_emb = embed_or_empty(model, "get_protein_repr", seq_values, batch_size=batch_size)
    text_emb = embed_or_empty(model, "get_text_repr", text_values, batch_size=batch_size)
    struct_emb = embed_or_empty(model, "get_structure_repr", struct_values, batch_size=batch_size)
    masks = np.array(
        [
            [
                float(item["sequence_present"]),
                float(item["text_present"]),
                float(item["structure_present"]),
            ]
            for item in modality_rows
        ],
        dtype=np.float32,
    )
    features = np.concatenate([seq_emb * masks[:, 0:1], text_emb * masks[:, 1:2], struct_emb * masks[:, 2:3], masks], axis=1)
    return features.astype(np.float32), modality_rows


def cluster_features(features: np.ndarray, rows: list[dict[str, Any]], *, max_clusters: int) -> list[str]:
    if len(rows) == 0:
        return []
    if len(rows) == 1:
        return ["protrek_cluster_0"]
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for ProTrek embedding clustering") from exc
    n_clusters = min(max(2, int(np.sqrt(len(rows)))), max_clusters, len(rows))
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=17, n_init="auto", batch_size=min(2048, max(64, len(rows))))
    labels = km.fit_predict(features)
    return [f"protrek_cluster_{int(label):06d}" for label in labels]


def write_split_rows(rows: list[dict[str, Any]], output_dir: Path, prefix: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        grouped[str(row["split"])].append(row)
    paths: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split, items in grouped.items():
        counts[split] = len(items)
        if not items:
            continue
        path = output_dir / f"{prefix}_{split}.parquet"
        pq.write_table(pa.Table.from_pylist(items), path, compression="zstd")
        paths[split] = str(path)
    return {"counts": counts, "paths": paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Input Parquet glob. May repeat.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prefix", default="toricblm_protrek_trimodal_split")
    parser.add_argument("--protrek-root", type=Path, default=Path("external/ProTrek"))
    parser.add_argument("--weights-dir", type=Path, default=Path("external/ProTrek/weights/ProTrek_35M"))
    parser.add_argument("--foldseek-bin", type=Path, default=Path("external/ProTrek/bin/foldseek"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-clusters", type=int, default=4096)
    parser.add_argument("--max-records", type=int, default=0, help="0 means use all input rows")
    parser.add_argument("--max-protein-length", type=int, default=1022)
    parser.add_argument("--max-structure-length", type=int, default=1022)
    parser.add_argument("--max-text-chars", type=int, default=1024)
    parser.add_argument("--require-protrek", action="store_true")
    args = parser.parse_args()

    readiness = protrek_readiness(args.protrek_root, args.weights_dir, args.foldseek_bin)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if not readiness["ready"]:
        report = {
            "schema": "toricblm.protrek_trimodal_split_report.v1",
            "protrek_inference_ran": False,
            "ready": False,
            "readiness": readiness,
            "output_written": False,
            "reason": "ProTrek weights or Foldseek executable unavailable",
            "policy": "do not claim ProTrek-derived split; graph/FoT rows remain trainable without structure",
        }
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.require_protrek:
            raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    input_paths = expand_inputs(args.input)
    rows = limit_rows(load_rows(input_paths), args.max_records)
    if not rows:
        raise SystemExit("No rows loaded from input files")
    try:
        features, modality_rows = make_feature_matrix(
            rows,
            protrek_root=args.protrek_root,
            weights_dir=args.weights_dir,
            foldseek_bin=args.foldseek_bin,
            device=args.device,
            batch_size=args.batch_size,
            max_protein_length=args.max_protein_length,
            max_structure_length=args.max_structure_length,
            max_text_chars=args.max_text_chars,
        )
    except Exception as exc:  # noqa: BLE001 - strict report beats traceback-only failure.
        report = {
            "schema": "toricblm.protrek_trimodal_split_report.v1",
            "protrek_inference_ran": False,
            "ready": True,
            "readiness": readiness,
            "output_written": False,
            "reason": f"ProTrek runtime failure: {type(exc).__name__}: {exc}",
            "records_requested": len(rows),
            "policy": "do not claim ProTrek-derived split until inference completes successfully",
        }
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    cluster_ids = cluster_features(features, rows, max_clusters=args.max_clusters)
    split_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    cluster_splits: dict[str, str] = {}
    out_rows: list[dict[str, Any]] = []
    modality_counts = {
        "sequence_present": 0,
        "text_present": 0,
        "structure_present": 0,
        "trimodal_present": 0,
    }
    for row, modality, cluster_id in zip(rows, modality_rows, cluster_ids, strict=True):
        split = cluster_splits.setdefault(cluster_id, split_for_cluster(cluster_id))
        split_counts[split] += 1
        for key in ("sequence_present", "text_present", "structure_present"):
            modality_counts[key] += int(bool(modality[key]))
        modality_counts["trimodal_present"] += int(bool(modality["sequence_present"] and modality["text_present"] and modality["structure_present"]))
        signature = {
            "schema": "toricblm.protrek_trimodal_signature.v1",
            "cluster_id": cluster_id,
            "split": split,
            "modality_presence": modality,
            "embedding_family": Path(args.weights_dir).name,
            "foldseek_bin": str(args.foldseek_bin),
            "protrek_root": str(args.protrek_root),
        }
        row = {k: v for k, v in row.items() if not k.startswith("_")}
        row["split"] = split
        row["split_cluster"] = cluster_id
        row["protrek_trimodal_signature_json"] = json.dumps(signature, ensure_ascii=True, sort_keys=True)
        out_rows.append(row)
    write_report = write_split_rows(out_rows, args.output_dir, args.prefix)
    report = {
        "schema": "toricblm.protrek_trimodal_split_report.v1",
        "protrek_inference_ran": True,
        "ready": True,
        "readiness": readiness,
        "input_files": [str(path) for path in input_paths],
        "records": len(rows),
        "cluster_count": len(set(cluster_ids)),
        "split_counts": split_counts,
        "modality_counts": modality_counts,
        "encoder_limits": {
            "max_protein_length": int(args.max_protein_length),
            "max_structure_length": int(args.max_structure_length),
            "max_text_chars": int(args.max_text_chars),
        },
        "output": write_report,
        "policy": "Rows missing structure stay trainable; only rows with structure_present=true are trimodal.",
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
