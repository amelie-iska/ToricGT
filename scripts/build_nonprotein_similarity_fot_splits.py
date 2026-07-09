#!/usr/bin/env python3
"""Build leakage-aware non-protein ToricBLM FoT split shards.

This script converts raw DNA, RNA/Rfam/RNAcentral, and PubChem SELFIES rows into
the same graph/FoT Parquet schema used by the UniProt FoT builder.  It enforces
split assignment before writing train shards:

* DNA/RNA rows are clustered by sequence-minhash bands plus available
  annotation keys such as family, clan, organism, type, and accession prefix.
* PubChem rows that decode through SELFIES/RDKit are clustered by Murcko
  scaffold plus Morgan-fingerprint locality buckets.
* PubChem rows in a legacy or extended SELFIES alphabet that cannot be decoded
  are still split by an explicit SELFIES token-graph MinHash cluster.  That
  cluster is labeled as token-graph evidence, not as a chemical scaffold.

The output is intentionally downsampled and diverse.  By default it targets
approximately 500k train rows for each of DNA, RNA, and small molecules, with
validation/test rows assigned from the same cluster policy.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_uniprot_fot_dataset import (  # noqa: E402
    DEFAULT_CONVEXTOK,
    RECORD_SCHEMA,
    graphify_row,
    index_convextok_tokens,
    load_convextok_tokens,
    stable_json,
)


RAW_ROOT = Path("/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale")
OUT_DIR = Path("/home/iska/Documents/amelie/bio/ToricGT/data/toricblm_nonprotein_fot_splits/v2_parallel")

_WORKER_CONFIG: dict[str, Any] = {}


def htext(value: Any, length: int = 16) -> str:
    text = value if isinstance(value, str) else stable_json(value)
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=16).hexdigest()[:length]


def split_for_cluster(cluster_key: str) -> str:
    bucket = int(htext(cluster_key, 8), 16) % 1000
    if bucket < 900:
        return "train"
    if bucket < 950:
        return "validation"
    return "test"


def expand_patterns(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for raw in sorted(glob.glob(str(pattern))):
            path = Path(raw)
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths


def clean_sequence(value: Any, max_chars: int = 200000) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", "", value[: max(1, int(max_chars))]).upper()


def kmers(seq: str, k: int) -> set[str]:
    if not seq:
        return set()
    if len(seq) < k:
        return {seq}
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def minhash_values(items: set[str], size: int = 64) -> list[str]:
    if not items:
        return []
    return sorted(htext(item, 20) for item in items)[: min(size, len(items))]


def sequence_minhash_values(seq: str, k: int, *, size: int = 64, max_kmers: int = 20000) -> list[str]:
    """Bounded MinHash sketch for long biological sequences.

    Full set materialization is unnecessarily expensive for large DNA/RNA rows.
    This samples k-mer positions deterministically when the sequence has more
    than ``max_kmers`` possible windows, then keeps the smallest hash values.
    Split assignment remains deterministic and cluster based.
    """
    if not seq:
        return []
    if len(seq) < k:
        return [htext(seq, 20)]
    n = len(seq) - k + 1
    if n <= max_kmers:
        positions = range(n)
    else:
        step = max(1, n // max_kmers)
        positions = range(0, n, step)
    hashes = {htext(seq[i : i + k], 20) for i in positions}
    return sorted(hashes)[: min(size, len(hashes))]


def seq_cluster(row: dict[str, Any], *, modality: str) -> dict[str, Any]:
    seq = clean_sequence(row.get("sequence"))
    if not seq:
        return {"ok": False, "reason": "missing_sequence"}
    k = 9 if modality == "dna" else 7
    sketch = sequence_minhash_values(seq, k, size=64)
    if not sketch:
        return {"ok": False, "reason": "empty_sequence_sketch"}
    # The first minimizer is a conservative locality anchor: near duplicates
    # often share it, so split assignment follows this anchor before row-level
    # hashing.  Additional bands remain in the audit signature.
    band_size = 4
    bands = [htext(sketch[i : i + band_size], 16) for i in range(0, min(len(sketch), 32), band_size)]
    family = row.get("family") or row.get("clan") or row.get("type") or ""
    organism = row.get("organism") or ""
    accession = row.get("accession") or row.get("id") or row.get("upi") or ""
    accession_prefix = str(accession)[:16]
    cluster = {
        "schema": "toricblm.sequence_similarity_cluster.v1",
        "modality": modality,
        "k": k,
        "length_bucket_log2": int(math.log2(max(len(seq), 1))),
        "primary_minhash": sketch[0],
        "bands": bands,
        "family_or_type": str(family)[:96],
        "organism": str(organism)[:96],
        "accession_prefix": accession_prefix,
    }
    cluster_key = "seqsim:" + htext(
        {
            "modality": modality,
            "primary_minhash": sketch[0],
            "family_or_type": str(family)[:96],
            "length_bucket_log2": cluster["length_bucket_log2"],
        },
        24,
    )
    return {"ok": True, "cluster_key": cluster_key, "signature": cluster}


def pubchem_cluster(row: dict[str, Any]) -> dict[str, Any]:
    selfies_value = row.get("SELFIES") or row.get("selfies")
    if not isinstance(selfies_value, str) or not selfies_value:
        return {"ok": False, "reason": "missing_selfies"}
    try:
        import selfies as sf
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PubChem split requires `selfies` and RDKit in the active environment") from exc
    tokens = re.findall(r"\[[^\]]+\]", selfies_value)
    if len(tokens) < 2:
        return {"ok": False, "reason": "empty_selfies_token_sequence"}
    try:
        smiles = sf.decoder(selfies_value)
    except Exception:
        token_ngrams = {" ".join(tokens[i : i + 3]) for i in range(max(1, len(tokens) - 2))}
        sketch = minhash_values(token_ngrams, size=64)
        if not sketch:
            return {"ok": False, "reason": "empty_selfies_token_sketch"}
        band_size = 4
        bands = [htext(sketch[i : i + band_size], 16) for i in range(0, min(len(sketch), 32), band_size)]
        cluster = {
            "schema": "toricblm.pubchem_selfies_token_cluster.v1",
            "modality": "small_molecule",
            "cluster_source": "selfies_token_graph_minhash",
            "decode_status": "selfies_decoder_failed_for_legacy_or_extended_alphabet",
            "token_count_bucket_log2": int(math.log2(max(len(tokens), 1))),
            "primary_token_minhash": sketch[0],
            "token_minhash_bands": bands,
            "token_prefix": tokens[:16],
        }
        cluster_key = "pubchem_selfies_token:" + htext(
            {
                "primary_token_minhash": sketch[0],
                "token_count_bucket_log2": cluster["token_count_bucket_log2"],
                "first_band": bands[0] if bands else "",
            },
            24,
        )
        return {"ok": True, "cluster_key": cluster_key, "signature": cluster}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"ok": False, "reason": "rdkit_parse_failed"}
    canonical = Chem.MolToSmiles(mol, canonical=True)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaffold = ""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    onbits = list(fp.GetOnBits())
    if not onbits:
        return {"ok": False, "reason": "empty_morgan_fingerprint"}
    bands = []
    for start in range(0, min(len(onbits), 96), 12):
        bands.append(htext(onbits[start : start + 12], 16))
    scaffold_key = scaffold or ("fpband:" + bands[0])
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    cluster = {
        "schema": "toricblm.pubchem_scaffold_cluster.v1",
        "modality": "small_molecule",
        "cluster_source": "rdkit_murcko_scaffold_and_morgan_fingerprint",
        "murcko_scaffold_smiles": scaffold,
        "canonical_smiles_hash": htext(canonical, 20),
        "morgan_radius": 2,
        "morgan_nbits": 2048,
        "morgan_onbit_bands": bands,
        "heavy_atom_bucket": int(math.log2(max(heavy_atoms, 1))),
    }
    cluster_key = "pubchem_scaffold:" + htext(
        {
            "scaffold": scaffold_key,
            "heavy_atom_bucket": cluster["heavy_atom_bucket"],
            "first_morgan_band": bands[0],
        },
        24,
    )
    return {"ok": True, "cluster_key": cluster_key, "signature": cluster, "canonical_smiles": canonical}


def update_record_split(
    record: dict[str, Any],
    *,
    split: str,
    cluster_key: str,
    cluster_signature: dict[str, Any],
    source_plan: str,
) -> dict[str, Any]:
    out = dict(record)
    cluster_payload = {
        "schema": "toricblm.nonprotein_similarity_split.v1",
        "cluster_id": cluster_key,
        "split": split,
        "source_plan": source_plan,
        "cluster_signature": cluster_signature,
        "policy": "cluster_hash_90_5_5_with_diverse_downsample",
    }
    out["split"] = split
    out["group_hash"] = htext(cluster_payload, 64)
    out["split_cluster"] = json.dumps(cluster_payload, ensure_ascii=True, sort_keys=True)
    out["leakage_signature_json"] = json.dumps(cluster_payload, ensure_ascii=True, sort_keys=True)
    try:
        graph = json.loads(out["graph_json"])
        graph["split_cluster"] = cluster_payload
        graph.setdefault("metadata", {})["leakage_control"] = cluster_payload
        out["graph_json"] = json.dumps(graph, ensure_ascii=True, sort_keys=True)
        out["metadata_json"] = json.dumps(graph.get("metadata", {}), ensure_ascii=True, sort_keys=True)
    except Exception:
        pass
    try:
        views = json.loads(out.get("training_views_json") or "{}")
        views.setdefault("views", {})["leakage_control"] = {
            "split_cluster_column": "split_cluster",
            "leakage_signature_column": "leakage_signature_json",
            "cluster_id": cluster_key,
            "split": split,
        }
        out["training_views_json"] = json.dumps(views, ensure_ascii=True, sort_keys=True)
    except Exception:
        pass
    return out


def init_graphify_worker(
    convextok_index: dict[int, list[dict[str, Any]]],
    convextok_max_bytes: int,
    max_sequence_chars: int,
    max_field_chars: int,
) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = {
        "convextok_index": convextok_index,
        "convextok_max_bytes": int(convextok_max_bytes),
        "max_sequence_chars": int(max_sequence_chars),
        "max_field_chars": int(max_field_chars),
    }


def graphify_worker_task(task: dict[str, Any]) -> dict[str, Any]:
    try:
        row = dict(task["row"])
        sequence = row.get("sequence")
        if isinstance(sequence, str):
            max_sequence_chars = int(_WORKER_CONFIG["max_sequence_chars"])
            if len(sequence) > max_sequence_chars:
                row["source_sequence_length"] = len(sequence)
        record = graphify_row(
            dataset=str(task["dataset"]),
            row=row,
            source_file=str(task["source_file"]),
            source_row_index=int(task["source_row_index"]),
            max_sequence_chars=int(_WORKER_CONFIG["max_sequence_chars"]),
            max_field_chars=int(_WORKER_CONFIG["max_field_chars"]),
            convextok_tokens=_WORKER_CONFIG["convextok_index"],
            convextok_max_bytes=int(_WORKER_CONFIG["convextok_max_bytes"]),
        )
        record = update_record_split(
            record,
            split=str(task["split"]),
            cluster_key=str(task["cluster_key"]),
            cluster_signature=dict(task["cluster_signature"]),
            source_plan=str(task["plan_name"]),
        )
        return {"ok": True, "record": record, "split": str(task["split"])}
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "split": str(task.get("split") or ""),
            "error": f"graphify_error:{type(exc).__name__}",
            "message": str(exc)[:500],
        }


def free_gb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return float(usage.free) / float(1024**3)


class SplitWriter:
    def __init__(self, out_dir: Path, modality: str, *, prefix: str, shard_size: int, min_free_gb: float) -> None:
        self.out_dir = out_dir / modality
        self.modality = modality
        self.prefix = prefix
        self.shard_size = int(shard_size)
        self.min_free_gb = float(min_free_gb)
        self.buffers: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
        self.counts: Counter[str] = Counter()
        self.shards: Counter[str] = Counter()
        self.paths: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
        for split in self.buffers:
            (self.out_dir / split).mkdir(parents=True, exist_ok=True)

    def add(self, row: dict[str, Any], split: str) -> None:
        self.buffers[split].append(row)
        self.counts[split] += 1
        if len(self.buffers[split]) >= self.shard_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        available_gb = free_gb(self.out_dir)
        if available_gb < self.min_free_gb:
            raise RuntimeError(
                f"disk_free_gb={available_gb:.2f} below min_free_gb={self.min_free_gb:.2f}; "
                f"refusing to flush {self.modality}/{split}"
            )
        path = self.out_dir / split / f"{self.prefix}_{self.modality}_{split}_{self.shards[split]:05d}.parquet"
        table = pa.Table.from_pylist(rows, schema=RECORD_SCHEMA)
        pq.write_table(table, path, compression="zstd", use_dictionary=True)
        self.paths[split].append(str(path))
        self.shards[split] += 1
        self.buffers[split] = []

    def close(self) -> None:
        for split in list(self.buffers):
            self.flush(split)


def iter_rows(paths: list[Path], *, columns: list[str] | None = None, batch_size: int = 4096):
    for path in paths:
        try:
            pf = pq.ParquetFile(path)
            schema_names = pf.schema_arrow.names
            use_columns = [name for name in (columns or schema_names) if name in schema_names]
            if not use_columns:
                continue
            row_offset = 0
            for batch in pf.iter_batches(batch_size=batch_size, columns=use_columns):
                for idx, row in enumerate(pa.Table.from_batches([batch]).to_pylist()):
                    yield path, row_offset + idx, row
                row_offset += batch.num_rows
        except Exception as exc:  # noqa: BLE001
            yield path, -1, {"_read_error": f"{type(exc).__name__}: {exc}"}


def target_counts(train_target: int, val_frac: float, test_frac: float) -> dict[str, int]:
    return {
        "train": int(train_target),
        "validation": max(1, int(round(train_target * val_frac))),
        "test": max(1, int(round(train_target * test_frac))),
    }


def process_plan(
    *,
    plan_name: str,
    modality: str,
    paths: list[Path],
    writer: SplitWriter,
    train_target: int,
    val_frac: float,
    test_frac: float,
    max_per_cluster: int,
    max_sequence_chars: int,
    max_field_chars: int,
    convextok_index: dict[int, list[dict[str, Any]]],
    convextok_max_bytes: int,
    columns: list[str],
    cluster_kind: str,
    num_workers: int,
    max_pending_per_worker: int,
    stop_when_train_target_met: bool,
) -> dict[str, Any]:
    targets = target_counts(train_target, val_frac, test_frac)
    accepted: Counter[str] = Counter()
    pending_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()
    scanned = 0
    max_workers = max(1, int(num_workers))
    max_pending = max_workers * max(1, int(max_pending_per_worker))

    def quota_full(include_pending: bool = False) -> bool:
        def observed(split: str) -> int:
            return int(accepted[split] + (pending_counts[split] if include_pending else 0))

        if stop_when_train_target_met:
            return observed("train") >= targets["train"]
        return all(observed(split) >= targets[split] for split in targets)

    def handle_result(result: dict[str, Any]) -> None:
        split = str(result.get("split") or "")
        if split:
            pending_counts[split] -= 1
        if result.get("ok"):
            record = result["record"]
            writer.add(record, str(result["split"]))
            accepted[str(result["split"])] += 1
        else:
            skipped[str(result.get("error") or "graphify_error")] += 1

    if max_workers <= 1:
        init_graphify_worker(convextok_index, convextok_max_bytes, max_sequence_chars, max_field_chars)
        executor_context = None
    else:
        init_graphify_worker(convextok_index, convextok_max_bytes, max_sequence_chars, max_field_chars)
        executor_context = ThreadPoolExecutor(
            max_workers=max_workers,
        )

    pending = set()
    try:
        for path, row_idx, row in iter_rows(paths, columns=columns):
            if quota_full(include_pending=True):
                break
            scanned += 1
            if row_idx < 0 or row.get("_read_error"):
                skipped[str(row.get("_read_error") or "read_error")] += 1
                continue
            cluster = pubchem_cluster(row) if cluster_kind == "pubchem" else seq_cluster(row, modality=modality)
            if not cluster.get("ok"):
                skipped[str(cluster.get("reason") or "cluster_failed")] += 1
                continue
            cluster_key = str(cluster["cluster_key"])
            split = split_for_cluster(cluster_key)
            if accepted[split] + pending_counts[split] >= targets[split]:
                skipped[f"quota_full:{split}"] += 1
                continue
            if cluster_counts[cluster_key] >= max_per_cluster:
                skipped["cluster_cap"] += 1
                continue
            row = dict(row)
            if cluster_kind == "pubchem" and cluster.get("canonical_smiles"):
                row["canonical_smiles"] = cluster["canonical_smiles"]
            task = {
                "dataset": f"toricblm_{plan_name}",
                "row": row,
                "source_file": str(path),
                "source_row_index": int(row_idx),
                "split": split,
                "cluster_key": cluster_key,
                "cluster_signature": cluster["signature"],
                "plan_name": plan_name,
            }
            cluster_counts[cluster_key] += 1
            pending_counts[split] += 1
            if executor_context is None:
                handle_result(graphify_worker_task(task))
            else:
                pending.add(executor_context.submit(graphify_worker_task, task))
                while len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        handle_result(future.result())
            if scanned % 100000 == 0:
                print(
                    f"{plan_name}: scanned={scanned} accepted={dict(accepted)} "
                    f"pending={dict(pending_counts)} workers={max_workers} "
                    f"clusters={len(cluster_counts)} skipped_top={skipped.most_common(3)}",
                    flush=True,
                )
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                handle_result(future.result())
    finally:
        if executor_context is not None:
            executor_context.shutdown(wait=True, cancel_futures=False)
    return {
        "plan": plan_name,
        "modality": modality,
        "input_files": [str(path) for path in paths],
        "targets": targets,
        "accepted": dict(accepted),
        "workers": max_workers,
        "scanned": scanned,
        "clusters": len(cluster_counts),
        "max_cluster_size_seen": max(cluster_counts.values()) if cluster_counts else 0,
        "skipped": dict(skipped),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--convextok-tokenizer", type=Path, default=DEFAULT_CONVEXTOK)
    parser.add_argument("--target-train-per-modality", type=int, default=500_000)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--max-per-cluster", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--max-sequence-chars", type=int, default=16384)
    parser.add_argument("--max-field-chars", type=int, default=8192)
    parser.add_argument("--convextok-max-bytes", type=int, default=1024)
    parser.add_argument("--min-free-gb", type=float, default=30.0, help="Stop before writing a shard if available disk falls below this floor.")
    parser.add_argument("--num-workers", type=int, default=1, help="Process workers for graph/FoT row materialization inside each selected plan.")
    parser.add_argument("--max-pending-per-worker", type=int, default=8, help="Bound queued graphify tasks per process worker.")
    parser.add_argument(
        "--stop-when-train-target-met",
        action="store_true",
        help="Stop each selected plan when its train target is met; validation/test rows remain opportunistic cluster-safe splits.",
    )
    parser.add_argument("--dna-train-target", type=int, default=None)
    parser.add_argument("--rna-train-target", type=int, default=None)
    parser.add_argument("--small-train-target", type=int, default=None)
    parser.add_argument(
        "--include-plan",
        action="append",
        default=None,
        help=(
            "Run only a named source plan; repeatable. Valid names are "
            "dna_coding_regions, rnacentral_8192, rfam_sequence, and pubchem_selfies. "
            "This is used to run modality curation in parallel without shard conflicts."
        ),
    )
    parser.add_argument(
        "--manifest-name",
        default="toricblm_nonprotein_similarity_fot_split_manifest.json",
        help="Manifest filename under out-dir/manifests.",
    )
    parser.add_argument("--smoke", action="store_true", help="Tiny targets for fast validation.")
    args = parser.parse_args()

    if args.smoke:
        args.dna_train_target = 64
        args.rna_train_target = 64
        args.small_train_target = 64
        args.val_frac = 0.10
        args.test_frac = 0.10
        args.shard_size = 32

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    convextok_tokens = load_convextok_tokens(args.convextok_tokenizer, max_token_bytes=96)
    convextok_index = index_convextok_tokens(convextok_tokens)

    dna_target = int(args.dna_train_target or args.target_train_per_modality)
    rna_target = int(args.rna_train_target or args.target_train_per_modality)
    small_target = int(args.small_train_target or args.target_train_per_modality)

    writers = {
        "dna": SplitWriter(args.out_dir, "dna", prefix="toricblm_nonprotein_fot", shard_size=args.shard_size, min_free_gb=args.min_free_gb),
        "rna": SplitWriter(args.out_dir, "rna", prefix="toricblm_nonprotein_fot", shard_size=args.shard_size, min_free_gb=args.min_free_gb),
        "small_molecule": SplitWriter(args.out_dir, "small_molecule", prefix="toricblm_nonprotein_fot", shard_size=args.shard_size, min_free_gb=args.min_free_gb),
    }

    plans = [
        {
            "plan_name": "dna_coding_regions",
            "modality": "dna",
            "paths": expand_patterns([str(args.raw_root / "dna_coding_regions_train/default/train/*.parquet")]),
            "writer": writers["dna"],
            "train_target": dna_target,
            "columns": ["accession", "organism", "sequence", "introns", "exons", "proteins"],
            "cluster_kind": "sequence",
        },
        {
            "plan_name": "rnacentral_8192",
            "modality": "rna",
            "paths": expand_patterns([str(args.raw_root / "rnacentral_8192_sequence_train/default/train/*.parquet")]),
            "writer": writers["rna"],
            "train_target": max(1, rna_target // 2),
            "columns": ["upi", "sequence", "type", "description"],
            "cluster_kind": "sequence",
        },
        {
            "plan_name": "rfam_sequence",
            "modality": "rna",
            "paths": expand_patterns([str(args.raw_root / "rfam_sequence_train/default/train/*.parquet")]),
            "writer": writers["rna"],
            "train_target": max(1, rna_target - max(1, rna_target // 2)),
            "columns": ["id", "sequence", "family", "clan", "description"],
            "cluster_kind": "sequence",
        },
        {
            "plan_name": "pubchem_selfies",
            "modality": "small_molecule",
            "paths": expand_patterns([str(args.raw_root / "pubchem10m_selfies_train/default/train/*.parquet")]),
            "writer": writers["small_molecule"],
            "train_target": small_target,
            "columns": ["SELFIES"],
            "cluster_kind": "pubchem",
        },
    ]
    if args.include_plan:
        allowed = set(args.include_plan)
        known = {str(plan["plan_name"]) for plan in plans}
        unknown = sorted(allowed - known)
        if unknown:
            raise SystemExit(f"Unknown --include-plan values {unknown}; known={sorted(known)}")
        plans = [plan for plan in plans if str(plan["plan_name"]) in allowed]
        if not plans:
            raise SystemExit("No curation plans selected.")

    reports = []
    for plan in plans:
        if not plan["paths"]:
            reports.append({"plan": plan["plan_name"], "error": "no_input_files"})
            continue
        reports.append(
            process_plan(
                plan_name=plan["plan_name"],
                modality=plan["modality"],
                paths=plan["paths"],
                writer=plan["writer"],
                train_target=int(plan["train_target"]),
                val_frac=float(args.val_frac),
                test_frac=float(args.test_frac),
                max_per_cluster=int(args.max_per_cluster),
                max_sequence_chars=int(args.max_sequence_chars),
                max_field_chars=int(args.max_field_chars),
                convextok_index=convextok_index,
                convextok_max_bytes=int(args.convextok_max_bytes),
                columns=plan["columns"],
                cluster_kind=plan["cluster_kind"],
                num_workers=int(args.num_workers),
                max_pending_per_worker=int(args.max_pending_per_worker),
                stop_when_train_target_met=bool(args.stop_when_train_target_met),
            )
        )
    for writer in writers.values():
        writer.close()

    manifest = {
        "schema": "toricblm.nonprotein_similarity_fot_split_manifest.v1",
        "out_dir": str(args.out_dir),
        "convextok_tokenizer": str(args.convextok_tokenizer),
        "convextok_priced_tokens_loaded": len(convextok_tokens),
        "targets": {
            "dna_train": dna_target,
            "rna_train": rna_target,
            "small_molecule_train": small_target,
            "validation_fraction": args.val_frac,
            "test_fraction": args.test_frac,
        },
        "plans": reports,
        "writer_counts": {modality: dict(writer.counts) for modality, writer in writers.items()},
        "writer_paths": {modality: writer.paths for modality, writer in writers.items()},
        "notes": [
            "Outputs preserve ToricBLM graph_json, forest_json, thought_forest_json, convextok_dag_json, and training_views_json.",
            "PubChem rows use RDKit Murcko scaffold plus Morgan fingerprint when SELFIES decodes; legacy/extended SELFIES rows use explicit SELFIES token-graph MinHash clusters and are labeled as such.",
            "RNA and DNA rows use sequence-minhash leakage clusters with available family/type/organism metadata.",
            "Training configs should consume only the train/ subdirectories from this output, not the raw upstream train folders.",
        ],
    }
    manifest_path = args.out_dir / "manifests" / str(args.manifest_name)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
