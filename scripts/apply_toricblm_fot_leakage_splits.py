#!/usr/bin/env python3
"""Apply leakage-aware train/validation/test splits to ToricBLM FoT records.

The splitter is conservative with the modalities available locally:

- sequence similarity: k-mer MinHash-style sketches plus length buckets;
- function similarity: GO/EC labels and annotation-text shingles;
- structure similarity: PDB/AFDB/structure-hook identifiers when present;
- FoT graph similarity: tree ids, edge-type histograms, depth/breadth counts,
  self-correction/consensus counts, and source graph node/edge type histograms.

When ProTrek embeddings or FoldSeek clusters are available later, their cluster
ids should be added to the same signature rather than replacing these source
grounded guards.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$", re.IGNORECASE)
GO_RE = re.compile(r"GO:\d{7}")
EC_RE = re.compile(r"\b\d+\.\d+\.\d+\.\d+\b")
PDB_RE = re.compile(r"\b[0-9][A-Za-z0-9]{3}\b")
AFDB_RE = re.compile(r"AF-[A-Za-z0-9]+-F\d+", re.IGNORECASE)


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


def kmers(seq: str, k: int) -> set[str]:
    seq = re.sub(r"\s+", "", seq.upper())
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def minhash_sketch(items: set[str], *, size: int = 24) -> list[str]:
    if not items:
        return []
    hashes = sorted(stable_hash(item, 20) for item in items)
    return hashes[: min(size, len(hashes))]


def sequence_signature(sequence: str) -> dict[str, Any]:
    seq = re.sub(r"\s+", "", str(sequence or "").upper())
    if not seq:
        return {"present": False, "kind": "none", "length_bucket": "none", "sketch": []}
    if AA_RE.match(seq):
        k = 5
        kind = "protein_like"
    else:
        k = 7
        kind = "nucleotide_or_other"
    length_bucket = int(math.log2(max(len(seq), 1)))
    sketch = minhash_sketch(kmers(seq, k), size=24)
    composition = Counter(seq)
    top_symbols = sorted(composition.items(), key=lambda item: (-item[1], item[0]))[:8]
    return {
        "present": True,
        "kind": kind,
        "length_bucket_log2": length_bucket,
        "k": k,
        "sketch": sketch,
        "top_symbols": top_symbols,
    }


def text_shingle_signature(text: str, *, n: int = 5, size: int = 20) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_:-]+", str(text or "").lower())
    if not words:
        return []
    shingles = {" ".join(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}
    return minhash_sketch(shingles, size=size)


def function_signature(row: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    annotation = str(row.get("annotation_text") or "")
    targets = graph.get("targets", {}).get("answer", {}) if isinstance(graph, dict) else {}
    go_terms = sorted(set(targets.get("go_terms") or []) | set(GO_RE.findall(annotation)))
    ec_numbers = sorted(set(targets.get("ec_numbers") or []) | set(EC_RE.findall(annotation)))
    return {
        "go_terms": go_terms[:64],
        "ec_numbers": ec_numbers[:32],
        "annotation_shingles": text_shingle_signature(annotation, n=5, size=20),
    }


def structure_signature(row: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    text = "\n".join(
        str(part or "")
        for part in (
            row.get("text"),
            row.get("graph_json"),
            row.get("enrichment_status_json"),
        )
    )
    targets = graph.get("targets", {}).get("answer", {}) if isinstance(graph, dict) else {}
    links = targets.get("structure_links") or []
    link_text = json.dumps(links, ensure_ascii=True, sort_keys=True)
    pdb = sorted(set(PDB_RE.findall(text + "\n" + link_text)))[:64]
    afdb = sorted(set(match.upper() for match in AFDB_RE.findall(text + "\n" + link_text)))[:64]
    return {
        "has_structure_hook": bool(links or pdb or afdb),
        "pdb_like_ids": pdb,
        "afdb_like_ids": afdb,
        "link_count": len(links) if isinstance(links, list) else 0,
    }


def histogram(items: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(items).items()))


def graph_signature(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    return {
        "node_count_bucket": int(math.log2(max(len(nodes), 1))),
        "edge_count_bucket": int(math.log2(max(len(edges), 1))),
        "node_types": histogram([str(node.get("type", "node")) for node in nodes if isinstance(node, dict)]),
        "edge_types": histogram([str(edge.get("type", "edge")) for edge in edges if isinstance(edge, dict)]),
    }


def fot_signature(fot: dict[str, Any]) -> dict[str, Any]:
    nodes = fot.get("nodes", []) if isinstance(fot, dict) else []
    edges = fot.get("edges", []) if isinstance(fot, dict) else []
    depths = [int(node.get("budget_level", 0)) for node in nodes if isinstance(node, dict)]
    tree_ids = [str(node.get("tree_id", "unknown")) for node in nodes if isinstance(node, dict)]
    edge_types = [str(edge.get("type", "edge")) for edge in edges if isinstance(edge, dict)]
    tree_edge_pairs = [
        f"{edge.get('tree_id', 'unknown')}::{edge.get('type', 'edge')}"
        for edge in edges
        if isinstance(edge, dict)
    ]
    return {
        "node_count_bucket": int(math.log2(max(len(nodes), 1))),
        "edge_count_bucket": int(math.log2(max(len(edges), 1))),
        "max_depth": max(depths) if depths else 0,
        "tree_ids": histogram(tree_ids),
        "edge_types": histogram(edge_types),
        "tree_edge_pairs": histogram(tree_edge_pairs),
        "self_correction_edges": sum(1 for item in edge_types if item == "self_correction"),
        "consensus_edges": sum(1 for item in edge_types if item == "consensus"),
    }


def record_signature(row: dict[str, Any]) -> dict[str, Any]:
    graph = json.loads(row.get("graph_json") or "{}")
    fot = json.loads(row.get("thought_forest_json") or "{}")
    seq_sig = sequence_signature(row.get("sequence") or "")
    function_sig = function_signature(row, graph)
    structure_sig = structure_signature(row, graph)
    graph_sig = graph_signature(graph)
    fot_sig = fot_signature(fot)
    dataset = str(row.get("dataset") or "unknown")
    cluster_basis = {
        "dataset": dataset,
        "sequence": seq_sig,
        "function": function_sig,
        "structure": structure_sig,
        "graph": graph_sig,
        "fot": fot_sig,
    }
    modality_keys = {
        "sequence_cluster": stable_hash({"dataset": dataset, "sequence": seq_sig}, 20),
        "function_cluster": stable_hash({"dataset": dataset, "function": function_sig}, 20),
        "structure_cluster": stable_hash({"dataset": dataset, "structure": structure_sig}, 20),
        "fot_graph_cluster": stable_hash({"dataset": dataset, "graph": graph_sig, "fot": fot_sig}, 20),
    }
    cluster_id = "leak_" + stable_hash({"modality_keys": modality_keys, "basis": cluster_basis}, 24)
    return {
        "schema": "toricblm.leakage_aware_split_signature.v1",
        "cluster_id": cluster_id,
        "cluster_basis": cluster_basis,
        "modality_keys": modality_keys,
        "split_policy": "deterministic_cluster_hash_90_5_5",
        "protrek_ready": True,
        "foldseek_ready": True,
    }


def apply_splits(input_paths: list[Path], output_path: Path, report_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cluster_splits: dict[str, str] = {}
    cluster_counts = Counter()
    split_counts = Counter()
    modality_counts: dict[str, Counter] = {
        "sequence_cluster": Counter(),
        "function_cluster": Counter(),
        "structure_cluster": Counter(),
        "fot_graph_cluster": Counter(),
    }
    for path in input_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=128):
            for row in pa.Table.from_batches([batch]).to_pylist():
                signature = record_signature(row)
                cluster_id = signature["cluster_id"]
                split = cluster_splits.setdefault(cluster_id, split_for_cluster(cluster_id))
                row = dict(row)
                row["split"] = split
                row["split_cluster"] = cluster_id
                row["leakage_signature_json"] = json.dumps(signature, ensure_ascii=True, sort_keys=True)
                rows.append(row)
                cluster_counts[cluster_id] += 1
                split_counts[split] += 1
                for key, value in signature["modality_keys"].items():
                    modality_counts[key][value] += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path, compression="zstd", use_dictionary=True)
    report = {
        "schema": "toricblm.leakage_aware_split_report.v1",
        "input_files": [str(path) for path in input_paths],
        "output_path": str(output_path),
        "records": len(rows),
        "split_counts": dict(split_counts),
        "cluster_count": len(cluster_counts),
        "max_cluster_size": max(cluster_counts.values()) if cluster_counts else 0,
        "modality_cluster_counts": {key: len(counter) for key, counter in modality_counts.items()},
        "largest_modality_clusters": {
            key: counter.most_common(8)
            for key, counter in modality_counts.items()
        },
        "notes": [
            "All rows in the same combined sequence/function/structure/FoT cluster are assigned to the same split.",
            "Structure similarity currently uses source-grounded PDB/AFDB hooks because coordinate-bearing structure files are not present in the local raw dataset.",
            "ProTrek trimodal embedding clusters and FoldSeek clusters can be added later as additional modality_keys without changing the downstream schema.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Input Parquet glob. May repeat.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    paths: list[Path] = []
    for pattern in args.input:
        matches = [Path(p) for p in sorted(glob.glob(str(Path(pattern).expanduser())))]
        if not matches:
            raise FileNotFoundError(f"no files matched {pattern}")
        paths.extend(matches)
    report = apply_splits(sorted(dict.fromkeys(paths)), Path(args.output), Path(args.report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
