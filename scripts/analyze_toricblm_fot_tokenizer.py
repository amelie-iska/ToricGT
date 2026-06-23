#!/usr/bin/env python3
"""Audit ConvexTok coverage on ToricBLM FoT biological graph records."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PG_ROOT = ROOT / "amelie-iska" / "parameter-golf"
if str(PG_ROOT) not in sys.path:
    sys.path.insert(0, str(PG_ROOT))

from convextok import BYTE_OFFSET, PRICED_OFFSET, ConvexTokTokenizer  # noqa: E402


FIELDS = (
    "text",
    "sequence",
    "annotation_text",
    "graph_json",
    "thought_forest_json",
    "training_views_json",
    "enrichment_status_json",
)


def iter_rows(patterns: list[str], *, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in sorted(glob.glob(str(Path(pattern).expanduser())))]
        if not matches:
            raise FileNotFoundError(f"no Parquet files matched: {pattern}")
        paths.extend(matches)
    emitted = 0
    for path in sorted(dict.fromkeys(paths)):
        pf = pq.ParquetFile(path)
        available = set(pf.schema.names)
        columns = [field for field in FIELDS if field in available]
        columns.extend([field for field in ("dataset", "task_family", "record_id") if field in available])
        for batch in pf.iter_batches(batch_size=128, columns=columns):
            table = pa.Table.from_batches([batch])
            for row in table.to_pylist():
                yield row
                emitted += 1
                if max_rows is not None and emitted >= max_rows:
                    return


def clean_text(value: Any, *, max_bytes: int) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ").strip()
    data = text.encode("utf-8", errors="replace")
    if len(data) > max_bytes:
        text = data[:max_bytes].decode("utf-8", errors="ignore")
    return text.strip()


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    ordered = sorted(values)
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p10": float(ordered[int(0.10 * (len(ordered) - 1))]),
        "p90": float(ordered[int(0.90 * (len(ordered) - 1))]),
    }


def audit_tokenizer(
    tok: ConvexTokTokenizer,
    docs: list[tuple[str, str, str]],
    *,
    reserve_limit: int | None,
    exact_trace: bool = False,
) -> dict[str, Any]:
    by_field: dict[str, dict[str, Any]] = {}
    aggregate = Counter()
    field_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for field, dataset, text in docs:
        data = text.encode("utf-8", errors="replace")
        if not data:
            continue
        if exact_trace:
            ids, trace = tok.encode_bytes_with_trace(data)
        else:
            ids = tok.encode_bytes(data)
            trace = {"mean_tropical_margin": 0.0, "active_path_entropy": 0.0}
        priced = sum(1 for token_id in ids if token_id >= PRICED_OFFSET)
        byte_fallback = sum(1 for token_id in ids if BYTE_OFFSET <= token_id < PRICED_OFFSET)
        reserve_hits = 0
        if reserve_limit:
            reserve_end = PRICED_OFFSET + int(reserve_limit)
            reserve_hits = sum(1 for token_id in ids if PRICED_OFFSET <= token_id < reserve_end)
        token_count = len(ids)
        byte_count = len(data)
        aggregate["docs"] += 1
        aggregate["bytes"] += byte_count
        aggregate["tokens"] += token_count
        aggregate["priced"] += priced
        aggregate["byte_fallback"] += byte_fallback
        aggregate["reserve_hits"] += reserve_hits
        values = field_values[field]
        values["tokens_per_byte"].append(token_count / max(byte_count, 1))
        values["bytes_per_token"].append(byte_count / max(token_count, 1))
        values["priced_fraction"].append(priced / max(token_count, 1))
        values["byte_fallback_fraction"].append(byte_fallback / max(token_count, 1))
        values["reserve_fraction"].append(reserve_hits / max(token_count, 1))
        values["mean_tropical_margin"].append(float(trace.get("mean_tropical_margin", 0.0)))
        values["active_path_entropy"].append(float(trace.get("active_path_entropy", 0.0)))
        by_dataset = values.setdefault(f"dataset::{dataset}::tokens_per_byte", [])
        by_dataset.append(token_count / max(byte_count, 1))
    for field, values in sorted(field_values.items()):
        by_field[field] = {
            key: summarize(vals)
            for key, vals in sorted(values.items())
            if not key.startswith("dataset::")
        }
        by_field[field]["dataset_tokens_per_byte_mean"] = {
            key.split("::", 2)[1]: float(statistics.fmean(vals))
            for key, vals in sorted(values.items())
            if key.startswith("dataset::") and vals
        }
    return {
        "docs": int(aggregate["docs"]),
        "bytes": int(aggregate["bytes"]),
        "tokens": int(aggregate["tokens"]),
        "tokens_per_byte": float(aggregate["tokens"] / max(aggregate["bytes"], 1)),
        "bytes_per_token": float(aggregate["bytes"] / max(aggregate["tokens"], 1)),
        "priced_fraction": float(aggregate["priced"] / max(aggregate["tokens"], 1)),
        "byte_fallback_fraction": float(aggregate["byte_fallback"] / max(aggregate["tokens"], 1)),
        "reserve_fraction": float(aggregate["reserve_hits"] / max(aggregate["tokens"], 1)),
        "by_field": by_field,
    }


def build_docs(rows: Iterable[dict[str, Any]], *, max_field_bytes: int) -> list[tuple[str, str, str]]:
    docs: list[tuple[str, str, str]] = []
    for row in rows:
        dataset = str(row.get("dataset") or row.get("task_family") or "unknown")
        for field in FIELDS:
            value = row.get(field)
            text = clean_text(value, max_bytes=max_field_bytes)
            if text:
                docs.append((field, dataset, text))
    return docs


def recommendation(payload: dict[str, Any]) -> list[str]:
    current = payload["current"]
    old = payload.get("old")
    notes: list[str] = []
    priced = float(current["priced_fraction"])
    fallback = float(current["byte_fallback_fraction"])
    bpt = float(current["bytes_per_token"])
    notes.append(
        f"Current ConvexTok bytes/token is {bpt:.3f}, priced-token fraction is {priced:.3f}, "
        f"and byte-fallback fraction is {fallback:.3f}."
    )
    if old:
        delta = 100.0 * (float(current["tokens_per_byte"]) / max(float(old["tokens_per_byte"]), 1e-12) - 1.0)
        notes.append(f"Relative to ConvexTok-2048, ConvexTok-8192 token count delta is {delta:.2f}% on audited fields.")
        if delta <= -8.0:
            notes.append("The 8192 tokenizer is materially better than 2048 on this dataset; keep it for the next full run.")
        elif delta <= -2.0:
            notes.append("The 8192 tokenizer improves coverage but not dramatically; keep it now and schedule a larger-vocab audit.")
        else:
            notes.append("The 8192 tokenizer is not clearly better on these fields; retraining/extension should be prioritized before a long run.")
    if fallback > 0.45:
        notes.append("Byte fallback is high; add a post-curation ConvexTok extension pass over graph_json, thought_forest_json, structure tags, GO/EC syntax, and sequence chunks.")
    elif fallback > 0.30:
        notes.append("Byte fallback is moderate; no immediate blocker, but a 12k/16k ConvexTok candidate sweep is justified before dynamics-scale training.")
    else:
        notes.append("Fallback is acceptable for a forward run; tokenizer retraining is optional rather than blocking.")
    sequence = current["by_field"].get("sequence", {})
    seq_bpt = sequence.get("bytes_per_token", {}).get("mean", 0.0) if sequence else 0.0
    if seq_bpt and seq_bpt < 2.0:
        notes.append("Sequence compression is weak; reserve more amino-acid/nucleotide motif tokens before large protein/dynamics training.")
    return notes


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ToricBLM FoT Tokenizer Audit",
        "",
        f"- Records/files audited: `{payload['source_patterns']}`",
        f"- Field-documents audited: `{payload['current']['docs']}`",
        f"- Current tokenizer: `{payload['current_tokenizer']}`",
    ]
    if payload.get("old_tokenizer"):
        lines.append(f"- Comparison tokenizer: `{payload['old_tokenizer']}`")
    lines.extend(["", "## Decision Notes", ""])
    for note in payload["recommendation"]:
        lines.append(f"- {note}")
    lines.extend(["", "## Aggregate Metrics", ""])
    rows = [
        ("current", payload["current"]),
    ]
    if payload.get("old"):
        rows.append(("old", payload["old"]))
    lines.append("| tokenizer | bytes/token | tokens/byte | priced fraction | byte fallback | reserve fraction |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, metrics in rows:
        lines.append(
            f"| {name} | {metrics['bytes_per_token']:.4f} | {metrics['tokens_per_byte']:.4f} | "
            f"{metrics['priced_fraction']:.4f} | {metrics['byte_fallback_fraction']:.4f} | {metrics['reserve_fraction']:.4f} |"
        )
    lines.extend(["", "## Current Tokenizer By Field", ""])
    lines.append("| field | bytes/token mean | priced fraction mean | byte fallback mean | reserve fraction mean | active entropy mean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for field, metrics in payload["current"]["by_field"].items():
        lines.append(
            f"| {field} | {metrics['bytes_per_token']['mean']:.4f} | "
            f"{metrics['priced_fraction']['mean']:.4f} | {metrics['byte_fallback_fraction']['mean']:.4f} | "
            f"{metrics['reserve_fraction']['mean']:.4f} | {metrics['active_path_entropy']['mean']:.4f} |"
        )
    lines.extend(["", "## Training Implication", ""])
    lines.append(
        "Use the existing ConvexTok-8192 tokenizer for the immediate full ToricBLM run if the aggregate fallback "
        "is acceptable. For future structure/dynamics phases, rerun this audit after adding coordinate, residue-pair, "
        "atom-type, chain-id, and trajectory metadata fields; only then decide whether 12k/16k vocabulary expansion "
        "is worth the parameter and artifact cost."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", action="append", required=True, help="Curated ToricBLM FoT Parquet glob. May repeat.")
    parser.add_argument("--current-tokenizer", required=True)
    parser.add_argument("--old-tokenizer", default="")
    parser.add_argument("--max-rows", type=int, default=1024)
    parser.add_argument("--max-field-bytes", type=int, default=4096)
    parser.add_argument("--reserve-limit", type=int, default=512)
    parser.add_argument("--exact-trace", action="store_true", help="Compute ConvexTok tropical margins and entropy. Slower.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    rows = list(iter_rows(args.parquet, max_rows=args.max_rows))
    docs = build_docs(rows, max_field_bytes=args.max_field_bytes)
    current = ConvexTokTokenizer.load_json(args.current_tokenizer)
    payload: dict[str, Any] = {
        "schema": "toricblm.fot_tokenizer_audit.v1",
        "source_patterns": args.parquet,
        "rows": len(rows),
        "field_documents": len(docs),
        "current_tokenizer": str(Path(args.current_tokenizer).resolve()),
        "old_tokenizer": str(Path(args.old_tokenizer).resolve()) if args.old_tokenizer else "",
        "current": audit_tokenizer(current, docs, reserve_limit=args.reserve_limit, exact_trace=args.exact_trace),
    }
    if args.old_tokenizer:
        old = ConvexTokTokenizer.load_json(args.old_tokenizer)
        payload["old"] = audit_tokenizer(old, docs, reserve_limit=0, exact_trace=False)
    payload["recommendation"] = recommendation(payload)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(Path(args.output_md), payload)
    print(json.dumps({"output_json": str(out), "output_md": args.output_md, "recommendation": payload["recommendation"]}, indent=2))


if __name__ == "__main__":
    main()
