#!/usr/bin/env python3
"""Compare ConvexTok token count and coverage deltas on local corpora."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PARAMETER_GOLF = ROOT / "amelie-iska" / "parameter-golf"
if str(PARAMETER_GOLF) not in sys.path:
    sys.path.insert(0, str(PARAMETER_GOLF))

from convextok import ConvexTokTokenizer, PRICED_OFFSET  # noqa: E402


def clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", " ").strip()


def iter_parquet_texts(pattern: str, *, text_column: str, max_docs: int, max_doc_bytes: int) -> Iterable[str]:
    import pyarrow.parquet as pq

    emitted = 0
    for raw in sorted(glob.glob(str(Path(pattern).expanduser()))):
        path = Path(raw)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1024, columns=[text_column]):
            for value in batch.column(0).to_pylist():
                text = clean_text(value)
                if not text:
                    continue
                data = text.encode("utf-8", errors="replace")
                if len(data) > max_doc_bytes:
                    text = data[:max_doc_bytes].decode("utf-8", errors="ignore").strip()
                if not text:
                    continue
                yield text
                emitted += 1
                if emitted >= max_docs:
                    return


def iter_jsonl_texts(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            text = clean_text(payload.get("text", ""))
            if text:
                yield text


def summarize(name: str, old_tok: ConvexTokTokenizer, new_tok: ConvexTokTokenizer, texts: list[str]) -> dict[str, float | int | str]:
    old_counts: list[int] = []
    new_counts: list[int] = []
    byte_counts: list[int] = []
    old_priced: list[float] = []
    new_priced: list[float] = []
    for text in texts:
        data = text.encode("utf-8", errors="replace")
        old_ids = old_tok.encode(text)
        new_ids = new_tok.encode(text)
        old_counts.append(len(old_ids))
        new_counts.append(len(new_ids))
        byte_counts.append(len(data))
        old_priced.append(sum(1 for token_id in old_ids if token_id >= PRICED_OFFSET) / max(len(old_ids), 1))
        new_priced.append(sum(1 for token_id in new_ids if token_id >= PRICED_OFFSET) / max(len(new_ids), 1))
    total_old = sum(old_counts)
    total_new = sum(new_counts)
    total_bytes = sum(byte_counts)
    doc_deltas = [100.0 * (new / old - 1.0) for old, new in zip(old_counts, new_counts, strict=True) if old]
    return {
        "corpus": name,
        "documents": len(texts),
        "bytes": total_bytes,
        "old_tokens": total_old,
        "new_tokens": total_new,
        "token_delta_pct": 100.0 * (total_new / max(total_old, 1) - 1.0),
        "old_bytes_per_token": total_bytes / max(total_old, 1),
        "new_bytes_per_token": total_bytes / max(total_new, 1),
        "old_priced_fraction_mean": statistics.mean(old_priced) if old_priced else 0.0,
        "new_priced_fraction_mean": statistics.mean(new_priced) if new_priced else 0.0,
        "median_doc_token_delta_pct": statistics.median(doc_deltas) if doc_deltas else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-tokenizer", required=True)
    parser.add_argument("--new-tokenizer", required=True)
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--seed-jsonl", required=True)
    parser.add_argument("--max-docs", type=int, default=1024)
    parser.add_argument("--max-doc-bytes", type=int, default=4096)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    old_tok = ConvexTokTokenizer.load_json(args.old_tokenizer)
    new_tok = ConvexTokTokenizer.load_json(args.new_tokenizer)
    corpora = [
        (
            "curated_train_sample",
            list(
                iter_parquet_texts(
                    args.parquet_glob,
                    text_column=str(args.text_column),
                    max_docs=int(args.max_docs),
                    max_doc_bytes=int(args.max_doc_bytes),
                )
            ),
        ),
        ("biomed_universal_seed", list(iter_jsonl_texts(Path(args.seed_jsonl).expanduser().resolve()))),
    ]
    results = [summarize(name, old_tok, new_tok, texts) for name, texts in corpora]
    payload = {"old_tokenizer": args.old_tokenizer, "new_tokenizer": args.new_tokenizer, "results": results}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
