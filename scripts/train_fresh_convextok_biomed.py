#!/usr/bin/env python3
"""Train a fresh ConvexTok vocabulary with a biomedical modality reserve.

The tokenizer is new: it does not preserve or copy an older ConvexTok
vocabulary.  A small reserve of exact biomedical/control tokens is included so
future UniProt, structure, atomistic, cell, and graph-control streams have
stable byte-exact tokens, while the remaining budget is selected by ConvexTok's
LP/rounding procedure on the current corpus sample.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PARAMETER_GOLF = ROOT / "amelie-iska" / "parameter-golf"
if str(PARAMETER_GOLF) not in sys.path:
    sys.path.insert(0, str(PARAMETER_GOLF))

from convextok import (  # noqa: E402
    ConvexTokTokenizer,
    PRICED_OFFSET,
    collect_candidates,
    rounded_candidate_indices,
    solve_convextok_lp,
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", " ").strip()


def iter_seed_texts(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            text = clean_text(payload.get("text", ""))
            if text:
                rows.append(text)
    return rows


def iter_parquet_texts(patterns: list[str], *, text_column: str, max_docs: int, max_doc_bytes: int) -> Iterable[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to sample Parquet text") from exc

    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(p).resolve() for p in glob.glob(str(Path(pattern).expanduser())))
    if not paths:
        raise FileNotFoundError(f"no parquet files matched {patterns}")
    emitted = 0
    for path in sorted(set(paths)):
        parquet_file = pq.ParquetFile(path)
        if text_column not in set(parquet_file.schema.names):
            raise ValueError(f"{path} does not contain text column {text_column!r}")
        for batch in parquet_file.iter_batches(batch_size=1024, columns=[text_column]):
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


def seed_token_bytes(seed_texts: list[str], *, max_token_bytes: int, limit: int) -> list[bytes]:
    pattern = re.compile(r"<[^>\s]+>|[A-Za-z0-9_:+./@\\-]{2,}")
    out: list[bytes] = []
    seen: set[bytes] = set()
    for text in seed_texts:
        for piece in pattern.findall(text):
            data = piece.encode("utf-8", errors="replace")
            if not (2 <= len(data) <= max_token_bytes):
                continue
            if data in seen:
                continue
            seen.add(data)
            out.append(data)
            if len(out) >= limit:
                return out
    return out


def token_preview(data: bytes) -> str:
    return data[:80].decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-tokenizer", required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--seed-jsonl", required=True)
    parser.add_argument("--seed-token-limit", type=int, default=384)
    parser.add_argument("--max-seed-token-bytes", type=int, default=72)
    parser.add_argument("--parquet-glob", action="append", required=True)
    parser.add_argument("--parquet-text-column", default="text")
    parser.add_argument("--sample-docs", type=int, default=512)
    parser.add_argument("--max-doc-bytes", type=int, default=2048)
    parser.add_argument("--max-candidates", type=int, default=9000)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--selection", default="lp", choices=["lp", "frequency"])
    parser.add_argument("--rounding", default="det", choices=["det", "bias", "int"])
    args = parser.parse_args()

    seed_path = Path(args.seed_jsonl).expanduser().resolve()
    out_path = Path(args.output_tokenizer).expanduser().resolve()
    seed_texts = iter_seed_texts(seed_path)
    reserve = seed_token_bytes(
        seed_texts,
        max_token_bytes=int(args.max_seed_token_bytes),
        limit=max(0, int(args.seed_token_limit)),
    )
    reserve_set = set(reserve)

    corpus_texts = list(
        iter_parquet_texts(
            args.parquet_glob,
            text_column=str(args.parquet_text_column),
            max_docs=int(args.sample_docs),
            max_doc_bytes=int(args.max_doc_bytes),
        )
    )
    candidates, docs = collect_candidates(
        corpus_texts,
        min_len=2,
        max_len=int(args.max_len),
        max_candidates=int(args.max_candidates),
        max_docs=None,
        max_doc_bytes=int(args.max_doc_bytes),
    )
    candidates = [candidate for candidate in candidates if candidate.data not in reserve_set]
    priced_budget = int(args.vocab_size) - PRICED_OFFSET
    if priced_budget <= len(reserve):
        raise ValueError(f"seed reserve {len(reserve)} leaves no corpus ConvexTok budget")
    corpus_budget = priced_budget - len(reserve)
    if str(args.selection) == "lp":
        lp_scores, objective, status, success = solve_convextok_lp(docs, candidates, budget=corpus_budget)
        selected_indices = rounded_candidate_indices(
            candidates,
            lp_scores,
            budget=corpus_budget,
            rounding=str(args.rounding),
        )
        selected_set = set(selected_indices)
        if len(selected_indices) < corpus_budget:
            remaining = [i for i in range(len(candidates)) if i not in selected_set]
            remaining.sort(key=lambda i: (-candidates[i].score, -candidates[i].length, candidates[i].data))
            selected_indices.extend(remaining[: corpus_budget - len(selected_indices)])
    else:
        lp_scores = [0.0] * len(candidates)
        objective = float("nan")
        status = "frequency_ranked_no_lp"
        success = True
        selected_indices = list(range(min(corpus_budget, len(candidates))))
    corpus_tokens = [candidates[i].data for i in selected_indices]
    corpus_scores = [float(lp_scores[i]) if i < len(lp_scores) else 0.0 for i in selected_indices]
    priced = reserve + corpus_tokens
    scores = [1.0] * len(reserve) + corpus_scores
    if len(priced) != priced_budget:
        raise RuntimeError(f"expected {priced_budget} priced tokens, got {len(priced)}")

    metadata = {
        "training_algorithm": "fresh_convextok_sparse_lp_with_biomed_reserve",
        "rounding": str(args.rounding),
        "vocab_size": int(args.vocab_size),
        "priced_budget": int(priced_budget),
        "seed_reserve_count": int(len(reserve)),
        "corpus_lp_token_count": int(len(corpus_tokens)),
        "selection": str(args.selection),
        "lp_objective": float(objective),
        "lp_status": str(status),
        "lp_success": bool(success),
        "candidate_count": int(len(candidates)),
        "sample_doc_count": int(len(docs)),
        "sample_byte_count": int(sum(len(doc) for doc in docs)),
        "seed_jsonl": str(seed_path),
        "parquet_globs": list(args.parquet_glob),
        "fresh_vocabulary": True,
        "base_tokenizer_reused": False,
    }
    tok = ConvexTokTokenizer(
        vocab_size=int(args.vocab_size),
        priced_tokens=priced,
        lp_scores=scores,
        rounding=str(args.rounding),
        metadata=metadata,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save_json(out_path)
    print(json.dumps({"output_tokenizer": str(out_path), **metadata}, indent=2))
    print("seed_preview:")
    for token in reserve[:20]:
        print(f"  {token_preview(token)}")


if __name__ == "__main__":
    main()
