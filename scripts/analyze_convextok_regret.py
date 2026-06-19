#!/usr/bin/env python3
"""Analyze ConvexTok regret and tokenisation DAG metrics on text samples."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PG_ROOT = ROOT / "amelie-iska" / "parameter-golf"
if str(PG_ROOT) not in sys.path:
    sys.path.insert(0, str(PG_ROOT))

from convextok import ConvexTokTokenizer, analyze_tokenizer_regret, tokenisation_dag_payload


def iter_docs_jsonl(path: Path, *, max_docs: int) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(out) >= max_docs:
                break
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            if text:
                out.append(text)
    return out


def iter_docs_parquet(patterns: list[str], *, text_column: str, max_docs: int) -> list[str]:
    import pyarrow.parquet as pq

    paths: list[Path] = []
    for raw_pattern in patterns:
        matches = [Path(p).resolve() for p in glob.glob(str(Path(raw_pattern).expanduser()))]
        if not matches:
            raise FileNotFoundError(f"no parquet files matched pattern: {raw_pattern}")
        paths.extend(matches)
    out: list[str] = []
    for path in sorted({p for p in paths}):
        parquet_file = pq.ParquetFile(path)
        if text_column not in set(parquet_file.schema.names):
            raise ValueError(f"{path} does not contain text column {text_column!r}")
        for batch in parquet_file.iter_batches(batch_size=1024, columns=[text_column]):
            for value in batch.column(0).to_pylist():
                text = "" if value is None else str(value).replace("\x00", " ").strip()
                if text:
                    out.append(text)
                if len(out) >= max_docs:
                    return out
    return out


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.docs_jsonl:
        return iter_docs_jsonl(Path(args.docs_jsonl), max_docs=int(args.max_docs))
    if args.docs_parquet:
        return iter_docs_parquet(args.docs_parquet, text_column=str(args.parquet_text_column), max_docs=int(args.max_docs))
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
        return [chunk for chunk in text.split("\n\n") if chunk.strip()][: int(args.max_docs)]
    return [
        "ToricGT embeds tropical shortest-path tokenisation into a toric vocabulary polytope.",
        "The byte-boundary DAG gives exact min-plus dynamic programming features for BPB training.",
        "Graphified FineWeb examples retain sequence flattening only for OAI scoring.",
    ][: int(args.max_docs)]


def maybe_sentencepiece_metrics(texts: list[str], path: str | None) -> dict[str, float]:
    if not path:
        return {}
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=path)
    token_count = 0
    byte_count = 0
    for text in texts:
        token_count += len(sp.encode(text, out_type=int))
        byte_count += len(text.encode("utf-8", errors="replace"))
    return {
        "tokenizer_regret/current_sentencepiece_path_tokens": float(token_count),
        "tokenizer_regret/current_sentencepiece_tokens_per_byte": float(token_count / max(byte_count, 1)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convextok-tokenizer", required=True)
    parser.add_argument("--sentencepiece-tokenizer", default=None)
    parser.add_argument("--docs-jsonl", default=None)
    parser.add_argument(
        "--docs-parquet",
        action="append",
        default=[],
        help="Local Parquet glob for text samples. May be repeated.",
    )
    parser.add_argument("--parquet-text-column", default="text")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--max-docs", type=int, default=16)
    parser.add_argument("--max-doc-bytes", type=int, default=2048)
    parser.add_argument("--max-candidates", type=int, default=1500)
    parser.add_argument("--dag-sample-index", type=int, default=0)
    parser.add_argument("--dag-max-bytes", type=int, default=512)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dag-output-json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    texts = load_texts(args)
    if not texts:
        raise ValueError("no texts available for ConvexTok regret analysis")
    tok = ConvexTokTokenizer.load_json(args.convextok_tokenizer)
    metrics: dict[str, Any] = analyze_tokenizer_regret(
        texts,
        tok,
        lp_vocab_size=int(tok.vocab_size),
        max_candidates=int(args.max_candidates),
        max_doc_bytes=int(args.max_doc_bytes),
    )
    metrics.update(maybe_sentencepiece_metrics(texts, args.sentencepiece_tokenizer))
    if "tokenizer_regret/current_sentencepiece_path_tokens" in metrics:
        lower = max(float(metrics["tokenizer_regret/lp_lower_bound_tokens"]), 1e-9)
        metrics["tokenizer_regret/current_sentencepiece_gap_ratio"] = float(
            metrics["tokenizer_regret/current_sentencepiece_path_tokens"] / lower
        )
        metrics["tokenizer_regret/convextok_minus_sentencepiece_tokens"] = float(
            metrics["tokenizer_regret/convextok_path_tokens"]
            - metrics["tokenizer_regret/current_sentencepiece_path_tokens"]
        )
    payload = {
        "schema": "toricgt.convextok_regret_report.v1",
        "convextok_tokenizer": str(Path(args.convextok_tokenizer).resolve()),
        "sentencepiece_tokenizer": None if args.sentencepiece_tokenizer is None else str(Path(args.sentencepiece_tokenizer).resolve()),
        "sample_docs": len(texts),
        "metrics": metrics,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dag_output_json:
        idx = max(0, min(int(args.dag_sample_index), len(texts) - 1))
        dag = tokenisation_dag_payload(tok, texts[idx], max_bytes=int(args.dag_max_bytes))
        dag_path = Path(args.dag_output_json)
        dag_path.parent.mkdir(parents=True, exist_ok=True)
        dag_path.write_text(json.dumps(dag, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
