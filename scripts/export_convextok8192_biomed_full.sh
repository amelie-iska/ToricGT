#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/parameter_golf_convextok8192_biomed_det_full}"
TOKENIZER_PATH="$OUTPUT_ROOT/tokenizers/fineweb_convextok_8192_biomed_det.convextok.json"
SEED_JSONL="$ROOT/amelie-iska/parameter-golf/data/convextok_biomed_universal_modality_seed.jsonl"

mkdir -p "$OUTPUT_ROOT/tokenizers" "$OUTPUT_ROOT/datasets"

if [[ ! -f "$TOKENIZER_PATH" ]]; then
  conda run --no-capture-output -n tokengt env PYTHONPATH="$ROOT/src:$ROOT/amelie-iska/parameter-golf" \
    python "$ROOT/scripts/train_fresh_convextok_biomed.py" \
      --output-tokenizer "$TOKENIZER_PATH" \
      --vocab-size 8192 \
      --seed-jsonl "$SEED_JSONL" \
      --seed-token-limit 512 \
      --parquet-glob "$DATA_ROOT/curated_hf_shards/train/*.parquet" \
      --sample-docs 384 \
      --max-doc-bytes 2048 \
      --max-candidates 8500 \
      --max-len 32 \
      --selection frequency
fi

MATCHED_FINEWEB_CONVEXTOK_ENCODE_WORKERS="${MATCHED_FINEWEB_CONVEXTOK_ENCODE_WORKERS:-24}"
MATCHED_FINEWEB_CONVEXTOK_ENCODE_BATCH_DOCS="${MATCHED_FINEWEB_CONVEXTOK_ENCODE_BATCH_DOCS:-128}"
export MATCHED_FINEWEB_CONVEXTOK_ENCODE_WORKERS
export MATCHED_FINEWEB_CONVEXTOK_ENCODE_BATCH_DOCS

conda run --no-capture-output -n tokengt env PYTHONPATH="$ROOT/amelie-iska/parameter-golf" \
  python "$ROOT/amelie-iska/parameter-golf/data/download_hf_docs_and_tokenize.py" \
    --output-root "$OUTPUT_ROOT" \
    --tokenizer-config "$ROOT/amelie-iska/parameter-golf/data/convextok_8192_biomed_det_reuse_tokenizer_specs.json" \
    --skip-byte \
    --docs-parquet-train-glob-local "$DATA_ROOT/curated_hf_shards/train/*.parquet" \
    --docs-parquet-val-glob-local "$DATA_ROOT/curated_hf_shards/validation/*.parquet" \
    --parquet-text-column text \
    --max-export-doc-bytes 8192 \
    --chunk-tokens 50000000
