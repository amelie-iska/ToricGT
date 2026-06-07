#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-}"
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(find checkpoints -maxdepth 1 -type d -name 'toricgt_full_tokengt_got_fineweb_derived_stable_*' -printf '%f\n' 2>/dev/null | sort | tail -n 1)"
fi
if [[ -z "$RUN_ID" ]]; then
  echo "RUN_ID is required and no stable TokenGT derived checkpoint directory was found" >&2
  exit 1
fi

TARGET_STEP="${TARGET_STEP:-1000}"
SESSION="${SESSION:-${RUN_ID}_derived_analysis_${TARGET_STEP}}"
PYTHON_BIN="${PYTHON_BIN:-/home/iska/miniconda3/envs/tokengt/bin/python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/$RUN_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/post_resume_analysis/$RUN_ID}"
RUN_PATH="${RUN_PATH:-}"
LOG_DIR="${LOG_DIR:-logs/full_tokengt_got_fineweb_derived/$RUN_ID/supervisor}"

mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for the default background watcher" >&2
  exit 1
fi

COMMAND=(env PYTHONPATH=src "$PYTHON_BIN" scripts/watch_training_analysis.py
  --checkpoint-dir "$CHECKPOINT_DIR"
  --start-step 0
  --target-step "$TARGET_STEP"
  --output-root "$OUTPUT_ROOT"
  --config config/train.full_tokengt_got_fineweb_derived.yaml
  --data-glob 'data/curated_hf_shards/validation/*.parquet'
  --device cpu
  --precision fp32
  --derived-category-example-dir "$CHECKPOINT_DIR/derived_category_examples"
  --derived-category-max-files 8
  --derived-category-max-objects 24
  --memory-trace-example-dir "$CHECKPOINT_DIR/memory_trace_examples"
  --memory-trace-max-files 8
  --memory-trace-max-queries 96)

if [[ -n "$RUN_PATH" ]]; then
  COMMAND+=(--run-path "$RUN_PATH")
fi

printf '%q ' "${COMMAND[@]}" > "$LOG_DIR/derived_analysis_watch_${TARGET_STEP}.sh"
printf '\n' >> "$LOG_DIR/derived_analysis_watch_${TARGET_STEP}.sh"
chmod +x "$LOG_DIR/derived_analysis_watch_${TARGET_STEP}.sh"

tmux new-session -d -s "$SESSION" "cd '$PWD' && bash '$LOG_DIR/derived_analysis_watch_${TARGET_STEP}.sh' 2>&1 | tee '$LOG_DIR/derived_analysis_watch_${TARGET_STEP}.log'"
echo "started $SESSION"
