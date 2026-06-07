#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-toricgt_full_tokengt_got_fineweb_derived_$(date -u +%Y%m%dT%H%M%SZ)}"
SESSION="${SESSION:-$RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-/home/iska/miniconda3/envs/tokengt/bin/python}"
CONFIG="${CONFIG:-config/train.full_tokengt_got_fineweb_derived.yaml}"
LOG_DIR="${LOG_DIR:-logs/full_tokengt_got_fineweb_derived/$RUN_ID}"

mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for the default background launch" >&2
  exit 1
fi

if ! ls data/curated_hf_shards/train/*.parquet >/dev/null 2>&1; then
  echo "missing full training shard parquet files under data/curated_hf_shards/train" >&2
  exit 1
fi

if ! ls data/curated_hf_shards/validation/*.parquet >/dev/null 2>&1; then
  echo "missing validation shard parquet files under data/curated_hf_shards/validation" >&2
  exit 1
fi

if ! ls amelie-iska/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin >/dev/null 2>&1; then
  echo "missing FineWeb sp1024 train bin shards under amelie-iska/parameter-golf/data/datasets/fineweb10B_sp1024" >&2
  exit 1
fi

if [[ ! -f amelie-iska/parameter-golf/data/tokenizers/fineweb_1024_bpe.model ]]; then
  echo "missing FineWeb sp1024 SentencePiece tokenizer" >&2
  exit 1
fi

COMMAND="PYTHONPATH=src TORICGT_WANDB_RAW_MODE=off WANDB_NAME=$RUN_ID $PYTHON_BIN scripts/train.py --config $CONFIG --checkpoint-dir checkpoints/$RUN_ID"
printf '%s\n' "$COMMAND" > "$LOG_DIR/command.sh"
chmod +x "$LOG_DIR/command.sh"

tmux new-session -d -s "$SESSION" "cd '$PWD' && bash '$LOG_DIR/command.sh' 2>&1 | tee '$LOG_DIR/train.log'"
echo "started $SESSION"
