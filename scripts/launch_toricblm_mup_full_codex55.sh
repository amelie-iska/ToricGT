#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/toricblm_mup_170m_codex55.env}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  echo "Run: conda run -n tokengt env PYTHONPATH=src:external/mup:amelie-iska/parameter-golf python scripts/generate_toricblm_mup_config.py" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${RUN_ID:-toricblm-mup-codex55-full-${RUN_STAMP}}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/oai_sidecar/${RUN_ID}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints/${RUN_ID}}"
mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR"

export RUN_ID
export CHECKPOINT_DIR
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_ID}"
export WANDB_NAME="${WANDB_NAME:-$RUN_ID}"
export PYTHONPATH="$ROOT/src:$ROOT/external/mup:$ROOT/amelie-iska/parameter-golf:${PYTHONPATH:-}"

echo "launching ToricBLM muP full run"
echo "root: $ROOT"
echo "config: $CONFIG_PATH"
echo "run_id: $RUN_ID"
echo "run_dir: $RUN_DIR"
echo "checkpoint_dir: $CHECKPOINT_DIR"
echo "data_path: ${DATA_PATH:-unset}"
echo "late_graph_train_glob: ${LATE_GRAPH_TRAIN_GLOB:-unset}"
echo "mup_base_shapes: ${MUP_BASE_SHAPES:-unset}"

exec conda run -n tokengt env PYTHONPATH="$PYTHONPATH" \
  python "$ROOT/amelie-iska/parameter-golf/train_gpt.py" 2>&1 | tee "$RUN_DIR/train.log"
