#!/usr/bin/env bash
# Launch a fresh native ToricGT all-phases Parameter-Golf training run.
#
# This is the full implemented-metrics path, not the external FineWeb scaffold:
# the native trainer logs byte BPB plus topology, toric geometry, tropical,
# complexity, trajectory-memory, Koszul persistence, and late Toric BGG metrics.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/train.parameter_golf_all_phases.yaml}"
CONDA_ENV="${CONDA_ENV:-tokengt}"
PROJECT="${WANDB_PROJECT:-toricgt-parameter-golf}"
ENTITY="${WANDB_ENTITY:-amelie-iska-math}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${RUN_ID:-toricgt-all-phases-${STAMP}}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"
TRAIN_SESSION="${TRAIN_SESSION:-toricgt_all_phases_${STAMP}}"
WATCH_SESSION="${WATCH_SESSION:-${TRAIN_SESSION}_watcher}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/parameter_golf_all_phases}"
LOG_DIR="${LOG_DIR:-logs/parameter_golf_all_phases/${RUN_ID}}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-outputs/post_resume_analysis/${RUN_ID}}"
TARGET_STEP="${TARGET_STEP:-1000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

export BPB_TARGET="${BPB_TARGET:-1.2}"
export BPB_MAX_REVIEW_ITERATIONS="${BPB_MAX_REVIEW_ITERATIONS:-100}"
export BPB_LOOP_STATE="${BPB_LOOP_STATE:-outputs/bpb_codex_loop_state_all_phases.json}"
export BPB_LOOP_STOP_FILE="${BPB_LOOP_STOP_FILE:-outputs/bpb_codex_loop_stop_all_phases}"
export BPB_LOOP_NAME="${BPB_LOOP_NAME:-all_phases_native_toricgt}"

mkdir -p "$LOG_DIR" "$ANALYSIS_ROOT" "$CHECKPOINT_DIR"
START_EPOCH="$(date +%s)"
printf '%s\n' "$START_EPOCH" > "${LOG_DIR}/start_epoch.txt"

if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
  echo "training tmux session already exists: $TRAIN_SESSION" >&2
  exit 1
fi
if tmux has-session -t "$WATCH_SESSION" 2>/dev/null; then
  echo "watcher tmux session already exists: $WATCH_SESSION" >&2
  exit 1
fi

TRAIN_CMD=(
  conda run --no-capture-output -n "$CONDA_ENV" env
  PYTHONPATH=src
  WANDB_RUN_ID="$RUN_ID"
  WANDB_RESUME=allow
  python scripts/train_parameter_golf_random_order.py
  --config "$CONFIG"
  --wandb
  --wandb-project "$PROJECT"
  --wandb-run-name "$RUN_NAME"
)

WATCH_CMD=(
  conda run --no-capture-output -n "$CONDA_ENV" env
  PYTHONPATH=src
  BPB_TARGET="$BPB_TARGET"
  BPB_MAX_REVIEW_ITERATIONS="$BPB_MAX_REVIEW_ITERATIONS"
  BPB_LOOP_STATE="$BPB_LOOP_STATE"
  BPB_LOOP_STOP_FILE="$BPB_LOOP_STOP_FILE"
  BPB_LOOP_NAME="$BPB_LOOP_NAME"
  python scripts/watch_training_analysis.py
  --checkpoint-dir "$CHECKPOINT_DIR"
  --start-step 0
  --target-step "$TARGET_STEP"
  --min-mtime-unix "$START_EPOCH"
  --poll-seconds "$POLL_SECONDS"
  --run-path "${ENTITY}/${PROJECT}/${RUN_ID}"
  --output-root "$ANALYSIS_ROOT"
  --config "$CONFIG"
  --data-glob "data/curated_hf_shards/validation/*.parquet"
  --seq-len 1024
  --simplex-samples 8
  --geometry-records 4
  --geometry-branches 6
  --device cuda
  --precision bf16
  --pause-training-before-analysis
  --pause-wait-seconds 12
  --training-tmux "$TRAIN_SESSION"
  --codex-review-hook scripts/codex_training_review_resume.sh
  --codex-review-tmux-prefix toricgt_codex_review_all_phases
)

printf '%q ' "${TRAIN_CMD[@]}" > "${LOG_DIR}/train_command.sh"
printf '\n' >> "${LOG_DIR}/train_command.sh"
printf '%q ' "${WATCH_CMD[@]}" > "${LOG_DIR}/watch_command.sh"
printf '\n' >> "${LOG_DIR}/watch_command.sh"
TRAIN_CMD_STR="$(printf '%q ' "${TRAIN_CMD[@]}")"
WATCH_CMD_STR="$(printf '%q ' "${WATCH_CMD[@]}")"

tmux new-session -d -s "$TRAIN_SESSION" \
  "cd '$REPO_ROOT' && ${TRAIN_CMD_STR} 2>&1 | tee '$LOG_DIR/train.log'"
tmux new-session -d -s "$WATCH_SESSION" \
  "cd '$REPO_ROOT' && ${WATCH_CMD_STR} 2>&1 | tee '$LOG_DIR/watcher.log'"

cat <<EOF
started native ToricGT all-phases training
  config:        $CONFIG
  run id:        $RUN_ID
  wandb:         https://wandb.ai/${ENTITY}/${PROJECT}/runs/${RUN_ID}
  training tmux: $TRAIN_SESSION
  watcher tmux:  $WATCH_SESSION
  first review:  checkpoint >= step $TARGET_STEP
  logs:          $LOG_DIR
  analysis root: $ANALYSIS_ROOT
  BPB loop:      target=$BPB_TARGET max_reviews=$BPB_MAX_REVIEW_ITERATIONS
EOF
