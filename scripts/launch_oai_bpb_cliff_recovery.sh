#!/usr/bin/env bash
# Launch the BPB cliff recovery run from the robust early-basin checkpoint and
# start the first interrupting GPU analysis/Codex-review gate.
#
# Run:
#   scripts/launch_oai_bpb_cliff_recovery.sh
#
# Optional:
#   RESUME_CKPT=checkpoints/parameter_golf_oai_dense/random_order_step_00001500.pt \
#   START_STEP=1500 \
#   TARGET_STEP=1700 \
#   scripts/launch_oai_bpb_cliff_recovery.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/train.parameter_golf_random_order_dense_valmix35_from1000.yaml}"
RESUME_CKPT="${RESUME_CKPT:-checkpoints/parameter_golf_oai_dense/random_order_step_00001500.pt}"
START_STEP="${START_STEP:-1500}"
TARGET_STEP="${TARGET_STEP:-1700}"
PROJECT="${WANDB_PROJECT:-toricgt-parameter-golf}"
ENTITY="${WANDB_ENTITY:-amelie-iska-math}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-oai-bpb-revealed-context-01500-${STAMP}}"
RUN_ID="${WANDB_RUN_ID:-oai01500revealed${STAMP//[^0-9A-Za-z]/}}"
TRAIN_TMUX="${TRAIN_TMUX:-toricgt_oai_bpb_revealed_01500_${STAMP}}"
WATCH_TMUX="${WATCH_TMUX:-toricgt_watch_bpb_revealed_01500_${STAMP}}"
CODEX_PREFIX="${CODEX_PREFIX:-toricgt_codex_review_bpb_cliff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/post_resume_analysis/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-logs/training}"
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}.log"
WATCH_LOG="${LOG_DIR}/${RUN_NAME}.watcher.log"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
date +%s > "${OUTPUT_ROOT}/start_epoch.txt"

if [[ ! -f "$RESUME_CKPT" ]]; then
  echo "missing checkpoint: $RESUME_CKPT" >&2
  exit 1
fi

for session in \
  toricgt_oai_graphcg_gfn_03600_recapture_20260602T142446Z \
  toricgt_watch_03600_recapture_20260602T142446Z \
  toricgt_oai_bpb_cliff_02000_20260602T143707Z \
  toricgt_watch_bpb_cliff_02000_20260602T143707Z; do
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux send-keys -t "$session" C-c || true
    sleep 2
  fi
done

if tmux has-session -t "$TRAIN_TMUX" 2>/dev/null; then
  echo "training tmux already exists: $TRAIN_TMUX" >&2
  exit 1
fi
if tmux has-session -t "$WATCH_TMUX" 2>/dev/null; then
  echo "watcher tmux already exists: $WATCH_TMUX" >&2
  exit 1
fi

TRAIN_CMD=(
  conda run --no-capture-output -n tokengt env
  "PYTHONPATH=src"
  "WANDB_RUN_ID=${RUN_ID}"
  "WANDB_RESUME=allow"
  python scripts/train_parameter_golf_random_order.py
  --config "$CONFIG"
  --resume "$RESUME_CKPT"
  --wandb
  --wandb-project "$PROJECT"
  --wandb-run-name "$RUN_NAME"
)

WATCH_CMD=(
  conda run --no-capture-output -n tokengt env
  "PYTHONPATH=src"
  python scripts/watch_training_analysis.py
  --checkpoint-dir checkpoints/parameter_golf_oai_dense
  --start-step "$START_STEP"
  --target-step "$TARGET_STEP"
  --run-path "${ENTITY}/${PROJECT}/${RUN_ID}"
  --output-root "$OUTPUT_ROOT"
  --min-mtime-unix "$(cat "${OUTPUT_ROOT}/start_epoch.txt")"
  --poll-seconds 30
  --config "$CONFIG"
  --device cuda
  --precision bf16
  --simplex-samples 8
  --geometry-records 4
  --geometry-branches 6
  --training-tmux "$TRAIN_TMUX"
  --pause-training-before-analysis
  --pause-wait-seconds 12
  --codex-review-hook scripts/codex_training_review_resume.sh
  --codex-review-tmux-prefix "$CODEX_PREFIX"
)

printf '%q ' "${TRAIN_CMD[@]}" > "${LOG_DIR}/${RUN_NAME}.command.sh"
printf '\n' >> "${LOG_DIR}/${RUN_NAME}.command.sh"
printf '%q ' "${WATCH_CMD[@]}" > "${LOG_DIR}/${RUN_NAME}.watcher.command.sh"
printf '\n' >> "${LOG_DIR}/${RUN_NAME}.watcher.command.sh"
chmod 700 "${LOG_DIR}/${RUN_NAME}.command.sh" "${LOG_DIR}/${RUN_NAME}.watcher.command.sh"

tmux new-session -d -s "$TRAIN_TMUX" "cd '$REPO_ROOT' && bash '${LOG_DIR}/${RUN_NAME}.command.sh' 2>&1 | tee '$TRAIN_LOG'"
tmux new-session -d -s "$WATCH_TMUX" "cd '$REPO_ROOT' && bash '${LOG_DIR}/${RUN_NAME}.watcher.command.sh' 2>&1 | tee '$WATCH_LOG'"

cat <<EOF
started BPB revealed-context cliff recovery
training tmux: $TRAIN_TMUX
watcher tmux:  $WATCH_TMUX
run name:      $RUN_NAME
resume ckpt:   $RESUME_CKPT
target step:   $TARGET_STEP
wandb path:    ${ENTITY}/${PROJECT}/${RUN_ID}
training log:  $TRAIN_LOG
watcher log:   $WATCH_LOG
analysis root: $OUTPUT_ROOT
EOF
