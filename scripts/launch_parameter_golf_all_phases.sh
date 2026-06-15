#!/usr/bin/env bash
# Launch a supervised native ToricGT all-phases Parameter-Golf training run.
#
# This is the full implemented-metrics path, not the external FineWeb scaffold:
# the native trainer logs byte BPB plus topology, toric geometry, tropical,
# complexity, trajectory-memory, Koszul persistence, and late Toric BGG metrics.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/train.parameter_golf_all_phases.yaml}"
CONDA_ENV="${CONDA_ENV:-tokengt}"
CONDA_BIN="${CONDA_BIN:-}"
if [[ -z "$CONDA_BIN" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [[ -x /home/iska/miniconda3/bin/conda ]]; then
    CONDA_BIN="/home/iska/miniconda3/bin/conda"
  else
    CONDA_BIN="conda"
  fi
fi
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PROJECT="${WANDB_PROJECT:-toricgt-parameter-golf}"
ENTITY="${WANDB_ENTITY:-amelie-iska-math}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${RUN_ID:-toricgt-sequential-all-phases-${STAMP}}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"
TRAIN_SESSION="${TRAIN_SESSION:-toricgt_all_phases_${STAMP}}"
WATCH_SESSION="${WATCH_SESSION:-${TRAIN_SESSION}_analysis}"
SUPERVISOR_SESSION="${SUPERVISOR_SESSION:-${TRAIN_SESSION}_supervisor}"
LOG_DIR="${LOG_DIR:-logs/parameter_golf_all_phases/${RUN_ID}}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-outputs/post_resume_analysis/${RUN_ID}}"
CAS_TARGET_DIR="${CAS_TARGET_DIR:-outputs/cas_training_targets/${RUN_ID}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/${RUN_ID}}"
if [[ -z "${CHECKPOINT_INTERVAL:-}" ]]; then
  CHECKPOINT_INTERVAL="$("$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print(int(((cfg.get("analysis") or {}).get("periodic_interval_steps")) or 250))
PY
)"
fi
POLL_SECONDS="${POLL_SECONDS:-60}"
ENABLE_CODEX_REVIEW="${ENABLE_CODEX_REVIEW:-1}"

export BPB_TARGET="${BPB_TARGET:-1.2}"
export BPB_GATE_STEP="${BPB_GATE_STEP:-4000}"
export BPB_MAX_REVIEW_ITERATIONS="${BPB_MAX_REVIEW_ITERATIONS:-100}"
export BPB_LOOP_STATE="${BPB_LOOP_STATE:-outputs/${RUN_ID}_bpb_codex_loop_state.json}"
export BPB_LOOP_STOP_FILE="${BPB_LOOP_STOP_FILE:-outputs/${RUN_ID}_bpb_codex_loop_stop}"
export BPB_LOOP_NAME="${BPB_LOOP_NAME:-all_phases_native_toricgt}"

mkdir -p "$LOG_DIR" "$ANALYSIS_ROOT" "$CHECKPOINT_DIR" "$CAS_TARGET_DIR"
START_EPOCH="$(date +%s)"
printf '%s\n' "$START_EPOCH" > "${LOG_DIR}/start_epoch.txt"

"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH=src \
  python scripts/build_toric_tropical_certificates.py \
    --output-dir "$CAS_TARGET_DIR" \
    --all-exact-cas \
    --require-cas \
  > "${LOG_DIR}/cas_training_targets.log" 2>&1

CAS_TORIC_IDEAL_CERT="$("$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python - "$CAS_TARGET_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "cache"
for path in sorted(root.glob("*/*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") == "macaulay2_toric_ideal_certificate":
        print(path)
        raise SystemExit(0)
raise SystemExit("macaulay2_toric_ideal_certificate was not generated")
PY
)"
printf '%s\n' "$CAS_TORIC_IDEAL_CERT" > "${LOG_DIR}/cas_toric_ideal_certificate_path.txt"

if tmux has-session -t "$SUPERVISOR_SESSION" 2>/dev/null; then
  echo "supervisor tmux session already exists: $SUPERVISOR_SESSION" >&2
  exit 1
fi

SUPERVISOR_CMD=(
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" env
  PYTHONPATH=src
  WANDB_ENTITY="$ENTITY"
  WANDB_PROJECT="$PROJECT"
  BPB_TARGET="$BPB_TARGET"
  BPB_MAX_REVIEW_ITERATIONS="$BPB_MAX_REVIEW_ITERATIONS"
  PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
  TORICGT_CAS_TORIC_IDEAL_CERT="$CAS_TORIC_IDEAL_CERT"
  CONDA_BIN="$CONDA_BIN"
  python scripts/supervise_parameter_golf_training.py
  --config "$CONFIG"
  --run-id "$RUN_ID"
  --run-name "$RUN_NAME"
  --wandb-entity "$ENTITY"
  --wandb-project "$PROJECT"
  --conda-env "$CONDA_ENV"
  --conda-bin "$CONDA_BIN"
  --train-session "$TRAIN_SESSION"
  --watch-session "$WATCH_SESSION"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --log-root "$LOG_DIR/supervisor"
  --analysis-root "$ANALYSIS_ROOT"
  --poll-seconds "$POLL_SECONDS"
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --target-bpb "$BPB_TARGET"
  --gate-step "$BPB_GATE_STEP"
  --max-analysis-iterations "$BPB_MAX_REVIEW_ITERATIONS"
)
if [[ "$ENABLE_CODEX_REVIEW" != "0" && "$ENABLE_CODEX_REVIEW" != "false" && "$ENABLE_CODEX_REVIEW" != "False" ]]; then
  SUPERVISOR_CMD+=(--enable-codex-review)
fi

printf '%q ' "${SUPERVISOR_CMD[@]}" > "${LOG_DIR}/supervisor_command.sh"
printf '\n' >> "${LOG_DIR}/supervisor_command.sh"
SUPERVISOR_CMD_STR="$(printf '%q ' "${SUPERVISOR_CMD[@]}")"

tmux new-session -d -s "$SUPERVISOR_SESSION" \
  "cd '$REPO_ROOT' && ${SUPERVISOR_CMD_STR} 2>&1 | tee '$LOG_DIR/supervisor.log'"

cat <<EOF
started supervised native ToricGT all-phases training
  config:        $CONFIG
  run id:        $RUN_ID
  wandb:         https://wandb.ai/${ENTITY}/${PROJECT}/runs/${RUN_ID}
  supervisor:    $SUPERVISOR_SESSION
  training tmux: $TRAIN_SESSION
  analysis tmux: $WATCH_SESSION
  review cadence: every $CHECKPOINT_INTERVAL checkpoint steps
  codex review:  ${ENABLE_CODEX_REVIEW}
  logs:          $LOG_DIR
  analysis root: $ANALYSIS_ROOT
  CAS targets:    $CAS_TARGET_DIR
  CAS toric ideal certificate: $CAS_TORIC_IDEAL_CERT
  BPB loop:      target=$BPB_TARGET gate_step=$BPB_GATE_STEP max_reviews=$BPB_MAX_REVIEW_ITERATIONS
EOF
