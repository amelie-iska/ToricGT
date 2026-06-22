#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-toricblm-mup-codex55-full-20260622T141538Z}"
REPO_ID="${REPO_ID:-AmelieSchreiber/ToricGT_160M_FoT}"
MIN_STEP="${MIN_STEP:-25000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TMUX_SESSION="${TMUX_SESSION:-toricblm_mup_full}"
PUBLISH_LOG_DIR="${PUBLISH_LOG_DIR:-$ROOT/logs}"
mkdir -p "$PUBLISH_LOG_DIR"
LOG_PATH="$PUBLISH_LOG_DIR/watch_publish_${RUN_ID}.log"

echo "watching run_id=$RUN_ID repo=$REPO_ID min_step=$MIN_STEP poll=${POLL_SECONDS}s" | tee -a "$LOG_PATH"

while true; do
  latest_ckpt="$(ls -1 "$ROOT/checkpoints/$RUN_ID"/*_step_*.pt 2>/dev/null | sort | tail -n 1 || true)"
  latest_step="-1"
  if [[ -n "$latest_ckpt" ]]; then
    latest_step="$(python3 - "$latest_ckpt" <<'PY'
import re, sys
m = re.search(r"_step_(\d+)\.pt$", sys.argv[1])
print(int(m.group(1)) if m else -1)
PY
)"
  fi

  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) still running latest_step=$latest_step latest_ckpt=${latest_ckpt:-none}" | tee -a "$LOG_PATH"
    sleep "$POLL_SECONDS"
    continue
  fi

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) training session ended latest_step=$latest_step latest_ckpt=${latest_ckpt:-none}" | tee -a "$LOG_PATH"
  if [[ "$latest_step" -lt "$MIN_STEP" ]]; then
    echo "refusing upload: latest_step=$latest_step < min_step=$MIN_STEP" | tee -a "$LOG_PATH"
    exit 2
  fi

  conda run -n tokengt env PYTHONPATH=src:external/mup:amelie-iska/parameter-golf \
    python scripts/publish_toricgt_160m_fot_to_hf.py \
      --repo-id "$REPO_ID" \
      --run-id "$RUN_ID" \
      --min-step "$MIN_STEP" 2>&1 | tee -a "$LOG_PATH"
  exit 0
done
