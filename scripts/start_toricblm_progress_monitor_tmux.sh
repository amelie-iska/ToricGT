#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 ]]; then
  RUN_DIR="$1"
else
  RUN_DIR="$(find "$ROOT/runs/oai_sidecar" -maxdepth 2 -type d -name 'epoch_001_structure_current' | sort | tail -1)"
fi
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "No structure-priority epoch directory found; pass RUN_DIR explicitly." >&2
  exit 1
fi
RUN_ID="$(basename "$(dirname "$RUN_DIR")")"
SESSION="${TORICBLM_PROGRESS_TMUX_SESSION:-toricblm_progress_monitor_${RUN_ID}}"
INTERVAL="${TORICBLM_PROGRESS_INTERVAL:-15}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux progress monitor already running: $SESSION"
  echo "attach: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && conda run --no-capture-output -n tokengt python scripts/monitor_toricblm_training_progress.py --run-dir '$RUN_DIR' --interval '$INTERVAL'"

echo "started tmux progress monitor: $SESSION"
echo "attach: tmux attach -t $SESSION"
