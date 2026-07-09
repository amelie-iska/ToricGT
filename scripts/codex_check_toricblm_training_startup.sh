#!/usr/bin/env bash
# Wait for the ToricBLM full training run to start, then hand a status bundle to
# Codex.  The Codex prompt asks for a concise report and explicitly authorizes
# repairs/restart if startup failed or the run is unhealthy.

set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/toricblm_mup_170m_convextok8192_biomed_fot_structure.env}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-$ROOT/scripts/launch_toricblm_mup_full_codex55.sh}"
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_MODEL="${CODEX_MODEL:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-21600}"
STARTUP_GRACE_SECONDS="${STARTUP_GRACE_SECONDS:-600}"
CHECK_ROOT="${CHECK_ROOT:-$ROOT/training_notes/toricblm_startup_codex_checks}"
TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  scripts/codex_check_toricblm_training_startup.sh [options]

Options:
  --poll-seconds N          Poll interval while waiting for training. Default: 60.
  --max-wait-seconds N      Max time to wait for training to appear. Default: 21600.
  --startup-grace-seconds N Wait this long after training appears before Codex. Default: 600.
  --config PATH             ToricBLM env config path.
  --launch-script PATH      Script Codex should use if restart is needed.
  --codex-bin PATH          Codex CLI binary. Default: codex.
  --codex-model NAME        Optional Codex model override.
  --tmux-session NAME       Launch this watchdog in tmux and return immediately.
  --dry-run                 Write prompt/report bundle but do not call Codex.
  -h, --help                Show this help.

Environment variables mirror the option names in uppercase where practical.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --max-wait-seconds) MAX_WAIT_SECONDS="$2"; shift 2 ;;
    --startup-grace-seconds) STARTUP_GRACE_SECONDS="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --launch-script) LAUNCH_SCRIPT="$2"; shift 2 ;;
    --codex-bin) CODEX_BIN="$2"; shift 2 ;;
    --codex-model) CODEX_MODEL="$2"; shift 2 ;;
    --tmux-session) TMUX_SESSION_NAME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "$TMUX_SESSION_NAME" ]]; then
  if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $TMUX_SESSION_NAME"
    exit 0
  fi
  quoted=(
    "cd '$ROOT' &&"
    "CONFIG_PATH='${CONFIG_PATH}'"
    "LAUNCH_SCRIPT='${LAUNCH_SCRIPT}'"
    "CODEX_BIN='${CODEX_BIN}'"
    "CODEX_MODEL='${CODEX_MODEL}'"
    "POLL_SECONDS='${POLL_SECONDS}'"
    "MAX_WAIT_SECONDS='${MAX_WAIT_SECONDS}'"
    "STARTUP_GRACE_SECONDS='${STARTUP_GRACE_SECONDS}'"
    "CHECK_ROOT='${CHECK_ROOT}'"
    "bash '$ROOT/scripts/codex_check_toricblm_training_startup.sh'"
  )
  log="$ROOT/logs/${TMUX_SESSION_NAME}.log"
  mkdir -p "$ROOT/logs"
  tmux new-session -d -s "$TMUX_SESSION_NAME" "${quoted[*]} 2>&1 | tee '$log'"
  echo "started tmux watchdog: $TMUX_SESSION_NAME"
  echo "log: $log"
  exit 0
fi

mkdir -p "$CHECK_ROOT" "$ROOT/training_notes" "$ROOT/logs"

training_processes() {
  ps -eo pid,ppid,stat,pcpu,pmem,etime,args \
    | awk '/train_gpt\\.py/ && !/awk/ {print}'
}

latest_file_for_pattern() {
  local pattern="$1"
  # shellcheck disable=SC2086
  ls -1t $pattern 2>/dev/null | head -1 || true
}

echo "waiting for ToricBLM training process..."
start_epoch="$(date +%s)"
while true; do
  if training_processes | grep -q 'train_gpt.py'; then
    break
  fi
  now="$(date +%s)"
  waited=$((now - start_epoch))
  if (( waited >= MAX_WAIT_SECONDS )); then
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    report_dir="$CHECK_ROOT/$stamp"
    mkdir -p "$report_dir"
    {
      echo "# ToricBLM Startup Watchdog Timeout"
      echo
      echo "- UTC: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
      echo "- Waited seconds: $waited"
      echo "- Config: $CONFIG_PATH"
      echo "- Launch script: $LAUNCH_SCRIPT"
      echo
      echo "No train_gpt.py process appeared before timeout."
    } | tee "$report_dir/STATUS.md"
    exit 3
  fi
  echo "training not active yet; waited ${waited}s, polling again in ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done

echo "training detected; waiting startup grace period: ${STARTUP_GRACE_SECONDS}s"
sleep "$STARTUP_GRACE_SECONDS"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
report_dir="$CHECK_ROOT/$stamp"
mkdir -p "$report_dir"

latest_watch_log="$(latest_file_for_pattern "$ROOT/logs/toricblm_nonprotein_fot_watch_v2_*.log")"
latest_train_log="$(latest_file_for_pattern "$ROOT/logs/toricblm_full_fot_split_safe_*.log")"
latest_launch_log="$(latest_file_for_pattern "$ROOT/logs/toricblm-mup-*.log")"

{
  echo "# ToricBLM Startup Status"
  echo
  echo "- UTC: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "- Root: $ROOT"
  echo "- Config: $CONFIG_PATH"
  echo "- Launch script: $LAUNCH_SCRIPT"
  echo "- Latest watcher log: ${latest_watch_log:-none}"
  echo "- Latest train log: ${latest_train_log:-${latest_launch_log:-none}}"
  echo
  echo "## Training Processes"
  echo '```text'
  training_processes || true
  echo '```'
  echo
  echo "## tmux Sessions"
  echo '```text'
  tmux list-sessions 2>/dev/null || true
  echo '```'
  echo
  echo "## Disk"
  echo '```text'
  df -h "$ROOT" || true
  du -sh "$ROOT/data/toricblm_nonprotein_fot_splits/v2_parallel" 2>/dev/null || true
  echo '```'
  echo
  echo "## GPU"
  echo '```text'
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || nvidia-smi 2>/dev/null || true
  echo '```'
} | tee "$report_dir/STATUS.md"

tmux list-sessions > "$report_dir/tmux_sessions.txt" 2>&1 || true
ps -eo pid,ppid,stat,pcpu,pmem,etime,args > "$report_dir/processes.txt" 2>&1 || true
df -h "$ROOT" > "$report_dir/disk.txt" 2>&1 || true
nvidia-smi > "$report_dir/nvidia_smi.txt" 2>&1 || true

if [[ -f "$CONFIG_PATH" ]]; then
  cp "$CONFIG_PATH" "$report_dir/active_config.env"
fi

if [[ -n "$latest_watch_log" && -f "$latest_watch_log" ]]; then
  tail -400 "$latest_watch_log" > "$report_dir/latest_watcher_tail.log" || true
fi
if [[ -n "$latest_train_log" && -f "$latest_train_log" ]]; then
  tail -800 "$latest_train_log" > "$report_dir/latest_training_tail.log" || true
elif [[ -n "$latest_launch_log" && -f "$latest_launch_log" ]]; then
  tail -800 "$latest_launch_log" > "$report_dir/latest_training_tail.log" || true
fi

(
  cd "$ROOT"
  conda run --no-capture-output -n tokengt python scripts/finalize_nonprotein_fot_manifest.py \
    --out-dir data/toricblm_nonprotein_fot_splits/v2_parallel \
    --min-train-rows 450000
) > "$report_dir/nonprotein_manifest_check.log" 2>&1 || true

cat > "$report_dir/codex_prompt.md" <<EOF
You are Codex running inside the ToricGT project at:

  $ROOT

Task: check the ToricBLM training run that just started, print a concise but complete status/progress report, and repair/restart training if startup failed or any serious errors are present.

Use these local evidence files first:

- Status summary: $report_dir/STATUS.md
- Process list: $report_dir/processes.txt
- tmux sessions: $report_dir/tmux_sessions.txt
- Disk: $report_dir/disk.txt
- GPU: $report_dir/nvidia_smi.txt
- Active config copy: $report_dir/active_config.env
- Nonprotein manifest check: $report_dir/nonprotein_manifest_check.log
- Watcher log tail, if present: $report_dir/latest_watcher_tail.log
- Training log tail, if present: $report_dir/latest_training_tail.log

Rules and requirements:

1. Do not expose or print secrets. In particular, do not read or print keys.txt.
2. Determine whether training is actually running, whether it is logging, whether W&B is likely initialized, whether GPU/VRAM use is plausible, and whether any preflight/data/config errors are visible.
3. Report current step/progress, losses/BPB if present, active tmux sessions, latest checkpoint/log path, GPU memory, disk headroom, and any errors.
4. If training is healthy, do not restart it. Write a report in training_notes with a timestamped filename and print the same summary.
5. If training failed, is missing, crashed, or is blocked by a repairable issue, repair the issue, then restart using:

   CONFIG_PATH=$CONFIG_PATH bash $LAUNCH_SCRIPT

   Prefer tmux for the restarted training run. Preserve logs. Do not delete important data or checkpoints.
6. If you restart training, write a timestamped report in training_notes explaining what failed, what you changed, exact command/session/log paths, and how to monitor it.
7. If the issue is not safely repairable, write a blocked report with the exact blocker and next command to run.

Be direct and evidence-based. Do not hand-wave. This is a startup health check, not a broad refactor.
EOF

if (( DRY_RUN )); then
  echo "dry-run: wrote startup bundle at $report_dir"
  exit 0
fi

CODEX_ARGS=(exec -C "$ROOT" --dangerously-bypass-approvals-and-sandbox)
if [[ -n "$CODEX_MODEL" ]]; then
  CODEX_ARGS=(-m "$CODEX_MODEL" "${CODEX_ARGS[@]}")
fi

echo "calling Codex for startup health check..."
"$CODEX_BIN" "${CODEX_ARGS[@]}" - < "$report_dir/codex_prompt.md" \
  > "$report_dir/codex_stdout.log" 2> "$report_dir/codex_stderr.log" || {
    code=$?
    echo "Codex startup check failed with exit code $code; see $report_dir/codex_stdout.log and codex_stderr.log"
    exit "$code"
  }

echo "Codex startup check complete."
echo "report bundle: $report_dir"
echo "stdout: $report_dir/codex_stdout.log"
