#!/usr/bin/env bash
# Trigger an automated Codex handoff after a training-analysis watcher finishes.
#
# Example:
#   scripts/codex_training_review_resume.sh \
#     --analysis-dir outputs/post_resume_analysis/oai-restart-01000-phased/step-00002500 \
#     --checkpoint checkpoints/parameter_golf_oai_dense/random_order_step_00002500.pt \
#     --step 2500 \
#     --run-path amelie-iska-math/toricgt-parameter-golf/1ouz53jk \
#     --training-tmux toricgt_pg_oai \
#     --tmux-session toricgt_codex_review_00002500
#
# The default handoff uses non-interactive `codex exec resume`, not interactive
# `codex resume`, so update prompts or TUI screens cannot block the restart
# decision loop.
#
# To resume a specific Codex session rather than the most recent one:
#   CODEX_RESUME_SESSION_ID=<session-id-or-thread-name> scripts/codex_training_review_resume.sh ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYSIS_DIR=""
CHECKPOINT=""
STEP=""
RUN_PATH=""
TRAINING_TMUX="toricgt_pg_oai"
SESSION_ID="${CODEX_RESUME_SESSION_ID:-}"
TMUX_SESSION=""
DRY_RUN=0
BPB_TARGET="${BPB_TARGET:-1.2}"
BPB_MAX_REVIEW_ITERATIONS="${BPB_MAX_REVIEW_ITERATIONS:-100}"
BPB_LOOP_STATE="${BPB_LOOP_STATE:-outputs/bpb_codex_loop_state.json}"
BPB_LOOP_STOP_FILE="${BPB_LOOP_STOP_FILE:-outputs/bpb_codex_loop_stop}"
BPB_LOOP_NAME="${BPB_LOOP_NAME:-parameter_golf_bpb_target}"

usage() {
  sed -n '1,32p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --analysis-dir)
      ANALYSIS_DIR="${2:?missing value for --analysis-dir}"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="${2:?missing value for --checkpoint}"
      shift 2
      ;;
    --step)
      STEP="${2:?missing value for --step}"
      shift 2
      ;;
    --run-path)
      RUN_PATH="${2:?missing value for --run-path}"
      shift 2
      ;;
    --training-tmux)
      TRAINING_TMUX="${2:?missing value for --training-tmux}"
      shift 2
      ;;
    --session-id)
      SESSION_ID="${2:?missing value for --session-id}"
      shift 2
      ;;
    --tmux-session)
      TMUX_SESSION="${2:?missing value for --tmux-session}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ANALYSIS_DIR" || -z "$CHECKPOINT" || -z "$STEP" ]]; then
  echo "--analysis-dir, --checkpoint, and --step are required" >&2
  usage >&2
  exit 2
fi

SYNOPSIS="$ANALYSIS_DIR/SYNOPSIS.md"
METRICS_SUMMARY="$ANALYSIS_DIR/metrics/category_summary.json"
GEOMETRY_SUMMARY="$ANALYSIS_DIR/geometry/reasoning_geometry_summary.json"
SIMPLEX_SUMMARY="$ANALYSIS_DIR/simplex/reasoning_simplex_summary.json"
HOOK_LOG_DIR="$ANALYSIS_DIR/logs"
mkdir -p "$HOOK_LOG_DIR"
LOOP_STATUS="$HOOK_LOG_DIR/bpb_codex_loop_status.json"

LOOP_EXPORTS="$(
  python3 - "$REPO_ROOT" "$ANALYSIS_DIR" "$CHECKPOINT" "$STEP" "$BPB_TARGET" \
    "$BPB_MAX_REVIEW_ITERATIONS" "$BPB_LOOP_STATE" "$BPB_LOOP_STOP_FILE" \
    "$BPB_LOOP_NAME" "$DRY_RUN" "$LOOP_STATUS" <<'PY'
import csv
import json
import math
import shlex
import sys
from pathlib import Path

repo = Path(sys.argv[1])
analysis_dir = Path(sys.argv[2])
checkpoint = sys.argv[3]
step = int(sys.argv[4])
target = float(sys.argv[5])
max_iterations = int(sys.argv[6])
state_path = Path(sys.argv[7])
stop_file = Path(sys.argv[8])
loop_name = sys.argv[9]
dry_run = sys.argv[10] == "1"
status_path = Path(sys.argv[11])

if not state_path.is_absolute():
    state_path = repo / state_path
if not stop_file.is_absolute():
    stop_file = repo / stop_file
if not status_path.is_absolute():
    status_path = repo / status_path


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def maybe_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def add_candidate(candidates, metric, value, source, priority):
    numeric = maybe_float(value)
    if numeric is None:
        return
    if numeric <= 0:
        return
    candidates.append(
        {
            "metric": metric,
            "value": numeric,
            "source": source,
            "priority": priority,
        }
    )


state = load_json(state_path)
if not isinstance(state, dict):
    state = {}
history = state.get("history")
if not isinstance(history, list):
    history = []
previous_iterations = int(state.get("iteration_count", 0) or 0)
iteration = previous_iterations + 1

candidates = []
checkpoint_meta = load_json(analysis_dir / "metrics" / "checkpoint_meta.json")
checkpoint_metrics = checkpoint_meta.get("metrics", {})
if isinstance(checkpoint_metrics, dict):
    for key, value in checkpoint_metrics.items():
        key_str = str(key)
        if "bpb" not in key_str.lower():
            continue
        priority = 40
        lowered = key_str.lower()
        if "fineweb" in lowered:
            priority = 0
        elif "best_val" in lowered or lowered in {"val_bpb", "val/bpb"}:
            priority = 10
        elif "complexity/val" in lowered:
            priority = 20
        elif "train" in lowered:
            priority = 30
        add_candidate(candidates, key_str, value, "checkpoint_meta", priority)

stats = load_json(analysis_dir / "metrics" / "metric_stats.json")
if isinstance(stats, list):
    for row in stats:
        if not isinstance(row, dict):
            continue
        metric = str(row.get("metric", ""))
        if "bpb" not in metric.lower():
            continue
        value = (
            row.get("last_median")
            if row.get("last_median") is not None
            else row.get("recent_median")
            if row.get("recent_median") is not None
            else row.get("last")
        )
        lowered = metric.lower()
        priority = 40
        if "fineweb" in lowered:
            priority = 0
        elif lowered in {"val/bpb", "val_bpb"} or "best_val" in lowered:
            priority = 10
        elif "complexity/val" in lowered:
            priority = 20
        elif "train" in lowered:
            priority = 30
        add_candidate(candidates, metric, value, "metric_stats", priority)

history_csv = analysis_dir / "metrics" / "wandb_history.csv"
if history_csv.exists():
    try:
        with history_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if rows:
            last = rows[-1]
            for key, value in last.items():
                if key and "bpb" in key.lower():
                    lowered = key.lower()
                    priority = 40
                    if "fineweb" in lowered:
                        priority = 0
                    elif lowered in {"val/bpb", "val_bpb"} or "best_val" in lowered:
                        priority = 10
                    elif "complexity/val" in lowered:
                        priority = 20
                    elif "train" in lowered:
                        priority = 30
                    add_candidate(candidates, key, value, "wandb_history_last", priority)
    except Exception:
        pass

primary = None
if candidates:
    primary = sorted(candidates, key=lambda item: (item["priority"], item["value"]))[0]

best_state_bpb = maybe_float(state.get("best_bpb"))
best_state_metric = str(state.get("best_bpb_metric", ""))
best_state_step = state.get("best_bpb_step")
if primary is not None and (best_state_bpb is None or primary["value"] < best_state_bpb):
    best_bpb = primary["value"]
    best_metric = primary["metric"]
    best_step = step
else:
    best_bpb = best_state_bpb
    best_metric = best_state_metric
    best_step = best_state_step

target_reached = bool(best_bpb is not None and best_bpb <= target)
max_reached = bool(iteration > max_iterations)
stop_requested = stop_file.exists()

record = {
    "iteration": iteration,
    "step": step,
    "checkpoint": checkpoint,
    "primary_bpb": primary,
    "target_bpb": target,
    "target_reached": target_reached,
    "max_iterations": max_iterations,
    "max_reached": max_reached,
    "stop_file": str(stop_file),
    "stop_requested": stop_requested,
}
history.append(record)
history = history[-250:]

state.update(
    {
        "loop_name": loop_name,
        "target_bpb": target,
        "max_iterations": max_iterations,
        "iteration_count": iteration,
        "last_step": step,
        "last_checkpoint": checkpoint,
        "last_primary_bpb": primary,
        "best_bpb": best_bpb,
        "best_bpb_metric": best_metric,
        "best_bpb_step": best_step,
        "target_reached": target_reached,
        "stop_requested": stop_requested,
        "history": history,
    }
)

status = {
    "state_path": str(state_path),
    "loop_name": loop_name,
    "iteration": iteration,
    "target_bpb": target,
    "max_iterations": max_iterations,
    "primary_bpb": primary,
    "best_bpb": best_bpb,
    "best_bpb_metric": best_metric,
    "best_bpb_step": best_step,
    "target_reached": target_reached,
    "max_reached": max_reached,
    "stop_requested": stop_requested,
    "stop_file": str(stop_file),
}
status_path.parent.mkdir(parents=True, exist_ok=True)
status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not dry_run:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

emit("BPB_REVIEW_ITERATION", iteration)
emit("BPB_LOOP_STATE_ABS", state_path)
emit("BPB_LOOP_STATUS", status_path)
emit("BPB_PRIMARY_METRIC", primary["metric"] if primary else "")
emit("BPB_PRIMARY_VALUE", primary["value"] if primary else "")
emit("BPB_BEST_VALUE", best_bpb if best_bpb is not None else "")
emit("BPB_BEST_METRIC", best_metric)
emit("BPB_BEST_STEP", best_step if best_step is not None else "")
emit("BPB_TARGET_REACHED", "1" if target_reached else "0")
emit("BPB_MAX_REACHED", "1" if max_reached else "0")
emit("BPB_STOP_REQUESTED", "1" if stop_requested else "0")
PY
)"
eval "$LOOP_EXPORTS"

if [[ "$BPB_TARGET_REACHED" == "1" ]]; then
  echo "BPB target reached; skipping Codex review. See $LOOP_STATUS"
  exit 0
fi
if [[ "$BPB_MAX_REACHED" == "1" ]]; then
  echo "BPB review iteration cap reached; skipping Codex review. See $LOOP_STATUS"
  exit 0
fi
if [[ "$BPB_STOP_REQUESTED" == "1" ]]; then
  echo "BPB loop stop sentinel exists; skipping Codex review. See $BPB_LOOP_STOP_FILE"
  exit 0
fi

read -r -d '' PROMPT <<EOF || true
Automated ToricGT training-analysis handoff.

The post-resume analysis watcher just completed. Please review the metrics,
statistics, and generated plots, then make an explicit resume decision.  The
watcher pauses the training tmux before analysis, so a "continue" decision must
actively resume from the analyzed checkpoint or another selected checkpoint; do
not assume the prior training process is still advancing.

Context:
- Repository: $REPO_ROOT
- Branch target: oai
- Training tmux session: $TRAINING_TMUX
- W&B run path: ${RUN_PATH:-not provided}
- Analyzed checkpoint: $CHECKPOINT
- Analyzed step: $STEP
- Analysis directory: $ANALYSIS_DIR
- BPB optimization loop: $BPB_LOOP_NAME
- BPB review iteration: $BPB_REVIEW_ITERATION / $BPB_MAX_REVIEW_ITERATIONS
- BPB target: <= $BPB_TARGET
- Best observed BPB in loop: ${BPB_BEST_VALUE:-unknown} (${BPB_BEST_METRIC:-unknown}, step ${BPB_BEST_STEP:-unknown})
- Current primary BPB signal: ${BPB_PRIMARY_VALUE:-unknown} (${BPB_PRIMARY_METRIC:-unknown})
- Loop state: $BPB_LOOP_STATE_ABS
- Loop status: $BPB_LOOP_STATUS
- Better-strategy stop sentinel: $BPB_LOOP_STOP_FILE
- Synopsis: $SYNOPSIS
- Metric categories: $METRICS_SUMMARY
- Reasoning simplex summary: $SIMPLEX_SUMMARY
- Reasoning geometry summary: $GEOMETRY_SUMMARY
- BPB amplification plan: $REPO_ROOT/planning/BPB-AMP.md
- BPB cliff plan: $REPO_ROOT/planning/BPB-CLIFF-RECOVERY.md

Requested work:
1. Inspect the W&B metrics export and all analysis summaries.
2. Optimize the BPB gate first.  The active target is BPB <= $BPB_TARGET.  Keep
   reviews, rollbacks, edits, resets, and restarts active until that target is
   reached, $BPB_MAX_REVIEW_ITERATIONS review iterations complete, or you find
   and document a better BPB optimization strategy.  If you discover a better
   strategy that should replace this loop, write a short explanation to
   $BPB_LOOP_STOP_FILE and stop launching ordinary cliff replays.
3. Inspect generated plots/contact sheets when present, including simplex plots,
   3D reasoning trajectories, Ramachandran-style phase plots, and energy or
   fitness landscapes.
4. Categorize metrics and plot behavior as:
   i. desired,
   ii. desired but too weak or slow,
   iii. undesirable.
5. Explain the behavior mathematically and statistically, focusing on:
   - official-style FineWeb BPB when available;
   - ToricGT curated hard-reasoning BPB and validation metrics;
   - train/validation loss;
   - GFlowNet graph-of-thought quality and branch replay;
   - Kolmogorov and relative-K proxies;
   - GraphCG basis behavior;
   - persistence/simplex/Koszul topology;
   - Toric BGG resolution consistency, standard leakage, Gale-dual consistency,
     and signature metrics when present;
   - MST efficiency, trajectory smoothness, and trajectory length;
   - toric entropy, noncommutative phase, tropical active faces, and chamber
     crossings;
   - branch/test-time-scaling diagnostics;
   - finite differences of checkpoint metrics, second differences, and Hessian
     probe sharpness/trace when available.
   Treat a strong FineWeb result after limited FineWeb exposure as possible OOD
   transfer from hard reasoning data, but require controls against tokenizer,
   n-gram, and dataset-easiness explanations.
6. Run scripts/propose_training_adjustments.py on the analysis directory and
   use its proposal as evidence, not as an automatic authority.
7. If adjustments are warranted, implement the smallest high-impact code/config
   changes that preserve the ToricGT Parameter-Golf architecture. Do not use
   JEPA. Keep random-order autoregressive graph decoding, tropical ring/hybrid
   attention, toric memory, dense contest weights, embedding-space GFlowNet
   graph-of-thought, hard GoT/ToT/CoT training data, GraphCG/topology/toric,
   Toric BGG diagnostics, and Kolmogorov diagnostics unless the data gives a
   clear reason to alter a scalar control.
8. Choose exactly one action:
   - CONTINUE: resume from the analyzed checkpoint with the same config.
   - ROLLBACK: resume from the last dense checkpoint before the first positive
     first derivative or before second-difference/Hessian evidence indicates a
     floor-bounce basin.
   - EDIT_AND_RESTART: make minimal scalar/config changes, then resume from the
     selected checkpoint.
   The decision must respect the two-gate rule in planning/BPB-AMP.md:
   FineWeb BPB is the competition calibration/evaluation gate, while hard
   ToricGT reasoning, GFlowNet, GraphCG, topology, toric/tropical, and memory
   metrics are the reasoning gate.  If the gates disagree, adjust mixture
   ratios and auxiliary weights rather than discarding either objective.
   In all three cases, start a fresh training tmux and report the tmux, log,
   checkpoint, and W&B details.  Do not leave the paused tmux as the active run.
9. After the resume/restart decision, schedule the next interrupting analysis.
   Use a target about 250 steps later while the run is still inside the
   2000--2500 BPB-cliff window, otherwise about 500--600 steps later.  The next
   watcher must use scripts/watch_training_analysis.py with --target-step,
   --pause-training-before-analysis, --device cuda, --precision bf16,
   --codex-review-hook scripts/codex_training_review_resume.sh, and
   --codex-review-tmux-prefix toricgt_codex_review. Use a fresh
   --min-mtime-unix captured at the restart time so old checkpoint filenames are
   ignored.  This creates the periodic pause -> analysis -> Codex review ->
   adjustment -> restart loop requested by the user.
   Preserve these loop environment variables on every restart:
   BPB_TARGET=$BPB_TARGET, BPB_MAX_REVIEW_ITERATIONS=$BPB_MAX_REVIEW_ITERATIONS,
   BPB_LOOP_STATE=$BPB_LOOP_STATE_ABS, BPB_LOOP_STOP_FILE=$BPB_LOOP_STOP_FILE,
   BPB_LOOP_NAME=$BPB_LOOP_NAME.
10. Update planning/METRICS.md with the analysis and decision. Push branch oai
   if code, config, docs, or planning files change.

Safety:
- Do not print or commit tokens or keys.txt.
- Do not delete checkpoints.
- Do not push training data or weight artifacts unless explicitly needed by the
  existing checkpoint-publishing path.
- Keep changes minimal and competition-focused.
EOF

if [[ -n "$SESSION_ID" ]]; then
  CODEX_CMD=(codex exec -C "$REPO_ROOT" --dangerously-bypass-approvals-and-sandbox resume "$SESSION_ID" "$PROMPT")
else
  CODEX_CMD=(codex exec -C "$REPO_ROOT" --dangerously-bypass-approvals-and-sandbox resume --last "$PROMPT")
fi

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf '%q ' "${CODEX_CMD[@]}"
  printf '\n'
} > "$HOOK_LOG_DIR/codex_resume_command.sh"
chmod 700 "$HOOK_LOG_DIR/codex_resume_command.sh"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "dry-run: wrote $HOOK_LOG_DIR/codex_resume_command.sh"
  exit 0
fi

if [[ -n "$TMUX_SESSION" ]]; then
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "tmux session already exists: $TMUX_SESSION" >&2
    exit 1
  fi
  tmux new-session -d -s "$TMUX_SESSION" "cd '$REPO_ROOT' && exec '$HOOK_LOG_DIR/codex_resume_command.sh'"
  echo "launched Codex review tmux: $TMUX_SESSION"
else
  exec "${CODEX_CMD[@]}"
fi
