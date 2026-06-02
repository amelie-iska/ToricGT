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
- Synopsis: $SYNOPSIS
- Metric categories: $METRICS_SUMMARY
- Reasoning simplex summary: $SIMPLEX_SUMMARY
- Reasoning geometry summary: $GEOMETRY_SUMMARY
- BPB amplification plan: $REPO_ROOT/planning/BPB-AMP.md
- BPB cliff plan: $REPO_ROOT/planning/BPB-CLIFF-RECOVERY.md

Requested work:
1. Inspect the W&B metrics export and all analysis summaries.
2. Inspect generated plots/contact sheets when present, including simplex plots,
   3D reasoning trajectories, Ramachandran-style phase plots, and energy or
   fitness landscapes.
3. Categorize metrics and plot behavior as:
   i. desired,
   ii. desired but too weak or slow,
   iii. undesirable.
4. Explain the behavior mathematically and statistically, focusing on:
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
5. If adjustments are warranted, implement the smallest high-impact code/config
   changes that preserve the ToricGT Parameter-Golf architecture. Do not use
   JEPA. Keep random-order autoregressive graph decoding, tropical ring/hybrid
   attention, toric memory, dense contest weights, embedding-space GFlowNet
   graph-of-thought, hard GoT/ToT/CoT training data, GraphCG/topology/toric,
   Toric BGG diagnostics, and Kolmogorov diagnostics unless the data gives a
   clear reason to alter a scalar control.
6. Choose exactly one action:
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
7. After the resume/restart decision, schedule the next interrupting analysis.
   Use a target about 250 steps later while the run is still inside the
   2000--2500 BPB-cliff window, otherwise about 500--600 steps later.  The next
   watcher must use scripts/watch_training_analysis.py with --target-step,
   --pause-training-before-analysis, --device cuda, --precision bf16,
   --codex-review-hook scripts/codex_training_review_resume.sh, and
   --codex-review-tmux-prefix toricgt_codex_review. Use a fresh
   --min-mtime-unix captured at the restart time so old checkpoint filenames are
   ignored.  This creates the periodic pause -> analysis -> Codex review ->
   adjustment -> restart loop requested by the user.
8. Update planning/METRICS.md with the analysis and decision. Push branch oai
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
