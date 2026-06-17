#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/iska/Documents/amelie/bio/ToricGT}"
PRIOR_ANALYSIS_DIR="${1:?usage: $0 PRIOR_ANALYSIS_DIR [CAMPAIGN_ID]}"
CAMPAIGN_ID="${2:-tg-bpb112-ad-$(date -u +%Y%m%dT%H%M%SZ)}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-21600}"

cd "${REPO_ROOT}"
mkdir -p "training_notes/${CAMPAIGN_ID}"

log="training_notes/${CAMPAIGN_ID}/analysis_driven_launch.log"
started_at="$(date -u +%s)"

{
  echo "[$(date -u --iso-8601=seconds)] waiting for prior analysis: ${PRIOR_ANALYSIS_DIR}"
  echo "campaign_id=${CAMPAIGN_ID}"
} >> "${log}"

while true; do
  if [[ -f "${PRIOR_ANALYSIS_DIR}/FULL-ITERATION-REPORT.md" && -f "${PRIOR_ANALYSIS_DIR}/next_profile_decision.json" ]]; then
    break
  fi
  now="$(date -u +%s)"
  elapsed=$((now - started_at))
  if (( elapsed > MAX_WAIT_SECONDS )); then
    {
      echo "[$(date -u --iso-8601=seconds)] timed out waiting for prior analysis after ${elapsed}s"
      echo "required files:"
      echo "  ${PRIOR_ANALYSIS_DIR}/FULL-ITERATION-REPORT.md"
      echo "  ${PRIOR_ANALYSIS_DIR}/next_profile_decision.json"
    } >> "${log}"
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

{
  echo "[$(date -u --iso-8601=seconds)] prior analysis complete; launching analysis-driven campaign"
  echo "prior_report=${PRIOR_ANALYSIS_DIR}/FULL-ITERATION-REPORT.md"
  echo "prior_decision=${PRIOR_ANALYSIS_DIR}/next_profile_decision.json"
} >> "${log}"

tmux new-session -d -s "toricgt_oai_fullanalysis_${CAMPAIGN_ID}" \
  "cd '${REPO_ROOT}' && /home/iska/miniconda3/bin/conda run --no-capture-output -n tokengt env PYTHONPATH=src python scripts/run_oai_sidecar_bpb_campaign.py --campaign-id '${CAMPAIGN_ID}' --max-runs 25 --followup-runs-after-meta 10 --target-bpb 1.19 --steps-per-run 1500 --full-analysis --strict-analysis --analysis-retries 1 --analysis-retry-timeout-multiplier 2.0 --analysis-records 3 --analysis-cas-max-points 6 --analysis-cas-macaulay2-timeout-seconds 900 --prior-analysis-dir '${PRIOR_ANALYSIS_DIR}' --prior-log /tmp/toricgt_no_prior_train.log --codex-review > 'training_notes/${CAMPAIGN_ID}/campaign_supervisor.log' 2>&1"

tmux new-session -d -s "toricgt_full_analysis_codex_watch_${CAMPAIGN_ID}" \
  "cd '${REPO_ROOT}' && /home/iska/miniconda3/bin/conda run --no-capture-output -n tokengt python scripts/watch_full_analysis_codex_reviews.py --campaign-notes-dir 'training_notes/${CAMPAIGN_ID}' --poll-seconds 60 --max-reviews 35 > 'training_notes/${CAMPAIGN_ID}/full_analysis_codex_watch.log' 2>&1"

{
  echo "[$(date -u --iso-8601=seconds)] launched"
  echo "training_tmux=toricgt_oai_fullanalysis_${CAMPAIGN_ID}"
  echo "watcher_tmux=toricgt_full_analysis_codex_watch_${CAMPAIGN_ID}"
} >> "${log}"

printf '%s\n' "${CAMPAIGN_ID}"
