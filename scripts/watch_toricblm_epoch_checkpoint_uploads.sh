#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:?set RUN_ID for the training run to watch}"
RUN_BASE_DIR="${RUN_BASE_DIR:-$ROOT/runs/oai_sidecar/${RUN_ID}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints/${RUN_ID}}"
SUMMARY_TSV="${SUMMARY_TSV:-}"
POLL_SECONDS="${HF_EPOCH_WATCH_POLL_SECONDS:-300}"
STATE_FILE="${HF_EPOCH_WATCH_STATE_FILE:-$RUN_BASE_DIR/hf_epoch_upload_watch_state.tsv}"

mkdir -p "$RUN_BASE_DIR" "$CHECKPOINT_DIR"
touch "$STATE_FILE"

find_epoch_dir() {
  local epoch_padded="$1"
  local candidate
  candidate="$(find "$RUN_BASE_DIR" -maxdepth 1 -type d -name "epoch_${epoch_padded}_*" | sort | head -1 || true)"
  if [[ -n "$candidate" ]]; then
    echo "$candidate"
  fi
}

find_manifest() {
  local epoch_padded="$1"
  local epoch_dir="$2"
  if [[ -n "$epoch_dir" && -f "$epoch_dir/epoch_state_manifest.json" ]]; then
    echo "$epoch_dir/epoch_state_manifest.json"
    return
  fi
  if [[ -f "$RUN_BASE_DIR/dynamic_epoch_manifests/epoch_${epoch_padded}_available_shards.json" ]]; then
    echo "$RUN_BASE_DIR/dynamic_epoch_manifests/epoch_${epoch_padded}_available_shards.json"
    return
  fi
}

find_summary() {
  if [[ -n "$SUMMARY_TSV" && -f "$SUMMARY_TSV" ]]; then
    echo "$SUMMARY_TSV"
    return
  fi
  if [[ -f "$RUN_BASE_DIR/structure_priority_curriculum_summary.tsv" ]]; then
    echo "$RUN_BASE_DIR/structure_priority_curriculum_summary.tsv"
    return
  fi
  if [[ -f "$RUN_BASE_DIR/dynamic_epoch_summary.tsv" ]]; then
    echo "$RUN_BASE_DIR/dynamic_epoch_summary.tsv"
    return
  fi
  echo "$RUN_BASE_DIR/epoch_upload_summary_placeholder.tsv"
}

already_uploaded() {
  local ckpt="$1"
  grep -Fq "$ckpt"$'\t'"uploaded" "$STATE_FILE"
}

mark_uploaded() {
  local ckpt="$1"
  local epoch="$2"
  printf '%s\tuploaded\tepoch=%s\t%s\n' "$ckpt" "$epoch" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATE_FILE"
}

upload_once() {
  local ckpt="$1"
  local name epoch_padded epoch_dir manifest summary
  name="$(basename "$ckpt")"
  epoch_padded="$(printf '%s' "$name" | sed -n 's/.*_epoch_\([0-9][0-9][0-9]\)_.*/\1/p')"
  if [[ -z "$epoch_padded" ]]; then
    echo "skip_non_epoch_checkpoint:$ckpt"
    return 0
  fi
  if already_uploaded "$ckpt"; then
    return 0
  fi
  epoch_dir="$(find_epoch_dir "$epoch_padded")"
  manifest="$(find_manifest "$epoch_padded" "$epoch_dir")"
  if [[ -z "$epoch_dir" || -z "$manifest" || ! -f "$manifest" ]]; then
    echo "epoch_upload_waiting_for_manifest epoch:$epoch_padded ckpt:$ckpt"
    return 0
  fi
  summary="$(find_summary)"
  [[ -f "$summary" ]] || printf 'epoch\tstatus\n' > "$summary"
  echo "epoch_upload_start epoch:$epoch_padded ckpt:$ckpt manifest:$manifest"
  "$ROOT/scripts/publish_toricblm_epoch_artifacts_to_hf.sh" \
    "$epoch_padded" "$RUN_ID" "$ckpt" "$manifest" "$epoch_dir" "$CHECKPOINT_DIR" "$summary"
  mark_uploaded "$ckpt" "$epoch_padded"
  echo "epoch_upload_done epoch:$epoch_padded ckpt:$ckpt"
}

echo "hf_epoch_upload_watcher_started run_id:$RUN_ID checkpoint_dir:$CHECKPOINT_DIR run_base_dir:$RUN_BASE_DIR poll_seconds:$POLL_SECONDS"
while true; do
  shopt -s nullglob
  for ckpt in "$CHECKPOINT_DIR"/"${RUN_ID}"_epoch_???_special_*.pt; do
    upload_once "$ckpt" || echo "epoch_upload_failed_nonfatal ckpt:$ckpt" >&2
  done
  shopt -u nullglob
  sleep "$POLL_SECONDS"
done
