#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${TORICBLM_TOKENGT_PYTHON:-/home/iska/miniconda3/envs/tokengt/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${PYTHON:-python3}"
fi

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 EPOCH_PADDED RUN_ID SPECIAL_CHECKPOINT MANIFEST_JSON EPOCH_RUN_DIR CHECKPOINT_DIR SUMMARY_TSV" >&2
  exit 2
fi

EPOCH_PADDED="$1"
RUN_ID="$2"
SPECIAL_CHECKPOINT="$3"
MANIFEST_JSON="$4"
EPOCH_RUN_DIR="$5"
CHECKPOINT_DIR="$6"
SUMMARY_TSV="$7"

if [[ ! -f "$SPECIAL_CHECKPOINT" ]]; then
  echo "Missing special checkpoint: $SPECIAL_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST_JSON" ]]; then
  echo "Missing dataset-state manifest: $MANIFEST_JSON" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" && -f "$ROOT/keys.txt" ]]; then
  HF_TOKEN="$(
    "$PY" - "$ROOT/keys.txt" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
patterns = [
    r"(?:^|\n)\s*(?:HF_TOKEN|HUGGINGFACE_TOKEN|HUGGING_FACE_HUB_TOKEN)\s*=\s*([^\s\"']+)",
    r"(hf_[A-Za-z0-9_=-]+)",
]
for pattern in patterns:
    match = re.search(pattern, text)
    if match:
        print(match.group(1).strip())
        raise SystemExit(0)
raise SystemExit(0)
PY
  )"
  export HF_TOKEN
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is unset and no token was found in keys.txt; skipping Hugging Face upload." >&2
  exit 3
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is not installed or not on PATH." >&2
  exit 4
fi

HF_CHECKPOINT_REPO="${HF_CHECKPOINT_REPO:-AmelieSchreiber/ToricGT_160M_FoT}"
HF_DATASET_STATE_REPO_PREFIX="${HF_DATASET_STATE_REPO_PREFIX:-AmelieSchreiber/toricblm-dataset-state}"
HF_UPLOAD_PRIVATE="${HF_UPLOAD_PRIVATE:-0}"
HF_UPLOAD_FLAGS=()
if [[ "$HF_UPLOAD_PRIVATE" == "1" ]]; then
  HF_UPLOAD_FLAGS+=(--private)
fi

safe_run_id="$(printf '%s' "$RUN_ID" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-' | sed 's/^-//;s/-$//')"
prefix_ns="${HF_DATASET_STATE_REPO_PREFIX%/*}"
prefix_name="${HF_DATASET_STATE_REPO_PREFIX#*/}"
if [[ "$prefix_ns" == "$prefix_name" ]]; then
  dataset_repo="${prefix_name}-${safe_run_id}-epoch-${EPOCH_PADDED}"
else
  dataset_repo="${prefix_ns}/${prefix_name}-${safe_run_id}-epoch-${EPOCH_PADDED}"
fi

state_dir="$EPOCH_RUN_DIR/hf_dataset_state_epoch_${EPOCH_PADDED}"
rm -rf "$state_dir"
mkdir -p "$state_dir"
cp "$MANIFEST_JSON" "$state_dir/available_shards.json"
[[ -f "$SUMMARY_TSV" ]] && cp "$SUMMARY_TSV" "$state_dir/dynamic_epoch_summary.tsv"
[[ -f "$EPOCH_RUN_DIR/toricblm_training_corpus_audit.json" ]] && cp "$EPOCH_RUN_DIR/toricblm_training_corpus_audit.json" "$state_dir/training_corpus_audit.json"
for cfg in "$EPOCH_RUN_DIR"/dynamic_epoch_*.env; do
  [[ -f "$cfg" ]] && cp "$cfg" "$state_dir/$(basename "$cfg")"
done
for cfg in "$EPOCH_RUN_DIR"/structure_priority_epoch_*.env "$EPOCH_RUN_DIR"/epoch_data_overrides.env; do
  [[ -f "$cfg" ]] && cp "$cfg" "$state_dir/$(basename "$cfg")"
done
if [[ -f "$EPOCH_RUN_DIR/train.log" ]]; then
  tail -n 500 "$EPOCH_RUN_DIR/train.log" > "$state_dir/train_log_tail.txt"
fi

"$PY" - "$MANIFEST_JSON" "$RUN_ID" "$EPOCH_PADDED" "$SPECIAL_CHECKPOINT" "$HF_CHECKPOINT_REPO" > "$state_dir/README.md" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
run_id = sys.argv[2]
epoch = sys.argv[3]
checkpoint = Path(sys.argv[4]).name
checkpoint_repo = sys.argv[5]
print("---")
print("license: apache-2.0")
print("task_categories:")
print("- text-generation")
print("- graph-ml")
print("pretty_name: ToricBLM Dynamic Epoch Dataset State")
print("---")
print(f"# ToricBLM dataset state: `{run_id}` epoch `{epoch}`")
print()
print("This dataset repo records the exact local training-data state visible to the dynamic epoch launcher.")
print("It intentionally stores manifests and audit records rather than duplicating large Parquet shards.")
print()
print(f"- Special checkpoint: `{checkpoint}`")
print(f"- Checkpoint repo: `{checkpoint_repo}`")
if manifest.get("schema") == "toricblm.structure_priority_curriculum_epoch_state.v1":
    print(f"- Curriculum mode: `{manifest.get('mode')}`")
    print(f"- Selected rows: `{manifest.get('selected_rows')}`")
    print(f"- Selected files: `{manifest.get('selected_file_count')}`")
    print(f"- Structure-flow rows: `{manifest.get('structure_rows_available_for_structure_flow')}`")
    print(f"- Coverage steps: `{manifest.get('coverage_steps')}`")
else:
    print(f"- Live AFDB glob: `{manifest.get('live_afdb_glob')}`")
    print(f"- Live AFDB shards: `{manifest.get('live_afdb_shards')}`")
    print(f"- Live AFDB size: `{manifest.get('live_afdb_gb', 0):.3f} GB`")
print()
print("Files:")
print("- `available_shards.json`: refreshed dataset-state snapshot for this epoch")
print("- `training_corpus_audit.json`: sampled corpus audit produced before training")
print("- `*_epoch_*.env` / `epoch_data_overrides.env`: resolved epoch launch configuration")
print("- `dynamic_epoch_summary.tsv`: epoch handoff/checkpoint summary")
print("- `train_log_tail.txt`: tail of the epoch training log")
PY

hf auth whoami >/dev/null

hf repos create "$HF_CHECKPOINT_REPO" --type model --exist-ok "${HF_UPLOAD_FLAGS[@]}" >/dev/null
hf repos create "$dataset_repo" --type dataset --exist-ok "${HF_UPLOAD_FLAGS[@]}" >/dev/null

hf upload "$HF_CHECKPOINT_REPO" "$SPECIAL_CHECKPOINT" "checkpoints/$RUN_ID/$(basename "$SPECIAL_CHECKPOINT")" \
  --type model --commit-message "Upload $RUN_ID epoch $EPOCH_PADDED special checkpoint"
hf upload "$HF_CHECKPOINT_REPO" "$MANIFEST_JSON" "checkpoints/$RUN_ID/manifests/epoch_${EPOCH_PADDED}_available_shards.json" \
  --type model --commit-message "Upload $RUN_ID epoch $EPOCH_PADDED dataset manifest"
hf upload "$dataset_repo" "$state_dir" . \
  --type dataset --commit-message "Upload $RUN_ID epoch $EPOCH_PADDED dataset state"

echo "hf_epoch_artifacts_uploaded checkpoint_repo:$HF_CHECKPOINT_REPO dataset_state_repo:$dataset_repo epoch:$EPOCH_PADDED"
