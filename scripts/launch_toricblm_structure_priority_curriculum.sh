#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/toricblm_mup_170m_convextok8192_biomed_fot_structure.env}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  exit 1
fi

PY="${TORICBLM_TOKENGT_PYTHON:-/home/iska/miniconda3/envs/tokengt/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "Missing tokengt Python: $PY" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${RUN_ID:-toricblm-structure-priority-curriculum-${RUN_STAMP}}"
RUN_BASE_DIR="${RUN_BASE_DIR:-$ROOT/runs/oai_sidecar/${RUN_ID}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints/${RUN_ID}}"
mkdir -p "$RUN_BASE_DIR" "$CHECKPOINT_DIR" logs

TOKENIZER_PATH="${TOKENIZER_PATH:?TOKENIZER_PATH must be set by the base config}"
TARGET_VRAM_GB="${CURRICULUM_TARGET_VRAM_GB:-20}"
SAMPLE_FILES="${CURRICULUM_TOKEN_SAMPLE_FILES:-48}"
ROWS_PER_SAMPLE_FILE="${CURRICULUM_ROWS_PER_SAMPLE_FILE:-8}"
VAL_LOSS_EVERY="${CURRICULUM_VAL_LOSS_EVERY:-50000}"
TRAIN_LOG_EVERY="${CURRICULUM_TRAIN_LOG_EVERY:-100}"
WARMUP_STEPS="${CURRICULUM_WARMUP_STEPS:-20}"
WARMDOWN_ITERS="${CURRICULUM_WARMDOWN_ITERS:-0}"
EVAL_INITIAL="${CURRICULUM_EVAL_INITIAL:-0}"
VAL_MAX_TOKENS="${CURRICULUM_VAL_MAX_TOKENS:-1048576}"
DRY_RUN="${CURRICULUM_DRY_RUN:-0}"

SUMMARY="$RUN_BASE_DIR/structure_priority_curriculum_summary.tsv"
printf 'epoch\tmode\tstart_step\tepoch_steps\ttarget_step\tselected_rows\tstructure_rows\tcheckpoint\tmanifest\n' > "$SUMMARY"

checkpoint_step() {
  local ckpt="$1"
  if [[ -z "$ckpt" ]]; then
    echo 0
    return
  fi
  "$PY" - "$ckpt" <<'PY'
import sys
from pathlib import Path
import torch
path = Path(sys.argv[1])
obj = torch.load(path, map_location="cpu", weights_only=False)
print(int(obj.get("step", 0)) if isinstance(obj, dict) else 0)
PY
}

json_field() {
  local manifest="$1"
  local field="$2"
  "$PY" - "$manifest" "$field" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
value = payload
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

run_epoch() {
  local epoch="$1"
  local mode="$2"
  local previous_manifest="${3:-}"
  local resume_checkpoint="${4:-}"
  local epoch_padded
  epoch_padded="$(printf '%03d' "$epoch")"
  local epoch_dir="$RUN_BASE_DIR/epoch_${epoch_padded}_${mode}"
  mkdir -p "$epoch_dir"

  local build_cmd=(
    "$PY" "$ROOT/scripts/build_toricblm_structure_priority_epoch_state.py"
    --mode "$mode"
    --epoch "$epoch"
    --output-dir "$epoch_dir"
    --tokenizer "$TOKENIZER_PATH"
    --sample-files "$SAMPLE_FILES"
    --rows-per-sample-file "$ROWS_PER_SAMPLE_FILE"
    --target-vram-gb "$TARGET_VRAM_GB"
  )
  if [[ -n "$previous_manifest" ]]; then
    build_cmd+=(--previous-structure-manifest "$previous_manifest")
  fi
  if [[ -n "${CURRICULUM_EXTRA_STRUCTURE_PATTERN:-}" ]]; then
    IFS=';' read -r -a extra_structure_patterns <<< "$CURRICULUM_EXTRA_STRUCTURE_PATTERN"
    for pattern in "${extra_structure_patterns[@]}"; do
      [[ -n "$pattern" ]] && build_cmd+=(--extra-structure-pattern "$pattern")
    done
  fi
  if [[ -n "${CURRICULUM_EXTRA_ALL_PATTERN:-}" ]]; then
    IFS=';' read -r -a extra_all_patterns <<< "$CURRICULUM_EXTRA_ALL_PATTERN"
    for pattern in "${extra_all_patterns[@]}"; do
      [[ -n "$pattern" ]] && build_cmd+=(--extra-all-pattern "$pattern")
    done
  fi

  echo "curriculum_epoch_manifest_build epoch:${epoch_padded} mode:${mode}"
  "${build_cmd[@]}" | tee "$epoch_dir/manifest_build.jsonl"

  local manifest="$epoch_dir/epoch_state_manifest.json"
  local data_overrides="$epoch_dir/epoch_data_overrides.env"
  if [[ ! -f "$manifest" || ! -f "$data_overrides" ]]; then
    echo "Epoch state build failed: $manifest / $data_overrides missing" >&2
    exit 1
  fi

  local epoch_steps selected_rows structure_rows start_step target_step
  epoch_steps="$(json_field "$manifest" "coverage_steps")"
  selected_rows="$(json_field "$manifest" "selected_rows")"
  structure_rows="$(json_field "$manifest" "structure_rows_available_for_structure_flow")"
  start_step="$(checkpoint_step "$resume_checkpoint")"
  target_step="$((start_step + epoch_steps))"

  echo "curriculum_epoch:${epoch_padded} mode:${mode} selected_rows:${selected_rows} structure_rows:${structure_rows} start_step:${start_step} epoch_steps:${epoch_steps} target_step:${target_step}"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$epoch" "$mode" "$start_step" "$epoch_steps" "$target_step" "$selected_rows" "$structure_rows" "dry_run" "$manifest" >> "$SUMMARY"
    LAST_MANIFEST="$manifest"
    LAST_CHECKPOINT="$resume_checkpoint"
    return
  fi

  if (( epoch_steps <= 0 )); then
    if [[ -z "$resume_checkpoint" || ! -f "$resume_checkpoint" ]]; then
      echo "Epoch ${epoch_padded} has no rows and no checkpoint to carry forward" >&2
      exit 1
    fi
    local no_op_ckpt="$CHECKPOINT_DIR/${RUN_ID}_epoch_${epoch_padded}_special_no_new_rows_step_$(printf '%06d' "$start_step").pt"
    ln -f "$resume_checkpoint" "$no_op_ckpt" 2>/dev/null || cp -p "$resume_checkpoint" "$no_op_ckpt"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$epoch" "$mode" "$start_step" "$epoch_steps" "$target_step" "$selected_rows" "$structure_rows" "$no_op_ckpt" "$manifest" >> "$SUMMARY"
    LAST_MANIFEST="$manifest"
    LAST_CHECKPOINT="$no_op_ckpt"
    return
  fi

  local epoch_config="$epoch_dir/structure_priority_epoch_${epoch_padded}.env"
  {
    printf 'source %q\n' "$CONFIG_PATH"
    printf 'source %q\n' "$data_overrides"
    printf 'export RUN_ID=%q\n' "$RUN_ID"
    printf 'export RUN_DIR=%q\n' "$epoch_dir"
    printf 'export CHECKPOINT_DIR=%q\n' "$CHECKPOINT_DIR"
    printf 'export RESUME_CHECKPOINT=%q\n' "$resume_checkpoint"
    if [[ -z "$resume_checkpoint" ]]; then
      printf 'export TORICBLM_START_FROM_STEP0=1\n'
      printf 'export START_STEP=0\n'
    else
      printf 'export TORICBLM_START_FROM_STEP0=0\n'
      printf 'unset START_STEP\n'
    fi
    printf 'export ITERATIONS=%q\n' "$target_step"
    printf 'export CHECKPOINT_EVERY=1\n'
    printf 'export VAL_LOSS_EVERY=%q\n' "$VAL_LOSS_EVERY"
    printf 'export TRAIN_LOG_EVERY=%q\n' "$TRAIN_LOG_EVERY"
    printf 'export WARMUP_STEPS=%q\n' "$WARMUP_STEPS"
    printf 'export WARMDOWN_ITERS=%q\n' "$WARMDOWN_ITERS"
    printf 'export EVAL_INITIAL=%q\n' "$EVAL_INITIAL"
    printf 'export VAL_MAX_TOKENS=%q\n' "$VAL_MAX_TOKENS"
    printf 'export TORICBLM_PREFLIGHT_READY_CHECK=%q\n' "${CURRICULUM_PREFLIGHT_READY_CHECK:-0}"
    printf 'export TORICBLM_TRAINING_CORPUS_AUDIT_SAMPLE_FILES=%q\n' "${CURRICULUM_AUDIT_SAMPLE_FILES:-6}"
    printf 'export TORICBLM_FOT_PREFLIGHT_SAMPLE_FILES=%q\n' "${CURRICULUM_FOT_PREFLIGHT_SAMPLE_FILES:-8}"
    printf 'export WANDB_RUN_ID=%q\n' "${WANDB_RUN_ID:-$RUN_ID}"
    printf 'export WANDB_NAME=%q\n' "${WANDB_NAME:-$RUN_ID}"
    printf 'export WANDB_RESUME=allow\n'
  } > "$epoch_config"

  (
    export CONFIG_PATH="$epoch_config"
    bash "$ROOT/scripts/launch_toricblm_mup_full_convextok8192_biomed.sh"
  )

  local target_ckpt="$CHECKPOINT_DIR/${RUN_ID}_step_$(printf '%06d' "$target_step").pt"
  if [[ ! -f "$target_ckpt" ]]; then
    echo "Expected epoch checkpoint missing: $target_ckpt" >&2
    exit 1
  fi
  local special_ckpt="$CHECKPOINT_DIR/${RUN_ID}_epoch_${epoch_padded}_special_${mode}_step_$(printf '%06d' "$target_step").pt"
  ln -f "$target_ckpt" "$special_ckpt" 2>/dev/null || cp -p "$target_ckpt" "$special_ckpt"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$epoch" "$mode" "$start_step" "$epoch_steps" "$target_step" "$selected_rows" "$structure_rows" "$special_ckpt" "$manifest" >> "$SUMMARY"

  if [[ "${HF_EPOCH_ARTIFACT_UPLOAD:-0}" == "1" ]]; then
    if ! "$ROOT/scripts/publish_toricblm_epoch_artifacts_to_hf.sh" \
      "$epoch_padded" "$RUN_ID" "$special_ckpt" "$manifest" "$epoch_dir" "$CHECKPOINT_DIR" "$SUMMARY"; then
      if [[ "${HF_EPOCH_ARTIFACT_UPLOAD_REQUIRED:-0}" == "1" ]]; then
        exit 1
      fi
      echo "hf_epoch_artifact_upload_failed_but_training_continues epoch:${epoch_padded}" >&2
    fi
  fi

  if [[ "$epoch" == "1" && "${PUSH_CODEBASE_ON_START:-0}" == "1" ]]; then
    if ! "$ROOT/scripts/push_toricgt_code_snapshot.sh"; then
      if [[ "${PUSH_CODEBASE_REQUIRED:-0}" == "1" ]]; then
        exit 1
      fi
      echo "codebase_push_failed_but_training_continues epoch:${epoch_padded}" >&2
    fi
  fi

  LAST_MANIFEST="$manifest"
  LAST_CHECKPOINT="$special_ckpt"
}

LAST_MANIFEST=""
LAST_CHECKPOINT=""
run_epoch 1 "structure_current" "" ""
EPOCH1_MANIFEST="$LAST_MANIFEST"
EPOCH1_CHECKPOINT="$LAST_CHECKPOINT"
run_epoch 2 "structure_delta" "$EPOCH1_MANIFEST" "$EPOCH1_CHECKPOINT"
EPOCH2_CHECKPOINT="$LAST_CHECKPOINT"
run_epoch 3 "all_entries" "" "$EPOCH2_CHECKPOINT"
EPOCH3_MANIFEST="$LAST_MANIFEST"
EPOCH3_CHECKPOINT="$LAST_CHECKPOINT"
run_epoch 4 "structure_delta" "$EPOCH3_MANIFEST" "$EPOCH3_CHECKPOINT"

echo "structure_priority_curriculum_complete run_id:$RUN_ID summary:$SUMMARY checkpoint_dir:$CHECKPOINT_DIR"
