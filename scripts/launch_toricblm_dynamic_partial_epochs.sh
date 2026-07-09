#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/toricblm_mup_170m_convextok8192_biomed_fot_structure.env}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  exit 1
fi
BASE_CONFIG_PATH="$CONFIG_PATH"

PY="${TORICBLM_TOKENGT_PYTHON:-/home/iska/miniconda3/envs/tokengt/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python"
fi

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${RUN_ID:-toricblm-dynamic-partial-epochs-${RUN_STAMP}}"
RUN_BASE_DIR="${RUN_BASE_DIR:-$ROOT/runs/oai_sidecar/${RUN_ID}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints/${RUN_ID}}"
mkdir -p "$RUN_BASE_DIR" "$CHECKPOINT_DIR" logs

EPOCHS="${DYNAMIC_EPOCHS:-3}"
EPOCH_STEPS="${DYNAMIC_EPOCH_STEPS:-1000}"
if (( EPOCHS < 1 )); then
  echo "DYNAMIC_EPOCHS must be >= 1" >&2
  exit 1
fi
if (( EPOCH_STEPS < 1 )); then
  echo "DYNAMIC_EPOCH_STEPS must be >= 1" >&2
  exit 1
fi

LIVE_AFDB_ROOT="${LIVE_AFDB_ROOT:-$ROOT/data/uniprot_fot/structures/afdb_gcs_proteome_tar_v4_large}"
LIVE_AFDB_GLOB="${LIVE_AFDB_GLOB:-$LIVE_AFDB_ROOT/worker_*/train/*.parquet}"
EXTRA_STRUCTURE_GLOBS="${EXTRA_STRUCTURE_GLOBS:-}"
BASE_RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
if [[ -z "$BASE_RESUME_CHECKPOINT" ]]; then
  # shellcheck disable=SC1090
  source "$BASE_CONFIG_PATH"
  BASE_RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
fi
if [[ "${DYNAMIC_START_FROM_STEP0:-1}" == "1" || "${TORICBLM_START_FROM_STEP0:-0}" == "1" || "${START_FROM_STEP0:-0}" == "1" ]]; then
  BASE_RESUME_CHECKPOINT=""
fi

if [[ -n "$BASE_RESUME_CHECKPOINT" && ! -f "$BASE_RESUME_CHECKPOINT" ]]; then
  echo "Resume checkpoint does not exist: $BASE_RESUME_CHECKPOINT" >&2
  exit 1
fi

MANIFEST_DIR="$RUN_BASE_DIR/dynamic_epoch_manifests"
mkdir -p "$MANIFEST_DIR"
SUMMARY="$RUN_BASE_DIR/dynamic_epoch_summary.tsv"
printf 'epoch\tstart_step\ttarget_step\tlive_afdb_shards\tlive_afdb_bytes\tresume_checkpoint\tspecial_checkpoint\n' > "$SUMMARY"

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

write_epoch_manifest() {
  local epoch="$1"
  local out_json="$2"
  "$PY" - "$epoch" "$out_json" "$LIVE_AFDB_GLOB" <<'PY'
import glob
import json
import os
import sys
from pathlib import Path

epoch = int(sys.argv[1])
out = Path(sys.argv[2])
pattern = sys.argv[3]
files = [Path(path) for path in sorted(glob.glob(pattern))]
total_bytes = sum(path.stat().st_size for path in files if path.exists())
payload = {
    "epoch": epoch,
    "created_utc": __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "live_afdb_glob": pattern,
    "live_afdb_shards": len(files),
    "live_afdb_bytes": total_bytes,
    "live_afdb_gb": total_bytes / 1_000_000_000,
    "first_shards": [str(path) for path in files[:16]],
    "last_shards": [str(path) for path in files[-16:]],
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"shards": len(files), "bytes": total_bytes}, sort_keys=True))
PY
}

CURRENT_RESUME="$BASE_RESUME_CHECKPOINT"
for epoch in $(seq 1 "$EPOCHS"); do
  start_step="$(checkpoint_step "$CURRENT_RESUME")"
  target_step="$((start_step + EPOCH_STEPS))"
  epoch_padded="$(printf '%03d' "$epoch")"
  epoch_run_dir="$RUN_BASE_DIR/epoch_${epoch_padded}"
  mkdir -p "$epoch_run_dir"

  manifest_json="$MANIFEST_DIR/epoch_${epoch_padded}_available_shards.json"
  manifest_line="$(write_epoch_manifest "$epoch" "$manifest_json")"
  shard_count="$("$PY" - "$manifest_json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["live_afdb_shards"])
PY
)"
  shard_bytes="$("$PY" - "$manifest_json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["live_afdb_bytes"])
PY
)"

  if (( shard_count < 1 )); then
    echo "No live AFDB shards matched $LIVE_AFDB_GLOB" >&2
    exit 1
  fi

  protein_structure_glob="$LIVE_AFDB_GLOB"
  if [[ -n "$EXTRA_STRUCTURE_GLOBS" ]]; then
    protein_structure_glob="${protein_structure_glob},${EXTRA_STRUCTURE_GLOBS}"
  fi

  epoch_config="$epoch_run_dir/dynamic_epoch_${epoch_padded}.env"
  {
    printf 'source %q\n' "$BASE_CONFIG_PATH"
    printf 'export RUN_ID=%q\n' "$RUN_ID"
    printf 'export RUN_DIR=%q\n' "$epoch_run_dir"
    printf 'export CHECKPOINT_DIR=%q\n' "$CHECKPOINT_DIR"
    printf 'export RESUME_CHECKPOINT=%q\n' "$CURRENT_RESUME"
    if [[ -z "$CURRENT_RESUME" ]]; then
      printf 'export TORICBLM_START_FROM_STEP0=1\n'
      printf 'export START_STEP=0\n'
    else
      printf 'export TORICBLM_START_FROM_STEP0=0\n'
      printf 'unset START_STEP\n'
    fi
    printf 'export PROTREK_STRUCTURE_TRAIN_GLOB=%q\n' "$protein_structure_glob"
    printf 'export PROTEIN_STRUCTURE_PROTREK_TRAIN_GLOB=%q\n' "$protein_structure_glob"
    cat <<'SH'
export GRAPH_TRAIN_GLOB=${PROTEIN_FOT_TRAIN_GLOB},${PROTEIN_STRUCTURE_PROTREK_TRAIN_GLOB},${NONPROTEIN_FOT_TRAIN_GLOB},${RAW_BIO_SEQUENCE_TRAIN_GLOB},/home/iska/Documents/amelie/bio/ToricGT/data/uniprot_fot/structures/pdb_modal/train/*.parquet,/home/iska/Documents/amelie/bio/ToricGT/data/uniprot_fot/structures/pdb_modal_parallel/worker_*/train/*.parquet,/home/iska/Documents/amelie/bio/ToricGT/data/uniprot_fot/structures/pubchem3d/train/*.parquet,/home/iska/Documents/amelie/bio/ToricGT/data/uniprot_fot/structures/pubchem3d_parallel/worker_*/train/*.parquet,/home/iska/Documents/amelie/bio/ToricGT/data/toricblm_late_mixed_fot_structure/train/*.parquet
export LONG_ENTRY_TRAIN_GLOB=${GRAPH_TRAIN_GLOB}
export TORICBLM_STRUCTURE_TRAIN_GLOB=${PROTEIN_STRUCTURE_PROTREK_TRAIN_GLOB},/home/iska/Documents/amelie/bio/ToricGT/data/uniprot_fot/structures/pdb_modal/train/*.parquet,/home/iska/Documents/amelie/bio/ToricGT/data/uniprot_fot/structures/pdb_modal_parallel/worker_*/train/*.parquet,/home/iska/Documents/amelie/bio/ToricGT/data/uniprot_fot/structures/pubchem3d/train/*.parquet,/home/iska/Documents/amelie/bio/ToricGT/data/uniprot_fot/structures/pubchem3d_parallel/worker_*/train/*.parquet
SH
    printf 'export ITERATIONS=%q\n' "$target_step"
    printf 'export CHECKPOINT_EVERY=%q\n' "${DYNAMIC_CHECKPOINT_EVERY:-1}"
    printf 'export VAL_LOSS_EVERY=%q\n' "${DYNAMIC_VAL_LOSS_EVERY:-$EPOCH_STEPS}"
    printf 'export TRAIN_LOG_EVERY=%q\n' "${DYNAMIC_TRAIN_LOG_EVERY:-1}"
    printf 'export WARMDOWN_ITERS=%q\n' "${DYNAMIC_WARMDOWN_ITERS:-0}"
    printf 'export WARMUP_STEPS=%q\n' "${DYNAMIC_WARMUP_STEPS:-20}"
    printf 'export EVAL_INITIAL=%q\n' "${DYNAMIC_EVAL_INITIAL:-0}"
    printf 'export VAL_MAX_TOKENS=%q\n' "${DYNAMIC_VAL_MAX_TOKENS:-1048576}"
    printf 'export TORICBLM_PREFLIGHT_READY_CHECK=%q\n' "${DYNAMIC_PREFLIGHT_READY_CHECK:-0}"
    printf 'export TORICBLM_ALLOW_STRUCTURE_TARGET_SHORTFALL=1\n'
    printf 'export TORICBLM_TRAINING_CORPUS_AUDIT_SAMPLE_FILES=%q\n' "${DYNAMIC_AUDIT_SAMPLE_FILES:-2}"
    printf 'export TORICBLM_FOT_PREFLIGHT_SAMPLE_FILES=%q\n' "${DYNAMIC_FOT_PREFLIGHT_SAMPLE_FILES:-8}"
    printf 'export WANDB_RUN_ID=%q\n' "${WANDB_RUN_ID:-$RUN_ID}"
    printf 'export WANDB_NAME=%q\n' "${WANDB_NAME:-$RUN_ID}"
    printf 'export WANDB_RESUME=allow\n'
  } > "$epoch_config"

  echo "dynamic_epoch:${epoch}/${EPOCHS} start_step:${start_step} target_step:${target_step} live_afdb_shards:${shard_count} manifest:${manifest_json}"
  (
    export CONFIG_PATH="$epoch_config"
    bash "$ROOT/scripts/launch_toricblm_mup_full_convextok8192_biomed.sh"
  )

  target_ckpt="$CHECKPOINT_DIR/${RUN_ID}_step_$(printf '%06d' "$target_step").pt"
  if [[ ! -f "$target_ckpt" ]]; then
    echo "Expected epoch checkpoint missing: $target_ckpt" >&2
    exit 1
  fi
  special_ckpt="$CHECKPOINT_DIR/${RUN_ID}_epoch_${epoch_padded}_special_step_$(printf '%06d' "$target_step").pt"
  ln -f "$target_ckpt" "$special_ckpt" 2>/dev/null || cp -p "$target_ckpt" "$special_ckpt"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$epoch" "$start_step" "$target_step" "$shard_count" "$shard_bytes" "$CURRENT_RESUME" "$special_ckpt" >> "$SUMMARY"

  if [[ "${HF_EPOCH_ARTIFACT_UPLOAD:-0}" == "1" ]]; then
    if ! "$ROOT/scripts/publish_toricblm_epoch_artifacts_to_hf.sh" \
      "$epoch_padded" "$RUN_ID" "$special_ckpt" "$manifest_json" "$epoch_run_dir" "$CHECKPOINT_DIR" "$SUMMARY"; then
      if [[ "${HF_EPOCH_ARTIFACT_UPLOAD_REQUIRED:-0}" == "1" ]]; then
        exit 1
      fi
      echo "hf_epoch_artifact_upload_failed_but_training_continues epoch:$epoch_padded" >&2
    fi
  fi

  if [[ "$epoch" == "1" && "${PUSH_CODEBASE_ON_START:-0}" == "1" ]]; then
    if ! "$ROOT/scripts/push_toricgt_code_snapshot.sh"; then
      if [[ "${PUSH_CODEBASE_REQUIRED:-0}" == "1" ]]; then
        exit 1
      fi
      echo "codebase_push_failed_but_training_continues epoch:$epoch_padded" >&2
    fi
  fi

  CURRENT_RESUME="$special_ckpt"
done

echo "dynamic_epoch_training_complete run_id:$RUN_ID summary:$SUMMARY checkpoint_dir:$CHECKPOINT_DIR"
