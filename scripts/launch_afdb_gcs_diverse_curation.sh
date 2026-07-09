#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/iska/miniconda3/envs/iska-net-2/bin/python}"
OUT_ROOT="${AFDB_GCS_OUT_ROOT:-data/uniprot_fot/structures/afdb_gcs_v4}"
PLAN_ROOT="${AFDB_GCS_PLAN_ROOT:-$OUT_ROOT/_diverse_selection_plan}"
WORKERS="${AFDB_GCS_WORKERS:-6}"
TARGET_RECORDS="${AFDB_GCS_TARGET_RECORDS:-5000000}"
MAX_SCAN_ROWS="${AFDB_GCS_MAX_SCAN_ROWS:-0}"
ENZYME_TARGET_FRACTION="${AFDB_GCS_ENZYME_TARGET_FRACTION:-0.15}"
ENZYME_HIGH_FRACTION="${AFDB_GCS_ENZYME_HIGH_FRACTION:-0.50}"
ENZYME_MID_FRACTION="${AFDB_GCS_ENZYME_MID_FRACTION:-0.25}"
ENZYME_LOW_FRACTION="${AFDB_GCS_ENZYME_LOW_FRACTION:-0.25}"
INPUT_GLOBS="${AFDB_GCS_INPUT_GLOBS:-/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale/uniprot_function_text_train/default/train/*.parquet,/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale/uniprot_uniref50_sequence_train/default/train/*.parquet}"
MAX_RESIDUES="${AFDB_MAX_RESIDUES:-512}"
SHARD_SIZE="${AFDB_SHARD_SIZE:-2048}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
BATCH_SIZE="${AFDB_GCS_BATCH_SIZE:-128}"
EMIT_3DI="${AFDB_EMIT_3DI:-1}"
REQUIRE_3DI="${AFDB_REQUIRE_3DI:-0}"
BUCKET="${AFDB_GCS_BUCKET:-gs://public-datasets-deepmind-alphafold-v4}"
VERSION="${AFDB_GCS_VERSION:-4}"
PREFIX="${AFDB_GCS_PREFIX:-toricblm_afdb_gcs_structure_fot}"
SESSION="${SESSION:-toricblm_afdb_gcs_curation_$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$OUT_ROOT" "$PLAN_ROOT" logs

PLAN_MANIFEST="$PLAN_ROOT/toricblm_afdb_gcs_diverse_plan_manifest.json"
if [[ ! -s "$PLAN_MANIFEST" || "${AFDB_GCS_REPLAN:-0}" == "1" ]]; then
  echo "[afdb-gcs] building capped diversity/enzyme selection plan out=$PLAN_ROOT target=$TARGET_RECORDS workers=$WORKERS"
  PLAN_ARGS=(
    scripts/plan_afdb_gcs_diverse_accessions.py
    --output-dir "$PLAN_ROOT"
    --workers "$WORKERS"
    --target-records "$TARGET_RECORDS"
    --max-scan-rows "$MAX_SCAN_ROWS"
    --enzyme-target-fraction "$ENZYME_TARGET_FRACTION"
    --enzyme-high-fraction "$ENZYME_HIGH_FRACTION"
    --enzyme-mid-fraction "$ENZYME_MID_FRACTION"
    --enzyme-low-fraction "$ENZYME_LOW_FRACTION"
  )
  IFS=',' read -r -a input_patterns <<< "$INPUT_GLOBS"
  for pattern in "${input_patterns[@]}"; do
    [[ -n "$pattern" ]] && PLAN_ARGS+=(--input "$pattern")
  done
  for pattern in \
    "data/uniprot_fot/structures/afdb_v6/toricblm_afdb_structure_fot_*.parquet" \
    "data/uniprot_fot/structures/afdb_v6_full/*/*.parquet" \
    "data/uniprot_fot/structures/afdb_uniref50_full/*/*.parquet" \
    "data/uniprot_fot/structures/afdb_parallel_uniref50_full/worker_*/*/*.parquet" \
    "data/uniprot_fot/structures/afdb_ebi_tar_v6/*/*.parquet" \
    "data/uniprot_fot/structures/afdb_ebi_tar_v6/worker_*/*/*.parquet" \
    "data/uniprot_fot/structures/afdb_gcs_v4/worker_*/*/*.parquet"; do
    PLAN_ARGS+=(--existing-parquet-glob "$pattern")
  done
  "$PY" "${PLAN_ARGS[@]}"
else
  echo "[afdb-gcs] using existing plan manifest=$PLAN_MANIFEST"
fi

tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash -lc '
set -euo pipefail
for idx in \$(seq 0 \$(( $WORKERS - 1 ))); do
  worker=\$(printf \"%02d\" \"\$idx\")
  plan=\"$PLAN_ROOT/worker_plans/worker_\${worker}.jsonl\"
  out=\"$OUT_ROOT/worker_\${worker}\"
  log=\"logs/afdb_gcs_worker_\${worker}_${SESSION}.log\"
  echo \"[afdb-gcs] launch worker=\$worker plan=\$plan out=\$out log=\$log\"
  mkdir -p \"\$out\"
  args=(
    scripts/build_afdb_gcs_structure_fot_dataset.py
    --plan-jsonl \"\$plan\"
    --out-dir \"\$out\"
    --bucket \"$BUCKET\"
    --version \"$VERSION\"
    --max-records 0
    --batch-size \"$BATCH_SIZE\"
    --max-residues \"$MAX_RESIDUES\"
    --shard-size \"$SHARD_SIZE\"
    --min-free-gb \"$MIN_FREE_GB\"
    --prefix \"$PREFIX\"
    --resume
  )
  if [[ \"$EMIT_3DI\" == \"1\" ]]; then
    args+=(--emit-foldseek-3di)
  fi
  if [[ \"$REQUIRE_3DI\" == \"1\" ]]; then
    args+=(--require-foldseek-3di)
  fi
  \"$PY\" \"\${args[@]}\" > \"\$log\" 2>&1 &
done
wait
echo \"[afdb-gcs] all workers completed \$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
'"

echo "$SESSION" > logs/active_afdb_gcs_curation_session.txt
echo "$SESSION"
