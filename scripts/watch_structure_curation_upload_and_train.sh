#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSIONS="${SESSIONS:-${SESSION:-$(cat logs/active_structure_curation_session.txt 2>/dev/null || true)}}"
if [[ -z "$SESSIONS" ]]; then
  echo "No curation tmux session supplied and logs/active_structure_curation_session.txt is missing." >&2
  exit 1
fi

PY="${PY:-/home/iska/miniconda3/envs/iska-net-2/bin/python}"
PROTREK_PY="${PROTREK_PY:-/home/iska/miniconda3/envs/protrek/bin/python}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
REPORT_DIR="${REPORT_DIR:-data/uniprot_fot/manifests}"
STRICT_MANIFEST="${STRICT_MANIFEST:-$REPORT_DIR/toricblm_multimodal_structure_readiness.json}"
PROTREK_REPORT="${PROTREK_REPORT:-$REPORT_DIR/toricblm_protrek_structure_split_report.json}"
PROTREK_SPLIT_DIR="${PROTREK_SPLIT_DIR:-data/uniprot_fot/splits/protrek_structure_current}"
RUN_PROTREK_SPLIT="${RUN_PROTREK_SPLIT:-1}"
RUN_HF_UPLOAD="${RUN_HF_UPLOAD:-1}"
RUN_TRAIN_IF_READY="${RUN_TRAIN_IF_READY:-1}"
TARGET_TOTAL_COORDINATE_ROWS="${TARGET_TOTAL_COORDINATE_ROWS:-10000000}"
TARGET_PROTEIN_STRUCTURES="${TARGET_PROTEIN_STRUCTURES:-5000000}"
TARGET_SMALL_MOLECULE_STRUCTURES="${TARGET_SMALL_MOLECULE_STRUCTURES:-5000000}"
AFDB_PARALLEL_ROOT="${AFDB_PARALLEL_ROOT:-data/uniprot_fot/structures/afdb_parallel_uniref50_full}"

mkdir -p "$REPORT_DIR" logs
WATCH_LOG="${WATCH_LOG:-logs/structure_curation_postprocess_$(date -u +%Y%m%dT%H%M%SZ).log}"

{
  echo "[watcher] sessions=$SESSIONS start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  while true; do
    RUNNING=0
    IFS=',' read -r -a SESSION_LIST <<< "$SESSIONS"
    for session in "${SESSION_LIST[@]}"; do
      if [[ -n "$session" ]] && tmux has-session -t "$session" 2>/dev/null; then
        RUNNING=1
      fi
    done
    if [[ "$RUNNING" == "0" ]]; then
      break
    fi
    echo "[watcher] one or more curation sessions still running $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sleep "$SLEEP_SECONDS"
  done
  echo "[watcher] curation sessions ended $(date -u +%Y-%m-%dT%H:%M:%SZ)"

  STRICT_CMD=(
    "$PY" scripts/build_toricblm_modality_readiness_manifest.py
    --output-json "$STRICT_MANIFEST"
    --input "data/uniprot_fot/structures/afdb_v6/toricblm_afdb_structure_fot_train.parquet"
    --input "data/uniprot_fot/structures/afdb_v6_full/train/*.parquet"
    --input "data/uniprot_fot/structures/afdb_uniref50_full/train/*.parquet"
    --input "$AFDB_PARALLEL_ROOT/worker_*/train/*.parquet"
    --input "data/uniprot_fot/structures/pdb_modal/train/*.parquet"
    --input "data/uniprot_fot/structures/pubchem3d/train/*.parquet"
    --require protein_afdb
    --require pdb_protein
    --require pdb_rna
    --require pdb_dna
    --require pdb_complex
    --require pubchem_3d_or_ligand
    --min-coordinate-rows "$TARGET_TOTAL_COORDINATE_ROWS"
    --min-count "protein_afdb=$TARGET_PROTEIN_STRUCTURES"
    --min-count "pubchem_3d_or_ligand=$TARGET_SMALL_MOLECULE_STRUCTURES"
  )
  set +e
  "${STRICT_CMD[@]}"
  READY_STATUS=$?
  set -e
  echo "[watcher] strict_readiness_exit=$READY_STATUS manifest=$STRICT_MANIFEST"

  if [[ "$RUN_PROTREK_SPLIT" == "1" ]]; then
    "$PROTREK_PY" scripts/split_with_protrek_trimodal.py \
      --input "data/uniprot_fot/structures/afdb_v6/toricblm_afdb_structure_fot_train.parquet" \
      --input "data/uniprot_fot/structures/afdb_v6_full/train/*.parquet" \
      --input "data/uniprot_fot/structures/afdb_uniref50_full/train/*.parquet" \
      --input "$AFDB_PARALLEL_ROOT/worker_*/train/*.parquet" \
      --input "data/uniprot_fot/structures/pdb_modal/train/*.parquet" \
      --output-dir "$PROTREK_SPLIT_DIR" \
      --report "$PROTREK_REPORT" \
      --prefix toricblm_protrek_structure_current \
      --batch-size "${PROTREK_BATCH_SIZE:-16}" \
      --max-clusters "${PROTREK_MAX_CLUSTERS:-8192}" \
      --require-protrek
    echo "[watcher] protrek_split_done report=$PROTREK_REPORT"
  fi

  if [[ "$RUN_HF_UPLOAD" == "1" ]]; then
    REPO_ID="${REPO_ID:-AmelieSchreiber/toricblm_fot}" NUM_WORKERS="${HF_UPLOAD_WORKERS:-8}" \
      bash scripts/upload_toricblm_fot_dataset_to_hf.sh
    echo "[watcher] hf_upload_done repo=${REPO_ID:-AmelieSchreiber/toricblm_fot}"
  fi

  if [[ "$RUN_TRAIN_IF_READY" == "1" && "$READY_STATUS" == "0" ]]; then
    echo "[watcher] strict readiness passed; launching structure training"
    CONFIG_PATH="$ROOT/configs/toricblm_mup_170m_convextok8192_biomed_fot_structure.env" \
      bash scripts/launch_toricblm_mup_full_convextok8192_biomed.sh
  else
    echo "[watcher] training not launched; strict readiness did not pass or RUN_TRAIN_IF_READY=0"
  fi
} 2>&1 | tee "$WATCH_LOG"
