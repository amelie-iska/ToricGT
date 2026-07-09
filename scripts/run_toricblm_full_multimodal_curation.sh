#!/usr/bin/env bash
set -euo pipefail

# Disk-safe ToricBLM multimodal curation launcher.
#
# This runs real-data curation only:
# - AFDB mmCIF coordinates with optional saved Foldseek/3Di sequences.
# - RCSB/PDB mmCIF modality queries for protein/RNA/DNA/complex records.
# - PubChem3D only from real SDF conformers when a CID file is supplied.
# - strict readiness manifests; missing required modalities fail loudly.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/iska/miniconda3/envs/iska-net-2/bin/python}"
OUT_ROOT="${OUT_ROOT:-data/uniprot_fot/structures}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"
AFDB_OUT="${AFDB_OUT:-$OUT_ROOT/afdb_v6_full}"
AFDB_UNIREF_OUT="${AFDB_UNIREF_OUT:-$OUT_ROOT/afdb_uniref50_full}"
PDB_OUT="${PDB_OUT:-$OUT_ROOT/pdb_modal}"
PUBCHEM_OUT="${PUBCHEM_OUT:-$OUT_ROOT/pubchem3d}"
PUBCHEM_CID_FILE="${PUBCHEM_CID_FILE:-$ROOT/data/uniprot_fot/pubchem/pubchem_cids_5m_from_naturelm_cid_smiles.txt}"
# The launcher writes an inventory manifest by default.  The training launcher
# independently writes/validates the strict readiness manifest before a run.
# This prevents a still-growing corpus from being mistaken for training-ready.
MANIFEST="${MANIFEST:-data/uniprot_fot/manifests/toricblm_multimodal_structure_curation_inventory.json}"
AFDB_FUNCTION_GLOB="${AFDB_FUNCTION_GLOB:-/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale/uniprot_function_text_train/default/train/*.parquet}"
AFDB_UNIREF_GLOB="${AFDB_UNIREF_GLOB:-/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale/uniprot_uniref50_sequence_train/default/train/*.parquet}"

AFDB_MAX_RECORDS="${AFDB_MAX_RECORDS:-5000000}"
AFDB_MAX_SCAN_ROWS="${AFDB_MAX_SCAN_ROWS:-0}"
AFDB_SHARD_SIZE="${AFDB_SHARD_SIZE:-1024}"
AFDB_MAX_RESIDUES="${AFDB_MAX_RESIDUES:-512}"
AFDB_EMIT_3DI="${AFDB_EMIT_3DI:-1}"

PDB_QUERY_LIMIT="${PDB_QUERY_LIMIT:-0}"
PDB_QUERY_PAGE_SIZE="${PDB_QUERY_PAGE_SIZE:-1000}"
PDB_MAX_RECORDS="${PDB_MAX_RECORDS:-0}"
PDB_MAX_ATOMS="${PDB_MAX_ATOMS:-512}"
PDB_MODALITIES="${PDB_MODALITIES-protein rna dna protein_rna protein_dna nucleic_acid complex ligand}"
PDB_LOCAL_STRUCTURE_GLOB="${PDB_LOCAL_STRUCTURE_GLOB:-}"
AFDB_LOCAL_CIF_GLOB="${AFDB_LOCAL_CIF_GLOB:-}"

TARGET_TOTAL_COORDINATE_ROWS="${TARGET_TOTAL_COORDINATE_ROWS:-10000000}"
TARGET_PROTEIN_STRUCTURES="${TARGET_PROTEIN_STRUCTURES:-5000000}"
TARGET_SMALL_MOLECULE_STRUCTURES="${TARGET_SMALL_MOLECULE_STRUCTURES:-5000000}"
ALLOW_STRUCTURE_TARGET_SHORTFALL="${ALLOW_STRUCTURE_TARGET_SHORTFALL:-0}"
PUBCHEM_USE_FTP_BULK="${PUBCHEM_USE_FTP_BULK:-1}"
PUBCHEM_FTP_MAX_FILES="${PUBCHEM_FTP_MAX_FILES:-0}"
PUBCHEM_FTP_START_AFTER="${PUBCHEM_FTP_START_AFTER:-}"

echo "[toricblm-curation] root=$ROOT"
echo "[toricblm-curation] output=$OUT_ROOT min_free_gb=$MIN_FREE_GB"

if [[ "${RUN_PDB:-1}" == "1" ]]; then
echo "[toricblm-curation] phase=pdb_all_available start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PDB_ARGS=(
  scripts/build_pdb_modal_structure_fot_dataset.py
  --out-dir "$PDB_OUT"
  --max-records "$PDB_MAX_RECORDS"
  --max-atoms "$PDB_MAX_ATOMS"
  --rcsb-query-limit "$PDB_QUERY_LIMIT"
  --rcsb-query-page-size "$PDB_QUERY_PAGE_SIZE"
  --min-free-gb "$MIN_FREE_GB"
  --include-ligands
  --remove-downloaded-cache
  --resume
)
if [[ -n "$PDB_LOCAL_STRUCTURE_GLOB" ]]; then
  IFS=',' read -r -a PDB_LOCAL_PATTERNS <<< "$PDB_LOCAL_STRUCTURE_GLOB"
  for pattern in "${PDB_LOCAL_PATTERNS[@]}"; do
    [[ -n "$pattern" ]] && PDB_ARGS+=(--local-structure-glob "$pattern")
  done
fi
for modality in $PDB_MODALITIES; do
  PDB_ARGS+=(--rcsb-query-modality "$modality")
done
"$PY" "${PDB_ARGS[@]}"
echo "[toricblm-curation] phase=pdb_all_available done=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
else
  echo "[toricblm-curation] phase=pdb_all_available skipped"
fi

if [[ "${RUN_PUBCHEM:-1}" == "1" ]]; then
echo "[toricblm-curation] phase=pubchem3d_5m_cid_stream start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PUBCHEM_ARGS=(
  scripts/build_pubchem3d_structure_fot_dataset.py
  --out-dir "$PUBCHEM_OUT"
  --max-records "${PUBCHEM_MAX_RECORDS:-5000000}"
  --max-atoms "${PUBCHEM_MAX_ATOMS:-256}"
  --shard-size "${PUBCHEM_SHARD_SIZE:-2048}"
  --cid-batch-size "${PUBCHEM_CID_BATCH_SIZE:-64}"
  --min-free-gb "$MIN_FREE_GB"
  --remove-downloaded-cache
  --resume
)
if [[ "$PUBCHEM_USE_FTP_BULK" == "1" ]]; then
  PUBCHEM_ARGS+=(--pubchem3d-ftp-bulk)
  if [[ "$PUBCHEM_FTP_MAX_FILES" != "0" ]]; then
    PUBCHEM_ARGS+=(--ftp-max-files "$PUBCHEM_FTP_MAX_FILES")
  fi
  if [[ -n "$PUBCHEM_FTP_START_AFTER" ]]; then
    PUBCHEM_ARGS+=(--ftp-start-after "$PUBCHEM_FTP_START_AFTER")
  fi
fi
if [[ -n "${PUBCHEM_CID_FILE:-}" && -s "$PUBCHEM_CID_FILE" ]]; then
  PUBCHEM_ARGS+=(--cid-file "$PUBCHEM_CID_FILE")
else
  echo "[toricblm-curation] PUBCHEM_CID_FILE missing or empty (${PUBCHEM_CID_FILE:-unset}); using FTP bulk only."
fi
"$PY" "${PUBCHEM_ARGS[@]}"
echo "[toricblm-curation] phase=pubchem3d_5m_cid_stream done=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
else
  echo "[toricblm-curation] phase=pubchem3d_5m_cid_stream skipped"
fi

if [[ "${RUN_AFDB:-1}" == "1" ]]; then
echo "[toricblm-curation] phase=afdb_5m_resume start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
AFDB_ARGS=(
  scripts/build_afdb_structure_fot_dataset.py
  --out-dir "$AFDB_OUT"
  --uniprot-function-glob "$AFDB_FUNCTION_GLOB"
  --max-records "${AFDB_MAX_RECORDS:-5000000}"
  --max-scan-rows "$AFDB_MAX_SCAN_ROWS"
  --max-residues "$AFDB_MAX_RESIDUES"
  --shard-size "$AFDB_SHARD_SIZE"
  --min-free-gb "$MIN_FREE_GB"
  --resume
)
if [[ -n "$AFDB_LOCAL_CIF_GLOB" ]]; then
  IFS=',' read -r -a AFDB_LOCAL_PATTERNS <<< "$AFDB_LOCAL_CIF_GLOB"
  for pattern in "${AFDB_LOCAL_PATTERNS[@]}"; do
    [[ -n "$pattern" ]] && AFDB_ARGS+=(--local-cif-glob "$pattern")
  done
fi
if [[ "$AFDB_EMIT_3DI" == "1" ]]; then
  AFDB_ARGS+=(--emit-foldseek-3di)
fi
"$PY" "${AFDB_ARGS[@]}"
echo "[toricblm-curation] phase=afdb_5m_resume done=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "[toricblm-curation] phase=afdb_uniref50_5m_resume start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
AFDB_UNIREF_ARGS=(
  scripts/build_afdb_structure_fot_dataset.py
  --out-dir "$AFDB_UNIREF_OUT"
  --uniprot-function-glob "$AFDB_UNIREF_GLOB"
  --max-records "${AFDB_UNIREF_MAX_RECORDS:-5000000}"
  --max-scan-rows "${AFDB_UNIREF_MAX_SCAN_ROWS:-0}"
  --max-residues "$AFDB_MAX_RESIDUES"
  --shard-size "$AFDB_SHARD_SIZE"
  --min-free-gb "$MIN_FREE_GB"
  --resume
  --prefix "toricblm_afdb_uniref50_structure_fot"
)
if [[ -n "$AFDB_LOCAL_CIF_GLOB" ]]; then
  IFS=',' read -r -a AFDB_UNIREF_LOCAL_PATTERNS <<< "$AFDB_LOCAL_CIF_GLOB"
  for pattern in "${AFDB_UNIREF_LOCAL_PATTERNS[@]}"; do
    [[ -n "$pattern" ]] && AFDB_UNIREF_ARGS+=(--local-cif-glob "$pattern")
  done
fi
if [[ "$AFDB_EMIT_3DI" == "1" ]]; then
  AFDB_UNIREF_ARGS+=(--emit-foldseek-3di)
fi
"$PY" "${AFDB_UNIREF_ARGS[@]}"
echo "[toricblm-curation] phase=afdb_uniref50_5m_resume done=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
else
  echo "[toricblm-curation] phase=afdb_5m_resume skipped"
  echo "[toricblm-curation] phase=afdb_uniref50_5m_resume skipped"
fi

MANIFEST_INPUTS=(
  "$AFDB_OUT/train/*.parquet"
  "$AFDB_OUT/validation/*.parquet"
  "$AFDB_OUT/test/*.parquet"
  "$AFDB_UNIREF_OUT/train/*.parquet"
  "$AFDB_UNIREF_OUT/validation/*.parquet"
  "$AFDB_UNIREF_OUT/test/*.parquet"
  "$PDB_OUT/train/*.parquet"
  "$PDB_OUT/validation/*.parquet"
  "$PDB_OUT/test/*.parquet"
  "$PUBCHEM_OUT/train/*.parquet"
  "$PUBCHEM_OUT/validation/*.parquet"
  "$PUBCHEM_OUT/test/*.parquet"
)
MANIFEST_ARGS=(scripts/build_toricblm_modality_readiness_manifest.py --output-json "$MANIFEST")
for pattern in "${MANIFEST_INPUTS[@]}"; do
  MANIFEST_ARGS+=(--input "$pattern")
done
MANIFEST_ARGS+=(
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
if [[ "$ALLOW_STRUCTURE_TARGET_SHORTFALL" == "1" ]]; then
  MANIFEST_ARGS+=(--allow-shortfall)
fi
"$PY" "${MANIFEST_ARGS[@]}"

echo "[toricblm-curation] wrote readiness manifest: $MANIFEST"
