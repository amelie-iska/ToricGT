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
PDB_OUT="${PDB_OUT:-$OUT_ROOT/pdb_modal}"
PUBCHEM_OUT="${PUBCHEM_OUT:-$OUT_ROOT/pubchem3d}"
MANIFEST="${MANIFEST:-data/uniprot_fot/manifests/toricblm_multimodal_structure_readiness.json}"

AFDB_MAX_RECORDS="${AFDB_MAX_RECORDS:-0}"
AFDB_MAX_SCAN_ROWS="${AFDB_MAX_SCAN_ROWS:-0}"
AFDB_SHARD_SIZE="${AFDB_SHARD_SIZE:-1024}"
AFDB_MAX_RESIDUES="${AFDB_MAX_RESIDUES:-512}"
AFDB_EMIT_3DI="${AFDB_EMIT_3DI:-1}"

PDB_QUERY_LIMIT="${PDB_QUERY_LIMIT:-512}"
PDB_MAX_RECORDS="${PDB_MAX_RECORDS:-0}"
PDB_MAX_ATOMS="${PDB_MAX_ATOMS:-512}"
PDB_MODALITIES="${PDB_MODALITIES:-protein rna dna protein_rna protein_dna complex ligand}"

echo "[toricblm-curation] root=$ROOT"
echo "[toricblm-curation] output=$OUT_ROOT min_free_gb=$MIN_FREE_GB"

AFDB_ARGS=(
  scripts/build_afdb_structure_fot_dataset.py
  --out-dir "$AFDB_OUT"
  --max-records "$AFDB_MAX_RECORDS"
  --max-scan-rows "$AFDB_MAX_SCAN_ROWS"
  --max-residues "$AFDB_MAX_RESIDUES"
  --shard-size "$AFDB_SHARD_SIZE"
  --min-free-gb "$MIN_FREE_GB"
  --resume
)
if [[ "$AFDB_EMIT_3DI" == "1" ]]; then
  AFDB_ARGS+=(--emit-foldseek-3di)
fi
"$PY" "${AFDB_ARGS[@]}"

PDB_ARGS=(
  scripts/build_pdb_modal_structure_fot_dataset.py
  --out-dir "$PDB_OUT"
  --max-records "$PDB_MAX_RECORDS"
  --max-atoms "$PDB_MAX_ATOMS"
  --rcsb-query-limit "$PDB_QUERY_LIMIT"
  --include-ligands
)
for modality in $PDB_MODALITIES; do
  PDB_ARGS+=(--rcsb-query-modality "$modality")
done
"$PY" "${PDB_ARGS[@]}"

if [[ -n "${PUBCHEM_CID_FILE:-}" ]]; then
  "$PY" scripts/build_pubchem3d_structure_fot_dataset.py \
    --cid-file "$PUBCHEM_CID_FILE" \
    --out-dir "$PUBCHEM_OUT" \
    --max-atoms "${PUBCHEM_MAX_ATOMS:-256}" \
    --shard-size "${PUBCHEM_SHARD_SIZE:-2048}"
else
  echo "[toricblm-curation] PUBCHEM_CID_FILE not set; skipping PubChem3D curation instead of fabricating conformers."
fi

MANIFEST_INPUTS=(
  "$AFDB_OUT/train/*.parquet"
  "$AFDB_OUT/validation/*.parquet"
  "$AFDB_OUT/test/*.parquet"
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
"$PY" "${MANIFEST_ARGS[@]}"

echo "[toricblm-curation] wrote readiness manifest: $MANIFEST"
