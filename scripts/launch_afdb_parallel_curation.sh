#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/iska/miniconda3/envs/iska-net-2/bin/python}"
WORKERS="${AFDB_PARALLEL_WORKERS:-16}"
OUT_ROOT="${AFDB_PARALLEL_OUT_ROOT:-data/uniprot_fot/structures/afdb_parallel_uniref50_full}"
INPUT_GLOBS="${AFDB_PARALLEL_INPUT_GLOBS:-/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale/uniprot_uniref50_sequence_train/default/train/*.parquet}"
MAX_RECORDS_TOTAL="${AFDB_PARALLEL_MAX_RECORDS:-5000000}"
MAX_SCAN_ROWS="${AFDB_PARALLEL_MAX_SCAN_ROWS:-0}"
MAX_RESIDUES="${AFDB_MAX_RESIDUES:-512}"
SHARD_SIZE="${AFDB_SHARD_SIZE:-1024}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
LOCAL_CIF_GLOB="${AFDB_LOCAL_CIF_GLOB:-../iska-net-2/data/raw/alphafold_db/cif/AF-*-F1-model_v6.cif}"
EMIT_3DI="${AFDB_EMIT_3DI:-1}"
REQUIRE_3DI="${AFDB_REQUIRE_3DI:-0}"
PREFIX="${AFDB_PARALLEL_PREFIX:-toricblm_afdb_parallel_structure_fot}"

mkdir -p "$OUT_ROOT" logs
MANIFEST_DIR="$OUT_ROOT/_worker_inputs"
rm -rf "$MANIFEST_DIR"
mkdir -p "$MANIFEST_DIR"

echo "[afdb-parallel] root=$ROOT"
echo "[afdb-parallel] workers=$WORKERS out_root=$OUT_ROOT"
echo "[afdb-parallel] input_globs=$INPUT_GLOBS"

"$PY" - "$WORKERS" "$MANIFEST_DIR" "$INPUT_GLOBS" <<'PY'
import glob
import sys
from pathlib import Path

workers = int(sys.argv[1])
manifest_dir = Path(sys.argv[2])
patterns = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
paths: list[str] = []
seen: set[str] = set()
for pattern in patterns:
    for path in sorted(glob.glob(pattern)):
        key = str(Path(path).resolve())
        if key not in seen:
            seen.add(key)
            paths.append(path)
if not paths:
    raise SystemExit(f"no input parquet files matched: {patterns}")
for worker in range(workers):
    worker_paths = paths[worker::workers]
    (manifest_dir / f"worker_{worker:02d}.txt").write_text("\n".join(worker_paths) + "\n", encoding="utf-8")
print(f"[afdb-parallel] matched_input_files={len(paths)}")
for worker in range(workers):
    count = len(paths[worker::workers])
    print(f"[afdb-parallel] worker={worker:02d} files={count}")
PY

quota=0
if [[ "$MAX_RECORDS_TOTAL" != "0" ]]; then
  quota=$(( (MAX_RECORDS_TOTAL + WORKERS - 1) / WORKERS ))
fi

pids=()
for worker in $(seq 0 $((WORKERS - 1))); do
  wid="$(printf '%02d' "$worker")"
  worker_out="$OUT_ROOT/worker_${wid}"
  worker_log="logs/afdb_parallel_worker_${wid}_$(date -u +%Y%m%dT%H%M%SZ).log"
  args=(
    scripts/build_afdb_structure_fot_dataset.py
    --input-list "$MANIFEST_DIR/worker_${wid}.txt"
    --out-dir "$worker_out"
    --max-records "$quota"
    --max-scan-rows "$MAX_SCAN_ROWS"
    --max-residues "$MAX_RESIDUES"
    --shard-size "$SHARD_SIZE"
    --min-free-gb "$MIN_FREE_GB"
    --resume
    --prefix "$PREFIX"
  )
  if [[ -n "$LOCAL_CIF_GLOB" ]]; then
    IFS=',' read -r -a local_patterns <<< "$LOCAL_CIF_GLOB"
    for pattern in "${local_patterns[@]}"; do
      [[ -n "$pattern" ]] && args+=(--local-cif-glob "$pattern")
    done
  fi
  if [[ "$EMIT_3DI" == "1" ]]; then
    args+=(--emit-foldseek-3di)
  fi
  if [[ "$REQUIRE_3DI" == "1" ]]; then
    args+=(--require-foldseek-3di)
  fi
  echo "[afdb-parallel] launch worker=$wid quota=$quota log=$worker_log"
  (
    set -euo pipefail
    "$PY" "${args[@]}"
  ) >"$worker_log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" != "0" ]]; then
  echo "[afdb-parallel] one or more workers failed" >&2
  exit 1
fi

echo "[afdb-parallel] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
