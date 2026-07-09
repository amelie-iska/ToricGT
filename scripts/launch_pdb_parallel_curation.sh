#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/iska/miniconda3/envs/iska-net-2/bin/python}"
LOCAL_GLOB="${LOCAL_GLOB:-../iska-net-2/data/raw/rcsb/cif/*.cif}"
OUT_ROOT="${OUT_ROOT:-data/uniprot_fot/structures/pdb_modal_parallel}"
PRIMARY_OUT="${PRIMARY_OUT:-data/uniprot_fot/structures/pdb_modal}"
WORKERS="${PDB_PARALLEL_WORKERS:-12}"
MAX_ATOMS="${PDB_MAX_ATOMS:-1024}"
SHARD_SIZE="${PDB_SHARD_SIZE:-2048}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
PREFIX="${PREFIX:-toricblm_pdb_modal_parallel_structure_fot}"
SESSION="${SESSION:-toricblm_pdb_parallel_curation_$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$OUT_ROOT/_worker_lists" logs

"$PY" - "$LOCAL_GLOB" "$PRIMARY_OUT" "$OUT_ROOT" "$WORKERS" "$PREFIX" <<'PY'
import glob
import sys
from pathlib import Path

import pyarrow.parquet as pq

local_glob = sys.argv[1]
primary_out = Path(sys.argv[2])
out_root = Path(sys.argv[3])
workers = int(sys.argv[4])
prefix = sys.argv[5]


def collect_seen(root: Path, prefixes: tuple[str, ...]) -> set[str]:
    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for pattern in prefixes:
            for path in split_dir.glob(f"{pattern}_{split}_*.parquet"):
                try:
                    table = pq.read_table(path, columns=["structure_id"])
                except Exception:
                    continue
                seen.update(str(value).lower() for value in table["structure_id"].to_pylist() if value)
    return seen


seen = set()
seen |= collect_seen(primary_out, ("toricblm_pdb_modal_structure_fot", "toricblm_pdb_rna_dna_targeted_structure_fot"))
for worker_dir in out_root.glob("worker_*"):
    seen |= collect_seen(worker_dir, (prefix,))

paths = []
for raw in glob.glob(local_glob):
    path = Path(raw)
    if path.stem.lower() not in seen:
        paths.append(path)
paths.sort(key=lambda p: p.name)

lists_dir = out_root / "_worker_lists"
lists_dir.mkdir(parents=True, exist_ok=True)
handles = [(lists_dir / f"worker_{idx:02d}.txt").open("w", encoding="utf-8") for idx in range(workers)]
try:
    for idx, path in enumerate(paths):
        handles[idx % workers].write(str(path) + "\n")
finally:
    for handle in handles:
        handle.close()

summary = {
    "local_glob": local_glob,
    "local_input_files": len(glob.glob(local_glob)),
    "already_curated_ids": len(seen),
    "remaining_files": len(paths),
    "workers": workers,
}
for key, value in summary.items():
    print(f"{key}={value}")
PY

tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash -lc '
set -euo pipefail
for idx in \$(seq 0 \$(( $WORKERS - 1 ))); do
  worker=\$(printf \"%02d\" \"\$idx\")
  list=\"$OUT_ROOT/_worker_lists/worker_\${worker}.txt\"
  out=\"$OUT_ROOT/worker_\${worker}\"
  log=\"logs/pdb_parallel_worker_\${worker}_${SESSION}.log\"
  echo \"[pdb-parallel] launch worker=\$worker list=\$list out=\$out log=\$log\"
  mkdir -p \"\$out\"
  \"$PY\" scripts/build_pdb_modal_structure_fot_dataset.py \
    --out-dir \"\$out\" \
    --prefix \"$PREFIX\" \
    --local-structure-list \"\$list\" \
    --max-records 0 \
    --max-atoms \"$MAX_ATOMS\" \
    --shard-size \"$SHARD_SIZE\" \
    --min-free-gb \"$MIN_FREE_GB\" \
    --include-ligands \
    --resume \
    > \"\$log\" 2>&1 &
done
wait
echo \"[pdb-parallel] all workers completed \$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
'"

echo "$SESSION" > logs/active_pdb_parallel_curation_session.txt
echo "$SESSION"
