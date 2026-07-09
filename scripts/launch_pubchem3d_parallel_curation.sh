#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/iska/miniconda3/envs/iska-net-2/bin/python}"
PRIMARY_OUT="${PRIMARY_OUT:-data/uniprot_fot/structures/pubchem3d}"
OUT_ROOT="${OUT_ROOT:-data/uniprot_fot/structures/pubchem3d_parallel}"
WORKERS="${PUBCHEM3D_PARALLEL_WORKERS:-6}"
TARGET_TOTAL="${PUBCHEM3D_TARGET_TOTAL:-5000000}"
MAX_ATOMS="${PUBCHEM3D_MAX_ATOMS:-256}"
SHARD_SIZE="${PUBCHEM3D_SHARD_SIZE:-2048}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
PREFIX="${PREFIX:-toricblm_pubchem3d_parallel_structure_fot}"
FTP_HOST="${PUBCHEM3D_FTP_HOST:-ftp.ncbi.nlm.nih.gov}"
FTP_DIR="${PUBCHEM3D_FTP_DIR:-pubchem/Compound_3D/01_conf_per_cmpd/SDF}"
SESSION="${SESSION:-toricblm_pubchem3d_parallel_curation_$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$OUT_ROOT/_worker_lists" logs

SUMMARY="$("$PY" - "$PRIMARY_OUT" "$OUT_ROOT" "$WORKERS" "$TARGET_TOTAL" "$PREFIX" "$FTP_HOST" "$FTP_DIR" <<'PY'
from __future__ import annotations

import ftplib
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

primary_out = Path(sys.argv[1])
out_root = Path(sys.argv[2])
workers = int(sys.argv[3])
target_total = int(sys.argv[4])
prefix = sys.argv[5]
host = sys.argv[6]
directory = sys.argv[7]


def compound_ids(root: Path, prefixes: tuple[str, ...]) -> set[str]:
    ids: set[str] = set()
    for split in ("train", "validation", "test"):
        split_dir = root / split
        if split_dir.exists():
            for pattern in prefixes:
                for path in split_dir.glob(f"{pattern}_{split}_*.parquet"):
                    try:
                        table = pq.read_table(path, columns=["compound_id"])
                    except Exception:
                        continue
                    ids.update(str(value) for value in table["compound_id"].to_pylist() if value)
        for worker_dir in root.glob("worker_*"):
            worker_split = worker_dir / split
            if not worker_split.exists():
                continue
            for pattern in prefixes:
                for path in worker_split.glob(f"{pattern}_{split}_*.parquet"):
                    try:
                        table = pq.read_table(path, columns=["compound_id"])
                    except Exception:
                        continue
                    ids.update(str(value) for value in table["compound_id"].to_pylist() if value)
    return ids


def ftp_file_range(name: str) -> tuple[int, int] | None:
    nums = re.findall(r"(\d+)", name)
    if len(nums) < 2:
        return None
    return int(nums[0]), int(nums[1])


def ftp_file_index(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


seen = compound_ids(primary_out, ("toricblm_pubchem3d_structure_fot",))
seen |= compound_ids(out_root, (prefix,))
numeric_seen = [int(value) for value in seen if str(value).isdigit()]
max_seen = max(numeric_seen) if numeric_seen else 0
remaining_target = max(0, target_total - len(seen))

with ftplib.FTP(host, timeout=120) as ftp:
    ftp.login()
    ftp.cwd(directory)
    names = sorted({name for name in ftp.nlst() if name.endswith(".sdf.gz")}, key=lambda n: (ftp_file_index(n), n))

retained = []
for name in names:
    rng = ftp_file_range(name)
    if rng is not None and rng[1] <= max_seen:
        continue
    retained.append(name)

lists_dir = out_root / "_worker_lists"
lists_dir.mkdir(parents=True, exist_ok=True)
handles = [(lists_dir / f"worker_{idx:02d}.txt").open("w", encoding="utf-8") for idx in range(workers)]
try:
    for idx, name in enumerate(retained):
        handles[idx % workers].write(name + "\n")
finally:
    for handle in handles:
        handle.close()

print(f"existing_total={len(seen)}")
print(f"max_seen_cid={max_seen}")
print(f"remaining_target={remaining_target}")
print(f"retained_ftp_files={len(retained)}")
print(f"per_worker_target={(remaining_target + workers - 1) // workers if workers else 0}")
PY
)"

echo "$SUMMARY"
PER_WORKER_TARGET="$(printf '%s\n' "$SUMMARY" | awk -F= '/per_worker_target=/{print $2}' | tail -1)"
if [[ -z "$PER_WORKER_TARGET" || "$PER_WORKER_TARGET" == "0" ]]; then
  echo "[pubchem-parallel] target already satisfied; not launching workers"
  exit 0
fi

tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash -lc '
set -euo pipefail
export PUBCHEM_ARIA2_CONNECTIONS=\"${PUBCHEM_ARIA2_CONNECTIONS:-8}\"
export PUBCHEM_ARIA2_SPLIT=\"${PUBCHEM_ARIA2_SPLIT:-8}\"
export PUBCHEM_ARIA2_TIMEOUT=\"${PUBCHEM_ARIA2_TIMEOUT:-900}\"
for idx in \$(seq 0 \$(( $WORKERS - 1 ))); do
  worker=\$(printf \"%02d\" \"\$idx\")
  list=\"$OUT_ROOT/_worker_lists/worker_\${worker}.txt\"
  out=\"$OUT_ROOT/worker_\${worker}\"
  cache=\"\$out/pubchem3d_ftp_sdf\"
  log=\"logs/pubchem3d_parallel_worker_\${worker}_${SESSION}.log\"
  echo \"[pubchem-parallel] launch worker=\$worker list=\$list out=\$out log=\$log\"
  mkdir -p \"\$out\"
  \"$PY\" scripts/build_pubchem3d_structure_fot_dataset.py \
    --pubchem3d-ftp-bulk \
    --ftp-file-list \"\$list\" \
    --out-dir \"\$out\" \
    --prefix \"$PREFIX\" \
    --max-records \"$PER_WORKER_TARGET\" \
    --max-atoms \"$MAX_ATOMS\" \
    --shard-size \"$SHARD_SIZE\" \
    --min-free-gb \"$MIN_FREE_GB\" \
    --ftp-cache-dir \"\$cache\" \
    --remove-downloaded-cache \
    --resume \
    > \"\$log\" 2>&1 &
done
wait
echo \"[pubchem-parallel] all workers completed \$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
'"

echo "$SESSION" > logs/active_pubchem3d_parallel_curation_session.txt
echo "$SESSION"
