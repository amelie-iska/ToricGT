#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/iska/miniconda3/envs/iska-net-2/bin/python}"
OUT_ROOT="${AFDB_EBI_TAR_OUT_ROOT:-data/uniprot_fot/structures/afdb_ebi_tar_v6}"
WORKERS="${AFDB_EBI_TAR_WORKERS:-4}"
MAX_RECORDS="${AFDB_EBI_TAR_MAX_RECORDS:-5000000}"
MAX_RESIDUES="${AFDB_MAX_RESIDUES:-512}"
SHARD_SIZE="${AFDB_SHARD_SIZE:-2048}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
EMIT_3DI="${AFDB_EMIT_3DI:-1}"
MAX_ARCHIVES="${AFDB_EBI_TAR_MAX_ARCHIVES:-0}"
INDEX_URL="${AFDB_EBI_INDEX_URL:-https://ftp.ebi.ac.uk/pub/databases/alphafold/v6}"
PREFIX="${AFDB_EBI_TAR_PREFIX:-toricblm_afdb_ebi_tar_structure_fot}"
DOWNLOAD_TIMEOUT="${AFDB_EBI_DOWNLOAD_TIMEOUT:-86400}"

mkdir -p "$OUT_ROOT/_worker_urls" logs

"$PY" - "$OUT_ROOT/_worker_urls" "$WORKERS" "$MAX_ARCHIVES" "$INDEX_URL" <<'PY'
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

out_dir = Path(sys.argv[1])
workers = int(sys.argv[2])
max_archives = int(sys.argv[3])
index_url = sys.argv[4].rstrip("/")
unit = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
html = urllib.request.urlopen(index_url, timeout=60).read().decode("utf-8", errors="replace")
items = []
for row in html.splitlines():
    m = re.search(r'href="([^"]+\.tar)"', row)
    if not m:
        continue
    name = m.group(1)
    if name.endswith("_pdb_v6.tar"):
        continue
    sm = re.search(r'align="right">\s*([0-9.]+)([KMGT]?)\s*</td>', row)
    size_mib = 1.0
    if sm:
        size_mib = float(sm.group(1)) * unit.get(sm.group(2), 1 / (1024 * 1024))
    items.append((name, size_mib, f"{index_url}/{name}"))
items.sort(key=lambda x: x[1], reverse=True)
if max_archives > 0:
    items = items[:max_archives]
bins = [{"size": 0.0, "urls": []} for _ in range(workers)]
for name, size, url in items:
    target = min(bins, key=lambda item: item["size"])
    target["urls"].append(url)
    target["size"] += size
out_dir.mkdir(parents=True, exist_ok=True)
for idx, item in enumerate(bins):
    path = out_dir / f"worker_{idx:02d}.txt"
    path.write_text("\n".join(item["urls"]) + ("\n" if item["urls"] else ""), encoding="utf-8")
    print(f"[afdb-ebi] worker={idx:02d} archives={len(item['urls'])} approx_mib={item['size']:.1f} list={path}")
PY

per_worker=$(( (MAX_RECORDS + WORKERS - 1) / WORKERS ))
echo "[afdb-ebi] root=$ROOT out=$OUT_ROOT workers=$WORKERS quota_per_worker=$per_worker max_records=$MAX_RECORDS"

for ((i=0; i<WORKERS; i++)); do
  wid="$(printf '%02d' "$i")"
  url_list="$OUT_ROOT/_worker_urls/worker_${wid}.txt"
  worker_out="$OUT_ROOT/worker_${wid}"
  worker_log="logs/afdb_ebi_tar_worker_${wid}_$(date -u +%Y%m%dT%H%M%SZ).log"
  args=(
    scripts/build_afdb_ebi_tar_structure_fot_dataset.py
    --out-dir "$worker_out"
    --tar-url-list "$url_list"
    --max-records "$per_worker"
    --max-residues "$MAX_RESIDUES"
    --shard-size "$SHARD_SIZE"
    --min-free-gb "$MIN_FREE_GB"
    --download-timeout "$DOWNLOAD_TIMEOUT"
    --prefix "$PREFIX"
    --resume
  )
  if [[ "$EMIT_3DI" == "1" ]]; then
    args+=(--emit-foldseek-3di)
  fi
  echo "[afdb-ebi] launch worker=$wid log=$worker_log"
  "$PY" "${args[@]}" >"$worker_log" 2>&1 &
done

wait
echo "[afdb-ebi] all workers completed"
