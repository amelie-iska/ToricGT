#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/iska/miniconda3/envs/iska-net-2/bin/python}"
OUT_ROOT="${AFDB_GCS_PROTEOME_TAR_OUT_ROOT:-data/uniprot_fot/structures/afdb_gcs_proteome_tar_v4}"
WORKERS="${AFDB_GCS_PROTEOME_TAR_WORKERS:-16}"
MAX_RECORDS="${AFDB_GCS_PROTEOME_TAR_MAX_RECORDS:-5000000}"
MAX_ARCHIVES="${AFDB_GCS_PROTEOME_TAR_MAX_ARCHIVES:-1200}"
MAX_RESIDUES="${AFDB_MAX_RESIDUES:-512}"
SHARD_SIZE="${AFDB_SHARD_SIZE:-2048}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
EMIT_3DI="${AFDB_EMIT_3DI:-1}"
PREFIX="${AFDB_GCS_PROTEOME_TAR_PREFIX:-toricblm_afdb_gcs_proteome_tar_structure_fot}"
BUCKET_PREFIX="${AFDB_GCS_PROTEOME_PREFIX:-gs://public-datasets-deepmind-alphafold-v4/proteomes}"
SESSION="${SESSION:-toricblm_afdb_gcs_proteome_tar_$(date -u +%Y%m%dT%H%M%SZ)}"
DOWNLOAD_TIMEOUT="${AFDB_GCS_PROTEOME_DOWNLOAD_TIMEOUT:-14400}"
LIST_LIMIT="${AFDB_GCS_PROTEOME_LIST_LIMIT:-120000}"
SELECTION_POLICY="${AFDB_GCS_PROTEOME_SELECTION_POLICY:-largest_then_diverse}"

mkdir -p "$OUT_ROOT/_worker_urls" logs

"$PY" - "$OUT_ROOT/_worker_urls" "$WORKERS" "$MAX_ARCHIVES" "$BUCKET_PREFIX" "$LIST_LIMIT" "$SELECTION_POLICY" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

out_dir = Path(sys.argv[1])
workers = int(sys.argv[2])
max_archives = int(sys.argv[3])
bucket_prefix = sys.argv[4].rstrip("/")
list_limit = int(sys.argv[5])
selection_policy = sys.argv[6]

match = re.match(r"^gs://([^/]+)/(.+)$", bucket_prefix)
if not match:
    raise SystemExit(f"Expected gs:// bucket prefix, got {bucket_prefix!r}")
bucket = match.group(1)
prefix = match.group(2).rstrip("/") + "/proteome-tax_id-"
page_token = ""
listed: list[tuple[int, str]] = []
while list_limit <= 0 or len(listed) < list_limit:
    query = (
        f"https://storage.googleapis.com/storage/v1/b/{quote(bucket)}/o"
        f"?prefix={quote(prefix, safe='')}&maxResults=1000"
    )
    if page_token:
        query += f"&pageToken={quote(page_token, safe='')}"
    with urlopen(query, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    for item in payload.get("items", []):
        name = item.get("name", "")
        if name.endswith("_v4.tar"):
            listed.append((int(item.get("size") or 0), f"gs://{bucket}/{name}"))
            if list_limit > 0 and len(listed) >= list_limit:
                break
    page_token = payload.get("nextPageToken") or ""
    if not page_token:
        break

by_tax: dict[str, list[tuple[int, str]]] = defaultdict(list)
for size, raw in listed:
    url = raw.strip()
    if not url:
        continue
    match = re.search(r"proteome-tax_id-([0-9]+)-([0-9]+)_v4\.tar$", url)
    if not match:
        continue
    by_tax[match.group(1)].append((size, url))

def key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

selected: list[str] = []
seen: set[str] = set()
if selection_policy == "largest_then_diverse":
    # High-yield shards keep curation bounded: AFDB proteome shards may contain
    # up to 10k proteins, while many tiny proteomes contain only one or two.
    # Use size as a proxy for record count, then add one extra per taxon by hash
    # if max_archives leaves room.
    for size, url in sorted((item for items in by_tax.values() for item in items), reverse=True):
        if url in seen:
            continue
        selected.append(url)
        seen.add(url)
        if max_archives > 0 and len(selected) >= max_archives:
            break
else:
    for tax_id in sorted(by_tax, key=key):
        items = sorted(by_tax[tax_id], key=lambda item: key(item[1]))
        if items and items[0][1] not in seen:
            selected.append(items[0][1])
            seen.add(items[0][1])
        if max_archives > 0 and len(selected) >= max_archives:
            break
    if max_archives <= 0 or len(selected) < max_archives:
        for size, url in sorted((item for items in by_tax.values() for item in items), key=lambda item: key(item[1])):
            if url in seen:
                continue
            selected.append(url)
            seen.add(url)
            if max_archives > 0 and len(selected) >= max_archives:
                break
if not selected:
    raise SystemExit("No AFDB GCS v4 proteome tar shards were found.")

bins = [[] for _ in range(workers)]
for idx, url in enumerate(selected):
    bins[idx % workers].append(url)

out_dir.mkdir(parents=True, exist_ok=True)
for idx, worker_urls in enumerate(bins):
    path = out_dir / f"worker_{idx:02d}.txt"
    path.write_text("\n".join(worker_urls) + ("\n" if worker_urls else ""), encoding="utf-8")
    print(f"[afdb-gcs-proteome] worker={idx:02d} archives={len(worker_urls)} list={path}")
(out_dir / "selected_archives.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
print(
    f"[afdb-gcs-proteome] listed_archives={len(listed)} selected_archives={len(selected)} "
    f"tax_ids={len(by_tax)} list_limit={list_limit} selection_policy={selection_policy}"
)
PY

per_worker=$(( (MAX_RECORDS + WORKERS - 1) / WORKERS ))
echo "[afdb-gcs-proteome] root=$ROOT out=$OUT_ROOT workers=$WORKERS quota_per_worker=$per_worker max_records=$MAX_RECORDS max_archives=$MAX_ARCHIVES list_limit=$LIST_LIMIT selection_policy=$SELECTION_POLICY"

tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash -lc '
set -euo pipefail
for i in \$(seq 0 \$(( $WORKERS - 1 ))); do
  wid=\$(printf \"%02d\" \"\$i\")
  url_list=\"$OUT_ROOT/_worker_urls/worker_\${wid}.txt\"
  worker_out=\"$OUT_ROOT/worker_\${wid}\"
  worker_log=\"logs/afdb_gcs_proteome_tar_worker_\${wid}_${SESSION}.log\"
  args=(
    scripts/build_afdb_ebi_tar_structure_fot_dataset.py
    --out-dir \"\$worker_out\"
    --tar-url-list \"\$url_list\"
    --max-records \"$per_worker\"
    --max-residues \"$MAX_RESIDUES\"
    --shard-size \"$SHARD_SIZE\"
    --min-free-gb \"$MIN_FREE_GB\"
    --download-timeout \"$DOWNLOAD_TIMEOUT\"
    --prefix \"$PREFIX\"
    --resume
  )
  if [[ \"$EMIT_3DI\" == \"1\" ]]; then
    args+=(--emit-foldseek-3di)
  fi
  echo \"[afdb-gcs-proteome] launch worker=\$wid log=\$worker_log\"
  \"$PY\" \"\${args[@]}\" >\"\$worker_log\" 2>&1 &
done
wait
echo \"[afdb-gcs-proteome] all workers completed \$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
'"

echo "$SESSION" > logs/active_afdb_gcs_proteome_tar_session.txt
echo "$SESSION"
