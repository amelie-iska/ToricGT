#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPO_ID="${REPO_ID:-AmelieSchreiber/toricblm_fot}"
LOCAL_ROOT="${LOCAL_ROOT:-data}"
NUM_WORKERS="${NUM_WORKERS:-8}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload ToricBLM FoT multimodal curated dataset}"

if [[ -z "${HF_TOKEN:-}" && -f "$ROOT/keys.txt" ]]; then
  HF_TOKEN="$(grep -Eo 'hf_[A-Za-z0-9_\-]+' "$ROOT/keys.txt" | head -1 || true)"
  export HF_TOKEN
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set and no Hugging Face token was found in keys.txt" >&2
  exit 1
fi

hf repos create "$REPO_ID" --type dataset --exist-ok >/dev/null

# upload-large-folder is resumable and keeps its own local cache.  We upload
# only curated dataset material and manifests, not raw secrets, logs, caches, or
# transient mmCIF/SDF download caches.
hf upload-large-folder "$REPO_ID" "$LOCAL_ROOT" \
  --type dataset \
  --num-workers "$NUM_WORKERS" \
  --include "uniprot_fot/**" \
  --include "toricblm_nonprotein_fot_splits/**" \
  --include "toricblm_late_mixed_fot_structure/**" \
  --exclude "**/mmcif_cache/**" \
  --exclude "**/rcsb_tmp_mmcif/**" \
  --exclude "**/pubchem3d_tmp_sdf/**" \
  --exclude "**/*.tmp" \
  --exclude "**/.cache/**"

echo "uploaded dataset material to https://huggingface.co/datasets/${REPO_ID}"
