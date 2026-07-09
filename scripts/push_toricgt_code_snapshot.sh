#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BRANCH="$(git branch --show-current)"
if [[ -z "$BRANCH" ]]; then
  echo "Could not determine current git branch." >&2
  exit 1
fi

git add \
  scripts/launch_toricblm_dynamic_partial_epochs.sh \
  scripts/launch_toricblm_mup_full_codex55.sh \
  scripts/publish_toricblm_epoch_artifacts_to_hf.sh \
  scripts/push_toricgt_code_snapshot.sh

if ! git diff --cached --quiet; then
  git commit -m "Add dynamic ToricBLM partial epoch training"
fi

git push origin "$BRANCH"
echo "code_snapshot_pushed branch:$BRANCH"
