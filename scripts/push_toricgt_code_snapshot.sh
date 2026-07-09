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

if ! git push origin "$BRANCH"; then
  TOKEN="$(
    /usr/bin/python3 - "$ROOT/keys.txt" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
patterns = [
    r"(?:^|\n)\s*(?:GITHUB_TOKEN|GH_TOKEN)\s*=\s*([^\s\"']+)",
    r"(gh[pousr]_[A-Za-z0-9_]+)",
    r"(github_pat_[A-Za-z0-9_]+)",
]
for pattern in patterns:
    match = re.search(pattern, text)
    if match:
        print(match.group(1).strip())
        break
PY
  )"
  if [[ -z "$TOKEN" ]]; then
    echo "Normal git push failed and no GitHub token was found in keys.txt." >&2
    exit 1
  fi
  BASIC="$(printf 'x-access-token:%s' "$TOKEN" | base64 -w0)"
  git -c http.extraHeader="AUTHORIZATION: Basic $BASIC" push origin "$BRANCH"
fi
echo "code_snapshot_pushed branch:$BRANCH"
