#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=${TORICMORA_PYTHON:-"$HOME/miniconda3/envs/tokengt/bin/python"}
if [[ ! -x "$python" ]]; then
    printf 'Existing tokengt Python not found at %s\n' "$python" >&2
    exit 2
fi

cd -- "$repo"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -B scripts/run_toricmora_reviewed.py "$@"
