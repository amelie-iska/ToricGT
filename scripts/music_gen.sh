#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="${TORICGT_CONDA_ENV:-tokengt}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEFAULT_OUTPUT="${ROOT_DIR}/outputs/music/toricgt_torus_music_${STAMP}.wav"

conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
  env PYTHONPATH="${ROOT_DIR}/src" \
  python -m toricgt.music \
  --output "${DEFAULT_OUTPUT}" \
  --play \
  "$@"
