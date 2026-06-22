#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/toricblm_mup_170m_convextok8192_biomed_codex55.env}"
exec "$ROOT/scripts/launch_toricblm_mup_full_codex55.sh"
