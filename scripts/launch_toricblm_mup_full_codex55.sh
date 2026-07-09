#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/toricblm_mup_170m_codex55.env}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  echo "Run: conda run -n tokengt env PYTHONPATH=src:external/mup:amelie-iska/parameter-golf python scripts/generate_toricblm_mup_config.py" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${RUN_ID:-toricblm-mup-codex55-full-${RUN_STAMP}}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/oai_sidecar/${RUN_ID}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints/${RUN_ID}}"
mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR"

export RUN_ID
export CHECKPOINT_DIR
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_ID}"
export WANDB_NAME="${WANDB_NAME:-$RUN_ID}"
export PYTHONPATH="$ROOT/src:$ROOT/external/mup:$ROOT/amelie-iska/parameter-golf:${PYTHONPATH:-}"
TOKENGT_PY="${TORICBLM_TOKENGT_PYTHON:-/home/iska/miniconda3/envs/tokengt/bin/python}"
if [[ ! -x "$TOKENGT_PY" ]]; then
  TOKENGT_PY="python"
fi

echo "launching ToricBLM muP full run"
echo "root: $ROOT"
echo "config: $CONFIG_PATH"
echo "run_id: $RUN_ID"
echo "run_dir: $RUN_DIR"
echo "checkpoint_dir: $CHECKPOINT_DIR"
echo "data_path: ${DATA_PATH:-unset}"
echo "late_graph_train_glob: ${LATE_GRAPH_TRAIN_GLOB:-unset}"
echo "mup_base_shapes: ${MUP_BASE_SHAPES:-unset}"

if [[ "${TORICBLM_REQUIRE_SPLIT_SAFE_BIO:-0}" == "1" ]]; then
  if [[ "${GRAPH_TRAIN_GLOB:-}" == *"/raw_hf_bio_scale/"* || "${LONG_ENTRY_TRAIN_GLOB:-}" == *"/raw_hf_bio_scale/"* ]]; then
    echo "TORICBLM_REQUIRE_SPLIT_SAFE_BIO=1 but a raw_hf_bio_scale path is still in GRAPH_TRAIN_GLOB or LONG_ENTRY_TRAIN_GLOB" >&2
    exit 1
  fi
  SPLIT_MANIFEST="${TORICBLM_NONPROTEIN_SPLIT_MANIFEST:-}"
  MIN_NONPROTEIN_ROWS="${TORICBLM_NONPROTEIN_MIN_TRAIN_ROWS_PER_MODALITY:-1}"
  if [[ -z "$SPLIT_MANIFEST" || ! -s "$SPLIT_MANIFEST" ]]; then
    echo "TORICBLM_REQUIRE_SPLIT_SAFE_BIO=1 but non-protein split manifest is missing: ${SPLIT_MANIFEST:-unset}" >&2
    exit 1
  fi
  "$TOKENGT_PY" - "$SPLIT_MANIFEST" "$MIN_NONPROTEIN_ROWS" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
minimum = int(sys.argv[2])
counts = manifest.get("writer_counts", {})
missing = []
for modality in ("dna", "rna", "small_molecule"):
    train_count = int(counts.get(modality, {}).get("train", 0))
    if train_count < minimum:
        missing.append(f"{modality}:train={train_count}<min={minimum}")
if missing:
    raise SystemExit("non-protein split manifest below required train counts: " + ", ".join(missing))
print("split_safe_bio_preflight:passed", json.dumps({m: counts.get(m, {}).get("train", 0) for m in ("dna", "rna", "small_molecule")}, sort_keys=True))
PY
fi

if [[ "${TORICBLM_REQUIRE_FOT_FORMAT:-0}" == "1" ]]; then
  FOT_COLUMNS="${TORICBLM_FOT_REQUIRED_COLUMNS:-graph_json,forest_json,thought_forest_json,convextok_dag_json,training_views_json}"
  FOT_SAMPLE_LIMIT="${TORICBLM_FOT_PREFLIGHT_SAMPLE_FILES:-32}"
  "$TOKENGT_PY" - "$GRAPH_TRAIN_GLOB" "$FOT_COLUMNS" "$FOT_SAMPLE_LIMIT" <<'PY'
import glob
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

patterns = [item for item in sys.argv[1].split(",") if item]
required = [item for item in sys.argv[2].split(",") if item]
limit = int(sys.argv[3])
files = []
for pattern in patterns:
    files.extend(sorted(glob.glob(pattern)))
files = [Path(path) for path in files]
if not files:
    raise SystemExit("TORICBLM_REQUIRE_FOT_FORMAT=1 but GRAPH_TRAIN_GLOB matched no Parquet files")
checked = 0
errors = []
for path in files:
    if checked >= limit:
        break
    try:
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        missing = [column for column in required if column not in names]
        if missing:
            errors.append({"path": str(path), "missing_columns": missing})
            continue
        row = pq.read_table(path, columns=required).slice(0, 1).to_pylist()
        if not row:
            errors.append({"path": str(path), "error": "empty_file"})
            continue
        empty = [column for column in required if row[0].get(column) in (None, "")]
        if empty:
            errors.append({"path": str(path), "empty_columns": empty})
            continue
        graph = json.loads(row[0]["graph_json"])
        thought = json.loads(row[0]["thought_forest_json"])
        convextok = json.loads(row[0]["convextok_dag_json"])
        if not graph.get("nodes") or not graph.get("edges"):
            errors.append({"path": str(path), "error": "graph_json_missing_nodes_or_edges"})
        if thought.get("schema") != "toricgt.biomed_source_grounded_forest_of_thought.v1":
            errors.append({"path": str(path), "error": "invalid_thought_forest_schema"})
        if convextok.get("schema") != "toricgt.convextok_tokenization_dag.v1":
            errors.append({"path": str(path), "error": "invalid_convextok_dag_schema"})
        checked += 1
    except Exception as exc:  # noqa: BLE001
        errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
if errors:
    raise SystemExit("FoT format preflight failed: " + json.dumps(errors[:16], sort_keys=True))
print("fot_format_preflight:passed", json.dumps({"checked_files": checked, "matched_files": len(files), "required_columns": required}, sort_keys=True))
PY
fi

if [[ "${TORICBLM_PREFLIGHT_READY_CHECK:-0}" == "1" ]]; then
  if [[ -z "${TORICBLM_STRUCTURE_TRAIN_GLOB:-}" ]]; then
    echo "TORICBLM_PREFLIGHT_READY_CHECK=1 but TORICBLM_STRUCTURE_TRAIN_GLOB is unset" >&2
    exit 1
  fi
  READY_MANIFEST="${TORICBLM_STRUCTURE_READINESS_MANIFEST:-$ROOT/data/uniprot_fot/manifests/toricblm_multimodal_structure_readiness.json}"
  PREFLIGHT_PY="${TORICBLM_PREFLIGHT_PYTHON:-/home/iska/miniconda3/envs/iska-net-2/bin/python}"
  if [[ ! -x "$PREFLIGHT_PY" ]]; then
    PREFLIGHT_PY="python"
  fi
  IFS=',' read -r -a STRUCTURE_INPUTS <<< "$TORICBLM_STRUCTURE_TRAIN_GLOB"
  PREFLIGHT_CMD=("$PREFLIGHT_PY" "$ROOT/scripts/build_toricblm_modality_readiness_manifest.py" "--output-json" "$READY_MANIFEST")
  for pattern in "${STRUCTURE_INPUTS[@]}"; do
    [[ -n "$pattern" ]] && PREFLIGHT_CMD+=("--input" "$pattern")
  done
  IFS=',' read -r -a REQUIRED_MODALITIES <<< "${TORICBLM_REQUIRED_STRUCTURE_MODALITIES:-protein_afdb}"
  for modality in "${REQUIRED_MODALITIES[@]}"; do
    [[ -n "$modality" ]] && PREFLIGHT_CMD+=("--require" "$modality")
  done
  echo "running ToricBLM structure readiness preflight"
  printf 'preflight_command:'
  printf ' %q' "${PREFLIGHT_CMD[@]}"
  printf '\n'
  "${PREFLIGHT_CMD[@]}"
fi

exec conda run --no-capture-output -n tokengt env PYTHONPATH="$PYTHONPATH" \
  python "$ROOT/amelie-iska/parameter-golf/train_gpt.py" 2>&1 | tee "$RUN_DIR/train.log"
