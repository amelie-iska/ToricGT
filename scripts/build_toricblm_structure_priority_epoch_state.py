#!/usr/bin/env python3
"""Build exact file-list manifests for the ToricBLM structure-priority curriculum.

The launcher uses this script at each epoch boundary.  It snapshots the files
currently available on disk, computes row counts from Parquet metadata, samples
actual graphified ConvexTok token lengths, and emits epoch-specific file lists
plus safe batch/context settings.

Epoch modes:
  * structure_current: all currently available coordinate-bearing files.
  * structure_delta: coordinate-bearing files not present in a previous snapshot.
  * all_entries: every graphifiable train file, with structures still routed to
    the structure-flow stream.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "amelie-iska" / "parameter-golf"))

from convextok import ConvexTokTokenizer  # type: ignore  # noqa: E402
from toricgt.oai_sidecar import GraphParquetTokenStream  # noqa: E402


COORD_COLUMNS = {
    "structure_coordinates",
    "ca_coordinates",
    "backbone_coordinates",
    "coordinate_mask",
}

STRUCTURE_PATTERNS = [
    "data/uniprot_fot/structures/afdb_v6/*.parquet",
    "data/uniprot_fot/structures/afdb_v6/train/*.parquet",
    "data/uniprot_fot/structures/afdb_v6_full/train/*.parquet",
    "data/uniprot_fot/structures/afdb_uniref50_full/train/*.parquet",
    "data/uniprot_fot/structures/afdb_parallel_uniref50_full/worker_*/train/*.parquet",
    "data/uniprot_fot/structures/afdb_ebi_tar_v6/train/*.parquet",
    "data/uniprot_fot/structures/afdb_ebi_tar_v6/worker_*/train/*.parquet",
    "data/uniprot_fot/structures/afdb_gcs_v4/train/*.parquet",
    "data/uniprot_fot/structures/afdb_gcs_v4/worker_*/train/*.parquet",
    "data/uniprot_fot/structures/afdb_gcs_proteome_tar_v4/train/*.parquet",
    "data/uniprot_fot/structures/afdb_gcs_proteome_tar_v4/worker_*/train/*.parquet",
    "data/uniprot_fot/structures/afdb_gcs_proteome_tar_v4_large/train/*.parquet",
    "data/uniprot_fot/structures/afdb_gcs_proteome_tar_v4_large/worker_*/train/*.parquet",
    "data/uniprot_fot/structures/pdb_modal/train/*.parquet",
    "data/uniprot_fot/structures/pdb_modal_parallel/worker_*/train/*.parquet",
    "data/uniprot_fot/structures/pubchem3d/train/*.parquet",
    "data/uniprot_fot/structures/pubchem3d_parallel/worker_*/train/*.parquet",
]

ALL_ENTRY_PATTERNS = [
    "data/uniprot_fot/splits/protrek_v2/*.parquet",
    "data/uniprot_fot/splits/protrek_v2/train/*.parquet",
    "data/toricblm_nonprotein_fot_splits/v2_parallel/dna/train/*.parquet",
    "data/toricblm_nonprotein_fot_splits/v2_parallel/rna/train/*.parquet",
    "data/toricblm_nonprotein_fot_splits/v2_parallel/small_molecule/train/*.parquet",
    "data/toricblm_late_mixed_fot_structure/train/*.parquet",
    "/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale/*/default/train/*.parquet",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def expand_one(pattern: str, seen_lists: set[str] | None = None) -> list[Path]:
    seen_lists = seen_lists if seen_lists is not None else set()
    pattern = pattern.strip()
    if not pattern:
        return []
    if pattern.startswith("@"):
        list_path = Path(pattern[1:]).expanduser()
        key = str(list_path.resolve()) if list_path.exists() else str(list_path)
        if key in seen_lists:
            return []
        seen_lists.add(key)
        try:
            lines = list_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        out: list[Path] = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                out.extend(expand_one(line, seen_lists))
        return out
    matches = sorted(glob.glob(pattern))
    if not matches and Path(pattern).exists():
        matches = [pattern]
    return [Path(item).resolve() for item in matches]


def expand_many(patterns: list[str]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    seen_lists: set[str] = set()
    for raw in patterns:
        for piece in re.split(r"[,;]", raw):
            for path in expand_one(piece, seen_lists):
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    out.append(path)
    return out


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def modality_hint(path: Path, columns: set[str]) -> str:
    text = str(path).lower()
    if "pubchem" in text or "selfies" in columns:
        return "small_molecule_3d"
    if "pdb" in text:
        return "pdb_complex_or_structure"
    if "afdb" in text:
        return "protein_afdb"
    if "/rna/" in text or "rnacentral" in text or "rfam" in text:
        return "rna_sequence_or_structure"
    if "/dna/" in text or "dna_" in text:
        return "dna_sequence_or_structure"
    if "uniprot" in text or "uniref" in text:
        return "protein_sequence_function"
    return "other_graphified_bio"


def file_info(path: Path) -> dict[str, Any] | None:
    try:
        pf = pq.ParquetFile(path)
        rows = int(pf.metadata.num_rows)
        columns = set(pf.schema_arrow.names)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": str(path),
            "relative_path": rel_or_abs(path),
            "readable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "rows": 0,
            "has_coordinates": False,
            "graphifiable": False,
            "modality": "unreadable",
        }
    graphifiable = bool(columns & set(GraphParquetTokenStream.TEXT_COLUMNS))
    has_coordinates = bool(columns & COORD_COLUMNS)
    return {
        "path": str(path),
        "relative_path": rel_or_abs(path),
        "readable": True,
        "rows": rows,
        "columns": sorted(columns),
        "has_coordinates": has_coordinates,
        "graphifiable": graphifiable,
        "modality": modality_hint(path, columns),
        "bytes": int(path.stat().st_size) if path.exists() else 0,
    }


def prior_structure_paths(manifest_path: Path | None) -> set[str]:
    if manifest_path is None or not manifest_path.exists():
        return set()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths: set[str] = set()
    for item in payload.get("selected_structure_files", []):
        if isinstance(item, dict):
            paths.add(str(Path(item["path"]).resolve()))
        else:
            paths.add(str(Path(item).resolve()))
    return paths


def parse_row_caps(values: list[str]) -> dict[str, int]:
    caps: dict[str, int] = {}
    for raw in values:
        for piece in re.split(r"[,;]", raw):
            piece = piece.strip()
            if not piece:
                continue
            if "=" not in piece:
                raise ValueError(f"row cap must be modality=rows, got {piece!r}")
            key, value = piece.split("=", 1)
            key = key.strip()
            try:
                cap = int(value.replace("_", "").strip())
            except ValueError as exc:
                raise ValueError(f"invalid row cap for {key!r}: {value!r}") from exc
            if cap < 0:
                raise ValueError(f"row cap must be nonnegative for {key!r}: {cap}")
            caps[key] = cap
    return caps


def apply_row_budgets(
    files: list[dict[str, Any]],
    *,
    modality_row_caps: dict[str, int],
    max_total_rows: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select whole Parquet shards under modality and total row budgets.

    The selection is deterministic and keeps non-capped modalities first so
    explicit caps, e.g. small_molecule_3d=250000, reduce overrepresented
    modalities without accidentally dropping protein/PDB structure shards.
    """

    if not modality_row_caps and (max_total_rows is None or max_total_rows <= 0):
        return files, {
            "enabled": False,
            "modality_row_caps": {},
            "max_total_rows": max_total_rows,
            "dropped_file_count": 0,
            "dropped_rows_by_modality": {},
        }

    selected: list[dict[str, Any]] = []
    rows_by_modality: Counter[str] = Counter()
    dropped_rows_by_modality: Counter[str] = Counter()
    dropped_files_by_modality: Counter[str] = Counter()
    total_rows = 0
    max_total = int(max_total_rows) if max_total_rows and max_total_rows > 0 else None

    def priority(info: dict[str, Any]) -> tuple[int, str, str]:
        modality = str(info.get("modality", "unknown"))
        capped = modality in modality_row_caps
        # Keep all uncapped modalities first; capped modalities fill after them.
        return (1 if capped else 0, modality, str(info.get("path", "")))

    for info in sorted(files, key=priority):
        modality = str(info.get("modality", "unknown"))
        rows = int(info.get("rows", 0))
        cap = modality_row_caps.get(modality)
        if cap is not None and rows_by_modality[modality] + rows > cap:
            dropped_rows_by_modality[modality] += rows
            dropped_files_by_modality[modality] += 1
            continue
        if max_total is not None and total_rows + rows > max_total:
            dropped_rows_by_modality[modality] += rows
            dropped_files_by_modality[modality] += 1
            continue
        selected.append(info)
        rows_by_modality[modality] += rows
        total_rows += rows

    return selected, {
        "enabled": True,
        "modality_row_caps": dict(sorted(modality_row_caps.items())),
        "max_total_rows": max_total,
        "selected_rows_by_modality": dict(sorted(rows_by_modality.items())),
        "dropped_rows_by_modality": dict(sorted(dropped_rows_by_modality.items())),
        "dropped_files_by_modality": dict(sorted(dropped_files_by_modality.items())),
        "dropped_file_count": int(sum(dropped_files_by_modality.values())),
    }


def write_filelist(path: Path, files: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(str(item["path"]) for item in files if item.get("readable", True)) + ("\n" if files else ""),
        encoding="utf-8",
    )


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))
    return float(ordered[idx])


def sample_token_stats(
    files: list[dict[str, Any]],
    tokenizer: Any,
    *,
    sample_files: int,
    rows_per_file: int,
) -> dict[str, Any]:
    lengths: list[int] = []
    policies: Counter[str] = Counter()
    sampled_rows = 0
    sampled_files = 0
    for info in files[: max(0, sample_files)]:
        path = Path(info["path"])
        try:
            pf = pq.ParquetFile(path)
            names = set(pf.schema_arrow.names)
            columns = [name for name in GraphParquetTokenStream.TEXT_COLUMNS if name in names]
            if not columns:
                continue
            batch_iter = pf.iter_batches(batch_size=max(1, rows_per_file), columns=columns)
            batch = next(batch_iter, None)
            if batch is None:
                continue
            table = pa.Table.from_batches([batch])
        except Exception:
            continue
        sampled_files += 1
        for row_index, row in enumerate(table.to_pylist()[:rows_per_file]):
            text, policy = GraphParquetTokenStream._graphify_row(row, f"{path.name}:{row_index}")
            if not text.strip():
                continue
            try:
                ids = tokenizer.encode(text, out_type=int)
            except TypeError:
                ids = tokenizer.encode(text)
            lengths.append(int(len(ids)) + 2)
            policies[policy] += 1
            sampled_rows += 1
    return {
        "sampled_files": sampled_files,
        "sampled_rows": sampled_rows,
        "token_length_min": min(lengths) if lengths else 0,
        "token_length_mean": float(mean(lengths)) if lengths else 0.0,
        "token_length_p50": percentile(lengths, 0.50),
        "token_length_p90": percentile(lengths, 0.90),
        "token_length_p95": percentile(lengths, 0.95),
        "token_length_p99": percentile(lengths, 0.99),
        "token_length_max": max(lengths) if lengths else 0,
        "graphification_policies": dict(sorted(policies.items())),
    }


def round_up(value: float, multiple: int) -> int:
    return int(math.ceil(max(value, 1.0) / multiple) * multiple)


def choose_batch_settings(stats: dict[str, Any], *, target_vram_gb: float) -> dict[str, Any]:
    p95 = float(stats.get("token_length_p95") or 4096.0)
    p99 = float(stats.get("token_length_p99") or p95)
    max_len = float(stats.get("token_length_max") or p99)

    # Full rows are segmented, so the memory driver is segment/full-context
    # length, not complete-row length.  Max tokens controls row truncation before
    # segmentation; keep it large enough for long annotations while avoiding
    # accidental million-token rows.
    long_entry_max_tokens = min(262144, max(24576, round_up(max(p99 * 1.25, p95 * 1.5), 1024)))
    if max_len > long_entry_max_tokens:
        long_entry_max_tokens = min(262144, max(long_entry_max_tokens, round_up(min(max_len, 262144), 1024)))

    if p95 <= 4096:
        long_entry_batch_size = 4
        graph_lm_batch_size = 8
        structure_batch_size = 24
    elif p95 <= 32768:
        long_entry_batch_size = 2
        graph_lm_batch_size = 6
        structure_batch_size = 16
    elif p95 <= 49152:
        long_entry_batch_size = 1
        graph_lm_batch_size = 4
        structure_batch_size = 12
    else:
        long_entry_batch_size = 1
        graph_lm_batch_size = 2
        structure_batch_size = 8

    if target_vram_gb < 20:
        graph_lm_batch_size = max(2, graph_lm_batch_size // 2)
        structure_batch_size = max(4, structure_batch_size // 2)

    full_context_seq_len = min(8192, max(4096, round_up(min(p95, 8192), 1024)))
    train_batch_tokens = 786432 if target_vram_gb >= 20 else 589824
    return {
        "train_batch_tokens": int(train_batch_tokens),
        "grad_accum_steps": 8,
        "graph_lm_batch_size": int(graph_lm_batch_size),
        "graph_lm_every": 1,
        "long_entry_batch_size": int(long_entry_batch_size),
        "long_entry_every": 1,
        "long_entry_max_tokens": int(long_entry_max_tokens),
        "long_entry_max_segments": 0,
        "long_entry_segment_checkpoint": 1,
        "long_entry_full_context_every": 64,
        "long_entry_full_context_seq_len": int(full_context_seq_len),
        "long_entry_full_context_min_tokens": 2048,
        "toricblm_structure_batch_size": int(structure_batch_size),
        "toricblm_structure_every": 1,
        "sidecar_batch_size": max(4, min(8, int(graph_lm_batch_size))),
        "sidecar_every": 2,
        "token_stats_basis": {
            "p95": p95,
            "p99": p99,
            "max": max_len,
            "target_vram_gb": float(target_vram_gb),
        },
    }


def coverage_steps(
    *,
    mode: str,
    selected_rows: int,
    structure_rows: int,
    settings: dict[str, Any],
    min_steps: int,
    max_steps: int | None,
) -> int:
    long_bs = max(1, int(settings["long_entry_batch_size"]))
    structure_bs = max(1, int(settings["toricblm_structure_batch_size"]))
    steps = 0
    if selected_rows > 0:
        steps = max(steps, math.ceil(selected_rows / long_bs))
    if structure_rows > 0 and mode != "all_entries":
        steps = max(steps, math.ceil(structure_rows / structure_bs))
    elif structure_rows > 0 and mode == "all_entries":
        # Epoch 3 trains every entry as full rows and still continues structure
        # flow over the currently available structure subset.
        steps = max(steps, math.ceil(structure_rows / structure_bs))
    computed = max(int(min_steps), int(steps))
    if max_steps is not None and max_steps > 0:
        computed = min(computed, int(max_steps))
    return computed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["structure_current", "structure_delta", "all_entries"])
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--previous-structure-manifest", type=Path)
    parser.add_argument("--sample-files", type=int, default=48)
    parser.add_argument("--rows-per-sample-file", type=int, default=8)
    parser.add_argument("--target-vram-gb", type=float, default=20.0)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--extra-structure-pattern", action="append", default=[])
    parser.add_argument("--extra-all-pattern", action="append", default=[])
    parser.add_argument(
        "--modality-row-cap",
        action="append",
        default=[],
        help="Whole-shard cap in the form modality=rows, e.g. small_molecule_3d=250000.",
    )
    parser.add_argument(
        "--max-selected-rows",
        type=int,
        default=0,
        help="Whole-shard total row cap after modality priority selection; 0 disables.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Hard cap on optimizer steps for this epoch state; 0 disables.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ConvexTokTokenizer.load_json(args.tokenizer)

    structure_patterns = [str(ROOT / pattern) if not pattern.startswith("/") else pattern for pattern in STRUCTURE_PATTERNS]
    all_patterns = [str(ROOT / pattern) if not pattern.startswith("/") else pattern for pattern in ALL_ENTRY_PATTERNS]
    structure_patterns.extend(args.extra_structure_pattern)
    all_patterns.extend(args.extra_all_pattern)
    all_patterns.extend(structure_patterns)

    structure_infos = [
        info for path in expand_many(structure_patterns) if (info := file_info(path)) and info.get("readable")
    ]
    structure_infos = [info for info in structure_infos if info.get("has_coordinates") and info.get("rows", 0) > 0]
    prior_paths = prior_structure_paths(args.previous_structure_manifest)
    current_structure_by_path = {str(Path(info["path"]).resolve()): info for info in structure_infos}
    if args.mode == "structure_delta":
        selected_structure = [
            info for key, info in sorted(current_structure_by_path.items()) if key not in prior_paths
        ]
        selected_files = selected_structure
    elif args.mode == "structure_current":
        selected_structure = [info for _, info in sorted(current_structure_by_path.items())]
        selected_files = selected_structure
    else:
        selected_structure = [info for _, info in sorted(current_structure_by_path.items())]
        all_infos = [info for path in expand_many(all_patterns) if (info := file_info(path)) and info.get("readable")]
        selected_by_path = {
            str(Path(info["path"]).resolve()): info
            for info in all_infos
            if info.get("graphifiable") and info.get("rows", 0) > 0
        }
        selected_files = [info for _, info in sorted(selected_by_path.items())]

    modality_row_caps = parse_row_caps(args.modality_row_cap)
    max_selected_rows = int(args.max_selected_rows) if int(args.max_selected_rows) > 0 else None
    selected_files, graph_budget_report = apply_row_budgets(
        selected_files,
        modality_row_caps=modality_row_caps,
        max_total_rows=max_selected_rows,
    )
    selected_file_paths = {str(Path(info["path"]).resolve()) for info in selected_files}
    selected_structure = [
        info for info in selected_structure if str(Path(info["path"]).resolve()) in selected_file_paths
    ]
    selected_structure, structure_budget_report = apply_row_budgets(
        selected_structure,
        modality_row_caps=modality_row_caps,
        max_total_rows=max_selected_rows,
    )

    selected_rows = int(sum(int(info.get("rows", 0)) for info in selected_files))
    structure_rows = int(sum(int(info.get("rows", 0)) for info in selected_structure))
    selected_modalities = Counter(str(info.get("modality", "unknown")) for info in selected_files)
    selected_rows_by_modality: Counter[str] = Counter()
    for info in selected_files:
        selected_rows_by_modality[str(info.get("modality", "unknown"))] += int(info.get("rows", 0))

    token_stats = sample_token_stats(
        selected_files,
        tokenizer,
        sample_files=max(1, int(args.sample_files)),
        rows_per_file=max(1, int(args.rows_per_sample_file)),
    )
    settings = choose_batch_settings(token_stats, target_vram_gb=float(args.target_vram_gb))
    steps = coverage_steps(
        mode=args.mode,
        selected_rows=selected_rows,
        structure_rows=structure_rows,
        settings=settings,
        min_steps=int(args.min_steps) if selected_rows > 0 else 0,
        max_steps=int(args.max_steps) if int(args.max_steps) > 0 else None,
    )
    uncapped_steps = coverage_steps(
        mode=args.mode,
        selected_rows=selected_rows,
        structure_rows=structure_rows,
        settings=settings,
        min_steps=int(args.min_steps) if selected_rows > 0 else 0,
        max_steps=None,
    )

    selected_filelist = args.output_dir / "selected_graph_and_long_entry_files.txt"
    structure_filelist = args.output_dir / "selected_structure_files.txt"
    write_filelist(selected_filelist, selected_files)
    write_filelist(structure_filelist, selected_structure)

    env_path = args.output_dir / "epoch_data_overrides.env"
    env_path.write_text(
        "\n".join(
            [
                f"export GRAPH_TRAIN_GLOB=@{selected_filelist}",
                f"export LONG_ENTRY_TRAIN_GLOB=@{selected_filelist}",
                f"export TORICBLM_STRUCTURE_TRAIN_GLOB=@{structure_filelist}",
                f"export PROTEIN_STRUCTURE_PROTREK_TRAIN_GLOB=@{structure_filelist}",
                "export LATE_GRAPH_TRAIN_GLOB=",
                "export GRAPH_LM_PRIMARY=1",
                "export LONG_ENTRY_TRAINING=1",
                "export TORICBLM_STRUCTURE_FLOW=1",
                f"export TRAIN_BATCH_TOKENS={settings['train_batch_tokens']}",
                f"export GRAD_ACCUM_STEPS={settings['grad_accum_steps']}",
                f"export GRAPH_LM_BATCH_SIZE={settings['graph_lm_batch_size']}",
                f"export GRAPH_LM_EVERY={settings['graph_lm_every']}",
                f"export LONG_ENTRY_BATCH_SIZE={settings['long_entry_batch_size']}",
                f"export LONG_ENTRY_EVERY={settings['long_entry_every']}",
                f"export LONG_ENTRY_MAX_TOKENS={settings['long_entry_max_tokens']}",
                f"export LONG_ENTRY_MAX_SEGMENTS={settings['long_entry_max_segments']}",
                f"export LONG_ENTRY_SEGMENT_CHECKPOINT={settings['long_entry_segment_checkpoint']}",
                f"export LONG_ENTRY_FULL_CONTEXT_EVERY={settings['long_entry_full_context_every']}",
                f"export LONG_ENTRY_FULL_CONTEXT_SEQ_LEN={settings['long_entry_full_context_seq_len']}",
                f"export LONG_ENTRY_FULL_CONTEXT_MIN_TOKENS={settings['long_entry_full_context_min_tokens']}",
                f"export TORICBLM_STRUCTURE_BATCH_SIZE={settings['toricblm_structure_batch_size']}",
                f"export TORICBLM_STRUCTURE_EVERY={settings['toricblm_structure_every']}",
                f"export TORICGT_SIDECAR_BATCH_SIZE={settings['sidecar_batch_size']}",
                f"export TORICGT_SIDECAR_EVERY={settings['sidecar_every']}",
                "export TORICBLM_REQUIRE_FOT_FORMAT=0",
                "export TORICBLM_REQUIRE_SEQUENCE_FUNCTION_TRAINABLE=0",
                "export TORICBLM_ALLOW_TRAIN_ONLY_RAW_SEQUENCE=1",
                "export TORICBLM_ALLOW_STRUCTURE_TARGET_SHORTFALL=1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema": "toricblm.structure_priority_curriculum_epoch_state.v1",
        "created_utc": utc_now(),
        "epoch": int(args.epoch),
        "mode": args.mode,
        "selected_rows": selected_rows,
        "selected_file_count": len(selected_files),
        "structure_rows_available_for_structure_flow": structure_rows,
        "structure_file_count_available_for_structure_flow": len(selected_structure),
        "previous_structure_manifest": str(args.previous_structure_manifest) if args.previous_structure_manifest else None,
        "coverage_steps": steps,
        "uncapped_coverage_steps": uncapped_steps,
        "coverage_policy": (
            "steps are ceil(row_count / batch_size) for full-row long-entry training, "
            "with structure-flow rows also covered for coordinate-bearing entries, "
            "then optionally capped by --max-steps for wall-clock bounded curricula"
        ),
        "selected_rows_by_modality": dict(sorted(selected_rows_by_modality.items())),
        "selected_file_count_by_modality": dict(sorted(selected_modalities.items())),
        "token_stats": token_stats,
        "batch_settings": settings,
        "row_budget": {
            "graph_and_long_entry": graph_budget_report,
            "structure_flow": structure_budget_report,
        },
        "selected_filelist": str(selected_filelist),
        "structure_filelist": str(structure_filelist),
        "env_overrides": str(env_path),
        "selected_structure_files": selected_structure,
        "selected_files_preview": selected_files[:64],
        "structure_patterns": structure_patterns,
        "all_entry_patterns": all_patterns,
        "multimodal_pairing_contract": (
            "Coordinate-bearing Parquet rows remain in GRAPH_TRAIN_GLOB and LONG_ENTRY_TRAIN_GLOB "
            "for structure-priority epochs, so coordinates are trained together with graph_json, "
            "thought_forest_json, convextok_dag_json, sequence/selfies fields, names, labels, "
            "GO/EC/function annotations, tags, and PDB/PubChem/UniProt metadata whenever those "
            "columns are present in the same row."
        ),
    }
    manifest_path = args.output_dir / "epoch_state_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "mode": args.mode,
                "epoch": args.epoch,
                "selected_rows": selected_rows,
                "structure_rows": structure_rows,
                "coverage_steps": steps,
                "batch_settings": settings,
                "token_stats": token_stats,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
