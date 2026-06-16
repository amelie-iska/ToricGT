#!/usr/bin/env python3
# Example:
# conda run --no-capture-output -n tokengt env PYTHONPATH=src \
#   python scripts/watch_training_analysis.py \
#   --checkpoint-dir checkpoints/parameter_golf_oai_dense \
#   --start-step 14750 --after-steps 1500 \
#   --run-path amelie-iska-math/toricgt-parameter-golf/oai-rescue-14750 \
#   --output-root outputs/post_resume_analysis/oai-rescue-14750 \
#   --min-mtime-unix "$(date +%s)"
#
# Optional Codex handoff after analysis completes:
#   conda run --no-capture-output -n tokengt env PYTHONPATH=src \
#     python scripts/watch_training_analysis.py \
#     --checkpoint-dir checkpoints/parameter_golf_oai_dense \
#     --start-step 1000 --after-steps 1500 \
#     --run-path amelie-iska-math/toricgt-parameter-golf/1ouz53jk \
#     --output-root outputs/post_resume_analysis/oai-restart-01000-phased \
#     --min-mtime-unix "$(cat outputs/post_resume_analysis/oai-restart-01000-phased/start_epoch.txt)" \
#     --target-step 2250 \
#     --pause-training-before-analysis \
#     --device cuda --precision bf16 \
#     --codex-review-hook scripts/codex_training_review_resume.sh \
#     --codex-review-tmux-prefix toricgt_codex_review
"""Wait for a future checkpoint and run non-interrupting metric analyses.

The watcher intentionally evaluates on CPU by default.  This keeps the analysis
from competing with the active training tmux for GPU memory.  It writes a
compact Markdown synopsis after W&B metrics, budget simplices, branch
trajectories, Ramachandran-style phase plots, and energy landscapes complete.
The geometry suite also writes directed nested-simplicial diagnostics for the
noncommutative topology induced by hidden-state relation arrows.
Reasoning simplex and geometry diagnostics are best-effort so checkpoint-family
format mismatches do not turn a sidecar review into a training blocker.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from toricgt.wandb_organization import organize_wandb_payload, update_wandb_summary


CHECKPOINT_PATTERN = re.compile(r"(?:random_order_step_|_step_)(\d+)\.pt$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="checkpoints/parameter_golf_oai_dense")
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--after-steps", type=int, default=1500)
    parser.add_argument(
        "--target-step",
        type=int,
        default=0,
        help="Analyze the first fresh checkpoint at or above this absolute step. Overrides start-step + after-steps.",
    )
    parser.add_argument(
        "--min-mtime-unix",
        type=float,
        default=0.0,
        help="Ignore checkpoints older than this Unix timestamp; useful when filenames already exist.",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--run-path", default="")
    parser.add_argument("--output-root", default="outputs/post_resume_analysis")
    parser.add_argument("--data-glob", default="data/curated_hf_shards/validation/*.parquet")
    parser.add_argument("--test-data-glob", default="data/curated_hf_shards/test/*.parquet")
    parser.add_argument("--config", default="config/train.parameter_golf_random_order_dense.yaml")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--simplex-samples", type=int, default=8)
    parser.add_argument("--geometry-records", type=int, default=4)
    parser.add_argument("--geometry-branches", type=int, default=6)
    parser.add_argument("--gudhi-persistence-audit", action="store_true")
    parser.add_argument("--gudhi-records", type=int, default=3)
    parser.add_argument("--gudhi-max-points", type=int, default=18)
    parser.add_argument("--gudhi-num-radii", type=int, default=5)
    parser.add_argument("--gudhi-num-levels", type=int, default=5)
    parser.add_argument("--gudhi-radius-quantile", type=float, default=0.62)
    parser.add_argument("--gudhi-macaulay2-timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--tokengt-bpb-batches",
        type=int,
        default=8,
        help="Validation batches for TokenGT graph-node SP1024 FineWeb BPB checkpoints.",
    )
    parser.add_argument("--tokengt-bpb-token-glob", default="")
    parser.add_argument("--tokengt-bpb-tokenizer-path", default="")
    parser.add_argument(
        "--skip-simplex-geometry",
        action="store_true",
        help="Skip random-order simplex/geometry scripts; useful for TokenGT checkpoints that only need exact derived-category analysis.",
    )
    parser.add_argument(
        "--derived-category-example-dir",
        default="",
        help="Directory containing trainer-emitted derived_category_examples JSON. Defaults to <checkpoint-dir>/derived_category_examples.",
    )
    parser.add_argument("--derived-category-max-files", type=int, default=8)
    parser.add_argument("--derived-category-max-objects", type=int, default=24)
    parser.add_argument("--skip-derived-category-analysis", action="store_true")
    parser.add_argument(
        "--memory-trace-example-dir",
        default="",
        help="Directory containing trainer-emitted memory_trace_examples JSON. Defaults to <checkpoint-dir>/memory_trace_examples.",
    )
    parser.add_argument("--memory-trace-max-files", type=int, default=8)
    parser.add_argument("--memory-trace-max-queries", type=int, default=96)
    parser.add_argument("--skip-memory-trace-analysis", action="store_true")
    parser.add_argument(
        "--cas-audit",
        action="store_true",
        help="Run a small exact CAS/closed-form certificate audit as part of periodic analysis.",
    )
    parser.add_argument(
        "--cas-audit-require-cas",
        action="store_true",
        help="Fail the optional CAS audit command when requested Sage/Macaulay2 checks are unavailable.",
    )
    parser.add_argument("--cas-audit-num-vertices", type=int, default=6)
    parser.add_argument("--cas-audit-sage-normal-fan", action="store_true")
    parser.add_argument("--cas-audit-macaulay2-smoke", action="store_true")
    parser.add_argument("--cas-audit-macaulay2-toric-ideal", action="store_true")
    parser.add_argument("--cas-audit-all-exact-cas", action="store_true")
    parser.add_argument(
        "--skip-toric-embedding-report",
        action="store_true",
        help="Skip exact tropical-to-toric embedding sidecar rendering and screenshots.",
    )
    parser.add_argument("--toric-embedding-sidecar-records", type=int, default=1)
    parser.add_argument("--toric-embedding-sidecar-max-points", type=int, default=10)
    parser.add_argument("--toric-embedding-exponent-dim", type=int, default=2)
    parser.add_argument("--toric-embedding-quantization-scale", type=int, default=8)
    parser.add_argument(
        "--toric-embedding-build-vector-bundle",
        action="store_true",
        help="Ask the toric report renderer to include the exact Macaulay2 vector-bundle certificate.",
    )
    parser.add_argument(
        "--skip-toric-embedding-screenshots",
        action="store_true",
        help="Render toric embedding HTML but skip Playwright screenshots.",
    )
    parser.add_argument(
        "--skip-branching-reasoning-report",
        action="store_true",
        help="Skip branch/merge reasoning trajectory simplex-tree HTML sidecar rendering.",
    )
    parser.add_argument("--branching-reasoning-max-nodes", type=int, default=160)
    parser.add_argument("--branching-reasoning-node-offset", type=int, default=0)
    parser.add_argument(
        "--skip-branching-reasoning-screenshots",
        action="store_true",
        help="Render branch/merge reasoning HTML but skip Playwright screenshots.",
    )
    parser.add_argument(
        "--assert-branching-reasoning-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When branch/merge screenshots are enabled, fail screenshot rendering if the report DOM contract is broken.",
    )
    parser.add_argument("--skip-test-time-scaling", action="store_true")
    parser.add_argument("--test-time-scaling-batches", type=int, default=8)
    parser.add_argument("--test-time-scaling-budgets", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--seed", type=int, default=10017)
    parser.add_argument("--target-bpb", type=float, default=float(os.environ.get("BPB_TARGET", "1.2")))
    parser.add_argument("--gate-step", type=int, default=int(os.environ.get("BPB_GATE_STEP", "4000")))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--skip-oai-competition-eval", action="store_true")
    parser.add_argument(
        "--codex-review-hook",
        default="",
        help="Optional shell script to call after analysis; receives analysis paths and can run codex resume.",
    )
    parser.add_argument(
        "--codex-review-session-id",
        default="",
        help="Optional Codex session/thread id passed to the review hook. Defaults to hook behavior.",
    )
    parser.add_argument(
        "--codex-review-tmux-prefix",
        default="",
        help="If set, the hook launches codex resume in <prefix>_<step> instead of inline.",
    )
    parser.add_argument("--training-tmux", default="toricgt_pg_oai")
    parser.add_argument(
        "--pause-training-before-analysis",
        action="store_true",
        help="Send Ctrl-C to --training-tmux after the target checkpoint appears and before analysis starts.",
    )
    parser.add_argument("--pause-wait-seconds", type=float, default=10.0)
    return parser.parse_args(argv)


def latest_checkpoint(checkpoint_dir: Path, min_step: int, min_mtime_unix: float = 0.0) -> tuple[int, Path] | None:
    best: tuple[int, Path] | None = None
    for path in checkpoint_dir.glob("*.pt"):
        if min_mtime_unix > 0:
            try:
                if path.stat().st_mtime < min_mtime_unix:
                    continue
            except OSError:
                continue
        match = CHECKPOINT_PATTERN.search(path.name)
        if not match:
            continue
        step = int(match.group(1))
        if step < min_step:
            continue
        if best is None or step < best[0]:
            best = (step, path)
    return best


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src:{env.get('PYTHONPATH', '')}"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        subprocess.run(command, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def run_optional_command(command: list[str], cwd: Path, log_path: Path) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src:{env.get('PYTHONPATH', '')}"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(command, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        handle.write(f"\nexit_code={result.returncode}\n")
        return result.returncode == 0


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


DEFAULT_REQUIRED_METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    "exact_ph": (
        "persistent_homology/h0/landscape_norm",
        "persistent_homology/h1/landscape_norm",
        "persistent_homology/h1/persistence_image_norm",
        "persistent_homology/h1/silhouette_norm",
        "persistent_homology/h1/entropy_vector_norm",
        "persistent_homology/h1/interval_count",
    ),
    "exact_cas": (
        "bgg_category_o/persistence/macaulay2_d_squared_zero",
        "bgg_category_o/persistence/macaulay2_identity_cone_acyclic",
        "analysis_control/exact_gudhi/macaulay2_resolution_ok",
    ),
    "bgg_category_o": (
        "bgg_category_o/persistence/two_parameter_square_residual",
        "bgg_category_o/persistence/gf2_exact_at_c1",
        "bgg_category_o/persistence/be_rank_residual_c1",
    ),
    "toric_vector_bundle": (
        "toric_vector_bundle/loss",
        "toric_vector_bundle/filtration_residual",
        "toric_vector_bundle/cech_gluing_residual",
        "toric_sheaf/cocycle_residual",
    ),
}


def metric_key_fragment(key: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in key).strip("_")


def missing_metric_alert_payload(
    observed: dict[str, Any],
    required_groups: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, float]:
    """Return strict missing-metric status flags for exact analysis groups."""

    groups = required_groups or DEFAULT_REQUIRED_METRIC_GROUPS
    payload: dict[str, float] = {}
    for group, keys in groups.items():
        missing = [key for key in keys if key not in observed]
        payload[f"metrics_status/{group}_missing_metric_count"] = float(len(missing))
        payload[f"metrics_status/{group}_all_metrics_present"] = 1.0 if not missing else 0.0
        for key in missing:
            payload[f"metrics_missing/{group}/{metric_key_fragment(key)}"] = 1.0
    return payload


def gudhi_wandb_payload(summary: dict[str, Any], *, step: int) -> dict[str, float]:
    """Stable metric aliases for exact GUDHI/Macaulay2 periodic audits."""

    fields = {
        "records": "records",
        "mean_points": "mean_points",
        "mean_num_simplices": "mean_num_simplices",
        "mean_betti0": "mean_betti0",
        "mean_betti1": "mean_betti1",
        "mean_h0_landscape_norm": "mean_h0_landscape_norm",
        "mean_h1_landscape_norm": "mean_h1_landscape_norm",
        "mean_h1_persistence_image_norm": "mean_h1_persistence_image_norm",
        "mean_two_parameter_commutative_square_residual": "mean_two_parameter_commutative_square_residual",
        "mean_macaulay2_homogeneous_d1": "mean_macaulay2_homogeneous_d1",
        "mean_macaulay2_homogeneous_d2": "mean_macaulay2_homogeneous_d2",
        "mean_macaulay2_d_squared_zero": "mean_macaulay2_d_squared_zero",
        "mean_macaulay2_identity_cone_acyclic": "mean_macaulay2_identity_cone_acyclic",
        "mean_finite_field_d_squared_zero": "mean_finite_field_d_squared_zero",
        "mean_finite_field_exact_at_c1": "mean_finite_field_exact_at_c1",
        "mean_be_rank_residual_c1": "mean_be_rank_residual_c1",
        "mean_simplicial_map_valid_fraction": "mean_simplicial_map_valid_fraction",
    }
    for dim in range(3):
        for suffix in (
            "interval_count",
            "total_persistence",
            "max_persistence",
            "mean_persistence",
            "persistence_entropy",
            "landscape_norm",
            "persistence_image_norm",
            "silhouette_norm",
            "entropy_vector_norm",
        ):
            key = f"mean_h{dim}_{suffix}"
            fields[key] = key
    payload: dict[str, float] = {
        "trainer/step": float(step),
        "metrics_status/gudhi_persistence_audit_available": 1.0,
        "metrics_status/macaulay2_bigraded_resolution_available": finite_float(
            summary.get("mean_macaulay2_d_squared_zero"), 0.0
        ),
    }
    for source, target in fields.items():
        value = finite_float(summary.get(source), 0.0)
        payload[f"gudhi_persistence/{target}"] = value
        payload[f"topology/exact_gudhi/{target}"] = value
    payload["bgg_category_o/persistence/exact_gudhi_betti0"] = finite_float(summary.get("mean_betti0"), 0.0)
    payload["bgg_category_o/persistence/exact_gudhi_betti1"] = finite_float(summary.get("mean_betti1"), 0.0)
    payload["bgg_category_o/persistence/two_parameter_square_residual"] = finite_float(
        summary.get("mean_two_parameter_commutative_square_residual"), 0.0
    )
    payload["bgg_category_o/persistence/macaulay2_homogeneous_d1"] = finite_float(
        summary.get("mean_macaulay2_homogeneous_d1"), 0.0
    )
    payload["bgg_category_o/persistence/macaulay2_homogeneous_d2"] = finite_float(
        summary.get("mean_macaulay2_homogeneous_d2"), 0.0
    )
    payload["bgg_category_o/persistence/macaulay2_d_squared_zero"] = finite_float(
        summary.get("mean_macaulay2_d_squared_zero"), 0.0
    )
    payload["bgg_category_o/persistence/macaulay2_identity_cone_acyclic"] = finite_float(
        summary.get("mean_macaulay2_identity_cone_acyclic"), 0.0
    )
    payload["topology/exact_gudhi/macaulay2_identity_cone_acyclic"] = finite_float(
        summary.get("mean_macaulay2_identity_cone_acyclic"), 0.0
    )
    payload["bgg_category_o/persistence/gf2_exact_at_c1"] = finite_float(
        summary.get("mean_finite_field_exact_at_c1"), 0.0
    )
    payload["bgg_category_o/persistence/be_rank_residual_c1"] = finite_float(
        summary.get("mean_be_rank_residual_c1"), 0.0
    )
    payload["topology/exact_gudhi/simplicial_map_valid_fraction"] = finite_float(
        summary.get("mean_simplicial_map_valid_fraction"), 0.0
    )
    payload["analysis_control/exact_gudhi/vectorized_ph_available"] = 1.0
    payload["analysis_control/exact_gudhi/f2_xy_module_available"] = 1.0
    payload["analysis_control/exact_gudhi/macaulay2_resolution_ok"] = finite_float(
        summary.get("mean_macaulay2_d_squared_zero"), 0.0
    )
    for dim in range(3):
        payload[f"persistent_homology/h{dim}/interval_count"] = finite_float(
            summary.get(f"mean_h{dim}_interval_count"), 0.0
        )
        payload[f"persistent_homology/h{dim}/total_persistence"] = finite_float(
            summary.get(f"mean_h{dim}_total_persistence"), 0.0
        )
        payload[f"persistent_homology/h{dim}/persistence_entropy"] = finite_float(
            summary.get(f"mean_h{dim}_persistence_entropy"), 0.0
        )
        payload[f"persistent_homology/h{dim}/landscape_norm"] = finite_float(
            summary.get(f"mean_h{dim}_landscape_norm"), 0.0
        )
        payload[f"persistent_homology/h{dim}/persistence_image_norm"] = finite_float(
            summary.get(f"mean_h{dim}_persistence_image_norm"), 0.0
        )
        payload[f"persistent_homology/h{dim}/silhouette_norm"] = finite_float(
            summary.get(f"mean_h{dim}_silhouette_norm"), 0.0
        )
        payload[f"persistent_homology/h{dim}/entropy_vector_norm"] = finite_float(
            summary.get(f"mean_h{dim}_entropy_vector_norm"), 0.0
        )
    payload.update(
        missing_metric_alert_payload(
            payload,
            {
                "exact_ph": DEFAULT_REQUIRED_METRIC_GROUPS["exact_ph"],
                "exact_cas": DEFAULT_REQUIRED_METRIC_GROUPS["exact_cas"],
                "bgg_category_o": DEFAULT_REQUIRED_METRIC_GROUPS["bgg_category_o"],
            },
        )
    )
    return payload


def log_gudhi_summary_to_wandb(run_path: str, summary: dict[str, Any], *, step: int, output_dir: Path) -> str:
    payload = gudhi_wandb_payload(summary, step=step)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "wandb_metrics.json"
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not run_path:
        return ""
    parts = [part for part in str(run_path).split("/") if part]
    if len(parts) != 3:
        raise ValueError(f"--run-path must be entity/project/run_id, got {run_path!r}")
    entity, project, run_id = parts
    import wandb  # type: ignore

    run = wandb.init(entity=entity, project=project, id=run_id, resume="allow")
    try:
        organized = organize_wandb_payload(payload)
        run.log(organized, step=int(step))
        update_wandb_summary(run, payload)
    finally:
        run.finish()
    return str(payload_path)


def pause_training_session(tmux_session: str, wait_seconds: float, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"requested training pause for tmux session: {tmux_session}\n")
        probe = subprocess.run(
            ["tmux", "has-session", "-t", tmux_session],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if probe.returncode != 0:
            handle.write("tmux session was not present; no pause signal sent\n")
            return
        subprocess.run(
            ["tmux", "send-keys", "-t", tmux_session, "C-c"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        handle.write(f"sent Ctrl-C; waiting {wait_seconds:.1f}s\n")
        handle.flush()
        time.sleep(max(0.0, wait_seconds))
        subprocess.run(
            ["tmux", "capture-pane", "-pt", tmux_session, "-S", "-20"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def is_compact_seq4096_checkpoint(path: Path) -> bool:
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    model = payload.get("model")
    return isinstance(model, dict) and "tok_emb.weight" in model


def is_tokengt_checkpoint(path: Path) -> bool:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    model = payload.get("model")
    return isinstance(model, dict) and "tokenizer.node_proj.weight" in model


def write_oai_unavailable_summary(base: Path, checkpoint: Path, step: int, reason: str) -> None:
    output_dir = base / "oai_competition"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint),
        "step": int(step),
        "source": "unavailable_for_checkpoint_family",
        "oai_competition/available": 0.0,
        "oai_competition/bpb_available": 0.0,
        "oai_competition/bpb": None,
        "oai_competition/loss": None,
        "reason": reason,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(
        "\n".join(
            [
                "# OAI Competition BPB",
                "",
                "This checkpoint was not evaluated for byte-level OAI BPB.",
                "",
                f"- Checkpoint: `{checkpoint}`",
                f"- Step: `{step}`",
                f"- Reason: {reason}",
                "",
                "Use this analysis run for TokenGT graph reconstruction, branch/merge topology, derived-category, and analogical-memory diagnostics. Run OAI BPB on a byte-level random-order or compact Seq4096 language-model checkpoint.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def seq4096_training_log_path(repo: Path, run_path: str, checkpoint: Path) -> Path:
    run_id = ""
    parts = [part for part in str(run_path).split("/") if part]
    if len(parts) >= 3:
        run_id = parts[-1]
    if not run_id:
        run_id = checkpoint.parent.name
    return repo / "amelie-iska" / "parameter-golf" / "logs" / f"{run_id}.txt"


def write_synopsis(base: Path, checkpoint: Path, step: int, run_path: str) -> Path:
    metrics_dir = base / "metrics"
    simplex_dir = base / "simplex"
    geometry_dir = base / "geometry"
    oai_dir = base / "oai_competition"
    derived_dir = base / "derived_category"
    memory_dir = base / "memory_trace"
    cas_dir = base / "cas_audit"
    gudhi_dir = base / "gudhi_persistence"
    test_time_dir = base / "test_time_scaling"
    toric_embedding_dir = base / "toric_embedding_report"
    branching_dir = base / "branching_reasoning_report"
    category = load_json(metrics_dir / "category_summary.json")
    geometry = load_json(geometry_dir / "reasoning_geometry_summary.json")
    simplex = load_json(simplex_dir / "reasoning_simplex_summary.json")
    oai = load_json(oai_dir / "summary.json")
    derived = load_json(derived_dir / "summary.json")
    memory = load_json(memory_dir / "summary.json")
    cas_audit = load_json(cas_dir / "cas_audit_summary.json")
    gudhi = load_json(gudhi_dir / "summary.json")
    test_time = load_json(test_time_dir / "summary.json")
    toric_embedding = load_json(toric_embedding_dir / "status.json")
    branching = load_json(branching_dir / "status.json")
    checkpoint_meta = load_json(metrics_dir / "checkpoint_meta.json")
    proposal = load_json(base / "training_adjustment_proposal.json")
    lines = [
        "# Post-Resume Analysis Synopsis",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Step: `{step}`",
        f"- W&B run: `{run_path or 'not requested'}`",
        "",
        "## Metric Categories",
    ]
    counts = category.get("counts", {})
    if isinstance(counts, dict) and counts:
        for key in sorted(counts):
            lines.append(f"- `{key}`: {counts[key]}")
    else:
        lines.append("- W&B metric category summary was not available.")
    if checkpoint_meta:
        metrics = checkpoint_meta.get("metrics", {})
        if isinstance(metrics, dict):
            lines.extend(
                [
                    "",
                    "## Checkpoint Metadata",
                    f"- `train_bpb`: {metrics.get('train_bpb', 'n/a')}",
                    f"- `best_val_bpb`: {metrics.get('best_val_bpb', 'n/a')}",
                    f"- `val_bpb`: {metrics.get('val_bpb', 'n/a')}",
                ]
            )
    if oai:
        lines.extend(
            [
                "",
                "## OAI Competition BPB",
                f"- Source: `{oai.get('source', 'n/a')}`",
                f"- `oai_competition/bpb`: {oai.get('oai_competition/bpb', 'n/a')}",
                f"- `oai_competition/loss`: {oai.get('oai_competition/loss', 'n/a')}",
                f"- Eval batches: `{oai.get('oai_competition/eval_batches', 'n/a')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Simplex Diagnostics",
            f"- Output directory: `{simplex_dir}`",
            f"- Records: `{simplex.get('records', 'n/a')}`",
            "",
            "## Geometry Diagnostics",
            f"- Output directory: `{geometry_dir}`",
            f"- Records: `{geometry.get('records', 'n/a')}`",
            f"- Branches: `{geometry.get('branches', 'n/a')}`",
            f"- Mean BPB: `{geometry.get('mean_bpb', 'n/a')}`",
            f"- Best BPB: `{geometry.get('best_bpb', 'n/a')}`",
            f"- Mean answer BPB: `{geometry.get('mean_answer_bpb', 'n/a')}`",
            f"- Best answer BPB: `{geometry.get('best_answer_bpb', 'n/a')}`",
            f"- Mean MST efficiency: `{geometry.get('mean_mst_efficiency', 'n/a')}`",
            f"- Mean path smoothness: `{geometry.get('mean_path_smoothness', 'n/a')}`",
            f"- Mean directed topology asymmetry: `{geometry.get('mean_topology_directed_asymmetry', 'n/a')}`",
            f"- Mean directed cycle flux: `{geometry.get('mean_topology_directed_cycle_flux', 'n/a')}`",
            f"- Mean radius-HDBSCAN clusters: `{geometry.get('mean_topology_hdbscan_cluster_count', 'n/a')}`",
            f"- Mean radius-HDBSCAN noise: `{geometry.get('mean_topology_hdbscan_noise_fraction', 'n/a')}`",
            f"- Mean radius-HDBSCAN stability: `{geometry.get('mean_topology_hdbscan_stability', 'n/a')}`",
            f"- Nested-simplicial topology plots: `{geometry_dir / 'topology'}`",
        ]
    )
    if derived:
        lines.extend(
            [
                "",
                "## Derived-Category / CCA Examples",
                f"- Output directory: `{derived_dir}`",
                f"- Objects analyzed: `{derived.get('object_count', 'n/a')}`",
                f"- Steps: `{derived.get('steps', 'n/a')}`",
                f"- Mean chain beta_1: `{derived.get('mean_chain_betti1', 'n/a')}`",
                f"- Mean symbolic projective dimension: `{derived.get('mean_symbolic_projective_dimension', 'n/a')}`",
                f"- Mean symbolic regularity: `{derived.get('mean_symbolic_regularity', 'n/a')}`",
                f"- Max Fitting maximal-minor count log10: `{derived.get('max_fitting_minor_count_log10', 'n/a')}`",
                f"- Report: `{derived_dir / 'REPORT.md'}`",
            ]
        )
    if toric_embedding:
        lines.extend(
            [
                "",
                "## Tropical-To-Toric Embedding Report",
                f"- Output directory: `{toric_embedding_dir}`",
                f"- Embedding manifest: `{toric_embedding.get('embedding_manifest', 'n/a')}`",
                f"- CAS sidecar ok: `{toric_embedding.get('cas_sidecar_ok', 'n/a')}`",
                f"- Report render ok: `{toric_embedding.get('report_render_ok', 'n/a')}`",
                f"- Screenshot render ok: `{toric_embedding.get('screenshot_render_ok', 'n/a')}`",
                f"- Report index: `{toric_embedding.get('report_index_html', 'n/a')}`",
            ]
        )
    if branching:
        lines.extend(
            [
                "",
                "## Branching Reasoning Simplex Report",
                f"- Output directory: `{branching_dir}`",
                f"- Embedding payload: `{branching.get('embedding_payload_npz', 'n/a')}`",
                f"- Report render ok: `{branching.get('report_render_ok', 'n/a')}`",
                f"- Screenshot render ok: `{branching.get('screenshot_render_ok', 'n/a')}`",
                f"- Node window: `{branching.get('embedding_node_offset', 'n/a')}` + `{branching.get('embedding_max_nodes', 'n/a')}`",
                f"- Report index: `{branching.get('report_index_html', 'n/a')}`",
            ]
        )
    if memory:
        lines.extend(
            [
                "",
                "## Analogical Memory Trace Examples",
                f"- Output directory: `{memory_dir}`",
                f"- Queries analyzed: `{memory.get('query_count', 'n/a')}`",
                f"- Candidates analyzed: `{memory.get('candidate_count', 'n/a')}`",
                f"- Rank-1 teacher-argmax agreement: `{memory.get('rank1_teacher_argmax_rate', 'n/a')}`",
                f"- Mean retrieval entropy: `{memory.get('mean_retrieval_entropy', 'n/a')}`",
                f"- Mean rank-1 model retrieval probability: `{memory.get('mean_rank1_retrieval_probability', 'n/a')}`",
                f"- Mean rank-1 teacher probability: `{memory.get('mean_rank1_teacher_probability', 'n/a')}`",
                f"- Mean branch / merge counts: `{memory.get('mean_dag_branch_count', 'n/a')}` / `{memory.get('mean_dag_merge_count', 'n/a')}`",
                f"- Mean rank-1 derived-category similarity: `{memory.get('mean_rank1_derived_category_similarity', 'n/a')}`",
                f"- Report: `{memory_dir / 'REPORT.md'}`",
            ]
        )
    if cas_audit:
        backends = cas_audit.get("backend_status", {}) if isinstance(cas_audit.get("backend_status", {}), dict) else {}
        toolchain = cas_audit.get("toolchain_status", {}) if isinstance(cas_audit.get("toolchain_status", {}), dict) else {}
        build_ok = cas_audit.get("build_returncode") == 0
        lines.extend(
            [
                "",
                "## CAS Exact-Certificate Audit",
                f"- Output directory: `{cas_dir}`",
                f"- Build command returned: `{cas_audit.get('build_returncode', 'n/a')}`",
                f"- Closed-form/CAS certificate build ok: `{build_ok}`",
            ]
        )
        for name, info in sorted(backends.items()):
            if isinstance(info, dict):
                lines.append(
                    f"- `{name}` available: `{info.get('available', 'n/a')}`; "
                    f"provenance: `{info.get('provenance', 'n/a')}`; "
                    f"version: `{info.get('version', 'n/a')}`"
                )
        if toolchain:
            available_tools = [
                name for name, info in sorted(toolchain.items())
                if isinstance(info, dict) and bool(info.get("available", False))
            ]
            missing_tools = [
                name for name, info in sorted(toolchain.items())
                if isinstance(info, dict) and not bool(info.get("available", False))
            ]
            lines.append(f"- Exact command-line tools available: `{', '.join(available_tools) or 'none'}`")
            lines.append(f"- Exact command-line tools unavailable: `{', '.join(missing_tools) or 'none'}`")
    if gudhi:
        lines.extend(
            [
                "",
                "## GUDHI / Macaulay2 Persistence Audit",
                f"- Output directory: `{gudhi_dir}`",
                f"- Browser index: `{gudhi_dir / 'index.html'}`",
                f"- Records: `{gudhi.get('records', 'n/a')}`",
                f"- Mean GUDHI simplex count: `{gudhi.get('mean_num_simplices', 'n/a')}`",
                f"- Mean Betti0 / Betti1: `{gudhi.get('mean_betti0', 'n/a')}` / `{gudhi.get('mean_betti1', 'n/a')}`",
                f"- Mean H0 landscape norm: `{gudhi.get('mean_h0_landscape_norm', 'n/a')}`",
                f"- Mean H1 persistence-image norm: `{gudhi.get('mean_h1_persistence_image_norm', 'n/a')}`",
                f"- M2 homogeneous d1/d2: `{gudhi.get('mean_macaulay2_homogeneous_d1', 'n/a')}` / `{gudhi.get('mean_macaulay2_homogeneous_d2', 'n/a')}`",
                f"- M2 d^2=0: `{gudhi.get('mean_macaulay2_d_squared_zero', 'n/a')}`",
                f"- 2-parameter square residual: `{gudhi.get('mean_two_parameter_commutative_square_residual', 'n/a')}`",
            ]
        )
    if test_time:
        test_summary = test_time.get("summary", {}) if isinstance(test_time.get("summary", {}), dict) else {}
        budgets = test_summary.get("summary", {}) if isinstance(test_summary.get("summary", {}), dict) else {}
        lines.extend(
            [
                "",
                "## Held-Out Test-Time Scaling",
                f"- Output directory: `{test_time_dir}`",
                f"- Data path: `{test_summary.get('data_path', 'n/a')}`",
                f"- Budgets: `{test_summary.get('budgets', 'n/a')}`",
            ]
        )
        for budget, values in sorted(budgets.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 0):
            if not isinstance(values, dict):
                continue
            lines.append(
                "- Budget "
                f"`{budget}`: best oracle MSE `{values.get('best_oracle_mse', 'n/a')}`, "
                f"reward `{values.get('best_reward_proxy', 'n/a')}`, "
                f"policy entropy `{values.get('policy_entropy', 'n/a')}`, "
                f"action diversity `{values.get('action_diversity', 'n/a')}`"
            )
    if proposal:
        decision = proposal.get("decision", {}) if isinstance(proposal.get("decision", {}), dict) else {}
        bpb_gate = proposal.get("bpb_gate", {}) if isinstance(proposal.get("bpb_gate", {}), dict) else {}
        structural = (
            proposal.get("structural_diagnostics", {})
            if isinstance(proposal.get("structural_diagnostics", {}), dict)
            else {}
        )
        lines.extend(
            [
                "",
                "## Training Control Proposal",
                f"- Target status: `{decision.get('target_status', 'n/a')}`",
                f"- Primary action: `{decision.get('primary_action', 'n/a')}`",
                f"- Current primary BPB: `{bpb_gate.get('current_primary_bpb', 'n/a')}`",
                f"- Best observed BPB: `{bpb_gate.get('best_observed_bpb', 'n/a')}`",
                f"- Gap to target: `{bpb_gate.get('gap_to_target', 'n/a')}`",
                f"- Required drop / 100 steps: `{bpb_gate.get('required_drop_per_100_steps', 'n/a')}`",
                f"- Recent drop / 100 steps: `{bpb_gate.get('recent_drop_per_100_steps', 'n/a')}`",
                f"- Structural pressure: `{structural.get('structural_pressure', 'n/a')}`",
                f"- Dominant structural family: `{structural.get('dominant_family', 'n/a')}`",
                f"- Proposal file: `{base / 'training_adjustment_proposal.md'}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Initial Interpretation",
            "",
            "Treat deterministic validation BPB as the checkpoint gate.  The",
            "GFlowNet and score-first values are test-time-scaling diagnostics,",
            "not promotion metrics.  Geometry is considered useful only when lower",
            "BPB or lower answer-span BPB accompanies higher MST efficiency, lower",
            "curvature, healthier toric entropy, and directed nested-complex metrics",
            "that show useful noncommutative structure without exploding cycle flux.",
        ]
    )
    out = base / "SYNOPSIS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_analysis_index(base: Path, checkpoint: Path, step: int, run_path: str) -> Path:
    """Write a browser landing page linking all periodic analysis families."""

    def exists_rel(path: Path) -> str:
        if path.exists():
            return path.relative_to(base).as_posix()
        return ""

    cards = [
        ("Synopsis", exists_rel(base / "SYNOPSIS.md"), "Human-readable checkpoint summary and training-control proposal."),
        ("GUDHI / M2 Persistence", exists_rel(base / "gudhi_persistence" / "index.html"), "Exact simplex trees, vectorized PH, F2[x_level,y_radius] modules, and Macaulay2 resolutions."),
        ("Tropical-To-Toric Embedding", exists_rel(base / "toric_embedding_report" / "index.html"), "Exact sidecar report for tropical attention exponents embedded into toric geometry, with multiplicity/balance/Chow audits."),
        ("Branching Reasoning Simplex Report", exists_rel(base / "branching_reasoning_report" / "branching_reasoning_trajectory.html"), "Branch/merge graph-of-thought trajectory, per-step simplex trees, analogical maps, and vectorized PH panels from checkpoint embeddings."),
        ("Geometry Summary", exists_rel(base / "geometry" / "reasoning_geometry_summary.json"), "Reasoning geometry, toric, tropical, topology, GraphCG, and analogical metrics."),
        ("Simplex Summary", exists_rel(base / "simplex" / "reasoning_simplex_summary.json"), "Reasoning/K/BPB triangle and tetrahedron projections."),
        ("CAS Audit", exists_rel(base / "cas_audit" / "cas_audit_summary.json"), "Exact Sage/Macaulay2/toric-toolchain certificate report."),
        ("OAI Competition BPB", exists_rel(base / "oai_competition" / "summary.json"), "Score-first byte-level evaluation summary when available."),
        ("Derived Category", exists_rel(base / "derived_category" / "REPORT.md"), "Derived-category, CCA, and chain-complex example analysis."),
        ("Memory Trace", exists_rel(base / "memory_trace" / "REPORT.md"), "Analogical trajectory-memory retrieval analysis."),
        ("Test-Time Scaling", exists_rel(base / "test_time_scaling" / "summary.json"), "Budgeted branch/search scaling metrics."),
    ]
    html_cards = []
    for title, href, desc in cards:
        link = f'<a class="pill" href="{href}">open</a>' if href else '<span class="pill muted">not generated</span>'
        html_cards.append(
            f"""<div class="card"><h2>{title}</h2><p>{desc}</p><div class="links">{link}</div></div>"""
        )
    css = """
:root{color-scheme:dark;--bg:#030712;--panel:#07111f;--text:#e8fbff;--muted:#91a8b7;--cyan:#37e8ff;--border:rgba(55,232,255,.28)}
body{margin:0;background:radial-gradient(circle at top left,#092238 0,var(--bg) 42rem);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:32px 24px 64px}.hero,.card{background:linear-gradient(180deg,rgba(11,23,40,.94),rgba(5,13,25,.96));border:1px solid var(--border);border-radius:8px;box-shadow:0 18px 50px rgba(0,0,0,.28)}
.hero{padding:24px;margin-bottom:20px}.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}.card{padding:16px}h1{margin:0 0 8px;font-size:28px;letter-spacing:0}h2{margin:0 0 10px;font-size:17px}p{color:var(--muted);line-height:1.5}.links{display:flex;gap:8px;flex-wrap:wrap}.pill{border:1px solid var(--border);border-radius:999px;padding:6px 10px;background:rgba(55,232,255,.07);color:var(--cyan);text-decoration:none}.muted{color:var(--muted)}
"""
    out = base / "index.html"
    out.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>ToricGT Periodic Analysis Step {step}</title><style>{css}</style></head>
<body><main><section class="hero"><h1>ToricGT Periodic Analysis</h1>
<p>Checkpoint <code>{checkpoint}</code> at step <code>{step}</code>. W&B run: <code>{run_path or 'not requested'}</code>.</p>
</section><section class="grid">{''.join(html_cards)}</section></main></body></html>
""",
        encoding="utf-8",
    )
    return out


def write_analysis_status(base: Path, checkpoint: Path, step: int, run_path: str) -> Path:
    """Write a machine-readable inventory for periodic analysis artifacts."""

    payload = {
        "schema": "toricgt.periodic_analysis_status.v1",
        "checkpoint": str(checkpoint),
        "step": int(step),
        "wandb_run": str(run_path or ""),
        "paths": {
            "index_html": str(base / "index.html"),
            "synopsis_md": str(base / "SYNOPSIS.md"),
            "geometry_summary": str(base / "geometry" / "reasoning_geometry_summary.json"),
            "gudhi_persistence_index": str(base / "gudhi_persistence" / "index.html"),
            "toric_embedding_report_index": str(base / "toric_embedding_report" / "index.html"),
            "toric_embedding_screenshot_index": str(base / "toric_embedding_report" / "html_screenshots" / "index.html"),
            "branching_reasoning_report_index": str(base / "branching_reasoning_report" / "branching_reasoning_trajectory.html"),
            "branching_reasoning_screenshot_index": str(base / "branching_reasoning_report" / "html_screenshots" / "index.html"),
            "oai_competition_summary": str(base / "oai_competition" / "summary.json"),
            "test_time_scaling_summary": str(base / "test_time_scaling" / "summary.json"),
        },
        "availability": {
            "index_html": (base / "index.html").exists(),
            "synopsis_md": (base / "SYNOPSIS.md").exists(),
            "geometry_summary": (base / "geometry" / "reasoning_geometry_summary.json").exists(),
            "gudhi_persistence_index": (base / "gudhi_persistence" / "index.html").exists(),
            "toric_embedding_report_index": (base / "toric_embedding_report" / "index.html").exists(),
            "toric_embedding_screenshot_index": (base / "toric_embedding_report" / "html_screenshots" / "index.html").exists(),
            "branching_reasoning_report_index": (base / "branching_reasoning_report" / "branching_reasoning_trajectory.html").exists(),
            "branching_reasoning_screenshot_index": (base / "branching_reasoning_report" / "html_screenshots" / "index.html").exists(),
            "oai_competition_summary": (base / "oai_competition" / "summary.json").exists(),
            "test_time_scaling_summary": (base / "test_time_scaling" / "summary.json").exists(),
        },
        "toric_embedding": load_json(base / "toric_embedding_report" / "status.json"),
        "branching_reasoning": load_json(base / "branching_reasoning_report" / "status.json"),
    }
    out = base / "analysis_status.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _resolve_embedding_payload_paths(embedding_manifest: Path) -> tuple[Path, Path | None]:
    manifest = load_json(embedding_manifest)
    rows = manifest.get("payloads", [])
    if not isinstance(rows, list) or not rows:
        raise FileNotFoundError(f"{embedding_manifest} does not contain embedding payload rows")
    row = rows[0] if isinstance(rows[0], dict) else {}
    candidates: list[Path] = []
    for key in ("npz", "relative_npz"):
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value))
        candidates.extend(
            [
                path,
                embedding_manifest.parent / path,
                embedding_manifest.parent.parent / path,
            ]
        )
    npz_path = next((path for path in candidates if path.exists()), None)
    if npz_path is None:
        raise FileNotFoundError(f"could not resolve first embedding payload NPZ from {embedding_manifest}")
    json_candidates: list[Path] = []
    for key in ("json", "relative_json"):
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value))
        json_candidates.extend(
            [
                path,
                embedding_manifest.parent / path,
                embedding_manifest.parent.parent / path,
            ]
        )
    json_path = next((path for path in json_candidates if path.exists()), None)
    return npz_path, json_path


def run_branching_reasoning_report(args: argparse.Namespace, base: Path, repo: Path) -> dict[str, Any]:
    """Render branch/merge simplex report from saved geometry embedding payloads."""

    report_dir = base / "branching_reasoning_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "enabled": not bool(args.skip_branching_reasoning_report),
        "embedding_manifest": str(base / "geometry" / "embeddings" / "manifest.json"),
        "embedding_payload_npz": "",
        "embedding_payload_json": "",
        "embedding_max_nodes": int(args.branching_reasoning_max_nodes),
        "embedding_node_offset": int(args.branching_reasoning_node_offset),
        "report_output_dir": str(report_dir),
        "report_render_ok": False,
        "screenshot_render_ok": False,
        "report_index_html": "",
        "trajectory_html": "",
        "screenshot_index_html": "",
        "reason": "",
    }
    status_path = report_dir / "status.json"
    if args.skip_branching_reasoning_report:
        status["reason"] = "disabled_by_cli"
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return status
    embedding_manifest = base / "geometry" / "embeddings" / "manifest.json"
    if not embedding_manifest.exists():
        status["reason"] = "no geometry embeddings manifest was emitted for this checkpoint family"
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return status
    try:
        npz_path, json_path = _resolve_embedding_payload_paths(embedding_manifest)
    except Exception as exc:
        status["reason"] = f"{type(exc).__name__}: {exc}"
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return status
    status["embedding_payload_npz"] = str(npz_path)
    status["embedding_payload_json"] = str(json_path or "")
    render_cmd = [
        sys.executable,
        "scripts/render_branching_reasoning_trajectory_report.py",
        "--output-dir",
        str(report_dir),
        "--embedding-payload-npz",
        str(npz_path),
        "--embedding-max-nodes",
        str(max(0, int(args.branching_reasoning_max_nodes))),
        "--embedding-node-offset",
        str(max(0, int(args.branching_reasoning_node_offset))),
    ]
    if json_path is not None:
        render_cmd.extend(["--embedding-payload-json", str(json_path)])
    status["report_render_ok"] = run_optional_command(
        render_cmd,
        cwd=repo,
        log_path=base / "logs" / "branching_reasoning_report.log",
    )
    if status["report_render_ok"]:
        status["report_index_html"] = str(report_dir / "index.html")
        status["trajectory_html"] = str(report_dir / "branching_reasoning_trajectory.html")
    if status["report_render_ok"] and not bool(args.skip_branching_reasoning_screenshots):
        screenshot_dir = report_dir / "html_screenshots"
        screenshot_cmd = [
            sys.executable,
            "scripts/render_html_screenshots.py",
            "--source-dir",
            str(report_dir),
            "--output-dir",
            str(screenshot_dir),
            "--width",
            "1600",
            "--height",
            "1000",
            "--wait-ms",
            "1800",
            "--timeout-ms",
            "90000",
            "--no-full-page",
            "--viewport-slices",
            "8",
            "--interaction-audit",
            "--interaction-delay-ms",
            "700",
        ]
        if bool(args.assert_branching_reasoning_report):
            screenshot_cmd.append("--assert-branching-report")
        status["screenshot_render_ok"] = run_optional_command(
            screenshot_cmd,
            cwd=repo,
            log_path=base / "logs" / "branching_reasoning_screenshots.log",
        )
        if status["screenshot_render_ok"]:
            status["screenshot_index_html"] = str(screenshot_dir / "index.html")
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


def run_toric_embedding_report(args: argparse.Namespace, base: Path, repo: Path) -> dict[str, Any]:
    """Render exact tropical-to-toric sidecar reports for periodic analysis."""

    status: dict[str, Any] = {
        "enabled": not bool(args.skip_toric_embedding_report),
        "embedding_manifest": "",
        "cas_sidecar_output_dir": "",
        "cas_sidecar_ok": False,
        "report_output_dir": str(base / "toric_embedding_report"),
        "report_render_ok": False,
        "screenshot_render_ok": False,
        "report_index_html": "",
        "screenshot_index_html": "",
        "reason": "",
    }
    report_dir = base / "toric_embedding_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    status_path = report_dir / "status.json"
    if args.skip_toric_embedding_report:
        status["reason"] = "disabled_by_cli"
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return status

    embedding_manifest = base / "geometry" / "embeddings" / "manifest.json"
    sidecar_dir = base / "embedding_cas_sidecar"
    if embedding_manifest.exists():
        status["embedding_manifest"] = str(embedding_manifest)
        status["cas_sidecar_output_dir"] = str(sidecar_dir)
        sidecar_cmd = [
            sys.executable,
            "scripts/run_embedding_cas_sidecar.py",
            "--embedding-manifest",
            str(embedding_manifest),
            "--output-dir",
            str(sidecar_dir),
            "--records",
            str(args.toric_embedding_sidecar_records),
            "--max-points",
            str(args.toric_embedding_sidecar_max_points),
            "--exponent-dim",
            str(args.toric_embedding_exponent_dim),
            "--quantization-scale",
            str(args.toric_embedding_quantization_scale),
        ]
        status["cas_sidecar_ok"] = run_optional_command(
            sidecar_cmd,
            cwd=repo,
            log_path=base / "logs" / "embedding_cas_sidecar.log",
        )
    else:
        status["reason"] = "no geometry embeddings manifest was emitted for this checkpoint family"

    has_records = bool(list((sidecar_dir / "records").glob("*_cas_sidecar.json"))) if sidecar_dir.exists() else False
    if not has_records:
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return status

    render_cmd = [
        sys.executable,
        "scripts/render_toric_embedding_report.py",
        "--sidecar-dir",
        str(sidecar_dir),
        "--output-dir",
        str(report_dir),
    ]
    if bool(args.toric_embedding_build_vector_bundle):
        render_cmd.append("--build-vector-bundle-certificate")
    status["report_render_ok"] = run_optional_command(
        render_cmd,
        cwd=repo,
        log_path=base / "logs" / "toric_embedding_report.log",
    )
    if status["report_render_ok"]:
        status["report_index_html"] = str(report_dir / "index.html")
    if status["report_render_ok"] and not bool(args.skip_toric_embedding_screenshots):
        screenshot_dir = report_dir / "html_screenshots"
        status["screenshot_render_ok"] = run_optional_command(
            [
                sys.executable,
                "scripts/render_html_screenshots.py",
                "--source-dir",
                str(report_dir),
                "--output-dir",
                str(screenshot_dir),
                "--width",
                "1600",
                "--height",
                "1000",
                "--wait-ms",
                "1600",
                "--timeout-ms",
                "60000",
            ],
            cwd=repo,
            log_path=base / "logs" / "toric_embedding_screenshots.log",
        )
        if status["screenshot_render_ok"]:
            status["screenshot_index_html"] = str(screenshot_dir / "index.html")
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


def trigger_codex_review_hook(args: argparse.Namespace, base: Path, checkpoint: Path, step: int) -> None:
    if not args.codex_review_hook:
        return
    hook = Path(args.codex_review_hook)
    if not hook.is_absolute():
        hook = Path.cwd() / hook
    command = [
        str(hook),
        "--analysis-dir",
        str(base),
        "--checkpoint",
        str(checkpoint),
        "--step",
        str(step),
        "--training-tmux",
        str(args.training_tmux),
        "--config",
        str(args.config),
    ]
    if args.run_path:
        command.extend(["--run-path", str(args.run_path)])
    if args.codex_review_session_id:
        command.extend(["--session-id", str(args.codex_review_session_id)])
    if args.codex_review_tmux_prefix:
        command.extend(["--tmux-session", f"{args.codex_review_tmux_prefix}_{step:08d}"])
    run_command(
        command,
        cwd=Path.cwd(),
        log_path=base / "logs" / "codex_review_hook.log",
    )


def main() -> None:
    args = parse_args()
    repo = Path.cwd()
    checkpoint_dir = Path(args.checkpoint_dir)
    target_step = int(args.target_step or (args.start_step + args.after_steps))
    print(f"waiting for checkpoint >= {target_step} in {checkpoint_dir}", flush=True)
    found: tuple[int, Path] | None = None
    while found is None:
        found = latest_checkpoint(checkpoint_dir, target_step, min_mtime_unix=float(args.min_mtime_unix or 0.0))
        if found is None:
            time.sleep(max(1.0, args.poll_seconds))
    step, checkpoint = found
    base = Path(args.output_root) / f"step-{step:08d}"
    base.mkdir(parents=True, exist_ok=True)
    if args.pause_training_before_analysis:
        pause_training_session(
            tmux_session=str(args.training_tmux),
            wait_seconds=float(args.pause_wait_seconds),
            log_path=base / "logs" / "training_pause.log",
        )
    print(f"analyzing checkpoint {checkpoint} at step {step}", flush=True)

    if args.run_path:
        run_command(
            [
                sys.executable,
                "scripts/analyze_wandb_metrics.py",
                "--run-path",
                args.run_path,
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(base / "metrics"),
            ],
            cwd=repo,
            log_path=base / "logs" / "metrics.log",
        )
    config = load_yaml(args.config)
    oai_config = config.get("oai_competition", {}) if isinstance(config.get("oai_competition", {}), dict) else {}
    token_gt_checkpoint = is_tokengt_checkpoint(checkpoint)
    if not args.skip_oai_competition_eval and token_gt_checkpoint:
        command = [
            sys.executable,
            "scripts/evaluate_tokengt_fineweb_bpb.py",
            "--checkpoint",
            str(checkpoint),
            "--config",
            args.config,
            "--output-json",
            str(base / "oai_competition" / "summary.json"),
            "--device",
            args.device,
            "--precision",
            args.precision,
            "--batches",
            str(args.tokengt_bpb_batches),
        ]
        if args.run_path:
            command.extend(["--wandb-run-path", args.run_path])
        if args.tokengt_bpb_token_glob:
            command.extend(["--token-glob", args.tokengt_bpb_token_glob])
        if args.tokengt_bpb_tokenizer_path:
            command.extend(["--tokenizer-path", args.tokengt_bpb_tokenizer_path])
        ok = run_optional_command(
            command,
            cwd=repo,
            log_path=base / "logs" / "tokengt_fineweb_bpb.log",
        )
        if not ok:
            write_oai_unavailable_summary(
                base,
                checkpoint,
                step,
                "TokenGT graph-node SP1024 FineWeb BPB evaluator failed; see logs/tokengt_fineweb_bpb.log.",
            )
    elif bool(oai_config.get("enabled", False)) and not args.skip_oai_competition_eval:
        command = [
            sys.executable,
            "scripts/evaluate_oai_competition_bpb.py",
            "--checkpoint",
            str(checkpoint),
            "--config",
            args.config,
            "--output-json",
            str(base / "oai_competition" / "summary.json"),
            "--device",
            args.device,
            "--precision",
            args.precision,
            "--seed",
            str(args.seed),
        ]
        if args.run_path:
            command.extend(["--wandb-run-path", args.run_path])
        run_optional_command(
            command,
            cwd=repo,
            log_path=base / "logs" / "oai_competition.log",
        )
    if args.skip_simplex_geometry:
        pass
    elif token_gt_checkpoint:
        command = [
            sys.executable,
            "scripts/evaluate_tokengt_reasoning_geometry_suite.py",
            "--checkpoint",
            str(checkpoint),
            "--config",
            args.config,
            "--output-dir",
            str(base / "geometry"),
            "--records",
            str(args.geometry_records),
            "--batch-size",
            "2",
            "--device",
            args.device,
            "--precision",
            args.precision,
            "--seed",
            str(args.seed),
        ]
        if args.data_glob:
            command.extend(["--data-glob", args.data_glob])
        if args.run_path:
            command.extend(["--wandb-run-path", args.run_path])
        if not args.skip_toric_embedding_report:
            command.append("--emit-embedding-payloads")
        run_optional_command(
            command,
            cwd=repo,
            log_path=base / "logs" / "tokengt_reasoning_geometry_suite.log",
        )
    elif is_compact_seq4096_checkpoint(checkpoint):
        run_optional_command(
            [
                sys.executable,
                "scripts/evaluate_seq4096_reasoning_geometry_suite.py",
                "--checkpoint",
                str(checkpoint),
                "--log",
                str(seq4096_training_log_path(repo, args.run_path, checkpoint)),
                "--output-dir",
                str(base),
                "--run-path",
                args.run_path,
                "--target-bpb",
                str(args.target_bpb),
                "--records",
                str(args.geometry_records),
                "--seed",
                str(args.seed),
            ],
            cwd=repo,
            log_path=base / "logs" / "seq4096_reasoning_geometry_suite.log",
        )
    else:
        run_optional_command(
            [
                sys.executable,
                "scripts/evaluate_reasoning_simplex.py",
                "--checkpoint",
                str(checkpoint),
                "--config",
                args.config,
                "--data-glob",
                args.data_glob,
                "--output-dir",
                str(base / "simplex"),
                "--samples",
                str(args.simplex_samples),
                "--batch-size",
                "2",
                "--seq-len",
                str(args.seq_len),
                "--budgets",
                "1",
                "2",
                "4",
                "8",
                "--device",
                args.device,
                "--precision",
                args.precision,
                "--seed",
                str(args.seed),
            ],
            cwd=repo,
            log_path=base / "logs" / "simplex.log",
        )
        run_optional_command(
            [
                sys.executable,
                "scripts/evaluate_reasoning_geometry_suite.py",
                "--checkpoint",
                str(checkpoint),
                "--config",
                args.config,
                "--data-glob",
                args.data_glob,
                "--output-dir",
                str(base / "geometry"),
                "--records",
                str(args.geometry_records),
                "--branches",
                str(args.geometry_branches),
                "--seq-len",
                str(args.seq_len),
                "--device",
                args.device,
                "--precision",
                args.precision,
                "--seed",
                str(args.seed),
            ],
            cwd=repo,
            log_path=base / "logs" / "geometry.log",
        )
    if not args.skip_derived_category_analysis:
        derived_example_dir = Path(args.derived_category_example_dir) if args.derived_category_example_dir else checkpoint_dir / "derived_category_examples"
        run_optional_command(
            [
                sys.executable,
                "scripts/analyze_derived_category_examples.py",
                "--example-dir",
                str(derived_example_dir),
                "--output-dir",
                str(base / "derived_category"),
                "--max-files",
                str(args.derived_category_max_files),
                "--max-objects",
                str(args.derived_category_max_objects),
            ],
            cwd=repo,
            log_path=base / "logs" / "derived_category.log",
        )
    if not args.skip_memory_trace_analysis:
        memory_example_dir = Path(args.memory_trace_example_dir) if args.memory_trace_example_dir else checkpoint_dir / "memory_trace_examples"
        run_optional_command(
            [
                sys.executable,
                "scripts/analyze_memory_trace_examples.py",
                "--example-dir",
                str(memory_example_dir),
                "--output-dir",
                str(base / "memory_trace"),
                "--max-files",
                str(args.memory_trace_max_files),
                "--max-queries",
                str(args.memory_trace_max_queries),
            ],
            cwd=repo,
            log_path=base / "logs" / "memory_trace.log",
        )
    if args.cas_audit:
        cas_command = [
            sys.executable,
            "scripts/run_periodic_cas_audit.py",
            "--output-dir",
            str(base / "cas_audit"),
            "--num-vertices",
            str(args.cas_audit_num_vertices),
            "--python-bin",
            sys.executable,
        ]
        if args.cas_audit_sage_normal_fan:
            cas_command.append("--sage-normal-fan")
        if args.cas_audit_macaulay2_smoke:
            cas_command.append("--macaulay2-smoke")
        if args.cas_audit_macaulay2_toric_ideal:
            cas_command.append("--macaulay2-toric-ideal")
        if args.cas_audit_all_exact_cas:
            cas_command.append("--all-exact-cas")
        if args.cas_audit_require_cas:
            cas_command.append("--require-cas")
        run_optional_command(
            cas_command,
            cwd=repo,
            log_path=base / "logs" / "cas_audit.log",
        )
    if args.gudhi_persistence_audit:
        run_command(
            [
                sys.executable,
                "scripts/run_gudhi_persistence_audit.py",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(base / "gudhi_persistence"),
                "--records",
                str(args.gudhi_records),
                "--max-points",
                str(args.gudhi_max_points),
                "--num-radii",
                str(args.gudhi_num_radii),
                "--num-levels",
                str(args.gudhi_num_levels),
                "--radius-quantile",
                str(args.gudhi_radius_quantile),
                "--macaulay2-timeout-seconds",
                str(args.gudhi_macaulay2_timeout_seconds),
            ],
            cwd=repo,
            log_path=base / "logs" / "gudhi_persistence.log",
        )
        gudhi_summary = load_json(base / "gudhi_persistence" / "summary.json")
        try:
            payload_path = log_gudhi_summary_to_wandb(
                args.run_path,
                gudhi_summary,
                step=step,
                output_dir=base / "gudhi_persistence",
            )
            (base / "logs" / "gudhi_wandb.log").write_text(
                f"wrote {payload_path or (base / 'gudhi_persistence' / 'wandb_metrics.json')}\n",
                encoding="utf-8",
            )
        except Exception as exc:
            (base / "logs" / "gudhi_wandb.log").write_text(f"gudhi W&B metric logging failed: {exc}\n", encoding="utf-8")
    if not args.skip_test_time_scaling:
        run_optional_command(
            [
                sys.executable,
                "scripts/test_time_scaling.py",
                "--checkpoint",
                str(checkpoint),
                "--config",
                args.config,
                "--data-path",
                args.test_data_glob,
                "--output-json",
                str(base / "test_time_scaling" / "summary.json"),
                "--device",
                args.device,
                "--batch-size",
                "2",
                "--batches",
                str(args.test_time_scaling_batches),
                "--budgets",
                *[str(budget) for budget in args.test_time_scaling_budgets],
                "--no-auto-curate-new-data",
            ],
            cwd=repo,
            log_path=base / "logs" / "test_time_scaling.log",
        )
    run_toric_embedding_report(args, base, repo)
    run_branching_reasoning_report(args, base, repo)
    proposal_command = [
        sys.executable,
        "scripts/propose_training_adjustments.py",
        "--analysis-dir",
        str(base),
        "--target-bpb",
        str(args.target_bpb),
        "--gate-step",
        str(args.gate_step),
        "--checkpoint-step",
        str(step),
    ]
    if args.run_path:
        proposal_command.extend(["--wandb-run-path", args.run_path])
    run_optional_command(
        proposal_command,
        cwd=repo,
        log_path=base / "logs" / "training_adjustment_proposal.log",
    )
    synopsis = write_synopsis(base, checkpoint, step, args.run_path)
    index = write_analysis_index(base, checkpoint, step, args.run_path)
    status = write_analysis_status(base, checkpoint, step, args.run_path)
    print(f"wrote {synopsis}", flush=True)
    print(f"wrote {index}", flush=True)
    print(f"wrote {status}", flush=True)
    trigger_codex_review_hook(args, base, checkpoint, step)


if __name__ == "__main__":
    main()
