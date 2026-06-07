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


CHECKPOINT_PATTERN = re.compile(r"(?:random_order_step_|_step_)(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--config", default="config/train.parameter_golf_random_order_dense.yaml")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--simplex-samples", type=int, default=8)
    parser.add_argument("--geometry-records", type=int, default=4)
    parser.add_argument("--geometry-branches", type=int, default=6)
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
    return parser.parse_args()


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
    category = load_json(metrics_dir / "category_summary.json")
    geometry = load_json(geometry_dir / "reasoning_geometry_summary.json")
    simplex = load_json(simplex_dir / "reasoning_simplex_summary.json")
    oai = load_json(oai_dir / "summary.json")
    derived = load_json(derived_dir / "summary.json")
    memory = load_json(memory_dir / "summary.json")
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
    if bool(oai_config.get("enabled", False)) and not args.skip_oai_competition_eval:
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
    print(f"wrote {synopsis}", flush=True)
    trigger_codex_review_hook(args, base, checkpoint, step)


if __name__ == "__main__":
    main()
