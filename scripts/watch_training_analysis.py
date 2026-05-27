#!/usr/bin/env python3
# Example:
# conda run --no-capture-output -n tokengt env PYTHONPATH=src \
#   python scripts/watch_training_analysis.py \
#   --checkpoint-dir checkpoints/parameter_golf_oai_dense \
#   --start-step 14750 --after-steps 1500 \
#   --run-path amelie-iska-math/toricgt-parameter-golf/oai-rescue-14750 \
#   --output-root outputs/post_resume_analysis/oai-rescue-14750
"""Wait for a future checkpoint and run non-interrupting metric analyses.

The watcher intentionally evaluates on CPU by default.  This keeps the analysis
from competing with the active training tmux for GPU memory.  It writes a
compact Markdown synopsis after W&B metrics, budget simplices, branch
trajectories, Ramachandran-style phase plots, and energy landscapes complete.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


CHECKPOINT_PATTERN = re.compile(r"random_order_step_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="checkpoints/parameter_golf_oai_dense")
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--after-steps", type=int, default=1500)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--run-path", default="")
    parser.add_argument("--output-root", default="outputs/post_resume_analysis")
    parser.add_argument("--data-glob", default="data/curated_hf_shards/validation/*.parquet")
    parser.add_argument("--config", default="config/train.parameter_golf_random_order_dense.yaml")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--simplex-samples", type=int, default=8)
    parser.add_argument("--geometry-records", type=int, default=4)
    parser.add_argument("--geometry-branches", type=int, default=6)
    parser.add_argument("--seed", type=int, default=10017)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="fp32", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def latest_checkpoint(checkpoint_dir: Path, min_step: int) -> tuple[int, Path] | None:
    best: tuple[int, Path] | None = None
    for path in checkpoint_dir.glob("random_order_step_*.pt"):
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_synopsis(base: Path, checkpoint: Path, step: int, run_path: str) -> Path:
    metrics_dir = base / "metrics"
    simplex_dir = base / "simplex"
    geometry_dir = base / "geometry"
    category = load_json(metrics_dir / "category_summary.json")
    geometry = load_json(geometry_dir / "reasoning_geometry_summary.json")
    simplex = load_json(simplex_dir / "reasoning_simplex_summary.json")
    checkpoint_meta = load_json(metrics_dir / "checkpoint_meta.json")
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
            "",
            "## Initial Interpretation",
            "",
            "Treat deterministic validation BPB as the checkpoint gate.  The",
            "GFlowNet and score-first values are test-time-scaling diagnostics,",
            "not promotion metrics.  Geometry is considered useful only when lower",
            "BPB or lower answer-span BPB accompanies higher MST efficiency, lower",
            "curvature, or healthier toric entropy.",
        ]
    )
    out = base / "SYNOPSIS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    repo = Path.cwd()
    checkpoint_dir = Path(args.checkpoint_dir)
    target_step = args.start_step + args.after_steps
    print(f"waiting for checkpoint >= {target_step} in {checkpoint_dir}", flush=True)
    found: tuple[int, Path] | None = None
    while found is None:
        found = latest_checkpoint(checkpoint_dir, target_step)
        if found is None:
            time.sleep(max(1.0, args.poll_seconds))
    step, checkpoint = found
    base = Path(args.output_root) / f"step-{step:08d}"
    base.mkdir(parents=True, exist_ok=True)
    print(f"analyzing checkpoint {checkpoint} at step {step}", flush=True)

    if args.run_path:
        run_command(
            [
                "python",
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
    run_command(
        [
            "python",
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
    run_command(
        [
            "python",
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
    synopsis = write_synopsis(base, checkpoint, step, args.run_path)
    print(f"wrote {synopsis}", flush=True)


if __name__ == "__main__":
    main()
