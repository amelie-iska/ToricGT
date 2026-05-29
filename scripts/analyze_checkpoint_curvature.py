#!/usr/bin/env python3
# Example:
# conda run --no-capture-output -n tokengt env PYTHONPATH=src \
#   python scripts/analyze_checkpoint_curvature.py \
#   --checkpoint-dir checkpoints/parameter_golf_oai_dense \
#   --steps 1000 1250 1500 1750 2000 2250 \
#   --output-dir outputs/checkpoint_curvature/oai-early-elbow
"""Finite-difference curvature analysis for ToricGT checkpoints.

The script reads saved checkpoint metadata and computes first and second
finite differences for scalar training metrics.  This is intentionally separate
from stochastic Hessian-vector probes: checkpoint curvature measures the
observed loss/BPB trajectory over training time, while HVP probes estimate the
local parameter-space sharpness at an individual checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="checkpoints/parameter_golf_oai_dense")
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--metrics", nargs="+", default=["train_bpb", "train_loss", "best_val_bpb"])
    parser.add_argument("--output-dir", default="outputs/checkpoint_curvature")
    return parser.parse_args()


def checkpoint_path(checkpoint_dir: Path, step: int) -> Path:
    return checkpoint_dir / f"random_order_step_{step:08d}.pt"


def load_metric_row(checkpoint_dir: Path, step: int, metric_names: list[str]) -> dict[str, Any]:
    path = checkpoint_path(checkpoint_dir, step)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    metrics = payload.get("metrics", {}) or {}
    row: dict[str, Any] = {"step": int(payload.get("step", step)), "checkpoint": str(path)}
    for metric in metric_names:
        value = metrics.get(metric)
        row[metric] = None if value is None else float(value)
    return row


def finite_differences(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    out = []
    for left, right in zip(rows, rows[1:]):
        if left.get(metric) is None or right.get(metric) is None:
            continue
        step_delta = float(right["step"] - left["step"])
        value_delta = float(right[metric] - left[metric])
        out.append(
            {
                "metric": metric,
                "kind": "first_difference",
                "left_step": int(left["step"]),
                "center_step": None,
                "right_step": int(right["step"]),
                "delta": value_delta,
                "per_1k_steps": value_delta / step_delta * 1000.0 if step_delta else None,
            }
        )
    for left, center, right in zip(rows, rows[1:], rows[2:]):
        if left.get(metric) is None or center.get(metric) is None or right.get(metric) is None:
            continue
        left_h = float(center["step"] - left["step"])
        right_h = float(right["step"] - center["step"])
        if abs(left_h - right_h) > 1e-9:
            curvature = 2.0 * (
                ((right[metric] - center[metric]) / right_h)
                - ((center[metric] - left[metric]) / left_h)
            ) / (left_h + right_h)
            second_delta = curvature * ((left_h + right_h) * 0.5) ** 2
        else:
            second_delta = float(right[metric] - 2.0 * center[metric] + left[metric])
            curvature = second_delta / (left_h * left_h)
        out.append(
            {
                "metric": metric,
                "kind": "second_difference",
                "left_step": int(left["step"]),
                "center_step": int(center["step"]),
                "right_step": int(right["step"]),
                "delta": second_delta,
                "per_1k_steps": curvature * 1_000_000.0,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows: list[dict[str, Any]], metric_names: list[str], output_dir: Path) -> None:
    steps = [row["step"] for row in rows]
    for metric in metric_names:
        values = [row.get(metric) for row in rows]
        if any(value is None for value in values):
            continue
        plt.figure(figsize=(8, 4.8))
        plt.plot(steps, values, marker="o", color="#38bdf8", linewidth=2.0)
        plt.title(f"Checkpoint Trajectory: {metric}")
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_trajectory.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = sorted(set(int(step) for step in args.steps))
    metric_names = [str(metric) for metric in args.metrics]
    rows = [load_metric_row(checkpoint_dir, step, metric_names) for step in steps]
    diff_rows = []
    for metric in metric_names:
        diff_rows.extend(finite_differences(rows, metric))
    write_csv(output_dir / "checkpoint_metrics.csv", rows)
    write_csv(output_dir / "finite_differences.csv", diff_rows)
    (output_dir / "checkpoint_curvature.json").write_text(
        json.dumps({"checkpoints": rows, "finite_differences": diff_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_metrics(rows, metric_names, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "checkpoints": len(rows), "differences": len(diff_rows)}, indent=2))


if __name__ == "__main__":
    main()
