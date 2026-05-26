#!/usr/bin/env python3
"""Project Parameter-Golf loss/BPB at an arbitrary step from early history.

The projection is intentionally conservative.  It fits several monotone
decreasing curves to the first `--fit-through-step` steps, combines them by
recent-fit error, and widens uncertainty as the requested step moves farther
from the observed fit window.  The target can be 50k or any earlier checkpoint
such as 5k, 10k, or 25k.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProjectionResult:
    metric: str
    mode: str
    observed_step_min: int
    observed_step_max: int
    observed_points: int
    target_step: int
    current_value: float
    projected_value: float
    lower: float
    upper: float
    model_weights: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-run", help="W&B run path, e.g. entity/project/run_id")
    parser.add_argument("--wandb-project", default="amelie-iska-math/toricgt-parameter-golf")
    parser.add_argument("--wandb-run-id", help="Run id only; combined with --wandb-project")
    parser.add_argument("--csv", help="CSV fallback with step and metric columns")
    parser.add_argument("--jsonl", help="JSONL fallback with step and metric fields")
    parser.add_argument("--output-dir", default="outputs/projections")
    parser.add_argument("--metrics", nargs="+", default=["train/loss", "train/bpb"])
    parser.add_argument("--step-key", default="_step")
    parser.add_argument("--target-step", type=int, default=50_000, help="Step to estimate, e.g. 5000, 10000, 50000")
    parser.add_argument(
        "--fit-through-step",
        "--max-observed-step",
        dest="max_observed_step",
        type=int,
        default=2_000,
        help="Only use history at or before this step for fitting. The old --max-observed-step alias still works.",
    )
    parser.add_argument("--min-points", type=int, default=25)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wandb-page-size", type=int, default=10_000)
    return parser.parse_args()


def wandb_history(run_path: str, metrics: list[str], page_size: int) -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    keys = ["_step", *metrics]
    rows = []
    for row in run.scan_history(keys=keys, page_size=page_size):
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no W&B history rows found for {run_path}")
    return pd.DataFrame(rows)


def local_history(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        return pd.read_csv(args.csv)
    if args.jsonl:
        rows = []
        with open(args.jsonl, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError("provide --wandb-run/--wandb-run-id, --csv, or --jsonl")


def load_history(args: argparse.Namespace) -> pd.DataFrame:
    run_path = args.wandb_run
    if not run_path and args.wandb_run_id:
        run_path = f"{args.wandb_project}/{args.wandb_run_id}"
    if run_path:
        return wandb_history(run_path, args.metrics, page_size=args.wandb_page_size)
    return local_history(args)


def clean_metric_frame(
    history: pd.DataFrame,
    metric: str,
    step_key: str,
    max_observed_step: int,
) -> pd.DataFrame:
    if step_key not in history.columns:
        raise KeyError(f"step key {step_key!r} not in history columns")
    if metric not in history.columns:
        raise KeyError(f"metric {metric!r} not in history columns")
    frame = history[[step_key, metric]].dropna().copy()
    frame.columns = ["step", "value"]
    frame["step"] = frame["step"].astype(int)
    frame["value"] = frame["value"].astype(float)
    frame = frame[(frame["step"] > 0) & (frame["step"] <= max_observed_step)]
    frame = frame[np.isfinite(frame["value"])]
    frame = frame.sort_values("step").drop_duplicates("step", keep="last")
    if frame.empty:
        raise ValueError(f"no usable points for {metric} through step {max_observed_step}")
    return frame


def design_matrix(step: np.ndarray, kind: str, target_step: int) -> np.ndarray:
    x = step.astype(np.float64)
    if kind == "log":
        return np.column_stack([np.ones_like(x), np.log1p(x)])
    if kind == "sqrt":
        return np.column_stack([np.ones_like(x), np.sqrt(x)])
    if kind == "power":
        return np.column_stack([np.ones_like(x), np.log1p(x), 1.0 / np.sqrt(x + 1.0)])
    if kind == "late_log":
        knot = min(2_000, max(10, target_step // 25))
        return np.column_stack([np.ones_like(x), np.log1p(x), np.log1p(np.maximum(x - knot, 0.0))])
    raise ValueError(kind)


def fit_curve(step: np.ndarray, value: np.ndarray, kind: str, target_step: int) -> tuple[np.ndarray, float]:
    matrix = design_matrix(step, kind, target_step)
    coef, *_ = np.linalg.lstsq(matrix, value, rcond=None)
    pred = matrix @ coef
    tail = max(5, len(step) // 3)
    rmse = float(np.sqrt(np.mean((pred[-tail:] - value[-tail:]) ** 2)))
    return coef, rmse


def predict_curve(step: np.ndarray, coef: np.ndarray, kind: str, target_step: int) -> np.ndarray:
    return design_matrix(step, kind, target_step) @ coef


def enforce_monotone_projection(
    pred: np.ndarray,
    observed_last: float,
    floor: float,
) -> np.ndarray:
    out = pred.copy()
    out[0] = min(out[0], observed_last)
    for idx in range(1, len(out)):
        out[idx] = min(out[idx], out[idx - 1])
    return np.maximum(out, floor)


def enforce_monotone_series(pred: np.ndarray, floor: float) -> np.ndarray:
    out = pred.copy()
    for idx in range(1, len(out)):
        out[idx] = min(out[idx], out[idx - 1])
    return np.maximum(out, floor)


def project_metric(
    frame: pd.DataFrame,
    metric: str,
    target_step: int,
    samples: int,
    seed: int,
) -> tuple[ProjectionResult, pd.DataFrame]:
    observed_steps = frame["step"].to_numpy(dtype=np.float64)
    observed_values = frame["value"].to_numpy(dtype=np.float64)
    current_step = int(observed_steps[-1])
    current_value = float(observed_values[-1])
    kinds = ["log", "sqrt", "power", "late_log"]
    coefs: dict[str, np.ndarray] = {}
    errors: dict[str, float] = {}
    for kind in kinds:
        coef, rmse = fit_curve(observed_steps, observed_values, kind, target_step)
        coefs[kind] = coef
        errors[kind] = max(rmse, 1e-5)
    inv_error = np.array([1.0 / errors[kind] for kind in kinds])
    weights = inv_error / inv_error.sum()
    floor = 0.25 if "bpb" in metric.lower() else 0.1
    if target_step >= current_step:
        mode = "forward_projection"
        horizon_steps = np.arange(current_step, target_step + 1, dtype=np.float64)
        model_preds = np.stack(
            [
                enforce_monotone_projection(
                    predict_curve(horizon_steps, coefs[kind], kind, target_step),
                    current_value,
                    floor=floor,
                )
                for kind in kinds
            ],
            axis=0,
        )
        distance_scale = max(1.0, target_step / max(current_step, 1))
        uncertainty_cap = current_value
    else:
        mode = "within_fit_window_estimate"
        start_step = max(1, min(int(observed_steps[0]), target_step))
        horizon_steps = np.arange(start_step, current_step + 1, dtype=np.float64)
        model_preds = np.stack(
            [
                enforce_monotone_series(
                    predict_curve(horizon_steps, coefs[kind], kind, target_step),
                    floor=floor,
                )
                for kind in kinds
            ],
            axis=0,
        )
        distance_scale = max(1.0, current_step / max(target_step, 1))
        uncertainty_cap = max(float(np.max(observed_values)), current_value)
    mean_path = (weights[:, None] * model_preds).sum(axis=0)
    model_spread = np.sqrt((weights[:, None] * (model_preds - mean_path[None, :]) ** 2).sum(axis=0))
    tail_values = observed_values[-max(5, len(observed_values) // 3) :]
    tail_noise = float(np.std(np.diff(tail_values))) if len(tail_values) > 2 else float(np.std(observed_values))
    tail_noise = max(tail_noise, 1e-5)
    rng = np.random.default_rng(seed)
    if target_step >= current_step:
        progress = (horizon_steps - current_step) / max(1.0, target_step - current_step)
        target_index = -1
    else:
        progress = np.abs(horizon_steps - target_step) / max(1.0, current_step - target_step)
        target_index = int(np.argmin(np.abs(horizon_steps - target_step)))
    widening = np.sqrt(progress) * (1.0 + 1.25 * progress)
    # Extra width accounts for the assumption that early improvements slow down
    # after step ~2K, while still allowing surprise late gains.
    sigma_path = model_spread + tail_noise * widening * math.sqrt(distance_scale)
    draws = rng.normal(loc=mean_path[target_index], scale=max(float(sigma_path[target_index]), 1e-5), size=samples)
    draws = np.minimum(draws, uncertainty_cap)
    draws = np.maximum(draws, floor)
    lower, upper = np.quantile(draws, [0.10, 0.90])
    projected = float(mean_path[target_index])
    result = ProjectionResult(
        metric=metric,
        mode=mode,
        observed_step_min=int(observed_steps[0]),
        observed_step_max=current_step,
        observed_points=len(frame),
        target_step=target_step,
        current_value=current_value,
        projected_value=projected,
        lower=float(lower),
        upper=float(upper),
        model_weights={kind: float(weight) for kind, weight in zip(kinds, weights)},
    )
    path = pd.DataFrame(
        {
            "step": horizon_steps.astype(int),
            "mean": mean_path,
            "lower": np.maximum(floor, mean_path - sigma_path),
            "upper": np.minimum(uncertainty_cap, mean_path + sigma_path),
        }
    )
    return result, path


def safe_name(metric: str) -> str:
    return metric.replace("/", "_").replace(" ", "_").replace(":", "_")


def plot_projection(
    frame: pd.DataFrame,
    path: pd.DataFrame,
    result: ProjectionResult,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#05070d")
    ax.set_facecolor("#05070d")
    ax.plot(frame["step"], frame["value"], color="#50f5ff", linewidth=1.7, label="observed")
    ax.plot(path["step"], path["mean"], color="#8cff6a", linewidth=1.9, label="projection")
    ax.fill_between(path["step"], path["lower"], path["upper"], color="#8cff6a", alpha=0.22, label="adaptive band")
    ax.scatter([result.target_step], [result.projected_value], color="#ffdf5d", s=80, edgecolor="white", zorder=5)
    ax.set_title(f"{result.metric} projection to step {result.target_step:,}")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(result.metric)
    ax.grid(color="#16333d", alpha=0.65)
    for spine in ax.spines.values():
        spine.set_color("#3cf4ff")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color("white")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.target_step <= 0:
        raise ValueError("--target-step must be positive")
    if args.max_observed_step <= 0:
        raise ValueError("--fit-through-step must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = load_history(args)
    results = []
    for metric in args.metrics:
        frame = clean_metric_frame(history, metric, args.step_key, args.max_observed_step)
        if len(frame) < args.min_points:
            raise RuntimeError(f"{metric} has only {len(frame)} points; need at least {args.min_points}")
        result, path = project_metric(
            frame=frame,
            metric=metric,
            target_step=args.target_step,
            samples=args.samples,
            seed=args.seed,
        )
        results.append(result)
        name = safe_name(metric)
        path.to_csv(output_dir / f"{name}_projection_path.csv", index=False)
        plot_projection(frame, path, result, output_dir / f"{name}_projection.png")
    summary = {result.metric: asdict(result) for result in results}
    (output_dir / "projection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
