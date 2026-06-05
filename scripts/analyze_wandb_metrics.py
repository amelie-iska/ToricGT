#!/usr/bin/env python3
# Example:
# conda run --no-capture-output -n tokengt env PYTHONPATH=src \
#   python scripts/analyze_wandb_metrics.py \
#   --run-path amelie-iska-math/toricgt-parameter-golf/8v4wnovf \
#   --checkpoint checkpoints/parameter_golf_oai_dense/random_order_step_00014750.pt \
#   --output-dir outputs/metrics_analysis/oai-8v4wnovf-step14750
"""Export and statistically analyze a ToricGT W&B run.

The analysis is deliberately lightweight and reproducible: it uses ordinary
least-squares trend estimates, rank correlations, median-window improvement,
and median-absolute-deviation spike detection.  The resulting CSV/JSON/PNG
outputs are intended to back the human metrics plan in planning/METRICS.md.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


CATEGORY_DESIRED = "as_desired"
CATEGORY_SLOW = "as_desired_but_not_strong_or_fast_enough"
CATEGORY_BAD = "not_as_desired"

DARK_BG = "#030712"
DARK_PANEL = "#06111f"
DARK_TEXT = "#e8fbff"
DARK_MUTED = "#9fb6c5"
DARK_GRID = "#143344"
DARK_SPINE = "#29536a"


@dataclass
class MetricStats:
    metric: str
    goal: str
    category: str
    n: int
    first_step: float
    last_step: float
    first_median: float
    last_median: float
    last_value: float
    best_value: float
    slope_per_1k: float
    slope_t: float
    recent_slope_per_1k: float
    recent_slope_t: float
    spearman_r: float
    relative_change: float
    spike_count: int
    max_abs_robust_zdiff: float
    cv: float
    rationale: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-path", required=True, help="entity/project/run_id")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default="outputs/metrics_analysis")
    parser.add_argument("--recent-fraction", type=float, default=0.3)
    parser.add_argument("--window-fraction", type=float, default=0.1)
    parser.add_argument("--max-plot-metrics", type=int, default=28)
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def finite_series(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = df[["_step", metric]].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.rename(columns={metric: "value"})
    out = out.sort_values("_step")
    return out


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def ols_slope_stats(steps: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    if len(values) < 3:
        return float("nan"), float("nan")
    x = (steps.astype(float) - float(np.min(steps))) / 1000.0
    y = values.astype(float)
    x_center = x - x.mean()
    sxx = float(np.sum(x_center * x_center))
    if sxx <= 0:
        return float("nan"), float("nan")
    slope = float(np.sum(x_center * (y - y.mean())) / sxx)
    intercept = float(y.mean() - slope * x.mean())
    resid = y - (intercept + slope * x)
    dof = max(1, len(y) - 2)
    sigma2 = float(np.sum(resid * resid) / dof)
    se = math.sqrt(max(0.0, sigma2 / sxx))
    t_stat = slope / se if se > 0 else float("inf") if slope != 0 else 0.0
    return slope, float(t_stat)


def metric_goal(name: str) -> str:
    lower_tokens = (
        "loss",
        "bpb",
        "error",
        "ncd",
        "bias_norm",
        "dissipation",
        "dominant_abs_curvature",
        "trace_abs_per_param",
        "hvp_norm",
        "probe_grad_norm",
        "sharpness",
        "qat_loss",
        "contrastive_loss",
        "trajectory_flow_loss",
        "future_permutation",
    )
    higher_tokens = ("accuracy", "diversity", "entropy")
    stable_prefixes = ("artifact/", "model/", "data/", "system/")
    stable_tokens = (
        "lr",
        "samples",
        "parameters",
        "bytes",
        "bits",
        "input_k_",
        "target_k_",
        "order_program_k_",
        "gflownet_action_trace_k_",
        "smear_temperature",
        "trajectory_kinetic_energy",
        "vram",
    )
    if "runtime_scale" in name:
        return "higher"
    if name.startswith(stable_prefixes) or any(token in name for token in stable_tokens):
        return "stable"
    if any(token in name for token in higher_tokens):
        return "higher"
    if any(token in name for token in lower_tokens):
        return "lower"
    if name.endswith("_weight"):
        return "stable"
    return "stable"


def robust_spikes(values: np.ndarray) -> tuple[int, float]:
    if len(values) < 5:
        return 0, 0.0
    diffs = np.diff(values.astype(float))
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med)))
    if mad <= 1e-12:
        return 0, 0.0
    z = 0.6745 * (diffs - med) / mad
    return int(np.sum(np.abs(z) > 6.0)), float(np.max(np.abs(z)))


def median_window(values: np.ndarray, fraction: float) -> tuple[float, float]:
    n = len(values)
    width = max(3, int(math.ceil(n * fraction)))
    width = min(width, n)
    return float(np.median(values[:width])), float(np.median(values[-width:]))


def categorize(goal: str, relative_change: float, recent_slope: float, recent_t: float, spike_count: int, cv: float) -> tuple[str, str]:
    if goal == "lower":
        if abs(relative_change) <= 1e-12 and abs(recent_slope) <= 1e-12:
            return CATEGORY_DESIRED, "lower-is-better metric is already at a stable floor"
        if abs(relative_change) < 10.0 and cv <= 2.0 and abs(recent_slope) < 1.0e-6:
            return CATEGORY_DESIRED, "lower-is-better metric is numerically near zero during a bounded ramp"
        if relative_change <= -0.03 and (recent_slope <= 0 or abs(recent_t) < 1.5) and spike_count <= 3:
            return CATEGORY_DESIRED, "lower-is-better metric improved materially without recent positive drift"
        if relative_change <= 0.01 and spike_count <= 6:
            return CATEGORY_SLOW, "lower-is-better metric is improving, flat, or noisy but not deteriorating strongly"
        return CATEGORY_BAD, "lower-is-better metric deteriorated, drifted upward, or showed repeated spikes"
    if goal == "higher":
        if relative_change >= 0.03 and (recent_slope >= 0 or abs(recent_t) < 1.5) and spike_count <= 3:
            return CATEGORY_DESIRED, "higher-is-better metric improved materially without recent negative drift"
        if relative_change >= -0.01 and spike_count <= 6:
            return CATEGORY_SLOW, "higher-is-better metric is improving, flat, or noisy but not strengthening quickly"
        return CATEGORY_BAD, "higher-is-better metric deteriorated or showed repeated spikes"
    if cv <= 0.02:
        return CATEGORY_DESIRED, "stability metric stayed essentially constant"
    if cv <= 0.08 and spike_count <= 3:
        return CATEGORY_DESIRED, "stability metric stayed tightly bounded"
    if cv <= 0.25 and spike_count <= 6:
        return CATEGORY_SLOW, "stability metric stayed usable but moved enough to monitor"
    return CATEGORY_BAD, "stability metric moved too much for a nominally stable diagnostic"


def analyze_metric(df: pd.DataFrame, metric: str, recent_fraction: float, window_fraction: float) -> MetricStats | None:
    series = finite_series(df, metric)
    if len(series) < 2:
        return None
    steps = series["_step"].to_numpy(dtype=float)
    values = series["value"].to_numpy(dtype=float)
    goal = metric_goal(metric)
    first_med, last_med = median_window(values, window_fraction)
    denom = max(abs(first_med), 1e-9)
    relative_change = (last_med - first_med) / denom
    if goal == "lower":
        best_value = float(np.min(values))
    elif goal == "higher":
        best_value = float(np.max(values))
    else:
        best_value = float(values[-1])
    slope, slope_t = ols_slope_stats(steps, values)
    recent_n = max(3, int(math.ceil(len(values) * recent_fraction)))
    recent_slope, recent_slope_t = ols_slope_stats(steps[-recent_n:], values[-recent_n:])
    spike_count, max_abs_z = robust_spikes(values)
    mean_abs = float(abs(np.mean(values)))
    cv = float(np.std(values) / max(mean_abs, 1e-9))
    category, rationale = categorize(goal, relative_change, recent_slope, recent_slope_t, spike_count, cv)
    return MetricStats(
        metric=metric,
        goal=goal,
        category=category,
        n=int(len(values)),
        first_step=float(steps[0]),
        last_step=float(steps[-1]),
        first_median=float(first_med),
        last_median=float(last_med),
        last_value=float(values[-1]),
        best_value=float(best_value),
        slope_per_1k=float(slope),
        slope_t=float(slope_t),
        recent_slope_per_1k=float(recent_slope),
        recent_slope_t=float(recent_slope_t),
        spearman_r=spearman(steps, values),
        relative_change=float(relative_change),
        spike_count=spike_count,
        max_abs_robust_zdiff=max_abs_z,
        cv=cv,
        rationale=rationale,
    )


def download_history(run_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    import wandb

    api = wandb.Api()
    requested_run_path = run_path
    try:
        run = api.run(run_path)
    except Exception:
        parts = run_path.split("/")
        if len(parts) != 3:
            raise
        entity, project, run_key = parts
        candidates = api.runs(f"{entity}/{project}", per_page=200)
        run = None
        for candidate in candidates:
            if candidate.id == run_key or candidate.name == run_key:
                run = candidate
                break
        if run is None:
            raise
        run_path = f"{entity}/{project}/{run.id}"
    rows = list(run.scan_history(page_size=1000))
    if not rows:
        raise RuntimeError(f"No W&B history rows found for {run_path}")
    df = pd.DataFrame(rows)
    if "_step" not in df.columns:
        raise RuntimeError("W&B history does not contain _step")
    meta = {
        "run_path": run_path,
        "requested_run_path": requested_run_path,
        "run_id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "last_history_step": run.lastHistoryStep,
        "summary": {k: v for k, v in dict(run.summary).items() if not str(k).startswith("_")},
    }
    return df, meta


def load_checkpoint_meta(path: str) -> dict[str, Any]:
    if not path:
        return {}
    checkpoint = Path(path)
    if not checkpoint.exists():
        return {"checkpoint": path, "exists": False}
    payload = torch.load(checkpoint, map_location="cpu")
    model_tensors = payload.get("model", {})
    parameter_count = int(sum(t.numel() for t in model_tensors.values() if torch.is_tensor(t)))
    return {
        "checkpoint": str(checkpoint),
        "exists": True,
        "step": int(payload.get("step", -1)),
        "model_type": payload.get("model_type", ""),
        "parameter_count": parameter_count,
        "metrics": payload.get("metrics", {}),
        "config": payload.get("config", {}),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        json.dumps(value)
        return value
    except TypeError:
        try:
            return dict(value)
        except Exception:
            return str(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True), encoding="utf-8")


def apply_dark_plot_style(fig: plt.Figure, axes: Any, *, grid: bool = True) -> None:
    fig.patch.set_facecolor(DARK_BG)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(DARK_PANEL)
        ax.title.set_color(DARK_TEXT)
        ax.xaxis.label.set_color(DARK_TEXT)
        ax.yaxis.label.set_color(DARK_TEXT)
        ax.tick_params(colors=DARK_MUTED)
        for spine in ax.spines.values():
            spine.set_color(DARK_SPINE)
        if grid:
            ax.grid(True, color=DARK_GRID, alpha=0.45, linewidth=0.6)


def save_dark_figure(fig: plt.Figure, out: Path, *, dpi: int = 180) -> None:
    fig.savefig(out, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")


def plot_core_timeseries(df: pd.DataFrame, out: Path, stats: list[MetricStats]) -> None:
    preferred = [
        "00_primary/oai_bpb",
        "00_primary/oai_best_bpb",
        "00_primary/train_bpb",
        "00_primary/train_loss",
        "00_primary/validation_bpb",
        "00_primary/toric_cca_topology_loss",
        "02_train/bpb",
        "02_train/loss",
        "03_validation/bpb",
        "03_validation/loss",
        "04_losses/train/total_loss",
        "04_losses/train/advanced_aux_loss",
        "06_graphcg/advanced/graphcg_loss",
        "07_topology_geometry/advanced/analogy_loss",
        "08_toric_tropical_bgg/advanced/toric_cca_topology_loss",
        "08_toric_tropical_bgg/advanced/toric_cca_binomial_residual",
        "08_toric_tropical_bgg/advanced/toric_cca_stanley_reisner_nonface_mass",
        "08_toric_tropical_bgg/advanced/koszul_bgg_loss",
        "08_toric_tropical_bgg/advanced/slepian_pollak_loss",
        "08_toric_tropical_bgg/advanced/toric_tropical_loss",
        "12_optimization/advanced/runtime_scale",
        "12_optimization/advanced/aux_loss",
        "12_optimization/time/step_avg_ms",
        "train/bpb",
        "train/loss",
        "train/total_loss",
        "val/bpb",
        "val/loss",
        "audit/future_permutation_logit_error",
        "complexity/val/bpb",
        "complexity/val/argmax_byte_accuracy",
        "complexity/val/prediction_target_ncd_lzma_mean",
        "complexity/train/prediction_target_ncd_lzma_mean",
    ]
    present = [m for m in preferred if m in df.columns and finite_series(df, m).shape[0] >= 2]
    if not present:
        return
    rows = int(math.ceil(len(present) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(14, max(4, 2.8 * rows)), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    apply_dark_plot_style(fig, axes)
    category_map = {s.metric: s.category for s in stats}
    colors = {CATEGORY_DESIRED: "#2dd4bf", CATEGORY_SLOW: "#f59e0b", CATEGORY_BAD: "#ef4444"}
    for ax, metric in zip(axes, present):
        series = finite_series(df, metric)
        ax.plot(series["_step"], series["value"], color=colors.get(category_map.get(metric), "#60a5fa"), linewidth=1.5)
        if len(series) >= 5:
            smooth = series["value"].ewm(span=min(25, max(5, len(series) // 8)), adjust=False).mean()
            ax.plot(series["_step"], smooth, color=DARK_TEXT, linewidth=1.0, alpha=0.75)
        ax.set_title(metric, fontsize=9)
    for ax in axes[len(present):]:
        ax.axis("off")
    fig.suptitle("Core ToricGT Parameter-Golf Metrics", fontsize=14, color=DARK_TEXT)
    save_dark_figure(fig, out)
    plt.close(fig)


def plot_category_counts(stats: list[MetricStats], out: Path) -> None:
    counts = pd.Series([s.category for s in stats]).value_counts().reindex(
        [CATEGORY_DESIRED, CATEGORY_SLOW, CATEGORY_BAD], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    apply_dark_plot_style(fig, ax, grid=False)
    ax.bar(counts.index, counts.values, color=["#2dd4bf", "#f59e0b", "#ef4444"])
    ax.set_ylabel("metric count")
    ax.set_title("Metric Behavior Categories")
    ax.tick_params(axis="x", rotation=15)
    save_dark_figure(fig, out)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out: Path) -> None:
    candidates = [
        "00_primary/oai_bpb",
        "00_primary/train_bpb",
        "00_primary/toric_cca_topology_loss",
        "02_train/bpb",
        "02_train/loss",
        "03_validation/bpb",
        "04_losses/train/advanced_aux_loss",
        "06_graphcg/advanced/graphcg_loss",
        "06_graphcg/advanced/graphcg_offdiag_coherence",
        "07_topology_geometry/advanced/analogy_loss",
        "08_toric_tropical_bgg/advanced/toric_cca_topology_loss",
        "08_toric_tropical_bgg/advanced/toric_cca_binomial_residual",
        "08_toric_tropical_bgg/advanced/toric_cca_stanley_reisner_nonface_mass",
        "08_toric_tropical_bgg/advanced/koszul_bgg_loss",
        "08_toric_tropical_bgg/advanced/slepian_pollak_loss",
        "08_toric_tropical_bgg/advanced/toric_tropical_loss",
        "12_optimization/advanced/runtime_scale",
        "12_optimization/advanced/aux_loss",
        "train/bpb",
        "train/loss",
        "val/bpb",
        "val/loss",
        "complexity/val/argmax_byte_accuracy",
        "complexity/val/prediction_target_ncd_lzma_mean",
        "complexity/val/prediction_k_lzma_mean",
        "complexity/val/gflownet_action_trace_k_lzma_mean",
    ]
    present = [m for m in candidates if m in df.columns and finite_series(df, m).shape[0] >= 3]
    if len(present) < 2:
        return
    aligned = df[["_step", *present]].sort_values("_step")
    filled = aligned[present].interpolate(limit_direction="both")
    corr = filled.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(present)), max(7, 0.7 * len(present))), constrained_layout=True)
    apply_dark_plot_style(fig, ax, grid=False)
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(present)), present, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(present)), present, fontsize=8)
    ax.set_title("Spearman Correlation of Selected Metrics")
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors=DARK_MUTED)
    colorbar.outline.set_edgecolor(DARK_SPINE)
    save_dark_figure(fig, out)
    plt.close(fig)


def plot_metric_slopes(stats: list[MetricStats], out: Path, max_metrics: int) -> None:
    ranked = sorted(
        [s for s in stats if math.isfinite(s.recent_slope_per_1k)],
        key=lambda s: abs(s.recent_slope_t),
        reverse=True,
    )[:max_metrics]
    if not ranked:
        return
    labels = [s.metric for s in ranked]
    values = [s.recent_slope_per_1k for s in ranked]
    colors = [
        "#2dd4bf" if s.category == CATEGORY_DESIRED else "#f59e0b" if s.category == CATEGORY_SLOW else "#ef4444"
        for s in ranked
    ]
    fig, ax = plt.subplots(figsize=(11, max(6, 0.35 * len(ranked))), constrained_layout=True)
    apply_dark_plot_style(fig, ax)
    y = np.arange(len(ranked))
    ax.barh(y, values, color=colors)
    ax.axvline(0.0, color=DARK_TEXT, linewidth=0.8)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("recent OLS slope per 1k steps")
    ax.set_title("Most Statistically Active Recent Metric Slopes")
    save_dark_figure(fig, out)
    plt.close(fig)


def markdown_summary(stats: list[MetricStats], meta: dict[str, Any], checkpoint_meta: dict[str, Any]) -> str:
    counts = pd.Series([s.category for s in stats]).value_counts().reindex(
        [CATEGORY_DESIRED, CATEGORY_SLOW, CATEGORY_BAD], fill_value=0
    )
    core_names = [
        "00_primary/oai_bpb",
        "00_primary/oai_best_bpb",
        "00_primary/train_bpb",
        "00_primary/train_loss",
        "00_primary/validation_bpb",
        "00_primary/toric_cca_topology_loss",
        "02_train/bpb",
        "02_train/loss",
        "03_validation/bpb",
        "03_validation/loss",
        "04_losses/train/advanced_aux_loss",
        "06_graphcg/advanced/graphcg_loss",
        "07_topology_geometry/advanced/analogy_loss",
        "08_toric_tropical_bgg/advanced/toric_cca_topology_loss",
        "08_toric_tropical_bgg/advanced/toric_cca_binomial_residual",
        "08_toric_tropical_bgg/advanced/toric_cca_stanley_reisner_nonface_mass",
        "08_toric_tropical_bgg/advanced/koszul_bgg_loss",
        "08_toric_tropical_bgg/advanced/slepian_pollak_loss",
        "08_toric_tropical_bgg/advanced/toric_tropical_loss",
        "12_optimization/advanced/runtime_scale",
        "12_optimization/advanced/aux_loss",
        "12_optimization/time/step_avg_ms",
        "train/bpb",
        "train/loss",
        "train/total_loss",
        "val/bpb",
        "val/loss",
        "complexity/val/argmax_byte_accuracy",
        "complexity/val/prediction_target_ncd_lzma_mean",
        "audit/future_permutation_logit_error",
    ]
    by_name = {s.metric: s for s in stats}
    lines = [
        "# Automatic Metrics Analysis",
        "",
        f"- W&B run: `{meta.get('run_path')}`",
        f"- W&B URL: {meta.get('url')}",
        f"- W&B state at export: `{meta.get('state')}`",
        f"- Last W&B history step: `{meta.get('last_history_step')}`",
    ]
    if checkpoint_meta:
        lines.extend(
            [
                f"- Checkpoint: `{checkpoint_meta.get('checkpoint')}`",
                f"- Checkpoint step: `{checkpoint_meta.get('step')}`",
                f"- Checkpoint parameters: `{checkpoint_meta.get('parameter_count')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Category Counts",
            "",
            f"- `{CATEGORY_DESIRED}`: {int(counts[CATEGORY_DESIRED])}",
            f"- `{CATEGORY_SLOW}`: {int(counts[CATEGORY_SLOW])}",
            f"- `{CATEGORY_BAD}`: {int(counts[CATEGORY_BAD])}",
            "",
            "## Core Metrics",
            "",
            "| metric | category | goal | first median | last median | relative change | recent slope/1k | recent t | spikes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in core_names:
        if name not in by_name:
            continue
        s = by_name[name]
        lines.append(
            f"| `{s.metric}` | {s.category} | {s.goal} | {s.first_median:.6g} | {s.last_median:.6g} | "
            f"{100*s.relative_change:.2f}% | {s.recent_slope_per_1k:.6g} | {s.recent_slope_t:.3g} | {s.spike_count} |"
        )
    lines.extend(["", "See `metric_stats.csv` for the full per-metric table."])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, meta = download_history(args.run_path)
    df = df.sort_values("_step")
    df.to_csv(output_dir / "wandb_history.csv", index=False)
    try:
        df.to_parquet(output_dir / "wandb_history.parquet", index=False)
    except Exception:
        pass
    checkpoint_meta = load_checkpoint_meta(args.checkpoint)
    write_json(output_dir / "wandb_run_meta.json", meta)
    write_json(output_dir / "checkpoint_meta.json", checkpoint_meta)
    if args.download_only:
        return

    numeric_metrics = []
    for column in df.columns:
        if column.startswith("_") or column in {"epoch"}:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().sum() >= 2:
            df[column] = converted
            numeric_metrics.append(column)

    stats = [
        stat
        for metric in numeric_metrics
        if (stat := analyze_metric(df, metric, args.recent_fraction, args.window_fraction)) is not None
    ]
    stats_dicts = [asdict(s) for s in sorted(stats, key=lambda item: (item.category, item.metric))]
    pd.DataFrame(stats_dicts).to_csv(output_dir / "metric_stats.csv", index=False)
    write_json(output_dir / "metric_stats.json", stats_dicts)

    counts = pd.Series([s.category for s in stats]).value_counts().to_dict()
    write_json(
        output_dir / "category_summary.json",
        {
            "counts": {key: int(value) for key, value in counts.items()},
            "run": meta,
            "checkpoint": checkpoint_meta,
        },
    )

    plot_core_timeseries(df, output_dir / "core_metric_timeseries.png", stats)
    plot_category_counts(stats, output_dir / "metric_category_counts.png")
    plot_correlation_heatmap(df, output_dir / "selected_metric_correlations.png")
    plot_metric_slopes(stats, output_dir / "recent_metric_slopes.png", args.max_plot_metrics)
    (output_dir / "automatic_metrics_report.md").write_text(
        markdown_summary(stats, meta, checkpoint_meta), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
