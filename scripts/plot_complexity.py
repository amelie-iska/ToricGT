#!/usr/bin/env python3
"""Plot Kolmogorov-style complexity summaries from W&B-exported JSON/CSV.

Use this after `scripts/evaluate_complexity.py` or after exporting W&B history.
The plots are intentionally simple: they make BPB/complexity and NCD/complexity
trends visible without changing training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", help="JSON from scripts/evaluate_complexity.py")
    parser.add_argument("--history-csv", help="CSV exported from W&B history")
    parser.add_argument("--output-dir", default="outputs/complexity/plots")
    return parser.parse_args()


def dark_axes(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#05070d")
    ax.set_facecolor("#05070d")
    ax.set_title(title, color="white")
    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#17323d", alpha=0.65)
    for spine in ax.spines.values():
        spine.set_color("#3cf4ff")
    return fig, ax


def plot_summary(path: Path, output_dir: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = sorted((key, value) for key, value in data.items() if key.startswith("complexity/") and key.endswith("_mean"))
    labels = [key.replace("complexity/", "").replace("_mean", "") for key, _ in items[:25]]
    values = [float(value) for _, value in items[:25]]
    fig, ax = dark_axes("Kolmogorov-style complexity summary", "metric", "mean value")
    ax.bar(range(len(values)), values, color="#50f5ff")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", color="white", fontsize=7)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "complexity_summary_bars.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_history(path: Path, output_dir: Path) -> None:
    frame = pd.read_csv(path)
    step = frame["_step"] if "_step" in frame.columns else frame.index
    candidates = [
        "train/bpb",
        "val/bpb",
        "complexity/train/target_cond_k_lzma_mean",
        "complexity/val/target_cond_k_lzma_mean",
        "complexity/train/prediction_target_ncd_lzma_mean",
    ]
    present = [column for column in candidates if column in frame.columns]
    if not present:
        return
    fig, ax = dark_axes("BPB and reasoning-complexity history", "step", "value")
    colors = ["#50f5ff", "#ffdf5d", "#8cff6a", "#ff66c4", "#ba76ff"]
    for color, column in zip(colors, present):
        ax.plot(step, frame[column], linewidth=1.6, label=column, color=color)
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color("white")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "complexity_history.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.summary_json:
        plot_summary(Path(args.summary_json), output_dir)
    if args.history_csv:
        plot_history(Path(args.history_csv), output_dir)


if __name__ == "__main__":
    main()
