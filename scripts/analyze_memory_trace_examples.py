#!/usr/bin/env python3
"""Analyze analogical memory retrieval traces emitted by TokenGT training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPONENT_KEYS = (
    "graphcg_chart_similarity",
    "toric_phase_similarity",
    "topology_similarity",
    "dag_similarity",
    "derived_category_similarity",
    "candidate_quality_z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-dir", default="checkpoints/memory_trace_examples")
    parser.add_argument("--output-dir", default="outputs/memory_trace_analysis")
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--max-queries", type=int, default=96)
    parser.add_argument("--dark-mode", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_rows(example_dir: str | Path, *, max_files: int, max_queries: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = Path(example_dir)
    if not base.exists():
        return [], []
    paths = sorted(base.glob("step_*.json"))[-max(1, int(max_files)) :]
    query_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        step = int(payload.get("step", 0) or 0)
        trace = payload.get("memory_trace", {})
        if not isinstance(trace, dict):
            continue
        for query in trace.get("queries", []) or []:
            if not isinstance(query, dict):
                continue
            query_row = {
                "step": step,
                "source_file": str(path),
                "query_index": int(query.get("query_index", 0) or 0),
                "teacher_argmax_index": int(query.get("teacher_argmax_index", -1) or -1),
                "retrieval_entropy": float(query.get("retrieval_entropy", 0.0) or 0.0),
                "query_quality_z": float(query.get("quality_z", 0.0) or 0.0),
            }
            dag = query.get("dag_features", {}) if isinstance(query.get("dag_features", {}), dict) else {}
            derived = (
                query.get("derived_category_features", {})
                if isinstance(query.get("derived_category_features", {}), dict)
                else {}
            )
            for key, value in dag.items():
                query_row[f"dag/{key}"] = float(value or 0.0)
            for key, value in derived.items():
                query_row[f"derived/{key}"] = float(value or 0.0)
            query_rows.append(query_row)
            for candidate in query.get("top_candidates", []) or []:
                if not isinstance(candidate, dict):
                    continue
                row = dict(query_row)
                row.update(
                    {
                        "rank": int(candidate.get("rank", 0) or 0),
                        "candidate_index": int(candidate.get("candidate_index", -1) or -1),
                        "model_logit": float(candidate.get("model_logit", 0.0) or 0.0),
                        "retrieval_probability": float(candidate.get("retrieval_probability", 0.0) or 0.0),
                        "teacher_probability": float(candidate.get("teacher_probability", 0.0) or 0.0),
                        "teacher_raw_score": float(candidate.get("teacher_raw_score", 0.0) or 0.0),
                        "is_teacher_argmax": float(bool(candidate.get("is_teacher_argmax", False))),
                    }
                )
                components = candidate.get("components", {}) if isinstance(candidate.get("components", {}), dict) else {}
                for key in COMPONENT_KEYS:
                    row[f"component/{key}"] = float(components.get(key, 0.0) or 0.0)
                candidate_rows.append(row)
            if len(query_rows) >= max_queries:
                return query_rows, candidate_rows
    return query_rows, candidate_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def set_dark(ax: plt.Axes) -> None:
    ax.set_facecolor("#0b1020")
    ax.tick_params(colors="#e5e7eb")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.title.set_color("#f8fafc")
    ax.xaxis.label.set_color("#e5e7eb")
    ax.yaxis.label.set_color("#e5e7eb")


def plot_probability_alignment(rows: list[dict[str, Any]], out: Path) -> None:
    if not rows:
        return
    x = np.asarray([row["teacher_probability"] for row in rows], dtype=float)
    y = np.asarray([row["retrieval_probability"] for row in rows], dtype=float)
    colors = np.asarray([row["is_teacher_argmax"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor="#020617")
    set_dark(ax)
    sc = ax.scatter(x, y, c=colors, cmap="cool", s=28, alpha=0.82, edgecolor="#06111f", linewidth=0.25)
    lim = max(float(np.nanmax([x.max(initial=0.0), y.max(initial=0.0)])), 1e-3)
    ax.plot([0, lim], [0, lim], color="#e5e7eb", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.set_xlabel("teacher probability")
    ax.set_ylabel("model retrieval probability")
    ax.set_title("Analogical Memory Retrieval: Model vs Teacher")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("teacher argmax candidate", color="#e5e7eb")
    cbar.ax.tick_params(colors="#e5e7eb")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_component_boxplot(rows: list[dict[str, Any]], out: Path) -> None:
    if not rows:
        return
    data = [np.asarray([row[f"component/{key}"] for row in rows], dtype=float) for key in COMPONENT_KEYS]
    labels = [key.replace("_similarity", "").replace("_", "\n") for key in COMPONENT_KEYS]
    fig, ax = plt.subplots(figsize=(10.5, 5.4), facecolor="#020617")
    set_dark(ax)
    box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    colors = ["#38bdf8", "#22c55e", "#a78bfa", "#f59e0b", "#f472b6", "#f8fafc"]
    for patch, color in zip(box["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for median in box["medians"]:
        median.set_color("#f8fafc")
    ax.set_title("Top-Candidate Analogical Similarity Components")
    ax.grid(axis="y", color="#1e293b", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_query_features(rows: list[dict[str, Any]], out: Path) -> None:
    if not rows:
        return
    steps = np.asarray([row["step"] for row in rows], dtype=float)
    entropy = np.asarray([row["retrieval_entropy"] for row in rows], dtype=float)
    branch = np.asarray([row.get("dag/branch_count", 0.0) for row in rows], dtype=float)
    merge = np.asarray([row.get("dag/merge_count", 0.0) for row in rows], dtype=float)
    pdim = np.asarray([row.get("derived/projective_dimension_norm", 0.0) for row in rows], dtype=float)
    regularity = np.asarray([row.get("derived/regularity_norm", 0.0) for row in rows], dtype=float)
    x = np.arange(len(rows)) if np.unique(steps).size <= 1 else steps
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), facecolor="#020617", sharex=True)
    for ax in axes:
        set_dark(ax)
        ax.grid(color="#1e293b", alpha=0.65)
    axes[0].plot(x, entropy, color="#38bdf8", marker="o", linewidth=1.4, label="retrieval entropy")
    axes[0].plot(x, branch, color="#ff4fd8", marker=".", linewidth=1.0, label="branch count")
    axes[0].plot(x, merge, color="#8cff6a", marker=".", linewidth=1.0, label="merge count")
    axes[0].legend(facecolor="#0b1020", edgecolor="#334155", labelcolor="#e5e7eb")
    axes[0].set_ylabel("trace / DAG")
    axes[1].plot(x, pdim, color="#ffd166", marker="o", linewidth=1.2, label="projective dimension norm")
    axes[1].plot(x, regularity, color="#a78bfa", marker="o", linewidth=1.2, label="regularity norm")
    axes[1].legend(facecolor="#0b1020", edgecolor="#334155", labelcolor="#e5e7eb")
    axes[1].set_ylabel("derived category")
    axes[1].set_xlabel("query index or checkpoint step")
    fig.suptitle("Query-Level DAG and Derived-Category Memory Features", color="#f8fafc")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def summarize(query_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rank1 = [row for row in candidate_rows if int(row.get("rank", 0)) == 1]

    def mean(rows: list[dict[str, Any]], key: str) -> float:
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        return float(np.mean(values)) if values else 0.0

    return {
        "trace_file_count": len({row["source_file"] for row in query_rows}),
        "query_count": len(query_rows),
        "candidate_count": len(candidate_rows),
        "steps": sorted({int(row["step"]) for row in query_rows}),
        "rank1_teacher_argmax_rate": mean(rank1, "is_teacher_argmax"),
        "mean_retrieval_entropy": mean(query_rows, "retrieval_entropy"),
        "mean_rank1_retrieval_probability": mean(rank1, "retrieval_probability"),
        "mean_rank1_teacher_probability": mean(rank1, "teacher_probability"),
        "mean_dag_branch_count": mean(query_rows, "dag/branch_count"),
        "mean_dag_merge_count": mean(query_rows, "dag/merge_count"),
        "mean_derived_projective_dimension_norm": mean(query_rows, "derived/projective_dimension_norm"),
        "mean_derived_regularity_norm": mean(query_rows, "derived/regularity_norm"),
        "mean_rank1_derived_category_similarity": mean(rank1, "component/derived_category_similarity"),
        "mean_rank1_dag_similarity": mean(rank1, "component/dag_similarity"),
        "mean_rank1_topology_similarity": mean(rank1, "component/topology_similarity"),
    }


def write_report(path: Path, summary: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Analogical Memory Trace Analysis",
        "",
        f"- Trace files: `{summary.get('trace_file_count', 0)}`",
        f"- Queries: `{summary.get('query_count', 0)}`",
        f"- Candidates: `{summary.get('candidate_count', 0)}`",
        f"- Steps: `{summary.get('steps', [])}`",
        f"- Rank-1 teacher-argmax agreement: `{summary.get('rank1_teacher_argmax_rate', 0.0):.4f}`",
        f"- Mean retrieval entropy: `{summary.get('mean_retrieval_entropy', 0.0):.4f}`",
        f"- Mean rank-1 model retrieval probability: `{summary.get('mean_rank1_retrieval_probability', 0.0):.4f}`",
        f"- Mean rank-1 teacher probability: `{summary.get('mean_rank1_teacher_probability', 0.0):.4f}`",
        f"- Mean branch / merge counts: `{summary.get('mean_dag_branch_count', 0.0):.4f}` / `{summary.get('mean_dag_merge_count', 0.0):.4f}`",
        f"- Mean derived projective-dimension norm: `{summary.get('mean_derived_projective_dimension_norm', 0.0):.4f}`",
        f"- Mean derived regularity norm: `{summary.get('mean_derived_regularity_norm', 0.0):.4f}`",
        f"- Mean rank-1 derived-category similarity: `{summary.get('mean_rank1_derived_category_similarity', 0.0):.4f}`",
        "",
        "## Files",
        "",
        f"- Candidate CSV: `{output_dir / 'memory_trace_candidates.csv'}`",
        f"- Query CSV: `{output_dir / 'memory_trace_queries.csv'}`",
        f"- Model/teacher probability plot: `{output_dir / 'memory_probability_alignment.png'}`",
        f"- Component boxplot: `{output_dir / 'memory_component_boxplot.png'}`",
        f"- Query feature trends: `{output_dir / 'memory_query_features.png'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    query_rows, candidate_rows = read_rows(args.example_dir, max_files=args.max_files, max_queries=args.max_queries)
    write_csv(out / "memory_trace_queries.csv", query_rows)
    write_csv(out / "memory_trace_candidates.csv", candidate_rows)
    summary = summarize(query_rows, candidate_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_probability_alignment(candidate_rows, out / "memory_probability_alignment.png")
    plot_component_boxplot(candidate_rows, out / "memory_component_boxplot.png")
    plot_query_features(query_rows, out / "memory_query_features.png")
    write_report(out / "REPORT.md", summary, out)


if __name__ == "__main__":
    main()
