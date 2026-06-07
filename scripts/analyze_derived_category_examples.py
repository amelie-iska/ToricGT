#!/usr/bin/env python3
"""Analyze exact derived-category examples emitted by TokenGT training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-dir", default="checkpoints/derived_category_examples")
    parser.add_argument("--output-dir", default="outputs/derived_category_analysis")
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=24)
    parser.add_argument("--dark-mode", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return value or "derived-category-object"


def read_examples(example_dir: str | Path, *, max_files: int, max_objects: int) -> list[dict[str, Any]]:
    base = Path(example_dir)
    if not base.exists():
        return []
    paths = sorted(base.glob("step_*.json"))[-max(1, int(max_files)) :]
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        step = int(payload.get("step", 0) or 0)
        for obj in payload.get("objects", []) or []:
            if not isinstance(obj, dict):
                continue
            item = dict(obj)
            item["source_file"] = str(path)
            item["step"] = step
            records.append(item)
            if len(records) >= max_objects:
                return records
    return records


def _resolution(obj: dict[str, Any]) -> dict[str, Any]:
    resolution_object = obj.get("projective_resolution", {})
    if isinstance(resolution_object, dict):
        resolution = resolution_object.get("resolution", {})
        if isinstance(resolution, dict):
            return resolution
    return {}


def _metrics(resolution: dict[str, Any]) -> dict[str, float]:
    metrics = resolution.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def summarize_object(obj: dict[str, Any], index: int) -> dict[str, Any]:
    chain = obj.get("chain_complex", {}) if isinstance(obj.get("chain_complex", {}), dict) else {}
    resolution = _resolution(obj)
    metrics = _metrics(resolution)
    betti = chain.get("betti", {}) if isinstance(chain.get("betti", {}), dict) else {}
    ranks = chain.get("ranks", {}) if isinstance(chain.get("ranks", {}), dict) else {}
    sr = resolution.get("stanley_reisner_ideal", {}) if isinstance(resolution.get("stanley_reisner_ideal", {}), dict) else {}
    fitting_rows = resolution.get("fitting_entry_ideal_rows", resolution.get("fitting_entry_ideal_generators", [])) or []
    fitting_summary = resolution.get("fitting_summary_rows", resolution.get("fitting_determinantal_summary", [])) or []
    dg_rows = resolution.get("dg_differential_entries", []) or []
    dg_products = resolution.get("dg_product_summary_rows", resolution.get("dg_product_summary_by_bidegree", [])) or []
    taylor_ranks = resolution.get("taylor_rank_by_homological_degree", {}) or {}
    max_minor_log10 = 0.0
    for row in fitting_summary:
        if isinstance(row, dict):
            max_minor_log10 = max(max_minor_log10, float(row.get("maximal_minor_count_log10", 0.0) or 0.0))
    return {
        "object_id": f"step{int(obj.get('step', 0)):08d}_sample{int(obj.get('sample_index', index)):03d}",
        "step": int(obj.get("step", 0) or 0),
        "sample_index": int(obj.get("sample_index", index) or 0),
        "source_file": str(obj.get("source_file", "")),
        "chain_vertices": int(chain.get("num_vertices", 0) or 0),
        "chain_edges": len(chain.get("edges", []) or []),
        "chain_triangles": len(chain.get("triangles", []) or []),
        "chain_rank_boundary_1": int(ranks.get("rank_boundary_1", 0) or 0),
        "chain_rank_boundary_2": int(ranks.get("rank_boundary_2", 0) or 0),
        "chain_betti0": int(betti.get("beta_0", 0) or 0),
        "chain_betti1": int(betti.get("beta_1", 0) or 0),
        "chain_betti2": int(betti.get("beta_2", 0) or 0),
        "chain_euler": int(chain.get("euler_characteristic", 0) or 0),
        "sr_generator_count": int(sr.get("generator_count", 0) or 0),
        "hochster_betti_row_count": len(resolution.get("hochster_betti_rows", []) or []),
        "taylor_homological_degrees": len(taylor_ranks),
        "dg_differential_entry_count": len(dg_rows),
        "fitting_entry_generator_count": len(fitting_rows),
        "fitting_maximal_minor_count_log10_max": max_minor_log10,
        "dg_product_bidegree_count": len(dg_products),
        "symbolic_projective_dimension": float(metrics.get("symbolic_resolution_projective_dimension", 0.0) or 0.0),
        "symbolic_regularity": float(metrics.get("symbolic_resolution_regularity", 0.0) or 0.0),
        "symbolic_minimal_total_betti": float(metrics.get("symbolic_resolution_minimal_total_betti", 0.0) or 0.0),
        "symbolic_nonminimality_log2": float(metrics.get("symbolic_resolution_nonminimality_log2", 0.0) or 0.0),
        "symbolic_betti_entropy": float(metrics.get("symbolic_resolution_betti_entropy", 0.0) or 0.0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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


def plot_chain_complex(obj: dict[str, Any], out: Path) -> None:
    chain = obj.get("chain_complex", {}) if isinstance(obj.get("chain_complex", {}), dict) else {}
    vertices = list(chain.get("vertices", []) or [])
    edges = [tuple(edge) for edge in chain.get("edges", []) or []]
    triangles = [tuple(tri) for tri in chain.get("triangles", []) or []]
    n = max(1, len(vertices))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = {vertex: np.array([math.cos(angles[i]), math.sin(angles[i])]) for i, vertex in enumerate(vertices)}
    fig, ax = plt.subplots(figsize=(7.0, 7.0), facecolor="#020617")
    set_dark(ax)
    for tri in triangles:
        pts = np.array([xy.get(v, np.zeros(2)) for v in tri])
        ax.fill(pts[:, 0], pts[:, 1], color="#7c3aed", alpha=0.18, edgecolor="#a78bfa", linewidth=1.0)
    for src, dst in edges:
        start = xy.get(src, np.zeros(2))
        end = xy.get(dst, np.zeros(2))
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#22d3ee", "lw": 1.8, "shrinkA": 12, "shrinkB": 12},
        )
    if vertices:
        pts = np.array([xy[v] for v in vertices])
        ax.scatter(pts[:, 0], pts[:, 1], s=280, color="#f8fafc", edgecolor="#38bdf8", linewidth=2.0, zorder=3)
        for vertex in vertices:
            ax.text(*xy[vertex], str(vertex), ha="center", va="center", color="#0f172a", weight="bold", zorder=4)
    ax.set_title("Reasoning-step chain complex: vertices, arrows, 2-simplices")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_betti_resolution(row: dict[str, Any], out: Path) -> None:
    labels = [
        "beta0",
        "beta1",
        "beta2",
        "pd",
        "reg",
        "log2 nonmin",
        "betti entropy",
    ]
    values = [
        row["chain_betti0"],
        row["chain_betti1"],
        row["chain_betti2"],
        row["symbolic_projective_dimension"],
        row["symbolic_regularity"],
        row["symbolic_nonminimality_log2"],
        row["symbolic_betti_entropy"],
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor="#020617")
    set_dark(ax)
    bars = ax.bar(labels, values, color=["#38bdf8", "#22c55e", "#a78bfa", "#f59e0b", "#fb7185", "#e879f9", "#f8fafc"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2g}", ha="center", va="bottom", color="#e5e7eb", fontsize=9)
    ax.set_ylabel("exact finite invariant")
    ax.set_title("Chain homology and symbolic projective-resolution invariants")
    ax.grid(axis="y", color="#1e293b", alpha=0.7)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_taylor_ranks(obj: dict[str, Any], out: Path) -> None:
    resolution = _resolution(obj)
    ranks = resolution.get("taylor_rank_by_homological_degree", {}) or {}
    if not ranks:
        return
    degrees = sorted(int(key) for key in ranks)
    values = [float(ranks[str(degree)]) for degree in degrees]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor="#020617")
    set_dark(ax)
    ax.plot(degrees, values, color="#38bdf8", marker="o", linewidth=2.0)
    ax.fill_between(degrees, values, color="#38bdf8", alpha=0.16)
    ax.set_xlabel("homological degree")
    ax.set_ylabel("Taylor free-module rank")
    ax.set_title("Exact Taylor free-resolution rank profile")
    ax.grid(color="#1e293b", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_fitting_dg_heatmap(obj: dict[str, Any], out: Path) -> None:
    resolution = _resolution(obj)
    fitting = resolution.get("fitting_entry_ideal_rows", resolution.get("fitting_entry_ideal_generators", [])) or []
    products = resolution.get("dg_product_summary_rows", resolution.get("dg_product_summary_by_bidegree", [])) or []
    max_degree = 1
    for row in fitting:
        if isinstance(row, dict):
            max_degree = max(max_degree, int(row.get("homological_degree", 0) or 0))
    for row in products:
        if isinstance(row, dict):
            max_degree = max(max_degree, int(row.get("result_homological_degree", 0) or 0))
    matrix = np.zeros((max_degree + 1, max_degree + 1), dtype=float)
    for row in fitting:
        if isinstance(row, dict):
            degree = int(row.get("homological_degree", 0) or 0)
            quotient_degree = int(row.get("quotient_degree", row.get("degree", 0)) or 0)
            matrix[min(degree, max_degree), min(quotient_degree, max_degree)] += float(row.get("multiplicity", 1.0) or 1.0)
    for row in products:
        if isinstance(row, dict):
            left = int(row.get("left_homological_degree", 0) or 0)
            right = int(row.get("right_homological_degree", 0) or 0)
            matrix[min(left, max_degree), min(right, max_degree)] += math.log1p(float(row.get("product_count", 0.0) or 0.0))
    fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor="#020617")
    set_dark(ax)
    image = ax.imshow(np.log1p(matrix), cmap="magma")
    ax.set_xlabel("quotient/right degree")
    ax.set_ylabel("homological/left degree")
    ax.set_title("Fitting-entry ideals and DG product bidegrees")
    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.tick_params(colors="#e5e7eb")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_similarity(rows: list[dict[str, Any]], out: Path) -> None:
    if len(rows) < 2:
        return
    fields = [
        "chain_betti0",
        "chain_betti1",
        "chain_betti2",
        "chain_euler",
        "sr_generator_count",
        "symbolic_projective_dimension",
        "symbolic_regularity",
        "symbolic_minimal_total_betti",
        "dg_differential_entry_count",
        "fitting_entry_generator_count",
    ]
    x = np.asarray([[float(row.get(field, 0.0) or 0.0) for field in fields] for row in rows], dtype=float)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)
    dists = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    sim = np.exp(-dists)
    fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor="#020617")
    set_dark(ax)
    image = ax.imshow(sim, cmap="viridis", vmin=0.0, vmax=1.0)
    labels = [row["object_id"].replace("step", "s").replace("_sample", " b") for row in rows]
    ax.set_xticks(range(len(rows)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color="#e5e7eb")
    ax.set_yticklabels(labels, fontsize=7, color="#e5e7eb")
    ax.set_title("Analogical derived-category object similarity")
    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.tick_params(colors="#e5e7eb")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, rows: list[dict[str, Any]], figures: list[Path], source_dir: Path) -> Path:
    summary = {
        "object_count": len(rows),
        "steps": sorted({int(row["step"]) for row in rows}),
        "mean_chain_betti1": float(np.mean([row["chain_betti1"] for row in rows])) if rows else 0.0,
        "mean_symbolic_projective_dimension": float(np.mean([row["symbolic_projective_dimension"] for row in rows])) if rows else 0.0,
        "mean_symbolic_regularity": float(np.mean([row["symbolic_regularity"] for row in rows])) if rows else 0.0,
        "max_fitting_minor_count_log10": float(max([row["fitting_maximal_minor_count_log10_max"] for row in rows], default=0.0)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Derived-Category / CCA Analysis",
        "",
        f"- Source examples: `{source_dir}`",
        f"- Objects analyzed: `{summary['object_count']}`",
        f"- Steps: `{summary['steps']}`",
        f"- Mean chain beta_1: `{summary['mean_chain_betti1']:.4f}`",
        f"- Mean symbolic projective dimension: `{summary['mean_symbolic_projective_dimension']:.4f}`",
        f"- Mean symbolic regularity: `{summary['mean_symbolic_regularity']:.4f}`",
        f"- Max Fitting maximal-minor count log10: `{summary['max_fitting_minor_count_log10']:.4f}`",
        "",
        "## Interpretation",
        "",
        "The objects below are exact finite combinatorial objects emitted by the TokenGT trainer: each reasoning sample has a simplicial chain complex and a bounded projective-resolution object in `D^b(grmod-S)` built from the multigraded Stanley-Reisner/Taylor certificate.  The similarity heatmap compares analogical memory candidates through Betti, Euler, projective-dimension, regularity, DG-differential, and Fitting data.",
        "",
        "## Figures",
    ]
    for fig in figures:
        lines.append(f"- `{fig}`")
    if rows:
        lines.extend(["", "## Object Table", ""])
        header = "| object | step | V | E | beta_1 | pd | reg | SR gens | DG entries | Fitting entries |"
        lines.extend([header, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for row in rows:
            lines.append(
                f"| `{row['object_id']}` | {row['step']} | {row['chain_vertices']} | {row['chain_edges']} | "
                f"{row['chain_betti1']} | {row['symbolic_projective_dimension']:.0f} | "
                f"{row['symbolic_regularity']:.0f} | {row['sr_generator_count']} | "
                f"{row['dg_differential_entry_count']} | {row['fitting_entry_generator_count']} |"
            )
    report = out_dir / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def analyze(example_dir: str | Path, output_dir: str | Path, *, max_files: int = 8, max_objects: int = 24) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    objects = read_examples(example_dir, max_files=max_files, max_objects=max_objects)
    rows = [summarize_object(obj, index) for index, obj in enumerate(objects)]
    write_csv(out_dir / "derived_category_summary.csv", rows)
    figures: list[Path] = []
    for obj, row in zip(objects, rows):
        object_dir = out_dir / slug(row["object_id"])
        object_dir.mkdir(parents=True, exist_ok=True)
        chain_path = object_dir / "chain_complex.png"
        betti_path = object_dir / "betti_resolution.png"
        ranks_path = object_dir / "taylor_ranks.png"
        fitting_path = object_dir / "fitting_dg_heatmap.png"
        plot_chain_complex(obj, chain_path)
        plot_betti_resolution(row, betti_path)
        plot_taylor_ranks(obj, ranks_path)
        plot_fitting_dg_heatmap(obj, fitting_path)
        figures.extend([chain_path, betti_path, ranks_path, fitting_path])
    similarity_path = out_dir / "derived_category_similarity_heatmap.png"
    plot_similarity(rows, similarity_path)
    if similarity_path.exists():
        figures.append(similarity_path)
    report = write_report(out_dir, rows, figures, Path(example_dir))
    return {"rows": rows, "figures": [str(path) for path in figures], "report": str(report)}


def main() -> None:
    args = parse_args()
    result = analyze(args.example_dir, args.output_dir, max_files=args.max_files, max_objects=args.max_objects)
    print(json.dumps({"report": result["report"], "figures": len(result["figures"]), "objects": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
