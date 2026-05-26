#!/usr/bin/env python3
"""Generate deterministic PDF figures for the ToricGT manuscript."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT = Path("assets/toricgt_pg_softmoe_figures")


COLORS = {
    "ink": "#111827",
    "muted": "#4b5563",
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#16a34a",
    "amber": "#d97706",
    "red": "#dc2626",
    "violet": "#7c3aed",
    "slate": "#334155",
    "light": "#f8fafc",
    "line": "#cbd5e1",
}


def setup(figsize=(11, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, xy, wh, text, fc="white", ec=None, size=9, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.2,
        facecolor=fc,
        edgecolor=ec or COLORS["line"],
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=COLORS["ink"], weight=weight)
    return patch


def arrow(ax, start, end, color=None, lw=1.5, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            lw=lw,
            color=color or COLORS["slate"],
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", format="pdf")
    plt.close(fig)


def tropical_active_faces():
    fig, ax = setup((11, 5))
    pts = np.array([[0.12, 0.22], [0.25, 0.70], [0.42, 0.34], [0.58, 0.78], [0.78, 0.32], [0.90, 0.62]])
    edges = [(0, 1, 3.0), (0, 2, 1.2), (2, 3, 1.4), (1, 3, 0.9), (3, 5, 1.1), (2, 4, 2.7), (4, 5, 0.8)]
    active = {(0, 2), (2, 3), (3, 5)}
    for u, v, w in edges:
        color = COLORS["green"] if (u, v) in active else COLORS["line"]
        arrow(ax, pts[u], pts[v], color=color, lw=2.4 if (u, v) in active else 1.1)
        mid = (pts[u] + pts[v]) / 2
        ax.text(mid[0], mid[1] + 0.025, f"{w:g}", fontsize=8, color=color)
    for i, (x, y) in enumerate(pts):
        ax.add_patch(Circle((x, y), 0.028, fc="#eff6ff", ec=COLORS["blue"], lw=1.2))
        ax.text(x, y, str(i), ha="center", va="center", fontsize=9)
    x = np.linspace(0.08, 0.92, 200)
    funcs = [0.18 + 0.50 * x, 0.58 - 0.23 * x, 0.18 + 0.85 * np.maximum(0, x - 0.45)]
    for f, c in zip(funcs, [COLORS["blue"], COLORS["amber"], COLORS["violet"]]):
        ax.plot(x, f, color=c, lw=1.3, alpha=0.45)
    ax.plot(x, np.maximum.reduce(funcs), color=COLORS["red"], lw=2.2)
    ax.text(0.08, 0.93, "Tropical active path and max-plus envelope", fontsize=14, weight="bold")
    ax.text(0.53, 0.12, "Active face changes mark discrete decisions; margins audit stability.", fontsize=10, color=COLORS["muted"])
    save(fig, "fig_tropical_active_faces.pdf")


def spiral_projection():
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    alpha = (np.sqrt(5) - 1) / 2
    beta = np.sqrt(2) - 1
    k = np.arange(700)
    ax1.plot(np.cos(2 * np.pi * alpha * k), np.sin(2 * np.pi * alpha * k), k / k.max(), color=COLORS["blue"], lw=1.2)
    u = 2 * np.pi * alpha * k
    v = 2 * np.pi * beta * k
    R, r = 1.2, 0.34
    ax2.scatter((R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v), s=3, color=COLORS["violet"], alpha=0.75)
    for ax, title in [(ax1, "Irrational circle lift"), (ax2, "Dense torus projection")]:
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 0.65))
    fig.suptitle("Spiral diagnostics for irrational rotations", fontsize=14, weight="bold")
    save(fig, "fig_spiral_projection.pdf")


def hebrew_root_graph():
    fig, ax = setup((11, 5.5))
    nodes = {
        "Root K-T-B": (0.18, 0.70, "#eff6ff"),
        "Radical K": (0.08, 0.45, "#ecfeff"),
        "Radical T": (0.18, 0.45, "#ecfeff"),
        "Radical B": (0.28, 0.45, "#ecfeff"),
        "Template C1aC2aC3": (0.50, 0.70, "#fef3c7"),
        "Binyan Qal": (0.50, 0.45, "#fef3c7"),
        "Form katav": (0.76, 0.70, "#f0fdf4"),
        "Lemma write": (0.76, 0.45, "#f0fdf4"),
        "Verse span": (0.76, 0.22, "#f8fafc"),
    }
    loc = {}
    for label, (x, y, fc) in nodes.items():
        box(ax, (x - 0.065, y - 0.04), (0.13, 0.08), label, fc=fc, size=8)
        loc[label] = (x, y)
    for rad in ["Radical K", "Radical T", "Radical B"]:
        arrow(ax, loc["Root K-T-B"], loc[rad], COLORS["blue"], lw=1.1)
        arrow(ax, loc[rad], loc["Template C1aC2aC3"], COLORS["cyan"], lw=1.1, rad=0.1)
    for src, dst in [("Template C1aC2aC3", "Form katav"), ("Binyan Qal", "Form katav"), ("Form katav", "Lemma write"), ("Form katav", "Verse span")]:
        arrow(ax, loc[src], loc[dst], COLORS["slate"], lw=1.2)
    ax.text(0.06, 0.90, "Root-template morphology as a typed graph", fontsize=14, weight="bold")
    ax.text(0.06, 0.12, "Records preserve analyzer uncertainty, paradigm edges, source spans, and semiring alignment losses.", fontsize=10, color=COLORS["muted"])
    save(fig, "fig_hebrew_root_graph.pdf")


def reasoning_capacity():
    fig, ax = setup((11, 5.8))
    centers = [(0.18, 0.58), (0.38, 0.58), (0.58, 0.58), (0.78, 0.58)]
    labels = ["Graph-token\nrouting", "Toric phase\ntransport", "Tropical DP\nselection", "Graph output\nverification"]
    colors = ["#eff6ff", "#faf5ff", "#f0fdf4", "#fff7ed"]
    for (x, y), label, fc in zip(centers, labels, colors):
        ax.add_patch(Circle((x, y), 0.085, fc=fc, ec=COLORS["line"], lw=1.3))
        ax.text(x, y, label, ha="center", va="center", fontsize=9, weight="bold")
    for a, b in zip(centers[:-1], centers[1:]):
        arrow(ax, (a[0] + 0.085, a[1]), (b[0] - 0.085, b[1]), COLORS["slate"])
    checks = ["orbit error", "cocycle residual", "active margin", "verifier score", "reward diversity"]
    for i, check in enumerate(checks):
        box(ax, (0.12 + i * 0.16, 0.20), (0.12, 0.08), check, fc="#f8fafc", size=8)
        arrow(ax, (0.18 + i * 0.16, 0.28), (0.48, 0.50), COLORS["muted"], lw=0.9, rad=0.12 - i * 0.04)
    ax.text(0.06, 0.88, "Reasoning capacity is audited at each computational layer", fontsize=14, weight="bold")
    save(fig, "fig_reasoning_capacity.pdf")


def visualization_dashboard():
    fig, ax = setup((11.5, 6))
    titles = ["Equivariance", "Active faces", "Toric phases", "Hebrew paradigms", "GFlowNet rewards", "Cache residuals"]
    for i, title in enumerate(titles):
        row, col = divmod(i, 3)
        x, y = 0.06 + col * 0.31, 0.58 - row * 0.34
        box(ax, (x, y), (0.24, 0.22), title, fc="#f8fafc", size=10, weight="bold")
        xs = np.linspace(x + 0.03, x + 0.21, 24)
        vals = y + 0.06 + 0.06 * np.sin(np.linspace(0, 2 * np.pi, 24) + i)
        ax.plot(xs, vals, color=[COLORS["blue"], COLORS["green"], COLORS["violet"], COLORS["amber"], COLORS["red"], COLORS["cyan"]][i], lw=1.6)
    ax.text(0.06, 0.92, "Visualization contract dashboard", fontsize=14, weight="bold")
    save(fig, "fig_visualization_dashboard.pdf")


def polar_ring_cache():
    fig, ax = setup((11.5, 5.5))
    labels = ["KV block", "Random\nprecondition", "Recursive\npolar transform", "Angle\ncodebooks", "Ring\ntraversal", "Dequantized\nattention"]
    fills = ["#eff6ff", "#ecfeff", "#f0fdf4", "#fef3c7", "#faf5ff", "#fff7ed"]
    for i, (label, fc) in enumerate(zip(labels, fills)):
        x = 0.04 + i * 0.155
        box(ax, (x, 0.55), (0.125, 0.16), label, fc=fc, size=8.5, weight="bold")
        if i < len(labels) - 1:
            arrow(ax, (x + 0.125, 0.63), (x + 0.155, 0.63), COLORS["slate"])
    ax.add_patch(Rectangle((0.12, 0.24), 0.76, 0.10, fc="#f8fafc", ec=COLORS["line"]))
    for i in range(12):
        ax.add_patch(Rectangle((0.13 + i * 0.06, 0.255), 0.04, 0.07, fc=COLORS["blue"] if i % 3 == 0 else COLORS["cyan"], alpha=0.65))
    ax.text(0.06, 0.89, "PolarQuant-style cache interleaved with ring attention", fontsize=14, weight="bold")
    ax.text(0.13, 0.18, "Packed radii and angles are unpacked blockwise; tropical margins bound active-face preservation.", fontsize=10, color=COLORS["muted"])
    save(fig, "fig_polar_ring_cache.pdf")


def soft_moe_toricgt():
    fig, ax = setup((11.5, 5.7))
    token_y = np.linspace(0.25, 0.78, 6)
    for y in token_y:
        box(ax, (0.06, y - 0.025), (0.13, 0.05), "graph token", fc="#eff6ff", size=7)
    expert_x = [0.48, 0.62, 0.76, 0.90]
    for e, x in enumerate(expert_x):
        box(ax, (x - 0.045, 0.53), (0.09, 0.12), f"Expert {e+1}", fc="#f0fdf4", size=8, weight="bold")
        box(ax, (x - 0.04, 0.35), (0.08, 0.08), "slot", fc="#fef3c7", size=8)
    for y in token_y:
        for x in expert_x:
            arrow(ax, (0.19, y), (x - 0.05, 0.39), COLORS["line"], lw=0.8)
            arrow(ax, (x - 0.045, 0.59), (0.26, y), COLORS["line"], lw=0.8, rad=-0.05)
    box(ax, (0.25, 0.48), (0.13, 0.14), "dispatch\nsoftmax over tokens", fc="#ecfeff", size=8)
    box(ax, (0.25, 0.28), (0.13, 0.14), "combine\nsoftmax over slots", fc="#faf5ff", size=8)
    ax.text(0.05, 0.90, "Default 4-expert Soft-MoE graph-token block", fontsize=14, weight="bold")
    ax.text(0.05, 0.12, "Shared routing preserves permutation equivariance; diagnostics report expert mass and slot entropy.", fontsize=10, color=COLORS["muted"])
    save(fig, "fig_soft_moe_toricgt_clean.pdf")


def parameter_golf_protocol():
    fig, ax = setup((11.5, 5.7))
    labels = ["Dense\nbaseline", "Distill\nToricGT traces", "Prefix-causal\nSoft-MoE", "QAT + pack", "Artifact\n<=16,000,000 B", "BPB\nvalidation"]
    fills = ["#eff6ff", "#ecfeff", "#f0fdf4", "#faf5ff", "#fef3c7", "#fff7ed"]
    for i, (label, fc) in enumerate(zip(labels, fills)):
        x = 0.04 + i * 0.155
        box(ax, (x, 0.56), (0.125, 0.15), label, fc=fc, size=8.5, weight="bold")
        if i < len(labels) - 1:
            arrow(ax, (x + 0.125, 0.635), (x + 0.155, 0.635), COLORS["slate"])
    checks = ["no network", "strict prefix", "roundtrip", "byte audit", "3 seeds"]
    for i, label in enumerate(checks):
        box(ax, (0.12 + i * 0.16, 0.25), (0.12, 0.08), label, fc="#f8fafc", size=8)
    ax.text(0.05, 0.90, "Parameter-Golf compression track", fontsize=14, weight="bold")
    ax.text(0.08, 0.15, "The contest LM is a separate deployment artifact; the full ToricGT graph model is the teacher and diagnostic scaffold.", fontsize=10, color=COLORS["muted"])
    save(fig, "fig_parameter_golf_protocol_clean.pdf")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": COLORS["line"],
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
        }
    )
    tropical_active_faces()
    spiral_projection()
    hebrew_root_graph()
    reasoning_capacity()
    visualization_dashboard()
    polar_ring_cache()
    soft_moe_toricgt()
    parameter_golf_protocol()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
