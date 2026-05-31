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
    fig.savefig(OUT / name, bbox_inches="tight", format="pdf", facecolor=fig.get_facecolor())
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


def graphcg_topology_analogy_map():
    fig, ax = plt.subplots(figsize=(12, 6.2), facecolor="#05070d")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("#05070d")
    title_color = "#7df9ff"
    text_color = "#e9fbff"
    dim_color = "#93a4b8"

    ax.text(0.05, 0.92, "GraphCG chart -> directed persistent complexes -> analogical maps", color=title_color, fontsize=15, weight="bold")
    ax.text(0.05, 0.86, "Disentangled basis directions act as concept coordinates; topology and analogy are computed in that chart.", color=text_color, fontsize=9.5)

    # Basis chart.
    origin = np.array([0.15, 0.49])
    dirs = np.array([[0.18, 0.11], [0.02, 0.20], [-0.13, 0.08], [0.12, -0.12]])
    for i, d in enumerate(dirs):
        ax.arrow(origin[0], origin[1], d[0], d[1], width=0.004, head_width=0.018, head_length=0.018, color=["#50f5ff", "#ff4fd8", "#ffd166", "#8cff6a"][i], length_includes_head=True)
        ax.text(origin[0] + d[0] * 1.10, origin[1] + d[1] * 1.10, f"$d_{i+1}$", color=text_color, fontsize=10)
    ax.scatter([origin[0]], [origin[1]], s=36, color="#ffffff", zorder=5)
    ax.text(0.06, 0.26, "GraphCG/NCE\nlearns steerable\nbasis $B=[d_i]$", color=text_color, fontsize=10, ha="left")

    def draw_complex(cx: float, cy: float, phase: float, label: str) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + phase
        radii = np.array([0.09, 0.06, 0.10, 0.07, 0.085, 0.055, 0.095])
        pts = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=-1)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 5), (5, 2), (0, 6), (6, 3)]
        tris = [(0, 1, 2), (2, 4, 5)]
        for tri in tris:
            ax.add_patch(Polygon(pts[list(tri)], closed=True, facecolor="#1c7c91", edgecolor="none", alpha=0.22))
        for i, j in edges:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#50f5ff", alpha=0.78, linewidth=1.2)
        for i, j in [(0, 2), (2, 5), (5, 4), (3, 6)]:
            ax.arrow(pts[i, 0], pts[i, 1], (pts[j, 0]-pts[i, 0])*0.72, (pts[j, 1]-pts[i, 1])*0.72, head_width=0.012, head_length=0.014, color="#ff4fd8", alpha=0.80, length_includes_head=True)
        ax.scatter(pts[:, 0], pts[:, 1], s=30, color="#e9fbff", edgecolor="#030712", linewidth=0.5, zorder=4)
        ax.text(cx - 0.11, cy - 0.16, label, color=text_color, fontsize=10)
        return pts

    pts_a = draw_complex(0.45, 0.56, 0.1, "$K_t(\\rho)$")
    pts_b = draw_complex(0.74, 0.53, 0.55, "$K_{t+1}(\\rho)$")
    for i in [0, 2, 4, 6]:
        start = pts_a[i]
        end = pts_b[(i + 1) % len(pts_b)]
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, lw=1.0, color="#ffd166", alpha=0.72, connectionstyle="arc3,rad=0.08"))
    ax.text(0.56, 0.74, "$P_t: C_\\bullet(K_t) \\to C_\\bullet(K_{t+1})$", color="#ffd166", fontsize=12)
    ax.text(0.39, 0.22, "nested radii\n$K(\\rho_1)\\subseteq K(\\rho_2)\\subseteq K(\\rho_3)$", color=dim_color, fontsize=9, ha="center")
    ax.text(0.70, 0.22, "losses: Dirichlet energy,\nchain-map residual,\ndirected map residual", color=dim_color, fontsize=9, ha="center")
    save(fig, "fig_graphcg_topology_analogy_map.pdf")


def toric_phase_simplicial_trajectory():
    fig = plt.figure(figsize=(10.8, 7.2), facecolor="#05070d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#05070d")
    theta = (np.sqrt(5) - 1) / 2
    beta = np.sqrt(2)
    steps = 180
    k = np.arange(steps)
    u = 2 * np.pi * theta * k
    v = 2 * np.pi * beta * k
    R, r = 1.0, 0.32
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    energy = 0.58 + 0.22 * np.sin(0.09 * k + 0.7) + 0.18 * np.cos(theta * k)
    uu, vv = np.meshgrid(np.linspace(0, 2*np.pi, 80), np.linspace(0, 2*np.pi, 28))
    xx = (R + r * np.cos(vv)) * np.cos(uu)
    yy = (R + r * np.cos(vv)) * np.sin(uu)
    zz = r * np.sin(vv)
    ax.plot_surface(xx, yy, zz, color="#0b2a36", alpha=0.16, linewidth=0)
    ax.plot_wireframe(xx, yy, zz, rstride=4, cstride=8, color="#1ad7e8", alpha=0.09, linewidth=0.35)
    points = np.stack([x, y, z], axis=-1)
    for start in range(0, steps - 24, 24):
        window = points[start:start+24]
        dist = np.linalg.norm(window[:, None, :] - window[None, :, :], axis=-1)
        threshold = np.quantile(dist[dist > 1e-8], 0.16)
        for i in range(window.shape[0]):
            for j in range(i + 1, window.shape[0]):
                if dist[i, j] <= threshold:
                    p, q = window[i], window[j]
                    ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#6df6ff", alpha=0.12, linewidth=0.5)
        if start + 48 < steps:
            c0 = window.mean(axis=0)
            c1 = points[start+24:start+48].mean(axis=0)
            delta = c1 - c0
            ax.quiver(c0[0], c0[1], c0[2], delta[0], delta[1], delta[2], color="#ff4fd8", linewidth=0.9, arrow_length_ratio=0.22)
    ax.plot(x, y, z, color="#50f5ff", linewidth=1.6, alpha=0.9)
    sc = ax.scatter(x, y, z, c=energy, cmap="magma", s=18, edgecolor="#06111f", linewidth=0.2)
    ax.scatter(x[0], y[0], z[0], s=80, color="#6df6ff", edgecolor="white")
    ax.scatter(x[-1], y[-1], z[-1], s=110, marker="*", color="#ffd166", edgecolor="white")
    ax.set_title("Irrational toric phase winding with local simplicial reasoning structure", color="white", fontsize=12)
    ax.text2D(0.02, 0.02, "cyan path = projected rotation-algebra phase; faint edges = local VR 1-skeleton; magenta arrows = analogical maps", transform=ax.transAxes, color="#e9fbff", fontsize=8.5)
    ax.set_axis_off()
    ax.view_init(elev=27, azim=40)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.68, pad=0.02)
    cbar.set_label("local tropical energy proxy", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    save(fig, "fig_toric_phase_simplicial_trajectory.pdf")


def dec_conservative_reasoning():
    fig, ax = plt.subplots(figsize=(12, 6.4), facecolor="#05070d")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("#05070d")
    title_color = "#7df9ff"
    text_color = "#e9fbff"
    muted = "#96a9bb"
    ax.text(0.05, 0.92, "DEC-style conservative audits on directed reasoning complexes", color=title_color, fontsize=15, weight="bold")
    ax.text(0.05, 0.86, "The Navier-Stokes DEC machinery becomes a lightweight consistency layer over graph-of-thought hidden flows.", color=text_color, fontsize=9.5)

    # Local directed complex.
    center = np.array([0.23, 0.53])
    pts = np.array([
        [-0.12, 0.02],
        [-0.04, 0.13],
        [0.07, 0.10],
        [0.13, -0.03],
        [0.02, -0.13],
        [-0.10, -0.10],
    ]) + center
    tris = [(0, 1, 2), (0, 2, 5), (2, 3, 4)]
    edges = [(0, 1), (1, 2), (2, 0), (0, 5), (5, 2), (2, 3), (3, 4), (4, 2), (1, 3), (5, 4)]
    for tri in tris:
        ax.add_patch(Polygon(pts[list(tri)], closed=True, facecolor="#1c7c91", edgecolor="none", alpha=0.24))
    for i, j in edges:
        start, end = pts[i], pts[j]
        delta = end - start
        ax.arrow(start[0], start[1], 0.73 * delta[0], 0.73 * delta[1], head_width=0.010, head_length=0.013, color="#50f5ff", alpha=0.78, length_includes_head=True, linewidth=0.8)
    ax.scatter(pts[:, 0], pts[:, 1], s=38, color="#e9fbff", edgecolor="#030712", linewidth=0.5, zorder=4)
    ax.text(0.09, 0.26, "directed flag complex\n$K_t(\\rho)$", color=text_color, fontsize=10)

    # DEC operators.
    op_x = 0.45
    boxes = [
        ("skew flow\n$u=A^\\to-(A^\\to)^T$", 0.71, "#102c46"),
        ("exterior derivative\n$d u$ / divergence", 0.57, "#102c46"),
        ("Hodge balance\n$*u$ via core radii", 0.43, "#102c46"),
        ("wedge/interior\n$u\\wedge\\omega$ vs $i_u\\omega$", 0.29, "#102c46"),
    ]
    for label, y, fc in boxes:
        patch = FancyBboxPatch((op_x, y - 0.045), 0.22, 0.09, boxstyle="round,pad=0.01,rounding_size=0.012", linewidth=1.0, facecolor=fc, edgecolor="#2bdff0")
        ax.add_patch(patch)
        ax.text(op_x + 0.11, y, label, color=text_color, fontsize=9.2, ha="center", va="center")
    for y in [0.71, 0.57, 0.43, 0.29]:
        ax.add_patch(FancyArrowPatch((0.34, 0.53), (op_x, y), arrowstyle="-|>", mutation_scale=12, color="#ffd166", lw=0.9, alpha=0.76, connectionstyle="arc3,rad=0.08"))

    # Loss summary.
    ax.add_patch(FancyBboxPatch((0.73, 0.32), 0.22, 0.34, boxstyle="round,pad=0.015,rounding_size=0.016", linewidth=1.0, facecolor="#111827", edgecolor="#ff4fd8"))
    ax.text(0.84, 0.60, "$\\mathcal{L}_{\\mathrm{DEC}}$", color="#ffb6ea", fontsize=17, weight="bold", ha="center")
    ax.text(0.84, 0.52, "$=\\|\\delta u\\|^2$", color=text_color, fontsize=11, ha="center")
    ax.text(0.84, 0.46, "$+\\lambda_\\omega\\|\\omega_{\\rho+}-\\omega_\\rho\\|^2$", color=text_color, fontsize=9.5, ha="center")
    ax.text(0.84, 0.40, "$+\\lambda_E|E_{\\rho+}-E_\\rho|^2$", color=text_color, fontsize=9.5, ha="center")
    ax.text(0.84, 0.34, "$+\\lambda_*\\mathcal{R}_*+\\lambda_\\wedge\\mathcal{R}_\\wedge$", color=text_color, fontsize=9.5, ha="center")
    ax.add_patch(FancyArrowPatch((0.67, 0.50), (0.73, 0.50), arrowstyle="-|>", mutation_scale=14, color="#ff4fd8", lw=1.2))

    ax.text(0.05, 0.08, "Healthy behavior: low divergence, bounded vorticity/energy drift, nonzero directed cycle structure, and no collapse to a symmetric trivial complex.", color=muted, fontsize=9.2)
    save(fig, "fig_dec_conservative_reasoning.pdf")


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
    graphcg_topology_analogy_map()
    toric_phase_simplicial_trajectory()
    dec_conservative_reasoning()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
