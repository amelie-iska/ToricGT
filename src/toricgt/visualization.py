"""Visualization helpers for toric and tropical graph states."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_unit_circle_braid(points: np.ndarray, output: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
    ax.scatter(points.real, points.imag, c=np.arange(len(points)), cmap="viridis", s=80)
    for idx, z in enumerate(points):
        ax.text(float(z.real), float(z.imag), str(idx), ha="center", va="center", color="white")
    ax.set_aspect("equal")
    ax.axis("off")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tropical_decision_boundary(points: np.ndarray, labels: np.ndarray, output: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=12)
    ax.set_xlabel("projection 1")
    ax.set_ylabel("projection 2")
    ax.set_title("Tropical embedding projection")
    fig.colorbar(scatter, ax=ax)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_graph(edge_index: np.ndarray, node_values: np.ndarray, output: str | Path) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(node_values)))
    graph.add_edges_from([tuple(map(int, e)) for e in edge_index])
    pos = nx.spring_layout(graph, seed=17)
    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.4)
    nx.draw_networkx_nodes(graph, pos, node_color=node_values, cmap="magma", ax=ax)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    ax.axis("off")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def reasoning_trajectory(seed: int = 17, steps: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic embedding-space graph-of-thought trajectory.

    The synthetic path is driven by irrational rotations and a dissipative
    velocity update.  It is a visualization proxy for prefix-visible GFlowNet
    hidden states: local minima in the energy curve correspond to more stable
    reasoning endpoints.
    """

    rng = np.random.default_rng(seed)
    theta = (np.sqrt(5.0) - 1.0) / 2.0
    beta = np.sqrt(2.0)
    gamma = np.sqrt(3.0)
    path = np.zeros((steps, 3), dtype=np.float64)
    velocity = rng.normal(scale=0.04, size=3)
    for t in range(1, steps):
        phase = np.array(
            [
                np.sin(2 * np.pi * theta * t),
                np.cos(2 * np.pi * beta * t),
                np.sin(2 * np.pi * gamma * t + theta * t * t),
            ]
        )
        force = 0.08 * phase - 0.025 * path[t - 1]
        velocity = 0.93 * velocity + force
        path[t] = path[t - 1] + velocity
    minima = np.array([[1.1, -0.7, 0.35], [-0.8, 0.85, -0.45], [0.25, 0.2, 0.95]])
    distances = np.stack([np.sum((path - minimum) ** 2, axis=1) for minimum in minima], axis=1)
    energy = -np.log(np.exp(-3.0 * distances).sum(axis=1) + 1e-8) / 3.0
    return path, energy


def plot_reasoning_trajectory_3d(path: np.ndarray, energy: np.ndarray, output: str | Path) -> None:
    fig = plt.figure(figsize=(8, 6), facecolor="#05070d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#05070d")
    scatter = ax.scatter(path[:, 0], path[:, 1], path[:, 2], c=energy, cmap="plasma", s=18)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#50f5ff", linewidth=1.5, alpha=0.8)
    best = int(np.argmin(energy))
    ax.scatter(path[best, 0], path[best, 1], path[best, 2], s=110, color="#8cff6a", edgecolor="white")
    ax.set_title("Embedding-Space Graph-of-Thought Trajectory", color="white")
    ax.set_xlabel("toric phase axis", color="white")
    ax.set_ylabel("tropical face axis", color="white")
    ax.set_zlabel("GFlowNet action axis", color="white")
    ax.tick_params(colors="white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.02, 0.03, 0.06, 1.0))
        axis.pane.set_edgecolor((0.2, 0.8, 0.9, 0.35))
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.08)
    cbar.set_label("energy / negative reward", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_ramachandran_style_reasoning(path: np.ndarray, energy: np.ndarray, output: str | Path) -> None:
    velocity = np.diff(path, axis=0, prepend=path[:1])
    phi = np.arctan2(velocity[:, 1], velocity[:, 0])
    psi = np.arctan2(velocity[:, 2], np.linalg.norm(velocity[:, :2], axis=1) + 1e-8)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#05070d")
    ax.set_facecolor("#05070d")
    sc = ax.scatter(phi, psi, c=energy, cmap="viridis_r", s=16)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi / 2, np.pi / 2)
    ax.set_xlabel("reasoning torsion phi")
    ax.set_ylabel("reasoning torsion psi")
    ax.set_title("Ramachandran-Style Reasoning Torsion Plot")
    for spine in ax.spines.values():
        spine.set_color("#3cf4ff")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("energy / negative reward", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_energy_landscape(path: np.ndarray, energy: np.ndarray, output: str | Path) -> None:
    x = np.linspace(path[:, 0].min() - 0.3, path[:, 0].max() + 0.3, 120)
    y = np.linspace(path[:, 1].min() - 0.3, path[:, 1].max() + 0.3, 120)
    xx, yy = np.meshgrid(x, y)
    minima = np.array([[1.1, -0.7], [-0.8, 0.85], [0.25, 0.2]])
    landscape = np.zeros_like(xx) + 0.15 * (xx**2 + yy**2)
    for idx, minimum in enumerate(minima):
        landscape -= np.exp(-((xx - minimum[0]) ** 2 + (yy - minimum[1]) ** 2) * (2.5 + idx))
    fig = plt.figure(figsize=(8, 6.8), facecolor="#05070d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#05070d")
    surface = ax.plot_surface(xx, yy, landscape, cmap="magma", linewidth=0, antialiased=True, alpha=0.78)
    path_z = (
        0.15 * (path[:, 0] ** 2 + path[:, 1] ** 2)
        - np.exp(-((path[:, 0] - minima[0, 0]) ** 2 + (path[:, 1] - minima[0, 1]) ** 2) * 2.5)
        - np.exp(-((path[:, 0] - minima[1, 0]) ** 2 + (path[:, 1] - minima[1, 1]) ** 2) * 3.5)
        - np.exp(-((path[:, 0] - minima[2, 0]) ** 2 + (path[:, 1] - minima[2, 1]) ** 2) * 4.5)
        + 0.035
    )
    ax.plot(path[:, 0], path[:, 1], path_z, color="#50f5ff", linewidth=1.8)
    ax.scatter(path[:, 0], path[:, 1], path_z, c=energy, cmap="viridis_r", s=22, edgecolor="none")
    best = int(np.argmin(energy))
    ax.scatter(path[best, 0], path[best, 1], path_z[best] + 0.04, s=120, color="#8cff6a", edgecolor="white")
    ax.set_title("Reasoning Energy/Fitness Landscape")
    ax.set_xlabel("embedding projection 1")
    ax.set_ylabel("embedding projection 2")
    ax.set_zlabel("landscape energy")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.view_init(elev=30, azim=-52)
    cbar = fig.colorbar(surface, ax=ax, fraction=0.035, pad=0.08)
    cbar.set_label("landscape energy", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_interactive_reasoning_plot(path: np.ndarray, energy: np.ndarray, output: str | Path) -> None:
    """Write a dependency-free interactive 3D HTML plot."""

    rows = [
        {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "energy": float(e), "step": int(i)}
        for i, (p, e) in enumerate(zip(path, energy, strict=True))
    ]
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ToricGT Reasoning Trajectory</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>body{{margin:0;background:#05070d;color:white;font-family:system-ui}}#plot{{width:100vw;height:100vh}}</style>
</head><body><div id="plot"></div><script>
const rows = {rows!r};
const trace = {{
  x: rows.map(r => r.x), y: rows.map(r => r.y), z: rows.map(r => r.z),
  mode: 'lines+markers', type: 'scatter3d',
  marker: {{size: 4, color: rows.map(r => r.energy), colorscale: 'Plasma', colorbar: {{title:'energy'}}}},
  line: {{color: '#50f5ff', width: 5}},
  text: rows.map(r => `step ${{r.step}}<br>energy ${{r.energy.toFixed(4)}}`)
}};
Plotly.newPlot('plot', [trace], {{
  paper_bgcolor:'#05070d', plot_bgcolor:'#05070d',
  scene: {{
    xaxis: {{title:'toric phase', color:'white', gridcolor:'#17323a'}},
    yaxis: {{title:'tropical face', color:'white', gridcolor:'#17323a'}},
    zaxis: {{title:'GFlowNet action', color:'white', gridcolor:'#17323a'}}
  }},
  title: {{text:'Embedding-Space Graph-of-Thought Reasoning Trajectory', font:{{color:'white'}}}}
}}, {{responsive:true}});
</script></body></html>"""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(html, encoding="utf-8")
