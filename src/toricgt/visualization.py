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

