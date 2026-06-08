"""Visualization helpers for toric and tropical graph states."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .got_trajectory import default_branch_merge_edges_np


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
    """Create deterministic embedding-space graph-of-thought vertex states.

    The vertex coordinates are driven by irrational rotations and a dissipative
    velocity update, but every three-step cell is placed as a visible diamond:
    a central source, two separated alternatives, and a central merge.  The
    returned array is indexed by reasoning-step id; it is not intended to imply
    that the graph-of-thought is a single chain.
    """

    rng = np.random.default_rng(seed)
    theta = (np.sqrt(5.0) - 1.0) / 2.0
    beta = np.sqrt(2.0)
    gamma = np.sqrt(3.0)
    centerline = np.zeros((steps, 3), dtype=np.float64)
    velocity = rng.normal(scale=0.04, size=3)
    for t in range(1, steps):
        phase = np.array(
            [
                np.sin(2 * np.pi * theta * t),
                np.cos(2 * np.pi * beta * t),
                np.sin(2 * np.pi * gamma * t + theta * t * t),
            ]
        )
        force = 0.08 * phase - 0.025 * centerline[t - 1]
        velocity = 0.93 * velocity + force
        centerline[t] = centerline[t - 1] + velocity
    path = centerline.copy()
    for t in range(steps):
        phase = np.array(
            [
                np.cos(2 * np.pi * theta * (t + 1)),
                np.sin(2 * np.pi * beta * (t + 1)),
                np.cos(2 * np.pi * gamma * (t + 1)),
            ]
        )
        branch_scale = 0.46 + 0.08 * np.sin(0.37 * t + seed)
        if t % 3 == 1:
            path[t] += branch_scale * np.array([0.08, 1.0, 0.32]) + 0.035 * phase
        elif t % 3 == 2:
            path[t] += branch_scale * np.array([-0.08, -1.0, -0.32]) - 0.035 * phase
    minima = np.array([[1.1, -0.7, 0.35], [-0.8, 0.85, -0.45], [0.25, 0.2, 0.95]])
    distances = np.stack([np.sum((path - minimum) ** 2, axis=1) for minimum in minima], axis=1)
    energy = -np.log(np.exp(-3.0 * distances).sum(axis=1) + 1e-8) / 3.0
    return path, energy


def _edge_role_colors(edges: np.ndarray, n: int) -> list[str]:
    out_degree = np.zeros((n,), dtype=np.int64)
    in_degree = np.zeros((n,), dtype=np.int64)
    for src, dst in edges:
        if 0 <= src < n and 0 <= dst < n and src != dst:
            out_degree[src] += 1
            in_degree[dst] += 1
    colors: list[str] = []
    for src, dst in edges:
        if out_degree[src] > 1 and in_degree[dst] > 1:
            colors.append("#ffd166")
        elif out_degree[src] > 1:
            colors.append("#ff4fd8")
        elif in_degree[dst] > 1:
            colors.append("#8cff6a")
        else:
            colors.append("#50f5ff")
    return colors


def _plot_branch_merge_edges_3d(
    ax,
    points: np.ndarray,
    *,
    edges: np.ndarray | None = None,
    linewidth: float = 1.25,
    alpha: float = 0.66,
    arrows: bool = False,
) -> None:
    if edges is None:
        edges = default_branch_merge_edges_np(len(points))
    edge_colors = _edge_role_colors(edges, len(points))
    for (src, dst), color in zip(edges, edge_colors, strict=False):
        if not (0 <= src < len(points) and 0 <= dst < len(points)):
            continue
        p = points[int(src)]
        q = points[int(dst)]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=color, linewidth=linewidth, alpha=alpha)
        if arrows:
            delta = q - p
            ax.quiver(
                p[0],
                p[1],
                p[2],
                0.72 * delta[0],
                0.72 * delta[1],
                0.72 * delta[2],
                color=color,
                linewidth=max(0.35, 0.42 * linewidth),
                arrow_length_ratio=0.18,
                alpha=min(0.92, alpha + 0.08),
            )


def plot_reasoning_trajectory_3d(
    path: np.ndarray,
    energy: np.ndarray,
    output: str | Path,
    *,
    edges: np.ndarray | None = None,
) -> None:
    fig = plt.figure(figsize=(8, 6), facecolor="#05070d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#05070d")
    if edges is None:
        edges = default_branch_merge_edges_np(len(path))
    _plot_branch_merge_edges_3d(ax, path, edges=edges, linewidth=1.25, alpha=0.66, arrows=True)
    scatter = ax.scatter(path[:, 0], path[:, 1], path[:, 2], c=energy, cmap="plasma", s=18)
    best = int(np.argmin(energy))
    ax.scatter(path[best, 0], path[best, 1], path[best, 2], s=110, color="#8cff6a", edgecolor="white")
    ax.set_title("Embedding-Space Graph-of-Thought Branch/Merge DAG", color="white")
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


def plot_energy_landscape(
    path: np.ndarray,
    energy: np.ndarray,
    output: str | Path,
    *,
    edges: np.ndarray | None = None,
) -> None:
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
    landscape_points = np.column_stack([path[:, 0], path[:, 1], path_z])
    _plot_branch_merge_edges_3d(ax, landscape_points, edges=edges, linewidth=1.45, alpha=0.70, arrows=False)
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


def _energy_landscape_grid(path: np.ndarray, resolution: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(path[:, 0].min() - 0.3, path[:, 0].max() + 0.3, resolution)
    y = np.linspace(path[:, 1].min() - 0.3, path[:, 1].max() + 0.3, resolution)
    xx, yy = np.meshgrid(x, y)
    minima = np.array([[1.1, -0.7], [-0.8, 0.85], [0.25, 0.2]], dtype=np.float64)
    landscape = np.zeros_like(xx) + 0.15 * (xx**2 + yy**2)
    for idx, minimum in enumerate(minima):
        landscape -= np.exp(-((xx - minimum[0]) ** 2 + (yy - minimum[1]) ** 2) * (2.5 + idx))
    return x, y, landscape, minima


def _energy_landscape_value(points: np.ndarray, minima: np.ndarray) -> np.ndarray:
    values = 0.15 * (points[:, 0] ** 2 + points[:, 1] ** 2)
    for idx, minimum in enumerate(minima):
        values -= np.exp(-((points[:, 0] - minimum[0]) ** 2 + (points[:, 1] - minimum[1]) ** 2) * (2.5 + idx))
    return values


def _bisector_segment(m0: np.ndarray, m1: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]) -> np.ndarray:
    """Return clipped points on the projected equal-basin wall between two minima."""

    midpoint = 0.5 * (m0 + m1)
    direction = np.array([-(m1 - m0)[1], (m1 - m0)[0]], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return np.zeros((0, 2), dtype=np.float64)
    direction /= norm
    span = 2.4 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    ts = np.linspace(-span, span, 220)
    segment = midpoint[None, :] + ts[:, None] * direction[None, :]
    keep = (
        (segment[:, 0] >= xlim[0])
        & (segment[:, 0] <= xlim[1])
        & (segment[:, 1] >= ylim[0])
        & (segment[:, 1] <= ylim[1])
    )
    return segment[keep]


def write_interactive_energy_landscape(
    path: np.ndarray,
    energy: np.ndarray,
    output: str | Path,
    *,
    edges: np.ndarray | None = None,
) -> None:
    """Write a rotatable HTML energy-landscape view for TokenGT trajectories.

    The surface is intentionally a smooth surrogate over the first two projected
    embedding axes; discontinuous toric/tropical chamber data is shown as
    overlays because PCA projection and interpolation otherwise hide most hard
    walls.
    """

    path = np.asarray(path, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] < 2:
        return
    if energy.shape[0] < path.shape[0]:
        return
    if edges is None:
        edges = default_branch_merge_edges_np(len(path))

    x, y, landscape, minima = _energy_landscape_grid(path, resolution=100)
    path_xy = path[:, :2]
    path_z = _energy_landscape_value(path_xy, minima) + 0.045
    finite_energy = np.nan_to_num(energy[: path.shape[0]], nan=float(np.nanmean(energy)) if np.isfinite(np.nanmean(energy)) else 0.0)
    rows = [
        {
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(z),
            "energy": float(e),
            "step": int(i),
        }
        for i, (p, z, e) in enumerate(zip(path_xy, path_z, finite_energy, strict=True))
    ]
    edge_rows = [
        {"src": int(src), "dst": int(dst), "role": role}
        for (src, dst), role in zip(edges, _edge_role_colors(edges, len(path)), strict=False)
        if 0 <= int(src) < len(path) and 0 <= int(dst) < len(path)
    ]

    xlim = (float(x.min()), float(x.max()))
    ylim = (float(y.min()), float(y.max()))
    wall_rows: list[dict[str, object]] = []
    for wall_idx, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        segment = _bisector_segment(minima[i], minima[j], xlim, ylim)
        if segment.shape[0] < 2:
            continue
        wall_z = _energy_landscape_value(segment, minima) + 0.095
        wall_rows.append(
            {
                "name": f"projected chamber wall {i}-{j}",
                "x": segment[:, 0].tolist(),
                "y": segment[:, 1].tolist(),
                "z": wall_z.tolist(),
                "color": ["#67e8f9", "#f0abfc", "#fde68a"][wall_idx],
            }
        )

    best = int(np.nanargmin(finite_energy)) if finite_energy.size else 0
    payload = {
        "x": x.tolist(),
        "y": y.tolist(),
        "z": landscape.tolist(),
        "rows": rows,
        "edges": edge_rows,
        "walls": wall_rows,
        "best": best,
    }
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ToricGT Energy Landscape</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
body{{margin:0;background:#05070d;color:white;font-family:system-ui}}
#plot{{width:100vw;height:100vh}}
.note{{position:absolute;z-index:5;left:18px;top:12px;max-width:980px;color:#e8fbff}}
.note h1{{font-size:20px;line-height:1.15;margin:0 0 5px 0}}
.note p{{font-size:12px;line-height:1.35;margin:0;opacity:.88}}
</style></head><body>
<div class="note">
  <h1>Interactive TokenGT Energy/Fitness Landscape</h1>
  <p>The surface is a smooth low-dimensional surrogate over projected embedding coordinates. Cyan/magenta/gold traces mark projected toric/tropical chamber-wall proxies; hard fan boundaries can be compressed or hidden by PCA, smoothing, and the learned continuous embeddings.</p>
</div>
<div id="plot"></div><script>
const payload = {json.dumps(payload)};
const surface = {{
  type:'surface', name:'smooth surrogate energy surface',
  x: payload.x, y: payload.y, z: payload.z,
  colorscale:'Magma', opacity:0.80, showscale:true,
  colorbar:{{title:'surrogate landscape energy'}}
}};
const wallTraces = payload.walls.map(w => ({{
  type:'scatter3d', mode:'lines', name:w.name,
  x:w.x, y:w.y, z:w.z,
  line:{{color:w.color, width:8}},
  hoverinfo:'name'
}}));
const edgeTraces = payload.edges.map((e, idx) => {{
  const a = payload.rows[e.src], b = payload.rows[e.dst];
  return {{
    type:'scatter3d', mode:'lines', showlegend:idx < 8,
    name:e.role === '#ff4fd8' ? 'branch edge' : (e.role === '#8cff6a' ? 'merge edge' : 'DAG edge'),
    x:[a.x,b.x], y:[a.y,b.y], z:[a.z,b.z],
    line:{{color:e.role, width:5}},
    text:[`${{e.src}} -> ${{e.dst}}`, `${{e.src}} -> ${{e.dst}}`],
    hoverinfo:'text'
  }};
}});
const nodeTrace = {{
  type:'scatter3d', mode:'markers', name:'reasoning vertices',
  x:payload.rows.map(r => r.x), y:payload.rows.map(r => r.y), z:payload.rows.map(r => r.z),
  marker:{{size:4.2, color:payload.rows.map(r => r.energy), colorscale:'Viridis', reversescale:true,
           colorbar:{{title:'model node energy'}}, line:{{color:'#06111f', width:0.6}}}},
  text:payload.rows.map(r => `step ${{r.step}}<br>model energy ${{r.energy.toFixed(5)}}<br>surface z ${{r.z.toFixed(5)}}`),
  hoverinfo:'text'
}};
const best = payload.rows[payload.best] || payload.rows[0];
const bestTrace = {{
  type:'scatter3d', mode:'markers', name:'lowest model-energy vertex',
  x:[best.x], y:[best.y], z:[best.z + 0.08],
  marker:{{size:10, color:'#8cff6a', line:{{color:'white', width:1.4}}}},
  text:[`best step ${{best.step}}<br>model energy ${{best.energy.toFixed(5)}}`],
  hoverinfo:'text'
}};
Plotly.newPlot('plot', [surface, ...wallTraces, ...edgeTraces, nodeTrace, bestTrace], {{
  paper_bgcolor:'#05070d', plot_bgcolor:'#05070d',
  legend:{{font:{{color:'white'}}, bgcolor:'rgba(5,7,13,.55)'}},
  scene:{{
    bgcolor:'#05070d',
    xaxis:{{title:'embedding projection 1', color:'white', gridcolor:'#17323a'}},
    yaxis:{{title:'embedding projection 2', color:'white', gridcolor:'#17323a'}},
    zaxis:{{title:'landscape energy', color:'white', gridcolor:'#17323a'}},
    camera:{{eye:{{x:1.55,y:-1.75,z:1.18}}}}
  }},
  margin:{{l:0,r:0,b:0,t:0}}
}}, {{responsive:true}});
</script></body></html>"""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(html, encoding="utf-8")


def write_interactive_reasoning_plot(
    path: np.ndarray,
    energy: np.ndarray,
    output: str | Path,
    *,
    edges: np.ndarray | None = None,
) -> None:
    """Write a dependency-free interactive 3D HTML plot."""

    if edges is None:
        edges = default_branch_merge_edges_np(len(path))
    rows = [
        {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "energy": float(e), "step": int(i)}
        for i, (p, e) in enumerate(zip(path, energy, strict=True))
    ]
    edge_rows = [
        {"src": int(src), "dst": int(dst), "role": role}
        for (src, dst), role in zip(edges, _edge_role_colors(edges, len(path)), strict=False)
        if 0 <= int(src) < len(path) and 0 <= int(dst) < len(path)
    ]
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ToricGT Reasoning Trajectory</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>body{{margin:0;background:#05070d;color:white;font-family:system-ui}}#plot{{width:100vw;height:100vh}}</style>
</head><body><div id="plot"></div><script>
const rows = {rows!r};
const edges = {edge_rows!r};
const nodeTrace = {{
  x: rows.map(r => r.x), y: rows.map(r => r.y), z: rows.map(r => r.z),
  mode: 'markers', type: 'scatter3d', name: 'reasoning vertices',
  marker: {{size: 4, color: rows.map(r => r.energy), colorscale: 'Plasma', colorbar: {{title:'energy'}}}},
  text: rows.map(r => `step ${{r.step}}<br>energy ${{r.energy.toFixed(4)}}`)
}};
const edgeTraces = edges.map((e, idx) => {{
  const a = rows[e.src], b = rows[e.dst];
  return {{
    x: [a.x, b.x], y: [a.y, b.y], z: [a.z, b.z],
    mode: 'lines', type: 'scatter3d', showlegend: idx < 8,
    name: e.role === '#ff4fd8' ? 'branch edge' : (e.role === '#8cff6a' ? 'merge edge' : 'DAG edge'),
    line: {{color: e.role, width: 5}},
    hoverinfo: 'text',
    text: [`${{e.src}} -> ${{e.dst}}`, `${{e.src}} -> ${{e.dst}}`]
  }};
}});
Plotly.newPlot('plot', [...edgeTraces, nodeTrace], {{
  paper_bgcolor:'#05070d', plot_bgcolor:'#05070d',
  scene: {{
    xaxis: {{title:'toric phase', color:'white', gridcolor:'#17323a'}},
    yaxis: {{title:'tropical face', color:'white', gridcolor:'#17323a'}},
    zaxis: {{title:'GFlowNet action', color:'white', gridcolor:'#17323a'}}
  }},
  title: {{text:'Embedding-Space Graph-of-Thought Branch/Merge DAG', font:{{color:'white'}}}}
}}, {{responsive:true}});
</script></body></html>"""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(html, encoding="utf-8")
