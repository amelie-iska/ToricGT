"""Interactive branch/merge reasoning-trajectory simplex visualizations.

This module generates a bounded analysis fixture for the graph-of-thought
reasoning visual audit.  It is intentionally independent of training: the
visual report is used to verify the UI contract and the topological analysis
contract without starting a model run.

All comparisons and topological signatures are computed in the original
embedding coordinates.  PCA coordinates are generated only for browser display.
Vectorized persistent-homology features are computed through GUDHI.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from .gudhi_persistence import (
    GudhiPersistenceConfig,
    build_rips_simplex_tree,
    finite_diagram,
    pairwise_distances,
    persistence_diagrams,
    radius_grid,
    sample_points,
    simplex_sets,
    standardize_points,
    vectorized_diagram_metrics,
)


CSS = """
:root{color-scheme:dark;--bg:#030712;--panel:#07111f;--panel2:#0b1728;--text:#e8fbff;--muted:#9fb1c9;--cyan:#37e8ff;--mag:#ff4fd8;--green:#8cff6a;--amber:#ffd166;--bad:#ff7b9c;--border:rgba(55,232,255,.28)}
body{margin:0;background:radial-gradient(circle at top left,#092238 0,#030712 44rem);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1480px;margin:0 auto;padding:30px 22px 72px}
.hero,.panel,.card{background:linear-gradient(180deg,rgba(11,23,40,.96),rgba(5,13,25,.98));border:1px solid var(--border);border-radius:8px;box-shadow:0 18px 50px rgba(0,0,0,.28)}
.hero{padding:24px;margin-bottom:18px}.panel,.card{padding:16px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
h1{margin:0 0 8px;font-size:29px;letter-spacing:0}h2{margin:0 0 10px;font-size:18px;letter-spacing:0}h3{margin:12px 0 8px;font-size:14px;color:#dff9ff}
p{color:var(--muted);line-height:1.55}.muted{color:var(--muted)}
.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin:12px 0}.control{border:1px solid rgba(145,168,183,.18);border-radius:8px;padding:10px;background:#020713}.control label{display:flex;justify-content:space-between;gap:8px;color:#dff9ff;font-size:13px}.control span{color:var(--cyan);font-variant-numeric:tabular-nums}input[type=range]{width:100%;accent-color:#37e8ff}
.plot{height:660px;border:1px solid rgba(55,232,255,.14);border-radius:8px;overflow:hidden;background:#020713;margin:10px 0}.plot.smallplot{height:420px}
.metric{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid rgba(145,168,183,.14);padding:7px 0}.metric span:first-child{color:var(--muted)}.metric span:last-child{text-align:right;color:white;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}
.pill{display:inline-block;border:1px solid var(--border);border-radius:999px;padding:6px 10px;margin:3px;background:rgba(55,232,255,.07);color:#dff9ff}.ok{color:var(--green)}.bad{color:var(--bad)}.warn{color:var(--amber)}
pre{white-space:pre-wrap;overflow:auto;max-height:520px;background:#020713;border:1px solid rgba(145,168,183,.18);border-radius:8px;padding:14px;color:#dff8ff}.card pre{max-height:300px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border:1px solid rgba(145,168,183,.16);padding:7px 8px;text-align:right}th:first-child,td:first-child{text-align:left;color:var(--muted)}
.scrollbox{max-height:430px;overflow:auto;border:1px solid rgba(145,168,183,.16);border-radius:8px;background:#020713}.scrollbox table{margin:0}.scrollbox th{position:sticky;top:0;background:#081425;z-index:1}
.banner{border:1px solid rgba(255,209,102,.42);border-radius:8px;background:rgba(255,209,102,.06);padding:10px 12px;margin:10px 0;color:#ffe8a8}.banner.bad{border-color:rgba(255,123,156,.55);background:rgba(255,123,156,.07);color:#ffd3df}
.status-badge-row{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.status-badge{display:inline-flex;align-items:center;gap:6px;border-radius:999px;border:1px solid rgba(145,168,183,.24);padding:6px 10px;background:#020713;color:#dff9ff;font-size:12px}.status-badge.pass{border-color:rgba(140,255,106,.55);background:rgba(140,255,106,.08)}.status-badge.fail{border-color:rgba(255,123,156,.55);background:rgba(255,123,156,.08)}.status-badge.info{border-color:rgba(55,232,255,.42);background:rgba(55,232,255,.07)}
.caption{border:1px solid rgba(145,168,183,.16);border-radius:8px;background:#020713;color:#dff9ff;padding:8px 10px;margin:8px 0;font-size:13px}
.detail{border:1px solid rgba(145,168,183,.18);border-radius:8px;background:#020713;padding:10px 12px;margin:10px 0;color:#dff9ff;min-height:54px}.detail strong{color:#fff}.detail code{color:#37e8ff}
"""


@dataclass(frozen=True)
class BranchingTrajectoryConfig:
    seed: int = 117
    embedding_dim: int = 24
    token_count_min: int = 8
    token_count_max: int = 12
    trajectory_levels: int = 18
    branch_lanes: int = 8
    side_branch_length: int = 5
    max_render_edges: int = 0
    max_render_triangles: int = 900
    max_map_image_records: int = 900
    radius_levels: int = 7
    ph_landscape_layers: int = 3
    ph_landscape_resolution: int = 32
    ph_image_resolution: int = 10
    analogy_map_threshold: float = 0.80
    analogy_ph_threshold: float = 0.55
    analogy_step_threshold: float = 0.70
    analogy_weak_map_threshold: float = 0.50
    analogy_weak_ph_threshold: float = 0.15
    analogy_weak_step_threshold: float = 0.30
    analogy_candidate_map_threshold: float = 0.35
    analogy_candidate_ph_threshold: float = 0.05
    analogy_candidate_step_threshold: float = 0.12


def _pca3(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    dim = min(3, vh.shape[0])
    projected = x @ vh[:dim].T
    if dim < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - dim)))
    scale = np.maximum(projected.std(axis=0, keepdims=True), 1e-8)
    return projected / scale


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return 1.0 if float(np.linalg.norm(aa - bb)) <= 1e-12 else 0.0
    return float(np.dot(aa, bb) / denom)


def _branch_dag() -> tuple[list[list[int]], list[tuple[int, int, str]]]:
    levels = [
        [0],
        [1, 2, 3],
        [4, 5, 6, 7, 8, 9],
        [10, 11, 12, 13],
        [14, 15, 16],
        [17, 18, 19, 20],
        [21, 22],
        [23, 24, 25],
        [26],
    ]
    edges = [
        (0, 1, "branch"), (0, 2, "branch"), (0, 3, "branch"),
        (1, 4, "expand"), (1, 5, "expand"), (2, 6, "expand"), (2, 7, "expand"), (3, 8, "expand"), (3, 9, "expand"),
        (4, 10, "merge"), (6, 10, "merge"), (5, 11, "merge"), (7, 11, "merge"), (8, 12, "expand"), (9, 13, "expand"),
        (10, 14, "merge"), (12, 14, "merge"), (11, 15, "merge"), (13, 15, "merge"), (10, 16, "compare"), (11, 16, "compare"),
        (14, 17, "refine"), (15, 18, "refine"), (16, 19, "refine"), (14, 20, "merge"), (15, 20, "merge"), (16, 20, "merge"),
        (17, 21, "merge"), (18, 21, "merge"), (19, 22, "merge"), (20, 22, "merge"),
        (21, 23, "branch"), (21, 24, "branch"), (22, 24, "branch"), (22, 25, "branch"),
        (23, 26, "final_merge"), (24, 26, "final_merge"), (25, 26, "final_merge"),
    ]
    return levels, edges


def _long_branch_dag(cfg: BranchingTrajectoryConfig) -> tuple[list[list[int]], list[tuple[int, int, str]]]:
    """Build a long branch/merge DAG with lane chains and side branches."""

    levels_count = max(9, int(cfg.trajectory_levels))
    lane_count = max(4, int(cfg.branch_lanes))
    side_len = max(2, int(cfg.side_branch_length))
    levels: list[list[int]] = [[0]]
    edges: list[tuple[int, int, str]] = []
    next_id = 1

    prev_main: list[int] = []
    first_level: list[int] = []
    for lane in range(lane_count):
        node = next_id
        next_id += 1
        first_level.append(node)
        edges.append((0, node, "root_branch"))
    levels.append(first_level)
    prev_main = first_level
    active_side: dict[int, tuple[int, int]] = {}

    for level in range(2, levels_count):
        current_level: list[int] = []
        current_main: list[int] = []
        for lane in range(lane_count):
            node = next_id
            next_id += 1
            current_level.append(node)
            current_main.append(node)
            edges.append((prev_main[lane], node, "long_branch_chain"))

        # Extend or merge existing side branches into the current lane node.
        next_active: dict[int, tuple[int, int]] = {}
        for lane, (tip, remaining) in active_side.items():
            if remaining <= 1:
                edges.append((tip, current_main[lane], "side_branch_merge"))
            else:
                side_node = next_id
                next_id += 1
                current_level.append(side_node)
                edges.append((tip, side_node, "side_branch_chain"))
                next_active[lane] = (side_node, remaining - 1)
        active_side = next_active

        # Start new side branches periodically, leaving some phases for long
        # uninterrupted chains so branches are visibly long rather than only
        # short one-step splits.
        if level % 4 == 2:
            for lane in range(lane_count):
                if lane in active_side:
                    continue
                side_node = next_id
                next_id += 1
                current_level.append(side_node)
                edges.append((prev_main[lane], side_node, "side_branch_start"))
                active_side[lane] = (side_node, side_len)

        # Cross-lane joins create graph-of-thought merging behavior without
        # destroying the main lane chains.
        if level % 5 == 0:
            for lane in range(0, lane_count - 1, 2):
                edges.append((prev_main[lane], current_main[lane + 1], "cross_lane_merge"))
                edges.append((prev_main[lane + 1], current_main[lane], "cross_lane_merge"))

        levels.append(current_level)
        prev_main = current_main

    final = next_id
    levels.append([final])
    for node in prev_main:
        edges.append((node, final, "final_merge"))
    for tip, _remaining in active_side.values():
        edges.append((tip, final, "final_side_merge"))
    return levels, edges


def _generate_step_embeddings(cfg: BranchingTrajectoryConfig) -> tuple[list[dict[str, Any]], list[tuple[int, int, str]], np.ndarray]:
    rng = np.random.default_rng(int(cfg.seed))
    if int(cfg.trajectory_levels) > 9 or int(cfg.branch_lanes) > 4:
        levels, edges = _long_branch_dag(cfg)
    else:
        levels, edges = _branch_dag()
    parents: dict[int, list[int]] = {}
    for src, dst, _kind in edges:
        parents.setdefault(dst, []).append(src)
    level_of = {node: level_idx for level_idx, nodes in enumerate(levels) for node in nodes}
    branch_axis = rng.normal(size=(max(4, int(cfg.branch_lanes)), cfg.embedding_dim))
    branch_axis /= np.maximum(np.linalg.norm(branch_axis, axis=1, keepdims=True), 1e-8)
    base_axis = rng.normal(size=(cfg.embedding_dim,))
    base_axis /= max(float(np.linalg.norm(base_axis)), 1e-8)
    embeddings: dict[int, np.ndarray] = {}
    branch_labels: dict[int, int] = {0: 0}
    for level_idx, nodes in enumerate(levels):
        for pos, node in enumerate(nodes):
            node_parents = parents.get(node, [])
            if not node_parents:
                branch = 0
                emb = 0.08 * rng.normal(size=(cfg.embedding_dim,))
            elif len(node_parents) == 1:
                branch = (branch_labels.get(node_parents[0], pos) + pos + 1) % branch_axis.shape[0]
                emb = (
                    embeddings[node_parents[0]]
                    + 0.42 * base_axis
                    + 0.55 * branch_axis[branch]
                    + 0.08 * rng.normal(size=(cfg.embedding_dim,))
                )
            else:
                branch = min(branch_labels.get(parent, pos) for parent in node_parents)
                emb = (
                    np.mean([embeddings[parent] for parent in node_parents], axis=0)
                    + 0.30 * base_axis
                    + 0.10 * branch_axis[branch]
                    + 0.06 * rng.normal(size=(cfg.embedding_dim,))
                )
            embeddings[node] = emb
            branch_labels[node] = branch
    pca = _pca3(np.stack([embeddings[idx] for idx in range(len(embeddings))], axis=0))
    nodes: list[dict[str, Any]] = []
    for idx in range(len(embeddings)):
        level = level_of[idx]
        in_degree = len(parents.get(idx, []))
        out_degree = sum(1 for src, _, _kind in edges if src == idx)
        nll = float(1.95 + 0.10 * level + 0.18 * out_degree + 0.22 * max(0, in_degree - 1) + 0.05 * math.sin(idx))
        nodes.append(
            {
                "id": idx,
                "label": f"r{idx}",
                "level": level,
                "branch": branch_labels[idx],
                "nll": nll,
                "in_degree": int(in_degree),
                "out_degree": int(out_degree),
                "node_kind": "merge" if in_degree > 1 else ("branch" if out_degree > 1 else "chain"),
                "is_merge": bool(in_degree > 1),
                "is_branch": bool(out_degree > 1),
                "embedding": embeddings[idx].astype(float).round(6).tolist(),
                "pca": pca[idx].astype(float).round(6).tolist(),
            }
        )
    return nodes, edges, np.stack([embeddings[idx] for idx in range(len(embeddings))], axis=0)


def _generate_step_tokens(
    cfg: BranchingTrajectoryConfig,
    nodes: list[dict[str, Any]],
    step_embeddings: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rng = np.random.default_rng(int(cfg.seed) + 1000)
    token_clouds: list[np.ndarray] = []
    flat: list[np.ndarray] = []
    for node in nodes:
        count = int(rng.integers(cfg.token_count_min, cfg.token_count_max + 1))
        base = step_embeddings[int(node["id"])]
        local_axis = rng.normal(size=(count, base.shape[0]))
        local_axis -= local_axis.mean(axis=0, keepdims=True)
        scale = np.linspace(0.12, 0.42, count).reshape(-1, 1)
        tokens = base.reshape(1, -1) + scale * local_axis / np.maximum(np.linalg.norm(local_axis, axis=1, keepdims=True), 1e-8)
        token_clouds.append(tokens)
        flat.extend(tokens)
    flat_pca = _pca3(np.stack(flat, axis=0))
    cursor = 0
    steps: list[dict[str, Any]] = []
    token_types = ["premise", "operator", "intermediate", "retrieval", "verifier", "answer"]
    for node, tokens in zip(nodes, token_clouds):
        coords = flat_pca[cursor : cursor + tokens.shape[0]]
        cursor += tokens.shape[0]
        dist = pairwise_distances(tokens)
        positive = dist[dist > 1e-10]
        max_radius = float(np.quantile(positive, 0.72)) if positive.size else 1.0
        radii = np.linspace(max_radius / max(2, cfg.radius_levels), max_radius, cfg.radius_levels)
        triangles = _diameter_triangles(dist)
        token_records = []
        for tok_idx in range(tokens.shape[0]):
            token_type = token_types[(tok_idx + int(node["id"])) % len(token_types)]
            token_nll = float(float(node["nll"]) + 0.04 * tok_idx + 0.10 * math.sin(tok_idx + int(node["id"])))
            logprob = -token_nll * math.log(2.0)
            entropy = float(0.42 + 0.06 * (tok_idx % 5) + 0.03 * int(node["is_branch"]) + 0.04 * int(node["is_merge"]))
            token_records.append(
                {
                    "id": tok_idx,
                    "global_token_id": int(cursor - tokens.shape[0] + tok_idx),
                    "text": f"{token_type}_{int(node['id']):03d}_{tok_idx:02d}",
                    "token_type": token_type,
                    "source_step": int(node["id"]),
                    "source_step_label": str(node["label"]),
                    "decode_order": tok_idx,
                    "nll": token_nll,
                    "logprob": float(logprob),
                    "entropy": entropy,
                    "rank": int(1 + (tok_idx * 7 + int(node["id"])) % 64),
                    "embedding": tokens[tok_idx].astype(float).round(6).tolist(),
                    "pca": coords[tok_idx].astype(float).round(6).tolist(),
                }
            )
        steps.append(
            {
                "step_index": int(node["id"]),
                "level": int(node["level"]),
                "label": str(node["label"]),
                "radius_values": radii.astype(float).round(6).tolist(),
                "tokens": token_records,
                "distances": dist.astype(float).round(6).tolist(),
                "triangles": triangles,
                "decode_edges": [[idx, idx + 1, idx + 1] for idx in range(tokens.shape[0] - 1)],
            }
        )
    return steps, np.stack(flat, axis=0)


def _level_assignment_from_edges(node_count: int, edges: np.ndarray) -> list[int]:
    """Assign coarse reasoning levels from directed graph edges.

    Checkpoint geometry payloads contain graph edges but not always explicit
    causal ranks.  This deterministic BFS-style assignment keeps the rendered
    trajectory tree-like when edges are mostly causal, and falls back to index
    order for disconnected or cyclic pieces.
    """

    if node_count <= 0:
        return []
    outgoing: dict[int, list[int]] = {idx: [] for idx in range(node_count)}
    incoming_count = [0 for _ in range(node_count)]
    for src, dst in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        if 0 <= int(src) < node_count and 0 <= int(dst) < node_count and int(src) != int(dst):
            outgoing[int(src)].append(int(dst))
            incoming_count[int(dst)] += 1
    roots = [idx for idx, degree in enumerate(incoming_count) if degree == 0]
    if not roots:
        roots = [0]
    levels = [-1 for _ in range(node_count)]
    queue: list[int] = []
    for root in roots:
        levels[root] = 0
        queue.append(root)
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for dst in sorted(set(outgoing.get(node, []))):
            next_level = levels[node] + 1
            if levels[dst] < 0 or next_level < levels[dst]:
                levels[dst] = next_level
                queue.append(dst)
    for idx, level in enumerate(levels):
        if level < 0:
            levels[idx] = max(levels[:idx] + [0]) + 1 if idx else 0
    return [int(level) for level in levels]


def _payload_nodes_from_embeddings(
    hidden: np.ndarray,
    nll: np.ndarray,
    edges: np.ndarray,
    *,
    label_prefix: str = "r",
) -> list[dict[str, Any]]:
    x = np.asarray(hidden, dtype=np.float64)
    pca = _pca3(x)
    node_count = int(x.shape[0])
    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2) if np.asarray(edges).size else np.zeros((0, 2), dtype=np.int64)
    valid = (
        (edge_array[:, 0] >= 0)
        & (edge_array[:, 1] >= 0)
        & (edge_array[:, 0] < node_count)
        & (edge_array[:, 1] < node_count)
        & (edge_array[:, 0] != edge_array[:, 1])
    )
    edge_array = edge_array[valid]
    levels = _level_assignment_from_edges(node_count, edge_array)
    in_degree = [0 for _ in range(node_count)]
    out_degree = [0 for _ in range(node_count)]
    for src, dst in edge_array:
        out_degree[int(src)] += 1
        in_degree[int(dst)] += 1
    nll_values = np.asarray(nll, dtype=np.float64).reshape(-1)
    if nll_values.shape[0] < node_count:
        nll_values = np.pad(nll_values, (0, node_count - nll_values.shape[0]), constant_values=float(np.nanmean(nll_values)) if nll_values.size else 0.0)
    nll_values = np.nan_to_num(nll_values[:node_count], nan=float(np.nanmean(nll_values[:node_count])) if node_count else 0.0)
    nodes: list[dict[str, Any]] = []
    for idx in range(node_count):
        nodes.append(
            {
                "id": int(idx),
                "label": f"{label_prefix}{idx}",
                "level": int(levels[idx]),
                "branch": int(idx % 8),
                "nll": float(nll_values[idx]),
                "in_degree": int(in_degree[idx]),
                "out_degree": int(out_degree[idx]),
                "node_kind": "merge" if in_degree[idx] > 1 else ("branch" if out_degree[idx] > 1 else "chain"),
                "is_merge": bool(in_degree[idx] > 1),
                "is_branch": bool(out_degree[idx] > 1),
                "embedding": x[idx].astype(float).round(6).tolist(),
                "pca": pca[idx].astype(float).round(6).tolist(),
            }
        )
    return nodes


def _real_payload_step_tokens(
    cfg: BranchingTrajectoryConfig,
    nodes: list[dict[str, Any]],
    hidden: np.ndarray,
    nll: np.ndarray,
    edges: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Build per-step token subcomplexes from actual saved embedding vectors."""

    x = np.asarray(hidden, dtype=np.float64)
    node_count = int(x.shape[0])
    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2) if np.asarray(edges).size else np.zeros((0, 2), dtype=np.int64)
    neighbors: dict[int, set[int]] = {idx: {idx} for idx in range(node_count)}
    for src, dst in edge_array:
        if 0 <= int(src) < node_count and 0 <= int(dst) < node_count and int(src) != int(dst):
            neighbors[int(src)].add(int(dst))
            neighbors[int(dst)].add(int(src))
    dist = pairwise_distances(x)
    nll_values = np.asarray(nll, dtype=np.float64).reshape(-1)
    if nll_values.shape[0] < node_count:
        fill = float(np.nanmean(nll_values)) if nll_values.size else 0.0
        nll_values = np.pad(nll_values, (0, node_count - nll_values.shape[0]), constant_values=fill)
    nll_values = np.nan_to_num(nll_values[:node_count], nan=float(np.nanmean(nll_values[:node_count])) if node_count else 0.0)
    steps: list[dict[str, Any]] = []
    flat_vectors: list[np.ndarray] = []
    step_token_sources: list[list[int]] = []
    for node in nodes:
        idx = int(node["id"])
        selected = set(neighbors.get(idx, {idx}))
        target_count = min(node_count, max(int(cfg.token_count_min), min(int(cfg.token_count_max), int(cfg.token_count_min) + len(selected))))
        nearest = np.argsort(dist[idx]).astype(int).tolist()
        for candidate in nearest:
            selected.add(int(candidate))
            if len(selected) >= target_count:
                break
        ordered = sorted(selected, key=lambda item: (dist[idx, item], item))[:target_count]
        step_token_sources.append(ordered)
        flat_vectors.extend([x[source] for source in ordered])
    flat_pca = _pca3(np.stack(flat_vectors, axis=0)) if flat_vectors else np.zeros((0, 3), dtype=np.float64)
    cursor = 0
    token_types = ["self", "graph_neighbor", "nearest_embedding", "retrieval_context", "verifier_context", "answer_context"]
    for node, sources in zip(nodes, step_token_sources):
        coords = flat_pca[cursor : cursor + len(sources)]
        local_points = np.asarray([x[source] for source in sources], dtype=np.float64)
        local_dist = pairwise_distances(local_points)
        positive = local_dist[local_dist > 1e-10]
        max_radius = float(np.quantile(positive, 0.72)) if positive.size else 1.0
        radii = np.linspace(max_radius / max(2, cfg.radius_levels), max_radius, cfg.radius_levels)
        triangles = _diameter_triangles(local_dist)
        token_records: list[dict[str, Any]] = []
        source_node = int(node["id"])
        for local_idx, source in enumerate(sources):
            if int(source) == source_node:
                token_type = "self"
            elif int(source) in neighbors.get(source_node, set()):
                token_type = "graph_neighbor"
            else:
                token_type = token_types[(local_idx + source_node) % len(token_types)]
            token_nll = float(nll_values[int(source)])
            token_records.append(
                {
                    "id": int(local_idx),
                    "global_token_id": int(source),
                    "text": f"{token_type}_node_{int(source):03d}",
                    "token_type": token_type,
                    "source_step": int(node["id"]),
                    "source_step_label": str(node["label"]),
                    "decode_order": int(local_idx),
                    "nll": token_nll,
                    "logprob": float(-token_nll * math.log(2.0)),
                    "entropy": float(0.35 + 0.04 * local_idx + 0.02 * int(node["is_branch"]) + 0.03 * int(node["is_merge"])),
                    "rank": int(1 + int(source) % 64),
                    "embedding": x[int(source)].astype(float).round(6).tolist(),
                    "pca": coords[local_idx].astype(float).round(6).tolist(),
                }
            )
        cursor += len(sources)
        steps.append(
            {
                "step_index": int(node["id"]),
                "level": int(node["level"]),
                "label": str(node["label"]),
                "radius_values": radii.astype(float).round(6).tolist(),
                "tokens": token_records,
                "distances": local_dist.astype(float).round(6).tolist(),
                "triangles": triangles,
                "decode_edges": [[idx, idx + 1, idx + 1] for idx in range(len(token_records) - 1)],
                "token_source_node_ids": [int(source) for source in sources],
            }
        )
    return steps, np.asarray(flat_vectors, dtype=np.float64)


def _diameter_triangles(dist: np.ndarray, *, limit: int | None = None) -> list[list[float]]:
    n = int(dist.shape[0])
    triangles: list[list[float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triangles.append([i, j, k, float(max(dist[i, j], dist[i, k], dist[j, k]))])
    if limit is not None and len(triangles) > int(limit):
        triangles = sorted(triangles, key=lambda row: row[3])[: int(limit)]
    return triangles


def _ph_feature_signature(points: np.ndarray, cfg: BranchingTrajectoryConfig) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    ph_cfg = GudhiPersistenceConfig(
        max_points=64,
        max_dimension=2,
        radius_quantile=0.72,
        num_radii=cfg.radius_levels,
        landscape_resolution=cfg.ph_landscape_resolution,
        landscape_layers=cfg.ph_landscape_layers,
        image_resolution=cfg.ph_image_resolution,
        macaulay2_resolutions=False,
    )
    x = standardize_points(sample_points(points, int(ph_cfg.max_points)))
    radii = radius_grid(x, ph_cfg)
    tree = build_rips_simplex_tree(x, float(radii[-1]), max_dimension=2)
    diagrams = persistence_diagrams(tree, max_dimension=2)
    features: dict[str, np.ndarray] = {}
    metrics: dict[str, float] = {
        "points": float(x.shape[0]),
        "num_simplices": float(tree.num_simplices()),
        "max_radius": float(radii[-1]),
    }
    for dim in range(3):
        vectorized = vectorized_diagram_metrics(diagrams[dim], ph_cfg)
        for name in ("landscape", "persistence_image", "silhouette", "entropy_vector"):
            features[f"h{dim}_{name}"] = np.asarray(vectorized[name], dtype=np.float64)
            metrics[f"h{dim}_{name}_norm"] = float(vectorized[f"{name}_norm"])
        metrics[f"h{dim}_interval_count"] = float(vectorized["count"])
        metrics[f"h{dim}_persistence_entropy"] = float(vectorized["persistence_entropy"])
    return features, metrics


def _feature_similarities(source: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {"landscape": [], "persistence_image": [], "silhouette": [], "entropy_vector": []}
    for key, source_vec in source.items():
        if key not in target:
            continue
        family = key.split("_", 1)[1]
        grouped.setdefault(family, []).append(_cosine(source_vec, target[key]))
    return {family: float(np.mean(values)) if values else 0.0 for family, values in grouped.items()}


def _max_lifetime(diagram: np.ndarray) -> float:
    diag = finite_diagram(diagram)
    if diag.size == 0:
        return 0.0
    return float(np.max(np.maximum(diag[:, 1] - diag[:, 0], 0.0)))


def _diagram_lifetime_l1(source: np.ndarray, target: np.ndarray) -> float:
    source_life = np.sort(np.maximum(finite_diagram(source)[:, 1] - finite_diagram(source)[:, 0], 0.0))
    target_life = np.sort(np.maximum(finite_diagram(target)[:, 1] - finite_diagram(target)[:, 0], 0.0))
    length = max(len(source_life), len(target_life))
    if length == 0:
        return 0.0
    source_pad = np.pad(source_life, (0, length - len(source_life)))
    target_pad = np.pad(target_life, (0, length - len(target_life)))
    return float(np.mean(np.abs(source_pad - target_pad)))


def _diagram_distance_summary(source_bundle: dict[str, Any], target_bundle: dict[str, Any]) -> dict[str, Any]:
    """Compute exact GUDHI bottleneck distances and optional Wasserstein distances.

    Wasserstein distance in GUDHI requires the optional POT dependency.  When it
    is absent, the payload records that status and exposes only deterministic
    finite-diagram lifetime summaries, without labeling them as Wasserstein.
    """

    try:
        import gudhi  # type: ignore
    except Exception as exc:  # pragma: no cover - GUDHI is required in tests/env
        return {"backend": "GUDHI unavailable", "error": f"{type(exc).__name__}: {exc}", "by_dimension": {}}

    try:
        from gudhi.wasserstein import wasserstein_distance  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional POT
        wasserstein_distance = None
        wasserstein_status = f"unavailable: {type(exc).__name__}: {exc}"
    else:
        wasserstein_status = "available"

    by_dimension: dict[str, dict[str, Any]] = {}
    for dim in range(3):
        key = str(dim)
        source = np.asarray(source_bundle["rendered"][key]["diagram"], dtype=np.float64)
        target = np.asarray(target_bundle["rendered"][key]["diagram"], dtype=np.float64)
        if source.size == 0:
            source = source.reshape(0, 2)
        if target.size == 0:
            target = target.reshape(0, 2)
        if source.shape[0] == 0 and target.shape[0] == 0:
            bottleneck = 0.0
        elif source.shape[0] == 0 or target.shape[0] == 0:
            bottleneck = max(_max_lifetime(source), _max_lifetime(target)) / 2.0
        else:
            bottleneck = float(gudhi.bottleneck_distance(source, target))
        if wasserstein_distance is None:
            wasserstein = None
        else:
            wasserstein = float(wasserstein_distance(source, target, order=1.0, internal_p=2.0))
        by_dimension[key] = {
            "bottleneck_distance": bottleneck,
            "wasserstein_distance": wasserstein,
            "wasserstein_status": wasserstein_status,
            "finite_lifetime_l1": _diagram_lifetime_l1(source, target),
            "source_interval_count": int(source.shape[0]),
            "memory_interval_count": int(target.shape[0]),
        }
    return {
        "backend": "GUDHI",
        "bottleneck_backend": "gudhi.bottleneck_distance",
        "wasserstein_backend": "gudhi.wasserstein.wasserstein_distance" if wasserstein_distance is not None else wasserstein_status,
        "by_dimension": by_dimension,
    }


def _simplex_diameter(dist: np.ndarray, simplex: tuple[int, ...]) -> float:
    if len(simplex) <= 1:
        return 0.0
    return float(max(dist[i, j] for pos, i in enumerate(simplex) for j in simplex[pos + 1 :]))


def _simplex_tree_payload(
    points: np.ndarray,
    display_points: np.ndarray,
    radii: np.ndarray,
    *,
    labels: list[str] | None = None,
    levels: list[int] | None = None,
    nll: list[float] | None = None,
    prefix: str = "v",
    max_edges: int | None = None,
    max_triangles: int = 900,
) -> dict[str, Any]:
    x = np.asarray(points, dtype=np.float64)
    display = np.asarray(display_points, dtype=np.float64)
    tree = build_rips_simplex_tree(x, float(radii[-1]), max_dimension=2)
    sets = simplex_sets(tree, max_dimension=2)
    dist = pairwise_distances(x)

    def birth_idx(simplex: tuple[int, ...]) -> int:
        diameter = _simplex_diameter(dist, simplex)
        idx = int(np.searchsorted(radii, diameter, side="left"))
        return min(max(idx, 0), len(radii) - 1)

    vertices = []
    for idx in range(x.shape[0]):
        vertices.append(
            {
                "id": int(idx),
                "label": labels[idx] if labels and idx < len(labels) else f"{prefix}{idx}",
                "level": int(levels[idx]) if levels and idx < len(levels) else int(idx),
                "nll": float(nll[idx]) if nll and idx < len(nll) else 0.0,
                "pca": display[idx].astype(float).round(6).tolist(),
            }
        )
    edges = []
    for edge in sets.get(1, []):
        diameter = _simplex_diameter(dist, edge)
        edges.append(
            {
                "simplex": [int(edge[0]), int(edge[1])],
                "diameter": float(diameter),
                "birth_radius_index": birth_idx(edge),
                "birth_radius": float(radii[birth_idx(edge)]),
            }
        )
    edge_count_exact = len(edges)
    triangles = []
    for tri in sets.get(2, []):
        diameter = _simplex_diameter(dist, tri)
        triangles.append(
            {
                "simplex": [int(tri[0]), int(tri[1]), int(tri[2])],
                "diameter": float(diameter),
                "birth_radius_index": birth_idx(tri),
                "birth_radius": float(radii[birth_idx(tri)]),
            }
        )
    triangle_count_exact = len(triangles)
    if len(triangles) > int(max_triangles):
        triangles = sorted(triangles, key=lambda row: row["diameter"])[: int(max_triangles)]
    return {
        "simplex_tree_backend": "gudhi.RipsComplex.create_simplex_tree",
        "radius_values": radii.astype(float).round(6).tolist(),
        "num_vertices": int(len(vertices)),
        "num_edges": int(edge_count_exact),
        "num_triangles": int(triangle_count_exact),
        "rendered_edge_count": int(len(edges)),
        "rendered_triangle_count": int(len(triangles)),
        "render_limits": {"visible_edges": "all", "max_triangles": int(max_triangles)},
        "visible_edge_policy": "all_visible_edges",
        "vertices": vertices,
        "edges": edges,
        "triangles": triangles,
    }


def _betti_curve_from_diagram(diagram: np.ndarray, grid: np.ndarray) -> list[float]:
    diag = finite_diagram(diagram)
    if diag.size == 0:
        return [0.0 for _ in grid]
    out: list[float] = []
    for value in grid:
        active = np.logical_and(diag[:, 0] <= value, value < diag[:, 1])
        out.append(float(active.sum()))
    return out


def _lifetime_histogram(diagram: np.ndarray, bins: int = 12) -> dict[str, list[float]]:
    diag = finite_diagram(diagram)
    if diag.size == 0:
        return {"bin_edges": [0.0, 1.0], "counts": [0.0]}
    lifetimes = np.maximum(diag[:, 1] - diag[:, 0], 0.0)
    upper = max(float(lifetimes.max()), 1e-8)
    counts, edges = np.histogram(lifetimes, bins=int(bins), range=(0.0, upper))
    return {"bin_edges": edges.astype(float).round(6).tolist(), "counts": counts.astype(float).tolist()}


def _ph_feature_bundle(points: np.ndarray, cfg: BranchingTrajectoryConfig) -> dict[str, Any]:
    ph_cfg = GudhiPersistenceConfig(
        max_points=64,
        max_dimension=2,
        radius_quantile=0.72,
        num_radii=cfg.radius_levels,
        landscape_resolution=cfg.ph_landscape_resolution,
        landscape_layers=cfg.ph_landscape_layers,
        image_resolution=cfg.ph_image_resolution,
        macaulay2_resolutions=False,
    )
    x = standardize_points(sample_points(points, int(ph_cfg.max_points)))
    radii = radius_grid(x, ph_cfg)
    tree = build_rips_simplex_tree(x, float(radii[-1]), max_dimension=2)
    diagrams = persistence_diagrams(tree, max_dimension=2)
    grid = np.linspace(0.0, float(radii[-1]), int(cfg.ph_landscape_resolution))
    features: dict[str, np.ndarray] = {}
    rendered: dict[str, Any] = {}
    metrics: dict[str, float] = {
        "points": float(x.shape[0]),
        "num_simplices": float(tree.num_simplices()),
        "max_radius": float(radii[-1]),
    }
    for dim in range(3):
        diag = finite_diagram(diagrams[dim])
        vectorized = vectorized_diagram_metrics(diagrams[dim], ph_cfg)
        landscape = np.asarray(vectorized["landscape"], dtype=np.float64).reshape(
            int(cfg.ph_landscape_layers), int(cfg.ph_landscape_resolution)
        )
        persistence_image = np.asarray(vectorized["persistence_image"], dtype=np.float64).reshape(
            int(cfg.ph_image_resolution), int(cfg.ph_image_resolution)
        )
        silhouette = np.asarray(vectorized["silhouette"], dtype=np.float64)
        entropy_vector = np.asarray(vectorized["entropy_vector"], dtype=np.float64)
        for name, arr in {
            "landscape": landscape.reshape(-1),
            "persistence_image": persistence_image.reshape(-1),
            "silhouette": silhouette.reshape(-1),
            "entropy_vector": entropy_vector.reshape(-1),
        }.items():
            features[f"h{dim}_{name}"] = arr
            metrics[f"h{dim}_{name}_norm"] = float(vectorized[f"{name}_norm"])
        metrics[f"h{dim}_interval_count"] = float(vectorized["count"])
        metrics[f"h{dim}_persistence_entropy"] = float(vectorized["persistence_entropy"])
        rendered[str(dim)] = {
            "diagram": diag.astype(float).round(6).tolist(),
            "landscape": landscape.astype(float).round(6).tolist(),
            "persistence_image": persistence_image.astype(float).round(6).tolist(),
            "silhouette": silhouette.astype(float).round(6).tolist(),
            "entropy_vector": entropy_vector.astype(float).round(6).tolist(),
            "betti_curve": _betti_curve_from_diagram(diagrams[dim], grid),
            "lifetime_histogram": _lifetime_histogram(diagrams[dim]),
            "persistence_entropy": float(vectorized["persistence_entropy"]),
            "interval_count": float(vectorized["count"]),
        }
    return {
        "backend": "GUDHI",
        "radius_grid": radii.astype(float).round(6).tolist(),
        "feature_grid": grid.astype(float).round(6).tolist(),
        "diagram_count_by_dimension": {str(dim): int(len(finite_diagram(diagrams[dim]))) for dim in range(3)},
        "rendered": rendered,
        "metrics": metrics,
        "_features": features,
    }


def _simplicial_map_payload(
    source: np.ndarray,
    target: np.ndarray,
    source_tree: dict[str, Any],
    target_tree: dict[str, Any],
    radius: float,
    max_image_records: int = 900,
) -> dict[str, Any]:
    source_tree_exact = build_rips_simplex_tree(source, float(radius), max_dimension=2)
    target_tree_exact = build_rips_simplex_tree(target, float(radius), max_dimension=2)
    source_sets = simplex_sets(source_tree_exact, max_dimension=2)
    target_sets = simplex_sets(target_tree_exact, max_dimension=2)
    target_edges = set(target_sets.get(1, []))
    target_triangles = set(target_sets.get(2, []))
    cross = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=-1)
    vertex_map_array = np.argmin(cross, axis=1).astype(int)
    vertex_map = vertex_map_array.tolist()
    vertex_map_distances = cross[np.arange(source.shape[0]), vertex_map_array].astype(float).tolist()

    def mapped(simplex: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(sorted(set(vertex_map[idx] for idx in simplex)))

    edge_images = []
    for edge in source_sets.get(1, []):
        image = mapped(edge)
        status = "collapsed" if len(image) <= 1 else ("valid" if image in target_edges else "invalid")
        edge_images.append({"source": list(edge), "image": list(image), "status": status})
    triangle_images = []
    for tri in source_sets.get(2, []):
        image = mapped(tri)
        status = "collapsed" if len(image) <= 2 else ("valid" if image in target_triangles else "invalid")
        triangle_images.append({"source": list(tri), "image": list(image), "status": status})
    valid_edges = sum(1 for item in edge_images if item["status"] in {"valid", "collapsed"})
    valid_triangles = sum(1 for item in triangle_images if item["status"] in {"valid", "collapsed"})
    edge_status_counts = {
        "valid": int(sum(1 for item in edge_images if item["status"] == "valid")),
        "collapsed": int(sum(1 for item in edge_images if item["status"] == "collapsed")),
        "invalid": int(sum(1 for item in edge_images if item["status"] == "invalid")),
    }
    triangle_status_counts = {
        "valid": int(sum(1 for item in triangle_images if item["status"] == "valid")),
        "collapsed": int(sum(1 for item in triangle_images if item["status"] == "collapsed")),
        "invalid": int(sum(1 for item in triangle_images if item["status"] == "invalid")),
    }
    total = len(edge_images) + len(triangle_images)
    valid = valid_edges + valid_triangles
    vertex_dist = np.asarray(vertex_map_distances, dtype=np.float64)
    rendered_edge_images = edge_images[: int(max_image_records)]
    rendered_triangle_images = triangle_images[: int(max_image_records)]
    return {
        "source_simplex_tree_payload": source_tree,
        "memory_simplex_tree_payload": target_tree,
        "vertex_map": vertex_map,
        "vertex_map_distances": vertex_map_distances,
        "radius": float(radius),
        "source_edges": len(edge_images),
        "source_triangles": len(triangle_images),
        "valid_edges": int(valid_edges),
        "valid_triangles": int(valid_triangles),
        "invalid_edges": int(edge_status_counts["invalid"]),
        "invalid_triangles": int(triangle_status_counts["invalid"]),
        "collapsed_edges": int(edge_status_counts["collapsed"]),
        "collapsed_triangles": int(triangle_status_counts["collapsed"]),
        "edge_status_counts": edge_status_counts,
        "triangle_status_counts": triangle_status_counts,
        "valid_fraction": float(valid / max(1, total)),
        "vertex_distance_mean": float(vertex_dist.mean()) if vertex_dist.size else 0.0,
        "vertex_distance_max": float(vertex_dist.max()) if vertex_dist.size else 0.0,
        "map_summary": {
            "source_vertices": int(source.shape[0]),
            "target_vertices": int(target.shape[0]),
            "source_edges": int(len(edge_images)),
            "source_triangles": int(len(triangle_images)),
            "valid_edges": int(edge_status_counts["valid"]),
            "collapsed_edges": int(edge_status_counts["collapsed"]),
            "invalid_edges": int(edge_status_counts["invalid"]),
            "valid_triangles": int(triangle_status_counts["valid"]),
            "collapsed_triangles": int(triangle_status_counts["collapsed"]),
            "invalid_triangles": int(triangle_status_counts["invalid"]),
            "valid_or_collapsed_fraction": float(valid / max(1, total)),
            "vertex_distance_mean": float(vertex_dist.mean()) if vertex_dist.size else 0.0,
            "vertex_distance_max": float(vertex_dist.max()) if vertex_dist.size else 0.0,
        },
        "rendered_edge_image_count": int(len(rendered_edge_images)),
        "rendered_triangle_image_count": int(len(rendered_triangle_images)),
        "map_image_record_limit": int(max_image_records),
        "edge_images": rendered_edge_images,
        "triangle_images": rendered_triangle_images,
    }


def _analogy_payload(
    cfg: BranchingTrajectoryConfig,
    step_embeddings: np.ndarray,
    steps: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    rng = np.random.default_rng(int(cfg.seed) + 2000)
    rotation = np.eye(step_embeddings.shape[1])
    noise = 0.035 * rng.normal(size=step_embeddings.shape)
    target = step_embeddings @ rotation + noise
    comparison_points = np.vstack([step_embeddings, target])
    display = _pca3(comparison_points)
    source_display = display[: step_embeddings.shape[0]]
    target_display = display[step_embeddings.shape[0] :]
    target_display = target_display + np.asarray([3.3, 0.0, 0.0], dtype=np.float64)
    dist = pairwise_distances(step_embeddings)
    positive = dist[dist > 1e-10]
    radius = float(np.quantile(positive, 0.70)) if positive.size else 1.0
    radii = np.linspace(radius / max(2, cfg.radius_levels), radius, cfg.radius_levels)
    labels = [str(node["label"]) for node in nodes]
    levels = [int(node["level"]) for node in nodes]
    nll = [float(node["nll"]) for node in nodes]
    source_tree = _simplex_tree_payload(
        step_embeddings,
        source_display,
        radii,
        labels=labels,
        levels=levels,
        nll=nll,
        prefix="s",
        max_edges=cfg.max_render_edges,
        max_triangles=cfg.max_render_triangles,
    )
    target_tree = _simplex_tree_payload(
        target,
        target_display,
        radii,
        labels=[f"m{idx}" for idx in range(target.shape[0])],
        levels=levels,
        nll=nll,
        prefix="m",
        max_edges=cfg.max_render_edges,
        max_triangles=cfg.max_render_triangles,
    )
    full_map = _simplicial_map_payload(
        step_embeddings,
        target,
        source_tree,
        target_tree,
        radius,
        max_image_records=cfg.max_map_image_records,
    )
    source_bundle = _ph_feature_bundle(step_embeddings, cfg)
    target_bundle = _ph_feature_bundle(target, cfg)
    feature_similarity = _feature_similarities(source_bundle["_features"], target_bundle["_features"])
    ph_distance_summary = _diagram_distance_summary(source_bundle, target_bundle)
    ph_cosine = float(np.mean(list(feature_similarity.values()))) if feature_similarity else 0.0
    ph_values = [float(value) for value in feature_similarity.values()]
    ph_best = float(max(ph_values)) if ph_values else 0.0
    ph_pass_count = int(sum(value >= cfg.analogy_weak_ph_threshold for value in ph_values))
    ph_gate_score = float(max(ph_cosine, ph_best))
    step_validities: list[float] = []
    for step in steps[: min(12, len(steps))]:
        points = np.asarray([tok["embedding"] for tok in step["tokens"]], dtype=np.float64)
        target_points = points + 0.025 * rng.normal(size=points.shape)
        radius_step = float(step["radius_values"][min(len(step["radius_values"]) - 1, max(1, len(step["radius_values"]) // 2))])
        local_radii = np.linspace(radius_step / max(2, cfg.radius_levels), radius_step, cfg.radius_levels)
        local_display = _pca3(np.vstack([points, target_points]))
        local_source_tree = _simplex_tree_payload(
            points,
            local_display[: points.shape[0]],
            local_radii,
            max_edges=cfg.max_render_edges,
            max_triangles=cfg.max_render_triangles,
        )
        local_target_tree = _simplex_tree_payload(
            target_points,
            local_display[points.shape[0] :],
            local_radii,
            max_edges=cfg.max_render_edges,
            max_triangles=cfg.max_render_triangles,
        )
        step_validities.append(
            _simplicial_map_payload(
                points,
                target_points,
                local_source_tree,
                local_target_tree,
                radius_step,
                max_image_records=cfg.max_map_image_records,
            )["valid_fraction"]
        )
    step_map_mean = float(np.mean(step_validities)) if step_validities else 0.0
    strong = (
        full_map["valid_fraction"] >= cfg.analogy_map_threshold
        and ph_cosine >= cfg.analogy_ph_threshold
        and step_map_mean >= cfg.analogy_step_threshold
    )
    weak = (
        full_map["valid_fraction"] >= cfg.analogy_weak_map_threshold
        and step_map_mean >= cfg.analogy_weak_step_threshold
        and (ph_cosine >= cfg.analogy_weak_ph_threshold or ph_pass_count >= 2)
    )
    candidate = (
        full_map["valid_fraction"] >= cfg.analogy_candidate_map_threshold
        and step_map_mean >= cfg.analogy_candidate_step_threshold
        and ph_gate_score >= cfg.analogy_candidate_ph_threshold
    )
    if strong:
        analogy_status = "strong_analogy"
    elif weak:
        analogy_status = "weak_analogy"
    elif candidate:
        analogy_status = "candidate_analogy"
    else:
        analogy_status = "no_analogy"
    emitted = analogy_status != "no_analogy"
    confidence_score = float(
        0.45 * full_map["valid_fraction"]
        + 0.30 * step_map_mean
        + 0.25 * max(0.0, min(1.0, ph_gate_score))
    )

    def condition(label: str, score: float, threshold: float, passed: bool, *, detail: str = "") -> dict[str, Any]:
        return {
            "label": label,
            "score": float(score),
            "threshold": float(threshold),
            "passed": bool(passed),
            "detail": detail,
        }

    strong_conditions = [
        condition(
            "full trajectory simplex-map",
            float(full_map["valid_fraction"]),
            cfg.analogy_map_threshold,
            float(full_map["valid_fraction"]) >= cfg.analogy_map_threshold,
        ),
        condition(
            "vectorized PH mean",
            ph_cosine,
            cfg.analogy_ph_threshold,
            ph_cosine >= cfg.analogy_ph_threshold,
        ),
        condition(
            "step simplex-map mean",
            step_map_mean,
            cfg.analogy_step_threshold,
            step_map_mean >= cfg.analogy_step_threshold,
        ),
    ]
    weak_ph_passed = bool(ph_cosine >= cfg.analogy_weak_ph_threshold or ph_pass_count >= 2)
    weak_conditions = [
        condition(
            "full trajectory simplex-map",
            float(full_map["valid_fraction"]),
            cfg.analogy_weak_map_threshold,
            float(full_map["valid_fraction"]) >= cfg.analogy_weak_map_threshold,
        ),
        condition(
            "step simplex-map mean",
            step_map_mean,
            cfg.analogy_weak_step_threshold,
            step_map_mean >= cfg.analogy_weak_step_threshold,
        ),
        condition(
            "weak PH evidence",
            ph_gate_score,
            cfg.analogy_weak_ph_threshold,
            weak_ph_passed,
            detail=f"PH mean {ph_cosine:.4f} >= {cfg.analogy_weak_ph_threshold:.4f} or at least 2 PH feature families pass; observed {ph_pass_count}",
        ),
    ]
    candidate_conditions = [
        condition(
            "full trajectory simplex-map",
            float(full_map["valid_fraction"]),
            cfg.analogy_candidate_map_threshold,
            float(full_map["valid_fraction"]) >= cfg.analogy_candidate_map_threshold,
        ),
        condition(
            "step simplex-map mean",
            step_map_mean,
            cfg.analogy_candidate_step_threshold,
            step_map_mean >= cfg.analogy_candidate_step_threshold,
        ),
        condition(
            "PH gate",
            ph_gate_score,
            cfg.analogy_candidate_ph_threshold,
            ph_gate_score >= cfg.analogy_candidate_ph_threshold,
            detail="candidate tier uses PH gate = max(vectorized PH mean, best PH feature-family cosine)",
        ),
    ]
    failed_strong = [item for item in strong_conditions if not item["passed"]]
    if strong:
        narrative = "Strong analogy emitted: all strong map, vectorized-PH, and step-map gates passed."
    elif weak:
        failed = "; ".join(
            f"{item['label']} {item['score']:.4f} < {item['threshold']:.4f}" for item in failed_strong
        )
        narrative = (
            "Weak analogy emitted: weak tier passed, but strong tier failed"
            + (f" because {failed}." if failed else ".")
        )
    elif candidate:
        narrative = "Candidate analogy emitted: candidate tier passed, but weak/strong evidence was incomplete."
    else:
        narrative = "No analogy emitted: candidate tier failed, so retrieval should not treat this memory as an analogy."

    return {
        "analogy_emitted": bool(emitted),
        "decision_rule": {
            "strong": {
                "full_reasoning_trajectory_simplex_map_threshold": cfg.analogy_map_threshold,
                "vectorized_persistence_similarity_threshold": cfg.analogy_ph_threshold,
                "step_simplicial_map_mean_threshold": cfg.analogy_step_threshold,
            },
            "weak": {
                "full_reasoning_trajectory_simplex_map_threshold": cfg.analogy_weak_map_threshold,
                "vectorized_persistence_similarity_threshold": cfg.analogy_weak_ph_threshold,
                "step_simplicial_map_mean_threshold": cfg.analogy_weak_step_threshold,
                "minimum_passing_ph_feature_families": 2,
            },
            "candidate": {
                "full_reasoning_trajectory_simplex_map_threshold": cfg.analogy_candidate_map_threshold,
                "vectorized_persistence_similarity_threshold": cfg.analogy_candidate_ph_threshold,
                "ph_gate_threshold": cfg.analogy_candidate_ph_threshold,
                "step_simplicial_map_mean_threshold": cfg.analogy_candidate_step_threshold,
            },
        },
        "analogy_status": analogy_status,
        "analogy_confidence_score": confidence_score,
        "decision_summary": {
            "narrative": narrative,
            "active_status": analogy_status,
            "strong": {"passed": bool(strong), "conditions": strong_conditions},
            "weak": {"passed": bool(weak), "conditions": weak_conditions},
            "candidate": {"passed": bool(candidate), "conditions": candidate_conditions},
            "scoreboard": {
                "full_trajectory_simplex_map": float(full_map["valid_fraction"]),
                "step_simplex_map_mean": float(step_map_mean),
                "vectorized_ph_mean": float(ph_cosine),
                "ph_gate": float(ph_gate_score),
                "ph_best_family_similarity": float(ph_best),
                "ph_weak_family_pass_count": int(ph_pass_count),
            },
        },
        "no_analogy_reason": "" if emitted else "No analogy emitted: map, step-map, and vectorized-PH evidence did not clear the candidate tier.",
        "analogy_passed_checks": {
            "strong": bool(strong),
            "weak": bool(weak),
            "candidate": bool(candidate),
            "ph_gate_score": ph_gate_score,
            "ph_best_family_similarity": ph_best,
            "ph_weak_family_pass_count": ph_pass_count,
        },
        "source_simplex_tree": source_tree,
        "memory_simplex_tree": target_tree,
        "candidate_map": full_map,
        "full_reasoning_trajectory_simplex_tree_map": full_map,
        "vectorized_persistence_comparisons": feature_similarity,
        "vectorized_persistence_cosine_mean": ph_cosine,
        "persistence_diagram_distances": ph_distance_summary,
        "source_ph_metrics": source_bundle["metrics"],
        "target_ph_metrics": target_bundle["metrics"],
        "ph_features": {
            "source": {key: value for key, value in source_bundle.items() if key != "_features"},
            "memory": {key: value for key, value in target_bundle.items() if key != "_features"},
            "similarity": feature_similarity,
        },
        "step_simplicial_map_valid_fraction_mean": step_map_mean,
        "target_memory_step_embeddings": target.astype(float).round(6).tolist(),
    }


def build_branching_reasoning_payload(cfg: BranchingTrajectoryConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BranchingTrajectoryConfig()
    nodes, dag_edges, step_embeddings = _generate_step_embeddings(cfg)
    steps, _flat_tokens = _generate_step_tokens(cfg, nodes, step_embeddings)
    step_dist = pairwise_distances(step_embeddings)
    positive = step_dist[step_dist > 1e-10]
    max_radius = float(np.quantile(positive, 0.72)) if positive.size else 1.0
    radii = np.linspace(max_radius / max(2, cfg.radius_levels), max_radius, cfg.radius_levels)
    payload = {
        "schema": "toricgt.branching_reasoning_trajectory.v1",
        "source_mode": "synthetic_long_branching_fixture",
        "source_metadata": {
            "seed": int(cfg.seed),
            "trajectory_levels": int(cfg.trajectory_levels),
            "branch_lanes": int(cfg.branch_lanes),
            "side_branch_length": int(cfg.side_branch_length),
            "embedding_dim": int(cfg.embedding_dim),
        },
        "embedding_comparison_space": "original_high_dimensional_embeddings",
        "pca_display_only": True,
        "controls": {
            "full": ["radius_slider", "reasoning_level_slider"],
            "step": ["step_radius_slider", "decoding_order_slider"],
            "analogy": ["analogy_radius_slider", "analogy_reasoning_level_slider"],
        },
        "nodes": nodes,
        "dag_edges": [[src, dst, kind] for src, dst, kind in dag_edges],
        "reasoning_order_edges": [[idx, idx + 1, idx + 1] for idx in range(len(nodes) - 1)],
        "radius_values": radii.astype(float).round(6).tolist(),
        "distances": step_dist.astype(float).round(6).tolist(),
        "triangles": _diameter_triangles(step_dist, limit=cfg.max_render_triangles),
        "triangle_count_exact": int(len(_diameter_triangles(step_dist))),
        "render_limits": {
            "visible_edges": "all",
            "max_render_edges": "deprecated_ignored_all_visible_edges",
            "max_render_triangles": int(cfg.max_render_triangles),
            "max_map_image_records": int(cfg.max_map_image_records),
        },
        "visible_edge_policy": "all_visible_edges",
        "steps": steps,
        "analogy": _analogy_payload(cfg, step_embeddings, steps, nodes),
        "gflownet_flow": {
            "training_objective": "continuous_embedding_space_trajectory_balance_contract",
            "log_flow_proxy": [float(1.0 + 0.06 * node["level"] - 0.03 * node["nll"]) for node in nodes],
            "branch_policy_entropy_proxy": [float(0.55 + 0.08 * node["is_branch"] + 0.04 * node["level"]) for node in nodes],
        },
    }
    return payload


def _load_embedding_payload_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        if "hidden" not in data.files:
            raise ValueError(f"{npz_path} is missing required `hidden` array")
        hidden = np.asarray(data["hidden"], dtype=np.float64)
        if hidden.ndim != 2 or hidden.shape[0] < 2:
            raise ValueError(f"{npz_path}: `hidden` must have shape [nodes, dim] with at least two nodes")
        nll = np.asarray(data["nll"], dtype=np.float64) if "nll" in data.files else np.zeros((hidden.shape[0],), dtype=np.float64)
        if nll.ndim != 1:
            nll = nll.reshape(-1)
        edges = np.asarray(data["edges"], dtype=np.int64).reshape(-1, 2) if "edges" in data.files else np.zeros((0, 2), dtype=np.int64)
        projected = np.asarray(data["projected"], dtype=np.float64) if "projected" in data.files else _pca3(hidden)
    return {"hidden": hidden, "nll": nll, "edges": edges, "projected": projected}


def _select_embedding_window(
    hidden: np.ndarray,
    nll: np.ndarray,
    edges: np.ndarray,
    *,
    max_nodes: int | None = None,
    node_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Select and reindex an explicit node window from a real payload.

    The visualization contract forbids silently dropping active visible
    one-dimensional simplex edges.  For large real checkpoint payloads, this
    function selects the point set *before* any simplex tree is built; all
    simplex edges inside that chosen point set remain visible.
    """

    x = np.asarray(hidden, dtype=np.float64)
    n = int(x.shape[0])
    if n <= 0:
        raise ValueError("embedding payload has no nodes")
    start = max(0, min(int(node_offset), n - 1))
    if max_nodes is None or int(max_nodes) <= 0:
        stop = n
    else:
        stop = min(n, start + max(2, int(max_nodes)))
        if stop - start < 2 and n >= 2:
            start = max(0, n - 2)
            stop = n
    selected_ids = list(range(start, stop))
    selected_set = set(selected_ids)
    reindex = {old: new for new, old in enumerate(selected_ids)}
    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2) if np.asarray(edges).size else np.zeros((0, 2), dtype=np.int64)
    kept_edges: list[list[int]] = []
    for src, dst in edge_array:
        src_i, dst_i = int(src), int(dst)
        if src_i in selected_set and dst_i in selected_set and src_i != dst_i:
            kept_edges.append([reindex[src_i], reindex[dst_i]])
    nll_array = np.asarray(nll, dtype=np.float64).reshape(-1)
    if nll_array.shape[0] < n:
        fill = float(np.nanmean(nll_array)) if nll_array.size else 0.0
        nll_array = np.pad(nll_array, (0, n - nll_array.shape[0]), constant_values=fill)
    metadata = {
        "original_node_count": int(n),
        "selected_node_count": int(stop - start),
        "selected_node_offset": int(start),
        "selected_node_stop": int(stop),
        "selected_original_node_ids": [int(idx) for idx in selected_ids],
        "node_window_applied": bool((stop - start) != n or start != 0),
        "edge_count_before_window": int(edge_array.shape[0]),
        "edge_count_after_window": int(len(kept_edges)),
        "visible_edge_policy_after_window": "all_visible_edges",
    }
    return (
        x[start:stop],
        nll_array[start:stop],
        np.asarray(kept_edges, dtype=np.int64).reshape(-1, 2),
        metadata,
    )


def build_branching_reasoning_payload_from_embedding_payload(
    npz_path: Path,
    *,
    metadata_json_path: Path | None = None,
    cfg: BranchingTrajectoryConfig | None = None,
    embedding_max_nodes: int | None = None,
    embedding_node_offset: int = 0,
) -> dict[str, Any]:
    """Render a saved checkpoint inference embedding payload as the report schema.

    The accepted NPZ format is the ``toricgt.embedding_payload.v1`` bundle
    emitted by ``scripts/evaluate_tokengt_reasoning_geometry_suite.py``.  The
    report uses the saved high-dimensional hidden vectors for PH and simplex
    maps.  PCA coordinates are recomputed only for browser display.
    """

    cfg = cfg or BranchingTrajectoryConfig()
    arrays = _load_embedding_payload_arrays(Path(npz_path))
    hidden, nll, edges, window_metadata = _select_embedding_window(
        arrays["hidden"],
        arrays["nll"],
        arrays["edges"],
        max_nodes=embedding_max_nodes,
        node_offset=embedding_node_offset,
    )
    node_count = int(hidden.shape[0])
    if edges.size == 0:
        edges = np.asarray([[idx, idx + 1] for idx in range(node_count - 1)], dtype=np.int64)
    nodes = _payload_nodes_from_embeddings(hidden, nll, edges)
    dag_edges = [(int(src), int(dst), "checkpoint_graph_edge") for src, dst in np.asarray(edges, dtype=np.int64).reshape(-1, 2) if 0 <= int(src) < node_count and 0 <= int(dst) < node_count and int(src) != int(dst)]
    if not dag_edges:
        dag_edges = [(idx, idx + 1, "checkpoint_index_order") for idx in range(node_count - 1)]
    steps, _flat_tokens = _real_payload_step_tokens(cfg, nodes, hidden, nll, np.asarray([[src, dst] for src, dst, _ in dag_edges], dtype=np.int64))
    step_dist = pairwise_distances(hidden)
    positive = step_dist[step_dist > 1e-10]
    max_radius = float(np.quantile(positive, 0.72)) if positive.size else 1.0
    radii = np.linspace(max_radius / max(2, cfg.radius_levels), max_radius, cfg.radius_levels)
    metadata: dict[str, Any] = {}
    if metadata_json_path is not None:
        metadata_path = Path(metadata_json_path)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = {
        "schema": "toricgt.branching_reasoning_trajectory.v1",
        "source_mode": "checkpoint_embedding_payload",
        "source_metadata": {
            "embedding_payload_npz": str(Path(npz_path)),
            "embedding_payload_json": str(metadata_json_path) if metadata_json_path is not None else "",
            "embedding_payload_schema": metadata.get("schema", ""),
            "record_index": metadata.get("record_index", None),
            "checkpoint_step": metadata.get("checkpoint_step", None),
            "nll_source": metadata.get("nll_source", ""),
            "original_hidden_shape": list(arrays["hidden"].shape),
            "hidden_shape": list(hidden.shape),
            **window_metadata,
        },
        "embedding_comparison_space": "original_high_dimensional_embeddings",
        "pca_display_only": True,
        "controls": {
            "full": ["radius_slider", "reasoning_level_slider"],
            "step": ["step_radius_slider", "decoding_order_slider"],
            "analogy": ["analogy_radius_slider", "analogy_reasoning_level_slider"],
        },
        "nodes": nodes,
        "dag_edges": [[src, dst, kind] for src, dst, kind in dag_edges],
        "reasoning_order_edges": [[idx, idx + 1, idx + 1] for idx in range(node_count - 1)],
        "radius_values": radii.astype(float).round(6).tolist(),
        "distances": step_dist.astype(float).round(6).tolist(),
        "triangles": _diameter_triangles(step_dist, limit=cfg.max_render_triangles),
        "triangle_count_exact": int(len(_diameter_triangles(step_dist))),
        "render_limits": {
            "visible_edges": "all",
            "max_render_edges": "deprecated_ignored_all_visible_edges",
            "max_render_triangles": int(cfg.max_render_triangles),
            "max_map_image_records": int(cfg.max_map_image_records),
        },
        "visible_edge_policy": "all_visible_edges",
        "steps": steps,
        "analogy": _analogy_payload(cfg, hidden, steps, nodes),
        "gflownet_flow": {
            "training_objective": "continuous_embedding_space_trajectory_balance_contract",
            "source": "checkpoint_embedding_payload_visual_audit",
            "log_flow_proxy": [float(1.0 + 0.06 * node["level"] - 0.03 * node["nll"]) for node in nodes],
            "branch_policy_entropy_proxy": [float(0.55 + 0.08 * node["is_branch"] + 0.04 * node["level"]) for node in nodes],
        },
    }
    return payload


def _plotly_bootstrap() -> str:
    fig = go.Figure()
    return '<div style="display:none">' + pio.to_html(fig, include_plotlyjs=True, full_html=False) + "</div>"


def _metrics_html(payload: dict[str, Any]) -> str:
    analogy = payload["analogy"]
    rows = {
        "source mode": payload.get("source_mode", "unknown"),
        "reasoning steps": len(payload["nodes"]),
        "DAG edges": len(payload["dag_edges"]),
        "radius levels": len(payload["radius_values"]),
        "analogy emitted": analogy["analogy_emitted"],
        "analogy status": analogy["analogy_status"],
        "analogy confidence": round(float(analogy["analogy_confidence_score"]), 4),
        "full simplex-map valid fraction": round(float(analogy["full_reasoning_trajectory_simplex_tree_map"]["valid_fraction"]), 4),
        "vectorized PH cosine mean": round(float(analogy["vectorized_persistence_cosine_mean"]), 4),
        "vectorized PH gate score": round(float(analogy["analogy_passed_checks"]["ph_gate_score"]), 4),
        "step simplex-map mean": round(float(analogy["step_simplicial_map_valid_fraction_mean"]), 4),
    }
    return "".join(f"<div class='metric'><span>{key}</span><span>{value}</span></div>" for key, value in rows.items())


def render_branching_reasoning_report(payload: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "branching_reasoning_payload.json"
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    js_payload = json.dumps(payload, separators=(",", ":"))
    html_path = output_dir / "branching_reasoning_trajectory.html"
    max_level = max(node["level"] for node in payload["nodes"])
    radius_default = max(0, min(len(payload["radius_values"]) - 1, len(payload["radius_values"]) // 2))
    step_radius_default = max(0, min(len(payload["steps"][0]["radius_values"]) - 1, len(payload["steps"][0]["radius_values"]) // 2))
    decode_default = max(0, len(payload["steps"][0]["tokens"]) - 1)
    html_path.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Branching Reasoning Trajectory Simplicial Audit</title><style>{CSS}</style></head>
<body><main>
{_plotly_bootstrap()}
<section class="hero"><h1>Branching Graph-Of-Thought Reasoning Trajectory</h1>
<p>This audit uses original high-dimensional embeddings for simplex maps and GUDHI vectorized persistent-homology comparisons.  The 3D coordinates are PCA display coordinates only.</p>
<p class="muted">Source mode: <code>{payload.get('source_mode', 'unknown')}</code>.  The source metadata is included in the raw payload panel.</p>
<p class="muted">Analogy tiers: <code>strong_analogy</code>, <code>weak_analogy</code>, <code>candidate_analogy</code>, and <code>no_analogy</code>.</p>
<p><a class="pill" href="branching_reasoning_payload.json">payload JSON</a><a class="pill" href="index.html">index</a></p></section>
<section class="grid"><div class="card"><h2>Audit Metrics</h2>{_metrics_html(payload)}</div><div class="card"><h2>Analogy Decision</h2><div id="top_analogy_decision_banner"></div><div id="top_analogy_decision_status_badges" class="status-badge-row"></div><div id="top_analogy_threshold_table"></div><details><summary class="muted">raw decision payload</summary><pre id="analogy_decision"></pre></details></div></section>
<section class="panel"><h2>Full Reasoning Trajectory Filtered Simplicial Complex</h2>
<div class="controls">
  <div class="control"><label>radius <span id="radius_value"></span></label><input id="radius_slider" type="range" min="0" max="{len(payload['radius_values'])-1}" step="1" value="{radius_default}"></div>
  <div class="control"><label>reasoning level / decoding order <span id="reasoning_level_value"></span></label><input id="reasoning_level_slider" type="range" min="0" max="{max_level}" step="1" value="{max_level}"></div>
  <div class="control"><label>filled 2-simplices <span id="full_triangle_state">faces hidden</span></label><input id="full_triangle_toggle" type="checkbox"></div>
  <div class="control"><label>step labels <span id="full_label_state">labels hidden</span></label><input id="full_label_toggle" type="checkbox"></div>
</div>
<div id="trajectory_state_caption" class="caption"></div><div id="trajectory_plot" class="plot"></div><div id="node_detail_panel" class="detail">Click a reasoning node to reveal its step metadata and token subcomplex.</div></section>
<section class="panel"><h2>Selected Reasoning Step Simplex Tree</h2>
<div class="controls">
  <div class="control"><label>step radius <span id="step_radius_value"></span></label><input id="step_radius_slider" type="range" min="0" max="{len(payload['steps'][0]['radius_values'])-1}" step="1" value="{step_radius_default}"></div>
  <div class="control"><label>decoding order <span id="decoding_order_value"></span></label><input id="decoding_order_slider" type="range" min="0" max="{len(payload['steps'][0]['tokens'])-1}" step="1" value="{decode_default}"></div>
  <div class="control"><label>filled 2-simplices <span id="step_triangle_state">faces hidden</span></label><input id="step_triangle_toggle" type="checkbox"></div>
  <div class="control"><label>token labels <span id="step_label_state">labels visible</span></label><input id="step_label_toggle" type="checkbox" checked></div>
</div>
<div id="step_title" class="muted"></div><div id="step_state_caption" class="caption"></div><div id="step_plot" class="plot"></div><div id="token_detail_panel" class="detail">Click a token point to reveal token text, likelihood, entropy, and source-step metadata.</div><div id="step_simplex_tree_table" class="detail"></div></section>
<section class="panel"><h2>Analogical Memory Retrieval: Simplex-Tree Maps And Vectorized PH</h2>
<div class="controls">
  <div class="control"><label>analogy radius <span id="analogy_radius_value"></span></label><input id="analogy_radius_slider" type="range" min="0" max="{len(payload['radius_values'])-1}" step="1" value="{radius_default}"></div>
  <div class="control"><label>analogy reasoning level <span id="analogy_reasoning_level_value"></span></label><input id="analogy_reasoning_level_slider" type="range" min="0" max="{max_level}" step="1" value="{max_level}"></div>
  <div class="control"><label>filled 2-simplices <span id="analogy_triangle_state">faces hidden</span></label><input id="analogy_triangle_toggle" type="checkbox"></div>
  <div class="control"><label>analogy labels <span id="analogy_label_state">labels hidden</span></label><input id="analogy_label_toggle" type="checkbox"></div>
</div>
<div id="no_analogy_banner"></div>
<div id="analogy_decision_status_badges" class="status-badge-row"></div>
<div id="analogy_state_caption" class="caption"></div>
<div id="analogy_map_summary" class="detail"></div>
<div id="ph_quick_summary" class="detail"></div>
<div id="analogy_plot" class="plot"></div>
<div id="analogy_detail_panel" class="detail">Click a source or memory vertex in the analogy plot to inspect the mapped reasoning-step metadata.</div>
<h3>Analogical Vertex Correspondence</h3><div id="analogy_vertex_map_table"></div>
<h3>Simplicial Map Validity</h3><div id="simplicial_map_validity_table"></div>
<h3>Vectorized Persistence Similarity</h3><div id="ph_table"></div>
<div id="ph_similarity_bar_plot" class="plot smallplot"></div>
<h3>Persistence Diagram Distances</h3><div id="ph_distance_table"></div>
<div id="persistence_diagram_plot" class="plot smallplot"></div>
<div id="persistence_landscape_plot" class="plot smallplot"></div>
<div id="ph_difference_plot" class="plot smallplot"></div>
<div id="persistence_image_heatmap" class="plot smallplot"></div>
<div id="silhouette_vector_plot" class="plot smallplot"></div>
<div id="entropy_vector_plot" class="plot smallplot"></div>
<div id="betti_curve_plot" class="plot smallplot"></div>
<div id="lifetime_histogram_plot" class="plot smallplot"></div>
</section>
<section class="panel"><h2>Raw Topological Payload</h2><pre id="raw_payload"></pre></section>
<script>
const trajectory_simplex_payload = {js_payload};
const simplex_step_payload = trajectory_simplex_payload.steps;
const dotted_decode_edges = trajectory_simplex_payload.reasoning_order_edges;
const half_arrow_payload = true;
const gflownet_flow = trajectory_simplex_payload.gflownet_flow;
let selectedStep = 0;

function visibleNodeIndices(level) {{
  return trajectory_simplex_payload.nodes.filter(n => n.level <= level).map(n => n.id);
}}
function coord(node) {{ return node.pca; }}
function lineTrace(name, xs, ys, zs, color, dash, width) {{
  return {{type:'scatter3d', mode:'lines', x:xs, y:ys, z:zs, name:name, line:{{color:color, width:width || 3, dash:dash || 'solid'}}, hoverinfo:'skip'}};
}}
function labelMode(enabled) {{
  return enabled ? 'markers+text' : 'markers';
}}
function fmt(value, digits=4) {{
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : 'n/a';
}}
function passStatusBadge(label, passed, detail) {{
  return '<span class="status-badge '+(passed ? 'pass' : 'fail')+'">'+(passed ? 'PASS' : 'FAIL')+' · '+label+(detail ? ' · '+detail : '')+'</span>';
}}
function infoStatusBadge(label, value) {{
  return '<span class="status-badge info">'+label+' · '+value+'</span>';
}}
function updateToggleState(toggleId, stateId, onText, offText) {{
  const el = document.getElementById(toggleId), state = document.getElementById(stateId);
  if (el && state) state.textContent = el.checked ? onText : offText;
}}
function conditionRows(tierName, tier) {{
  return tier.conditions.map(item => '<tr><td>'+tierName+'</td><td>'+item.label+'</td><td>'+fmt(item.score)+'</td><td>>= '+fmt(item.threshold)+'</td><td>'+(item.passed ? '<span class="ok">pass</span>' : '<span class="bad">fail</span>')+'</td><td>'+(item.detail || '')+'</td></tr>').join('');
}}
function decisionConditionRowsHTML() {{
  const decision = trajectory_simplex_payload.analogy.decision_summary;
  return [
    conditionRows('strong threshold', decision.strong),
    conditionRows('weak threshold', decision.weak),
    conditionRows('candidate threshold', decision.candidate)
  ].join('');
}}
function decisionThresholdTableHTML() {{
  return '<table><tr><th>tier</th><th>measure</th><th>score</th><th>threshold</th><th>result</th><th>meaning</th></tr>'+
    decisionConditionRowsHTML()+
    '</table>';
}}
function decisionStatusBadgesHTML() {{
  const analogy = trajectory_simplex_payload.analogy;
  const summary = analogy.decision_summary;
  const scores = summary.scoreboard;
  return [
    passStatusBadge('strong tier', summary.strong.passed, summary.strong.passed ? 'all strong gates passed' : 'one or more strong gates failed'),
    passStatusBadge('weak tier', summary.weak.passed, summary.weak.passed ? 'weak retrieval accepted' : 'weak gates failed'),
    passStatusBadge('candidate tier', summary.candidate.passed, summary.candidate.passed ? 'minimum evidence present' : 'minimum evidence failed'),
    infoStatusBadge('full trajectory map', fmt(scores.full_trajectory_simplex_map)),
    infoStatusBadge('step map mean', fmt(scores.step_simplex_map_mean)),
    infoStatusBadge('vectorized PH mean', fmt(scores.vectorized_ph_mean)),
    infoStatusBadge('PH gate', fmt(scores.ph_gate))
  ].join('');
}}
function mapSummaryHTML() {{
  const map = trajectory_simplex_payload.analogy.candidate_map;
  const summary = map.map_summary || {{}};
  return '<strong>Compact simplex-map summary</strong><br>'+
    'vertices <code>'+summary.source_vertices+' -> '+summary.target_vertices+'</code> · '+
    'edges valid/collapsed/invalid <code>'+summary.valid_edges+'/'+summary.collapsed_edges+'/'+summary.invalid_edges+'</code> · '+
    'triangles valid/collapsed/invalid <code>'+summary.valid_triangles+'/'+summary.collapsed_triangles+'/'+summary.invalid_triangles+'</code><br>'+
    'valid-or-collapsed fraction <code>'+fmt(summary.valid_or_collapsed_fraction)+'</code> · '+
    'mean/max vertex-map distance <code>'+fmt(summary.vertex_distance_mean)+' / '+fmt(summary.vertex_distance_max)+'</code>';
}}
function phQuickSummaryHTML() {{
  const analogy = trajectory_simplex_payload.analogy;
  const comp = analogy.vectorized_persistence_comparisons;
  const keys = Object.keys(comp).sort((a,b) => Number(comp[b])-Number(comp[a]));
  const top = keys.slice(0, 5).map(k => '<span class="status-badge info">'+k+' '+fmt(comp[k])+'</span>').join('');
  return '<strong>Vectorized PH evidence near the map</strong><br>'+
    'mean <code>'+fmt(analogy.vectorized_persistence_cosine_mean)+'</code> · '+
    'best family <code>'+fmt(analogy.analogy_passed_checks.ph_best_family_similarity)+'</code> · '+
    'weak-family pass count <code>'+analogy.analogy_passed_checks.ph_weak_family_pass_count+'</code>'+
    '<div class="status-badge-row">'+top+'</div>';
}}
function markerTrace(name, nodes, showLabels) {{
  return {{
    type:'scatter3d', mode:labelMode(showLabels), name:name,
    x:nodes.map(n=>n.pca[0]), y:nodes.map(n=>n.pca[1]), z:nodes.map(n=>n.pca[2]),
    text:nodes.map(n=>n.label), textposition:'top center',
    customdata:nodes.map(n=>[n.id,n.level,n.node_kind,n.in_degree,n.out_degree,n.branch]),
    marker:{{size:7,color:nodes.map(n=>n.nll),colorscale:'Turbo',showscale:true,colorbar:{{title:'node NLL'}},line:{{color:'#e8fbff',width:0.5}}}},
    hovertemplate:'step %{{customdata[0]}} / %{{text}}<br>level %{{customdata[1]}}<br>kind %{{customdata[2]}}<br>in/out %{{customdata[3]}}/%{{customdata[4]}}<br>branch %{{customdata[5]}}<br>NLL %{{marker.color:.3f}}<extra></extra>'
  }};
}}
function nodeDetailsHTML(node) {{
  return '<strong>Reasoning step '+node.id+' ('+node.label+')</strong><br>'+
    'level <code>'+node.level+'</code> · kind <code>'+node.node_kind+'</code> · branch <code>'+node.branch+'</code><br>'+
    'in/out degree <code>'+node.in_degree+'/'+node.out_degree+'</code> · NLL <code>'+Number(node.nll).toFixed(4)+'</code><br>'+
    'token count <code>'+(simplex_step_payload[node.id] ? simplex_step_payload[node.id].tokens.length : 0)+'</code>';
}}
function tokenDetailsHTML(token, step) {{
  return '<strong>Token '+token.id+' / '+token.text+'</strong><br>'+
    'type <code>'+token.token_type+'</code> · source step <code>'+token.source_step_label+'</code> · decode order <code>'+token.decode_order+'</code><br>'+
    'NLL <code>'+Number(token.nll).toFixed(4)+'</code> · logprob <code>'+Number(token.logprob).toFixed(4)+'</code> · entropy <code>'+Number(token.entropy).toFixed(4)+'</code> · rank <code>'+token.rank+'</code><br>'+
    'global token id <code>'+token.global_token_id+'</code> · step level <code>'+step.level+'</code>';
}}
function edgeLineFromPairs(pairs, nodesById, color, dash, name, width) {{
  let xs=[], ys=[], zs=[];
  for (const item of pairs) {{
    const src = nodesById[item[0]], dst = nodesById[item[1]];
    if (!src || !dst) continue;
    xs.push(src.pca[0], dst.pca[0], null); ys.push(src.pca[1], dst.pca[1], null); zs.push(src.pca[2], dst.pca[2], null);
  }}
  return lineTrace(name, xs, ys, zs, color, dash, width);
}}
function midpointArrowTrace(pairs, nodesById) {{
  let x=[], y=[], z=[], labels=[];
  for (const item of pairs) {{
    const src = nodesById[item[0]], dst = nodesById[item[1]];
    if (!src || !dst) continue;
    x.push((src.pca[0]+dst.pca[0])/2); y.push((src.pca[1]+dst.pca[1])/2); z.push((src.pca[2]+dst.pca[2])/2); labels.push(item[0]+' -> '+item[1]);
  }}
  return {{type:'scatter3d', mode:'markers', x:x, y:y, z:z, text:labels, name:'small midpoint decoding-order arrowheads', marker:{{size:3,color:'rgba(255,209,102,.82)',symbol:'diamond'}}, hovertemplate:'decode arrow %{{text}}<extra></extra>'}};
}}
function fullSimplicialEdges(visible, radius) {{
  const visibleSet = new Set(visible);
  let out=[];
  for (let i=0;i<trajectory_simplex_payload.nodes.length;i++) for (let j=i+1;j<trajectory_simplex_payload.nodes.length;j++) {{
    if (visibleSet.has(i) && visibleSet.has(j) && trajectory_simplex_payload.distances[i][j] <= radius) out.push([i,j]);
  }}
  return out;
}}
function fullTriangles(visible, radius) {{
  const visibleSet = new Set(visible);
  return trajectory_simplex_payload.triangles.filter(t => visibleSet.has(t[0]) && visibleSet.has(t[1]) && visibleSet.has(t[2]) && t[3] <= radius);
}}
function meshTrace(tris, nodesById, name, color) {{
  if (!tris.length) return {{type:'mesh3d', x:[], y:[], z:[], i:[], j:[], k:[], name:name, opacity:0.10, color:color}};
  let points=[], index={{}}, ii=[], jj=[], kk=[];
  function add(id) {{ if (index[id] === undefined) {{ const n=nodesById[id]; index[id]=points.length; points.push(n.pca); }} return index[id]; }}
  for (const tri of tris) {{ ii.push(add(tri[0])); jj.push(add(tri[1])); kk.push(add(tri[2])); }}
  return {{type:'mesh3d', x:points.map(p=>p[0]), y:points.map(p=>p[1]), z:points.map(p=>p[2]), i:ii, j:jj, k:kk, name:name, opacity:0.13, color:color, hoverinfo:'skip'}};
}}
function renderTrajectory() {{
  const rIdx = parseInt(document.getElementById('radius_slider').value);
  const level = parseInt(document.getElementById('reasoning_level_slider').value);
  const radius = trajectory_simplex_payload.radius_values[rIdx];
  document.getElementById('radius_value').textContent = radius.toFixed(3);
  document.getElementById('reasoning_level_value').textContent = level;
  const visible = visibleNodeIndices(level);
  const visibleSet = new Set(visible);
  const showTriangles = document.getElementById('full_triangle_toggle').checked;
  const showLabels = document.getElementById('full_label_toggle').checked;
  updateToggleState('full_triangle_toggle', 'full_triangle_state', 'faces visible', 'faces hidden');
  updateToggleState('full_label_toggle', 'full_label_state', 'labels visible', 'labels hidden');
  const nodes = trajectory_simplex_payload.nodes.filter(n => visibleSet.has(n.id));
  const nodesById = Object.fromEntries(nodes.map(n => [n.id, n]));
  const dagPairs = trajectory_simplex_payload.dag_edges.filter(e => visibleSet.has(e[0]) && visibleSet.has(e[1]));
  const orderPairs = dotted_decode_edges.filter(e => e[2] <= Math.max(1, visible.length-1) && visibleSet.has(e[0]) && visibleSet.has(e[1]));
  const complexPairs = fullSimplicialEdges(visible, radius);
  const tris = showTriangles ? fullTriangles(visible, radius) : [];
  document.getElementById('trajectory_state_caption').innerHTML =
    'radius <code>'+radius.toFixed(3)+'</code> · reasoning level <code>'+level+'</code> · visible vertices <code>'+nodes.length+'</code> · visible one-dimensional simplex edges <code>'+complexPairs.length+'</code> · visible 2-simplex faces <code>'+tris.length+'</code> · label state <code>'+(showLabels ? 'visible' : 'hidden')+'</code>';
  const traces = [
    meshTrace(tris, nodesById, 'radius-controlled 2-simplices', '#37e8ff'),
    edgeLineFromPairs(complexPairs, nodesById, 'rgba(55,232,255,.40)', 'solid', 'radius-controlled simplex edges', 2),
    edgeLineFromPairs(dagPairs, nodesById, 'rgba(140,255,106,.70)', 'solid', 'graph-of-thought DAG branch/merge edges', 5),
    edgeLineFromPairs(orderPairs, nodesById, 'rgba(255,209,102,.50)', 'dot', 'faint dotted decoding order arrows', 2),
    midpointArrowTrace(orderPairs, nodesById),
    markerTrace('reasoning steps', nodes, showLabels)
  ];
  Plotly.react('trajectory_plot', traces, {{template:'plotly_dark', paper_bgcolor:'#020713', plot_bgcolor:'#020713', scene:{{aspectmode:'data', xaxis:{{title:'PC1'}},yaxis:{{title:'PC2'}},zaxis:{{title:'PC3'}}}}, legend:{{x:0.01,y:0.99,bgcolor:'rgba(2,7,19,.58)'}}, margin:{{l:0,r:0,t:28,b:0}}, showlegend:true}});
}}
function simplexTableHTML(step, tokens, pairs, tris) {{
  const tokenById = Object.fromEntries(step.tokens.map(t => [t.id, t]));
  const vertexRows = tokens.map(t => '<tr><td>0</td><td>('+t.id+')</td><td>0.000</td><td>'+t.text+'</td></tr>');
  const edgeRows = pairs.map(e => '<tr><td>1</td><td>('+e[0]+', '+e[1]+')</td><td>'+Number(step.distances[e[0]][e[1]]).toFixed(3)+'</td><td>'+tokenById[e[0]].text+' -> '+tokenById[e[1]].text+'</td></tr>');
  const triRows = tris.map(t => '<tr><td>2</td><td>('+t[0]+', '+t[1]+', '+t[2]+')</td><td>'+Number(t[3]).toFixed(3)+'</td><td>faces ('+t[0]+', '+t[1]+'), ('+t[0]+', '+t[2]+'), ('+t[1]+', '+t[2]+')</td></tr>');
  return '<strong>Selected-step simplex tree filtration</strong><br>'+
    '<span class="muted">Rows are the active simplices at the current radius and decoding-order sliders.  Edges are all visible one-dimensional simplices; no edge cap is applied.</span>'+
    '<table><tr><th>dim</th><th>simplex</th><th>birth radius</th><th>token / face metadata</th></tr>'+vertexRows.concat(edgeRows, triRows).join('')+'</table>';
}}
function renderStep(stepId) {{
  selectedStep = stepId;
  const step = simplex_step_payload[stepId] || simplex_step_payload[0];
  document.getElementById('decoding_order_slider').max = Math.max(0, step.tokens.length - 1);
  document.getElementById('step_radius_slider').max = Math.max(0, step.radius_values.length - 1);
  const rIdx = Math.min(parseInt(document.getElementById('step_radius_slider').value), step.radius_values.length - 1);
  const order = Math.min(parseInt(document.getElementById('decoding_order_slider').value), step.tokens.length - 1);
  const radius = step.radius_values[rIdx];
  document.getElementById('step_radius_value').textContent = radius.toFixed(3);
  document.getElementById('decoding_order_value').textContent = order;
  document.getElementById('step_title').textContent = 'selected reasoning step ' + step.step_index + ' at level ' + step.level;
  const tokens = step.tokens.filter(t => t.decode_order <= order);
  const tokenSet = new Set(tokens.map(t=>t.id));
  const tokenById = Object.fromEntries(tokens.map(t=>[t.id,t]));
  const showTriangles = document.getElementById('step_triangle_toggle').checked;
  const showLabels = document.getElementById('step_label_toggle').checked;
  updateToggleState('step_triangle_toggle', 'step_triangle_state', 'faces visible', 'faces hidden');
  updateToggleState('step_label_toggle', 'step_label_state', 'labels visible', 'labels hidden');
  let pairs=[];
  for (let i=0;i<step.tokens.length;i++) for (let j=i+1;j<step.tokens.length;j++) {{
    if (tokenSet.has(i) && tokenSet.has(j) && step.distances[i][j] <= radius) pairs.push([i,j]);
  }}
  const decodePairs = step.decode_edges.filter(e => e[2] <= order && tokenSet.has(e[0]) && tokenSet.has(e[1]));
  const activeTris = step.triangles.filter(t => tokenSet.has(t[0]) && tokenSet.has(t[1]) && tokenSet.has(t[2]) && t[3] <= radius);
  const tris = showTriangles ? activeTris : [];
  document.getElementById('step_state_caption').innerHTML =
    'step <code>'+step.step_index+'</code> · radius <code>'+radius.toFixed(3)+'</code> · decoding order <code>'+order+'</code> · visible tokens <code>'+tokens.length+'</code> · visible one-dimensional simplex edges <code>'+pairs.length+'</code> · active 2-simplex faces <code>'+activeTris.length+'</code> · rendered faces <code>'+tris.length+'</code> · label state <code>'+(showLabels ? 'visible' : 'hidden')+'</code>';
  const traces = [
    meshTrace(tris, tokenById, 'step 2-simplices', '#ff4fd8'),
    edgeLineFromPairs(pairs, tokenById, 'rgba(55,232,255,.45)', 'solid', 'radius simplex edges', 2),
    edgeLineFromPairs(decodePairs, tokenById, 'rgba(255,209,102,.55)', 'dot', 'dotted token decoding order arrows', 2),
    midpointArrowTrace(decodePairs, tokenById),
    {{type:'scatter3d', mode:labelMode(showLabels), name:'token embeddings', x:tokens.map(t=>t.pca[0]), y:tokens.map(t=>t.pca[1]), z:tokens.map(t=>t.pca[2]), text:tokens.map(t=>String(t.id)), customdata:tokens.map(t=>[t.id,t.text,t.token_type,t.decode_order,t.logprob,t.entropy,t.rank,t.global_token_id]), marker:{{size:6,color:tokens.map(t=>t.nll),colorscale:'Turbo',showscale:true,colorbar:{{title:'token NLL'}}}}, hovertemplate:'token %{{customdata[0]}}: %{{customdata[1]}}<br>type %{{customdata[2]}}<br>decode %{{customdata[3]}}<br>NLL %{{marker.color:.3f}}<br>logprob %{{customdata[4]:.3f}}<br>entropy %{{customdata[5]:.3f}}<br>rank %{{customdata[6]}}<extra></extra>'}}
  ];
  document.getElementById('step_simplex_tree_table').innerHTML = simplexTableHTML(step, tokens, pairs, activeTris);
  Plotly.react('step_plot', traces, {{template:'plotly_dark', paper_bgcolor:'#020713', plot_bgcolor:'#020713', scene:{{aspectmode:'data', xaxis:{{title:'PC1'}},yaxis:{{title:'PC2'}},zaxis:{{title:'PC3'}}}}, legend:{{x:0.01,y:0.99,bgcolor:'rgba(2,7,19,.58)'}}, margin:{{l:0,r:0,t:28,b:0}}, showlegend:true}});
}}
function treeNodesAtLevel(tree, level) {{
  return tree.vertices.filter(v => v.level <= level);
}}
function treeEdgesAtRadius(tree, ids, rIdx) {{
  const visible = new Set(ids);
  return tree.edges.filter(e => e.birth_radius_index <= rIdx && visible.has(e.simplex[0]) && visible.has(e.simplex[1]));
}}
function treeTrianglesAtRadius(tree, ids, rIdx) {{
  const visible = new Set(ids);
  return tree.triangles.filter(t => t.birth_radius_index <= rIdx && visible.has(t.simplex[0]) && visible.has(t.simplex[1]) && visible.has(t.simplex[2]));
}}
function treeMarkerTrace(tree, name, level, showLabels) {{
  const verts = treeNodesAtLevel(tree, level);
  return {{
    type:'scatter3d', mode:labelMode(showLabels), name:name,
    x:verts.map(v=>v.pca[0]), y:verts.map(v=>v.pca[1]), z:verts.map(v=>v.pca[2]),
    text:verts.map(v=>v.label), textposition:'top center', customdata:verts.map(v=>[v.id,v.label,v.level,v.nll,name]),
    marker:{{size:7,color:verts.map(v=>v.nll),colorscale:'Turbo',showscale:false,line:{{color:'#e8fbff',width:0.5}}}},
    hovertemplate:name+'<br>vertex %{{customdata[0]}} / %{{customdata[1]}}<br>level %{{customdata[2]}}<br>NLL %{{customdata[3]:.3f}}<extra></extra>'
  }};
}}
function treeEdgeTrace(tree, name, level, rIdx, color) {{
  const verts = treeNodesAtLevel(tree, level);
  const byId = Object.fromEntries(verts.map(v => [v.id, v]));
  const ids = verts.map(v => v.id);
  let xs=[], ys=[], zs=[];
  for (const edge of treeEdgesAtRadius(tree, ids, rIdx)) {{
    const a = byId[edge.simplex[0]], b = byId[edge.simplex[1]];
    if (!a || !b) continue;
    xs.push(a.pca[0], b.pca[0], null); ys.push(a.pca[1], b.pca[1], null); zs.push(a.pca[2], b.pca[2], null);
  }}
  return lineTrace(name, xs, ys, zs, color, 'solid', 1.35);
}}
function dagSkeletonTraceForTree(tree, name, level, color) {{
  const verts = treeNodesAtLevel(tree, level);
  const byId = Object.fromEntries(verts.map(v => [v.id, v]));
  let xs=[], ys=[], zs=[];
  for (const edge of trajectory_simplex_payload.dag_edges) {{
    const src = byId[edge[0]], dst = byId[edge[1]];
    if (!src || !dst) continue;
    xs.push(src.pca[0], dst.pca[0], null); ys.push(src.pca[1], dst.pca[1], null); zs.push(src.pca[2], dst.pca[2], null);
  }}
  return lineTrace(name, xs, ys, zs, color, 'solid', 5);
}}
function treeMeshTrace(tree, name, level, rIdx, color) {{
  const verts = treeNodesAtLevel(tree, level);
  const byId = Object.fromEntries(verts.map(v => [v.id, v]));
  const ids = verts.map(v => v.id);
  const tris = treeTrianglesAtRadius(tree, ids, rIdx);
  let points=[], index={{}}, ii=[], jj=[], kk=[];
  function add(id) {{ if (index[id] === undefined) {{ const v=byId[id]; index[id]=points.length; points.push(v.pca); }} return index[id]; }}
  for (const tri of tris) {{
    if (!byId[tri.simplex[0]] || !byId[tri.simplex[1]] || !byId[tri.simplex[2]]) continue;
    ii.push(add(tri.simplex[0])); jj.push(add(tri.simplex[1])); kk.push(add(tri.simplex[2]));
  }}
  return {{type:'mesh3d', x:points.map(p=>p[0]), y:points.map(p=>p[1]), z:points.map(p=>p[2]), i:ii, j:jj, k:kk, name:name, opacity:0.12, color:color, hoverinfo:'skip'}};
}}
function mapArrowTraces(sourceTree, memoryTree, level) {{
  const analogy = trajectory_simplex_payload.analogy;
  const sourceVerts = Object.fromEntries(treeNodesAtLevel(sourceTree, level).map(v => [v.id, v]));
  const memoryVerts = Object.fromEntries(treeNodesAtLevel(memoryTree, level).map(v => [v.id, v]));
  const distances = analogy.candidate_map.vertex_map_distances || [];
  const sorted = distances.slice().sort((a,b) => Number(a)-Number(b));
  const q50 = sorted.length ? Number(sorted[Math.floor(sorted.length * 0.50)]) : 0;
  const q85 = sorted.length ? Number(sorted[Math.floor(sorted.length * 0.85)]) : q50;
  const buckets = [
    {{name:'candidate nearest-neighbor simplex-tree map arrows · close original-embedding distance', color:'rgba(140,255,106,.76)', xs:[], ys:[], zs:[]}},
    {{name:'candidate nearest-neighbor simplex-tree map arrows · medium original-embedding distance', color:'rgba(255,209,102,.70)', xs:[], ys:[], zs:[]}},
    {{name:'candidate nearest-neighbor simplex-tree map arrows · far original-embedding distance', color:'rgba(255,123,156,.66)', xs:[], ys:[], zs:[]}}
  ];
  for (let src=0; src<analogy.candidate_map.vertex_map.length; src++) {{
    const dst = analogy.candidate_map.vertex_map[src];
    const s = sourceVerts[src], m = memoryVerts[dst];
    if (!s || !m) continue;
    const dist = Number(distances[src] || 0);
    const bucket = dist <= q50 ? buckets[0] : (dist <= q85 ? buckets[1] : buckets[2]);
    bucket.xs.push(s.pca[0], m.pca[0], null); bucket.ys.push(s.pca[1], m.pca[1], null); bucket.zs.push(s.pca[2], m.pca[2], null);
  }}
  return buckets.map(bucket => lineTrace(bucket.name, bucket.xs, bucket.ys, bucket.zs, bucket.color, 'dot', 3));
}}
function analogyBannerHTML(analogy) {{
  const narrative = analogy.decision_summary ? analogy.decision_summary.narrative : '';
  if (!analogy.analogy_emitted) {{
    return '<div class="banner bad"><strong>No analogy emitted.</strong> '+analogy.no_analogy_reason+
      ' Candidate map arrows are still drawn so the rejected nearest-neighbor simplex-tree map can be inspected. '+
      'Map <code>'+Number(analogy.candidate_map.valid_fraction).toFixed(4)+'</code>, step-map <code>'+
      Number(analogy.step_simplicial_map_valid_fraction_mean).toFixed(4)+'</code>, PH gate <code>'+
      Number(analogy.analogy_passed_checks.ph_gate_score).toFixed(4)+'</code>. '+narrative+'</div>';
  }}
  const cls = analogy.analogy_status === 'strong_analogy' ? 'ok' : (analogy.analogy_status === 'weak_analogy' ? 'warn' : 'warn');
  return '<div class="banner"><strong class="'+cls+'">'+analogy.analogy_status.replaceAll('_',' ')+'</strong> emitted. '+
    'Confidence <code>'+Number(analogy.analogy_confidence_score).toFixed(4)+'</code>; '+
    'map <code>'+Number(analogy.candidate_map.valid_fraction).toFixed(4)+'</code>, '+
    'step-map <code>'+Number(analogy.step_simplicial_map_valid_fraction_mean).toFixed(4)+'</code>, '+
    'PH gate <code>'+Number(analogy.analogy_passed_checks.ph_gate_score).toFixed(4)+'</code>. '+narrative+'</div>';
}}
function renderTopDecisionPanel() {{
  const analogy = trajectory_simplex_payload.analogy;
  document.getElementById('top_analogy_decision_banner').innerHTML = analogyBannerHTML(analogy);
  document.getElementById('top_analogy_decision_status_badges').innerHTML = decisionStatusBadgesHTML();
  document.getElementById('top_analogy_threshold_table').innerHTML = decisionThresholdTableHTML();
}}
function renderAnalogy() {{
  const analogy = trajectory_simplex_payload.analogy;
  const sourceTree = analogy.source_simplex_tree;
  const memoryTree = analogy.memory_simplex_tree;
  const rIdx = parseInt(document.getElementById('analogy_radius_slider').value);
  const level = parseInt(document.getElementById('analogy_reasoning_level_slider').value);
  const showTriangles = document.getElementById('analogy_triangle_toggle').checked;
  const showLabels = document.getElementById('analogy_label_toggle').checked;
  const radius = sourceTree.radius_values[Math.min(rIdx, sourceTree.radius_values.length - 1)];
  document.getElementById('analogy_radius_value').textContent = radius.toFixed(3);
  document.getElementById('analogy_reasoning_level_value').textContent = level;
  updateToggleState('analogy_triangle_toggle', 'analogy_triangle_state', 'faces visible', 'faces hidden');
  updateToggleState('analogy_label_toggle', 'analogy_label_state', 'labels visible', 'labels hidden');
  document.getElementById('no_analogy_banner').innerHTML = analogyBannerHTML(analogy);
  document.getElementById('analogy_decision_status_badges').innerHTML = decisionStatusBadgesHTML();
  renderTopDecisionPanel();
  document.getElementById('analogy_map_summary').innerHTML = mapSummaryHTML();
  document.getElementById('ph_quick_summary').innerHTML = phQuickSummaryHTML();
  const sourceIds = treeNodesAtLevel(sourceTree, level).map(v => v.id);
  const memoryIds = treeNodesAtLevel(memoryTree, level).map(v => v.id);
  const sourceEdges = treeEdgesAtRadius(sourceTree, sourceIds, rIdx);
  const memoryEdges = treeEdgesAtRadius(memoryTree, memoryIds, rIdx);
  const sourceTris = treeTrianglesAtRadius(sourceTree, sourceIds, rIdx);
  const memoryTris = treeTrianglesAtRadius(memoryTree, memoryIds, rIdx);
  const visibleArrows = sourceIds.filter(id => analogy.candidate_map.vertex_map[id] !== undefined && memoryIds.includes(analogy.candidate_map.vertex_map[id])).length;
  document.getElementById('analogy_state_caption').innerHTML =
    'radius <code>'+radius.toFixed(3)+'</code> · reasoning level <code>'+level+'</code> · source/memory vertices <code>'+sourceIds.length+' / '+memoryIds.length+'</code> · one-dimensional simplex edges <code>'+sourceEdges.length+' / '+memoryEdges.length+'</code> · available 2-simplex faces <code>'+sourceTris.length+' / '+memoryTris.length+'</code> · rendered map arrows <code>'+visibleArrows+'</code> · label state <code>'+(showLabels ? 'visible' : 'hidden')+'</code>';
  const traces = [
    showTriangles ? treeMeshTrace(sourceTree, 'source simplex-tree 2-simplices', level, rIdx, '#37e8ff') : {{type:'mesh3d', x:[], y:[], z:[], i:[], j:[], k:[], name:'source simplex-tree 2-simplices hidden', opacity:0.0}},
    showTriangles ? treeMeshTrace(memoryTree, 'memory simplex-tree 2-simplices', level, rIdx, '#ff4fd8') : {{type:'mesh3d', x:[], y:[], z:[], i:[], j:[], k:[], name:'memory simplex-tree 2-simplices hidden', opacity:0.0}},
    treeEdgeTrace(sourceTree, 'source radius simplex-tree one-dimensional edges', level, rIdx, 'rgba(55,232,255,.34)'),
    treeEdgeTrace(memoryTree, 'memory radius simplex-tree one-dimensional edges', level, rIdx, 'rgba(255,79,216,.34)'),
    dagSkeletonTraceForTree(sourceTree, 'source graph-of-thought branch/merge skeleton', level, 'rgba(140,255,106,.78)'),
    dagSkeletonTraceForTree(memoryTree, 'memory graph-of-thought branch/merge skeleton', level, 'rgba(255,79,216,.72)'),
    ...mapArrowTraces(sourceTree, memoryTree, level),
    treeMarkerTrace(sourceTree, 'source reasoning simplex tree', level, showLabels),
    treeMarkerTrace(memoryTree, 'retrieved memory simplex tree', level, showLabels)
  ];
  Plotly.react('analogy_plot', traces, {{template:'plotly_dark', paper_bgcolor:'#020713', plot_bgcolor:'#020713', scene:{{aspectmode:'data', xaxis:{{title:'PC1 display'}},yaxis:{{title:'PC2 display'}},zaxis:{{title:'PC3 display'}}}}, legend:{{x:0.01,y:0.99,bgcolor:'rgba(2,7,19,.58)'}}, margin:{{l:0,r:0,t:28,b:0}}, showlegend:true}});
  renderValidityTable();
  renderVertexMapTable();
}}
function renderValidityTable() {{
  const map = trajectory_simplex_payload.analogy.candidate_map;
  const rows = [
    '<tr><td>map counts</td><td>vertices</td><td>'+map.vertex_map.length+'</td><td>'+map.vertex_map.length+'</td><td><span class="ok">mapped</span></td><td>nearest-neighbor vertex map in original embedding space</td></tr>',
    '<tr><td>map counts</td><td>one-dimensional simplex images</td><td>'+map.valid_edges+'</td><td>'+map.source_edges+'</td><td>'+(map.invalid_edges === 0 ? '<span class="ok">all valid/collapsed</span>' : '<span class="warn">some invalid</span>')+'</td><td>edge images must be target edges or collapsed vertices</td></tr>',
    '<tr><td>map counts</td><td>2-simplex images</td><td>'+map.valid_triangles+'</td><td>'+map.source_triangles+'</td><td>'+(map.invalid_triangles === 0 ? '<span class="ok">all valid/collapsed</span>' : '<span class="warn">some invalid</span>')+'</td><td>triangle images must be target triangles or collapsed faces</td></tr>',
    decisionConditionRowsHTML()
  ];
  document.getElementById('simplicial_map_validity_table').innerHTML =
    '<table><tr><th>group</th><th>measure</th><th>score/count</th><th>threshold/total</th><th>result</th><th>meaning</th></tr>'+
    rows.join('')+
    '</table>';
}}
function analogyVertexDetailsHTML(side, vertex) {{
  return '<strong>'+side+' analogy vertex '+vertex.id+' / '+vertex.label+'</strong><br>'+
    'level <code>'+vertex.level+'</code> · NLL <code>'+Number(vertex.nll).toFixed(4)+'</code><br>'+
    'This vertex is a reasoning-step node in the full trajectory simplex tree; comparisons and nearest-neighbor maps use the original embedding coordinates.';
}}
function renderVertexMapTable() {{
  const analogy = trajectory_simplex_payload.analogy;
  const sourceTree = analogy.source_simplex_tree;
  const memoryTree = analogy.memory_simplex_tree;
  const map = analogy.candidate_map;
  const memoryById = Object.fromEntries(memoryTree.vertices.map(v => [v.id, v]));
  const rows = sourceTree.vertices.map(src => {{
    const targetId = map.vertex_map[src.id];
    const dst = memoryById[targetId];
    const distance = map.vertex_map_distances ? map.vertex_map_distances[src.id] : NaN;
    return '<tr><td>'+src.id+'</td><td>'+src.label+'</td><td>'+src.level+'</td><td>'+Number(src.nll).toFixed(3)+'</td><td>'+targetId+'</td><td>'+(dst ? dst.label : '')+'</td><td>'+(dst ? dst.level : '')+'</td><td>'+(dst ? Number(dst.nll).toFixed(3) : '')+'</td><td>'+Number(distance).toFixed(4)+'</td></tr>';
  }}).join('');
  document.getElementById('analogy_vertex_map_table').innerHTML =
    '<div class="muted">Nearest-neighbor vertex map in the original embedding space.  The 3D PCA view is display-only.</div>'+
    '<div class="scrollbox"><table><tr><th>source id</th><th>source label</th><th>source level</th><th>source NLL</th><th>memory id</th><th>memory label</th><th>memory level</th><th>memory NLL</th><th>embedding distance</th></tr>'+rows+'</table></div>';
}}
function phTable() {{
  const comp = trajectory_simplex_payload.analogy.vectorized_persistence_comparisons;
  let rows = Object.keys(comp).map(k => '<tr><td>'+k+'</td><td>'+Number(comp[k]).toFixed(4)+'</td></tr>').join('');
  document.getElementById('ph_table').innerHTML = '<table><tr><th>vectorized PH feature</th><th>cosine similarity</th></tr>'+rows+'</table>';
  renderPHSimilarityBars();
  renderPHDistanceTable();
  document.getElementById('analogy_decision').textContent = JSON.stringify(trajectory_simplex_payload.analogy, null, 2);
  document.getElementById('raw_payload').textContent = JSON.stringify({{schema:trajectory_simplex_payload.schema, source_mode:trajectory_simplex_payload.source_mode, source_metadata:trajectory_simplex_payload.source_metadata, controls:trajectory_simplex_payload.controls, analogy:trajectory_simplex_payload.analogy, gflownet_flow:gflownet_flow}}, null, 2);
}}
function renderPHSimilarityBars() {{
  const comp = trajectory_simplex_payload.analogy.vectorized_persistence_comparisons;
  const keys = Object.keys(comp);
  const values = keys.map(k => comp[k]);
  Plotly.react('ph_similarity_bar_plot', [{{type:'bar', x:keys, y:values, marker:{{color:'#37e8ff'}}, hovertemplate:'%{{x}}<br>cosine %{{y:.4f}}<extra></extra>'}}], {{template:'plotly_dark', title:'Vectorized PH Similarity By Feature Family', paper_bgcolor:'#020713', plot_bgcolor:'#020713', yaxis:{{title:'cosine similarity', range:[0,1]}}, margin:{{l:44,r:18,t:42,b:76}}}});
}}
function renderPHDistanceTable() {{
  const dist = trajectory_simplex_payload.analogy.persistence_diagram_distances;
  const dims = Object.keys(dist.by_dimension || {{}}).sort();
  const rows = dims.map(dim => {{
    const item = dist.by_dimension[dim];
    const wasserstein = item.wasserstein_distance === null || item.wasserstein_distance === undefined ? item.wasserstein_status : Number(item.wasserstein_distance).toFixed(4);
    return '<tr><td>H'+dim+'</td><td>'+Number(item.bottleneck_distance).toFixed(4)+'</td><td>'+wasserstein+'</td><td>'+Number(item.finite_lifetime_l1).toFixed(4)+'</td><td>'+item.source_interval_count+'</td><td>'+item.memory_interval_count+'</td></tr>';
  }}).join('');
  document.getElementById('ph_distance_table').innerHTML =
    '<table><tr><th>dimension</th><th>exact bottleneck</th><th>Wasserstein status/value</th><th>finite lifetime L1</th><th>source intervals</th><th>memory intervals</th></tr>'+rows+'</table>';
}}
function phSide(side) {{ return trajectory_simplex_payload.analogy.ph_features[side].rendered; }}
function phGrid(side) {{ return trajectory_simplex_payload.analogy.ph_features[side].feature_grid; }}
function renderPersistenceDiagrams() {{
  const source = phSide('source'), memory = phSide('memory');
  let traces=[];
  for (const dim of ['0','1','2']) {{
    traces.push({{type:'scatter', mode:'markers', name:'source H'+dim+' diagram', x:source[dim].diagram.map(p=>p[0]), y:source[dim].diagram.map(p=>p[1]), marker:{{size:7}}, hovertemplate:'source H'+dim+'<br>birth %{{x:.3f}}<br>death %{{y:.3f}}<extra></extra>'}});
    traces.push({{type:'scatter', mode:'markers', name:'memory H'+dim+' diagram', x:memory[dim].diagram.map(p=>p[0]), y:memory[dim].diagram.map(p=>p[1]), marker:{{size:7,symbol:'diamond'}}, hovertemplate:'memory H'+dim+'<br>birth %{{x:.3f}}<br>death %{{y:.3f}}<extra></extra>'}});
  }}
  traces.push({{type:'scatter', mode:'lines', name:'birth=death diagonal', x:[0, Math.max(...phGrid('source'), ...phGrid('memory'))], y:[0, Math.max(...phGrid('source'), ...phGrid('memory'))], line:{{color:'rgba(232,251,255,.35)',dash:'dash'}}}});
  Plotly.react('persistence_diagram_plot', traces, {{template:'plotly_dark', title:'GUDHI Persistence Diagrams H0/H1/H2', paper_bgcolor:'#020713', plot_bgcolor:'#020713', xaxis:{{title:'birth'}}, yaxis:{{title:'death'}}, margin:{{l:44,r:18,t:42,b:42}}}});
}}
function renderLandscapePlot() {{
  const source = phSide('source'), memory = phSide('memory'), grid = phGrid('source');
  let traces=[];
  for (const dim of ['0','1','2']) {{
    for (let layer=0; layer<source[dim].landscape.length; layer++) {{
      traces.push({{type:'scatter', mode:'lines', name:'source H'+dim+' landscape '+layer, x:grid, y:source[dim].landscape[layer], line:{{width:1.4}}, opacity:0.70}});
      traces.push({{type:'scatter', mode:'lines', name:'memory H'+dim+' landscape '+layer, x:grid, y:memory[dim].landscape[layer], line:{{width:1.4,dash:'dot'}}, opacity:0.70}});
    }}
  }}
  Plotly.react('persistence_landscape_plot', traces, {{template:'plotly_dark', title:'Persistence Landscape Plot', paper_bgcolor:'#020713', plot_bgcolor:'#020713', xaxis:{{title:'filtration value'}}, yaxis:{{title:'landscape value'}}, margin:{{l:44,r:18,t:42,b:42}}}});
}}
function flatten2d(arr) {{
  let out=[];
  for (const row of arr) for (const value of row) out.push(value);
  return out;
}}
function vectorDifference(a, b) {{
  const n = Math.min(a.length, b.length);
  let out=[];
  for (let i=0;i<n;i++) out.push(Number(a[i])-Number(b[i]));
  return out;
}}
function renderPHDifferencePlot() {{
  const source = phSide('source'), memory = phSide('memory');
  const dim = '1';
  const traces = [
    {{type:'scatter', mode:'lines', name:'H1 landscape source-minus-memory', y:vectorDifference(flatten2d(source[dim].landscape), flatten2d(memory[dim].landscape)), line:{{color:'#37e8ff',width:2}}}},
    {{type:'scatter', mode:'lines', name:'H1 persistence image source-minus-memory', y:vectorDifference(flatten2d(source[dim].persistence_image), flatten2d(memory[dim].persistence_image)), line:{{color:'#ff4fd8',width:2}}}},
    {{type:'scatter', mode:'lines', name:'H1 silhouette source-minus-memory', y:vectorDifference(source[dim].silhouette, memory[dim].silhouette), line:{{color:'#8cff6a',width:2}}}},
    {{type:'scatter', mode:'lines', name:'H1 entropy vector source-minus-memory', y:vectorDifference(source[dim].entropy_vector, memory[dim].entropy_vector), line:{{color:'#ffd166',width:2}}}},
    {{type:'scatter', mode:'lines', name:'H1 Betti curve source-minus-memory', y:vectorDifference(source[dim].betti_curve, memory[dim].betti_curve), line:{{color:'#f4978e',width:2}}}}
  ];
  Plotly.react('ph_difference_plot', traces, {{template:'plotly_dark', title:'Vectorized PH Difference Plot (Source - Memory, H1)', paper_bgcolor:'#020713', plot_bgcolor:'#020713', xaxis:{{title:'vector coordinate'}}, yaxis:{{title:'difference'}}, margin:{{l:44,r:18,t:42,b:42}}}});
}}
function renderImageHeatmap() {{
  const source = phSide('source'), memory = phSide('memory');
  const dim = '1';
  const traces = [
    {{type:'heatmap', z:source[dim].persistence_image, colorscale:'Viridis', name:'source H1 persistence image', xaxis:'x', yaxis:'y', colorbar:{{title:'source', x:0.46}}}},
    {{type:'heatmap', z:memory[dim].persistence_image, colorscale:'Magma', name:'memory H1 persistence image', xaxis:'x2', yaxis:'y2', colorbar:{{title:'memory', x:1.0}}}}
  ];
  Plotly.react('persistence_image_heatmap', traces, {{template:'plotly_dark', title:'Persistence Image Heatmap H1', paper_bgcolor:'#020713', plot_bgcolor:'#020713', xaxis:{{domain:[0,0.45], title:'birth bin'}}, yaxis:{{domain:[0,1], title:'persistence bin'}}, xaxis2:{{domain:[0.55,1], title:'birth bin'}}, yaxis2:{{domain:[0,1], anchor:'x2', title:'persistence bin'}}, margin:{{l:44,r:18,t:42,b:42}}}});
}}
function vectorLinePlot(divId, key, title) {{
  const source = phSide('source'), memory = phSide('memory');
  let traces=[];
  for (const dim of ['0','1','2']) {{
    traces.push({{type:'scatter', mode:'lines', name:'source H'+dim+' '+key, y:source[dim][key], line:{{width:2}}}});
    traces.push({{type:'scatter', mode:'lines', name:'memory H'+dim+' '+key, y:memory[dim][key], line:{{width:2,dash:'dot'}}}});
  }}
  Plotly.react(divId, traces, {{template:'plotly_dark', title:title, paper_bgcolor:'#020713', plot_bgcolor:'#020713', xaxis:{{title:'vector coordinate'}}, yaxis:{{title:key}}, margin:{{l:44,r:18,t:42,b:42}}}});
}}
function renderLifetimeHistogram() {{
  const source = phSide('source'), memory = phSide('memory');
  let traces=[];
  for (const dim of ['0','1','2']) {{
    traces.push({{type:'bar', name:'source H'+dim+' lifetime histogram', x:source[dim].lifetime_histogram.bin_edges.slice(0,-1), y:source[dim].lifetime_histogram.counts, opacity:0.65}});
    traces.push({{type:'bar', name:'memory H'+dim+' lifetime histogram', x:memory[dim].lifetime_histogram.bin_edges.slice(0,-1), y:memory[dim].lifetime_histogram.counts, opacity:0.45}});
  }}
  Plotly.react('lifetime_histogram_plot', traces, {{template:'plotly_dark', title:'Lifetime Histogram Plot', barmode:'overlay', paper_bgcolor:'#020713', plot_bgcolor:'#020713', xaxis:{{title:'lifetime'}}, yaxis:{{title:'count'}}, margin:{{l:44,r:18,t:42,b:42}}}});
}}
function renderPHPanels() {{
  renderPersistenceDiagrams();
  renderLandscapePlot();
  renderPHDifferencePlot();
  renderImageHeatmap();
  vectorLinePlot('silhouette_vector_plot', 'silhouette', 'Silhouette Vector Plot');
  vectorLinePlot('entropy_vector_plot', 'entropy_vector', 'Entropy Vector Plot');
  vectorLinePlot('betti_curve_plot', 'betti_curve', 'Betti Curve Plot');
  renderLifetimeHistogram();
}}
function setAuditRange(id, fraction) {{
  const el = document.getElementById(id);
  if (!el) return false;
  const min = Number(el.min || 0), max = Number(el.max || 0);
  el.value = String(Math.round(min + (max - min) * fraction));
  el.dispatchEvent(new Event('input', {{bubbles:true}}));
  return true;
}}
function setAuditCheck(id, checked) {{
  const el = document.getElementById(id);
  if (!el) return false;
  el.checked = Boolean(checked);
  el.dispatchEvent(new Event('change', {{bubbles:true}}));
  return true;
}}
function revealAuditDetails() {{
  const visibleNodes = trajectory_simplex_payload.nodes.filter(n => n.level <= Number(document.getElementById('reasoning_level_slider').value));
  const node = visibleNodes[Math.max(0, Math.floor(visibleNodes.length * 0.62) - 1)] || trajectory_simplex_payload.nodes[0];
  if (node) {{
    document.getElementById('node_detail_panel').innerHTML = nodeDetailsHTML(node);
    renderStep(node.id);
    const step = simplex_step_payload[node.id] || simplex_step_payload[0];
    const token = step.tokens[Math.max(0, Math.floor(step.tokens.length * 0.50) - 1)] || step.tokens[0];
    if (token) document.getElementById('token_detail_panel').innerHTML = tokenDetailsHTML(token, step);
  }}
  const sourceVertex = trajectory_simplex_payload.analogy.source_simplex_tree.vertices[0];
  const memoryVertex = trajectory_simplex_payload.analogy.memory_simplex_tree.vertices[0];
  if (sourceVertex && memoryVertex) {{
    document.getElementById('analogy_detail_panel').innerHTML =
      analogyVertexDetailsHTML('source', sourceVertex) + '<hr>' + analogyVertexDetailsHTML('memory', memoryVertex);
  }}
}}
window.toricgtScreenshotAudit = function(mode) {{
  let touched = false;
  if (mode === 'full_mid') {{
    touched = setAuditRange('radius_slider', 0.55) || touched;
    touched = setAuditRange('reasoning_level_slider', 0.55) || touched;
    touched = setAuditCheck('full_label_toggle', true) || touched;
    document.getElementById('trajectory_plot')?.scrollIntoView({{block:'center'}});
  }} else if (mode === 'full_triangles') {{
    touched = setAuditRange('radius_slider', 1.0) || touched;
    touched = setAuditRange('reasoning_level_slider', 1.0) || touched;
    touched = setAuditCheck('full_triangle_toggle', true) || touched;
    touched = setAuditCheck('full_label_toggle', false) || touched;
    document.getElementById('trajectory_plot')?.scrollIntoView({{block:'center'}});
  }} else if (mode === 'step_detail') {{
    touched = setAuditRange('step_radius_slider', 1.0) || touched;
    touched = setAuditRange('decoding_order_slider', 1.0) || touched;
    touched = setAuditCheck('step_label_toggle', true) || touched;
    revealAuditDetails();
    document.getElementById('step_plot')?.scrollIntoView({{block:'center'}});
  }} else if (mode === 'step_triangles') {{
    touched = setAuditRange('step_radius_slider', 1.0) || touched;
    touched = setAuditRange('decoding_order_slider', 1.0) || touched;
    touched = setAuditCheck('step_triangle_toggle', true) || touched;
    touched = setAuditCheck('step_label_toggle', false) || touched;
    revealAuditDetails();
    document.getElementById('step_plot')?.scrollIntoView({{block:'center'}});
  }} else if (mode === 'analogy_detail') {{
    touched = setAuditRange('analogy_radius_slider', 0.65) || touched;
    touched = setAuditRange('analogy_reasoning_level_slider', 1.0) || touched;
    touched = setAuditCheck('analogy_label_toggle', true) || touched;
    revealAuditDetails();
    document.getElementById('analogy_plot')?.scrollIntoView({{block:'center'}});
  }} else if (mode === 'analogy_triangles') {{
    touched = setAuditRange('analogy_radius_slider', 1.0) || touched;
    touched = setAuditRange('analogy_reasoning_level_slider', 1.0) || touched;
    touched = setAuditCheck('analogy_triangle_toggle', true) || touched;
    touched = setAuditCheck('analogy_label_toggle', false) || touched;
    revealAuditDetails();
    document.getElementById('analogy_plot')?.scrollIntoView({{block:'center'}});
  }}
  return touched;
}};
document.getElementById('radius_slider').addEventListener('input', renderTrajectory);
document.getElementById('reasoning_level_slider').addEventListener('input', renderTrajectory);
document.getElementById('step_radius_slider').addEventListener('input', () => renderStep(selectedStep));
document.getElementById('decoding_order_slider').addEventListener('input', () => renderStep(selectedStep));
document.getElementById('analogy_radius_slider').addEventListener('input', renderAnalogy);
document.getElementById('analogy_reasoning_level_slider').addEventListener('input', renderAnalogy);
document.getElementById('full_triangle_toggle').addEventListener('change', renderTrajectory);
document.getElementById('step_triangle_toggle').addEventListener('change', () => renderStep(selectedStep));
document.getElementById('analogy_triangle_toggle').addEventListener('change', renderAnalogy);
document.getElementById('full_label_toggle').addEventListener('change', renderTrajectory);
document.getElementById('step_label_toggle').addEventListener('change', () => renderStep(selectedStep));
document.getElementById('analogy_label_toggle').addEventListener('change', renderAnalogy);
function initializeToricGTReport() {{
  phTable();
  renderTopDecisionPanel();
  renderTrajectory();
  renderStep(0);
  renderAnalogy();
  registerToricGTClickHandlers();
  const renderHeavyPH = () => {{
    try {{
      renderPHPanels();
    }} catch (err) {{
      console.error('ToricGT PH panel render failed', err);
    }}
  }};
  if (window.requestIdleCallback) {{
    window.requestIdleCallback(renderHeavyPH, {{timeout: 1800}});
  }} else {{
    window.setTimeout(renderHeavyPH, 800);
  }}
}}
window.addEventListener('load', () => window.setTimeout(initializeToricGTReport, 0));
function registerToricGTClickHandlers() {{
  const trajectoryGraphDiv = document.getElementById('trajectory_plot');
  if (trajectoryGraphDiv && trajectoryGraphDiv.on && !trajectoryGraphDiv._toricgtBound) {{
    trajectoryGraphDiv._toricgtBound = true;
    trajectoryGraphDiv.on('plotly_click', function(data) {{
      if (data.points && data.points.length && data.points[0].customdata !== undefined) {{
        const raw = data.points[0].customdata;
        const nodeId = Array.isArray(raw) ? Number(raw[0]) : Number(raw);
        const node = trajectory_simplex_payload.nodes.find(n => n.id === nodeId);
        if (node) document.getElementById('node_detail_panel').innerHTML = nodeDetailsHTML(node);
        renderStep(nodeId);
        registerToricGTClickHandlers();
      }}
    }});
  }}
  const stepGraphDiv = document.getElementById('step_plot');
  if (stepGraphDiv && stepGraphDiv.on && !stepGraphDiv._toricgtBound) {{
    stepGraphDiv._toricgtBound = true;
    stepGraphDiv.on('plotly_click', function(data) {{
      if (data.points && data.points.length && data.points[0].customdata !== undefined) {{
        const raw = data.points[0].customdata;
        const tokenId = Array.isArray(raw) ? Number(raw[0]) : Number(raw);
        const step = simplex_step_payload[selectedStep] || simplex_step_payload[0];
        const token = step.tokens.find(t => t.id === tokenId);
        if (token) document.getElementById('token_detail_panel').innerHTML = tokenDetailsHTML(token, step);
      }}
    }});
  }}
  const analogyGraphDiv = document.getElementById('analogy_plot');
  if (analogyGraphDiv && analogyGraphDiv.on && !analogyGraphDiv._toricgtBound) {{
    analogyGraphDiv._toricgtBound = true;
    analogyGraphDiv.on('plotly_click', function(data) {{
      if (data.points && data.points.length && data.points[0].customdata !== undefined) {{
        const raw = data.points[0].customdata;
        if (!Array.isArray(raw) || raw.length < 5) return;
        const side = String(raw[4]).includes('memory') || String(raw[4]).includes('retrieved') ? 'memory' : 'source';
        const tree = side === 'memory' ? trajectory_simplex_payload.analogy.memory_simplex_tree : trajectory_simplex_payload.analogy.source_simplex_tree;
        const vertex = tree.vertices.find(v => v.id === Number(raw[0]));
        if (vertex) document.getElementById('analogy_detail_panel').innerHTML = analogyVertexDetailsHTML(side, vertex);
      }}
    }});
  }}
}}
</script>
</main></body></html>""",
        encoding="utf-8",
    )
    index = output_dir / "index.html"
    index.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Branching Reasoning Trajectory Report</title><style>{CSS}</style></head>
<body><main><section class="hero"><h1>Branching Reasoning Trajectory Report</h1><p>Interactive graph-of-thought branch/merge trajectory with full and per-step filtered simplicial complexes, dual sliders, simplex-map analogical retrieval checks, and GUDHI vectorized PH similarities.</p><p><a class="pill" href="branching_reasoning_trajectory.html">open trajectory report</a><a class="pill" href="branching_reasoning_payload.json">payload JSON</a></p></section><section class="grid"><div class="card"><h2>Summary</h2>{_metrics_html(payload)}</div></section></main></body></html>""",
        encoding="utf-8",
    )
    manifest = {
        "schema": "toricgt.branching_reasoning_visual_report.v1",
        "index_html": "index.html",
        "trajectory_html": "branching_reasoning_trajectory.html",
        "payload_json": "branching_reasoning_payload.json",
        "source_mode": str(payload.get("source_mode", "unknown")),
        "nodes": len(payload["nodes"]),
        "edges": len(payload["dag_edges"]),
        "analogy_emitted": bool(payload["analogy"]["analogy_emitted"]),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
