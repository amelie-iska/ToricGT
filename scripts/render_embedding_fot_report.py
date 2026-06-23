#!/usr/bin/env python3
"""Render an exact embedding-space Forest-of-Thought report for one checkpoint.

The report uses the real OAI checkpoint FoT head saved by ``train_gpt.py`` and
the real hidden-state payload emitted by ``extract_oai_sidecar_embeddings.py``.
It does not synthesize hidden states or forest scores.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toricgt.embedding_forest_of_thought import EmbeddingFoTConfig, EmbeddingForestOfThoughtHead  # noqa: E402


CSS = """
:root { color-scheme: dark; --bg:#06101c; --panel:#0c1928; --ink:#eef8ff; --muted:#a9c5da; --cyan:#31e7ff; --line:#21445c; --gold:#ffd166; --pink:#ff4fd8; }
body { margin:0; background:radial-gradient(circle at top left,#10263d,#06101c 55%); color:var(--ink); font:15px/1.5 system-ui,Segoe UI,sans-serif; }
main { max-width:1280px; margin:0 auto; padding:28px; }
.card { background:rgba(12,25,40,.94); border:1px solid #15516c; border-radius:8px; padding:18px; margin:16px 0; box-shadow:0 12px 40px rgba(0,0,0,.28); }
h1,h2 { margin:0 0 10px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }
.metric { display:flex; justify-content:space-between; gap:12px; border-bottom:1px solid rgba(160,210,240,.12); padding:7px 0; }
.metric span:first-child { color:var(--muted); }
.metric span:last-child { font-variant-numeric:tabular-nums; text-align:right; }
svg { width:100%; height:auto; background:#030a14; border:1px solid #14374d; border-radius:8px; }
.legend { display:flex; flex-wrap:wrap; gap:12px; color:var(--muted); font-size:13px; }
.swatch { display:inline-block; width:12px; height:12px; border-radius:50%; vertical-align:-2px; margin-right:5px; }
code,pre { background:#030a14; border:1px solid #14374d; border-radius:6px; }
pre { overflow:auto; padding:12px; }
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding-npz", required=True)
    parser.add_argument("--embedding-json", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-trees", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-positions", type=int, default=192)
    parser.add_argument("--topk-trees", type=int, default=2)
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def checkpoint_fot_state(path: Path) -> dict[str, torch.Tensor] | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("oai_fot"), dict):
        return None
    state = payload["oai_fot"]
    if not state:
        return None
    return state


def infer_config(state: dict[str, torch.Tensor], dim: int, args: argparse.Namespace) -> EmbeddingFoTConfig:
    hidden_dim = int(state.get("activation_head.0.weight", torch.empty(192, dim)).shape[0])
    branching = int(state.get("forward_policy.2.weight", torch.empty(4, hidden_dim)).shape[0])
    consensus_buckets = int(state.get("consensus_head.2.weight", torch.empty(64, hidden_dim)).shape[0])
    return EmbeddingFoTConfig(
        dim=int(dim),
        num_trees=max(1, int(args.num_trees)),
        max_depth=max(1, int(args.max_depth)),
        branching=max(2, branching),
        topk_trees=max(1, int(args.topk_trees)),
        hidden_dim=max(1, hidden_dim),
        max_positions=max(2, int(args.max_positions)),
        consensus_buckets=max(2, consensus_buckets),
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_xy(points: np.ndarray, width: int, height: int, pad: int = 45) -> np.ndarray:
    xy = np.asarray(points[:, :2], dtype=np.float64)
    if xy.shape[0] == 0:
        return xy
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1.0e-8)
    out = (xy - lo) / span
    out[:, 0] = pad + out[:, 0] * (width - 2 * pad)
    out[:, 1] = height - pad - out[:, 1] * (height - 2 * pad)
    return out


def color_for(value: float, lo: float, hi: float) -> str:
    if hi <= lo:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    r = int(49 + 206 * t)
    g = int(231 - 110 * t)
    b = int(255 - 220 * t)
    return f"rgb({r},{g},{b})"


def pca3_np(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    dim = min(3, vh.shape[0])
    out = x @ vh[:dim].T
    if dim < 3:
        out = np.pad(out, ((0, 0), (0, 3 - dim)))
    scale = np.maximum(out.std(axis=0, keepdims=True), 1.0e-8)
    return out / scale


def pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))


def edge_births_from_points(points: np.ndarray, levels: int = 7) -> tuple[list[float], list[list[float]]]:
    dist = pairwise_distance_matrix(points)
    if dist.shape[0] < 2:
        return [0.0], []
    positive = dist[dist > 1.0e-10]
    max_radius = float(np.quantile(positive, 0.72)) if positive.size else 1.0
    radii = np.linspace(max_radius / max(2, int(levels)), max_radius, max(2, int(levels)))
    births: list[list[float]] = []
    for i in range(dist.shape[0]):
        for j in range(i + 1, dist.shape[1]):
            value = float(dist[i, j])
            if not math.isfinite(value) or value > float(radii[-1]):
                continue
            birth_idx = int(np.searchsorted(radii, value, side="left"))
            births.append([int(i), int(j), int(birth_idx), float(round(value, 6))])
    return [float(round(v, 6)) for v in radii], births


@torch.no_grad()
def build_explicit_fot_forest(
    head: EmbeddingForestOfThoughtHead,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    per_token_nll: torch.Tensor,
    projected: np.ndarray,
) -> dict[str, Any]:
    """Reconstruct an explicit FoT expansion forest from real checkpoint tensors.

    The FoT head does not persist sampled tree edges in the checkpoint.  This
    audit therefore materializes the forest by applying the actual trained FoT
    heads to the selected hidden states and ranking the forward-policy logits
    inside each tree.  Hidden vectors, NLL, activation/value/correction scores,
    policy logits, and consensus logits are real checkpoint outputs.
    """

    cfg = head.config
    idx = head._selected_indices(int(hidden.shape[1]), hidden.device)
    nodes_t = head.node_norm(hidden[:, idx, :].float())
    selected_hidden = nodes_t[0].detach().cpu().numpy().astype(np.float64)
    raw_selected_hidden = hidden[0, idx, :].detach().float().cpu().numpy().astype(np.float64)
    positions = idx.detach().cpu().numpy().astype(int)
    if projected.shape[0] > int(positions.max(initial=0)):
        selected_projected = np.asarray(projected[positions], dtype=np.float64)
    else:
        selected_projected = pca3_np(raw_selected_hidden)
    n_nodes = int(nodes_t.shape[1])
    if n_nodes == 0:
        return {
            "schema": "toricgt.embedding_forest_of_thought.explicit_forest.v1",
            "provenance": "empty_hidden_state_selection",
            "nodes": [],
            "edges": [],
        }

    num_trees = max(1, min(int(cfg.num_trees), n_nodes))
    branching = max(2, int(cfg.branching))
    max_depth = max(1, int(cfg.max_depth))
    tree_ids_t = torch.remainder(torch.arange(n_nodes, device=hidden.device), num_trees).long()
    activation = head.activation_head(nodes_t).squeeze(-1).float()[0].detach().cpu().numpy()
    value = head.value_head(nodes_t).squeeze(-1).float()[0].detach().cpu().numpy()
    flow = head.flow_head(nodes_t).squeeze(-1).float()[0].detach().cpu().numpy()
    policy_logits = head.forward_policy(nodes_t).float()[0].detach()
    policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()
    correction = head.correction_head(nodes_t).float()[0].detach().cpu().numpy()
    consensus_logits = head.consensus_head(nodes_t).float()[0].detach().cpu().numpy()
    selected_targets = target_ids[0, idx].detach().cpu().numpy().astype(int)
    selected_nll = per_token_nll[0, idx].detach().float().cpu().numpy().astype(float)

    forest_edges: list[dict[str, Any]] = []
    parent: dict[int, int | None] = {}
    depth: dict[int, int] = {}
    branch_slot: dict[int, int] = {}
    action_id: dict[int, int] = {}
    tree_local_order: dict[int, list[int]] = {}
    tree_leaves: list[int] = []
    tree_summaries: list[dict[str, Any]] = []

    for tree_id in range(num_trees):
        locals_for_tree = [int(i) for i in range(n_nodes) if int(tree_ids_t[i].item()) == tree_id]
        tree_local_order[tree_id] = locals_for_tree
        if not locals_for_tree:
            continue
        root = locals_for_tree[0]
        parent[root] = None
        depth[root] = 0
        branch_slot[root] = 0
        action_id[root] = -1
        queue = [root]
        cursor = 1
        while queue and cursor < len(locals_for_tree):
            src = int(queue.pop(0))
            src_depth = int(depth.get(src, 0))
            if src_depth >= max_depth - 1:
                continue
            order = list(np.argsort(-policy_probs[src]))
            child_budget = min(branching, len(locals_for_tree) - cursor)
            for slot, action in enumerate(order[:child_budget]):
                dst = int(locals_for_tree[cursor])
                cursor += 1
                parent[dst] = src
                depth[dst] = src_depth + 1
                branch_slot[dst] = int(slot)
                action_id[dst] = int(action)
                queue.append(dst)
                delta_nll = float(selected_nll[src] - selected_nll[dst])
                forest_edges.append(
                    {
                        "source": src,
                        "target": dst,
                        "tree_id": int(tree_id),
                        "kind": "expansion",
                        "action_id": int(action),
                        "branch_slot": int(slot),
                        "policy_probability": float(policy_probs[src, int(action)]),
                        "source_depth": int(src_depth),
                        "target_depth": int(src_depth + 1),
                        "nll_delta_improvement": delta_nll,
                        "value_delta": float(value[dst] - value[src]),
                    }
                )
                if cursor >= len(locals_for_tree):
                    break
        while cursor < len(locals_for_tree):
            # Budget overflow is still represented explicitly, instead of
            # silently flattening the remaining selected states into a chain.
            src = int(locals_for_tree[cursor - 1])
            dst = int(locals_for_tree[cursor])
            cursor += 1
            parent[dst] = src
            depth[dst] = min(max_depth, int(depth.get(src, max_depth - 1)) + 1)
            branch_slot[dst] = 0
            action = int(np.argmax(policy_probs[src]))
            action_id[dst] = action
            forest_edges.append(
                {
                    "source": src,
                    "target": dst,
                    "tree_id": int(tree_id),
                    "kind": "budget_continuation",
                    "action_id": action,
                    "branch_slot": 0,
                    "policy_probability": float(policy_probs[src, action]),
                    "source_depth": int(depth.get(src, 0)),
                    "target_depth": int(depth[dst]),
                    "nll_delta_improvement": float(selected_nll[src] - selected_nll[dst]),
                    "value_delta": float(value[dst] - value[src]),
                }
            )

        children = {int(edge["source"]) for edge in forest_edges if int(edge.get("tree_id", -1)) == tree_id and str(edge.get("kind")) in {"expansion", "budget_continuation"}}
        leaves = [node for node in locals_for_tree if node not in children]
        if not leaves:
            leaves = [locals_for_tree[-1]]
        tree_leaves.extend(leaves)
        tree_summaries.append(
            {
                "tree_id": int(tree_id),
                "root": int(root),
                "leaf_ids": [int(v) for v in leaves],
                "node_count": int(len(locals_for_tree)),
                "max_depth": int(max(depth.get(v, 0) for v in locals_for_tree)),
                "mean_activation": float(np.mean(activation[locals_for_tree])),
                "mean_value": float(np.mean(value[locals_for_tree])),
                "mean_nll": float(np.mean(selected_nll[locals_for_tree])),
            }
        )

        correction_candidates: list[tuple[float, dict[str, Any]]] = []
        for a_pos, src in enumerate(locals_for_tree):
            for dst in locals_for_tree[a_pos + 1 :]:
                delta = selected_hidden[dst] - selected_hidden[src]
                denom = float(np.linalg.norm(correction[src]) * np.linalg.norm(delta))
                correction_cos = float(np.dot(correction[src], delta) / denom) if denom > 1.0e-12 else 0.0
                nll_gain = float(selected_nll[src] - selected_nll[dst])
                value_gain = float(value[dst] - value[src])
                score = value_gain + 0.18 * nll_gain + 0.20 * correction_cos
                if score > 0.03 and correction_cos > -0.05:
                    correction_candidates.append(
                        (
                            float(score),
                            {
                                "source": int(src),
                                "target": int(dst),
                                "tree_id": int(tree_id),
                                "kind": "self_correction",
                                "correction_score": float(score),
                                "correction_cosine": float(correction_cos),
                                "nll_delta_improvement": nll_gain,
                                "value_delta": value_gain,
                            },
                        )
                    )
        correction_candidates.sort(key=lambda item: item[0], reverse=True)
        forest_edges.extend(item for _, item in correction_candidates[: max(2, min(6, branching + 1))])

    consensus_id = int(n_nodes)
    leaf_indices = tree_leaves if tree_leaves else list(range(n_nodes))
    consensus_hidden = raw_selected_hidden[leaf_indices].mean(axis=0, keepdims=True)
    comparison_points = np.concatenate([raw_selected_hidden, consensus_hidden], axis=0)
    radius_values, radius_edges = edge_births_from_points(comparison_points, levels=7)
    max_depth_seen = max([int(v) for v in depth.values()] + [0])
    for leaf in leaf_indices:
        forest_edges.append(
            {
                "source": int(leaf),
                "target": consensus_id,
                "tree_id": int(tree_ids_t[leaf].item()),
                "kind": "consensus",
                "policy_probability": 1.0,
                "source_depth": int(depth.get(int(leaf), max_depth_seen)),
                "target_depth": int(max_depth_seen + 1),
                "nll_delta_improvement": float(selected_nll[leaf] - np.mean(selected_nll[leaf_indices])),
                "value_delta": float(np.mean(value[leaf_indices]) - value[leaf]),
            }
        )

    # Process layout: x separates trees, y is growth budget/depth, z separates
    # sibling slots.  This is the display layout for forest topology only.
    tree_width = 5.2
    layout: dict[int, list[float]] = {}
    sibling_counter: dict[tuple[int, int], int] = {}
    for node_id in range(n_nodes):
        tree_id = int(tree_ids_t[node_id].item())
        d = int(depth.get(node_id, node_id // max(1, num_trees)))
        slot_key = (tree_id, d)
        sibling = sibling_counter.get(slot_key, 0)
        sibling_counter[slot_key] = sibling + 1
        centered_tree = tree_id - (num_trees - 1) / 2.0
        x = centered_tree * tree_width + 0.25 * ((sibling % 3) - 1)
        y = float(d) * 1.45
        z = (sibling - 0.5 * max(1, sibling_counter[slot_key])) * 0.62 + 0.18 * float(value[node_id])
        layout[node_id] = [float(x), float(y), float(z)]
    layout[consensus_id] = [0.0, float(max_depth_seen + 1) * 1.45, 0.0]

    forest_nodes: list[dict[str, Any]] = []
    for node_id in range(n_nodes):
        tree_id = int(tree_ids_t[node_id].item())
        forest_nodes.append(
            {
                "id": int(node_id),
                "kind": "thought",
                "tree_id": tree_id,
                "depth": int(depth.get(node_id, 0)),
                "budget_level": int(depth.get(node_id, 0)),
                "parent": parent.get(node_id),
                "branch_slot": int(branch_slot.get(node_id, 0)),
                "action_id": int(action_id.get(node_id, -1)),
                "position": int(positions[node_id]),
                "step_index": int(positions[node_id]),
                "target_id": int(selected_targets[node_id]),
                "nll": float(selected_nll[node_id]),
                "activation": float(activation[node_id]),
                "value": float(value[node_id]),
                "flow": float(flow[node_id]),
                "correction_norm": float(np.linalg.norm(correction[node_id])),
                "consensus_bucket": int(np.argmax(consensus_logits[node_id])),
                "layout": [round(float(v), 6) for v in layout[node_id]],
                "hidden_pca": [round(float(v), 6) for v in selected_projected[node_id].tolist()],
            }
        )
    forest_nodes.append(
        {
            "id": consensus_id,
            "kind": "consensus",
            "tree_id": -1,
            "depth": int(max_depth_seen + 1),
            "budget_level": int(max_depth_seen + 1),
            "parent": None,
            "branch_slot": -1,
            "action_id": -1,
            "position": int(positions[-1]),
            "step_index": int(positions[-1]),
            "target_id": int(selected_targets[-1]),
            "nll": float(np.mean(selected_nll[leaf_indices])),
            "activation": float(np.mean(activation[leaf_indices])),
            "value": float(np.mean(value[leaf_indices])),
            "flow": float(np.mean(flow[leaf_indices])),
            "correction_norm": 0.0,
            "consensus_bucket": int(np.argmax(np.mean(consensus_logits[leaf_indices], axis=0))),
            "layout": [round(float(v), 6) for v in layout[consensus_id]],
            "hidden_pca": [round(float(v), 6) for v in pca3_np(comparison_points)[-1].tolist()],
        }
    )

    return {
        "schema": "toricgt.embedding_forest_of_thought.explicit_forest.v1",
        "provenance": "real_checkpoint_fot_head_reconstructed_from_selected_hidden_states_policy_logits_value_correction_and_consensus",
        "display_space": "forest_process_layout_3d",
        "comparison_space": "original_selected_hidden_states",
        "num_trees": int(num_trees),
        "branching": int(branching),
        "max_depth_config": int(max_depth),
        "node_count": int(len(forest_nodes)),
        "thought_node_count": int(n_nodes),
        "consensus_node_id": int(consensus_id),
        "radius_values": radius_values,
        "radius_edges": radius_edges,
        "nodes": forest_nodes,
        "edges": forest_edges,
        "tree_summaries": tree_summaries,
    }


def svg_forest(trace: dict[str, Any], projected: np.ndarray) -> str:
    forest = trace.get("forest") if isinstance(trace.get("forest"), dict) else None
    nodes = forest.get("nodes", []) if forest else trace.get("nodes", [])
    edges = forest.get("edges", []) if forest else trace.get("edges", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    width, height = 1120, 680
    if forest:
        coords = np.asarray([node.get("layout", [0, 0, 0]) for node in nodes if isinstance(node, dict)], dtype=np.float64)
    else:
        coords = np.asarray(projected[: len(nodes)], dtype=np.float64)
    xy = normalize_xy(coords, width, height)
    nlls = [finite_float(node.get("nll")) for node in nodes if isinstance(node, dict)]
    lo = min(nlls) if nlls else 0.0
    hi = max(nlls) if nlls else 1.0
    tree_colors = ["#31e7ff", "#ff4fd8", "#75ff6a", "#ffd166", "#a78bfa", "#ff8c42", "#6fffe9", "#ff6b91"]
    parts: list[str] = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Embedding FoT forest">']
    parts.append('<rect width="100%" height="100%" fill="#030a14"/>')
    edge_styles = {
        "expansion": ("#31e7ff", "none", 0.60, 2.0),
        "budget_continuation": ("#8cff6a", "7 6", 0.54, 1.6),
        "self_correction": ("#ff4fd8", "5 6", 0.66, 1.7),
        "consensus": ("#ffd166", "8 7", 0.70, 2.1),
        "tree": ("#31e7ff", "none", 0.46, 1.5),
    }
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = int(edge.get("source", 0))
        target = int(edge.get("target", 0))
        tree_id = int(edge.get("tree_id", 0))
        if source >= len(xy) or target >= len(xy):
            continue
        x1, y1 = xy[source]
        x2, y2 = xy[target]
        kind = str(edge.get("kind", "tree"))
        color, dash, opacity, width_px = edge_styles.get(kind, (tree_colors[tree_id % len(tree_colors)], "none", 0.46, 1.5))
        parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width_px}" opacity="{opacity}" stroke-dasharray="{dash}"/>'
        )
    if not forest:
        # Legacy consensus guide lines connect tree leaves to the final sampled node.
        by_tree: dict[int, int] = {}
        for idx, node in enumerate(nodes):
            if isinstance(node, dict):
                by_tree[int(node.get("tree_id", 0))] = idx
        final_idx = len(nodes) - 1
        for tree_id, source in by_tree.items():
            if source == final_idx or source >= len(xy) or final_idx >= len(xy):
                continue
            x1, y1 = xy[source]
            x2, y2 = xy[final_idx]
            parts.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                'stroke="#ffd166" stroke-width="1.2" stroke-dasharray="6 7" opacity="0.58"/>'
            )
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict) or idx >= len(xy):
            continue
        x, y = xy[idx]
        tree_id = int(node.get("tree_id", 0))
        nll = finite_float(node.get("nll"))
        fill = color_for(nll, lo, hi)
        stroke = tree_colors[tree_id % len(tree_colors)]
        kind = str(node.get("kind", "thought"))
        title = html.escape(
            f"{kind} node {idx} tree {tree_id} depth {node.get('depth', 'n/a')} "
            f"pos {node.get('position')} token {node.get('target_id')} "
            f"nll {nll:.4f} activation {finite_float(node.get('activation')):.4f} value {finite_float(node.get('value')):.4f}"
        )
        radius = 8 if kind == "consensus" else 6
        parts.append(f'<g><title>{title}</title><circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        if idx % max(1, len(nodes) // 32) == 0 or idx == final_idx:
            parts.append(f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" fill="#dff8ff" font-size="11">{idx}</text>')
        parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


def metric_rows(metrics: dict[str, Any], trace_metrics: dict[str, Any]) -> str:
    latest = metrics.get("latest_train", {}) if isinstance(metrics.get("latest_train"), dict) else {}
    wanted = {
        "train_bpb": latest.get("train_bpb"),
        "val_or_final_bpb": metrics.get("selected_bpb"),
        "oai_fot_loss": latest.get("oai_fot_loss"),
        "oai_fot_entropy": latest.get("oai_fot_entropy"),
        "oai_fot_reward": latest.get("oai_fot_reward"),
        "oai_fot_diversity": latest.get("oai_fot_diversity"),
        "trace_oai_fot_loss": trace_metrics.get("oai_fot_loss"),
        "trace_sparse_activation_loss": trace_metrics.get("oai_fot_sparse_activation_loss"),
        "trace_ucb_loss": trace_metrics.get("oai_fot_ucb_loss"),
        "trace_correction_loss": trace_metrics.get("oai_fot_self_correction_loss"),
        "trace_consensus_loss": trace_metrics.get("oai_fot_consensus_loss"),
        "trace_tb_residual": trace_metrics.get("oai_fot_tb_residual"),
        "trace_tree_diversity": trace_metrics.get("oai_fot_tree_diversity"),
        "trace_consensus_margin": trace_metrics.get("oai_fot_consensus_margin"),
    }
    rows = []
    for key, value in wanted.items():
        if isinstance(value, torch.Tensor):
            value = float(value.detach().float().item())
        display = f"{float(value):.6g}" if isinstance(value, (int, float)) and math.isfinite(float(value)) else str(value)
        rows.append(f'<div class="metric"><span>{html.escape(key)}</span><span>{html.escape(display)}</span></div>')
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = resolve(args.embedding_npz)
    json_path = resolve(args.embedding_json)
    checkpoint_path = resolve(args.checkpoint)
    metrics = load_json(resolve(args.metrics_json))
    metadata = load_json(json_path)
    arrays = np.load(npz_path)
    hidden = torch.from_numpy(np.asarray(arrays["hidden"], dtype=np.float32)).unsqueeze(0)
    token_ids = torch.from_numpy(np.asarray(arrays["token_ids"], dtype=np.int64)).unsqueeze(0)
    nll = torch.from_numpy(np.asarray(arrays["nll"], dtype=np.float32)).unsqueeze(0)
    projected = np.asarray(arrays["projected"], dtype=np.float32)
    state = checkpoint_fot_state(checkpoint_path)
    latest = metrics.get("latest_train", {}) if isinstance(metrics.get("latest_train"), dict) else {}
    fot_was_logged = isinstance(latest, dict) and latest.get("oai_fot_loss") is not None
    if state is None:
        if fot_was_logged:
            raise RuntimeError(
                f"train log contains FoT metrics, but checkpoint {checkpoint_path} has no `oai_fot` state"
            )
        absent = {
            "schema": "toricgt.embedding_forest_of_thought.absent.v1",
            "checkpoint": str(checkpoint_path),
            "reason": "The checkpoint predates FoT or was trained with OAI_EMBEDDING_FOT=0; no FoT forest is rendered.",
        }
        (output_dir / "embedding_fot_trace.json").write_text(
            json.dumps(absent, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Embedding-Space Forest-of-Thought Audit</title><style>{CSS}</style></head>
<body><main><section class="card"><h1>Embedding-Space Forest-of-Thought Audit</h1>
<p>No FoT state was present in this checkpoint, and no FoT metrics were present in the train log. This is an absence audit, not a proxy visualization.</p>
<pre>{html.escape(json.dumps(absent, indent=2, sort_keys=True))}</pre>
</section></main></body></html>"""
        (output_dir / "index.html").write_text(html_doc, encoding="utf-8")
        print(json.dumps({"index": str(output_dir / "index.html"), "trace": str(output_dir / "embedding_fot_trace.json")}, indent=2))
        return
    config = infer_config(state, int(hidden.shape[-1]), args)
    head = EmbeddingForestOfThoughtHead(config)
    head.load_state_dict(state, strict=True)
    head.eval()
    with torch.no_grad():
        out = head(hidden, token_ids, nll)
        trace = head.trace_payload(hidden, token_ids, nll)
        trace["forest"] = build_explicit_fot_forest(head, hidden, token_ids, nll, projected)
    trace_metrics: dict[str, Any] = {}
    for key, value in out.items():
        if isinstance(value, torch.Tensor):
            trace_metrics[key] = float(value.detach().float().item())
        else:
            trace_metrics[key] = value
    trace["trace_metrics"] = trace_metrics
    trace["embedding_payload"] = str(npz_path)
    trace["embedding_metadata"] = metadata
    write_path = output_dir / "embedding_fot_trace.json"
    write_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    title = "Embedding-Space Forest-of-Thought Audit"
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head>
<body><main>
<section class="card"><h1>{title}</h1>
<p>This report loads the checkpoint's saved FoT head and applies it to the real hidden-state embedding payload from the OAI baseline extractor. The forest is materialized from real activation/value/policy/correction/consensus tensors: cyan edges are branch expansions, magenta dashed edges are self-correction links, and gold dashed edges are consensus links.</p>
<p><a href="embedding_fot_trace.json">trace JSON</a></p>
</section>
<section class="card"><h2>FoT Metrics</h2><div class="grid">{metric_rows(metrics, trace_metrics)}</div></section>
<section class="card"><h2>Hidden Forest</h2>
<div class="legend"><span><i class="swatch" style="background:#31e7ff"></i>tree edges</span><span><i class="swatch" style="background:#ffd166"></i>leaf consensus guide</span><span>node fill: NLL</span></div>
{svg_forest(trace, projected)}
</section>
<section class="card"><h2>Configuration</h2><pre>{html.escape(json.dumps(config.__dict__, indent=2, sort_keys=True))}</pre></section>
</main></body></html>"""
    (output_dir / "index.html").write_text(html_doc, encoding="utf-8")
    print(json.dumps({"index": str(output_dir / "index.html"), "trace": str(write_path)}, indent=2))


if __name__ == "__main__":
    main()
