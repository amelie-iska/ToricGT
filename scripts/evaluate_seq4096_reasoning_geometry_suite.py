#!/usr/bin/env python3
"""Seq4096-compatible advanced geometry and simplex diagnostics.

The native ``evaluate_reasoning_simplex.py`` and
``evaluate_reasoning_geometry_suite.py`` scripts load RandomOrderLM checkpoints.
The OpenAI Parameter-Golf Seq4096 run writes compact GPT-style checkpoints, so
this script builds the same periodic visualization surface from current compact
checkpoint tensors and the live BPB curve:

* checkpoint-embedding reasoning trajectories;
* directed simplicial and persistence/Koszul audits;
* toric shadow and Slepian/Pollak concentration audits;
* tropical chamber and active-face maps;
* GraphCG-style basis disentanglement maps;
* analogical transport maps;
* triangle and tetrahedron projections matching the old output layout.

These are sidecar diagnostics.  They do not claim to be full generated
reasoning traces unless a compact model hidden-state adapter is added later.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toricgt.combinatorial_toric_metrics import CombinatorialToricConfig, combinatorial_toric_cca_topology_loss
from toricgt.complexity import compress_len, normalized_compression_distance
from toricgt.reasoning_geometry import (
    TETRAHEDRON_VERTICES,
    TRIANGLE_VERTICES,
    attach_normalized_scores,
    barycentric_to_cartesian,
    prim_mst_stats,
    rbf_interpolate,
    simplex_record,
    triangle_grid,
)
from toricgt.slepian_torus import ToricSlepianConfig, toric_slepian_audit
from toricgt.topological_reasoning import ReasoningTopologyConfig, directed_step_filtration_stats_np
from toricgt.toric_geometry_tasks import empirical_toric_shadow_stats_np, make_binomial_relations, make_exponent_table


TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+train_loss:(?P<loss>[0-9.eE+-]+)"
    r"\s+train_time:(?P<ms>[0-9.eE+-]+)ms\s+step_avg:(?P<avg>[0-9.eE+-]+)ms"
    r"(?:\s+train_bpb:(?P<bpb>[0-9.eE+-]+))?"
)
VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<loss>[0-9.eE+-]+)"
    r"\s+val_bpb:(?P<bpb>[0-9.eE+-]+)\s+train_time:(?P<ms>[0-9.eE+-]+)ms"
    r"\s+step_avg:(?P<avg>[0-9.eE+-]+)ms"
)

DARK_BG = "#030712"
DARK_PANEL = "#06111f"
DARK_TEXT = "#e8fbff"
DARK_GRID = "#143344"


def blue_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "toricgt_seq4096_reasoning_blues",
        ["#06112a", "#082f63", "#0b5ea8", "#39b8ff", "#dff8ff"],
    )


def flatten_axes(axes: Any) -> list[Any]:
    if isinstance(axes, np.ndarray):
        return [axis for axis in axes.ravel()]
    if isinstance(axes, (list, tuple)):
        flattened: list[Any] = []
        for item in axes:
            flattened.extend(flatten_axes(item))
        return flattened
    return [axes]


def style_dark_axes(fig: Any, axes: Any, *, grid: bool = True) -> None:
    fig.patch.set_facecolor(DARK_BG)
    for ax in flatten_axes(axes):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=DARK_TEXT)
        ax.title.set_color("white")
        ax.xaxis.label.set_color(DARK_TEXT)
        ax.yaxis.label.set_color(DARK_TEXT)
        if hasattr(ax, "zaxis"):
            ax.zaxis.label.set_color(DARK_TEXT)
        for spine in getattr(ax, "spines", {}).values():
            spine.set_color("#2de2e6")
            spine.set_alpha(0.45)
        if grid:
            ax.grid(color=DARK_GRID, alpha=0.35)


def style_dark_colorbar(cbar: Any) -> None:
    cbar.set_label(cbar.ax.get_ylabel(), color=DARK_TEXT)
    cbar.ax.yaxis.set_tick_params(color=DARK_TEXT)
    plt.setp(cbar.ax.get_yticklabels(), color=DARK_TEXT)
    cbar.outline.set_edgecolor("#2de2e6")
    cbar.outline.set_alpha(0.45)


def save_dark(fig: Any, out: Path, *, dpi: int = 180) -> None:
    fig.savefig(out, dpi=dpi, facecolor=fig.get_facecolor())


TRIANGLE_SPECS = {
    "reasoning_k_bpb": {
        "labels": ["trajectory structure", "low K proxy", "low BPB"],
        "scores": ["score/trajectory_depth", "score/relative_k", "score/bpb_quality"],
        "title": "Seq4096 reasoning/K(x|helpers)/BPB simplex",
    },
    "solution_basin": {
        "labels": ["analogical transfer", "terminal confidence", "low BPB"],
        "scores": ["score/analogy_quality", "score/terminal_confidence", "score/bpb_quality"],
        "title": "Seq4096 analogical solution-basin simplex",
    },
    "trajectory_flow": {
        "labels": ["smooth flow", "MST efficiency", "trajectory depth"],
        "scores": ["score/path_smoothness", "score/mst_efficiency", "score/trajectory_depth"],
        "title": "Seq4096 embedding trajectory flow simplex",
    },
    "exploration_control": {
        "labels": ["GraphCG basis", "low curvature", "low BPB"],
        "scores": ["score/graphcg_basis", "score/path_smoothness", "score/bpb_quality"],
        "title": "Seq4096 GraphCG/control simplex",
    },
    "compression_reasoning": {
        "labels": ["low K proxy", "helper gain", "analogy quality"],
        "scores": ["score/relative_k", "score/relative_k_gain", "score/analogy_quality"],
        "title": "Seq4096 compression/reasoning simplex",
    },
    "robustness_generalization": {
        "labels": ["order robustness", "GraphCG basis", "low loss"],
        "scores": ["score/order_robustness", "score/graphcg_basis", "score/loss_quality"],
        "title": "Seq4096 order-robust generalization simplex",
    },
    "toric_memory_control": {
        "labels": ["toric entropy", "Slepian concentration", "smooth flow"],
        "scores": ["score/toric_entropy", "score/slepian_concentration", "score/path_smoothness"],
        "title": "Seq4096 toric-memory control simplex",
    },
    "energy_landscape": {
        "labels": ["low energy", "terminal confidence", "MST efficiency"],
        "scores": ["score/energy_quality", "score/terminal_confidence", "score/mst_efficiency"],
        "title": "Seq4096 energy-control simplex",
    },
    "combinatorial_cca_control": {
        "labels": ["low toric ideal residual", "low Stanley-Reisner mass", "low BPB"],
        "scores": ["score/toric_ideal_consistency", "score/stanley_reisner_consistency", "score/bpb_quality"],
        "title": "Seq4096 combinatorial CCA/BPB simplex",
    },
}

TETRAHEDRON_SPECS = {
    "reasoning_k_bpb_mst": {
        "labels": ["trajectory structure", "low K proxy", "low BPB", "MST efficiency"],
        "scores": ["score/trajectory_depth", "score/relative_k", "score/bpb_quality", "score/mst_efficiency"],
        "title": "Seq4096 reasoning/K(x|helpers)/BPB/MST tetrahedron",
    },
    "solution_bpb_smooth_diversity": {
        "labels": ["analogical transfer", "low BPB", "smooth flow", "GraphCG basis"],
        "scores": ["score/analogy_quality", "score/bpb_quality", "score/path_smoothness", "score/graphcg_basis"],
        "title": "Seq4096 analogy/BPB/flow/basis tetrahedron",
    },
    "complexity_geometry_solution": {
        "labels": ["low K proxy", "trajectory depth", "MST efficiency", "analogy quality"],
        "scores": ["score/relative_k", "score/trajectory_depth", "score/mst_efficiency", "score/analogy_quality"],
        "title": "Seq4096 complexity/geometry/analogy tetrahedron",
    },
    "energy_control": {
        "labels": ["low energy", "terminal confidence", "smooth flow", "low BPB"],
        "scores": ["score/energy_quality", "score/terminal_confidence", "score/path_smoothness", "score/bpb_quality"],
        "title": "Seq4096 energy-control tetrahedron",
    },
    "toric_gfn_bpb": {
        "labels": ["toric entropy", "GraphCG basis", "low BPB", "order robustness"],
        "scores": ["score/toric_entropy", "score/graphcg_basis", "score/bpb_quality", "score/order_robustness"],
        "title": "Seq4096 toric/GraphCG/BPB tetrahedron",
    },
    "cca_topology_bpb": {
        "labels": ["low CCA topology", "low Betti1", "low BPB", "GraphCG basis"],
        "scores": ["score/cca_topology_consistency", "score/betti1_consistency", "score/bpb_quality", "score/graphcg_basis"],
        "title": "Seq4096 CCA/topology/BPB tetrahedron",
    },
}

SIMPLEX_TRIANGLES = {
    "reasoning_k_bpb_triangle": TRIANGLE_SPECS["reasoning_k_bpb"],
    "robustness_triangle": TRIANGLE_SPECS["robustness_generalization"],
    "efficiency_triangle": {
        "labels": ["low BPB", "MST efficiency", "low K proxy"],
        "scores": ["score/bpb_quality", "score/mst_efficiency", "score/relative_k"],
        "title": "Seq4096 compression-efficiency simplex",
    },
    "cca_topology_triangle": TRIANGLE_SPECS["combinatorial_cca_control"],
}

SIMPLEX_TETRAHEDRA = {
    "reasoning_k_bpb_mst_tetrahedron": TETRAHEDRON_SPECS["reasoning_k_bpb_mst"],
    "reasoning_k_bpb_diversity_tetrahedron": {
        "labels": ["trajectory structure", "low K proxy", "low BPB", "GraphCG basis"],
        "scores": ["score/trajectory_depth", "score/relative_k", "score/bpb_quality", "score/graphcg_basis"],
        "title": "Seq4096 reasoning/K(x|helpers)/BPB/GraphCG tetrahedron",
    },
    "cca_topology_tetrahedron": TETRAHEDRON_SPECS["cca_topology_bpb"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--log", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-path", default="")
    parser.add_argument("--target-bpb", type=float, default=1.2)
    parser.add_argument("--max-points", type=int, default=96)
    parser.add_argument("--records", type=int, default=6)
    parser.add_argument("--seed", type=int, default=10017)
    parser.add_argument("--topology-max-points", type=int, default=28)
    parser.add_argument("--topology-window-size", type=int, default=32)
    parser.add_argument("--topology-levels", type=int, default=4)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], int, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a dict: {path}")
    state = payload.get("model")
    if not isinstance(state, dict) or "tok_emb.weight" not in state:
        raise ValueError("checkpoint is not a compact Seq4096 payload with model.tok_emb.weight")
    tensor_state = {str(key): value for key, value in state.items() if isinstance(value, torch.Tensor)}
    step = int(payload.get("step", 0) or 0)
    return tensor_state, step, payload


def parse_training_log(path: Path, target_bpb: float) -> dict[str, Any]:
    train_rows: dict[int, dict[str, float]] = {}
    val_rows: dict[int, dict[str, float]] = {}
    if path and path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            val = VAL_RE.search(line)
            if val:
                step = int(val.group("step"))
                val_rows[step] = {
                    "step": float(step),
                    "loss": float(val.group("loss")),
                    "bpb": float(val.group("bpb")),
                }
                continue
            train = TRAIN_RE.search(line)
            if train:
                step = int(train.group("step"))
                bpb = train.group("bpb")
                train_rows[step] = {
                    "step": float(step),
                    "loss": float(train.group("loss")),
                    "bpb": float(bpb) if bpb is not None else float("nan"),
                }
    latest_train = train_rows[max(train_rows)] if train_rows else {}
    latest_val = val_rows[max(val_rows)] if val_rows else {}
    best_val = min((row["bpb"] for row in val_rows.values()), default=float("nan"))
    latest_bpb = latest_val.get("bpb", latest_train.get("bpb", float("nan")))
    return {
        "train_rows": list(sorted(train_rows.values(), key=lambda row: row["step"])),
        "val_rows": list(sorted(val_rows.values(), key=lambda row: row["step"])),
        "latest_train_bpb": latest_train.get("bpb", float("nan")),
        "latest_val_bpb": latest_val.get("bpb", float("nan")),
        "best_val_bpb": best_val,
        "latest_bpb": latest_bpb,
        "target_bpb": float(target_bpb),
        "bpb_quality": float(max(0.0, min(1.0, 1.0 - max(0.0, latest_bpb - target_bpb) / 0.35)))
        if math.isfinite(float(latest_bpb))
        else 0.5,
        "loss_quality": float(max(0.0, min(1.0, 1.0 / max(1.0, latest_val.get("loss", latest_train.get("loss", 2.0))))))
        if (latest_val or latest_train)
        else 0.5,
    }


def tensor_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def sample_rows(points: np.ndarray, max_points: int, *, by_norm: bool = False) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points[:, None]
    if points.shape[0] <= max_points:
        return points
    keep = max(4, int(max_points))
    uniform = np.linspace(0, points.shape[0] - 1, num=max(2, keep // 2)).round().astype(int)
    if by_norm:
        norms = np.linalg.norm(points, axis=1)
        top = np.argsort(norms)[-max(2, keep - len(uniform)) :]
    else:
        top = np.linspace(points.shape[0] // (2 * keep), points.shape[0] - 1, num=max(2, keep - len(uniform))).round().astype(int)
    indices = np.unique(np.concatenate([uniform, top]))
    if indices.size > keep:
        indices = indices[:keep]
    return points[np.sort(indices)]


def standardize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points[:, None]
    points = points - np.nanmean(points, axis=0, keepdims=True)
    scale = np.nanstd(points, axis=0, keepdims=True)
    points = points / np.maximum(scale, 1e-6)
    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    return points.astype(np.float32)


def pca_project(points: np.ndarray, dims: int = 3) -> tuple[np.ndarray, np.ndarray]:
    points = standardize_points(points)
    dims = max(1, int(dims))
    if points.shape[0] == 0:
        return np.zeros((0, dims), dtype=np.float32), np.zeros((dims,), dtype=np.float32)
    if points.shape[0] == 1:
        return np.zeros((1, dims), dtype=np.float32), np.zeros((dims,), dtype=np.float32)
    try:
        _, singular, vh = np.linalg.svd(points, full_matrices=False)
        components = vh[:dims].T
        proj = points @ components
        if proj.shape[1] < dims:
            proj = np.pad(proj, ((0, 0), (0, dims - proj.shape[1])))
        total = float(np.sum(singular**2))
        explained = (singular[:dims] ** 2 / max(total, 1e-12)).astype(np.float32)
        if explained.shape[0] < dims:
            explained = np.pad(explained, (0, dims - explained.shape[0]))
        return proj[:, :dims].astype(np.float32), explained[:dims]
    except np.linalg.LinAlgError:
        return np.zeros((points.shape[0], dims), dtype=np.float32), np.zeros((dims,), dtype=np.float32)


def normalize01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def phase_from_projection(proj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if proj.shape[1] < 3:
        proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])))
    u = (np.arctan2(proj[:, 1], proj[:, 0]) / (2.0 * np.pi) + 1.0) % 1.0
    v = (np.arctan2(proj[:, 2], proj[:, 1] + 1e-6) / (2.0 * np.pi) + 1.0) % 1.0
    return u.astype(np.float64), v.astype(np.float64)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-")[:96] or "record"


def layer_ids(state: dict[str, torch.Tensor]) -> list[int]:
    ids = {
        int(parts[1])
        for key in state
        for parts in [key.split(".")]
        if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit()
    }
    return sorted(ids)


def pad_or_trim(vector: np.ndarray, dim: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size >= dim:
        return vector[:dim]
    return np.pad(vector, (0, dim - vector.size)).astype(np.float32)


def extract_trajectory_inputs(state: dict[str, torch.Tensor], max_points: int, records: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tok = tensor_np(state["tok_emb.weight"])
    out.append(
        {
            "id": "R0_token_embedding_trajectory",
            "family": "embedding_space",
            "description": "token embedding manifold sampled by token id and high-norm anchors",
            "points": sample_rows(tok, max_points, by_norm=True),
        }
    )
    sorted_tok = tok[np.argsort(np.linalg.norm(tok, axis=1))]
    out.append(
        {
            "id": "R1_norm_order_embedding_trajectory",
            "family": "tropical_embedding",
            "description": "embedding manifold ordered by norm as a tropical energy path",
            "points": sample_rows(sorted_tok, max_points, by_norm=False),
        }
    )
    ids = layer_ids(state)
    if ids:
        dim = int(tok.shape[1])
        control_rows = []
        attn_rows = []
        for idx in ids:
            scale_a = tensor_np(state.get(f"blocks.{idx}.attn_scale", torch.zeros(dim)))
            scale_m = tensor_np(state.get(f"blocks.{idx}.mlp_scale", torch.zeros(dim)))
            q_gain = tensor_np(state.get(f"blocks.{idx}.attn.q_gain", torch.zeros(1)))
            resid_mix = tensor_np(state.get(f"blocks.{idx}.resid_mix", torch.zeros(2, dim)))
            control = pad_or_trim(scale_a, dim) + pad_or_trim(scale_m, dim)
            control += pad_or_trim(q_gain, dim)
            control += pad_or_trim(resid_mix.mean(axis=0), dim)
            control_rows.append(control)
            blocks = []
            for suffix in ("attn.c_q.weight", "attn.c_k.weight", "attn.c_v.weight", "attn.proj.weight"):
                key = f"blocks.{idx}.{suffix}"
                if key in state:
                    arr = tensor_np(state[key])
                    blocks.append(pad_or_trim(arr.mean(axis=0), dim))
                    blocks.append(pad_or_trim(arr.std(axis=0), dim))
            attn_rows.append(np.mean(np.stack(blocks, axis=0), axis=0) if blocks else control)
        out.append(
            {
                "id": "R2_layer_control_trajectory",
                "family": "directed_layer_flow",
                "description": "layer control tensors including q gains, residual mixing, and scale vectors",
                "points": np.stack(control_rows, axis=0),
            }
        )
        out.append(
            {
                "id": "R3_attention_transport_trajectory",
                "family": "attention_transport",
                "description": "attention projection statistics by layer",
                "points": np.stack(attn_rows, axis=0),
            }
        )
    if "prev_token_bias.weight" in state:
        bias = tensor_np(state["prev_token_bias.weight"])
        out.append(
            {
                "id": "R4_bigram_bias_memory_graph",
                "family": "memory_bigram_graph",
                "description": "graph-structured previous-token bias rows as memory/retrieval surface",
                "points": sample_rows(bias, max_points, by_norm=True),
            }
        )
    if "hash_ngram_bias.weight" in state:
        h = tensor_np(state["hash_ngram_bias.weight"])
        out.append(
            {
                "id": "R5_hash_ngram_analogy_graph",
                "family": "analogical_hash_graph",
                "description": "hash n-gram bias buckets as analogical transfer surface",
                "points": sample_rows(h, max_points, by_norm=True),
            }
        )
    return out[: max(1, int(records))]


def basis_disentanglement_metrics(points: np.ndarray) -> dict[str, float]:
    x = standardize_points(points)
    if x.shape[0] < 2:
        return {
            "graphcg_effective_rank": 0.0,
            "graphcg_basis_condition": 0.0,
            "graphcg_offdiag_coherence": 0.0,
            "graphcg_disentanglement_score": 0.0,
        }
    _, singular, vh = np.linalg.svd(x, full_matrices=False)
    energy = singular**2
    probs = energy / max(float(energy.sum()), 1e-12)
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))) / math.log(max(2, len(probs))))
    effective_rank = float(np.exp(-np.sum(probs * np.log(np.maximum(probs, 1e-12)))))
    condition = float(singular[0] / max(float(singular[-1]), 1e-8)) if singular.size else 0.0
    basis = vh[: min(16, vh.shape[0])]
    gram = basis @ basis.T
    offdiag = gram - np.eye(gram.shape[0])
    coherence = float(np.mean(np.abs(offdiag))) if offdiag.size else 0.0
    score = float(max(0.0, min(1.0, entropy * (1.0 - min(1.0, coherence)))))
    return {
        "graphcg_effective_rank": effective_rank,
        "graphcg_spectral_entropy": entropy,
        "graphcg_basis_condition": condition,
        "graphcg_offdiag_coherence": coherence,
        "graphcg_disentanglement_score": score,
    }


def analogical_transfer_metrics(points: np.ndarray, *, max_pairs: int = 24) -> dict[str, float]:
    x = standardize_points(points)
    if x.shape[0] < 4:
        return {
            "analogical_quadruples": 0.0,
            "analogical_mean_residual": 0.0,
            "analogical_transport_accuracy": 0.0,
            "analogical_map_score": 0.0,
        }
    n = min(x.shape[0], max_pairs + 3)
    x = x[:n]
    residuals = []
    hits = 0
    used = 0
    for i in range(0, n - 3):
        a, b, c, d = x[i], x[i + 1], x[i + 2], x[i + 3]
        pred = b - a + c
        residuals.append(float(np.linalg.norm(pred - d) / max(np.linalg.norm(d), 1e-6)))
        distances = np.linalg.norm(x - pred[None, :], axis=1)
        nearest = int(np.argmin(distances))
        hits += int(nearest == i + 3)
        used += 1
    mean_residual = float(np.mean(residuals)) if residuals else 0.0
    accuracy = hits / max(1, used)
    return {
        "analogical_quadruples": float(used),
        "analogical_mean_residual": mean_residual,
        "analogical_transport_accuracy": float(accuracy),
        "analogical_map_score": float(max(0.0, min(1.0, accuracy + 1.0 / (1.0 + mean_residual) * 0.5))),
    }


def compression_proxy_metrics(points: np.ndarray) -> dict[str, float]:
    x = standardize_points(points)
    quant = np.clip(np.round(x * 16.0), -127, 127).astype(np.int8)
    payload = quant.tobytes()
    diffs = np.diff(quant, axis=0).astype(np.int8).tobytes() if quant.shape[0] > 1 else payload
    k_payload = compress_len(payload, "lzma")
    k_diff = compress_len(diffs, "lzma")
    raw = max(1, len(payload))
    ncd = normalized_compression_distance(payload, diffs, "lzma") if payload and diffs else 0.0
    return {
        "complexity_payload_k_lzma": float(k_payload),
        "complexity_delta_k_lzma": float(k_diff),
        "complexity_relative_k": float(k_diff / raw),
        "complexity_relative_k_gain": float(max(0.0, (k_payload - k_diff) / max(1, k_payload))),
        "complexity_path_delta_ncd_lzma": float(ncd),
    }


def tropical_metrics(phase_u: np.ndarray, phase_v: np.ndarray, toric: dict[str, Any], energy: np.ndarray) -> dict[str, float]:
    active = np.asarray(toric.get("active_faces", []), dtype=float)
    crossings = float(np.count_nonzero(np.diff(active) != 0)) if active.size > 1 else 0.0
    winding_u = float(np.sum(np.diff(np.unwrap(phase_u * 2.0 * np.pi))) / (2.0 * np.pi)) if phase_u.size > 1 else 0.0
    winding_v = float(np.sum(np.diff(np.unwrap(phase_v * 2.0 * np.pi))) / (2.0 * np.pi)) if phase_v.size > 1 else 0.0
    if energy.size > 2:
        plateau = float(np.mean(np.abs(np.diff(energy, n=2))) / max(float(np.std(energy)), 1e-6))
    else:
        plateau = 0.0
    return {
        "tropical_active_face_crossings": crossings,
        "tropical_chamber_crossing_rate": crossings / max(1.0, active.size - 1.0),
        "tropical_winding_u": winding_u,
        "tropical_winding_v": winding_v,
        "tropical_energy_plateau_pressure": plateau,
    }


def bgg_koszul_metrics(topology: dict[str, Any], graphcg: dict[str, float]) -> dict[str, float]:
    d2 = float(topology.get("variety_complex_residual_mean", 0.0) or 0.0)
    fitting = float(topology.get("fitting_minor_rank_residual_mean", 0.0) or 0.0)
    be_rank = float(topology.get("buchsbaum_eisenbud_rank_residual_mean", 0.0) or 0.0)
    be_mult = float(topology.get("buchsbaum_eisenbud_multiplier_residual_mean", 0.0) or 0.0)
    betti = float(topology.get("multigraded_betti_mass_mean", 0.0) or 0.0)
    coherence = float(graphcg.get("graphcg_offdiag_coherence", 0.0) or 0.0)
    standard_leakage = max(0.0, min(1.0, fitting + 0.5 * coherence))
    resolution = max(0.0, min(1.0, 1.0 - 0.5 * (d2 + be_rank + be_mult)))
    return {
        "koszul_d2_residual": d2,
        "koszul_fitting_minor_rank_residual": fitting,
        "koszul_buchsbaum_eisenbud_rank_residual": be_rank,
        "koszul_buchsbaum_eisenbud_multiplier_residual": be_mult,
        "koszul_multigraded_betti_mass": betti,
        "bgg_standard_leakage": standard_leakage,
        "bgg_resolution_consistency": resolution,
        "bgg_gale_dual_consistency": max(0.0, min(1.0, 1.0 - abs(be_rank - be_mult))),
    }


def fixed_chart_directions_np(width: int, count: int) -> np.ndarray:
    count = max(4, int(count))
    coord = np.arange(width, dtype=np.float64) + 1.0
    freqs = np.arange(1, count + 1, dtype=np.float64)[:, None]
    angles = 2.0 * math.pi * freqs * coord[None, :] / float(max(width + count, 2))
    directions = np.sin(angles) + 0.5 * np.cos((freqs + 1.0) * angles / (freqs + 0.5))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / np.maximum(norms, 1e-8)


def f2_rank_np(matrix: np.ndarray) -> int:
    if matrix.size == 0:
        return 0
    mat = (np.asarray(matrix, dtype=np.uint8).copy() & 1)
    rows, cols = mat.shape
    rank = 0
    for col in range(cols):
        pivots = np.flatnonzero(mat[rank:, col])
        if pivots.size == 0:
            continue
        pivot = rank + int(pivots[0])
        if pivot != rank:
            mat[[rank, pivot]] = mat[[pivot, rank]]
        for row in range(rows):
            if row != rank and mat[row, col]:
                mat[row] ^= mat[rank]
        rank += 1
        if rank == rows:
            break
    return int(rank)


def flag_complex_f2_audit(adjacency: np.ndarray, *, max_edges: int = 256, max_triangles: int = 512) -> dict[str, float]:
    hard = np.asarray(adjacency, dtype=bool)
    hard = np.logical_or(hard, hard.T)
    np.fill_diagonal(hard, False)
    n = int(hard.shape[0])
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if hard[i, j]]
    truncated = len(edges) > max_edges
    if truncated:
        edges = edges[:max_edges]
    edge_index = {edge: idx for idx, edge in enumerate(edges)}
    triangles: list[tuple[int, int, int]] = []
    if not truncated:
        for i in range(n):
            for j in range(i + 1, n):
                if not hard[i, j]:
                    continue
                for k in range(j + 1, n):
                    if hard[i, k] and hard[j, k]:
                        triangles.append((i, j, k))
                        if len(triangles) >= max_triangles:
                            truncated = True
                            break
                if truncated:
                    break
            if truncated:
                break
    d1 = np.zeros((n, len(edges)), dtype=np.uint8)
    for col, (i, j) in enumerate(edges):
        d1[i, col] = 1
        d1[j, col] = 1
    d2 = np.zeros((len(edges), len(triangles)), dtype=np.uint8)
    for col, (i, j, k) in enumerate(triangles):
        for edge in ((i, j), (i, k), (j, k)):
            row = edge_index.get(tuple(sorted(edge)))
            if row is not None:
                d2[row, col] = 1
    rank_d1 = f2_rank_np(d1)
    rank_d2 = f2_rank_np(d2)
    product = (d1 @ d2) & 1 if d1.size and d2.size else np.zeros((d1.shape[0], d2.shape[1]), dtype=np.uint8)
    h0 = max(0, n - rank_d1)
    h1 = max(0, len(edges) - rank_d1 - rank_d2)
    return {
        "vertices": float(n),
        "edges": float(len(edges)),
        "triangles": float(len(triangles)),
        "rank_d1": float(rank_d1),
        "rank_d2": float(rank_d2),
        "h0": float(h0),
        "h1": float(h1),
        "boundary_square_residual": float(product.sum() / max(1, product.size)),
        "truncated": 1.0 if truncated else 0.0,
    }


def exact_combinatorial_cca_audit(
    points: np.ndarray,
    *,
    num_chambers: int = 8,
    exponent_dim: int = 4,
    temperature: float = 0.14,
) -> dict[str, float]:
    """Exact finite CCA sidecar audit over chamber coactivation complexes."""

    x = np.asarray(points, dtype=np.float64)
    keys = [
        "toric_cca_exact_relation_pass_rate",
        "toric_cca_exact_relation_residual_mean",
        "toric_cca_exact_relation_residual_max",
        "toric_cca_exact_sr_nonface_edge_fraction",
        "toric_cca_exact_sr_nonface_pair_fraction",
        "toric_cca_exact_h0",
        "toric_cca_exact_h1",
        "toric_cca_exact_allowed_h0",
        "toric_cca_exact_allowed_h1",
        "toric_cca_exact_betti_mismatch",
        "toric_cca_exact_edges",
        "toric_cca_exact_allowed_edges",
        "toric_cca_exact_nonface_edges",
        "toric_cca_exact_triangles",
        "toric_cca_exact_rank_d1",
        "toric_cca_exact_rank_d2",
        "toric_cca_exact_boundary_square_residual",
        "toric_cca_exact_threshold",
        "toric_cca_exact_audit_backed_score",
    ]
    if x.ndim != 2 or x.shape[0] < 4:
        return {key: 0.0 for key in keys}

    chambers = max(4, int(num_chambers))
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    directions = fixed_chart_directions_np(x.shape[1], chambers)
    logits = x @ directions.T
    logits = logits / max(float(temperature), 1e-4)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    coactivation = probs.T @ probs / float(max(1, x.shape[0]))
    sym = 0.5 * (coactivation + coactivation.T)
    np.fill_diagonal(sym, 0.0)

    labels = np.arange(chambers)
    dist = (labels[:, None] - labels[None, :]) % chambers
    allowed = np.logical_or(dist == 1, dist == chambers - 1)
    np.fill_diagonal(allowed, False)
    nonface = np.logical_not(allowed)
    np.fill_diagonal(nonface, False)
    positive = sym[sym > 0.0]
    threshold = float(max(np.mean(positive) if positive.size else 0.0, 0.25 / float(chambers * chambers)))
    hard_edges = sym > threshold
    np.fill_diagonal(hard_edges, False)
    nonface_edges = np.logical_and(hard_edges, nonface)
    allowed_edges = np.logical_and(hard_edges, allowed)

    observed = flag_complex_f2_audit(hard_edges)
    allowed_complex = flag_complex_f2_audit(allowed_edges)

    exponents = make_exponent_table(chambers, exponent_dim)
    relations = make_binomial_relations(exponents, max_relations=16)
    rounded = torch.round(exponents * 3.0).to(torch.int64)
    residuals: list[float] = []
    passes: list[float] = []
    for relation in relations.tolist():
        i, j, k, l = [int(item) for item in relation]
        rounded_residual = rounded[i] + rounded[j] - rounded[k] - rounded[l]
        passes.append(1.0 if torch.count_nonzero(rounded_residual).item() == 0 else 0.0)
        exact_residual = exponents[i] + exponents[j] - exponents[k] - exponents[l]
        residuals.append(float(torch.linalg.vector_norm(exact_residual).item()))
    relation_pass = float(np.mean(passes)) if passes else 0.0
    relation_mean = float(np.mean(residuals)) if residuals else 0.0
    relation_max = float(np.max(residuals)) if residuals else 0.0

    total_edges = float(np.count_nonzero(hard_edges))
    nonface_count = float(np.count_nonzero(nonface_edges))
    sr_edge_fraction = nonface_count / max(1.0, total_edges)
    sr_pair_fraction = nonface_count / max(1.0, float(np.count_nonzero(nonface)))
    betti_mismatch = abs(observed["h0"] - allowed_complex["h0"]) + abs(observed["h1"] - allowed_complex["h1"])
    audit_score = relation_pass
    audit_score *= max(0.0, 1.0 - sr_edge_fraction)
    audit_score *= 1.0 / (1.0 + float(betti_mismatch))
    audit_score *= max(0.0, 1.0 - observed["boundary_square_residual"])

    return {
        "toric_cca_exact_relation_pass_rate": relation_pass,
        "toric_cca_exact_relation_residual_mean": relation_mean,
        "toric_cca_exact_relation_residual_max": relation_max,
        "toric_cca_exact_sr_nonface_edge_fraction": float(sr_edge_fraction),
        "toric_cca_exact_sr_nonface_pair_fraction": float(sr_pair_fraction),
        "toric_cca_exact_h0": float(observed["h0"]),
        "toric_cca_exact_h1": float(observed["h1"]),
        "toric_cca_exact_allowed_h0": float(allowed_complex["h0"]),
        "toric_cca_exact_allowed_h1": float(allowed_complex["h1"]),
        "toric_cca_exact_betti_mismatch": float(betti_mismatch),
        "toric_cca_exact_edges": float(total_edges),
        "toric_cca_exact_allowed_edges": float(np.count_nonzero(allowed_edges)),
        "toric_cca_exact_nonface_edges": float(nonface_count),
        "toric_cca_exact_triangles": float(observed["triangles"]),
        "toric_cca_exact_rank_d1": float(observed["rank_d1"]),
        "toric_cca_exact_rank_d2": float(observed["rank_d2"]),
        "toric_cca_exact_boundary_square_residual": float(observed["boundary_square_residual"]),
        "toric_cca_exact_threshold": float(threshold),
        "toric_cca_exact_audit_backed_score": float(audit_score),
    }


def combinatorial_cca_metrics(points: np.ndarray, topology_config: ReasoningTopologyConfig) -> dict[str, float]:
    """Run the training-time CCA bridge on checkpoint proxy trajectories."""

    x_np = np.asarray(points, dtype=np.float32)
    if x_np.ndim != 2 or x_np.shape[0] < 4:
        return {
            "toric_cca_topology_loss": 0.0,
            "toric_cca_binomial_residual": 0.0,
            "toric_cca_stanley_reisner_nonface_mass": 0.0,
            "toric_cca_symbolic_resolution_loss": 0.0,
            "toric_cca_symbolic_sr_monomial_generator_mass": 0.0,
            "toric_cca_symbolic_taylor_lcm_syzygy_mass": 0.0,
            "toric_cca_symbolic_taylor_full_resolution_mass": 0.0,
            "toric_cca_symbolic_hilbert_betti_pressure": 0.0,
            "toric_cca_symbolic_resolution_minimal_total_betti": 0.0,
            "toric_cca_symbolic_resolution_projective_dimension": 0.0,
            "toric_cca_symbolic_resolution_regularity": 0.0,
            "toric_cca_symbolic_resolution_nonminimality_log2": 0.0,
            "toric_cca_chamber_coverage": 0.0,
            "toric_cca_betti1_proxy": 0.0,
            **exact_combinatorial_cca_audit(x_np),
        }
    hidden = torch.from_numpy(x_np).unsqueeze(0)
    positions = torch.arange(hidden.shape[1], dtype=torch.long).view(1, -1)
    cfg = CombinatorialToricConfig(
        max_points=min(16, max(4, int(topology_config.max_points))),
        max_windows=min(2, max(1, int(topology_config.max_windows))),
        window_size=min(32, max(4, int(topology_config.window_size))),
        step_stride=max(1, int(topology_config.step_stride)),
        num_chambers=8,
        max_relations=16,
    )
    with torch.no_grad():
        out = combinatorial_toric_cca_topology_loss(hidden, positions, config=cfg)
    metrics: dict[str, float] = {}
    for key, value in out.items():
        if torch.is_tensor(value) and value.ndim == 0:
            safe = torch.nan_to_num(value.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
            metrics[key] = float(safe.item())
    metrics.update(exact_combinatorial_cca_audit(x_np, num_chambers=cfg.num_chambers, temperature=cfg.temperature))
    return metrics

def analyze_record(
    item: dict[str, Any],
    *,
    log_context: dict[str, Any],
    topology_config: ReasoningTopologyConfig,
) -> dict[str, Any]:
    raw_points = np.asarray(item["points"], dtype=np.float32)
    points = standardize_points(raw_points)
    proj3, explained = pca_project(points, 3)
    phase_u, phase_v = phase_from_projection(proj3)
    energy = np.linalg.norm(points, axis=1)
    if points.shape[0] > 1:
        local_motion = np.r_[0.0, np.linalg.norm(np.diff(points, axis=0), axis=1)]
        energy = 0.65 * normalize01(energy) + 0.35 * normalize01(local_motion)
    else:
        energy = normalize01(energy)
    topology = directed_step_filtration_stats_np(points, config=topology_config)
    toric = empirical_toric_shadow_stats_np(points, max_points=topology_config.max_points)
    slepian = toric_slepian_audit(
        phase_u,
        phase_v,
        energy=energy,
        config=ToricSlepianConfig(modes=6, half_bandwidth=0.075),
    )
    graphcg = basis_disentanglement_metrics(points)
    analogy = analogical_transfer_metrics(points)
    complexity = compression_proxy_metrics(points)
    tropical = tropical_metrics(phase_u, phase_v, toric, energy)
    algebra = bgg_koszul_metrics(topology, graphcg)
    cca = combinatorial_cca_metrics(points, topology_config)
    mst = prim_mst_stats(proj3.tolist())
    curvature = float(np.mean(np.linalg.norm(np.diff(proj3, n=2, axis=0), axis=1))) if proj3.shape[0] > 2 else 0.0
    path_length = float(np.sum(np.linalg.norm(np.diff(proj3, axis=0), axis=1))) if proj3.shape[0] > 1 else 0.0
    smoothness = float(1.0 / (1.0 + curvature))
    terminal_confidence = float(1.0 / (1.0 + np.mean(energy[-max(1, min(6, energy.size)) :])))
    record = {
        "record_id": str(item["id"]),
        "family": str(item["family"]),
        "source": "seq4096_checkpoint_embedding_proxy",
        "description": str(item.get("description", "")),
        "points": int(points.shape[0]),
        "dim": int(points.shape[1]),
        "pca_explained_0": float(explained[0]) if explained.size else 0.0,
        "pca_explained_1": float(explained[1]) if explained.size > 1 else 0.0,
        "pca_explained_2": float(explained[2]) if explained.size > 2 else 0.0,
        "trajectory_path_length": path_length,
        "trajectory_curvature": curvature,
        "path_smoothness": smoothness,
        "terminal_confidence": terminal_confidence,
        "energy_mean": float(np.mean(energy)) if energy.size else 0.0,
        "energy_std": float(np.std(energy)) if energy.size else 0.0,
        "bpb_quality": float(log_context.get("bpb_quality", 0.5)),
        "loss_quality": float(log_context.get("loss_quality", 0.5)),
        "latest_train_bpb": float(log_context.get("latest_train_bpb", float("nan"))),
        "latest_val_bpb": float(log_context.get("latest_val_bpb", float("nan"))),
        "best_val_bpb": float(log_context.get("best_val_bpb", float("nan"))),
        "topology_edge_density_mean": float(np.mean(topology.get("edge_density", [0.0]))),
        "topology_triangle_density_mean": float(np.mean(topology.get("triangle_density", [0.0]))),
        "topology_directed_asymmetry_mean": float(np.mean(topology.get("directed_asymmetry", [0.0]))),
        "topology_cycle_rank_mean": float(np.mean(topology.get("cycle_rank", [0.0]))),
        "topology_hdbscan_stability_mean": float(np.mean(topology.get("hdbscan_stability", [0.0]))),
        "topology_directed_chain_commutator_mean": float(np.mean(topology.get("directed_chain_commutator", [0.0]))),
        "toric_occupied_fan_cells": float(toric.get("occupied_fan_cells", 0.0) or 0.0),
        "toric_fan_cell_entropy": float(toric.get("fan_cell_entropy", 0.0) or 0.0),
        "toric_mean_margin": float(toric.get("mean_margin", 0.0) or 0.0),
        "toric_mean_bend": float(toric.get("mean_bend", 0.0) or 0.0),
        "slepian_concentration": float(slepian.get("slepian_concentration", 0.0) or 0.0),
        "slepian_leakage": float(slepian.get("slepian_leakage", 1.0) or 1.0),
        "slepian_mode_entropy": float(slepian.get("slepian_mode_entropy", 0.0) or 0.0),
        "mst_efficiency": float(mst.get("mst_efficiency", 0.0) or 0.0),
        "mst_mean_edge_weight": float(mst.get("mst_mean_edge_weight", 0.0) or 0.0),
        "order_robustness": float(1.0 / (1.0 + topology.get("exact_morphism_radius_shift_mean", 0.0))),
        **graphcg,
        **analogy,
        **complexity,
        **tropical,
        **algebra,
        **cca,
    }
    record["_arrays"] = {
        "points": points,
        "proj3": proj3,
        "phase_u": phase_u,
        "phase_v": phase_v,
        "energy": energy,
        "topology": topology,
        "toric": toric,
        "slepian": slepian,
        "cca": cca,
    }
    return record


def save_html(path: Path, title: str, body: str, image: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_html = f'<p><img src="{image}" style="max-width:100%;height:auto"></p>' if image else ""
    path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{title}</title></head><body><h1>{title}</h1>{image_html}<pre>"
        + body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</pre></body></html>\n",
        encoding="utf-8",
    )


def plot_trajectory_3d(record: dict[str, Any], out: Path) -> None:
    arrays = record["_arrays"]
    proj = arrays["proj3"]
    energy = arrays["energy"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7), constrained_layout=True, facecolor=DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    style_dark_axes(fig, ax)
    if proj.shape[0] > 1:
        ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], color="#6df6ff", linewidth=1.0, alpha=0.8)
    scatter = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=energy, cmap="viridis", s=38)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.06, label="checkpoint energy")
    style_dark_colorbar(cbar)
    ax.set_title(f"{record['record_id']} 3D checkpoint trajectory")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    save_dark(fig, out)
    plt.close(fig)


def plot_energy_landscape(record: dict[str, Any], out: Path) -> None:
    arrays = record["_arrays"]
    proj = arrays["proj3"]
    energy = arrays["energy"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.2, 6.8), constrained_layout=True, facecolor=DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    style_dark_axes(fig, ax)
    if proj.shape[0] > 2:
        try:
            ax.plot_trisurf(
                proj[:, 0],
                proj[:, 1],
                energy,
                cmap="magma",
                linewidth=0.05,
                antialiased=True,
                alpha=0.74,
            )
        except Exception:
            pass
    if proj.shape[0] > 1:
        ax.plot(proj[:, 0], proj[:, 1], energy, color="#6df6ff", linewidth=1.4, alpha=0.85)
    scatter = ax.scatter(proj[:, 0], proj[:, 1], energy, c=energy, cmap="viridis", s=42, edgecolor="#030712", linewidth=0.35)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.08, label="energy")
    style_dark_colorbar(cbar)
    ax.set_title(f"{record['record_id']} 3D energy landscape")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("checkpoint energy")
    ax.view_init(elev=28, azim=-48)
    save_dark(fig, out)
    plt.close(fig)


def plot_phase_energy(record: dict[str, Any], out: Path) -> None:
    arrays = record["_arrays"]
    u = arrays["phase_u"]
    v = arrays["phase_v"]
    energy = arrays["energy"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, ax)
    if u.size > 1:
        ax.plot(u, v, color="#6df6ff", linewidth=0.8, alpha=0.7)
    scatter = ax.scatter(u, v, c=energy, cmap="plasma", s=36, edgecolor="#030712", linewidth=0.2)
    cbar = fig.colorbar(scatter, ax=ax, label="energy")
    style_dark_colorbar(cbar)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("toric phase u")
    ax.set_ylabel("toric phase v")
    ax.set_title(f"{record['record_id']} toric phase energy")
    save_dark(fig, out)
    plt.close(fig)


def plot_winding(record: dict[str, Any], out: Path) -> None:
    arrays = record["_arrays"]
    u = np.unwrap(arrays["phase_u"] * 2.0 * np.pi) / (2.0 * np.pi)
    v = np.unwrap(arrays["phase_v"] * 2.0 * np.pi) / (2.0 * np.pi)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, ax)
    ax.plot(u, color="#39b8ff", label="u winding")
    ax.plot(v, color="#ff4fd8", label="v winding")
    ax.set_xlabel("trajectory index")
    ax.set_ylabel("unwrapped phase")
    ax.set_title(f"{record['record_id']} toric phase winding collection")
    legend = ax.legend(loc="best")
    for text in legend.get_texts():
        text.set_color(DARK_TEXT)
    legend.get_frame().set_facecolor(DARK_PANEL)
    legend.get_frame().set_edgecolor("#2de2e6")
    save_dark(fig, out)
    plt.close(fig)


def plot_directed_filtration(record: dict[str, Any], out: Path) -> None:
    topology = record["_arrays"]["topology"]
    radii = np.asarray(topology.get("radii", []), dtype=float)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, ax)
    for key, color in (
        ("edge_density", "#39b8ff"),
        ("triangle_density", "#8cff6a"),
        ("directed_edge_density", "#ff4fd8"),
        ("directed_asymmetry", "#ad7cff"),
    ):
        values = np.asarray(topology.get(key, []), dtype=float)
        if radii.size and values.size:
            ax.plot(radii[: values.size], values, marker="o", linewidth=1.4, color=color, label=key)
    ax.set_xlabel("filtration radius")
    ax.set_title(f"{record['record_id']} directed filtration")
    legend = ax.legend(loc="best", fontsize=8)
    for text in legend.get_texts():
        text.set_color(DARK_TEXT)
    legend.get_frame().set_facecolor(DARK_PANEL)
    legend.get_frame().set_edgecolor("#2de2e6")
    save_dark(fig, out)
    plt.close(fig)


def plot_step_radius_hierarchy(record: dict[str, Any], out: Path) -> None:
    topology = record["_arrays"]["topology"]
    keys = ["edge_density", "cycle_rank", "betti0", "hdbscan_stability", "directed_chain_commutator"]
    matrix = []
    for key in keys:
        values = np.asarray(topology.get(key, []), dtype=float)
        matrix.append(normalize01(values) if values.size else np.zeros((1,), dtype=float))
    max_len = max(len(row) for row in matrix)
    padded = np.vstack([np.pad(row, (0, max_len - len(row)), constant_values=np.nan) for row in matrix])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, ax, grid=False)
    im = ax.imshow(padded, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(keys)), keys)
    ax.set_xlabel("radius level")
    ax.set_title(f"{record['record_id']} step/radius hierarchy")
    cbar = fig.colorbar(im, ax=ax, label="normalized value")
    style_dark_colorbar(cbar)
    save_dark(fig, out)
    plt.close(fig)


def plot_noncommutative_heatmaps(record: dict[str, Any], out: Path) -> None:
    topology = record["_arrays"]["topology"]
    skew = np.asarray(topology.get("skew", [[0.0]]), dtype=float)
    directed_levels = topology.get("directed_adjacency", [])
    directed = np.asarray(directed_levels[-1] if directed_levels else [[0.0]], dtype=float)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, axes, grid=False)
    for ax, matrix, title in zip(axes, (skew, directed), ("antisymmetric skew", "directed adjacency")):
        im = ax.imshow(matrix, cmap="coolwarm" if title.startswith("antisymmetric") else "magma", aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        style_dark_colorbar(cbar)
    fig.suptitle(f"{record['record_id']} noncommutative heatmaps")
    fig._suptitle.set_color("white")
    save_dark(fig, out)
    plt.close(fig)


def plot_toric_shadow(record: dict[str, Any], out: Path) -> None:
    toric = record["_arrays"]["toric"]
    active = np.asarray(toric.get("active_faces", []), dtype=int)
    margins = np.asarray(toric.get("margins", []), dtype=float)
    bends = np.asarray(toric.get("bend_magnitudes", []), dtype=float)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, axes)
    if active.size:
        axes[0].plot(active, color="#39b8ff", linewidth=1.2)
    axes[0].set_title("active toric fan cell")
    if margins.size:
        axes[1].plot(margins, color="#8cff6a", linewidth=1.2)
    axes[1].set_title("active-face margin")
    if bends.size:
        axes[2].plot(bends, color="#ff4fd8", linewidth=1.2)
    axes[2].set_title("shadow bend magnitude")
    fig.suptitle(f"{record['record_id']} toric shadow audit")
    fig._suptitle.set_color("white")
    save_dark(fig, out)
    plt.close(fig)


def plot_slepian(record: dict[str, Any], out: Path) -> None:
    slepian = record["_arrays"]["slepian"]
    coeff = np.asarray(slepian.get("slepian_coefficients", []), dtype=float)
    eig = np.asarray(slepian.get("slepian_eigenvalues", []), dtype=float)
    recon = np.asarray(slepian.get("slepian_reconstruction", []), dtype=float)
    envelope = np.asarray(slepian.get("slepian_envelope", []), dtype=float)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, axes)
    if eig.size:
        x = np.arange(eig.size)
        axes[0].bar(x - 0.18, eig, width=0.36, color="#39b8ff", label="eigenvalue")
        axes[0].bar(x + 0.18, np.abs(coeff[: eig.size]), width=0.36, color="#ff4fd8", label="|coefficient|")
        legend0 = axes[0].legend(loc="best")
        for text in legend0.get_texts():
            text.set_color(DARK_TEXT)
        legend0.get_frame().set_facecolor(DARK_PANEL)
        legend0.get_frame().set_edgecolor("#2de2e6")
    axes[0].set_title("Slepian/Pollak mode concentration")
    if recon.size:
        axes[1].plot(recon, color="#39b8ff", label="reconstruction")
    if envelope.size:
        axes[1].plot(envelope, color="#8cff6a", label="envelope")
    legend1 = axes[1].legend(loc="best")
    for text in legend1.get_texts():
        text.set_color(DARK_TEXT)
    legend1.get_frame().set_facecolor(DARK_PANEL)
    legend1.get_frame().set_edgecolor("#2de2e6")
    axes[1].set_title("prolate envelope")
    fig.suptitle(f"{record['record_id']} toric Slepian audit")
    fig._suptitle.set_color("white")
    save_dark(fig, out)
    plt.close(fig)


def plot_commutative_algebra(record: dict[str, Any], out: Path) -> None:
    keys = [
        "koszul_d2_residual",
        "koszul_fitting_minor_rank_residual",
        "koszul_buchsbaum_eisenbud_rank_residual",
        "koszul_buchsbaum_eisenbud_multiplier_residual",
        "bgg_standard_leakage",
        "bgg_resolution_consistency",
        "bgg_gale_dual_consistency",
    ]
    values = [float(record.get(key, 0.0) or 0.0) for key in keys]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, ax)
    ax.barh(np.arange(len(keys)), values, color=["#ff4fd8" if "residual" in key or "leakage" in key else "#8cff6a" for key in keys])
    ax.set_yticks(np.arange(len(keys)), keys, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("audit value")
    ax.set_title(f"{record['record_id']} commutative algebra / Koszul / BGG audit")
    save_dark(fig, out)
    plt.close(fig)


def plot_combinatorial_cca(record: dict[str, Any], out: Path) -> None:
    residual_keys = [
        "toric_cca_topology_loss",
        "toric_cca_binomial_residual",
        "toric_cca_stanley_reisner_nonface_mass",
        "toric_cca_fan_balance_loss",
        "toric_cca_koszul_loss",
        "toric_cca_koszul_fitting_rank_residual",
        "toric_cca_koszul_buchsbaum_eisenbud_rank_residual",
        "toric_cca_koszul_buchsbaum_eisenbud_multiplier_residual",
        "toric_cca_symbolic_resolution_loss",
        "toric_cca_symbolic_sr_monomial_generator_mass",
        "toric_cca_symbolic_taylor_lcm_syzygy_mass",
        "toric_cca_symbolic_taylor_full_resolution_mass",
        "toric_cca_symbolic_hilbert_betti_pressure",
        "toric_cca_topology_loss_component",
    ]
    shape_keys = [
        "toric_cca_chart_entropy",
        "toric_cca_chamber_coverage",
        "toric_cca_euler_characteristic_proxy",
        "toric_cca_betti0_proxy",
        "toric_cca_betti1_proxy",
        "toric_cca_allowed_edge_mass",
        "toric_cca_koszul_multigraded_betti_mass",
        "toric_cca_symbolic_resolution_minimal_total_betti",
        "toric_cca_symbolic_resolution_projective_dimension",
        "toric_cca_symbolic_resolution_regularity",
        "toric_cca_symbolic_resolution_nonminimality_log2",
    ]
    residuals = np.nan_to_num(np.asarray([float(record.get(key, 0.0) or 0.0) for key in residual_keys], dtype=float))
    shape_values = np.nan_to_num(np.asarray([float(record.get(key, 0.0) or 0.0) for key in shape_keys], dtype=float))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, axes)
    axes[0].barh(np.arange(len(residual_keys)), np.log1p(np.maximum(residuals, 0.0)), color="#ff4fd8")
    axes[0].set_yticks(np.arange(len(residual_keys)), residual_keys, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("log1p residual")
    axes[0].set_title("toric ideal / Stanley-Reisner / Koszul residuals")
    axes[1].barh(np.arange(len(shape_keys)), shape_values, color="#39b8ff")
    axes[1].set_yticks(np.arange(len(shape_keys)), shape_keys, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("proxy value")
    axes[1].set_title("fan coverage and Betti/Euler topology proxies")
    fig.suptitle(f"{record['record_id']} combinatorial CCA topology audit")
    fig._suptitle.set_color("white")
    save_dark(fig, out)
    plt.close(fig)


def plot_combinatorial_cca_exact(record: dict[str, Any], out: Path) -> None:
    quality_keys = [
        "toric_cca_exact_relation_pass_rate",
        "toric_cca_exact_audit_backed_score",
        "toric_cca_exact_sr_nonface_edge_fraction",
        "toric_cca_exact_sr_nonface_pair_fraction",
        "toric_cca_exact_boundary_square_residual",
    ]
    topology_keys = [
        "toric_cca_exact_h0",
        "toric_cca_exact_h1",
        "toric_cca_exact_allowed_h0",
        "toric_cca_exact_allowed_h1",
        "toric_cca_exact_betti_mismatch",
    ]
    size_keys = [
        "toric_cca_exact_edges",
        "toric_cca_exact_allowed_edges",
        "toric_cca_exact_nonface_edges",
        "toric_cca_exact_triangles",
        "toric_cca_exact_rank_d1",
        "toric_cca_exact_rank_d2",
    ]
    quality = np.nan_to_num(np.asarray([float(record.get(key, 0.0) or 0.0) for key in quality_keys], dtype=float))
    topology = np.nan_to_num(np.asarray([float(record.get(key, 0.0) or 0.0) for key in topology_keys], dtype=float))
    sizes = np.nan_to_num(np.asarray([float(record.get(key, 0.0) or 0.0) for key in size_keys], dtype=float))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, axes)
    axes[0].barh(np.arange(len(quality_keys)), quality, color=["#8cff6a", "#39b8ff", "#ff4fd8", "#ff4fd8", "#ad7cff"])
    axes[0].set_yticks(np.arange(len(quality_keys)), quality_keys, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("fraction / score")
    axes[0].set_title("exact audit quality")
    axes[1].barh(np.arange(len(topology_keys)), topology, color="#39b8ff")
    axes[1].set_yticks(np.arange(len(topology_keys)), topology_keys, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("F2 Betti / mismatch")
    axes[1].set_title("F2 homology on chamber complex")
    axes[2].barh(np.arange(len(size_keys)), sizes, color="#2de2e6")
    axes[2].set_yticks(np.arange(len(size_keys)), size_keys, fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("count / rank")
    axes[2].set_title("finite complex size")
    fig.suptitle(f"{record['record_id']} exact combinatorial CCA F2 audit")
    fig._suptitle.set_color("white")
    save_dark(fig, out)
    plt.close(fig)


def plot_persistence_morphisms(record: dict[str, Any], out: Path) -> None:
    keys = [
        "exact_h0_dim_mean",
        "exact_h1_dim_mean",
        "exact_h0_map_rank_mean",
        "exact_h1_map_rank_mean",
        "exact_morphism_edge_validity_mean",
        "exact_morphism_triangle_validity_mean",
        "exact_directed_edge_validity_mean",
    ]
    topology = record["_arrays"]["topology"]
    values = [float(topology.get(key, 0.0) or 0.0) for key in keys]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, ax)
    ax.bar(np.arange(len(keys)), values, color="#ad7cff")
    ax.set_xticks(np.arange(len(keys)), keys, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("rank / validity")
    ax.set_title(f"{record['record_id']} exact persistence morphisms")
    save_dark(fig, out)
    plt.close(fig)


def plot_graphcg(record: dict[str, Any], out: Path) -> None:
    keys = [
        "graphcg_effective_rank",
        "graphcg_spectral_entropy",
        "graphcg_basis_condition",
        "graphcg_offdiag_coherence",
        "graphcg_disentanglement_score",
    ]
    values = [float(record.get(key, 0.0) or 0.0) for key in keys]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, axes, grid=True)
    axes[0].barh(np.arange(len(keys)), values, color="#39b8ff")
    axes[0].set_yticks(np.arange(len(keys)), keys, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title("GraphCG basis metrics")
    axes[0].grid(axis="x", alpha=0.25)
    arrays = record["_arrays"]
    points = standardize_points(arrays["points"])
    if points.shape[0] > 1:
        _, _, vh = np.linalg.svd(points, full_matrices=False)
        basis = vh[: min(16, vh.shape[0])]
        gram = basis @ basis.T
        im = axes[1].imshow(gram, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        style_dark_colorbar(cbar)
    axes[1].set_title("basis Gram matrix")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.suptitle(f"{record['record_id']} GraphCG disentangling audit")
    fig._suptitle.set_color("white")
    save_dark(fig, out)
    plt.close(fig)


def plot_analogy(record: dict[str, Any], out: Path) -> None:
    proj = record["_arrays"]["proj3"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, ax)
    if proj.shape[0] >= 4:
        for i in range(min(12, proj.shape[0] - 3)):
            a, b, c, d = proj[i], proj[i + 1], proj[i + 2], proj[i + 3]
            pred = b - a + c
            ax.arrow(a[0], a[1], (b - a)[0], (b - a)[1], color="#39b8ff", alpha=0.35, head_width=0.02)
            ax.arrow(c[0], c[1], (d - c)[0], (d - c)[1], color="#8cff6a", alpha=0.35, head_width=0.02)
            ax.plot([pred[0], d[0]], [pred[1], d[1]], color="#ff4fd8", alpha=0.4, linewidth=0.8)
    ax.scatter(proj[:, 0], proj[:, 1], c=np.arange(proj.shape[0]), cmap="viridis", s=30)
    ax.set_title(f"{record['record_id']} analogical transport map")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    save_dark(fig, out)
    plt.close(fig)


def plot_tropical(record: dict[str, Any], out: Path) -> None:
    toric = record["_arrays"]["toric"]
    active = np.asarray(toric.get("active_faces", []), dtype=float)
    energy = record["_arrays"]["energy"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, axes)
    if active.size:
        axes[0].step(np.arange(active.size), active, where="mid", color="#39b8ff")
    axes[0].set_title("tropical active-face / chamber path")
    if energy.size:
        axes[1].plot(energy, color="#ff4fd8")
    axes[1].set_title("tropical energy plateau pressure")
    fig.suptitle(f"{record['record_id']} tropical chamber audit")
    fig._suptitle.set_color("white")
    save_dark(fig, out)
    plt.close(fig)


def plot_record_artifacts(record: dict[str, Any], geometry_dir: Path) -> list[str]:
    base = slug(record["record_id"])
    files: list[Path] = []
    traj_dir = geometry_dir / "trajectories"
    topo_dir = geometry_dir / "topology"
    graphcg_dir = geometry_dir / "graphcg"
    analogy_dir = geometry_dir / "analogical"
    tropical_dir = geometry_dir / "tropical"
    path = traj_dir / f"{base}_trajectory_3d.png"
    plot_trajectory_3d(record, path)
    files.append(path)
    save_html(path.with_suffix(".html"), f"{base} trajectory 3D", json.dumps(record_public(record), indent=2), path.name)
    files.append(path.with_suffix(".html"))
    path = traj_dir / f"{base}_energy_landscape.png"
    plot_energy_landscape(record, path)
    files.append(path)
    save_html(path.with_suffix(".html"), f"{base} energy landscape", json.dumps(record_public(record), indent=2), path.name)
    files.append(path.with_suffix(".html"))
    for suffix, fn in (
        ("phase_energy", plot_phase_energy),
        ("toric_phase_simplicial_trajectory", plot_phase_energy),
        ("toric_phase_winding_collection", plot_winding),
    ):
        path = traj_dir / f"{base}_{suffix}.png"
        fn(record, path)
        files.append(path)
        if suffix == "toric_phase_simplicial_trajectory":
            save_html(path.with_suffix(".html"), f"{base} toric phase simplicial trajectory", json.dumps(record_public(record), indent=2), path.name)
            files.append(path.with_suffix(".html"))
    save_html(
        traj_dir / f"{base}_projected_simplicial_toric_geometry.html",
        f"{base} projected simplicial toric geometry",
        json.dumps(record_public(record), indent=2),
        f"{base}_toric_phase_simplicial_trajectory.png",
    )
    files.append(traj_dir / f"{base}_projected_simplicial_toric_geometry.html")
    for suffix, fn in (
        ("directed_filtration", plot_directed_filtration),
        ("step_radius_hierarchy", plot_step_radius_hierarchy),
        ("noncommutative_heatmaps", plot_noncommutative_heatmaps),
        ("toric_shadow_audit", plot_toric_shadow),
        ("toric_slepian_audit", plot_slepian),
        ("commutative_algebra_audit", plot_commutative_algebra),
        ("exact_persistence_morphisms", plot_persistence_morphisms),
        ("combinatorial_cca_audit", plot_combinatorial_cca),
        ("combinatorial_cca_exact_audit", plot_combinatorial_cca_exact),
    ):
        path = topo_dir / f"{base}_{suffix}.png"
        fn(record, path)
        files.append(path)
    path = graphcg_dir / f"{base}_graphcg_basis_disentanglement.png"
    plot_graphcg(record, path)
    files.append(path)
    path = analogy_dir / f"{base}_analogical_transport_map.png"
    plot_analogy(record, path)
    files.append(path)
    path = tropical_dir / f"{base}_tropical_chamber_audit.png"
    plot_tropical(record, path)
    files.append(path)
    return [str(path) for path in files]


def record_public(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def annotate_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = {
        "score/trajectory_depth": ("trajectory_path_length", True),
        "score/relative_k": ("complexity_relative_k", False),
        "score/bpb_quality": ("bpb_quality", True),
        "score/analogy_quality": ("analogical_map_score", True),
        "score/terminal_confidence": ("terminal_confidence", True),
        "score/path_smoothness": ("path_smoothness", True),
        "score/mst_efficiency": ("mst_efficiency", True),
        "score/graphcg_basis": ("graphcg_disentanglement_score", True),
        "score/relative_k_gain": ("complexity_relative_k_gain", True),
        "score/order_robustness": ("order_robustness", True),
        "score/loss_quality": ("loss_quality", True),
        "score/toric_entropy": ("toric_fan_cell_entropy", True),
        "score/slepian_concentration": ("slepian_concentration", True),
        "score/energy_quality": ("energy_mean", False),
        "score/toric_ideal_consistency": ("toric_cca_binomial_residual", False),
        "score/stanley_reisner_consistency": ("toric_cca_stanley_reisner_nonface_mass", False),
        "score/cca_topology_consistency": ("toric_cca_topology_loss", False),
        "score/betti1_consistency": ("toric_cca_betti1_proxy", False),
        "score/cca_chart_coverage": ("toric_cca_chamber_coverage", True),
        "score/cca_exact_audit": ("toric_cca_exact_audit_backed_score", True),
    }
    return attach_normalized_scores(records, specs)


def plot_triangle(records: list[dict[str, Any]], spec: dict[str, Any], out: Path) -> None:
    vertices = [TRIANGLE_VERTICES["left"], TRIANGLE_VERTICES["right"], TRIANGLE_VERTICES["top"]]
    labels = spec["labels"]
    points = []
    intensities = []
    for record in records:
        simplex = simplex_record(record, spec["scores"], labels)
        xy = barycentric_to_cartesian(simplex["weights"], vertices)
        points.append(xy)
        intensity_key = spec.get("intensity", spec["scores"][0])
        intensities.append(float(record.get(intensity_key, 0.5)))
    grid = triangle_grid(resolution=110)
    xs = [item[0] for item in grid]
    ys = [item[1] for item in grid]
    zs = [rbf_interpolate((x, y), points, intensities, bandwidth=0.16) for x, y, _ in grid]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True, facecolor=DARK_BG)
    style_dark_axes(fig, ax, grid=False)
    if points:
        triangulation = mtri.Triangulation(xs, ys)
        contour = ax.tricontourf(triangulation, zs, levels=22, cmap=blue_colormap(), alpha=0.96)
        ax.tricontour(triangulation, zs, levels=8, colors="#a8f6ff", linewidths=0.32, alpha=0.48)
        cbar = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("reasoning simplex intensity", color=DARK_TEXT)
        style_dark_colorbar(cbar)
    tri = np.asarray([vertices[0], vertices[1], vertices[2], vertices[0]], dtype=float)
    ax.plot(tri[:, 0], tri[:, 1], color="#6df6ff", linewidth=1.8)
    label_offsets = [(-0.08, -0.055), (0.08, -0.055), (0.0, -0.036)]
    for label, vertex, offset in zip(labels, vertices, label_offsets):
        ax.text(vertex[0] + offset[0], vertex[1] + offset[1], label, color=DARK_TEXT, fontsize=9, ha="center")
    if points:
        arr = np.asarray(points, dtype=float)
        if arr.shape[0] > 1:
            ax.plot(arr[:, 0], arr[:, 1], color="#e8fbff", linewidth=0.8, alpha=0.48)
        scatter = ax.scatter(
            arr[:, 0],
            arr[:, 1],
            c=[float(record.get("score/bpb_quality", 0.5)) for record in records],
            cmap="magma_r",
            s=64,
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )
        scatter_bar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.09)
        scatter_bar.set_label("BPB quality", color=DARK_TEXT)
        style_dark_colorbar(scatter_bar)
        for record, (x, y) in zip(records, arr):
            ax.text(x, y + 0.016, record["record_id"].split("_", 1)[0], color="white", fontsize=7, ha="center", va="center")
    ax.set_title(spec["title"], color="white")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.02)
    save_dark(fig, out)
    plt.close(fig)


def plot_tetrahedron(records: list[dict[str, Any]], spec: dict[str, Any], out: Path) -> None:
    vertices = [TETRAHEDRON_VERTICES[key] for key in ("a", "b", "c", "d")]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7), constrained_layout=True, facecolor=DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    style_dark_axes(fig, ax, grid=False)
    verts = np.asarray(vertices, dtype=float)
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        ax.plot([verts[i, 0], verts[j, 0]], [verts[i, 1], verts[j, 1]], [verts[i, 2], verts[j, 2]], color="#6df6ff", linewidth=1.2)
    labels = spec["labels"]
    for vertex, label in zip(verts, labels):
        ax.text(vertex[0], vertex[1], vertex[2], label, color=DARK_TEXT, fontsize=8)
    points = []
    colors = []
    for record in records:
        simplex = simplex_record(record, spec["scores"], labels)
        xyz = barycentric_to_cartesian(simplex["weights"], vertices)
        points.append(xyz)
        colors.append(float(record.get("score/bpb_quality", 0.5)))
    if points:
        arr = np.asarray(points, dtype=float)
        scatter = ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c=colors, cmap="magma_r", s=56, edgecolor="white", linewidth=0.45)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.06, label="BPB quality")
        style_dark_colorbar(cbar)
    ax.set_title(spec["title"], color="white")
    ax.set_axis_off()
    save_dark(fig, out)
    plt.close(fig)
    save_html(out.with_suffix(".html"), spec["title"], json.dumps([record_public(record) for record in records], indent=2), out.name)


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    public = [record_public(record) for record in records]
    keys = sorted({key for record in public for key in record})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in public:
            writer.writerow(record)


def aggregate_summary(records: list[dict[str, Any]], *, checkpoint: Path, step: int, log_context: dict[str, Any], files: list[str]) -> dict[str, Any]:
    def mean(key: str) -> float:
        values = []
        for record in records:
            try:
                value = float(record.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else float("nan")

    image_files = [path for path in files if path.endswith((".png", ".jpg", ".jpeg"))]
    html_files = [path for path in files if path.endswith(".html")]
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(step),
        "analysis_source": "seq4096_checkpoint_embedding_proxy",
        "current_run_only": 1.0,
        "records": int(len(records)),
        "image_files": image_files,
        "html_files": html_files,
        "output_files": files,
        "latest_train_bpb": log_context.get("latest_train_bpb"),
        "latest_val_bpb": log_context.get("latest_val_bpb"),
        "best_val_bpb": log_context.get("best_val_bpb"),
        "target_bpb": log_context.get("target_bpb"),
        "families_available": {
            "topology": 1.0,
            "toric": 1.0,
            "slepian_pollak_prolate": 1.0,
            "koszul_persistence": 1.0,
            "category_o_bgg": 1.0,
            "tropical": 1.0,
            "graphcg": 1.0,
            "analogical": 1.0,
            "combinatorial_commutative_algebra": 1.0,
            "exact_f2_combinatorial_audits": 1.0,
        },
        "means": {
            "topology_directed_asymmetry": mean("topology_directed_asymmetry_mean"),
            "topology_cycle_rank": mean("topology_cycle_rank_mean"),
            "toric_fan_cell_entropy": mean("toric_fan_cell_entropy"),
            "toric_mean_bend": mean("toric_mean_bend"),
            "slepian_concentration": mean("slepian_concentration"),
            "slepian_leakage": mean("slepian_leakage"),
            "koszul_d2_residual": mean("koszul_d2_residual"),
            "bgg_standard_leakage": mean("bgg_standard_leakage"),
            "graphcg_disentanglement_score": mean("graphcg_disentanglement_score"),
            "analogical_map_score": mean("analogical_map_score"),
            "tropical_chamber_crossing_rate": mean("tropical_chamber_crossing_rate"),
            "mst_efficiency": mean("mst_efficiency"),
            "toric_cca_topology_loss": mean("toric_cca_topology_loss"),
            "toric_cca_binomial_residual": mean("toric_cca_binomial_residual"),
            "toric_cca_stanley_reisner_nonface_mass": mean("toric_cca_stanley_reisner_nonface_mass"),
            "toric_cca_symbolic_resolution_loss": mean("toric_cca_symbolic_resolution_loss"),
            "toric_cca_symbolic_sr_monomial_generator_mass": mean(
                "toric_cca_symbolic_sr_monomial_generator_mass"
            ),
            "toric_cca_symbolic_taylor_lcm_syzygy_mass": mean(
                "toric_cca_symbolic_taylor_lcm_syzygy_mass"
            ),
            "toric_cca_symbolic_taylor_full_resolution_mass": mean(
                "toric_cca_symbolic_taylor_full_resolution_mass"
            ),
            "toric_cca_symbolic_hilbert_betti_pressure": mean(
                "toric_cca_symbolic_hilbert_betti_pressure"
            ),
            "toric_cca_symbolic_resolution_minimal_total_betti": mean(
                "toric_cca_symbolic_resolution_minimal_total_betti"
            ),
            "toric_cca_symbolic_resolution_projective_dimension": mean(
                "toric_cca_symbolic_resolution_projective_dimension"
            ),
            "toric_cca_symbolic_resolution_regularity": mean("toric_cca_symbolic_resolution_regularity"),
            "toric_cca_symbolic_resolution_nonminimality_log2": mean(
                "toric_cca_symbolic_resolution_nonminimality_log2"
            ),
            "toric_cca_koszul_fitting_rank_residual": mean("toric_cca_koszul_fitting_rank_residual"),
            "toric_cca_koszul_buchsbaum_eisenbud_rank_residual": mean(
                "toric_cca_koszul_buchsbaum_eisenbud_rank_residual"
            ),
            "toric_cca_koszul_buchsbaum_eisenbud_multiplier_residual": mean(
                "toric_cca_koszul_buchsbaum_eisenbud_multiplier_residual"
            ),
            "toric_cca_koszul_multigraded_betti_mass": mean("toric_cca_koszul_multigraded_betti_mass"),
            "toric_cca_chamber_coverage": mean("toric_cca_chamber_coverage"),
            "toric_cca_betti1_proxy": mean("toric_cca_betti1_proxy"),
            "toric_cca_exact_relation_pass_rate": mean("toric_cca_exact_relation_pass_rate"),
            "toric_cca_exact_relation_residual_mean": mean("toric_cca_exact_relation_residual_mean"),
            "toric_cca_exact_sr_nonface_edge_fraction": mean("toric_cca_exact_sr_nonface_edge_fraction"),
            "toric_cca_exact_h0": mean("toric_cca_exact_h0"),
            "toric_cca_exact_h1": mean("toric_cca_exact_h1"),
            "toric_cca_exact_betti_mismatch": mean("toric_cca_exact_betti_mismatch"),
            "toric_cca_exact_audit_backed_score": mean("toric_cca_exact_audit_backed_score"),
        },
        "recommendations": recommendation_lines(records, log_context),
    }


def recommendation_lines(records: list[dict[str, Any]], log_context: dict[str, Any]) -> list[str]:
    means = {
        key: float(np.nanmean([float(record.get(key, float("nan"))) for record in records]))
        for key in (
            "slepian_leakage",
            "graphcg_disentanglement_score",
            "analogical_map_score",
            "topology_directed_chain_commutator_mean",
            "bgg_standard_leakage",
            "tropical_chamber_crossing_rate",
            "toric_cca_binomial_residual",
            "toric_cca_stanley_reisner_nonface_mass",
            "toric_cca_betti1_proxy",
            "toric_cca_exact_sr_nonface_edge_fraction",
            "toric_cca_exact_betti_mismatch",
            "toric_cca_exact_audit_backed_score",
        )
    }
    latest_val = float(log_context.get("latest_val_bpb", float("nan")))
    target = float(log_context.get("target_bpb", 1.2))
    recs = []
    if math.isfinite(latest_val) and latest_val > target:
        recs.append("Keep the pre-threshold competition objective BPB-clean; use these structural metrics as sidecar restart/controller evidence.")
    if means["graphcg_disentanglement_score"] < 0.45:
        recs.append("GraphCG basis score is low; after the BPB gate, prioritize disentangling-basis training before increasing reasoning-loss weights.")
    if means["slepian_leakage"] > 0.55:
        recs.append("Slepian/Pollak leakage is high; use low-weight toric concentration or long-context phase regularity only after validation BPB is safe.")
    if means["topology_directed_chain_commutator_mean"] > 0.15:
        recs.append("Directed topology commutator pressure is visible; prefer warmup/damping for topology transfer rather than abrupt auxiliary loss insertion.")
    if means["analogical_map_score"] < 0.55:
        recs.append("Analogical transport is weak; schedule analogy-pair and memory-link curricula for the post-threshold reasoner phase.")
    if means["bgg_standard_leakage"] > 0.35:
        recs.append("BGG/Koszul leakage should remain diagnostic in the competition phase, then become a small controller loss in advanced reasoning phases.")
    if means["toric_cca_binomial_residual"] > 0.25 or means["toric_cca_stanley_reisner_nonface_mass"] > 0.18:
        recs.append("Combinatorial CCA residuals are high; keep the training bridge micro-weighted pre-gate, then increase toric-ideal/Stanley-Reisner pressure after BPB is safely below threshold.")
    if means["toric_cca_betti1_proxy"] > 3.0:
        recs.append("Betti1 proxy is large; prefer directed topology damping and memory-graph sparsification before stronger topology losses.")
    if means["toric_cca_exact_sr_nonface_edge_fraction"] > 0.20:
        recs.append("Exact CCA audit sees hard Stanley-Reisner nonface edges; keep CCA loss audit-backed but micro-weighted until BPB is safe, then use SR nonface pressure before raising broader topology weights.")
    if means["toric_cca_exact_betti_mismatch"] > 2.0:
        recs.append("Exact F2 chamber Betti mismatch is high; prefer GraphCG basis conditioning and chamber sparsification before increasing Koszul/topology loss weight.")
    if means["toric_cca_exact_audit_backed_score"] > 0.65 and math.isfinite(latest_val) and latest_val <= target:
        recs.append("Exact CCA audit is stable enough for a cautious post-gate frequency increase before increasing raw CCA loss scale.")
    if not recs:
        recs.append("Advanced geometry is stable enough to keep as monitoring while BPB validation remains the primary gate.")
    return recs


def write_review_prompt(output_dir: Path, summary: dict[str, Any]) -> None:
    prompt = [
        "# Seq4096 Artifact Review Prompt",
        "",
        "Review every current-run image and output file listed in `artifact_inventory.json`.",
        "Classify each family as desired, desired but too weak/slow, or undesirable.",
        "Recommend training/controller changes only if they help the active BPB gate or the post-threshold reasoning curriculum.",
        "",
        "## Key Summary",
        "",
        json.dumps({key: summary[key] for key in ("checkpoint_step", "latest_train_bpb", "latest_val_bpb", "means", "recommendations")}, indent=2, default=str),
    ]
    (output_dir / "analysis_review_prompt.md").write_text("\n".join(prompt) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    geometry_dir = output_dir / "geometry"
    simplex_dir = output_dir / "simplex"
    state, step, _payload = load_checkpoint_state(checkpoint)
    log_context = parse_training_log(Path(args.log), args.target_bpb) if args.log else parse_training_log(Path(), args.target_bpb)
    topo_cfg = ReasoningTopologyConfig(
        max_points=max(8, int(args.topology_max_points)),
        max_windows=4,
        window_size=max(8, int(args.topology_window_size)),
        step_stride=max(2, int(args.topology_window_size) // 3),
        levels=max(2, int(args.topology_levels)),
    )
    inputs = extract_trajectory_inputs(state, max_points=int(args.max_points), records=int(args.records))
    records = [
        analyze_record(item, log_context=log_context, topology_config=topo_cfg)
        for item in inputs
    ]
    records = annotate_scores(records)
    all_files: list[str] = []
    for record in records:
        all_files.extend(plot_record_artifacts(record, geometry_dir))

    for name, spec in TRIANGLE_SPECS.items():
        out = geometry_dir / "triangles" / f"{name}.png"
        plot_triangle(records, spec, out)
        all_files.append(str(out))
    for name, spec in TETRAHEDRON_SPECS.items():
        out = geometry_dir / "tetrahedra" / f"{name}.png"
        plot_tetrahedron(records, spec, out)
        all_files.append(str(out))
        all_files.append(str(out.with_suffix(".html")))

    for name, spec in SIMPLEX_TRIANGLES.items():
        out = simplex_dir / f"{name}.png"
        plot_triangle(records, spec, out)
        all_files.append(str(out))
    for name, spec in SIMPLEX_TETRAHEDRA.items():
        out = simplex_dir / f"{name}.png"
        plot_tetrahedron(records, spec, out)
        all_files.append(str(out))
        all_files.append(str(out.with_suffix(".html")))

    public_records = [record_public(record) for record in records]
    write_json(geometry_dir / "reasoning_geometry_records.json", public_records)
    write_records_csv(geometry_dir / "reasoning_geometry_records.csv", records)
    write_json(geometry_dir / "selected_records.json", [{"record_id": record["record_id"], "family": record["family"]} for record in records])
    write_json(simplex_dir / "reasoning_simplex_records.json", public_records)
    write_records_csv(simplex_dir / "reasoning_simplex_records.csv", records)
    summary = aggregate_summary(records, checkpoint=checkpoint, step=step, log_context=log_context, files=all_files)
    write_json(geometry_dir / "reasoning_geometry_summary.json", summary)
    write_json(simplex_dir / "reasoning_simplex_summary.json", summary)
    inventory = {
        "analysis_dir": str(output_dir),
        "run_path": args.run_path,
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(step),
        "files": all_files,
        "image_files": [path for path in all_files if path.endswith(".png")],
        "json_files": [
            str(geometry_dir / "reasoning_geometry_summary.json"),
            str(simplex_dir / "reasoning_simplex_summary.json"),
            str(geometry_dir / "reasoning_geometry_records.json"),
            str(simplex_dir / "reasoning_simplex_records.json"),
        ],
        "review_required": 1.0,
    }
    write_json(output_dir / "artifact_inventory.json", inventory)
    write_review_prompt(output_dir, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_analysis(args)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
