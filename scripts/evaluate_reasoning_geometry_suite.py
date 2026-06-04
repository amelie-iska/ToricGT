#!/usr/bin/env python3
# Example:
# conda run --no-capture-output -n tokengt env PYTHONPATH=src \
#   python scripts/evaluate_reasoning_geometry_suite.py \
#   --checkpoint checkpoints/parameter_golf_oai_dense/random_order_step_00014750.pt \
#   --data-glob 'data/curated_hf_shards/validation/*.parquet' \
#   --output-dir outputs/reasoning_geometry_suite/oai-step14750 \
#   --records 4 --branches 6 --seq-len 1024 --seed 10017
"""Extended geometry diagnostics for random-order ToricGT reasoning.

This script complements ``evaluate_reasoning_simplex.py``.  It selects
reasoning-heavy validation records with solution/answer fields, evaluates
multiple random-order and GFlowNet branches from a checkpoint, projects real
embedding-space hidden trajectories to three dimensions, and renders richer
triangle/tetrahedron diagnostics from model-computed quantities.

The terminal markers are solution-conditioned terminals: the plotted sequence
contains the source solution/answer span when available.  They are not claimed
to be free-form generated solutions unless the branch likelihood and decoded
evaluation explicitly support that downstream.
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import math
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
try:
    import wandb
except ImportError:  # pragma: no cover - optional analysis dependency
    wandb = None

from evaluate_reasoning_simplex import blue_colormap, load_model
from toricgt.complexity import random_order_complexity_metrics
from toricgt.random_order_lm import DenseRandomOrderToricLM, byte_encode, random_order_batch
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
from toricgt.topological_reasoning import ReasoningTopologyConfig, directed_step_filtration_stats_np
from toricgt.slepian_torus import ToricSlepianConfig, toric_slepian_audit
from toricgt.toric_geometry_tasks import empirical_toric_shadow_stats_np
from train_parameter_golf_random_order import ParquetByteChunkDataset


SELECT_COLUMNS = (
    "record_id",
    "dataset",
    "task_family",
    "language",
    "question",
    "answer",
    "solution",
    "reasoning",
    "metadata_json",
    "text",
    "graph_json",
    "estimated_tokens",
    "quality_flags_json",
)

DEFAULT_DOMAIN_KEYWORDS = (
    "math",
    "code",
    "algorithm",
    "graph",
    "got",
    "tree",
    "physics",
    "spatial",
    "health",
    "medical",
    "medicine",
    "biomed",
    "chem",
    "protein",
    "hebrew",
    "morph",
    "toric",
    "tropical",
)

TRIANGLE_SPECS = {
    "reasoning_k_bpb": {
        "labels": ["reasoning time", "low K(x|helpers)", "low BPB"],
        "scores": ["score/reasoning", "score/relative_k", "score/bpb_quality"],
        "title": "Reasoning/K(x|helpers)/BPB simplex",
        "intensity": "score/reasoning",
    },
    "solution_basin": {
        "labels": ["answer-span likelihood", "terminal confidence", "low BPB"],
        "scores": ["score/answer_quality", "score/terminal_confidence", "score/bpb_quality"],
        "title": "Actual-solution basin simplex",
        "intensity": "score/answer_quality",
    },
    "trajectory_flow": {
        "labels": ["smooth flow", "MST efficiency", "trajectory depth"],
        "scores": ["score/path_smoothness", "score/mst_efficiency", "score/trajectory_depth"],
        "title": "Embedding trajectory flow simplex",
        "intensity": "score/path_smoothness",
    },
    "exploration_control": {
        "labels": ["GFlowNet diversity", "low curvature", "low BPB"],
        "scores": ["score/action_diversity", "score/path_smoothness", "score/bpb_quality"],
        "title": "Exploration/control simplex",
        "intensity": "score/action_diversity",
    },
    "compression_reasoning": {
        "labels": ["low K(x|helpers)", "helper gain", "answer likelihood"],
        "scores": ["score/relative_k", "score/relative_k_gain", "score/answer_quality"],
        "title": "Conditional-compression/reasoning simplex",
        "intensity": "score/compression_efficiency",
    },
    "robustness_generalization": {
        "labels": ["order robustness", "GFlowNet diversity", "low loss"],
        "scores": ["score/order_robustness", "score/action_diversity", "score/loss_quality"],
        "title": "Order-robust generalization simplex",
        "intensity": "score/order_robustness",
    },
    "toric_memory_control": {
        "labels": ["toric entropy", "low BPB", "smooth flow"],
        "scores": ["score/toric_entropy", "score/bpb_quality", "score/path_smoothness"],
        "title": "Toric-memory control simplex",
        "intensity": "score/toric_entropy",
    },
    "energy_landscape": {
        "labels": ["low energy", "terminal confidence", "MST efficiency"],
        "scores": ["score/energy_quality", "score/terminal_confidence", "score/mst_efficiency"],
        "title": "Energy-landscape simplex",
        "intensity": "score/energy_quality",
    },
}

TETRAHEDRON_SPECS = {
    "reasoning_k_bpb_mst": {
        "labels": ["reasoning time", "low K(x|helpers)", "low BPB", "MST efficiency"],
        "scores": ["score/reasoning", "score/relative_k", "score/bpb_quality", "score/mst_efficiency"],
        "title": "Reasoning/K(x|helpers)/BPB/MST tetrahedron",
    },
    "solution_bpb_smooth_diversity": {
        "labels": ["answer likelihood", "low BPB", "smooth flow", "GFlowNet diversity"],
        "scores": ["score/answer_quality", "score/bpb_quality", "score/path_smoothness", "score/action_diversity"],
        "title": "Solution/BPB/flow/diversity tetrahedron",
    },
    "complexity_geometry_solution": {
        "labels": ["low K(x|helpers)", "trajectory depth", "MST efficiency", "answer likelihood"],
        "scores": ["score/relative_k", "score/trajectory_depth", "score/mst_efficiency", "score/answer_quality"],
        "title": "Conditional-complexity/geometry/solution tetrahedron",
    },
    "energy_control": {
        "labels": ["low energy", "terminal confidence", "smooth flow", "low BPB"],
        "scores": ["score/energy_quality", "score/terminal_confidence", "score/path_smoothness", "score/bpb_quality"],
        "title": "Energy-control tetrahedron",
    },
    "toric_gfn_bpb": {
        "labels": ["toric entropy", "GFlowNet diversity", "low BPB", "order robustness"],
        "scores": ["score/toric_entropy", "score/action_diversity", "score/bpb_quality", "score/order_robustness"],
        "title": "Toric/GFlowNet/BPB tetrahedron",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/parameter_golf_oai_dense/random_order_step_00014750.pt")
    parser.add_argument("--config", default="config/train.parameter_golf_random_order_dense.yaml")
    parser.add_argument("--data-glob", default="data/curated_hf_shards/validation/*.parquet")
    parser.add_argument("--output-dir", default="outputs/reasoning_geometry_suite")
    parser.add_argument("--records", type=int, default=4)
    parser.add_argument("--branches", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=10017)
    parser.add_argument("--max-files", type=int, default=24)
    parser.add_argument("--scan-batches-per-file", type=int, default=4)
    parser.add_argument("--scan-batch-size", type=int, default=512)
    parser.add_argument("--problem-chars", type=int, default=2500)
    parser.add_argument("--solution-chars", type=int, default=6500)
    parser.add_argument("--graph-chars", type=int, default=1600)
    parser.add_argument("--max-pca-points", type=int, default=4096)
    parser.add_argument("--max-plot-points", type=int, default=180)
    parser.add_argument("--max-mst-nodes", type=int, default=96)
    parser.add_argument("--simplicial-windows", type=int, default=6)
    parser.add_argument("--simplicial-radius-quantiles", default="0.08,0.14,0.22")
    parser.add_argument("--simplicial-default-level", default="default", choices=["sparse", "default", "dense"])
    parser.add_argument("--simplicial-max-edges-per-window", type=int, default=34)
    parser.add_argument("--simplicial-max-triangles-per-window", type=int, default=12)
    parser.add_argument("--compressors", nargs="+", default=["zlib", "lzma"])
    parser.add_argument("--domain-keywords", nargs="+", default=list(DEFAULT_DOMAIN_KEYWORDS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Log analysis summaries and images to Weights & Biases")
    parser.add_argument("--wandb-project", default="toricgt-parameter-golf")
    parser.add_argument("--wandb-run-name", default="")
    return parser.parse_args()


def safe_text(value: Any) -> str:
    return value if isinstance(value, str) and value.strip() else ""


def slug(value: str, max_len: int = 72) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower())
    return value.strip("-")[:max_len] or "record"


def parse_float_csv(value: str, *, fallback: tuple[float, ...]) -> tuple[float, ...]:
    parsed: list[float] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = float(raw)
        except ValueError:
            continue
        if 0.0 < item < 1.0:
            parsed.append(item)
    return tuple(parsed) if parsed else fallback


def row_domain_prefix(row: dict[str, Any]) -> str:
    formatter = ParquetByteChunkDataset(
        parquet_glob="",
        seq_len=128,
        byte_offset=4,
        repeat=False,
        include_graph_projection=True,
        graph_projection_max_chars=1024,
    )
    return formatter._domain_prefix(row)  # noqa: SLF001 - reuse training formatter intentionally.


def graph_projection(row: dict[str, Any], max_chars: int, byte_offset: int) -> str:
    formatter = ParquetByteChunkDataset(
        parquet_glob="",
        seq_len=128,
        byte_offset=byte_offset,
        repeat=False,
        include_graph_projection=True,
        graph_projection_max_chars=max_chars,
    )
    return formatter._graph_projection(row.get("graph_json"))  # noqa: SLF001


def bounded_marker(value: str, max_chars: int = 512) -> str:
    """Return a compact actual-solution marker for span overlays."""

    value = value.strip()
    if not value:
        return ""
    answer_match = re.search(r"(?:final answer|answer)\s*[:：]\s*(.{16,})", value, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()[:max_chars]
    if "</think>" in value:
        tail = value.split("</think>", 1)[1].strip()
        if tail:
            return tail[:max_chars]
    return value[:max_chars]


def analysis_text(row: dict[str, Any], args: argparse.Namespace, byte_offset: int) -> tuple[str, str, tuple[int, int] | None]:
    question = safe_text(row.get("question"))
    answer = safe_text(row.get("answer"))
    solution = safe_text(row.get("solution"))
    reasoning = safe_text(row.get("reasoning"))
    primary = safe_text(row.get("text"))
    solution_body = "\n\n".join(part for part in (solution, reasoning, answer) if part).strip()
    if not solution_body and primary:
        solution_body = primary
    prefix = row_domain_prefix(row)
    graph_text = graph_projection(row, args.graph_chars, byte_offset=byte_offset)
    sections = []
    if prefix:
        sections.append(prefix)
    if question:
        sections.append("<problem>\n" + question[: args.problem_chars])
    if solution_body:
        sections.append("<solution>\n" + solution_body[: args.solution_chars])
    elif primary:
        sections.append("<text>\n" + primary[: args.problem_chars + args.solution_chars])
    if graph_text:
        sections.append(graph_text[: args.graph_chars])
    text = "\n\n".join(sections).strip()
    marker = bounded_marker(answer or solution or reasoning)
    encoded = byte_encode(text, byte_offset=byte_offset)
    span = None
    if marker:
        marker_encoded = byte_encode(marker, byte_offset=byte_offset)
        span = find_subsequence(encoded, marker_encoded)
        if span is None and len(marker_encoded) > 64:
            span = find_subsequence(encoded, marker_encoded[: min(len(marker_encoded), 256)])
    return text, marker, span


def find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int] | None:
    if not needle or len(needle) > len(haystack):
        return None
    first = needle[0]
    limit = len(haystack) - len(needle) + 1
    for start in range(limit):
        if haystack[start] == first and haystack[start : start + len(needle)] == needle:
            return (start, start + len(needle))
    return None


def record_score(row: dict[str, Any], keywords: tuple[str, ...]) -> float:
    haystack = " ".join(
        safe_text(row.get(key)).lower()
        for key in ("dataset", "task_family", "language", "question", "answer", "solution", "reasoning", "metadata_json")
    )
    score = 0.0
    score += 4.0 if safe_text(row.get("question")) else 0.0
    score += 5.0 if safe_text(row.get("answer")) or safe_text(row.get("solution")) else 0.0
    score += 2.0 if safe_text(row.get("graph_json")) else 0.0
    score += min(4.0, len(safe_text(row.get("answer")) + safe_text(row.get("solution"))) / 2000.0)
    score += sum(1.0 for keyword in keywords if keyword.lower() in haystack)
    try:
        estimated = int(row.get("estimated_tokens") or 0)
    except (TypeError, ValueError):
        estimated = 0
    if estimated:
        score += min(3.0, math.log1p(estimated) / 3.0)
    return float(score)


def select_reasoning_records(args: argparse.Namespace, byte_offset: int) -> list[dict[str, Any]]:
    files = sorted(glob.glob(args.data_glob))[: max(1, args.max_files)]
    if not files:
        raise FileNotFoundError(f"no parquet files matched {args.data_glob!r}")
    keywords = tuple(keyword.lower() for keyword in args.domain_keywords)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for path in files:
        parquet = pq.ParquetFile(path)
        columns = [name for name in SELECT_COLUMNS if name in parquet.schema.names]
        for batch_index, batch in enumerate(parquet.iter_batches(batch_size=args.scan_batch_size, columns=columns)):
            if batch_index >= args.scan_batches_per_file:
                break
            table = batch.to_pydict()
            if not table:
                continue
            rows = [dict(zip(table, values)) for values in zip(*table.values())]
            for row in rows:
                text, _, span = analysis_text(row, args, byte_offset=byte_offset)
                encoded = byte_encode(text, byte_offset=byte_offset)
                if len(encoded) < args.seq_len:
                    continue
                score = record_score(row, keywords)
                if span is not None:
                    score += 3.0
                if score <= 0:
                    continue
                candidates.append((score, row))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[dict[str, Any]] = []
    seen_domains: set[str] = set()
    seen_ids: set[str] = set()
    for _, row in candidates:
        record_id = safe_text(row.get("record_id")) or safe_text(row.get("content_hash")) or str(len(seen_ids))
        if record_id in seen_ids:
            continue
        domain = f"{safe_text(row.get('dataset'))}:{safe_text(row.get('task_family'))}"
        if domain in seen_domains and len(seen_domains) < args.records:
            continue
        selected.append(row)
        seen_ids.add(record_id)
        seen_domains.add(domain)
        if len(selected) >= args.records:
            break
    if len(selected) < args.records:
        for _, row in candidates:
            record_id = safe_text(row.get("record_id")) or str(len(seen_ids))
            if record_id in seen_ids:
                continue
            selected.append(row)
            seen_ids.add(record_id)
            if len(selected) >= args.records:
                break
    if not selected:
        raise RuntimeError("no reasoning-heavy records with enough bytes were found")
    return selected


def token_window_for_record(
    row: dict[str, Any],
    args: argparse.Namespace,
    byte_offset: int,
) -> dict[str, Any]:
    text, marker, marker_span = analysis_text(row, args, byte_offset=byte_offset)
    encoded = byte_encode(text, byte_offset=byte_offset)
    if len(encoded) < args.seq_len:
        raise ValueError("record is shorter than seq_len after analysis formatting")
    start = 0
    span = marker_span
    if span is not None and span[1] > args.seq_len:
        center = (span[0] + span[1]) // 2
        start = max(0, min(center - args.seq_len // 2, len(encoded) - args.seq_len))
    tokens = encoded[start : start + args.seq_len]
    adjusted_span = None
    if span is not None:
        left = max(span[0], start) - start
        right = min(span[1], start + args.seq_len) - start
        if right > left:
            adjusted_span = (left, right)
    return {
        "text": text,
        "marker": marker,
        "answer_span": adjusted_span,
        "window_start": start,
        "tokens": tokens,
    }


def hidden_pca(hidden_by_branch: list[np.ndarray], max_points: int, seed: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    all_points = np.concatenate(hidden_by_branch, axis=0).astype(np.float32)
    rng = np.random.default_rng(seed)
    if all_points.shape[0] > max_points:
        indices = rng.choice(all_points.shape[0], size=max_points, replace=False)
        fit_points = all_points[indices]
    else:
        fit_points = all_points
    mean = fit_points.mean(axis=0, keepdims=True)
    centered = fit_points - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3].astype(np.float32)
    projected = [(points.astype(np.float32) - mean) @ components.T for points in hidden_by_branch]
    return projected, mean.squeeze(0), components


def path_stats(points: np.ndarray) -> dict[str, float]:
    if points.shape[0] < 3:
        return {
            "path_length": 0.0,
            "mean_speed": 0.0,
            "curvature": 0.0,
            "terminal_radius": float(np.linalg.norm(points[-1])) if points.size else 0.0,
        }
    velocity = np.diff(points, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0)
    curvature = np.linalg.norm(acceleration, axis=1) / np.maximum(speed[:-1] ** 2, 1e-8)
    return {
        "path_length": float(speed.sum()),
        "mean_speed": float(speed.mean()),
        "curvature": float(np.mean(curvature)),
        "terminal_radius": float(np.linalg.norm(points[-1])),
    }


def helper_conditional_k(record: dict[str, Any], compressor: str = "lzma") -> float:
    """Return the helper-conditioned K(x|y) proxy for simplex plots.

    This deliberately avoids plotting an absolute K(x) estimate.  The primary
    value is the shortest-known conditional target code under the random-order
    helper family: strict prefix, public order/tree program, graph-projected
    byte chunk, GFlowNet action trace, and optional graph/tree payloads.
    """

    candidates = [
        record.get(f"complexity/target_helper_cond_k_{compressor}_mean"),
        record.get(f"complexity/target_cond_k_{compressor}_mean"),
        record.get("complexity/target_helper_cond_k_zlib_mean"),
        record.get("complexity/target_cond_k_zlib_mean"),
    ]
    for value in candidates:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return 0.0


def analogical_helper_gain(record: dict[str, Any], compressor: str = "lzma") -> float:
    """Return positive gain when analogy helpers reduce K(x|helpers)."""

    gain_candidates = [
        record.get(f"complexity/analogical_transfer_gain_k_{compressor}_mean"),
        record.get("complexity/analogical_transfer_gain_k_zlib_mean"),
    ]
    for value in gain_candidates:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return max(0.0, float(value))
    relative_candidates = [
        record.get(f"complexity/analogical_transfer_relative_k_{compressor}_mean"),
        record.get("complexity/analogical_transfer_relative_k_zlib_mean"),
    ]
    for value in relative_candidates:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return max(0.0, -float(value))
    return 0.0


@torch.no_grad()
def evaluate_branch(
    model: DenseRandomOrderToricLM,
    tokens: torch.Tensor,
    sample_id: torch.Tensor,
    pass_id: int,
    branch_index: int,
    answer_span: tuple[int, int] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = tokens.device
    batch = random_order_batch(
        tokens,
        seed=model.config.seed,
        sample_ids=sample_id,
        pass_id=pass_id,
        bos_token_id=model.bos_token_id,
    )
    aux = model.forward_from_previous(
        batch.previous_tokens,
        batch.target_positions,
        sample_gflownet=model.config.use_gflownet_policy,
        return_aux=True,
    )
    if not isinstance(aux, dict):
        raise RuntimeError("expected auxiliary output")
    logits = aux["logits"]
    per_token_nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        batch.target_tokens.reshape(-1),
        reduction="none",
    ).view_as(batch.target_tokens)
    loss = per_token_nll.mean()
    log_probs = F.log_softmax(logits.float(), dim=-1)
    target_logp = log_probs.gather(-1, batch.target_tokens.unsqueeze(-1)).squeeze(-1)
    terminal_confidence = float(target_logp[:, -max(1, min(32, target_logp.shape[1])) :].mean().exp().detach().cpu())
    original_nll = torch.empty_like(per_token_nll)
    original_nll.scatter_(1, batch.target_positions, per_token_nll)
    answer_nll = None
    answer_steps: list[int] = []
    if answer_span is not None:
        left, right = answer_span
        if right > left:
            answer_nll = float(original_nll[:, left:right].mean().detach().cpu())
            mask = (batch.target_positions[0] >= left) & (batch.target_positions[0] < right)
            answer_steps = torch.where(mask)[0].detach().cpu().tolist()
    complexity = random_order_complexity_metrics(
        tokens=tokens,
        previous_tokens=batch.previous_tokens,
        target_tokens=batch.target_tokens,
        permutation=batch.permutation,
        logits=logits,
        action_ids=aux.get("action_ids"),
        byte_offset=model.config.byte_offset,
        prefix="complexity",
        compressors=tuple(args.compressors),
        max_samples=1,
    )
    hidden = aux["hidden"].detach().float().cpu().numpy()[0]
    toric_probe_metrics: dict[str, float] = {}
    if getattr(model, "toric_geometry_probe", None) is not None:
        with torch.no_grad():
            probe_out = model.toric_geometry_probe(aux["hidden"], batch.target_positions, batch.target_tokens)
        for key, value in probe_out.items():
            if key == "toric_geometry_loss" or not torch.is_tensor(value):
                continue
            if value.ndim == 0:
                toric_probe_metrics[f"toric_geometry/{key}"] = float(value.detach().cpu())
    target_positions_np = batch.target_positions.detach().cpu().numpy()[0]
    phase_u = (float(model.config.theta) * target_positions_np.astype(float)) % 1.0
    phase_v = (float(model.config.beta) * target_positions_np.astype(float)) % 1.0
    phase_cocycle = (
        float(model.config.theta) * target_positions_np.astype(float) ** 2
        + float(model.config.beta) * target_positions_np.astype(float)
    ) % 1.0
    torus_r = 0.32
    torus_R = 1.08
    toric_torus_path = np.stack(
        [
            (torus_R + torus_r * np.cos(2 * np.pi * phase_v)) * np.cos(2 * np.pi * phase_u),
            (torus_R + torus_r * np.cos(2 * np.pi * phase_v)) * np.sin(2 * np.pi * phase_u),
            torus_r * np.sin(2 * np.pi * phase_v),
        ],
        axis=-1,
    ).astype(np.float32)
    phase_points = np.stack([phase_u, phase_v], axis=-1)
    phase_delta = phase_points[:, None, :] - phase_points[None, :, :]
    phase_delta = np.minimum(np.abs(phase_delta), 1.0 - np.abs(phase_delta))
    phase_dist = np.linalg.norm(phase_delta, axis=-1)
    phase_recurrence = float(np.mean(np.partition(phase_dist + np.eye(phase_dist.shape[0]) * 1.0e6, kth=1, axis=1)[:, 1]))
    nll_np = per_token_nll.detach().float().cpu().numpy()[0]
    slepian_audit = toric_slepian_audit(
        phase_u.astype(np.float64),
        phase_v.astype(np.float64),
        energy=nll_np.astype(np.float64),
        config=ToricSlepianConfig(
            half_bandwidth=0.075,
            modes=6,
            theta=float(model.config.theta),
            beta=float(model.config.beta),
        ),
    )
    if getattr(model, "graphcg_direction_basis", None) is not None:
        basis = model.graphcg_direction_basis.detach().float().cpu().numpy()
        basis = basis / np.maximum(np.linalg.norm(basis, axis=-1, keepdims=True), 1e-8)
        chart = hidden @ basis.T
        chart_abs = np.abs(chart)
        chart_axis = np.argmax(chart_abs, axis=-1).astype(np.int32)
        sorted_abs = np.sort(chart_abs, axis=-1)
        chart_margin = (sorted_abs[:, -1] - sorted_abs[:, -2]) if chart_abs.shape[1] > 1 else sorted_abs[:, -1]
    else:
        chart_axis = np.zeros(hidden.shape[0], dtype=np.int32)
        chart_margin = np.zeros(hidden.shape[0], dtype=np.float32)
    hidden_for_mst = hidden
    if hidden_for_mst.shape[0] > args.max_mst_nodes:
        idx = np.linspace(0, hidden_for_mst.shape[0] - 1, num=args.max_mst_nodes).round().astype(int)
        hidden_for_mst = hidden_for_mst[idx]
    mst = prim_mst_stats(hidden_for_mst.tolist())
    action_ids = aux.get("action_ids")
    if action_ids is not None:
        flat_actions = action_ids.detach().cpu().reshape(-1).tolist()
        action_diversity = len(set(int(x) for x in flat_actions)) / max(1, int(model.config.gflownet_num_actions))
    else:
        flat_actions = []
        action_diversity = 0.0
    record = {
        "branch_index": branch_index,
        "pass_id": pass_id,
        "loss": float(loss.detach().cpu()),
        "bpb": float(loss.detach().cpu() / math.log(2.0)),
        "answer_nll": float(answer_nll) if answer_nll is not None else float(loss.detach().cpu()),
        "answer_bpb": float(answer_nll / math.log(2.0)) if answer_nll is not None else float(loss.detach().cpu() / math.log(2.0)),
        "terminal_confidence": terminal_confidence,
        "gflownet_entropy": float(aux.get("policy_entropy", torch.zeros(())).detach().cpu()),
        "gflownet_action_diversity": float(action_diversity),
        "toric_memory_entropy": float(aux.get("toric_memory_entropy", torch.zeros(())).detach().cpu()),
        "smear_temperature": float(aux.get("smear_temperature", torch.zeros(())).detach().cpu()),
        "trajectory_tokens": float(tokens.shape[1] * max(1, model.config.recurrent_passes)),
        "hidden": hidden,
        "per_token_nll": nll_np,
        "target_positions": target_positions_np,
        "toric_torus_path": toric_torus_path,
        "toric_phase_u": phase_u.astype(np.float32),
        "toric_phase_v": phase_v.astype(np.float32),
        "toric_phase_cocycle": phase_cocycle.astype(np.float32),
        "toric_phase_recurrence": phase_recurrence,
        "toric_slepian_eigenvalues": np.asarray(slepian_audit["slepian_eigenvalues"], dtype=np.float32),
        "toric_slepian_coefficients": np.asarray(slepian_audit["slepian_coefficients"], dtype=np.float32),
        "toric_slepian_reconstruction": np.asarray(slepian_audit["slepian_reconstruction"], dtype=np.float32),
        "toric_slepian_envelope": np.asarray(slepian_audit["slepian_envelope"], dtype=np.float32),
        "toric_slepian_concentration": float(slepian_audit["slepian_concentration"]),
        "toric_slepian_leakage": float(slepian_audit["slepian_leakage"]),
        "toric_slepian_mode_entropy": float(slepian_audit["slepian_mode_entropy"]),
        "toric_slepian_effective_modes": float(slepian_audit["slepian_effective_modes"]),
        "toric_slepian_bandwidth": float(slepian_audit["slepian_bandwidth"]),
        "graphcg_chart_axis": chart_axis,
        "graphcg_chart_margin": chart_margin.astype(np.float32),
        "toric_shadow": empirical_toric_shadow_stats_np(hidden, max_points=int(args.max_plot_points)),
        "answer_steps": answer_steps,
        "action_ids": flat_actions,
    }
    record.update(toric_probe_metrics)
    for key, value in complexity.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            record[key] = float(value)
    for key, value in mst.items():
        record[f"mst/{key}"] = float(value)
    helper_k = helper_conditional_k(record)
    record["helper_cond_k_proxy"] = float(helper_k)
    record["helper_cond_k_per_byte"] = float(helper_k / max(1, int(tokens.shape[1])))
    record["analogical_k_gain_proxy"] = float(analogical_helper_gain(record))
    # Backward-compatible field name used by older JSON consumers.  It now
    # means helper-conditioned K(x|helpers), not absolute K(x).
    record["k_proxy"] = record["helper_cond_k_proxy"]
    return record


def enrich_branch_scores(records: list[dict[str, Any]]) -> None:
    by_record: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_record.setdefault(str(record["record_id"]), []).append(record)
    for group in by_record.values():
        bpb_values = np.array([float(item["bpb"]) for item in group], dtype=float)
        std = float(np.std(bpb_values)) if len(bpb_values) > 1 else 0.0
        for item in group:
            item["order_bpb_std"] = std
            item["order_robustness_raw"] = 1.0 / (1.0 + std)
    for record in records:
        path = record.get("projected_path")
        if isinstance(path, np.ndarray):
            stats = path_stats(path)
            record.update(stats)
        else:
            record.update({"path_length": 0.0, "mean_speed": 0.0, "curvature": 0.0, "terminal_radius": 0.0})
        record["mst_efficiency"] = float(record.get("mst/mst_efficiency", 0.5))
        record["energy"] = float(record.get("loss", 0.0))
        record["trajectory_depth"] = float(math.log1p(record.get("trajectory_tokens", 0.0)) + 0.02 * record["path_length"])
        record["path_smoothness_raw"] = float(1.0 / (1.0 + max(0.0, record["curvature"])))
        record["relative_k_proxy"] = float(record.get("helper_cond_k_per_byte", record.get("k_proxy", 0.0)))
        record["compression_efficiency_raw"] = float(
            (1.0 + record.get("analogical_k_gain_proxy", 0.0))
            / (1.0 + max(0.0, record["relative_k_proxy"]) + max(record["bpb"], 1e-6))
        )
    attach_normalized_scores(
        records,
        {
            "score/reasoning": ("trajectory_depth", True),
            "score/k": ("relative_k_proxy", False),
            "score/relative_k": ("relative_k_proxy", False),
            "score/relative_k_gain": ("analogical_k_gain_proxy", True),
            "score/bpb_quality": ("bpb", False),
            "score/loss_quality": ("loss", False),
            "score/answer_quality": ("answer_bpb", False),
            "score/terminal_confidence": ("terminal_confidence", True),
            "score/mst_efficiency": ("mst_efficiency", True),
            "score/action_diversity": ("gflownet_action_diversity", True),
            "score/path_smoothness": ("path_smoothness_raw", True),
            "score/trajectory_depth": ("trajectory_depth", True),
            "score/compression_efficiency": ("compression_efficiency_raw", True),
            "score/order_robustness": ("order_robustness_raw", True),
            "score/toric_entropy": ("toric_memory_entropy", True),
            "score/energy_quality": ("energy", False),
        },
    )


def plot_triangle(records: list[dict[str, Any]], spec: dict[str, Any], output_path: Path) -> None:
    vertices = [TRIANGLE_VERTICES["left"], TRIANGLE_VERTICES["right"], TRIANGLE_VERTICES["top"]]
    labels = spec["labels"]
    points: list[tuple[float, float]] = []
    intensities: list[float] = []
    for record in records:
        simplex = simplex_record(record, spec["scores"], labels)
        points.append(barycentric_to_cartesian(simplex["weights"], vertices))
        intensities.append(float(record.get(spec.get("intensity", "score/reasoning"), 0.0)))
    grid = triangle_grid(resolution=120)
    xs = [item[0] for item in grid]
    ys = [item[1] for item in grid]
    zs = [rbf_interpolate((x, y), points, intensities, bandwidth=0.15) for x, y, _ in grid]
    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor="#030712")
    ax.set_facecolor("#030712")
    tri = mtri.Triangulation(xs, ys)
    contour = ax.tricontourf(tri, zs, levels=24, cmap=blue_colormap(), alpha=0.96)
    ax.tricontour(tri, zs, levels=9, colors="#a8f6ff", linewidths=0.33, alpha=0.5)
    boundary_x = [vertices[0][0], vertices[1][0], vertices[2][0], vertices[0][0]]
    boundary_y = [vertices[0][1], vertices[1][1], vertices[2][1], vertices[0][1]]
    ax.plot(boundary_x, boundary_y, color="#6df6ff", linewidth=1.9)
    scatter = ax.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        c=[float(record["bpb"]) for record in records],
        cmap="magma_r",
        s=76,
        edgecolor="white",
        linewidth=0.55,
        zorder=5,
    )
    for record, (x, y) in zip(records, points):
        ax.text(
            x,
            y + 0.018,
            f"R{record['record_index']}:{record['branch_index']}",
            color="white",
            fontsize=6.5,
            ha="center",
        )
    label_offsets = [(-0.08, -0.055), (0.08, -0.055), (0.0, -0.036)]
    for label, vertex, offset in zip(labels, vertices, label_offsets):
        ax.text(vertex[0] + offset[0], vertex[1] + offset[1], label, color="#e8fbff", fontsize=10, ha="center")
    ax.set_title(spec["title"], color="white", fontsize=11, pad=14)
    ax.set_aspect("equal")
    ax.set_axis_off()
    cbar = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("simplex intensity", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    sb = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.09)
    sb.set_label("BPB", color="white")
    sb.ax.yaxis.set_tick_params(color="white")
    plt.setp(sb.ax.get_yticklabels(), color="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def tetrahedron_coords(records: list[dict[str, Any]], spec: dict[str, Any]) -> list[tuple[float, float, float]]:
    vertices = [TETRAHEDRON_VERTICES[key] for key in ("a", "b", "c", "d")]
    coords = []
    for record in records:
        simplex = simplex_record(record, spec["scores"], spec["labels"])
        coords.append(barycentric_to_cartesian(simplex["weights"], vertices))
    return coords


def plot_tetrahedron(records: list[dict[str, Any]], spec: dict[str, Any], output_path: Path) -> None:
    vertices = [TETRAHEDRON_VERTICES[key] for key in ("a", "b", "c", "d")]
    coords = tetrahedron_coords(records, spec)
    fig = plt.figure(figsize=(8.5, 7.5), facecolor="#030712")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#030712")
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for left, right in edges:
        ax.plot(
            [vertices[left][0], vertices[right][0]],
            [vertices[left][1], vertices[right][1]],
            [vertices[left][2], vertices[right][2]],
            color="#6df6ff",
            linewidth=1.5,
        )
    sc = ax.scatter(
        [coord[0] for coord in coords],
        [coord[1] for coord in coords],
        [coord[2] for coord in coords],
        c=[float(record["bpb"]) for record in records],
        cmap="magma_r",
        s=66,
        edgecolor="white",
        linewidth=0.6,
    )
    for record, coord in zip(records, coords):
        ax.text(coord[0], coord[1], coord[2] + 0.022, f"R{record['record_index']}:{record['branch_index']}", color="white", fontsize=6)
    for label, vertex in zip(spec["labels"], vertices):
        ax.text(vertex[0], vertex[1], vertex[2] + 0.045, label, color="#e8fbff", fontsize=8.5, ha="center")
    ax.set_title(spec["title"], color="white", pad=12)
    ax.set_axis_off()
    ax.view_init(elev=23, azim=42)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("BPB", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_interactive_tetrahedron(records: list[dict[str, Any]], spec: dict[str, Any], output_path: Path) -> None:
    vertices = [TETRAHEDRON_VERTICES[key] for key in ("a", "b", "c", "d")]
    coords = tetrahedron_coords(records, spec)
    payload = {
        "title": spec["title"],
        "labels": spec["labels"],
        "vertices": vertices,
        "coords": coords,
        "bpb": [float(record["bpb"]) for record in records],
        "hover": [
            (
                f"record=R{record['record_index']} branch={record['branch_index']}<br>"
                f"dataset={html.escape(str(record.get('dataset', '')))}<br>"
                f"BPB={record['bpb']:.4f}<br>K_hat(x|helpers)={record['k_proxy']:.2f}<br>"
                f"answer BPB={record['answer_bpb']:.4f}<br>MST={record['mst_efficiency']:.4f}"
            )
            for record in records
        ],
    }
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(spec['title'])}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script></head>
<body style="margin:0;background:#030712;color:white;font-family:system-ui">
<div id="plot" style="width:100vw;height:100vh"></div>
<script>
const payload = {json.dumps(payload)};
const edges = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]];
const traces = edges.map(([i,j]) => ({{
  type:'scatter3d', mode:'lines',
  x:[payload.vertices[i][0], payload.vertices[j][0]],
  y:[payload.vertices[i][1], payload.vertices[j][1]],
  z:[payload.vertices[i][2], payload.vertices[j][2]],
  line:{{color:'#6df6ff', width:4}}, hoverinfo:'skip', showlegend:false
}}));
traces.push({{
  type:'scatter3d', mode:'markers',
  x:payload.coords.map(p => p[0]), y:payload.coords.map(p => p[1]), z:payload.coords.map(p => p[2]),
  hovertext:payload.hover, hoverinfo:'text',
  marker:{{size:6, color:payload.bpb, colorscale:'Magma', reversescale:true, colorbar:{{title:'BPB'}}, line:{{color:'white', width:1}}}},
  showlegend:false
}});
traces.push({{
  type:'scatter3d', mode:'text',
  x:payload.vertices.map(p => p[0]), y:payload.vertices.map(p => p[1]), z:payload.vertices.map(p => p[2] + 0.05),
  text:payload.labels, textfont:{{color:'#e8fbff', size:13}}, hoverinfo:'skip', showlegend:false
}});
Plotly.newPlot('plot', traces, {{
  title:{{text:payload.title, font:{{color:'white'}}}},
  paper_bgcolor:'#030712', plot_bgcolor:'#030712',
  scene:{{xaxis:{{visible:false}}, yaxis:{{visible:false}}, zaxis:{{visible:false}}, bgcolor:'#030712'}},
  margin:{{l:0,r:0,b:0,t:48}}
}});
</script></body></html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def subsample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length)
    return np.linspace(0, length - 1, num=max_points).round().astype(int)


def _focus_transform_params(
    branches: list[dict[str, Any]],
    *,
    dims: int,
    max_points: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust center/scale for readable projected hidden-space plots."""

    chunks: list[np.ndarray] = []
    for branch in branches:
        path = np.asarray(branch.get("projected_path", np.zeros((0, dims))), dtype=float)
        if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] < dims:
            continue
        idx = subsample_indices(path.shape[0], max(8, max_points // max(1, len(branches))))
        chunks.append(path[idx, :dims])
    if not chunks:
        return np.zeros((dims,), dtype=float), np.ones((dims,), dtype=float)
    points = np.concatenate(chunks, axis=0)
    center = np.nanmedian(points, axis=0)
    q10 = np.nanpercentile(points, 10.0, axis=0)
    q90 = np.nanpercentile(points, 90.0, axis=0)
    scale = 0.5 * (q90 - q10)
    fallback = np.nanstd(points, axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(fallback > 1e-6, fallback, 1.0))
    return center.astype(float), scale.astype(float)


def _focus_project(points: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Asinh-compressed robust projection preserving local geometry visibility."""

    arr = np.asarray(points, dtype=float)
    dims = min(arr.shape[-1], center.shape[0])
    out = arr.copy()
    out[..., :dims] = np.arcsinh((arr[..., :dims] - center[:dims]) / np.maximum(scale[:dims], 1e-8))
    return out


def _triangle_long_edge_mask(
    triangles: np.ndarray,
    xs: list[float] | np.ndarray,
    ys: list[float] | np.ndarray,
    *,
    edge_quantile: float = 0.82,
) -> np.ndarray:
    """Mask projection-triangulation artifacts with very long edges."""

    tri = np.asarray(triangles, dtype=int)
    if tri.ndim != 2 or tri.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    pts = np.stack([x, y], axis=-1)
    pa = pts[tri[:, 0]]
    pb = pts[tri[:, 1]]
    pc = pts[tri[:, 2]]
    e01 = np.linalg.norm(pa - pb, axis=-1)
    e12 = np.linalg.norm(pb - pc, axis=-1)
    e20 = np.linalg.norm(pc - pa, axis=-1)
    all_edges = np.concatenate([e01, e12, e20])
    finite = all_edges[np.isfinite(all_edges) & (all_edges > 1e-10)]
    if finite.size == 0:
        return np.zeros((tri.shape[0],), dtype=bool)
    threshold = float(np.quantile(finite, min(0.98, max(0.50, edge_quantile))))
    max_edge = np.maximum.reduce([e01, e12, e20])
    area = 0.5 * np.abs((pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1]) - (pc[:, 0] - pa[:, 0]) * (pb[:, 1] - pa[:, 1]))
    mask = (max_edge > threshold) | (area < 1e-10)
    if np.count_nonzero(~mask) < min(12, tri.shape[0]):
        threshold = float(np.quantile(finite, 0.94))
        mask = (max_edge > threshold) | (area < 1e-10)
    return mask


def plot_trajectory_3d(record_meta: dict[str, Any], branches: list[dict[str, Any]], output_path: Path) -> None:
    fig = plt.figure(figsize=(10, 8), facecolor="#030712")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#030712")
    cmap = plt.get_cmap("turbo")
    bpbs = np.array([float(branch["bpb"]) for branch in branches], dtype=float)
    lo, hi = float(bpbs.min()), float(bpbs.max())
    best_index = int(np.argmin(bpbs))
    focus_center, focus_scale = _focus_transform_params(branches, dims=3, max_points=4096)
    for branch in branches:
        path = branch["projected_path"]
        idx = subsample_indices(path.shape[0], int(branch.get("max_plot_points", 180)))
        plotted = _focus_project(path[idx, :3], focus_center, focus_scale)
        norm = 0.5 if abs(hi - lo) < 1e-8 else (float(branch["bpb"]) - lo) / (hi - lo)
        color = cmap(1.0 - norm)
        ax.plot(plotted[:, 0], plotted[:, 1], plotted[:, 2], color=color, linewidth=1.4, alpha=0.78)
        ax.scatter(plotted[0, 0], plotted[0, 1], plotted[0, 2], s=28, color="#6df6ff", edgecolor="white", linewidth=0.4)
        marker = "*" if int(branch["branch_index"]) == best_index else "o"
        size = 115 if marker == "*" else 44
        ax.scatter(
            plotted[-1, 0],
            plotted[-1, 1],
            plotted[-1, 2],
            s=size,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        answer_steps = set(int(step) for step in branch.get("answer_steps", []))
        if answer_steps:
            answer_idx = [i for i, step in enumerate(idx.tolist()) if int(step) in answer_steps]
            if answer_idx:
                answer_points = plotted[answer_idx]
                ax.scatter(
                    answer_points[:, 0],
                    answer_points[:, 1],
                    answer_points[:, 2],
                    s=16,
                    color="#ffd166",
                    alpha=0.86,
                    edgecolor="none",
                )
    title = (
        f"Embedding-space GoT branches: R{record_meta['record_index']} "
        f"{record_meta.get('task_family', '')}"
    )
    ax.set_title(title, color="white", fontsize=13)
    ax.text2D(
        0.02,
        0.02,
        "focused coordinates: asinh((PC-median)/robust-scale) | cyan=start, gold=answer span, star=best terminal",
        transform=ax.transAxes,
        color="#e8fbff",
        fontsize=9,
    )
    ax.set_axis_off()
    ax.view_init(elev=24, azim=36)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_interactive_trajectory(record_meta: dict[str, Any], branches: list[dict[str, Any]], output_path: Path) -> None:
    traces: list[dict[str, Any]] = []
    if not branches:
        return
    bpbs = np.array([float(branch["bpb"]) for branch in branches], dtype=float)
    lo, hi = float(bpbs.min()), float(bpbs.max())
    ranked = sorted(branches, key=lambda item: float(item["bpb"]))
    best = ranked[0]
    focus_center, focus_scale = _focus_transform_params(branches, dims=3, max_points=4096)
    best_path = np.asarray(best.get("projected_path", np.zeros((0, 3))), dtype=float)
    if best_path.ndim == 2 and best_path.shape[0] >= 6:
        best_idx = subsample_indices(best_path.shape[0], min(150, int(best.get("max_plot_points", 180))))
        best_raw_points = best_path[best_idx, :3]
        best_points = _focus_project(best_raw_points, focus_center, focus_scale)
        active_faces = _sample_toric_shadow_vector(best, "active_faces", best_idx, default=0.0, integer=True)
        margins = _sample_toric_shadow_vector(best, "margins", best_idx, default=0.0)
        traces.extend(
            _clean_windowed_simplicial_traces(
                best_points,
                name_prefix="best-branch reasoning-step",
                windows=int(best.get("simplicial_windows", 6)),
                radius_quantiles=tuple(best.get("simplicial_radius_quantiles", (0.08, 0.14, 0.22))),
                default_level=str(best.get("simplicial_default_level", "default")),
                max_edges_per_window=int(best.get("simplicial_max_edges_per_window", 34)),
                max_triangles_per_window=int(best.get("simplicial_max_triangles_per_window", 12)),
                opacity=0.20,
            )
        )
        traces.extend(
            _projected_chamber_traces(
                best_points,
                active_faces,
                margins,
                best_idx,
                z_offset=0.0,
                name_prefix="projected toric/tropical",
            )
        )
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": best_points[:, 0].tolist(),
                "y": best_points[:, 1].tolist(),
                "z": best_points[:, 2].tolist(),
                "marker": {
                    "size": 4,
                    "color": active_faces.tolist(),
                    "colorscale": "Turbo",
                    "opacity": 0.78,
                    "line": {"color": "#06111f", "width": 0.7},
                    "colorbar": {"title": {"text": "active face", "font": {"color": "white"}}, "tickfont": {"color": "white"}},
                },
                "name": "best-branch active-face vertices",
                "hovertext": [
                    (
                        f"step={int(step)}<br>active face={int(face)}<br>tropical margin={float(margin):.4f}<br>"
                        f"raw PC=({raw[0]:.3f}, {raw[1]:.3f}, {raw[2]:.3f})"
                    )
                    for step, face, margin, raw in zip(
                        best_idx.tolist(),
                        active_faces.tolist(),
                        margins.tolist(),
                        best_raw_points.tolist(),
                        strict=True,
                    )
                ],
                "hoverinfo": "text",
            }
        )
    for branch in branches:
        path = np.asarray(branch["projected_path"], dtype=float)
        idx = subsample_indices(path.shape[0], int(branch.get("max_plot_points", 180)))
        raw_plotted = path[idx, :3]
        plotted = _focus_project(raw_plotted, focus_center, focus_scale)
        norm = 0.5 if abs(hi - lo) < 1e-8 else (float(branch["bpb"]) - lo) / (hi - lo)
        color = f"hsl({int(200 - 160 * norm)}, 95%, 58%)"
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "x": plotted[:, 0].tolist(),
                "y": plotted[:, 1].tolist(),
                "z": plotted[:, 2].tolist(),
                "line": {"color": color, "width": 4},
                "name": f"branch {branch['branch_index']} BPB={branch['bpb']:.3f}",
                "hoverinfo": "name",
            }
        )
        terminal = plotted[-1]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": [float(terminal[0])],
                "y": [float(terminal[1])],
                "z": [float(terminal[2])],
                "marker": {"size": 6, "color": color, "symbol": "diamond", "line": {"color": "white", "width": 1}},
                "name": f"terminal {branch['branch_index']}",
                "hovertext": (
                    f"branch={branch['branch_index']}<br>BPB={branch['bpb']:.4f}<br>"
                    f"answer BPB={branch['answer_bpb']:.4f}<br>K_hat(x|helpers)={branch['k_proxy']:.2f}"
                ),
                "hoverinfo": "text",
            }
        )
        answer_steps = set(int(step) for step in branch.get("answer_steps", []))
        if answer_steps:
            answer_idx = [i for i, step in enumerate(idx.tolist()) if int(step) in answer_steps]
            if answer_idx:
                answer_points = plotted[answer_idx]
                traces.append(
                    {
                        "type": "scatter3d",
                        "mode": "markers",
                        "x": answer_points[:, 0].tolist(),
                        "y": answer_points[:, 1].tolist(),
                        "z": answer_points[:, 2].tolist(),
                        "marker": {"size": 3, "color": "#ffd166"},
                        "name": f"answer span {branch['branch_index']}",
                        "hoverinfo": "name",
                    }
                )
    density_menu = _simplicial_density_menu(traces)
    payload = {
        "title": f"R{record_meta['record_index']} {record_meta.get('dataset', '')} / {record_meta.get('task_family', '')}",
        "subtitle": (
            "focused coordinates: asinh((PC-median)/robust-scale) | lines: random-order/GFlowNet branches | "
            "window-colored local VR complexes: best branch | gold diamonds: chamber crossings | magenta open circles: low tropical margins"
        ),
        "traces": traces,
        "updatemenus": density_menu,
    }
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Reasoning trajectory</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script></head>
<body style="margin:0;background:#030712;color:white;font-family:system-ui">
<div style="position:absolute;z-index:5;left:18px;top:12px;max-width:1040px;color:#e8fbff">
  <div style="font-size:20px;font-weight:650">{html.escape(payload['title'])}</div>
  <div style="font-size:12px;opacity:.84;margin-top:4px">{payload['subtitle']}</div>
</div>
<div id="plot" style="width:100vw;height:100vh"></div>
<script>
const payload = {json.dumps(payload)};
Plotly.newPlot('plot', payload.traces, {{
  paper_bgcolor:'#030712', plot_bgcolor:'#030712',
  scene:{{
    xaxis:{{title:'focused PC1', color:'#e8fbff', gridcolor:'#143344', zerolinecolor:'#2de2e6'}},
    yaxis:{{title:'focused PC2', color:'#e8fbff', gridcolor:'#143344', zerolinecolor:'#2de2e6'}},
    zaxis:{{title:'focused PC3', color:'#e8fbff', gridcolor:'#143344', zerolinecolor:'#ffd166'}},
    bgcolor:'#030712',
    aspectmode:'data',
    camera:{{eye:{{x:1.35,y:-1.45,z:1.05}}}}
  }},
  updatemenus: payload.updatemenus,
  legend:{{font:{{color:'white'}}, x:0.02, y:0.80, bgcolor:'rgba(3,7,18,0.48)'}},
  margin:{{l:0,r:0,b:0,t:0}}
}}, {{responsive:true, displaylogo:false}});
</script></body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def plot_phase_energy(record_meta: dict[str, Any], branches: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#030712")
    ax.set_facecolor("#030712")
    xs: list[float] = []
    ys: list[float] = []
    cs: list[float] = []
    for branch in branches:
        path = branch["projected_path"]
        if path.shape[0] < 3:
            continue
        velocity = np.diff(path[:, :2], axis=0)
        phi = np.arctan2(velocity[:-1, 1], velocity[:-1, 0])
        psi = np.arctan2(velocity[1:, 1], velocity[1:, 0])
        energy = np.asarray(branch["per_token_nll"], dtype=float)[1 : 1 + phi.shape[0]]
        xs.extend(phi.tolist())
        ys.extend(psi.tolist())
        cs.extend(energy.tolist())
    if xs:
        sc = ax.scatter(xs, ys, c=cs, s=5, cmap="magma", alpha=0.72, edgecolor="none")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("local NLL energy", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white")
    ax.axhline(0.0, color="#6df6ff", linewidth=0.6, alpha=0.55)
    ax.axvline(0.0, color="#6df6ff", linewidth=0.6, alpha=0.55)
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_xlabel("phi: projected velocity angle", color="white")
    ax.set_ylabel("psi: next-step projected velocity angle", color="white")
    ax.set_title(f"Ramachandran-style phase plot R{record_meta['record_index']}", color="white")
    ax.tick_params(colors="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_energy_landscape(record_meta: dict[str, Any], branches: list[dict[str, Any]], output_path: Path) -> None:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    focus_center, focus_scale = _focus_transform_params(branches, dims=2, max_points=4096)
    for branch in branches:
        path = branch["projected_path"]
        energy = np.asarray(branch["per_token_nll"], dtype=float)
        idx = subsample_indices(path.shape[0], 800)
        focused = _focus_project(path[idx, :2], focus_center, focus_scale)
        xs.extend(focused[:, 0].tolist())
        ys.extend(focused[:, 1].tolist())
        zs.extend(energy[idx].tolist())
    fig = plt.figure(figsize=(8.4, 7.2), facecolor="#030712")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#030712")
    if len(xs) > 12:
        x_arr = np.asarray(xs, dtype=float)
        y_arr = np.asarray(ys, dtype=float)
        z_arr = np.asarray(zs, dtype=float)
        try:
            surface = ax.plot_trisurf(
                x_arr,
                y_arr,
                z_arr,
                cmap="magma",
                linewidth=0.035,
                antialiased=True,
                alpha=0.82,
            )
            cbar = fig.colorbar(surface, ax=ax, fraction=0.04, pad=0.08)
        except Exception:
            scatter = ax.scatter(x_arr, y_arr, z_arr, c=z_arr, cmap="magma", s=7, alpha=0.82)
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.08)
    else:
        scatter = ax.scatter(xs, ys, zs, c=zs, cmap="magma", s=14)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.08)
    cbar.set_label("local NLL energy", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    for branch in branches:
        path = branch["projected_path"]
        energy = np.asarray(branch["per_token_nll"], dtype=float)
        step = max(1, path.shape[0] // 160)
        focused_path = _focus_project(path[::step, :2], focus_center, focus_scale)
        energy_path = energy[::step][: focused_path.shape[0]]
        ax.plot(focused_path[:, 0], focused_path[:, 1], energy_path, color="#6df6ff", alpha=0.35, linewidth=0.7)
    ax.set_xlabel("focused PC1", color="white")
    ax.set_ylabel("focused PC2", color="white")
    ax.set_zlabel("local NLL energy", color="white")
    ax.set_title(f"3D embedding energy landscape R{record_meta['record_index']}", color="white")
    ax.text(
        0.02,
        0.02,
        0.98,
        "asinh-focused PCA; local-NLL surface with branch overlays",
        transform=ax.transAxes,
        color="#e8fbff",
        fontsize=8,
    )
    ax.tick_params(colors="white")
    ax.view_init(elev=30, azim=-48)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_interactive_energy_landscape(
    record_meta: dict[str, Any],
    branches: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write a rotatable 3D embedding-energy landscape.

    The static landscape and this companion HTML both lift model-computed local
    NLL values into the z-axis over the first two hidden-state PCA coordinates.
    The mesh is built from sampled reasoning states, while each branch path is
    drawn on top of the surface so low-energy basins and high-energy ridges can
    be inspected by rotating the plot.
    """

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    hover: list[str] = []
    branch_paths: list[dict[str, Any]] = []
    bpbs = np.asarray([float(branch["bpb"]) for branch in branches], dtype=float)
    best_branch_index = int(np.argmin(bpbs)) if bpbs.size else -1
    bpb_lo = float(np.nanmin(bpbs)) if bpbs.size else 0.0
    bpb_hi = float(np.nanmax(bpbs)) if bpbs.size else 1.0
    focus_center, focus_scale = _focus_transform_params(branches, dims=2, max_points=4096)

    for branch in branches:
        path = np.asarray(branch.get("projected_path", np.zeros((0, 3))), dtype=float)
        energy = np.asarray(branch.get("per_token_nll", np.zeros((path.shape[0],))), dtype=float)
        if path.ndim != 2 or path.shape[0] < 3 or energy.shape[0] < path.shape[0]:
            continue
        idx = subsample_indices(path.shape[0], min(800, int(branch.get("max_plot_points", 800))))
        focused_path = _focus_project(path[:, :2], focus_center, focus_scale)
        sampled_path = focused_path[idx]
        raw_sampled_path = path[idx, :2]
        sampled_energy = energy[idx]
        branch_id = int(branch.get("branch_index", len(branch_paths)))
        branch_bpb = float(branch.get("bpb", 0.0))
        answer_bpb = float(branch.get("answer_bpb", branch_bpb))
        xs.extend(sampled_path[:, 0].tolist())
        ys.extend(sampled_path[:, 1].tolist())
        zs.extend(sampled_energy.tolist())
        hover.extend(
            [
                (
                    f"B{branch_id} step {int(step)}<br>local NLL={float(e):.4f}<br>BPB={branch_bpb:.4f}<br>"
                    f"answer BPB={answer_bpb:.4f}<br>raw PC1={raw[0]:.3f}<br>raw PC2={raw[1]:.3f}"
                )
                for step, e, raw in zip(idx.tolist(), sampled_energy.tolist(), raw_sampled_path.tolist(), strict=True)
            ]
        )
        branch_paths.append(
            {
                "branch_index": branch_id,
                "bpb": branch_bpb,
                "answer_bpb": answer_bpb,
                "x": sampled_path[:, 0].tolist(),
                "y": sampled_path[:, 1].tolist(),
                "z": (sampled_energy + 0.035).tolist(),
                "steps": idx.astype(int).tolist(),
                "is_best": branch_id == int(branches[best_branch_index].get("branch_index", best_branch_index))
                if best_branch_index >= 0
                else False,
            }
        )

    if len(xs) < 12:
        return

    # Triangulation can fail if many PCA samples are repeated or nearly
    # collinear.  Fall back to a point cloud in that case, but keep the same
    # interactive branch overlays.
    mesh_trace: dict[str, Any] | None = None
    try:
        tri = mtri.Triangulation(xs, ys)
        triangles = np.asarray(tri.triangles, dtype=int)
        tri_mask = _triangle_long_edge_mask(triangles, xs, ys, edge_quantile=0.68)
        triangles = triangles[~tri_mask]
        if triangles.size:
            mesh_trace = {
                "type": "mesh3d",
                "x": xs,
                "y": ys,
                "z": zs,
                "i": triangles[:, 0].tolist(),
                "j": triangles[:, 1].tolist(),
                "k": triangles[:, 2].tolist(),
                "intensity": zs,
                "colorscale": "Magma",
                "opacity": 0.78,
                "flatshading": False,
                "name": "local NLL energy surface",
                "hovertext": hover,
                "hoverinfo": "text",
                "colorbar": {"title": "local NLL energy"},
            }
    except Exception:
        mesh_trace = None

    traces: list[dict[str, Any]] = []
    if mesh_trace is not None:
        traces.append(mesh_trace)
    else:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": xs,
                "y": ys,
                "z": zs,
                "marker": {
                    "size": 3,
                    "color": zs,
                    "colorscale": "Magma",
                    "opacity": 0.78,
                    "colorbar": {"title": "local NLL energy"},
                },
                "hovertext": hover,
                "hoverinfo": "text",
                "name": "local NLL samples",
            }
        )

    if best_branch_index >= 0:
        best_branch = branches[best_branch_index]
        best_path = np.asarray(best_branch.get("projected_path", np.zeros((0, 3))), dtype=float)
        best_energy = np.asarray(best_branch.get("per_token_nll", np.zeros((best_path.shape[0],))), dtype=float)
        if best_path.ndim == 2 and best_path.shape[0] >= 6 and best_energy.shape[0] >= best_path.shape[0]:
            best_idx = subsample_indices(best_path.shape[0], min(150, int(best_branch.get("max_plot_points", 180))))
            best_focus = _focus_project(best_path[:, :2], focus_center, focus_scale)
            best_points = np.column_stack(
                [
                    best_focus[best_idx, 0],
                    best_focus[best_idx, 1],
                    best_energy[best_idx] + 0.055,
                ]
            )
            active_faces = _sample_toric_shadow_vector(best_branch, "active_faces", best_idx, default=0.0, integer=True)
            margins = _sample_toric_shadow_vector(best_branch, "margins", best_idx, default=0.0)
            traces.extend(
                _clean_windowed_simplicial_traces(
                    best_points,
                    name_prefix="energy-surface reasoning-step",
                    z_values=best_energy[best_idx],
                    windows=int(best_branch.get("simplicial_windows", 6)),
                    radius_quantiles=tuple(best_branch.get("simplicial_radius_quantiles", (0.08, 0.14, 0.22))),
                    default_level=str(best_branch.get("simplicial_default_level", "default")),
                    max_edges_per_window=int(best_branch.get("simplicial_max_edges_per_window", 34)),
                    max_triangles_per_window=int(best_branch.get("simplicial_max_triangles_per_window", 12)),
                    opacity=0.18,
                )
            )
            traces.extend(
                _projected_chamber_traces(
                    best_points,
                    active_faces,
                    margins,
                    best_idx,
                    z_offset=0.08,
                    name_prefix="energy toric/tropical",
                )
            )
            traces.extend(_newton_polytope_shadow_traces(best_branch.get("toric_shadow"), best_points, name_prefix="energy toric"))
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "markers",
                    "x": best_points[:, 0].tolist(),
                    "y": best_points[:, 1].tolist(),
                    "z": best_points[:, 2].tolist(),
                    "marker": {
                        "size": 4.2,
                        "color": active_faces.tolist(),
                        "colorscale": "Turbo",
                        "opacity": 0.86,
                        "line": {"color": "#06111f", "width": 0.8},
                        "colorbar": {"title": {"text": "active face", "font": {"color": "white"}}, "tickfont": {"color": "white"}},
                    },
                    "name": "energy-surface toric vertices",
                    "hovertext": [
                        f"step={int(step)}<br>active face={int(face)}<br>tropical margin={float(margin):.4f}<br>local NLL={float(e):.4f}"
                        for step, face, margin, e in zip(
                            best_idx.tolist(),
                            active_faces.tolist(),
                            margins.tolist(),
                            best_energy[best_idx].tolist(),
                            strict=True,
                        )
                    ],
                    "hoverinfo": "text",
                }
            )

    branch_palette = ["#38f2ff", "#ff4fd8", "#ffd166", "#8cff6a", "#ad7cff", "#ff7a45", "#7bdff2", "#f15bb5"]
    for rank, item in enumerate(sorted(branch_paths, key=lambda row: row["bpb"])):
        denom = max(1e-8, bpb_hi - bpb_lo)
        norm = 0.5 if abs(denom) < 1e-8 else (float(item["bpb"]) - bpb_lo) / denom
        color = branch_palette[rank % len(branch_palette)]
        width = 8 if item["is_best"] else max(2, int(5 - 2 * norm))
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines+markers" if item["is_best"] else "lines",
                "x": item["x"],
                "y": item["y"],
                "z": item["z"],
                "line": {"color": color, "width": width},
                "marker": {"size": 3.5, "color": color},
                "opacity": 0.96 if item["is_best"] else 0.36,
                "name": f"{'best ' if item['is_best'] else ''}B{item['branch_index']} BPB={item['bpb']:.3f}",
                "hovertext": [
                    f"B{item['branch_index']} step {int(step)}<br>BPB={item['bpb']:.4f}<br>answer BPB={item['answer_bpb']:.4f}"
                    for step in item["steps"]
                ],
                "hoverinfo": "text",
            }
        )
        if item["x"]:
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "markers",
                    "x": [item["x"][-1]],
                    "y": [item["y"][-1]],
                    "z": [item["z"][-1]],
                    "marker": {
                        "size": 8 if item["is_best"] else 5,
                        "color": color,
                        "symbol": "diamond",
                        "line": {"color": "white", "width": 1.2},
                    },
                    "name": f"terminal B{item['branch_index']}",
                    "hovertext": f"terminal B{item['branch_index']}<br>BPB={item['bpb']:.4f}<br>answer BPB={item['answer_bpb']:.4f}",
                    "hoverinfo": "text",
                    "showlegend": False,
                }
            )

    z_arr = np.asarray(zs, dtype=float)
    low_count = min(10, max(3, int(0.01 * z_arr.shape[0])))
    low_idx = np.argsort(z_arr)[:low_count]
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": [xs[int(i)] for i in low_idx],
            "y": [ys[int(i)] for i in low_idx],
            "z": [zs[int(i)] + 0.08 for i in low_idx],
            "marker": {"size": 6, "color": "#b9ff66", "symbol": "circle", "line": {"color": "white", "width": 1.0}},
            "name": "low-energy basin samples",
            "hovertext": [hover[int(i)] for i in low_idx],
            "hoverinfo": "text",
        }
    )

    density_menu = _simplicial_density_menu(traces)
    payload = {
        "title": (
            f"Interactive embedding energy landscape R{record_meta['record_index']} "
            f"{html.escape(str(record_meta.get('dataset', '')))} / {html.escape(str(record_meta.get('task_family', '')))}"
        ),
        "subtitle": (
            "Focused hidden-space coordinates use asinh((PC-median)/robust-scale); mesh triangles with long projection "
            "edges are masked. Branches, local VR complexes, chamber crossings, and low-margin wall samples are overlaid."
        ),
        "traces": traces,
        "updatemenus": density_menu,
    }
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Embedding energy landscape</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script></head>
<body style="margin:0;background:#030712;color:white;font-family:system-ui">
<div style="position:absolute;z-index:5;left:18px;top:12px;max-width:1040px;color:#e8fbff">
  <div style="font-size:20px;font-weight:650">{payload['title']}</div>
  <div style="font-size:12px;opacity:.84;margin-top:4px">{payload['subtitle']}</div>
</div>
<div id="plot" style="width:100vw;height:100vh"></div>
<script>
const payload = {json.dumps(payload)};
Plotly.newPlot('plot', payload.traces, {{
  paper_bgcolor:'#030712',
  plot_bgcolor:'#030712',
  scene:{{
    bgcolor:'#030712',
    xaxis:{{title:'focused PC1', color:'#e8fbff', gridcolor:'#143344', zerolinecolor:'#2de2e6'}},
    yaxis:{{title:'focused PC2', color:'#e8fbff', gridcolor:'#143344', zerolinecolor:'#2de2e6'}},
    zaxis:{{title:'local NLL energy', color:'#e8fbff', gridcolor:'#143344', zerolinecolor:'#ffd166'}},
    camera:{{eye:{{x:1.45, y:-1.65, z:1.20}}}}
  }},
  updatemenus: payload.updatemenus,
  legend:{{font:{{color:'white'}}, bgcolor:'rgba(3,7,18,0.58)'}},
  margin:{{l:0,r:0,b:0,t:0}},
  annotations:[{{
    text:'mesh: local NLL over hidden-state PCA | adjustable local VR complexes | gold=chamber crossings | magenta=open low-margin wall samples',
    x:0.02, y:0.02, xref:'paper', yref:'paper', showarrow:false,
    font:{{color:'#e8fbff', size:12}}, align:'left'
  }}]
}}, {{responsive:true}});
</script></body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def _rips_edges_and_triangles(
    points: np.ndarray,
    *,
    radius_quantile: float | None = None,
    max_edges: int = 520,
    max_triangles: int = 220,
) -> tuple[float, list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Build a capped Vietoris-Rips 1/2-complex on projected reasoning steps."""

    if points.ndim != 2 or points.shape[0] < 3:
        return 0.0, [], []
    diffs = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    nonzero = dist[dist > 1e-8]
    if nonzero.size == 0:
        return 0.0, [], []
    chosen_radius = float(np.quantile(nonzero, 0.14))
    edges_with_dist: list[tuple[float, int, int]] = []
    quantile_schedule = (float(radius_quantile),) if radius_quantile is not None else (0.10, 0.14, 0.18, 0.22, 0.28)
    for quantile in quantile_schedule:
        quantile = min(0.95, max(0.02, float(quantile)))
        radius = float(np.quantile(nonzero, quantile))
        candidates: list[tuple[float, int, int]] = []
        for i in range(points.shape[0]):
            for j in range(i + 1, points.shape[0]):
                if dist[i, j] <= radius:
                    candidates.append((float(dist[i, j]), i, j))
        if radius_quantile is not None or len(candidates) >= min(24, max(6, points.shape[0] // 3)) or quantile == 0.28:
            chosen_radius = radius
            edges_with_dist = candidates
            break
    edges_with_dist.sort(key=lambda item: item[0])
    edges = [(i, j) for _, i, j in edges_with_dist[:max_edges]]
    edge_set = set(edges)
    adjacency: dict[int, list[int]] = {i: [] for i in range(points.shape[0])}
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    triangles: list[tuple[int, int, int]] = []
    for i in range(points.shape[0]):
        nbrs = sorted(j for j in adjacency[i] if j > i)
        for a_pos in range(len(nbrs)):
            j = nbrs[a_pos]
            for k in nbrs[a_pos + 1 :]:
                if (min(j, k), max(j, k)) in edge_set:
                    triangles.append((i, j, k))
                    if len(triangles) >= max_triangles:
                        return chosen_radius, edges, triangles
    return chosen_radius, edges, triangles


def _sample_toric_shadow_vector(
    branch: dict[str, Any],
    key: str,
    idx: np.ndarray,
    *,
    default: float = 0.0,
    integer: bool = False,
) -> np.ndarray:
    """Align a sampled toric-shadow sequence to the plotted hidden states."""

    values = np.asarray(branch.get("toric_shadow", {}).get(key, []), dtype=float)
    if values.ndim != 1 or values.size == 0:
        fill = int(default) if integer else float(default)
        dtype = int if integer else float
        return np.full((idx.shape[0],), fill, dtype=dtype)
    if values.size >= idx.shape[0] and values.size < int(np.max(idx)) + 1:
        sampled = values[: idx.shape[0]]
    else:
        sampled = values[np.minimum(idx, values.size - 1)]
    return sampled.astype(int if integer else float)


def _clean_windowed_simplicial_traces(
    points: np.ndarray,
    *,
    name_prefix: str,
    z_values: np.ndarray | None = None,
    windows: int = 6,
    radius_quantiles: tuple[float, ...] = (0.08, 0.14, 0.22),
    default_level: str = "default",
    max_edges_per_window: int = 34,
    max_triangles_per_window: int = 12,
    opacity: float = 0.22,
) -> list[dict[str, Any]]:
    """Build legible local Rips traces instead of one dense global complex."""

    if points.ndim != 2 or points.shape[0] < 6:
        return []
    traces: list[dict[str, Any]] = []
    palette = ["#5efcff", "#a4ff5e", "#ffd166", "#ff6bcb", "#9b8cff", "#ff8f5e"]
    quantiles = tuple(float(q) for q in radius_quantiles if 0.0 < float(q) < 1.0) or (0.08, 0.14, 0.22)
    if len(quantiles) == 1:
        level_specs = [("default", quantiles[0])]
    else:
        ordered = sorted(quantiles)
        level_specs = [
            ("sparse", ordered[0]),
            ("default", ordered[len(ordered) // 2]),
            ("dense", ordered[-1]),
        ]
    default_level = default_level if default_level in {label for label, _ in level_specs} else "default"
    n = points.shape[0]
    window_size = max(6, int(math.ceil(n / max(1, windows))))
    for level_label, level_quantile in level_specs:
        visible = level_label == default_level
        for window_id, start in enumerate(range(0, n, window_size)):
            stop = min(n, start + window_size)
            if stop - start < 4:
                continue
            local = points[start:stop]
            radius, edges, triangles = _rips_edges_and_triangles(
                local,
                radius_quantile=level_quantile,
                max_edges=max_edges_per_window,
                max_triangles=max_triangles_per_window,
            )
            color = palette[window_id % len(palette)]
            density_meta = {
                "simplicial_density": level_label,
                "radius_quantile": level_quantile,
                "radius": radius,
            }
            if triangles:
                tri_i, tri_j, tri_k = zip(*triangles, strict=True)
                mesh: dict[str, Any] = {
                    "type": "mesh3d",
                    "x": local[:, 0].tolist(),
                    "y": local[:, 1].tolist(),
                    "z": local[:, 2].tolist(),
                    "i": list(tri_i),
                    "j": list(tri_j),
                    "k": list(tri_k),
                    "color": color,
                    "opacity": opacity,
                    "flatshading": False,
                    "showscale": False,
                    "name": f"{name_prefix} {level_label} 2-simplices W{window_id} q={level_quantile:.2f}",
                    "legendgroup": f"{name_prefix}-simplices-{level_label}",
                    "hoverinfo": "skip",
                    "visible": visible,
                    "meta": density_meta,
                }
                if z_values is not None and z_values.shape[0] >= stop:
                    local_z = z_values[start:stop]
                    mesh["intensity"] = local_z.tolist()
                    mesh["colorscale"] = [[0, "#2de2e6"], [0.5, "#ffd166"], [1, "#ff4fd8"]]
                traces.append(mesh)
            if edges:
                ex: list[float | None] = []
                ey: list[float | None] = []
                ez: list[float | None] = []
                for i, j in edges:
                    ex.extend([float(local[i, 0]), float(local[j, 0]), None])
                    ey.extend([float(local[i, 1]), float(local[j, 1]), None])
                    ez.extend([float(local[i, 2]), float(local[j, 2]), None])
                traces.append(
                    {
                        "type": "scatter3d",
                        "mode": "lines",
                        "x": ex,
                        "y": ey,
                        "z": ez,
                        "line": {"color": color, "width": 2},
                        "opacity": min(0.58, opacity + 0.18),
                        "name": f"{name_prefix} {level_label} 1-skeleton W{window_id}",
                        "legendgroup": f"{name_prefix}-simplices-{level_label}",
                        "hoverinfo": "skip",
                        "visible": visible,
                        "meta": density_meta,
                    }
                )
            centroid = np.mean(local, axis=0)
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "markers+text",
                    "x": [float(centroid[0])],
                    "y": [float(centroid[1])],
                    "z": [float(centroid[2])],
                    "marker": {"size": 5, "color": color, "symbol": "circle", "line": {"color": "white", "width": 0.6}},
                    "text": [f"W{window_id}"],
                    "textposition": "top center",
                    "textfont": {"color": "#e8fbff", "size": 9},
                    "name": f"{name_prefix} {level_label} local window W{window_id}",
                    "legendgroup": f"{name_prefix}-simplices-{level_label}",
                    "hovertext": (
                        f"local reasoning window W{window_id}<br>"
                        f"steps {start}-{stop - 1}<br>radius q={level_quantile:.2f}<br>radius={radius:.4f}"
                    ),
                    "hoverinfo": "text",
                    "visible": visible,
                    "meta": density_meta,
                    "showlegend": False,
                }
            )
    return traces


def _simplicial_density_menu(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    levels = ["sparse", "default", "dense"]
    present = {
        str(trace.get("meta", {}).get("simplicial_density"))
        for trace in traces
        if isinstance(trace.get("meta"), dict) and trace.get("meta", {}).get("simplicial_density")
    }
    levels = [level for level in levels if level in present]
    if len(levels) <= 1:
        return []
    buttons = []
    for level in levels:
        visible = []
        for trace in traces:
            density = trace.get("meta", {}).get("simplicial_density") if isinstance(trace.get("meta"), dict) else None
            visible.append(True if density is None else str(density) == level)
        buttons.append(
            {
                "label": f"simplicial {level}",
                "method": "update",
                "args": [{"visible": visible}],
            }
        )
    return [
        {
            "type": "buttons",
            "direction": "right",
            "x": 0.02,
            "y": 0.94,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(3,7,18,0.72)",
            "bordercolor": "#2de2e6",
            "font": {"color": "white", "size": 11},
            "buttons": buttons,
        }
    ]


def _projected_chamber_traces(
    points: np.ndarray,
    active_faces: np.ndarray,
    margins: np.ndarray,
    idx: np.ndarray,
    *,
    z_offset: float = 0.0,
    name_prefix: str = "toric/tropical",
) -> list[dict[str, Any]]:
    """Create empirical chamber, wall-crossing, and low-margin traces."""

    traces: list[dict[str, Any]] = []
    if points.ndim != 2 or points.shape[0] < 3:
        return traces
    active = active_faces.astype(int) if active_faces.size else np.zeros((points.shape[0],), dtype=int)
    margin = margins.astype(float) if margins.size else np.zeros((points.shape[0],), dtype=float)
    if active.shape[0] != points.shape[0]:
        active = np.resize(active, points.shape[0])
    if margin.shape[0] != points.shape[0]:
        margin = np.resize(margin, points.shape[0])
    lifted = points.copy()
    lifted[:, 2] = lifted[:, 2] + z_offset
    palette = ["#2de2e6", "#ff4fd8", "#ffd166", "#8cff6a", "#ad7cff", "#ff7a45", "#7bdff2", "#f15bb5", "#b9fbc0", "#fee440"]
    face_centroids: dict[int, np.ndarray] = {}
    face_counts: dict[int, int] = {}
    for face in np.unique(active):
        face_int = int(face)
        mask = active == face_int
        count = int(np.count_nonzero(mask))
        if count:
            face_centroids[face_int] = lifted[mask].mean(axis=0)
            face_counts[face_int] = count
    transitions = np.flatnonzero(np.diff(active) != 0) + 1 if active.size > 1 else np.asarray([], dtype=int)
    if transitions.size:
        trans = lifted[transitions]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": trans[:, 0].tolist(),
                "y": trans[:, 1].tolist(),
                "z": trans[:, 2].tolist(),
                "marker": {"size": 9, "color": "#ffd166", "symbol": "diamond", "line": {"color": "white", "width": 1.0}},
                "name": f"{name_prefix} chamber crossings",
                "hovertext": [
                    f"step {int(idx[pos])}<br>active face {int(active[pos - 1])} → {int(active[pos])}<br>tropical margin={float(margin[pos]):.4f}"
                    for pos in transitions.tolist()
                ],
                "hoverinfo": "text",
            }
        )
        extent = float(np.linalg.norm(np.ptp(lifted, axis=0))) if lifted.size else 1.0
        wall_radius = max(0.055, min(0.42, 0.050 * extent))
        wall_positions = transitions
        if wall_positions.size > 18:
            wall_positions = wall_positions[np.linspace(0, wall_positions.size - 1, num=18).round().astype(int)]
        for wall_rank, pos in enumerate(wall_positions.tolist()):
            prev_face = int(active[max(0, pos - 1)])
            next_face = int(active[pos])
            center = 0.5 * (lifted[max(0, pos - 1)] + lifted[pos])
            normal = face_centroids.get(next_face, lifted[pos]) - face_centroids.get(prev_face, lifted[max(0, pos - 1)])
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm < 1e-8:
                normal = lifted[pos] - lifted[max(0, pos - 1)]
                normal_norm = float(np.linalg.norm(normal))
            if normal_norm < 1e-8:
                continue
            normal = normal / normal_norm
            ref = np.asarray([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(normal, ref))) > 0.90:
                ref = np.asarray([0.0, 1.0, 0.0], dtype=float)
            u = np.cross(normal, ref)
            u = u / max(float(np.linalg.norm(u)), 1e-8)
            v = np.cross(normal, u)
            v = v / max(float(np.linalg.norm(v)), 1e-8)
            corners = np.stack(
                [
                    center - wall_radius * u - wall_radius * v,
                    center + wall_radius * u - wall_radius * v,
                    center + wall_radius * u + wall_radius * v,
                    center - wall_radius * u + wall_radius * v,
                ],
                axis=0,
            )
            traces.append(
                {
                    "type": "mesh3d",
                    "x": corners[:, 0].tolist(),
                    "y": corners[:, 1].tolist(),
                    "z": corners[:, 2].tolist(),
                    "i": [0, 0],
                    "j": [1, 2],
                    "k": [2, 3],
                    "color": "#ffd166" if wall_rank % 2 == 0 else "#ff4fd8",
                    "opacity": 0.18,
                    "name": f"{name_prefix} fitted wall {prev_face}->{next_face}",
                    "hovertext": (
                        f"empirical wall patch<br>step {int(idx[pos])}<br>"
                        f"active face {prev_face} → {next_face}<br>"
                        f"margin={float(margin[pos]):.4f}"
                    ),
                    "hoverinfo": "text",
                    "showlegend": wall_rank < 3,
                }
            )
    finite_margin = margin[np.isfinite(margin)]
    if finite_margin.size:
        low_threshold = float(np.quantile(finite_margin, 0.12))
        low_margin_idx = np.flatnonzero(margin <= low_threshold)
        if low_margin_idx.size:
            low_points = lifted[low_margin_idx]
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "markers",
                    "x": low_points[:, 0].tolist(),
                    "y": low_points[:, 1].tolist(),
                    "z": low_points[:, 2].tolist(),
                    "marker": {"size": 6, "color": "#ff4fd8", "symbol": "circle-open", "line": {"color": "#ff4fd8", "width": 2.0}},
                    "name": f"{name_prefix} low-margin tropical wall samples",
                    "hovertext": [
                        f"step {int(idx[pos])}<br>active face={int(active[pos])}<br>tropical margin={float(margin[pos]):.4f}"
                        for pos in low_margin_idx.tolist()
                    ],
                    "hoverinfo": "text",
                }
            )
    ranked_faces = sorted(face_counts, key=lambda face: face_counts[face], reverse=True)[:8]
    for face_rank, face in enumerate(ranked_faces):
        mask = active == int(face)
        if int(np.count_nonzero(mask)) < 4:
            continue
        cluster = lifted[mask]
        if cluster.shape[0] > 120:
            select = np.linspace(0, cluster.shape[0] - 1, num=120).round().astype(int)
            cluster = cluster[select]
        traces.append(
            {
                "type": "mesh3d",
                "x": cluster[:, 0].tolist(),
                "y": cluster[:, 1].tolist(),
                "z": cluster[:, 2].tolist(),
                "alphahull": 0,
                "color": palette[face_rank % len(palette)],
                "opacity": 0.105,
                "name": f"{name_prefix} chamber polytope face {int(face)}",
                "hovertext": (
                    f"empirical chamber polytope<br>active face={int(face)}<br>"
                    f"states={face_counts[int(face)]}<br>"
                    f"mean margin={float(np.mean(margin[mask])):.4f}"
                ),
                "hoverinfo": "text",
                "showlegend": face_rank < 6,
            }
        )
    centroid = lifted.mean(axis=0)
    ray_x: list[float | None] = []
    ray_y: list[float | None] = []
    ray_z: list[float | None] = []
    chamber_x: list[float] = []
    chamber_y: list[float] = []
    chamber_z: list[float] = []
    chamber_text: list[str] = []
    chamber_labels: list[str] = []
    for face in np.unique(active)[:12]:
        mask = active == int(face)
        if np.count_nonzero(mask) < 2:
            continue
        cluster = lifted[mask]
        endpoint = cluster.mean(axis=0)
        ray_x.extend([float(centroid[0]), float(endpoint[0]), None])
        ray_y.extend([float(centroid[1]), float(endpoint[1]), None])
        ray_z.extend([float(centroid[2]), float(endpoint[2]), None])
        chamber_x.append(float(endpoint[0]))
        chamber_y.append(float(endpoint[1]))
        chamber_z.append(float(endpoint[2]))
        chamber_text.append(
            f"active face {int(face)}<br>states={int(np.count_nonzero(mask))}<br>mean margin={float(np.mean(margin[mask])):.4f}"
        )
        chamber_labels.append(str(int(face)))
    if ray_x:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "x": ray_x,
                "y": ray_y,
                "z": ray_z,
                "line": {"color": "#ff4fd8", "width": 3},
                "opacity": 0.50,
                "name": f"{name_prefix} empirical normal-fan rays",
                "hoverinfo": "skip",
            }
        )
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers+text",
                "x": chamber_x,
                "y": chamber_y,
                "z": chamber_z,
                "text": chamber_labels,
                "textposition": "top center",
                "marker": {"size": 7, "color": "#2de2e6", "symbol": "square", "line": {"color": "white", "width": 0.8}},
                "textfont": {"color": "#e8fbff", "size": 10},
                "name": f"{name_prefix} chamber centroids",
                "hovertext": chamber_text,
                "hoverinfo": "text",
            }
        )
    return traces


def _newton_polytope_shadow_traces(
    shadow: dict[str, Any] | None,
    plotted: np.ndarray,
    *,
    name_prefix: str = "toric",
) -> list[dict[str, Any]]:
    """Create a compact Newton-polytope shadow from toric audit exponents."""

    if not shadow or plotted.ndim != 2 or plotted.shape[1] < 3:
        return []
    exponents = np.asarray(shadow.get("pseudo_exponents", []), dtype=float)
    if exponents.ndim != 2 or exponents.shape[0] < 4:
        return []
    exponents = exponents - exponents.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(exponents, full_matrices=False)
        coords = exponents @ vh[:3].T
    except np.linalg.LinAlgError:
        coords = exponents[:, :3]
        if coords.shape[1] < 3:
            coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    coords = coords - coords.mean(axis=0, keepdims=True)
    norm = float(np.max(np.linalg.norm(coords, axis=1))) if coords.size else 1.0
    coords = coords / max(norm, 1e-8)
    span = np.ptp(plotted, axis=0)
    scale = max(0.20, min(0.65, 0.16 * float(np.linalg.norm(span))))
    anchor = np.asarray(
        [
            float(np.nanmin(plotted[:, 0]) + 0.14 * max(span[0], 1e-6)),
            float(np.nanmax(plotted[:, 1]) - 0.14 * max(span[1], 1e-6)),
            float(np.nanmax(plotted[:, 2]) - 0.10 * max(span[2], 1e-6)),
        ],
        dtype=float,
    )
    coords = anchor[None, :] + scale * coords
    traces: list[dict[str, Any]] = [
        {
            "type": "mesh3d",
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "z": coords[:, 2].tolist(),
            "alphahull": 0,
            "color": "#2de2e6",
            "opacity": 0.16,
            "name": f"{name_prefix} Newton-polytope shadow",
            "hovertext": (
                "Newton-polytope shadow<br>"
                "vertices are deterministic pseudo-exponents used by the empirical toric fan audit"
            ),
            "hoverinfo": "text",
        },
        {
            "type": "scatter3d",
            "mode": "markers+text",
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "z": coords[:, 2].tolist(),
            "marker": {"size": 5, "color": "#ffd166", "line": {"color": "white", "width": 0.5}},
            "text": [f"m{j}" for j in range(coords.shape[0])],
            "textfont": {"color": "#e8fbff", "size": 8},
            "textposition": "top center",
            "name": f"{name_prefix} pseudo-exponent vertices",
            "hoverinfo": "text",
            "hovertext": [f"pseudo-exponent m{j}" for j in range(coords.shape[0])],
        },
    ]
    return traces


def write_interactive_projected_simplicial_toric_geometry(
    record_meta: dict[str, Any],
    branches: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write a 3D hidden-space trajectory with Rips simplices and toric chambers.

    This is the embedding-space companion to the torus-projected phase plot.  It
    keeps the actual model trajectory in the PCA projection used by the static
    graph-of-thought plots, then overlays the local Vietoris-Rips 1/2-complex
    built from the same reasoning-step embeddings.  Toric active faces color
    the vertices; chamber transitions and normal-fan rays expose where tropical
    or toric geometry changes along the trajectory.
    """

    usable = [
        branch
        for branch in branches
        if np.asarray(branch.get("projected_path", [])).ndim == 2
        and np.asarray(branch.get("projected_path", [])).shape[0] >= 4
    ]
    if not usable:
        return
    ranked = sorted(usable, key=lambda item: float(item["bpb"]))
    best = ranked[0]
    path = np.asarray(best["projected_path"], dtype=float)
    n_steps = path.shape[0]
    idx = subsample_indices(n_steps, min(180, int(best.get("max_plot_points", 180))))
    focus_center, focus_scale = _focus_transform_params(usable, dims=3, max_points=4096)
    raw_plotted = path[idx, :3]
    plotted = _focus_project(raw_plotted, focus_center, focus_scale)
    energy = np.asarray(best.get("per_token_nll", np.zeros((n_steps,))), dtype=float)
    plotted_energy = energy[np.minimum(idx, energy.shape[0] - 1)] if energy.size else np.zeros((idx.shape[0],), dtype=float)
    active_faces = _sample_toric_shadow_vector(best, "active_faces", idx, default=0.0, integer=True)
    margins = _sample_toric_shadow_vector(best, "margins", idx, default=0.0)
    chart_axis = np.asarray(best.get("graphcg_chart_axis", np.zeros((n_steps,))), dtype=float)
    chart_margin = np.asarray(best.get("graphcg_chart_margin", np.zeros((n_steps,))), dtype=float)
    plotted_axis = chart_axis[np.minimum(idx, chart_axis.shape[0] - 1)] if chart_axis.size else np.zeros((idx.shape[0],), dtype=float)
    plotted_chart_margin = (
        chart_margin[np.minimum(idx, chart_margin.shape[0] - 1)] if chart_margin.size else np.zeros((idx.shape[0],), dtype=float)
    )

    traces: list[dict[str, Any]] = []
    traces.extend(
        _clean_windowed_simplicial_traces(
            plotted,
            name_prefix="projected reasoning-step",
            z_values=plotted_energy,
            windows=int(best.get("simplicial_windows", 6)),
            radius_quantiles=tuple(best.get("simplicial_radius_quantiles", (0.08, 0.14, 0.22))),
            default_level=str(best.get("simplicial_default_level", "default")),
            max_edges_per_window=int(best.get("simplicial_max_edges_per_window", 34)),
            max_triangles_per_window=int(best.get("simplicial_max_triangles_per_window", 12)),
            opacity=0.20,
        )
    )

    branch_palette = ["#38f2ff", "#ff4fd8", "#ffd166", "#8cff6a", "#ad7cff", "#ff7a45", "#7bdff2", "#f15bb5"]
    for rank, branch in enumerate(ranked[:8]):
        branch_path = np.asarray(branch["projected_path"], dtype=float)
        bidx = subsample_indices(branch_path.shape[0], min(220, int(branch.get("max_plot_points", 180))))
        sampled = _focus_project(branch_path[bidx, :3], focus_center, focus_scale)
        color = branch_palette[rank % len(branch_palette)]
        is_best = branch is best
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "x": sampled[:, 0].tolist(),
                "y": sampled[:, 1].tolist(),
                "z": sampled[:, 2].tolist(),
                "line": {"color": color, "width": 7 if is_best else 2},
                "opacity": 0.94 if is_best else 0.28,
                "name": f"{'best ' if is_best else ''}B{branch['branch_index']} BPB={float(branch['bpb']):.3f}",
                "hoverinfo": "name",
            }
        )

    traces.extend(
        _projected_chamber_traces(
            plotted,
            active_faces,
            margins,
            idx,
            z_offset=0.0,
            name_prefix="projected toric/tropical",
        )
    )
    traces.extend(_newton_polytope_shadow_traces(best.get("toric_shadow"), plotted, name_prefix="projected toric"))

    margin_lo = float(np.nanmin(plotted_chart_margin)) if plotted_chart_margin.size else 0.0
    margin_hi = float(np.nanmax(plotted_chart_margin)) if plotted_chart_margin.size else 1.0
    marker_sizes = 3.5 + 8.5 * (plotted_chart_margin - margin_lo) / max(1e-8, margin_hi - margin_lo)
    hover = [
        (
            f"reasoning step={int(step)}<br>"
            f"toric active face={int(face)}<br>"
            f"active-face margin={float(margin):.4f}<br>"
            f"local NLL={float(e):.4f}<br>"
            f"GraphCG axis={int(axis)}<br>"
            f"chart margin={float(chart_m):.4f}<br>"
            f"raw PC=({float(raw[0]):.3f}, {float(raw[1]):.3f}, {float(raw[2]):.3f})"
        )
        for step, face, margin, e, axis, chart_m, raw in zip(
            idx.tolist(),
            active_faces.tolist(),
            margins.tolist(),
            plotted_energy.tolist(),
            plotted_axis.tolist(),
            plotted_chart_margin.tolist(),
            raw_plotted.tolist(),
            strict=True,
        )
    ]
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": plotted[:, 0].tolist(),
            "y": plotted[:, 1].tolist(),
            "z": plotted[:, 2].tolist(),
            "marker": {
                "size": marker_sizes.tolist(),
                "color": active_faces.tolist(),
                "colorscale": "Turbo",
                "opacity": 0.94,
                "line": {"color": "#06111f", "width": 1},
                "colorbar": {"title": {"text": "toric active face", "font": {"color": "white"}}, "tickfont": {"color": "white"}},
            },
            "text": hover,
            "hoverinfo": "text",
            "name": "reasoning-step vertices",
        }
    )
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": [float(plotted[0, 0]), float(plotted[-1, 0])],
            "y": [float(plotted[0, 1]), float(plotted[-1, 1])],
            "z": [float(plotted[0, 2]), float(plotted[-1, 2])],
            "marker": {"size": [9, 12], "color": ["#6df6ff", "#ffd166"], "symbol": ["circle", "diamond"], "line": {"color": "white", "width": 1}},
            "name": "start / terminal",
            "hoverinfo": "name",
        }
    )

    title = (
        f"Projected simplicial-toric reasoning geometry R{record_meta['record_index']} "
        f"{html.escape(str(record_meta.get('dataset', '')))} / {html.escape(str(record_meta.get('task_family', '')))}"
    )
    subtitle = (
        "Focused hidden-state coordinates use asinh((PC-median)/robust-scale). Translucent triangles and cyan edges "
        "are local step-level Vietoris-Rips complexes; translucent chamber hulls are empirical active-face polytopes; "
        "gold/magenta sheets are fitted wall patches; the inset hull is the Newton-polytope shadow used by the fan audit."
    )
    density_menu = _simplicial_density_menu(traces)
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Projected simplicial toric reasoning geometry</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script></head>
<body style="margin:0;background:#030712;color:white;font-family:system-ui">
<div style="position:absolute;z-index:5;left:18px;top:12px;max-width:1040px;color:#e8fbff">
  <div style="font-size:20px;font-weight:650">{title}</div>
  <div style="font-size:12px;opacity:.84;margin-top:4px">{subtitle}</div>
</div>
<div id="plot" style="width:100vw;height:100vh"></div>
<script>
const traces = {json.dumps(traces)};
Plotly.newPlot('plot', traces, {{
  paper_bgcolor:'#030712',
  plot_bgcolor:'#030712',
  scene:{{
    bgcolor:'#030712',
    xaxis:{{title:'focused PC1', color:'#e8fbff', gridcolor:'#143344', zerolinecolor:'#2de2e6'}},
    yaxis:{{title:'focused PC2', color:'#e8fbff', gridcolor:'#143344', zerolinecolor:'#2de2e6'}},
    zaxis:{{title:'focused PC3', color:'#e8fbff', gridcolor:'#143344', zerolinecolor:'#ffd166'}},
    aspectmode:'data',
    camera:{{eye:{{x:1.35,y:-1.45,z:1.05}}}}
  }},
  updatemenus: {json.dumps(density_menu)},
  legend:{{font:{{color:'white'}}, x:0.02, y:0.82, bgcolor:'rgba(3,7,18,0.48)'}},
  margin:{{l:0,r:0,b:0,t:0}}
}}, {{responsive:true, displaylogo:false}});
</script></body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def plot_toric_phase_simplicial_trajectory(
    record_meta: dict[str, Any],
    branches: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Render toric phase winding with local simplex edges and analogy maps."""

    if not branches:
        return
    best = min(branches, key=lambda item: float(item["bpb"]))
    torus = np.asarray(best.get("toric_torus_path", np.zeros((0, 3))), dtype=float)
    energy = np.asarray(best.get("per_token_nll", np.zeros((torus.shape[0],))), dtype=float)
    chart_axis = np.asarray(best.get("graphcg_chart_axis", np.zeros((torus.shape[0],))), dtype=float)
    chart_margin = np.asarray(best.get("graphcg_chart_margin", np.zeros((torus.shape[0],))), dtype=float)
    if torus.shape[0] < 4:
        return
    idx = subsample_indices(torus.shape[0], int(best.get("max_plot_points", 180)))
    plotted = torus[idx]
    plotted_energy = energy[idx]
    plotted_axis = chart_axis[idx]
    plotted_margin = chart_margin[idx]

    fig = plt.figure(figsize=(12.8, 9.2), facecolor="#030712")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#030712")
    u = np.linspace(0, 2 * np.pi, 72)
    v = np.linspace(0, 2 * np.pi, 24)
    uu, vv = np.meshgrid(u, v)
    R, r = 1.08, 0.32
    xx = (R + r * np.cos(vv)) * np.cos(uu)
    yy = (R + r * np.cos(vv)) * np.sin(uu)
    zz = r * np.sin(vv)
    ax.plot_surface(xx, yy, zz, color="#0a2635", alpha=0.16, linewidth=0, shade=True)
    ax.plot_wireframe(xx, yy, zz, rstride=4, cstride=8, color="#1ad7e8", alpha=0.08, linewidth=0.35)

    diffs = plotted[:, None, :] - plotted[None, :, :]
    chord = np.linalg.norm(diffs, axis=-1)
    local_edges: list[tuple[int, int]] = []
    window = min(10, max(5, plotted.shape[0] // 14))
    for start in range(0, plotted.shape[0], max(4, window // 2)):
        stop = min(plotted.shape[0], start + window)
        sub = chord[start:stop, start:stop]
        if sub.shape[0] < 4:
            continue
        threshold = float(np.quantile(sub[sub > 1e-8], 0.035)) if np.any(sub > 1e-8) else 0.0
        for i in range(start, stop):
            for j in range(i + 1, stop):
                if chord[i, j] <= threshold and len(local_edges) < 42:
                    local_edges.append((i, j))
    for i, j in local_edges:
        ax.plot(
            [plotted[i, 0], plotted[j, 0]],
            [plotted[i, 1], plotted[j, 1]],
            [plotted[i, 2], plotted[j, 2]],
            color="#6df6ff",
            alpha=0.08,
            linewidth=0.32,
        )

    for start in range(0, plotted.shape[0] - window, max(6, window)):
        source = plotted[start : start + window]
        target = plotted[start + window : start + 2 * window]
        if source.shape[0] < 3 or target.shape[0] < 3:
            continue
        c0 = source.mean(axis=0)
        c1 = target.mean(axis=0)
        delta = c1 - c0
        ax.quiver(
            c0[0],
            c0[1],
            c0[2],
            delta[0],
            delta[1],
            delta[2],
            color="#ff4fd8",
            linewidth=1.0,
            arrow_length_ratio=0.22,
            alpha=0.72,
        )

    line_color = "#50f5ff"
    ax.plot(plotted[:, 0], plotted[:, 1], plotted[:, 2], color=line_color, linewidth=1.7, alpha=0.86)
    size = 18.0 + 44.0 * (plotted_margin - np.nanmin(plotted_margin)) / max(
        1e-8,
        float(np.nanmax(plotted_margin) - np.nanmin(plotted_margin)),
    )
    sc = ax.scatter(
        plotted[:, 0],
        plotted[:, 1],
        plotted[:, 2],
        c=plotted_energy,
        s=size,
        cmap="magma",
        alpha=0.92,
        edgecolor="#06111f",
        linewidth=0.25,
    )
    for axis_id in np.unique(plotted_axis.astype(int))[:8]:
        mask = plotted_axis.astype(int) == int(axis_id)
        if np.count_nonzero(mask) < 3:
            continue
        points = plotted[mask]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=8, alpha=0.30, label=f"basis axis {axis_id}")
    ax.scatter(plotted[0, 0], plotted[0, 1], plotted[0, 2], s=85, color="#6df6ff", edgecolor="white", linewidth=0.8)
    ax.scatter(plotted[-1, 0], plotted[-1, 1], plotted[-1, 2], s=120, marker="*", color="#ffd166", edgecolor="white", linewidth=0.8)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label("local NLL / tropical energy proxy", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    ax.set_title(
        f"Toric irrational-phase simplicial trajectory R{record_meta['record_index']} B{best['branch_index']}",
        color="white",
        fontsize=13,
    )
    ax.text2D(
        0.02,
        0.025,
        "cyan line: phase-wound reasoning path | faint edges: local VR 1-skeleton | magenta arrows: soft analogy maps between windows | marker size: GraphCG chart margin",
        transform=ax.transAxes,
        color="#e8fbff",
        fontsize=8.5,
    )
    ax.set_axis_off()
    ax.view_init(elev=28, azim=38)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_interactive_toric_phase_simplicial_trajectory(
    record_meta: dict[str, Any],
    branches: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write an interactive torus-projected reasoning trajectory audit.

    The static PNG is useful for reports, but the toric path is genuinely a
    three-dimensional object: the dense phase winding, local Rips edges, and
    analogical transports are much easier to audit when the torus can be
    rotated.  This HTML artifact uses only the already-computed model
    quantities from the branch records.
    """

    usable = [
        branch
        for branch in branches
        if np.asarray(branch.get("toric_torus_path", [])).ndim == 2
        and np.asarray(branch.get("toric_torus_path", [])).shape[0] >= 4
    ]
    if not usable:
        return
    ranked = sorted(usable, key=lambda item: float(item["bpb"]))
    best = ranked[0]
    torus = np.asarray(best["toric_torus_path"], dtype=float)
    energy = np.asarray(best.get("per_token_nll", np.zeros((torus.shape[0],))), dtype=float)
    phase_u = np.asarray(best.get("toric_phase_u", np.zeros((torus.shape[0],))), dtype=float)
    phase_v = np.asarray(best.get("toric_phase_v", np.zeros((torus.shape[0],))), dtype=float)
    chart_axis = np.asarray(best.get("graphcg_chart_axis", np.zeros((torus.shape[0],))), dtype=float)
    chart_margin = np.asarray(best.get("graphcg_chart_margin", np.zeros((torus.shape[0],))), dtype=float)
    idx = subsample_indices(torus.shape[0], int(best.get("max_plot_points", 180)))
    plotted = torus[idx]
    plotted_energy = energy[idx] if energy.shape[0] >= torus.shape[0] else np.zeros((idx.shape[0],), dtype=float)
    plotted_u = phase_u[idx] if phase_u.shape[0] >= torus.shape[0] else np.zeros((idx.shape[0],), dtype=float)
    plotted_v = phase_v[idx] if phase_v.shape[0] >= torus.shape[0] else np.zeros((idx.shape[0],), dtype=float)
    plotted_axis = chart_axis[idx] if chart_axis.shape[0] >= torus.shape[0] else np.zeros((idx.shape[0],), dtype=float)
    plotted_margin = chart_margin[idx] if chart_margin.shape[0] >= torus.shape[0] else np.zeros((idx.shape[0],), dtype=float)

    traces: list[dict[str, Any]] = []
    uu, vv = np.meshgrid(np.linspace(0, 2 * np.pi, 64), np.linspace(0, 2 * np.pi, 28))
    major_radius, minor_radius = 1.08, 0.32
    xx = (major_radius + minor_radius * np.cos(vv)) * np.cos(uu)
    yy = (major_radius + minor_radius * np.cos(vv)) * np.sin(uu)
    zz = minor_radius * np.sin(vv)
    traces.append(
        {
            "type": "surface",
            "x": xx.tolist(),
            "y": yy.tolist(),
            "z": zz.tolist(),
            "colorscale": [[0, "#082033"], [1, "#0b3a4a"]],
            "showscale": False,
            "opacity": 0.20,
            "name": "commutative torus shadow",
            "hoverinfo": "skip",
        }
    )

    branch_colors = ["#38f2ff", "#ff4fd8", "#ffd166", "#8cff6a", "#ad7cff", "#ff7a45", "#7bdff2", "#f15bb5"]
    for rank, branch in enumerate(ranked[:8]):
        path = np.asarray(branch["toric_torus_path"], dtype=float)
        branch_idx = subsample_indices(path.shape[0], min(180, int(branch.get("max_plot_points", 180))))
        sampled = path[branch_idx]
        color = branch_colors[rank % len(branch_colors)]
        is_best = branch is best
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "x": sampled[:, 0].tolist(),
                "y": sampled[:, 1].tolist(),
                "z": sampled[:, 2].tolist(),
                "line": {"color": color, "width": 7 if is_best else 2},
                "opacity": 0.92 if is_best else 0.28,
                "name": f"{'best ' if is_best else ''}B{branch['branch_index']} BPB={float(branch['bpb']):.3f}",
                "hoverinfo": "name",
            }
        )

    diffs = plotted[:, None, :] - plotted[None, :, :]
    chord = np.linalg.norm(diffs, axis=-1)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    local_edges: list[tuple[int, int]] = []
    window = min(10, max(5, plotted.shape[0] // 14))
    default_level = str(best.get("simplicial_default_level", "sparse"))
    if default_level not in {"sparse", "default", "dense"}:
        default_level = "sparse"
    for level_label, edge_quantile, max_edges, opacity in (
        ("sparse", 0.035, 42, 0.10),
        ("default", 0.075, 80, 0.14),
        ("dense", 0.13, 145, 0.18),
    ):
        edge_x = []
        edge_y = []
        edge_z = []
        local_edges = []
        for start in range(0, plotted.shape[0], max(4, window // 2)):
            stop = min(plotted.shape[0], start + window)
            sub = chord[start:stop, start:stop]
            if sub.shape[0] < 4:
                continue
            threshold = float(np.quantile(sub[sub > 1e-8], edge_quantile)) if np.any(sub > 1e-8) else 0.0
            for i in range(start, stop):
                for j in range(i + 1, stop):
                    if chord[i, j] <= threshold and len(local_edges) < max_edges:
                        local_edges.append((i, j))
                        edge_x.extend([float(plotted[i, 0]), float(plotted[j, 0]), None])
                        edge_y.extend([float(plotted[i, 1]), float(plotted[j, 1]), None])
                        edge_z.extend([float(plotted[i, 2]), float(plotted[j, 2]), None])
        if edge_x:
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "x": edge_x,
                    "y": edge_y,
                    "z": edge_z,
                    "line": {"color": "#6df6ff", "width": 1},
                    "opacity": opacity,
                    "name": f"torus local VR 1-skeleton {level_label}",
                    "hoverinfo": "skip",
                    "visible": level_label == default_level,
                    "meta": {"simplicial_density": level_label, "radius_quantile": edge_quantile},
                }
            )

    arrow_x: list[float | None] = []
    arrow_y: list[float | None] = []
    arrow_z: list[float | None] = []
    cone_x: list[float] = []
    cone_y: list[float] = []
    cone_z: list[float] = []
    cone_u: list[float] = []
    cone_v: list[float] = []
    cone_w: list[float] = []
    for start in range(0, plotted.shape[0] - window, max(6, window)):
        source = plotted[start : start + window]
        target = plotted[start + window : start + 2 * window]
        if source.shape[0] < 3 or target.shape[0] < 3:
            continue
        c0 = source.mean(axis=0)
        c1 = target.mean(axis=0)
        delta = c1 - c0
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-8:
            continue
        arrow_x.extend([float(c0[0]), float(c1[0]), None])
        arrow_y.extend([float(c0[1]), float(c1[1]), None])
        arrow_z.extend([float(c0[2]), float(c1[2]), None])
        direction = delta / norm
        cone_x.append(float(c1[0]))
        cone_y.append(float(c1[1]))
        cone_z.append(float(c1[2]))
        cone_u.append(float(direction[0]))
        cone_v.append(float(direction[1]))
        cone_w.append(float(direction[2]))
    if arrow_x:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "x": arrow_x,
                "y": arrow_y,
                "z": arrow_z,
                "line": {"color": "#ff4fd8", "width": 4},
                "opacity": 0.62,
                "name": "analogical window transports",
                "hoverinfo": "skip",
            }
        )
        traces.append(
            {
                "type": "cone",
                "x": cone_x,
                "y": cone_y,
                "z": cone_z,
                "u": cone_u,
                "v": cone_v,
                "w": cone_w,
                "sizemode": "absolute",
                "sizeref": 0.045,
                "anchor": "tip",
                "colorscale": [[0, "#ff4fd8"], [1, "#ff4fd8"]],
                "showscale": False,
                "opacity": 0.72,
                "name": "transport arrowheads",
                "hoverinfo": "skip",
            }
        )

    if plotted_margin.size:
        margin_lo = float(np.nanmin(plotted_margin))
        margin_hi = float(np.nanmax(plotted_margin))
        marker_sizes = 3.5 + 8.0 * (plotted_margin - margin_lo) / max(1e-8, margin_hi - margin_lo)
    else:
        marker_sizes = np.full((plotted.shape[0],), 4.0)
    hover = [
        (
            f"substep={int(step)}<br>"
            f"kθ={float(u):.4f}<br>kβ={float(v):.4f}<br>"
            f"NLL={float(e):.4f}<br>GraphCG axis={int(axis)}<br>"
            f"chart margin={float(margin):.4f}"
        )
        for step, u, v, e, axis, margin in zip(idx, plotted_u, plotted_v, plotted_energy, plotted_axis, plotted_margin)
    ]
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": plotted[:, 0].tolist(),
            "y": plotted[:, 1].tolist(),
            "z": plotted[:, 2].tolist(),
            "marker": {
                "size": marker_sizes.tolist(),
                "color": plotted_energy.tolist(),
                "colorscale": "Magma",
                "opacity": 0.92,
                "line": {"color": "#06111f", "width": 1},
                "colorbar": {"title": {"text": "local NLL", "font": {"color": "white"}}, "tickfont": {"color": "white"}},
            },
            "text": hover,
            "hoverinfo": "text",
            "name": "best-branch phase samples",
        }
    )
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": [float(plotted[0, 0]), float(plotted[-1, 0])],
            "y": [float(plotted[0, 1]), float(plotted[-1, 1])],
            "z": [float(plotted[0, 2]), float(plotted[-1, 2])],
            "marker": {"size": [8, 11], "color": ["#6df6ff", "#ffd166"], "symbol": ["circle", "diamond"], "line": {"color": "white", "width": 1}},
            "name": "start / terminal",
            "hoverinfo": "name",
        }
    )

    payload = {
        "title": (
            f"Interactive toric reasoning trajectory R{record_meta['record_index']} "
            f"{html.escape(str(record_meta.get('dataset', '')))} / {html.escape(str(record_meta.get('task_family', '')))}"
        ),
        "subtitle": (
            "Commutative torus shadow of the noncommutative phase memory; branch paths follow projected irrational "
            "phase leaves. Local VR edges are adjustable; magenta arrows are analogical window transports."
        ),
        "traces": traces,
    }
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Interactive toric reasoning trajectory</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script></head>
<body style="margin:0;background:#030712;color:white;font-family:system-ui">
<div style="position:absolute;z-index:5;left:18px;top:12px;max-width:980px;color:#e8fbff">
  <div style="font-size:20px;font-weight:650">{payload['title']}</div>
  <div style="font-size:12px;opacity:.82;margin-top:4px">{payload['subtitle']}</div>
</div>
<div id="plot" style="width:100vw;height:100vh"></div>
<script>
const traces = {json.dumps(traces)};
Plotly.newPlot('plot', traces, {{
  paper_bgcolor:'#030712',
  plot_bgcolor:'#030712',
  scene:{{
    bgcolor:'#030712',
    xaxis:{{visible:false}},
    yaxis:{{visible:false}},
    zaxis:{{visible:false}},
    aspectmode:'data',
    camera:{{eye:{{x:1.45,y:1.45,z:0.95}}}}
  }},
  updatemenus: {json.dumps(_simplicial_density_menu(traces))},
  legend:{{font:{{color:'white'}}, x:0.02, y:0.82, bgcolor:'rgba(3,7,18,0.45)'}},
  margin:{{l:0,r:0,b:0,t:0}}
}}, {{responsive:true, displaylogo:false}});
</script></body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def _plot_wrapped_torus_square_path(
    ax: Any,
    phase_u: np.ndarray,
    phase_v: np.ndarray,
    *,
    color: str,
    alpha: float,
    linewidth: float,
    label: str | None = None,
) -> None:
    """Plot a path on [0,1)^2 without drawing false chords across wrap cuts."""

    if phase_u.size < 2 or phase_v.size < 2:
        return
    x: list[float] = [float(phase_u[0])]
    y: list[float] = [float(phase_v[0])]
    for i in range(1, phase_u.size):
        du = abs(float(phase_u[i]) - float(phase_u[i - 1]))
        dv = abs(float(phase_v[i]) - float(phase_v[i - 1]))
        if du > 0.5 or dv > 0.5:
            x.append(float("nan"))
            y.append(float("nan"))
        x.append(float(phase_u[i]))
        y.append(float(phase_v[i]))
    ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth, label=label)


def plot_toric_phase_winding_collection(
    record_meta: dict[str, Any],
    branches: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Render flat irrational phase winding and the embedded torus version together."""

    usable = [
        branch
        for branch in branches
        if np.asarray(branch.get("toric_phase_u", [])).size >= 4
        and np.asarray(branch.get("toric_phase_v", [])).size >= 4
        and np.asarray(branch.get("toric_torus_path", [])).ndim == 2
    ]
    if not usable:
        return
    ranked = sorted(usable, key=lambda item: float(item["bpb"]))
    best = ranked[0]
    phase_u = np.asarray(best["toric_phase_u"], dtype=float)
    phase_v = np.asarray(best["toric_phase_v"], dtype=float)
    cocycle = np.asarray(best.get("toric_phase_cocycle", np.zeros_like(phase_u)), dtype=float)
    torus = np.asarray(best["toric_torus_path"], dtype=float)
    energy = np.asarray(best.get("per_token_nll", np.zeros((phase_u.shape[0],))), dtype=float)
    chart_margin = np.asarray(best.get("graphcg_chart_margin", np.zeros((phase_u.shape[0],))), dtype=float)
    idx = subsample_indices(phase_u.shape[0], int(best.get("max_plot_points", 180)))
    u_best = phase_u[idx]
    v_best = phase_v[idx]
    cocycle_best = cocycle[idx]
    torus_best = torus[idx]
    energy_best = energy[idx] if energy.shape[0] >= phase_u.shape[0] else np.zeros_like(u_best)
    margin_best = chart_margin[idx] if chart_margin.shape[0] >= phase_u.shape[0] else np.zeros_like(u_best)

    fig = plt.figure(figsize=(17.5, 9.6), facecolor="#030712")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.02, 1.18], height_ratios=[1.0, 0.34], wspace=0.08, hspace=0.14)
    ax_flat = fig.add_subplot(grid[0, 0])
    ax_torus = fig.add_subplot(grid[0, 1], projection="3d")
    ax_phase = fig.add_subplot(grid[1, :])
    for ax in (ax_flat, ax_phase):
        ax.set_facecolor("#030712")
        ax.tick_params(colors="#dff8ff")
        for spine in ax.spines.values():
            spine.set_color("#18536a")
    ax_torus.set_facecolor("#030712")

    branch_colors = ["#38f2ff", "#ff4fd8", "#ffd166", "#8cff6a", "#ad7cff", "#ff7a45", "#7bdff2", "#f15bb5"]
    for b_index, branch in enumerate(ranked[:8]):
        u = np.asarray(branch["toric_phase_u"], dtype=float)
        v = np.asarray(branch["toric_phase_v"], dtype=float)
        branch_idx = subsample_indices(u.shape[0], min(160, int(branch.get("max_plot_points", 180))))
        color = branch_colors[b_index % len(branch_colors)]
        label = f"B{branch['branch_index']} BPB={float(branch['bpb']):.3f}" if b_index < 5 else None
        _plot_wrapped_torus_square_path(
            ax_flat,
            u[branch_idx],
            v[branch_idx],
            color=color,
            alpha=0.22 if branch is not best else 0.88,
            linewidth=0.75 if branch is not best else 1.7,
            label=label,
        )

    phase_points = np.stack([u_best, v_best], axis=-1)
    delta = np.abs(phase_points[:, None, :] - phase_points[None, :, :])
    periodic_delta = np.minimum(delta, 1.0 - delta)
    phase_chord = np.linalg.norm(periodic_delta, axis=-1)
    local_edges: list[tuple[int, int]] = []
    window = min(10, max(5, phase_points.shape[0] // 14))
    for start in range(0, phase_points.shape[0], max(4, window // 2)):
        stop = min(phase_points.shape[0], start + window)
        sub = phase_chord[start:stop, start:stop]
        if sub.shape[0] < 4:
            continue
        threshold = float(np.quantile(sub[sub > 1e-8], 0.035)) if np.any(sub > 1e-8) else 0.0
        for i in range(start, stop):
            for j in range(i + 1, stop):
                if phase_chord[i, j] <= threshold and len(local_edges) < 42:
                    local_edges.append((i, j))
    for i, j in local_edges:
        if abs(u_best[i] - u_best[j]) <= 0.5 and abs(v_best[i] - v_best[j]) <= 0.5:
            ax_flat.plot([u_best[i], u_best[j]], [v_best[i], v_best[j]], color="#b8fbff", alpha=0.08, linewidth=0.28)

    ax_flat.scatter(u_best, v_best, c=energy_best, s=18, cmap="magma", alpha=0.90, edgecolor="#06111f", linewidth=0.15)
    ax_flat.scatter(u_best[0], v_best[0], s=72, color="#6df6ff", edgecolor="white", linewidth=0.8, zorder=5)
    ax_flat.scatter(u_best[-1], v_best[-1], s=105, marker="*", color="#ffd166", edgecolor="white", linewidth=0.8, zorder=5)
    for x in np.linspace(0, 1, 5):
        ax_flat.axvline(x, color="#11384b", alpha=0.25, linewidth=0.5)
        ax_flat.axhline(x, color="#11384b", alpha=0.25, linewidth=0.5)
    ax_flat.set_xlim(0, 1)
    ax_flat.set_ylim(0, 1)
    ax_flat.set_aspect("equal", adjustable="box")
    ax_flat.set_xlabel(r"$k\theta\;\mathrm{mod}\;1$", color="#e8fbff")
    ax_flat.set_ylabel(r"$k\beta\;\mathrm{mod}\;1$", color="#e8fbff")
    ax_flat.set_title("Flat irrational phase winding on T^2", color="white", fontsize=13)
    ax_flat.legend(loc="upper right", fontsize=7, facecolor="#07111f", edgecolor="#18536a", labelcolor="#e8fbff")

    uu, vv = np.meshgrid(np.linspace(0, 2 * np.pi, 84), np.linspace(0, 2 * np.pi, 30))
    R, r = 1.08, 0.32
    xx = (R + r * np.cos(vv)) * np.cos(uu)
    yy = (R + r * np.cos(vv)) * np.sin(uu)
    zz = r * np.sin(vv)
    ax_torus.plot_surface(xx, yy, zz, color="#0a2635", alpha=0.16, linewidth=0, shade=True)
    ax_torus.plot_wireframe(xx, yy, zz, rstride=4, cstride=8, color="#1ad7e8", alpha=0.08, linewidth=0.35)
    for branch in ranked[:5]:
        tpath = np.asarray(branch["toric_torus_path"], dtype=float)
        branch_idx = subsample_indices(tpath.shape[0], min(160, int(branch.get("max_plot_points", 180))))
        color = branch_colors[int(branch["branch_index"]) % len(branch_colors)]
        ax_torus.plot(tpath[branch_idx, 0], tpath[branch_idx, 1], tpath[branch_idx, 2], color=color, alpha=0.18, linewidth=0.65)
    for i, j in local_edges:
        ax_torus.plot(
            [torus_best[i, 0], torus_best[j, 0]],
            [torus_best[i, 1], torus_best[j, 1]],
            [torus_best[i, 2], torus_best[j, 2]],
            color="#6df6ff",
            alpha=0.08,
            linewidth=0.28,
        )
    for start in range(0, torus_best.shape[0] - window, max(6, window)):
        source = torus_best[start : start + window]
        target = torus_best[start + window : start + 2 * window]
        if source.shape[0] < 3 or target.shape[0] < 3:
            continue
        c0 = source.mean(axis=0)
        c1 = target.mean(axis=0)
        d = c1 - c0
        ax_torus.quiver(c0[0], c0[1], c0[2], d[0], d[1], d[2], color="#ff4fd8", linewidth=1.0, arrow_length_ratio=0.22, alpha=0.72)
    sizes = 18.0 + 42.0 * (margin_best - np.nanmin(margin_best)) / max(1e-8, float(np.nanmax(margin_best) - np.nanmin(margin_best)))
    sc = ax_torus.scatter(
        torus_best[:, 0],
        torus_best[:, 1],
        torus_best[:, 2],
        c=energy_best,
        s=sizes,
        cmap="magma",
        alpha=0.93,
        edgecolor="#06111f",
        linewidth=0.18,
    )
    ax_torus.plot(torus_best[:, 0], torus_best[:, 1], torus_best[:, 2], color="#50f5ff", alpha=0.90, linewidth=1.55)
    ax_torus.scatter(torus_best[0, 0], torus_best[0, 1], torus_best[0, 2], s=80, color="#6df6ff", edgecolor="white", linewidth=0.8)
    ax_torus.scatter(torus_best[-1, 0], torus_best[-1, 1], torus_best[-1, 2], s=120, marker="*", color="#ffd166", edgecolor="white", linewidth=0.8)
    ax_torus.set_title("Embedded torus with local simplicial edges", color="white", fontsize=13)
    ax_torus.set_axis_off()
    ax_torus.view_init(elev=27, azim=40)
    cbar = fig.colorbar(sc, ax=ax_torus, shrink=0.62, pad=0.02)
    cbar.set_label("local NLL / energy proxy", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")

    t = np.arange(u_best.shape[0])
    ax_phase.plot(t, u_best, color="#38f2ff", linewidth=1.0, label=r"$k\theta$")
    ax_phase.plot(t, v_best, color="#ffd166", linewidth=1.0, label=r"$k\beta$")
    ax_phase.plot(t, cocycle_best, color="#ff4fd8", linewidth=1.0, label="quadratic cocycle")
    if energy_best.size:
        e = (energy_best - np.nanmin(energy_best)) / max(1e-8, float(np.nanmax(energy_best) - np.nanmin(energy_best)))
        ax_phase.fill_between(t, 0, e, color="#6df6ff", alpha=0.10, label="normalized NLL")
    ax_phase.set_ylim(-0.03, 1.03)
    ax_phase.set_xlabel("subsampled reasoning step", color="#e8fbff")
    ax_phase.set_title("Phase coordinates and noncommutative cocycle along the chosen branch", color="white", fontsize=11)
    ax_phase.legend(loc="upper right", ncol=4, fontsize=8, facecolor="#07111f", edgecolor="#18536a", labelcolor="#e8fbff")

    fig.suptitle(
        f"Irrational toric phase winding collection R{record_meta['record_index']} best B{best['branch_index']}",
        color="white",
        fontsize=16,
        y=0.985,
    )
    fig.text(
        0.015,
        0.012,
        "left: flat Kronecker winding on the commutative torus shadow | right: same points embedded on a torus surface | faint edges: local VR 1-skeleton | magenta arrows: analogical window transports",
        color="#e8fbff",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_toric_shadow_audit(
    record_meta: dict[str, Any],
    branches: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Render empirical toric fan, bend, and phase-leaf diagnostics."""

    if not branches:
        return
    best = min(branches, key=lambda item: float(item["bpb"]))
    shadow = best.get("toric_shadow")
    if not isinstance(shadow, dict):
        return
    active = np.asarray(shadow.get("active_faces", []), dtype=float)
    margins = np.asarray(shadow.get("margins", []), dtype=float)
    bends = np.asarray(shadow.get("bend_magnitudes", []), dtype=float)
    if active.size < 2:
        return
    branch_ids = np.asarray([float(branch["branch_index"]) for branch in branches], dtype=float)
    bpbs = np.asarray([float(branch["bpb"]) for branch in branches], dtype=float)
    occupied = np.asarray(
        [
            float(branch.get("toric_shadow", {}).get("occupied_fan_cells", 0.0))
            if isinstance(branch.get("toric_shadow"), dict)
            else 0.0
            for branch in branches
        ],
        dtype=float,
    )
    entropy = np.asarray(
        [
            float(branch.get("toric_shadow", {}).get("fan_cell_entropy", 0.0))
            if isinstance(branch.get("toric_shadow"), dict)
            else 0.0
            for branch in branches
        ],
        dtype=float,
    )
    mean_margin = np.asarray(
        [
            float(branch.get("toric_shadow", {}).get("mean_margin", 0.0))
            if isinstance(branch.get("toric_shadow"), dict)
            else 0.0
            for branch in branches
        ],
        dtype=float,
    )
    mean_bend = np.asarray(
        [
            float(branch.get("toric_shadow", {}).get("mean_bend", 0.0))
            if isinstance(branch.get("toric_shadow"), dict)
            else 0.0
            for branch in branches
        ],
        dtype=float,
    )
    leaf_residual = np.asarray(
        [float(branch.get("toric_geometry/toric_leaf_residual", np.nan)) for branch in branches],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4), facecolor="#030712")
    for ax in axes.reshape(-1):
        ax.set_facecolor("#030712")
        ax.tick_params(colors="#d7f7ff")
        for spine in ax.spines.values():
            spine.set_color("#164b63")

    x_active = np.arange(active.size)
    axes[0, 0].step(x_active, active, where="mid", color="#62f7ff", linewidth=1.5, alpha=0.9)
    axes[0, 0].scatter(x_active, active, c=margins[: active.size], cmap="viridis", s=18, alpha=0.92)
    axes[0, 0].set_title("Newton fan active cells along reasoning path", color="white", fontsize=11)
    axes[0, 0].set_xlabel("subsampled reasoning step", color="#d7f7ff")
    axes[0, 0].set_ylabel("active pseudo-face", color="#d7f7ff")

    x_margin = np.arange(margins.size)
    axes[0, 1].plot(x_margin, margins, color="#6df6ff", linewidth=1.3, label="active-face margin")
    if bends.size:
        bend_x = np.linspace(0, max(1, margins.size - 1), bends.size)
        axes[0, 1].plot(bend_x, bends, color="#ff4fd8", linewidth=1.1, alpha=0.84, label="bend magnitude")
    axes[0, 1].axhline(0.0, color="#ffffff", linewidth=0.5, alpha=0.25)
    axes[0, 1].set_title("Tropical margins and toric bends", color="white", fontsize=11)
    axes[0, 1].set_xlabel("subsampled reasoning step", color="#d7f7ff")
    axes[0, 1].legend(facecolor="#07111f", edgecolor="#164b63", labelcolor="white", fontsize=8)

    sc = axes[1, 0].scatter(
        occupied,
        mean_margin,
        c=bpbs,
        s=70 + 12 * np.maximum(0.0, entropy),
        cmap="magma_r",
        edgecolor="#e8fbff",
        linewidth=0.35,
        alpha=0.94,
    )
    for idx, branch_id in enumerate(branch_ids):
        axes[1, 0].text(occupied[idx], mean_margin[idx], f"B{int(branch_id)}", color="#d7f7ff", fontsize=7)
    axes[1, 0].set_title("Branch fan coverage vs. stability", color="white", fontsize=11)
    axes[1, 0].set_xlabel("occupied fan cells", color="#d7f7ff")
    axes[1, 0].set_ylabel("mean active-face margin", color="#d7f7ff")
    cbar = fig.colorbar(sc, ax=axes[1, 0], fraction=0.04, pad=0.02)
    cbar.set_label("BPB", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")

    axes[1, 1].plot(branch_ids, entropy, color="#62f7ff", marker="o", linewidth=1.2, label="fan entropy")
    axes[1, 1].plot(branch_ids, mean_bend, color="#ffd166", marker="s", linewidth=1.2, label="mean bend")
    if np.isfinite(leaf_residual).any():
        axes[1, 1].plot(
            branch_ids,
            np.nan_to_num(leaf_residual, nan=np.nanmean(leaf_residual)),
            color="#ff4fd8",
            marker="^",
            linewidth=1.2,
            label="phase-leaf residual",
        )
    axes[1, 1].set_title("Toric shadow summary by branch", color="white", fontsize=11)
    axes[1, 1].set_xlabel("branch", color="#d7f7ff")
    axes[1, 1].legend(facecolor="#07111f", edgecolor="#164b63", labelcolor="white", fontsize=8)

    fig.suptitle(
        f"Empirical toric shadow audit R{record_meta['record_index']} best B{best['branch_index']}",
        color="white",
        fontsize=14,
    )
    fig.text(
        0.012,
        0.015,
        "fan cells: fitted active Newton faces | margins: tropical stability | bends: Cartier-style piecewise-linear curvature | leaf residual: noncommutative phase-foliation consistency",
        color="#d7f7ff",
        fontsize=8.4,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_toric_slepian_audit(
    record_meta: dict[str, Any],
    branches: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Render DPSS/Slepian concentration of the noncommutative torus phase path."""

    if not branches:
        return
    best = min(branches, key=lambda item: float(item["bpb"]))
    eig = np.asarray(best.get("toric_slepian_eigenvalues", []), dtype=float)
    coeff = np.asarray(best.get("toric_slepian_coefficients", []), dtype=float)
    recon = np.asarray(best.get("toric_slepian_reconstruction", []), dtype=float)
    envelope = np.asarray(best.get("toric_slepian_envelope", []), dtype=float)
    energy = np.asarray(best.get("per_token_nll", []), dtype=float)
    phase_u = np.asarray(best.get("toric_phase_u", []), dtype=float)
    phase_v = np.asarray(best.get("toric_phase_v", []), dtype=float)
    if eig.size == 0 or phase_u.size < 4:
        return

    concentration = float(best.get("toric_slepian_concentration", 0.0))
    leakage = float(best.get("toric_slepian_leakage", 1.0))
    entropy = float(best.get("toric_slepian_mode_entropy", 0.0))
    branch_ids = np.asarray([float(branch["branch_index"]) for branch in branches], dtype=float)
    branch_conc = np.asarray([float(branch.get("toric_slepian_concentration", 0.0)) for branch in branches], dtype=float)
    branch_bpb = np.asarray([float(branch.get("bpb", 0.0)) for branch in branches], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4), facecolor="#030712")
    for ax in axes.reshape(-1):
        ax.set_facecolor("#030712")
        ax.tick_params(colors="#d7f7ff")
        for spine in ax.spines.values():
            spine.set_color("#164b63")

    modes = np.arange(eig.size)
    axes[0, 0].bar(modes, eig, color="#62f7ff", alpha=0.78, edgecolor="#e8fbff", linewidth=0.35)
    axes[0, 0].plot(modes, np.abs(coeff[: eig.size]), color="#ff4fd8", marker="o", linewidth=1.2, label="|coeff|")
    axes[0, 0].set_title("Toric Slepian concentration spectrum", color="white", fontsize=11)
    axes[0, 0].set_xlabel("DPSS mode", color="#d7f7ff")
    axes[0, 0].set_ylabel("eigenvalue / coefficient", color="#d7f7ff")
    axes[0, 0].legend(facecolor="#07111f", edgecolor="#164b63", labelcolor="white", fontsize=8)

    if energy.size:
        e = energy - float(np.nanmin(energy))
        axes[0, 1].plot(np.arange(energy.size), e / max(float(np.nanmax(e)), 1e-8), color="#ffd166", linewidth=1.0, label="NLL energy")
    if recon.size:
        r = recon - float(np.nanmin(recon))
        axes[0, 1].plot(np.arange(recon.size), r / max(float(np.nanmax(r)), 1e-8), color="#62f7ff", linewidth=1.3, label="Slepian reconstruction")
    if envelope.size:
        axes[0, 1].plot(np.arange(envelope.size), envelope, color="#ff4fd8", linewidth=1.2, alpha=0.84, label="phase envelope")
    axes[0, 1].set_title("Time-limited phase signal vs. energy", color="white", fontsize=11)
    axes[0, 1].set_xlabel("random-order reasoning step", color="#d7f7ff")
    axes[0, 1].legend(facecolor="#07111f", edgecolor="#164b63", labelcolor="white", fontsize=8)

    color = envelope if envelope.size == phase_u.size else energy[: phase_u.size] if energy.size >= phase_u.size else phase_u
    sc = axes[1, 0].scatter(
        phase_u,
        phase_v,
        c=color,
        cmap="viridis",
        s=16,
        alpha=0.90,
        edgecolor="#06111f",
        linewidth=0.12,
    )
    axes[1, 0].set_title("Projected Kronecker leaf colored by DPSS envelope", color="white", fontsize=11)
    axes[1, 0].set_xlabel(r"$e^{2\pi i k\theta}$ phase", color="#d7f7ff")
    axes[1, 0].set_ylabel(r"$e^{2\pi i k\beta}$ phase", color="#d7f7ff")
    cbar = fig.colorbar(sc, ax=axes[1, 0], fraction=0.04, pad=0.02)
    cbar.set_label("envelope", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")

    axes[1, 1].scatter(branch_bpb, branch_conc, c=branch_ids, cmap="magma", s=80, edgecolor="#e8fbff", linewidth=0.35)
    for idx, branch_id in enumerate(branch_ids):
        axes[1, 1].text(branch_bpb[idx], branch_conc[idx], f"B{int(branch_id)}", color="#d7f7ff", fontsize=7)
    axes[1, 1].set_title("Branch BPB vs. toric phase concentration", color="white", fontsize=11)
    axes[1, 1].set_xlabel("BPB", color="#d7f7ff")
    axes[1, 1].set_ylabel("concentration", color="#d7f7ff")
    axes[1, 1].axhline(np.nanmean(branch_conc), color="#6df6ff", linewidth=0.8, alpha=0.45)

    fig.suptitle(
        (
            f"Toric Slepian/PSWF audit R{record_meta['record_index']} B{best['branch_index']} "
            f"C={concentration:.3f}, leakage={leakage:.3f}, entropy={entropy:.3f}"
        ),
        color="white",
        fontsize=14,
    )
    fig.text(
        0.012,
        0.015,
        "DPSS modes are finite prolate-spheroidal analogues: high concentration means the projected noncommutative torus phase path has coherent band-limited structure; leakage flags diffuse or unstable reasoning-phase drift.",
        color="#d7f7ff",
        fontsize=8.4,
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12, top=0.89, wspace=0.26, hspace=0.36)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def directed_filtration_stats(
    hidden: np.ndarray,
    *,
    max_points: int = 64,
    levels: int = 4,
    radius_min: float = 0.55,
    radius_max: float = 1.65,
    skew_scale: float = 0.35,
    temperature: float = 0.12,
) -> dict[str, Any]:
    """Compute scale-normalized directed filtered-complex diagnostics.

    The vertices are local hidden states in reasoning-step windows.  Each
    window builds a radius-parametrized Vietoris--Rips/flag hierarchy, and a
    separate time/skew directed hierarchy records noncommutative flow.  The
    result keeps the historical keys used by downstream plots while adding
    Betti/cycle-rank, boundary, Dirichlet, simplex-tree, and HDBSCAN summaries.
    """

    return directed_step_filtration_stats_np(
        hidden,
        config=ReasoningTopologyConfig(
            max_points=max_points,
            max_windows=6,
            window_size=max(8, min(32, int(hidden.shape[0]) if hidden.ndim else 8)),
            step_stride=max(1, max_points // 8),
            levels=levels,
            radius_min=radius_min,
            radius_max=radius_max,
            skew_scale=skew_scale,
            temperature=temperature,
        ),
    )

    if hidden.shape[0] < 4:
        zeros = np.zeros(max(1, levels), dtype=float)
        return {
            "radii": np.linspace(radius_min, radius_max, max(1, levels)).tolist(),
            "edge_density": zeros.tolist(),
            "triangle_density": zeros.tolist(),
            "directed_edge_density": zeros.tolist(),
            "directed_asymmetry": zeros.tolist(),
            "directed_cycle_flux": zeros.tolist(),
            "directed_transitive_loss": zeros.tolist(),
            "directed_chain_commutator": zeros.tolist(),
            "inclusion_violation": zeros.tolist(),
            "hdbscan_cluster_count": zeros.tolist(),
            "hdbscan_noise_fraction": zeros.tolist(),
            "hdbscan_stability": zeros.tolist(),
            "hdbscan_persistent_edge_density": zeros.tolist(),
            "hdbscan_core_radius": 0.0,
            "skew_norm": 0.0,
            "distance": np.zeros((1, 1), dtype=float),
            "mutual_reachability": np.zeros((1, 1), dtype=float),
            "hdbscan_persistence_adjacency": np.zeros((1, 1), dtype=float),
            "skew": np.zeros((1, 1), dtype=float),
            "directed_adjacency": [],
        }
    arrows = np.diff(hidden.astype(np.float32), axis=0)
    if arrows.shape[0] > max_points:
        idx = np.linspace(0, arrows.shape[0] - 1, num=max_points).round().astype(int)
        arrows = arrows[idx]
    norms = np.linalg.norm(arrows, axis=1, keepdims=True)
    arrows = arrows / np.maximum(norms, 1e-8)
    diff = arrows[:, None, :] - arrows[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    positive = dist[dist > 1e-8]
    scale = float(np.median(positive)) if positive.size else 1.0
    dist = dist / max(scale, 1e-6)
    n = int(arrows.shape[0])
    eye = np.eye(n, dtype=float)
    half = max(1, arrows.shape[1] // 2)
    left = arrows[:, :half]
    right = arrows[:, half : half + half]
    if right.shape[1] < left.shape[1]:
        right = np.pad(right, ((0, 0), (0, left.shape[1] - right.shape[1])), mode="constant")
    skew = left @ right.T - right @ left.T
    skew_abs = np.abs(skew[np.triu_indices(n, k=1)])
    skew_unit = skew / max(float(np.median(skew_abs)) if skew_abs.size else 1.0, 1e-6)
    skew_unit = np.clip(skew_unit, -3.0, 3.0) * float(skew_scale)
    core_k = min(4, max(1, n - 1))
    masked_dist = dist + eye * 1.0e6
    core_radius = np.partition(masked_dist, kth=core_k - 1, axis=1)[:, core_k - 1]
    mutual_reachability = np.maximum(dist, np.maximum(core_radius[:, None], core_radius[None, :]))
    mutual_reachability = mutual_reachability + eye * 1.0e6
    radii = np.linspace(float(radius_min), float(radius_max), max(1, int(levels)))
    edge_density: list[float] = []
    triangle_density: list[float] = []
    directed_edge_density: list[float] = []
    directed_asymmetry: list[float] = []
    directed_cycle_flux: list[float] = []
    directed_transitive_loss: list[float] = []
    directed_chain_commutator: list[float] = []
    inclusion_violation: list[float] = []
    hdbscan_cluster_count: list[float] = []
    hdbscan_noise_fraction: list[float] = []
    hdbscan_stability: list[float] = []
    hdbscan_persistent_edge_density: list[float] = []
    directed_adjacency: list[np.ndarray] = []
    hdbscan_persistence = np.zeros((n, n), dtype=float)
    previous_sym: np.ndarray | None = None
    previous_dir: np.ndarray | None = None
    def _component_sizes(adjacency: np.ndarray) -> list[list[int]]:
        seen = np.zeros(adjacency.shape[0], dtype=bool)
        components: list[list[int]] = []
        for start in range(adjacency.shape[0]):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            component: list[int] = []
            while stack:
                node = stack.pop()
                component.append(node)
                for nxt in np.flatnonzero(adjacency[node]):
                    if not seen[nxt]:
                        seen[nxt] = True
                        stack.append(int(nxt))
            components.append(component)
        return components
    for radius in radii:
        sym = sigmoid_np((radius - dist) / max(float(temperature), 1e-6)) * (1.0 - eye)
        directed = sigmoid_np((radius - dist + skew_unit) / max(float(temperature), 1e-6)) * (1.0 - eye)
        density_adj = sigmoid_np((radius - mutual_reachability) / max(float(temperature), 1e-6)) * (1.0 - eye)
        hdbscan_persistence += density_adj / max(1, len(radii))
        hard_density = (mutual_reachability <= radius) & (~np.eye(n, dtype=bool))
        components = _component_sizes(hard_density | hard_density.T)
        stable_components = [component for component in components if len(component) >= 4]
        stable_nodes = set(node for component in stable_components for node in component)
        hdbscan_cluster_count.append(float(len(stable_components)))
        hdbscan_noise_fraction.append(float(1.0 - len(stable_nodes) / max(1, n)))
        hdbscan_stability.append(float(len(stable_nodes) / max(1, n)))
        hdbscan_persistent_edge_density.append(float(density_adj.sum() / max(1, n * (n - 1))))
        directed_adjacency.append(directed)
        edge_density.append(float(sym.sum() / max(1, n * (n - 1))))
        directed_edge_density.append(float(directed.sum() / max(1, n * (n - 1))))
        triangle_mass = np.einsum("ij,jk,ik->", sym, sym, sym)
        triangle_density.append(float(triangle_mass / max(1, n * (n - 1) * (n - 2))))
        directed_asymmetry.append(float(np.mean(np.abs(directed - directed.T))))
        flux = directed * np.maximum(0.0, skew_unit)
        directed_cycle_flux.append(float(np.einsum("ij,jk,ki->", flux, flux, flux) / max(1, n**3)))
        directed_square = directed @ directed
        directed_transitive_loss.append(float(np.mean(np.maximum(0.0, directed_square - directed) ** 2)))
        rows = directed + eye
        rows = rows / np.maximum(rows.sum(axis=1, keepdims=True), 1e-8)
        if previous_dir is None:
            directed_chain_commutator.append(0.0)
        else:
            prev_rows = previous_dir + eye
            prev_rows = prev_rows / np.maximum(prev_rows.sum(axis=1, keepdims=True), 1e-8)
            directed_chain_commutator.append(float(np.mean((prev_rows @ rows - rows @ prev_rows) ** 2)))
        if previous_sym is None:
            inclusion_violation.append(0.0)
        else:
            inclusion_violation.append(float(np.mean(np.maximum(0.0, previous_sym - sym) ** 2)))
        previous_sym = sym
        previous_dir = directed
    return {
        "radii": radii.tolist(),
        "edge_density": edge_density,
        "triangle_density": triangle_density,
        "directed_edge_density": directed_edge_density,
        "directed_asymmetry": directed_asymmetry,
        "directed_cycle_flux": directed_cycle_flux,
        "directed_transitive_loss": directed_transitive_loss,
        "directed_chain_commutator": directed_chain_commutator,
        "inclusion_violation": inclusion_violation,
        "hdbscan_cluster_count": hdbscan_cluster_count,
        "hdbscan_noise_fraction": hdbscan_noise_fraction,
        "hdbscan_stability": hdbscan_stability,
        "hdbscan_persistent_edge_density": hdbscan_persistent_edge_density,
        "hdbscan_core_radius": float(np.mean(core_radius)),
        "skew_norm": float(np.mean(np.abs(skew_unit))),
        "distance": dist,
        "mutual_reachability": mutual_reachability,
        "hdbscan_persistence_adjacency": hdbscan_persistence,
        "skew": skew_unit,
        "directed_adjacency": directed_adjacency,
    }


def attach_topology_stats(branch: dict[str, Any], stats: dict[str, Any]) -> None:
    branch["topology_edge_density"] = float(np.mean(stats["edge_density"]))
    branch["topology_triangle_density"] = float(np.mean(stats["triangle_density"]))
    branch["topology_directed_edge_density"] = float(np.mean(stats["directed_edge_density"]))
    branch["topology_directed_asymmetry"] = float(np.mean(stats["directed_asymmetry"]))
    branch["topology_directed_cycle_flux"] = float(np.mean(stats["directed_cycle_flux"]))
    branch["topology_directed_transitive_loss"] = float(np.mean(stats["directed_transitive_loss"]))
    branch["topology_directed_chain_commutator"] = float(np.mean(stats["directed_chain_commutator"]))
    branch["topology_inclusion_violation"] = float(np.mean(stats["inclusion_violation"]))
    branch["topology_boundary_residual"] = float(np.mean(stats.get("boundary_residual", [0.0])))
    branch["topology_dirichlet_energy"] = float(np.mean(stats.get("dirichlet_energy", [0.0])))
    branch["topology_dec_conservation_loss"] = float(np.mean(stats.get("dec_conservation_loss", [0.0])))
    branch["topology_dec_mass_residual"] = float(np.mean(stats.get("dec_mass_residual", [0.0])))
    branch["topology_dec_vorticity_drift"] = float(np.mean(stats.get("dec_vorticity_drift", [0.0])))
    branch["topology_dec_kinetic_energy"] = float(np.mean(stats.get("dec_kinetic_energy", [0.0])))
    branch["topology_dec_kinetic_energy_drift"] = float(np.mean(stats.get("dec_kinetic_energy_drift", [0.0])))
    branch["topology_dec_hodge_balance"] = float(np.mean(stats.get("dec_hodge_balance", [0.0])))
    branch["topology_dec_wedge_interior_residual"] = float(
        np.mean(stats.get("dec_wedge_interior_residual", [0.0]))
    )
    branch["topology_analogical_map_loss"] = float(stats.get("analogical_map_loss", 0.0))
    branch["topology_directed_map_loss"] = float(stats.get("directed_map_loss", 0.0))
    branch["topology_transport_entropy"] = float(stats.get("transport_entropy", 0.0))
    branch["topology_exact_h0_dim_mean"] = float(stats.get("exact_h0_dim_mean", 0.0))
    branch["topology_exact_h1_dim_mean"] = float(stats.get("exact_h1_dim_mean", 0.0))
    branch["topology_variety_complex_residual_mean"] = float(stats.get("variety_complex_residual_mean", 0.0))
    branch["topology_fitting_minor_rank_residual_mean"] = float(
        stats.get("fitting_minor_rank_residual_mean", 0.0)
    )
    branch["topology_buchsbaum_eisenbud_rank_residual_mean"] = float(
        stats.get("buchsbaum_eisenbud_rank_residual_mean", 0.0)
    )
    branch["topology_buchsbaum_eisenbud_multiplier_residual_mean"] = float(
        stats.get("buchsbaum_eisenbud_multiplier_residual_mean", 0.0)
    )
    branch["topology_multigraded_betti_mass_mean"] = float(stats.get("multigraded_betti_mass_mean", 0.0))
    branch["topology_exact_h0_map_rank_mean"] = float(stats.get("exact_h0_map_rank_mean", 0.0))
    branch["topology_exact_h1_map_rank_mean"] = float(stats.get("exact_h1_map_rank_mean", 0.0))
    branch["topology_exact_morphism_radius_shift_mean"] = float(stats.get("exact_morphism_radius_shift_mean", 0.0))
    branch["topology_exact_morphism_edge_validity_mean"] = float(
        stats.get("exact_morphism_edge_validity_mean", 0.0)
    )
    branch["topology_exact_morphism_triangle_validity_mean"] = float(
        stats.get("exact_morphism_triangle_validity_mean", 0.0)
    )
    branch["topology_exact_directed_edge_validity_mean"] = float(
        stats.get("exact_directed_edge_validity_mean", 0.0)
    )
    branch["topology_exact_morphism_computed"] = float(stats.get("exact_morphism_computed", 0.0))
    branch["topology_exact_morphism_truncated_complexes"] = float(
        stats.get("exact_morphism_truncated_complexes", 0.0)
    )
    branch["topology_betti0"] = float(np.mean(stats.get("betti0", [0.0])))
    branch["topology_cycle_rank"] = float(np.mean(stats.get("cycle_rank", [0.0])))
    branch["topology_hdbscan_cluster_count"] = float(np.mean(stats["hdbscan_cluster_count"]))
    branch["topology_hdbscan_noise_fraction"] = float(np.mean(stats["hdbscan_noise_fraction"]))
    branch["topology_hdbscan_stability"] = float(np.mean(stats["hdbscan_stability"]))
    branch["topology_hdbscan_persistent_edge_density"] = float(np.mean(stats["hdbscan_persistent_edge_density"]))
    branch["topology_hdbscan_core_radius"] = float(stats["hdbscan_core_radius"])
    branch["topology_skew_norm"] = float(stats["skew_norm"])


def plot_directed_filtration(record_meta: dict[str, Any], branches: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(6, 2, figsize=(12.5, 18.5), facecolor="#030712", constrained_layout=True)
    metrics = [
        ("edge_density", "symmetric edge density"),
        ("triangle_density", "soft triangle density"),
        ("betti0", "0D components"),
        ("cycle_rank", "1D cycle-rank proxy"),
        ("dirichlet_energy", "Dirichlet energy"),
        ("boundary_residual", "oriented boundary residual"),
        ("analogical_map_by_radius", "soft analogical map residual"),
        ("directed_map_by_radius", "directed map residual"),
        ("dec_conservation_loss", "DEC conservation residual"),
        ("dec_mass_residual", "DEC mass / divergence residual"),
        ("directed_asymmetry", "directed asymmetry"),
        ("directed_cycle_flux", "noncommutative cycle flux"),
    ]
    bpbs = np.array([float(branch["bpb"]) for branch in branches], dtype=float)
    lo, hi = float(bpbs.min()), float(bpbs.max())
    cmap = plt.get_cmap("turbo")
    for ax, (metric, title) in zip(axes.reshape(-1), metrics):
        ax.set_facecolor("#030712")
        for branch in branches:
            stats = branch.get("topology_stats")
            if not isinstance(stats, dict):
                continue
            norm = 0.5 if abs(hi - lo) < 1e-8 else (float(branch["bpb"]) - lo) / (hi - lo)
            color = cmap(1.0 - norm)
            ax.plot(
                stats["radii"],
                stats[metric],
                color=color,
                linewidth=1.45,
                alpha=0.84,
                label=f"B{branch['branch_index']} BPB={branch['bpb']:.2f}",
            )
        ax.set_title(title, color="white", fontsize=8.5, pad=5)
        ax.set_xlabel("filtration radius", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.grid(color="#1f3b52", linewidth=0.5, alpha=0.6)
    axes[0, 0].legend(loc="best", fontsize=7, framealpha=0.18, facecolor="#07111e", labelcolor="white")
    fig.suptitle(
        f"Directed nested simplicial diagnostics R{record_meta['record_index']}",
        color="white",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_topology_heatmaps(record_meta: dict[str, Any], branches: list[dict[str, Any]], output_path: Path) -> None:
    if not branches:
        return
    best = min(branches, key=lambda item: float(item["bpb"]))
    stats = best.get("topology_stats")
    if not isinstance(stats, dict):
        return
    directed = stats.get("directed_adjacency", [])
    if not directed:
        return
    mid = len(directed) // 2
    panels = [
        (np.asarray(stats["distance"]), "scale-normalized distance", "viridis"),
        (np.asarray(stats["mutual_reachability"]), "mutual reachability", "viridis"),
        (np.asarray(stats["hdbscan_persistence_adjacency"]), "density persistence adjacency", "magma"),
        (np.asarray(stats["skew"]), "antisymmetric toric skew", "coolwarm"),
        (np.asarray(directed[0]), f"directed adjacency r={stats['radii'][0]:.2f}", "viridis"),
        (np.asarray(directed[mid]), f"directed adjacency r={stats['radii'][mid]:.2f}", "viridis"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), facecolor="#030712")
    for ax, (matrix, title, cmap_name) in zip(axes.reshape(-1), panels):
        ax.set_facecolor("#030712")
        im = ax.imshow(matrix, cmap=cmap_name, interpolation="nearest", aspect="auto")
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white", fontsize=7)
    fig.suptitle(
        f"Best branch noncommutative simplex-tree map R{record_meta['record_index']} B{best['branch_index']}",
        color="white",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_step_radius_heatmaps(record_meta: dict[str, Any], branches: list[dict[str, Any]], output_path: Path) -> None:
    if not branches:
        return
    best = min(branches, key=lambda item: float(item["bpb"]))
    stats = best.get("topology_stats")
    if not isinstance(stats, dict):
        return
    heatmaps = stats.get("step_radius_heatmaps", {})
    if not isinstance(heatmaps, dict) or not heatmaps:
        return
    panels = [
        ("edge_density", "step x radius edge density", "viridis"),
        ("cycle_rank", "step x radius cycle rank", "magma"),
        ("betti0", "step x radius Betti-0", "cividis"),
        ("dec_conservation_loss", "DEC conservation residual", "inferno"),
        ("dec_kinetic_energy", "DEC kinetic energy", "turbo"),
        ("dec_wedge_interior_residual", "DEC wedge/interior residual", "plasma"),
        ("analogical_map_loss", "transition x radius map loss", "plasma"),
        ("directed_map_loss", "transition x radius directed map loss", "inferno"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 8.2), facecolor="#030712")
    for ax, (key, title, cmap_name) in zip(np.asarray(axes).reshape(-1), panels):
        ax.set_facecolor("#030712")
        matrix = np.asarray(heatmaps.get(key, np.zeros((1, len(stats.get("radii", [0.0]))))), dtype=float)
        if matrix.size == 0:
            matrix = np.zeros((1, len(stats.get("radii", [0.0]))), dtype=float)
        im = ax.imshow(matrix, cmap=cmap_name, aspect="auto", interpolation="nearest")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("radius level", color="white")
        ax.set_ylabel("reasoning window", color="white")
        ax.tick_params(colors="white", labelsize=7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white", fontsize=7)
    fig.suptitle(
        f"Step-local nested simplex hierarchy R{record_meta['record_index']} B{best['branch_index']}",
        color="white",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_exact_persistence_morphism_heatmaps(
    record_meta: dict[str, Any],
    branches: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not branches:
        return
    best = min(branches, key=lambda item: float(item["bpb"]))
    stats = best.get("topology_stats")
    if not isinstance(stats, dict):
        return
    heatmaps = stats.get("step_radius_heatmaps", {})
    panels = [
        ("exact_h0_dims", "exact $\\dim H_0(K_s(\\rho))$", "cividis"),
        ("exact_h1_dims", "exact $\\dim H_1(K_s(\\rho))$", "magma"),
        ("exact_h0_map_rank", "rank $H_0(P_s)$", "viridis"),
        ("exact_h1_map_rank", "rank $H_1(P_s)$", "plasma"),
        ("exact_radius_shift", "minimal radius shift", "inferno"),
        ("exact_edge_validity", "simplicial edge validity", "Greens"),
        ("exact_triangle_validity", "2-simplex validity", "Blues"),
        ("exact_directed_edge_validity", "directed edge validity", "Purples"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 8.2), facecolor="#030712")
    for ax, (key, title, cmap_name) in zip(axes.reshape(-1), panels):
        ax.set_facecolor("#030712")
        matrix = np.asarray(heatmaps.get(key, np.zeros((1, len(stats.get("radii", [0.0]))))), dtype=float)
        if matrix.size == 0:
            matrix = np.zeros((1, len(stats.get("radii", [0.0]))), dtype=float)
        im = ax.imshow(matrix, cmap=cmap_name, aspect="auto", interpolation="nearest")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("radius level", color="white")
        ax.set_ylabel("window / transition", color="white")
        ax.tick_params(colors="white", labelsize=7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white", fontsize=7)
    fig.suptitle(
        f"Exact F2 persistence-module morphism audit R{record_meta['record_index']} B{best['branch_index']}",
        color="white",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_commutative_algebra_audit(
    record_meta: dict[str, Any],
    branches: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not branches:
        return
    best = min(branches, key=lambda item: float(item["bpb"]))
    stats = best.get("topology_stats")
    if not isinstance(stats, dict):
        return
    heatmaps = stats.get("step_radius_heatmaps", {})
    panels = [
        ("variety_complex_residual", "variety of complexes residual $d_1d_2$", "inferno"),
        ("fitting_minor_rank_residual", "Fitting/minor rank residual", "magma"),
        ("buchsbaum_eisenbud_rank_residual", "Buchsbaum-Eisenbud rank residual", "plasma"),
        ("buchsbaum_eisenbud_multiplier_residual", "BE complementary-minor residual", "coolwarm"),
        ("multigraded_betti_mass", "multigraded Betti mass proxy", "cividis"),
        ("exact_h1_dims", "exact $\\dim H_1$ obstruction", "viridis"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.4), facecolor="#030712", constrained_layout=True)
    for ax, (key, title, cmap_name) in zip(axes.reshape(-1), panels):
        ax.set_facecolor("#030712")
        matrix = np.asarray(heatmaps.get(key, np.zeros((1, len(stats.get("radii", [0.0]))))), dtype=float)
        if matrix.size == 0:
            matrix = np.zeros((1, len(stats.get("radii", [0.0]))), dtype=float)
        im = ax.imshow(matrix, cmap=cmap_name, aspect="auto", interpolation="nearest")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("radius level", color="white", fontsize=8)
        ax.set_ylabel("window", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white", fontsize=7)
    fig.suptitle(
        f"Affine/Koszul commutative-algebra audit R{record_meta['record_index']} B{best['branch_index']}",
        color="white",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def serializable_record(record: dict[str, Any]) -> dict[str, Any]:
    skip = {
        "hidden",
        "per_token_nll",
        "projected_path",
        "target_positions",
        "topology_stats",
        "toric_torus_path",
        "toric_phase_u",
        "toric_phase_v",
        "toric_phase_cocycle",
        "toric_slepian_reconstruction",
        "toric_slepian_envelope",
        "graphcg_chart_axis",
        "graphcg_chart_margin",
    }
    out: dict[str, Any] = {}
    for key, value in record.items():
        if key in skip:
            continue
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def write_branch_records(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = [serializable_record(record) for record in records]
    (output_dir / "reasoning_geometry_records.json").write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
    keys = sorted({key for record in serializable for key in record if not isinstance(record.get(key), (list, dict))})
    with (output_dir / "reasoning_geometry_records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in serializable:
            writer.writerow({key: record.get(key) for key in keys})


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args)
    seq_len = min(int(args.seq_len), int(model.config.max_seq_len))
    if seq_len != args.seq_len:
        args.seq_len = seq_len
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    model.config = replace(model.config, dropout=0.0)
    records = select_reasoning_records(args, byte_offset=model.config.byte_offset)
    branch_records: list[dict[str, Any]] = []
    record_manifest: list[dict[str, Any]] = []
    use_amp = device.type == "cuda" and args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    start_time = time.perf_counter()
    for record_index, row in enumerate(records):
        token_info = token_window_for_record(row, args, byte_offset=model.config.byte_offset)
        tokens = torch.tensor([token_info["tokens"]], dtype=torch.long, device=device)
        sample_id = torch.tensor([args.seed * 10_000 + record_index], dtype=torch.long, device=device)
        row_id = safe_text(row.get("record_id")) or f"record-{record_index}"
        meta = {
            "record_index": record_index,
            "record_id": row_id,
            "dataset": safe_text(row.get("dataset")),
            "task_family": safe_text(row.get("task_family")),
            "language": safe_text(row.get("language")),
            "answer_span": token_info["answer_span"],
            "window_start": token_info["window_start"],
            "marker_preview": token_info["marker"][:256],
        }
        record_manifest.append(meta)
        branches: list[dict[str, Any]] = []
        print(f"evaluating R{record_index}: {meta['dataset']} / {meta['task_family']}", flush=True)
        for branch_index in range(args.branches):
            pass_id = args.seed + record_index * 100_003 + branch_index * 1_009
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                branch = evaluate_branch(
                    model=model,
                    tokens=tokens,
                    sample_id=sample_id,
                    pass_id=pass_id,
                    branch_index=branch_index,
                    answer_span=token_info["answer_span"],
                    args=args,
                )
            branch.update(meta)
            branch["max_plot_points"] = int(args.max_plot_points)
            branches.append(branch)
        projected_paths, _, _ = hidden_pca([branch["hidden"] for branch in branches], max_points=args.max_pca_points, seed=args.seed + record_index)
        for branch, projected in zip(branches, projected_paths):
            branch["projected_path"] = projected
            topology_stats = directed_filtration_stats(branch["hidden"])
            branch["topology_stats"] = topology_stats
            attach_topology_stats(branch, topology_stats)
            branch["simplicial_windows"] = int(args.simplicial_windows)
            branch["simplicial_radius_quantiles"] = parse_float_csv(
                str(args.simplicial_radius_quantiles),
                fallback=(0.08, 0.14, 0.22),
            )
            branch["simplicial_default_level"] = str(args.simplicial_default_level)
            branch["simplicial_max_edges_per_window"] = int(args.simplicial_max_edges_per_window)
            branch["simplicial_max_triangles_per_window"] = int(args.simplicial_max_triangles_per_window)
        enrich_branch_scores(branches)
        branch_records.extend(branches)
        record_slug = f"R{record_index}_{slug(meta['dataset'] + '_' + meta['task_family'])}"
        traj_dir = output_dir / "trajectories"
        topology_dir = output_dir / "topology"
        plot_trajectory_3d(meta, branches, traj_dir / f"{record_slug}_trajectory_3d.png")
        write_interactive_trajectory(meta, branches, traj_dir / f"{record_slug}_trajectory_3d.html")
        plot_phase_energy(meta, branches, traj_dir / f"{record_slug}_phase_energy.png")
        plot_energy_landscape(meta, branches, traj_dir / f"{record_slug}_energy_landscape.png")
        write_interactive_energy_landscape(meta, branches, traj_dir / f"{record_slug}_energy_landscape.html")
        plot_toric_phase_simplicial_trajectory(
            meta,
            branches,
            traj_dir / f"{record_slug}_toric_phase_simplicial_trajectory.png",
        )
        write_interactive_toric_phase_simplicial_trajectory(
            meta,
            branches,
            traj_dir / f"{record_slug}_toric_phase_simplicial_trajectory.html",
        )
        write_interactive_projected_simplicial_toric_geometry(
            meta,
            branches,
            traj_dir / f"{record_slug}_projected_simplicial_toric_geometry.html",
        )
        plot_toric_phase_winding_collection(
            meta,
            branches,
            traj_dir / f"{record_slug}_toric_phase_winding_collection.png",
        )
        plot_toric_shadow_audit(meta, branches, topology_dir / f"{record_slug}_toric_shadow_audit.png")
        plot_toric_slepian_audit(meta, branches, topology_dir / f"{record_slug}_toric_slepian_audit.png")
        plot_directed_filtration(meta, branches, topology_dir / f"{record_slug}_directed_filtration.png")
        plot_topology_heatmaps(meta, branches, topology_dir / f"{record_slug}_noncommutative_heatmaps.png")
        plot_step_radius_heatmaps(meta, branches, topology_dir / f"{record_slug}_step_radius_hierarchy.png")
        plot_exact_persistence_morphism_heatmaps(
            meta,
            branches,
            topology_dir / f"{record_slug}_exact_persistence_morphisms.png",
        )
        plot_commutative_algebra_audit(
            meta,
            branches,
            topology_dir / f"{record_slug}_commutative_algebra_audit.png",
        )
    enrich_branch_scores(branch_records)
    write_branch_records(branch_records, output_dir)
    (output_dir / "selected_records.json").write_text(json.dumps(record_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    triangle_dir = output_dir / "triangles"
    tetra_dir = output_dir / "tetrahedra"
    for name, spec in TRIANGLE_SPECS.items():
        plot_triangle(branch_records, spec, triangle_dir / f"{name}.png")
    for name, spec in TETRAHEDRON_SPECS.items():
        plot_tetrahedron(branch_records, spec, tetra_dir / f"{name}.png")
        write_interactive_tetrahedron(branch_records, spec, tetra_dir / f"{name}.html")

    def mean_record_key(key: str, default: float = 0.0) -> float:
        values = [float(record.get(key, default)) for record in branch_records]
        return float(np.mean(values)) if values else float(default)

    def mean_shadow_key(key: str, default: float = 0.0) -> float:
        values = [
            float(record.get("toric_shadow", {}).get(key, default))
            for record in branch_records
            if isinstance(record.get("toric_shadow"), dict)
        ]
        return float(np.mean(values)) if values else float(default)

    summary = {
        "checkpoint": args.checkpoint,
        "records": len(records),
        "branches": len(branch_records),
        "elapsed_seconds": time.perf_counter() - start_time,
        "mean_bpb": float(np.mean([record["bpb"] for record in branch_records])),
        "best_bpb": float(np.min([record["bpb"] for record in branch_records])),
        "mean_answer_bpb": float(np.mean([record["answer_bpb"] for record in branch_records])),
        "best_answer_bpb": float(np.min([record["answer_bpb"] for record in branch_records])),
        "mean_mst_efficiency": float(np.mean([record["mst_efficiency"] for record in branch_records])),
        "mean_path_smoothness": float(np.mean([record["path_smoothness_raw"] for record in branch_records])),
        "mean_topology_directed_asymmetry": float(np.mean([record["topology_directed_asymmetry"] for record in branch_records])),
        "mean_topology_directed_cycle_flux": float(np.mean([record["topology_directed_cycle_flux"] for record in branch_records])),
        "mean_topology_triangle_density": float(np.mean([record["topology_triangle_density"] for record in branch_records])),
        "mean_topology_betti0": float(np.mean([record["topology_betti0"] for record in branch_records])),
        "mean_topology_cycle_rank": float(np.mean([record["topology_cycle_rank"] for record in branch_records])),
        "mean_topology_boundary_residual": float(np.mean([record["topology_boundary_residual"] for record in branch_records])),
        "mean_topology_dirichlet_energy": float(np.mean([record["topology_dirichlet_energy"] for record in branch_records])),
        "mean_topology_dec_conservation_loss": float(
            np.mean([record["topology_dec_conservation_loss"] for record in branch_records])
        ),
        "mean_topology_dec_mass_residual": float(
            np.mean([record["topology_dec_mass_residual"] for record in branch_records])
        ),
        "mean_topology_dec_vorticity_drift": float(
            np.mean([record["topology_dec_vorticity_drift"] for record in branch_records])
        ),
        "mean_topology_dec_kinetic_energy": float(
            np.mean([record["topology_dec_kinetic_energy"] for record in branch_records])
        ),
        "mean_topology_dec_kinetic_energy_drift": float(
            np.mean([record["topology_dec_kinetic_energy_drift"] for record in branch_records])
        ),
        "mean_topology_dec_hodge_balance": float(
            np.mean([record["topology_dec_hodge_balance"] for record in branch_records])
        ),
        "mean_topology_dec_wedge_interior_residual": float(
            np.mean([record["topology_dec_wedge_interior_residual"] for record in branch_records])
        ),
        "mean_topology_analogical_map_loss": float(np.mean([record["topology_analogical_map_loss"] for record in branch_records])),
        "mean_topology_directed_map_loss": float(np.mean([record["topology_directed_map_loss"] for record in branch_records])),
        "mean_topology_transport_entropy": float(np.mean([record["topology_transport_entropy"] for record in branch_records])),
        "mean_topology_exact_h0_dim": float(np.mean([record["topology_exact_h0_dim_mean"] for record in branch_records])),
        "mean_topology_exact_h1_dim": float(np.mean([record["topology_exact_h1_dim_mean"] for record in branch_records])),
        "mean_topology_variety_complex_residual": float(
            np.mean([record["topology_variety_complex_residual_mean"] for record in branch_records])
        ),
        "mean_topology_fitting_minor_rank_residual": float(
            np.mean([record["topology_fitting_minor_rank_residual_mean"] for record in branch_records])
        ),
        "mean_topology_buchsbaum_eisenbud_rank_residual": float(
            np.mean([record["topology_buchsbaum_eisenbud_rank_residual_mean"] for record in branch_records])
        ),
        "mean_topology_buchsbaum_eisenbud_multiplier_residual": float(
            np.mean([record["topology_buchsbaum_eisenbud_multiplier_residual_mean"] for record in branch_records])
        ),
        "mean_topology_multigraded_betti_mass": float(
            np.mean([record["topology_multigraded_betti_mass_mean"] for record in branch_records])
        ),
        "mean_topology_exact_h0_map_rank": float(
            np.mean([record["topology_exact_h0_map_rank_mean"] for record in branch_records])
        ),
        "mean_topology_exact_h1_map_rank": float(
            np.mean([record["topology_exact_h1_map_rank_mean"] for record in branch_records])
        ),
        "mean_topology_exact_radius_shift": float(
            np.mean([record["topology_exact_morphism_radius_shift_mean"] for record in branch_records])
        ),
        "mean_topology_exact_edge_validity": float(
            np.mean([record["topology_exact_morphism_edge_validity_mean"] for record in branch_records])
        ),
        "mean_topology_exact_triangle_validity": float(
            np.mean([record["topology_exact_morphism_triangle_validity_mean"] for record in branch_records])
        ),
        "mean_topology_exact_directed_edge_validity": float(
            np.mean([record["topology_exact_directed_edge_validity_mean"] for record in branch_records])
        ),
        "mean_topology_exact_morphisms_computed": float(
            np.mean([record["topology_exact_morphism_computed"] for record in branch_records])
        ),
        "mean_topology_exact_truncated_complexes": float(
            np.mean([record["topology_exact_morphism_truncated_complexes"] for record in branch_records])
        ),
        "mean_toric_phase_recurrence": float(np.mean([record["toric_phase_recurrence"] for record in branch_records])),
        "mean_toric_slepian_concentration": mean_record_key("toric_slepian_concentration"),
        "mean_toric_slepian_leakage": mean_record_key("toric_slepian_leakage"),
        "mean_toric_slepian_mode_entropy": mean_record_key("toric_slepian_mode_entropy"),
        "mean_toric_slepian_effective_modes": mean_record_key("toric_slepian_effective_modes"),
        "mean_topology_inclusion_violation": float(np.mean([record["topology_inclusion_violation"] for record in branch_records])),
        "mean_topology_hdbscan_cluster_count": float(
            np.mean([record["topology_hdbscan_cluster_count"] for record in branch_records])
        ),
        "mean_topology_hdbscan_noise_fraction": float(
            np.mean([record["topology_hdbscan_noise_fraction"] for record in branch_records])
        ),
        "mean_topology_hdbscan_stability": float(
            np.mean([record["topology_hdbscan_stability"] for record in branch_records])
        ),
        "mean_toric_shadow_occupied_fan_cells": mean_shadow_key("occupied_fan_cells"),
        "mean_toric_shadow_fan_cell_entropy": mean_shadow_key("fan_cell_entropy"),
        "mean_toric_shadow_mean_margin": mean_shadow_key("mean_margin"),
        "mean_toric_shadow_min_margin": mean_shadow_key("min_margin"),
        "mean_toric_shadow_mean_bend": mean_shadow_key("mean_bend"),
        "mean_toric_shadow_slope_residual": mean_shadow_key("slope_residual"),
        "mean_toric_geometry_active_face_margin": mean_record_key("toric_geometry/toric_active_face_margin"),
        "mean_toric_geometry_active_face_entropy": mean_record_key("toric_geometry/toric_active_face_entropy"),
        "mean_toric_geometry_bend_magnitude": mean_record_key("toric_geometry/toric_bend_magnitude"),
        "mean_toric_geometry_binomial_residual": mean_record_key("toric_geometry/toric_binomial_residual"),
        "mean_toric_geometry_affine_wall_distance": mean_record_key("toric_geometry/toric_affine_wall_distance"),
        "mean_toric_geometry_coxeter_loss": mean_record_key("toric_geometry/toric_coxeter_loss"),
        "mean_toric_geometry_braid_loss": mean_record_key("toric_geometry/toric_braid_loss"),
        "mean_toric_geometry_leaf_residual": mean_record_key("toric_geometry/toric_leaf_residual"),
        "outputs": {
            "records": str(output_dir / "reasoning_geometry_records.json"),
            "selected_records": str(output_dir / "selected_records.json"),
            "triangles": str(triangle_dir),
            "tetrahedra": str(tetra_dir),
            "trajectories": str(output_dir / "trajectories"),
            "topology": str(output_dir / "topology"),
        },
    }
    (output_dir / "reasoning_geometry_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb logging requested but wandb is not installed")
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"reasoning-geometry-{Path(args.checkpoint).stem}",
            config=vars(args),
            job_type="reasoning_geometry_analysis",
        )
        log_payload = {
            f"analysis/{key}": value
            for key, value in summary.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        image_paths = []
        for pattern in (
            "trajectories/*_trajectory_3d.png",
            "trajectories/*_toric_phase_simplicial_trajectory.png",
            "trajectories/*_toric_phase_winding_collection.png",
            "topology/*_directed_filtration.png",
            "topology/*_toric_shadow_audit.png",
            "topology/*_toric_slepian_audit.png",
            "topology/*_step_radius_hierarchy.png",
            "topology/*_exact_persistence_morphisms.png",
            "topology/*_commutative_algebra_audit.png",
            "triangles/*.png",
            "tetrahedra/*.png",
        ):
            image_paths.extend(sorted(output_dir.glob(pattern))[:8])
        for image_path in image_paths[:48]:
            log_payload[f"analysis/images/{image_path.stem}"] = wandb.Image(str(image_path))
        wandb.log(log_payload)
        run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
