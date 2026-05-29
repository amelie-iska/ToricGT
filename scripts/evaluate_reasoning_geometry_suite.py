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
        "labels": ["reasoning time", "K(x)", "low BPB"],
        "scores": ["score/reasoning", "score/k", "score/bpb_quality"],
        "title": "Reasoning/K/BPB simplex",
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
        "labels": ["K(x)", "compression efficiency", "answer likelihood"],
        "scores": ["score/k", "score/compression_efficiency", "score/answer_quality"],
        "title": "Compression/reasoning simplex",
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
        "labels": ["reasoning time", "K(x)", "low BPB", "MST efficiency"],
        "scores": ["score/reasoning", "score/k", "score/bpb_quality", "score/mst_efficiency"],
        "title": "Reasoning/K/BPB/MST tetrahedron",
    },
    "solution_bpb_smooth_diversity": {
        "labels": ["answer likelihood", "low BPB", "smooth flow", "GFlowNet diversity"],
        "scores": ["score/answer_quality", "score/bpb_quality", "score/path_smoothness", "score/action_diversity"],
        "title": "Solution/BPB/flow/diversity tetrahedron",
    },
    "complexity_geometry_solution": {
        "labels": ["K(x)", "trajectory depth", "MST efficiency", "answer likelihood"],
        "scores": ["score/k", "score/trajectory_depth", "score/mst_efficiency", "score/answer_quality"],
        "title": "Complexity/geometry/solution tetrahedron",
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
    parser.add_argument("--compressors", nargs="+", default=["zlib", "lzma"])
    parser.add_argument("--domain-keywords", nargs="+", default=list(DEFAULT_DOMAIN_KEYWORDS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--random-init", action="store_true")
    return parser.parse_args()


def safe_text(value: Any) -> str:
    return value if isinstance(value, str) and value.strip() else ""


def slug(value: str, max_len: int = 72) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower())
    return value.strip("-")[:max_len] or "record"


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
        "per_token_nll": per_token_nll.detach().float().cpu().numpy()[0],
        "target_positions": batch.target_positions.detach().cpu().numpy()[0],
        "answer_steps": answer_steps,
        "action_ids": flat_actions,
    }
    for key, value in complexity.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            record[key] = float(value)
    for key, value in mst.items():
        record[f"mst/{key}"] = float(value)
    record["k_proxy"] = float(
        record.get("complexity/target_cond_k_lzma_mean", record.get("complexity/target_cond_k_zlib_mean", 0.0))
        + 0.25
        * record.get(
            "complexity/gflownet_action_trace_k_lzma_mean",
            record.get("complexity/order_program_k_lzma_mean", 0.0),
        )
    )
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
        record["compression_efficiency_raw"] = float(record["k_proxy"] / max(record["bpb"], 1e-6))
    attach_normalized_scores(
        records,
        {
            "score/reasoning": ("trajectory_depth", True),
            "score/k": ("k_proxy", True),
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
    label_offsets = [(-0.08, -0.055), (0.08, -0.055), (0.0, 0.048)]
    for label, vertex, offset in zip(labels, vertices, label_offsets):
        ax.text(vertex[0] + offset[0], vertex[1] + offset[1], label, color="#e8fbff", fontsize=10, ha="center")
    ax.set_title(spec["title"], color="white", fontsize=14, pad=16)
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
                f"BPB={record['bpb']:.4f}<br>K={record['k_proxy']:.2f}<br>"
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


def plot_trajectory_3d(record_meta: dict[str, Any], branches: list[dict[str, Any]], output_path: Path) -> None:
    fig = plt.figure(figsize=(10, 8), facecolor="#030712")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#030712")
    cmap = plt.get_cmap("turbo")
    bpbs = np.array([float(branch["bpb"]) for branch in branches], dtype=float)
    lo, hi = float(bpbs.min()), float(bpbs.max())
    best_index = int(np.argmin(bpbs))
    for branch in branches:
        path = branch["projected_path"]
        idx = subsample_indices(path.shape[0], int(branch.get("max_plot_points", 180)))
        plotted = path[idx]
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
        "cyan=start, gold=actual answer/solution span, star=best-likelihood terminal",
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
    traces = []
    bpbs = np.array([float(branch["bpb"]) for branch in branches], dtype=float)
    lo, hi = float(bpbs.min()), float(bpbs.max())
    for branch in branches:
        path = branch["projected_path"]
        idx = subsample_indices(path.shape[0], int(branch.get("max_plot_points", 180)))
        plotted = path[idx]
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
                    f"answer BPB={branch['answer_bpb']:.4f}<br>K={branch['k_proxy']:.2f}"
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
    payload = {
        "title": f"R{record_meta['record_index']} {record_meta.get('dataset', '')} / {record_meta.get('task_family', '')}",
        "traces": traces,
    }
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Reasoning trajectory</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script></head>
<body style="margin:0;background:#030712;color:white;font-family:system-ui">
<div id="plot" style="width:100vw;height:100vh"></div>
<script>
const payload = {json.dumps(payload)};
Plotly.newPlot('plot', payload.traces, {{
  title:{{text:payload.title, font:{{color:'white'}}}},
  paper_bgcolor:'#030712', plot_bgcolor:'#030712',
  scene:{{xaxis:{{visible:false}}, yaxis:{{visible:false}}, zaxis:{{visible:false}}, bgcolor:'#030712'}},
  legend:{{font:{{color:'white'}}}},
  margin:{{l:0,r:0,b:0,t:52}}
}});
</script></body></html>
"""
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
    for branch in branches:
        path = branch["projected_path"]
        energy = np.asarray(branch["per_token_nll"], dtype=float)
        idx = subsample_indices(path.shape[0], 800)
        xs.extend(path[idx, 0].tolist())
        ys.extend(path[idx, 1].tolist())
        zs.extend(energy[idx].tolist())
    fig, ax = plt.subplots(figsize=(8.2, 7), facecolor="#030712")
    ax.set_facecolor("#030712")
    if len(xs) > 12:
        tri = mtri.Triangulation(xs, ys)
        contour = ax.tricontourf(tri, zs, levels=28, cmap="magma", alpha=0.92)
        ax.tricontour(tri, zs, levels=9, colors="#dff8ff", linewidths=0.22, alpha=0.35)
        cbar = fig.colorbar(contour, ax=ax, fraction=0.04, pad=0.02)
    else:
        scatter = ax.scatter(xs, ys, c=zs, cmap="magma", s=8)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("local NLL energy", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    for branch in branches:
        path = branch["projected_path"]
        ax.plot(path[:: max(1, path.shape[0] // 160), 0], path[:: max(1, path.shape[0] // 160), 1], color="#6df6ff", alpha=0.28, linewidth=0.8)
    ax.set_xlabel("PC1", color="white")
    ax.set_ylabel("PC2", color="white")
    ax.set_title(f"Embedding energy landscape R{record_meta['record_index']}", color="white")
    ax.tick_params(colors="white")
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

    The vertices are relation arrows between consecutive hidden states.  The
    symmetric distance builds a small soft Vietoris-Rips filtration, while an
    antisymmetric bilinear form biases edges into a directed noncommutative
    filtration.  These quantities are diagnostics, not persistent-homology
    replacements: they are cheap enough to run during periodic training
    analyses and stable under global hidden-state rescalings.
    """

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
            "skew_norm": 0.0,
            "distance": np.zeros((1, 1), dtype=float),
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
    radii = np.linspace(float(radius_min), float(radius_max), max(1, int(levels)))
    edge_density: list[float] = []
    triangle_density: list[float] = []
    directed_edge_density: list[float] = []
    directed_asymmetry: list[float] = []
    directed_cycle_flux: list[float] = []
    directed_transitive_loss: list[float] = []
    directed_chain_commutator: list[float] = []
    inclusion_violation: list[float] = []
    directed_adjacency: list[np.ndarray] = []
    previous_sym: np.ndarray | None = None
    previous_dir: np.ndarray | None = None
    for radius in radii:
        sym = sigmoid_np((radius - dist) / max(float(temperature), 1e-6)) * (1.0 - eye)
        directed = sigmoid_np((radius - dist + skew_unit) / max(float(temperature), 1e-6)) * (1.0 - eye)
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
        "skew_norm": float(np.mean(np.abs(skew_unit))),
        "distance": dist,
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
    branch["topology_skew_norm"] = float(stats["skew_norm"])


def plot_directed_filtration(record_meta: dict[str, Any], branches: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor="#030712")
    metrics = [
        ("edge_density", "symmetric edge density"),
        ("triangle_density", "soft triangle density"),
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
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel("filtration radius", color="white")
        ax.tick_params(colors="white")
        ax.grid(color="#1f3b52", linewidth=0.5, alpha=0.6)
    axes[0, 0].legend(loc="best", fontsize=7, framealpha=0.18, facecolor="#07111e", labelcolor="white")
    fig.suptitle(
        f"Directed nested simplicial diagnostics R{record_meta['record_index']}",
        color="white",
        fontsize=14,
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
        (np.asarray(stats["skew"]), "antisymmetric toric skew", "coolwarm"),
        (np.asarray(directed[0]), f"directed adjacency r={stats['radii'][0]:.2f}", "viridis"),
        (np.asarray(directed[mid]), f"directed adjacency r={stats['radii'][mid]:.2f}", "viridis"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.2), facecolor="#030712")
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


def serializable_record(record: dict[str, Any]) -> dict[str, Any]:
    skip = {"hidden", "per_token_nll", "projected_path", "target_positions", "topology_stats"}
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
        enrich_branch_scores(branches)
        branch_records.extend(branches)
        record_slug = f"R{record_index}_{slug(meta['dataset'] + '_' + meta['task_family'])}"
        traj_dir = output_dir / "trajectories"
        topology_dir = output_dir / "topology"
        plot_trajectory_3d(meta, branches, traj_dir / f"{record_slug}_trajectory_3d.png")
        write_interactive_trajectory(meta, branches, traj_dir / f"{record_slug}_trajectory_3d.html")
        plot_phase_energy(meta, branches, traj_dir / f"{record_slug}_phase_energy.png")
        plot_energy_landscape(meta, branches, traj_dir / f"{record_slug}_energy_landscape.png")
        plot_directed_filtration(meta, branches, topology_dir / f"{record_slug}_directed_filtration.png")
        plot_topology_heatmaps(meta, branches, topology_dir / f"{record_slug}_noncommutative_heatmaps.png")
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
        "mean_topology_inclusion_violation": float(np.mean([record["topology_inclusion_violation"] for record in branch_records])),
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
