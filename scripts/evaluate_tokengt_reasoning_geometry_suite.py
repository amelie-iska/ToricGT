#!/usr/bin/env python3
"""TokenGT graph-of-thought geometry diagnostics.

This suite is the TokenGT counterpart to the random-order language-model
geometry scripts.  It loads a graph-to-graph ``ToricTokenGT`` checkpoint,
streams graph-structured validation/FineWeb records, and plots real node
embedding trajectories as branch/merge DAGs.  The score fields are graph
reconstruction and topology diagnostics, not byte-level OAI BPB.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toricgt.cli_config import flatten_cli_config, load_yaml_config  # noqa: E402
from toricgt.config import ModelConfig  # noqa: E402
from toricgt.derived_category_metrics import (  # noqa: E402
    DerivedCategoryConfig,
    derived_category_feature_summary,
    derived_category_objects_from_batch,
)
from toricgt.got_trajectory import got_dag_metrics  # noqa: E402
from toricgt.graph_dataset import CuratedGraphIterableDataset, collate_graph_items  # noqa: E402
from toricgt.graph_tokenizer import GraphBatch  # noqa: E402
from toricgt.metrics import masked_mse  # noqa: E402
from toricgt.model import ToricTokenGT  # noqa: E402
from toricgt.reasoning_geometry import (  # noqa: E402
    TRIANGLE_VERTICES,
    attach_normalized_scores,
    barycentric_to_cartesian,
    prim_mst_stats,
    rbf_interpolate,
    simplex_record,
    triangle_grid,
)
from toricgt.topological_reasoning import ReasoningTopologyConfig, directed_step_filtration_stats_np  # noqa: E402
from toricgt.visualization import (  # noqa: E402
    plot_energy_landscape,
    plot_reasoning_trajectory_3d,
    write_interactive_energy_landscape,
    write_interactive_reasoning_plot,
)
from toricgt.wandb_organization import configure_wandb_metrics, organize_wandb_payload  # noqa: E402


TRIANGLE_SPECS = {
    "graph_energy_topology": {
        "labels": ["low graph energy", "branch/merge structure", "MST efficiency"],
        "scores": ["score/low_energy", "score/dag_structure", "score/mst_efficiency"],
        "title": "TokenGT graph energy / DAG topology / MST simplex",
        "intensity": "score/low_energy",
    },
    "derived_memory_geometry": {
        "labels": ["derived-category regularity", "topology stability", "low merge scatter"],
        "scores": ["score/derived_regular", "score/topology_stability", "score/merge_tightness"],
        "title": "Derived-category / topology / memory-geometry simplex",
        "intensity": "score/derived_regular",
    },
    "noncommutative_flow": {
        "labels": ["low directed flux", "low asymmetry", "branch diversity"],
        "scores": ["score/low_cycle_flux", "score/low_asymmetry", "score/branch_diversity"],
        "title": "Directed noncommutative flow simplex",
        "intensity": "score/low_cycle_flux",
    },
    "branch_merge_non_linear": {
        "labels": ["branch/merge edges", "low chain collapse", "branch diversity"],
        "scores": ["score/branch_merge_edges", "score/low_linear_chain", "score/branch_diversity"],
        "title": "Branch/Merge DAG vs. Linear-Chain Collapse Simplex",
        "intensity": "score/low_linear_chain",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/train.full_tokengt_got_fineweb_derived.yaml")
    parser.add_argument("--data-glob", action="append", default=[])
    parser.add_argument("--output-dir", default="outputs/tokengt_reasoning_geometry")
    parser.add_argument("--records", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--parquet-batch-size", type=int, default=512)
    parser.add_argument("--fineweb-tokenizer-path", default="")
    parser.add_argument("--fineweb-tokens-per-graph", type=int, default=0)
    parser.add_argument("--fineweb-stride-tokens", type=int, default=0)
    parser.add_argument("--derived-category-max-vertices", type=int, default=8)
    parser.add_argument("--topology-max-points", type=int, default=32)
    parser.add_argument("--topology-max-windows", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wandb-run-path", default="")
    return parser.parse_args()


def config_get(flat: dict[str, Any], key: str, default: Any) -> Any:
    value = flat.get(key, default)
    return default if value is None else value


def load_flat_config(path: str | Path) -> dict[str, Any]:
    try:
        return flatten_cli_config(load_yaml_config(path))
    except Exception:
        return {}


def model_config_from_checkpoint(payload: dict[str, Any], flat_config: dict[str, Any]) -> ModelConfig:
    raw = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raw = flat_config
    allowed = {field.name for field in fields(ModelConfig)}
    filtered = {key: value for key, value in dict(raw).items() if key in allowed}
    return ModelConfig(**filtered)


def autocast_context(device: str, precision: str):
    if not str(device).startswith("cuda") or precision == "fp32":
        return torch.no_grad()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def move_batch(batch: GraphBatch, target: torch.Tensor, device: str) -> tuple[GraphBatch, torch.Tensor]:
    return (
        GraphBatch(
            node_features=batch.node_features.to(device),
            edge_features=batch.edge_features.to(device),
            edge_index=batch.edge_index.to(device),
            node_mask=batch.node_mask.to(device),
            edge_mask=batch.edge_mask.to(device),
            lm_input_ids=batch.lm_input_ids.to(device) if batch.lm_input_ids is not None else None,
            lm_target_ids=batch.lm_target_ids.to(device) if batch.lm_target_ids is not None else None,
            lm_mask=batch.lm_mask.to(device) if batch.lm_mask is not None else None,
            lm_target_byte_lengths=batch.lm_target_byte_lengths.to(device)
            if batch.lm_target_byte_lengths is not None
            else None,
            lm_target_positions=batch.lm_target_positions.to(device) if batch.lm_target_positions is not None else None,
            lm_input_features=batch.lm_input_features.to(device) if batch.lm_input_features is not None else None,
            node_causal_rank=batch.node_causal_rank.to(device) if batch.node_causal_rank is not None else None,
        ),
        target.to(device),
    )


def pca_project(points: np.ndarray, dims: int = 3) -> np.ndarray:
    if points.size == 0:
        return np.zeros((1, dims), dtype=np.float32)
    x = np.asarray(points, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 2:
        out = np.zeros((x.shape[0], dims), dtype=np.float32)
        out[:, : min(dims, x.shape[1])] = x[:, : min(dims, x.shape[1])]
        return out
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    basis = vt[: min(dims, vt.shape[0])].T
    projected = x @ basis
    if projected.shape[1] < dims:
        projected = np.pad(projected, ((0, 0), (0, dims - projected.shape[1])))
    return projected.astype(np.float32)


def finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else 0.0


def valid_edges(edge_index: torch.Tensor, edge_mask: torch.Tensor, n: int) -> np.ndarray:
    raw = edge_index.detach().cpu().numpy()
    mask = edge_mask.detach().cpu().numpy().astype(bool)
    edges = raw[mask].reshape(-1, 2)
    keep = (edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (edges[:, 0] < n) & (edges[:, 1] < n) & (edges[:, 0] != edges[:, 1])
    return edges[keep].astype(np.int64)


def per_node_mse(out_node: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    err = (out_node.float() - target.float()).pow(2).mean(dim=-1)
    return torch.where(node_mask, err, torch.zeros_like(err))


def topology_summary(stats: dict[str, Any]) -> dict[str, float]:
    return {
        "topology_edge_density": finite_mean([float(x) for x in stats.get("edge_density", [])]),
        "topology_triangle_density": finite_mean([float(x) for x in stats.get("triangle_density", [])]),
        "topology_directed_asymmetry": finite_mean([float(x) for x in stats.get("directed_asymmetry", [])]),
        "topology_directed_cycle_flux": finite_mean([float(x) for x in stats.get("directed_cycle_flux", [])]),
        "topology_hdbscan_cluster_count": finite_mean([float(x) for x in stats.get("hdbscan_cluster_count", [])]),
        "topology_hdbscan_noise_fraction": finite_mean([float(x) for x in stats.get("hdbscan_noise_fraction", [])]),
        "topology_hdbscan_stability": finite_mean([float(x) for x in stats.get("hdbscan_stability", [])]),
        "topology_exact_h0_dim_mean": float(stats.get("exact_h0_dim_mean", 0.0) or 0.0),
        "topology_exact_h1_dim_mean": float(stats.get("exact_h1_dim_mean", 0.0) or 0.0),
        "topology_variety_complex_residual_mean": float(stats.get("variety_complex_residual_mean", 0.0) or 0.0),
        "topology_fitting_minor_rank_residual_mean": float(stats.get("fitting_minor_rank_residual_mean", 0.0) or 0.0),
        "topology_buchsbaum_eisenbud_multiplier_residual_mean": float(
            stats.get("buchsbaum_eisenbud_multiplier_residual_mean", 0.0) or 0.0
        ),
        "topology_multigraded_betti_mass_mean": float(stats.get("multigraded_betti_mass_mean", 0.0) or 0.0),
    }


def plot_topology_heatmap(stats: dict[str, Any], output: Path) -> None:
    matrices = []
    titles = []
    distance = np.asarray(stats.get("distance", []), dtype=float)
    if distance.ndim == 2 and distance.size:
        matrices.append(distance)
        titles.append("normalized hidden distance")
    directed = stats.get("directed_adjacency", [])
    if isinstance(directed, list) and directed:
        matrix = np.asarray(directed[-1], dtype=float)
        if matrix.ndim == 2 and matrix.size:
            matrices.append(matrix)
            titles.append("directed adjacency")
    persistence = np.asarray(stats.get("hdbscan_persistence_adjacency", []), dtype=float)
    if persistence.ndim == 2 and persistence.size:
        matrices.append(persistence)
        titles.append("persistent reachability")
    if not matrices:
        return
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.2 * len(matrices), 4.8), facecolor="#020617")
    if len(matrices) == 1:
        axes = [axes]
    for ax, matrix, title in zip(axes, matrices, titles, strict=False):
        ax.set_facecolor("#020617")
        image = ax.imshow(matrix, cmap="magma", interpolation="nearest")
        ax.set_title(title, color="#f8fafc")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#cbd5e1")
        plt.setp(cbar.ax.get_yticklabels(), color="#cbd5e1")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_triangle(records: list[dict[str, Any]], spec: dict[str, Any], output: Path) -> None:
    vertices = [TRIANGLE_VERTICES["left"], TRIANGLE_VERTICES["right"], TRIANGLE_VERTICES["top"]]
    labels = spec["labels"]
    points: list[tuple[float, float]] = []
    intensity: list[float] = []
    for record in records:
        simplex = simplex_record(record, spec["scores"], labels)
        points.append(barycentric_to_cartesian(simplex["weights"], vertices))
        intensity.append(float(record.get(spec["intensity"], 0.0)))
    grid = triangle_grid(resolution=90)
    xs = [item[0] for item in grid]
    ys = [item[1] for item in grid]
    zs = [rbf_interpolate((x, y), points, intensity, bandwidth=0.20) for x, y, _ in grid]
    fig, ax = plt.subplots(figsize=(8.0, 7.0), facecolor="#020617")
    ax.set_facecolor("#020617")
    tri = mtri.Triangulation(xs, ys)
    contour = ax.tricontourf(tri, zs, levels=18, cmap="viridis", alpha=0.95)
    ax.tricontour(tri, zs, levels=8, colors="#67e8f9", linewidths=0.35, alpha=0.48)
    boundary_x = [vertices[0][0], vertices[1][0], vertices[2][0], vertices[0][0]]
    boundary_y = [vertices[0][1], vertices[1][1], vertices[2][1], vertices[0][1]]
    ax.plot(boundary_x, boundary_y, color="#67e8f9", linewidth=1.7)
    scatter = ax.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        c=[record["graph_reconstruction_mse"] for record in records],
        cmap="magma_r",
        s=92,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    for idx, (x, y) in enumerate(points):
        ax.text(x, y + 0.025, f"G{idx}", color="#f8fafc", fontsize=8, ha="center")
    offsets = [(-0.06, -0.055), (0.06, -0.055), (0.0, -0.065)]
    for label, vertex, offset in zip(labels, vertices, offsets, strict=False):
        ax.text(vertex[0] + offset[0], vertex[1] + offset[1], label, color="#e0f2fe", fontsize=10, ha="center")
    ax.set_title(spec["title"], color="#f8fafc", fontsize=14, pad=20)
    ax.set_aspect("equal")
    ax.set_axis_off()
    cbar = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("diagnostic intensity", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#cbd5e1")
    plt.setp(cbar.ax.get_yticklabels(), color="#cbd5e1")
    sb = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.09)
    sb.set_label("graph reconstruction MSE", color="#f8fafc")
    sb.ax.yaxis.set_tick_params(color="#cbd5e1")
    plt.setp(sb.ax.get_yticklabels(), color="#cbd5e1")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def extract_resolution_metrics(obj: dict[str, Any]) -> dict[str, float]:
    resolution = obj.get("projective_resolution", {})
    if isinstance(resolution, dict):
        resolution = resolution.get("resolution", {})
    metrics = resolution.get("metrics", {}) if isinstance(resolution, dict) else {}
    return {
        "symbolic_projective_dimension": float(metrics.get("symbolic_resolution_projective_dimension", 0.0) or 0.0),
        "symbolic_regularity": float(metrics.get("symbolic_resolution_regularity", 0.0) or 0.0),
        "symbolic_minimal_total_betti": float(metrics.get("symbolic_resolution_minimal_total_betti", 0.0) or 0.0),
        "symbolic_betti_entropy": float(metrics.get("symbolic_resolution_betti_entropy", 0.0) or 0.0),
    }


def evaluate_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flat = load_flat_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = model_config_from_checkpoint(payload, flat)
    model = ToricTokenGT(cfg)
    model.load_state_dict(payload["model"], strict=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)
    model.eval()
    paths = args.data_glob or config_get(flat, "val_data_path", ["data/curated_hf_shards/validation/*.parquet"])
    if isinstance(paths, str):
        paths = [paths]
    dataset = CuratedGraphIterableDataset(
        paths,
        cfg,
        parquet_batch_size=int(config_get(flat, "parquet_batch_size", args.parquet_batch_size)),
        interleave_paths=bool(config_get(flat, "interleave_data_paths", False)),
        fineweb_tokenizer_path=args.fineweb_tokenizer_path or str(config_get(flat, "fineweb_tokenizer_path", "")),
        fineweb_tokens_per_graph=int(args.fineweb_tokens_per_graph or config_get(flat, "fineweb_tokens_per_graph", 1024)),
        fineweb_stride_tokens=int(args.fineweb_stride_tokens or config_get(flat, "fineweb_stride_tokens", 1024)),
    )
    loader = DataLoader(dataset, batch_size=max(1, int(args.batch_size)), collate_fn=collate_graph_items, num_workers=0)
    out_dir = Path(args.output_dir)
    traj_dir = out_dir / "trajectories"
    topo_dir = out_dir / "topology"
    records: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    topology_cfg = ReasoningTopologyConfig(
        max_points=max(4, int(args.topology_max_points)),
        max_windows=max(1, int(args.topology_max_windows)),
        window_size=max(4, min(64, int(cfg.max_nodes))),
        step_stride=8,
    )
    with torch.no_grad():
        for batch, target in loader:
            batch, target = move_batch(batch, target, device)
            if device.startswith("cuda") and args.precision != "fp32":
                dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out = model(batch)
                    loss = masked_mse(out["node"], target, batch.node_mask)
            else:
                out = model(batch)
                loss = masked_mse(out["node"], target, batch.node_mask)
            node_energy = per_node_mse(out["node"], target, batch.node_mask)
            got = got_dag_metrics(
                out["node_embeddings"],
                node_mask=batch.node_mask,
                edge_index=batch.edge_index,
                edge_mask=batch.edge_mask,
            )
            derived = derived_category_feature_summary(
                out["node_embeddings"],
                node_mask=batch.node_mask,
                edge_index=batch.edge_index,
                edge_mask=batch.edge_mask,
                config=DerivedCategoryConfig(max_vertices=int(args.derived_category_max_vertices)),
            ).detach().float().cpu()
            batch_objects = derived_category_objects_from_batch(
                batch.edge_index,
                node_mask=batch.node_mask,
                edge_mask=batch.edge_mask,
                max_vertices=int(args.derived_category_max_vertices),
            )
            hidden_cpu = out["node_embeddings"].detach().float().cpu()
            energy_cpu = node_energy.detach().float().cpu()
            for sample_index in range(hidden_cpu.shape[0]):
                if len(records) >= int(args.records):
                    break
                n = int(batch.node_mask[sample_index].detach().cpu().sum().item())
                if n <= 0:
                    continue
                hidden = hidden_cpu[sample_index, :n].numpy()
                projected = pca_project(hidden, dims=3)
                energy = energy_cpu[sample_index, :n].numpy()
                if not np.isfinite(energy).all():
                    energy = np.nan_to_num(energy, nan=float(np.nanmean(energy)) if np.isfinite(np.nanmean(energy)) else 0.0)
                edges = valid_edges(batch.edge_index[sample_index], batch.edge_mask[sample_index], n)
                record_id = len(records)
                topology = directed_step_filtration_stats_np(hidden, config=topology_cfg)
                topology_metrics = topology_summary(topology)
                mst = prim_mst_stats(projected.tolist())
                graph_mse = float(energy[energy >= 0].mean()) if energy.size else float(loss.detach().cpu())
                path_speed = np.linalg.norm(np.diff(projected, axis=0), axis=1) if projected.shape[0] > 1 else np.zeros((1,))
                path_smoothness = float(1.0 / (1.0 + np.std(path_speed)))
                resolution_metrics = extract_resolution_metrics(batch_objects[sample_index])
                record = {
                    "record_index": int(record_id),
                    "checkpoint_step": int(payload.get("step", 0) or 0),
                    "graph_reconstruction_mse": graph_mse,
                    "global_masked_mse": float(loss.detach().cpu()),
                    "node_count": float(n),
                    "edge_count": float(edges.shape[0]),
                    "mst_efficiency": float(mst.get("mst_efficiency", 0.0)),
                    "mst_total_weight": float(mst.get("mst_total_weight", 0.0)),
                    "path_smoothness": path_smoothness,
                    "got_dag_branch_count": float(got["got_dag_branch_count_batch"][sample_index].detach().cpu()),
                    "got_dag_merge_count": float(got["got_dag_merge_count_batch"][sample_index].detach().cpu()),
                    "got_dag_back_edge_fraction": float(got["got_dag_back_edge_fraction_batch"][sample_index].detach().cpu()),
                    "got_dag_branch_diversity": float(got["got_dag_branch_diversity_batch"][sample_index].detach().cpu()),
                    "got_dag_merge_scatter": float(got["got_dag_merge_scatter_batch"][sample_index].detach().cpu()),
                    "got_dag_simplex_edge_density": float(got["got_dag_simplex_edge_density_batch"][sample_index].detach().cpu()),
                    "got_dag_triangle_density": float(got["got_dag_triangle_density_batch"][sample_index].detach().cpu()),
                    "got_dag_linear_chain_fraction": float(
                        got["got_dag_linear_chain_fraction_batch"][sample_index].detach().cpu()
                    ),
                    "got_dag_branch_merge_edge_fraction": float(
                        got["got_dag_branch_merge_edge_fraction_batch"][sample_index].detach().cpu()
                    ),
                    "derived_feature_projective_dimension": float(derived[sample_index, 7]),
                    "derived_feature_regularity": float(derived[sample_index, 8]),
                    "derived_feature_total_betti": float(derived[sample_index, 9]),
                    "derived_feature_betti_entropy": float(derived[sample_index, 10]),
                    **resolution_metrics,
                    **topology_metrics,
                }
                records.append(record)
                obj = dict(batch_objects[sample_index])
                obj["record_index"] = int(record_id)
                objects.append(obj)
                prefix = f"record_{record_id:03d}"
                plot_reasoning_trajectory_3d(projected, energy, traj_dir / f"{prefix}_trajectory_3d.png", edges=edges)
                plot_energy_landscape(projected, energy, traj_dir / f"{prefix}_energy_landscape.png", edges=edges)
                write_interactive_energy_landscape(projected, energy, traj_dir / f"{prefix}_energy_landscape.html", edges=edges)
                write_interactive_reasoning_plot(projected, energy, traj_dir / f"{prefix}_trajectory_3d.html", edges=edges)
                plot_topology_heatmap(topology, topo_dir / f"{prefix}_topology_heatmaps.png")
            if len(records) >= int(args.records):
                break
    if not records:
        raise RuntimeError("no TokenGT graph records were evaluated")
    return records, objects


def enrich_scores(records: list[dict[str, Any]]) -> None:
    attach_normalized_scores(
        records,
        {
            "score/low_energy": ("graph_reconstruction_mse", False),
            "score/dag_structure": ("got_dag_branch_count", True),
            "score/mst_efficiency": ("mst_efficiency", True),
            "score/derived_regular": ("derived_feature_betti_entropy", True),
            "score/topology_stability": ("topology_hdbscan_stability", True),
            "score/merge_tightness": ("got_dag_merge_scatter", False),
            "score/low_cycle_flux": ("topology_directed_cycle_flux", False),
            "score/low_asymmetry": ("topology_directed_asymmetry", False),
            "score/branch_diversity": ("got_dag_branch_diversity", True),
            "score/branch_merge_edges": ("got_dag_branch_merge_edge_fraction", True),
            "score/low_linear_chain": ("got_dag_linear_chain_fraction", False),
        },
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    def mean(key: str) -> float:
        return finite_mean([float(record.get(key, 0.0)) for record in records])

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_family": "tokengt_graph",
        "records": len(records),
        "oai_competition_bpb_available": 0.0,
        "oai_competition_bpb_note": "TokenGT graph checkpoints do not expose byte-level language-model logits; use graph reconstruction and topology metrics here, and evaluate OAI BPB on random-order/compact Seq4096 LM checkpoints.",
        "mean_graph_reconstruction_mse": mean("graph_reconstruction_mse"),
        "best_graph_reconstruction_mse": min(float(record["graph_reconstruction_mse"]) for record in records),
        "mean_mst_efficiency": mean("mst_efficiency"),
        "mean_path_smoothness": mean("path_smoothness"),
        "mean_got_dag_branch_count": mean("got_dag_branch_count"),
        "mean_got_dag_merge_count": mean("got_dag_merge_count"),
        "mean_got_dag_branch_diversity": mean("got_dag_branch_diversity"),
        "mean_got_dag_merge_scatter": mean("got_dag_merge_scatter"),
        "mean_got_dag_linear_chain_fraction": mean("got_dag_linear_chain_fraction"),
        "mean_got_dag_branch_merge_edge_fraction": mean("got_dag_branch_merge_edge_fraction"),
        "mean_topology_directed_asymmetry": mean("topology_directed_asymmetry"),
        "mean_topology_directed_cycle_flux": mean("topology_directed_cycle_flux"),
        "mean_topology_triangle_density": mean("topology_triangle_density"),
        "mean_topology_betti0": mean("topology_exact_h0_dim_mean"),
        "mean_topology_cycle_rank": mean("topology_exact_h1_dim_mean"),
        "mean_topology_hdbscan_cluster_count": mean("topology_hdbscan_cluster_count"),
        "mean_topology_hdbscan_noise_fraction": mean("topology_hdbscan_noise_fraction"),
        "mean_topology_hdbscan_stability": mean("topology_hdbscan_stability"),
        "mean_topology_variety_complex_residual": mean("topology_variety_complex_residual_mean"),
        "mean_topology_fitting_minor_rank_residual": mean("topology_fitting_minor_rank_residual_mean"),
        "mean_topology_buchsbaum_eisenbud_multiplier_residual": mean("topology_buchsbaum_eisenbud_multiplier_residual_mean"),
        "mean_topology_multigraded_betti_mass": mean("topology_multigraded_betti_mass_mean"),
        "mean_symbolic_projective_dimension": mean("symbolic_projective_dimension"),
        "mean_symbolic_regularity": mean("symbolic_regularity"),
        "mean_symbolic_minimal_total_betti": mean("symbolic_minimal_total_betti"),
        "mean_symbolic_betti_entropy": mean("symbolic_betti_entropy"),
    }
    return summary


def log_to_wandb(run_path: str, summary: dict[str, Any], step: int) -> None:
    if not run_path:
        return
    parts = [part for part in run_path.split("/") if part]
    if len(parts) != 3:
        return
    entity, project, run_id = parts
    import wandb

    run = wandb.init(entity=entity, project=project, id=run_id, resume="allow")
    try:
        configure_wandb_metrics(wandb)
        payload = {
            "trainer/step": step,
            "analysis_control/tokengt_geometry/mean_graph_reconstruction_mse": summary["mean_graph_reconstruction_mse"],
            "analysis_control/tokengt_geometry/best_graph_reconstruction_mse": summary["best_graph_reconstruction_mse"],
            "analysis_control/tokengt_geometry/mean_mst_efficiency": summary["mean_mst_efficiency"],
            "analysis_control/tokengt_geometry/mean_path_smoothness": summary["mean_path_smoothness"],
            "analysis_control/tokengt_geometry/mean_got_dag_branch_count": summary["mean_got_dag_branch_count"],
            "analysis_control/tokengt_geometry/mean_got_dag_merge_count": summary["mean_got_dag_merge_count"],
            "analysis_control/tokengt_geometry/mean_got_dag_linear_chain_fraction": summary[
                "mean_got_dag_linear_chain_fraction"
            ],
            "analysis_control/tokengt_geometry/mean_got_dag_branch_merge_edge_fraction": summary[
                "mean_got_dag_branch_merge_edge_fraction"
            ],
            "analysis_control/tokengt_geometry/mean_topology_directed_asymmetry": summary["mean_topology_directed_asymmetry"],
            "analysis_control/tokengt_geometry/mean_topology_directed_cycle_flux": summary["mean_topology_directed_cycle_flux"],
            "analysis_control/tokengt_geometry/oai_competition_bpb_available": summary["oai_competition_bpb_available"],
        }
        run.log(organize_wandb_payload(payload), step=step)
    finally:
        run.finish()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    out_dir = Path(args.output_dir)
    records, objects = evaluate_records(args)
    enrich_scores(records)
    write_csv(out_dir / "tokengt_geometry_records.csv", records)
    (out_dir / "tokengt_geometry_records.json").write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "derived_category_objects.json").write_text(json.dumps(objects, indent=2, sort_keys=True), encoding="utf-8")
    for name, spec in TRIANGLE_SPECS.items():
        plot_triangle(records, spec, out_dir / f"{name}_triangle.png")
    summary = summarize(records, args)
    summary["outputs"] = sorted(path.name for path in out_dir.glob("*") if path.is_file())
    summary["trajectory_html_outputs"] = sorted(
        str(path.relative_to(out_dir)) for path in (out_dir / "trajectories").glob("*_trajectory_3d.html")
    )
    summary["interactive_energy_landscape_outputs"] = sorted(
        str(path.relative_to(out_dir)) for path in (out_dir / "trajectories").glob("*_energy_landscape.html")
    )
    (out_dir / "reasoning_geometry_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    step = int(records[0].get("checkpoint_step", 0) or 0)
    log_to_wandb(args.wandb_run_path, summary, step=step)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
