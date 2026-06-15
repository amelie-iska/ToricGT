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
import glob
import html
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import evaluate_seq4096_reasoning_geometry_suite as rich_geometry  # noqa: E402
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
from toricgt.gudhi_persistence import GudhiPersistenceConfig, vectorized_point_cloud_signature  # noqa: E402
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
from toricgt.slepian_torus import ToricSlepianConfig, toric_slepian_audit  # noqa: E402
from toricgt.topological_reasoning import ReasoningTopologyConfig, directed_step_filtration_stats_np  # noqa: E402
from toricgt.visualization import (  # noqa: E402
    plot_energy_landscape,
    plot_pca_nll_4d,
    plot_reasoning_step_complex_pca_3d,
    plot_reasoning_trajectory_3d,
    write_interactive_energy_landscape,
    write_interactive_pca_nll_4d,
    write_interactive_reasoning_step_complex_pca_3d,
    write_interactive_reasoning_plot,
)
from toricgt.wandb_organization import configure_wandb_metrics, organize_wandb_payload  # noqa: E402


DEFAULT_FINEWEB_TOKEN_GLOB = "amelie-iska/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"


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

ANALOGY_RETRIEVAL_THRESHOLDS = {
    "ph_signature_cosine": 0.45,
    "simplicial_edge_valid_fraction": 0.55,
    "trajectory_tree_map_score": 0.20,
}


PLOT_FAMILY_REGISTRY = [
    {
        "family": "trajectory_3d",
        "introduced": "previous",
        "dimension": "3D",
        "static_pattern": "trajectories/*_trajectory_3d.png",
        "interactive_pattern": "trajectories/*_trajectory_3d.html",
        "interactive_required": True,
        "description": "Raw node-embedding reasoning trajectory with branch/merge graph edges and energy color.",
    },
    {
        "family": "energy_landscape",
        "introduced": "previous",
        "dimension": "3D surface",
        "static_pattern": "trajectories/*_energy_landscape.png",
        "interactive_pattern": "trajectories/*_energy_landscape.html",
        "interactive_required": True,
        "description": "Smooth projected energy/fitness surface with trajectory overlay and toric/tropical chamber-wall proxies.",
    },
    {
        "family": "pca_nll_4d",
        "introduced": "previous",
        "dimension": "4D",
        "static_pattern": "trajectories/record_*_pca_nll_4d.png",
        "interactive_pattern": "trajectories/record_*_pca_nll_4d.html",
        "interactive_required": True,
        "description": "3D PCA coordinates of node embeddings with per-node NLL as the fourth plotted coordinate.",
    },
    {
        "family": "complex_pca_nll_3d",
        "introduced": "current",
        "dimension": "3D/4D complex",
        "static_pattern": "trajectories/record_*_complex_pca_nll_3d.png",
        "interactive_pattern": "trajectories/record_*_complex_pca_nll_3d.html",
        "interactive_required": True,
        "description": "Reasoning-step filtered simplicial complexes as trajectory vertices, colored by mean NLL and sized by simplicial mass, with local-complex thought bubbles.",
    },
    {
        "family": "reasoning_step_simplex_tree_3d",
        "introduced": "staircase_simplextree_slepian_20260615",
        "dimension": "3D PCA filtered simplex trees",
        "static_pattern": "",
        "interactive_pattern": "trajectories/record_*_reasoning_step_simplex_tree_3d.html",
        "interactive_required": True,
        "description": "Token embedding subcollections per reasoning step, slider-controlled Rips simplex trees, NLL colors, and decode-order half-arrows.",
    },
    {
        "family": "analogical_simplex_maps_3d",
        "introduced": "staircase_simplextree_slepian_20260615",
        "dimension": "3D PCA analogical maps",
        "static_pattern": "",
        "interactive_pattern": "analogical/record_*_analogical_simplex_maps_3d.html",
        "interactive_required": True,
        "description": "Simplicial maps between reasoning-step simplex trees, vectorized persistence comparisons, and full-trajectory filtered-complex maps.",
    },
    {
        "family": "slepian_torus_surface",
        "introduced": "staircase_simplextree_slepian_20260615",
        "dimension": "3D foliated torus",
        "static_pattern": "",
        "interactive_pattern": "trajectories/record_*_slepian_torus_surface.html",
        "interactive_required": True,
        "description": "Slepian/Pollak DPSS reconstruction and envelope colormap on the finite irrational torus foliation.",
    },
    {
        "family": "topology_heatmaps",
        "introduced": "previous",
        "dimension": "2D",
        "static_pattern": "topology/*_topology_heatmaps.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Distance, directed adjacency, and persistent reachability heatmaps.",
    },
    {
        "family": "phase_energy",
        "introduced": "r97_rich_baseline",
        "dimension": "2D",
        "static_pattern": "trajectories/*_phase_energy.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Toric phase-energy projection for the selected graph/inference trajectory.",
    },
    {
        "family": "toric_phase_simplicial_trajectory",
        "introduced": "r97_rich_baseline",
        "dimension": "3D",
        "static_pattern": "trajectories/*_toric_phase_simplicial_trajectory.png",
        "interactive_pattern": "trajectories/*_toric_phase_simplicial_trajectory.html",
        "interactive_required": True,
        "description": "Toric phase trajectory with local simplicial structure and analogical transport overlay.",
    },
    {
        "family": "projected_simplicial_toric_geometry",
        "introduced": "r97_rich_baseline",
        "dimension": "3D/4D",
        "static_pattern": "",
        "interactive_pattern": "trajectories/*_projected_simplicial_toric_geometry.html",
        "interactive_required": True,
        "description": "Projected hidden-space simplicial toric geometry HTML companion.",
    },
    {
        "family": "toric_phase_winding_collection",
        "introduced": "r97_rich_baseline",
        "dimension": "2D",
        "static_pattern": "trajectories/*_toric_phase_winding_collection.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Flat irrational toric phase winding and recurrence audit.",
    },
    {
        "family": "directed_filtration",
        "introduced": "r97_rich_baseline",
        "dimension": "2D",
        "static_pattern": "topology/*_directed_filtration.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Directed nested simplicial filtration curves across radii.",
    },
    {
        "family": "noncommutative_heatmaps",
        "introduced": "r97_rich_baseline",
        "dimension": "2D",
        "static_pattern": "topology/*_noncommutative_heatmaps.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Skew, directed adjacency, and noncommutative flow heatmaps.",
    },
    {
        "family": "step_radius_hierarchy",
        "introduced": "r97_rich_baseline",
        "dimension": "2D",
        "static_pattern": "topology/*_step_radius_hierarchy.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Step-by-radius hierarchy for topology, DEC, and transport diagnostics.",
    },
    {
        "family": "exact_persistence_morphisms",
        "introduced": "r97_rich_baseline",
        "dimension": "2D",
        "static_pattern": "topology/*_exact_persistence_morphisms.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Exact persistence-module and simplicial morphism audits.",
    },
    {
        "family": "commutative_algebra_audit",
        "introduced": "r97_rich_baseline",
        "dimension": "2D",
        "static_pattern": "topology/*_commutative_algebra_audit.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Koszul/BGG/Fitting/Buchsbaum-Eisenbud commutative-algebra audit.",
    },
    {
        "family": "toric_shadow_audit",
        "introduced": "r97_rich_baseline",
        "dimension": "2D",
        "static_pattern": "topology/*_toric_shadow_audit.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Empirical toric fan, chamber, margin, and bend diagnostics.",
    },
    {
        "family": "toric_slepian_audit",
        "introduced": "r97_rich_baseline",
        "dimension": "2D",
        "static_pattern": "topology/*_toric_slepian_audit.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Slepian/Pollak prolate concentration and leakage audit.",
    },
    {
        "family": "graphcg_basis_disentanglement",
        "introduced": "r97_rich_baseline",
        "dimension": "2D",
        "static_pattern": "graphcg/*_graphcg_basis_disentanglement.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "GraphCG basis disentanglement and Gram-matrix audit.",
    },
    {
        "family": "analogical_transport_map",
        "introduced": "r97_rich_baseline",
        "dimension": "2D",
        "static_pattern": "analogical/*_analogical_transport_map.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Analogical transport parallelogram residual map.",
    },
    {
        "family": "tropical_chamber_audit",
        "introduced": "r97_rich_baseline",
        "dimension": "2D",
        "static_pattern": "tropical/*_tropical_chamber_audit.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Tropical active-face/chamber path and energy plateau audit.",
    },
    {
        "family": "rich_triangles",
        "introduced": "r97_rich_baseline",
        "dimension": "2D simplex",
        "static_pattern": "triangles/*.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "R97-style BPB, compression, topology, toric, CCA, and control triangle plots.",
    },
    {
        "family": "rich_tetrahedra",
        "introduced": "r97_rich_baseline",
        "dimension": "3D",
        "static_pattern": "tetrahedra/*.png",
        "interactive_pattern": "tetrahedra/*.html",
        "interactive_required": True,
        "description": "R97-style tetrahedra with interactive HTML companions.",
    },
    {
        "family": "symbolic_resolution_artifacts",
        "introduced": "r97_rich_baseline",
        "dimension": "symbolic data",
        "static_pattern": "topology/*_symbolic_resolution_certificate*",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Exact multigraded symbolic-resolution certificate JSON/CSV files.",
    },
    {
        "family": "legacy_reasoning_geometry_records",
        "introduced": "r97_rich_baseline",
        "dimension": "table",
        "static_pattern": "reasoning_geometry_records.*",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "R97-compatible public record table for rich geometry diagnostics.",
    },
    {
        "family": "selected_records",
        "introduced": "r97_rich_baseline",
        "dimension": "metadata",
        "static_pattern": "selected_records.json",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Selected record manifest matching the older output layout.",
    },
    {
        "family": "fineweb_curve_diagnostic_payload",
        "introduced": "r97_rich_baseline",
        "dimension": "metadata",
        "static_pattern": "fineweb_curve_diagnostic_payload.json",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "TokenGT geometry diagnostic payload; true byte BPB is produced only by the FineWeb BPB evaluator.",
    },
    {
        "family": "graph_energy_topology_triangle",
        "introduced": "previous",
        "dimension": "2D simplex",
        "static_pattern": "graph_energy_topology_triangle.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Triangle diagnostic for graph energy, DAG topology, and MST efficiency.",
    },
    {
        "family": "derived_memory_geometry_triangle",
        "introduced": "previous",
        "dimension": "2D simplex",
        "static_pattern": "derived_memory_geometry_triangle.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Triangle diagnostic for derived-category regularity, topology stability, and merge tightness.",
    },
    {
        "family": "noncommutative_flow_triangle",
        "introduced": "previous",
        "dimension": "2D simplex",
        "static_pattern": "noncommutative_flow_triangle.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Triangle diagnostic for directed flux, asymmetry, and branch diversity.",
    },
    {
        "family": "branch_merge_non_linear_triangle",
        "introduced": "previous",
        "dimension": "2D simplex",
        "static_pattern": "branch_merge_non_linear_triangle.png",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Triangle diagnostic for branch/merge structure versus linear-chain collapse.",
    },
    {
        "family": "tokengt_geometry_records",
        "introduced": "previous",
        "dimension": "table",
        "static_pattern": "tokengt_geometry_records.*",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Per-record CSV/JSON metrics backing the geometry plots.",
    },
    {
        "family": "derived_category_objects",
        "introduced": "previous",
        "dimension": "symbolic data",
        "static_pattern": "derived_category_objects.json",
        "interactive_pattern": "",
        "interactive_required": False,
        "description": "Derived-category, projective-resolution, and symbolic CCA objects used by the diagnostics.",
    },
]


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
    parser.add_argument(
        "--emit-embedding-payloads",
        dest="emit_embedding_payloads",
        action="store_true",
        default=True,
        help="Persist exact hidden-state, PCA, energy, NLL, edge, and local-complex arrays for CAS/inference audits.",
    )
    parser.add_argument(
        "--no-emit-embedding-payloads",
        dest="emit_embedding_payloads",
        action="store_false",
        help="Skip NPZ embedding payload output.",
    )
    parser.add_argument(
        "--rich-legacy-geometry",
        dest="rich_legacy_geometry",
        action="store_true",
        default=True,
        help="Generate the R97-style rich geometry bundle in addition to TokenGT-native plots.",
    )
    parser.add_argument(
        "--no-rich-legacy-geometry",
        dest="rich_legacy_geometry",
        action="store_false",
        help="Skip R97-style rich geometry outputs for a faster narrow pass.",
    )
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


def expand_glob_path(path: str | Path) -> list[str]:
    path_str = str(path)
    matches = sorted(glob.glob(path_str))
    return matches or [path_str]


def geometry_data_paths(args: argparse.Namespace, flat: dict[str, Any]) -> list[str]:
    requested = args.data_glob or config_get(flat, "val_data_path", ["data/curated_hf_shards/validation/*.parquet"])
    if isinstance(requested, str):
        requested = [requested]
    requested_paths = [str(path) for path in requested]
    if any(path.endswith(".bin") or ".bin" in path for path in requested_paths):
        return [path for item in requested_paths for path in expand_glob_path(item)]
    token_glob = str(config_get(flat, "val_token_glob", DEFAULT_FINEWEB_TOKEN_GLOB))
    token_paths = expand_glob_path(token_glob)
    if token_paths and any(path.endswith(".bin") for path in token_paths):
        return token_paths
    return [path for item in requested_paths for path in expand_glob_path(item)]


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


def write_embedding_payload(
    embedding_dir: Path,
    *,
    record_id: int,
    checkpoint_step: int,
    hidden: np.ndarray,
    projected: np.ndarray,
    energy: np.ndarray,
    nll: np.ndarray,
    edges: np.ndarray,
    complex_projected: np.ndarray,
    complex_nll: np.ndarray,
    complex_mass: np.ndarray,
    complex_edges: np.ndarray,
    complex_rows: list[dict[str, Any]],
    nll_source: str,
) -> dict[str, Any]:
    """Persist the exact arrays consumed by geometry and CAS sidecar audits."""

    embedding_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"record_{record_id:03d}"
    npz_path = embedding_dir / f"{prefix}_embedding_payload.npz"
    json_path = embedding_dir / f"{prefix}_embedding_payload.json"
    np.savez_compressed(
        npz_path,
        hidden=np.asarray(hidden, dtype=np.float32),
        projected=np.asarray(projected, dtype=np.float32),
        energy=np.asarray(energy, dtype=np.float32),
        nll=np.asarray(nll, dtype=np.float32),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        complex_projected=np.asarray(complex_projected, dtype=np.float32),
        complex_nll=np.asarray(complex_nll, dtype=np.float32),
        complex_mass=np.asarray(complex_mass, dtype=np.float32),
        complex_edges=np.asarray(complex_edges, dtype=np.int64).reshape(-1, 2),
    )
    metadata = {
        "schema": "toricgt.embedding_payload.v1",
        "record_index": int(record_id),
        "checkpoint_step": int(checkpoint_step),
        "nll_source": str(nll_source),
        "npz": npz_path.name,
        "array_shapes": {
            "hidden": list(np.asarray(hidden).shape),
            "projected": list(np.asarray(projected).shape),
            "energy": list(np.asarray(energy).shape),
            "nll": list(np.asarray(nll).shape),
            "edges": list(np.asarray(edges).reshape(-1, 2).shape),
            "complex_projected": list(np.asarray(complex_projected).shape),
            "complex_nll": list(np.asarray(complex_nll).shape),
            "complex_mass": list(np.asarray(complex_mass).shape),
            "complex_edges": list(np.asarray(complex_edges).reshape(-1, 2).shape),
        },
        "complex_rows": complex_rows,
    }
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "record_index": int(record_id),
        "npz": str(npz_path),
        "json": str(json_path),
        "relative_npz": f"embeddings/{npz_path.name}",
        "relative_json": f"embeddings/{json_path.name}",
        "node_count": int(np.asarray(hidden).shape[0]),
        "hidden_dim": int(np.asarray(hidden).shape[1]) if np.asarray(hidden).ndim == 2 else 0,
    }


def connected_components_count(node_count: int, edges: list[tuple[int, int]]) -> int:
    if node_count <= 0:
        return 0
    parent = list(range(node_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for src, dst in edges:
        if 0 <= src < node_count and 0 <= dst < node_count and src != dst:
            union(src, dst)
    return len({find(idx) for idx in range(node_count)})


def reasoning_step_complexes(
    hidden: np.ndarray,
    nll: np.ndarray,
    edges: np.ndarray,
    *,
    max_vertices: int = 11,
    radius_quantile: float = 0.42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build one local filtered simplicial complex per reasoning step.

    Each anchor vertex contributes a small metric-neighborhood complex.  Edges
    are included when they are present in the trajectory graph or appear before
    the local filtration radius; triangles are clique 2-simplices in that local
    filtered graph.  The returned global coordinates are PCA coordinates of the
    resulting complex signatures, not raw token embeddings.
    """

    hidden = np.asarray(hidden, dtype=np.float32)
    nll = np.asarray(nll, dtype=np.float32)
    n = int(hidden.shape[0])
    if n <= 0:
        empty_edges = np.zeros((0, 2), dtype=np.int64)
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), empty_edges, []
    max_vertices = max(3, min(int(max_vertices), n))
    if nll.shape[0] < n:
        nll = np.pad(nll, (0, n - nll.shape[0]), constant_values=0.0)
    nll = np.nan_to_num(nll[:n], nan=float(np.nanmean(nll[:n])) if np.isfinite(np.nanmean(nll[:n])) else 0.0)
    graph_adj = [set() for _ in range(n)]
    directed_edges: list[tuple[int, int]] = []
    for src_raw, dst_raw in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        src = int(src_raw)
        dst = int(dst_raw)
        if 0 <= src < n and 0 <= dst < n and src != dst:
            directed_edges.append((src, dst))
            graph_adj[src].add(dst)
            graph_adj[dst].add(src)
    if directed_edges:
        trajectory_edges = np.asarray(directed_edges, dtype=np.int64)
    else:
        trajectory_edges = np.asarray([[idx, idx + 1] for idx in range(n - 1)], dtype=np.int64)

    diff = hidden[:, None, :] - hidden[None, :, :]
    distance = np.linalg.norm(diff, axis=-1)
    positive = distance[distance > 1e-8]
    global_radius = float(np.quantile(positive, radius_quantile)) if positive.size else 0.0
    signatures: list[np.ndarray] = []
    mean_nlls: list[float] = []
    masses: list[float] = []
    rows: list[dict[str, Any]] = []
    for anchor in range(n):
        neighbors = set(graph_adj[anchor])
        neighbors.add(anchor)
        for candidate in np.argsort(distance[anchor]):
            if len(neighbors) >= max_vertices:
                break
            neighbors.add(int(candidate))
        ordered = sorted(neighbors, key=lambda idx: (0.0 if idx == anchor else float(distance[anchor, idx]), idx))
        local = sorted(ordered[:max_vertices])
        local_index = {vertex: idx for idx, vertex in enumerate(local)}
        local_count = len(local)
        local_distance = distance[np.ix_(local, local)]
        local_positive = local_distance[local_distance > 1e-8]
        radius = float(np.quantile(local_positive, radius_quantile)) if local_positive.size else global_radius
        if radius <= 1e-8 and global_radius > 1e-8:
            radius = global_radius
        local_edges: set[tuple[int, int]] = set()
        for src, dst in directed_edges:
            if src in local_index and dst in local_index:
                a, b = sorted((local_index[src], local_index[dst]))
                if a != b:
                    local_edges.add((a, b))
        for i in range(local_count):
            for j in range(i + 1, local_count):
                if local_distance[i, j] <= radius:
                    local_edges.add((i, j))
        sorted_local_edges = sorted(local_edges)
        edge_lookup = set(sorted_local_edges)
        triangles = 0
        for i in range(local_count):
            for j in range(i + 1, local_count):
                if (i, j) not in edge_lookup:
                    continue
                for k in range(j + 1, local_count):
                    if (i, k) in edge_lookup and (j, k) in edge_lookup:
                        triangles += 1
        components = connected_components_count(local_count, sorted_local_edges)
        cycle_rank = max(0, len(sorted_local_edges) - local_count + components)
        local_hidden = hidden[local]
        local_projected = pca_project(local_hidden, dims=3)
        local_nll = nll[local]
        mean_nll = float(np.mean(local_nll)) if local_nll.size else 0.0
        mass = float(local_count + len(sorted_local_edges) + triangles)
        max_possible_edges = max(1, local_count * (local_count - 1) // 2)
        max_possible_triangles = max(1, local_count * (local_count - 1) * (local_count - 2) // 6)
        topological_scalars = np.asarray(
            [
                local_count / max(1, max_vertices),
                len(sorted_local_edges) / max_possible_edges,
                triangles / max_possible_triangles,
                components / max(1, local_count),
                cycle_rank / max(1, max_possible_edges),
                radius / (global_radius + 1e-8),
                mean_nll,
                float(np.max(local_nll)) if local_nll.size else mean_nll,
                mass / max(1, max_vertices + max_possible_edges + max_possible_triangles),
            ],
            dtype=np.float32,
        )
        signature = np.concatenate(
            [
                np.mean(local_hidden, axis=0),
                np.std(local_hidden, axis=0),
                topological_scalars,
            ]
        )
        signatures.append(signature)
        mean_nlls.append(mean_nll)
        masses.append(mass)
        rows.append(
            {
                "anchor": int(anchor),
                "anchor_local_index": int(local_index.get(anchor, 0)),
                "vertices": [int(vertex) for vertex in local],
                "vertex_count": int(local_count),
                "local_edges": [[int(src), int(dst)] for src, dst in sorted_local_edges],
                "local_edge_count": int(len(sorted_local_edges)),
                "triangle_count": int(triangles),
                "component_count": int(components),
                "cycle_rank": float(cycle_rank),
                "filtration_radius": float(radius),
                "mean_nll": float(mean_nll),
                "max_nll": float(np.max(local_nll)) if local_nll.size else float(mean_nll),
                "simplicial_mass": float(mass),
                "local_points": local_projected.astype(float).round(6).tolist(),
                "local_nll": np.asarray(local_nll, dtype=np.float64).round(6).tolist(),
            }
        )
    projected = pca_project(np.stack(signatures, axis=0), dims=3)
    return (
        projected,
        np.asarray(mean_nlls, dtype=np.float32),
        np.asarray(masses, dtype=np.float32),
        trajectory_edges,
        rows,
    )


def reasoning_level_step_groups(
    hidden: np.ndarray,
    nll: np.ndarray,
    edges: np.ndarray,
    *,
    causal_rank: np.ndarray | None = None,
    max_levels: int = 8,
    max_branches: int = 3,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Group token embeddings into readable graph-of-thought reasoning steps.

    The grouping is deterministic and uses actual trajectory information:
    causal/order rank gives the reasoning level, and embedding-space PC2
    quantiles split each level into branch lanes.  Actual graph edges crossing
    groups define branch/merge edges when present; decode-order edges are kept
    separately by the interactive plot.
    """

    emb = np.asarray(hidden, dtype=np.float64)
    n = int(emb.shape[0])
    if n <= 0:
        return [], np.zeros((0, 2), dtype=np.int64)
    values = np.asarray(nll, dtype=np.float64).reshape(-1)
    if values.shape[0] < n:
        values = np.pad(values, (0, n - values.shape[0]), constant_values=0.0)
    if causal_rank is None or np.asarray(causal_rank).reshape(-1).shape[0] < n:
        order_key = np.arange(n, dtype=np.float64)
    else:
        raw_rank = np.asarray(causal_rank, dtype=np.float64).reshape(-1)[:n]
        finite = np.isfinite(raw_rank)
        if not finite.any():
            order_key = np.arange(n, dtype=np.float64)
        else:
            replacement = float(np.nanmax(raw_rank[finite]) + 1.0)
            order_key = np.where(finite, raw_rank, replacement)
    sorted_idx = np.lexsort((np.arange(n), order_key))
    levels = max(1, min(int(max_levels), n))
    bins = np.array_split(sorted_idx, levels)
    proj = pca_project(emb, dims=3)
    rows: list[dict[str, Any]] = []
    vertex_to_group: dict[int, int] = {}
    for level_idx, bin_vertices in enumerate(bins):
        vertices = [int(v) for v in bin_vertices.tolist()]
        if not vertices:
            continue
        branch_count = max(1, min(int(max_branches), len(vertices)))
        branch_key = proj[vertices, 1] if len(vertices) > 1 else np.zeros((len(vertices),), dtype=np.float64)
        branch_order = [vertices[i] for i in np.argsort(branch_key)]
        branch_bins = np.array_split(np.asarray(branch_order, dtype=np.int64), branch_count)
        for branch_idx, branch_vertices_raw in enumerate(branch_bins):
            branch_vertices = [int(v) for v in branch_vertices_raw.tolist()]
            if not branch_vertices:
                continue
            group_id = len(rows)
            for vertex in branch_vertices:
                vertex_to_group[vertex] = group_id
            local_points = pca_project(emb[branch_vertices], dims=3)
            local_dist = np.linalg.norm(local_points[:, None, :] - local_points[None, :, :], axis=-1)
            local_positive = local_dist[local_dist > 1e-8]
            local_radius = float(np.quantile(local_positive, 0.55)) if local_positive.size else 0.0
            local_edges, local_triangles, components, cycle_rank = _rips_edges_triangles(local_points, local_radius)
            rows.append(
                {
                    "step_id": int(group_id),
                    "level": int(level_idx),
                    "branch": int(branch_idx),
                    "vertices": branch_vertices,
                    "vertex_count": int(len(branch_vertices)),
                    "mean_nll": float(np.mean(values[branch_vertices])) if branch_vertices else 0.0,
                    "max_nll": float(np.max(values[branch_vertices])) if branch_vertices else 0.0,
                    "local_edges": [[int(a), int(b)] for a, b in local_edges],
                    "local_edge_count": int(len(local_edges)),
                    "triangle_count": int(len(local_triangles)),
                    "component_count": int(components),
                    "cycle_rank": int(cycle_rank),
                    "local_points": local_points.astype(float).round(6).tolist(),
                    "local_nll": np.asarray(values[branch_vertices], dtype=np.float64).round(6).tolist(),
                }
            )
    dag_edges: set[tuple[int, int]] = set()
    for src_raw, dst_raw in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        src = int(src_raw)
        dst = int(dst_raw)
        if src in vertex_to_group and dst in vertex_to_group:
            gs = vertex_to_group[src]
            gd = vertex_to_group[dst]
            if gs != gd and rows[gs]["level"] <= rows[gd]["level"]:
                dag_edges.add((gs, gd))
    return rows, np.asarray(sorted(dag_edges), dtype=np.int64).reshape(-1, 2)


def _radius_schedule_from_points(points: np.ndarray, *, levels: int = 6) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return np.asarray([0.0], dtype=np.float64)
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    positive = dist[dist > 1e-8]
    if positive.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    lo = float(np.quantile(positive, 0.10))
    hi = float(np.quantile(positive, 0.82))
    if hi <= lo:
        hi = float(np.max(positive))
    return np.linspace(0.0, max(hi, lo, 1e-6), num=max(2, int(levels)))


def _rips_edges_triangles(points: np.ndarray, radius: float) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], int, int]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = int(pts.shape[0])
    if n <= 0:
        return [], [], 0, 0
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if dist[i, j] <= float(radius)]
    edge_set = set(edges)
    triangles: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                continue
            for k in range(j + 1, n):
                if (i, k) in edge_set and (j, k) in edge_set:
                    triangles.append((i, j, k))
    components = connected_components_count(n, edges)
    cycle_rank = max(0, len(edges) - n + components)
    return edges, triangles, components, cycle_rank


def _rips_edges_triangles_any(points: np.ndarray, radius: float) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], int, int]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        pts = pts.reshape(-1, 1)
    n = int(pts.shape[0])
    if n <= 0:
        return [], [], 0, 0
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if dist[i, j] <= float(radius)]
    edge_set = set(edges)
    triangles: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                continue
            for k in range(j + 1, n):
                if (i, k) in edge_set and (j, k) in edge_set:
                    triangles.append((i, j, k))
    components = connected_components_count(n, edges)
    cycle_rank = max(0, len(edges) - n + components)
    return edges, triangles, components, cycle_rank


def _standardize_embedding_points(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2:
        x = x.reshape(-1, 1)
    if x.shape[0] == 0:
        return np.zeros((0, max(1, x.shape[1] if x.ndim == 2 else 1)), dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x, axis=0, keepdims=True)
    scale = np.std(x, axis=0, keepdims=True)
    x = x / np.maximum(scale, 1e-8)
    row_norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(row_norm.mean(), 1e-8)


def _radius_schedule_from_embeddings(points: np.ndarray, *, levels: int = 6) -> np.ndarray:
    pts = _standardize_embedding_points(points)
    if pts.shape[0] < 2:
        return np.asarray([0.0], dtype=np.float64)
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    positive = dist[dist > 1e-8]
    if positive.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    return np.linspace(0.0, float(np.quantile(positive, 0.72)), num=max(2, int(levels)))


def _level_prefix_sizes(n: int, *, levels: int = 6) -> list[int]:
    if n <= 0:
        return [0]
    return sorted({max(1, min(n, int(math.ceil(n * frac)))) for frac in np.linspace(1.0 / max(1, levels), 1.0, max(1, levels))})


def _step_rows_to_centroids(projected: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    pts = np.asarray(projected, dtype=np.float64).reshape(-1, 3)
    centroids: list[np.ndarray] = []
    for row in rows:
        vertices = [int(v) for v in row.get("vertices", []) if 0 <= int(v) < pts.shape[0]]
        if vertices:
            centroids.append(np.mean(pts[vertices], axis=0))
    if not centroids:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(centroids, axis=0).astype(np.float64)


def _dotted_3d_segments(points: np.ndarray, edges: list[tuple[int, int]], *, dash_parts: int = 10) -> tuple[list[float], list[float], list[float]]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for src, dst in edges:
        if not (0 <= src < pts.shape[0] and 0 <= dst < pts.shape[0]):
            continue
        a = pts[src]
        b = pts[dst]
        for part in range(max(2, dash_parts)):
            if part % 2:
                continue
            t0 = part / max(1, dash_parts)
            t1 = min(1.0, (part + 0.58) / max(1, dash_parts))
            p0 = a * (1.0 - t0) + b * t0
            p1 = a * (1.0 - t1) + b * t1
            xs.extend([float(p0[0]), float(p1[0]), None])
            ys.extend([float(p0[1]), float(p1[1]), None])
            zs.extend([float(p0[2]), float(p1[2]), None])
    return xs, ys, zs


def _edge_line_xyz(points: np.ndarray, edges: list[tuple[int, int]]) -> tuple[list[float], list[float], list[float]]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for src, dst in edges:
        if 0 <= src < pts.shape[0] and 0 <= dst < pts.shape[0]:
            a, b = pts[src], pts[dst]
            xs.extend([float(a[0]), float(b[0]), None])
            ys.extend([float(a[1]), float(b[1]), None])
            zs.extend([float(a[2]), float(b[2]), None])
    return xs, ys, zs


def _cone_trace_for_edges(points: np.ndarray, edges: list[tuple[int, int]], *, color: str, name: str, visible: bool) -> go.Cone:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    locs: list[np.ndarray] = []
    vecs: list[np.ndarray] = []
    for src, dst in edges:
        if 0 <= src < pts.shape[0] and 0 <= dst < pts.shape[0]:
            a, b = pts[src], pts[dst]
            v = b - a
            norm = float(np.linalg.norm(v))
            if norm <= 1e-9:
                continue
            locs.append(a + 0.55 * v)
            vecs.append(v / norm)
    if not locs:
        locs = [np.zeros(3, dtype=np.float64)]
        vecs = [np.zeros(3, dtype=np.float64)]
    xyz = np.stack(locs, axis=0)
    uvw = np.stack(vecs, axis=0)
    return go.Cone(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        u=uvw[:, 0],
        v=uvw[:, 1],
        w=uvw[:, 2],
        sizemode="absolute",
        sizeref=0.08,
        anchor="tail",
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name=name,
        visible=visible,
        opacity=0.48,
        hoverinfo="skip",
    )


def _token_color_values(nll: np.ndarray, available: bool) -> tuple[np.ndarray, str, str]:
    values = np.asarray(nll, dtype=np.float64).reshape(-1)
    if not bool(available):
        return np.zeros_like(values), "NLL unavailable", "Greys"
    values = np.nan_to_num(values, nan=float(np.nanmean(values)) if np.isfinite(np.nanmean(values)) else 0.0)
    return values, "node NLL", "Turbo"


def write_interactive_reasoning_step_simplex_tree_3d(
    projected: np.ndarray,
    nll: np.ndarray,
    rows: list[dict[str, Any]],
    out: Path,
    *,
    nll_available: bool,
    nll_source: str,
    step_edges: np.ndarray | None = None,
) -> None:
    """Render token subcollections and per-step filtered simplex trees in 3D PCA.

    The simplicial computations are performed on the per-step token subcollections.
    PCA is only the visual coordinate system.  Two independent controls expose
    the filtration: reasoning level selects a prefix of the graph-of-thought
    trajectory, and radius selects the Vietoris-Rips scale inside each step.
    """

    out.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(projected, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        points = np.zeros((1, 3), dtype=np.float64)
    if not rows:
        rows = [
            {
                "step_id": 0,
                "level": 0,
                "branch": 0,
                "vertices": [int(i) for i in range(points.shape[0])],
                "mean_nll": 0.0,
                "max_nll": 0.0,
            }
        ]
    values, color_label, colorscale = _token_color_values(nll, bool(nll_available))
    if values.shape[0] < points.shape[0]:
        values = np.pad(values, (0, points.shape[0] - values.shape[0]), constant_values=0.0)
    centroids = _step_rows_to_centroids(points, rows)
    if centroids.shape[0] == 0:
        centroids = points[:1].copy()
    raw_step_edges = np.asarray(step_edges if step_edges is not None else np.zeros((0, 2)), dtype=np.int64).reshape(-1, 2)
    levels = sorted({int(row.get("level", idx)) for idx, row in enumerate(rows)})
    if not levels:
        levels = [0]
    max_level = max(levels)
    radius_schedule = _radius_schedule_from_points(points, levels=6)
    fig = go.Figure()
    pair_trace_indices: dict[str, list[int]] = {}
    all_step_payload: dict[str, Any] = {
        "schema": "toricgt.reasoning_step_simplex_tree_3d.v1",
        "filtration": "two_parameter_reasoning_level_and_radius",
        "nll_source": str(nll_source),
        "nll_available": bool(nll_available),
        "reasoning_levels": [int(v) for v in levels],
        "radius_schedule": [float(v) for v in radius_schedule],
        "steps": [],
    }

    for level_idx, level in enumerate(levels):
        active_step_indices = [
            idx for idx, row in enumerate(rows[: centroids.shape[0]]) if int(row.get("level", idx)) <= int(level)
        ]
        active_vertex_set: set[int] = set()
        for idx in active_step_indices:
            active_vertex_set.update(int(v) for v in rows[idx].get("vertices", []) if 0 <= int(v) < points.shape[0])
        active_vertices = sorted(active_vertex_set)
        for radius_idx, radius in enumerate(radius_schedule):
            visible = level_idx == 0 and radius_idx == 0
            start_idx = len(fig.data)
            local_edges_global: set[tuple[int, int]] = set()
            local_summaries: list[dict[str, Any]] = []
            for step_idx in active_step_indices:
                row = rows[step_idx]
                vertices = [int(v) for v in row.get("vertices", []) if 0 <= int(v) < points.shape[0]]
                if not vertices:
                    continue
                local_points = points[vertices]
                local_edges, triangles, components, cycle_rank = _rips_edges_triangles(local_points, float(radius))
                for src, dst in local_edges:
                    local_edges_global.add(tuple(sorted((vertices[src], vertices[dst]))))
                local_summaries.append(
                    {
                        "reasoning_step": int(step_idx),
                        "level": int(row.get("level", step_idx)),
                        "branch": int(row.get("branch", 0)),
                        "vertices": vertices,
                        "radius": float(radius),
                        "simplex_tree": {
                            "num_vertices": int(len(vertices)),
                            "num_edges": int(len(local_edges)),
                            "num_triangles": int(len(triangles)),
                            "component_count": int(components),
                            "cycle_rank": int(cycle_rank),
                        },
                        "mean_nll": float(np.mean(values[vertices])) if bool(nll_available) and vertices else None,
                    }
                )
            if level_idx == 0 and radius_idx == 0:
                all_step_payload["steps"] = local_summaries

            token_xyz = points[active_vertices] if active_vertices else np.zeros((0, 3), dtype=np.float64)
            token_values = values[active_vertices] if active_vertices else np.zeros((0,), dtype=np.float64)
            token_custom = [
                [int(vertex), "unavailable" if not bool(nll_available) else f"{float(values[vertex]):.6g}"]
                for vertex in active_vertices
            ]
            edge_xyz = _edge_line_xyz(points, sorted(local_edges_global))

            active_set = set(active_step_indices)
            actual_edges = [
                (int(src), int(dst))
                for src, dst in raw_step_edges.tolist()
                if int(src) in active_set and int(dst) in active_set and 0 <= int(src) < centroids.shape[0] and 0 <= int(dst) < centroids.shape[0]
            ]
            actual_xyz = _edge_line_xyz(centroids, actual_edges)
            decode_edges = [
                (active_step_indices[idx], active_step_indices[idx + 1])
                for idx in range(max(0, len(active_step_indices) - 1))
                if int(level) > min(levels)
            ]
            dotted_xyz = _dotted_3d_segments(centroids, decode_edges)
            centroid_values = []
            centroid_custom = []
            centroid_text = []
            for step_idx in active_step_indices:
                row = rows[step_idx]
                vertices = [int(v) for v in row.get("vertices", []) if 0 <= int(v) < values.shape[0]]
                step_value = float(np.mean(values[vertices])) if bool(nll_available) and vertices else 0.0
                summary = next((item for item in local_summaries if int(item["reasoning_step"]) == int(step_idx)), {})
                centroid_values.append(step_value)
                centroid_text.append(f"L{int(row.get('level', step_idx))}:B{int(row.get('branch', 0))}")
                centroid_custom.append(
                    [
                        int(step_idx),
                        int(row.get("level", step_idx)),
                        int(row.get("branch", 0)),
                        "unavailable" if not bool(nll_available) else f"{step_value:.6g}",
                        json.dumps(summary, sort_keys=True),
                    ]
                )

            fig.add_trace(
                go.Scatter3d(
                    x=token_xyz[:, 0] if token_xyz.size else [],
                    y=token_xyz[:, 1] if token_xyz.size else [],
                    z=token_xyz[:, 2] if token_xyz.size else [],
                    mode="markers",
                    marker={
                        "size": 5,
                        "color": token_values,
                        "colorscale": colorscale,
                        "showscale": visible,
                        "colorbar": {"title": color_label},
                    },
                    name="active token embedding vectors",
                    visible=visible,
                    customdata=token_custom,
                    hovertemplate="token %{customdata[0]}<br>NLL %{customdata[1]}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=edge_xyz[0],
                    y=edge_xyz[1],
                    z=edge_xyz[2],
                    mode="lines",
                    line={"color": "rgba(55,232,255,0.34)", "width": 2},
                    name="radius-controlled local simplex-tree edges",
                    visible=visible,
                    hoverinfo="skip",
                )
            )
            active_centroids = centroids[active_step_indices] if active_step_indices else np.zeros((0, 3), dtype=np.float64)
            fig.add_trace(
                go.Scatter3d(
                    x=active_centroids[:, 0] if active_centroids.size else [],
                    y=active_centroids[:, 1] if active_centroids.size else [],
                    z=active_centroids[:, 2] if active_centroids.size else [],
                    mode="markers+text",
                    marker={"size": 10, "color": centroid_values, "colorscale": colorscale, "line": {"color": "#e8fbff", "width": 1}},
                    text=centroid_text,
                    textposition="top center",
                    name="reasoning-step simplex-tree nodes",
                    visible=visible,
                    customdata=centroid_custom,
                    hovertemplate="step %{customdata[0]} L%{customdata[1]} B%{customdata[2]}<br>NLL %{customdata[3]}<br>simplex_tree_payload=%{customdata[4]}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=actual_xyz[0],
                    y=actual_xyz[1],
                    z=actual_xyz[2],
                    mode="lines",
                    line={"color": "rgba(140,255,106,0.45)", "width": 4},
                    name="graph-of-thought branch/merge edges",
                    visible=visible,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=dotted_xyz[0],
                    y=dotted_xyz[1],
                    z=dotted_xyz[2],
                    mode="lines",
                    line={"color": "rgba(255,211,155,0.50)", "width": 3},
                    name="reasoning-level slider dotted decode order",
                    visible=visible and bool(decode_edges),
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                _cone_trace_for_edges(
                    centroids,
                    decode_edges,
                    color="#ffd39b",
                    name="half-arrows appear after reasoning level advances",
                    visible=visible and bool(decode_edges),
                )
            )
            pair_trace_indices[f"{level_idx}|{radius_idx}"] = list(range(start_idx, len(fig.data)))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#030712",
        plot_bgcolor="#07111f",
        title="Reasoning Trajectory Simplex Trees In 3D PCA",
        scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"},
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
        height=760,
    )
    payload = json.dumps(all_step_payload, sort_keys=True)
    pair_payload = json.dumps(pair_trace_indices, sort_keys=True)
    radii_payload = json.dumps([float(v) for v in radius_schedule])
    levels_payload = json.dumps([int(v) for v in levels])
    html_text = fig.to_html(include_plotlyjs="cdn", full_html=True, div_id="reasoning_step_simplex_tree_3d")
    click_panel = f"""
<section style="max-width:1180px;margin:0 auto 24px;padding:16px;background:#07111f;border:1px solid rgba(55,232,255,.26);border-radius:8px;color:#e8fbff;font-family:Inter,Arial,sans-serif">
  <h2 style="margin:0 0 8px;font-size:18px">Two-Parameter Simplex Tree Payload</h2>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:12px 0">
    <label>reasoning level <strong id="reasoning_level_label">0</strong><br>
      <input id="reasoning_level_slider" type="range" min="0" max="{max(0, len(levels) - 1)}" value="0" step="1" style="width:100%">
    </label>
    <label>radius <strong id="radius_label">{float(radius_schedule[0]):.4f}</strong><br>
      <input id="radius_slider" type="range" min="0" max="{max(0, len(radius_schedule) - 1)}" value="0" step="1" style="width:100%">
    </label>
  </div>
  <p style="color:#91a8b7">The reasoning-level slider reveals prefix levels of the graph-of-thought trajectory. Dotted decode-order edges and half-arrows appear only after the level advances. The radius slider adds Rips edges inside each step subcollection.</p>
  <pre id="simplex_tree_payload" style="white-space:pre-wrap;color:#c8f7ff;background:#020713;padding:12px;border-radius:6px;max-height:280px;overflow:auto">{html.escape(payload)}</pre>
</section>
<script type="application/json" id="simplex_tree_trace_index">{html.escape(pair_payload)}</script>
<script type="application/json" id="simplex_tree_levels">{html.escape(levels_payload)}</script>
<script type="application/json" id="simplex_tree_radii">{html.escape(radii_payload)}</script>
<script>
const gd = document.getElementById('reasoning_step_simplex_tree_3d');
const traceIndex = JSON.parse(document.getElementById('simplex_tree_trace_index').textContent);
const levels = JSON.parse(document.getElementById('simplex_tree_levels').textContent);
const radii = JSON.parse(document.getElementById('simplex_tree_radii').textContent);
function setSimplexTreeScene() {{
  const l = Number(document.getElementById('reasoning_level_slider').value);
  const r = Number(document.getElementById('radius_slider').value);
  document.getElementById('reasoning_level_label').textContent = levels[l];
  document.getElementById('radius_label').textContent = Number(radii[r]).toFixed(4);
  const n = gd ? gd.data.length : 0;
  const visible = Array(n).fill(false);
  const active = traceIndex[`${{l}}|${{r}}`] || [];
  active.forEach((idx) => {{ if (idx >= 0 && idx < visible.length) visible[idx] = true; }});
  if (gd) Plotly.restyle(gd, {{visible: visible}}, Array.from({{length:n}}, (_, i) => i));
}}
document.getElementById('reasoning_level_slider').addEventListener('input', setSimplexTreeScene);
document.getElementById('radius_slider').addEventListener('input', setSimplexTreeScene);
if (gd) {{
  gd.on('plotly_click', function(data) {{
    const point = data.points && data.points[0];
    const cd = point && point.customdata;
    const panel = document.getElementById('simplex_tree_payload');
    if (panel && cd) {{
      panel.textContent = Array.isArray(cd) && cd.length > 2 ? cd[2] : JSON.stringify(cd, null, 2);
    }}
  }});
}}
</script>
"""
    out.write_text(html_text.replace("</body>", click_panel + "</body>"), encoding="utf-8")


def _step_persistence_vector(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    values: list[float] = []
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    for radius in radii:
        edges, triangles, components, cycle_rank = _rips_edges_triangles(pts, float(radius))
        n = max(1, pts.shape[0])
        values.extend(
            [
                float(pts.shape[0]),
                float(len(edges)),
                float(len(triangles)),
                float(components),
                float(cycle_rank),
                float(len(edges)) / max(1, n * (n - 1) / 2),
            ]
        )
    return np.asarray(values, dtype=np.float64)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.shape[0]:
        size = min(x.shape[0], y.shape[0])
        x = x[:size]
        y = y[:size]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def _joint_standardize_pair(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2:
        raise ValueError("source and target embeddings must be 2D")
    if src.shape[1] != dst.shape[1]:
        dim = min(src.shape[1], dst.shape[1])
        src = src[:, :dim]
        dst = dst[:, :dim]
    joined = np.concatenate([src, dst], axis=0)
    joined = np.nan_to_num(joined, nan=0.0, posinf=0.0, neginf=0.0)
    joined = joined - joined.mean(axis=0, keepdims=True)
    scale = joined.std(axis=0, keepdims=True)
    joined = joined / np.maximum(scale, 1e-8)
    return joined[: src.shape[0]], joined[src.shape[0] :]


def _trajectory_ph_signature(hidden: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    points = np.asarray(hidden, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        return np.zeros((1,), dtype=np.float32), {
            "backend_gudhi": 0.0,
            "points": float(points.shape[0] if points.ndim == 2 else 0),
            "vector_norm": 0.0,
        }
    cfg = GudhiPersistenceConfig(
        max_points=max(2, min(32, int(points.shape[0]))),
        max_dimension=2,
        landscape_resolution=24,
        landscape_layers=3,
        image_resolution=8,
        macaulay2_resolutions=False,
    )
    return vectorized_point_cloud_signature(points, cfg)


def _radius_schedule_from_pair(source: np.ndarray, target: np.ndarray, *, levels: int = 6) -> np.ndarray:
    src, dst = _joint_standardize_pair(source, target)
    joined = np.concatenate([src, dst], axis=0)
    if joined.shape[0] < 2:
        return np.asarray([0.0], dtype=np.float64)
    dist = np.linalg.norm(joined[:, None, :] - joined[None, :, :], axis=-1)
    positive = dist[dist > 1e-8]
    if positive.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    return np.linspace(0.0, float(np.quantile(positive, 0.72)), num=max(2, int(levels)))


def _nearest_vertex_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape[0] == 0 or target.shape[0] == 0:
        return np.zeros((source.shape[0],), dtype=np.int64)
    dist = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=-1)
    return np.argmin(dist, axis=1).astype(np.int64)


def _full_trajectory_simplicial_map_audit(
    source_hidden: np.ndarray,
    target_hidden: np.ndarray,
    *,
    levels: int = 6,
    radii: int = 6,
) -> dict[str, Any]:
    """Audit nearest-neighbor maps between full filtered trajectory complexes."""

    src_all, dst_all = _joint_standardize_pair(source_hidden, target_hidden)
    src_levels = _level_prefix_sizes(src_all.shape[0], levels=levels)
    dst_levels = _level_prefix_sizes(dst_all.shape[0], levels=levels)
    radius_grid = _radius_schedule_from_pair(source_hidden, target_hidden, levels=radii)
    edge_scores: list[float] = []
    triangle_scores: list[float] = []
    coverage_scores: list[float] = []
    rows: list[dict[str, Any]] = []
    for level_idx, (src_count, dst_count) in enumerate(zip(src_levels, dst_levels)):
        src = src_all[:src_count]
        dst = dst_all[:dst_count]
        vertex_map = _nearest_vertex_map(src, dst)
        for radius_idx, radius in enumerate(radius_grid):
            src_edges, src_triangles, _, _ = _rips_edges_triangles_any(src, float(radius))
            dst_edges, dst_triangles, _, _ = _rips_edges_triangles_any(dst, float(radius))
            dst_edge_set = {tuple(sorted(edge)) for edge in dst_edges}
            dst_triangle_set = {tuple(sorted(tri)) for tri in dst_triangles}
            valid_edges = 0
            for src_i, src_j in src_edges:
                mapped = tuple(sorted((int(vertex_map[src_i]), int(vertex_map[src_j]))))
                if mapped[0] == mapped[1] or mapped in dst_edge_set:
                    valid_edges += 1
            valid_triangles = 0
            for src_i, src_j, src_k in src_triangles:
                mapped_vertices = sorted({int(vertex_map[src_i]), int(vertex_map[src_j]), int(vertex_map[src_k])})
                if len(mapped_vertices) < 3 or tuple(mapped_vertices) in dst_triangle_set:
                    valid_triangles += 1
            edge_fraction = float(valid_edges / len(src_edges)) if src_edges else 1.0
            triangle_fraction = float(valid_triangles / len(src_triangles)) if src_triangles else 1.0
            coverage = float(len(set(int(v) for v in vertex_map.tolist())) / max(1, dst.shape[0]))
            if src_edges:
                edge_scores.append(edge_fraction)
            if src_triangles:
                triangle_scores.append(triangle_fraction)
            coverage_scores.append(coverage)
            rows.append(
                {
                    "level_index": int(level_idx),
                    "source_prefix_vertices": int(src_count),
                    "target_prefix_vertices": int(dst_count),
                    "radius_index": int(radius_idx),
                    "radius": float(radius),
                    "source_edges": int(len(src_edges)),
                    "target_edges": int(len(dst_edges)),
                    "source_triangles": int(len(src_triangles)),
                    "target_triangles": int(len(dst_triangles)),
                    "edge_valid_fraction": edge_fraction,
                    "triangle_valid_fraction": triangle_fraction,
                    "target_vertex_coverage": coverage,
                }
            )
    edge_mean = float(np.mean(edge_scores)) if edge_scores else 0.0
    triangle_mean = float(np.mean(triangle_scores)) if triangle_scores else 0.0
    coverage_mean = float(np.mean(coverage_scores)) if coverage_scores else 0.0
    score = float(0.45 * edge_mean + 0.25 * triangle_mean + 0.30 * coverage_mean)
    return {
        "simplicial_edge_valid_fraction": edge_mean,
        "simplicial_triangle_valid_fraction": triangle_mean,
        "target_vertex_coverage": coverage_mean,
        "full_filtered_complex_map_score": score,
        "radius_schedule": [float(v) for v in radius_grid],
        "level_radius_rows": rows,
    }


def _step_tree_map_audit(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    src_rows = list(source.get("step_rows", []))
    dst_rows = list(target.get("step_rows", []))
    src_hidden = np.asarray(source.get("hidden", np.zeros((0, 1))), dtype=np.float64)
    dst_hidden = np.asarray(target.get("hidden", np.zeros((0, 1))), dtype=np.float64)
    if not src_rows or not dst_rows:
        return {"trajectory_tree_map_score": 0.0, "step_edge_valid_fraction": 0.0, "step_vertex_coverage": 0.0, "step_map_pairs": []}

    def hidden_centroids(hidden: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
        centroids: list[np.ndarray] = []
        for row in rows:
            vertices = [int(v) for v in row.get("vertices", []) if 0 <= int(v) < hidden.shape[0]]
            if vertices:
                centroids.append(np.mean(hidden[vertices], axis=0))
        return np.stack(centroids, axis=0) if centroids else np.zeros((0, hidden.shape[1] if hidden.ndim == 2 else 1))

    src_centroids_raw = hidden_centroids(src_hidden, src_rows)
    dst_centroids_raw = hidden_centroids(dst_hidden, dst_rows)
    src_centroids, dst_centroids = _joint_standardize_pair(src_centroids_raw, dst_centroids_raw)
    step_map = _nearest_vertex_map(src_centroids, dst_centroids)
    src_edges = np.asarray(source.get("step_edges", np.zeros((0, 2))), dtype=np.int64).reshape(-1, 2)
    dst_edges = np.asarray(target.get("step_edges", np.zeros((0, 2))), dtype=np.int64).reshape(-1, 2)
    if src_edges.size == 0 and src_centroids.shape[0] > 1:
        src_edges = np.asarray([[idx, idx + 1] for idx in range(src_centroids.shape[0] - 1)], dtype=np.int64)
    if dst_edges.size == 0 and dst_centroids.shape[0] > 1:
        dst_edges = np.asarray([[idx, idx + 1] for idx in range(dst_centroids.shape[0] - 1)], dtype=np.int64)
    dst_edge_set = {tuple(sorted((int(src), int(dst)))) for src, dst in dst_edges.tolist() if int(src) != int(dst)}
    valid = 0
    total = 0
    mapped_edges: list[dict[str, Any]] = []
    for src_step, dst_step in src_edges.tolist():
        if not (0 <= int(src_step) < step_map.shape[0] and 0 <= int(dst_step) < step_map.shape[0]):
            continue
        mapped = (int(step_map[int(src_step)]), int(step_map[int(dst_step)]))
        mapped_simplex_edge = tuple(sorted(mapped))
        ok = mapped[0] == mapped[1] or mapped_simplex_edge in dst_edge_set
        valid += int(ok)
        total += 1
        mapped_edges.append({"source_edge": [int(src_step), int(dst_step)], "mapped_target_edge": list(mapped), "valid": bool(ok)})
    edge_fraction = float(valid / total) if total else 0.0
    coverage = float(len(set(int(v) for v in step_map.tolist())) / max(1, dst_centroids.shape[0]))
    score = float(0.70 * edge_fraction + 0.30 * coverage)
    return {
        "trajectory_tree_map_score": score,
        "step_edge_valid_fraction": edge_fraction,
        "step_vertex_coverage": coverage,
        "step_map_pairs": [[int(i), int(v)] for i, v in enumerate(step_map.tolist())],
        "mapped_step_edges": mapped_edges,
    }


def compare_analogy_candidate(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    source_signature = np.asarray(source.get("ph_signature"), dtype=np.float64)
    target_signature = np.asarray(target.get("ph_signature"), dtype=np.float64)
    ph_cosine = _cosine_similarity(source_signature, target_signature)
    simplicial = _full_trajectory_simplicial_map_audit(source["hidden"], target["hidden"])
    step_tree = _step_tree_map_audit(source, target)
    metrics: dict[str, Any] = {
        "source_record_id": int(source["record_id"]),
        "target_record_id": int(target["record_id"]),
        "ph_signature_cosine": float(ph_cosine),
        **simplicial,
        **step_tree,
        "thresholds": dict(ANALOGY_RETRIEVAL_THRESHOLDS),
    }
    accepted = (
        metrics["ph_signature_cosine"] >= ANALOGY_RETRIEVAL_THRESHOLDS["ph_signature_cosine"]
        and metrics["simplicial_edge_valid_fraction"] >= ANALOGY_RETRIEVAL_THRESHOLDS["simplicial_edge_valid_fraction"]
        and metrics["trajectory_tree_map_score"] >= ANALOGY_RETRIEVAL_THRESHOLDS["trajectory_tree_map_score"]
    )
    metrics["accepted_analogy"] = bool(accepted)
    metrics["rejection_reasons"] = [
        name
        for name, threshold in ANALOGY_RETRIEVAL_THRESHOLDS.items()
        if float(metrics.get(name, 0.0)) < float(threshold)
    ]
    return metrics


def _active_vertices_for_level(rows: list[dict[str, Any]], level: int, n: int) -> list[int]:
    vertices: set[int] = set()
    for row in rows:
        if int(row.get("level", 0)) <= int(level):
            vertices.update(int(v) for v in row.get("vertices", []) if 0 <= int(v) < n)
    return sorted(vertices)


def write_interactive_analogical_simplex_maps_3d(
    source: dict[str, Any],
    candidates: list[dict[str, Any]],
    out: Path,
) -> None:
    """Render gated analogical maps between full reasoning trajectory simplex trees."""

    out.parent.mkdir(parents=True, exist_ok=True)
    accepted = [candidate for candidate in candidates if bool(candidate.get("accepted_analogy"))]
    selected = max(
        accepted,
        key=lambda item: (
            float(item.get("ph_signature_cosine", 0.0)),
            float(item.get("full_filtered_complex_map_score", 0.0)),
            float(item.get("trajectory_tree_map_score", 0.0)),
        ),
        default=None,
    )
    browser_candidates = [
        {key: value for key, value in candidate.items() if key != "target_payload"}
        for candidate in candidates
    ]
    payload = {
        "schema": "toricgt.analogical_simplex_maps_3d.v2",
        "retrieval_rule": "emit analogy only when PH signature, full filtered-complex map, and reasoning-step tree-map gates all pass",
        "source_record_id": int(source["record_id"]),
        "thresholds": dict(ANALOGY_RETRIEVAL_THRESHOLDS),
        "candidate_count": int(len(candidates)),
        "accepted_candidate_count": int(len(accepted)),
        "candidates": browser_candidates,
        "selected_target_record_id": None if selected is None else int(selected["target_record_id"]),
    }
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.68, 0.32],
        subplot_titles=("Full reasoning-trajectory simplex-tree map in 3D PCA", "Retrieval gates"),
    )
    metric_names = ["ph_signature_cosine", "simplicial_edge_valid_fraction", "trajectory_tree_map_score"]
    if selected is None:
        fig.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers+text",
                marker={"size": 10, "color": ["#ff6b6b"]},
                text=["No analogy emitted"],
                textposition="top center",
                name="No analogy emitted",
                hovertemplate="No candidate satisfied all analogical retrieval gates.<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=[ANALOGY_RETRIEVAL_THRESHOLDS[name] for name in metric_names],
                marker={"color": ["#ffd166", "#ffd166", "#ffd166"]},
                name="thresholds",
                hovertemplate="%{x}<br>threshold=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#030712",
            plot_bgcolor="#07111f",
            title="Analogical Reasoning Maps Between Full Simplex Trees",
            annotations=[
                {
                    "text": "No analogy emitted: at least one required gate failed.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.36,
                    "y": 0.95,
                    "showarrow": False,
                    "font": {"color": "#ffb4a8", "size": 14},
                }
            ],
            height=760,
            margin={"l": 20, "r": 20, "t": 70, "b": 45},
        )
        fig.update_scenes(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3")
        payload_text = json.dumps(payload, sort_keys=True)
        html_text = fig.to_html(include_plotlyjs="cdn", full_html=True, div_id="analogical_simplex_maps_3d")
        html_text = html_text.replace(
            "</body>",
            f'<script type="application/json" id="vectorized_persistence_comparisons">{html.escape(payload_text)}</script></body>',
        )
        out.write_text(html_text, encoding="utf-8")
        return

    target_payload = selected["target_payload"]
    src_projected = np.asarray(source["projected"], dtype=np.float64).reshape(-1, 3)
    dst_projected = np.asarray(target_payload["projected"], dtype=np.float64).reshape(-1, 3)
    src_values, color_label, colorscale = _token_color_values(source.get("nll", np.zeros((src_projected.shape[0],))), bool(source.get("nll_available", False)))
    dst_values, _, _ = _token_color_values(target_payload.get("nll", np.zeros((dst_projected.shape[0],))), bool(target_payload.get("nll_available", False)))
    offset = float(max(np.ptp(src_projected[:, 0]) if src_projected.size else 1.0, np.ptp(dst_projected[:, 0]) if dst_projected.size else 1.0, 1.0) + 1.5)
    src_vis = src_projected.copy()
    dst_vis = dst_projected.copy()
    src_vis[:, 0] -= offset
    dst_vis[:, 0] += offset
    src_rows = list(source.get("step_rows", []))
    dst_rows = list(target_payload.get("step_rows", []))
    src_levels = sorted({int(row.get("level", 0)) for row in src_rows}) or [0]
    dst_levels = sorted({int(row.get("level", 0)) for row in dst_rows}) or [0]
    level_count = max(len(src_levels), len(dst_levels))
    level_labels = list(range(level_count))
    radius_schedule = [float(v) for v in selected.get("radius_schedule", _radius_schedule_from_pair(source["hidden"], target_payload["hidden"]))]
    src_centroids = _step_rows_to_centroids(src_vis, src_rows)
    dst_centroids = _step_rows_to_centroids(dst_vis, dst_rows)
    step_map_pairs = [(int(a), int(b)) for a, b in selected.get("step_map_pairs", [])]
    pair_trace_indices: dict[str, list[int]] = {}
    for level_idx, _level_label in enumerate(level_labels):
        src_level = src_levels[min(level_idx, len(src_levels) - 1)]
        dst_level = dst_levels[min(level_idx, len(dst_levels) - 1)]
        src_vertices = _active_vertices_for_level(src_rows, int(src_level), src_vis.shape[0])
        dst_vertices = _active_vertices_for_level(dst_rows, int(dst_level), dst_vis.shape[0])
        src_active_steps = [idx for idx, row in enumerate(src_rows[: src_centroids.shape[0]]) if int(row.get("level", 0)) <= int(src_level)]
        dst_active_steps = [idx for idx, row in enumerate(dst_rows[: dst_centroids.shape[0]]) if int(row.get("level", 0)) <= int(dst_level)]
        for radius_idx, radius in enumerate(radius_schedule):
            visible = level_idx == 0 and radius_idx == 0
            start_idx = len(fig.data)
            src_local_edges, _, _, _ = _rips_edges_triangles(src_vis[src_vertices] if src_vertices else np.zeros((0, 3)), float(radius))
            dst_local_edges, _, _, _ = _rips_edges_triangles(dst_vis[dst_vertices] if dst_vertices else np.zeros((0, 3)), float(radius))
            src_edges_global = [(src_vertices[a], src_vertices[b]) for a, b in src_local_edges]
            dst_edges_global = [(dst_vertices[a], dst_vertices[b]) for a, b in dst_local_edges]
            src_edge_xyz = _edge_line_xyz(src_vis, src_edges_global)
            dst_edge_xyz = _edge_line_xyz(dst_vis, dst_edges_global)
            map_edges = [
                (src_idx, dst_idx)
                for src_idx, dst_idx in step_map_pairs
                if src_idx in set(src_active_steps) and dst_idx in set(dst_active_steps) and 0 <= src_idx < src_centroids.shape[0] and 0 <= dst_idx < dst_centroids.shape[0]
            ]
            map_x: list[float] = []
            map_y: list[float] = []
            map_z: list[float] = []
            for src_idx, dst_idx in map_edges:
                a = src_centroids[src_idx]
                b = dst_centroids[dst_idx]
                map_x.extend([float(a[0]), float(b[0]), None])
                map_y.extend([float(a[1]), float(b[1]), None])
                map_z.extend([float(a[2]), float(b[2]), None])
            fig.add_trace(
                go.Scatter3d(
                    x=src_vis[src_vertices, 0] if src_vertices else [],
                    y=src_vis[src_vertices, 1] if src_vertices else [],
                    z=src_vis[src_vertices, 2] if src_vertices else [],
                    mode="markers",
                    marker={"size": 4, "color": src_values[src_vertices] if src_vertices else [], "colorscale": colorscale, "showscale": visible, "colorbar": {"title": color_label}},
                    name="source trajectory token vectors",
                    visible=visible,
                    hovertemplate="source token %{text}<extra></extra>",
                    text=[str(v) for v in src_vertices],
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=dst_vis[dst_vertices, 0] if dst_vertices else [],
                    y=dst_vis[dst_vertices, 1] if dst_vertices else [],
                    z=dst_vis[dst_vertices, 2] if dst_vertices else [],
                    mode="markers",
                    marker={"size": 4, "color": dst_values[dst_vertices] if dst_vertices else [], "colorscale": colorscale, "showscale": False},
                    name="target memory token vectors",
                    visible=visible,
                    hovertemplate="target token %{text}<extra></extra>",
                    text=[str(v) for v in dst_vertices],
                ),
                row=1,
                col=1,
            )
            fig.add_trace(go.Scatter3d(x=src_edge_xyz[0], y=src_edge_xyz[1], z=src_edge_xyz[2], mode="lines", line={"color": "rgba(55,232,255,0.28)", "width": 2}, name="source radius complex", visible=visible, hoverinfo="skip"), row=1, col=1)
            fig.add_trace(go.Scatter3d(x=dst_edge_xyz[0], y=dst_edge_xyz[1], z=dst_edge_xyz[2], mode="lines", line={"color": "rgba(255,211,155,0.28)", "width": 2}, name="target radius complex", visible=visible, hoverinfo="skip"), row=1, col=1)
            fig.add_trace(
                go.Scatter3d(
                    x=src_centroids[src_active_steps, 0] if src_active_steps else [],
                    y=src_centroids[src_active_steps, 1] if src_active_steps else [],
                    z=src_centroids[src_active_steps, 2] if src_active_steps else [],
                    mode="markers+text",
                    marker={"size": 9, "color": "#38e8ff", "line": {"color": "#e8fbff", "width": 1}},
                    text=[f"S{idx}" for idx in src_active_steps],
                    name="source simplex-tree step nodes",
                    visible=visible,
                    hovertemplate="source step %{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=dst_centroids[dst_active_steps, 0] if dst_active_steps else [],
                    y=dst_centroids[dst_active_steps, 1] if dst_active_steps else [],
                    z=dst_centroids[dst_active_steps, 2] if dst_active_steps else [],
                    mode="markers+text",
                    marker={"size": 9, "color": "#ffd39b", "line": {"color": "#e8fbff", "width": 1}},
                    text=[f"T{idx}" for idx in dst_active_steps],
                    name="target memory simplex-tree step nodes",
                    visible=visible,
                    hovertemplate="target step %{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=map_x,
                    y=map_y,
                    z=map_z,
                    mode="lines",
                    line={"color": "rgba(140,255,106,0.72)", "width": 5},
                    name="accepted full_reasoning_trajectory_simplex_tree_map",
                    visible=visible,
                    hovertemplate="accepted_analogy simplicial map<extra></extra>",
                ),
                row=1,
                col=1,
            )
            pair_trace_indices[f"{level_idx}|{radius_idx}"] = list(range(start_idx, len(fig.data)))
    gate_values = [float(selected.get(name, 0.0)) for name in metric_names]
    gate_thresholds = [float(ANALOGY_RETRIEVAL_THRESHOLDS[name]) for name in metric_names]
    fig.add_trace(
        go.Bar(
            x=metric_names,
            y=gate_values,
            marker={"color": ["#38e8ff", "#8cff6a", "#ffd39b"]},
            name="accepted candidate metrics",
            customdata=[json.dumps({name: float(selected.get(name, 0.0)), "threshold": float(ANALOGY_RETRIEVAL_THRESHOLDS[name])}, sort_keys=True) for name in metric_names],
            hovertemplate="%{x}<br>value=%{y:.3f}<br>analogical_simplex_map=%{customdata}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=metric_names,
            y=gate_thresholds,
            mode="markers",
            marker={"size": 12, "symbol": "line-ew", "color": "#ff6b6b"},
            name="gate thresholds",
            hovertemplate="%{x}<br>threshold=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#030712",
        plot_bgcolor="#07111f",
        title="Analogical Reasoning Maps Between Full Simplex Trees",
        height=760,
        margin={"l": 20, "r": 20, "t": 70, "b": 45},
    )
    fig.update_scenes(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3")
    fig.update_yaxes(title_text="gate value", row=1, col=2, range=[0.0, 1.05])
    payload_text = json.dumps(payload, sort_keys=True)
    pair_payload = json.dumps(pair_trace_indices, sort_keys=True)
    radii_payload = json.dumps(radius_schedule)
    levels_payload = json.dumps(level_labels)
    html_text = fig.to_html(include_plotlyjs="cdn", full_html=True, div_id="analogical_simplex_maps_3d")
    control_panel = f"""
<section style="max-width:1180px;margin:0 auto 24px;padding:16px;background:#07111f;border:1px solid rgba(140,255,106,.28);border-radius:8px;color:#e8fbff;font-family:Inter,Arial,sans-serif">
  <h2 style="margin:0 0 8px;font-size:18px">Accepted Analogical Memory Map</h2>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:12px 0">
    <label>reasoning level <strong id="analogical_reasoning_level_label">0</strong><br>
      <input id="reasoning_level_slider" type="range" min="0" max="{max(0, len(level_labels) - 1)}" value="0" step="1" style="width:100%">
    </label>
    <label>radius <strong id="analogical_radius_label">{float(radius_schedule[0]):.4f}</strong><br>
      <input id="radius_slider" type="range" min="0" max="{max(0, len(radius_schedule) - 1)}" value="0" step="1" style="width:100%">
    </label>
  </div>
  <p style="color:#91a8b7">The retrieval head should use these same gates: full-trajectory simplex-tree map, filtered-complex simplicial map, and vectorized PH similarity. PCA is used only for this rendering.</p>
  <pre style="white-space:pre-wrap;color:#c8f7ff;background:#020713;padding:12px;border-radius:6px;max-height:280px;overflow:auto">accepted_analogy {html.escape(payload_text)}</pre>
</section>
<script type="application/json" id="vectorized_persistence_comparisons">{html.escape(payload_text)}</script>
<script type="application/json" id="analogical_trace_index">{html.escape(pair_payload)}</script>
<script type="application/json" id="analogical_levels">{html.escape(levels_payload)}</script>
<script type="application/json" id="analogical_radii">{html.escape(radii_payload)}</script>
<script>
const analogicalGd = document.getElementById('analogical_simplex_maps_3d');
const analogicalTraceIndex = JSON.parse(document.getElementById('analogical_trace_index').textContent);
const analogicalLevels = JSON.parse(document.getElementById('analogical_levels').textContent);
const analogicalRadii = JSON.parse(document.getElementById('analogical_radii').textContent);
function setAnalogicalScene() {{
  const l = Number(document.getElementById('reasoning_level_slider').value);
  const r = Number(document.getElementById('radius_slider').value);
  document.getElementById('analogical_reasoning_level_label').textContent = analogicalLevels[l];
  document.getElementById('analogical_radius_label').textContent = Number(analogicalRadii[r]).toFixed(4);
  const metricTraces = 2;
  const n = analogicalGd ? analogicalGd.data.length : 0;
  const visible = Array(n).fill(false);
  const active = analogicalTraceIndex[`${{l}}|${{r}}`] || [];
  active.forEach((idx) => {{ if (idx >= 0 && idx < n) visible[idx] = true; }});
  for (let i = Math.max(0, n - metricTraces); i < n; i++) visible[i] = true;
  if (analogicalGd) Plotly.restyle(analogicalGd, {{visible: visible}}, Array.from({{length:n}}, (_, i) => i));
}}
document.getElementById('reasoning_level_slider').addEventListener('input', setAnalogicalScene);
document.getElementById('radius_slider').addEventListener('input', setAnalogicalScene);
</script>
"""
    out.write_text(html_text.replace("</body>", control_panel + "</body>"), encoding="utf-8")


def write_interactive_slepian_torus_surface(
    nll: np.ndarray,
    out: Path,
    *,
    nll_available: bool,
) -> None:
    """Render a finite Slepian/Pollak signal as a color field on T^2."""

    out.parent.mkdir(parents=True, exist_ok=True)
    length = max(16, int(np.asarray(nll).reshape(-1).shape[0]))
    cfg = ToricSlepianConfig()
    idx = np.arange(length, dtype=np.float64)
    phase_u = (idx * cfg.theta) % 1.0
    phase_v = (idx * cfg.beta) % 1.0
    energy = np.asarray(nll, dtype=np.float64).reshape(-1)
    if not bool(nll_available) or energy.shape[0] != length:
        energy = None
    audit = toric_slepian_audit(phase_u, phase_v, energy=energy, config=cfg)
    reconstruction = np.asarray(audit["slepian_reconstruction"], dtype=np.float64).reshape(-1)
    envelope = np.asarray(audit["slepian_envelope"], dtype=np.float64).reshape(-1)
    if reconstruction.shape[0] < length:
        reconstruction = np.pad(reconstruction, (0, length - reconstruction.shape[0]), constant_values=0.0)
    if envelope.shape[0] < length:
        envelope = np.pad(envelope, (0, length - envelope.shape[0]), constant_values=1.0)
    grid = 54
    u = np.linspace(0.0, 2.0 * np.pi, grid)
    v = np.linspace(0.0, 2.0 * np.pi, grid)
    uu, vv = np.meshgrid(u, v)
    major, minor = 2.4, 0.72
    x = (major + minor * np.cos(vv)) * np.cos(uu)
    y = (major + minor * np.cos(vv)) * np.sin(uu)
    z = minor * np.sin(vv)
    surface_color = np.zeros_like(uu)
    for r in range(grid):
        for c in range(grid):
            du = np.minimum(np.abs((uu[r, c] / (2.0 * np.pi)) - phase_u), 1.0 - np.abs((uu[r, c] / (2.0 * np.pi)) - phase_u))
            dv = np.minimum(np.abs((vv[r, c] / (2.0 * np.pi)) - phase_v), 1.0 - np.abs((vv[r, c] / (2.0 * np.pi)) - phase_v))
            weight = np.exp(-(du * du + dv * dv) / 0.010)
            surface_color[r, c] = float(np.sum(weight * reconstruction[:length]) / max(np.sum(weight), 1e-12))
    path_x = (major + minor * np.cos(2.0 * np.pi * phase_v)) * np.cos(2.0 * np.pi * phase_u)
    path_y = (major + minor * np.cos(2.0 * np.pi * phase_v)) * np.sin(2.0 * np.pi * phase_u)
    path_z = minor * np.sin(2.0 * np.pi * phase_v)
    custom = [
        [
            int(i),
            float(phase_u[i]),
            float(phase_v[i]),
            float(reconstruction[i]),
            float(envelope[i]),
            None if energy is None else float(energy[i]),
        ]
        for i in range(length)
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=surface_color,
            colorscale="Turbo",
            opacity=0.92,
            colorbar={"title": "Slepian value"},
            name="slepian_torus_surface",
            hovertemplate="torus surface<br>Slepian value=%{surfacecolor:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=path_x,
            y=path_y,
            z=path_z,
            mode="lines+markers",
            line={"color": "#e8fbff", "width": 4},
            marker={"size": 4, "color": envelope[:length], "colorscale": "Viridis", "line": {"color": "#020713", "width": 0.5}},
            name="irrational phase foliation",
            customdata=custom,
            hovertemplate=(
                "trajectory index %{customdata[0]}<br>u=%{customdata[1]:.4f}<br>v=%{customdata[2]:.4f}"
                "<br>Slepian=%{customdata[3]:.4f}<br>envelope=%{customdata[4]:.4f}<br>local NLL/energy=%{customdata[5]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#030712",
        plot_bgcolor="#07111f",
        title=(
            "Slepian/Pollak Prolate Signal On A Foliated Torus "
            f"(concentration {float(audit['slepian_concentration']):.3f})"
        ),
        scene={"xaxis_title": "torus x", "yaxis_title": "torus y", "zaxis_title": "torus z"},
        height=760,
        margin={"l": 0, "r": 0, "t": 58, "b": 0},
    )
    payload = json.dumps(
        {
            "schema": "toricgt.slepian_torus_surface.v1",
            "slepian_customdata": custom,
            "slepian_concentration": float(audit["slepian_concentration"]),
            "slepian_leakage": float(audit["slepian_leakage"]),
            "slepian_bandwidth": float(audit["slepian_bandwidth"]),
        },
        sort_keys=True,
    )
    html_text = fig.to_html(include_plotlyjs="cdn", full_html=True, div_id="slepian_torus_surface")
    html_text = html_text.replace("</body>", f'<script type="application/json" id="slepian_torus_payload">{html.escape(payload)}</script></body>')
    out.write_text(html_text, encoding="utf-8")


def per_node_mse(out_node: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    err = (out_node.float() - target.float()).pow(2).mean(dim=-1)
    return torch.where(node_mask, err, torch.zeros_like(err))


def per_node_lm_nll(outputs: dict[str, torch.Tensor], batch: GraphBatch) -> tuple[torch.Tensor | None, str]:
    logits = outputs.get("lm_logits")
    targets = batch.lm_target_ids
    if logits is None or targets is None:
        return None, "lm_token_nll_unavailable"
    vocab = int(logits.shape[-1])
    node_count = min(int(logits.shape[1]), int(targets.shape[1]), int(batch.node_mask.shape[1]))
    if node_count <= 0 or vocab <= 1:
        return None, "lm_token_nll_unavailable"
    logits = logits[:, :node_count, :].float()
    targets = targets[:, :node_count].to(device=logits.device, dtype=torch.long)
    node_mask = batch.node_mask[:, :node_count].to(device=logits.device, dtype=torch.bool)
    if batch.lm_mask is not None:
        lm_mask = batch.lm_mask[:, :node_count].to(device=logits.device, dtype=torch.bool)
        valid = node_mask & lm_mask
    else:
        valid = node_mask
    valid = valid & (targets >= 0) & (targets < vocab)
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    flat_nll = F.cross_entropy(logits.reshape(-1, vocab), safe_targets.reshape(-1), reduction="none")
    nll = flat_nll.reshape(logits.shape[0], node_count)
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    if node_count < batch.node_mask.shape[1]:
        pad = nll.new_zeros((nll.shape[0], int(batch.node_mask.shape[1]) - node_count))
        nll = torch.cat([nll, pad], dim=1)
    return nll, "lm_token_nll"


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


def token_gt_rich_geometry_record(
    *,
    record_id: int,
    hidden: np.ndarray,
    mean_node_nll: float,
    graph_mse: float,
    topology_cfg: ReasoningTopologyConfig,
    checkpoint_step: int,
    nll_source: str,
) -> dict[str, Any]:
    bpb_proxy = float(mean_node_nll / math.log(2.0)) if math.isfinite(float(mean_node_nll)) else float("nan")
    log_context = {
        "latest_train_bpb": bpb_proxy,
        "latest_val_bpb": bpb_proxy,
        "best_val_bpb": bpb_proxy,
        "latest_bpb": bpb_proxy,
        "bpb_quality": float(max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, bpb_proxy - 1.2))))),
        "loss_quality": float(1.0 / (1.0 + max(0.0, float(graph_mse)))),
    }
    item = {
        "id": f"record_{record_id:03d}_tokengt_node_embedding_trajectory",
        "family": "tokengt_graph_embedding",
        "description": (
            "TokenGT inference/checkpoint node embeddings interpreted through the R97 rich geometry sidecar; "
            f"NLL source={nll_source}, checkpoint_step={checkpoint_step}."
        ),
        "points": np.asarray(hidden, dtype=np.float32),
    }
    return rich_geometry.analyze_record(item, log_context=log_context, topology_config=topology_cfg)


def write_rich_legacy_geometry_outputs(records: list[dict[str, Any]], out_dir: Path, *, checkpoint: str) -> list[str]:
    if not records:
        return []
    out_dir = out_dir.resolve()

    def rel(path: str | Path) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = (out_dir / candidate).resolve()
        try:
            return str(candidate.relative_to(out_dir))
        except ValueError:
            return str(candidate)

    annotated = rich_geometry.annotate_scores(records)
    files: list[str] = []
    for record in annotated:
        files.extend(rich_geometry.plot_record_artifacts(record, out_dir))

    triangle_dir = out_dir / "triangles"
    tetra_dir = out_dir / "tetrahedra"
    for name, spec in rich_geometry.TRIANGLE_SPECS.items():
        path = triangle_dir / f"{name}.png"
        rich_geometry.plot_triangle(annotated, spec, path)
        files.append(str(path))
    for name, spec in rich_geometry.TETRAHEDRON_SPECS.items():
        path = tetra_dir / f"{name}.png"
        rich_geometry.plot_tetrahedron(annotated, spec, path)
        files.append(str(path))
        html_path = path.with_suffix(".html")
        if html_path.exists():
            files.append(str(html_path))

    public_records = [rich_geometry.record_public(record) for record in annotated]
    rich_geometry.write_json(out_dir / "reasoning_geometry_records.json", public_records)
    rich_geometry.write_records_csv(out_dir / "reasoning_geometry_records.csv", annotated)
    rich_geometry.write_json(
        out_dir / "selected_records.json",
        [{"record_id": record["record_id"], "family": record.get("family", "")} for record in annotated],
    )
    fineweb_payload = {
        "source": "tokengt_geometry_sidecar_not_byte_bpb",
        "checkpoint": checkpoint,
        "records": len(annotated),
        "mean_energy": finite_mean([float(record.get("energy_mean", 0.0)) for record in annotated]),
        "mean_graphcg_disentanglement": finite_mean(
            [float(record.get("graphcg_disentanglement_score", 0.0)) for record in annotated]
        ),
        "mean_analogical_map_score": finite_mean(
            [float(record.get("analogical_map_score", 0.0)) for record in annotated]
        ),
        "mean_slepian_concentration": finite_mean(
            [float(record.get("slepian_concentration", 0.0)) for record in annotated]
        ),
        "mean_toric_cca_topology_loss": finite_mean(
            [float(record.get("toric_cca_topology_loss", 0.0)) for record in annotated]
        ),
    }
    rich_geometry.write_json(out_dir / "fineweb_curve_diagnostic_payload.json", fineweb_payload)
    files.extend(
        str(path)
        for path in (
            out_dir / "reasoning_geometry_records.json",
            out_dir / "reasoning_geometry_records.csv",
            out_dir / "selected_records.json",
            out_dir / "fineweb_curve_diagnostic_payload.json",
        )
    )
    return sorted(rel(path) for path in files)


def evaluate_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
    paths = geometry_data_paths(args, flat)
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
    embedding_dir = out_dir / "embeddings"
    records: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    rich_records: list[dict[str, Any]] = []
    embedding_payloads: list[dict[str, Any]] = []
    analogical_payloads: list[dict[str, Any]] = []
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
            node_nll, nll_source = per_node_lm_nll(out, batch)
            node_nll_available = node_nll is not None
            if node_nll is None:
                node_nll = node_energy
                nll_source = "graph_reconstruction_mse_visualization_energy"
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
            nll_cpu = node_nll.detach().float().cpu()
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
                nll = nll_cpu[sample_index, :n].numpy()
                if not np.isfinite(nll).all():
                    nll = np.nan_to_num(nll, nan=float(np.nanmean(nll)) if np.isfinite(np.nanmean(nll)) else 0.0)
                edges = valid_edges(batch.edge_index[sample_index], batch.edge_mask[sample_index], n)
                complex_projected, complex_nll, complex_mass, complex_edges, complex_rows = reasoning_step_complexes(
                    hidden,
                    nll,
                    edges,
                    max_vertices=min(11, max(3, n)),
                )
                causal_rank = None
                if batch.node_causal_rank is not None:
                    causal_rank = batch.node_causal_rank[sample_index, :n].detach().float().cpu().numpy()
                step_rows, step_edges = reasoning_level_step_groups(
                    hidden,
                    nll,
                    edges,
                    causal_rank=causal_rank,
                    max_levels=min(8, max(2, n)),
                    max_branches=3,
                )
                record_id = len(records)
                topology = directed_step_filtration_stats_np(hidden, config=topology_cfg)
                topology_metrics = topology_summary(topology)
                mst = prim_mst_stats(projected.tolist())
                graph_mse = float(energy[energy >= 0].mean()) if energy.size else float(loss.detach().cpu())
                mean_node_nll = float(nll.mean()) if nll.size else graph_mse
                max_node_nll = float(nll.max()) if nll.size else graph_mse
                mean_complex_mass = float(complex_mass.mean()) if complex_mass.size else 0.0
                max_complex_mass = float(complex_mass.max()) if complex_mass.size else 0.0
                mean_complex_vertices = finite_mean([float(row.get("vertex_count", 0.0)) for row in complex_rows])
                mean_complex_edges = finite_mean([float(row.get("local_edge_count", 0.0)) for row in complex_rows])
                mean_complex_triangles = finite_mean([float(row.get("triangle_count", 0.0)) for row in complex_rows])
                mean_complex_cycle_rank = finite_mean([float(row.get("cycle_rank", 0.0)) for row in complex_rows])
                path_speed = np.linalg.norm(np.diff(projected, axis=0), axis=1) if projected.shape[0] > 1 else np.zeros((1,))
                path_smoothness = float(1.0 / (1.0 + np.std(path_speed)))
                resolution_metrics = extract_resolution_metrics(batch_objects[sample_index])
                record = {
                    "record_index": int(record_id),
                    "checkpoint_step": int(payload.get("step", 0) or 0),
                    "graph_reconstruction_mse": graph_mse,
                    "global_masked_mse": float(loss.detach().cpu()),
                    "mean_node_nll": mean_node_nll,
                    "max_node_nll": max_node_nll,
                    "node_nll_source": nll_source,
                    "node_nll_available": float(1.0 if node_nll_available else 0.0),
                    "node_count": float(n),
                    "edge_count": float(edges.shape[0]),
                    "reasoning_step_complex_mean_nll": float(complex_nll.mean()) if complex_nll.size else mean_node_nll,
                    "reasoning_step_complex_mean_mass": mean_complex_mass,
                    "reasoning_step_complex_max_mass": max_complex_mass,
                    "reasoning_step_complex_mean_vertices": mean_complex_vertices,
                    "reasoning_step_complex_mean_edges": mean_complex_edges,
                    "reasoning_step_complex_mean_triangles": mean_complex_triangles,
                    "reasoning_step_complex_mean_cycle_rank": mean_complex_cycle_rank,
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
                if bool(getattr(args, "emit_embedding_payloads", True)):
                    payload_record = write_embedding_payload(
                        embedding_dir,
                        record_id=record_id,
                        checkpoint_step=int(payload.get("step", 0) or 0),
                        hidden=hidden,
                        projected=projected,
                        energy=energy,
                        nll=nll,
                        edges=edges,
                        complex_projected=complex_projected,
                        complex_nll=complex_nll,
                        complex_mass=complex_mass,
                        complex_edges=complex_edges,
                        complex_rows=complex_rows,
                        nll_source=nll_source,
                    )
                    embedding_payloads.append(payload_record)
                    record["embedding_payload_npz"] = payload_record["relative_npz"]
                    record["embedding_payload_json"] = payload_record["relative_json"]
                records.append(record)
                if bool(getattr(args, "rich_legacy_geometry", True)):
                    rich_records.append(
                        token_gt_rich_geometry_record(
                            record_id=record_id,
                            hidden=hidden,
                            mean_node_nll=mean_node_nll,
                            graph_mse=graph_mse,
                            topology_cfg=topology_cfg,
                            checkpoint_step=int(payload.get("step", 0) or 0),
                            nll_source=nll_source,
                        )
                    )
                obj = dict(batch_objects[sample_index])
                obj["record_index"] = int(record_id)
                objects.append(obj)
                prefix = f"record_{record_id:03d}"
                plot_reasoning_trajectory_3d(projected, energy, traj_dir / f"{prefix}_trajectory_3d.png", edges=edges)
                plot_energy_landscape(projected, energy, traj_dir / f"{prefix}_energy_landscape.png", edges=edges)
                plot_pca_nll_4d(projected, nll, traj_dir / f"{prefix}_pca_nll_4d.png", edges=edges)
                plot_reasoning_step_complex_pca_3d(
                    complex_projected,
                    complex_nll,
                    complex_mass,
                    traj_dir / f"{prefix}_complex_pca_nll_3d.png",
                    edges=complex_edges,
                )
                write_interactive_energy_landscape(projected, energy, traj_dir / f"{prefix}_energy_landscape.html", edges=edges)
                write_interactive_reasoning_plot(projected, energy, traj_dir / f"{prefix}_trajectory_3d.html", edges=edges)
                write_interactive_pca_nll_4d(projected, nll, traj_dir / f"{prefix}_pca_nll_4d.html", edges=edges)
                write_interactive_reasoning_step_complex_pca_3d(
                    complex_projected,
                    complex_nll,
                    complex_mass,
                    traj_dir / f"{prefix}_complex_pca_nll_3d.html",
                    edges=complex_edges,
                    complex_rows=complex_rows,
                )
                write_interactive_reasoning_step_simplex_tree_3d(
                    projected,
                    nll,
                    step_rows,
                    traj_dir / f"{prefix}_reasoning_step_simplex_tree_3d.html",
                    nll_available=bool(node_nll_available),
                    nll_source=nll_source,
                    step_edges=step_edges,
                )
                ph_signature, ph_metrics = _trajectory_ph_signature(hidden)
                analogical_payloads.append(
                    {
                        "record_id": int(record_id),
                        "hidden": hidden.astype(np.float32),
                        "projected": projected.astype(np.float32),
                        "nll": nll.astype(np.float32),
                        "nll_available": bool(node_nll_available),
                        "step_rows": step_rows,
                        "step_edges": step_edges.astype(np.int64),
                        "ph_signature": ph_signature.astype(np.float32),
                        "ph_metrics": ph_metrics,
                    }
                )
                write_interactive_slepian_torus_surface(
                    nll,
                    traj_dir / f"{prefix}_slepian_torus_surface.html",
                    nll_available=bool(node_nll_available),
                )
                plot_topology_heatmap(topology, topo_dir / f"{prefix}_topology_heatmaps.png")
            if len(records) >= int(args.records):
                break
    if not records:
        raise RuntimeError("no TokenGT graph records were evaluated")
    if bool(getattr(args, "emit_embedding_payloads", True)):
        embedding_manifest = {
            "schema": "toricgt.embedding_payload_manifest.v1",
            "checkpoint": str(args.checkpoint),
            "records": len(embedding_payloads),
            "payloads": embedding_payloads,
        }
        embedding_dir.mkdir(parents=True, exist_ok=True)
        (embedding_dir / "manifest.json").write_text(
            json.dumps(embedding_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    analogical_dir = out_dir / "analogical"
    analogical_dir.mkdir(parents=True, exist_ok=True)
    for source in analogical_payloads:
        render_candidates: list[dict[str, Any]] = []
        for target_payload in analogical_payloads:
            if int(target_payload["record_id"]) == int(source["record_id"]):
                continue
            comparison = compare_analogy_candidate(source, target_payload)
            comparison["target_payload"] = target_payload
            render_candidates.append(comparison)
        write_interactive_analogical_simplex_maps_3d(
            source,
            render_candidates,
            analogical_dir / f"record_{int(source['record_id']):03d}_analogical_simplex_maps_3d.html",
        )
    return records, objects, rich_records


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


def relative_matches(out_dir: Path, pattern: str) -> list[str]:
    if not pattern:
        return []
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.glob(pattern) if path.is_file())


def build_plot_family_manifest(out_dir: Path) -> dict[str, Any]:
    families = []
    for spec in PLOT_FAMILY_REGISTRY:
        static_pattern = str(spec.get("static_pattern", ""))
        interactive_pattern = str(spec.get("interactive_pattern", ""))
        static_outputs = relative_matches(out_dir, static_pattern)
        interactive_outputs = relative_matches(out_dir, interactive_pattern)
        interactive_required = bool(spec.get("interactive_required", False))
        if static_pattern and not static_outputs:
            status = "missing_static"
        elif interactive_required and not interactive_outputs:
            status = "missing_interactive"
        else:
            status = "ok"
        families.append(
            {
                "family": spec["family"],
                "introduced": spec["introduced"],
                "dimension": spec["dimension"],
                "interactive_required": interactive_required,
                "status": status,
                "static_count": len(static_outputs),
                "interactive_count": len(interactive_outputs),
                "static_outputs": static_outputs,
                "interactive_outputs": interactive_outputs,
                "description": spec["description"],
            }
        )
    required_3d4d = [item for item in families if item["interactive_required"]]
    return {
        "schema": "toricgt.geometry_plot_family_manifest.v1",
        "families": families,
        "previous_families": sorted(item["family"] for item in families if item["introduced"] == "previous"),
        "current_new_families": sorted(
            item["family"]
            for item in families
            if item["introduced"] == "current" or str(item["introduced"]).startswith("staircase_simplextree_slepian_")
        ),
        "union_families": sorted(item["family"] for item in families),
        "interactive_3d4d_complete": all(item["status"] == "ok" for item in required_3d4d),
        "missing_required_interactive": [
            item["family"] for item in required_3d4d if item["status"] == "missing_interactive"
        ],
        "missing_static": [item["family"] for item in families if item["status"] == "missing_static"],
    }


def write_plot_family_manifest(out_dir: Path) -> dict[str, Any]:
    manifest = build_plot_family_manifest(out_dir)
    (out_dir / "plot_family_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Geometry Plot Family Manifest",
        "",
        "This manifest records the union of previous and current geometry-analysis outputs. "
        "All 3D/4D families are expected to have an interactive HTML companion.",
        "",
        f"- Previous families: {', '.join(manifest['previous_families'])}",
        f"- Current new families: {', '.join(manifest['current_new_families'])}",
        f"- 3D/4D interactive complete: {manifest['interactive_3d4d_complete']}",
        "",
        "| family | introduced | dimension | static | interactive | status | description |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for item in manifest["families"]:
        description = str(item["description"]).replace("|", "/")
        row = dict(item)
        row["description"] = description
        lines.append(
            "| {family} | {introduced} | {dimension} | {static_count} | {interactive_count} | {status} | {description} |".format(
                **row,
            )
        )
    lines.append("")
    lines.append("## Required Interactive 3D/4D Families")
    for item in manifest["families"]:
        if item["interactive_required"]:
            lines.append(f"- `{item['family']}`: `{item['status']}`")
    (out_dir / "plot_family_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    def mean(key: str) -> float:
        return finite_mean([float(record.get(key, 0.0)) for record in records])

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_family": "tokengt_graph",
        "records": len(records),
        "oai_competition_bpb_available": 0.0,
        "oai_competition_bpb_note": "TokenGT geometry reports graph reconstruction, topology, and per-node SP1024 LM NLL when FineWeb graph targets are available; byte-normalized BPB is reported by evaluate_tokengt_fineweb_bpb.py.",
        "mean_graph_reconstruction_mse": mean("graph_reconstruction_mse"),
        "best_graph_reconstruction_mse": min(float(record["graph_reconstruction_mse"]) for record in records),
        "mean_node_nll": mean("mean_node_nll"),
        "max_node_nll": max(float(record.get("max_node_nll", 0.0)) for record in records),
        "node_nll_source": records[0].get("node_nll_source", "unknown"),
        "mean_reasoning_step_complex_nll": mean("reasoning_step_complex_mean_nll"),
        "mean_reasoning_step_complex_mass": mean("reasoning_step_complex_mean_mass"),
        "max_reasoning_step_complex_mass": max(
            float(record.get("reasoning_step_complex_max_mass", 0.0)) for record in records
        ),
        "mean_reasoning_step_complex_vertices": mean("reasoning_step_complex_mean_vertices"),
        "mean_reasoning_step_complex_edges": mean("reasoning_step_complex_mean_edges"),
        "mean_reasoning_step_complex_triangles": mean("reasoning_step_complex_mean_triangles"),
        "mean_reasoning_step_complex_cycle_rank": mean("reasoning_step_complex_mean_cycle_rank"),
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
            "analysis_control/checkpoint_step": step,
            "analysis_control/tokengt_geometry/mean_graph_reconstruction_mse": summary["mean_graph_reconstruction_mse"],
            "analysis_control/tokengt_geometry/best_graph_reconstruction_mse": summary["best_graph_reconstruction_mse"],
            "analysis_control/tokengt_geometry/mean_node_nll": summary["mean_node_nll"],
            "analysis_control/tokengt_geometry/max_node_nll": summary["max_node_nll"],
            "analysis_control/tokengt_geometry/mean_reasoning_step_complex_nll": summary[
                "mean_reasoning_step_complex_nll"
            ],
            "analysis_control/tokengt_geometry/mean_reasoning_step_complex_mass": summary[
                "mean_reasoning_step_complex_mass"
            ],
            "analysis_control/tokengt_geometry/mean_reasoning_step_complex_triangles": summary[
                "mean_reasoning_step_complex_triangles"
            ],
            "analysis_control/tokengt_geometry/mean_reasoning_step_complex_cycle_rank": summary[
                "mean_reasoning_step_complex_cycle_rank"
            ],
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
        run.log(organize_wandb_payload(payload))
    finally:
        run.finish()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    out_dir = Path(args.output_dir)
    records, objects, rich_records = evaluate_records(args)
    enrich_scores(records)
    write_csv(out_dir / "tokengt_geometry_records.csv", records)
    (out_dir / "tokengt_geometry_records.json").write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "derived_category_objects.json").write_text(json.dumps(objects, indent=2, sort_keys=True), encoding="utf-8")
    rich_legacy_outputs = (
        write_rich_legacy_geometry_outputs(rich_records, out_dir, checkpoint=str(args.checkpoint))
        if bool(getattr(args, "rich_legacy_geometry", True))
        else []
    )
    for name, spec in TRIANGLE_SPECS.items():
        plot_triangle(records, spec, out_dir / f"{name}_triangle.png")
    plot_manifest = write_plot_family_manifest(out_dir)
    summary = summarize(records, args)
    summary["outputs"] = sorted(path.name for path in out_dir.glob("*") if path.is_file())
    summary["trajectory_html_outputs"] = sorted(
        str(path.relative_to(out_dir)) for path in (out_dir / "trajectories").glob("*_trajectory_3d.html")
    )
    summary["interactive_energy_landscape_outputs"] = sorted(
        str(path.relative_to(out_dir)) for path in (out_dir / "trajectories").glob("*_energy_landscape.html")
    )
    summary["pca_nll_4d_png_outputs"] = sorted(
        str(path.relative_to(out_dir)) for path in (out_dir / "trajectories").glob("*_pca_nll_4d.png")
    )
    summary["interactive_pca_nll_4d_outputs"] = sorted(
        str(path.relative_to(out_dir)) for path in (out_dir / "trajectories").glob("*_pca_nll_4d.html")
    )
    summary["reasoning_step_complex_png_outputs"] = sorted(
        str(path.relative_to(out_dir)) for path in (out_dir / "trajectories").glob("*_complex_pca_nll_3d.png")
    )
    summary["interactive_reasoning_step_complex_outputs"] = sorted(
        str(path.relative_to(out_dir)) for path in (out_dir / "trajectories").glob("*_complex_pca_nll_3d.html")
    )
    summary["plot_family_manifest"] = plot_manifest
    summary["plot_family_manifest_outputs"] = ["plot_family_manifest.json", "plot_family_manifest.md"]
    summary["rich_legacy_geometry_enabled"] = bool(getattr(args, "rich_legacy_geometry", True))
    summary["rich_legacy_geometry_record_count"] = len(rich_records)
    summary["rich_legacy_geometry_outputs"] = rich_legacy_outputs
    summary["embedding_payloads_enabled"] = bool(getattr(args, "emit_embedding_payloads", True))
    summary["embedding_payload_manifest"] = "embeddings/manifest.json" if (out_dir / "embeddings" / "manifest.json").exists() else ""
    summary["embedding_payload_outputs"] = sorted(
        str(path.relative_to(out_dir)) for path in (out_dir / "embeddings").glob("*_embedding_payload.npz")
    )
    (out_dir / "reasoning_geometry_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    step = int(records[0].get("checkpoint_step", 0) or 0)
    log_to_wandb(args.wandb_run_path, summary, step=step)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
