#!/usr/bin/env python3
"""Review every ToricGT sidecar metric logged to W&B for one run.

The OAI baseline trainer logs detailed sidecar metrics under
``toricgt_sidecar/*``.  This script downloads the run history, exports every
observed sidecar metric, attaches a concise mathematical description and
weighting guidance, and computes simple trend/correlation statistics for the
5K restart decision report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIDECAR_PREFIX = "toricgt_sidecar/"
RELATED_PREFIXES = (
    "06_graphcg/",
    "07_topology_geometry/",
    "08_toric_tropical_bgg/",
    "16_status/metrics_status/",
)


@dataclass
class SidecarMetricReview:
    metric: str
    family: str
    goal: str
    n: int
    first_step: float | None
    last_step: float | None
    first_value: float | None
    last_value: float | None
    min_value: float | None
    max_value: float | None
    mean_value: float | None
    slope_per_1k: float | None
    recent_slope_per_1k: float | None
    corr_train_bpb: float | None
    corr_train_loss: float | None
    status: str
    description: str
    weight_guidance: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-path", required=True, help="W&B entity/project/run_id or entity/project/run_name")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recent-fraction", type=float, default=0.30)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument(
        "--include-related-aliases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also include organized aliases for sidecar-related graph/toric/topology namespaces if present.",
    )
    return parser.parse_args()


def download_history(run_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    import wandb

    api = wandb.Api()
    requested = run_path
    parts_for_meta = run_path.split("/")
    entity_for_meta = parts_for_meta[0] if len(parts_for_meta) >= 3 else ""
    project_for_meta = parts_for_meta[1] if len(parts_for_meta) >= 3 else ""
    try:
        run = api.run(run_path)
    except Exception:
        parts = run_path.split("/")
        if len(parts) != 3:
            raise
        entity, project, run_key = parts
        run = None
        for candidate in api.runs(f"{entity}/{project}", per_page=300):
            if candidate.id == run_key or candidate.name == run_key:
                run = candidate
                break
        if run is None:
            raise
    rows = list(run.scan_history(page_size=1000))
    if not rows:
        raise RuntimeError(f"No W&B history rows found for {requested}")
    df = pd.DataFrame(rows)
    if "_step" not in df.columns:
        raise RuntimeError("W&B history has no _step column")
    meta = {
        "requested_run_path": requested,
        "resolved_run_path": f"{entity_for_meta}/{project_for_meta}/{run.id}" if entity_for_meta and project_for_meta else run.id,
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "url": run.url,
        "history_rows": int(len(df)),
        "last_history_step": run.lastHistoryStep,
    }
    return df, meta


def numeric_series(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    values = pd.to_numeric(df[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({"_step": pd.to_numeric(df["_step"], errors="coerce"), "value": values})
    return out.dropna().sort_values("_step")


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return None
    x = x.astype(float)
    y = y.astype(float)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1e-12:
        return None
    return float(np.sum(x * y) / denom)


def slope_per_1k(steps: np.ndarray, values: np.ndarray) -> float | None:
    if len(values) < 3:
        return None
    x = (steps.astype(float) - float(np.min(steps))) / 1000.0
    y = values.astype(float)
    x_center = x - x.mean()
    denom = float(np.sum(x_center * x_center))
    if denom <= 1e-12:
        return None
    return float(np.sum(x_center * (y - y.mean())) / denom)


def paired_corr(df: pd.DataFrame, metric: str, target: str) -> float | None:
    if target not in df.columns:
        return None
    metric_values = pd.to_numeric(df[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
    target_values = pd.to_numeric(df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    pair = pd.DataFrame({"metric": metric_values, "target": target_values}).dropna()
    if len(pair) < 3:
        return None
    return pearson(pair["metric"].to_numpy(dtype=float), pair["target"].to_numpy(dtype=float))


def sidecar_metric_columns(df: pd.DataFrame, include_related_aliases: bool) -> list[str]:
    cols = [str(col) for col in df.columns if str(col).startswith(SIDECAR_PREFIX)]
    if include_related_aliases:
        cols.extend(
            str(col)
            for col in df.columns
            if any(str(col).startswith(prefix) for prefix in RELATED_PREFIXES)
        )
    return sorted(set(cols))


def stripped_name(metric: str) -> str:
    if metric.startswith(SIDECAR_PREFIX):
        return metric[len(SIDECAR_PREFIX) :]
    return metric.split("/", 1)[-1]


def metric_family(metric: str) -> str:
    name = stripped_name(metric)
    if name in {"loss", "toricgt_sidecar_loss", "lm_loss"}:
        return "sidecar_total"
    if name.startswith("graphcg_"):
        return "graphcg_full_rank_concept_chart"
    if name.startswith("analogy_"):
        return "analogical_lattice"
    if name.startswith("tokengt_graph_"):
        return "tokengt_causal_graph_tokenization"
    if name.startswith("trajectory_memory_"):
        return "trajectory_memory_retrieval"
    if name.startswith("toric_geometry_") or name.startswith("toric_") and not name.startswith("toric_bgg") and not name.startswith("toric_cca"):
        return "toric_tropical_geometry"
    if name.startswith("toric_vector_bundle_") or name.startswith("toric_sheaf_"):
        return "toric_vector_bundle_sheaf"
    if name.startswith("toric_bgg_"):
        return "toric_bgg_category_o"
    if name.startswith("koszul_"):
        return "koszul_persistence"
    if name.startswith("toric_cca_"):
        return "combinatorial_commutative_algebra"
    if name.endswith("_active") or name.endswith("_enabled") or name.endswith("_available") or name.endswith("_weight"):
        return "activation_and_weight_status"
    return "other_sidecar"


def metric_goal(metric: str) -> str:
    name = stripped_name(metric)
    if name.endswith("_weight") or name.endswith("_active") or name.endswith("_enabled") or name.endswith("_available"):
        return "status"
    if any(token in name for token in ("allowed_mass", "recall1", "consistency", "margin", "density", "active_1d_cone_mass")):
        return "higher"
    if any(token in name for token in ("entropy", "count", "rank", "chart_dim", "probe_rank", "batch", "backend")):
        return "context"
    if any(token in name for token in ("loss", "residual", "leakage", "ce", "bce", "nonface", "gap", "norm")):
        return "lower"
    if name in {"lm_loss", "loss", "toricgt_sidecar_loss"}:
        return "lower"
    return "context"


def description_for(metric: str) -> str:
    name = stripped_name(metric)
    family = metric_family(metric)
    exact = {
        "lm_loss": "Auxiliary graph-text language-model loss on the curated graph stream; it measures how costly the sidecar batch is under the base model but is not itself added to the sidecar objective.",
        "loss": "Weighted scalar sidecar objective actually backpropagated for the auxiliary ToricGT heads at the log step.",
        "toricgt_sidecar_loss": "Unscaled internal weighted sum of active sidecar objectives before the trainer-level sidecar multiplier.",
        "graphcg_loss": "Full-rank GraphCG concept-chart objective combining basis orthogonality, coordinate covariance control, and axis entropy.",
        "graphcg_orthogonal_loss": "Squared off-diagonal Gram penalty for the learned full-rank GraphCG basis.",
        "graphcg_covariance_loss": "Decorrelation penalty for GraphCG coordinates across hidden-state samples.",
        "graphcg_axis_entropy": "Normalized entropy of concept-axis assignments; it measures whether hidden states use multiple interpretable axes.",
        "graphcg_chart_dim": "Dimension of the active GraphCG chart, expected to equal the hidden width for full-rank training.",
        "analogy_lattice_loss": "Analogical relation loss asking repeated byte-class transitions to share relation vectors and GraphCG-basis reconstructions.",
        "analogy_functor_loss": "Variance of relation vectors within repeated transition classes, used as an approximate functoriality residual.",
        "analogy_basis_loss": "Reconstruction error of relation vectors from GraphCG concept-axis coordinates.",
        "analogy_lattice_margin": "Top-axis separation margin for relation vectors; larger means cleaner analogical basis selection.",
        "analogy_relation_groups": "Number of repeated relation groups available in the sidecar batch.",
        "tokengt_graph_loss": "TokenGT-style causal graph reconstruction loss on sampled token nodes and local one-dimensional edges.",
        "tokengt_graph_edge_bce": "Binary cross-entropy for causal local edge prediction between token nodes.",
        "tokengt_graph_byte_class_loss": "Binary cross-entropy for same-byte-class token relation prediction.",
        "tokengt_graph_edge_density": "Density of target one-dimensional causal graph edges in the sampled token graph.",
        "tokengt_graph_causal_edge_fraction": "Fraction of sampled token pairs that are valid causal local graph edges.",
        "trajectory_memory_loss": "Training objective for analogical trajectory-memory retrieval over in-batch graph-of-thought summaries.",
        "trajectory_memory_ce": "Cross-entropy for selecting the teacher-preferred analogy/memory trajectory.",
        "trajectory_memory_distill_loss": "KL distillation from the teacher similarity distribution over memory candidates.",
        "trajectory_memory_quality_loss": "Regression loss for predicting candidate quality from trajectory summaries.",
        "trajectory_memory_recall1": "Fraction of queries whose top memory matches the teacher argmax.",
        "trajectory_memory_entropy": "Normalized entropy of retrieval probabilities; it measures spread versus collapse.",
        "trajectory_memory_score_gap": "Average top-vs-second retrieval logit gap.",
        "trajectory_memory_persistence_similarity": "Mean similarity of vectorized persistent-homology signatures across memory candidates.",
        "trajectory_memory_persistence_norm": "Norm of vectorized persistence features used by the retrieval head.",
        "trajectory_memory_persistence_entropy": "Entropy of persistence intervals/features in trajectory summaries.",
        "trajectory_memory_persistence_total": "Total persistence magnitude in trajectory summaries.",
        "trajectory_memory_dag_similarity": "Similarity of graph-of-thought DAG branch/merge summaries between candidates.",
        "trajectory_memory_dag_branch_count": "Average branch count in graph-of-thought trajectory summaries.",
        "trajectory_memory_dag_merge_count": "Average merge count in graph-of-thought trajectory summaries.",
        "trajectory_memory_derived_similarity": "Similarity of derived-category feature summaries for trajectory complexes.",
        "trajectory_memory_derived_projective_dimension": "Average projective-dimension proxy in derived-category summaries.",
        "trajectory_memory_derived_regularity": "Average regularity proxy in derived-category summaries.",
        "toric_geometry_loss": "Low-rank toric/tropical geometry objective combining active-face, moment, bend, binomial, Coxeter, braid, and phase-leaf terms.",
        "toric_fan_loss": "Active normal-fan classification and margin loss for tropical ring attention embedded into toric geometry.",
        "toric_active_face_ce": "Cross-entropy for predicting the teacher active face of the Newton polytope.",
        "toric_active_face_margin": "Mean margin between active and competing toric/tropical faces; larger means more stable fan-cell decisions.",
        "toric_active_face_entropy": "Entropy of active-face assignments; it tracks whether fan cells are diverse or collapsed.",
        "toric_bend_loss": "Consistency loss for bends across adjacent toric fan cells.",
        "toric_binomial_loss": "Loss enforcing toric ideal/binomial relation consistency in probe logits.",
        "toric_binomial_residual": "Raw residual for binomial relations among exponent candidates.",
        "toric_binomial_relation_source_exact": "Status flag indicating whether binomial relations came from an exact CAS certificate.",
        "toric_moment_loss": "Moment-map consistency loss between predicted and teacher toric active-face moments.",
        "toric_coxeter_loss": "Affine-Coxeter wall/reflection consistency loss.",
        "toric_affine_wall_distance": "Distance to the nearest affine wall in the toric/Coxeter chart.",
        "toric_braid_loss": "Braid-relation consistency loss for wall-crossing paths.",
        "toric_leaf_residual": "Noncommutative torus phase-leaf residual for projected toric phase paths.",
        "toric_probe_rank": "Rank of the low-rank toric geometry probe.",
        "toric_vector_bundle_1d_cone_ce_loss": "Total Klyachko/sheaf vector-bundle objective for one-dimensional-cone labels, filtrations, splittings, and gluing.",
        "toric_vector_bundle_1d_cone_ce": "Cross-entropy for one-dimensional-cone labels in the finite Klyachko certificate.",
        "toric_vector_bundle_filtration_level_ce": "Cross-entropy for filtration-level labels attached to one-dimensional cones.",
        "toric_vector_bundle_filtration_residual": "Membership residual for finite filtration subspaces.",
        "toric_vector_bundle_1d_cone_klyachko_nesting_residual": "Residual for Klyachko decreasing-filtration nesting.",
        "toric_vector_bundle_cone_splitting_residual": "Residual for local splitting over affine toric cones.",
        "toric_vector_bundle_cech_gluing_residual": "Cech gluing residual for local sheaf sections on overlaps.",
        "toric_sheaf_chart_gluing_residual": "Alias of chart-gluing residual for sheaf compatibility.",
        "toric_sheaf_cocycle_residual": "Cech cocycle residual for triple overlaps.",
        "toric_vector_bundle_1d_cone_entropy": "Entropy of one-dimensional-cone predictions.",
        "toric_vector_bundle_active_1d_cone_mass": "Mean predicted mass on the certificate-active one-dimensional cone.",
        "toric_vector_bundle_rank": "Rank of the finite vector-bundle fiber certificate.",
        "toric_bgg_loss": "Finite Toric BGG/category-O objective over boundary-square, standard-filtration, Koszul, Gale-dual, and signature terms.",
        "toric_bgg_resolution_consistency": "Resolution consistency score 1/(1+d^2 residual); larger means closer to a chain complex.",
        "toric_bgg_d2_residual": "Boundary-square residual measuring failure of d_{k-1} d_k = 0.",
        "toric_bgg_standard_leakage": "Probability mass outside the allowed standard-filtration order ideal.",
        "toric_bgg_standard_allowed_mass": "Probability mass inside the allowed standard-filtration order ideal.",
        "toric_bgg_koszul_linearity_residual": "Residual for mass away from the expected linear Koszul degree profile.",
        "toric_bgg_gale_dual_consistency": "Consistency loss for finite Gale-dual signature pairing.",
        "toric_bgg_signature_smoothness": "Smoothness residual for BGG trajectory signatures.",
        "toric_bgg_standard_entropy": "Entropy of standard-label predictions.",
        "koszul_persistence_loss": "Koszul/persistence objective over multiparameter filtration modules and exactness proxies.",
        "toric_cca_topology_loss": "Combinatorial commutative algebra/topology objective coupling toric charts, Stanley-Reisner nonfaces, Koszul terms, and topology.",
        "toric_cca_allowed_edge_mass": "Average predicted mass assigned to combinatorially allowed one-dimensional edges in the toric chart complex.",
        "toric_cca_betti0_proxy": "Finite-window proxy for connected-component Betti mass in the chart/topology audit.",
        "toric_cca_betti1_proxy": "Finite-window proxy for one-cycle Betti mass in the chart/topology audit.",
        "toric_cca_binomial_residual": "Residual for toric binomial relations in the combinatorial commutative algebra certificate.",
        "toric_cca_chamber_coverage": "Coverage of occupied toric/Coxeter chambers in the finite chart sample.",
        "toric_cca_chart_entropy": "Entropy of active affine toric chart assignments; low values indicate chart collapse.",
        "toric_cca_euler_characteristic_proxy": "Finite-window Euler-characteristic proxy from the simplex/chart counts.",
        "toric_cca_fan_balance_loss": "Loss measuring imbalance across toric fan cells or chamber assignments.",
        "toric_cca_topology_loss_component": "The topological component inside the combinatorial toric objective, separated from symbolic algebra terms.",
        "toric_cca_windows": "Number of finite windows used for the combinatorial toric/topology sidecar audit.",
        "enabled": "Global sidecar status flag indicating whether ToricGT sidecar computation is enabled.",
        "sidecar_compute_all_metrics": "Status flag indicating that the trainer is computing the complete sidecar metric set, not only compact console metrics.",
        "toric_bgg_late_gate_required": "Status flag documenting whether the Toric BGG family is configured to be delayed by a late-training gate.",
        "toric_bgg_provenance_exact_finite_chain": "Status flag documenting that the Toric BGG supervision comes from exact finite chain-complex certificates.",
    }
    if name in exact:
        return exact[name]
    if name.endswith("_weight"):
        return "Configured family loss weight logged for audit; it controls how strongly the corresponding sidecar family contributes to training."
    if name.endswith("_active") or name.endswith("_enabled") or name.endswith("_available"):
        return "Binary status flag showing whether the corresponding sidecar family, metric source, or certificate was active/available."
    if "buchsbaum_eisenbud" in name:
        return "Buchsbaum-Eisenbud-style exactness/minor residual from the combinatorial commutative algebra/Koszul audit."
    if "stanley_reisner" in name or "nonface" in name:
        return "Stanley-Reisner/nonface mass metric measuring whether forbidden simplices or monomial nonfaces are being activated."
    if "symbolic" in name or "taylor" in name or "syzygy" in name or "resolution" in name:
        return "Symbolic resolution/syzygy metric from the combinatorial commutative algebra audit."
    if "koszul" in name:
        return "Koszul-complex metric for exactness, syzygies, Betti mass, or affine toric chart behavior."
    if family == "toric_tropical_geometry":
        return "Toric/tropical geometry probe metric for active fan cells, toric ideals, affine walls, moments, or phase leaves."
    if family == "trajectory_memory_retrieval":
        return "Trajectory-memory retrieval metric for analogical graph-of-thought summaries, topology, persistence, DAG, or derived-category features."
    return "Observed ToricGT sidecar metric; review its trend and BPB correlation before changing its family weight."


def weight_guidance_for(metric: str) -> str:
    name = stripped_name(metric)
    family = metric_family(metric)
    if name in {"lm_loss"}:
        return "Not directly weighted. Use it as a distribution-mismatch diagnostic for the graph sidecar stream."
    if name in {"loss", "toricgt_sidecar_loss"}:
        return "Raising global sidecar pressure makes all weighted sidecar families affect hidden states more; lower it if BPB worsens or train loss becomes noisy."
    if name.endswith("_active") or name.endswith("_enabled") or name.endswith("_available") or name == "enabled":
        return "Status metric only. It should confirm the family is being observed; it is not directly optimized."
    if name.endswith("_weight"):
        return "Logged configured weight. Raising it gives this family more gradient influence; lowering it makes the family more diagnostic and less likely to interfere with BPB."
    if name.startswith("sidecar_compute_") or name.startswith("toric_bgg_provenance_") or name.endswith("_gate_required"):
        return "Status/provenance metric only. It should be reviewed for audit completeness but is not directly optimized."
    if family == "graphcg_full_rank_concept_chart":
        return "Increase `GRAPHCG_LOSS_WEIGHT` when concept axes are unstable, high-covariance, or low-rank; decrease if train/val BPB rises or axis entropy dominates."
    if family == "analogical_lattice":
        return "Increase `ANALOGY_LOSS_WEIGHT` when relation groups exist but analogy/functor residuals stay high; decrease if it correlates positively with BPB or reduces token likelihood."
    if family == "tokengt_causal_graph_tokenization":
        return "Increase `TOKENGT_GRAPH_LOSS_WEIGHT` or first-class TokenGT strength when causal graph edges are poorly learned; decrease if graph BCE tracks higher BPB."
    if family == "trajectory_memory_retrieval":
        return "Increase `TRAJECTORY_MEMORY_LOSS_WEIGHT` when recall and PH/DAG/derived similarities are low but BPB is stable; decrease if memory CE/noise correlates with worse BPB."
    if family == "toric_tropical_geometry":
        return "Increase `TORIC_GEOMETRY_LOSS_WEIGHT` when active-face margins/binomial/moment residuals are weak and BPB is stable; decrease if toric residuals track BPB increases."
    if family == "toric_vector_bundle_sheaf":
        return "Increase `TORIC_VECTOR_BUNDLE_LOSS_WEIGHT` when one-dimensional-cone, filtration, splitting, or sheaf gluing residuals remain high; decrease if it competes with BPB."
    if family == "toric_bgg_category_o":
        return "Increase `TORIC_BGG_LOSS_WEIGHT` when d^2, standard leakage, Koszul linearity, or Gale-dual residuals remain high after BPB stabilizes; keep low early if BPB is above target."
    if family == "koszul_persistence":
        return "Increase `KOSZUL_PERSISTENCE_LOSS_WEIGHT` when PH/Koszul exactness or Betti metrics are poor and memory retrieval needs structure; reduce if it adds noise."
    if family == "combinatorial_commutative_algebra":
        return "Increase `COMBINATORIAL_TORIC_LOSS_WEIGHT` when Stanley-Reisner, syzygy, Taylor-resolution, or Buchsbaum-Eisenbud residuals are high; keep low unless BPB tolerates the algebraic pressure."
    if family == "activation_and_weight_status":
        return "Status metric only. It should be observed for audit completeness but is not itself optimized."
    return "Use the owning family weight conservatively; only raise it when the metric improves reasoning/structure without hurting train or validation BPB."


def metric_status(goal: str, first: float | None, last: float | None, corr_bpb: float | None) -> str:
    if first is None or last is None:
        return "insufficient_data"
    delta = last - first
    if goal == "lower":
        if delta < 0 and (corr_bpb is None or corr_bpb >= -0.2):
            return "improving"
        if delta > 0 and (corr_bpb is not None and corr_bpb > 0.25):
            return "bpb_risk"
        return "watch"
    if goal == "higher":
        if delta > 0 and (corr_bpb is None or corr_bpb <= 0.25):
            return "improving"
        if delta < 0:
            return "weakening"
        return "watch"
    return "context"


def review_metric(df: pd.DataFrame, metric: str, recent_fraction: float) -> SidecarMetricReview | None:
    series = numeric_series(df, metric)
    if series.empty:
        return None
    steps = series["_step"].to_numpy(dtype=float)
    values = series["value"].to_numpy(dtype=float)
    recent_n = max(3, int(math.ceil(len(values) * recent_fraction)))
    goal = metric_goal(metric)
    first = float(values[0])
    last = float(values[-1])
    corr_bpb = paired_corr(df, metric, "train/bpb")
    corr_loss = paired_corr(df, metric, "train/loss")
    return SidecarMetricReview(
        metric=metric,
        family=metric_family(metric),
        goal=goal,
        n=int(len(values)),
        first_step=float(steps[0]),
        last_step=float(steps[-1]),
        first_value=first,
        last_value=last,
        min_value=float(np.min(values)),
        max_value=float(np.max(values)),
        mean_value=float(np.mean(values)),
        slope_per_1k=slope_per_1k(steps, values),
        recent_slope_per_1k=slope_per_1k(steps[-recent_n:], values[-recent_n:]) if len(values) >= 3 else None,
        corr_train_bpb=corr_bpb,
        corr_train_loss=corr_loss,
        status=metric_status(goal, first, last, corr_bpb),
        description=description_for(metric),
        weight_guidance=weight_guidance_for(metric),
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.6g}"
    return "n/a"


def write_markdown(path: Path, meta: dict[str, Any], reviews: list[SidecarMetricReview]) -> None:
    by_family: dict[str, list[SidecarMetricReview]] = {}
    for row in reviews:
        by_family.setdefault(row.family, []).append(row)
    lines = [
        "# ToricGT Sidecar W&B Metric Review",
        "",
        f"- W&B run: `{meta.get('resolved_run_path')}`",
        f"- URL: {meta.get('url')}",
        f"- observed sidecar/related metrics: `{len(reviews)}`",
        f"- history rows: `{meta.get('history_rows')}`",
        "",
        "This report is generated at the 5K full-analysis interval. It lists every observed `toricgt_sidecar/*` metric, plus sidecar-related organized aliases when present, so the next hyperparameter decision considers the detailed mathematical behavior rather than only the compact console subset.",
        "",
        "## Family Summary",
        "",
        "| family | metrics | BPB-risk | improving | watch/context |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, rows in sorted(by_family.items()):
        risk = sum(1 for row in rows if row.status == "bpb_risk")
        improving = sum(1 for row in rows if row.status == "improving")
        watch = len(rows) - risk - improving
        lines.append(f"| {family} | {len(rows)} | {risk} | {improving} | {watch} |")
    lines.extend(["", "## Metric Glossary and Trend Review", ""])
    for family, rows in sorted(by_family.items()):
        lines.extend([f"### {family}", ""])
        lines.extend(
            [
                "| metric | goal | n | last | slope/1k | corr train BPB | status | meaning | weight guidance |",
                "|---|---|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for row in sorted(rows, key=lambda item: item.metric):
            lines.append(
                "| {metric} | {goal} | {n} | {last} | {slope} | {corr} | {status} | {desc} | {guidance} |".format(
                    metric=row.metric,
                    goal=row.goal,
                    n=row.n,
                    last=fmt(row.last_value),
                    slope=fmt(row.slope_per_1k),
                    corr=fmt(row.corr_train_bpb),
                    status=row.status,
                    desc=row.description.replace("|", "\\|"),
                    guidance=row.weight_guidance.replace("|", "\\|"),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def plot_correlations(path: Path, reviews: list[SidecarMetricReview]) -> None:
    rows = [row for row in reviews if row.corr_train_bpb is not None]
    rows = sorted(rows, key=lambda row: abs(float(row.corr_train_bpb or 0.0)), reverse=True)[:30]
    if not rows:
        return
    labels = [row.metric.replace(SIDECAR_PREFIX, "") for row in rows]
    values = [float(row.corr_train_bpb or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(12, max(6, 0.28 * len(rows))), constrained_layout=True)
    fig.patch.set_facecolor("#030712")
    ax.set_facecolor("#07111f")
    ax.barh(labels[::-1], values[::-1], color=["#37e8ff" if v >= 0 else "#ef476f" for v in values[::-1]])
    ax.axvline(0.0, color="#e8fbff", linewidth=1.0)
    ax.set_title("ToricGT sidecar metrics most correlated with train BPB", color="#e8fbff")
    ax.set_xlabel("Pearson correlation with train/bpb", color="#e8fbff")
    ax.tick_params(colors="#9fb6c5")
    for spine in ax.spines.values():
        spine.set_color("#29536a")
    ax.grid(True, axis="x", color="#143344", alpha=0.5)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df, meta = download_history(args.run_path)
    columns = sidecar_metric_columns(df, bool(args.include_related_aliases))
    reviews = []
    for metric in columns:
        series = numeric_series(df, metric)
        if len(series) < int(args.min_points):
            continue
        row = review_metric(df, metric, float(args.recent_fraction))
        if row is not None:
            reviews.append(row)
    reviews = sorted(reviews, key=lambda row: (row.family, row.metric))
    rows = [asdict(row) for row in reviews]
    raw_history_path = output_dir / "sidecar_wandb_history.csv"
    export_cols = ["_step"] + columns
    df[[col for col in export_cols if col in df.columns]].to_csv(raw_history_path, index=False)
    write_json(output_dir / "sidecar_metric_review.json", {"meta": meta, "metrics": rows})
    write_csv(output_dir / "sidecar_metric_review.csv", rows)
    write_markdown(output_dir / "SIDECAR-METRIC-REVIEW.md", meta, reviews)
    plot_correlations(output_dir / "sidecar_metric_train_bpb_correlations.png", reviews)
    by_family: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in reviews:
        by_family[row.family] = by_family.get(row.family, 0) + 1
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
    risky = [
        asdict(row)
        for row in sorted(
            [row for row in reviews if row.status == "bpb_risk"],
            key=lambda item: abs(float(item.corr_train_bpb or 0.0)),
            reverse=True,
        )[:12]
    ]
    summary = {
        "meta": meta,
        "observed_metric_count": len(reviews),
        "raw_sidecar_metric_count": sum(1 for row in reviews if row.metric.startswith(SIDECAR_PREFIX)),
        "related_alias_metric_count": sum(1 for row in reviews if not row.metric.startswith(SIDECAR_PREFIX)),
        "family_counts": by_family,
        "status_counts": status_counts,
        "bpb_risk_metrics_top": risky,
        "review_markdown": str(output_dir / "SIDECAR-METRIC-REVIEW.md"),
        "review_json": str(output_dir / "sidecar_metric_review.json"),
        "review_csv": str(output_dir / "sidecar_metric_review.csv"),
        "history_csv": str(raw_history_path),
        "correlation_plot": str(output_dir / "sidecar_metric_train_bpb_correlations.png"),
    }
    write_json(output_dir / "sidecar_metric_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
