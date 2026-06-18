#!/usr/bin/env python3
"""Adaptive BPB annealing policy for ToricGT OAI campaigns.

The controller is deliberately conservative with the score-defining FineWeb BPB
stream and experimental with the *distribution* of advanced pressure.  It keeps
all advanced metric families observable, then changes training pressure family by
family based on BPB slope, W&B sidecar correlations, retrieval evidence,
uncertainty gates, gradient-routing conflicts, graph-LM difficulty, and artifact
margin.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any


TARGET_BPB_DEFAULT = 1.19


@dataclass(frozen=True)
class FamilyRule:
    name: str
    env: str
    metric_aliases: tuple[str, ...]
    min_value: float
    max_value: float
    default_value: float
    exploration: float
    description: str


FAMILIES: tuple[FamilyRule, ...] = (
    FamilyRule(
        "graphcg",
        "GRAPHCG_LOSS_WEIGHT",
        ("graphcg", "graphcg_loss", "toricgt_sidecar/graphcg_loss"),
        1.0e-6,
        3.0e-5,
        3.0e-6,
        0.18,
        "Full-rank GraphCG basis pressure; keep orthogonality on, damp covariance when BPB-conflict rises.",
    ),
    FamilyRule(
        "analogy",
        "ANALOGY_LOSS_WEIGHT",
        ("analogy", "analogy_loss", "toricgt_sidecar/analogy_lattice_loss"),
        5.0e-6,
        4.5e-5,
        2.0e-5,
        0.25,
        "Analogy lattice and functorial relation pressure; raise only when retrieval evidence is present.",
    ),
    FamilyRule(
        "tokengt_graph",
        "TOKENGT_GRAPH_LOSS_WEIGHT",
        ("tokengt", "tokengt_graph_loss", "toricgt_sidecar/tokengt_graph_loss"),
        1.0e-5,
        8.0e-5,
        4.0e-5,
        0.20,
        "Causal graph edge/node supervision for graph-in/graph-out training.",
    ),
    FamilyRule(
        "trajectory_memory",
        "TRAJECTORY_MEMORY_LOSS_WEIGHT",
        ("memory", "memory_loss", "trajectory_memory_loss", "toricgt_sidecar/trajectory_memory_loss"),
        5.0e-6,
        4.5e-5,
        2.0e-5,
        0.24,
        "Analogical memory retrieval and trajectory-key pressure.",
    ),
    FamilyRule(
        "toric_geometry",
        "TORIC_GEOMETRY_LOSS_WEIGHT",
        ("toric", "toric_geometry_loss", "toricgt_sidecar/toric_geometry_loss"),
        2.5e-7,
        8.0e-6,
        2.0e-6,
        0.30,
        "Tropical active-face, moment, binomial, divisor, fan, and toric embedding pressure.",
    ),
    FamilyRule(
        "toric_vector_bundle_1d_cone",
        "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT",
        (
            "vector_bundle_1d_cone",
            "toric_vector_bundle_1d_cone_ce_loss",
            "toricgt_sidecar/toric_vector_bundle_1d_cone_ce_loss",
        ),
        2.5e-7,
        5.0e-6,
        1.0e-6,
        0.30,
        "Vector-bundle/sheaf one-dimensional-cone compatibility, filtrations, and gluing pressure.",
    ),
    FamilyRule(
        "toric_bgg",
        "TORIC_BGG_LOSS_WEIGHT",
        ("bgg", "bgg_loss", "toric_bgg_loss", "toricgt_sidecar/toric_bgg_loss"),
        2.5e-7,
        7.0e-6,
        2.0e-6,
        0.32,
        "Finite hypertoric/category-O BGG complex, d^2, standard-filtration, and Gale-dual pressure.",
    ),
    FamilyRule(
        "koszul_persistence",
        "KOSZUL_PERSISTENCE_LOSS_WEIGHT",
        ("koszul", "koszul_persistence_loss", "toricgt_sidecar/koszul_persistence_loss"),
        2.5e-7,
        5.0e-6,
        1.0e-6,
        0.28,
        "GUDHI vectorized PH, Koszul exactness, Betti and persistence-module pressure.",
    ),
    FamilyRule(
        "combinatorial_toric",
        "COMBINATORIAL_TORIC_LOSS_WEIGHT",
        ("cca", "combinatorial_toric", "toric_cca_topology_loss", "toricgt_sidecar/toric_cca_topology_loss"),
        2.5e-7,
        5.0e-6,
        1.0e-6,
        0.28,
        "Combinatorial commutative algebra, syzygy, resolution, and Buchsbaum-Eisenbud audit pressure.",
    ),
    FamilyRule(
        "oai_embedding_gflownet",
        "OAI_GFLOWNET_LOSS_WEIGHT",
        (
            "oai_gflownet_loss",
            "oai_gflownet_entropy",
            "oai_gflownet_reward",
            "oai_gflownet/loss",
            "oai_gflownet/tb_residual",
            "oai_gflownet/reward_mean",
            "oai_gflownet/action_diversity",
        ),
        5.0e-6,
        8.0e-5,
        2.0e-5,
        0.34,
        "Training-only embedding-space GFlowNet graph-of-thought pressure for BPB-facing hidden trajectories.",
    ),
    FamilyRule(
        "oai_multi_token_prediction",
        "OAI_MTP_LOSS_WEIGHT",
        ("oai_mtp_loss", "oai_mtp/loss", "oai_mtp/weighted_loss"),
        1.0e-3,
        1.2e-2,
        3.0e-3,
        0.25,
        "FineWeb-only multi-token prediction pressure using existing hidden states and the tied LM head.",
    ),
)


def finite(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else default
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fmt_float(value: float) -> str:
    if abs(value) < 1.0e-3:
        return f"{value:.6g}"
    if abs(value) < 1.0:
        return f"{value:.5f}".rstrip("0").rstrip(".")
    return f"{value:.6g}"


def series(metrics: dict[str, Any], key: str) -> list[tuple[float, float]]:
    rows = metrics.get("train_rows") or []
    out: list[tuple[float, float]] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        step = finite(row.get("step"))
        value = finite(row.get(key))
        if step is not None and value is not None:
            out.append((step, value))
    return out


def latest(metrics: dict[str, Any], key: str, default: float | None = None) -> float | None:
    train = metrics.get("latest_train")
    if isinstance(train, dict):
        value = finite(train.get(key), None)
        if value is not None:
            return value
    rows = series(metrics, key)
    return rows[-1][1] if rows else default


def slope_per_1k(points: list[tuple[float, float]], *, tail: int = 8) -> float | None:
    pts = points[-tail:]
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom <= 0:
        return None
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True)) / denom
    return 1000.0 * slope


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    xvar = sum((x - xbar) ** 2 for x in xs)
    yvar = sum((y - ybar) ** 2 for y in ys)
    if xvar <= 0 or yvar <= 0:
        return None
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True)) / math.sqrt(xvar * yvar)


def train_corr(metrics: dict[str, Any], key: str) -> float | None:
    bp = dict(series(metrics, "train_bpb"))
    vals = dict(series(metrics, key))
    common = sorted(set(bp) & set(vals))
    if len(common) < 3:
        return None
    return pearson([vals[s] for s in common], [bp[s] for s in common])


def sidecar_corr_from_review(sidecar_review: dict[str, Any], rule: FamilyRule) -> float | None:
    return metric_corr_from_review(sidecar_review, rule.metric_aliases + (rule.name,))


def metric_corr_from_review(sidecar_review: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    rows = sidecar_review.get("bpb_risk_metrics_top") or []
    if not isinstance(rows, list):
        return None
    best: float | None = None
    aliases = tuple(alias.lower() for alias in aliases)
    for row in rows:
        if not isinstance(row, dict):
            continue
        metric = str(row.get("metric") or "").lower()
        if any(alias in metric for alias in aliases):
            corr = finite(row.get("corr_train_bpb"))
            if corr is not None and (best is None or abs(corr) > abs(best)):
                best = corr
    return best


def route_conflict(metrics: dict[str, Any], family: str) -> float:
    # W&B metrics are richer than train.log, but this gives the controller a
    # stable default until sidecar review exports routing aliases.
    route_name = "sidecar"
    if family == "tokengt_graph":
        route_name = "graph_lm"
    if family in {"graphcg", "analogy", "trajectory_memory", "toric_geometry", "toric_bgg", "koszul_persistence", "combinatorial_toric", "toric_vector_bundle_1d_cone"}:
        route_name = "sidecar"
    return finite(latest(metrics, f"aux_grad_routing/{route_name}_conflict"), 0.0) or 0.0


def deterministic_jitter(run_id: str, family: str, scale: float) -> float:
    digest = hashlib.sha256(f"{run_id}:{family}".encode("utf-8")).digest()
    raw = int.from_bytes(digest[:4], "big") / 2**32
    centered = 2.0 * raw - 1.0
    return math.exp(scale * centered)


def graph_lm_decision(metrics: dict[str, Any], target_bpb: float) -> tuple[dict[str, str], dict[str, Any]]:
    train_bpb = latest(metrics, "train_bpb", float("inf")) or float("inf")
    graph_bpb = latest(metrics, "graph_lm_bpb", float("inf")) or float("inf")
    graph_weight = latest(metrics, "graph_lm_weight", 0.0) or 0.0
    train_slope = slope_per_1k(series(metrics, "train_bpb"))
    graph_slope = slope_per_1k(series(metrics, "graph_lm_bpb"))
    high_bpb = train_bpb > target_bpb + 0.18
    stalled = train_slope is not None and train_slope > -0.10
    graph_easy = graph_bpb < max(0.45, train_bpb - 0.60)
    overrides: dict[str, str] = {}
    reason: list[str] = []
    if high_bpb and (stalled or graph_easy):
        peak = 0.045 if graph_easy else 0.060
        start = 0.008
        warmup = 850
        hold = 120
        reason.append(
            "FineWeb train BPB is high/stalled while graph-LM is easier or not clearly helping; delay graph-LM pressure."
        )
    elif train_bpb <= target_bpb + 0.08 and (train_slope is None or train_slope < -0.04):
        peak = 0.125
        start = 0.025
        warmup = 275
        hold = 25
        reason.append("FineWeb BPB is near target and still improving; allow stronger graph-LM structural pressure.")
    else:
        peak = 0.080
        start = 0.015
        warmup = 500
        hold = 60
        reason.append("Use balanced graph-LM pressure while FineWeb remains the controlling objective.")
    overrides.update(
        {
            "GRAPH_LM_LOSS_WEIGHT": fmt_float(peak),
            "GRAPH_LM_LOSS_WEIGHT_START": fmt_float(start),
            "GRAPH_LM_WARMUP_STEPS": str(int(warmup)),
            "GRAPH_LM_HOLD_STEPS": str(int(hold)),
        }
    )
    return overrides, {
        "train_bpb": train_bpb,
        "graph_lm_bpb": graph_bpb,
        "graph_lm_weight_observed": graph_weight,
        "train_bpb_slope_per_1k": train_slope,
        "graph_lm_bpb_slope_per_1k": graph_slope,
        "graph_easy": graph_easy,
        "reason": reason,
    }


def graph_structure_decision(
    metrics: dict[str, Any],
    target_bpb: float,
    sidecar_review: dict[str, Any] | None = None,
    *,
    run_id: str = "",
) -> tuple[dict[str, str], dict[str, Any]]:
    sidecar_review = sidecar_review or {}
    train_bpb = latest(metrics, "train_bpb", float("inf")) or float("inf")
    tokengt_corr = train_corr(metrics, "tokengt")
    train_slope = slope_per_1k(series(metrics, "train_bpb"))
    flatten_lift = latest(metrics, "graph_output_flattening_ce_lift", None)
    flatten_corr = metric_corr_from_review(
        sidecar_review,
        (
            "graph_output_flattening/ce_lift",
            "graph_output_flattening/calibration_loss",
            "graph_output_flattening/ce_regression",
            "fineweb_graphify/edge_weight",
            "fineweb_graphify/edge_token_weight",
        ),
    )
    # ``tokengt`` in the train log is a graph loss, not an activation magnitude.
    # Positive correlation means graph loss is high when BPB is high, so pressure
    # that lowers it may be helpful.  Negative correlation is the risky case:
    # the graph loss is already low while BPB is high, suggesting conflict or
    # over-regularization rather than useful structure.
    graph_signal_risky = tokengt_corr is not None and tokengt_corr < -0.20
    flatten_signal_helpful = (
        (flatten_lift is not None and flatten_lift > 0.0)
        or (flatten_corr is not None and flatten_corr < -0.10)
    )
    improving_fast = train_slope is not None and train_slope < -0.20
    near_target = train_bpb <= target_bpb + 0.10
    if train_bpb > target_bpb + 0.22 or graph_signal_risky:
        radius = 2
        radius_mode = "local_bpb_recovery"
    elif improving_fast and flatten_signal_helpful and not graph_signal_risky:
        radius = 4 if near_target else 3
        radius_mode = "evidence_widening"
    elif near_target and not graph_signal_risky:
        radius = 4
        radius_mode = "near_target_generous"
    else:
        radius = 3
        radius_mode = "balanced_exploration"
    # Deliberately explore a little, but only upward when the BPB curve is
    # already healthy.  This makes wider graph neighborhoods a late-stage
    # hypothesis rather than an early source of sequence-noise.
    if near_target and improving_fast and flatten_signal_helpful:
        jitter = deterministic_jitter(run_id or "radius", "graph_radius", 0.20)
        if jitter > 1.10:
            radius = min(6, radius + 1)
            radius_mode = f"{radius_mode}_generous_jitter"
    radius = int(clamp(float(radius), 2.0, 6.0))
    overrides = {
        "FINEWEB_GRAPHIFY": "1",
        "TOKENGT_FIRST_CLASS": "1",
        "GRAPH_OUTPUT_FLATTENING": "1",
        "OAI_FINEWEB_OUTPUT_FLATTENING": "1",
        "GRAPH_OUTPUT_EDGE_RADIUS": str(radius),
        "TOKENGT_GRAPH_RADIUS": str(radius),
        "TOKENGT_IDENTIFIER_DIM": "24",
        "GRAPH_OUTPUT_VIRTUAL_EDGE_TOKENS": "1",
        "GRAPH_OUTPUT_SCORE_CORRECTION": "1",
        "GRAPH_OUTPUT_CALIBRATION_LOSS_WEIGHT": "0.012",
        "GRAPH_OUTPUT_CALIBRATION_MARGIN": "0.0",
        "GRAPH_OUTPUT_CALIBRATION_EVERY": "25",
        "GRAPH_OUTPUT_CALIBRATION_MAX_SEQUENCES": "2",
    }
    rationale: list[str] = [
        "Keep graphification first-class and keep OAI-FineWeb flattening scoped to BPB scoring.",
        f"Adaptive graph radius selected {radius} via {radius_mode}; widen later only when BPB slope and graph/flattening evidence support it.",
    ]
    if train_bpb > target_bpb + 0.18 and (tokengt_corr is None or graph_signal_risky):
        overrides.update(
            {
                "TOKENGT_FIRST_CLASS_LR": "1.2e-4",
                "TOKENGT_STRUCTURAL_WEIGHT": "0.032",
                "TOKENGT_EDGE_WEIGHT": "0.022",
                "TOKENGT_TORUS_WEIGHT": "0.007",
                "TOKENGT_IDENTIFIER_WEIGHT": "0.008",
                "TOKENGT_ENDPOINT_WEIGHT": "0.010",
                "TOKENGT_EDGE_TOKEN_WEIGHT": "0.009",
                "GRAPH_OUTPUT_FLATTENING_LR": "1.2e-4",
                "GRAPH_OUTPUT_NODE_WEIGHT": "0.040",
                "GRAPH_OUTPUT_EDGE_WEIGHT": "0.040",
                "GRAPH_OUTPUT_EDGE_TOKEN_WEIGHT": "0.020",
                "GRAPH_OUTPUT_SCORE_CORRECTION_WEIGHT": "0.014",
                "GRAPH_OUTPUT_CALIBRATION_LOSS_WEIGHT": "0.018",
            }
        )
        rationale.append(
            "TokenGT graph pressure appears BPB-risky or BPB is still high; keep graph structure on, "
            "but use gentle identifier/endpoint/virtual-edge and sequence-correction weights."
        )
    elif train_bpb <= target_bpb + 0.10 or (train_slope is not None and train_slope < -0.20):
        overrides.update(
            {
                "TOKENGT_FIRST_CLASS_LR": "2.2e-4",
                "TOKENGT_STRUCTURAL_WEIGHT": "0.060",
                "TOKENGT_EDGE_WEIGHT": "0.042",
                "TOKENGT_TORUS_WEIGHT": "0.014",
                "TOKENGT_IDENTIFIER_WEIGHT": "0.022",
                "TOKENGT_ENDPOINT_WEIGHT": "0.028",
                "TOKENGT_EDGE_TOKEN_WEIGHT": "0.026",
                "GRAPH_OUTPUT_FLATTENING_LR": "2.0e-4",
                "GRAPH_OUTPUT_NODE_WEIGHT": "0.065",
                "GRAPH_OUTPUT_EDGE_WEIGHT": "0.065",
                "GRAPH_OUTPUT_EDGE_TOKEN_WEIGHT": "0.060",
                "GRAPH_OUTPUT_SCORE_CORRECTION_WEIGHT": "0.045",
                "GRAPH_OUTPUT_CALIBRATION_LOSS_WEIGHT": "0.008",
            }
        )
        rationale.append(
            "FineWeb BPB is improving enough to test stronger node identifiers, endpoint maps, "
            "virtual local edge tokens, and sequence score correction."
        )
    else:
        overrides.update(
            {
                "TOKENGT_FIRST_CLASS_LR": "1.6e-4",
                "TOKENGT_STRUCTURAL_WEIGHT": "0.045",
                "TOKENGT_EDGE_WEIGHT": "0.030",
                "TOKENGT_TORUS_WEIGHT": "0.010",
                "TOKENGT_IDENTIFIER_WEIGHT": "0.014",
                "TOKENGT_ENDPOINT_WEIGHT": "0.018",
                "TOKENGT_EDGE_TOKEN_WEIGHT": "0.016",
                "GRAPH_OUTPUT_FLATTENING_LR": "1.6e-4",
                "GRAPH_OUTPUT_NODE_WEIGHT": "0.050",
                "GRAPH_OUTPUT_EDGE_WEIGHT": "0.050",
                "GRAPH_OUTPUT_EDGE_TOKEN_WEIGHT": "0.035",
                "GRAPH_OUTPUT_SCORE_CORRECTION_WEIGHT": "0.024",
                "GRAPH_OUTPUT_CALIBRATION_LOSS_WEIGHT": "0.012",
            }
        )
        rationale.append(
            "Use moderate first-class graph structure while BPB evidence is ambiguous; keep BPB-safe flattening correction active."
        )
    return overrides, {
        "tokengt_corr_train_bpb": tokengt_corr,
        "flattening_lift": flatten_lift,
        "flattening_corr_train_bpb": flatten_corr,
        "radius": radius,
        "radius_mode": radius_mode,
        "radius_policy": "2-3 early, 4-6 only after BPB/flattening evidence supports wider neighborhoods",
        "reason": rationale,
    }


def family_decisions(metrics: dict[str, Any], sidecar_review: dict[str, Any], run_id: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    overrides: dict[str, str] = {}
    decisions: list[dict[str, Any]] = []
    train_bpb = latest(metrics, "train_bpb", float("inf")) or float("inf")
    train_slope = slope_per_1k(series(metrics, "train_bpb"))
    retrieval_gate = latest(metrics, "sidecar_retrieval_gate", None)
    uncertainty = latest(metrics, "sidecar_uncertainty_weight", None)
    for rule in FAMILIES:
        corr_candidates = [train_corr(metrics, alias) for alias in rule.metric_aliases]
        corr_candidates = [c for c in corr_candidates if c is not None]
        corr = sidecar_corr_from_review(sidecar_review, rule)
        if corr is None and corr_candidates:
            corr = max(corr_candidates, key=lambda c: abs(c))
        residual = next((latest(metrics, alias, None) for alias in rule.metric_aliases if latest(metrics, alias, None) is not None), None)
        conflict = route_conflict(metrics, rule.name)
        score = 0.0
        reasons: list[str] = []
        if corr is not None:
            score += 0.45 * corr
            if corr > 0.25:
                reasons.append(f"loss is high when BPB is high (corr {corr:.3f}); pressure may help if routed safely")
            elif corr < -0.20:
                score -= 0.40
                reasons.append(f"loss is anticorrelated with BPB (corr {corr:.3f}); reduce pressure and keep as diagnostic")
            else:
                reasons.append(f"weak train-BPB correlation {corr:.3f}; allow bounded exploration")
        else:
            reasons.append("insufficient correlation evidence; use bounded exploratory prior")
        if train_slope is not None and train_slope < -0.15:
            score += 0.15
            reasons.append("FineWeb BPB is improving; safe to test slightly more structure")
        elif train_bpb > 1.45 or (train_slope is not None and train_slope > -0.03):
            score -= 0.20
            reasons.append("FineWeb BPB is high/stalled; dampen nonessential pressure")
        if conflict > 0.25:
            score -= 0.35 * conflict
            reasons.append(f"gradient conflict {conflict:.3f}; PCGrad says lower direct pressure")
        if rule.name in {"analogy", "trajectory_memory"}:
            if retrieval_gate is not None and retrieval_gate < 0.45:
                score -= 0.35
                reasons.append(f"retrieval gate {retrieval_gate:.3f} is weak; keep diagnostics but lower gradient")
            elif retrieval_gate is not None:
                score += 0.15
                reasons.append(f"retrieval gate {retrieval_gate:.3f} supports analogy/memory pressure")
        if rule.name in {
            "toric_geometry",
            "toric_vector_bundle_1d_cone",
            "toric_bgg",
            "koszul_persistence",
            "combinatorial_toric",
        }:
            if uncertainty is not None and uncertainty > 1.20:
                score += 0.15
                reasons.append(f"uncertainty weight {uncertainty:.3f} localizes advanced pressure to hard tokens")
            elif train_bpb > 1.45:
                score -= 0.10
                reasons.append("advanced algebraic pressure remains nonzero but lighter until BPB stabilizes")
        if rule.name == "oai_embedding_gflownet":
            entropy_corr = metric_corr_from_review(sidecar_review, ("oai_gflownet/entropy", "oai_gflownet/action_diversity"))
            residual_corr = metric_corr_from_review(sidecar_review, ("oai_gflownet/tb_residual", "oai_gflownet/loss"))
            if entropy_corr is not None and entropy_corr < -0.10:
                score += 0.12
                reasons.append(f"GFlowNet entropy/diversity anticorrelates with BPB ({entropy_corr:.3f}); keep exploration pressure active")
            if residual_corr is not None and residual_corr > 0.20:
                score += 0.10
                reasons.append(f"GFlowNet residual tracks high BPB ({residual_corr:.3f}); lowering it may help hidden trajectory organization")
            if train_bpb > 1.55:
                score -= 0.12
                reasons.append("primary BPB is very high; keep GFlowNet light until the base likelihood descends")
        if rule.name == "oai_multi_token_prediction":
            mtp_corr = metric_corr_from_review(sidecar_review, ("oai_mtp/loss", "oai_mtp/weighted_loss"))
            if mtp_corr is not None and mtp_corr > 0.20:
                score += 0.16
                reasons.append(f"MTP loss tracks high BPB ({mtp_corr:.3f}); future-token auxiliary may accelerate early descent")
            elif mtp_corr is not None and mtp_corr < -0.20:
                score -= 0.18
                reasons.append(f"MTP loss anticorrelates with BPB ({mtp_corr:.3f}); reduce to avoid overfitting a local auxiliary")
            if train_bpb > 1.60:
                score += 0.08
                reasons.append("primary BPB is high; keep a small MTP acceleration signal active")
        jitter = deterministic_jitter(run_id, rule.name, rule.exploration)
        multiplier = math.exp(clamp(score, -0.70, 0.60)) * jitter
        value = clamp(rule.default_value * multiplier, rule.min_value, rule.max_value)
        if train_bpb > 1.65 and rule.name not in {"tokengt_graph", "graphcg"}:
            value = min(value, rule.default_value)
            reasons.append("very high BPB: cap advanced exploratory pressure at default")
        overrides[rule.env] = fmt_float(value)
        decisions.append(
            {
                "family": rule.name,
                "env": rule.env,
                "value": value,
                "default": rule.default_value,
                "min": rule.min_value,
                "max": rule.max_value,
                "score": score,
                "jitter": jitter,
                "corr_train_bpb": corr,
                "residual": residual,
                "gradient_conflict": conflict,
                "description": rule.description,
                "reasons": reasons,
            }
        )
    return overrides, decisions


def optimizer_decision(metrics: dict[str, Any], target_bpb: float) -> tuple[dict[str, str], dict[str, Any]]:
    train_bpb = latest(metrics, "train_bpb", float("inf")) or float("inf")
    train_slope = slope_per_1k(series(metrics, "train_bpb"))
    artifact = finite(metrics.get("artifact_bytes"), 0.0) or 0.0
    overrides: dict[str, str]
    reasons: list[str] = []
    if artifact > 15_850_000:
        overrides = {
            "TRAIN_BATCH_TOKENS": "917504",
            "MATRIX_LR": "0.036",
            "SCALAR_LR": "0.036",
            "TIED_EMBED_LR": "0.046",
            "WARMDOWN_ITERS": "650",
        }
        reasons.append("artifact margin is tight; avoid shape changes and use moderate LR")
    elif train_bpb > target_bpb + 0.30 and (train_slope is None or train_slope > -0.20):
        overrides = {
            "TRAIN_BATCH_TOKENS": "983040",
            "MATRIX_LR": "0.050",
            "SCALAR_LR": "0.050",
            "TIED_EMBED_LR": "0.060",
            "WARMDOWN_ITERS": "500",
        }
        reasons.append("BPB is high and not dropping fast; use faster main-model optimization at ~20GB VRAM")
    elif train_slope is not None and train_slope < -0.40:
        overrides = {
            "TRAIN_BATCH_TOKENS": "983040",
            "MATRIX_LR": "0.040",
            "SCALAR_LR": "0.040",
            "TIED_EMBED_LR": "0.050",
            "WARMDOWN_ITERS": "700",
        }
        reasons.append("BPB is dropping quickly; reduce LR slightly and preserve the trajectory")
    else:
        overrides = {
            "TRAIN_BATCH_TOKENS": "983040",
            "MATRIX_LR": "0.044",
            "SCALAR_LR": "0.044",
            "TIED_EMBED_LR": "0.054",
            "WARMDOWN_ITERS": "600",
        }
        reasons.append("Use high VRAM utilization with moderate-fast LR for the next exploratory run")
    overrides.update({"TRAIN_SEQ_LEN": "1024", "WARMUP_STEPS": "10", "VAL_LOSS_EVERY": "1500", "CHECKPOINT_EVERY": "1500"})
    return overrides, {"train_bpb": train_bpb, "train_bpb_slope_per_1k": train_slope, "artifact_bytes": artifact, "reason": reasons}


def profile_hint(metrics: dict[str, Any], target_bpb: float, family_plan: list[dict[str, Any]]) -> str:
    selected = finite(metrics.get("selected_bpb"), float("inf")) or float("inf")
    train_bpb = latest(metrics, "train_bpb", float("inf")) or float("inf")
    graph_bpb = latest(metrics, "graph_lm_bpb", float("inf")) or float("inf")
    train_slope = slope_per_1k(series(metrics, "train_bpb"))
    if selected > 1.30 or train_bpb > target_bpb + 0.25:
        return "gate1500_fast_main_lr_light_graphcg"
    if graph_bpb < 0.35 and train_bpb > target_bpb + 0.08:
        return "gate1500_fast_main_lr_light_graphcg"
    helpful = sorted(family_plan, key=lambda row: float(row.get("score") or 0.0), reverse=True)
    top = helpful[0]["family"] if helpful else ""
    if top in {"toric_geometry", "toric_bgg"} and (train_slope is None or train_slope < -0.05):
        return "gate1500_structural_toric_heavy"
    if top in {"trajectory_memory", "analogy", "koszul_persistence"}:
        return "gate1500_bgg_memory_probe"
    return "gate1500_high_batch_toric_bgg_memory"


def build_adaptive_decision(
    metrics: dict[str, Any],
    validation: dict[str, Any] | None = None,
    sidecar_review: dict[str, Any] | None = None,
    *,
    run_id: str = "",
    target_bpb: float = TARGET_BPB_DEFAULT,
) -> dict[str, Any]:
    validation = validation or {}
    sidecar_review = sidecar_review or {}
    strict_ok = bool(validation.get("strict_validation_passed", False))
    selected_bpb = finite(metrics.get("selected_bpb"), float("inf")) or float("inf")
    train_bpb = latest(metrics, "train_bpb", float("inf")) or float("inf")
    graph_overrides, graph_plan = graph_lm_decision(metrics, target_bpb)
    structure_overrides, structure_plan = graph_structure_decision(
        metrics,
        target_bpb,
        sidecar_review,
        run_id=run_id,
    )
    family_overrides, family_plan = family_decisions(metrics, sidecar_review, run_id)
    opt_overrides, opt_plan = optimizer_decision(metrics, target_bpb)
    env_overrides: dict[str, str] = {}
    for block in (opt_overrides, structure_overrides, graph_overrides, family_overrides):
        env_overrides.update(block)
    env_overrides.update(
        {
            "AUX_GRAD_ROUTING": "1",
            "AUX_GRAD_ROUTE_GRAPH_LM": "1",
            "AUX_GRAD_ROUTE_SIDECAR": "1",
            "OAI_GFLOWNET": "1",
            "OAI_GFLOWNET_EVERY": "1",
            "OAI_GFLOWNET_LR": "2e-4",
            "OAI_GFLOWNET_ENTROPY_WEIGHT": "2e-6",
            "OAI_GFLOWNET_ENTROPY_TARGET": "1.8",
            "OAI_GFLOWNET_NUM_ACTIONS": "16",
            "OAI_GFLOWNET_MAX_SEQUENCES": "2",
            "OAI_GFLOWNET_MAX_POSITIONS": "192",
            "OAI_MTP": "1",
            "OAI_MTP_EVERY": "1",
            "OAI_MTP_OFFSETS": "2",
            "OAI_MTP_MAX_SEQUENCES": "2",
            "SCORE_FIRST_TTA": "1",
            "SCORE_FIRST_TTA_STEPS": "64",
            "SCORE_FIRST_TTA_LR": "2e-5",
            "SCORE_FIRST_TTA_COMMIT": "0",
            "RETRIEVAL_CONDITIONED_AUX": "1",
            "SIDECAR_UNCERTAINTY_WEIGHTING": "1",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "GRAPHCG_BPB_ORTHOGONAL_WEIGHT": "0.025",
            "GRAPHCG_COVARIANCE_CONFLICT_DAMPING": "0.65",
            "TORICGT_SIDECAR_SEQ_LEN": "256",
            "TORICGT_SIDECAR_BATCH_SIZE": "8",
            "GRAPH_LM_SEQ_LEN": "1024",
            "GRAPH_LM_BATCH_SIZE": "4",
        }
    )
    if train_bpb > target_bpb + 0.25:
        env_overrides.update(
            {
                "TORICGT_SIDECAR_LOSS_WEIGHT": "1.0",
                "TORICGT_SIDECAR_LOSS_WEIGHT_START": "0.02",
                "TORICGT_SIDECAR_WARMUP_STEPS": "750",
                "TORICGT_SIDECAR_HOLD_STEPS": "150",
            }
        )
    elif train_bpb <= target_bpb + 0.08:
        env_overrides.update(
            {
                "TORICGT_SIDECAR_LOSS_WEIGHT": "1.0",
                "TORICGT_SIDECAR_LOSS_WEIGHT_START": "0.08",
                "TORICGT_SIDECAR_WARMUP_STEPS": "300",
                "TORICGT_SIDECAR_HOLD_STEPS": "50",
            }
        )
    else:
        env_overrides.update(
            {
                "TORICGT_SIDECAR_LOSS_WEIGHT": "1.0",
                "TORICGT_SIDECAR_LOSS_WEIGHT_START": "0.05",
                "TORICGT_SIDECAR_WARMUP_STEPS": "500",
                "TORICGT_SIDECAR_HOLD_STEPS": "100",
            }
        )
    hint = profile_hint(metrics, target_bpb, family_plan)
    if not strict_ok:
        hint = "analysis_blocked_repeat_only_after_fix"
    missed = (math.isfinite(selected_bpb) and selected_bpb > target_bpb) or train_bpb > target_bpb
    reason = (
        "Adaptive BPB controller built family-specific env overrides from train BPB slope, validation/int8 BPB, "
        "graph-LM difficulty, sidecar metric correlations, retrieval/uncertainty gates, PCGrad conflicts, and artifact margin. "
    )
    if missed:
        reason += "The gate missed the target, so the next run should restart from step 0 with these overrides. "
    else:
        reason += "The target appears met; keep the decision as a reproducible continuation profile if more exploration is needed. "
    if not strict_ok:
        reason += "Strict full-analysis validation failed, so the campaign should not launch a blind new profile."
    return {
        "next_profile_hint": hint,
        "reason": reason,
        "target_bpb": target_bpb,
        "selected_bpb": selected_bpb,
        "train_bpb": train_bpb,
        "env_overrides": env_overrides,
        "adaptive_annealing_enabled": True,
        "sweep_strategy": "family_specific_evidence_weighted_exploration",
        "score_context": {
            "selected_bpb": selected_bpb,
            "train_bpb": train_bpb,
            "train_bpb_slope_per_1k": slope_per_1k(series(metrics, "train_bpb")),
            "graph_lm_bpb": latest(metrics, "graph_lm_bpb", None),
            "graph_lm_weight": latest(metrics, "graph_lm_weight", None),
            "artifact_bytes": metrics.get("artifact_bytes"),
            "strict_validation_passed": strict_ok,
            "sidecar_metric_count": sidecar_review.get("observed_metric_count"),
            "adaptive_graph_radius": structure_plan.get("radius"),
            "adaptive_graph_radius_mode": structure_plan.get("radius_mode"),
        },
        "optimizer_plan": opt_plan,
        "graph_lm_plan": graph_plan,
        "graph_structure_plan": structure_plan,
        "family_plan": family_plan,
        "sidecar_metric_review": sidecar_review,
    }
