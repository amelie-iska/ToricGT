#!/usr/bin/env python3
# Example:
#   conda run -n tokengt env PYTHONPATH=src python scripts/propose_training_adjustments.py \
#     --analysis-dir outputs/manual_analysis/oai-bpb-postbounce-02000_step2250_20260602T002659Z \
#     --output-json outputs/manual_analysis/oai-bpb-postbounce-02000_step2250_20260602T002659Z/training_adjustment_proposal.json
"""Propose training-control changes from a completed analysis gate.

The proposal is deliberately auditable.  BPB remains the primary competition
gate; GraphCG, Slepian/Pollak, toric, topology, BGG/Koszul, trajectory-memory,
and analogy diagnostics are compressed into sidecar pressures that decide how
aggressively to branch, damp, or promote the richer reasoning phases.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from toricgt.wandb_organization import configure_wandb_metrics, organize_wandb_payload, update_wandb_summary


ACTION_IDS = {
    "unknown": 0.0,
    "continue_until_first_gate_signal": 1.0,
    "hold_bpb_clean_schedule": 2.0,
    "prepare_midrun_recovery_branch": 3.0,
    "restart_from_best_checkpoint_with_damped_structural_sidecars": 4.0,
    "preserve_competition_checkpoint_then_expand_reasoning": 5.0,
}

STATUS_IDS = {
    "warming_up_unestablished": 0.0,
    "on_track": 1.0,
    "off_track": 2.0,
    "target_reached": 3.0,
}

LOSS_POLICY_IDS = {
    "hold": 0.0,
    "sidecar_tiny": 1.0,
    "enable_tiny": 2.0,
    "damp": 3.0,
    "post_threshold": 4.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--target-bpb", type=float, default=1.2)
    parser.add_argument("--gate-step", type=int, default=4000)
    parser.add_argument("--checkpoint-step", type=int, default=0)
    parser.add_argument("--wandb-run-path", default="", help="Optional entity/project/run_id to log control signals.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def positive_finite(value: Any) -> bool:
    out = finite_or_nan(value)
    return math.isfinite(out) and out > 0.0


def bounded(value: float, scale: float = 1.0, floor: float = 0.0) -> float:
    if not math.isfinite(value) or scale <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (value - floor) / scale))


def low_pressure(value: float, target: float, scale: float = 1.0) -> float:
    if not math.isfinite(value) or scale <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (target - value) / scale))


def metric_map(stats: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("metric", "")): row for row in stats if isinstance(row, dict)}


def metric(stats_by_name: dict[str, dict[str, Any]], *names: str) -> dict[str, Any]:
    for name in names:
        row = stats_by_name.get(name)
        if row is not None:
            return row
    return {}


def metric_value(
    stats_by_name: dict[str, dict[str, Any]],
    names: tuple[str, ...],
    key: str = "last_value",
    default: float = 0.0,
) -> float:
    return finite(metric(stats_by_name, *names).get(key), default=default)


def metric_category(stats_by_name: dict[str, dict[str, Any]], *names: str) -> str:
    return str(metric(stats_by_name, *names).get("category", "missing"))


def any_bad(stats_by_name: dict[str, dict[str, Any]], *names: str) -> bool:
    return any(metric_category(stats_by_name, name) == "not_as_desired" for name in names)


def json_value(data: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in data:
            return finite(data.get(key), default=default)
    return default


def analysis_step(root: Path, explicit_step: int) -> int:
    if explicit_step > 0:
        return int(explicit_step)
    match = re.search(r"step-(\d+)", root.name)
    if match:
        return int(match.group(1))
    match = re.search(r"step[-_]?(\d+)", str(root))
    return int(match.group(1)) if match else 0


def first_positive(*values: Any) -> float:
    for value in values:
        number = finite_or_nan(value)
        if math.isfinite(number) and number > 0.0:
            return number
    return float("nan")


def min_positive(*values: Any) -> float:
    numbers = [finite_or_nan(value) for value in values]
    positives = [value for value in numbers if math.isfinite(value) and value > 0.0]
    return min(positives) if positives else float("nan")


def oai_bpb_value(oai: dict[str, Any]) -> float:
    return first_positive(
        oai.get("oai_competition/bpb"),
        oai.get("bpb/oai_competition"),
        oai.get("competition/oai_bpb"),
    )


def is_sampled_oai_eval(oai: dict[str, Any]) -> bool:
    scope = str(oai.get("oai_competition/eval_scope", oai.get("eval_scope", ""))).lower()
    max_sequences = first_positive(
        oai.get("oai_competition/val_max_sequences"),
        oai.get("val_max_sequences"),
    )
    return "sample" in scope or (math.isfinite(max_sequences) and max_sequences > 0.0)


def compute_bpb_gate(
    *,
    stats_by_name: dict[str, dict[str, Any]],
    bpb_report: dict[str, Any],
    oai: dict[str, Any],
    geometry: dict[str, Any],
    step: int,
    target_bpb: float,
    gate_step: int,
) -> dict[str, Any]:
    train_bpb_last = metric_value(stats_by_name, ("train/bpb", "bpb/train", "fineweb/train_bpb"), default=float("nan"))
    val_bpb_last = metric_value(stats_by_name, ("val/bpb", "bpb/val", "fineweb/val_bpb"), default=float("nan"))
    train_bpb_best = metric_value(
        stats_by_name,
        ("train/bpb", "bpb/train", "fineweb/train_bpb"),
        key="best_value",
        default=float("nan"),
    )
    val_bpb_best = metric_value(
        stats_by_name,
        ("val/bpb", "bpb/val", "fineweb/best_val_bpb"),
        key="best_value",
        default=float("nan"),
    )
    oai_bpb = oai_bpb_value(oai)
    authoritative_oai_bpb = float("nan") if is_sampled_oai_eval(oai) else oai_bpb
    best_val_bpb = first_positive(
        bpb_report.get("best_val_bpb"),
        val_bpb_best,
        bpb_report.get("latest_val_bpb"),
        val_bpb_last,
    )
    geometry_best_bpb = first_positive(geometry.get("best_bpb"), geometry.get("best_answer_bpb"))
    current_primary_bpb = first_positive(
        bpb_report.get("latest_val_bpb"),
        val_bpb_last,
        best_val_bpb,
        authoritative_oai_bpb,
    )
    best_gate_bpb = min_positive(authoritative_oai_bpb, best_val_bpb)
    best_observed_bpb = min_positive(oai_bpb, best_val_bpb, geometry_best_bpb, train_bpb_best)
    if not math.isfinite(best_observed_bpb):
        best_observed_bpb = current_primary_bpb
    gap = current_primary_bpb - float(target_bpb) if math.isfinite(current_primary_bpb) else float("nan")

    val_slope_per_100 = finite_or_nan(bpb_report.get("val_bpb_recent_slope_per_100_steps"))
    if not math.isfinite(val_slope_per_100):
        val_slope_per_1k = metric_value(
            stats_by_name,
            ("val/bpb", "bpb/val", "fineweb/val_bpb"),
            key="recent_slope_per_1k",
            default=float("nan"),
        )
        val_slope_per_100 = val_slope_per_1k / 10.0 if math.isfinite(val_slope_per_1k) else float("nan")
    recent_drop_per_100 = max(0.0, -val_slope_per_100) if math.isfinite(val_slope_per_100) else 0.0
    remaining_steps = max(0, int(gate_step) - int(step))
    required_drop_per_100 = (
        max(0.0, gap) / max(1.0, float(remaining_steps)) * 100.0
        if math.isfinite(gap)
        else float("nan")
    )
    projected_target_step = finite_or_nan(bpb_report.get("projected_target_step_from_val"))
    if not math.isfinite(projected_target_step) and recent_drop_per_100 > 0 and math.isfinite(gap):
        projected_target_step = float(step) + max(0.0, gap) / recent_drop_per_100 * 100.0
    velocity_shortfall = (
        max(0.0, required_drop_per_100 - recent_drop_per_100)
        if math.isfinite(required_drop_per_100)
        else float("nan")
    )
    target_reached = math.isfinite(best_gate_bpb) and best_gate_bpb <= float(target_bpb)
    on_track = target_reached or (
        math.isfinite(projected_target_step)
        and projected_target_step <= float(gate_step)
        and math.isfinite(current_primary_bpb)
    )
    warmup_unestablished = step < 500 and not math.isfinite(projected_target_step) and not positive_finite(oai_bpb)
    if target_reached:
        status = "target_reached"
    elif warmup_unestablished:
        status = "warming_up_unestablished"
    elif on_track:
        status = "on_track"
    else:
        status = "off_track"

    return {
        "status": status,
        "step": int(step),
        "gate_step": int(gate_step),
        "target_bpb": float(target_bpb),
        "current_primary_bpb": current_primary_bpb,
        "best_observed_bpb": best_observed_bpb,
        "best_gate_bpb": best_gate_bpb,
        "train_bpb_last": train_bpb_last,
        "val_bpb_last": val_bpb_last,
        "oai_competition_bpb": oai_bpb,
        "authoritative_oai_bpb": authoritative_oai_bpb,
        "oai_competition_eval_scope": str(oai.get("oai_competition/eval_scope", oai.get("eval_scope", ""))),
        "geometry_best_bpb": geometry_best_bpb,
        "gap_to_target": gap,
        "recent_drop_per_100_steps": recent_drop_per_100,
        "required_drop_per_100_steps": required_drop_per_100,
        "velocity_shortfall_per_100_steps": velocity_shortfall,
        "projected_target_step": projected_target_step,
        "remaining_steps_to_gate": remaining_steps,
        "target_reached": target_reached,
        "on_track": on_track,
    }


def compute_structural_pressures(
    *,
    stats_by_name: dict[str, dict[str, Any]],
    geometry: dict[str, Any],
    fineweb_diag: dict[str, Any],
) -> dict[str, Any]:
    graphcg_loss = metric_value(stats_by_name, ("train/graphcg_loss",), default=0.0)
    graphcg_cov = metric_value(stats_by_name, ("train/graphcg_covariance_loss",), default=0.0)
    graphcg_basis = metric_value(stats_by_name, ("graphcg/basis_loss", "train/analogy_basis_loss"), default=0.0)
    graphcg_coherence = metric_value(stats_by_name, ("train/graphcg_basis_coherence",), default=0.0)
    graphcg_pressure = max(
        bounded(graphcg_loss, 1.0),
        bounded(graphcg_cov, 1.0),
        bounded(graphcg_basis, 1.0),
        bounded(graphcg_coherence, 1.0),
        0.65 if any_bad(stats_by_name, "train/graphcg_loss", "train/graphcg_covariance_loss") else 0.0,
    )

    slepian_leakage = max(
        metric_value(stats_by_name, ("slepian_pollak/leakage", "toric/slepian_leakage"), default=0.0),
        json_value(fineweb_diag, "toric/slepian_leakage", "diagnostics/latest/slepian_leakage", default=0.0),
        json_value(geometry, "mean_toric_slepian_leakage", default=0.0),
    )
    slepian_concentration = first_positive(
        metric_value(stats_by_name, ("slepian_pollak/concentration", "toric/slepian_concentration"), default=0.0),
        json_value(fineweb_diag, "toric/slepian_concentration", "diagnostics/latest/slepian_concentration", default=0.0),
        json_value(geometry, "mean_toric_slepian_concentration", default=0.0),
    )
    slepian_pressure = max(bounded(slepian_leakage, 1.0), low_pressure(slepian_concentration, 0.72, 0.72))

    topology_loss = max(
        metric_value(stats_by_name, ("topology/step_loss", "topology/lattice_loss", "train/analogy_step_topology_loss"), default=0.0),
        json_value(fineweb_diag, "topology/topology_loss", "diagnostics/latest/topology_loss", default=0.0),
    )
    directed_loss = max(
        metric_value(stats_by_name, ("topology/step_directed_loss", "topology/directed_loss"), default=0.0),
        json_value(geometry, "mean_topology_directed_map_loss", default=0.0),
    )
    hdbscan_noise = max(
        metric_value(stats_by_name, ("topology/step_hdbscan_noise_fraction",), default=0.0),
        json_value(geometry, "mean_topology_hdbscan_noise_fraction", default=0.0),
    )
    hdbscan_stability = first_positive(
        metric_value(stats_by_name, ("topology/step_hdbscan_stability", "topology/hdbscan_stability"), default=0.0),
        json_value(geometry, "mean_topology_hdbscan_stability", default=0.0),
    )
    cycle_flux = abs(
        max(
            metric_value(stats_by_name, ("topology/step_directed_cycle_flux",), default=0.0),
            json_value(geometry, "mean_topology_directed_cycle_flux", default=0.0),
        )
    )
    topology_pressure = max(
        bounded(topology_loss, 1.2),
        bounded(directed_loss, 0.30),
        bounded(hdbscan_noise, 0.50),
        low_pressure(hdbscan_stability, 0.60, 0.60),
        bounded(cycle_flux, 0.20),
    )

    toric_margin = min(
        metric_value(stats_by_name, ("toric/active_face_margin", "train/toric_active_face_margin"), default=0.0),
        json_value(geometry, "mean_toric_geometry_active_face_margin", default=0.0),
    )
    toric_shadow_margin = json_value(geometry, "mean_toric_shadow_mean_margin", default=0.0)
    toric_bend = max(
        metric_value(stats_by_name, ("toric/bend_magnitude", "train/toric_bend_magnitude"), default=0.0),
        json_value(geometry, "mean_toric_shadow_mean_bend", default=0.0),
    )
    toric_binom = max(
        metric_value(stats_by_name, ("toric/binomial_residual", "train/toric_binomial_residual"), default=0.0),
        json_value(geometry, "mean_toric_geometry_binomial_residual", default=0.0),
    )
    toric_pressure = max(
        low_pressure(toric_margin, 0.04, 2.0),
        low_pressure(toric_shadow_margin, 0.04, 0.04),
        bounded(toric_bend, 2.0),
        bounded(toric_binom, 1.0),
    )

    bgg_d2 = max(
        metric_value(stats_by_name, ("bgg_category_o/d2_residual", "category_o/d2_residual"), default=0.0),
        json_value(fineweb_diag, "bgg_category_o/d2_residual", "diagnostics/latest/bgg_d2_residual", default=0.0),
    )
    bgg_leak = max(
        metric_value(stats_by_name, ("bgg_category_o/standard_leakage", "category_o/standard_leakage"), default=0.0),
        json_value(fineweb_diag, "bgg_category_o/standard_leakage", "diagnostics/latest/bgg_standard_leakage", default=0.0),
    )
    koszul_loss = metric_value(stats_by_name, ("koszul_persistence/loss", "train/koszul_persistence_loss"), default=0.0)
    koszul_exact = metric_value(stats_by_name, ("koszul_persistence/exactness_residual", "train/koszul_exactness_residual"), default=0.0)
    bgg_koszul_pressure = max(bounded(bgg_d2, 0.10), bounded(bgg_leak, 1.0), bounded(koszul_loss, 1.0), bounded(koszul_exact, 0.25))

    memory_loss = metric_value(stats_by_name, ("train/trajectory_memory_loss",), default=0.0)
    memory_recall = metric_value(stats_by_name, ("train/trajectory_memory_recall1",), default=0.0)
    memory_entropy = metric_value(stats_by_name, ("train/trajectory_memory_entropy",), default=0.0)
    trajectory_flow = metric_value(stats_by_name, ("train/trajectory_flow_loss",), default=0.0)
    memory_pressure = max(
        bounded(memory_loss, 1.0),
        low_pressure(memory_recall, 0.25, 0.25),
        low_pressure(memory_entropy, 0.50, 0.50),
        bounded(trajectory_flow, 1.0),
    )

    analogy_map = metric_value(stats_by_name, ("topology/step_analogical_map_loss", "train/analogy_step_analogical_map_loss"), default=0.0)
    analogy_directed_map = metric_value(stats_by_name, ("topology/step_directed_map_loss", "train/analogy_step_directed_map_loss"), default=0.0)
    contrastive = metric_value(stats_by_name, ("train/contrastive_loss",), default=0.0)
    analogy_pressure = max(
        bounded(analogy_map, 0.30),
        bounded(analogy_directed_map, 0.30),
        bounded(contrastive, 1.0),
        0.65 if any_bad(stats_by_name, "train/analogy_step_analogical_map_loss", "topology/step_analogical_map_loss") else 0.0,
    )

    complexity_ncd = max(
        json_value(fineweb_diag, "complexity/recent_full_log_ncd_lzma", "diagnostics/latest/complexity_recent_full_log_ncd_lzma", default=0.0),
        metric_value(stats_by_name, ("complexity/train/prediction_target_ncd_lzma_mean", "complexity/val/prediction_target_ncd_lzma_mean"), default=0.0),
    )
    complexity_pressure = bounded(complexity_ncd, 1.0)

    family_pressures = {
        "graphcg_disentanglement": graphcg_pressure,
        "slepian_pollak": slepian_pressure,
        "directed_topology": topology_pressure,
        "toric_tropical": toric_pressure,
        "bgg_koszul": bgg_koszul_pressure,
        "trajectory_memory": memory_pressure,
        "analogical_reasoning": analogy_pressure,
        "complexity_compression": complexity_pressure,
    }
    structural_pressure = max(
        0.0,
        min(
            1.0,
            0.16 * graphcg_pressure
            + 0.16 * slepian_pressure
            + 0.18 * topology_pressure
            + 0.16 * toric_pressure
            + 0.12 * bgg_koszul_pressure
            + 0.08 * memory_pressure
            + 0.08 * analogy_pressure
            + 0.06 * complexity_pressure,
        ),
    )
    dominant_family = max(family_pressures.items(), key=lambda item: (item[1], item[0]))[0]
    return {
        "structural_pressure": structural_pressure,
        "dominant_family": dominant_family,
        "dominant_family_pressure": family_pressures[dominant_family],
        "family_pressures": family_pressures,
        "raw": {
            "graphcg_loss": graphcg_loss,
            "graphcg_covariance_loss": graphcg_cov,
            "graphcg_basis_loss": graphcg_basis,
            "slepian_leakage": slepian_leakage,
            "slepian_concentration": slepian_concentration,
            "topology_loss": topology_loss,
            "directed_topology_loss": directed_loss,
            "hdbscan_noise_fraction": hdbscan_noise,
            "hdbscan_stability": hdbscan_stability,
            "toric_active_face_margin": toric_margin,
            "toric_shadow_margin": toric_shadow_margin,
            "toric_bend": toric_bend,
            "toric_binomial_residual": toric_binom,
            "bgg_d2_residual": bgg_d2,
            "bgg_standard_leakage": bgg_leak,
            "koszul_loss": koszul_loss,
            "memory_loss": memory_loss,
            "memory_recall1": memory_recall,
            "analogy_map_loss": analogy_map,
            "complexity_ncd_lzma": complexity_ncd,
        },
    }


def choose_action(bpb_gate: dict[str, Any], structural: dict[str, Any]) -> tuple[str, dict[str, str], list[str]]:
    status = str(bpb_gate["status"])
    step = int(bpb_gate["step"])
    structural_pressure = finite(structural.get("structural_pressure"), default=0.0)
    dominant = str(structural.get("dominant_family", "none"))
    shortfall = finite(bpb_gate.get("velocity_shortfall_per_100_steps"), default=0.0)
    projected = finite_or_nan(bpb_gate.get("projected_target_step"))
    gate_step = int(bpb_gate["gate_step"])

    loss_policy = {
        "graphcg_loss_weight": "sidecar_tiny",
        "slepian_pollak_loss_weight": "sidecar_tiny",
        "analogy_lattice_loss_weight": "hold",
        "toric_geometry_loss_weight": "hold",
        "toric_bgg_loss_weight": "hold",
        "koszul_persistence_loss_weight": "hold",
        "trajectory_memory_loss_weight": "hold",
        "trajectory_flow_loss_weight": "hold",
        "contrastive_loss_weight": "sidecar_tiny",
        "gflownet_loss_weight": "sidecar_tiny",
    }
    rationale: list[str] = []

    if status == "target_reached":
        action = "preserve_competition_checkpoint_then_expand_reasoning"
        for key in loss_policy:
            loss_policy[key] = "post_threshold"
        rationale.append("Best observed BPB is at or below target; freeze the competition artifact before broadening reasoning losses.")
        return action, loss_policy, rationale

    if status == "warming_up_unestablished":
        action = "continue_until_first_gate_signal"
        rationale.append("The run has not produced a checkpoint-level BPB projection yet; continue to the first periodic analysis before restarting.")
        rationale.append("Advanced losses should remain tiny because early train BPB is not a reliable target-gate signal.")
        return action, loss_policy, rationale

    if status == "on_track":
        action = "hold_bpb_clean_schedule"
        loss_policy["graphcg_loss_weight"] = "enable_tiny"
        loss_policy["slepian_pollak_loss_weight"] = "enable_tiny"
        loss_policy["gflownet_loss_weight"] = "enable_tiny"
        if structural_pressure < 0.45:
            loss_policy["analogy_lattice_loss_weight"] = "enable_tiny"
            loss_policy["trajectory_flow_loss_weight"] = "enable_tiny"
        rationale.append("BPB projects to the target before the gate, so avoid disrupting the clean BPB descent.")
        rationale.append("Use GraphCG and Slepian/Pollak as tiny regularizers and promote heavier topology/memory phases only after the threshold artifact is preserved.")
        return action, loss_policy, rationale

    if step >= gate_step or (math.isfinite(projected) and projected > float(gate_step) and step >= 3000):
        action = "restart_from_best_checkpoint_with_damped_structural_sidecars"
        for key in loss_policy:
            if key not in {"graphcg_loss_weight", "slepian_pollak_loss_weight", "contrastive_loss_weight"}:
                loss_policy[key] = "damp"
        rationale.append("BPB is projected to miss the 4K gate; restart from the best mid-run checkpoint instead of continuing the same slope.")
        rationale.append(f"Dominant advanced pressure is {dominant}; use it to choose damping/branch controls, not as a heavy primary loss.")
        return action, loss_policy, rationale

    action = "prepare_midrun_recovery_branch"
    if shortfall > 0:
        rationale.append(f"Validation BPB needs {shortfall:.6f} more drop per 100 steps to hit the gate.")
    rationale.append(f"Structural pressure is {structural_pressure:.3f}; keep the main run BPB-first while preparing a branched recapture.")
    rationale.append(f"Dominant family {dominant} should decide which sidecar gets promoted on the branch.")
    return action, loss_policy, rationale


def wandb_payload(proposal: dict[str, Any]) -> dict[str, float]:
    bpb = proposal["bpb_gate"]
    structural = proposal["structural_diagnostics"]
    action = str(proposal["decision"]["primary_action"])
    payload = {
        "trainer/step": float(bpb["step"]),
        "analysis_control/target_status_id": STATUS_IDS.get(str(bpb["status"]), 0.0),
        "analysis_control/primary_action_id": ACTION_IDS.get(action, 0.0),
        "analysis_control/current_primary_bpb": finite(bpb.get("current_primary_bpb"), default=0.0),
        "analysis_control/best_observed_bpb": finite(bpb.get("best_observed_bpb"), default=0.0),
        "analysis_control/gap_to_target": finite(bpb.get("gap_to_target"), default=0.0),
        "analysis_control/recent_drop_per_100_steps": finite(bpb.get("recent_drop_per_100_steps"), default=0.0),
        "analysis_control/required_drop_per_100_steps": finite(bpb.get("required_drop_per_100_steps"), default=0.0),
        "analysis_control/velocity_shortfall_per_100_steps": finite(bpb.get("velocity_shortfall_per_100_steps"), default=0.0),
        "analysis_control/projected_target_step": finite(bpb.get("projected_target_step"), default=0.0),
        "analysis_control/target_reached": float(bool(bpb.get("target_reached"))),
        "analysis_control/on_track": float(bool(bpb.get("on_track"))),
        "analysis_control/structural_pressure": finite(structural.get("structural_pressure"), default=0.0),
        "analysis_control/dominant_family_pressure": finite(structural.get("dominant_family_pressure"), default=0.0),
    }
    families = structural.get("family_pressures")
    if isinstance(families, dict):
        for key, value in families.items():
            payload[f"analysis_control/family_pressure/{key}"] = finite(value, default=0.0)
    loss_policy = proposal["decision"].get("loss_policy", {})
    if isinstance(loss_policy, dict):
        for key, value in loss_policy.items():
            payload[f"analysis_control/loss_policy/{key}_id"] = LOSS_POLICY_IDS.get(str(value), 0.0)
    return payload


def maybe_log_wandb(run_path: str, proposal: dict[str, Any]) -> str:
    if not run_path:
        return ""
    parts = run_path.split("/")
    if len(parts) != 3:
        return "invalid run path; expected entity/project/run_id"
    entity, project, run_id = parts
    try:
        import wandb

        run = wandb.init(
            entity=entity,
            project=project,
            id=run_id,
            resume="allow",
            name=run_id,
            config={"analysis_control_source": "scripts/propose_training_adjustments.py"},
        )
        configure_wandb_metrics(wandb)
        payload = wandb_payload(proposal)
        organized = organize_wandb_payload(payload)
        wandb.log(organized, step=int(proposal["bpb_gate"]["step"]))
        update_wandb_summary(run, {key: value for key, value in payload.items() if key.startswith("analysis_control/")})
        run.finish()
        return ""
    except Exception as exc:  # pragma: no cover - W&B failures should not invalidate local analysis.
        return f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    root = Path(args.analysis_dir)
    step = analysis_step(root, int(args.checkpoint_step or 0))
    stats_raw = load_json(root / "metrics" / "metric_stats.json")
    stats = stats_raw if isinstance(stats_raw, list) else []
    stats_by_name = metric_map(stats)
    geometry = load_json(root / "geometry" / "reasoning_geometry_summary.json")
    simplex = load_json(root / "simplex" / "reasoning_simplex_summary.json")
    bpb_report = load_json(root / "bpb" / "bpb_acceleration_report.json")
    fineweb_diag = load_json(root / "geometry" / "fineweb_curve_diagnostic_payload.json")
    oai = load_json(root / "oai_competition" / "summary.json")

    bpb_gate = compute_bpb_gate(
        stats_by_name=stats_by_name,
        bpb_report=bpb_report if isinstance(bpb_report, dict) else {},
        oai=oai if isinstance(oai, dict) else {},
        geometry=geometry if isinstance(geometry, dict) else {},
        step=step,
        target_bpb=float(args.target_bpb),
        gate_step=int(args.gate_step),
    )
    structural = compute_structural_pressures(
        stats_by_name=stats_by_name,
        geometry=geometry if isinstance(geometry, dict) else {},
        fineweb_diag=fineweb_diag if isinstance(fineweb_diag, dict) else {},
    )
    action, loss_policy, rationale = choose_action(bpb_gate, structural)

    proposal: dict[str, Any] = {
        "analysis_dir": str(root),
        "search_policy": "bpb_first_advanced_metric_recapture_controller",
        "bpb_gate": bpb_gate,
        "structural_diagnostics": structural,
        "simplex_summary": simplex if isinstance(simplex, dict) else {},
        "decision": {
            "primary_action": action,
            "target_status": bpb_gate["status"],
            "loss_policy": loss_policy,
            "restart_policy": (
                "use_best_checkpoint_at_or_before_3000_to_3500_with_damped_sidecars"
                if action.startswith("restart")
                else "no_restart_until_next_gate"
            ),
            "competition_checkpoint_policy": (
                "preserve_immediately"
                if bpb_gate["target_reached"]
                else "wait_for_target_bpb"
            ),
        },
        "recommended_phase_delta": {},
        "rationale": rationale,
    }

    delta = proposal["recommended_phase_delta"]
    if action == "prepare_midrun_recovery_branch":
        delta["primary_bpb_controls"] = "prepare_best_checkpoint_branch_with_velocity_or_transfer_intervention"
        delta["advanced_metric_usage"] = "rank_branch_controls_by_dominant_family_pressure"
    elif action == "restart_from_best_checkpoint_with_damped_structural_sidecars":
        delta["primary_bpb_controls"] = "restart_from_best_midrun_checkpoint_with_optimizer_state_reset_or_damped_lr"
        delta["advanced_metric_usage"] = "use_graphcg_slepian_topology_pressures_as damping and branch-selection signals"
    elif action == "hold_bpb_clean_schedule":
        delta["primary_bpb_controls"] = "hold_current_schedule"
        delta["advanced_metric_usage"] = "tiny_graphcg_slepian_regularization_plus_sidecar_analysis"
    elif action == "preserve_competition_checkpoint_then_expand_reasoning":
        delta["primary_bpb_controls"] = "freeze_parameter_golf_artifact"
        delta["advanced_metric_usage"] = "begin_graph_of_thought_memory_analogy_topology_curriculum_from_preserved_checkpoint"
    else:
        delta["primary_bpb_controls"] = "continue_until_checkpoint_signal"
        delta["advanced_metric_usage"] = "observe_all_families_without_promoting_heavy_losses"

    wandb_error = maybe_log_wandb(args.wandb_run_path, proposal)
    if wandb_error:
        proposal["wandb_log_error"] = wandb_error

    output_json = Path(args.output_json) if args.output_json else root / "training_adjustment_proposal.json"
    output_json.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md = Path(args.output_md) if args.output_md else root / "training_adjustment_proposal.md"
    lines = [
        "# Training Adjustment Proposal",
        "",
        f"- Analysis directory: `{root}`",
        f"- Search policy: `{proposal['search_policy']}`",
        f"- Target status: `{proposal['decision']['target_status']}`",
        f"- Primary action: `{proposal['decision']['primary_action']}`",
        "",
        "## BPB Gate",
    ]
    for key, value in proposal["bpb_gate"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Structural Pressures"])
    lines.append(f"- `structural_pressure`: `{structural['structural_pressure']}`")
    lines.append(f"- `dominant_family`: `{structural['dominant_family']}`")
    for key, value in structural.get("family_pressures", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Loss Policy"])
    for key, value in loss_policy.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Recommended Phase Delta"])
    for key, value in delta.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Rationale"])
    for item in rationale:
        lines.append(f"- {item}")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(proposal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
