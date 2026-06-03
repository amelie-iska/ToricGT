#!/usr/bin/env python3
# Example:
#   conda run -n tokengt env PYTHONPATH=src python scripts/propose_training_adjustments.py \
#     --analysis-dir outputs/manual_analysis/oai-bpb-postbounce-02000_step2250_20260602T002659Z \
#     --output-json outputs/manual_analysis/oai-bpb-postbounce-02000_step2250_20260602T002659Z/training_adjustment_proposal.json
"""Propose small training-control changes from a completed analysis gate.

This is intentionally conservative.  It is not an optimizer that edits configs
blindly; it converts the same metric files reviewed by Codex into an auditable
proposal that can be accepted, rejected, or modified in the next review turn.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--target-bpb", type=float, default=1.2)
    parser.add_argument("--gate-step", type=int, default=4000)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def metric(stats: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in stats:
        if row.get("metric") == name:
            return row
    return {}


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def positive_finite(value: Any) -> bool:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out) and out > 0.0


def main() -> None:
    args = parse_args()
    root = Path(args.analysis_dir)
    stats_raw = load_json(root / "metrics" / "metric_stats.json")
    stats = stats_raw if isinstance(stats_raw, list) else []
    geometry = load_json(root / "geometry" / "reasoning_geometry_summary.json")
    bpb_report = load_json(root / "bpb" / "bpb_acceleration_report.json")
    fineweb_diag = load_json(root / "geometry" / "fineweb_curve_diagnostic_payload.json")

    train_bpb = metric(stats, "train/bpb")
    train_loss = metric(stats, "train/loss")
    graphcg_loss = metric(stats, "train/graphcg_loss")
    graphcg_cov = metric(stats, "train/graphcg_covariance_loss")
    analogy_map = metric(stats, "train/analogy_step_analogical_map_loss")
    toric_entropy = metric(stats, "train/toric_active_face_entropy")

    bpb_slope = f(train_bpb, "recent_slope_per_1k")
    loss_slope = f(train_loss, "recent_slope_per_1k")
    best_bpb = float(geometry.get("best_bpb", 0.0) or 0.0)
    mean_bpb = float(geometry.get("mean_bpb", 0.0) or 0.0)
    branch_gap = max(0.0, mean_bpb - best_bpb) if mean_bpb and best_bpb else 0.0
    toric_margin = float(geometry.get("mean_toric_shadow_mean_margin", 0.0) or 0.0)
    active_face_margin = float(geometry.get("mean_toric_geometry_active_face_margin", 0.0) or 0.0)
    best_val_bpb = finite(bpb_report.get("best_val_bpb"), default=0.0)
    target_gap = finite(bpb_report.get("target_gap"), default=0.0)
    val_recent_slope = finite(bpb_report.get("val_bpb_recent_slope_per_100_steps"), default=0.0)
    train_recent_slope = finite(bpb_report.get("train_bpb_recent_slope_per_100_steps"), default=0.0)
    projected_target_step = finite(bpb_report.get("projected_target_step_from_val"), default=0.0)
    projected_steps_to_target = finite(bpb_report.get("projected_steps_to_target_from_val"), default=0.0)
    topology_loss = finite(fineweb_diag.get("topology/topology_loss"), default=0.0)
    bgg_d2_residual = finite(fineweb_diag.get("bgg_category_o/d2_residual"), default=0.0)
    koszul_persistence = finite(fineweb_diag.get("bgg_category_o/persistence/betti_entropy"), default=0.0)
    slepian_leakage = finite(fineweb_diag.get("toric/slepian_leakage"), default=0.0)
    slepian_concentration = finite(fineweb_diag.get("toric/slepian_concentration"), default=0.0)
    tropical_plateau_pressure = finite(fineweb_diag.get("tropical/bpb_plateau_pressure"), default=0.0)
    complexity_ncd = finite(fineweb_diag.get("complexity/recent_full_log_ncd_lzma"), default=0.0)

    proposal: dict[str, Any] = {
        "analysis_dir": str(root),
        "signals": {
            "best_validation_bpb": best_val_bpb,
            "target_gap": target_gap,
            "validation_recent_slope_per_100_steps": val_recent_slope,
            "validation_projected_steps_to_target": projected_steps_to_target,
            "validation_projected_target_step": projected_target_step,
            "training_gate_step": int(args.gate_step),
            "train_bpb_recent_slope_per_1k": bpb_slope,
            "train_bpb_recent_slope_per_100_steps": train_recent_slope,
            "train_loss_recent_slope_per_1k": loss_slope,
            "branch_gap_mean_minus_best_bpb": branch_gap,
            "toric_shadow_mean_margin": toric_margin,
            "toric_active_face_margin": active_face_margin,
            "diagnostic_topology_loss": topology_loss,
            "diagnostic_bgg_d2_residual": bgg_d2_residual,
            "diagnostic_koszul_persistence_betti_entropy": koszul_persistence,
            "diagnostic_slepian_leakage": slepian_leakage,
            "diagnostic_slepian_concentration": slepian_concentration,
            "diagnostic_tropical_plateau_pressure": tropical_plateau_pressure,
            "diagnostic_complexity_ncd_lzma": complexity_ncd,
            "graphcg_loss_category": graphcg_loss.get("category"),
            "graphcg_covariance_category": graphcg_cov.get("category"),
            "analogy_map_category": analogy_map.get("category"),
            "toric_entropy_category": toric_entropy.get("category"),
        },
        "recommended_phase_delta": {},
        "rationale": [],
        "search_policy": "multi_fidelity_codex_gate_with_structural_sidecar",
    }

    delta = proposal["recommended_phase_delta"]
    rationale = proposal["rationale"]

    target_reached = bool(best_val_bpb and best_val_bpb <= float(args.target_bpb))
    projected_before_gate = positive_finite(projected_target_step) and projected_target_step <= float(args.gate_step)
    validation_descending = val_recent_slope < 0.0
    validation_stalled = not validation_descending or abs(val_recent_slope) < 0.002

    if target_reached:
        delta["competition_checkpoint"] = "preserve_current_best_as_openai_parameter_golf_artifact"
        delta["post_threshold_phase"] = "begin_reasoning_memory_analogy_curriculum_from_preserved_checkpoint"
        rationale.append("Validation BPB is at or below target; freeze the threshold artifact before broader reasoning phases.")
    elif validation_descending and projected_before_gate:
        delta["primary_bpb_controls"] = "hold_current_seq4096_schedule"
        delta["primary_structural_losses"] = "keep_off_or_tiny_until_threshold_checkpoint"
        delta["advanced_structural_branch"] = "continue_graphcg_slepian_memory_analogy_sidecar"
        rationale.append("Validation BPB is still descending and projects below target before the 4K gate; avoid disrupting the BPB-clean primary run.")
    elif validation_descending and positive_finite(projected_target_step):
        delta["primary_bpb_controls"] = "gate_risk_restart_from_best_checkpoint_or_step3000_recapture"
        delta["candidate_bpb_controls"] = "tied_embedding_lr_plus_5_to_10pct_batch_tokens_hold_or_plus_if_memory_allows_shorter_warmdown"
        rationale.append("Validation BPB is improving but projected to miss the 4K gate; prepare a mid-run restart from the best checkpoint.")
    elif validation_stalled:
        delta["primary_bpb_controls"] = "restart_from_best_checkpoint_with_small_lr_or_batch_intervention"
        delta["candidate_bpb_controls"] = "reset_optimizer_rng_loader_keep_best_weights_try_tied_embedding_lr_plus_10pct"
        rationale.append("Validation BPB slope is weak or nonnegative; the primary run needs a conservative recapture experiment.")

    validation_gate_on_track = target_reached or (validation_descending and projected_before_gate)
    if validation_gate_on_track:
        delta["lr_multiplier"] = "hold_validation_gate_on_track"
        rationale.append("Validation gate status overrides noisy train-only BPB slope heuristics.")
    elif bpb_slope < -1.0 and loss_slope < -0.6:
        delta["lr_multiplier"] = "increase_small_or_hold"
        rationale.append("BPB and loss have strong negative recent slopes; avoid large resets.")
    elif bpb_slope < -0.25:
        delta["lr_multiplier"] = "hold_or_increase_10pct"
        rationale.append("BPB is improving but not dramatically; a small LR increase is viable.")
    else:
        delta["lr_multiplier"] = "decrease_or_rollback"
        rationale.append("BPB slope is weak or positive; prefer rollback or lower LR.")

    if graphcg_cov.get("category") == "not_as_desired" or graphcg_loss.get("category") != "as_desired":
        delta["graphcg_loss_weight"] = "enable_5e-5_to_1e-4"
        delta["analogy_lattice_loss_weight"] = "tiny_1e-5_to_2e-5"
        rationale.append("GraphCG basis/covariance diagnostics are drifting; activate a small chart-learning loss.")

    if topology_loss > 0.0 or bgg_d2_residual > 0.0 or koszul_persistence > 0.0:
        delta["topology_bgg_koszul_policy"] = "analyze_and_train_on_auxiliary_branch_before_primary_bpb_gate"
        rationale.append("Topology, BGG, or Koszul diagnostics are available; use them to steer sidecar transfer while keeping the competition BPB objective dominant.")

    if slepian_leakage > 0.5 or (slepian_concentration > 0.0 and slepian_concentration < 0.5):
        delta["slepian_pollak_policy"] = "continue_auxiliary_leakage_reduction_do_not_raise_primary_loss_before_threshold"
        rationale.append("Slepian/Pollak concentration is still low; it is useful diagnostic pressure, but heavy primary weighting would risk BPB throughput.")

    if tropical_plateau_pressure > 0.0 and validation_stalled:
        delta["tropical_attention_policy"] = "try_short_recapture_with_tropical_long_context_controls_after_best_checkpoint"
        rationale.append("Tropical plateau pressure and validation stall coincide; a targeted recapture is preferable to ordinary continuation.")

    if branch_gap > 0.75:
        delta["gflownet_loss_weight"] = "enable_tiny_2.5e-4_to_5e-4"
        delta["gflownet_entropy_weight"] = "tiny_5e-5_to_1e-4_at_target_2.0"
        rationale.append("Branch search finds much better terminals than the mean branch; promote a tiny GFlowNet objective so the policy begins learning that branch gap while BPB remains dominant.")

    if toric_margin < 0.04 or active_face_margin < 0.0:
        delta["toric_geometry_loss_weight"] = "keep_zero_until_margins_improve"
        delta["koszul_persistence_loss_weight"] = "keep_zero_until_margins_improve"
        rationale.append("Toric active-face margins are weak; heavy toric/Koszul losses would likely fight BPB.")

    if analogy_map.get("category") == "not_as_desired":
        delta["contrastive_loss_weight"] = "increase_small_while_graphcg_on"
        rationale.append("Analogical map residual is drifting; a small contrastive increase should help before topology losses are promoted.")

    if not rationale:
        rationale.append("No decisive metric failure detected; continue current phase and recheck at the next gate.")

    output_json = Path(args.output_json) if args.output_json else root / "training_adjustment_proposal.json"
    output_json.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md = Path(args.output_md) if args.output_md else root / "training_adjustment_proposal.md"
    lines = [
        "# Training Adjustment Proposal",
        "",
        f"- Analysis directory: `{root}`",
        f"- Search policy: `{proposal['search_policy']}`",
        "",
        "## Signals",
    ]
    for key, value in proposal["signals"].items():
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
