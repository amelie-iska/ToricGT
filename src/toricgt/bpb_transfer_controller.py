"""BPB-transfer controller for advanced ToricGT metric families.

The controller is intentionally lightweight and JSON-friendly.  It does not
replace the trainer's primary BPB objective; it converts periodic analysis
signals into conservative family-level recommendations for whether advanced
losses should stay as sidecars, receive a guarded low-scale auxiliary weight,
or move into post-threshold reasoning phases.
"""

from __future__ import annotations

import math
from typing import Any


DEFAULT_FAMILIES: tuple[str, ...] = (
    "bpb_gap",
    "topology_directed",
    "toric_slepian",
    "bgg_koszul",
    "tropical_complexity",
)

FAMILY_LOSS_KEYS: dict[str, tuple[str, ...]] = {
    "bpb_gap": ("primary_cross_entropy", "fineweb_bpb"),
    "topology_directed": (
        "analogy_lattice_loss_weight",
        "trajectory_flow_loss_weight",
    ),
    "toric_slepian": (
        "toric_geometry_loss_weight",
        "slepian_pollak_loss_weight",
        "toric_entropy_loss_weight",
    ),
    "bgg_koszul": (
        "toric_bgg_loss_weight",
        "koszul_persistence_loss_weight",
    ),
    "tropical_complexity": (
        "tropical_attention_mode",
        "complexity_sidecar_weight",
        "qat_loss_weight",
    ),
}


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _first_finite(mapping: dict[str, Any] | None, keys: tuple[str, ...], default: float = float("nan")) -> float:
    if not mapping:
        return default
    for key in keys:
        value = _finite_float(mapping.get(key), default=float("nan"))
        if math.isfinite(value):
            return value
    return default


def _artifact_total_bytes(report: dict[str, Any] | None) -> float:
    return _first_finite(
        report,
        (
            "artifact/int8_zlib_total_bytes",
            "artifact_total_bytes",
            "int8_zlib_total_bytes",
            "total_bytes",
        ),
    )


def _artifact_under_limit(report: dict[str, Any] | None, total_bytes: float, limit: int) -> bool | None:
    explicit = _first_finite(
        report,
        (
            "artifact/under_size_limit",
            "artifact_under_size_limit",
            "under_size_limit",
        ),
    )
    if math.isfinite(explicit):
        return bool(explicit >= 0.5)
    if math.isfinite(total_bytes):
        return bool(total_bytes <= float(limit))
    return None


def _artifact_policy(
    artifact_report: dict[str, Any] | None,
    *,
    artifact_size_limit_bytes: int,
    min_artifact_margin_bytes: int,
) -> tuple[str, float, bool | None, float]:
    total_bytes = _artifact_total_bytes(artifact_report)
    under = _artifact_under_limit(artifact_report, total_bytes, artifact_size_limit_bytes)
    if not math.isfinite(total_bytes):
        return "unknown", float("nan"), under, total_bytes
    margin = float(artifact_size_limit_bytes) - total_bytes
    if under is False or margin < 0.0:
        return "over_limit_export_guard_required", margin, False, total_bytes
    if margin < float(min_artifact_margin_bytes):
        return "under_limit_tight_margin", margin, True, total_bytes
    return "under_limit", margin, True, total_bytes


def _target_state(current_report: dict[str, Any], target_bpb: float) -> tuple[str, float]:
    best = _first_finite(current_report, ("best_val_bpb", "latest_val_bpb", "model_bpb"))
    if math.isfinite(best) and best <= float(target_bpb):
        return "threshold_reached", best
    if math.isfinite(best):
        return "pre_threshold", best
    return "unknown", best


def _family_evidence_action(evidence_report: dict[str, Any] | None, family: str) -> str:
    evidence = (evidence_report or {}).get("family_evidence")
    if isinstance(evidence, dict):
        item = evidence.get(family)
        if isinstance(item, dict):
            action = item.get("evidence_action")
            if isinstance(action, str) and action:
                return action
    return "insufficient_history"


def _pre_threshold_family_control(
    *,
    family: str,
    pressure: float,
    evidence_action: str,
    velocity_shortfall_pressure: float,
    artifact_size_policy: str,
) -> tuple[str, float, str]:
    if family == "bpb_gap":
        return "primary_bpb_objective", 1.0, "Primary competition BPB remains the objective."
    if artifact_size_policy == "over_limit_export_guard_required":
        return "artifact_guard_sidecar_only", 0.0, "Export is over the 16MB limit, so avoid auxiliary promotion."

    pressure = max(0.0, pressure if math.isfinite(pressure) else 0.0)
    shortfall = max(0.0, velocity_shortfall_pressure if math.isfinite(velocity_shortfall_pressure) else 0.0)
    if evidence_action == "supports_guarded_velocity" and pressure >= 0.20 and shortfall >= 0.25:
        scale = min(0.30, 0.06 + 0.40 * min(pressure, 0.60) + 0.08 * min(shortfall, 1.0))
        return "guarded_aux_loss", float(scale), "Historical evidence supports a low-scale BPB-transfer probe."
    if evidence_action == "sidecar_or_damp":
        return "sidecar_or_damp", 0.0, "Historical evidence is adverse, so use as a damping/diagnostic signal."
    if evidence_action == "monitor_as_context" and pressure >= 0.35 and shortfall >= 0.60:
        return "watch_low_scale_probe", min(0.08, 0.02 + 0.10 * min(pressure, 0.60)), "Pressure is high but evidence is mixed."
    return "sidecar_only", 0.0, "Insufficient BPB-transfer evidence before the threshold checkpoint."


def _post_threshold_family_control(
    *,
    family: str,
    pressure: float,
    evidence_action: str,
    artifact_size_policy: str,
) -> tuple[str, float, str]:
    if family == "bpb_gap":
        return "preserved_competition_checkpoint", 1.0, "Competition BPB checkpoint should remain immutable."
    if artifact_size_policy == "over_limit_export_guard_required":
        return "advanced_phase_after_export_guard", 0.15, "Fix competition export first, then train advanced phases separately."
    if evidence_action == "supports_guarded_velocity":
        return "post_threshold_advanced_phase", 1.0, "Promote after preserving the competition checkpoint."
    if evidence_action == "sidecar_or_damp":
        return "post_threshold_damped_phase", 0.15, "Use cautiously; evidence was adverse during BPB competition phase."
    if evidence_action == "monitor_as_context":
        return "post_threshold_advanced_phase", 0.35, "Mixed evidence is acceptable after checkpoint preservation."
    scale = 0.25 + 0.25 * min(max(pressure, 0.0), 1.0)
    return "post_threshold_advanced_phase", float(scale), "No history yet; use a measured advanced-phase ramp."


def bpb_transfer_control_report(
    current_report: dict[str, Any] | None,
    evidence_report: dict[str, Any] | None = None,
    artifact_report: dict[str, Any] | None = None,
    *,
    target_bpb: float = 1.2,
    gate_step: int = 4000,
    artifact_size_limit_bytes: int = 16_000_000,
    min_artifact_margin_bytes: int = 250_000,
) -> dict[str, Any]:
    """Return family-level BPB-transfer and artifact-size recommendations."""

    current_report = current_report or {}
    evidence_report = evidence_report or {}
    artifact_report = artifact_report or current_report
    target_state, best_bpb = _target_state(current_report, target_bpb)
    artifact_size_policy, artifact_margin, artifact_under, artifact_total = _artifact_policy(
        artifact_report,
        artifact_size_limit_bytes=int(artifact_size_limit_bytes),
        min_artifact_margin_bytes=int(min_artifact_margin_bytes),
    )
    velocity_shortfall_pressure = _first_finite(
        current_report,
        ("bpb_velocity_shortfall_pressure", "current_velocity_shortfall_pressure"),
        default=0.0,
    )
    family_pressures_raw = current_report.get("structural_family_pressures") or {}
    family_pressures = family_pressures_raw if isinstance(family_pressures_raw, dict) else {}
    families = sorted(set(DEFAULT_FAMILIES) | {str(key) for key in family_pressures})

    family_controls: dict[str, dict[str, Any]] = {}
    for family in families:
        pressure = _finite_float(family_pressures.get(family), default=0.0)
        evidence_action = _family_evidence_action(evidence_report, family)
        if target_state == "threshold_reached":
            mode, scale, rationale = _post_threshold_family_control(
                family=family,
                pressure=pressure,
                evidence_action=evidence_action,
                artifact_size_policy=artifact_size_policy,
            )
        else:
            mode, scale, rationale = _pre_threshold_family_control(
                family=family,
                pressure=pressure,
                evidence_action=evidence_action,
                velocity_shortfall_pressure=velocity_shortfall_pressure,
                artifact_size_policy=artifact_size_policy,
            )
        family_controls[family] = {
            "family": family,
            "pressure": pressure,
            "evidence_action": evidence_action,
            "mode": mode,
            "recommended_loss_scale": float(scale),
            "loss_weight_keys": list(FAMILY_LOSS_KEYS.get(family, ())),
            "rationale": rationale,
        }

    if artifact_size_policy == "over_limit_export_guard_required":
        competition_policy = "export_guard_before_auxiliary_promotion"
    elif target_state == "threshold_reached":
        competition_policy = "preserve_threshold_checkpoint_then_advanced_phases"
    elif any(item["mode"] == "guarded_aux_loss" for item in family_controls.values()):
        competition_policy = "pre_threshold_guarded_bpb_transfer"
    else:
        competition_policy = "pre_threshold_primary_bpb_clean"

    return {
        "controller_version": "bpb_transfer_controller_v1",
        "target_bpb": float(target_bpb),
        "gate_step": int(gate_step),
        "target_checkpoint_state": target_state,
        "best_val_bpb_for_controller": best_bpb,
        "artifact_size_limit_bytes": int(artifact_size_limit_bytes),
        "artifact_total_bytes": artifact_total,
        "artifact_size_margin_bytes": artifact_margin,
        "artifact_under_size_limit": artifact_under,
        "artifact_size_policy": artifact_size_policy,
        "velocity_shortfall_pressure": velocity_shortfall_pressure,
        "competition_phase_policy": competition_policy,
        "family_controls": family_controls,
        "recommended_loss_scales": {
            family: item["recommended_loss_scale"] for family, item in family_controls.items()
        },
        "recommended_modes": {
            family: item["mode"] for family, item in family_controls.items()
        },
    }
