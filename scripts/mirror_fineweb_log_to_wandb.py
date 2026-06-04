#!/usr/bin/env python3
"""Mirror Parameter-Golf FineWeb log metrics to W&B.

The local Parameter-Golf scaffold writes plain-text logs and does not depend on
W&B.  This tailer parses `train_loss` and `val_bpb` lines from an active log and
logs them to W&B under a separate monitoring run without touching the trainer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path


TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+train_loss:(?P<loss>[0-9.]+)"
    r"\s+train_time:(?P<ms>[0-9.]+)ms\s+step_avg:(?P<avg>[0-9.]+)ms"
    r"(?:\s+train_bpb:(?P<bpb>[0-9.]+))?"
)
VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<loss>[0-9.]+)"
    r"\s+val_bpb:(?P<bpb>[0-9.]+)\s+train_time:(?P<ms>[0-9.]+)ms"
    r"\s+step_avg:(?P<avg>[0-9.]+)ms"
)
LOW_TRAIN_BPB_TRIGGER_VAL_RE = re.compile(
    r"low_train_bpb_trigger_val\s+step:(?P<step>\d+)/(?P<total>\d+)"
    r"\s+train_bpb:(?P<train_bpb>[0-9.]+)"
    r"\s+threshold:(?P<threshold>[0-9.]+)"
    r"\s+val_loss:(?P<val_loss>[0-9.]+)"
    r"\s+val_bpb:(?P<val_bpb>[0-9.]+)"
    r"\s+train_time:(?P<ms>[0-9.]+)ms"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "toricgt-parameter-golf"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "amelie-iska-math"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--target-bpb", type=float, default=1.2)
    parser.add_argument(
        "--diagnostics-json",
        default="",
        help="Optional latest full-diagnostics JSON to mirror into dense W&B rows.",
    )
    parser.add_argument(
        "--gate-state-json",
        default="",
        help="Optional 4K recovery gate-state JSON to mirror into dense W&B rows.",
    )
    return parser.parse_args()


def default_diagnostics_json(log_path: Path) -> Path:
    return Path("logs") / f"{log_path.stem}.full_diag.latest.json"


def default_gate_state_json(run_id: str) -> Path:
    return Path("outputs") / f"{run_id}.4k_recovery_state.json"


STRUCTURAL_FAMILY_IDS = {
    "none": 0.0,
    "bpb_gap": 1.0,
    "topology_directed": 2.0,
    "toric_slepian": 3.0,
    "bgg_koszul": 4.0,
    "tropical_complexity": 5.0,
}

CONTROL_POLICY_IDS = {
    "base_controls": 0.0,
    "low_train_bpb_generalization_relief": 1.0,
    "structural_pressure_damped": 2.0,
    "curvature_aware_velocity_recovery": 3.0,
    "velocity_shortfall_recovery": 4.0,
    "validation_gap_recapture": 5.0,
    "lexical_transition_bias_recapture": 6.0,
    "curvature_aware_structural_relief_velocity_recapture": 7.0,
    "structural_relief_velocity_recapture": 8.0,
    "structural_pressure_recapture": 9.0,
    "post_hot_probe_damped_transfer": 10.0,
    "bpb_velocity_recapture": 11.0,
    "failed_train_wave_damped_transfer_probe": 12.0,
}

ADVANCED_METRIC_POLICY_IDS = {
    "primary_bpb_clean": 0.0,
    "monitor_structural_sidecars": 1.0,
    "damp_structural_pressure": 2.0,
    "curvature_recovery": 3.0,
    "graphcg_slepian_sidecar_primary_bpb_clean": 4.0,
    "bigram_bias_primary_bpb_clean_structural_sidecars": 5.0,
    "gate_velocity_shortfall_structural_relief_bpb_velocity": 6.0,
    "guarded_transfer_relief_bpb_velocity": 7.0,
    "toric_topology_slepian_guarded_bpb_recapture": 8.0,
    "hot_probe_failed_validation_damping_structural_sidecars": 9.0,
    "proposal_guided_bpb_recapture_structural_sidecars": 10.0,
    "failed_train_wave_analogue_damp_graphcg_slepian_sidecars": 11.0,
}

RISK_SOURCE_IDS = {
    "validation_projection": 1.0,
    "failed_trajectory_analogue": 2.0,
    "failed_train_wave_analogue": 3.0,
}


def _finite_float(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def add_finite_metric(target: dict[str, float], key: str, value) -> None:
    numeric = _finite_float(value)
    if math.isfinite(numeric):
        target[key] = numeric


def add_numeric_tree(target: dict[str, float], prefix: str, value, *, depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            clean = str(child_key).replace(" ", "_")
            add_numeric_tree(target, f"{prefix}/{clean}", child_value, depth=depth + 1)
        return
    if isinstance(value, bool):
        target[prefix] = float(value)
        return
    add_finite_metric(target, prefix, value)


def latest_diagnostics_aliases(path: Path, step: int) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    aliases: dict[str, float] = {}
    for key, value in payload.items():
        if not (
            key.startswith("diagnostics/latest/")
            or key.startswith("diagnostics/families/")
            or key.startswith("diagnostics/structural_recapture")
            or key.startswith("fineweb_curve/")
        ):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            aliases[key] = numeric
    diag_step = payload.get("trainer/step")
    try:
        diag_step_value = float(diag_step)
    except (TypeError, ValueError):
        diag_step_value = float("nan")
    if math.isfinite(diag_step_value):
        aliases["diagnostics/latest_full_metrics_step"] = diag_step_value
        aliases["diagnostics/latest/staleness_steps"] = max(0.0, float(step) - diag_step_value)
    return aliases


def latest_gate_state_aliases(path: Path, step: int, target_bpb: float) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    aliases: dict[str, float] = {"gate/state_available": 1.0}
    add_finite_metric(aliases, "gate/latest_step", payload.get("latest_step"))
    add_finite_metric(aliases, "gate/gate_step", payload.get("gate_step"))
    add_finite_metric(aliases, "gate/restart_index", payload.get("restart_index"))
    add_finite_metric(aliases, "gate/target_bpb", payload.get("target_bpb", target_bpb))
    if isinstance(payload.get("target_reached"), bool):
        aliases["gate/target_reached"] = float(bool(payload.get("target_reached")))
    if isinstance(payload.get("advanced_metric_controls_enabled"), bool):
        aliases["advanced_control/enabled"] = float(bool(payload.get("advanced_metric_controls_enabled")))
    if isinstance(payload.get("analogue_risk_enabled"), bool):
        aliases["advanced_control/analogue_risk_enabled"] = float(bool(payload.get("analogue_risk_enabled")))

    gate_step = _finite_float(payload.get("gate_step"))
    latest_step = _finite_float(payload.get("latest_step", step))
    if math.isfinite(gate_step) and math.isfinite(latest_step):
        aliases["gate/remaining_steps_to_gate"] = max(0.0, gate_step - latest_step)
        aliases["gate/current_step_over_gate"] = latest_step / max(gate_step, 1.0)

    latest_validation = payload.get("latest_validation") if isinstance(payload.get("latest_validation"), dict) else {}
    best_validation = (
        payload.get("best_validation_at_or_before_gate")
        if isinstance(payload.get("best_validation_at_or_before_gate"), dict)
        else {}
    )
    latest_bpb = _finite_float(latest_validation.get("val_bpb"))
    best_bpb = _finite_float(best_validation.get("val_bpb"))
    if math.isfinite(latest_bpb):
        aliases["gate/latest_val_bpb"] = latest_bpb
        aliases["gate/latest_gap_to_target"] = latest_bpb - float(target_bpb)
    if math.isfinite(best_bpb):
        aliases["gate/best_val_bpb"] = best_bpb
        aliases["gate/best_gap_to_target"] = best_bpb - float(target_bpb)

    diagnostics = payload.get("advanced_diagnostics")
    if isinstance(diagnostics, dict):
        add_numeric_tree(aliases, "gate/advanced_diagnostics", diagnostics)
        family = str(diagnostics.get("dominant_structural_family", "none"))
        aliases["gate/advanced_diagnostics/dominant_structural_family_id"] = STRUCTURAL_FAMILY_IDS.get(
            family, 0.0
        )
        if isinstance(diagnostics.get("structural_pressure_high"), bool):
            aliases["gate/advanced_diagnostics/structural_pressure_high"] = float(
                bool(diagnostics.get("structural_pressure_high"))
            )

    for control_key in ("base_recovery_launch_controls", "recovery_launch_controls"):
        controls = payload.get(control_key)
        if not isinstance(controls, dict):
            continue
        prefix = "advanced_control/base" if control_key.startswith("base") else "advanced_control/recovery"
        add_numeric_tree(aliases, prefix, controls)
        aliases[f"{prefix}/policy_id"] = CONTROL_POLICY_IDS.get(str(controls.get("policy", "")), 0.0)
        aliases[f"{prefix}/advanced_metric_policy_id"] = ADVANCED_METRIC_POLICY_IDS.get(
            str(controls.get("advanced_metric_policy", "")), 0.0
        )

    for risk_key in ("preemptive_gate_risk", "projection_gate_risk", "analogue_gate_risk"):
        risk = payload.get(risk_key)
        prefix = f"gate/{risk_key}"
        aliases[f"{prefix}/active"] = 1.0 if isinstance(risk, dict) else 0.0
        if isinstance(risk, dict):
            add_numeric_tree(aliases, prefix, risk)
            aliases[f"{prefix}/risk_source_id"] = RISK_SOURCE_IDS.get(str(risk.get("risk_source", "")), 0.0)
            matched_runs = risk.get("analogue_matched_runs")
            if isinstance(matched_runs, (list, tuple)):
                aliases[f"{prefix}/analogue_matched_run_count"] = float(len(matched_runs))
    return aliases


def latest_diagnostics_summary_pin_payload(path: Path, step: int) -> dict[str, float]:
    """Return latest diagnostics aliases for summary-only re-pinning.

    Multiple W&B writers resume the same run during BPB recovery.  The primary
    trainer logs every step and can leave the public summary without sidecar
    diagnostics even though the metrics are present in history.  The dense mirror
    periodically re-pins this compact payload so the run summary stays useful.
    """

    payload = latest_diagnostics_aliases(path, step)
    if payload:
        payload["diagnostics/summary_pin_active"] = 1.0
    return payload


def latest_gate_state_summary_pin_payload(path: Path, step: int, target_bpb: float) -> dict[str, float]:
    payload = latest_gate_state_aliases(path, step, target_bpb)
    if payload:
        payload["gate/summary_pin_active"] = 1.0
    return payload


def summary_payload_fingerprint(payload: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(key), round(float(value), 12)) for key, value in payload.items()))


def sync_public_summary(
    wandb_module,
    *,
    entity: str,
    project: str,
    run_id: str,
    summary_payload: dict[str, float],
) -> bool:
    try:
        api_run = wandb_module.Api().run(f"{entity}/{project}/{run_id}")
        for key, value in summary_payload.items():
            api_run.summary[key] = value
        api_run.summary.update()
    except Exception as exc:  # pragma: no cover - W&B availability should not stop the mirror.
        print(f"wandb_summary_sync_failed:{type(exc).__name__}:{exc}", flush=True)
        return False
    return True


def main() -> None:
    args = parse_args()
    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=args.run_id,
        name=args.run_name or args.run_id,
        resume="allow",
        config={
            "source_log": args.log,
            "target_bpb": args.target_bpb,
            "monitor_kind": "fineweb_log_tail",
        },
    )
    wandb.define_metric("*", step_metric="trainer/step")
    path = Path(args.log)
    diagnostics_json = Path(args.diagnostics_json) if args.diagnostics_json else default_diagnostics_json(path)
    gate_state_json = Path(args.gate_state_json) if args.gate_state_json else default_gate_state_json(args.run_id)
    seen: set[tuple[str, int]] = set()
    best_bpb: float | None = None
    initial_bpb: float | None = None
    latest_seen_step = 0
    last_diagnostics_summary_pin: tuple[tuple[str, float], ...] = ()
    while True:
        stop_requested = False
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            text_tail = "\n".join(text.splitlines()[-20:])
            for line in text.splitlines():
                trigger_val = LOW_TRAIN_BPB_TRIGGER_VAL_RE.search(line)
                if trigger_val:
                    step = int(trigger_val.group("step"))
                    latest_seen_step = max(latest_seen_step, step)
                    key = ("low_train_bpb_trigger_val", step)
                    if key in seen:
                        continue
                    seen.add(key)
                    total = int(trigger_val.group("total"))
                    train_bpb = float(trigger_val.group("train_bpb"))
                    threshold = float(trigger_val.group("threshold"))
                    bpb = float(trigger_val.group("val_bpb"))
                    val_loss = float(trigger_val.group("val_loss"))
                    train_time_ms = float(trigger_val.group("ms"))
                    best_bpb = bpb if best_bpb is None else min(best_bpb, bpb)
                    if initial_bpb is None:
                        initial_bpb = bpb
                    target_gap = bpb - args.target_bpb
                    payload = {
                        "trainer/step": step,
                        "trainer/total_steps": total,
                        "trainer/progress": step / max(total, 1),
                        "time/train_ms": train_time_ms,
                        "time/train_seconds": train_time_ms / 1000.0,
                        "progress/step": step,
                        "progress/total_steps": total,
                        "progress/fraction": step / max(total, 1),
                        "progress/remaining_steps": max(total - step, 0),
                        "fineweb/train_bpb": train_bpb,
                        "fineweb/val_loss": val_loss,
                        "fineweb/val_bpb": bpb,
                        "fineweb/best_val_bpb": best_bpb,
                        "fineweb/target_bpb": args.target_bpb,
                        "fineweb/target_gap_bpb": target_gap,
                        "fineweb/target_reached": float(best_bpb <= args.target_bpb),
                        "fineweb/train_time_ms": train_time_ms,
                        "train/bpb": train_bpb,
                        "train_bpb": train_bpb,
                        "val/loss": val_loss,
                        "val/perplexity": math.exp(min(val_loss, 20.0)),
                        "val/bpb": bpb,
                        "val_bpb": bpb,
                        "bpb": bpb,
                        "bpb/train": train_bpb,
                        "bpb/val": bpb,
                        "bpb/best": best_bpb,
                        "bpb/target": args.target_bpb,
                        "bpb/gap_to_target": target_gap,
                        "bpb/improvement_from_initial": (
                            0.0 if initial_bpb is None else initial_bpb - bpb
                        ),
                        "bpb/target_reached": float(best_bpb <= args.target_bpb),
                        "openai_parameter_golf/bpb": bpb,
                        "openai_parameter_golf/best_bpb": best_bpb,
                        "openai_parameter_golf/train_bpb": train_bpb,
                        "openai_parameter_golf/target_bpb": args.target_bpb,
                        "openai_parameter_golf/gap_to_target": target_gap,
                        "trigger/low_train_bpb": 1.0,
                        "trigger/low_train_bpb_threshold": threshold,
                        "trigger/train_bpb": train_bpb,
                        "trigger/val_loss": val_loss,
                        "trigger/val_bpb": bpb,
                        "trigger/best_val_bpb": best_bpb,
                        "checkpoint/reason_low_train_bpb": 1.0,
                    }
                    diagnostic_payload = latest_diagnostics_aliases(diagnostics_json, step)
                    gate_payload = latest_gate_state_aliases(gate_state_json, step, args.target_bpb)
                    payload.update(diagnostic_payload)
                    payload.update(gate_payload)
                    wandb.log(payload)
                    summary_payload = {
                        "fineweb/train_bpb": train_bpb,
                        "fineweb/val_bpb": bpb,
                        "fineweb/best_val_bpb": best_bpb,
                        "fineweb/target_bpb": args.target_bpb,
                        "train/bpb": train_bpb,
                        "train_bpb": train_bpb,
                        "val/bpb": bpb,
                        "val/loss": val_loss,
                        "val_bpb": bpb,
                        "bpb": bpb,
                        "bpb/train": train_bpb,
                        "bpb/val": bpb,
                        "bpb/best": best_bpb,
                        "bpb/target": args.target_bpb,
                        "bpb/gap_to_target": target_gap,
                        "openai_parameter_golf/bpb": bpb,
                        "openai_parameter_golf/best_bpb": best_bpb,
                        "openai_parameter_golf/train_bpb": train_bpb,
                        "openai_parameter_golf/target_bpb": args.target_bpb,
                        "openai_parameter_golf/gap_to_target": target_gap,
                        "trigger/low_train_bpb": 1.0,
                        "trigger/low_train_bpb_threshold": threshold,
                        "trigger/train_bpb": train_bpb,
                        "trigger/val_bpb": bpb,
                        "trigger/best_val_bpb": best_bpb,
                        "checkpoint/reason_low_train_bpb": 1.0,
                        "progress/step": step,
                        "progress/fraction": step / max(total, 1),
                    }
                    summary_payload.update(diagnostic_payload)
                    summary_payload.update(gate_payload)
                    run.summary.update(summary_payload)
                    sync_public_summary(
                        wandb,
                        entity=args.entity,
                        project=args.project,
                        run_id=args.run_id,
                        summary_payload=summary_payload,
                    )
                    print(
                        "wandb_low_train_bpb_trigger "
                        f"step={step} train_bpb={train_bpb:.4f} val_bpb={bpb:.4f} best={best_bpb:.4f}",
                        flush=True,
                    )
                    continue
                val = VAL_RE.search(line)
                if val:
                    step = int(val.group("step"))
                    latest_seen_step = max(latest_seen_step, step)
                    key = ("val", step)
                    if key in seen:
                        continue
                    seen.add(key)
                    total = int(val.group("total"))
                    bpb = float(val.group("bpb"))
                    val_loss = float(val.group("loss"))
                    train_time_ms = float(val.group("ms"))
                    step_avg_ms = float(val.group("avg"))
                    best_bpb = bpb if best_bpb is None else min(best_bpb, bpb)
                    if initial_bpb is None:
                        initial_bpb = bpb
                    target_gap = bpb - args.target_bpb
                    payload = {
                        "trainer/step": step,
                        "trainer/total_steps": total,
                        "trainer/progress": step / max(total, 1),
                        "time/train_ms": train_time_ms,
                        "time/train_seconds": train_time_ms / 1000.0,
                        "time/step_avg_ms": step_avg_ms,
                        "time/steps_per_second": 1000.0 / max(step_avg_ms, 1e-9),
                        "progress/step": step,
                        "progress/total_steps": total,
                        "progress/fraction": step / max(total, 1),
                        "progress/remaining_steps": max(total - step, 0),
                        "fineweb/val_loss": val_loss,
                        "fineweb/val_bpb": bpb,
                        "fineweb/best_val_bpb": best_bpb,
                        "fineweb/target_bpb": args.target_bpb,
                        "fineweb/target_gap_bpb": target_gap,
                        "fineweb/target_reached": float(best_bpb <= args.target_bpb),
                        "fineweb/train_time_ms": train_time_ms,
                        # Generic aliases keep normal BPB dashboards useful.
                        "val/loss": val_loss,
                        "val/perplexity": math.exp(min(val_loss, 20.0)),
                        "val/bpb": bpb,
                        "val_bpb": bpb,
                        "bpb": bpb,
                        "bpb/val": bpb,
                        "bpb/best": best_bpb,
                        "bpb/target": args.target_bpb,
                        "bpb/gap_to_target": target_gap,
                        "bpb/improvement_from_initial": (
                            0.0 if initial_bpb is None else initial_bpb - bpb
                        ),
                        "bpb/target_reached": float(best_bpb <= args.target_bpb),
                        "openai_parameter_golf/bpb": bpb,
                        "openai_parameter_golf/best_bpb": best_bpb,
                        "openai_parameter_golf/target_bpb": args.target_bpb,
                        "openai_parameter_golf/gap_to_target": target_gap,
                    }
                    diagnostic_payload = latest_diagnostics_aliases(diagnostics_json, step)
                    gate_payload = latest_gate_state_aliases(gate_state_json, step, args.target_bpb)
                    payload.update(diagnostic_payload)
                    payload.update(gate_payload)
                    wandb.log(payload)
                    summary_payload = {
                        "fineweb/val_bpb": bpb,
                        "fineweb/best_val_bpb": best_bpb,
                        "fineweb/target_bpb": args.target_bpb,
                        "val/bpb": bpb,
                        "val/loss": val_loss,
                        "val_bpb": bpb,
                        "bpb": bpb,
                        "bpb/val": bpb,
                        "bpb/best": best_bpb,
                        "bpb/target": args.target_bpb,
                        "bpb/gap_to_target": target_gap,
                        "openai_parameter_golf/bpb": bpb,
                        "openai_parameter_golf/best_bpb": best_bpb,
                        "openai_parameter_golf/target_bpb": args.target_bpb,
                        "openai_parameter_golf/gap_to_target": target_gap,
                        "progress/step": step,
                        "progress/fraction": step / max(total, 1),
                    }
                    summary_payload.update(diagnostic_payload)
                    summary_payload.update(gate_payload)
                    run.summary.update(summary_payload)
                    sync_public_summary(
                        wandb,
                        entity=args.entity,
                        project=args.project,
                        run_id=args.run_id,
                        summary_payload=summary_payload,
                    )
                    print(f"wandb_val step={step} val_bpb={bpb:.4f} best={best_bpb:.4f}", flush=True)
                    continue
                train = TRAIN_RE.search(line)
                if train:
                    step = int(train.group("step"))
                    latest_seen_step = max(latest_seen_step, step)
                    key = ("train", step)
                    if key in seen:
                        continue
                    seen.add(key)
                    total = int(train.group("total"))
                    train_loss = float(train.group("loss"))
                    train_bpb = float(train.group("bpb")) if train.group("bpb") is not None else None
                    train_time_ms = float(train.group("ms"))
                    step_avg_ms = float(train.group("avg"))
                    payload = {
                        "trainer/step": step,
                        "trainer/total_steps": total,
                        "trainer/progress": step / max(total, 1),
                        "time/train_ms": train_time_ms,
                        "time/train_seconds": train_time_ms / 1000.0,
                        "time/step_avg_ms": step_avg_ms,
                        "time/steps_per_second": 1000.0 / max(step_avg_ms, 1e-9),
                        "progress/step": step,
                        "progress/total_steps": total,
                        "progress/fraction": step / max(total, 1),
                        "progress/remaining_steps": max(total - step, 0),
                        "fineweb/train_loss": train_loss,
                        "fineweb/train_time_ms": train_time_ms,
                        "train/loss": train_loss,
                        "train/perplexity": math.exp(min(train_loss, 20.0)),
                    }
                    if train_bpb is not None:
                        payload.update(
                            {
                                "fineweb/train_bpb": train_bpb,
                                "train/bpb": train_bpb,
                                "train_bpb": train_bpb,
                                "bpb/train": train_bpb,
                                "openai_parameter_golf/train_bpb": train_bpb,
                            }
                        )
                    diagnostic_payload = latest_diagnostics_aliases(diagnostics_json, step)
                    gate_payload = latest_gate_state_aliases(gate_state_json, step, args.target_bpb)
                    payload.update(diagnostic_payload)
                    payload.update(gate_payload)
                    wandb.log(payload)
                    summary_payload = {
                        "fineweb/train_loss": train_loss,
                        "train/loss": train_loss,
                        "progress/step": step,
                        "progress/fraction": step / max(total, 1),
                    }
                    if train_bpb is not None:
                        summary_payload.update(
                            {
                                "fineweb/train_bpb": train_bpb,
                                "train/bpb": train_bpb,
                                "train_bpb": train_bpb,
                                "bpb/train": train_bpb,
                                "openai_parameter_golf/train_bpb": train_bpb,
                            }
                        )
                    summary_payload.update(diagnostic_payload)
                    summary_payload.update(gate_payload)
                    run.summary.update(summary_payload)
                    sync_public_summary(
                        wandb,
                        entity=args.entity,
                        project=args.project,
                        run_id=args.run_id,
                        summary_payload=summary_payload,
                    )
            stop_requested = any(
                marker in text_tail
                for marker in ("final_int8_zlib_roundtrip", "Traceback", "RuntimeError")
            )
        if latest_seen_step > 0:
            diagnostics_summary_pin = latest_diagnostics_summary_pin_payload(diagnostics_json, latest_seen_step)
            diagnostics_summary_pin.update(
                latest_gate_state_summary_pin_payload(gate_state_json, latest_seen_step, args.target_bpb)
            )
            if diagnostics_summary_pin:
                fingerprint = summary_payload_fingerprint(diagnostics_summary_pin)
                changed = fingerprint != last_diagnostics_summary_pin
                if changed:
                    history_payload = dict(diagnostics_summary_pin)
                    history_payload["trainer/step"] = latest_seen_step
                    history_payload["progress/step"] = latest_seen_step
                    wandb.log(history_payload)
                run.summary.update(diagnostics_summary_pin)
                if sync_public_summary(
                    wandb,
                    entity=args.entity,
                    project=args.project,
                    run_id=args.run_id,
                    summary_payload=diagnostics_summary_pin,
                ) and changed:
                    print(
                        "wandb_diagnostics_summary_pin "
                        f"step={latest_seen_step} keys={len(diagnostics_summary_pin)}",
                        flush=True,
                    )
                last_diagnostics_summary_pin = fingerprint
        if stop_requested:
            break
        time.sleep(max(1.0, args.poll_seconds))
    run.finish()


if __name__ == "__main__":
    main()
