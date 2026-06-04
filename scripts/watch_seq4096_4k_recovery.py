#!/usr/bin/env python3
"""Gate Seq4096 FineWeb runs at 4K and relaunch from the best checkpoint.

If the run reaches the gate step without a validation BPB at or below target,
this watcher kills the active training tmux and starts a fresh recovery run
from the best checkpoint observed at or before the gate.  The recovery gets a
fresh W&B run id and its own analysis/mirror sidecars so replayed step numbers
remain visible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<loss>[0-9.eE+-]+)"
    r"\s+val_bpb:(?P<bpb>[0-9.eE+-]+)"
)
LOW_TRAIN_BPB_TRIGGER_VAL_RE = re.compile(
    r"low_train_bpb_trigger_val\s+step:(?P<step>\d+)/(?P<total>\d+)"
    r"\s+train_bpb:(?P<train_bpb>[0-9.eE+-]+)"
    r"\s+threshold:(?P<threshold>[0-9.eE+-]+)"
    r"\s+val_loss:(?P<val_loss>[0-9.eE+-]+)"
    r"\s+val_bpb:(?P<val_bpb>[0-9.eE+-]+)"
)
TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+train_loss:(?P<loss>[0-9.eE+-]+)"
    r"(?:.*?\btrain_bpb:(?P<bpb>[0-9.eE+-]+))?"
)
CHECKPOINT_RE_TEMPLATE = "{run_id}_step_{step:06d}.pt"


@dataclass(frozen=True)
class ValRow:
    step: int
    total: int
    val_loss: float
    val_bpb: float
    source: str = "scheduled"


@dataclass(frozen=True)
class TrainRow:
    step: int
    total: int
    train_loss: float
    train_bpb: float = float("nan")


@dataclass(frozen=True)
class ParsedLog:
    train_rows: list[TrainRow]
    val_rows: list[ValRow]

    @property
    def latest_step(self) -> int:
        return max([row.step for row in self.train_rows] + [row.step for row in self.val_rows], default=0)


@dataclass(frozen=True)
class RecoveryLaunch:
    training_command: list[str]
    training_shell: str


@dataclass(frozen=True)
class PreemptiveGateRisk:
    analysis_root: str
    latest_analysis_step: int
    latest_projected_target_step: float
    latest_best_val_bpb: float
    latest_recent_val_slope: float
    latest_projected_gate_overrun_steps: float
    latest_velocity_shortfall_pressure: float
    latest_val_velocity_shortfall: float
    missed_projection_count: int
    required_patience: int
    gate_step: int
    target_bpb: float
    risk_source: str = "validation_projection"
    analogue_failed_count: int = 0
    analogue_train_rmse_mean: float = float("nan")
    analogue_train_rmse_min: float = float("nan")
    analogue_projected_target_step_mean: float = float("nan")
    analogue_matched_runs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecoveryControls:
    train_batch_tokens: int
    tied_embed_lr: float
    matrix_lr: float
    scalar_lr: float
    muon_momentum: float
    muon_momentum_warmup_steps: int
    muon_momentum_warmup_start: float
    grad_clip_norm: float | None
    policy: str = "base_controls"
    advanced_metric_policy: str = "primary_bpb_clean"
    rationale: tuple[str, ...] = ()
    bigram_bias: bool = False
    bigram_bias_lr: float = 0.05
    bigram_bias_init_from_data: bool = False
    bigram_bias_init_tokens: int = 100_000_000
    bigram_bias_init_alpha: float = 0.1
    bigram_bias_init_strength: float = 0.35
    bigram_bias_scale: float = 1.0
    advanced_loss_scale: float = 0.0
    graphcg_loss_weight: float = 0.0
    toric_tropical_loss_weight: float = 0.0
    slepian_loss_weight: float = 0.0
    koszul_bgg_loss_weight: float = 0.0
    analogy_loss_weight: float = 0.0
    advanced_loss_sample_tokens: int = 256
    toric_tropical_fan_bins: int = 8
    advanced_loss_log_only: bool = False
    advanced_loss_start_step: int = 0
    advanced_loss_end_step: int = 0
    advanced_loss_every: int = 1
    advanced_loss_warmup_steps: int = 0
    advanced_loss_min_best_val_bpb: float = 0.0
    advanced_loss_max_ce_ratio: float = 0.0
    reset_optimizer_on_resume: bool = True
    reset_rng_on_resume: bool = True
    reset_loader_on_resume: bool = True

    def launch_dict(self) -> dict[str, Any]:
        return {
            "train_batch_tokens": int(self.train_batch_tokens),
            "tied_embed_lr": float(self.tied_embed_lr),
            "matrix_lr": float(self.matrix_lr),
            "scalar_lr": float(self.scalar_lr),
            "muon_momentum": float(self.muon_momentum),
            "muon_momentum_warmup_steps": int(self.muon_momentum_warmup_steps),
            "muon_momentum_warmup_start": float(self.muon_momentum_warmup_start),
            "grad_clip_norm": None if self.grad_clip_norm is None else float(self.grad_clip_norm),
            "policy": self.policy,
            "advanced_metric_policy": self.advanced_metric_policy,
            "rationale": list(self.rationale),
            "bigram_bias": bool(self.bigram_bias),
            "bigram_bias_lr": float(self.bigram_bias_lr),
            "bigram_bias_init_from_data": bool(self.bigram_bias_init_from_data),
            "bigram_bias_init_tokens": int(self.bigram_bias_init_tokens),
            "bigram_bias_init_alpha": float(self.bigram_bias_init_alpha),
            "bigram_bias_init_strength": float(self.bigram_bias_init_strength),
            "bigram_bias_scale": float(self.bigram_bias_scale),
            "advanced_loss_scale": float(self.advanced_loss_scale),
            "graphcg_loss_weight": float(self.graphcg_loss_weight),
            "toric_tropical_loss_weight": float(self.toric_tropical_loss_weight),
            "slepian_loss_weight": float(self.slepian_loss_weight),
            "koszul_bgg_loss_weight": float(self.koszul_bgg_loss_weight),
            "analogy_loss_weight": float(self.analogy_loss_weight),
            "advanced_loss_sample_tokens": int(self.advanced_loss_sample_tokens),
            "toric_tropical_fan_bins": int(self.toric_tropical_fan_bins),
            "advanced_loss_log_only": bool(self.advanced_loss_log_only),
            "advanced_loss_start_step": int(self.advanced_loss_start_step),
            "advanced_loss_end_step": int(self.advanced_loss_end_step),
            "advanced_loss_every": int(self.advanced_loss_every),
            "advanced_loss_warmup_steps": int(self.advanced_loss_warmup_steps),
            "advanced_loss_min_best_val_bpb": float(self.advanced_loss_min_best_val_bpb),
            "advanced_loss_max_ce_ratio": float(self.advanced_loss_max_ce_ratio),
            "reset_optimizer_on_resume": bool(self.reset_optimizer_on_resume),
            "reset_rng_on_resume": bool(self.reset_rng_on_resume),
            "reset_loader_on_resume": bool(self.reset_loader_on_resume),
        }


def recovery_observation_interval(controls: RecoveryControls) -> int:
    """Use tighter observation for advanced-loss recovery probes near 4K."""

    return 50 if controls.advanced_loss_scale > 0 else 250


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sanitize_tmux_name(value: str, limit: int = 96) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)
    return clean[:limit].rstrip("_") or "toricgt_4k_recovery"


def build_recovery_run_id(parent_run_id: str, restart_index: int, stamp: str, max_len: int = 95) -> str:
    """Build a compact W&B-safe id instead of recursively appending parents."""

    parent = str(parent_run_id or "")
    if "seq4096" in parent:
        stem = "toricgt_seq4096"
    else:
        stem = parent.split("_4k_recovery_r", 1)[0] or "toricgt"
        stem = re.sub(r"[^A-Za-z0-9_.:-]+", "_", stem).strip("_") or "toricgt"
        stem = stem[:40].rstrip("_") or "toricgt"
    suffix = f"_4k_recovery_r{int(restart_index)}_{stamp}"
    budget = max(8, int(max_len) - len(suffix))
    return f"{stem[:budget].rstrip('_')}{suffix}"


def parse_seq4096_log(path: Path | str) -> ParsedLog:
    train: dict[int, TrainRow] = {}
    vals: dict[int, ValRow] = {}
    path = Path(path)
    if not path.exists():
        return ParsedLog([], [])
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        low_trigger = LOW_TRAIN_BPB_TRIGGER_VAL_RE.search(line)
        if low_trigger:
            step = int(low_trigger.group("step"))
            total = int(low_trigger.group("total"))
            train[step] = TrainRow(
                step=step,
                total=total,
                train_loss=float("nan"),
                train_bpb=float(low_trigger.group("train_bpb")),
            )
            vals[step] = ValRow(
                step=step,
                total=total,
                val_loss=float(low_trigger.group("val_loss")),
                val_bpb=float(low_trigger.group("val_bpb")),
                source="low_train_bpb",
            )
            continue
        val = VAL_RE.search(line)
        if val:
            step = int(val.group("step"))
            vals[step] = ValRow(
                step=step,
                total=int(val.group("total")),
                val_loss=float(val.group("loss")),
                val_bpb=float(val.group("bpb")),
            )
            continue
        train_match = TRAIN_RE.search(line)
        if train_match:
            step = int(train_match.group("step"))
            train[step] = TrainRow(
                step=step,
                total=int(train_match.group("total")),
                train_loss=float(train_match.group("loss")),
                train_bpb=finite_float(train_match.group("bpb")),
            )
    return ParsedLog(
        train_rows=[train[key] for key in sorted(train)],
        val_rows=[vals[key] for key in sorted(vals)],
    )


def select_best_validation(rows: list[ValRow], gate_step: int) -> ValRow | None:
    candidates = [row for row in rows if row.step <= int(gate_step) and math.isfinite(row.val_bpb)]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row.val_bpb, -row.step))


def select_recovery_validation(
    rows: list[ValRow],
    gate_step: int,
    min_recovery_runway_steps: int = 500,
    late_better_margin_bpb: float = 5e-4,
) -> ValRow | None:
    """Select the checkpoint to restart from after a missed gate.

    If the gate step itself is the best-but-still-missed checkpoint, restarting
    there gives the recovery run no training room before the same gate.  Prefer
    the best validation with enough room to retrain before the gate; fall back
    to the best earlier validation only when no roomy validation exists, and
    then to the inclusive best only when no earlier validation exists.
    """

    min_step = int(gate_step) - max(1, int(min_recovery_runway_steps))
    roomy = [row for row in rows if row.step <= min_step and math.isfinite(row.val_bpb)]
    before_gate = [row for row in rows if row.step < int(gate_step) and math.isfinite(row.val_bpb)]
    best_before_gate = min(before_gate, key=lambda row: (row.val_bpb, -row.step)) if before_gate else None
    if roomy:
        roomy_best = min(roomy, key=lambda row: (row.val_bpb, -row.step))
        if (
            best_before_gate is not None
            and best_before_gate.step > roomy_best.step
            and best_before_gate.val_bpb <= roomy_best.val_bpb - max(0.0, float(late_better_margin_bpb))
        ):
            return best_before_gate
        return roomy_best
    if best_before_gate is not None:
        return best_before_gate
    return select_best_validation(rows, gate_step)


def checkpoint_for_step(checkpoint_dir: Path | str, run_id: str, step: int) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    exact = checkpoint_dir / CHECKPOINT_RE_TEMPLATE.format(run_id=run_id, step=int(step))
    if exact.exists():
        return exact.resolve()
    candidates = sorted(checkpoint_dir.glob(f"*_step_{int(step):06d}.pt"))
    return candidates[-1].resolve() if candidates else None


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def analysis_step_from_path(path: Path, payload: dict[str, Any]) -> int:
    for key in ("checkpoint_step", "latest_step"):
        value = payload.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"step-(\d+)", str(path))
    return int(match.group(1)) if match else 0


def load_analysis_status(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_preemptive_gate_risk(
    analysis_root: Path | str,
    *,
    gate_step: int,
    target_bpb: float,
    min_step: int,
    patience: int,
) -> PreemptiveGateRisk | None:
    """Load repeated validation-ETA gate risk from completed analysis reports."""

    root = Path(analysis_root)
    if not root.exists():
        return None
    rows: list[tuple[int, dict[str, Any]]] = []
    for status_path in sorted(root.glob("step-*/analysis_status.json")):
        payload = load_analysis_status(status_path)
        if not payload:
            continue
        step = analysis_step_from_path(status_path, payload)
        if step >= int(min_step):
            rows.append((step, payload))
    if not rows:
        return None
    rows.sort(key=lambda item: item[0])
    missed = 0
    for step, payload in reversed(rows):
        projected = finite_float(payload.get("projected_target_step_from_val"))
        best_val = finite_float(payload.get("best_val_bpb"))
        recent_slope = finite_float(payload.get("val_bpb_recent_slope_per_100_steps"))
        velocity_shortfall_pressure = finite_float(payload.get("bpb_velocity_shortfall_pressure"))
        val_velocity_shortfall = finite_float(payload.get("val_velocity_shortfall_to_gate_per_100_steps"))
        material_velocity_shortfall = (
            velocity_shortfall_pressure >= 0.10
            or (math.isfinite(val_velocity_shortfall) and val_velocity_shortfall >= 0.0005)
        )
        effective_projected = projected
        if (
            not math.isfinite(effective_projected)
            and math.isfinite(recent_slope)
            and recent_slope >= 0.0
            and material_velocity_shortfall
        ):
            effective_projected = float(gate_step) + 1000.0
        projected_overrun = (
            effective_projected - float(gate_step)
            if math.isfinite(effective_projected)
            else float("nan")
        )
        material_projected_miss = effective_projected > float(gate_step) and (
            not math.isfinite(velocity_shortfall_pressure)
            or projected_overrun >= 50.0
            or material_velocity_shortfall
        )
        is_risk = (
            best_val > float(target_bpb)
            and math.isfinite(recent_slope)
            and (recent_slope < 0.0 or material_velocity_shortfall)
            and material_projected_miss
        )
        if not is_risk:
            break
        missed += 1
    latest_step, latest = rows[-1]
    latest_projected = finite_float(latest.get("projected_target_step_from_val"))
    latest_recent_slope = finite_float(latest.get("val_bpb_recent_slope_per_100_steps"))
    latest_velocity_shortfall_pressure = finite_float(latest.get("bpb_velocity_shortfall_pressure"))
    latest_val_velocity_shortfall = finite_float(latest.get("val_velocity_shortfall_to_gate_per_100_steps"))
    latest_material_velocity_shortfall = (
        latest_velocity_shortfall_pressure >= 0.10
        or (math.isfinite(latest_val_velocity_shortfall) and latest_val_velocity_shortfall >= 0.0005)
    )
    if (
        not math.isfinite(latest_projected)
        and math.isfinite(latest_recent_slope)
        and latest_recent_slope >= 0.0
        and latest_material_velocity_shortfall
    ):
        latest_projected = float(gate_step) + 1000.0
    return PreemptiveGateRisk(
        analysis_root=str(root),
        latest_analysis_step=int(latest_step),
        latest_projected_target_step=latest_projected,
        latest_best_val_bpb=finite_float(latest.get("best_val_bpb")),
        latest_recent_val_slope=latest_recent_slope,
        latest_projected_gate_overrun_steps=(
            latest_projected - float(gate_step) if math.isfinite(latest_projected) else float("nan")
        ),
        latest_velocity_shortfall_pressure=latest_velocity_shortfall_pressure,
        latest_val_velocity_shortfall=latest_val_velocity_shortfall,
        missed_projection_count=int(missed),
        required_patience=max(1, int(patience)),
        gate_step=int(gate_step),
        target_bpb=float(target_bpb),
    )


def should_preempt_for_gate_risk(risk: PreemptiveGateRisk | None) -> bool:
    return bool(risk and risk.missed_projection_count >= risk.required_patience)


def should_preempt_at_latest_validation(
    latest_val: ValRow | None,
    risk: PreemptiveGateRisk | None,
    *,
    gate_step: int,
    min_step: int,
) -> bool:
    """Return whether a preemptive restart should fire at the latest validation.

    Low-train-BPB probes are intentionally dense validation check-ins. They are
    useful for saving/analyzing promising train-BPB dips, but a single probe can
    be a noisy transfer readout. Only scheduled validation intervals should be
    allowed to trigger another restart, and only when the risk analysis is at
    least as current as the latest validation row.
    """

    if latest_val is None:
        return False
    if getattr(latest_val, "source", "scheduled") == "low_train_bpb":
        return False
    if latest_val.step < int(min_step) or latest_val.step >= int(gate_step):
        return False
    if risk is None or int(risk.latest_analysis_step) < int(latest_val.step):
        return False
    return should_preempt_for_gate_risk(risk)


def effective_restart_ceiling(restart_index: int, max_restarts: int) -> int:
    """Return an absolute restart ceiling from mixed absolute/remaining inputs.

    Older live gate commands sometimes pass an absolute restart index such as 53
    with a small max-restarts value such as 8. Treat positive low ceilings as a
    remaining restart budget so the gate does not silently refuse to recover.
    """

    restart_index = int(restart_index)
    max_restarts = int(max_restarts)
    if max_restarts <= 0 or max_restarts > restart_index:
        return max_restarts
    return restart_index + max(1, max_restarts)


def should_hold_train_wave_for_validation_probe(
    risk: PreemptiveGateRisk | None,
    *,
    tied_embed_lr: float,
    hot_probe_lr: float = 0.037,
) -> bool:
    """Let explicit hot-velocity probes reach validation instead of relaunching.

    Failed train-wave analogues are useful early branch-rejection signals, but
    repeated replays can otherwise ladder through near-identical 3250 restarts
    without ever producing a validation point for altered controls.  Hot
    velocity probes get one validation readout; damped probes are allowed to
    preempt when they are already replaying a failed damped branch.
    """

    return bool(
        risk
        and risk.risk_source == "failed_train_wave_analogue"
        and should_preempt_for_gate_risk(risk)
        and float(tied_embed_lr) >= float(hot_probe_lr)
    )


def plan_failed_train_wave_recovery_controls(
    base: RecoveryControls,
    *,
    parsed: ParsedLog,
    risk: PreemptiveGateRisk,
    gate_step: int,
) -> RecoveryControls:
    """Turn a repeated failed train-wave analogue into a transfer-stability probe."""

    latest_val = parsed.val_rows[-1] if parsed.val_rows else None
    latest_step = max(parsed.latest_step, latest_val.step if latest_val is not None else 0)
    repeated_damped_branch = base.tied_embed_lr <= 0.0325 and base.bigram_bias_lr <= 0.0065
    if repeated_damped_branch:
        previous_advanced_probe = base.advanced_loss_scale > 0.0
        if previous_advanced_probe:
            advanced_metric_policy = "light_graphcg_slepian_transfer_probe_after_val_regression"
            advanced_loss_scale = max(0.03, min(0.05, base.advanced_loss_scale * 0.25))
            graphcg_loss_weight = max(0.01, min(0.02, base.graphcg_loss_weight * 0.50))
            toric_tropical_loss_weight = max(0.0, min(0.008, base.toric_tropical_loss_weight * 0.25))
            slepian_loss_weight = max(0.005, min(0.01, base.slepian_loss_weight * 0.50))
            koszul_bgg_loss_weight = 0.0
            analogy_loss_weight = max(0.0, min(0.005, base.analogy_loss_weight * 0.50))
            advanced_loss_every = 8
            advanced_loss_max_ce_ratio = 0.0005
            advanced_rationale = (
                "previous advanced-loss branch produced a low train BPB but worsened validation transfer",
                "reduce auxiliary pressure to a scheduled GraphCG/Slepian transfer probe and hold Koszul/BGG for post-threshold",
            )
        else:
            advanced_metric_policy = "guarded_graphcg_toric_slepian_koszul_analogy_losses"
            advanced_loss_scale = 0.02
            graphcg_loss_weight = 0.02
            toric_tropical_loss_weight = 0.005
            slepian_loss_weight = 0.01
            koszul_bgg_loss_weight = 0.0
            analogy_loss_weight = 0.0
            advanced_loss_every = 4
            advanced_loss_max_ce_ratio = 0.001
            advanced_rationale = (
                "begin bounded advanced-loss microprobe: GraphCG basis disentanglement, toric/tropical chamber pressure, and "
                "Slepian/Pollak trajectory concentration with CE-ratio clipping and log-only sidecar metrics",
                "hold Koszul/BGG exactness and analogy transport losses for the post-threshold reasoning-memory phase",
            )
        return replace(
            base,
            train_batch_tokens=min(base.train_batch_tokens, 917_504),
            tied_embed_lr=round(max(0.029, min(base.tied_embed_lr * 0.94, base.tied_embed_lr - 0.0015)), 6),
            matrix_lr=round(max(0.0165, min(base.matrix_lr * 0.92, base.matrix_lr - 0.001)), 6),
            scalar_lr=round(max(0.0165, min(base.scalar_lr * 0.92, base.scalar_lr - 0.001)), 6),
            muon_momentum_warmup_steps=max(
                base.muon_momentum_warmup_steps,
                int(gate_step) + 750,
                int(latest_step) + 1250,
            ),
            bigram_bias=True,
            bigram_bias_lr=round(max(0.004, min(base.bigram_bias_lr * 0.70, base.bigram_bias_lr - 0.001)), 6),
            bigram_bias_scale=round(max(base.bigram_bias_scale, 1.15), 6),
            bigram_bias_init_from_data=False,
            policy="repeated_damped_train_wave_diversity_probe",
            advanced_metric_policy=advanced_metric_policy,
            advanced_loss_scale=advanced_loss_scale,
            graphcg_loss_weight=graphcg_loss_weight,
            toric_tropical_loss_weight=toric_tropical_loss_weight,
            slepian_loss_weight=slepian_loss_weight,
            koszul_bgg_loss_weight=koszul_bgg_loss_weight,
            analogy_loss_weight=analogy_loss_weight,
            advanced_loss_sample_tokens=max(base.advanced_loss_sample_tokens, 256),
            toric_tropical_fan_bins=max(base.toric_tropical_fan_bins, 8),
            advanced_loss_log_only=True,
            advanced_loss_start_step=max(0, int(latest_step) + 50),
            advanced_loss_every=advanced_loss_every,
            advanced_loss_warmup_steps=100,
            advanced_loss_min_best_val_bpb=1.205,
            advanced_loss_max_ce_ratio=advanced_loss_max_ce_ratio,
            rationale=(
                "failed damped train-wave branch has reached the tied/bigram LR floor without validation transfer",
                "matched failed analogue count "
                f"{risk.analogue_failed_count} suggests this is the same recovery basin, not a fresh BPB descent",
                "lower tied, matrix, scalar, and bigram LR and reduce batch tokens to change the optimizer trajectory",
                *advanced_rationale,
            ),
        )
    return replace(
        base,
        tied_embed_lr=round(max(0.032, min(base.tied_embed_lr * 0.92, base.tied_embed_lr - 0.001)), 6),
        matrix_lr=base.matrix_lr,
        scalar_lr=base.scalar_lr,
        muon_momentum_warmup_steps=max(
            base.muon_momentum_warmup_steps,
            int(gate_step) + 500,
            int(latest_step) + 1000,
        ),
        bigram_bias=True,
        bigram_bias_lr=round(max(0.006, min(base.bigram_bias_lr, base.bigram_bias_lr * 0.55, 0.012)), 6),
        bigram_bias_init_from_data=False,
        policy="failed_train_wave_damped_transfer_probe",
        advanced_metric_policy="failed_train_wave_analogue_damp_graphcg_slepian_sidecars",
        rationale=(
            "failed train-wave analogue matched "
            f"{risk.analogue_failed_count} prior runs before a new validation point",
            f"matched analogue family projects target near step {risk.latest_projected_target_step:.1f}, "
            f"beyond gate {gate_step}",
            "damp tied-embedding and bigram transition LR instead of replaying another velocity escalation",
            "hold matrix/scalar LR fixed and keep GraphCG/Slepian/topology/toric/BGG/Koszul signals as sidecar transfer diagnostics",
        ),
    )


def round_up_to_multiple(value: float, multiple: int) -> int:
    if multiple <= 0:
        return int(math.ceil(value))
    return int(math.ceil(float(value) / float(multiple)) * multiple)


def latest_train_bpb_at_or_before(parsed: ParsedLog, step: int) -> float:
    candidates = [
        row.train_bpb
        for row in parsed.train_rows
        if row.step <= int(step) and math.isfinite(row.train_bpb)
    ]
    return candidates[-1] if candidates else float("nan")


def projected_target_step_from_validation(rows: list[ValRow], target_bpb: float) -> tuple[float, float]:
    """Project the target step from the last two validation points."""

    candidates = [row for row in rows if math.isfinite(row.val_bpb)]
    if len(candidates) < 2:
        return float("nan"), float("nan")
    previous, latest = candidates[-2], candidates[-1]
    step_delta = latest.step - previous.step
    if step_delta <= 0:
        return float("nan"), float("nan")
    slope_per_100 = (latest.val_bpb - previous.val_bpb) / (float(step_delta) / 100.0)
    velocity_per_100 = -slope_per_100
    if velocity_per_100 <= 0.0:
        return float("nan"), slope_per_100
    steps_to_target = (latest.val_bpb - float(target_bpb)) / velocity_per_100 * 100.0
    return float(latest.step) + steps_to_target, slope_per_100


def train_profile_rmse(current: ParsedLog, history: ParsedLog, steps: list[int]) -> float:
    current_by_step = {
        row.step: row.train_bpb
        for row in current.train_rows
        if math.isfinite(row.train_bpb)
    }
    history_by_step = {
        row.step: row.train_bpb
        for row in history.train_rows
        if math.isfinite(row.train_bpb)
    }
    diffs = [
        (current_by_step[step] - history_by_step[step]) ** 2
        for step in steps
        if step in current_by_step and step in history_by_step
    ]
    return math.sqrt(sum(diffs) / float(len(diffs))) if diffs else float("nan")


def load_failed_trajectory_analogue_risk(
    current_log: Path | str,
    *,
    history_logs: list[Path | str] | None = None,
    gate_step: int,
    target_bpb: float,
    min_step: int = 3250,
    train_rmse_threshold: float = 0.001,
    min_failed_analogues: int = 2,
    current_projection_gate_margin_steps: float = 100.0,
    min_profile_points: int = 4,
) -> PreemptiveGateRisk | None:
    """Preempt runs replaying prior train-BPB trajectories that later missed the gate."""

    current_path = Path(current_log)
    current = parse_seq4096_log(current_path)
    latest_val = current.val_rows[-1] if current.val_rows else None
    if latest_val is None or latest_val.step < int(min_step) or latest_val.step >= int(gate_step):
        return None
    best_current = select_best_validation(current.val_rows, gate_step)
    if best_current is None or best_current.val_bpb <= float(target_bpb):
        return None
    current_projected, current_slope = projected_target_step_from_validation(current.val_rows, target_bpb)
    if not math.isfinite(current_projected):
        return None
    if current_projected <= float(gate_step):
        return None

    current_train_steps = [
        row.step
        for row in current.train_rows
        if row.step <= latest_val.step and row.step >= latest_val.step - 250 and math.isfinite(row.train_bpb)
    ]
    profile_steps = sorted(current_train_steps)[-5:]
    if len(profile_steps) < int(min_profile_points):
        return None

    if history_logs is None:
        history_logs = sorted(current_path.parent.glob("toricgt_seq4096_4k_recovery_r*.txt"))
    matched: list[dict[str, Any]] = []
    for history_log in history_logs:
        history_path = Path(history_log)
        if history_path.resolve() == current_path.resolve():
            continue
        history = parse_seq4096_log(history_path)
        if len(history.val_rows) < 2:
            continue
        history_after_current = [
            row
            for row in history.val_rows
            if row.step > latest_val.step and row.step <= int(gate_step)
        ]
        if not history_after_current:
            continue
        projected_history, _history_slope = projected_target_step_from_validation(
            [row for row in history.val_rows if row.step <= history_after_current[-1].step],
            target_bpb,
        )
        best_history = select_best_validation(
            [row for row in history.val_rows if row.step <= history_after_current[-1].step],
            gate_step,
        )
        if (
            best_history is None
            or best_history.val_bpb <= float(target_bpb)
            or not math.isfinite(projected_history)
            or projected_history <= float(gate_step)
        ):
            continue
        rmse = train_profile_rmse(current, history, profile_steps)
        if math.isfinite(rmse) and rmse <= float(train_rmse_threshold):
            matched.append(
                {
                    "path": str(history_path),
                    "rmse": rmse,
                    "projected_target_step": projected_history,
                    "latest_history_step": history_after_current[-1].step,
                    "latest_history_val_bpb": history_after_current[-1].val_bpb,
                }
            )

    if len(matched) < int(min_failed_analogues):
        return None
    rmse_values = [float(item["rmse"]) for item in matched]
    projected_values = [float(item["projected_target_step"]) for item in matched]
    analogue_projected = sum(projected_values) / float(len(projected_values))
    return PreemptiveGateRisk(
        analysis_root=str(current_path.parent),
        latest_analysis_step=int(latest_val.step),
        latest_projected_target_step=max(float(current_projected), float(analogue_projected)),
        latest_best_val_bpb=float(best_current.val_bpb),
        latest_recent_val_slope=float(current_slope),
        latest_projected_gate_overrun_steps=max(float(current_projected), float(analogue_projected))
        - float(gate_step),
        latest_velocity_shortfall_pressure=0.0,
        latest_val_velocity_shortfall=0.0,
        missed_projection_count=int(len(matched)),
        required_patience=max(1, int(min_failed_analogues)),
        gate_step=int(gate_step),
        target_bpb=float(target_bpb),
        risk_source="failed_trajectory_analogue",
        analogue_failed_count=int(len(matched)),
        analogue_train_rmse_mean=sum(rmse_values) / float(len(rmse_values)),
        analogue_train_rmse_min=min(rmse_values),
        analogue_projected_target_step_mean=analogue_projected,
        analogue_matched_runs=tuple(str(item["path"]) for item in matched),
    )


def load_train_wave_analogue_risk(
    current_log: Path | str,
    *,
    history_logs: list[Path | str] | None = None,
    gate_step: int,
    target_bpb: float,
    min_step: int = 3350,
    train_rmse_threshold: float = 0.001,
    min_failed_analogues: int = 2,
    min_profile_points: int = 2,
) -> PreemptiveGateRisk | None:
    """Preempt post-resume train waves that match prior failed validation branches."""

    current_path = Path(current_log)
    current = parse_seq4096_log(current_path)
    latest_val = current.val_rows[-1] if current.val_rows else None
    if latest_val is None or latest_val.val_bpb <= float(target_bpb):
        return None
    latest_step = current.latest_step
    if latest_step < int(min_step) or latest_val.step >= int(gate_step):
        return None
    profile_steps = [
        row.step
        for row in current.train_rows
        if row.step > latest_val.step
        and row.step <= latest_step
        and math.isfinite(row.train_bpb)
    ]
    profile_steps = sorted(profile_steps)[-5:]
    if len(profile_steps) < int(min_profile_points):
        return None

    if history_logs is None:
        history_logs = sorted(current_path.parent.glob("toricgt_seq4096_4k_recovery_r*.txt"))
    matched: list[dict[str, Any]] = []
    for history_log in history_logs:
        history_path = Path(history_log)
        if history_path.resolve() == current_path.resolve():
            continue
        history = parse_seq4096_log(history_path)
        history_future_vals = [
            row
            for row in history.val_rows
            if row.step > latest_val.step and row.step <= int(gate_step)
        ]
        if not history_future_vals:
            continue
        best_history = select_best_validation(
            [row for row in history.val_rows if row.step <= history_future_vals[-1].step],
            gate_step,
        )
        if best_history is None or best_history.val_bpb <= float(target_bpb):
            continue
        projected_history, history_slope = projected_target_step_from_validation(
            [row for row in history.val_rows if row.step <= history_future_vals[-1].step],
            target_bpb,
        )
        failed_projection = (
            not math.isfinite(projected_history)
            or projected_history > float(gate_step)
            or (math.isfinite(history_slope) and history_slope >= 0.0)
        )
        if not failed_projection:
            continue
        rmse = train_profile_rmse(current, history, profile_steps)
        if math.isfinite(rmse) and rmse <= float(train_rmse_threshold):
            matched.append(
                {
                    "path": str(history_path),
                    "rmse": rmse,
                    "projected_target_step": projected_history
                    if math.isfinite(projected_history)
                    else float(gate_step) + 1000.0,
                    "latest_history_step": history_future_vals[-1].step,
                    "latest_history_val_bpb": history_future_vals[-1].val_bpb,
                }
            )

    if len(matched) < int(min_failed_analogues):
        return None
    rmse_values = [float(item["rmse"]) for item in matched]
    projected_values = [float(item["projected_target_step"]) for item in matched]
    analogue_projected = sum(projected_values) / float(len(projected_values))
    return PreemptiveGateRisk(
        analysis_root=str(current_path.parent),
        latest_analysis_step=int(latest_step),
        latest_projected_target_step=max(float(gate_step) + 1000.0, float(analogue_projected)),
        latest_best_val_bpb=float(latest_val.val_bpb),
        latest_recent_val_slope=float("nan"),
        latest_projected_gate_overrun_steps=max(float(gate_step) + 1000.0, float(analogue_projected))
        - float(gate_step),
        latest_velocity_shortfall_pressure=1.0,
        latest_val_velocity_shortfall=float("nan"),
        missed_projection_count=int(len(matched)),
        required_patience=max(1, int(min_failed_analogues)),
        gate_step=int(gate_step),
        target_bpb=float(target_bpb),
        risk_source="failed_train_wave_analogue",
        analogue_failed_count=int(len(matched)),
        analogue_train_rmse_mean=sum(rmse_values) / float(len(rmse_values)),
        analogue_train_rmse_min=min(rmse_values),
        analogue_projected_target_step_mean=analogue_projected,
        analogue_matched_runs=tuple(str(item["path"]) for item in matched),
    )


def load_advanced_diagnostics(repo_root: Path | str, run_id: str) -> dict[str, Any]:
    """Load the latest full-diagnostics sidecar payload for a Seq4096 run."""

    path = Path(repo_root) / "logs" / f"{run_id}.full_diag.latest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    payload = dict(payload)
    payload["_source_path"] = str(path)
    return payload


def diagnostic_float(payload: dict[str, Any] | None, *keys: str, default: float = float("nan")) -> float:
    if not payload:
        return default
    for key in keys:
        if key in payload:
            return finite_float(payload.get(key), default=default)
    return default


def diagnostic_available(payload: dict[str, Any] | None, key: str) -> bool:
    return diagnostic_float(payload, key, default=0.0) >= 0.5


def bounded_pressure(value: float, scale: float, *, invert: bool = False, floor: float = 0.0) -> float:
    if not math.isfinite(value) or scale <= 0.0:
        return 0.0
    measured = max(0.0, floor - value) if invert else max(0.0, value - floor)
    return max(0.0, min(1.0, measured / float(scale)))


def structural_family_pressures(components: dict[str, float] | None) -> dict[str, float]:
    """Group fine-grained recapture components into controller-level families."""

    components = components or {}

    def value(key: str) -> float:
        return finite_float(components.get(key), default=0.0)

    return {
        "bpb_gap": value("bpb_gap_pressure"),
        "topology_directed": value("topology_loss") + value("directed_topology_loss"),
        "toric_slepian": value("slepian_leakage")
        + value("toric_negative_margin")
        + value("toric_shadow_bend"),
        "bgg_koszul": value("bgg_standard_leakage") + value("bgg_d2_residual"),
        "tropical_complexity": value("complexity_ncd") + value("tropical_plateau"),
    }


def dominant_structural_family(families: dict[str, float] | None) -> tuple[str, float]:
    families = {
        str(key): finite_float(value, default=0.0)
        for key, value in (families or {}).items()
        if finite_float(value, default=0.0) > 0.0
    }
    if not families:
        return "none", 0.0
    key, value = max(families.items(), key=lambda item: (item[1], item[0]))
    return key, float(value)


def summarize_advanced_diagnostics(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Compact structural-pressure summary used by the recovery controller."""

    if not payload:
        return {"available": False, "structural_pressure_high": False, "structural_recapture_score": 0.0}
    bpb_pressure = diagnostic_float(payload, "diagnostics/latest/bpb_intervention_pressure", default=0.0)
    topology_loss = diagnostic_float(payload, "diagnostics/latest/topology_loss", "topology/topology_loss")
    directed_topology_loss = diagnostic_float(
        payload,
        "diagnostics/latest/directed_topology_loss",
        "topology/directed_topology_loss",
    )
    slepian_leakage = diagnostic_float(
        payload,
        "diagnostics/latest/slepian_leakage",
        "diagnostics/latest/pollak_prolate_slepian_leakage",
        "toric/slepian_leakage",
    )
    toric_margin = diagnostic_float(
        payload,
        "diagnostics/latest/toric_active_face_margin",
        "toric/active_face_margin",
    )
    toric_bend = diagnostic_float(
        payload,
        "diagnostics/latest/toric_shadow_mean_bend",
        "toric/shadow_mean_bend",
    )
    tropical_plateau = diagnostic_float(
        payload,
        "tropical/bpb_plateau_pressure",
        "diagnostics/latest/tropical_bpb_plateau_pressure",
        default=0.0,
    )
    bgg_standard_leakage = diagnostic_float(
        payload,
        "diagnostics/latest/bgg_standard_leakage",
        "bgg_category_o/standard_leakage",
        default=0.0,
    )
    bgg_d2_residual = diagnostic_float(
        payload,
        "diagnostics/latest/bgg_d2_residual",
        "bgg_category_o/d2_residual",
        default=0.0,
    )
    complexity_ncd = diagnostic_float(
        payload,
        "diagnostics/latest/complexity_recent_full_log_ncd_lzma",
        "complexity/recent_full_log_ncd_lzma",
        default=0.0,
    )
    family_available = any(
        diagnostic_available(payload, key)
        for key in (
            "diagnostics/families/topology_available",
            "diagnostics/families/toric_available",
            "diagnostics/families/slepian_pollak_prolate_available",
            "diagnostics/families/koszul_persistence_available",
            "diagnostics/families/category_o_bgg_available",
            "diagnostics/families/tropical_available",
        )
    )
    recapture_components = {
        "bpb_gap_pressure": 0.16 * bounded_pressure(bpb_pressure, 0.12),
        "topology_loss": 0.17 * bounded_pressure(topology_loss, 1.40),
        "directed_topology_loss": 0.12 * bounded_pressure(directed_topology_loss, 0.24),
        "slepian_leakage": 0.13 * bounded_pressure(slepian_leakage, 1.0),
        "toric_negative_margin": 0.13 * bounded_pressure(toric_margin, 2.0, invert=True),
        "toric_shadow_bend": 0.12 * bounded_pressure(toric_bend, 2.0),
        "bgg_standard_leakage": 0.08 * bounded_pressure(bgg_standard_leakage, 1.0),
        "bgg_d2_residual": 0.04 * bounded_pressure(bgg_d2_residual, 0.10),
        "complexity_ncd": 0.03 * bounded_pressure(complexity_ncd, 1.0),
        "tropical_plateau": 0.02 * bounded_pressure(tropical_plateau, 0.04),
    }
    family_pressures = structural_family_pressures(recapture_components)
    dominant_family, dominant_family_pressure = dominant_structural_family(family_pressures)
    structural_recapture_score = max(0.0, min(1.0, sum(recapture_components.values())))
    structural_pressure_high = bool(
        family_available
        and (
            structural_recapture_score >= 0.50
            or bpb_pressure >= 0.16
        )
    )
    if not family_available:
        structural_band = "bpb_curve_only"
    elif structural_recapture_score >= 0.75:
        structural_band = "high"
    elif structural_recapture_score >= 0.50:
        structural_band = "guarded"
    elif structural_recapture_score >= 0.30:
        structural_band = "watch"
    else:
        structural_band = "low"
    return {
        "available": True,
        "source_path": str(payload.get("_source_path", "")),
        "structural_pressure_high": structural_pressure_high,
        "structural_recapture_score": structural_recapture_score,
        "structural_recapture_band": structural_band,
        "structural_recapture_components": recapture_components,
        "structural_family_pressures": family_pressures,
        "dominant_structural_family": dominant_family,
        "dominant_structural_family_pressure": dominant_family_pressure,
        "bpb_intervention_pressure": bpb_pressure,
        "topology_loss": topology_loss,
        "directed_topology_loss": directed_topology_loss,
        "slepian_leakage": slepian_leakage,
        "toric_active_face_margin": toric_margin,
        "toric_shadow_mean_bend": toric_bend,
        "tropical_bpb_plateau_pressure": tropical_plateau,
        "bgg_standard_leakage": bgg_standard_leakage,
        "bgg_d2_residual": bgg_d2_residual,
        "complexity_recent_full_log_ncd_lzma": complexity_ncd,
    }


def plan_metric_driven_recovery_controls(
    base: RecoveryControls,
    *,
    parsed: ParsedLog,
    target_bpb: float,
    gate_step: int,
    projected_target_step: float = float("nan"),
    max_train_batch_tokens: int | None = None,
    validation_gap_threshold: float = 0.04,
    low_train_bpb_margin: float = 0.0,
    enabled: bool = True,
    advanced_diagnostics: dict[str, Any] | None = None,
) -> RecoveryControls:
    """Adapt the next recovery launch to BPB and structural-analysis signals.

    The Seq4096 competition runner does not carry the richer GraphCG/Slepian
    losses.  When the analyses indicate structural pressure, the safe action
    for the primary BPB run is to keep that runner BPB-clean while using its
    train/validation geometry to choose optimizer controls.
    """

    if not enabled:
        return replace(base, policy="base_controls", rationale=("advanced metric controls disabled",))

    latest_val = parsed.val_rows[-1] if parsed.val_rows else None
    if latest_val is None:
        return replace(base, policy="base_controls", rationale=("waiting for validation BPB",))

    train_bpb = latest_train_bpb_at_or_before(parsed, latest_val.step)
    validation_gap = (
        latest_val.val_bpb - train_bpb if math.isfinite(train_bpb) else float("nan")
    )
    diagnostics_summary = summarize_advanced_diagnostics(advanced_diagnostics)
    projected_miss = (
        math.isfinite(projected_target_step)
        and projected_target_step > float(gate_step)
        and latest_val.step < int(gate_step)
        and latest_val.val_bpb > float(target_bpb)
    )
    gate_miss = latest_val.step >= int(gate_step) and latest_val.val_bpb > float(target_bpb)
    train_already_low = (
        math.isfinite(train_bpb)
        and train_bpb <= float(target_bpb) + float(low_train_bpb_margin)
    )
    validation_lagging = (
        math.isfinite(validation_gap)
        and validation_gap >= float(validation_gap_threshold)
        and latest_val.val_bpb > float(target_bpb)
    )

    if train_already_low and validation_lagging:
        batch_cap = (
            int(max_train_batch_tokens)
            if max_train_batch_tokens is not None and int(max_train_batch_tokens) > 0
            else int(base.train_batch_tokens)
        )
        raised_batch = round_up_to_multiple(base.train_batch_tokens * 1.07, 65_536)
        return replace(
            base,
            train_batch_tokens=max(base.train_batch_tokens, min(batch_cap, raised_batch)),
            tied_embed_lr=round(max(0.028, base.tied_embed_lr * 0.875), 6),
            matrix_lr=round(max(0.017, base.matrix_lr * 0.90), 6),
            scalar_lr=round(max(0.017, base.scalar_lr * 0.90), 6),
            muon_momentum_warmup_steps=max(base.muon_momentum_warmup_steps, 500),
            policy="validation_gap_recapture",
            advanced_metric_policy="graphcg_slepian_sidecar_primary_bpb_clean",
            rationale=(
                f"train BPB {train_bpb:.4f} is already at/below target while validation BPB "
                f"{latest_val.val_bpb:.4f} lags by {validation_gap:.4f}",
                "increase effective batch for steadier validation transfer",
                "lower tied/matrix/scalar learning rates instead of pushing train BPB harder",
                "keep GraphCG/Slepian/topology losses in sidecar transfer until the competition checkpoint is preserved",
            ),
        )

    if projected_miss or gate_miss:
        batch_cap = (
            int(max_train_batch_tokens)
            if max_train_batch_tokens is not None and int(max_train_batch_tokens) > 0
            else int(base.train_batch_tokens)
        )
        raised_batch = round_up_to_multiple(base.train_batch_tokens * 1.07, 65_536)
        if not base.bigram_bias and base.tied_embed_lr >= 0.0395:
            return replace(
                base,
                train_batch_tokens=max(base.train_batch_tokens, min(batch_cap, raised_batch)),
                bigram_bias=True,
                bigram_bias_init_from_data=False,
                policy="lexical_transition_bias_recapture",
                advanced_metric_policy="bigram_bias_primary_bpb_clean_structural_sidecars",
                rationale=(
                    f"validation projects target at step {projected_target_step:.1f}, beyond gate {gate_step}",
                    "tied-embedding LR is already at the recapture cap",
                    "enable a zero-initialized learned bigram transition-bias head so initial BPB is preserved",
                    "keep GraphCG/Slepian/topology losses in sidecar transfer while adding only BPB-native lexical bias",
                ),
            )
        if base.bigram_bias and diagnostics_summary.get("structural_pressure_high"):
            structural_score = finite_float(diagnostics_summary.get("structural_recapture_score"), default=0.0)
            structural_band = str(diagnostics_summary.get("structural_recapture_band", "unknown"))
            dominant_family = str(diagnostics_summary.get("dominant_structural_family", "none"))
            dominant_pressure = finite_float(
                diagnostics_summary.get("dominant_structural_family_pressure"),
                default=0.0,
            )
            family_pressures = diagnostics_summary.get("structural_family_pressures") or {}
            topology_pressure = finite_float(
                family_pressures.get("topology_directed") if isinstance(family_pressures, dict) else None,
                default=0.0,
            )
            transfer_efficiency = diagnostic_float(
                advanced_diagnostics,
                "diagnostics/latest/transfer_efficiency_recent",
                "fineweb_curve/transfer_efficiency_recent",
            )
            validation_lag_pressure = diagnostic_float(
                advanced_diagnostics,
                "diagnostics/latest/validation_lag_pressure",
                "fineweb_curve/validation_lag_pressure",
                default=0.0,
            )
            required_val_velocity = diagnostic_float(
                advanced_diagnostics,
                "diagnostics/latest/required_val_velocity_to_gate_per_100_steps",
                "fineweb_curve/required_val_velocity_to_gate_per_100_steps",
            )
            recent_val_velocity = diagnostic_float(
                advanced_diagnostics,
                "diagnostics/latest/val_bpb_velocity_recent_per_100_steps",
                "fineweb_curve/val_bpb_velocity_recent_per_100_steps",
            )
            val_velocity_shortfall = diagnostic_float(
                advanced_diagnostics,
                "diagnostics/latest/val_velocity_shortfall_to_gate_per_100_steps",
                "fineweb_curve/val_velocity_shortfall_to_gate_per_100_steps",
                default=0.0,
            )
            velocity_shortfall_pressure = diagnostic_float(
                advanced_diagnostics,
                "diagnostics/latest/bpb_velocity_shortfall_pressure",
                "fineweb_curve/bpb_velocity_shortfall_pressure",
                default=0.0,
            )
            resume_step_aware_warmup_steps = max(
                base.muon_momentum_warmup_steps,
                int(gate_step) + 250,
                int(latest_val.step) + 750,
            )
            validation_transfer_relieved = (
                0.50 <= structural_score < 0.75
                and math.isfinite(validation_gap)
                and validation_gap <= 0.005
                and (
                    not math.isfinite(transfer_efficiency)
                    or transfer_efficiency >= 0.45
                    or validation_gap <= 0.0
                )
                and validation_lag_pressure <= 0.35
                and topology_pressure < 0.28
            )
            if validation_transfer_relieved:
                velocity_shortfall_active = (
                    velocity_shortfall_pressure >= 0.45
                    or val_velocity_shortfall >= 0.0025
                )
                toric_slepian_guarded = (
                    dominant_family == "toric_slepian"
                    and structural_score >= 0.65
                    and dominant_pressure >= 0.35
                )
                if toric_slepian_guarded and velocity_shortfall_active:
                    return replace(
                        base,
                        train_batch_tokens=max(base.train_batch_tokens, min(batch_cap, raised_batch)),
                        tied_embed_lr=round(max(0.034, min(0.03672, base.tied_embed_lr * 0.94)), 6),
                        matrix_lr=base.matrix_lr,
                        scalar_lr=base.scalar_lr,
                        muon_momentum_warmup_steps=resume_step_aware_warmup_steps,
                        bigram_bias=True,
                        bigram_bias_lr=round(max(0.016, min(0.025, base.bigram_bias_lr * 0.70)), 6),
                        bigram_bias_init_from_data=False,
                        policy="toric_slepian_guarded_transfer_probe",
                        advanced_metric_policy="dominant_toric_slepian_damped_transfer_bpb_velocity",
                        rationale=(
                            f"validation projects target at step {projected_target_step:.1f}, beyond gate {gate_step}",
                            "guarded structural recapture is dominated by toric/Slepian pressure: "
                            f"score={structural_score:.3f} band={structural_band} "
                            f"pressure={dominant_pressure:.3f}",
                            "hot tied-embedding and bigram probes have underperformed this family, so use the advanced diagnostics to switch branch families",
                            "damp lexical transition heat while preserving the larger batch and long Muon warmup for smoother validation transfer",
                            "keep GraphCG/Slepian/topology/toric/BGG/Koszul losses in sidecar transfer until the competition checkpoint is preserved",
                        ),
                    )
                if velocity_shortfall_active:
                    return replace(
                        base,
                        train_batch_tokens=max(base.train_batch_tokens, min(batch_cap, raised_batch)),
                        tied_embed_lr=round(max(base.tied_embed_lr, min(0.0392, base.tied_embed_lr * 1.12)), 6),
                        matrix_lr=base.matrix_lr,
                        scalar_lr=base.scalar_lr,
                        muon_momentum_warmup_steps=resume_step_aware_warmup_steps,
                        bigram_bias=True,
                        bigram_bias_lr=round(
                            max(base.bigram_bias_lr, min(0.035, base.bigram_bias_lr * 1.35)),
                            6,
                        ),
                        bigram_bias_init_from_data=False,
                        policy="curvature_aware_structural_relief_velocity_recapture",
                        advanced_metric_policy="gate_velocity_shortfall_structural_relief_bpb_velocity",
                        rationale=(
                            f"validation projects target at step {projected_target_step:.1f}, beyond gate {gate_step}",
                            "structural recapture pressure has eased enough for guarded velocity: "
                            f"score={structural_score:.3f} band={structural_band}",
                            "dominant advanced-metric family is "
                            f"{dominant_family} pressure={dominant_pressure:.3f}",
                            "validation velocity shortfall is active: "
                            f"recent={recent_val_velocity:.4f}/100 required={required_val_velocity:.4f}/100 "
                            f"shortfall={val_velocity_shortfall:.4f}/100 pressure={velocity_shortfall_pressure:.3f}",
                            f"validation BPB {latest_val.val_bpb:.4f} is no longer lagging train BPB {train_bpb:.4f} "
                            f"(gap={validation_gap:.4f})",
                            "use a stronger tied-embedding and bigram-LR nudge while keeping matrix/scalar LR fixed",
                            "keep GraphCG/Slepian/topology/toric/BGG/Koszul losses in sidecar transfer until the competition checkpoint is preserved",
                        ),
                    )
                return replace(
                    base,
                    train_batch_tokens=max(base.train_batch_tokens, min(batch_cap, raised_batch)),
                    tied_embed_lr=round(max(base.tied_embed_lr, min(0.038, base.tied_embed_lr * 1.08)), 6),
                    matrix_lr=base.matrix_lr,
                    scalar_lr=base.scalar_lr,
                    muon_momentum_warmup_steps=resume_step_aware_warmup_steps,
                    bigram_bias=True,
                    bigram_bias_lr=round(
                        max(base.bigram_bias_lr, min(0.03, base.bigram_bias_lr * 1.25)),
                        6,
                    ),
                    bigram_bias_init_from_data=False,
                    policy="structural_relief_velocity_recapture",
                    advanced_metric_policy="guarded_transfer_relief_bpb_velocity",
                    rationale=(
                        f"validation projects target at step {projected_target_step:.1f}, beyond gate {gate_step}",
                        "structural recapture pressure has eased into the guarded band: "
                        f"score={structural_score:.3f} band={structural_band}",
                        "dominant advanced-metric family is "
                        f"{dominant_family} pressure={dominant_pressure:.3f}",
                        f"validation BPB {latest_val.val_bpb:.4f} is no longer lagging train BPB {train_bpb:.4f} "
                        f"(gap={validation_gap:.4f})",
                        "use a small tied-embedding and bigram-LR velocity push while keeping matrix/scalar LR fixed",
                        "keep GraphCG/Slepian/topology/toric/BGG/Koszul losses in sidecar transfer until the competition checkpoint is preserved",
                    ),
                )
            return replace(
                base,
                train_batch_tokens=max(base.train_batch_tokens, min(batch_cap, raised_batch)),
                tied_embed_lr=round(max(0.034, base.tied_embed_lr * 0.925), 6),
                matrix_lr=round(max(0.018, base.matrix_lr * 0.95), 6),
                scalar_lr=round(max(0.018, base.scalar_lr * 0.95), 6),
                muon_momentum_warmup_steps=resume_step_aware_warmup_steps,
                bigram_bias=True,
                bigram_bias_lr=round(max(0.012, base.bigram_bias_lr * 0.50), 6),
                bigram_bias_init_from_data=False,
                policy="structural_pressure_recapture",
                advanced_metric_policy="toric_topology_slepian_guarded_bpb_recapture",
                rationale=(
                    f"validation projects target at step {projected_target_step:.1f}, beyond gate {gate_step}",
                    "bigram transition bias is already enabled, so stop escalating the tied embedding LR cap",
                    "advanced diagnostics show structural pressure: "
                    f"topology={diagnostics_summary.get('topology_loss')}, "
                    f"directed_topology={diagnostics_summary.get('directed_topology_loss')}, "
                    f"slepian_leakage={diagnostics_summary.get('slepian_leakage')}, "
                    f"toric_margin={diagnostics_summary.get('toric_active_face_margin')}",
                    f"structural recapture score={structural_score:.3f} band={structural_band}",
                    "dominant advanced-metric family is "
                    f"{dominant_family} pressure={dominant_pressure:.3f}",
                    "use resume-step-aware Muon warmup so structural recapture still changes the optimizer flow after a 3K checkpoint resume",
                    "damp tied-embedding and bigram LR to improve validation transfer without injecting heavy structural losses",
                ),
            )
        if base.bigram_bias and base.tied_embed_lr >= 0.037:
            return replace(
                base,
                train_batch_tokens=max(base.train_batch_tokens, min(batch_cap, raised_batch)),
                tied_embed_lr=round(max(0.034, base.tied_embed_lr * 0.92), 6),
                matrix_lr=base.matrix_lr,
                scalar_lr=base.scalar_lr,
                muon_momentum_warmup_steps=max(
                    base.muon_momentum_warmup_steps,
                    int(gate_step) + 500,
                    int(latest_val.step) + 1000,
                ),
                bigram_bias=True,
                bigram_bias_lr=round(max(0.012, base.bigram_bias_lr * 0.65), 6),
                bigram_bias_init_from_data=False,
                policy="post_hot_probe_damped_transfer",
                advanced_metric_policy="hot_probe_failed_validation_damping_structural_sidecars",
                rationale=(
                    f"validation projects target at step {projected_target_step:.1f}, beyond gate {gate_step}",
                    f"hot tied-embedding probe lr={base.tied_embed_lr:.6f} did not produce enough validation velocity",
                    "do not escalate toward the known too-hot LR band",
                    "damp tied-embedding and bigram LR while holding matrix/scalar LR fixed",
                    "use the next run as a transfer-stability probe before switching to the native GraphCG/Slepian reasoning-memory trainer",
                ),
            )
        return replace(
            base,
            train_batch_tokens=max(base.train_batch_tokens, min(batch_cap, raised_batch)),
            tied_embed_lr=round(min(0.040, base.tied_embed_lr * 1.05), 6),
            policy="bpb_velocity_recapture",
            advanced_metric_policy="proposal_guided_bpb_recapture_structural_sidecars",
            rationale=(
                f"validation projects target at step {projected_target_step:.1f}, beyond gate {gate_step}",
                "increase effective batch and tied-embedding LR to accelerate validation BPB",
                "hold matrix/scalar LR to avoid disrupting the current basin",
                "use GraphCG/Slepian/topology diagnostics as sidecar transfer signals, not heavy primary losses",
            ),
        )

    return replace(
        base,
        policy="base_controls",
        advanced_metric_policy="monitor_structural_sidecars",
        rationale=("no metric-driven recovery-control change selected",),
    )


def shell_env(env: dict[str, Any]) -> str:
    return " ".join(f"{key}={shlex.quote(str(value))}" for key, value in env.items())


def build_recovery_launch(
    *,
    repo_root: Path,
    parameter_golf_root: Path,
    run_id: str,
    checkpoint_dir: Path,
    log_path: Path,
    resume_checkpoint: Path,
    seed: int,
    target_bpb: float,
    python: str = "/home/iska/miniconda3/envs/tokengt/bin/python",
    iterations: int = 4_000,
    train_seq_len: int = 4096,
    train_batch_tokens: int = 393_216,
    tied_embed_lr: float = 0.032,
    matrix_lr: float = 0.018,
    scalar_lr: float = 0.018,
    muon_momentum: float = 0.985,
    muon_momentum_warmup_steps: int = 600,
    muon_momentum_warmup_start: float = 0.90,
    grad_clip_norm: float | None = None,
    warmdown_iters: int = 1000,
    val_loss_every: int = 50,
    train_log_every: int = 50,
    checkpoint_every: int = 50,
    wandb_project: str = "toricgt-parameter-golf",
    wandb_entity: str = "amelie-iska-math",
    bigram_bias: bool = False,
    bigram_bias_lr: float = 0.05,
    bigram_bias_init_from_data: bool = False,
    bigram_bias_init_tokens: int = 100_000_000,
    bigram_bias_init_alpha: float = 0.1,
    bigram_bias_init_strength: float = 0.35,
    bigram_bias_scale: float = 1.0,
    advanced_loss_scale: float = 0.0,
    graphcg_loss_weight: float = 0.0,
    toric_tropical_loss_weight: float = 0.0,
    slepian_loss_weight: float = 0.0,
    koszul_bgg_loss_weight: float = 0.0,
    analogy_loss_weight: float = 0.0,
    advanced_loss_sample_tokens: int = 256,
    toric_tropical_fan_bins: int = 8,
    advanced_loss_log_only: bool = False,
    advanced_loss_start_step: int = 0,
    advanced_loss_end_step: int = 0,
    advanced_loss_every: int = 1,
    advanced_loss_warmup_steps: int = 0,
    advanced_loss_min_best_val_bpb: float = 0.0,
    advanced_loss_max_ce_ratio: float = 0.0,
    reset_optimizer_on_resume: bool = True,
    reset_rng_on_resume: bool = True,
    reset_loader_on_resume: bool = True,
    checkpoint_on_train_bpb_below: float = 1.13,
    checkpoint_on_train_bpb_cooldown_steps: int = 10,
    checkpoint_on_train_bpb_max: int = 6,
    val_on_train_bpb_checkpoint: bool = True,
) -> RecoveryLaunch:
    _ = repo_root
    env = {
        "RUN_ID": run_id,
        "WANDB_RUN_ID": run_id,
        "WANDB_RUN_NAME": run_id,
        "WANDB_PROJECT": wandb_project,
        "WANDB_ENTITY": wandb_entity,
        "WANDB_RESUME": "allow",
        "TARGET_BPB": target_bpb,
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": 1024,
        "TRAIN_SEQ_LEN": train_seq_len,
        "TRAIN_BATCH_TOKENS": train_batch_tokens,
        "TIED_EMBED_LR": tied_embed_lr,
        "MATRIX_LR": matrix_lr,
        "SCALAR_LR": scalar_lr,
        "MUON_MOMENTUM": muon_momentum,
        "MUON_MOMENTUM_WARMUP_STEPS": muon_momentum_warmup_steps,
        "MUON_MOMENTUM_WARMUP_START": muon_momentum_warmup_start,
        "WARMDOWN_ITERS": warmdown_iters,
        "MAX_WALLCLOCK_SECONDS": 0,
        "VAL_LOSS_EVERY": val_loss_every,
        "TRAIN_LOG_EVERY": train_log_every,
        "ITERATIONS": iterations,
        "CHECKPOINT_EVERY": checkpoint_every,
        "CHECKPOINT_DIR": checkpoint_dir,
        "SEED": seed,
        "WANDB_LOG_EVERY": 1,
        "RESUME_CHECKPOINT": resume_checkpoint,
        "RESET_OPTIMIZER_ON_RESUME": 1 if reset_optimizer_on_resume else 0,
        "RESET_RNG_ON_RESUME": 1 if reset_rng_on_resume else 0,
        "RESET_LOADER_ON_RESUME": 1 if reset_loader_on_resume else 0,
        "CHECKPOINT_ON_TRAIN_BPB_BELOW": checkpoint_on_train_bpb_below,
        "CHECKPOINT_ON_TRAIN_BPB_COOLDOWN_STEPS": checkpoint_on_train_bpb_cooldown_steps,
        "CHECKPOINT_ON_TRAIN_BPB_MAX": checkpoint_on_train_bpb_max,
        "VAL_ON_TRAIN_BPB_CHECKPOINT": 1 if val_on_train_bpb_checkpoint else 0,
        "ADVANCED_LOSS_SCALE": advanced_loss_scale,
        "GRAPHCG_LOSS_WEIGHT": graphcg_loss_weight,
        "TORIC_TROPICAL_LOSS_WEIGHT": toric_tropical_loss_weight,
        "SLEPIAN_LOSS_WEIGHT": slepian_loss_weight,
        "KOSZUL_BGG_LOSS_WEIGHT": koszul_bgg_loss_weight,
        "ANALOGY_LOSS_WEIGHT": analogy_loss_weight,
        "ADVANCED_LOSS_SAMPLE_TOKENS": advanced_loss_sample_tokens,
        "TORIC_TROPICAL_FAN_BINS": toric_tropical_fan_bins,
        "ADVANCED_LOSS_LOG_ONLY": 1 if advanced_loss_log_only else 0,
        "ADVANCED_LOSS_START_STEP": advanced_loss_start_step,
        "ADVANCED_LOSS_END_STEP": advanced_loss_end_step,
        "ADVANCED_LOSS_EVERY": advanced_loss_every,
        "ADVANCED_LOSS_WARMUP_STEPS": advanced_loss_warmup_steps,
        "ADVANCED_LOSS_MIN_BEST_VAL_BPB": advanced_loss_min_best_val_bpb,
        "ADVANCED_LOSS_MAX_CE_RATIO": advanced_loss_max_ce_ratio,
    }
    if grad_clip_norm is not None:
        env["GRAD_CLIP_NORM"] = grad_clip_norm
    if bigram_bias:
        env.update(
            {
                "BIGRAM_BIAS": 1,
                "BIGRAM_BIAS_LR": bigram_bias_lr,
                "BIGRAM_BIAS_SCALE": bigram_bias_scale,
                "BIGRAM_BIAS_INIT_FROM_DATA": 1 if bigram_bias_init_from_data else 0,
                "BIGRAM_BIAS_INIT_TOKENS": int(bigram_bias_init_tokens),
                "BIGRAM_BIAS_INIT_ALPHA": bigram_bias_init_alpha,
                "BIGRAM_BIAS_INIT_STRENGTH": bigram_bias_init_strength,
            }
        )
    training_command = [f"{key}={value}" for key, value in env.items()] + [
        python,
        "-u",
        "records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py",
    ]
    shell = (
        f"mkdir -p {shlex.quote(str(log_path.parent))} {shlex.quote(str(checkpoint_dir))} && "
        f"cd {shlex.quote(str(parameter_golf_root))} && "
        f"export {shell_env(env)} && "
        f"{shlex.quote(str(python))} -u records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py "
        f"2>&1 | tee -a {shlex.quote(str(log_path))}"
    )
    return RecoveryLaunch(training_command=training_command, training_shell=shell)


def run(command: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def tmux_has(session: str) -> bool:
    return run(["tmux", "has-session", "-t", session]).returncode == 0


def tmux_kill(session: str) -> None:
    if tmux_has(session):
        run(["tmux", "kill-session", "-t", session])


def tmux_start(session: str, command: str, dry_run: bool) -> None:
    if dry_run:
        print(f"dry_run_tmux_start {session}: {command}", flush=True)
        return
    run(["tmux", "new-session", "-d", "-s", session, command], check=True)


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_analysis_shell(
    *,
    repo_root: Path,
    run_id: str,
    train_tmux: str,
    checkpoint_dir: Path,
    log_path: Path,
    target_bpb: float,
    start_step: int,
    python: str,
    wandb_entity: str,
    wandb_project: str,
) -> str:
    analysis_log = repo_root / "logs" / f"{run_id}.analysis_watcher.txt"
    output_root = repo_root / "outputs" / "post_resume_analysis" / run_id
    codex_review_hook = repo_root / "scripts" / "codex_training_review_resume.sh"
    codex_review_args = ""
    if codex_review_hook.exists():
        codex_review_args = (
            f"--codex-review-hook {shlex.quote(str(codex_review_hook))} "
            f"--codex-review-tmux-prefix {shlex.quote(f'toricgt_codex_review_{run_id}')} "
        )
    default_loop_state = repo_root / "outputs" / f"{run_id}_bpb_codex_loop_state.json"
    default_loop_stop_file = repo_root / "outputs" / f"{run_id}_bpb_codex_loop_stop"
    env = {
        "PYTHONPATH": "src",
        "WANDB_PROJECT": wandb_project,
        "WANDB_ENTITY": wandb_entity,
        "BPB_TARGET": os.environ.get("BPB_TARGET", str(target_bpb)),
        "BPB_MAX_REVIEW_ITERATIONS": os.environ.get("BPB_MAX_REVIEW_ITERATIONS", "100"),
        "BPB_LOOP_STATE": os.environ.get("BPB_LOOP_STATE", str(default_loop_state)),
        "BPB_LOOP_STOP_FILE": os.environ.get("BPB_LOOP_STOP_FILE", str(default_loop_stop_file)),
        "BPB_LOOP_NAME": os.environ.get("BPB_LOOP_NAME", "parameter_golf_bpb_target"),
    }
    command = (
        f"cd {shlex.quote(str(repo_root))} && export {shell_env(env)} && "
        f"{shlex.quote(str(python))} scripts/watch_seq4096_analysis.py "
        f"--checkpoint-dir {shlex.quote(str(checkpoint_dir))} "
        f"--log {shlex.quote(str(log_path))} "
        f"--run-path {shlex.quote(f'{wandb_entity}/{wandb_project}/{run_id}')} "
        f"--output-root {shlex.quote(str(output_root))} "
        f"--start-step {int(start_step)} --interval-steps 1 --poll-seconds 30 "
        f"--analyze-start-step "
        f"--target-bpb {float(target_bpb)} --training-tmux {shlex.quote(train_tmux)} "
        f"{codex_review_args}"
        f"2>&1 | tee -a {shlex.quote(str(analysis_log))}"
    )
    return command


def build_dense_mirror_shell(*, repo_root: Path, run_id: str, log_path: Path, target_bpb: float, python: str) -> str:
    mirror_log = repo_root / "logs" / f"{run_id}.wandb_mirror.txt"
    diagnostics_json = repo_root / "logs" / f"{run_id}.full_diag.latest.json"
    gate_state_json = repo_root / "outputs" / f"{run_id}.4k_recovery_state.json"
    return (
        f"cd {shlex.quote(str(repo_root))} && export PYTHONPATH=src && "
        f"{shlex.quote(str(python))} scripts/mirror_fineweb_log_to_wandb.py "
        f"--log {shlex.quote(str(log_path))} --run-id {shlex.quote(run_id)} "
        f"--target-bpb {float(target_bpb)} --poll-seconds 15 "
        f"--diagnostics-json {shlex.quote(str(diagnostics_json))} "
        f"--gate-state-json {shlex.quote(str(gate_state_json))} "
        f"2>&1 | tee -a {shlex.quote(str(mirror_log))}"
    )


def build_full_diag_shell(*, repo_root: Path, run_id: str, log_path: Path, target_bpb: float, python: str) -> str:
    diag_log = repo_root / "logs" / f"{run_id}.full_diag.txt"
    diagnostics_json = repo_root / "logs" / f"{run_id}.full_diag.latest.json"
    return (
        f"cd {shlex.quote(str(repo_root))} && export PYTHONPATH=src && "
        f"{shlex.quote(str(python))} scripts/mirror_fineweb_full_diagnostics_to_wandb.py "
        f"--log {shlex.quote(str(log_path))} --run-id {shlex.quote(run_id)} "
        f"--target-bpb {float(target_bpb)} --poll-seconds 60 "
        f"--output-json {shlex.quote(str(diagnostics_json))} "
        f"2>&1 | tee -a {shlex.quote(str(diag_log))}"
    )


def build_gate_shell(
    *,
    repo_root: Path,
    run_id: str,
    train_tmux: str,
    checkpoint_dir: Path,
    log_path: Path,
    target_bpb: float,
    gate_step: int,
    restart_index: int,
    max_restarts: int,
    min_recovery_runway_steps: int,
    recovery_train_batch_tokens: int,
    recovery_tied_embed_lr: float,
    recovery_matrix_lr: float,
    recovery_scalar_lr: float,
    recovery_muon_momentum: float,
    recovery_muon_momentum_warmup_steps: int,
    recovery_muon_momentum_warmup_start: float,
    recovery_grad_clip_norm: float | None,
    recovery_bigram_bias: bool,
    recovery_bigram_bias_lr: float,
    recovery_bigram_bias_init_from_data: bool,
    recovery_bigram_bias_init_tokens: int,
    recovery_bigram_bias_init_alpha: float,
    recovery_bigram_bias_init_strength: float,
    recovery_bigram_bias_scale: float,
    recovery_advanced_loss_scale: float,
    recovery_graphcg_loss_weight: float,
    recovery_toric_tropical_loss_weight: float,
    recovery_slepian_loss_weight: float,
    recovery_koszul_bgg_loss_weight: float,
    recovery_analogy_loss_weight: float,
    recovery_advanced_loss_sample_tokens: int,
    recovery_toric_tropical_fan_bins: int,
    recovery_advanced_loss_log_only: bool,
    recovery_advanced_loss_start_step: int,
    recovery_advanced_loss_end_step: int,
    recovery_advanced_loss_every: int,
    recovery_advanced_loss_warmup_steps: int,
    recovery_advanced_loss_min_best_val_bpb: float,
    recovery_advanced_loss_max_ce_ratio: float,
    recovery_reset_optimizer: bool,
    recovery_reset_rng: bool,
    recovery_reset_loader: bool,
    recovery_checkpoint_on_train_bpb_below: float,
    recovery_checkpoint_on_train_bpb_cooldown_steps: int,
    recovery_checkpoint_on_train_bpb_max: int,
    recovery_val_on_train_bpb_checkpoint: bool,
    preempt_on_projected_miss: bool,
    preempt_min_step: int,
    preempt_patience: int,
    advanced_metric_controls: bool,
    recovery_max_train_batch_tokens: int,
    validation_gap_threshold: float,
    low_train_bpb_margin: float,
    python: str,
) -> str:
    gate_log = repo_root / "logs" / f"{run_id}.4k_gate.txt"
    state_path = repo_root / "outputs" / f"{run_id}.4k_recovery_state.json"
    default_loop_state = repo_root / "outputs" / f"{run_id}_bpb_codex_loop_state.json"
    default_loop_stop_file = repo_root / "outputs" / f"{run_id}_bpb_codex_loop_stop"
    loop_env = {
        "BPB_TARGET": os.environ.get("BPB_TARGET", str(target_bpb)),
        "BPB_MAX_REVIEW_ITERATIONS": os.environ.get("BPB_MAX_REVIEW_ITERATIONS", "100"),
        "BPB_LOOP_STATE": os.environ.get("BPB_LOOP_STATE", str(default_loop_state)),
        "BPB_LOOP_STOP_FILE": os.environ.get("BPB_LOOP_STOP_FILE", str(default_loop_stop_file)),
        "BPB_LOOP_NAME": os.environ.get("BPB_LOOP_NAME", "parameter_golf_bpb_target"),
    }
    grad_clip_arg = (
        ""
        if recovery_grad_clip_norm is None
        else f"--recovery-grad-clip-norm {float(recovery_grad_clip_norm)} "
    )
    preempt_arg = (
        f"--preempt-on-projected-miss --preempt-min-step {int(preempt_min_step)} "
        f"--preempt-patience {int(preempt_patience)} "
        if preempt_on_projected_miss
        else ""
    )
    advanced_metric_arg = (
        ""
        if advanced_metric_controls
        else "--no-advanced-metric-controls "
    )
    bigram_arg = ""
    if recovery_bigram_bias:
        bigram_arg = (
            f"--recovery-bigram-bias "
            f"--recovery-bigram-bias-lr {float(recovery_bigram_bias_lr)} "
            f"--recovery-bigram-bias-init-tokens {int(recovery_bigram_bias_init_tokens)} "
            f"--recovery-bigram-bias-init-alpha {float(recovery_bigram_bias_init_alpha)} "
            f"--recovery-bigram-bias-init-strength {float(recovery_bigram_bias_init_strength)} "
            f"--recovery-bigram-bias-scale {float(recovery_bigram_bias_scale)} "
        )
        if recovery_bigram_bias_init_from_data:
            bigram_arg += "--recovery-bigram-bias-init-from-data "
    advanced_loss_arg = (
        f"--recovery-advanced-loss-scale {float(recovery_advanced_loss_scale)} "
        f"--recovery-graphcg-loss-weight {float(recovery_graphcg_loss_weight)} "
        f"--recovery-toric-tropical-loss-weight {float(recovery_toric_tropical_loss_weight)} "
        f"--recovery-slepian-loss-weight {float(recovery_slepian_loss_weight)} "
        f"--recovery-koszul-bgg-loss-weight {float(recovery_koszul_bgg_loss_weight)} "
        f"--recovery-analogy-loss-weight {float(recovery_analogy_loss_weight)} "
        f"--recovery-advanced-loss-sample-tokens {int(recovery_advanced_loss_sample_tokens)} "
        f"--recovery-toric-tropical-fan-bins {int(recovery_toric_tropical_fan_bins)} "
        f"--recovery-advanced-loss-start-step {int(recovery_advanced_loss_start_step)} "
        f"--recovery-advanced-loss-end-step {int(recovery_advanced_loss_end_step)} "
        f"--recovery-advanced-loss-every {int(recovery_advanced_loss_every)} "
        f"--recovery-advanced-loss-warmup-steps {int(recovery_advanced_loss_warmup_steps)} "
        f"--recovery-advanced-loss-min-best-val-bpb {float(recovery_advanced_loss_min_best_val_bpb)} "
        f"--recovery-advanced-loss-max-ce-ratio {float(recovery_advanced_loss_max_ce_ratio)} "
    )
    if recovery_advanced_loss_log_only:
        advanced_loss_arg += "--recovery-advanced-loss-log-only "
    reset_arg = ""
    if not recovery_reset_optimizer:
        reset_arg += "--no-recovery-reset-optimizer "
    if not recovery_reset_rng:
        reset_arg += "--no-recovery-reset-rng "
    if not recovery_reset_loader:
        reset_arg += "--no-recovery-reset-loader "
    low_bpb_trigger_arg = (
        f"--recovery-checkpoint-on-train-bpb-below {float(recovery_checkpoint_on_train_bpb_below)} "
        f"--recovery-checkpoint-on-train-bpb-cooldown-steps {int(recovery_checkpoint_on_train_bpb_cooldown_steps)} "
        f"--recovery-checkpoint-on-train-bpb-max {int(recovery_checkpoint_on_train_bpb_max)} "
    )
    if not recovery_val_on_train_bpb_checkpoint:
        low_bpb_trigger_arg += "--no-recovery-val-on-train-bpb-checkpoint "
    return (
        f"cd {shlex.quote(str(repo_root))} && export PYTHONPATH=src {shell_env(loop_env)} && "
        f"{shlex.quote(str(python))} scripts/watch_seq4096_4k_recovery.py "
        f"--log {shlex.quote(str(log_path))} --checkpoint-dir {shlex.quote(str(checkpoint_dir))} "
        f"--run-id {shlex.quote(run_id)} --train-tmux {shlex.quote(train_tmux)} "
        f"--target-bpb {float(target_bpb)} --gate-step {int(gate_step)} "
        f"--restart-index {int(restart_index)} --max-restarts {int(max_restarts)} "
        f"--min-recovery-runway-steps {int(min_recovery_runway_steps)} "
        f"--recovery-train-batch-tokens {int(recovery_train_batch_tokens)} "
        f"--recovery-tied-embed-lr {float(recovery_tied_embed_lr)} "
        f"--recovery-matrix-lr {float(recovery_matrix_lr)} "
        f"--recovery-scalar-lr {float(recovery_scalar_lr)} "
        f"--recovery-muon-momentum {float(recovery_muon_momentum)} "
        f"--recovery-muon-momentum-warmup-steps {int(recovery_muon_momentum_warmup_steps)} "
        f"--recovery-muon-momentum-warmup-start {float(recovery_muon_momentum_warmup_start)} "
        f"{grad_clip_arg}"
        f"{bigram_arg}"
        f"{advanced_loss_arg}"
        f"{reset_arg}"
        f"{low_bpb_trigger_arg}"
        f"{preempt_arg}"
        f"{advanced_metric_arg}"
        f"--recovery-max-train-batch-tokens {int(recovery_max_train_batch_tokens)} "
        f"--validation-gap-threshold {float(validation_gap_threshold)} "
        f"--low-train-bpb-margin {float(low_train_bpb_margin)} "
        f"--state {shlex.quote(str(state_path))} "
        f"2>&1 | tee -a {shlex.quote(str(gate_log))}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-tmux", required=True)
    parser.add_argument("--target-bpb", type=float, default=1.2)
    parser.add_argument("--gate-step", type=int, default=4000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-restarts", type=int, default=6)
    parser.add_argument("--restart-index", type=int, default=0)
    parser.add_argument("--min-recovery-runway-steps", type=int, default=500)
    parser.add_argument("--state", default="")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--parameter-golf-root", default="amelie-iska/parameter-golf")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "toricgt-parameter-golf"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "amelie-iska-math"))
    parser.add_argument("--seed", type=int, default=7331)
    parser.add_argument("--recovery-train-batch-tokens", type=int, default=393_216)
    parser.add_argument("--recovery-tied-embed-lr", type=float, default=0.032)
    parser.add_argument("--recovery-matrix-lr", type=float, default=0.018)
    parser.add_argument("--recovery-scalar-lr", type=float, default=0.018)
    parser.add_argument("--recovery-muon-momentum", type=float, default=0.985)
    parser.add_argument("--recovery-muon-momentum-warmup-steps", type=int, default=600)
    parser.add_argument("--recovery-muon-momentum-warmup-start", type=float, default=0.90)
    parser.add_argument("--recovery-grad-clip-norm", type=float, default=None)
    parser.add_argument("--recovery-bigram-bias", action="store_true")
    parser.add_argument("--recovery-bigram-bias-lr", type=float, default=0.05)
    parser.add_argument("--recovery-bigram-bias-init-from-data", action="store_true")
    parser.add_argument("--recovery-bigram-bias-init-tokens", type=int, default=100_000_000)
    parser.add_argument("--recovery-bigram-bias-init-alpha", type=float, default=0.1)
    parser.add_argument("--recovery-bigram-bias-init-strength", type=float, default=0.35)
    parser.add_argument("--recovery-bigram-bias-scale", type=float, default=1.0)
    parser.add_argument("--recovery-advanced-loss-scale", type=float, default=0.0)
    parser.add_argument("--recovery-graphcg-loss-weight", type=float, default=0.0)
    parser.add_argument("--recovery-toric-tropical-loss-weight", type=float, default=0.0)
    parser.add_argument("--recovery-slepian-loss-weight", type=float, default=0.0)
    parser.add_argument("--recovery-koszul-bgg-loss-weight", type=float, default=0.0)
    parser.add_argument("--recovery-analogy-loss-weight", type=float, default=0.0)
    parser.add_argument("--recovery-advanced-loss-sample-tokens", type=int, default=256)
    parser.add_argument("--recovery-toric-tropical-fan-bins", type=int, default=8)
    parser.add_argument("--recovery-advanced-loss-log-only", action="store_true")
    parser.add_argument("--recovery-advanced-loss-start-step", type=int, default=0)
    parser.add_argument("--recovery-advanced-loss-end-step", type=int, default=0)
    parser.add_argument("--recovery-advanced-loss-every", type=int, default=1)
    parser.add_argument("--recovery-advanced-loss-warmup-steps", type=int, default=0)
    parser.add_argument("--recovery-advanced-loss-min-best-val-bpb", type=float, default=0.0)
    parser.add_argument("--recovery-advanced-loss-max-ce-ratio", type=float, default=0.0)
    parser.add_argument("--recovery-reset-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recovery-reset-rng", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recovery-reset-loader", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recovery-checkpoint-on-train-bpb-below", type=float, default=1.13)
    parser.add_argument("--recovery-checkpoint-on-train-bpb-cooldown-steps", type=int, default=10)
    parser.add_argument("--recovery-checkpoint-on-train-bpb-max", type=int, default=6)
    parser.add_argument(
        "--recovery-val-on-train-bpb-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--analysis-root", default="")
    parser.add_argument("--preempt-on-projected-miss", action="store_true")
    parser.add_argument("--preempt-min-step", type=int, default=2500)
    parser.add_argument("--preempt-patience", type=int, default=2)
    parser.add_argument("--no-analogue-risk", action="store_true")
    parser.add_argument("--analogue-risk-min-step", type=int, default=3250)
    parser.add_argument("--analogue-risk-train-rmse-threshold", type=float, default=0.001)
    parser.add_argument("--analogue-risk-min-failed", type=int, default=2)
    parser.add_argument("--analogue-risk-gate-margin-steps", type=float, default=100.0)
    parser.add_argument("--no-advanced-metric-controls", action="store_true")
    parser.add_argument("--recovery-max-train-batch-tokens", type=int, default=983_040)
    parser.add_argument("--validation-gap-threshold", type=float, default=0.04)
    parser.add_argument("--low-train-bpb-margin", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configured_max_restarts = int(args.max_restarts)
    max_restarts = effective_restart_ceiling(args.restart_index, args.max_restarts)
    repo_root = Path(args.repo_root).resolve()
    parameter_golf_root = Path(args.parameter_golf_root)
    if not parameter_golf_root.is_absolute():
        parameter_golf_root = repo_root / parameter_golf_root
    log_path = Path(args.log)
    checkpoint_dir = Path(args.checkpoint_dir)
    state_path = Path(args.state) if args.state else repo_root / "outputs" / f"{args.run_id}.4k_recovery_state.json"
    last_status = ""

    while True:
        parsed = parse_seq4096_log(log_path)
        latest_val = parsed.val_rows[-1] if parsed.val_rows else None
        best_val = select_best_validation(parsed.val_rows, args.gate_step)
        target_reached = bool(best_val and best_val.val_bpb <= args.target_bpb)
        analysis_root = (
            Path(args.analysis_root)
            if args.analysis_root
            else repo_root / "outputs" / "post_resume_analysis" / args.run_id
        )
        advanced_diagnostics = load_advanced_diagnostics(repo_root, args.run_id)
        advanced_diagnostics_summary = summarize_advanced_diagnostics(advanced_diagnostics)
        projection_gate_risk = (
            load_preemptive_gate_risk(
                analysis_root,
                gate_step=args.gate_step,
                target_bpb=args.target_bpb,
                min_step=args.preempt_min_step,
                patience=args.preempt_patience,
            )
            if args.preempt_on_projected_miss
            else None
        )
        analogue_gate_risk = (
            load_failed_trajectory_analogue_risk(
                log_path,
                history_logs=sorted(log_path.parent.glob("toricgt_seq4096_4k_recovery_r*.txt")),
                gate_step=args.gate_step,
                target_bpb=args.target_bpb,
                min_step=args.analogue_risk_min_step,
                train_rmse_threshold=args.analogue_risk_train_rmse_threshold,
                min_failed_analogues=args.analogue_risk_min_failed,
                current_projection_gate_margin_steps=args.analogue_risk_gate_margin_steps,
            )
            if args.preempt_on_projected_miss and not args.no_analogue_risk
            else None
        )
        train_wave_gate_risk = (
            load_train_wave_analogue_risk(
                log_path,
                history_logs=sorted(log_path.parent.glob("toricgt_seq4096_4k_recovery_r*.txt")),
                gate_step=args.gate_step,
                target_bpb=args.target_bpb,
                min_step=max(args.analogue_risk_min_step, 3350),
                train_rmse_threshold=args.analogue_risk_train_rmse_threshold,
                min_failed_analogues=1,
            )
            if args.preempt_on_projected_miss and not args.no_analogue_risk
            else None
        )
        train_wave_validation_probe_hold = should_hold_train_wave_for_validation_probe(
            train_wave_gate_risk,
            tied_embed_lr=args.recovery_tied_embed_lr,
        )
        preemptive_risk = projection_gate_risk
        if not should_preempt_for_gate_risk(preemptive_risk) and should_preempt_for_gate_risk(analogue_gate_risk):
            preemptive_risk = analogue_gate_risk
        if (
            not train_wave_validation_probe_hold
            and not should_preempt_for_gate_risk(preemptive_risk)
            and should_preempt_for_gate_risk(train_wave_gate_risk)
        ):
            preemptive_risk = train_wave_gate_risk
        preempt_for_gate_risk = should_preempt_at_latest_validation(
            latest_val,
            preemptive_risk,
            gate_step=args.gate_step,
            min_step=args.preempt_min_step,
        )
        base_controls = RecoveryControls(
            train_batch_tokens=args.recovery_train_batch_tokens,
            tied_embed_lr=args.recovery_tied_embed_lr,
            matrix_lr=args.recovery_matrix_lr,
            scalar_lr=args.recovery_scalar_lr,
            muon_momentum=args.recovery_muon_momentum,
            muon_momentum_warmup_steps=args.recovery_muon_momentum_warmup_steps,
            muon_momentum_warmup_start=args.recovery_muon_momentum_warmup_start,
            grad_clip_norm=args.recovery_grad_clip_norm,
            bigram_bias=args.recovery_bigram_bias,
            bigram_bias_lr=args.recovery_bigram_bias_lr,
            bigram_bias_init_from_data=args.recovery_bigram_bias_init_from_data,
            bigram_bias_init_tokens=args.recovery_bigram_bias_init_tokens,
            bigram_bias_init_alpha=args.recovery_bigram_bias_init_alpha,
            bigram_bias_init_strength=args.recovery_bigram_bias_init_strength,
            bigram_bias_scale=args.recovery_bigram_bias_scale,
            advanced_loss_scale=args.recovery_advanced_loss_scale,
            graphcg_loss_weight=args.recovery_graphcg_loss_weight,
            toric_tropical_loss_weight=args.recovery_toric_tropical_loss_weight,
            slepian_loss_weight=args.recovery_slepian_loss_weight,
            koszul_bgg_loss_weight=args.recovery_koszul_bgg_loss_weight,
            analogy_loss_weight=args.recovery_analogy_loss_weight,
            advanced_loss_sample_tokens=args.recovery_advanced_loss_sample_tokens,
            toric_tropical_fan_bins=args.recovery_toric_tropical_fan_bins,
            advanced_loss_log_only=args.recovery_advanced_loss_log_only,
            advanced_loss_start_step=args.recovery_advanced_loss_start_step,
            advanced_loss_end_step=args.recovery_advanced_loss_end_step,
            advanced_loss_every=args.recovery_advanced_loss_every,
            advanced_loss_warmup_steps=args.recovery_advanced_loss_warmup_steps,
            advanced_loss_min_best_val_bpb=args.recovery_advanced_loss_min_best_val_bpb,
            advanced_loss_max_ce_ratio=args.recovery_advanced_loss_max_ce_ratio,
            reset_optimizer_on_resume=args.recovery_reset_optimizer,
            reset_rng_on_resume=args.recovery_reset_rng,
            reset_loader_on_resume=args.recovery_reset_loader,
        )
        projected_target_step = (
            float("nan")
            if preemptive_risk is None
            else preemptive_risk.latest_projected_target_step
        )
        recovery_controls = plan_metric_driven_recovery_controls(
            base_controls,
            parsed=parsed,
            target_bpb=args.target_bpb,
            gate_step=args.gate_step,
            projected_target_step=projected_target_step,
            max_train_batch_tokens=args.recovery_max_train_batch_tokens,
            validation_gap_threshold=args.validation_gap_threshold,
            low_train_bpb_margin=args.low_train_bpb_margin,
            enabled=not args.no_advanced_metric_controls,
            advanced_diagnostics=advanced_diagnostics,
        )
        if (
            preemptive_risk is not None
            and preemptive_risk.risk_source == "failed_train_wave_analogue"
            and should_preempt_for_gate_risk(preemptive_risk)
            and not train_wave_validation_probe_hold
        ):
            recovery_controls = plan_failed_train_wave_recovery_controls(
                base_controls,
                parsed=parsed,
                risk=preemptive_risk,
                gate_step=args.gate_step,
            )
        status: dict[str, Any] = {
            "run_id": args.run_id,
            "log": str(log_path),
            "checkpoint_dir": str(checkpoint_dir),
            "gate_step": args.gate_step,
            "target_bpb": args.target_bpb,
            "latest_step": parsed.latest_step,
            "latest_validation": None if latest_val is None else latest_val.__dict__,
            "best_validation_at_or_before_gate": None if best_val is None else best_val.__dict__,
            "target_reached": target_reached,
            "restart_index": args.restart_index,
            "configured_max_restarts": configured_max_restarts,
            "effective_max_restarts": max_restarts,
            "min_recovery_runway_steps": args.min_recovery_runway_steps,
            "preempt_on_projected_miss": bool(args.preempt_on_projected_miss),
            "preemptive_gate_risk": None if preemptive_risk is None else preemptive_risk.__dict__,
            "projection_gate_risk": None if projection_gate_risk is None else projection_gate_risk.__dict__,
            "analogue_gate_risk": None if analogue_gate_risk is None else analogue_gate_risk.__dict__,
            "train_wave_gate_risk": None if train_wave_gate_risk is None else train_wave_gate_risk.__dict__,
            "train_wave_validation_probe_hold": bool(train_wave_validation_probe_hold),
            "analogue_risk_enabled": not args.no_analogue_risk,
            "base_recovery_launch_controls": base_controls.launch_dict(),
            "recovery_launch_controls": recovery_controls.launch_dict(),
            "advanced_metric_controls_enabled": not args.no_advanced_metric_controls,
            "advanced_diagnostics": advanced_diagnostics_summary,
            "recovery_max_train_batch_tokens": args.recovery_max_train_batch_tokens,
            "validation_gap_threshold": args.validation_gap_threshold,
            "low_train_bpb_margin": args.low_train_bpb_margin,
            "updated_unix": time.time(),
        }
        rendered = json.dumps(status, sort_keys=True)
        if rendered != last_status:
            write_state(state_path, status)
            if latest_val:
                best_text = "n/a" if best_val is None else f"{best_val.val_bpb:.4f}@{best_val.step}"
                print(
                    f"4k_gate_status run={args.run_id} latest={latest_val.val_bpb:.4f}@{latest_val.step} "
                    f"best={best_text} target={args.target_bpb:.4f}",
                    flush=True,
                )
            last_status = rendered

        if target_reached:
            status["event"] = "target_reached_before_or_at_gate"
            write_state(state_path, status)
            return

        if latest_val is None or (latest_val.step < args.gate_step and not preempt_for_gate_risk):
            time.sleep(max(1.0, args.poll_seconds))
            continue

        if args.restart_index >= max_restarts:
            status["event"] = "restart_limit_reached"
            write_state(state_path, status)
            print(json.dumps(status, sort_keys=True), flush=True)
            return

        if best_val is None:
            time.sleep(max(1.0, args.poll_seconds))
            continue

        recovery_val = select_recovery_validation(
            parsed.val_rows,
            args.gate_step,
            min_recovery_runway_steps=args.min_recovery_runway_steps,
        )
        if recovery_val is None:
            time.sleep(max(1.0, args.poll_seconds))
            continue

        resume_checkpoint = checkpoint_for_step(checkpoint_dir, args.run_id, recovery_val.step)
        if resume_checkpoint is None:
            status["event"] = "waiting_for_best_checkpoint"
            status["best_checkpoint_step"] = recovery_val.step
            write_state(state_path, status)
            time.sleep(max(1.0, args.poll_seconds))
            continue

        stamp = utc_stamp()
        next_index = args.restart_index + 1
        recovery_run_id = build_recovery_run_id(args.run_id, next_index, stamp)
        recovery_train_tmux = sanitize_tmux_name(f"toricgt_seq4096_4k_recovery_r{next_index}_{stamp}")
        recovery_checkpoint_dir = parameter_golf_root / "checkpoints" / recovery_run_id
        recovery_log = parameter_golf_root / "logs" / f"{recovery_run_id}.txt"
        recovery_eval_interval = recovery_observation_interval(recovery_controls)
        recovery_checkpoint_interval = recovery_eval_interval
        launch = build_recovery_launch(
            repo_root=repo_root,
            parameter_golf_root=parameter_golf_root,
            run_id=recovery_run_id,
            checkpoint_dir=recovery_checkpoint_dir,
            log_path=recovery_log,
            resume_checkpoint=resume_checkpoint,
            seed=args.seed + next_index,
            target_bpb=args.target_bpb,
            python=args.python,
            val_loss_every=recovery_eval_interval,
            checkpoint_every=recovery_checkpoint_interval,
            train_batch_tokens=recovery_controls.train_batch_tokens,
            tied_embed_lr=recovery_controls.tied_embed_lr,
            matrix_lr=recovery_controls.matrix_lr,
            scalar_lr=recovery_controls.scalar_lr,
            muon_momentum=recovery_controls.muon_momentum,
            muon_momentum_warmup_steps=recovery_controls.muon_momentum_warmup_steps,
            muon_momentum_warmup_start=recovery_controls.muon_momentum_warmup_start,
            grad_clip_norm=recovery_controls.grad_clip_norm,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            bigram_bias=recovery_controls.bigram_bias,
            bigram_bias_lr=recovery_controls.bigram_bias_lr,
            bigram_bias_init_from_data=recovery_controls.bigram_bias_init_from_data,
            bigram_bias_init_tokens=recovery_controls.bigram_bias_init_tokens,
            bigram_bias_init_alpha=recovery_controls.bigram_bias_init_alpha,
            bigram_bias_init_strength=recovery_controls.bigram_bias_init_strength,
            bigram_bias_scale=recovery_controls.bigram_bias_scale,
            advanced_loss_scale=recovery_controls.advanced_loss_scale,
            graphcg_loss_weight=recovery_controls.graphcg_loss_weight,
            toric_tropical_loss_weight=recovery_controls.toric_tropical_loss_weight,
            slepian_loss_weight=recovery_controls.slepian_loss_weight,
            koszul_bgg_loss_weight=recovery_controls.koszul_bgg_loss_weight,
            analogy_loss_weight=recovery_controls.analogy_loss_weight,
            advanced_loss_sample_tokens=recovery_controls.advanced_loss_sample_tokens,
            toric_tropical_fan_bins=recovery_controls.toric_tropical_fan_bins,
            advanced_loss_log_only=recovery_controls.advanced_loss_log_only,
            advanced_loss_start_step=recovery_controls.advanced_loss_start_step,
            advanced_loss_end_step=recovery_controls.advanced_loss_end_step,
            advanced_loss_every=recovery_controls.advanced_loss_every,
            advanced_loss_warmup_steps=recovery_controls.advanced_loss_warmup_steps,
            advanced_loss_min_best_val_bpb=recovery_controls.advanced_loss_min_best_val_bpb,
            advanced_loss_max_ce_ratio=recovery_controls.advanced_loss_max_ce_ratio,
            reset_optimizer_on_resume=recovery_controls.reset_optimizer_on_resume,
            reset_rng_on_resume=recovery_controls.reset_rng_on_resume,
            reset_loader_on_resume=recovery_controls.reset_loader_on_resume,
            checkpoint_on_train_bpb_below=args.recovery_checkpoint_on_train_bpb_below,
            checkpoint_on_train_bpb_cooldown_steps=args.recovery_checkpoint_on_train_bpb_cooldown_steps,
            checkpoint_on_train_bpb_max=args.recovery_checkpoint_on_train_bpb_max,
            val_on_train_bpb_checkpoint=args.recovery_val_on_train_bpb_checkpoint,
        )
        command_dir = repo_root / "logs" / recovery_run_id / "supervisor"
        command_dir.mkdir(parents=True, exist_ok=True)
        (command_dir / "train_command.sh").write_text(launch.training_shell + "\n", encoding="utf-8")

        analysis_shell = build_analysis_shell(
            repo_root=repo_root,
            run_id=recovery_run_id,
            train_tmux=recovery_train_tmux,
            checkpoint_dir=recovery_checkpoint_dir,
            log_path=recovery_log,
            target_bpb=args.target_bpb,
            start_step=recovery_val.step,
            python=args.python,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
        )
        dense_shell = build_dense_mirror_shell(
            repo_root=repo_root,
            run_id=recovery_run_id,
            log_path=recovery_log,
            target_bpb=args.target_bpb,
            python=args.python,
        )
        diag_shell = build_full_diag_shell(
            repo_root=repo_root,
            run_id=recovery_run_id,
            log_path=recovery_log,
            target_bpb=args.target_bpb,
            python=args.python,
        )
        gate_shell = build_gate_shell(
            repo_root=repo_root,
            run_id=recovery_run_id,
            train_tmux=recovery_train_tmux,
            checkpoint_dir=recovery_checkpoint_dir,
            log_path=recovery_log,
            target_bpb=args.target_bpb,
            gate_step=args.gate_step,
            restart_index=next_index,
            max_restarts=max_restarts,
            min_recovery_runway_steps=args.min_recovery_runway_steps,
            recovery_train_batch_tokens=recovery_controls.train_batch_tokens,
            recovery_tied_embed_lr=recovery_controls.tied_embed_lr,
            recovery_matrix_lr=recovery_controls.matrix_lr,
            recovery_scalar_lr=recovery_controls.scalar_lr,
            recovery_muon_momentum=recovery_controls.muon_momentum,
            recovery_muon_momentum_warmup_steps=recovery_controls.muon_momentum_warmup_steps,
            recovery_muon_momentum_warmup_start=recovery_controls.muon_momentum_warmup_start,
            recovery_grad_clip_norm=recovery_controls.grad_clip_norm,
            recovery_bigram_bias=recovery_controls.bigram_bias,
            recovery_bigram_bias_lr=recovery_controls.bigram_bias_lr,
            recovery_bigram_bias_init_from_data=recovery_controls.bigram_bias_init_from_data,
            recovery_bigram_bias_init_tokens=recovery_controls.bigram_bias_init_tokens,
            recovery_bigram_bias_init_alpha=recovery_controls.bigram_bias_init_alpha,
            recovery_bigram_bias_init_strength=recovery_controls.bigram_bias_init_strength,
            recovery_bigram_bias_scale=recovery_controls.bigram_bias_scale,
            recovery_advanced_loss_scale=recovery_controls.advanced_loss_scale,
            recovery_graphcg_loss_weight=recovery_controls.graphcg_loss_weight,
            recovery_toric_tropical_loss_weight=recovery_controls.toric_tropical_loss_weight,
            recovery_slepian_loss_weight=recovery_controls.slepian_loss_weight,
            recovery_koszul_bgg_loss_weight=recovery_controls.koszul_bgg_loss_weight,
            recovery_analogy_loss_weight=recovery_controls.analogy_loss_weight,
            recovery_advanced_loss_sample_tokens=recovery_controls.advanced_loss_sample_tokens,
            recovery_toric_tropical_fan_bins=recovery_controls.toric_tropical_fan_bins,
            recovery_advanced_loss_log_only=recovery_controls.advanced_loss_log_only,
            recovery_advanced_loss_start_step=recovery_controls.advanced_loss_start_step,
            recovery_advanced_loss_end_step=recovery_controls.advanced_loss_end_step,
            recovery_advanced_loss_every=recovery_controls.advanced_loss_every,
            recovery_advanced_loss_warmup_steps=recovery_controls.advanced_loss_warmup_steps,
            recovery_advanced_loss_min_best_val_bpb=recovery_controls.advanced_loss_min_best_val_bpb,
            recovery_advanced_loss_max_ce_ratio=recovery_controls.advanced_loss_max_ce_ratio,
            recovery_reset_optimizer=recovery_controls.reset_optimizer_on_resume,
            recovery_reset_rng=recovery_controls.reset_rng_on_resume,
            recovery_reset_loader=recovery_controls.reset_loader_on_resume,
            recovery_checkpoint_on_train_bpb_below=args.recovery_checkpoint_on_train_bpb_below,
            recovery_checkpoint_on_train_bpb_cooldown_steps=args.recovery_checkpoint_on_train_bpb_cooldown_steps,
            recovery_checkpoint_on_train_bpb_max=args.recovery_checkpoint_on_train_bpb_max,
            recovery_val_on_train_bpb_checkpoint=args.recovery_val_on_train_bpb_checkpoint,
            preempt_on_projected_miss=args.preempt_on_projected_miss,
            preempt_min_step=args.preempt_min_step,
            preempt_patience=args.preempt_patience,
            advanced_metric_controls=not args.no_advanced_metric_controls,
            recovery_max_train_batch_tokens=args.recovery_max_train_batch_tokens,
            validation_gap_threshold=args.validation_gap_threshold,
            low_train_bpb_margin=args.low_train_bpb_margin,
            python=args.python,
        )
        (command_dir / "analysis_command.sh").write_text(analysis_shell + "\n", encoding="utf-8")
        (command_dir / "dense_mirror_command.sh").write_text(dense_shell + "\n", encoding="utf-8")
        (command_dir / "full_diag_command.sh").write_text(diag_shell + "\n", encoding="utf-8")
        (command_dir / "gate_command.sh").write_text(gate_shell + "\n", encoding="utf-8")

        status.update(
            {
                "event": "launching_preemptive_gate_risk_recovery"
                if preempt_for_gate_risk
                else "launching_4k_recovery",
                "selected_recovery_validation": recovery_val.__dict__,
                "resume_checkpoint": str(resume_checkpoint),
                "recovery_run_id": recovery_run_id,
                "recovery_train_tmux": recovery_train_tmux,
                "recovery_log": str(recovery_log),
                "recovery_checkpoint_dir": str(recovery_checkpoint_dir),
                "command_dir": str(command_dir),
                "applied_recovery_launch_controls": recovery_controls.launch_dict(),
                "recovery_validation_interval_steps": recovery_eval_interval,
                "recovery_checkpoint_interval_steps": recovery_checkpoint_interval,
            }
        )
        write_state(state_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)

        if not args.dry_run:
            tmux_kill(args.train_tmux)
        tmux_start(recovery_train_tmux, launch.training_shell, args.dry_run)
        tmux_start(
            sanitize_tmux_name(f"toricgt_seq4096_4k_analysis_r{next_index}_{stamp}"),
            analysis_shell,
            args.dry_run,
        )
        tmux_start(
            sanitize_tmux_name(f"toricgt_seq4096_4k_mirror_r{next_index}_{stamp}"),
            dense_shell,
            args.dry_run,
        )
        tmux_start(
            sanitize_tmux_name(f"toricgt_seq4096_4k_full_diag_r{next_index}_{stamp}"),
            diag_shell,
            args.dry_run,
        )
        tmux_start(
            sanitize_tmux_name(f"toricgt_seq4096_4k_gate_r{next_index}_{stamp}"),
            gate_shell,
            args.dry_run,
        )
        return


if __name__ == "__main__":
    main()
