#!/usr/bin/env python3
"""Mirror full ToricGT diagnostic namespaces for a FineWeb BPB run.

The official-style Parameter-Golf FineWeb scaffold is an external byte-LM
trainer.  It prints loss/BPB/timing, but it does not expose ToricGT hidden
states or graph trajectories.  This companion process keeps the W&B metric
surface complete without interrupting the trainer:

* BPB/timing remain the primary optimization metrics.
* topology/toric/tropical/BGG metrics are finite audits over the training curve
  and deterministic certificates unless a future trainer exposes hidden states.
* status flags make that scope explicit in W&B.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from toricgt.complexity import compress_len, conditional_compress_len, normalized_compression_distance
from toricgt.koszul_persistence import KoszulPersistenceConfig, koszul_persistence_loss
from toricgt.slepian_torus import toric_slepian_audit
from toricgt.topological_reasoning import ReasoningTopologyConfig, reasoning_step_topology_loss
from toricgt.toric_bgg import ToricBGGConfig, ToricBGGProbe, boundary_square_residual, toy_bgg_certificate
from toricgt.toric_geometry_tasks import (
    LowRankToricGeometryProbe,
    ToricGeometryConfig,
    empirical_toric_shadow_stats_np,
)
from toricgt.wandb_organization import configure_wandb_metrics, organize_wandb_payload, update_wandb_summary


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
    r"\s+train_bpb:(?P<train_bpb>[0-9.eE+-]+)"
    r"\s+threshold:(?P<threshold>[0-9.eE+-]+)"
    r"\s+val_loss:(?P<val_loss>[0-9.eE+-]+)"
    r"\s+val_bpb:(?P<val_bpb>[0-9.eE+-]+)"
    r"\s+train_time:(?P<ms>[0-9.eE+-]+)ms"
)
ADVANCED_LOSSES_RE = re.compile(r"advanced_losses:(?P<body>.*)")


def parse_colon_metrics(body: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for token in body.strip().split():
        if ":" not in token:
            continue
        key, raw_value = token.split(":", 1)
        try:
            metrics[key] = float(raw_value)
        except ValueError:
            continue
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "toricgt-parameter-golf"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "amelie-iska-math"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--target-bpb", type=float, default=1.2)
    parser.add_argument("--gate-step", type=int, default=4000)
    parser.add_argument("--max-points", type=int, default=160)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--once", action="store_true", help="Log one diagnostics payload and exit.")
    parser.add_argument("--no-wandb", action="store_true", help="Write a one-shot JSON payload without opening W&B.")
    return parser.parse_args()


def parse_log(path: Path) -> dict[str, Any]:
    train: dict[int, dict[str, float]] = {}
    vals: dict[int, dict[str, float]] = {}
    advanced_losses: dict[str, float] = {}
    if not path.exists():
        return {"train": train, "vals": vals, "advanced_losses": advanced_losses, "latest_step": 0, "total": 0}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        advanced_match = ADVANCED_LOSSES_RE.search(line)
        if advanced_match:
            advanced_losses.update(parse_colon_metrics(advanced_match.group("body")))
            continue
        val = VAL_RE.search(line)
        if val:
            step = int(val.group("step"))
            vals[step] = {
                "step": float(step),
                "total": float(val.group("total")),
                "val_loss": float(val.group("loss")),
                "val_bpb": float(val.group("bpb")),
                "train_time_ms": float(val.group("ms")),
                "step_avg_ms": float(val.group("avg")),
            }
            continue
        low_train_val = LOW_TRAIN_BPB_TRIGGER_VAL_RE.search(line)
        if low_train_val:
            step = int(low_train_val.group("step"))
            total = float(low_train_val.group("total"))
            train_bpb = float(low_train_val.group("train_bpb"))
            train_time_ms = float(low_train_val.group("ms"))
            vals[step] = {
                "step": float(step),
                "total": total,
                "val_loss": float(low_train_val.group("val_loss")),
                "val_bpb": float(low_train_val.group("val_bpb")),
                "train_time_ms": train_time_ms,
                "step_avg_ms": float("nan"),
            }
            if step not in train or not math.isfinite(float(train[step].get("train_bpb", float("nan")))):
                train[step] = {
                    "step": float(step),
                    "total": total,
                    "train_loss": float(low_train_val.group("val_loss")),
                    "train_bpb": train_bpb,
                    "train_time_ms": train_time_ms,
                    "step_avg_ms": 0.0,
                }
            continue
        train_match = TRAIN_RE.search(line)
        if train_match:
            step = int(train_match.group("step"))
            train[step] = {
                "step": float(step),
                "total": float(train_match.group("total")),
                "train_loss": float(train_match.group("loss")),
                "train_bpb": float(train_match.group("bpb")) if train_match.group("bpb") is not None else float("nan"),
                "train_time_ms": float(train_match.group("ms")),
                "step_avg_ms": float(train_match.group("avg")),
            }
    latest_step = max([0, *train.keys(), *vals.keys()])
    total = int(max([0.0, *[row["total"] for row in train.values()], *[row["total"] for row in vals.values()]]))
    return {"train": train, "vals": vals, "advanced_losses": advanced_losses, "latest_step": latest_step, "total": total}


def _last_at_or_before(rows: dict[int, dict[str, float]], step: int, field: str, default: float) -> float:
    keys = [key for key in rows if key <= step]
    if not keys:
        return float(default)
    return float(rows[max(keys)].get(field, default))


def build_curve_tensor(parsed: dict[str, Any], max_points: int, target_bpb: float) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    train: dict[int, dict[str, float]] = parsed["train"]
    vals: dict[int, dict[str, float]] = parsed["vals"]
    steps = sorted(set(train) | set(vals))
    if not steps:
        hidden = torch.zeros((1, 0, 18), dtype=torch.float32)
        positions = torch.zeros((1, 0), dtype=torch.long)
        return hidden, positions, np.zeros((0,), dtype=np.float64)
    if len(steps) > max_points:
        idx = np.linspace(0, len(steps) - 1, num=max_points).round().astype(int)
        steps = [steps[int(i)] for i in idx]
    total = float(parsed.get("total") or max(steps) or 1)
    val_bpbi: list[float] = []
    rows: list[list[float]] = []
    best = float("inf")
    prev_val = _last_at_or_before(vals, steps[0], "val_bpb", target_bpb + 1.0)
    prev_train = _last_at_or_before(train, steps[0], "train_loss", 0.0)
    prev_train_bpb = _last_at_or_before(train, steps[0], "train_bpb", prev_val)
    for step in steps:
        train_loss = _last_at_or_before(train, step, "train_loss", prev_train)
        train_bpb = _last_at_or_before(train, step, "train_bpb", prev_train_bpb)
        if not math.isfinite(train_bpb):
            train_bpb = prev_train_bpb
        val_loss = _last_at_or_before(vals, step, "val_loss", train_loss)
        val_bpb = _last_at_or_before(vals, step, "val_bpb", prev_val)
        best = min(best, val_bpb)
        step_avg = _last_at_or_before(train, step, "step_avg_ms", 0.0)
        progress = float(step) / max(total, 1.0)
        gap = val_bpb - float(target_bpb)
        val_delta = val_bpb - prev_val
        train_delta = train_loss - prev_train
        train_bpb_delta = train_bpb - prev_train_bpb
        train_gap = train_bpb - float(target_bpb)
        generalization_gap = val_bpb - train_bpb
        angle_theta = 2.0 * math.pi * 0.6180339887498948 * step
        angle_beta = 2.0 * math.pi * (math.sqrt(2.0) - 1.0) * step
        rows.append(
            [
                progress,
                math.log1p(float(step)) / math.log1p(max(total, 1.0)),
                train_loss,
                train_bpb,
                val_loss,
                val_bpb,
                best,
                gap,
                train_gap,
                generalization_gap,
                val_delta,
                train_delta,
                train_bpb_delta,
                step_avg / 1000.0,
                math.sin(angle_theta),
                math.cos(angle_theta),
                math.sin(angle_beta),
                math.cos(angle_beta),
            ]
        )
        val_bpbi.append(val_bpb)
        prev_val = val_bpb
        prev_train = train_loss
        prev_train_bpb = train_bpb
    arr = np.asarray(rows, dtype=np.float32)
    if arr.shape[0] > 1:
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        arr = (arr - mean) / np.maximum(std, 1e-5)
    hidden = torch.from_numpy(arr[None, :, :]).float()
    positions = torch.tensor([steps], dtype=torch.long)
    return hidden, positions, np.asarray(val_bpbi, dtype=np.float64)


def _tensor_value(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(torch.nan_to_num(value.detach().float(), nan=0.0, posinf=1.0e6, neginf=-1.0e6).cpu())
    return float(value)


def bounded_pressure(value: float, scale: float, *, invert: bool = False, floor: float = 0.0) -> float:
    if not math.isfinite(value) or scale <= 0.0:
        return 0.0
    measured = max(0.0, floor - value) if invert else max(0.0, value - floor)
    return max(0.0, min(1.0, measured / float(scale)))


def finite_metric(payload: dict[str, float], *keys: str, default: float = float("nan")) -> float:
    for key in keys:
        try:
            value = float(payload.get(key, default))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return default


STRUCTURAL_FAMILY_IDS = {
    "none": 0.0,
    "bpb_gap": 1.0,
    "topology_directed": 2.0,
    "toric_slepian": 3.0,
    "bgg_koszul": 4.0,
    "tropical_complexity": 5.0,
}


def structural_family_pressures(components: dict[str, float] | None) -> dict[str, float]:
    components = components or {}

    def value(key: str) -> float:
        try:
            numeric = float(components.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0
        return numeric if math.isfinite(numeric) else 0.0

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
    valid = {
        str(key): float(value)
        for key, value in (families or {}).items()
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0
    }
    if not valid:
        return "none", 0.0
    key, value = max(valid.items(), key=lambda item: (item[1], item[0]))
    return key, float(value)


def add_prefixed_torch_metrics(
    target: dict[str, float],
    metrics: dict[str, torch.Tensor],
    *,
    prefix: str,
    strip_prefixes: tuple[str, ...] = (),
) -> None:
    for key, value in metrics.items():
        clean = key
        for strip in strip_prefixes:
            if clean.startswith(strip):
                clean = clean[len(strip) :]
        target[f"{prefix}/{clean}"] = _tensor_value(value)


def _latest_at_or_before(
    rows: dict[int, dict[str, float]],
    step: int,
    field: str,
) -> float | None:
    keys = [key for key in rows if key <= step]
    if not keys:
        return None
    value = rows[max(keys)].get(field)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def add_fineweb_curve_status(payload: dict[str, float], parsed: dict[str, Any], target_bpb: float) -> None:
    latest_step = int(parsed.get("latest_step") or 0)
    train: dict[int, dict[str, float]] = parsed["train"]
    vals: dict[int, dict[str, float]] = parsed["vals"]
    latest_train_bpb = _latest_at_or_before(train, latest_step, "train_bpb")
    latest_val_bpb = _latest_at_or_before(vals, latest_step, "val_bpb")
    if latest_train_bpb is not None:
        payload["fineweb_curve/latest_train_bpb"] = latest_train_bpb
        payload["fineweb_curve/latest_train_gap_to_target"] = latest_train_bpb - float(target_bpb)
    if latest_val_bpb is not None:
        payload["fineweb_curve/latest_val_bpb"] = latest_val_bpb
        payload["fineweb_curve/latest_val_gap_to_target"] = latest_val_bpb - float(target_bpb)
    if latest_train_bpb is not None and latest_val_bpb is not None:
        payload["fineweb_curve/latest_generalization_gap_bpb"] = latest_val_bpb - latest_train_bpb


def _slope_per_100_steps(steps: np.ndarray, values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    x = steps.astype(np.float64) / 100.0
    y = values.astype(np.float64)
    x = x - float(x.mean())
    denom = float(np.sum(x * x))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x * (y - float(y.mean()))) / denom)


def fineweb_transfer_metrics(parsed: dict[str, Any], target_bpb: float) -> dict[str, float]:
    train: dict[int, dict[str, float]] = parsed["train"]
    vals: dict[int, dict[str, float]] = parsed["vals"]
    rows: list[tuple[float, float, float]] = []
    for step in sorted(vals):
        val_bpb = _latest_at_or_before(vals, step, "val_bpb")
        train_bpb = _latest_at_or_before(train, step, "train_bpb")
        if val_bpb is None or train_bpb is None:
            continue
        rows.append((float(step), float(train_bpb), float(val_bpb)))
    if not rows:
        return {"fineweb_curve/transfer_pairs": 0.0}

    arr = np.asarray(rows, dtype=np.float64)
    steps = arr[:, 0]
    train_bpb = arr[:, 1]
    val_bpb = arr[:, 2]
    gaps = val_bpb - train_bpb
    metrics: dict[str, float] = {
        "fineweb_curve/transfer_pairs": float(len(rows)),
        "fineweb_curve/latest_train_bpb_at_validation": float(train_bpb[-1]),
        "fineweb_curve/latest_val_bpb_for_transfer": float(val_bpb[-1]),
        "fineweb_curve/latest_generalization_gap_bpb": float(gaps[-1]),
        "fineweb_curve/generalization_gap_slope_per_100_steps": _slope_per_100_steps(steps, gaps),
    }
    if len(rows) < 2:
        metrics.update(
            {
                "fineweb_curve/cumulative_train_drop_bpb": 0.0,
                "fineweb_curve/cumulative_val_drop_bpb": 0.0,
                "fineweb_curve/validation_transfer_pressure": 0.0,
            }
        )
        return metrics

    train_drop = train_bpb[:-1] - train_bpb[1:]
    val_drop = val_bpb[:-1] - val_bpb[1:]
    valid = np.isfinite(train_drop) & np.isfinite(val_drop) & (train_drop > 1e-9)
    efficiencies = np.full_like(train_drop, np.nan, dtype=np.float64)
    efficiencies[valid] = val_drop[valid] / train_drop[valid]
    recent_efficiency = float(efficiencies[np.isfinite(efficiencies)][-1]) if np.isfinite(efficiencies).any() else float("nan")
    recent_train_drop = float(train_drop[-1])
    recent_val_drop = float(val_drop[-1])
    cumulative_train_drop = float(train_bpb[0] - train_bpb[-1])
    cumulative_val_drop = float(val_bpb[0] - val_bpb[-1])
    cumulative_efficiency = (
        cumulative_val_drop / cumulative_train_drop
        if cumulative_train_drop > 1e-9
        else float("nan")
    )
    transfer_pressure = max(0.0, recent_train_drop - recent_val_drop)
    lag_pressure = max(0.0, gaps[-1] - 0.04) + max(0.0, 0.55 - recent_efficiency if math.isfinite(recent_efficiency) else 0.0)
    metrics.update(
        {
            "fineweb_curve/recent_train_drop_bpb": recent_train_drop,
            "fineweb_curve/recent_val_drop_bpb": recent_val_drop,
            "fineweb_curve/transfer_efficiency_recent": recent_efficiency,
            "fineweb_curve/cumulative_train_drop_bpb": cumulative_train_drop,
            "fineweb_curve/cumulative_val_drop_bpb": cumulative_val_drop,
            "fineweb_curve/transfer_efficiency_cumulative": cumulative_efficiency,
            "fineweb_curve/validation_transfer_pressure": transfer_pressure,
            "fineweb_curve/validation_lag_pressure": lag_pressure,
            "fineweb_curve/target_gap_at_transfer_step": float(val_bpb[-1] - float(target_bpb)),
        }
    )
    return metrics


def fineweb_gate_velocity_metrics(parsed: dict[str, Any], target_bpb: float, gate_step: int) -> dict[str, float]:
    vals: dict[int, dict[str, float]] = parsed["vals"]
    rows: list[tuple[float, float]] = []
    for step in sorted(vals):
        val_bpb = _latest_at_or_before(vals, step, "val_bpb")
        if val_bpb is not None:
            rows.append((float(step), float(val_bpb)))
    if not rows:
        return {"fineweb_curve/val_velocity_pairs": 0.0}

    latest_step, latest_val = rows[-1]
    remaining_steps = max(1.0, float(gate_step) - latest_step)
    target_gap = max(0.0, latest_val - float(target_bpb))
    required_velocity = target_gap / remaining_steps * 100.0
    recent_velocity = float("nan")
    previous_velocity = float("nan")
    velocity_delta = float("nan")
    if len(rows) >= 2:
        prev_step, prev_val = rows[-2]
        recent_velocity = max(0.0, (prev_val - latest_val) / max(1.0, latest_step - prev_step) * 100.0)
    if len(rows) >= 3:
        prev2_step, prev2_val = rows[-3]
        prev_step, prev_val = rows[-2]
        previous_velocity = max(0.0, (prev2_val - prev_val) / max(1.0, prev_step - prev2_step) * 100.0)
    if math.isfinite(recent_velocity) and math.isfinite(previous_velocity):
        velocity_delta = recent_velocity - previous_velocity
    effective_recent = recent_velocity if math.isfinite(recent_velocity) else 0.0
    shortfall = max(0.0, required_velocity - effective_recent)
    decay = max(0.0, previous_velocity - effective_recent) if math.isfinite(previous_velocity) else 0.0
    return {
        "fineweb_curve/val_velocity_pairs": float(len(rows)),
        "fineweb_curve/val_bpb_velocity_recent_per_100_steps": recent_velocity,
        "fineweb_curve/val_bpb_velocity_previous_per_100_steps": previous_velocity,
        "fineweb_curve/val_bpb_velocity_delta_per_100_steps": velocity_delta,
        "fineweb_curve/val_bpb_velocity_decay_per_100_steps": decay,
        "fineweb_curve/required_val_velocity_to_gate_per_100_steps": required_velocity,
        "fineweb_curve/val_velocity_shortfall_to_gate_per_100_steps": shortfall,
        "fineweb_curve/bpb_velocity_shortfall_pressure": max(0.0, min(1.0, shortfall / 0.005)),
    }


def _latest_step_and_value(rows: dict[int, dict[str, float]], field: str) -> tuple[int, float]:
    if not rows:
        return 0, float("nan")
    step = max(rows)
    value = rows[step].get(field, float("nan"))
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float("nan")
    return int(step), numeric if math.isfinite(numeric) else float("nan")


def diagnostic_log_signature(parsed: dict[str, Any]) -> tuple[int, int, int, float, float, int, int]:
    """Detect a same-step validation row arriving after the train row."""

    train: dict[int, dict[str, float]] = parsed["train"]
    vals: dict[int, dict[str, float]] = parsed["vals"]
    latest_step = int(parsed.get("latest_step") or 0)
    latest_train_step, latest_train_bpb = _latest_step_and_value(train, "train_bpb")
    latest_val_step, latest_val_bpb = _latest_step_and_value(vals, "val_bpb")
    return (
        latest_step,
        latest_train_step,
        latest_val_step,
        round(latest_train_bpb, 8) if math.isfinite(latest_train_bpb) else float("nan"),
        round(latest_val_bpb, 8) if math.isfinite(latest_val_bpb) else float("nan"),
        len(train),
        len(vals),
    )


DIAGNOSTIC_ALIAS_KEYS = {
    "topology/topology_loss": "diagnostics/latest/topology_loss",
    "topology/directed_topology_loss": "diagnostics/latest/directed_topology_loss",
    "topology/simplex_closure_loss": "diagnostics/latest/simplex_closure_loss",
    "topology/directed_chain_commutator": "diagnostics/latest/directed_chain_commutator",
    "topology/cycle_rank": "diagnostics/latest/cycle_rank",
    "topology/hdbscan_stability": "diagnostics/latest/hdbscan_stability",
    "topology/affine_toric/persistence_loss": "diagnostics/latest/persistence_loss",
    "bgg_category_o/persistence/exactness_residual": "diagnostics/latest/koszul_exactness_residual",
    "bgg_category_o/persistence/syzygy_residual": "diagnostics/latest/koszul_syzygy_residual",
    "bgg_category_o/persistence/buchsbaum_eisenbud_rank_residual": "diagnostics/latest/koszul_buchsbaum_eisenbud_rank_residual",
    "bgg_category_o/d2_residual": "diagnostics/latest/bgg_d2_residual",
    "bgg_category_o/loss": "diagnostics/latest/bgg_loss",
    "bgg_category_o/resolution_consistency": "diagnostics/latest/bgg_resolution_consistency",
    "bgg_category_o/standard_leakage": "diagnostics/latest/bgg_standard_leakage",
    "toric/geometry_loss": "diagnostics/latest/toric_geometry_loss",
    "toric/shadow_fan_cell_entropy": "diagnostics/latest/toric_shadow_fan_cell_entropy",
    "toric/shadow_mean_bend": "diagnostics/latest/toric_shadow_mean_bend",
    "toric/active_face_entropy": "diagnostics/latest/toric_active_face_entropy",
    "toric/active_face_margin": "diagnostics/latest/toric_active_face_margin",
    "toric/braid_loss": "diagnostics/latest/toric_braid_loss",
    "toric/binomial_residual": "diagnostics/latest/toric_binomial_residual",
    "toric/slepian_concentration": "diagnostics/latest/slepian_concentration",
    "toric/slepian_leakage": "diagnostics/latest/slepian_leakage",
    "toric/slepian_mode_entropy": "diagnostics/latest/slepian_mode_entropy",
    "toric/slepian_effective_modes": "diagnostics/latest/slepian_effective_modes",
    "tropical/bpb_recent_slope": "diagnostics/latest/tropical_bpb_recent_slope",
    "tropical/bpb_target_gap": "diagnostics/latest/tropical_bpb_target_gap",
    "tropical/bpb_active_face_entropy": "diagnostics/latest/tropical_bpb_active_face_entropy",
    "tropical/bpb_plateau_pressure": "diagnostics/latest/tropical_bpb_plateau_pressure",
    "complexity/recent_full_log_ncd_lzma": "diagnostics/latest/complexity_recent_full_log_ncd_lzma",
    "fineweb_curve/latest_train_bpb": "diagnostics/latest/train_bpb",
    "fineweb_curve/latest_val_bpb": "diagnostics/latest/val_bpb",
    "fineweb_curve/latest_generalization_gap_bpb": "diagnostics/latest/generalization_gap_bpb",
    "fineweb_curve/latest_train_bpb_at_validation": "diagnostics/latest/train_bpb_at_validation",
    "fineweb_curve/latest_val_bpb_for_transfer": "diagnostics/latest/val_bpb_for_transfer",
    "fineweb_curve/generalization_gap_slope_per_100_steps": "diagnostics/latest/generalization_gap_slope_per_100_steps",
    "fineweb_curve/transfer_efficiency_recent": "diagnostics/latest/transfer_efficiency_recent",
    "fineweb_curve/transfer_efficiency_cumulative": "diagnostics/latest/transfer_efficiency_cumulative",
    "fineweb_curve/recent_train_drop_bpb": "diagnostics/latest/recent_train_drop_bpb",
    "fineweb_curve/recent_val_drop_bpb": "diagnostics/latest/recent_val_drop_bpb",
    "fineweb_curve/validation_transfer_pressure": "diagnostics/latest/validation_transfer_pressure",
    "fineweb_curve/validation_lag_pressure": "diagnostics/latest/validation_lag_pressure",
    "fineweb_curve/val_bpb_velocity_recent_per_100_steps": "diagnostics/latest/val_bpb_velocity_recent_per_100_steps",
    "fineweb_curve/val_bpb_velocity_previous_per_100_steps": "diagnostics/latest/val_bpb_velocity_previous_per_100_steps",
    "fineweb_curve/val_bpb_velocity_delta_per_100_steps": "diagnostics/latest/val_bpb_velocity_delta_per_100_steps",
    "fineweb_curve/val_bpb_velocity_decay_per_100_steps": "diagnostics/latest/val_bpb_velocity_decay_per_100_steps",
    "fineweb_curve/required_val_velocity_to_gate_per_100_steps": "diagnostics/latest/required_val_velocity_to_gate_per_100_steps",
    "fineweb_curve/val_velocity_shortfall_to_gate_per_100_steps": "diagnostics/latest/val_velocity_shortfall_to_gate_per_100_steps",
    "fineweb_curve/bpb_velocity_shortfall_pressure": "diagnostics/latest/bpb_velocity_shortfall_pressure",
}


def diagnostic_alias_payload(payload: dict[str, float]) -> dict[str, float]:
    aliases: dict[str, float] = {
        "diagnostics/families/topology_available": 1.0 if any(key.startswith("topology/") for key in payload) else 0.0,
        "diagnostics/families/toric_available": 1.0 if any(key.startswith("toric/") for key in payload) else 0.0,
        "diagnostics/families/tropical_available": 1.0 if any(key.startswith("tropical/") for key in payload) else 0.0,
        "diagnostics/families/category_o_bgg_available": 1.0 if any(key.startswith("bgg_category_o/") for key in payload) else 0.0,
        "diagnostics/families/koszul_persistence_available": 1.0
        if any(key.startswith("bgg_category_o/persistence/") for key in payload)
        else 0.0,
        "diagnostics/families/slepian_pollak_prolate_available": 1.0
        if any(key.startswith("toric/slepian_") for key in payload)
        else 0.0,
        "diagnostics/families/complexity_available": 1.0 if any(key.startswith("complexity/") for key in payload) else 0.0,
        "diagnostics/families/fineweb_bpb_curve_available": 1.0
        if any(key.startswith("fineweb_curve/") for key in payload)
        else 0.0,
    }
    for source, alias in DIAGNOSTIC_ALIAS_KEYS.items():
        value = payload.get(source)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            aliases[alias] = numeric
    if "diagnostics/latest/slepian_concentration" in aliases:
        aliases["diagnostics/latest/pollak_prolate_slepian_concentration"] = aliases[
            "diagnostics/latest/slepian_concentration"
        ]
    if "diagnostics/latest/slepian_leakage" in aliases:
        aliases["diagnostics/latest/pollak_prolate_slepian_leakage"] = aliases["diagnostics/latest/slepian_leakage"]
    gap = aliases.get("diagnostics/latest/tropical_bpb_target_gap")
    plateau = aliases.get("diagnostics/latest/tropical_bpb_plateau_pressure", 0.0)
    topology_loss = aliases.get("diagnostics/latest/topology_loss", 0.0)
    slepian_leakage = aliases.get("diagnostics/latest/slepian_leakage", 0.0)
    if gap is not None:
        aliases["diagnostics/latest/bpb_intervention_pressure"] = float(
            max(0.0, gap) + max(0.0, plateau) + 0.05 * max(0.0, topology_loss) + 0.02 * max(0.0, slepian_leakage)
        )
    return aliases


def structural_recapture_payload(payload: dict[str, float]) -> dict[str, float]:
    bpb_pressure = finite_metric(payload, "diagnostics/latest/bpb_intervention_pressure", default=0.0)
    topology_loss = finite_metric(payload, "diagnostics/latest/topology_loss", "topology/topology_loss")
    directed_topology_loss = finite_metric(
        payload,
        "diagnostics/latest/directed_topology_loss",
        "topology/directed_topology_loss",
    )
    slepian_leakage = finite_metric(
        payload,
        "diagnostics/latest/slepian_leakage",
        "diagnostics/latest/pollak_prolate_slepian_leakage",
        "toric/slepian_leakage",
    )
    toric_margin = finite_metric(
        payload,
        "diagnostics/latest/toric_active_face_margin",
        "toric/active_face_margin",
    )
    toric_bend = finite_metric(
        payload,
        "diagnostics/latest/toric_shadow_mean_bend",
        "toric/shadow_mean_bend",
    )
    bgg_standard_leakage = finite_metric(
        payload,
        "diagnostics/latest/bgg_standard_leakage",
        "bgg_category_o/standard_leakage",
        default=0.0,
    )
    bgg_d2_residual = finite_metric(
        payload,
        "diagnostics/latest/bgg_d2_residual",
        "bgg_category_o/d2_residual",
        default=0.0,
    )
    complexity_ncd = finite_metric(
        payload,
        "diagnostics/latest/complexity_recent_full_log_ncd_lzma",
        "complexity/recent_full_log_ncd_lzma",
        default=0.0,
    )
    tropical_plateau = finite_metric(
        payload,
        "diagnostics/latest/tropical_bpb_plateau_pressure",
        "tropical/bpb_plateau_pressure",
        default=0.0,
    )
    components = {
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
    family_pressures = structural_family_pressures(components)
    dominant_family, dominant_pressure = dominant_structural_family(family_pressures)
    score = max(0.0, min(1.0, sum(components.values())))
    families_available = any(
        finite_metric(payload, key, default=0.0) >= 0.5
        for key in (
            "diagnostics/families/topology_available",
            "diagnostics/families/toric_available",
            "diagnostics/families/slepian_pollak_prolate_available",
            "diagnostics/families/koszul_persistence_available",
            "diagnostics/families/category_o_bgg_available",
            "diagnostics/families/tropical_available",
        )
    )
    if not families_available:
        band_id = 1.0
    elif score >= 0.75:
        band_id = 4.0
    elif score >= 0.50:
        band_id = 3.0
    elif score >= 0.30:
        band_id = 2.0
    else:
        band_id = 0.0
    out = {
        "diagnostics/latest/structural_recapture_score": score,
        "diagnostics/latest/structural_pressure_high": float(families_available and score >= 0.50),
        "diagnostics/latest/structural_recapture_band_id": band_id,
        "diagnostics/structural_recapture_score": score,
        "diagnostics/structural_recapture_band_id": band_id,
        "diagnostics/families/structural_recapture_available": 1.0 if families_available else 0.0,
    }
    for key, value in components.items():
        out[f"diagnostics/structural_recapture_components/{key}"] = float(value)
    for key, value in family_pressures.items():
        out[f"diagnostics/structural_family_pressure/{key}"] = float(value)
        out[f"diagnostics/latest/structural_family_pressure_{key}"] = float(value)
    out["diagnostics/latest/dominant_structural_pressure_id"] = float(
        STRUCTURAL_FAMILY_IDS.get(dominant_family, 0.0)
    )
    out["diagnostics/latest/dominant_structural_pressure_value"] = float(dominant_pressure)
    out["diagnostics/dominant_structural_pressure_id"] = float(STRUCTURAL_FAMILY_IDS.get(dominant_family, 0.0))
    out["diagnostics/dominant_structural_pressure_value"] = float(dominant_pressure)
    return out


def tropical_curve_metrics(values: np.ndarray, target_bpb: float) -> dict[str, float]:
    if values.size == 0:
        return {
            "tropical/bpb_best_so_far": 0.0,
            "tropical/bpb_margin_to_second_best": 0.0,
            "tropical/bpb_target_gap": 0.0,
            "tropical/bpb_recent_slope": 0.0,
            "tropical/bpb_active_face_entropy": 0.0,
        }
    best = float(np.min(values))
    sorted_vals = np.sort(values)
    second = float(sorted_vals[1]) if sorted_vals.size > 1 else best
    recent = values[-min(6, values.size) :]
    slope = 0.0
    if recent.size > 1:
        x = np.arange(recent.size, dtype=np.float64)
        slope = float(np.polyfit(x, recent, deg=1)[0])
    deltas = np.diff(values, prepend=values[0])
    faces = np.digitize(deltas, bins=np.asarray([-0.02, -0.01, -0.005, 0.0, 0.005], dtype=np.float64))
    counts = np.bincount(faces, minlength=7).astype(np.float64)
    probs = counts[counts > 0] / max(1.0, counts.sum())
    entropy = float(-(probs * np.log(probs)).sum() / math.log(max(2, counts.size)))
    return {
        "tropical/bpb_best_so_far": best,
        "tropical/bpb_margin_to_second_best": float(second - best),
        "tropical/bpb_target_gap": float(best - target_bpb),
        "tropical/bpb_recent_slope": slope,
        "tropical/bpb_active_face_entropy": entropy,
        "tropical/bpb_minplus_improvement": float(values[0] - best),
        "tropical/bpb_plateau_pressure": float(max(0.0, slope)),
    }


def complexity_metrics(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {"complexity/run_log_available": 0.0}
    data = log_path.read_bytes()
    recent = b"\n".join(data.splitlines()[-200:])
    metrics: dict[str, float] = {"complexity/run_log_available": 1.0, "complexity/run_log_bytes": float(len(data))}
    for compressor in ("zlib", "lzma"):
        full_k = float(compress_len(data, compressor))
        recent_k = float(compress_len(recent, compressor))
        cond_k = float(conditional_compress_len(recent, data[-max(1, min(len(data), 64_000)) :], compressor))
        metrics[f"complexity/run_log_k_{compressor}"] = full_k
        metrics[f"complexity/recent_log_k_{compressor}"] = recent_k
        metrics[f"complexity/recent_to_tail_cond_k_{compressor}"] = cond_k
        metrics[f"complexity/run_log_k_{compressor}_per_byte"] = full_k / max(1.0, float(len(data)))
    if len(data) and len(recent):
        metrics["complexity/recent_full_log_ncd_lzma"] = float(normalized_compression_distance(recent, data, "lzma"))
    return metrics


def advanced_training_control_metrics(parsed: dict[str, Any]) -> dict[str, float]:
    controls = {str(k): float(v) for k, v in dict(parsed.get("advanced_losses") or {}).items()}
    scale = controls.get("scale", 0.0)
    log_only = controls.get("log_only", 0.0) > 0.5
    enabled = scale > 0.0 and not log_only
    family_weights = {
        "graphcg": controls.get("graphcg", 0.0),
        "toric_tropical": controls.get("toric_tropical", 0.0),
        "slepian_pollak_prolate": controls.get("slepian", 0.0),
        "koszul_bgg": controls.get("koszul_bgg", 0.0),
        "analogical_reasoning": controls.get("analogy", 0.0),
    }
    metrics: dict[str, float] = {
        "metrics_status/advanced_loss_controls_available": 1.0 if controls else 0.0,
        "metrics_status/advanced_loss_training_enabled": 1.0 if enabled else 0.0,
        "metrics_status/full_toricgt_training_metrics_available": 1.0 if enabled and any(v > 0.0 for v in family_weights.values()) else 0.0,
        "metrics_status/toric_bgg_training_enabled": 1.0
        if enabled and (family_weights["toric_tropical"] > 0.0 or family_weights["koszul_bgg"] > 0.0)
        else 0.0,
        "metrics_status/graphcg_training_enabled": 1.0 if enabled and family_weights["graphcg"] > 0.0 else 0.0,
        "metrics_status/toric_tropical_training_enabled": 1.0
        if enabled and family_weights["toric_tropical"] > 0.0
        else 0.0,
        "metrics_status/slepian_pollak_training_enabled": 1.0
        if enabled and family_weights["slepian_pollak_prolate"] > 0.0
        else 0.0,
        "metrics_status/koszul_bgg_training_enabled": 1.0 if enabled and family_weights["koszul_bgg"] > 0.0 else 0.0,
        "metrics_status/analogical_reasoning_training_enabled": 1.0
        if enabled and family_weights["analogical_reasoning"] > 0.0
        else 0.0,
        "diagnostics/families/graphcg_training_enabled": 1.0 if enabled and family_weights["graphcg"] > 0.0 else 0.0,
        "diagnostics/families/toric_tropical_training_enabled": 1.0
        if enabled and family_weights["toric_tropical"] > 0.0
        else 0.0,
        "diagnostics/families/slepian_pollak_training_enabled": 1.0
        if enabled and family_weights["slepian_pollak_prolate"] > 0.0
        else 0.0,
        "diagnostics/families/koszul_bgg_training_enabled": 1.0
        if enabled and family_weights["koszul_bgg"] > 0.0
        else 0.0,
        "diagnostics/families/analogical_reasoning_training_enabled": 1.0
        if enabled and family_weights["analogical_reasoning"] > 0.0
        else 0.0,
    }
    for key, value in controls.items():
        metrics[f"advanced_control/log/{key}"] = value
    for key, value in family_weights.items():
        metrics[f"advanced_control/log/{key}_effective_weight"] = scale * value if enabled else 0.0
    return metrics


def compute_payload(
    parsed: dict[str, Any],
    log_path: Path,
    target_bpb: float,
    max_points: int,
    gate_step: int = 4000,
) -> dict[str, float]:
    hidden, positions, val_bpbs = build_curve_tensor(parsed, max_points, target_bpb)
    payload: dict[str, float] = {
        "metrics_status/model_hidden_state_available": 0.0,
        "metrics_status/fineweb_curve_diagnostics_available": 1.0 if hidden.shape[1] >= 4 else 0.0,
        "metrics_status/full_toricgt_training_metrics_available": 0.0,
        "metrics_status/toric_bgg_training_enabled": 0.0,
        "metrics_status/metric_scope_fineweb_curve_proxy": 1.0,
        "diagnostics/source_code": 1.0,
    }
    payload.update(advanced_training_control_metrics(parsed))
    if hidden.shape[1] >= 4:
        topology = reasoning_step_topology_loss(
            hidden,
            config=ReasoningTopologyConfig(max_points=24, max_windows=6, window_size=32, step_stride=8),
        )
        add_prefixed_torch_metrics(
            payload,
            topology,
            prefix="topology",
            strip_prefixes=("reasoning_step_",),
        )
        koszul = koszul_persistence_loss(
            hidden,
            positions,
            config=KoszulPersistenceConfig(max_points=24, max_windows=4, window_size=32, step_stride=8),
        )
        add_prefixed_torch_metrics(payload, koszul, prefix="bgg_category_o/persistence", strip_prefixes=("koszul_",))
        add_prefixed_torch_metrics(payload, koszul, prefix="topology/affine_toric", strip_prefixes=("koszul_",))

        torch.manual_seed(17)
        toric_probe = LowRankToricGeometryProbe(
            hidden.shape[-1],
            ToricGeometryConfig(enabled=True, max_positions=max_points, num_exponents=16, probe_rank=12),
        )
        toric_probe.eval()
        with torch.no_grad():
            toric = toric_probe(hidden, positions)
        add_prefixed_torch_metrics(payload, toric, prefix="toric", strip_prefixes=("toric_",))
        shadow = empirical_toric_shadow_stats_np(hidden.reshape(-1, hidden.shape[-1]).numpy(), max_points=max_points)
        for key in ("occupied_fan_cells", "fan_cell_entropy", "mean_margin", "min_margin", "mean_bend", "slope_residual"):
            payload[f"toric/shadow_{key}"] = float(shadow.get(key, 0.0))

        phase_u = ((positions.numpy().reshape(-1) * 0.6180339887498948) % 1.0).astype(np.float64)
        phase_v = ((positions.numpy().reshape(-1) * (math.sqrt(2.0) - 1.0)) % 1.0).astype(np.float64)
        energy = val_bpbs if val_bpbs.size == phase_u.size else None
        slepian = toric_slepian_audit(phase_u, phase_v, energy=energy)
        for key in ("slepian_concentration", "slepian_leakage", "slepian_mode_entropy", "slepian_effective_modes", "slepian_bandwidth"):
            payload[f"toric/slepian_{key.removeprefix('slepian_')}"] = float(slepian[key])

        torch.manual_seed(23)
        bgg_probe = ToricBGGProbe(
            hidden.shape[-1],
            ToricBGGConfig(num_standard_tokens=8, probe_rank=8, signature_dim=16, max_positions=64),
        )
        bgg_probe.eval()
        with torch.no_grad():
            bgg = bgg_probe(hidden, target_positions=positions)
        add_prefixed_torch_metrics(payload, bgg, prefix="bgg_category_o", strip_prefixes=("toric_bgg_",))
        add_prefixed_torch_metrics(payload, bgg, prefix="category_o", strip_prefixes=("toric_bgg_",))
    cert = toy_bgg_certificate()
    toy_d2 = boundary_square_residual(cert.boundaries)
    payload["bgg_category_o/toy_koszul_d2_residual"] = _tensor_value(toy_d2)
    payload["bgg_category_o/toy_standard_poset_size"] = float(cert.standard_mask.shape[0])
    payload["bgg_category_o/late_phase_toggle_enabled"] = 0.0
    payload["category_o/late_phase_toggle_enabled"] = 0.0
    payload.update(tropical_curve_metrics(val_bpbs, target_bpb))
    payload.update(complexity_metrics(log_path))
    add_fineweb_curve_status(payload, parsed, target_bpb)
    payload.update(fineweb_transfer_metrics(parsed, target_bpb))
    payload.update(fineweb_gate_velocity_metrics(parsed, target_bpb, gate_step))
    payload.update(diagnostic_alias_payload(payload))
    payload.update(structural_recapture_payload(payload))
    return payload


def write_json(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sync_public_summary(
    wandb_module: Any,
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
    except Exception as exc:  # pragma: no cover - network/API failure should not stop diagnostics.
        print(f"wandb_summary_sync_failed:{type(exc).__name__}:{exc}", flush=True)
        return False
    return True


def run_once_without_wandb(
    *,
    log_path: Path,
    output_json: Path | str,
    target_bpb: float,
    max_points: int,
    gate_step: int = 4000,
) -> dict[str, float]:
    parsed = parse_log(log_path)
    latest_step = int(parsed.get("latest_step") or 0)
    total = int(parsed.get("total") or 0)
    payload = compute_payload(parsed, log_path, target_bpb, max_points, gate_step)
    payload.update(
        {
            "trainer/step": latest_step,
            "trainer/total_steps": total,
            "progress/step": latest_step,
            "progress/total_steps": total,
            "progress/fraction": float(latest_step) / max(float(total), 1.0),
        }
    )
    write_json(str(output_json), payload)
    return payload


def main() -> None:
    args = parse_args()
    if args.no_wandb:
        payload = run_once_without_wandb(
            log_path=Path(args.log),
            output_json=args.output_json,
            target_bpb=args.target_bpb,
            max_points=args.max_points,
            gate_step=args.gate_step,
        )
        print(
            "fineweb_full_diag_json "
            f"step={int(payload.get('trainer/step', 0.0))} "
            f"topology={payload.get('topology/topology_loss', 0.0):.4f} "
            f"toric_entropy={payload.get('toric/shadow_fan_cell_entropy', 0.0):.4f}",
            flush=True,
        )
        return

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
            "monitor_kind": "fineweb_full_toricgt_diagnostics",
            "diagnostic_scope": "fineweb_training_curve_proxy",
        },
    )
    configure_wandb_metrics(wandb)
    log_path = Path(args.log)
    seen_signatures: set[tuple[int, int, int, float, float, int, int]] = set()
    while True:
        parsed = parse_log(log_path)
        latest_step = int(parsed.get("latest_step") or 0)
        total = int(parsed.get("total") or 0)
        signature = diagnostic_log_signature(parsed)
        should_log = latest_step > 0 and signature not in seen_signatures
        if should_log:
            payload = compute_payload(parsed, log_path, args.target_bpb, args.max_points, args.gate_step)
            payload.update(
                {
                    "trainer/step": latest_step,
                    "trainer/total_steps": total,
                    "progress/step": latest_step,
                    "progress/total_steps": total,
                    "progress/fraction": float(latest_step) / max(float(total), 1.0),
                }
            )
            wandb.log(organize_wandb_payload(payload))
            summary_payload = {
                "diagnostics/latest_full_metrics_step": latest_step,
                "metrics_status/model_hidden_state_available": 0.0,
                "metrics_status/fineweb_curve_diagnostics_available": payload[
                    "metrics_status/fineweb_curve_diagnostics_available"
                ],
                "metrics_status/full_toricgt_training_metrics_available": payload[
                    "metrics_status/full_toricgt_training_metrics_available"
                ],
                "metrics_status/toric_bgg_training_enabled": payload["metrics_status/toric_bgg_training_enabled"],
                "bgg_category_o/late_phase_toggle_enabled": 0.0,
                "toric/shadow_fan_cell_entropy": payload.get("toric/shadow_fan_cell_entropy", 0.0),
                "topology/topology_loss": payload.get("topology/topology_loss", 0.0),
                "tropical/bpb_best_so_far": payload.get("tropical/bpb_best_so_far", 0.0),
            }
            summary_payload.update({key: value for key, value in payload.items() if key.startswith("metrics_status/")})
            summary_payload.update({key: value for key, value in payload.items() if key.startswith("advanced_control/log/")})
            summary_payload.update({key: value for key, value in payload.items() if key.startswith("diagnostics/latest/")})
            summary_payload.update({key: value for key, value in payload.items() if key.startswith("diagnostics/families/")})
            summary_payload.update(
                {
                    key: value
                    for key, value in payload.items()
                    if key.startswith("diagnostics/structural_recapture")
                }
            )
            update_wandb_summary(run, summary_payload)
            if sync_public_summary(
                wandb,
                entity=args.entity,
                project=args.project,
                run_id=args.run_id,
                summary_payload=summary_payload,
            ):
                print(f"wandb_summary_sync step={latest_step} keys={len(summary_payload)}", flush=True)
            write_json(args.output_json, payload)
            print(
                "wandb_full_diag "
                f"step={latest_step} topology={payload.get('topology/topology_loss', 0.0):.4f} "
                f"toric_entropy={payload.get('toric/shadow_fan_cell_entropy', 0.0):.4f} "
                f"bgg_d2={payload.get('bgg_category_o/d2_residual', 0.0):.4f} "
                f"structural_recapture={payload.get('diagnostics/latest/structural_recapture_score', 0.0):.4f}",
                flush=True,
            )
            seen_signatures.add(signature)
            if args.once:
                break
        if log_path.exists():
            text_tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
            if any(marker in text_tail for marker in ("final_int8_zlib_roundtrip", "Traceback", "RuntimeError")):
                break
        if args.once:
            break
        time.sleep(max(1.0, args.poll_seconds))
    run.finish()


if __name__ == "__main__":
    main()
