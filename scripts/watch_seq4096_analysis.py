#!/usr/bin/env python3
"""Checkpoint-gated analysis watcher for Seq4096 Parameter-Golf runs.

The historical ToricGT watcher is built around native
``random_order_step_*.pt`` checkpoints.  The OpenAI Parameter-Golf Seq4096
trainer writes checkpoints named like ``<run>_step_000500.pt`` and logs BPB in
a plain text stream.  This watcher keeps the same periodic review habit while
using the real Seq4096 artifacts:

* wait for every checkpoint interval;
* export W&B metric statistics and plots;
* run one full FineWeb curve diagnostic payload into W&B and JSON;
* produce BPB-specific acceleration, simplex, velocity, and phase-plane plots;
* write a synopsis that points to the next training intervention.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+train_loss:(?P<loss>[0-9.eE+-]+)"
    r"\s+train_time:(?P<ms>[0-9.eE+-]+)ms\s+step_avg:(?P<avg>[0-9.eE+-]+)ms"
    r"(?:\s+train_bpb:(?P<bpb>[0-9.eE+-]+))?"
)
VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<loss>[0-9.eE+-]+)"
    r"\s+val_bpb:(?P<bpb>[0-9.eE+-]+)\s+train_time:(?P<ms>[0-9.eE+-]+)ms"
    r"\s+step_avg:(?P<avg>[0-9.eE+-]+)ms"
)
SEQ4096_CHECKPOINT_RE = re.compile(r"(?:^|_)step_(?P<step>\d+)\.pt$")


class TrainRow(NamedTuple):
    step: int
    total: int
    train_loss: float
    train_time_ms: float
    step_avg_ms: float
    train_bpb: float | None


class ValRow(NamedTuple):
    step: int
    total: int
    val_loss: float
    val_bpb: float
    train_time_ms: float
    step_avg_ms: float


class Seq4096TrainingLog(NamedTuple):
    train_rows: list[TrainRow]
    val_rows: list[ValRow]

    @property
    def latest_step(self) -> int:
        steps = [row.step for row in self.train_rows] + [row.step for row in self.val_rows]
        return max(steps, default=0)

    @property
    def total_steps(self) -> int:
        totals = [row.total for row in self.train_rows] + [row.total for row in self.val_rows]
        return max(totals, default=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--run-path", required=True, help="entity/project/run_id")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--interval-steps", type=int, default=500)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--target-bpb", type=float, default=1.2)
    parser.add_argument("--gate-step", type=int, default=4000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "toricgt-parameter-golf"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "amelie-iska-math"))
    parser.add_argument("--run-id", default="", help="Defaults to the final component of --run-path")
    parser.add_argument("--training-tmux", default="")
    parser.add_argument("--codex-review-hook", default="")
    parser.add_argument("--codex-review-tmux-prefix", default="toricgt_codex_review_seq4096")
    parser.add_argument("--max-checkpoints", type=int, default=0, help="0 means run forever")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--skip-wandb-metrics", action="store_true")
    parser.add_argument("--skip-full-diagnostics", action="store_true")
    parser.add_argument("--wandb-settle-seconds", type=float, default=5.0)
    return parser.parse_args()


def maybe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_seq4096_log(path: Path) -> Seq4096TrainingLog:
    train: dict[int, TrainRow] = {}
    vals: dict[int, ValRow] = {}
    if not path.exists():
        return Seq4096TrainingLog([], [])
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        val = VAL_RE.search(line)
        if val:
            step = int(val.group("step"))
            vals[step] = ValRow(
                step=step,
                total=int(val.group("total")),
                val_loss=float(val.group("loss")),
                val_bpb=float(val.group("bpb")),
                train_time_ms=float(val.group("ms")),
                step_avg_ms=float(val.group("avg")),
            )
            continue
        train_match = TRAIN_RE.search(line)
        if train_match:
            step = int(train_match.group("step"))
            train[step] = TrainRow(
                step=step,
                total=int(train_match.group("total")),
                train_loss=float(train_match.group("loss")),
                train_time_ms=float(train_match.group("ms")),
                step_avg_ms=float(train_match.group("avg")),
                train_bpb=maybe_float(train_match.group("bpb")),
            )
    return Seq4096TrainingLog(
        train_rows=[train[key] for key in sorted(train)],
        val_rows=[vals[key] for key in sorted(vals)],
    )


def training_dataframe(parsed: Seq4096TrainingLog, target_bpb: float = 1.2) -> pd.DataFrame:
    steps = sorted({row.step for row in parsed.train_rows} | {row.step for row in parsed.val_rows})
    train_by_step = {row.step: row for row in parsed.train_rows}
    val_by_step = {row.step: row for row in parsed.val_rows}
    rows: list[dict[str, float]] = []
    for step in steps:
        train = train_by_step.get(step)
        val = val_by_step.get(step)
        total = parsed.total_steps
        if train is not None:
            total = train.total
        if val is not None:
            total = val.total
        rows.append(
            {
                "step": float(step),
                "total_steps": float(total),
                "train_loss": float(train.train_loss) if train else np.nan,
                "train_bpb": float(train.train_bpb) if train and train.train_bpb is not None else np.nan,
                "val_loss": float(val.val_loss) if val else np.nan,
                "val_bpb": float(val.val_bpb) if val else np.nan,
                "train_time_ms": float((val or train).train_time_ms) if (val or train) else np.nan,
                "step_avg_ms": float((val or train).step_avg_ms) if (val or train) else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "step",
                "total_steps",
                "train_loss",
                "train_bpb",
                "val_loss",
                "val_bpb",
                "target_gap",
                "model_bpb",
            ]
        )
    frame = frame.sort_values("step").reset_index(drop=True)
    frame["train_bpb_filled"] = frame["train_bpb"].ffill()
    frame["val_bpb_filled"] = frame["val_bpb"].ffill()
    frame["model_bpb"] = frame["val_bpb_filled"].combine_first(frame["train_bpb_filled"])
    frame["target_gap"] = frame["model_bpb"] - float(target_bpb)
    frame["generalization_gap"] = frame["val_bpb_filled"] - frame["train_bpb_filled"]
    return frame


def extract_checkpoint_step(path: Path) -> int | None:
    if path.name.startswith("random_order_step_"):
        return None
    match = SEQ4096_CHECKPOINT_RE.search(path.name)
    if not match:
        return None
    return int(match.group("step"))


def find_checkpoint_at_step(checkpoint_dir: Path, step: int) -> Path | None:
    candidates: list[Path] = []
    if not checkpoint_dir.exists():
        return None
    for path in checkpoint_dir.glob("*.pt"):
        if extract_checkpoint_step(path) == step:
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item.stat().st_mtime, item.name))[-1]


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates: list[tuple[int, float, Path]] = []
    if not checkpoint_dir.exists():
        return None
    for path in checkpoint_dir.glob("*.pt"):
        step = extract_checkpoint_step(path)
        if step is None:
            continue
        candidates.append((step, path.stat().st_mtime, path))
    if not candidates:
        return None
    return sorted(candidates)[-1][2]


def finite_xy(frame: pd.DataFrame, y_field: str) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty or y_field not in frame.columns:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    selected = frame[["step", y_field]].replace([np.inf, -np.inf], np.nan).dropna()
    return selected["step"].to_numpy(dtype=float), selected[y_field].to_numpy(dtype=float)


def slope_per_steps(steps: np.ndarray, values: np.ndarray, unit_steps: float = 100.0) -> float:
    if len(values) < 2:
        return float("nan")
    x = steps.astype(float) / float(unit_steps)
    y = values.astype(float)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * (y - y.mean())) / denom)


def estimate_steps_to_target(current_bpb: float | None, slope_per_100: float, target_bpb: float) -> float:
    if current_bpb is None or not math.isfinite(current_bpb):
        return float("nan")
    if not math.isfinite(slope_per_100) or slope_per_100 >= 0:
        return float("nan")
    return max(0.0, (float(current_bpb) - float(target_bpb)) / (-float(slope_per_100)) * 100.0)


def last_finite(frame: pd.DataFrame, field: str) -> float | None:
    if field not in frame.columns:
        return None
    values = pd.to_numeric(frame[field], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def min_finite(frame: pd.DataFrame, field: str) -> float | None:
    if field not in frame.columns:
        return None
    values = pd.to_numeric(frame[field], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    return float(values.min())


def bpb_acceleration_report(
    frame: pd.DataFrame,
    target_bpb: float = 1.2,
    checkpoint_step: int | None = None,
    gate_step: int = 4000,
) -> dict[str, Any]:
    latest_step = int(frame["step"].max()) if not frame.empty else 0
    latest_train = last_finite(frame, "train_bpb")
    latest_val = last_finite(frame, "val_bpb")
    best_val = min_finite(frame, "val_bpb")
    if best_val is None:
        best_val = min_finite(frame, "model_bpb")
    model_bpb = last_finite(frame, "model_bpb")
    gap_source = best_val if best_val is not None else model_bpb
    target_gap = float(gap_source - target_bpb) if gap_source is not None else float("nan")

    train_steps, train_values = finite_xy(frame, "train_bpb")
    train_slope = slope_per_steps(train_steps, train_values, unit_steps=100.0)
    recent_train_slope = train_slope
    if len(train_values) >= 3:
        recent_n = min(max(3, len(train_values) // 2), len(train_values))
        recent_train_slope = slope_per_steps(train_steps[-recent_n:], train_values[-recent_n:], unit_steps=100.0)

    val_steps, val_values = finite_xy(frame, "val_bpb")
    val_slope = slope_per_steps(val_steps, val_values, unit_steps=100.0)
    recent_val_slope = val_slope
    if len(val_values) >= 3:
        recent_n = min(max(3, len(val_values) // 2), len(val_values))
        recent_val_slope = slope_per_steps(val_steps[-recent_n:], val_values[-recent_n:], unit_steps=100.0)
    latest_val_step = int(val_steps[-1]) if len(val_steps) else latest_step

    state = "insufficient_data"
    if gap_source is not None and gap_source <= target_bpb:
        state = "target_reached"
    elif math.isfinite(target_gap) and target_gap <= 0.20:
        state = "near_target"
    elif math.isfinite(recent_train_slope) and recent_train_slope <= -0.05:
        state = "fast_descent"
    elif math.isfinite(recent_train_slope) and recent_train_slope <= -0.01:
        state = "descending"
    elif latest_train is not None or latest_val is not None:
        state = "plateau_pressure"

    recommendations: list[str] = []
    if state == "target_reached":
        recommendations.append(
            "Save this checkpoint as the OpenAI Parameter-Golf threshold artifact and keep an immutable manifest."
        )
        recommendations.append(
            "Start the post-threshold reasoning phases while preserving the BPB checkpoint for competition export."
        )
    elif state == "near_target":
        recommendations.append(
            "Keep validation and checkpoint gates dense; the run is close enough that every checkpoint can become the target artifact."
        )
        if math.isfinite(recent_val_slope) and recent_val_slope < 0:
            recommendations.append(
                "Continue the current high-throughput schedule while validation BPB keeps descending; treat short train-BPB ETA noise as secondary."
            )
        else:
            recommendations.append(
                "Do not restart on train-BPB noise alone; wait for the next validation checkpoint unless validation stalls."
            )
    elif state == "fast_descent":
        recommendations.append(
            "Do not interrupt the current schedule; train BPB is still dropping quickly."
        )
        recommendations.append(
            "Use the next validation checkpoint to confirm the descent transfers to competition BPB."
        )
    elif state == "descending":
        recommendations.append(
            "Continue training, but monitor whether the recent train BPB slope weakens before validation improves."
        )
    elif state == "plateau_pressure":
        recommendations.append(
            "If this repeats at two checkpoints, try a BPB acceleration intervention: higher tied-embedding LR, shorter warmdown, or a brief batch-token increase."
        )
        recommendations.append(
            "Check W&B for train/val divergence before changing learning rate."
        )
    else:
        recommendations.append("Wait for at least one train BPB and one validation BPB observation.")

    projected_steps_to_target_from_train = estimate_steps_to_target(
        latest_train, recent_train_slope, target_bpb
    )
    val_projection_bpb = best_val if best_val is not None else latest_val
    projected_steps_to_target_from_val = estimate_steps_to_target(
        val_projection_bpb, recent_val_slope, target_bpb
    )
    projected_target_step_from_val = (
        float(latest_val_step) + projected_steps_to_target_from_val
        if math.isfinite(projected_steps_to_target_from_val)
        else float("nan")
    )
    projection_source = "validation" if math.isfinite(projected_steps_to_target_from_val) else "train"
    if not math.isfinite(projected_steps_to_target_from_val) and not math.isfinite(projected_steps_to_target_from_train):
        projection_source = "unavailable"

    remaining_steps_to_gate = max(0.0, float(gate_step) - float(latest_val_step))
    gate_gap_source = latest_val if latest_val is not None else gap_source
    gate_target_gap = (
        max(0.0, float(gate_gap_source) - float(target_bpb))
        if gate_gap_source is not None and math.isfinite(float(gate_gap_source))
        else float("nan")
    )
    required_val_velocity = (
        gate_target_gap / max(1.0, remaining_steps_to_gate) * 100.0
        if math.isfinite(gate_target_gap)
        else float("nan")
    )
    recent_val_velocity = (
        max(0.0, -float(recent_val_slope))
        if math.isfinite(recent_val_slope)
        else float("nan")
    )
    val_velocity_shortfall = (
        max(0.0, required_val_velocity - (recent_val_velocity if math.isfinite(recent_val_velocity) else 0.0))
        if math.isfinite(required_val_velocity)
        else float("nan")
    )
    velocity_shortfall_pressure = (
        max(0.0, min(1.0, val_velocity_shortfall / 0.005))
        if math.isfinite(val_velocity_shortfall)
        else float("nan")
    )
    projected_gate_miss = (
        not math.isfinite(projected_target_step_from_val)
        or projected_target_step_from_val > float(gate_step)
    )
    near_gate_window = latest_val_step >= int(gate_step) - 1000
    material_velocity_shortfall = (
        math.isfinite(val_velocity_shortfall)
        and (val_velocity_shortfall >= 0.0005 or velocity_shortfall_pressure >= 0.10)
    )
    off_track_unreachable = (
        state != "target_reached"
        and gap_source is not None
        and gap_source > float(target_bpb)
        and remaining_steps_to_gate > 0.0
        and near_gate_window
        and material_velocity_shortfall
        and projected_gate_miss
    )
    if off_track_unreachable:
        state = "off_track_unreachable"
        recommendations = [
            "Treat this near-target run as unreachable by the 4K gate; validation BPB velocity is below the required gate velocity.",
            "Trigger an advanced metric intervention now: use structural recapture, toric/Slepian/topology pressure, and failed-trajectory analogues to pick the next restart.",
            "Keep the primary competition run BPB-clean while using GraphCG/Koszul/BGG/topology diagnostics as controller inputs or sidecar transfer signals.",
        ]

    return {
        "state": state,
        "latest_step": latest_step,
        "checkpoint_step": int(checkpoint_step) if checkpoint_step is not None else None,
        "gate_step": int(gate_step),
        "latest_val_step": latest_val_step,
        "remaining_steps_to_gate": remaining_steps_to_gate,
        "latest_train_bpb": latest_train,
        "latest_val_bpb": latest_val,
        "best_val_bpb": best_val,
        "target_bpb": float(target_bpb),
        "target_gap": target_gap,
        "train_bpb_slope_per_100_steps": train_slope,
        "train_bpb_recent_slope_per_100_steps": recent_train_slope,
        "val_bpb_slope_per_100_steps": val_slope,
        "val_bpb_recent_slope_per_100_steps": recent_val_slope,
        "projected_steps_to_target_from_train": projected_steps_to_target_from_train,
        "projected_steps_to_target_from_val": projected_steps_to_target_from_val,
        "projected_target_step_from_val": projected_target_step_from_val,
        "projection_source": projection_source,
        "required_val_velocity_to_gate_per_100_steps": required_val_velocity,
        "val_bpb_velocity_recent_per_100_steps": recent_val_velocity,
        "val_velocity_shortfall_to_gate_per_100_steps": val_velocity_shortfall,
        "bpb_velocity_shortfall_pressure": velocity_shortfall_pressure,
        "recommendations": recommendations,
    }


def validation_transfer_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Pair validation checkpoints with the latest train BPB at the same step."""

    columns = [
        "step",
        "train_bpb_at_val",
        "val_bpb",
        "generalization_gap",
        "train_drop_bpb",
        "val_drop_bpb",
        "transfer_efficiency",
        "transfer_pressure",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    work = frame.copy()
    work["train_bpb_filled"] = pd.to_numeric(work.get("train_bpb_filled"), errors="coerce").ffill()
    work["val_bpb"] = pd.to_numeric(work.get("val_bpb"), errors="coerce")
    paired = work.dropna(subset=["val_bpb", "train_bpb_filled"]).copy()
    if paired.empty:
        return pd.DataFrame(columns=columns)
    paired["train_bpb_at_val"] = paired["train_bpb_filled"]
    paired["generalization_gap"] = paired["val_bpb"] - paired["train_bpb_at_val"]
    paired["train_drop_bpb"] = paired["train_bpb_at_val"].shift(1) - paired["train_bpb_at_val"]
    paired["val_drop_bpb"] = paired["val_bpb"].shift(1) - paired["val_bpb"]
    train_drop = paired["train_drop_bpb"].to_numpy(dtype=float)
    val_drop = paired["val_drop_bpb"].to_numpy(dtype=float)
    efficiency = np.full_like(train_drop, np.nan, dtype=float)
    valid = np.isfinite(train_drop) & np.isfinite(val_drop) & (train_drop > 1e-9)
    efficiency[valid] = val_drop[valid] / train_drop[valid]
    paired["transfer_efficiency"] = efficiency
    paired["transfer_pressure"] = np.maximum(0.0, paired["train_drop_bpb"] - paired["val_drop_bpb"])
    return paired[columns].reset_index(drop=True)


def transfer_efficiency_report(frame: pd.DataFrame, target_bpb: float = 1.2) -> dict[str, Any]:
    """Measure how much train BPB descent transfers to validation BPB."""

    paired = validation_transfer_frame(frame)
    if paired.empty:
        return {
            "transfer_regime": "insufficient_data",
            "transfer_pairs": 0,
            "latest_generalization_gap_bpb": float("nan"),
            "generalization_gap_slope_per_100_steps": float("nan"),
            "cumulative_train_drop_bpb": float("nan"),
            "cumulative_val_drop_bpb": float("nan"),
            "cumulative_transfer_efficiency": float("nan"),
            "recent_train_drop_bpb": float("nan"),
            "recent_val_drop_bpb": float("nan"),
            "recent_transfer_efficiency": float("nan"),
            "validation_transfer_pressure": float("nan"),
            "transfer_recommendations": [
                "Wait for paired train/validation BPB checkpoints before judging transfer."
            ],
        }

    steps = paired["step"].to_numpy(dtype=float)
    gaps = paired["generalization_gap"].to_numpy(dtype=float)
    latest_gap = float(gaps[-1]) if len(gaps) else float("nan")
    gap_slope = slope_per_steps(steps, gaps, unit_steps=100.0) if len(gaps) >= 2 else float("nan")
    latest_val = float(paired["val_bpb"].iloc[-1])
    latest_train = float(paired["train_bpb_at_val"].iloc[-1])

    if len(paired) >= 2:
        cumulative_train_drop = float(
            paired["train_bpb_at_val"].iloc[0] - paired["train_bpb_at_val"].iloc[-1]
        )
        cumulative_val_drop = float(paired["val_bpb"].iloc[0] - paired["val_bpb"].iloc[-1])
    else:
        cumulative_train_drop = 0.0
        cumulative_val_drop = 0.0
    cumulative_efficiency = (
        cumulative_val_drop / cumulative_train_drop
        if cumulative_train_drop > 1e-9
        else float("nan")
    )
    recent_train_drop = (
        float(paired["train_drop_bpb"].dropna().iloc[-1])
        if paired["train_drop_bpb"].notna().any()
        else float("nan")
    )
    recent_val_drop = (
        float(paired["val_drop_bpb"].dropna().iloc[-1])
        if paired["val_drop_bpb"].notna().any()
        else float("nan")
    )
    recent_efficiency = (
        recent_val_drop / recent_train_drop
        if math.isfinite(recent_train_drop) and recent_train_drop > 1e-9
        else float("nan")
    )
    recent_pressure = (
        max(0.0, recent_train_drop - recent_val_drop)
        if math.isfinite(recent_train_drop) and math.isfinite(recent_val_drop)
        else float("nan")
    )

    if latest_val <= target_bpb:
        regime = "target_reached"
    elif len(paired) < 2:
        regime = "single_validation_pair"
    elif math.isfinite(recent_efficiency) and recent_efficiency < 0.35 and latest_gap >= 0.04:
        regime = "train_descent_validation_lag"
    elif math.isfinite(gap_slope) and gap_slope > 0.0 and latest_gap >= 0.03:
        regime = "generalization_gap_widening"
    elif math.isfinite(cumulative_efficiency) and cumulative_efficiency >= 0.55 and latest_val > target_bpb:
        regime = "healthy_transfer"
    else:
        regime = "watch_transfer"

    recommendations: list[str] = []
    if regime == "target_reached":
        recommendations.append("Preserve this checkpoint before adding advanced reasoning or memory losses.")
    elif regime in {"train_descent_validation_lag", "generalization_gap_widening"}:
        recommendations.append(
            "Treat low train BPB as insufficient; prefer batch/regularity and structural-transfer controls that improve validation BPB."
        )
        recommendations.append(
            "Use topology/Slepian/toric/BGG sidecars to choose guarded restarts rather than raising the primary LR blindly."
        )
    elif regime == "healthy_transfer":
        recommendations.append("Keep the current schedule while validation transfer remains healthy.")
    else:
        recommendations.append("Keep transfer checkpoints dense until two or more paired validation intervals are available.")

    return {
        "transfer_regime": regime,
        "transfer_pairs": int(len(paired)),
        "latest_transfer_step": int(steps[-1]),
        "latest_train_bpb_at_validation": latest_train,
        "latest_val_bpb_for_transfer": latest_val,
        "latest_generalization_gap_bpb": latest_gap,
        "generalization_gap_slope_per_100_steps": gap_slope,
        "cumulative_train_drop_bpb": cumulative_train_drop,
        "cumulative_val_drop_bpb": cumulative_val_drop,
        "cumulative_transfer_efficiency": cumulative_efficiency,
        "recent_train_drop_bpb": recent_train_drop,
        "recent_val_drop_bpb": recent_val_drop,
        "recent_transfer_efficiency": recent_efficiency,
        "validation_transfer_pressure": recent_pressure,
        "transfer_recommendations": recommendations,
    }


def gate_velocity_report(frame: pd.DataFrame, target_bpb: float = 1.2, gate_step: int = 4000) -> dict[str, Any]:
    val_steps, val_values = finite_xy(frame, "val_bpb")
    if len(val_values) == 0:
        return {
            "gate_step": int(gate_step),
            "val_velocity_pairs": 0,
            "val_bpb_velocity_recent_per_100_steps": float("nan"),
            "val_bpb_velocity_previous_per_100_steps": float("nan"),
            "val_bpb_velocity_delta_per_100_steps": float("nan"),
            "val_bpb_velocity_decay_per_100_steps": float("nan"),
            "required_val_velocity_to_gate_per_100_steps": float("nan"),
            "val_velocity_shortfall_to_gate_per_100_steps": float("nan"),
            "bpb_velocity_shortfall_pressure": float("nan"),
        }
    latest_step = float(val_steps[-1])
    latest_val = float(val_values[-1])
    remaining_steps = max(1.0, float(gate_step) - latest_step)
    target_gap = max(0.0, latest_val - float(target_bpb))
    required_velocity = target_gap / remaining_steps * 100.0
    recent_velocity = float("nan")
    previous_velocity = float("nan")
    velocity_delta = float("nan")
    if len(val_values) >= 2:
        recent_velocity = max(
            0.0,
            (float(val_values[-2]) - latest_val) / max(1.0, latest_step - float(val_steps[-2])) * 100.0,
        )
    if len(val_values) >= 3:
        previous_velocity = max(
            0.0,
            (float(val_values[-3]) - float(val_values[-2]))
            / max(1.0, float(val_steps[-2]) - float(val_steps[-3]))
            * 100.0,
        )
    if math.isfinite(recent_velocity) and math.isfinite(previous_velocity):
        velocity_delta = recent_velocity - previous_velocity
    effective_recent = recent_velocity if math.isfinite(recent_velocity) else 0.0
    shortfall = max(0.0, required_velocity - effective_recent)
    decay = max(0.0, previous_velocity - effective_recent) if math.isfinite(previous_velocity) else 0.0
    pressure = max(0.0, min(1.0, shortfall / 0.005))
    return {
        "gate_step": int(gate_step),
        "val_velocity_pairs": int(len(val_values)),
        "val_bpb_velocity_recent_per_100_steps": recent_velocity,
        "val_bpb_velocity_previous_per_100_steps": previous_velocity,
        "val_bpb_velocity_delta_per_100_steps": velocity_delta,
        "val_bpb_velocity_decay_per_100_steps": decay,
        "required_val_velocity_to_gate_per_100_steps": required_velocity,
        "val_velocity_shortfall_to_gate_per_100_steps": shortfall,
        "bpb_velocity_shortfall_pressure": pressure,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def finite_payload_value(payload: dict[str, Any], *keys: str, default: float = float("nan")) -> float:
    for key in keys:
        if key not in payload:
            continue
        try:
            value = float(payload.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return default


def bounded_pressure(value: float, scale: float, *, invert: bool = False, floor: float = 0.0) -> float:
    if not math.isfinite(value) or scale <= 0.0:
        return 0.0
    measured = max(0.0, floor - value) if invert else max(0.0, value - floor)
    return max(0.0, min(1.0, measured / float(scale)))


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


def structural_recapture_report(diagnostic_payload: dict[str, Any]) -> dict[str, Any]:
    """Score whether advanced diagnostics should shape the next BPB restart."""

    if not diagnostic_payload:
        return {
            "structural_recapture_score": 0.0,
            "structural_recapture_band": "missing",
            "structural_control_prior": "wait_for_diagnostics",
            "structural_recapture_components": {},
        }
    bpb_pressure = finite_payload_value(
        diagnostic_payload,
        "diagnostics/latest/bpb_intervention_pressure",
        default=0.0,
    )
    topology_loss = finite_payload_value(
        diagnostic_payload,
        "diagnostics/latest/topology_loss",
        "topology/topology_loss",
    )
    directed_topology_loss = finite_payload_value(
        diagnostic_payload,
        "diagnostics/latest/directed_topology_loss",
        "topology/directed_topology_loss",
    )
    slepian_leakage = finite_payload_value(
        diagnostic_payload,
        "diagnostics/latest/slepian_leakage",
        "diagnostics/latest/pollak_prolate_slepian_leakage",
        "toric/slepian_leakage",
    )
    toric_margin = finite_payload_value(
        diagnostic_payload,
        "diagnostics/latest/toric_active_face_margin",
        "toric/active_face_margin",
    )
    toric_bend = finite_payload_value(
        diagnostic_payload,
        "diagnostics/latest/toric_shadow_mean_bend",
        "toric/shadow_mean_bend",
    )
    bgg_standard_leakage = finite_payload_value(
        diagnostic_payload,
        "diagnostics/latest/bgg_standard_leakage",
        "bgg_category_o/standard_leakage",
        default=0.0,
    )
    bgg_d2_residual = finite_payload_value(
        diagnostic_payload,
        "diagnostics/latest/bgg_d2_residual",
        "bgg_category_o/d2_residual",
        default=0.0,
    )
    complexity_ncd = finite_payload_value(
        diagnostic_payload,
        "diagnostics/latest/complexity_recent_full_log_ncd_lzma",
        "complexity/recent_full_log_ncd_lzma",
        default=0.0,
    )
    tropical_plateau = finite_payload_value(
        diagnostic_payload,
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
        finite_payload_value(diagnostic_payload, key, default=0.0) >= 0.5
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
        band = "bpb_curve_only"
        control_prior = "hold_primary_bpb_clean_until_structural_payload_available"
    elif score >= 0.75:
        band = "high"
        control_prior = f"restart_guarded_damp_lr_use_{dominant_family}_sidecar_transfer"
    elif score >= 0.50:
        band = "guarded"
        if dominant_family == "topology_directed":
            control_prior = "guard_directed_topology_with_warmup_and_bigram_lr_damping"
        elif dominant_family == "toric_slepian":
            control_prior = "guard_toric_slepian_concentration_before_velocity_push"
        else:
            control_prior = "restart_guarded_warmup_bigram_lr_damping"
    elif score >= 0.30:
        band = "watch"
        control_prior = "monitor_do_not_add_heavy_structural_losses"
    else:
        band = "low"
        control_prior = "prefer_bpb_velocity_or_validation_gap_controls"
    return {
        "structural_recapture_score": score,
        "structural_recapture_band": band,
        "structural_control_prior": control_prior,
        "structural_recapture_components": components,
        "structural_family_pressures": family_pressures,
        "dominant_structural_pressure_family": dominant_family,
        "dominant_structural_pressure_value": dominant_pressure,
        "structural_raw_signals": {
            "bpb_intervention_pressure": bpb_pressure,
            "topology_loss": topology_loss,
            "directed_topology_loss": directed_topology_loss,
            "slepian_leakage": slepian_leakage,
            "toric_active_face_margin": toric_margin,
            "toric_shadow_mean_bend": toric_bend,
            "bgg_standard_leakage": bgg_standard_leakage,
            "bgg_d2_residual": bgg_d2_residual,
            "complexity_recent_full_log_ncd_lzma": complexity_ncd,
            "tropical_bpb_plateau_pressure": tropical_plateau,
        },
    }


def plot_bpb_timeseries(frame: pd.DataFrame, out: Path, target_bpb: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    train_steps, train_values = finite_xy(frame, "train_bpb")
    val_steps, val_values = finite_xy(frame, "val_bpb")
    if len(train_values):
        ax.plot(train_steps, train_values, color="#2563eb", linewidth=1.8, marker="o", markersize=3, label="train BPB")
    if len(val_values):
        ax.plot(val_steps, val_values, color="#dc2626", linewidth=2.2, marker="s", markersize=4, label="validation BPB")
        best_idx = int(np.argmin(val_values))
        ax.scatter([val_steps[best_idx]], [val_values[best_idx]], color="#16a34a", s=70, zorder=5, label="best validation")
    ax.axhline(target_bpb, color="#111827", linestyle="--", linewidth=1.2, label=f"target {target_bpb:.3f}")
    ax.set_xlabel("trainer step")
    ax.set_ylabel("bits per byte")
    ax.set_title("Seq4096 BPB Descent Toward OpenAI Parameter-Golf Target")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_bpb_velocity(frame: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    plotted = False
    for field, color, label in (("train_bpb", "#2563eb", "train BPB drop/100 steps"), ("val_bpb", "#dc2626", "validation BPB drop/100 steps")):
        steps, values = finite_xy(frame, field)
        if len(values) < 2:
            continue
        delta_steps = np.diff(steps)
        velocity = -np.diff(values) / np.maximum(delta_steps, 1.0) * 100.0
        ax.plot(steps[1:], velocity, color=color, marker="o", linewidth=1.6, label=label)
        plotted = True
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.set_xlabel("trainer step")
    ax.set_ylabel("positive means BPB is falling")
    ax.set_title("BPB Descent Velocity")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(loc="best")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def train_drop_segments(frame: pd.DataFrame) -> pd.DataFrame:
    steps, values = finite_xy(frame, "train_bpb")
    if len(values) < 2:
        return pd.DataFrame(columns=["start_step", "end_step", "drop", "drop_per_100", "end_bpb"])
    rows = []
    for start_step, end_step, start_bpb, end_bpb in zip(steps[:-1], steps[1:], values[:-1], values[1:]):
        span = max(float(end_step - start_step), 1.0)
        drop = float(start_bpb - end_bpb)
        rows.append(
            {
                "start_step": float(start_step),
                "end_step": float(end_step),
                "drop": drop,
                "drop_per_100": drop / span * 100.0,
                "end_bpb": float(end_bpb),
            }
        )
    return pd.DataFrame(rows)


def rolling_eta_to_target(frame: pd.DataFrame, target_bpb: float, field: str = "train_bpb") -> pd.DataFrame:
    steps, values = finite_xy(frame, field)
    rows = []
    if len(values) < 2:
        return pd.DataFrame(columns=["step", "eta_steps", "recent_drop_per_100"])
    for idx in range(1, len(values)):
        start = max(0, idx - 3)
        recent_steps = steps[start : idx + 1]
        recent_values = values[start : idx + 1]
        slope = slope_per_steps(recent_steps, recent_values, unit_steps=100.0)
        drop_per_100 = -slope if math.isfinite(slope) else float("nan")
        if math.isfinite(drop_per_100) and drop_per_100 > 0:
            eta = max(0.0, (float(values[idx]) - float(target_bpb)) / drop_per_100 * 100.0)
        else:
            eta = float("nan")
        rows.append({"step": float(steps[idx]), "eta_steps": eta, "recent_drop_per_100": drop_per_100})
    return pd.DataFrame(rows)


def plot_bpb_target_zone(frame: pd.DataFrame, out: Path, target_bpb: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    train_steps, train_values = finite_xy(frame, "train_bpb")
    val_steps, val_values = finite_xy(frame, "val_bpb")
    if len(train_values):
        ax.plot(train_steps, train_values, color="#2563eb", marker="o", linewidth=1.6, label="train BPB")
    if len(val_values):
        ax.plot(val_steps, val_values, color="#dc2626", marker="s", linewidth=2.0, label="validation BPB")
    ax.axhline(target_bpb, color="#111827", linestyle="--", linewidth=1.1, label=f"target {target_bpb:.3f}")
    values = np.concatenate([arr for arr in (train_values, val_values) if len(arr)]) if (len(train_values) or len(val_values)) else np.asarray([target_bpb])
    recent_values = values[np.isfinite(values)]
    if recent_values.size:
        low = min(float(target_bpb) - 0.05, float(np.nanmin(recent_values[-6:])) - 0.08)
        high = max(float(target_bpb) + 0.35, float(np.nanmax(recent_values[-8:])) + 0.08)
        ax.set_ylim(max(0.8, low), min(max(high, target_bpb + 0.25), 2.4))
    ax.set_xlabel("trainer step")
    ax.set_ylabel("bits per byte")
    ax.set_title("BPB Target Zone")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_bpb_drop_waterfall(frame: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    segments = train_drop_segments(frame)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    if not segments.empty:
        labels = [f"{int(row.start_step)}-{int(row.end_step)}" for row in segments.itertuples()]
        colors = ["#16a34a" if value >= 0 else "#dc2626" for value in segments["drop_per_100"]]
        x = np.arange(len(segments))
        ax.bar(x, segments["drop_per_100"], color=colors)
        ax.set_xticks(x, labels, rotation=45, ha="right")
    else:
        ax.text(0.5, 0.5, "need at least two train BPB points", ha="center", va="center")
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.set_ylabel("BPB drop per 100 steps")
    ax.set_title("BPB Drop Waterfall")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_bpb_eta_to_target(frame: pd.DataFrame, out: Path, target_bpb: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    train_eta = rolling_eta_to_target(frame, target_bpb, field="train_bpb")
    val_eta = rolling_eta_to_target(frame, target_bpb, field="val_bpb")
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    plotted = False
    if not train_eta.empty:
        ax.plot(train_eta["step"], train_eta["eta_steps"], color="#7c3aed", marker="o", linewidth=1.7, label="train-slope ETA")
        ax.fill_between(train_eta["step"], train_eta["eta_steps"], color="#c4b5fd", alpha=0.20)
        plotted = True
    if not val_eta.empty:
        ax.plot(val_eta["step"], val_eta["eta_steps"], color="#dc2626", marker="s", linewidth=1.9, label="validation-slope ETA")
        plotted = True
    if plotted:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "need recent negative BPB slope", ha="center", va="center")
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.set_xlabel("trainer step")
    ax.set_ylabel("projected steps to BPB target")
    ax.set_title("Rolling ETA To BPB Target")
    ax.grid(alpha=0.25)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_bpb_rockfall_dashboard(frame: pd.DataFrame, out: Path, target_bpb: float, report: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    train_steps, train_values = finite_xy(frame, "train_bpb")
    val_steps, val_values = finite_xy(frame, "val_bpb")
    if len(train_values):
        ax.plot(train_steps, train_values, color="#2563eb", marker="o", linewidth=1.5, label="train")
    if len(val_values):
        ax.plot(val_steps, val_values, color="#dc2626", marker="s", linewidth=1.8, label="validation")
    ax.axhline(target_bpb, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_title("Target-zone BPB")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    if len(train_values) or len(val_values):
        vals = np.concatenate([arr for arr in (train_values, val_values) if len(arr)])
        ax.set_ylim(max(0.8, min(target_bpb - 0.05, float(np.nanmin(vals[-8:])) - 0.08)), min(2.4, max(target_bpb + 0.35, float(np.nanmax(vals[-8:])) + 0.08)))

    ax = axes[0, 1]
    segments = train_drop_segments(frame)
    if not segments.empty:
        x = np.arange(len(segments))
        ax.bar(x, segments["drop_per_100"], color=["#16a34a" if value >= 0 else "#dc2626" for value in segments["drop_per_100"]])
        ax.set_xticks(x, [f"{int(row.start_step)}-{int(row.end_step)}" for row in segments.itertuples()], rotation=45, ha="right", fontsize=8)
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.set_title("Drop waterfall")
    ax.set_ylabel("drop/100 steps")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    train_eta = rolling_eta_to_target(frame, target_bpb, field="train_bpb")
    val_eta = rolling_eta_to_target(frame, target_bpb, field="val_bpb")
    if not train_eta.empty:
        ax.plot(train_eta["step"], train_eta["eta_steps"], color="#7c3aed", marker="o", linewidth=1.5, label="train")
    if not val_eta.empty:
        ax.plot(val_eta["step"], val_eta["eta_steps"], color="#dc2626", marker="s", linewidth=1.6, label="validation")
    if not train_eta.empty or not val_eta.empty:
        ax.legend(loc="best", fontsize=8)
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.set_title("ETA from BPB slopes")
    ax.set_xlabel("trainer step")
    ax.set_ylabel("steps")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.axis("off")
    summary_lines = [
        f"state: {report.get('state')}",
        f"best val BPB: {report.get('best_val_bpb')}",
        f"target gap: {report.get('target_gap')}",
        f"recent train slope/100: {report.get('train_bpb_recent_slope_per_100_steps')}",
        f"recent val slope/100: {report.get('val_bpb_recent_slope_per_100_steps')}",
        f"train-projected steps: {report.get('projected_steps_to_target_from_train')}",
        f"val-projected steps: {report.get('projected_steps_to_target_from_val')}",
        f"val-projected target step: {report.get('projected_target_step_from_val')}",
        f"projection source: {report.get('projection_source')}",
    ]
    ax.text(0.02, 0.95, "\n".join(summary_lines), va="top", ha="left", fontsize=11)
    ax.set_title("Intervention readout")
    fig.suptitle("BPB Rockfall Dashboard", fontsize=15)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def barycentric_to_xy(weights: np.ndarray) -> np.ndarray:
    vertices = np.asarray([[0.05, 0.05], [0.95, 0.05], [0.50, 0.90]], dtype=float)
    return weights @ vertices


def plot_bpb_simplex(frame: pd.DataFrame, out: Path, target_bpb: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame = pd.DataFrame({"step": [0.0], "model_bpb": [target_bpb + 1.0]})
    work = frame.copy()
    work["model_bpb"] = pd.to_numeric(work.get("model_bpb"), errors="coerce").ffill().bfill()
    work["train_bpb_filled"] = pd.to_numeric(work.get("train_bpb_filled"), errors="coerce").ffill().bfill()
    work["val_bpb_filled"] = pd.to_numeric(work.get("val_bpb_filled"), errors="coerce").ffill().bfill()
    steps = pd.to_numeric(work["step"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    model = pd.to_numeric(work["model_bpb"], errors="coerce").fillna(target_bpb + 1.0).to_numpy(dtype=float)
    train = pd.to_numeric(work["train_bpb_filled"], errors="coerce").fillna(pd.Series(model, index=work.index)).to_numpy(dtype=float)
    val = pd.to_numeric(work["val_bpb_filled"], errors="coerce").fillna(pd.Series(model, index=work.index)).to_numpy(dtype=float)
    target_gap = np.maximum(model - target_bpb, 0.0)
    generalization_gap = np.maximum(val - train, 0.0)
    velocity = np.zeros_like(model)
    if len(model) > 1:
        velocity[1:] = np.maximum(-np.diff(model) / np.maximum(np.diff(steps), 1.0) * 100.0, 0.0)
    components = np.stack([target_gap, generalization_gap, velocity + 1e-6], axis=1)
    denom = components.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    weights = components / denom
    xy = barycentric_to_xy(weights)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    triangle = np.asarray([[0.05, 0.05], [0.95, 0.05], [0.50, 0.90], [0.05, 0.05]])
    ax.plot(triangle[:, 0], triangle[:, 1], color="#111827", linewidth=1.2)
    ax.text(0.02, 0.00, "target gap", ha="left", va="top", fontsize=10)
    ax.text(0.98, 0.00, "generalization gap", ha="right", va="top", fontsize=10)
    ax.text(0.50, 0.94, "descent velocity", ha="center", va="bottom", fontsize=10)
    if len(xy) > 1:
        ax.plot(xy[:, 0], xy[:, 1], color="#64748b", linewidth=1.0, alpha=0.8)
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=steps, cmap="viridis", s=38, edgecolor="#111827", linewidth=0.25)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="trainer step")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("BPB Descent Simplex")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_bpb_phase_plane(frame: pd.DataFrame, out: Path, target_bpb: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    work = frame.copy()
    work["train_bpb_filled"] = pd.to_numeric(work.get("train_bpb_filled"), errors="coerce").ffill()
    work["val_bpb_filled"] = pd.to_numeric(work.get("val_bpb_filled"), errors="coerce").ffill()
    work = work.dropna(subset=["train_bpb_filled", "val_bpb_filled"])
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if not work.empty:
        steps = work["step"].to_numpy(dtype=float)
        scatter = ax.scatter(
            work["train_bpb_filled"],
            work["val_bpb_filled"],
            c=steps,
            cmap="plasma",
            s=44,
            edgecolor="#111827",
            linewidth=0.25,
        )
        ax.plot(work["train_bpb_filled"], work["val_bpb_filled"], color="#64748b", linewidth=1.0, alpha=0.8)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="trainer step")
    ax.axhline(target_bpb, color="#111827", linestyle="--", linewidth=1.0)
    ax.axvline(target_bpb, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_xlabel("train BPB")
    ax.set_ylabel("validation BPB")
    ax.set_title("Train vs Validation BPB Phase Plane")
    ax.grid(alpha=0.25)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_bpb_transfer_efficiency(frame: pd.DataFrame, out: Path, report: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    paired = validation_transfer_frame(frame)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    ax = axes[0]
    intervals = paired.dropna(subset=["train_drop_bpb", "val_drop_bpb"]) if not paired.empty else paired
    if not intervals.empty:
        x = np.arange(len(intervals))
        width = 0.36
        labels = [str(int(row.step)) for row in intervals.itertuples()]
        ax.bar(x - width / 2, intervals["train_drop_bpb"], width=width, color="#2563eb", label="train BPB drop")
        ax.bar(x + width / 2, intervals["val_drop_bpb"], width=width, color="#dc2626", label="validation BPB drop")
        ax.set_xticks(x, labels)
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "need at least two paired validation checkpoints", ha="center", va="center")
        ax.set_xticks([])
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.set_ylabel("BPB drop over interval")
    ax.set_title("Train-To-Validation BPB Transfer")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    if not paired.empty:
        steps = paired["step"].to_numpy(dtype=float)
        ax.plot(steps, paired["generalization_gap"], color="#7c3aed", marker="o", linewidth=1.8, label="generalization gap")
        if "transfer_efficiency" in paired:
            eff = paired[["step", "transfer_efficiency"]].replace([np.inf, -np.inf], np.nan).dropna()
            if not eff.empty:
                ax2 = ax.twinx()
                ax2.plot(
                    eff["step"],
                    eff["transfer_efficiency"],
                    color="#16a34a",
                    marker="s",
                    linewidth=1.6,
                    label="transfer efficiency",
                )
                ax2.axhline(0.55, color="#16a34a", linestyle="--", linewidth=0.9, alpha=0.7)
                ax2.set_ylabel("validation drop / train drop")
                ax2.set_ylim(min(-0.2, float(eff["transfer_efficiency"].min()) - 0.1), max(1.2, float(eff["transfer_efficiency"].max()) + 0.1))
                ax2.legend(loc="upper right")
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "transfer report unavailable", ha="center", va="center")
        ax.set_xticks([])
    ax.axhline(0.04, color="#dc2626", linestyle="--", linewidth=0.9, alpha=0.75)
    ax.set_xlabel("validation checkpoint step")
    ax.set_ylabel("validation BPB minus train BPB")
    ax.set_title(f"Transfer Regime: {report.get('transfer_regime')}")
    ax.grid(alpha=0.25)

    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_bpb_gate_velocity_requirement(report: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    recent = report.get("val_bpb_velocity_recent_per_100_steps")
    required = report.get("required_val_velocity_to_gate_per_100_steps")
    shortfall = report.get("val_velocity_shortfall_to_gate_per_100_steps")
    pressure = report.get("bpb_velocity_shortfall_pressure")
    values = [
        float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else 0.0
        for value in (recent, required, shortfall)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax = axes[0]
    labels = ["recent val velocity", "required to gate", "shortfall"]
    colors = ["#16a34a", "#2563eb", "#dc2626"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("BPB drop per 100 steps")
    ax.set_title("Gate Velocity Requirement")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    ax.axis("off")
    lines = [
        f"gate step: {report.get('gate_step')}",
        f"validation velocity pairs: {report.get('val_velocity_pairs')}",
        f"recent velocity/100: {recent}",
        f"required velocity/100: {required}",
        f"shortfall/100: {shortfall}",
        f"shortfall pressure: {pressure}",
        "",
        "control reading:",
        "- shortfall near zero: hold or damp",
        "- shortfall with low topology pressure: velocity recapture",
        "- shortfall with high directed topology: warmup before more LR",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=10, wrap=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_diagnostic_proxy_geometry(payload: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "topology/topology_loss",
        "topology/affine_toric/persistence_loss",
        "toric/shadow_fan_cell_entropy",
        "toric/shadow_mean_bend",
        "toric/slepian_mode_entropy",
        "bgg_category_o/d2_residual",
        "tropical/bpb_recent_slope",
        "tropical/bpb_target_gap",
        "complexity/recent_full_log_ncd_lzma",
    ]
    values: list[tuple[str, float]] = []
    for key in preferred:
        value = payload.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append((key, numeric))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * max(1, len(values)))), constrained_layout=True)
    if values:
        labels = [key for key, _ in values]
        xs = [value for _, value in values]
        colors = ["#2563eb" if value >= 0 else "#dc2626" for value in xs]
        y = np.arange(len(values))
        ax.barh(y, xs, color=colors)
        ax.set_yticks(y, labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.0, color="#111827", linewidth=0.8)
    else:
        ax.text(0.5, 0.5, "diagnostic payload not available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("ToricGT FineWeb Curve Proxy Geometry")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_structural_recapture_map(report: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    components = report.get("structural_recapture_components") or {}
    components = {
        str(key): float(value)
        for key, value in components.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    score = float(report.get("structural_recapture_score") or 0.0)
    band = str(report.get("structural_recapture_band") or "unknown")
    prior = str(report.get("structural_control_prior") or "unknown")
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.45 * max(6, len(components)))), constrained_layout=True)

    ax = axes[0]
    ax.barh([0], [score], color="#2563eb")
    ax.axvline(0.30, color="#f59e0b", linestyle="--", linewidth=1.0, label="watch")
    ax.axvline(0.50, color="#dc2626", linestyle="--", linewidth=1.0, label="guarded")
    ax.axvline(0.75, color="#7c2d12", linestyle="--", linewidth=1.0, label="high")
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks([0], ["score"])
    ax.set_xlabel("bounded structural recapture score")
    ax.set_title(f"Structural Control Band: {band}")
    ax.set_ylim(-0.9, 0.9)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.02, -0.45, prior, ha="left", va="top", fontsize=9, wrap=True)

    ax = axes[1]
    if components:
        ordered = sorted(components.items(), key=lambda item: item[1], reverse=True)
        labels = [item[0] for item in ordered]
        xs = [item[1] for item in ordered]
        colors = ["#dc2626" if value >= 0.10 else "#2563eb" for value in xs]
        y = np.arange(len(ordered))
        ax.barh(y, xs, color=colors)
        ax.set_yticks(y, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("weighted contribution")
    else:
        ax.text(0.5, 0.5, "structural diagnostics unavailable", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Score Components")
    ax.grid(axis="x", alpha=0.25)
    fig.suptitle("BPB Structural Recapture Map", fontsize=15)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_advanced_metric_control_map(report: dict[str, Any], out: Path) -> None:
    """Plot family-level pressures that drive the next optimizer-control choice."""

    out.parent.mkdir(parents=True, exist_ok=True)
    families = report.get("structural_family_pressures") or {}
    families = {
        str(key): float(value)
        for key, value in families.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    dominant = str(report.get("dominant_structural_pressure_family") or "none")
    transfer_regime = str(report.get("transfer_regime") or "unknown")
    recent_efficiency = report.get("recent_transfer_efficiency")
    validation_gap = report.get("latest_generalization_gap_bpb")
    control_prior = str(report.get("structural_control_prior") or "unknown")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    ax = axes[0]
    if families:
        ordered = sorted(families.items(), key=lambda item: item[1], reverse=True)
        labels = [item[0] for item in ordered]
        xs = [item[1] for item in ordered]
        colors = ["#dc2626" if label == dominant else "#2563eb" for label in labels]
        y = np.arange(len(ordered))
        ax.barh(y, xs, color=colors)
        ax.set_yticks(y, labels, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0.20, color="#f59e0b", linestyle="--", linewidth=0.9, label="watch")
        ax.axvline(0.28, color="#dc2626", linestyle="--", linewidth=0.9, label="guard")
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "family pressures unavailable", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Advanced-Metric Family Pressure")
    ax.set_xlabel("weighted pressure")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    ax.axis("off")
    lines = [
        f"dominant family: {dominant}",
        f"transfer regime: {transfer_regime}",
        f"recent transfer efficiency: {recent_efficiency}",
        f"latest generalization gap: {validation_gap}",
        f"gate velocity shortfall: {report.get('val_velocity_shortfall_to_gate_per_100_steps')}",
        f"velocity shortfall pressure: {report.get('bpb_velocity_shortfall_pressure')}",
        f"structural control prior: {control_prior}",
        "",
        "control reading:",
        "- healthy transfer + guarded pressure: BPB velocity nudge",
        "- topology/toric/Slepian pressure: warmup and LR damping",
        "- validation lag: batch/regularity before more LR",
        "- threshold reached: preserve checkpoint before advanced losses",
    ]
    ax.text(0.02, 0.96, "\n".join(lines), va="top", ha="left", fontsize=10, wrap=True)
    ax.set_title("Controller Readout")

    fig.suptitle("Advanced Metric Control Map", fontsize=15)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def parse_analysis_step_dir(path: Path, payload: dict[str, Any]) -> int:
    for key in ("checkpoint_step", "latest_step", "latest_val_step"):
        try:
            return int(payload.get(key))
        except (TypeError, ValueError):
            pass
    match = re.search(r"step-(\d+)", str(path))
    return int(match.group(1)) if match else 0


def load_historical_analysis_statuses(history_root: Path | str | None) -> list[dict[str, Any]]:
    """Load prior periodic analysis statuses grouped by run for evidence scoring."""

    if history_root is None:
        return []
    root = Path(history_root)
    if not root.exists():
        return []
    statuses: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/step-*/analysis_status.json")):
        payload = load_json(path)
        if not payload:
            continue
        run_id = path.parts[-3] if len(path.parts) >= 3 else ""
        item = dict(payload)
        item["_status_path"] = str(path)
        item["_run_id"] = run_id
        item["_analysis_step"] = parse_analysis_step_dir(path, payload)
        statuses.append(item)
    return statuses


def historical_next_validation_rows(statuses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each analysis status with the next validation status from the same run."""

    by_run: dict[str, list[dict[str, Any]]] = {}
    for status in statuses:
        run_id = str(status.get("_run_id") or "")
        if not run_id:
            continue
        by_run.setdefault(run_id, []).append(status)

    rows: list[dict[str, Any]] = []
    for run_id, run_statuses in by_run.items():
        ordered = sorted(run_statuses, key=lambda item: int(item.get("_analysis_step", 0)))
        for current, nxt in zip(ordered[:-1], ordered[1:]):
            current_val = finite_payload_value(current, "latest_val_bpb", "best_val_bpb")
            next_val = finite_payload_value(nxt, "latest_val_bpb", "best_val_bpb")
            current_step = int(current.get("_analysis_step", 0))
            next_step = int(nxt.get("_analysis_step", 0))
            if (
                not math.isfinite(current_val)
                or not math.isfinite(next_val)
                or next_step <= current_step
            ):
                continue
            family_pressures = current.get("structural_family_pressures") or {}
            if not isinstance(family_pressures, dict):
                family_pressures = {}
            rows.append(
                {
                    "run_id": run_id,
                    "step": current_step,
                    "next_step": next_step,
                    "val_bpb": current_val,
                    "next_val_bpb": next_val,
                    "next_val_drop_bpb": current_val - next_val,
                    "next_val_drop_per_100_steps": (current_val - next_val)
                    / max(1.0, float(next_step - current_step))
                    * 100.0,
                    "bpb_velocity_shortfall_pressure": finite_payload_value(
                        current,
                        "bpb_velocity_shortfall_pressure",
                        default=0.0,
                    ),
                    "dominant_structural_pressure_family": str(
                        current.get("dominant_structural_pressure_family") or "none"
                    ),
                    "structural_family_pressures": {
                        str(key): finite_payload_value(family_pressures, str(key), default=0.0)
                        for key in family_pressures
                    },
                }
            )
    return rows


def evidence_action(
    *,
    observations: int,
    active_count: int,
    active_lift: float,
    pressure_outcome_corr: float,
) -> str:
    if observations < 3 or active_count == 0:
        return "insufficient_history"
    supportive = (
        (math.isfinite(active_lift) and active_lift >= 0.001)
        or (math.isfinite(pressure_outcome_corr) and pressure_outcome_corr >= 0.45)
    )
    adverse = (
        (math.isfinite(active_lift) and active_lift <= -0.001)
        or (math.isfinite(pressure_outcome_corr) and pressure_outcome_corr <= -0.45)
    )
    if supportive and not adverse:
        return "supports_guarded_velocity"
    if adverse and not supportive:
        return "sidecar_or_damp"
    return "monitor_as_context"


def pressure_outcome_correlation(pressures: np.ndarray, outcomes: np.ndarray) -> float:
    finite = np.isfinite(pressures) & np.isfinite(outcomes)
    if int(finite.sum()) < 2:
        return float("nan")
    x = pressures[finite]
    y = outcomes[finite]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def advanced_metric_evidence_report(
    history_root: Path | str | None,
    *,
    current_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score advanced metric families by historical next-validation BPB transfer."""

    statuses = load_historical_analysis_statuses(history_root)
    rows = historical_next_validation_rows(statuses)
    current_report = current_report or {}
    families = sorted(
        {
            str(key)
            for row in rows
            for key in (row.get("structural_family_pressures") or {}).keys()
        }
        | {
            str(key)
            for key in (current_report.get("structural_family_pressures") or {}).keys()
        }
    )
    outcomes = np.asarray([float(row["next_val_drop_bpb"]) for row in rows], dtype=float)
    baseline = float(np.nanmean(outcomes)) if outcomes.size else float("nan")
    family_evidence: dict[str, dict[str, Any]] = {}
    for family in families:
        pressures = np.asarray(
            [
                float((row.get("structural_family_pressures") or {}).get(family, 0.0))
                for row in rows
            ],
            dtype=float,
        )
        if outcomes.size == 0:
            family_evidence[family] = {
                "observations": 0,
                "active_observations": 0,
                "mean_pressure": float("nan"),
                "mean_next_val_drop_bpb": float("nan"),
                "active_mean_next_val_drop_bpb": float("nan"),
                "inactive_mean_next_val_drop_bpb": float("nan"),
                "active_lift_next_val_drop_bpb": float("nan"),
                "pressure_outcome_corr": float("nan"),
                "evidence_action": "insufficient_history",
            }
            continue
        active_threshold = 0.20
        active = np.isfinite(pressures) & (pressures >= active_threshold)
        inactive = np.isfinite(pressures) & ~active
        active_mean = float(np.nanmean(outcomes[active])) if int(active.sum()) else float("nan")
        inactive_mean = float(np.nanmean(outcomes[inactive])) if int(inactive.sum()) else float("nan")
        active_lift = (
            active_mean - inactive_mean
            if math.isfinite(active_mean) and math.isfinite(inactive_mean)
            else float("nan")
        )
        corr = pressure_outcome_correlation(pressures, outcomes)
        action = evidence_action(
            observations=int(outcomes.size),
            active_count=int(active.sum()),
            active_lift=active_lift,
            pressure_outcome_corr=corr,
        )
        weighted_mean = (
            float(np.nansum(pressures * outcomes) / max(float(np.nansum(pressures)), 1e-12))
            if int(np.isfinite(pressures).sum())
            else float("nan")
        )
        family_evidence[family] = {
            "observations": int(outcomes.size),
            "active_observations": int(active.sum()),
            "active_threshold": active_threshold,
            "mean_pressure": float(np.nanmean(pressures)) if pressures.size else float("nan"),
            "mean_next_val_drop_bpb": baseline,
            "pressure_weighted_next_val_drop_bpb": weighted_mean,
            "active_mean_next_val_drop_bpb": active_mean,
            "inactive_mean_next_val_drop_bpb": inactive_mean,
            "active_lift_next_val_drop_bpb": active_lift,
            "pressure_outcome_corr": corr,
            "evidence_action": action,
        }

    current_families = current_report.get("structural_family_pressures") or {}
    if not isinstance(current_families, dict):
        current_families = {}
    current_dominant = str(current_report.get("dominant_structural_pressure_family") or "none")
    current_pressure = finite_payload_value(current_families, current_dominant, default=0.0)
    current_velocity_shortfall_pressure = finite_payload_value(
        current_report,
        "bpb_velocity_shortfall_pressure",
        default=0.0,
    )
    current_action = family_evidence.get(current_dominant, {}).get(
        "evidence_action",
        "insufficient_history",
    )
    if current_action == "supports_guarded_velocity" and current_velocity_shortfall_pressure >= 0.35:
        current_policy = "metric_supported_velocity_push"
    elif current_action == "sidecar_or_damp" and current_pressure >= 0.20:
        current_policy = "metric_pressure_requires_sidecar_or_damping"
    elif current_action == "insufficient_history":
        current_policy = "bpb_velocity_primary_until_history_accumulates"
    else:
        current_policy = "monitor_metric_context_keep_bpb_primary"

    return {
        "history_root": str(history_root or ""),
        "evidence_observations": int(len(rows)),
        "baseline_next_val_drop_bpb": baseline,
        "family_evidence": family_evidence,
        "current_dominant_family": current_dominant,
        "current_dominant_pressure": current_pressure,
        "current_family_evidence_action": current_action,
        "current_velocity_shortfall_pressure": current_velocity_shortfall_pressure,
        "current_evidence_policy": current_policy,
        "rows": rows[-32:],
    }


def plot_advanced_metric_evidence_map(report: dict[str, Any], out: Path) -> None:
    """Plot historical next-validation evidence for advanced metric families."""

    out.parent.mkdir(parents=True, exist_ok=True)
    evidence = report.get("family_evidence") or {}
    evidence = {str(key): value for key, value in evidence.items() if isinstance(value, dict)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    ax = axes[0]
    if evidence:
        ordered = sorted(
            evidence.items(),
            key=lambda item: finite_payload_value(item[1], "active_lift_next_val_drop_bpb", default=-999.0),
        )
        labels = [item[0] for item in ordered]
        lifts = [
            finite_payload_value(item[1], "active_lift_next_val_drop_bpb", default=0.0)
            for item in ordered
        ]
        colors = ["#16a34a" if value >= 0 else "#dc2626" for value in lifts]
        y = np.arange(len(ordered))
        ax.barh(y, lifts, color=colors)
        ax.set_yticks(y, labels, fontsize=9)
        ax.axvline(0.0, color="#111827", linewidth=0.9)
        ax.set_xlabel("active-family lift in next validation BPB drop")
        ax.grid(axis="x", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "need prior paired analyses", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Historical Transfer Evidence")

    ax = axes[1]
    ax.axis("off")
    lines = [
        f"observations: {report.get('evidence_observations')}",
        f"baseline next-val drop: {report.get('baseline_next_val_drop_bpb')}",
        f"current dominant family: {report.get('current_dominant_family')}",
        f"current dominant pressure: {report.get('current_dominant_pressure')}",
        f"current family evidence: {report.get('current_family_evidence_action')}",
        f"velocity shortfall pressure: {report.get('current_velocity_shortfall_pressure')}",
        f"evidence policy: {report.get('current_evidence_policy')}",
        "",
        "reading:",
        "- positive lift: pressure historically preceded validation BPB drops",
        "- negative lift: use as sidecar/damping context",
        "- insufficient history: keep BPB velocity primary",
    ]
    ax.text(0.02, 0.96, "\n".join(lines), va="top", ha="left", fontsize=10, wrap=True)
    ax.set_title("Evidence-Gated Controller Readout")
    fig.suptitle("Advanced Metric Evidence Map", fontsize=15)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def phase_bpb_breakdown_plan(target_bpb: float) -> dict[str, Any]:
    return {
        "target_bpb": float(target_bpb),
        "principle": (
            "Keep the OpenAI FineWeb BPB gate primary until a <= target checkpoint is preserved; "
            "if it stalls, activate structured transfer phases and analyze each phase with its own BPB surface."
        ),
        "phases": {
            "competition_fineweb": {
                "objective": "OpenAI Parameter-Golf FineWeb byte-level BPB <= target.",
                "bpb_metrics": [
                    "openai_parameter_golf/bpb",
                    "fineweb/val_bpb",
                    "val/bpb",
                    "bpb/gap_to_target",
                ],
                "advanced_metrics": [
                    "tropical/bpb_recent_slope",
                    "tropical/bpb_target_gap",
                    "topology/topology_loss",
                    "toric/shadow_fan_cell_entropy",
                ],
                "activation": "Always active before threshold checkpoint preservation.",
            },
            "reasoning_got_tot_cot": {
                "objective": "Train graph-of-thought, tree-of-thought, and chain-of-thought trajectories as explicit reasoning surfaces.",
                "bpb_metrics": [
                    "reasoning/got_bpb",
                    "reasoning/tot_bpb",
                    "reasoning/cot_bpb",
                    "reasoning/heldout_bpb",
                ],
                "advanced_metrics": [
                    "gflownet/action_trace_complexity",
                    "gflownet/trajectory_entropy",
                    "topology/directed_topology_loss",
                    "topology/simplex_closure_loss",
                ],
                "activation": "Use if FineWeb BPB slope stalls above target or after preserving the competition checkpoint.",
            },
            "embedding_gflownet": {
                "objective": "Use embedding-space graph-of-thought GFlowNet training, inference, and test-time scaling.",
                "bpb_metrics": [
                    "gflownet/replay_bpb",
                    "gflownet/branch_bpb",
                    "gflownet/test_time_scaled_bpb",
                ],
                "advanced_metrics": [
                    "gflownet/branch_diversity",
                    "gflownet/flow_matching_loss",
                    "gflownet/reward_calibration",
                    "topology/directed_chain_commutator",
                ],
                "activation": "Use as a BPB-transfer phase when pure FineWeb training flattens.",
            },
            "memory_retrieval": {
                "objective": "Train graph-structured memory read/write/retrieval paths and analyze retrieval-conditioned BPB.",
                "bpb_metrics": [
                    "memory/retrieval_bpb",
                    "memory/write_bpb",
                    "memory/read_bpb",
                    "memory/consolidation_bpb",
                ],
                "advanced_metrics": [
                    "memory/retrieval_accuracy",
                    "memory/link_consistency",
                    "complexity/memory_trace_ncd",
                    "topology/memory_graph_stability",
                ],
                "activation": "Use for long dependency transfer and after adding memory boundary tokens or markup.",
            },
            "analogical_transfer": {
                "objective": "Train analogy pairs and structure-preserving maps to improve compression and reasoning transfer.",
                "bpb_metrics": [
                    "analogy/source_bpb",
                    "analogy/target_bpb",
                    "analogy/transfer_bpb",
                    "analogy/heldout_family_bpb",
                ],
                "advanced_metrics": [
                    "topology/analogical_map_loss",
                    "toric/braid_loss",
                    "bgg_category_o/resolution_consistency",
                    "complexity/analogy_ncd",
                ],
                "activation": "Use when BPB improvements require nonlocal structural transfer rather than more FineWeb exposure.",
            },
            "long_context_tropical": {
                "objective": "Extend beyond 4096 tokens with tropical ring attention for long reasoning and memory trajectories.",
                "bpb_metrics": ["long_context/8192_bpb", "long_context/16384_bpb", "long_context/trajectory_bpb"],
                "advanced_metrics": ["tropical/active_face_entropy", "toric/slepian_concentration", "topology/cycle_rank"],
                "activation": "Use after threshold preservation or as a controlled transfer phase if short-context BPB stalls.",
            },
        },
    }


def write_synopsis(
    output_dir: Path,
    report: dict[str, Any],
    run_path: str,
    checkpoint_path: Path | None,
    diagnostic_payload: dict[str, Any],
) -> None:
    lines = [
        "# Seq4096 Periodic Training Analysis",
        "",
        f"- W&B run: `{run_path}`",
        f"- checkpoint: `{checkpoint_path}`" if checkpoint_path else "- checkpoint: pending",
        f"- state: `{report.get('state')}`",
        f"- latest step: `{report.get('latest_step')}`",
        f"- best validation BPB: `{report.get('best_val_bpb')}`",
        f"- target BPB: `{report.get('target_bpb')}`",
        f"- target gap: `{report.get('target_gap')}`",
        f"- recent train BPB slope per 100 steps: `{report.get('train_bpb_recent_slope_per_100_steps')}`",
        f"- recent validation BPB slope per 100 steps: `{report.get('val_bpb_recent_slope_per_100_steps')}`",
        f"- validation-projected steps to target: `{report.get('projected_steps_to_target_from_val')}`",
        f"- validation-projected target step: `{report.get('projected_target_step_from_val')}`",
        f"- train-to-validation transfer regime: `{report.get('transfer_regime')}`",
        f"- recent transfer efficiency: `{report.get('recent_transfer_efficiency')}`",
        f"- latest generalization gap BPB: `{report.get('latest_generalization_gap_bpb')}`",
        f"- structural recapture score: `{report.get('structural_recapture_score')}`",
        f"- structural recapture band: `{report.get('structural_recapture_band')}`",
        f"- dominant structural pressure family: `{report.get('dominant_structural_pressure_family')}`",
        f"- dominant structural pressure value: `{report.get('dominant_structural_pressure_value')}`",
        f"- structural control prior: `{report.get('structural_control_prior')}`",
        f"- advanced metric evidence policy: `{report.get('current_evidence_policy')}`",
        f"- current family evidence action: `{report.get('current_family_evidence_action')}`",
        "",
        "## Plots",
        "",
        "- `bpb/bpb_descent_timeseries.png`: train and validation BPB against the 1.2 target.",
        "- `bpb/bpb_velocity.png`: positive values mean BPB is falling faster.",
        "- `bpb/bpb_descent_simplex.png`: BPB Descent Simplex over target gap, generalization gap, and descent velocity.",
        "- `bpb/bpb_target_zone.png`: zoomed view of the target band where <1.2 decisions are made.",
        "- `bpb/bpb_drop_waterfall.png`: interval-by-interval BPB drop pressure.",
        "- `bpb/bpb_eta_to_target.png`: rolling projection of steps remaining to the target from train and validation BPB slopes.",
        "- `bpb/bpb_rockfall_dashboard.png`: combined intervention readout for deciding whether to leave the run alone or adjust scalar controls.",
        "- `bpb/bpb_phase_plane.png`: train BPB versus validation BPB trajectory.",
        "- `bpb/bpb_transfer_efficiency.png`: Train-To-Validation BPB Transfer Efficiency, showing whether train BPB gains are becoming validation BPB gains.",
        "- `bpb/bpb_gate_velocity_requirement.png`: Gate Velocity Requirement, comparing recent validation BPB velocity to the velocity needed to cross the 1.2 target by the 4K gate.",
        "- `bpb/diagnostic_proxy_geometry.png`: topology, toric, Slepian, BGG, tropical, and complexity proxy readout.",
        "- `bpb/bpb_structural_recapture_map.png`: BPB Structural Recapture Map showing which advanced metrics should shape the next restart.",
        "- `bpb/advanced_metric_control_map.png`: Advanced Metric Control Map showing family-level pressure used to decide velocity, damping, or transfer stabilization.",
        "- `bpb/advanced_metric_evidence_map.png`: Advanced Metric Evidence Map showing whether each advanced metric family historically predicted the next validation BPB drop.",
        "- `training_adjustment_proposal.json` and `.md`: conservative intervention recommendation combining BPB trajectory, W&B metric statistics, and structural diagnostics.",
        "",
        "## Recommendations",
        "",
    ]
    for item in report.get("recommendations", []):
        lines.append(f"- {item}")
    transfer_items = report.get("transfer_recommendations", [])
    if transfer_items:
        lines.extend(["", "## Transfer Readout", ""])
        for item in transfer_items:
            lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Post-Threshold Reasoning And Memory Phase",
            "",
            "- After the OpenAI Parameter-Golf BPB checkpoint is preserved, start a graph-structured reasoning curriculum with explicit reasoning boundary tokens for step starts, step ends, directed edges, and simplex/cell identifiers.",
            "- If straightforward FineWeb competition training stalls above target, use embedding-space GFlowNet graph-of-thought training, inference, and test-time scaling as a BPB-transfer phase rather than a blind optimizer tweak.",
            "- Break down GoT/ToT/CoT reasoning BPB separately from competition BPB: track `reasoning/got_bpb`, `reasoning/tot_bpb`, `reasoning/cot_bpb`, and held-out reasoning BPB alongside topology, GFlowNet, and complexity metrics.",
            "- Add graph-structured memory examples with memory boundary tokens for memory read, memory write, memory link, and memory consolidation events.",
            "- Analyze memory-retrieval BPB, memory read/write BPB, and retrieval-conditioned held-out BPB so memory improvements are not hidden inside aggregate loss.",
            "- Train analogical reasoning as a separate transfer phase and track analogical-transfer BPB, analogy source/target BPB, and topology/toric/BGG consistency metrics.",
            "- Keep these as existing-token markup for the current tokenizer, or train a clean advanced-phase tokenizer with user-defined symbols if the model shape is intentionally reset.",
            "- Analyze the resulting trajectories as directed noncommutative paths through reasoning-step simplicial complexes and graph memory cells.",
            "- Expand beyond 4096 tokens in later phases with long-context tropical ring attention so multi-step graph-of-thought trajectories and memory paths can be trained and audited in one context.",
            "- See `bpb/phase_bpb_breakdown_plan.json` for the phase-specific BPB surfaces and advanced metrics to activate when the competition gate stalls or after the threshold checkpoint is saved.",
        ]
    )
    if diagnostic_payload:
        lines.extend(
            [
                "",
                "## Diagnostic Payload Highlights",
                "",
            ]
        )
        for key in sorted(diagnostic_payload):
            if any(token in key for token in ("topology", "toric", "slepian", "bgg", "tropical/bpb")):
                value = diagnostic_payload[key]
                if isinstance(value, (int, float)):
                    lines.append(f"- `{key}`: {float(value):.6g}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "SYNOPSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bpb_analysis_artifacts(
    parsed: Seq4096TrainingLog,
    output_dir: Path,
    target_bpb: float = 1.2,
    gate_step: int = 4000,
    checkpoint_path: Path | None = None,
    run_path: str = "",
    diagnostic_payload: dict[str, Any] | None = None,
    history_root: Path | None = None,
) -> dict[str, Any]:
    diagnostic_payload = diagnostic_payload or {}
    output_dir.mkdir(parents=True, exist_ok=True)
    bpb_dir = output_dir / "bpb"
    bpb_dir.mkdir(parents=True, exist_ok=True)
    frame = training_dataframe(parsed, target_bpb=target_bpb)
    checkpoint_step = extract_checkpoint_step(checkpoint_path) if checkpoint_path else None
    report = bpb_acceleration_report(
        frame,
        target_bpb=target_bpb,
        checkpoint_step=checkpoint_step,
        gate_step=gate_step,
    )
    structural_report = structural_recapture_report(diagnostic_payload)
    transfer_report = transfer_efficiency_report(frame, target_bpb=target_bpb)
    velocity_report = gate_velocity_report(frame, target_bpb=target_bpb, gate_step=gate_step)
    report.update(structural_report)
    report.update(transfer_report)
    report.update(velocity_report)
    evidence_report = advanced_metric_evidence_report(history_root, current_report=report)
    report.update(
        {
            "advanced_metric_evidence_policy": evidence_report.get("current_evidence_policy"),
            "current_evidence_policy": evidence_report.get("current_evidence_policy"),
            "current_family_evidence_action": evidence_report.get("current_family_evidence_action"),
            "advanced_metric_evidence_observations": evidence_report.get("evidence_observations"),
            "advanced_metric_evidence_baseline_next_val_drop_bpb": evidence_report.get(
                "baseline_next_val_drop_bpb"
            ),
        }
    )
    report["recommendations"] = list(report.get("recommendations", [])) + list(
        transfer_report.get("transfer_recommendations", [])
    )
    frame.to_csv(bpb_dir / "seq4096_training_log_metrics.csv", index=False)
    write_json(bpb_dir / "bpb_acceleration_report.json", report)
    write_json(bpb_dir / "structural_recapture_report.json", structural_report)
    write_json(bpb_dir / "bpb_transfer_efficiency_report.json", transfer_report)
    write_json(bpb_dir / "bpb_gate_velocity_requirement_report.json", velocity_report)
    write_json(bpb_dir / "advanced_metric_evidence_report.json", evidence_report)
    write_json(
        bpb_dir / "checkpoint_manifest.json",
        {
            "checkpoint": str(checkpoint_path) if checkpoint_path else "",
            "checkpoint_step": checkpoint_step,
            "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
            "target_bpb": target_bpb,
        },
    )
    write_json(bpb_dir / "phase_bpb_breakdown_plan.json", phase_bpb_breakdown_plan(target_bpb))
    if diagnostic_payload:
        write_json(output_dir / "geometry" / "fineweb_curve_diagnostic_payload.json", diagnostic_payload)
    plot_bpb_timeseries(frame, bpb_dir / "bpb_descent_timeseries.png", target_bpb)
    plot_bpb_velocity(frame, bpb_dir / "bpb_velocity.png")
    plot_bpb_simplex(frame, bpb_dir / "bpb_descent_simplex.png", target_bpb)
    plot_bpb_target_zone(frame, bpb_dir / "bpb_target_zone.png", target_bpb)
    plot_bpb_drop_waterfall(frame, bpb_dir / "bpb_drop_waterfall.png")
    plot_bpb_eta_to_target(frame, bpb_dir / "bpb_eta_to_target.png", target_bpb)
    plot_bpb_rockfall_dashboard(frame, bpb_dir / "bpb_rockfall_dashboard.png", target_bpb, report)
    plot_bpb_phase_plane(frame, bpb_dir / "bpb_phase_plane.png", target_bpb)
    plot_bpb_transfer_efficiency(frame, bpb_dir / "bpb_transfer_efficiency.png", report)
    plot_bpb_gate_velocity_requirement(report, bpb_dir / "bpb_gate_velocity_requirement.png")
    plot_diagnostic_proxy_geometry(diagnostic_payload, bpb_dir / "diagnostic_proxy_geometry.png")
    plot_structural_recapture_map(report, bpb_dir / "bpb_structural_recapture_map.png")
    plot_advanced_metric_control_map(report, bpb_dir / "advanced_metric_control_map.png")
    plot_advanced_metric_evidence_map(evidence_report, bpb_dir / "advanced_metric_evidence_map.png")
    write_synopsis(output_dir, report, run_path, checkpoint_path, diagnostic_payload)
    return report


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, check=False)
        handle.write(f"\nexit_code={result.returncode}\n")
    return int(result.returncode)


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_periodic_analysis(args: argparse.Namespace, checkpoint: Path, step: int) -> dict[str, Any]:
    root = repo_root()
    output_dir = Path(args.output_root) / f"step-{step:08d}"
    logs_dir = output_dir / "logs"
    metrics_dir = output_dir / "metrics"
    geometry_dir = output_dir / "geometry"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("WANDB_PROJECT", args.project)
    env.setdefault("WANDB_ENTITY", args.entity)
    run_id = args.run_id or args.run_path.split("/")[-1]

    command_status: dict[str, int] = {}
    diagnostic_payload: dict[str, Any] = {}
    diagnostic_json = geometry_dir / "fineweb_curve_diagnostic_payload.json"
    if not args.skip_full_diagnostics:
        diag_cmd = [
            args.python,
            str(root / "scripts" / "mirror_fineweb_full_diagnostics_to_wandb.py"),
            "--log",
            str(args.log),
            "--project",
            args.project,
            "--entity",
            args.entity,
            "--run-id",
            run_id,
            "--run-name",
            run_id,
            "--target-bpb",
            str(args.target_bpb),
            "--gate-step",
            str(args.gate_step),
            "--output-json",
            str(diagnostic_json),
            "--once",
            "--no-wandb",
        ]
        command_status["full_diagnostics"] = run_command(diag_cmd, logs_dir / "full_diagnostics.log", env)
        diagnostic_payload = load_json(diagnostic_json)

    if not args.skip_wandb_metrics:
        if command_status.get("full_diagnostics") == 0 and args.wandb_settle_seconds > 0:
            time.sleep(float(args.wandb_settle_seconds))
        metrics_cmd = [
            args.python,
            str(root / "scripts" / "analyze_wandb_metrics.py"),
            "--run-path",
            args.run_path,
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(metrics_dir),
            "--max-plot-metrics",
            "36",
        ]
        command_status["wandb_metrics"] = run_command(metrics_cmd, logs_dir / "analyze_wandb_metrics.log", env)

    parsed = parse_seq4096_log(Path(args.log))
    report = write_bpb_analysis_artifacts(
        parsed,
        output_dir,
        target_bpb=args.target_bpb,
        gate_step=args.gate_step,
        checkpoint_path=checkpoint,
        run_path=args.run_path,
        diagnostic_payload=diagnostic_payload,
        history_root=Path(args.output_root).parent,
    )
    proposal_cmd = [
        args.python,
        str(root / "scripts" / "propose_training_adjustments.py"),
        "--analysis-dir",
        str(output_dir),
        "--target-bpb",
        str(args.target_bpb),
        "--gate-step",
        str(args.gate_step),
    ]
    command_status["training_adjustment_proposal"] = run_command(
        proposal_cmd,
        logs_dir / "training_adjustment_proposal.log",
        env,
    )
    report["command_status"] = command_status
    write_json(output_dir / "analysis_status.json", report)
    maybe_write_target_artifact_manifest(args, output_dir, checkpoint, report)
    if args.codex_review_hook:
        run_codex_review_hook(args, output_dir, checkpoint, step, env)
    return report


def maybe_write_target_artifact_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    checkpoint: Path,
    report: dict[str, Any],
) -> None:
    best = report.get("best_val_bpb")
    try:
        best_float = float(best)
    except (TypeError, ValueError):
        return
    if not math.isfinite(best_float) or best_float > float(args.target_bpb):
        return
    target_dir = Path(args.output_root) / "parameter_golf_threshold_checkpoint"
    target_dir.mkdir(parents=True, exist_ok=True)
    link = target_dir / checkpoint.name
    if not link.exists():
        try:
            link.symlink_to(checkpoint.resolve())
        except OSError:
            link.write_text(str(checkpoint.resolve()) + "\n", encoding="utf-8")
    write_json(
        target_dir / "manifest.json",
        {
            "checkpoint": str(checkpoint.resolve()),
            "analysis_dir": str(output_dir.resolve()),
            "best_val_bpb": best_float,
            "target_bpb": float(args.target_bpb),
            "state": report.get("state"),
        },
    )


def run_codex_review_hook(args: argparse.Namespace, output_dir: Path, checkpoint: Path, step: int, env: dict[str, str]) -> None:
    hook = Path(args.codex_review_hook)
    if not hook.exists():
        return
    session = f"{args.codex_review_tmux_prefix}_{step:08d}"
    command = [
        str(hook),
        "--analysis-dir",
        str(output_dir),
        "--checkpoint",
        str(checkpoint),
        "--step",
        str(step),
        "--run-path",
        args.run_path,
        "--tmux-session",
        session,
    ]
    if args.training_tmux:
        command.extend(["--training-tmux", args.training_tmux])
    run_command(command, output_dir / "logs" / "codex_review_hook.log", env)


def next_interval_step(start_step: int, interval_steps: int, processed: set[int]) -> int:
    step = start_step + interval_steps
    while step in processed:
        step += interval_steps
    return step


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    log_path = Path(args.log)
    processed: set[int] = set()
    completed = 0
    target_step = next_interval_step(args.start_step, args.interval_steps, processed)
    print(
        f"seq4096_analysis_watcher run_path={args.run_path} checkpoint_dir={checkpoint_dir} "
        f"target_step={target_step} interval={args.interval_steps}",
        flush=True,
    )
    while True:
        parsed = parse_seq4096_log(log_path)
        latest = parsed.latest_step
        checkpoint = find_checkpoint_at_step(checkpoint_dir, target_step)
        if checkpoint is not None and latest >= target_step:
            print(f"analysis_start step={target_step} checkpoint={checkpoint}", flush=True)
            report = run_periodic_analysis(args, checkpoint, target_step)
            print(
                "analysis_done "
                f"step={target_step} state={report.get('state')} "
                f"best_val_bpb={report.get('best_val_bpb')} gap={report.get('target_gap')}",
                flush=True,
            )
            processed.add(target_step)
            completed += 1
            if args.once or (args.max_checkpoints and completed >= args.max_checkpoints):
                break
            target_step = next_interval_step(args.start_step, args.interval_steps, processed)
            continue
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        latest_checkpoint_step = extract_checkpoint_step(latest_checkpoint) if latest_checkpoint else None
        print(
            f"waiting target_step={target_step} latest_log_step={latest} "
            f"latest_checkpoint_step={latest_checkpoint_step}",
            flush=True,
        )
        if args.once and latest >= target_step and checkpoint is None:
            break
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
