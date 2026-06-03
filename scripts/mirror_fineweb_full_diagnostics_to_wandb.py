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


TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+train_loss:(?P<loss>[0-9.]+)"
    r"\s+train_time:(?P<ms>[0-9.]+)ms\s+step_avg:(?P<avg>[0-9.]+)ms"
)
VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<loss>[0-9.]+)"
    r"\s+val_bpb:(?P<bpb>[0-9.]+)\s+train_time:(?P<ms>[0-9.]+)ms"
    r"\s+step_avg:(?P<avg>[0-9.]+)ms"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "toricgt-parameter-golf"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "amelie-iska-math"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--target-bpb", type=float, default=1.2)
    parser.add_argument("--max-points", type=int, default=160)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--once", action="store_true", help="Log one diagnostics payload and exit.")
    parser.add_argument("--no-wandb", action="store_true", help="Write a one-shot JSON payload without opening W&B.")
    return parser.parse_args()


def parse_log(path: Path) -> dict[str, Any]:
    train: dict[int, dict[str, float]] = {}
    vals: dict[int, dict[str, float]] = {}
    if not path.exists():
        return {"train": train, "vals": vals, "latest_step": 0, "total": 0}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
        train_match = TRAIN_RE.search(line)
        if train_match:
            step = int(train_match.group("step"))
            train[step] = {
                "step": float(step),
                "total": float(train_match.group("total")),
                "train_loss": float(train_match.group("loss")),
                "train_time_ms": float(train_match.group("ms")),
                "step_avg_ms": float(train_match.group("avg")),
            }
    latest_step = max([0, *train.keys(), *vals.keys()])
    total = int(max([0.0, *[row["total"] for row in train.values()], *[row["total"] for row in vals.values()]]))
    return {"train": train, "vals": vals, "latest_step": latest_step, "total": total}


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
        hidden = torch.zeros((1, 0, 12), dtype=torch.float32)
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
    for step in steps:
        train_loss = _last_at_or_before(train, step, "train_loss", prev_train)
        val_loss = _last_at_or_before(vals, step, "val_loss", train_loss)
        val_bpb = _last_at_or_before(vals, step, "val_bpb", prev_val)
        best = min(best, val_bpb)
        step_avg = _last_at_or_before(train, step, "step_avg_ms", 0.0)
        progress = float(step) / max(total, 1.0)
        gap = val_bpb - float(target_bpb)
        val_delta = val_bpb - prev_val
        train_delta = train_loss - prev_train
        angle_theta = 2.0 * math.pi * 0.6180339887498948 * step
        angle_beta = 2.0 * math.pi * (math.sqrt(2.0) - 1.0) * step
        rows.append(
            [
                progress,
                math.log1p(float(step)) / math.log1p(max(total, 1.0)),
                train_loss,
                val_loss,
                val_bpb,
                best,
                gap,
                val_delta,
                train_delta,
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


def compute_payload(parsed: dict[str, Any], log_path: Path, target_bpb: float, max_points: int) -> dict[str, float]:
    hidden, positions, val_bpbs = build_curve_tensor(parsed, max_points, target_bpb)
    payload: dict[str, float] = {
        "metrics_status/model_hidden_state_available": 0.0,
        "metrics_status/fineweb_curve_diagnostics_available": 1.0 if hidden.shape[1] >= 4 else 0.0,
        "metrics_status/full_toricgt_training_metrics_available": 0.0,
        "metrics_status/toric_bgg_training_enabled": 0.0,
        "metrics_status/metric_scope_fineweb_curve_proxy": 1.0,
        "diagnostics/source_code": 1.0,
    }
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
    return payload


def write_json(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_once_without_wandb(
    *,
    log_path: Path,
    output_json: Path | str,
    target_bpb: float,
    max_points: int,
) -> dict[str, float]:
    parsed = parse_log(log_path)
    latest_step = int(parsed.get("latest_step") or 0)
    total = int(parsed.get("total") or 0)
    payload = compute_payload(parsed, log_path, target_bpb, max_points)
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
    wandb.define_metric("*", step_metric="trainer/step")
    log_path = Path(args.log)
    seen_steps: set[int] = set()
    while True:
        parsed = parse_log(log_path)
        latest_step = int(parsed.get("latest_step") or 0)
        total = int(parsed.get("total") or 0)
        should_log = latest_step > 0 and latest_step not in seen_steps
        if should_log:
            payload = compute_payload(parsed, log_path, args.target_bpb, args.max_points)
            payload.update(
                {
                    "trainer/step": latest_step,
                    "trainer/total_steps": total,
                    "progress/step": latest_step,
                    "progress/total_steps": total,
                    "progress/fraction": float(latest_step) / max(float(total), 1.0),
                }
            )
            wandb.log(payload)
            run.summary.update(
                {
                    "diagnostics/latest_full_metrics_step": latest_step,
                    "metrics_status/model_hidden_state_available": 0.0,
                    "metrics_status/fineweb_curve_diagnostics_available": payload[
                        "metrics_status/fineweb_curve_diagnostics_available"
                    ],
                    "bgg_category_o/late_phase_toggle_enabled": 0.0,
                    "toric/shadow_fan_cell_entropy": payload.get("toric/shadow_fan_cell_entropy", 0.0),
                    "topology/topology_loss": payload.get("topology/topology_loss", 0.0),
                    "tropical/bpb_best_so_far": payload.get("tropical/bpb_best_so_far", 0.0),
                }
            )
            write_json(args.output_json, payload)
            print(
                "wandb_full_diag "
                f"step={latest_step} topology={payload.get('topology/topology_loss', 0.0):.4f} "
                f"toric_entropy={payload.get('toric/shadow_fan_cell_entropy', 0.0):.4f} "
                f"bgg_d2={payload.get('bgg_category_o/d2_residual', 0.0):.4f}",
                flush=True,
            )
            seen_steps.add(latest_step)
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
