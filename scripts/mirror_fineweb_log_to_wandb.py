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
    return parser.parse_args()


def default_diagnostics_json(log_path: Path) -> Path:
    return Path("logs") / f"{log_path.stem}.full_diag.latest.json"


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
    seen: set[tuple[str, int]] = set()
    best_bpb: float | None = None
    initial_bpb: float | None = None
    while True:
        stop_requested = False
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            text_tail = "\n".join(text.splitlines()[-20:])
            for line in text.splitlines():
                val = VAL_RE.search(line)
                if val:
                    step = int(val.group("step"))
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
                    payload.update(diagnostic_payload)
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
                    payload.update(diagnostic_payload)
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
        if stop_requested:
            break
        time.sleep(max(1.0, args.poll_seconds))
    run.finish()


if __name__ == "__main__":
    main()
