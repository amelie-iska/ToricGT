#!/usr/bin/env python3
"""Mirror Parameter-Golf FineWeb log metrics to W&B.

The local Parameter-Golf scaffold writes plain-text logs and does not depend on
W&B.  This tailer parses `train_loss` and `val_bpb` lines from an active log and
logs them to W&B under a separate monitoring run without touching the trainer.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from pathlib import Path


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
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--target-bpb", type=float, default=1.2)
    return parser.parse_args()


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
    seen: set[tuple[str, int]] = set()
    best_bpb: float | None = None
    while True:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
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
                        "bpb/val": bpb,
                        "bpb/best": best_bpb,
                        "bpb/target": args.target_bpb,
                        "bpb/gap_to_target": target_gap,
                        "bpb/improvement_from_initial": (
                            0.0 if best_bpb is None else 4.1077 - bpb
                        ),
                        "bpb/target_reached": float(best_bpb <= args.target_bpb),
                    }
                    wandb.log(payload, step=step)
                    run.summary.update(
                        {
                            "fineweb/val_bpb": bpb,
                            "fineweb/best_val_bpb": best_bpb,
                            "fineweb/target_bpb": args.target_bpb,
                            "val/bpb": bpb,
                            "val/loss": val_loss,
                            "bpb/val": bpb,
                            "bpb/best": best_bpb,
                            "bpb/target": args.target_bpb,
                            "bpb/gap_to_target": target_gap,
                            "progress/step": step,
                            "progress/fraction": step / max(total, 1),
                        }
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
                    wandb.log(payload, step=step)
                    run.summary.update(
                        {
                            "fineweb/train_loss": train_loss,
                            "train/loss": train_loss,
                            "progress/step": step,
                            "progress/fraction": step / max(total, 1),
                        }
                    )
        if any(marker in text for marker in ("final_int8_zlib_roundtrip", "Traceback", "RuntimeError")) if path.exists() else False:
            break
        time.sleep(max(1.0, args.poll_seconds))
    run.finish()


if __name__ == "__main__":
    main()
