#!/usr/bin/env python3
"""Mirror Parameter-Golf FineWeb log metrics to W&B.

The local Parameter-Golf scaffold writes plain-text logs and does not depend on
W&B.  This tailer parses `train_loss` and `val_bpb` lines from an active log and
logs them to W&B under a separate monitoring run without touching the trainer.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path


TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+train_loss:(?P<loss>[0-9.]+)\s+train_time:(?P<ms>[0-9.]+)ms"
)
VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<loss>[0-9.]+)\s+val_bpb:(?P<bpb>[0-9.]+)\s+train_time:(?P<ms>[0-9.]+)ms"
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
                    bpb = float(val.group("bpb"))
                    best_bpb = bpb if best_bpb is None else min(best_bpb, bpb)
                    wandb.log(
                        {
                            "fineweb/val_loss": float(val.group("loss")),
                            "fineweb/val_bpb": bpb,
                            "fineweb/best_val_bpb": best_bpb,
                            "fineweb/target_bpb": args.target_bpb,
                            "fineweb/target_reached": float(best_bpb <= args.target_bpb),
                            "fineweb/train_time_ms": float(val.group("ms")),
                        },
                        step=step,
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
                    wandb.log(
                        {
                            "fineweb/train_loss": float(train.group("loss")),
                            "fineweb/train_time_ms": float(train.group("ms")),
                        },
                        step=step,
                    )
        if any(marker in text for marker in ("final_int8_zlib_roundtrip", "Traceback", "RuntimeError")) if path.exists() else False:
            break
        time.sleep(max(1.0, args.poll_seconds))
    run.finish()


if __name__ == "__main__":
    main()
