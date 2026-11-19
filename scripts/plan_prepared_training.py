#!/usr/bin/env python3
"""Admit an exact symbolic manifest and print a two-epoch CUDA launch plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from toricgt.prepared_stream import open_prepared_stream


def build_plan(
    admission: dict, manifest: Path, output: Path, *, batch_size: int, max_accum: int, seed: int,
    config: Path | None = None, precision: str = "fp16", warmup_steps: int | None = None,
    checkpoint_every: int | None = None,
) -> dict:
    if min(batch_size, max_accum) < 1:
        raise ValueError("batch size and maximum accumulation must be positive")
    batches = (admission["train_records"] + batch_size - 1) // batch_size
    if batches < 1:
        raise ValueError("a two-epoch plan requires at least one training batch")
    # An accumulation boundary must coincide with each exact epoch boundary.
    accumulation = next(n for n in range(min(max_accum, batches), 0, -1) if batches % n == 0)
    epoch_steps = batches // accumulation
    total_steps = 2 * epoch_steps
    if warmup_steps is None:
        warmup_steps = min(2000, total_steps // 10)
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup must end before the last planned update")
    digest = admission["manifest_sha256"]
    common = [
        sys.executable, "-B", "scripts/train.py", "--device", "cuda:0",
        "--prepared-stream-manifest", str(manifest), "--prepared-stream-sha256", digest,
        "--seed", str(seed), "--precision", precision, "--batch-size", str(batch_size),
        "--grad-accum-steps", str(accumulation), "--steps", str(epoch_steps),
        "--schedule-total-steps", str(total_steps), "--warmup-steps", str(warmup_steps),
        "--checkpoint-every", str(epoch_steps if checkpoint_every is None else checkpoint_every),
    ]
    if config is not None:
        common.extend(["--config", str(config)])
    if val_rows := admission["validation_records"]:
        validation_batches = (val_rows + batch_size - 1) // batch_size
        common += ["--eval-every", str(epoch_steps), "--eval-batches", str(validation_batches)]

    segments = []
    previous = None
    for epoch in (1, 2):
        directory = output / f"epoch-{epoch}"
        argv = common + ["--checkpoint-dir", str(directory)]
        if previous is not None:
            argv.extend(["--resume", str(previous)])
        previous = directory / "toricgt_final.pt"
        segments.append({
            "epoch": epoch, "end_step": epoch * epoch_steps,
            "expected_batches_consumed": epoch * batches,
            "checkpoint": str(previous), "argv": argv,
        })
    return {
        "admission": admission, "seed": seed, "batch_size": batch_size,
        "grad_accum_steps": accumulation, "batches_per_epoch": batches,
        "schedule_total_steps": total_steps, "segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New experiment path. The planner creates no files.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("choose a new experiment path")
    try:
        with open_prepared_stream(args.manifest, args.sha256) as (_, admission):
            plan = build_plan(admission, args.manifest.resolve(), args.output.absolute(),
                              batch_size=args.batch_size, max_accum=args.max_accum, seed=args.seed)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
