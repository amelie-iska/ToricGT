#!/usr/bin/env python3
"""Verify the two ToricMora boundary checkpoints from a trusted launch plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import torch


def load_checkpoint(path: Path):
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
        stream.seek(0)
        with ZipFile(stream) as archive:
            if archive.testzip() is not None:
                raise ValueError("checkpoint archive has a damaged member")
        stream.seek(0)
        # The operator supplies their own launch plan and local checkpoints.
        return torch.load(stream, map_location="cpu", weights_only=False), checksum.hexdigest()


def inspect_run(plan):
    segments = plan["segments"]
    if [segment["epoch"] for segment in segments] != [1, 2]:
        raise ValueError("expected exactly two epoch boundaries")
    horizon = plan["schedule_total_steps"]
    accumulation = plan["grad_accum_steps"]
    batches_per_epoch = plan["batches_per_epoch"]
    manifest_pin = plan["admission"]["manifest_sha256"]
    validation_size = plan["admission"]["validation_records"]
    if min(horizon, accumulation, batches_per_epoch) < 1 or horizon * accumulation != 2 * batches_per_epoch:
        raise ValueError("the plan does not cover exactly two complete epochs")

    report = []
    first_weights = None
    first_recipe_pin = None
    for segment in segments:
        epoch = segment["epoch"]
        step = segment["end_step"]
        cursor = segment["expected_batches_consumed"]
        if 2 * step != epoch * horizon or cursor != epoch * batches_per_epoch:
            raise ValueError("invalid epoch or cursor boundary in plan")
        path = Path(segment["checkpoint"])
        checkpoint, sha256 = load_checkpoint(path)
        recovery = checkpoint["prepared_recovery"]
        recipe_pin = recovery["plan_sha256"]
        if (checkpoint["step"] != step or recovery["batches_consumed"] != cursor
                or recovery["schedule_total_steps"] != horizon or recovery["manifest_sha256"] != manifest_pin):
            raise ValueError(f"epoch {epoch} checkpoint differs from the planned step, cursor, or input pin")
        if not isinstance(recipe_pin, str) or len(recipe_pin) != 64:
            raise ValueError("checkpoint is missing its prepared recipe pin")
        if not checkpoint["optimizer"]["state"]:
            raise ValueError(f"epoch {epoch} checkpoint contains no optimizer updates")
        weights = checkpoint["model"]
        if not weights or any(not torch.is_tensor(w) or not torch.isfinite(w).all() for w in weights.values()):
            raise ValueError(f"epoch {epoch} checkpoint contains invalid weights")
        if first_weights is None:
            first_weights = weights
            first_recipe_pin = recipe_pin
        else:
            if first_recipe_pin != recipe_pin or first_weights.keys() != weights.keys():
                raise ValueError("checkpoint recipe or parameter inventory drifted between epochs")
            if all(torch.equal(first_weights[key], value) for key, value in weights.items()):
                raise ValueError("epoch two checkpoint weights did not change")
        validation = recovery.get("validation")
        if validation_size and (
            not isinstance(validation, dict) or validation["step"] != step
            or validation["records"] != validation_size or not math.isfinite(validation["masked_mse"])
        ):
            raise ValueError(f"epoch {epoch} validation is missing, incomplete, stale, or non-finite")
        report.append({
            "epoch": epoch, "step": step, "batches_consumed": cursor,
            "checkpoint_sha256": sha256, "validation": validation,
        })
    return {"manifest_sha256": manifest_pin, "recipe_sha256": first_recipe_pin, "epochs": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-plan", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = inspect_run(json.loads(args.trusted_plan.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, KeyError, RuntimeError, BadZipFile) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
