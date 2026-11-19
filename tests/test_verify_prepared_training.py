"""Prepared-run verifier tests using hand-authored checkpoint archives."""

from pathlib import Path

import pytest
import torch

from scripts.plan_prepared_training import build_plan
from scripts.verify_prepared_training import inspect_run
from toricgt.synthetic_resume import atomic_save_checkpoint


@pytest.fixture
def launch_plan(tmp_path):
    return build_plan(
        {"manifest_sha256": "a" * 64, "train_records": 5, "validation_records": 2},
        tmp_path / "unused.json", tmp_path / "future",
        batch_size=2, max_accum=2, seed=29,
    )


def write_receipts(plan, fault=None):
    for segment in plan["segments"]:
        epoch = segment["epoch"]
        checkpoint = Path(segment["checkpoint"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        recovery = {
            "manifest_sha256": plan["admission"]["manifest_sha256"],
            "plan_sha256": "b" * 64,
            "schedule_total_steps": plan["schedule_total_steps"],
            "batches_consumed": segment["expected_batches_consumed"],
            "validation": {"step": segment["end_step"], "records": 2, "masked_mse": 1.0 / epoch},
        }
        value = float(epoch)
        if epoch == 2:
            if fault == "cursor":
                recovery["batches_consumed"] -= 1
            elif fault == "recipe":
                recovery["plan_sha256"] = "c" * 64
            elif fault == "validation":
                recovery["validation"]["records"] = 1
            elif fault == "weights":
                value = 1.0
        atomic_save_checkpoint({
            "step": segment["end_step"], "prepared_recovery": recovery,
            "model": {"weight": torch.tensor([value])}, "optimizer": {"state": {0: {"step": epoch}}},
        }, checkpoint)


def test_prepared_run_verifies_two_checkpoints_without_accessing_the_manifest(launch_plan):
    write_receipts(launch_plan)
    report = inspect_run(launch_plan)
    assert report["manifest_sha256"] == "a" * 64
    assert report["recipe_sha256"] == "b" * 64
    assert [row["batches_consumed"] for row in report["epochs"]] == [3, 6]
    assert [row["validation"]["masked_mse"] for row in report["epochs"]] == [1.0, 0.5]
    assert all(len(row["checkpoint_sha256"]) == 64 for row in report["epochs"])


@pytest.mark.parametrize("fault", ["cursor", "recipe", "validation", "weights"])
def test_prepared_run_rejects_checkpoint_drift(launch_plan, fault):
    write_receipts(launch_plan, fault)
    with pytest.raises(ValueError, match=fault):
        inspect_run(launch_plan)
