from argparse import Namespace
import random

import numpy as np
import pytest
import torch

from scripts.train import save_checkpoint, scheduled_lr, set_seed, synthetic_batch
from toricgt.config import ModelConfig, TrainConfig
from toricgt.model import ToricTokenGT
from toricgt.synthetic_resume import (
    atomic_save_checkpoint,
    capture_synthetic_recovery,
    restore_synthetic_recovery,
    synthetic_resume_plan,
)


def options(**overrides):
    values = {
        "exact_synthetic_resume": True,
        "resume": None,
        "steps": 3,
        "schedule_total_steps": 8,
        "data_path": [],
        "val_data_path": [],
        "wandb": False,
        "lr_schedule": "cosine",
        "checkpoint_dir": "fixture",
        "seed": 29,
    }
    return Namespace(**(values | overrides))


def test_generated_graph_resume_plan_requires_same_settings_and_horizon():
    horizon, digest = synthetic_resume_plan(options())
    checkpoint = {
        "optimizer": {},
        "step": 3,
        "synthetic_recovery": {"version": 1, "schedule_total_steps": horizon, "plan_sha256": digest},
    }
    args = options(resume="authored.pt", steps=5, schedule_total_steps=None, checkpoint_dir="other-fixture")
    assert synthetic_resume_plan(args, checkpoint) == (horizon, digest)
    with pytest.raises(ValueError, match="fixed synthetic schedule"):
        synthetic_resume_plan(options(steps=6), checkpoint)
    with pytest.raises(ValueError, match="configuration differs"):
        synthetic_resume_plan(options(seed=17), checkpoint)
    with pytest.raises(ValueError, match="lacks exact"):
        synthetic_resume_plan(args, {"step": 3, "optimizer": {}})
    for override in ({"wandb": True}, {"data_path": ["unused"]}, {"val_data_path": ["unused"]}):
        with pytest.raises(ValueError, match="requires generated graphs"):
            synthetic_resume_plan(options(**override))
    assert synthetic_resume_plan(options(exact_synthetic_resume=False, schedule_total_steps=None), checkpoint) == (6, None)


def test_native_no_update_restore_matches_next_generated_batch_and_forward(tmp_path):
    seed = 29
    set_seed(seed)
    cfg = ModelConfig(d_model=16, num_heads=2, num_layers=1, max_nodes=4, max_edges=6, use_soft_moe=False)
    train_cfg = TrainConfig(device="cpu", precision="fp32", use_wandb=False)
    model = ToricTokenGT(cfg).train()
    optimizer = torch.optim.AdamW(model.parameters())
    parameter = next(model.parameters())
    optimizer.state[parameter] = {
        "step": torch.tensor(3.0),
        "exp_avg": torch.full_like(parameter, 0.125),
        "exp_avg_sq": torch.full_like(parameter, 0.25),
    }
    scaler = torch.amp.GradScaler("cpu", init_scale=64.0)
    scaler.scale(torch.ones(()))
    horizon, digest = synthetic_resume_plan(options())
    recovery = capture_synthetic_recovery(scaler, "cpu", horizon, digest)
    save_checkpoint(
        tmp_path, "authored.pt", model, optimizer, cfg, train_cfg, 3, synthetic_recovery=recovery,
    )

    @torch.no_grad()
    def next_observations(net):
        batch, target = synthetic_batch(cfg, 2, "cpu")
        prediction = net(batch)["node"]
        return random.random(), np.random.random(), batch.edge_index, target, prediction

    uninterrupted = next_observations(model)
    set_seed(88)
    restored_model = ToricTokenGT(cfg).train()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters())
    restored_scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    checkpoint = torch.load(tmp_path / "authored.pt", map_location="cpu", weights_only=False)
    resumed_args = options(resume="authored.pt", steps=5, schedule_total_steps=None)
    resume_horizon, _ = synthetic_resume_plan(resumed_args, checkpoint)
    restored_model.load_state_dict(checkpoint["model"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restore_synthetic_recovery(checkpoint["synthetic_recovery"], restored_scaler, "cpu")
    resumed = next_observations(restored_model)
    assert resumed[:2] == uninterrupted[:2]
    for actual, expected in zip(resumed[2:], uninterrupted[2:], strict=True):
        assert torch.equal(actual, expected)
    assert restored_scaler.state_dict() == scaler.state_dict()
    original_state = optimizer.state[parameter]
    resumed_state = restored_optimizer.state[next(restored_model.parameters())]
    for key in original_state:
        assert torch.equal(resumed_state[key], original_state[key])
    expected_lr = scheduled_lr(0.001, 3, horizon, 2, 0.1, "cosine")
    resumed_lr = scheduled_lr(0.001, checkpoint["step"], resume_horizon, 2, 0.1, "cosine")
    assert resumed_lr == expected_lr


@pytest.mark.parametrize("corrupt_archive", [False, True])
def test_atomic_save_retains_existing_checkpoint_on_failure(tmp_path, monkeypatch, corrupt_archive):
    target = tmp_path / "authored.pt"
    atomic_save_checkpoint({"step": 1}, target)
    original = target.read_bytes()
    write = torch.save

    def interrupted_save(payload, stream):
        if corrupt_archive:
            stream.write(b"incomplete zip")
            return
        write(payload, stream)
        raise OSError("interrupted write")

    monkeypatch.setattr("toricgt.synthetic_resume.torch.save", interrupted_save)
    from zipfile import BadZipFile

    with pytest.raises((OSError, BadZipFile)):
        atomic_save_checkpoint({"step": 2}, target)
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]
