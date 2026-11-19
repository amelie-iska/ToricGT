"""CUDA AdamW restart parity on authored gradients and disposable parameters."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.train import apply_optimizer_update, scheduled_lr, set_optimizer_lr
from toricgt.synthetic_resume import (
    atomic_save_checkpoint,
    capture_synthetic_recovery,
    configure_prepared_cuda,
    restore_synthetic_recovery,
)


def replay_updates(root: Path, resumed: bool) -> None:
    device = "cuda:0"
    configure_prepared_cuda(device)
    params = torch.nn.ParameterList([
        torch.nn.Parameter(torch.tensor([0.5, -0.25, 1.5], device=device)),
        torch.nn.Parameter(torch.tensor([-0.125, 0.875], device=device)),
    ])
    optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", init_scale=64.0 if resumed else 16.0, growth_interval=1)
    path = root / "before-boundary.pt"
    if resumed:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        params.load_state_dict(checkpoint["params"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_synthetic_recovery(checkpoint["prepared_recovery"], scaler, device)
        start_step = checkpoint["step"]
    else:
        optimizer.state[params[0]] = {
            "step": torch.tensor(2.0),
            "exp_avg": torch.full_like(params[0], 0.125),
            "exp_avg_sq": torch.full_like(params[0], 0.25),
        }
        start_step = 2
        atomic_save_checkpoint({
            "params": params.state_dict(), "optimizer": optimizer.state_dict(), "step": start_step,
            "prepared_recovery": capture_synthetic_recovery(scaler, device, 6, "authored-gradients"),
        }, path)

    receipts = []
    for index, gradient in enumerate(([3.0, -2.0, 5.0], [float("inf"), 0.0, -1.0], [-4.0, 2.0, 3.0])):
        step = start_step + index
        lr = scheduled_lr(0.001, step, 6, 1, 0.1, "cosine")
        set_optimizer_lr(optimizer, lr)
        gradients = (gradient, [1.0, -2.0])
        for parameter, values in zip(params, gradients, strict=True):
            parameter.grad = scaler.scale(torch.tensor(values, device=device))
        previous = [p.detach().clone() for p in params]
        norm = apply_optimizer_update(params.parameters(), optimizer, scaler, 1.0)
        assert all(p.grad is None for p in params)
        changed = any(not torch.equal(old, param) for old, param in zip(previous, params, strict=True))
        receipts.append({
            "changed": changed, "lr": lr, "norm": norm.cpu(),
            "params": [param.detach().cpu() for param in params],
            "moments": [{key: value.detach().cpu().clone() for key, value in optimizer.state[param].items()}
                        for param in params],
            "scaler": scaler.state_dict(),
        })
    torch.save(receipts, root / ("restart.pt" if resumed else "continuous.pt"))


@pytest.mark.skipif(os.environ.get("TORICMORA_CUDA_PARITY") != "1", reason="opt-in authored CUDA parity")
def test_cuda_native_adamw_boundary_matches_across_restart(tmp_path):
    for resumed in (False, True):
        subprocess.run([
            sys.executable, "-B", "-c",
            "import runpy,sys; from pathlib import Path;"
            "runpy.run_path(sys.argv[1])['replay_updates'](Path(sys.argv[2]),sys.argv[3]=='True')",
            str(Path(__file__).resolve()), str(tmp_path), str(resumed),
        ], check=True)
    uninterrupted = torch.load(tmp_path / "continuous.pt", weights_only=True)
    restarted = torch.load(tmp_path / "restart.pt", weights_only=True)
    torch.testing.assert_close(restarted, uninterrupted, rtol=0, atol=0)
    assert [receipt["changed"] for receipt in restarted] == [True, False, True]
    assert [receipt["scaler"]["scale"] for receipt in restarted] == [32.0, 16.0, 32.0]
    assert [receipt["moments"][0]["step"].item() for receipt in restarted] == [3.0, 3.0, 4.0]
    assert [receipt["moments"][1]["step"].item() for receipt in restarted] == [1.0, 1.0, 2.0]
