"""Exact-state checkpoints for the native trainer."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch


def atomic_save_checkpoint(payload: dict, path: Path) -> None:
    """Verify the serialized checkpoint before replacing the destination."""
    staging_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            staging_path = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        with ZipFile(staging_path) as archive:
            damaged = archive.testzip()
            if damaged is not None:
                raise OSError(f"checkpoint archive failed integrity verification: {damaged}")
        os.replace(staging_path, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)


def synthetic_resume_plan(args, checkpoint: dict | None = None) -> tuple[int, str | None]:
    """Keep an absolute learning-rate horizon across segmented invocations."""
    recovery = checkpoint.get("synthetic_recovery") if checkpoint is not None else None
    step = int(checkpoint.get("step", 0)) if checkpoint is not None else 0
    end = step + args.steps
    if not args.exact_synthetic_resume:
        if args.schedule_total_steps is not None:
            raise ValueError("--schedule-total-steps requires --exact-synthetic-resume")
        return end, None
    if args.data_path or args.val_data_path or args.wandb:
        raise ValueError("exact synthetic recovery requires generated graphs and offline logging")
    if checkpoint is not None and (not isinstance(recovery, dict) or recovery.get("version") != 1):
        raise ValueError("checkpoint lacks exact synthetic recovery state")
    if checkpoint is not None and ("optimizer" not in checkpoint or step < 0):
        raise ValueError("checkpoint lacks a valid optimizer state or step")
    horizon = args.schedule_total_steps
    if horizon is None:
        horizon = recovery.get("schedule_total_steps") if recovery is not None else end
    if type(horizon) is not int or args.steps < 1 or end > horizon:
        raise ValueError("requested steps exceed the fixed synthetic schedule horizon")
    return _check_plan_digest(args, horizon, recovery, "synthetic")


def _check_plan_digest(args, horizon, recovery, label, **extra):
    invocation_keys = {
        "resume", "steps", "checkpoint_dir", "config", "schedule_total_steps",
        "prepared_stream_manifest", "prepared_stream_sha256", "check_prepared_stream",
    }
    settings = {key: value for key, value in vars(args).items() if key not in invocation_keys}
    settings.update(schedule_total_steps=horizon, torch_version=str(torch.__version__), **extra)
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if recovery is not None and recovery.get("plan_sha256") != digest:
        raise ValueError(f"exact {label} recovery configuration differs from the checkpoint")
    return horizon, digest


def configure_prepared_cuda(device: str) -> dict:
    import pyarrow

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("prepared recovery requires a CUDA training device")
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in (":4096:8", ":16:8"):
        raise ValueError("prepared recovery requires a deterministic cuBLAS workspace configuration")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(target),
        "capability": list(torch.cuda.get_device_capability(target)),
        "cublas_workspace": workspace,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "python": list(sys.version_info[:3]),
    }


def prepared_resume_plan(args, checkpoint, manifest_sha256, runtime) -> tuple[int, str]:
    recovery = checkpoint.get("prepared_recovery") if checkpoint is not None else None
    step = checkpoint.get("step") if checkpoint is not None else 0
    if checkpoint is not None and (
        not isinstance(recovery, dict) or recovery.get("version") != 1 or "optimizer" not in checkpoint
    ):
        raise ValueError("checkpoint lacks exact prepared recovery state")
    if type(step) is not int or step < 0 or args.grad_accum_steps < 1 or args.steps < 1:
        raise ValueError("invalid prepared checkpoint step or run length")
    end = step + args.steps
    horizon = args.schedule_total_steps
    if horizon is None:
        horizon = recovery.get("schedule_total_steps") if recovery is not None else end
    if type(horizon) is not int or end > horizon:
        raise ValueError("requested steps exceed the fixed prepared schedule horizon")
    return _check_plan_digest(
        args, horizon, recovery, "prepared", manifest_sha256=manifest_sha256, runtime=runtime,
    )


def capture_synthetic_recovery(scaler, device: str, horizon: int, digest: str) -> dict:
    target = torch.device(device)
    return {
        "version": 1,
        "schedule_total_steps": horizon,
        "plan_sha256": digest,
        "scaler": scaler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(target) if target.type == "cuda" else None,
    }


def restore_synthetic_recovery(recovery: dict, scaler, device: str) -> None:
    target = torch.device(device)
    scaler.load_state_dict(recovery["scaler"])
    random.setstate(recovery["python_rng"])
    np.random.set_state(recovery["numpy_rng"])
    torch.random.set_rng_state(recovery["torch_rng"])
    if target.type == "cuda":
        torch.cuda.set_rng_state(recovery["cuda_rng"], target)
