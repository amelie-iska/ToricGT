#!/usr/bin/env python3
"""Check, plan, or explicitly launch the reviewed symbolic CUDA recipe."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml

from plan_prepared_training import build_plan
from verify_prepared_training import inspect_run
from toricgt.cli_config import flatten_cli_config
from toricgt.prepared_stream import open_prepared_stream


BINDINGS = {"manifest", "manifest_sha256", "output", "gpu_lease", "cuda_visible_device", "checkpoint_every"}
PLANNER_FIELDS = {
    "steps", "device", "checkpoint_dir", "resume", "prepared_stream_manifest", "prepared_stream_sha256",
    "check_prepared_stream", "schedule_total_steps", "checkpoint_every", "eval_every", "eval_batches",
}
DISALLOWED_WHEN_SET = {
    "data_path", "val_data_path", "interleave_data_paths", "expert_cyclic_curriculum", "wandb",
    "exact_synthetic_resume", "use_lm_head", "use_lm_token_embeddings", "fineweb_lm_loss_weight",
    "write_derived_category_examples", "write_memory_trace_examples", "artifact_probe_every",
    "artifact_probe_at_start",
}


def _read_recipe(config: Path):
    raw = config.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("configuration requires a metadata section")
    launcher = payload["metadata"].get("prepared_launch")
    if not isinstance(launcher, dict) or launcher.keys() != BINDINGS:
        raise ValueError("metadata.prepared_launch must declare the paths, pin, GPU and checkpoint cadence")
    settings = flatten_cli_config(payload)
    if unexpected := PLANNER_FIELDS & settings.keys():
        raise ValueError("launcher-generated settings must be omitted: " + ", ".join(sorted(unexpected)))
    if incompatible := {key for key in DISALLOWED_WHEN_SET if settings.get(key)}:
        raise ValueError("incompatible settings for reviewed graph training: " + ", ".join(sorted(incompatible)))
    choices = {
        "attention": {"tropical_ring", "hybrid", "tropical", "softmax"},
        "precision": {"fp16", "bf16", "fp32"},
        "lr_schedule": {"constant", "cosine"},
        "gflownet_space": {"embedding"},
    }
    for name, options in choices.items():
        if settings.get(name) not in options:
            raise ValueError(f"invalid {name} for reviewed graph training")
    positive = (
        "d_model", "num_heads", "num_layers", "max_nodes", "max_edges", "batch_size", "grad_accum_steps",
        "soft_moe_experts", "soft_moe_slots", "recurrent_passes", "log_interval",
        "ffn_multiplier", "ring_block_size", "torus_rank",
    )
    for name in positive:
        if type(settings.get(name)) is not int or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if settings["d_model"] % settings["num_heads"]:
        raise ValueError("d_model must be divisible by num_heads")
    for name in ("no_soft_moe", "soft_moe_all_layers", "use_causal_graph_attention", "activation_checkpointing"):
        if type(settings.get(name)) is not bool:
            raise ValueError(f"{name} must be a YAML boolean")
    for name in ("gflownet_loss_weight", "got_dag_loss_weight", "trajectory_memory_loss_weight",
                 "derived_category_loss_weight"):
        value = settings.get(name, 0.0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    for name in ("lr", "weight_decay", "min_lr_ratio", "dropout", "theta", "soft_moe_residual_scale",
                 "clip_grad_norm", "adam_beta1", "adam_beta2", "adam_eps"):
        value = settings.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if settings["min_lr_ratio"] > 1:
        raise ValueError("min_lr_ratio must be at most one")
    if any(settings[name] == 0 for name in ("lr", "adam_eps", "clip_grad_norm")):
        raise ValueError("learning rate, adam_eps and clip_grad_norm must be positive")
    if any(settings[name] >= 1 for name in ("dropout", "adam_beta1", "adam_beta2")):
        raise ValueError("dropout and AdamW beta values must be less than one")
    if type(settings.get("seed")) is not int or not 0 <= settings["seed"] <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    warmup = settings.get("warmup_steps")
    if warmup is not None and (type(warmup) is not int or warmup < 0):
        raise ValueError("warmup_steps must be null or a nonnegative integer")
    checkpoint_every = launcher["checkpoint_every"]
    if checkpoint_every is not None and (type(checkpoint_every) is not int or checkpoint_every < 0):
        raise ValueError("checkpoint_every must be null or a nonnegative integer")
    if type(launcher["cuda_visible_device"]) is not int or launcher["cuda_visible_device"] < 0:
        raise ValueError("cuda_visible_device must be a nonnegative device index")
    # Reuse the exact native argparse option registry to reject unknown model knobs.
    checker = subprocess.run(
        [sys.executable, "-B", "scripts/train.py", "--config", str(config), "--help"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
    )
    if checker.returncode:
        raise ValueError(checker.stderr.strip().splitlines()[-1] if checker.stderr else "invalid trainer YAML")
    return raw, launcher, settings


def _configured_path(config: Path, launcher: dict, key: str, *, must_exist: bool) -> Path:
    value = launcher[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"set {key} in metadata.prepared_launch")
    path = config.parent / Path(value).expanduser()
    if must_exist:
        return path.resolve(strict=True)
    if path.exists() or path.is_symlink():
        raise ValueError("choose a new output directory")
    return path.absolute()


def _plan(config, launcher, settings):
    manifest = _configured_path(config, launcher, "manifest", must_exist=True)
    output = _configured_path(config, launcher, "output", must_exist=False)
    with open_prepared_stream(manifest, launcher["manifest_sha256"]) as (_, admission):
        plan = build_plan(
            admission, manifest, output, config=config, batch_size=settings["batch_size"],
            max_accum=settings["grad_accum_steps"], seed=settings["seed"], precision=settings["precision"],
            warmup_steps=settings.get("warmup_steps"), checkpoint_every=launcher["checkpoint_every"],
        )
    return output, plan


def _launch(raw, launcher, output, plan):
    lease = _configured_path(config, launcher, "gpu_lease", must_exist=True)
    if not lease.is_file():
        raise ValueError("gpu_lease must refer to the existing GPU lock file")
    environment = os.environ | {
        "CUDA_VISIBLE_DEVICES": str(launcher["cuda_visible_device"]),
        "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1",
    }
    with lease.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        snapshot = output / "config.yml"
        snapshot.write_bytes(raw)
        for entry in plan["segments"]:
            argv = entry["argv"]
            argv[argv.index("--config") + 1] = str(snapshot)
        (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (output / "training.log").open("x", encoding="utf-8") as log:
            for segment in plan["segments"]:
                subprocess.run(segment["argv"], env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = inspect_run(plan)
        (output / "verification.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "plan", "train"), nargs="?", default="check")
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    parser.add_argument("--approve-manifest-sha256", help="Required for train after reviewing the filtered input.")
    opts = parser.parse_args()
    config = opts.config.resolve()
    try:
        raw, launcher, settings = _read_recipe(config)
        digest = hashlib.sha256(raw).hexdigest()
        if opts.mode == "train" and (
            opts.approve_manifest_sha256 is None
            or opts.approve_manifest_sha256 != launcher["manifest_sha256"]
        ):
            raise ValueError("train requires explicit approval matching the filtered manifest SHA-256")
        if opts.mode == "check":
            result = {"config_sha256": digest, "configuration_checked": True}
        else:
            output, plan = _plan(config, launcher, settings)
            plan["config_sha256"] = digest
            result = plan if opts.mode == "plan" else _launch(raw, launcher, output, plan)
    except (OSError, TypeError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
