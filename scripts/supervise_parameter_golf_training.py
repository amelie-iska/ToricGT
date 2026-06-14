#!/usr/bin/env python3
"""Keep a Parameter-Golf training run alive and schedule sidecar analyses.

The supervisor is intentionally boring: it does not tune hyperparameters by
itself and it never pauses training for analysis.  Its job is to keep exactly
one training tmux alive, restart from the newest checkpoint when the run dies
or stalls, and keep a non-blocking analysis/Codex-review sidecar moving.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


CHECKPOINT_RE = re.compile(r"random_order_step_(\d+)\.pt$")
ANALYSIS_RE = re.compile(r"step-(\d+)$")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train.parameter_golf_all_phases.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "amelie-iska-math"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "toricgt-parameter-golf"))
    parser.add_argument("--conda-env", default=os.environ.get("CONDA_ENV", "tokengt"))
    parser.add_argument("--conda-bin", default=os.environ.get("CONDA_BIN", "conda"))
    parser.add_argument("--train-session", default="toricgt_all_phases_live")
    parser.add_argument("--watch-session", default="toricgt_all_phases_analysis")
    parser.add_argument("--codex-prefix", default="toricgt_codex_review_all_phases")
    parser.add_argument("--codex-model", default=os.environ.get("CODEX_REVIEW_MODEL", "gpt-5-codex"))
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--log-root", default="")
    parser.add_argument("--analysis-root", default="")
    parser.add_argument("--state-path", default="")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--stale-log-seconds", type=float, default=1200.0)
    parser.add_argument("--stale-checkpoint-seconds", type=float, default=3600.0)
    parser.add_argument("--max-analysis-seconds", type=float, default=7200.0)
    parser.add_argument("--analysis-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--analysis-precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--target-bpb", type=float, default=float(os.environ.get("BPB_TARGET", "1.2")))
    parser.add_argument("--gate-step", type=int, default=int(os.environ.get("BPB_GATE_STEP", "4000")))
    parser.add_argument("--max-restarts", type=int, default=100)
    parser.add_argument("--max-analysis-iterations", type=int, default=100)
    parser.add_argument("--data-glob", default="data/curated_hf_shards/validation/*.parquet")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--simplex-samples", type=int, default=8)
    parser.add_argument("--geometry-records", type=int, default=4)
    parser.add_argument("--geometry-branches", type=int, default=6)
    parser.add_argument("--seed", type=int, default=10017)
    parser.add_argument(
        "--enable-codex-review",
        action="store_true",
        help="Opt in to automated Codex review subagents after periodic analysis.",
    )
    parser.add_argument(
        "--no-codex-review",
        action="store_true",
        help="Compatibility flag; Codex review subagents are disabled unless --enable-codex-review is set.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def config_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run(command: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def tmux_exact_target(session: str) -> str:
    return f"={session}"


def tmux_has(session: str) -> bool:
    return run(["tmux", "has-session", "-t", tmux_exact_target(session)]).returncode == 0


def tmux_kill(session: str) -> None:
    if tmux_has(session):
        run(["tmux", "kill-session", "-t", tmux_exact_target(session)])


def tmux_start(session: str, command: str, dry_run: bool) -> None:
    if dry_run:
        print(f"dry-run tmux {session}: {command}", flush=True)
        return
    run(["tmux", "new-session", "-d", "-s", session, command], check=True)


def latest_checkpoint(checkpoint_dir: Path) -> tuple[int, Path, float] | None:
    best: tuple[int, Path, float] | None = None
    for path in checkpoint_dir.glob("random_order_step_*.pt"):
        match = CHECKPOINT_RE.search(path.name)
        if not match:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        step = int(match.group(1))
        if best is None or step > best[0]:
            best = (step, path, mtime)
    return best


def highest_analysis_step(analysis_root: Path) -> int:
    best = 0
    if not analysis_root.exists():
        return best
    for path in analysis_root.glob("step-*"):
        if not path.is_dir():
            continue
        match = ANALYSIS_RE.search(path.name)
        if match:
            best = max(best, int(match.group(1)))
    return best


def load_checkpoint_bpb(path: Path) -> dict[str, float]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
    except Exception:
        return {}
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    if not isinstance(metrics, dict):
        return {}
    out: dict[str, float] = {}
    for key in (
        "train_bpb",
        "val_bpb",
        "best_val_bpb",
        "oai_competition/bpb",
        "oai_competition/best_bpb",
        "best_oai_competition_bpb",
    ):
        value = metrics.get(key)
        if isinstance(value, (float, int)):
            out[key] = float(value)
    return out


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"events": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"events": []}
    if not isinstance(data, dict):
        return {"events": []}
    data.setdefault("events", [])
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_event(state: dict[str, Any], name: str, **payload: Any) -> None:
    event = {"time_utc": utc_iso(), "event": name, **payload}
    events = state.setdefault("events", [])
    if isinstance(events, list):
        events.append(event)
        del events[:-200]
    print(json.dumps(event, sort_keys=True), flush=True)


def most_recent_log_mtime(log_root: Path) -> float:
    best = 0.0
    for path in log_root.glob("*/train.log"):
        try:
            best = max(best, path.stat().st_mtime)
        except OSError:
            continue
    return best


def start_training(args: argparse.Namespace, state: dict[str, Any], checkpoint: Path | None) -> None:
    restarts = int(state.get("restarts", 0))
    if restarts >= args.max_restarts:
        append_event(state, "restart_limit_reached", max_restarts=args.max_restarts)
        return
    if tmux_has(args.train_session):
        tmux_kill(args.train_session)
    cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cuda_alloc_env = []
    if cuda_alloc_conf:
        cuda_alloc_env.append(f"PYTORCH_CUDA_ALLOC_CONF={cuda_alloc_conf}")
    cas_env = []
    cas_toric_ideal_cert = os.environ.get("TORICGT_CAS_TORIC_IDEAL_CERT", "").strip()
    if cas_toric_ideal_cert:
        cas_env.append(f"TORICGT_CAS_TORIC_IDEAL_CERT={cas_toric_ideal_cert}")

    stamp = utc_stamp()
    run_name = args.run_name or args.run_id
    log_dir = Path(args.log_root) / f"restart_{restarts:03d}_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.conda_bin,
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "env",
        "PYTHONPATH=src",
        f"WANDB_ENTITY={args.wandb_entity}",
        f"WANDB_PROJECT={args.wandb_project}",
        f"WANDB_RUN_ID={args.run_id}",
        f"CONDA_BIN={args.conda_bin}",
        *cuda_alloc_env,
        *cas_env,
        "WANDB_RESUME=allow",
        "python",
        "scripts/train_parameter_golf_random_order.py",
        "--config",
        args.config,
        "--wandb",
        "--wandb-project",
        args.wandb_project,
        "--wandb-run-name",
        run_name,
    ]
    if checkpoint is not None:
        command.extend(["--resume", str(checkpoint)])
    command_text = shell_join(command)
    (log_dir / "train_command.sh").write_text(command_text + "\n", encoding="utf-8")
    tmux_start(
        args.train_session,
        f"cd {shlex.quote(str(Path.cwd()))} && {command_text} 2>&1 | tee {shlex.quote(str(log_dir / 'train.log'))}",
        dry_run=bool(args.dry_run),
    )
    state["restarts"] = restarts + 1
    state["active_train_log"] = str(log_dir / "train.log")
    state["last_train_start_utc"] = utc_iso()
    append_event(
        state,
        "training_started",
        session=args.train_session,
        resume=str(checkpoint) if checkpoint else "",
        log=str(log_dir / "train.log"),
    )


def watcher_age_seconds(state: dict[str, Any]) -> float:
    started = state.get("watcher_started_unix")
    if not isinstance(started, (float, int)):
        return 0.0
    return max(0.0, time.time() - float(started))


def start_watcher(args: argparse.Namespace, state: dict[str, Any], target_step: int) -> None:
    if int(state.get("analysis_iterations_started", 0)) >= args.max_analysis_iterations:
        append_event(state, "analysis_iteration_limit_reached", max_iterations=args.max_analysis_iterations)
        return
    if tmux_has(args.watch_session):
        return
    stamp = utc_stamp()
    log_dir = Path(args.log_root) / f"analysis_target_{target_step:08d}_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_path = f"{args.wandb_entity}/{args.wandb_project}/{args.run_id}"
    loop_state = os.environ.get("BPB_LOOP_STATE", str(Path(args.log_root) / "bpb_codex_loop_state.json"))
    loop_stop_file = os.environ.get("BPB_LOOP_STOP_FILE", str(Path(args.log_root) / "bpb_codex_loop_stop"))
    loop_name = os.environ.get("BPB_LOOP_NAME", "all_phases_supervised_watchdog")
    cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cuda_alloc_env = []
    if cuda_alloc_conf:
        cuda_alloc_env.append(f"PYTORCH_CUDA_ALLOC_CONF={cuda_alloc_conf}")
    codex_review_enabled = bool(args.enable_codex_review and not args.no_codex_review)
    codex_review_env = []
    if codex_review_enabled and args.codex_model:
        codex_review_env.append(f"CODEX_REVIEW_MODEL={args.codex_model}")
    command = [
        args.conda_bin,
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "env",
        "PYTHONPATH=src",
        f"BPB_TARGET={args.target_bpb}",
        f"BPB_MAX_REVIEW_ITERATIONS={args.max_analysis_iterations}",
        f"BPB_LOOP_STATE={loop_state}",
        f"CONDA_BIN={args.conda_bin}",
        *cuda_alloc_env,
        f"BPB_LOOP_STOP_FILE={loop_stop_file}",
        f"BPB_LOOP_NAME={loop_name}",
        "CODEX_REVIEW_FALLBACK_CONTINUE=0",
        "CODEX_REVIEW_TIMEOUT_SECONDS=1200",
        *codex_review_env,
        "python",
        "scripts/watch_training_analysis.py",
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--start-step",
        "0",
        "--target-step",
        str(target_step),
        "--poll-seconds",
        str(max(10.0, args.poll_seconds)),
        "--run-path",
        run_path,
        "--output-root",
        args.analysis_root,
        "--config",
        args.config,
        "--data-glob",
        args.data_glob,
        "--seq-len",
        str(args.seq_len),
        "--simplex-samples",
        str(args.simplex_samples),
        "--geometry-records",
        str(args.geometry_records),
        "--geometry-branches",
        str(args.geometry_branches),
        "--device",
        args.analysis_device,
        "--precision",
        args.analysis_precision,
        "--seed",
        str(args.seed),
        "--target-bpb",
        str(args.target_bpb),
        "--gate-step",
        str(args.gate_step),
        "--training-tmux",
        args.train_session,
        "--cas-audit",
        "--cas-audit-all-exact-cas",
        "--cas-audit-require-cas",
    ]
    if codex_review_enabled:
        command.extend(
            [
                "--codex-review-hook",
                "scripts/codex_training_review_resume.sh",
                "--codex-review-tmux-prefix",
                args.codex_prefix,
            ]
        )
    command_text = shell_join(command)
    (log_dir / "watch_command.sh").write_text(command_text + "\n", encoding="utf-8")
    tmux_start(
        args.watch_session,
        f"cd {shlex.quote(str(Path.cwd()))} && {command_text} 2>&1 | tee {shlex.quote(str(log_dir / 'watcher.log'))}",
        dry_run=bool(args.dry_run),
    )
    state["watcher_started_unix"] = time.time()
    state["watcher_target_step"] = target_step
    state["analysis_iterations_started"] = int(state.get("analysis_iterations_started", 0)) + 1
    append_event(
        state,
        "watcher_started",
        session=args.watch_session,
        target_step=target_step,
        log=str(log_dir / "watcher.log"),
        codex_review=codex_review_enabled,
        codex_model=args.codex_model if codex_review_enabled else "",
    )


def should_restart(args: argparse.Namespace, state: dict[str, Any], checkpoint: tuple[int, Path, float] | None) -> str:
    if not tmux_has(args.train_session):
        return "training_tmux_missing"
    now = time.time()
    log_mtime = most_recent_log_mtime(Path(args.log_root))
    ckpt_mtime = checkpoint[2] if checkpoint else 0.0
    if log_mtime > 0 and now - log_mtime <= args.stale_log_seconds:
        return ""
    if ckpt_mtime > 0 and now - ckpt_mtime <= args.stale_checkpoint_seconds:
        return ""
    age_log = now - log_mtime if log_mtime > 0 else None
    age_ckpt = now - ckpt_mtime if ckpt_mtime > 0 else None
    state["last_stale_probe"] = {"log_age_seconds": age_log, "checkpoint_age_seconds": age_ckpt}
    return "training_stale"


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    if not args.checkpoint_dir:
        args.checkpoint_dir = str(config_get(config, "training", "checkpoint_dir", default="checkpoints/parameter_golf_all_phases"))
    if not args.log_root:
        args.log_root = f"logs/parameter_golf_all_phases/{args.run_id}/supervisor"
    if not args.analysis_root:
        args.analysis_root = f"outputs/post_resume_analysis/{args.run_id}"
    if not args.state_path:
        args.state_path = str(Path(args.log_root) / "supervisor_state.json")

    Path(args.log_root).mkdir(parents=True, exist_ok=True)
    Path(args.analysis_root).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state_path)
    state = load_state(state_path)
    state.update(
        {
            "run_id": args.run_id,
            "wandb_run": f"https://wandb.ai/{args.wandb_entity}/{args.wandb_project}/runs/{args.run_id}",
            "config": args.config,
            "checkpoint_dir": args.checkpoint_dir,
            "conda_bin": args.conda_bin,
            "train_session": args.train_session,
            "watch_session": args.watch_session,
            "target_bpb": args.target_bpb,
            "gate_step": args.gate_step,
            "updated_utc": utc_iso(),
        }
    )
    append_event(state, "supervisor_started", log_root=args.log_root, analysis_root=args.analysis_root)
    if tmux_has(args.train_session):
        append_event(state, "adopted_existing_training", session=args.train_session)
    if tmux_has(args.watch_session) and "watcher_started_unix" not in state:
        state["watcher_started_unix"] = time.time()
        state["watcher_target_step"] = state.get("watcher_target_step", "adopted")
        append_event(state, "adopted_existing_watcher", session=args.watch_session)
    save_state(state_path, state)

    while True:
        checkpoint = latest_checkpoint(Path(args.checkpoint_dir))
        if checkpoint:
            step, ckpt_path, ckpt_mtime = checkpoint
            state["latest_checkpoint_step"] = step
            state["latest_checkpoint"] = str(ckpt_path)
            state["latest_checkpoint_mtime"] = ckpt_mtime
            bpb = load_checkpoint_bpb(ckpt_path)
            if bpb:
                state["latest_checkpoint_bpb"] = bpb
                best = min(
                    [value for key, value in bpb.items() if "bpb" in key],
                    default=float("inf"),
                )
                if best <= args.target_bpb:
                    append_event(state, "target_bpb_reached", checkpoint=str(ckpt_path), best_bpb=best)
                    save_state(state_path, state)
                    break

        reason = should_restart(args, state, checkpoint)
        if reason:
            resume = checkpoint[1] if checkpoint else None
            append_event(state, "restart_required", reason=reason, resume=str(resume) if resume else "")
            start_training(args, state, resume)

        if tmux_has(args.watch_session) and watcher_age_seconds(state) > args.max_analysis_seconds:
            append_event(
                state,
                "watcher_timeout_kill",
                session=args.watch_session,
                age_seconds=watcher_age_seconds(state),
                target_step=state.get("watcher_target_step"),
            )
            tmux_kill(args.watch_session)

        if not tmux_has(args.watch_session):
            latest_step = checkpoint[0] if checkpoint else 0
            analyzed_step = highest_analysis_step(Path(args.analysis_root))
            state["highest_analysis_step"] = analyzed_step
            base_step = max(latest_step, analyzed_step)
            next_target = max(args.checkpoint_interval, ((base_step // args.checkpoint_interval) + 1) * args.checkpoint_interval)
            start_watcher(args, state, next_target)

        state["updated_utc"] = utc_iso()
        save_state(state_path, state)
        time.sleep(max(5.0, args.poll_seconds))


if __name__ == "__main__":
    main()
