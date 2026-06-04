#!/usr/bin/env python3
"""Gate Seq4096 FineWeb runs at 4K and relaunch from the best checkpoint.

If the run reaches the gate step without a validation BPB at or below target,
this watcher kills the active training tmux and starts a fresh recovery run
from the best checkpoint observed at or before the gate.  The recovery gets a
fresh W&B run id and its own analysis/mirror sidecars so replayed step numbers
remain visible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<loss>[0-9.eE+-]+)"
    r"\s+val_bpb:(?P<bpb>[0-9.eE+-]+)"
)
TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+train_loss:(?P<loss>[0-9.eE+-]+)"
    r"(?:.*?\btrain_bpb:(?P<bpb>[0-9.eE+-]+))?"
)
CHECKPOINT_RE_TEMPLATE = "{run_id}_step_{step:06d}.pt"


@dataclass(frozen=True)
class ValRow:
    step: int
    total: int
    val_loss: float
    val_bpb: float


@dataclass(frozen=True)
class TrainRow:
    step: int
    total: int
    train_loss: float
    train_bpb: float = float("nan")


@dataclass(frozen=True)
class ParsedLog:
    train_rows: list[TrainRow]
    val_rows: list[ValRow]

    @property
    def latest_step(self) -> int:
        return max([row.step for row in self.train_rows] + [row.step for row in self.val_rows], default=0)


@dataclass(frozen=True)
class RecoveryLaunch:
    training_command: list[str]
    training_shell: str


@dataclass(frozen=True)
class PreemptiveGateRisk:
    analysis_root: str
    latest_analysis_step: int
    latest_projected_target_step: float
    latest_best_val_bpb: float
    latest_recent_val_slope: float
    missed_projection_count: int
    required_patience: int
    gate_step: int
    target_bpb: float


@dataclass(frozen=True)
class RecoveryControls:
    train_batch_tokens: int
    tied_embed_lr: float
    matrix_lr: float
    scalar_lr: float
    muon_momentum: float
    muon_momentum_warmup_steps: int
    muon_momentum_warmup_start: float
    grad_clip_norm: float | None
    policy: str = "base_controls"
    advanced_metric_policy: str = "primary_bpb_clean"
    rationale: tuple[str, ...] = ()
    bigram_bias: bool = False
    bigram_bias_lr: float = 0.05
    bigram_bias_init_from_data: bool = False
    bigram_bias_init_tokens: int = 100_000_000
    bigram_bias_init_alpha: float = 0.1
    bigram_bias_init_strength: float = 0.35
    bigram_bias_scale: float = 1.0

    def launch_dict(self) -> dict[str, Any]:
        return {
            "train_batch_tokens": int(self.train_batch_tokens),
            "tied_embed_lr": float(self.tied_embed_lr),
            "matrix_lr": float(self.matrix_lr),
            "scalar_lr": float(self.scalar_lr),
            "muon_momentum": float(self.muon_momentum),
            "muon_momentum_warmup_steps": int(self.muon_momentum_warmup_steps),
            "muon_momentum_warmup_start": float(self.muon_momentum_warmup_start),
            "grad_clip_norm": None if self.grad_clip_norm is None else float(self.grad_clip_norm),
            "policy": self.policy,
            "advanced_metric_policy": self.advanced_metric_policy,
            "rationale": list(self.rationale),
            "bigram_bias": bool(self.bigram_bias),
            "bigram_bias_lr": float(self.bigram_bias_lr),
            "bigram_bias_init_from_data": bool(self.bigram_bias_init_from_data),
            "bigram_bias_init_tokens": int(self.bigram_bias_init_tokens),
            "bigram_bias_init_alpha": float(self.bigram_bias_init_alpha),
            "bigram_bias_init_strength": float(self.bigram_bias_init_strength),
            "bigram_bias_scale": float(self.bigram_bias_scale),
        }


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sanitize_tmux_name(value: str, limit: int = 96) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)
    return clean[:limit].rstrip("_") or "toricgt_4k_recovery"


def build_recovery_run_id(parent_run_id: str, restart_index: int, stamp: str, max_len: int = 95) -> str:
    """Build a compact W&B-safe id instead of recursively appending parents."""

    parent = str(parent_run_id or "")
    if "seq4096" in parent:
        stem = "toricgt_seq4096"
    else:
        stem = parent.split("_4k_recovery_r", 1)[0] or "toricgt"
        stem = re.sub(r"[^A-Za-z0-9_.:-]+", "_", stem).strip("_") or "toricgt"
        stem = stem[:40].rstrip("_") or "toricgt"
    suffix = f"_4k_recovery_r{int(restart_index)}_{stamp}"
    budget = max(8, int(max_len) - len(suffix))
    return f"{stem[:budget].rstrip('_')}{suffix}"


def parse_seq4096_log(path: Path | str) -> ParsedLog:
    train: dict[int, TrainRow] = {}
    vals: dict[int, ValRow] = {}
    path = Path(path)
    if not path.exists():
        return ParsedLog([], [])
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        val = VAL_RE.search(line)
        if val:
            step = int(val.group("step"))
            vals[step] = ValRow(
                step=step,
                total=int(val.group("total")),
                val_loss=float(val.group("loss")),
                val_bpb=float(val.group("bpb")),
            )
            continue
        train_match = TRAIN_RE.search(line)
        if train_match:
            step = int(train_match.group("step"))
            train[step] = TrainRow(
                step=step,
                total=int(train_match.group("total")),
                train_loss=float(train_match.group("loss")),
                train_bpb=finite_float(train_match.group("bpb")),
            )
    return ParsedLog(
        train_rows=[train[key] for key in sorted(train)],
        val_rows=[vals[key] for key in sorted(vals)],
    )


def select_best_validation(rows: list[ValRow], gate_step: int) -> ValRow | None:
    candidates = [row for row in rows if row.step <= int(gate_step) and math.isfinite(row.val_bpb)]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row.val_bpb, -row.step))


def select_recovery_validation(
    rows: list[ValRow],
    gate_step: int,
    min_recovery_runway_steps: int = 500,
) -> ValRow | None:
    """Select the checkpoint to restart from after a missed gate.

    If the gate step itself is the best-but-still-missed checkpoint, restarting
    there gives the recovery run no training room before the same gate.  Prefer
    the best validation with enough room to retrain before the gate; fall back
    to the best earlier validation only when no roomy validation exists, and
    then to the inclusive best only when no earlier validation exists.
    """

    min_step = int(gate_step) - max(1, int(min_recovery_runway_steps))
    roomy = [row for row in rows if row.step <= min_step and math.isfinite(row.val_bpb)]
    if roomy:
        return min(roomy, key=lambda row: (row.val_bpb, -row.step))
    before_gate = [row for row in rows if row.step < int(gate_step) and math.isfinite(row.val_bpb)]
    if before_gate:
        return min(before_gate, key=lambda row: (row.val_bpb, -row.step))
    return select_best_validation(rows, gate_step)


def checkpoint_for_step(checkpoint_dir: Path | str, run_id: str, step: int) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    exact = checkpoint_dir / CHECKPOINT_RE_TEMPLATE.format(run_id=run_id, step=int(step))
    if exact.exists():
        return exact.resolve()
    candidates = sorted(checkpoint_dir.glob(f"*_step_{int(step):06d}.pt"))
    return candidates[-1].resolve() if candidates else None


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def analysis_step_from_path(path: Path, payload: dict[str, Any]) -> int:
    for key in ("checkpoint_step", "latest_step"):
        value = payload.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"step-(\d+)", str(path))
    return int(match.group(1)) if match else 0


def load_analysis_status(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_preemptive_gate_risk(
    analysis_root: Path | str,
    *,
    gate_step: int,
    target_bpb: float,
    min_step: int,
    patience: int,
) -> PreemptiveGateRisk | None:
    """Load repeated validation-ETA gate risk from completed analysis reports."""

    root = Path(analysis_root)
    if not root.exists():
        return None
    rows: list[tuple[int, dict[str, Any]]] = []
    for status_path in sorted(root.glob("step-*/analysis_status.json")):
        payload = load_analysis_status(status_path)
        if not payload:
            continue
        step = analysis_step_from_path(status_path, payload)
        if step >= int(min_step):
            rows.append((step, payload))
    if not rows:
        return None
    rows.sort(key=lambda item: item[0])
    missed = 0
    for step, payload in reversed(rows):
        projected = finite_float(payload.get("projected_target_step_from_val"))
        best_val = finite_float(payload.get("best_val_bpb"))
        recent_slope = finite_float(payload.get("val_bpb_recent_slope_per_100_steps"))
        is_risk = (
            best_val > float(target_bpb)
            and recent_slope < 0.0
            and projected > float(gate_step)
        )
        if not is_risk:
            break
        missed += 1
    latest_step, latest = rows[-1]
    return PreemptiveGateRisk(
        analysis_root=str(root),
        latest_analysis_step=int(latest_step),
        latest_projected_target_step=finite_float(latest.get("projected_target_step_from_val")),
        latest_best_val_bpb=finite_float(latest.get("best_val_bpb")),
        latest_recent_val_slope=finite_float(latest.get("val_bpb_recent_slope_per_100_steps")),
        missed_projection_count=int(missed),
        required_patience=max(1, int(patience)),
        gate_step=int(gate_step),
        target_bpb=float(target_bpb),
    )


def should_preempt_for_gate_risk(risk: PreemptiveGateRisk | None) -> bool:
    return bool(risk and risk.missed_projection_count >= risk.required_patience)


def round_up_to_multiple(value: float, multiple: int) -> int:
    if multiple <= 0:
        return int(math.ceil(value))
    return int(math.ceil(float(value) / float(multiple)) * multiple)


def latest_train_bpb_at_or_before(parsed: ParsedLog, step: int) -> float:
    candidates = [
        row.train_bpb
        for row in parsed.train_rows
        if row.step <= int(step) and math.isfinite(row.train_bpb)
    ]
    return candidates[-1] if candidates else float("nan")


def plan_metric_driven_recovery_controls(
    base: RecoveryControls,
    *,
    parsed: ParsedLog,
    target_bpb: float,
    gate_step: int,
    projected_target_step: float = float("nan"),
    max_train_batch_tokens: int | None = None,
    validation_gap_threshold: float = 0.04,
    low_train_bpb_margin: float = 0.0,
    enabled: bool = True,
) -> RecoveryControls:
    """Adapt the next recovery launch to BPB and structural-analysis signals.

    The Seq4096 competition runner does not carry the richer GraphCG/Slepian
    losses.  When the analyses indicate structural pressure, the safe action
    for the primary BPB run is to keep that runner BPB-clean while using its
    train/validation geometry to choose optimizer controls.
    """

    if not enabled:
        return replace(base, policy="base_controls", rationale=("advanced metric controls disabled",))

    latest_val = parsed.val_rows[-1] if parsed.val_rows else None
    if latest_val is None:
        return replace(base, policy="base_controls", rationale=("waiting for validation BPB",))

    train_bpb = latest_train_bpb_at_or_before(parsed, latest_val.step)
    validation_gap = (
        latest_val.val_bpb - train_bpb if math.isfinite(train_bpb) else float("nan")
    )
    projected_miss = (
        math.isfinite(projected_target_step)
        and projected_target_step > float(gate_step)
        and latest_val.step < int(gate_step)
        and latest_val.val_bpb > float(target_bpb)
    )
    train_already_low = (
        math.isfinite(train_bpb)
        and train_bpb <= float(target_bpb) + float(low_train_bpb_margin)
    )
    validation_lagging = (
        math.isfinite(validation_gap)
        and validation_gap >= float(validation_gap_threshold)
        and latest_val.val_bpb > float(target_bpb)
    )

    if train_already_low and validation_lagging:
        batch_cap = (
            int(max_train_batch_tokens)
            if max_train_batch_tokens is not None and int(max_train_batch_tokens) > 0
            else int(base.train_batch_tokens)
        )
        raised_batch = round_up_to_multiple(base.train_batch_tokens * 1.07, 65_536)
        return replace(
            base,
            train_batch_tokens=max(base.train_batch_tokens, min(batch_cap, raised_batch)),
            tied_embed_lr=round(max(0.028, base.tied_embed_lr * 0.875), 6),
            matrix_lr=round(max(0.017, base.matrix_lr * 0.90), 6),
            scalar_lr=round(max(0.017, base.scalar_lr * 0.90), 6),
            muon_momentum_warmup_steps=max(base.muon_momentum_warmup_steps, 500),
            policy="validation_gap_recapture",
            advanced_metric_policy="graphcg_slepian_sidecar_primary_bpb_clean",
            rationale=(
                f"train BPB {train_bpb:.4f} is already at/below target while validation BPB "
                f"{latest_val.val_bpb:.4f} lags by {validation_gap:.4f}",
                "increase effective batch for steadier validation transfer",
                "lower tied/matrix/scalar learning rates instead of pushing train BPB harder",
                "keep GraphCG/Slepian/topology losses in sidecar transfer until the competition checkpoint is preserved",
            ),
        )

    if projected_miss:
        batch_cap = (
            int(max_train_batch_tokens)
            if max_train_batch_tokens is not None and int(max_train_batch_tokens) > 0
            else int(base.train_batch_tokens)
        )
        raised_batch = round_up_to_multiple(base.train_batch_tokens * 1.07, 65_536)
        if not base.bigram_bias and base.tied_embed_lr >= 0.0395:
            return replace(
                base,
                train_batch_tokens=max(base.train_batch_tokens, min(batch_cap, raised_batch)),
                bigram_bias=True,
                bigram_bias_init_from_data=False,
                policy="lexical_transition_bias_recapture",
                advanced_metric_policy="bigram_bias_primary_bpb_clean_structural_sidecars",
                rationale=(
                    f"validation projects target at step {projected_target_step:.1f}, beyond gate {gate_step}",
                    "tied-embedding LR is already at the recapture cap",
                    "enable a zero-initialized learned bigram transition-bias head so initial BPB is preserved",
                    "keep GraphCG/Slepian/topology losses in sidecar transfer while adding only BPB-native lexical bias",
                ),
            )
        return replace(
            base,
            train_batch_tokens=max(base.train_batch_tokens, min(batch_cap, raised_batch)),
            tied_embed_lr=round(min(0.040, base.tied_embed_lr * 1.05), 6),
            policy="bpb_velocity_recapture",
            advanced_metric_policy="proposal_guided_bpb_recapture_structural_sidecars",
            rationale=(
                f"validation projects target at step {projected_target_step:.1f}, beyond gate {gate_step}",
                "increase effective batch and tied-embedding LR to accelerate validation BPB",
                "hold matrix/scalar LR to avoid disrupting the current basin",
                "use GraphCG/Slepian/topology diagnostics as sidecar transfer signals, not heavy primary losses",
            ),
        )

    return replace(
        base,
        policy="base_controls",
        advanced_metric_policy="monitor_structural_sidecars",
        rationale=("no metric-driven recovery-control change selected",),
    )


def shell_env(env: dict[str, Any]) -> str:
    return " ".join(f"{key}={shlex.quote(str(value))}" for key, value in env.items())


def build_recovery_launch(
    *,
    repo_root: Path,
    parameter_golf_root: Path,
    run_id: str,
    checkpoint_dir: Path,
    log_path: Path,
    resume_checkpoint: Path,
    seed: int,
    target_bpb: float,
    python: str = "/home/iska/miniconda3/envs/tokengt/bin/python",
    iterations: int = 20_000,
    train_seq_len: int = 4096,
    train_batch_tokens: int = 393_216,
    tied_embed_lr: float = 0.032,
    matrix_lr: float = 0.018,
    scalar_lr: float = 0.018,
    muon_momentum: float = 0.985,
    muon_momentum_warmup_steps: int = 600,
    muon_momentum_warmup_start: float = 0.90,
    grad_clip_norm: float | None = None,
    warmdown_iters: int = 2500,
    val_loss_every: int = 250,
    train_log_every: int = 50,
    checkpoint_every: int = 250,
    wandb_project: str = "toricgt-parameter-golf",
    wandb_entity: str = "amelie-iska-math",
    bigram_bias: bool = False,
    bigram_bias_lr: float = 0.05,
    bigram_bias_init_from_data: bool = False,
    bigram_bias_init_tokens: int = 100_000_000,
    bigram_bias_init_alpha: float = 0.1,
    bigram_bias_init_strength: float = 0.35,
    bigram_bias_scale: float = 1.0,
) -> RecoveryLaunch:
    _ = repo_root
    env = {
        "RUN_ID": run_id,
        "WANDB_RUN_ID": run_id,
        "WANDB_RUN_NAME": run_id,
        "WANDB_PROJECT": wandb_project,
        "WANDB_ENTITY": wandb_entity,
        "WANDB_RESUME": "allow",
        "TARGET_BPB": target_bpb,
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": 1024,
        "TRAIN_SEQ_LEN": train_seq_len,
        "TRAIN_BATCH_TOKENS": train_batch_tokens,
        "TIED_EMBED_LR": tied_embed_lr,
        "MATRIX_LR": matrix_lr,
        "SCALAR_LR": scalar_lr,
        "MUON_MOMENTUM": muon_momentum,
        "MUON_MOMENTUM_WARMUP_STEPS": muon_momentum_warmup_steps,
        "MUON_MOMENTUM_WARMUP_START": muon_momentum_warmup_start,
        "WARMDOWN_ITERS": warmdown_iters,
        "MAX_WALLCLOCK_SECONDS": 0,
        "VAL_LOSS_EVERY": val_loss_every,
        "TRAIN_LOG_EVERY": train_log_every,
        "ITERATIONS": iterations,
        "CHECKPOINT_EVERY": checkpoint_every,
        "CHECKPOINT_DIR": checkpoint_dir,
        "SEED": seed,
        "WANDB_LOG_EVERY": 1,
        "RESUME_CHECKPOINT": resume_checkpoint,
        "RESET_OPTIMIZER_ON_RESUME": 1,
        "RESET_RNG_ON_RESUME": 1,
        "RESET_LOADER_ON_RESUME": 1,
    }
    if grad_clip_norm is not None:
        env["GRAD_CLIP_NORM"] = grad_clip_norm
    if bigram_bias:
        env.update(
            {
                "BIGRAM_BIAS": 1,
                "BIGRAM_BIAS_LR": bigram_bias_lr,
                "BIGRAM_BIAS_SCALE": bigram_bias_scale,
                "BIGRAM_BIAS_INIT_FROM_DATA": 1 if bigram_bias_init_from_data else 0,
                "BIGRAM_BIAS_INIT_TOKENS": int(bigram_bias_init_tokens),
                "BIGRAM_BIAS_INIT_ALPHA": bigram_bias_init_alpha,
                "BIGRAM_BIAS_INIT_STRENGTH": bigram_bias_init_strength,
            }
        )
    training_command = [f"{key}={value}" for key, value in env.items()] + [
        python,
        "-u",
        "records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py",
    ]
    shell = (
        f"mkdir -p {shlex.quote(str(log_path.parent))} {shlex.quote(str(checkpoint_dir))} && "
        f"cd {shlex.quote(str(parameter_golf_root))} && "
        f"export {shell_env(env)} && "
        f"{shlex.quote(str(python))} -u records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py "
        f"2>&1 | tee -a {shlex.quote(str(log_path))}"
    )
    return RecoveryLaunch(training_command=training_command, training_shell=shell)


def run(command: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def tmux_has(session: str) -> bool:
    return run(["tmux", "has-session", "-t", session]).returncode == 0


def tmux_kill(session: str) -> None:
    if tmux_has(session):
        run(["tmux", "kill-session", "-t", session])


def tmux_start(session: str, command: str, dry_run: bool) -> None:
    if dry_run:
        print(f"dry_run_tmux_start {session}: {command}", flush=True)
        return
    run(["tmux", "new-session", "-d", "-s", session, command], check=True)


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_analysis_shell(
    *,
    repo_root: Path,
    run_id: str,
    train_tmux: str,
    checkpoint_dir: Path,
    log_path: Path,
    target_bpb: float,
    start_step: int,
    python: str,
    wandb_entity: str,
    wandb_project: str,
) -> str:
    analysis_log = repo_root / "logs" / f"{run_id}.analysis_watcher.txt"
    output_root = repo_root / "outputs" / "post_resume_analysis" / run_id
    env = {
        "PYTHONPATH": "src",
        "WANDB_PROJECT": wandb_project,
        "WANDB_ENTITY": wandb_entity,
        "BPB_LOOP_STATE": repo_root / "outputs" / f"{run_id}_bpb_codex_loop_state.json",
        "BPB_LOOP_STOP_FILE": repo_root / "outputs" / f"{run_id}_bpb_codex_loop_stop",
    }
    command = (
        f"cd {shlex.quote(str(repo_root))} && export {shell_env(env)} && "
        f"{shlex.quote(str(python))} scripts/watch_seq4096_analysis.py "
        f"--checkpoint-dir {shlex.quote(str(checkpoint_dir))} "
        f"--log {shlex.quote(str(log_path))} "
        f"--run-path {shlex.quote(f'{wandb_entity}/{wandb_project}/{run_id}')} "
        f"--output-root {shlex.quote(str(output_root))} "
        f"--start-step {int(start_step)} --interval-steps 250 --poll-seconds 30 "
        f"--target-bpb {float(target_bpb)} --training-tmux {shlex.quote(train_tmux)} "
        f"2>&1 | tee -a {shlex.quote(str(analysis_log))}"
    )
    return command


def build_dense_mirror_shell(*, repo_root: Path, run_id: str, log_path: Path, target_bpb: float, python: str) -> str:
    mirror_log = repo_root / "logs" / f"{run_id}.wandb_mirror.txt"
    diagnostics_json = repo_root / "logs" / f"{run_id}.full_diag.latest.json"
    return (
        f"cd {shlex.quote(str(repo_root))} && export PYTHONPATH=src && "
        f"{shlex.quote(str(python))} scripts/mirror_fineweb_log_to_wandb.py "
        f"--log {shlex.quote(str(log_path))} --run-id {shlex.quote(run_id)} "
        f"--target-bpb {float(target_bpb)} --poll-seconds 15 "
        f"--diagnostics-json {shlex.quote(str(diagnostics_json))} "
        f"2>&1 | tee -a {shlex.quote(str(mirror_log))}"
    )


def build_full_diag_shell(*, repo_root: Path, run_id: str, log_path: Path, target_bpb: float, python: str) -> str:
    diag_log = repo_root / "logs" / f"{run_id}.full_diag.txt"
    diagnostics_json = repo_root / "logs" / f"{run_id}.full_diag.latest.json"
    return (
        f"cd {shlex.quote(str(repo_root))} && export PYTHONPATH=src && "
        f"{shlex.quote(str(python))} scripts/mirror_fineweb_full_diagnostics_to_wandb.py "
        f"--log {shlex.quote(str(log_path))} --run-id {shlex.quote(run_id)} "
        f"--target-bpb {float(target_bpb)} --poll-seconds 60 "
        f"--output-json {shlex.quote(str(diagnostics_json))} "
        f"2>&1 | tee -a {shlex.quote(str(diag_log))}"
    )


def build_gate_shell(
    *,
    repo_root: Path,
    run_id: str,
    train_tmux: str,
    checkpoint_dir: Path,
    log_path: Path,
    target_bpb: float,
    gate_step: int,
    restart_index: int,
    max_restarts: int,
    min_recovery_runway_steps: int,
    recovery_train_batch_tokens: int,
    recovery_tied_embed_lr: float,
    recovery_matrix_lr: float,
    recovery_scalar_lr: float,
    recovery_muon_momentum: float,
    recovery_muon_momentum_warmup_steps: int,
    recovery_muon_momentum_warmup_start: float,
    recovery_grad_clip_norm: float | None,
    recovery_bigram_bias: bool,
    recovery_bigram_bias_lr: float,
    recovery_bigram_bias_init_from_data: bool,
    recovery_bigram_bias_init_tokens: int,
    recovery_bigram_bias_init_alpha: float,
    recovery_bigram_bias_init_strength: float,
    recovery_bigram_bias_scale: float,
    preempt_on_projected_miss: bool,
    preempt_min_step: int,
    preempt_patience: int,
    advanced_metric_controls: bool,
    recovery_max_train_batch_tokens: int,
    validation_gap_threshold: float,
    low_train_bpb_margin: float,
    python: str,
) -> str:
    gate_log = repo_root / "logs" / f"{run_id}.4k_gate.txt"
    state_path = repo_root / "outputs" / f"{run_id}.4k_recovery_state.json"
    grad_clip_arg = (
        ""
        if recovery_grad_clip_norm is None
        else f"--recovery-grad-clip-norm {float(recovery_grad_clip_norm)} "
    )
    preempt_arg = (
        f"--preempt-on-projected-miss --preempt-min-step {int(preempt_min_step)} "
        f"--preempt-patience {int(preempt_patience)} "
        if preempt_on_projected_miss
        else ""
    )
    advanced_metric_arg = (
        ""
        if advanced_metric_controls
        else "--no-advanced-metric-controls "
    )
    bigram_arg = ""
    if recovery_bigram_bias:
        bigram_arg = (
            f"--recovery-bigram-bias "
            f"--recovery-bigram-bias-lr {float(recovery_bigram_bias_lr)} "
            f"--recovery-bigram-bias-init-tokens {int(recovery_bigram_bias_init_tokens)} "
            f"--recovery-bigram-bias-init-alpha {float(recovery_bigram_bias_init_alpha)} "
            f"--recovery-bigram-bias-init-strength {float(recovery_bigram_bias_init_strength)} "
            f"--recovery-bigram-bias-scale {float(recovery_bigram_bias_scale)} "
        )
        if recovery_bigram_bias_init_from_data:
            bigram_arg += "--recovery-bigram-bias-init-from-data "
    return (
        f"cd {shlex.quote(str(repo_root))} && export PYTHONPATH=src && "
        f"{shlex.quote(str(python))} scripts/watch_seq4096_4k_recovery.py "
        f"--log {shlex.quote(str(log_path))} --checkpoint-dir {shlex.quote(str(checkpoint_dir))} "
        f"--run-id {shlex.quote(run_id)} --train-tmux {shlex.quote(train_tmux)} "
        f"--target-bpb {float(target_bpb)} --gate-step {int(gate_step)} "
        f"--restart-index {int(restart_index)} --max-restarts {int(max_restarts)} "
        f"--min-recovery-runway-steps {int(min_recovery_runway_steps)} "
        f"--recovery-train-batch-tokens {int(recovery_train_batch_tokens)} "
        f"--recovery-tied-embed-lr {float(recovery_tied_embed_lr)} "
        f"--recovery-matrix-lr {float(recovery_matrix_lr)} "
        f"--recovery-scalar-lr {float(recovery_scalar_lr)} "
        f"--recovery-muon-momentum {float(recovery_muon_momentum)} "
        f"--recovery-muon-momentum-warmup-steps {int(recovery_muon_momentum_warmup_steps)} "
        f"--recovery-muon-momentum-warmup-start {float(recovery_muon_momentum_warmup_start)} "
        f"{grad_clip_arg}"
        f"{bigram_arg}"
        f"{preempt_arg}"
        f"{advanced_metric_arg}"
        f"--recovery-max-train-batch-tokens {int(recovery_max_train_batch_tokens)} "
        f"--validation-gap-threshold {float(validation_gap_threshold)} "
        f"--low-train-bpb-margin {float(low_train_bpb_margin)} "
        f"--state {shlex.quote(str(state_path))} "
        f"2>&1 | tee -a {shlex.quote(str(gate_log))}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-tmux", required=True)
    parser.add_argument("--target-bpb", type=float, default=1.2)
    parser.add_argument("--gate-step", type=int, default=4000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-restarts", type=int, default=6)
    parser.add_argument("--restart-index", type=int, default=0)
    parser.add_argument("--min-recovery-runway-steps", type=int, default=500)
    parser.add_argument("--state", default="")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--parameter-golf-root", default="amelie-iska/parameter-golf")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "toricgt-parameter-golf"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "amelie-iska-math"))
    parser.add_argument("--seed", type=int, default=7331)
    parser.add_argument("--recovery-train-batch-tokens", type=int, default=393_216)
    parser.add_argument("--recovery-tied-embed-lr", type=float, default=0.032)
    parser.add_argument("--recovery-matrix-lr", type=float, default=0.018)
    parser.add_argument("--recovery-scalar-lr", type=float, default=0.018)
    parser.add_argument("--recovery-muon-momentum", type=float, default=0.985)
    parser.add_argument("--recovery-muon-momentum-warmup-steps", type=int, default=600)
    parser.add_argument("--recovery-muon-momentum-warmup-start", type=float, default=0.90)
    parser.add_argument("--recovery-grad-clip-norm", type=float, default=None)
    parser.add_argument("--recovery-bigram-bias", action="store_true")
    parser.add_argument("--recovery-bigram-bias-lr", type=float, default=0.05)
    parser.add_argument("--recovery-bigram-bias-init-from-data", action="store_true")
    parser.add_argument("--recovery-bigram-bias-init-tokens", type=int, default=100_000_000)
    parser.add_argument("--recovery-bigram-bias-init-alpha", type=float, default=0.1)
    parser.add_argument("--recovery-bigram-bias-init-strength", type=float, default=0.35)
    parser.add_argument("--recovery-bigram-bias-scale", type=float, default=1.0)
    parser.add_argument("--analysis-root", default="")
    parser.add_argument("--preempt-on-projected-miss", action="store_true")
    parser.add_argument("--preempt-min-step", type=int, default=2500)
    parser.add_argument("--preempt-patience", type=int, default=2)
    parser.add_argument("--no-advanced-metric-controls", action="store_true")
    parser.add_argument("--recovery-max-train-batch-tokens", type=int, default=983_040)
    parser.add_argument("--validation-gap-threshold", type=float, default=0.04)
    parser.add_argument("--low-train-bpb-margin", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    parameter_golf_root = Path(args.parameter_golf_root)
    if not parameter_golf_root.is_absolute():
        parameter_golf_root = repo_root / parameter_golf_root
    log_path = Path(args.log)
    checkpoint_dir = Path(args.checkpoint_dir)
    state_path = Path(args.state) if args.state else repo_root / "outputs" / f"{args.run_id}.4k_recovery_state.json"
    last_status = ""

    while True:
        parsed = parse_seq4096_log(log_path)
        latest_val = parsed.val_rows[-1] if parsed.val_rows else None
        best_val = select_best_validation(parsed.val_rows, args.gate_step)
        target_reached = bool(best_val and best_val.val_bpb <= args.target_bpb)
        analysis_root = (
            Path(args.analysis_root)
            if args.analysis_root
            else repo_root / "outputs" / "post_resume_analysis" / args.run_id
        )
        preemptive_risk = (
            load_preemptive_gate_risk(
                analysis_root,
                gate_step=args.gate_step,
                target_bpb=args.target_bpb,
                min_step=args.preempt_min_step,
                patience=args.preempt_patience,
            )
            if args.preempt_on_projected_miss
            else None
        )
        preempt_for_gate_risk = (
            bool(latest_val)
            and latest_val.step >= int(args.preempt_min_step)
            and latest_val.step < int(args.gate_step)
            and should_preempt_for_gate_risk(preemptive_risk)
        )
        base_controls = RecoveryControls(
            train_batch_tokens=args.recovery_train_batch_tokens,
            tied_embed_lr=args.recovery_tied_embed_lr,
            matrix_lr=args.recovery_matrix_lr,
            scalar_lr=args.recovery_scalar_lr,
            muon_momentum=args.recovery_muon_momentum,
            muon_momentum_warmup_steps=args.recovery_muon_momentum_warmup_steps,
            muon_momentum_warmup_start=args.recovery_muon_momentum_warmup_start,
            grad_clip_norm=args.recovery_grad_clip_norm,
            bigram_bias=args.recovery_bigram_bias,
            bigram_bias_lr=args.recovery_bigram_bias_lr,
            bigram_bias_init_from_data=args.recovery_bigram_bias_init_from_data,
            bigram_bias_init_tokens=args.recovery_bigram_bias_init_tokens,
            bigram_bias_init_alpha=args.recovery_bigram_bias_init_alpha,
            bigram_bias_init_strength=args.recovery_bigram_bias_init_strength,
            bigram_bias_scale=args.recovery_bigram_bias_scale,
        )
        projected_target_step = (
            float("nan")
            if preemptive_risk is None
            else preemptive_risk.latest_projected_target_step
        )
        recovery_controls = plan_metric_driven_recovery_controls(
            base_controls,
            parsed=parsed,
            target_bpb=args.target_bpb,
            gate_step=args.gate_step,
            projected_target_step=projected_target_step,
            max_train_batch_tokens=args.recovery_max_train_batch_tokens,
            validation_gap_threshold=args.validation_gap_threshold,
            low_train_bpb_margin=args.low_train_bpb_margin,
            enabled=not args.no_advanced_metric_controls,
        )
        status: dict[str, Any] = {
            "run_id": args.run_id,
            "log": str(log_path),
            "checkpoint_dir": str(checkpoint_dir),
            "gate_step": args.gate_step,
            "target_bpb": args.target_bpb,
            "latest_step": parsed.latest_step,
            "latest_validation": None if latest_val is None else latest_val.__dict__,
            "best_validation_at_or_before_gate": None if best_val is None else best_val.__dict__,
            "target_reached": target_reached,
            "restart_index": args.restart_index,
            "min_recovery_runway_steps": args.min_recovery_runway_steps,
            "preempt_on_projected_miss": bool(args.preempt_on_projected_miss),
            "preemptive_gate_risk": None if preemptive_risk is None else preemptive_risk.__dict__,
            "base_recovery_launch_controls": base_controls.launch_dict(),
            "recovery_launch_controls": recovery_controls.launch_dict(),
            "advanced_metric_controls_enabled": not args.no_advanced_metric_controls,
            "recovery_max_train_batch_tokens": args.recovery_max_train_batch_tokens,
            "validation_gap_threshold": args.validation_gap_threshold,
            "low_train_bpb_margin": args.low_train_bpb_margin,
            "updated_unix": time.time(),
        }
        rendered = json.dumps(status, sort_keys=True)
        if rendered != last_status:
            write_state(state_path, status)
            if latest_val:
                best_text = "n/a" if best_val is None else f"{best_val.val_bpb:.4f}@{best_val.step}"
                print(
                    f"4k_gate_status run={args.run_id} latest={latest_val.val_bpb:.4f}@{latest_val.step} "
                    f"best={best_text} target={args.target_bpb:.4f}",
                    flush=True,
                )
            last_status = rendered

        if target_reached:
            status["event"] = "target_reached_before_or_at_gate"
            write_state(state_path, status)
            return

        if latest_val is None or (latest_val.step < args.gate_step and not preempt_for_gate_risk):
            time.sleep(max(1.0, args.poll_seconds))
            continue

        if args.restart_index >= args.max_restarts:
            status["event"] = "restart_limit_reached"
            write_state(state_path, status)
            print(json.dumps(status, sort_keys=True), flush=True)
            return

        if best_val is None:
            time.sleep(max(1.0, args.poll_seconds))
            continue

        recovery_val = select_recovery_validation(
            parsed.val_rows,
            args.gate_step,
            min_recovery_runway_steps=args.min_recovery_runway_steps,
        )
        if recovery_val is None:
            time.sleep(max(1.0, args.poll_seconds))
            continue

        resume_checkpoint = checkpoint_for_step(checkpoint_dir, args.run_id, recovery_val.step)
        if resume_checkpoint is None:
            status["event"] = "waiting_for_best_checkpoint"
            status["best_checkpoint_step"] = recovery_val.step
            write_state(state_path, status)
            time.sleep(max(1.0, args.poll_seconds))
            continue

        stamp = utc_stamp()
        next_index = args.restart_index + 1
        recovery_run_id = build_recovery_run_id(args.run_id, next_index, stamp)
        recovery_train_tmux = sanitize_tmux_name(f"toricgt_seq4096_4k_recovery_r{next_index}_{stamp}")
        recovery_checkpoint_dir = parameter_golf_root / "checkpoints" / recovery_run_id
        recovery_log = parameter_golf_root / "logs" / f"{recovery_run_id}.txt"
        launch = build_recovery_launch(
            repo_root=repo_root,
            parameter_golf_root=parameter_golf_root,
            run_id=recovery_run_id,
            checkpoint_dir=recovery_checkpoint_dir,
            log_path=recovery_log,
            resume_checkpoint=resume_checkpoint,
            seed=args.seed + next_index,
            target_bpb=args.target_bpb,
            python=args.python,
            train_batch_tokens=recovery_controls.train_batch_tokens,
            tied_embed_lr=recovery_controls.tied_embed_lr,
            matrix_lr=recovery_controls.matrix_lr,
            scalar_lr=recovery_controls.scalar_lr,
            muon_momentum=recovery_controls.muon_momentum,
            muon_momentum_warmup_steps=recovery_controls.muon_momentum_warmup_steps,
            muon_momentum_warmup_start=recovery_controls.muon_momentum_warmup_start,
            grad_clip_norm=recovery_controls.grad_clip_norm,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            bigram_bias=recovery_controls.bigram_bias,
            bigram_bias_lr=recovery_controls.bigram_bias_lr,
            bigram_bias_init_from_data=recovery_controls.bigram_bias_init_from_data,
            bigram_bias_init_tokens=recovery_controls.bigram_bias_init_tokens,
            bigram_bias_init_alpha=recovery_controls.bigram_bias_init_alpha,
            bigram_bias_init_strength=recovery_controls.bigram_bias_init_strength,
            bigram_bias_scale=recovery_controls.bigram_bias_scale,
        )
        command_dir = repo_root / "logs" / recovery_run_id / "supervisor"
        command_dir.mkdir(parents=True, exist_ok=True)
        (command_dir / "train_command.sh").write_text(launch.training_shell + "\n", encoding="utf-8")

        analysis_shell = build_analysis_shell(
            repo_root=repo_root,
            run_id=recovery_run_id,
            train_tmux=recovery_train_tmux,
            checkpoint_dir=recovery_checkpoint_dir,
            log_path=recovery_log,
            target_bpb=args.target_bpb,
            start_step=recovery_val.step,
            python=args.python,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
        )
        dense_shell = build_dense_mirror_shell(
            repo_root=repo_root,
            run_id=recovery_run_id,
            log_path=recovery_log,
            target_bpb=args.target_bpb,
            python=args.python,
        )
        diag_shell = build_full_diag_shell(
            repo_root=repo_root,
            run_id=recovery_run_id,
            log_path=recovery_log,
            target_bpb=args.target_bpb,
            python=args.python,
        )
        gate_shell = build_gate_shell(
            repo_root=repo_root,
            run_id=recovery_run_id,
            train_tmux=recovery_train_tmux,
            checkpoint_dir=recovery_checkpoint_dir,
            log_path=recovery_log,
            target_bpb=args.target_bpb,
            gate_step=args.gate_step,
            restart_index=next_index,
            max_restarts=args.max_restarts,
            min_recovery_runway_steps=args.min_recovery_runway_steps,
            recovery_train_batch_tokens=recovery_controls.train_batch_tokens,
            recovery_tied_embed_lr=recovery_controls.tied_embed_lr,
            recovery_matrix_lr=recovery_controls.matrix_lr,
            recovery_scalar_lr=recovery_controls.scalar_lr,
            recovery_muon_momentum=recovery_controls.muon_momentum,
            recovery_muon_momentum_warmup_steps=recovery_controls.muon_momentum_warmup_steps,
            recovery_muon_momentum_warmup_start=recovery_controls.muon_momentum_warmup_start,
            recovery_grad_clip_norm=recovery_controls.grad_clip_norm,
            recovery_bigram_bias=recovery_controls.bigram_bias,
            recovery_bigram_bias_lr=recovery_controls.bigram_bias_lr,
            recovery_bigram_bias_init_from_data=recovery_controls.bigram_bias_init_from_data,
            recovery_bigram_bias_init_tokens=recovery_controls.bigram_bias_init_tokens,
            recovery_bigram_bias_init_alpha=recovery_controls.bigram_bias_init_alpha,
            recovery_bigram_bias_init_strength=recovery_controls.bigram_bias_init_strength,
            recovery_bigram_bias_scale=recovery_controls.bigram_bias_scale,
            preempt_on_projected_miss=args.preempt_on_projected_miss,
            preempt_min_step=args.preempt_min_step,
            preempt_patience=args.preempt_patience,
            advanced_metric_controls=not args.no_advanced_metric_controls,
            recovery_max_train_batch_tokens=args.recovery_max_train_batch_tokens,
            validation_gap_threshold=args.validation_gap_threshold,
            low_train_bpb_margin=args.low_train_bpb_margin,
            python=args.python,
        )
        (command_dir / "analysis_command.sh").write_text(analysis_shell + "\n", encoding="utf-8")
        (command_dir / "dense_mirror_command.sh").write_text(dense_shell + "\n", encoding="utf-8")
        (command_dir / "full_diag_command.sh").write_text(diag_shell + "\n", encoding="utf-8")
        (command_dir / "gate_command.sh").write_text(gate_shell + "\n", encoding="utf-8")

        status.update(
            {
                "event": "launching_preemptive_gate_risk_recovery"
                if preempt_for_gate_risk
                else "launching_4k_recovery",
                "selected_recovery_validation": recovery_val.__dict__,
                "resume_checkpoint": str(resume_checkpoint),
                "recovery_run_id": recovery_run_id,
                "recovery_train_tmux": recovery_train_tmux,
                "recovery_log": str(recovery_log),
                "recovery_checkpoint_dir": str(recovery_checkpoint_dir),
                "command_dir": str(command_dir),
                "applied_recovery_launch_controls": recovery_controls.launch_dict(),
            }
        )
        write_state(state_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)

        if not args.dry_run:
            tmux_kill(args.train_tmux)
        tmux_start(recovery_train_tmux, launch.training_shell, args.dry_run)
        tmux_start(
            sanitize_tmux_name(f"toricgt_seq4096_4k_analysis_r{next_index}_{stamp}"),
            analysis_shell,
            args.dry_run,
        )
        tmux_start(
            sanitize_tmux_name(f"toricgt_seq4096_4k_mirror_r{next_index}_{stamp}"),
            dense_shell,
            args.dry_run,
        )
        tmux_start(
            sanitize_tmux_name(f"toricgt_seq4096_4k_full_diag_r{next_index}_{stamp}"),
            diag_shell,
            args.dry_run,
        )
        tmux_start(
            sanitize_tmux_name(f"toricgt_seq4096_4k_gate_r{next_index}_{stamp}"),
            gate_shell,
            args.dry_run,
        )
        return


if __name__ == "__main__":
    main()
