#!/usr/bin/env python3
"""Watch a Seq4096 BPB run and decide whether it passes a continuation gate."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<loss>[0-9.eE+-]+)"
    r"\s+val_bpb:(?P<bpb>[0-9.eE+-]+)"
)


@dataclass(frozen=True)
class ValPoint:
    step: int
    total: int
    loss: float
    bpb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-tmux", required=True)
    parser.add_argument("--gate-step", type=int, required=True)
    parser.add_argument("--target-bpb", type=float, required=True)
    parser.add_argument("--continue-threshold-bpb", type=float, default=1.17)
    parser.add_argument("--state", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--kill-on-miss", action="store_true")
    return parser.parse_args()


def parse_vals(path: Path) -> list[ValPoint]:
    if not path.exists():
        return []
    points: dict[int, ValPoint] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = VAL_RE.search(line)
        if not match:
            continue
        point = ValPoint(
            step=int(match.group("step")),
            total=int(match.group("total")),
            loss=float(match.group("loss")),
            bpb=float(match.group("bpb")),
        )
        points[point.step] = point
    return [points[step] for step in sorted(points)]


def tmux_has(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    ).returncode == 0


def tmux_kill(session: str) -> None:
    subprocess.run(["tmux", "kill-session", "-t", session], check=False, text=True)


def write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    state_path = Path(args.state)
    last_state = ""
    while True:
        vals = parse_vals(log_path)
        latest = vals[-1] if vals else None
        gate_vals = [point for point in vals if point.step <= args.gate_step]
        best_gate = min(gate_vals, key=lambda point: point.bpb) if gate_vals else None
        latest_step = latest.step if latest else 0
        train_alive = tmux_has(args.train_tmux)
        target_reached = bool(best_gate and best_gate.bpb <= args.target_bpb)
        continuation_pass = bool(best_gate and best_gate.bpb <= args.continue_threshold_bpb)
        gate_observed = latest_step >= args.gate_step or (not train_alive and latest_step > 0)

        action = "waiting"
        if target_reached:
            action = "target_reached_continue"
        elif continuation_pass and gate_observed:
            action = "continuation_pass_continue"
        elif gate_observed and best_gate is not None:
            action = "gate_missed_stop_for_retune"
            if args.kill_on_miss and train_alive:
                tmux_kill(args.train_tmux)
                train_alive = False
                action = "gate_missed_training_stopped_for_retune"
        elif not train_alive and latest_step == 0:
            action = "waiting_for_training_log"

        payload = {
            "run_id": args.run_id,
            "log": str(log_path),
            "gate_step": args.gate_step,
            "target_bpb": args.target_bpb,
            "continue_threshold_bpb": args.continue_threshold_bpb,
            "latest_validation": None if latest is None else asdict(latest),
            "best_validation_at_or_before_gate": None if best_gate is None else asdict(best_gate),
            "target_reached": target_reached,
            "continuation_pass": continuation_pass,
            "gate_observed": gate_observed,
            "train_tmux": args.train_tmux,
            "train_alive": train_alive,
            "action": action,
            "updated_unix": time.time(),
        }
        rendered = json.dumps(payload, sort_keys=True)
        if rendered != last_state:
            write_state(state_path, payload)
            latest_text = "n/a" if latest is None else f"{latest.bpb:.4f}@{latest.step}"
            best_text = "n/a" if best_gate is None else f"{best_gate.bpb:.4f}@{best_gate.step}"
            print(
                f"bpb_gate_status run={args.run_id} latest={latest_text} best_gate={best_text} "
                f"target={args.target_bpb:.4f} continue={args.continue_threshold_bpb:.4f} action={action}",
                flush=True,
            )
            last_state = rendered

        if action in {
            "target_reached_continue",
            "continuation_pass_continue",
            "gate_missed_stop_for_retune",
            "gate_missed_training_stopped_for_retune",
        }:
            return
        time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    main()
