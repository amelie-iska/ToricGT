#!/usr/bin/env python3
"""Watch a Parameter-Golf FineWeb log and write BPB loop status files.

This monitor is intentionally non-interrupting.  The FineWeb scaffold does not
emit intermediate checkpoints, so the monitor records validation BPB progress
from the log and leaves the active trainer alone until a human/Codex review
chooses a restart strategy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any


VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<loss>[0-9.]+)\s+val_bpb:(?P<bpb>[0-9.]+)"
)
TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+train_loss:(?P<loss>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--target-bpb", type=float, default=1.2)
    parser.add_argument("--max-review-iterations", type=int, default=100)
    parser.add_argument("--state", default="outputs/bpb_codex_loop_state.json")
    parser.add_argument("--status", default="outputs/fineweb_bpb_status.json")
    parser.add_argument("--report", default="outputs/fineweb_bpb_status.md")
    parser.add_argument("--stop-file", default="outputs/bpb_codex_loop_stop")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def parse_log(path: Path) -> dict[str, Any]:
    latest_train: dict[str, Any] | None = None
    vals: list[dict[str, Any]] = []
    if not path.exists():
        return {"latest_train": None, "validations": []}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = VAL_RE.search(line)
        if match:
            vals.append(
                {
                    "step": int(match.group("step")),
                    "total": int(match.group("total")),
                    "val_loss": float(match.group("loss")),
                    "val_bpb": float(match.group("bpb")),
                }
            )
            continue
        match = TRAIN_RE.search(line)
        if match:
            latest_train = {
                "step": int(match.group("step")),
                "total": int(match.group("total")),
                "train_loss": float(match.group("loss")),
            }
    return {"latest_train": latest_train, "validations": vals}


def write_report(path: Path, status: dict[str, Any]) -> None:
    vals = status.get("validations") or []
    latest = status.get("latest_validation") or {}
    best = status.get("best_validation") or {}
    lines = [
        "# FineWeb BPB Monitor",
        "",
        f"- Run id: `{status.get('run_id') or 'unknown'}`",
        f"- Log: `{status.get('log')}`",
        f"- Target BPB: `{status.get('target_bpb')}`",
        f"- Latest train step: `{(status.get('latest_train') or {}).get('step', 'n/a')}`",
        f"- Latest validation BPB: `{latest.get('val_bpb', 'n/a')}` at step `{latest.get('step', 'n/a')}`",
        f"- Best validation BPB: `{best.get('val_bpb', 'n/a')}` at step `{best.get('step', 'n/a')}`",
        f"- Target reached: `{status.get('target_reached')}`",
        f"- Stop requested: `{status.get('stop_requested')}`",
        "",
        "## Validation History",
        "",
        "| step | val_loss | val_bpb |",
        "|---:|---:|---:|",
    ]
    for row in vals:
        lines.append(f"| {row['step']} | {row['val_loss']:.4f} | {row['val_bpb']:.4f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_loop_state(state_path: Path, status: dict[str, Any], max_iterations: int) -> None:
    state = load_json(state_path)
    history = state.get("fineweb_history")
    if not isinstance(history, list):
        history = []
    latest = status.get("latest_validation")
    best = status.get("best_validation")
    if latest:
        seen = {(item.get("source"), item.get("step")) for item in history if isinstance(item, dict)}
        key = ("fineweb", latest.get("step"))
        if key not in seen:
            history.append(
                {
                    "source": "fineweb",
                    "step": latest.get("step"),
                    "val_bpb": latest.get("val_bpb"),
                    "val_loss": latest.get("val_loss"),
                    "target_bpb": status.get("target_bpb"),
                    "target_reached": status.get("target_reached"),
                }
            )
    state.update(
        {
            "fineweb_active": True,
            "fineweb_latest_validation": latest,
            "fineweb_best_validation": best,
            "fineweb_target_reached": status.get("target_reached"),
            "fineweb_target_bpb": status.get("target_bpb"),
            "max_iterations": max_iterations,
            "fineweb_history": history[-250:],
        }
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    state_path = Path(args.state)
    status_path = Path(args.status)
    report_path = Path(args.report)
    stop_file = Path(args.stop_file)
    last_payload = ""
    while True:
        parsed = parse_log(log_path)
        vals = parsed["validations"]
        best = min(vals, key=lambda row: row["val_bpb"]) if vals else None
        latest = vals[-1] if vals else None
        target_reached = bool(best and best["val_bpb"] <= args.target_bpb)
        status = {
            "run_id": args.run_id,
            "log": str(log_path),
            "target_bpb": args.target_bpb,
            "max_review_iterations": args.max_review_iterations,
            "latest_train": parsed["latest_train"],
            "latest_validation": latest,
            "best_validation": best,
            "validations": vals,
            "target_reached": target_reached,
            "stop_file": str(stop_file),
            "stop_requested": stop_file.exists(),
            "updated_unix": time.time(),
        }
        payload = json.dumps(status, indent=2, sort_keys=True)
        if payload != last_payload:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(payload + "\n", encoding="utf-8")
            write_report(report_path, status)
            update_loop_state(state_path, status, args.max_review_iterations)
            if latest:
                print(
                    "fineweb_status "
                    f"step={latest['step']} val_bpb={latest['val_bpb']:.4f} "
                    f"best={best['val_bpb']:.4f} target={args.target_bpb:.4f}",
                    flush=True,
                )
            last_payload = payload
        if target_reached or stop_file.exists():
            break
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
