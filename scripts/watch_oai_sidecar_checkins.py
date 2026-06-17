#!/usr/bin/env python3
"""Write periodic checkin reports for OAI-baseline ToricGT sidecar runs."""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


VAL_RE = re.compile(r"step:(?P<step>\d+)/(?P<total>\d+) val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+)")
TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+) train_loss:(?P<loss>[0-9.]+).*?"
    r"step_avg:(?P<avg>[0-9.]+)ms(?: sidecar_loss:(?P<sidecar>[0-9.eE+-]+) "
    r"graphcg:(?P<graphcg>[0-9.eE+-]+) analogy:(?P<analogy>[0-9.eE+-]+) "
    r"tokengt_graph:(?P<tokengt>[0-9.eE+-]+) memory:(?P<memory>[0-9.eE+-]+))?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--notes-dir", required=True)
    parser.add_argument("--target-step", type=int, default=5000)
    parser.add_argument("--repeat-interval", type=int, default=0)
    parser.add_argument("--stop-step", type=int, default=0)
    parser.add_argument("--target-bpb", type=float, default=1.12)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--codex-review", action="store_true")
    return parser.parse_args()


def latest_rows(log_path: Path) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    vals: list[dict[str, float]] = []
    trains: list[dict[str, float]] = []
    if not log_path.exists():
        return vals, trains
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := VAL_RE.search(line):
            vals.append({key: float(value) for key, value in match.groupdict().items()})
        if match := TRAIN_RE.search(line):
            row = {key: float(value) for key, value in match.groupdict(default="nan").items()}
            trains.append(row)
    return vals, trains


def write_report(args: argparse.Namespace, checkpoint: Path) -> Path:
    log_path = Path(args.log)
    vals, trains = latest_rows(log_path)
    notes_dir = Path(args.notes_dir)
    notes_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = notes_dir / f"CHECKIN-step-{args.target_step:06d}-{stamp}.md"
    latest_val = vals[-1] if vals else None
    latest_train = trains[-1] if trains else None
    lines = [
        f"# OAI ToricGT Sidecar Checkin: {args.run_id}",
        "",
        f"- timestamp UTC: `{stamp}`",
        f"- checkpoint: `{checkpoint}`",
        f"- log: `{log_path}`",
        f"- target BPB: `{args.target_bpb}`",
        "",
        "## Latest Validation",
    ]
    if latest_val:
        gap = latest_val["bpb"] - float(args.target_bpb)
        lines.append(f"- step `{int(latest_val['step'])}` val_loss `{latest_val['loss']:.4f}` val_bpb `{latest_val['bpb']:.4f}` target_gap `{gap:.4f}`")
    else:
        lines.append("- no validation row parsed yet")
    lines.extend(["", "## Latest Training And Sidecar Metrics"])
    if latest_train:
        lines.extend(
            [
                f"- step `{int(latest_train['step'])}` train_loss `{latest_train['loss']:.4f}` step_avg_ms `{latest_train['avg']:.2f}`",
                f"- sidecar_loss `{latest_train.get('sidecar', float('nan')):.6f}`",
                f"- graphcg_loss `{latest_train.get('graphcg', float('nan')):.6f}`",
                f"- analogy_lattice_loss `{latest_train.get('analogy', float('nan')):.6f}`",
                f"- tokengt_graph_loss `{latest_train.get('tokengt', float('nan')):.6f}`",
                f"- trajectory_memory_loss `{latest_train.get('memory', float('nan')):.6f}`",
            ]
        )
    else:
        lines.append("- no train row parsed yet")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This run uses the OAI SP1024 FineWeb baseline as the primary BPB objective. "
            "The ToricGT sidecar is active only as an auxiliary graph/reasoning signal over external graph Parquet rows. "
            "A sidecar metric of `nan` indicates a logging or activation failure and should block continuation.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    target_step = int(args.target_step)
    while True:
        args.target_step = target_step
        target = checkpoint_dir / f"{args.run_id}_step_{target_step:06d}.pt"
        while not target.exists():
            time.sleep(float(args.poll_seconds))
        report = write_report(args, target)
        print(f"wrote {report}", flush=True)
        if args.codex_review:
            prompt = (
                f"Review this OAI ToricGT sidecar training checkin report and write concise recommendations "
                f"for improving BPB while keeping the sidecar active: {report}"
            )
            subprocess.run(["codex", "exec", prompt], check=False)
        if args.repeat_interval <= 0:
            break
        target_step += int(args.repeat_interval)
        if args.stop_step > 0 and target_step > args.stop_step:
            break


if __name__ == "__main__":
    main()
