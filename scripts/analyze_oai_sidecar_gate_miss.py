#!/usr/bin/env python3
"""Analyze an OAI-sidecar run that missed the configured BPB gate.

This is used when an already-running campaign was launched with an older
cadence and therefore cannot produce the full checkpoint-backed visualization
suite at the new gate step.  It still performs the complete W&B metric audit
available for the run, including the full ``toricgt_sidecar/*`` review, and
writes an explicit restart plan.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+) train_loss:(?P<loss>[0-9.]+)"
    r"(?: train_bpb:(?P<train_bpb>[0-9.eE+-]+) train_bpt:(?P<train_bpt>[0-9.eE+-]+))?.*?"
    r"step_avg:(?P<step_avg>[0-9.]+)ms.*?"
    r"(?:graph_lm_loss:(?P<graph_lm_loss>[0-9.eE+-]+) graph_lm_bpb:(?P<graph_lm_bpb>[0-9.eE+-]+) "
    r"graph_lm_w:(?P<graph_lm_weight>[0-9.eE+-]+) )?.*?"
    r"sidecar_loss:(?P<sidecar>[0-9.eE+-]+) graphcg:(?P<graphcg>[0-9.eE+-]+) "
    r"analogy:(?P<analogy>[0-9.eE+-]+) tokengt_graph:(?P<tokengt>[0-9.eE+-]+) "
    r"memory:(?P<memory>[0-9.eE+-]+)"
    r"(?: toric:(?P<toric>[0-9.eE+-]+) (?:vb1d|vb):(?P<vector_bundle_1d_cone>[0-9.eE+-]+) "
    r"bgg:(?P<bgg>[0-9.eE+-]+) koszul:(?P<koszul>[0-9.eE+-]+) "
    r"cca:(?P<cca>[0-9.eE+-]+))?"
)
VAL_RE = re.compile(r"step:(?P<step>\d+)/(?P<total>\d+) val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+)")
CHECKPOINT_RE = re.compile(r"checkpoint_saved:(?P<path>.*?) step:(?P<step>\d+) val_bpb:(?P<bpb>[-+0-9.eE]+|None)")


@dataclass
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-path", required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-step", type=int, default=1500)
    parser.add_argument("--target-bpb", type=float, default=1.19)
    parser.add_argument("--next-profile", default="gate1500_high_batch_toric_bgg_memory")
    parser.add_argument("--codex-review", action="store_true")
    return parser.parse_args()


def safe_float(value: str | None) -> float | None:
    if value is None or value == "None":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def parse_log(path: Path) -> dict[str, Any]:
    trains: list[dict[str, Any]] = []
    vals: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    for line in text.splitlines():
        if match := TRAIN_RE.search(line):
            row: dict[str, Any] = {}
            for key, raw in match.groupdict().items():
                value = safe_float(raw)
                if value is not None:
                    row[key] = value
            trains.append(row)
        if match := VAL_RE.search(line):
            vals.append({key: safe_float(value) for key, value in match.groupdict().items()})
        if match := CHECKPOINT_RE.search(line):
            checkpoints.append(
                {
                    "path": match.group("path"),
                    "step": int(match.group("step")),
                    "val_bpb": safe_float(match.group("bpb")),
                }
            )
    return {"train_rows": trains, "val_rows": vals, "checkpoint_paths": checkpoints, "train_log": str(path)}


def latest_at_or_before(rows: list[dict[str, Any]], step: int) -> dict[str, Any] | None:
    candidates = [row for row in rows if int(row.get("step", -1)) <= int(step)]
    if not candidates:
        return None
    return max(candidates, key=lambda row: int(row.get("step", -1)))


def latest_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: int(row.get("step", -1)))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(name: str, command: list[str], output_dir: Path) -> CommandResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout = output_dir / f"{name}.stdout.log"
    stderr = output_dir / f"{name}.stderr.log"
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, text=True, check=False)
    return CommandResult(name, command, int(proc.returncode), str(stdout), str(stderr))


def plot_train_curves(output_dir: Path, rows: list[dict[str, Any]], gate_step: int, target_bpb: float) -> str | None:
    points = [
        (float(row["step"]), float(row["train_bpb"]))
        for row in rows
        if row.get("step") is not None and row.get("train_bpb") is not None
    ]
    if not points:
        return None
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("#030712")
    ax.set_facecolor("#07111f")
    ax.plot(x, y, color="#37e8ff", marker="o", markersize=3, linewidth=1.8, label="train BPB")
    ax.axhline(float(target_bpb), color="#ef476f", linestyle="--", linewidth=1.2, label=f"target {target_bpb:.3f}")
    ax.axvline(float(gate_step), color="#ffd166", linestyle=":", linewidth=1.2, label=f"gate step {gate_step}")
    ax.set_title(f"Current run BPB curve at gate step {gate_step}", color="#e8fbff")
    ax.set_xlabel("step", color="#e8fbff")
    ax.set_ylabel("train BPB", color="#e8fbff")
    ax.tick_params(colors="#9fb6c5")
    ax.grid(True, color="#143344", alpha=0.55)
    for spine in ax.spines.values():
        spine.set_color("#29536a")
    ax.legend(facecolor="#07111f", edgecolor="#29536a", labelcolor="#e8fbff")
    out = output_dir / "train_bpb_gate_curve.png"
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return str(out)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_metrics = parse_log(Path(args.train_log))
    gate_train = latest_at_or_before(log_metrics["train_rows"], int(args.gate_step))
    latest_train = latest_row(log_metrics["train_rows"])
    latest_val = latest_row(log_metrics["val_rows"])
    gate_bpb = safe_float(str(gate_train.get("train_bpb"))) if gate_train else None
    gate_passed = bool(gate_bpb is not None and gate_bpb < float(args.target_bpb))
    plot_path = plot_train_curves(output_dir, log_metrics["train_rows"], int(args.gate_step), float(args.target_bpb))
    commands = [
        run_command(
            "wandb_metrics",
            [
                PYTHON,
                "scripts/analyze_wandb_metrics.py",
                "--run-path",
                args.run_path,
                "--output-dir",
                str(output_dir / "wandb_metrics"),
            ],
            output_dir / "command_logs",
        ),
        run_command(
            "sidecar_metric_review",
            [
                PYTHON,
                "scripts/analyze_toricgt_sidecar_wandb_metrics.py",
                "--run-path",
                args.run_path,
                "--output-dir",
                str(output_dir / "sidecar_metric_review"),
            ],
            output_dir / "command_logs",
        ),
    ]
    summary = {
        "generated_utc": utc_iso(),
        "run_id": args.run_id,
        "run_path": args.run_path,
        "gate_step": int(args.gate_step),
        "target_bpb": float(args.target_bpb),
        "gate_train_row": gate_train,
        "latest_train_row": latest_train,
        "latest_val_row": latest_val,
        "gate_passed": gate_passed,
        "train_bpb_gate_curve": plot_path,
        "log_metrics": log_metrics,
        "commands": [asdict(result) for result in commands],
        "next_profile": args.next_profile,
        "analysis_limitation": (
            "The current run was launched under the previous 5K cadence and only has a step-0 checkpoint. "
            "Checkpoint-backed GUDHI/Sage/Macaulay2/visual full analysis is therefore intentionally deferred "
            f"to the restarted {int(args.gate_step)}-step run, which will checkpoint and validate at the gate."
        ),
    }
    write_json(output_dir / "gate_miss_summary.json", summary)
    sidecar_review = output_dir / "sidecar_metric_review" / "SIDECAR-METRIC-REVIEW.md"
    wandb_report = output_dir / "wandb_metrics" / "automatic_metrics_report.md"
    report = output_dir / "GATE-MISS-REPORT-AND-RESTART-PLAN.md"
    lines = [
        "# OAI Sidecar Gate-Miss Report And Restart Plan",
        "",
        f"- generated UTC: `{utc_iso()}`",
        f"- run id: `{args.run_id}`",
        f"- W&B run: `{args.run_path}`",
        f"- gate: `train BPB < {float(args.target_bpb):.3f}` by step `{int(args.gate_step)}`",
        f"- gate result: `{'passed' if gate_passed else 'failed'}`",
        f"- gate train row: `{json.dumps(gate_train, sort_keys=True)}`",
        f"- latest train row: `{json.dumps(latest_train, sort_keys=True)}`",
        f"- latest validation row: `{json.dumps(latest_val, sort_keys=True)}`",
        "",
        "## Generated Analyses",
        "",
        f"- W&B metric report: `{wandb_report}`",
        f"- sidecar metric review: `{sidecar_review}`",
        f"- train BPB gate curve: `{plot_path}`",
        f"- raw summary JSON: `{output_dir / 'gate_miss_summary.json'}`",
        "",
        "## Decision",
        "",
        f"The active run missed or has not yet satisfied the new {int(args.gate_step)}-step BPB gate.  Because it was launched before the new cadence existed, "
        f"it cannot provide a checkpoint-backed full visualization suite at step {int(args.gate_step)}.  The correct implementation is to stop it, "
        f"restart from step 0 with validation/checkpointing at {int(args.gate_step)}, and make the restarted run's full analysis mandatory before any subsequent restart.",
        "",
        "## Training Plan Update To Implement",
        "",
        f"- Set campaign cadence to `{int(args.gate_step)}` steps.",
        f"- Set target to `BPB < {float(args.target_bpb):.3f}` at the gate.",
        "- Set `VAL_LOSS_EVERY` and `CHECKPOINT_EVERY` equal to the cadence so checkpoint-backed visualizations exist at every decision point.",
        f"- Start next profile `{args.next_profile}`: faster primary BPB optimization, bonafide graph-LM primary training, and all sidecar metrics still logged.",
        "- If the restarted run misses the gate, run the full checkpoint-backed analysis suite, Codex review, write the report/plan, then restart from step 0 with the next profile.",
        "",
        "## Command Results",
        "",
        "```json",
        json.dumps([asdict(result) for result in commands], indent=2, sort_keys=True),
        "```",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    if args.codex_review:
        prompt = (
            f"Review this ToricGT OAI-sidecar {int(args.gate_step)}-step gate-miss report. Read the linked W&B metrics and "
            "sidecar metric review. Confirm the restart plan or recommend exact hyperparameter changes for "
            f"the next run. Report path: {report}"
        )
        subprocess.run(["codex", "exec", prompt], check=False)
    print(json.dumps({"report": str(report), "summary": str(output_dir / "gate_miss_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
