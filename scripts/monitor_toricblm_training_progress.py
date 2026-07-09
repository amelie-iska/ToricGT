#!/usr/bin/env python3
"""Live progress and ETA monitor for ToricBLM/ToricGT training logs.

The trainer writes detailed step lines only every configured log interval.  This
monitor reads the latest emitted line, estimates the in-flight step from the
most recent step average, and renders a tqdm progress bar with ETA and key
metrics.  It is intentionally read-only: it never signals or modifies the
training process.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


STEP_RE = re.compile(r"\bstep:(?P<step>\d+)/(?P<total>\d+)\b")
METRIC_RE = re.compile(r"(?P<key>[A-Za-z0-9_./-]+):(?P<value>[^ \n]+)")


@dataclass
class StepStatus:
    step: int
    total: int
    metrics: dict[str, str]
    log_mtime: float
    line: str


def _tail_lines(path: Path, max_bytes: int = 1_000_000) -> list[str]:
    size = path.stat().st_size
    with path.open("rb") as fh:
        fh.seek(max(0, size - max_bytes))
        data = fh.read().decode("utf-8", errors="replace")
    return data.splitlines()


def latest_step_status(log_file: Path) -> StepStatus:
    if not log_file.exists():
        raise FileNotFoundError(log_file)
    mtime = log_file.stat().st_mtime
    for line in reversed(_tail_lines(log_file)):
        match = STEP_RE.search(line)
        if not match:
            continue
        metrics = {m.group("key"): m.group("value") for m in METRIC_RE.finditer(line)}
        return StepStatus(
            step=int(match.group("step")),
            total=int(match.group("total")),
            metrics=metrics,
            log_mtime=mtime,
            line=line,
        )
    raise RuntimeError(f"No step line found in {log_file}")


def parse_ms(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    if value.endswith("ms"):
        value = value[:-2]
    try:
        return float(value)
    except ValueError:
        return None


def human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def gpu_summary() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
    except Exception:
        return "gpu:n/a"
    if not out:
        return "gpu:n/a"
    used, total, util = [part.strip() for part in out.splitlines()[0].split(",")[:3]]
    return f"gpu:{used}/{total}MiB {util}%"


def load_epoch_summary(run_dir: Path) -> dict[str, object]:
    manifest = run_dir / "epoch_state_manifest.json"
    if not manifest.exists():
        return {}
    try:
        obj = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "epoch": obj.get("epoch"),
        "mode": obj.get("mode"),
        "selected_rows": obj.get("selected_rows"),
        "coverage_steps": obj.get("coverage_steps"),
        "selected_rows_by_modality": obj.get("selected_rows_by_modality"),
        "selected_file_count_by_modality": obj.get("selected_file_count_by_modality"),
    }


def find_default_run_dir() -> Path:
    root = Path("runs/oai_sidecar")
    candidates = sorted(root.glob("toricblm-structure-priority-curriculum-*/epoch_001_structure_current"))
    if not candidates:
        raise SystemExit("No structure-priority run dir found; pass --run-dir.")
    return candidates[-1]


def report_once(run_dir: Path, log_file: Path) -> None:
    status = latest_step_status(log_file)
    step_avg_ms = parse_ms(status.metrics.get("step_avg"))
    now = time.time()
    estimate_extra = 0
    if step_avg_ms and step_avg_ms > 0:
        estimate_extra = int(max(0.0, now - status.log_mtime) / (step_avg_ms / 1000.0))
    estimated_step = min(status.total, status.step + estimate_extra)
    basis_step = max(status.step, estimated_step)
    eta_seconds = None
    if step_avg_ms and step_avg_ms > 0:
        eta_seconds = (status.total - basis_step) * step_avg_ms / 1000.0
    epoch = load_epoch_summary(run_dir)
    print(f"run_dir: {run_dir}")
    print(f"log_file: {log_file}")
    print(f"last_logged_step: {status.step}/{status.total}")
    print(f"estimated_current_step: {estimated_step}/{status.total}")
    print(f"step_avg: {step_avg_ms / 1000.0 if step_avg_ms else 'unknown'} sec")
    if eta_seconds is not None:
        print(f"eta_from_estimated_step: {human_duration(eta_seconds)}")
    print(f"train_bpb: {status.metrics.get('train_bpb', 'n/a')}")
    print(f"graph_lm_bpb: {status.metrics.get('graph_lm_bpb', 'n/a')}")
    print(f"long_entry_bpb: {status.metrics.get('long_entry_bpb', 'n/a')}")
    print(f"structure_rmsd: {status.metrics.get('structure_rmsd', 'n/a')}")
    print(gpu_summary())
    if epoch:
        print("epoch_summary:")
        print(json.dumps(epoch, indent=2, sort_keys=True))


def monitor(run_dir: Path, log_file: Path, interval: float) -> None:
    status = latest_step_status(log_file)
    bar = tqdm(total=status.total, initial=status.step, unit="step", dynamic_ncols=True)
    try:
        while True:
            status = latest_step_status(log_file)
            step_avg_ms = parse_ms(status.metrics.get("step_avg"))
            estimated_step = status.step
            eta = "eta:n/a"
            if step_avg_ms and step_avg_ms > 0:
                estimated_step = min(
                    status.total,
                    status.step + int(max(0.0, time.time() - status.log_mtime) / (step_avg_ms / 1000.0)),
                )
                eta = f"eta:{human_duration((status.total - estimated_step) * step_avg_ms / 1000.0)}"
            bar.total = status.total
            bar.n = estimated_step
            postfix = {
                "logged": f"{status.step}/{status.total}",
                "train_bpb": status.metrics.get("train_bpb", "n/a"),
                "graph": status.metrics.get("graph_lm_bpb", "n/a"),
                "long": status.metrics.get("long_entry_bpb", "n/a"),
                "rmsd": status.metrics.get("structure_rmsd", "n/a"),
                "avg_s": f"{(step_avg_ms or 0.0) / 1000.0:.2f}" if step_avg_ms else "n/a",
                "eta": eta,
                "gpu": gpu_summary().replace("gpu:", ""),
            }
            bar.set_postfix(postfix, refresh=True)
            time.sleep(interval)
    finally:
        bar.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--once", action="store_true", help="Print one ETA report and exit.")
    args = parser.parse_args()

    run_dir = args.run_dir or find_default_run_dir()
    log_file = args.log_file or (run_dir / "train.log")
    if args.once:
        report_once(run_dir, log_file)
    else:
        monitor(run_dir, log_file, args.interval)


if __name__ == "__main__":
    main()
