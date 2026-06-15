#!/usr/bin/env python3
"""Report ToricGT-related processes, tmux sessions, and recent outputs.

This command is read-only.  It does not start, stop, restart, or signal any
process.  It is meant to make background training/analysis terminals explicit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


KEYWORDS = (
    "ToricGT",
    "toricgt",
    "TropicalGT",
    "tropicalgt",
    "train_",
    "infer_tokengt",
    "run_gudhi",
    "run_embedding_cas_sidecar",
    "watch_training_analysis",
    "wandb",
)


def run_text(command: list[str]) -> str:
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.stdout


def process_rows() -> list[dict[str, Any]]:
    output = run_text(["ps", "-eo", "pid,ppid,stat,etime,cmd"])
    rows: list[dict[str, Any]] = []
    for line in output.splitlines()[1:]:
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid, ppid, stat, etime, cmd = parts
        if not any(keyword.lower() in cmd.lower() for keyword in KEYWORDS):
            continue
        rows.append({"pid": int(pid), "ppid": int(ppid), "stat": stat, "etime": etime, "cmd": cmd})
    return rows


def tmux_sessions() -> list[str]:
    output = run_text(["tmux", "list-sessions"])
    if "no server running" in output.lower() or "failed to connect" in output.lower():
        return []
    return [line for line in output.splitlines() if line.strip()]


def latest_paths(root: Path, pattern: str, *, limit: int = 8) -> list[str]:
    if not root.exists():
        return []
    files = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path) for path in files[: max(1, int(limit))]]


def status_payload(repo: Path) -> dict[str, Any]:
    return {
        "schema": "toricgt.status.v1",
        "repo": str(repo),
        "read_only": True,
        "processes": process_rows(),
        "tmux_sessions": tmux_sessions(),
        "latest_checkpoints": latest_paths(repo / "checkpoints", "**/*.pt"),
        "latest_output_indexes": latest_paths(repo / "outputs", "**/index.html"),
        "latest_analysis_summaries": latest_paths(repo / "outputs", "**/AUDIT_SUMMARY.md"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Repository root.")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = status_payload(Path(args.repo).resolve())
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"ToricGT status for {payload['repo']}")
    print("read_only: true")
    print("\nProcesses:")
    if payload["processes"]:
        for row in payload["processes"]:
            print(f"  pid={row['pid']} ppid={row['ppid']} stat={row['stat']} etime={row['etime']} cmd={row['cmd']}")
    else:
        print("  none")
    print("\ntmux sessions:")
    if payload["tmux_sessions"]:
        for session in payload["tmux_sessions"]:
            print(f"  {session}")
    else:
        print("  none")
    print("\nLatest checkpoints:")
    for path in payload["latest_checkpoints"]:
        print(f"  {path}")
    print("\nLatest output indexes:")
    for path in payload["latest_output_indexes"]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
