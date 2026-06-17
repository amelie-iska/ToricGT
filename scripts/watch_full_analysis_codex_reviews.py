#!/usr/bin/env python3
"""Watch a campaign notes directory and run Codex on full analysis reports.

The training supervisor already blocks on full analysis before choosing the
next profile.  This watcher makes the Codex review target explicit: it waits
for each ``RUN-*-full-analysis*/FULL-ITERATION-REPORT.md`` file, invokes
``codex exec`` with that report path, and writes a marker so each analysis
bundle is reviewed once.  Retry analysis directories are included because the
retry result is the one that should inform the next training profile.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-notes-dir", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-reviews", type=int, default=25)
    parser.add_argument("--codex-bin", default="codex")
    return parser.parse_args()


def review_report(codex_bin: str, report: Path, marker: Path) -> int:
    prompt = (
        "Review this ToricGT 5K-step full iteration analysis. Inspect the report's linked W&B metrics, "
        "exact GUDHI/Macaulay2 persistent-homology outputs, Sage/Macaulay2 toric embedding sidecars, "
        "toric vector-bundle/sheaf report, Toric BGG/category O report, branching reasoning visualizations, "
        "screenshots, sidecar losses, BPB, artifact size, and next_profile_decision.json. "
        "Write a concise but complete recommendation: continue, reduce auxiliary weights, isolate a subset, "
        "change LR/warmdown, or stop/fix instrumentation. "
        f"Full analysis report path: {report}"
    )
    proc = subprocess.run([codex_bin, "exec", prompt], check=False)
    marker.write_text(
        f"reviewed_utc: {utc_iso()}\nreturncode: {int(proc.returncode)}\nreport: {report}\n",
        encoding="utf-8",
    )
    return int(proc.returncode)


def main() -> None:
    args = parse_args()
    root = Path(args.campaign_notes_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reviewed = 0
    while reviewed < int(args.max_reviews):
        reports = sorted(root.glob("RUN-*-full-analysis*/FULL-ITERATION-REPORT.md"))
        did_work = False
        for report in reports:
            marker = report.with_suffix(report.suffix + ".codex_reviewed")
            if marker.exists():
                continue
            print(f"[{utc_iso()}] Codex reviewing {report}", flush=True)
            review_report(str(args.codex_bin), report, marker)
            reviewed += 1
            did_work = True
            if reviewed >= int(args.max_reviews):
                break
        if not did_work:
            time.sleep(max(5.0, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
