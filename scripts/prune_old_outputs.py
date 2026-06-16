#!/usr/bin/env python3
"""Dry-run cleanup for generated ToricGT output bundles.

The command is intentionally conservative: it only considers directories
matching explicit generated-output patterns, skips symlinks and ``latest_*``
entries, and defaults to dry-run mode.  Use ``--apply`` to delete candidates
after reviewing the JSON summary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any


DEFAULT_PATTERNS = [
    "branching_reasoning_trajectory_*",
    "branching_reasoning_embedding_payload_*",
    "rendered_html_screenshots_*",
    "outputs_index_screenshots_*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--pattern", action="append", default=[], help="Glob pattern to consider. May be repeated.")
    parser.add_argument("--keep-latest", type=int, default=3, help="Keep this many newest directories per pattern.")
    parser.add_argument("--min-age-hours", type=float, default=0.0, help="Only consider directories at least this old.")
    parser.add_argument("--apply", action="store_true", help="Actually delete candidates. Default is dry-run.")
    parser.add_argument("--manifest", default="", help="Optional JSON manifest path to write.")
    return parser.parse_args()


def dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return int(total)


def collect_candidates(root: Path, patterns: list[str], *, keep_latest: int, min_age_hours: float) -> list[dict[str, Any]]:
    now = time.time()
    records: list[dict[str, Any]] = []
    for pattern in patterns:
        matches = [
            path
            for path in root.glob(pattern)
            if path.is_dir() and not path.is_symlink() and not path.name.startswith("latest_")
        ]
        matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for rank, path in enumerate(matches):
            age_hours = max(0.0, (now - path.stat().st_mtime) / 3600.0)
            delete_candidate = rank >= max(0, int(keep_latest)) and age_hours >= float(min_age_hours)
            records.append(
                {
                    "path": str(path),
                    "pattern": pattern,
                    "rank_newest_first": int(rank),
                    "age_hours": round(float(age_hours), 3),
                    "bytes": dir_size(path) if delete_candidate else 0,
                    "delete_candidate": bool(delete_candidate),
                    "deleted": False,
                }
            )
    return records


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    patterns = list(args.pattern or []) or DEFAULT_PATTERNS
    records = collect_candidates(
        root,
        patterns,
        keep_latest=int(args.keep_latest),
        min_age_hours=float(args.min_age_hours),
    )
    if args.apply:
        for record in records:
            if not record["delete_candidate"]:
                continue
            path = Path(str(record["path"]))
            if path.exists() and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
                record["deleted"] = True
    payload = {
        "schema": "toricgt.prune_old_outputs.v1",
        "output_root": str(root),
        "dry_run": not bool(args.apply),
        "patterns": patterns,
        "keep_latest": int(args.keep_latest),
        "min_age_hours": float(args.min_age_hours),
        "candidate_count": int(sum(1 for record in records if record["delete_candidate"])),
        "candidate_bytes": int(sum(int(record["bytes"]) for record in records if record["delete_candidate"])),
        "records": records,
    }
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
