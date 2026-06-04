#!/usr/bin/env python3
"""Prune old Parameter-Golf checkpoint step files while preserving useful anchors."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STEP_RE = re.compile(r"(?:random_order_step_|.*_step_)(\d+)\.pt$")
ALWAYS_KEEP_NAMES = {
    "best.pt",
    "final.pt",
    "latest.pt",
    "initial_parameter_golf_artifact.zip",
    "initial_parameter_golf_artifact.zip.json",
    "final_parameter_golf_artifact.zip",
    "final_parameter_golf_artifact.zip.json",
    "hf_best_publish_state.json",
    "parameter_golf_oai_best.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Checkpoint directories to prune.")
    parser.add_argument("--keep-latest", type=int, default=10)
    parser.add_argument("--keep-every", type=int, default=5000)
    parser.add_argument("--keep-step", type=int, action="append", default=[])
    parser.add_argument("--apply", action="store_true", help="Delete files. Without this, only reports.")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def step_for_path(path: Path) -> int | None:
    match = STEP_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def prune_dir(path: Path, *, keep_latest: int, keep_every: int, keep_steps: set[int], apply: bool) -> dict[str, object]:
    step_files = [(step_for_path(file), file) for file in path.glob("*.pt")]
    step_files = [(step, file) for step, file in step_files if step is not None]
    step_files.sort(key=lambda item: item[0])
    latest_steps = {step for step, _ in step_files[-max(keep_latest, 0):]}
    keep: set[Path] = set()
    delete: list[Path] = []
    for step, file in step_files:
        should_keep = (
            file.name in ALWAYS_KEEP_NAMES
            or step in latest_steps
            or step in keep_steps
            or (keep_every > 0 and step % keep_every == 0)
        )
        if should_keep:
            keep.add(file)
        else:
            delete.append(file)
    deleted_bytes = 0
    for file in delete:
        try:
            deleted_bytes += file.stat().st_size
            if apply:
                file.unlink()
        except OSError:
            pass
    return {
        "path": str(path),
        "step_files": len(step_files),
        "kept_step_files": len(keep),
        "deleted_step_files": len(delete) if apply else 0,
        "would_delete_step_files": len(delete),
        "deleted_bytes": deleted_bytes if apply else 0,
        "would_delete_bytes": deleted_bytes,
        "kept_steps": sorted(step for step, file in step_files if file in keep),
        "deleted_steps": sorted(step for step, file in step_files if file in delete),
        "applied": bool(apply),
    }


def main() -> None:
    args = parse_args()
    report = {
        "keep_latest": int(args.keep_latest),
        "keep_every": int(args.keep_every),
        "keep_steps": sorted(set(int(step) for step in args.keep_step)),
        "directories": [],
    }
    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_dir():
            report["directories"].append({"path": raw_path, "error": "missing_or_not_directory"})
            continue
        report["directories"].append(
            prune_dir(
                path,
                keep_latest=int(args.keep_latest),
                keep_every=int(args.keep_every),
                keep_steps=set(int(step) for step in args.keep_step),
                apply=bool(args.apply),
            )
        )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
