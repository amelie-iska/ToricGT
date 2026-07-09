#!/usr/bin/env python3
"""Repartition an existing AFDB GCS accession plan across more workers.

This preserves the exact selected accessions and selection metadata from a
previous plan while changing only the worker JSONL layout.  It is intended for
throughput tuning after the biological selection has already been accepted.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def iter_plan_rows(input_dir: Path) -> Any:
    worker_dir = input_dir / "worker_plans"
    files = sorted(worker_dir.glob("worker_*.jsonl"))
    if not files:
        files = sorted(input_dir.glob("worker_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no worker_*.jsonl files found under {input_dir}")
    for fp in files:
        with fp.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                raw = raw.strip()
                if raw:
                    yield raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--prefix", default="toricblm_afdb_gcs_diverse_plan")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(args.output_dir)
    worker_dir = args.output_dir / "worker_plans"
    worker_dir.mkdir(parents=True, exist_ok=True)

    handles = [
        (worker_dir / f"worker_{idx:02d}.jsonl").open("w", encoding="utf-8")
        for idx in range(args.workers)
    ]
    counts: Counter[str] = Counter()
    total = 0
    try:
        for total, raw in enumerate(iter_plan_rows(args.input_dir), start=1):
            item = json.loads(raw)
            selection = item.get("selection") if isinstance(item.get("selection"), dict) else {}
            counts[str(selection.get("tier") or item.get("enzyme_tier") or "unknown")] += 1
            handles[(total - 1) % args.workers].write(json.dumps(item, sort_keys=True, ensure_ascii=True) + "\n")
    finally:
        for handle in handles:
            handle.close()

    source_manifest = read_manifest(args.input_dir / f"{args.prefix}_manifest.json")
    if not source_manifest:
        source_manifest = read_manifest(args.input_dir / "toricblm_afdb_gcs_diverse_plan_manifest.json")
    manifest = {
        "schema": "toricblm.afdb_gcs_diverse_accession_plan_manifest.v1",
        "selection_preserved_from": str(args.input_dir),
        "source_manifest": source_manifest,
        "accepted_records": total,
        "workers": args.workers,
        "counts": dict(sorted(counts.items())),
        "worker_plan_dir": str(worker_dir),
        "worker_plans": [str(worker_dir / f"worker_{idx:02d}.jsonl") for idx in range(args.workers)],
        "policy": "Repartition only; accession selection and per-row metadata are unchanged.",
    }
    manifest_path = args.output_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compatibility_path = args.output_dir / "toricblm_afdb_gcs_diverse_plan_manifest.json"
    if compatibility_path != manifest_path:
        compatibility_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"repartitioned_records={total}")
    print(f"workers={args.workers}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
