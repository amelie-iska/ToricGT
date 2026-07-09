#!/usr/bin/env python3
"""Repair AFDB curation progress after an interrupted writer.

The AFDB builder appends accepted accessions to ``build_progress.jsonl`` after
adding them to an in-memory shard buffer.  If an older process is interrupted
before ``writer.close()``, some accepted accessions may exist in progress but
not in any written Parquet shard.  This utility keeps skipped/error progress
records and accepted records that are actually present in output shards, while
removing unflushed accepted records so ``--resume`` can retry them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def shard_accessions(out_dir: Path) -> set[str]:
    accessions: set[str] = set()
    for path in sorted(out_dir.glob("*/*.parquet")):
        try:
            pf = pq.ParquetFile(path)
            if "uniprot_accession" not in pf.schema_arrow.names:
                continue
            table = pq.read_table(path, columns=["uniprot_accession"])
            for row in table.to_pylist():
                accession = row.get("uniprot_accession")
                if accession:
                    accessions.add(str(accession))
        except Exception:
            continue
    return accessions


def repair(out_dir: Path, *, dry_run: bool) -> dict[str, int | str | bool]:
    progress_path = out_dir / "build_progress.jsonl"
    if not progress_path.exists():
        return {"progress_path": str(progress_path), "dry_run": dry_run, "status": "missing_progress", "kept": 0, "removed": 0}
    present = shard_accessions(out_dir)
    kept_lines: list[str] = []
    removed = 0
    accepted_seen = 0
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                kept_lines.append(line)
                continue
            if item.get("status") == "accepted":
                accepted_seen += 1
                accession = str(item.get("accession") or "")
                if accession and accession not in present:
                    removed += 1
                    continue
            kept_lines.append(line if line.endswith("\n") else line + "\n")
    if removed and not dry_run:
        backup = progress_path.with_suffix(progress_path.suffix + ".pre_repair")
        if not backup.exists():
            backup.write_text(progress_path.read_text(encoding="utf-8"), encoding="utf-8")
        progress_path.write_text("".join(kept_lines), encoding="utf-8")
    return {
        "progress_path": str(progress_path),
        "dry_run": dry_run,
        "parquet_accessions": len(present),
        "accepted_progress_seen": accepted_seen,
        "kept": len(kept_lines),
        "removed_unflushed_accepted": removed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(repair(args.out_dir, dry_run=bool(args.dry_run)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
