#!/usr/bin/env python3
"""Summarize Hebrew niqqud coverage in curated ToricGT Parquet splits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm.auto import tqdm


def parse_flags(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def empty_counts() -> dict[str, int]:
    return {
        "rows": 0,
        "hebrew_rows": 0,
        "pointed_hebrew_rows": 0,
        "unpointed_hebrew_rows": 0,
        "hebrew_letters": 0,
        "niqqud_marks": 0,
    }


def add_row(counts: dict[str, int], flags: dict[str, Any]) -> None:
    counts["rows"] += 1
    if not flags.get("contains_hebrew"):
        return
    counts["hebrew_rows"] += 1
    counts["hebrew_letters"] += int(flags.get("hebrew_letters") or 0)
    counts["niqqud_marks"] += int(flags.get("niqqud_marks") or 0)
    if flags.get("has_niqqud"):
        counts["pointed_hebrew_rows"] += 1
    if flags.get("has_hebrew_without_niqqud"):
        counts["unpointed_hebrew_rows"] += 1


def finalize(counts: dict[str, int]) -> dict[str, int | float]:
    out: dict[str, int | float] = dict(counts)
    hebrew_rows = counts["hebrew_rows"]
    letters = counts["hebrew_letters"]
    out["pointed_hebrew_row_fraction"] = (
        counts["pointed_hebrew_rows"] / hebrew_rows if hebrew_rows else 0.0
    )
    out["unpointed_hebrew_row_fraction"] = (
        counts["unpointed_hebrew_rows"] / hebrew_rows if hebrew_rows else 0.0
    )
    out["niqqud_per_hebrew_letter"] = counts["niqqud_marks"] / letters if letters else 0.0
    return out


def summarize_file(path: Path, batch_size: int) -> dict[str, Any]:
    totals = empty_counts()
    by_dataset: dict[str, dict[str, int]] = defaultdict(empty_counts)
    parquet = pq.ParquetFile(path)
    for batch in tqdm(
        parquet.iter_batches(columns=["dataset", "quality_flags_json"], batch_size=batch_size),
        total=parquet.metadata.num_row_groups,
        desc=f"niqqud-report {path.name}",
    ):
        datasets = batch.column(0).to_pylist()
        flags_values = batch.column(1).to_pylist()
        for dataset, raw_flags in zip(datasets, flags_values, strict=True):
            flags = parse_flags(raw_flags)
            add_row(totals, flags)
            if flags.get("contains_hebrew"):
                add_row(by_dataset[str(dataset or "")], flags)
    return {
        "totals": finalize(totals),
        "by_dataset": {
            dataset: finalize(counts)
            for dataset, counts in sorted(
                by_dataset.items(),
                key=lambda item: item[1]["hebrew_rows"],
                reverse=True,
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=65_536)
    args = parser.parse_args()

    root = Path(args.curated_dir)
    output = Path(args.output) if args.output else root / "niqqud_report.json"
    split_paths = {
        split: root / f"{split}.parquet" for split in ("train", "validation", "test")
    }
    report: dict[str, Any] = {
        "policy": (
            "Preserve existing niqqud; prefer same-consonant pointed variants when present; "
            "flag unpointed Hebrew for filtering or later restoration. Do not synthesize "
            "niqqud into attested source text unless a vetted diacritization pipeline is "
            "explicitly enabled."
        ),
        "strict_pointed_filter": "quality_flags_json.has_hebrew_without_niqqud == false",
        "splits": {},
    }
    aggregate = empty_counts()
    aggregate_by_dataset: dict[str, dict[str, int]] = defaultdict(empty_counts)
    for split, path in split_paths.items():
        if not path.exists():
            continue
        split_report = summarize_file(path, batch_size=args.batch_size)
        report["splits"][split] = split_report
        totals = split_report["totals"]
        for key in empty_counts():
            aggregate[key] += int(totals[key])
        for dataset, counts in split_report["by_dataset"].items():
            target = aggregate_by_dataset[dataset]
            for key in empty_counts():
                target[key] += int(counts[key])
    report["aggregate"] = finalize(aggregate)
    report["aggregate_by_dataset"] = {
        dataset: finalize(counts)
        for dataset, counts in sorted(
            aggregate_by_dataset.items(),
            key=lambda item: item[1]["hebrew_rows"],
            reverse=True,
        )
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
