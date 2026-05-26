#!/usr/bin/env python3
"""Annotate curated Parquet rows with Hebrew niqqud quality flags."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from toricgt.hebrew import hebrew_niqqud_stats

NIQQUD_POLICY = (
    "preserve existing niqqud; prefer same-consonant pointed variants when present; "
    "flag unpointed Hebrew for filtering or later restoration"
)


def parse_json(value: str | None) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def compact_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def likely_hebrew_row(dataset: str, language: str, task_family: str) -> bool:
    language = language.lower()
    task_family = task_family.lower()
    dataset = dataset.lower()
    return (
        "he" in language
        or "arc" in language
        or "hebrew" in task_family
        or "rabbinic" in task_family
        or "hebrew" in dataset
        or "sefaria" in dataset
        or "unimorph" in dataset
    )


def niqqud_texts_for_row(row: dict) -> list[str]:
    dataset = str(row.get("dataset") or "")
    if dataset in {
        "unimorph/universal_morphologies",
        "Sefaria/Rabbinic-Hebrew-English-Pairs",
        "Sefaria/hebrew_library",
    }:
        return [str(row.get("answer") or "")]
    return [
        str(row.get("question") or ""),
        str(row.get("answer") or ""),
        str(row.get("solution") or ""),
        str(row.get("reasoning") or ""),
    ]


def update_row(row: dict, drop_unpointed_hebrew: bool, update_metadata: bool = False) -> dict | None:
    if (
        not likely_hebrew_row(
            str(row.get("dataset") or ""),
            str(row.get("language") or ""),
            str(row.get("task_family") or ""),
        )
    ):
        return row
    stats = hebrew_niqqud_stats(*dict.fromkeys(part for part in niqqud_texts_for_row(row) if part))
    if not stats.contains_hebrew:
        return row
    if drop_unpointed_hebrew and stats.has_hebrew_without_niqqud:
        return None
    flags = parse_json(row.get("quality_flags_json"))
    payload = stats.to_dict()
    flags.update(payload)
    flags["toricgt_niqqud_policy"] = NIQQUD_POLICY
    if update_metadata:
        metadata = parse_json(row.get("metadata_json"))
        metadata["toricgt_hebrew_niqqud"] = payload
        metadata["toricgt_niqqud_policy"] = NIQQUD_POLICY
        row["metadata_json"] = compact_json(metadata)
    row["quality_flags_json"] = compact_json(flags)
    return row


def enrich_file(
    path: Path,
    in_place: bool,
    drop_unpointed_hebrew: bool,
    batch_size: int,
    update_metadata: bool,
) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    kept = 0
    dropped = 0
    tmp_path = Path(tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))[1]) if in_place else None
    writer: pq.ParquetWriter | None = None
    try:
        for batch in tqdm(parquet.iter_batches(batch_size=batch_size), desc=f"niqqud {path.name}"):
            table = pa.Table.from_batches([batch], schema=schema)
            dataset_values = table.column("dataset").to_pylist()
            language_values = table.column("language").to_pylist()
            task_values = table.column("task_family").to_pylist()
            likely_mask = [
                likely_hebrew_row(str(ds or ""), str(lang or ""), str(task or ""))
                for ds, lang, task in zip(dataset_values, language_values, task_values, strict=True)
            ]
            if not any(likely_mask):
                rows_kept = table.num_rows
                if in_place and rows_kept:
                    if writer is None:
                        assert tmp_path is not None
                        writer = pq.ParquetWriter(tmp_path, schema, compression="zstd")
                    writer.write_table(table)
                kept += rows_kept
                continue

            columns = {
                name: table.column(name).to_pylist()
                for name in (
                    "dataset",
                    "language",
                    "task_family",
                    "question",
                    "answer",
                    "solution",
                    "reasoning",
                    "metadata_json",
                    "quality_flags_json",
                )
            }
            keep_mask: list[bool] = []
            flags_out = list(columns["quality_flags_json"])
            metadata_out = list(columns["metadata_json"])
            for idx, likely in enumerate(likely_mask):
                if not likely:
                    keep_mask.append(True)
                    continue
                row = {name: values[idx] for name, values in columns.items()}
                updated = update_row(
                    row,
                    drop_unpointed_hebrew=drop_unpointed_hebrew,
                    update_metadata=update_metadata,
                )
                if updated is None:
                    keep_mask.append(False)
                    dropped += 1
                    continue
                keep_mask.append(True)
                flags_out[idx] = updated["quality_flags_json"]
                metadata_out[idx] = updated["metadata_json"]

            if not all(keep_mask):
                table = table.filter(pa.array(keep_mask))
                flags_out = [value for value, keep in zip(flags_out, keep_mask, strict=True) if keep]
                metadata_out = [value for value, keep in zip(metadata_out, keep_mask, strict=True) if keep]
            flags_index = schema.get_field_index("quality_flags_json")
            table = table.set_column(flags_index, "quality_flags_json", pa.array(flags_out, type=pa.string()))
            if update_metadata:
                metadata_index = schema.get_field_index("metadata_json")
                table = table.set_column(metadata_index, "metadata_json", pa.array(metadata_out, type=pa.string()))
            if table.num_rows and in_place:
                if writer is None:
                    assert tmp_path is not None
                    writer = pq.ParquetWriter(tmp_path, schema, compression="zstd")
                writer.write_table(table)
            kept += table.num_rows
        if writer is not None:
            writer.close()
            writer = None
        if in_place and tmp_path is not None:
            shutil.move(str(tmp_path), str(path))
    finally:
        if writer is not None:
            writer.close()
        if in_place and tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--drop-unpointed-hebrew", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32_768)
    parser.add_argument("--include-all", action="store_true")
    parser.add_argument("--update-metadata", action="store_true")
    args = parser.parse_args()

    root = Path(args.curated_dir)
    paths = [root / "train.parquet", root / "validation.parquet", root / "test.parquet"]
    if args.include_all:
        paths.insert(0, root / "all.parquet")
    summary = {}
    for path in paths:
        kept, dropped = enrich_file(
            path,
            in_place=args.in_place,
            drop_unpointed_hebrew=args.drop_unpointed_hebrew,
            batch_size=args.batch_size,
            update_metadata=args.update_metadata,
        )
        summary[str(path)] = {"kept": kept, "dropped": dropped}
    report_path = root / "niqqud_report.json"
    report_path.write_text(compact_json(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
