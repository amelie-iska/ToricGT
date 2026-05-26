#!/usr/bin/env python
"""Shard curated train/validation/test Parquet files for Hugging Face upload."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


def shard_split(
    source: Path,
    output_dir: Path,
    split: str,
    target_shard_bytes: int,
    batch_size: int,
    compression: str,
    overwrite: bool,
) -> dict[str, object]:
    if not source.exists():
        raise FileNotFoundError(source)
    split_dir = output_dir / split
    if split_dir.exists() and overwrite:
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(source)
    total_rows = parquet_file.metadata.num_rows
    source_bytes = source.stat().st_size
    rows_per_shard = max(1, int(total_rows * target_shard_bytes / max(source_bytes, 1)))
    existing = sorted(split_dir.glob(f"{split}-*.parquet"))
    if existing and not overwrite:
        return {
            "split": split,
            "source": str(source),
            "source_bytes": source_bytes,
            "total_rows": total_rows,
            "rows_per_shard": rows_per_shard,
            "target_shard_bytes": target_shard_bytes,
            "compression": compression,
            "skipped_existing": True,
            "shards": [{"path": str(path), "bytes": path.stat().st_size} for path in existing],
        }

    writer = None
    shard_idx = 0
    shard_rows = 0
    written_rows = 0
    current_path: Path | None = None
    shards: list[dict[str, object]] = []

    def close_writer() -> None:
        nonlocal writer, shard_rows, current_path
        if writer is not None:
            writer.close()
            assert current_path is not None
            shards.append(
                {
                    "path": str(current_path),
                    "rows": shard_rows,
                    "bytes": current_path.stat().st_size,
                }
            )
        writer = None
        shard_rows = 0
        current_path = None

    progress = tqdm(total=total_rows, desc=f"shard {split}", unit="rows")
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        offset = 0
        while offset < table.num_rows:
            if writer is None:
                current_path = split_dir / f"{split}-{shard_idx:05d}.parquet"
                writer = pq.ParquetWriter(
                    current_path,
                    table.schema,
                    compression=compression,
                    use_dictionary=True,
                )
                shard_idx += 1
            remaining = rows_per_shard - shard_rows
            take = min(remaining, table.num_rows - offset)
            writer.write_table(table.slice(offset, take))
            offset += take
            shard_rows += take
            written_rows += take
            progress.update(take)
            size_limit_reached = current_path is not None and current_path.exists() and current_path.stat().st_size >= target_shard_bytes
            if shard_rows >= rows_per_shard or size_limit_reached:
                close_writer()
    close_writer()
    progress.close()
    return {
        "split": split,
        "source": str(source),
        "source_bytes": source_bytes,
        "total_rows": total_rows,
        "written_rows": written_rows,
        "rows_per_shard": rows_per_shard,
        "target_shard_bytes": target_shard_bytes,
        "compression": compression,
        "skipped_existing": False,
        "shards": shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/curated")
    parser.add_argument("--output-dir", default="data/curated_hf_shards")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--target-shard-mb", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target_shard_mb": args.target_shard_mb,
        "splits": [],
    }
    target_bytes = args.target_shard_mb * 1024 * 1024
    for split in args.splits:
        result = shard_split(
            input_dir / f"{split}.parquet",
            output_dir,
            split,
            target_bytes,
            args.batch_size,
            args.compression,
            args.overwrite,
        )
        manifest["splits"].append(result)

    manifest_path = output_dir / "shard_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
