#!/usr/bin/env python3
"""Publish curated ToricGT split files to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from tqdm.auto import tqdm


def write_dataset_card(curated_dir: Path, destination: Path, repo_id: str) -> None:
    report_path = curated_dir / "split_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    stats = report.get("stats", {})
    lines = [
        "---",
        "license: other",
        "task_categories:",
        "- text-generation",
        "- graph-ml",
        "language:",
        "- en",
        "- he",
        "pretty_name: ToricGT Curated Graph Reasoning Splits",
        "---",
        "",
        "# ToricGT Curated Graph Reasoning Splits",
        "",
        "Curated working dataset repository for ToricGT.",
        "",
        "The upload contains only curated split Parquet files and metadata generated locally.",
        "Raw upstream downloads are not uploaded. Each row preserves source dataset, license, split, hashes, and graph JSON fields for audit.",
        "",
        "Hebrew/Jewish-text records are sourced from Sefaria and UniMorph Hebrew sources; Christian-branded biblical-language datasets are intentionally excluded.",
        "",
        "## Files",
        "",
        "- `train.parquet`",
        "- `validation.parquet`",
        "- `test.parquet`",
        "- `all.parquet` if `--include-all` was used",
        "- `manifest.json`",
        "- `split_report.json`",
        "- `split_report.md`",
        "",
        "## Split Summary",
        "",
        f"- Repository: `{repo_id}`",
        f"- Total rows: `{stats.get('total_records', 'unknown')}`",
        f"- Train rows: `{stats.get('split_counts', {}).get('train', 'unknown')}`",
        f"- Validation rows: `{stats.get('split_counts', {}).get('validation', 'unknown')}`",
        f"- Test rows: `{stats.get('split_counts', {}).get('test', 'unknown')}`",
        "",
        "## License Notes",
        "",
        "This is a mixed-source research dataset. Preserve attribution columns and review upstream licenses before model redistribution.",
        "GPL-licensed and share-alike sources should be handled separately when downstream release terms matter.",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def selected_files(curated_dir: Path, include_all: bool, include_by_dataset: bool) -> list[Path]:
    names = [
        "train.parquet",
        "validation.parquet",
        "test.parquet",
        "manifest.json",
        "split_report.json",
        "split_report.md",
        "niqqud_report.json",
    ]
    if include_all:
        names.insert(0, "all.parquet")
    files = [curated_dir / name for name in names if (curated_dir / name).exists()]
    if include_by_dataset:
        files.extend(sorted((curated_dir / "by_dataset").glob("*.parquet")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--repo-id", required=True, help="Example: AmelieSchreiber/toricgt-curated-splits")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", action="store_true", help="Create or update the dataset repo as private.")
    visibility.add_argument("--public", action="store_true", help="Create or update the dataset repo as public.")
    parser.add_argument("--include-all", action="store_true", help="Also upload all.parquet.")
    parser.add_argument("--include-by-dataset", action="store_true", help="Also upload by_dataset shards.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    curated_dir = Path(args.curated_dir)
    files = selected_files(curated_dir, include_all=args.include_all, include_by_dataset=args.include_by_dataset)
    if not files:
        raise SystemExit(f"no curated files found under {curated_dir}")

    with tempfile.TemporaryDirectory(prefix="toricgt_hf_card_") as tmp:
        card = Path(tmp) / "README.md"
        write_dataset_card(curated_dir, card, args.repo_id)
        upload_plan = [(card, "README.md")]
        for path in files:
            if path.parent.name == "by_dataset":
                repo_path = f"by_dataset/{path.name}"
            else:
                repo_path = path.name
            upload_plan.append((path, repo_path))

        total_bytes = sum(path.stat().st_size for path, _ in upload_plan if path.exists())
        print({"repo_id": args.repo_id, "files": len(upload_plan), "bytes": total_bytes, "dry_run": args.dry_run})
        for path, repo_path in upload_plan:
            print(f"{path} -> {repo_path}")
        if args.dry_run:
            return

        private = bool(args.private and not args.public)
        create_repo(args.repo_id, repo_type="dataset", private=private, exist_ok=True)
        api = HfApi()
        for path, repo_path in tqdm(upload_plan, desc="hf upload"):
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=repo_path,
                repo_id=args.repo_id,
                repo_type="dataset",
            )


if __name__ == "__main__":
    main()
