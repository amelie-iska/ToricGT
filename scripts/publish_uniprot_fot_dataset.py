#!/usr/bin/env python3
"""Publish the ToricGT UniProt FoT data workspace to Hugging Face."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from tqdm.auto import tqdm


def write_card(data_dir: Path, destination: Path, repo_id: str) -> None:
    manifest_path = data_dir / "manifests" / "uniprot_fot_build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    totals = manifest.get("raw_manifest", {}).get("totals", {})
    derived = manifest.get("derived", {})
    lines = [
        "---",
        "license: other",
        "task_categories:",
        "- text-generation",
        "- graph-ml",
        "- feature-extraction",
        "language:",
        "- en",
        "pretty_name: UniProt Forest-of-Thought Graphification for ToricGT",
        "---",
        "",
        "# UniProt Forest-of-Thought Graphification for ToricGT",
        "",
        "This repository stores Parquet-first biological graph/FoT curation artifacts for ToricGT.",
        "The current upload is an initial deterministic graphification sample, not the final authored reasoning corpus.",
        "",
        "Raw upstream Parquet data is not uploaded in this initial push. Local raw sources remain organized by symlink under the ToricGT data tree and require separate license review before redistribution.",
        "",
        "## Files",
        "",
        "- `derived/*.parquet` graphified source-data slices",
        "- `authored/accepted/accepted_records.parquet` when authored records are present",
        "- `authored/accepted/validation_report.json`",
        "- `manifests/uniprot_fot_build_manifest.json`",
        "- `schema/uniprot_fot_record.schema.json`",
        "- `LOCAL_README.md`",
        "- `authored/README.md`",
        "- `authored/progress_log.md`",
        "- `GOAL.md`",
        "",
        "## Current Local Build",
        "",
        f"- Repository: `{repo_id}`",
        f"- Derived sample records: `{derived.get('records', 'unknown')}`",
        f"- Raw datasets scanned: `{totals.get('datasets', 'unknown')}`",
        f"- Raw Parquet files scanned: `{totals.get('parquet_files', 'unknown')}`",
        f"- Readable raw Parquet files: `{totals.get('readable_parquet_files', 'unknown')}`",
        f"- Raw rows indexed in manifest: `{totals.get('rows', 'unknown')}`",
        "",
        "## Format",
        "",
        "Each row contains flat loader columns plus `graph_json`, `forest_json`, `metadata_json`, hashes, split cluster metadata, and quality flags. Graph JSON records include directed nodes/edges, GFlowNet reward metadata, continuous embedding proxies, TokenGT/TropicalGT/ToricGT fields, active support nodes, and leakage-resistant split metadata.",
        "",
        "## Safety And License Notes",
        "",
        "Biomedical content is for model-training curation and mechanism-focused annotation reasoning, not patient-specific medical advice. Source licenses and redistribution rights must be reviewed before uploading larger raw-derived corpora.",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def upload_plan(data_dir: Path, include_jsonl: bool) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for parquet_path in sorted((data_dir / "derived").glob("uniprot_fot_graphified*.parquet")):
        files.append((parquet_path, f"derived/{parquet_path.name}"))
    files.extend(
        [
        (data_dir / "manifests" / "uniprot_fot_build_manifest.json", "manifests/uniprot_fot_build_manifest.json"),
        (data_dir / "schema" / "uniprot_fot_record.schema.json", "schema/uniprot_fot_record.schema.json"),
        (data_dir / "README.md", "LOCAL_README.md"),
        (data_dir / "GOAL.md", "GOAL.md"),
        (data_dir / "authored" / "README.md", "authored/README.md"),
        (data_dir / "authored" / "progress_log.md", "authored/progress_log.md"),
        (data_dir / "authored" / "accepted" / "accepted_records.parquet", "authored/accepted/accepted_records.parquet"),
        (data_dir / "authored" / "accepted" / "validation_report.json", "authored/accepted/validation_report.json"),
        ]
    )
    if include_jsonl:
        for jsonl_path in sorted((data_dir / "derived").glob("uniprot_fot_graphified*.jsonl")):
            files.append((jsonl_path, f"derived/{jsonl_path.name}"))
        files.append((data_dir / "authored" / "accepted" / "accepted_records.jsonl", "authored/accepted/accepted_records.jsonl"))
    return [(path, repo_path) for path, repo_path in files if path.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/uniprot_fot")
    parser.add_argument("--repo-id", default="AmelieSchreiber/uniprot_fot")
    parser.add_argument("--include-jsonl", action="store_true", help="Also upload inspectable JSONL.")
    parser.add_argument("--private", action="store_true", help="Create/update as a private dataset.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = upload_plan(data_dir, include_jsonl=args.include_jsonl)
    if not files:
        raise SystemExit(f"no uploadable files found under {data_dir}")

    with tempfile.TemporaryDirectory(prefix="uniprot_fot_hf_") as tmp:
        card = Path(tmp) / "README.md"
        write_card(data_dir, card, args.repo_id)
        plan = [(card, "README.md"), *files]
        total_bytes = sum(path.stat().st_size for path, _ in plan)
        print(json.dumps({"repo_id": args.repo_id, "files": len(plan), "bytes": total_bytes, "dry_run": args.dry_run}, indent=2))
        for path, repo_path in plan:
            print(f"{path} -> {repo_path}")
        if args.dry_run:
            return

        create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        api = HfApi()
        for path, repo_path in tqdm(plan, desc="hf upload"):
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=repo_path,
                repo_id=args.repo_id,
                repo_type="dataset",
            )


if __name__ == "__main__":
    main()
