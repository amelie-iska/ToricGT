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
        "# ToricBLM Bio Forest-of-Thought Graphification for ToricGT",
        "",
        "This repository stores Parquet-first biological graph/FoT curation artifacts for ToricGT/ToricBLM.",
        "The upload contains deterministic graph/FoT rows plus any locally curated real-coordinate structure shards selected by the publish command.",
        "",
        "Raw upstream Parquet data is not uploaded in this initial push. Local raw sources remain organized by symlink under the ToricGT data tree and require separate license review before redistribution.",
        "",
        "## Files",
        "",
        "- `derived/*.parquet` graphified source-data slices",
        "- `splits/*/*.parquet` leakage-aware or ProTrek-aware graph/FoT splits when present",
        "- `structures/*/{train,validation,test}/*.parquet` real coordinate-bearing AFDB/PDB/PubChem3D shards when present",
        "- `authored/accepted/accepted_records.parquet` when authored records are present",
        "- `authored/accepted/validation_report.json`",
        "- `authored/records/*.json` inspectable handwritten source records",
        "- `authored/source_anchors/*.parquet` source-row anchors for future handwritten batches",
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
        "Each row contains flat loader columns plus `graph_json`, `forest_json`, `thought_forest_json`, `convextok_dag_json`, `training_views_json`, `enrichment_status_json`, `metadata_json`, hashes, split cluster metadata, and quality flags.",
        "",
        "`thought_forest_json` materializes source-grounded Forest-of-Thought trees with expansion, self-correction, and consensus edges. `convextok_dag_json` stores a compact ConvexTok byte-boundary DAG with free byte fallback edges, matched priced-token edges, and a min-plus shortest path. Graph JSON records include directed nodes/edges, GFlowNet reward metadata, continuous embedding proxies, TokenGT/TropicalGT/ToricGT fields, active support nodes, and leakage-resistant split metadata.",
        "",
        "## Safety And License Notes",
        "",
        "Biomedical content is for model-training curation and mechanism-focused annotation reasoning, not patient-specific medical advice. Coordinate-native structure shards are generated only from real AFDB/PDB/PubChem3D records; raw upstream caches such as mmCIF/SDF downloads are not uploaded by this script.",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def upload_plan(data_dir: Path, include_jsonl: bool, include_structures: bool, include_splits: bool) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for pattern in ("uniprot_fot_graphified*.parquet", "toricblm_fot_graphified*.parquet"):
        for parquet_path in sorted((data_dir / "derived").glob(pattern)):
            files.append((parquet_path, f"derived/{parquet_path.name}"))
    if include_splits:
        for parquet_path in sorted((data_dir / "splits").glob("*/*.parquet")):
            files.append((parquet_path, str(parquet_path.relative_to(data_dir))))
        for json_path in sorted((data_dir / "splits").glob("*/*.json")):
            files.append((json_path, str(json_path.relative_to(data_dir))))
    if include_structures:
        for parquet_path in sorted((data_dir / "structures").glob("*/*/*.parquet")):
            # Upload train/validation/test shards, never raw mmCIF/SDF caches.
            files.append((parquet_path, str(parquet_path.relative_to(data_dir))))
        for json_path in sorted((data_dir / "structures").glob("*/*.json")):
            files.append((json_path, str(json_path.relative_to(data_dir))))
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
    for record_path in sorted((data_dir / "authored" / "records").glob("*.json")):
        files.append((record_path, f"authored/records/{record_path.name}"))
    for anchor_path in sorted((data_dir / "authored" / "source_anchors").glob("*.parquet")):
        files.append((anchor_path, f"authored/source_anchors/{anchor_path.name}"))
    for anchor_manifest in sorted((data_dir / "authored" / "source_anchors").glob("*_manifest.json")):
        files.append((anchor_manifest, f"authored/source_anchors/{anchor_manifest.name}"))
    if include_jsonl:
        for pattern in ("uniprot_fot_graphified*.jsonl", "toricblm_fot_graphified*.jsonl"):
            for jsonl_path in sorted((data_dir / "derived").glob(pattern)):
                files.append((jsonl_path, f"derived/{jsonl_path.name}"))
        files.append((data_dir / "authored" / "accepted" / "accepted_records.jsonl", "authored/accepted/accepted_records.jsonl"))
    return [(path, repo_path) for path, repo_path in files if path.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/uniprot_fot")
    parser.add_argument("--repo-id", default="AmelieSchreiber/toricblm_fot")
    parser.add_argument("--include-jsonl", action="store_true", help="Also upload inspectable JSONL.")
    parser.add_argument("--include-structures", action="store_true", help="Upload curated real-coordinate structure Parquet shards and manifests, not raw caches.")
    parser.add_argument("--include-splits", action="store_true", help="Upload leakage/ProTrek split Parquet files and reports.")
    parser.add_argument("--private", action="store_true", help="Create/update as a private dataset.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = upload_plan(
        data_dir,
        include_jsonl=args.include_jsonl,
        include_structures=args.include_structures,
        include_splits=args.include_splits,
    )
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
