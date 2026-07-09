#!/usr/bin/env python3
"""Fill an existing AFDB GCS selection plan to a target count.

This is used when the enzyme-biased selection has enough enzyme coverage and
the remaining plan budget should be filled with general diverse UniProt
accessions so downstream coordinate curation can proceed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from plan_afdb_gcs_diverse_accessions import (
    diversity_bucket,
    enzyme_profile,
    length_bucket,
    load_existing_accessions,
    parquet_columns,
    row_accessions,
    row_entry_name,
    row_function,
    row_protein_name,
    row_sequence,
    safe_text,
    source_paths,
)


def read_existing_plan(plan_dir: Path) -> tuple[set[str], Counter[str], Counter[str], int]:
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    accepted = 0
    for path in sorted(plan_dir.glob("worker_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                accession = str(row.get("accession") or "").strip()
                if accession:
                    seen.add(accession)
                selection = row.get("selection") or {}
                tier = str(selection.get("tier") or "unknown")
                bucket = str(selection.get("diversity_bucket") or "")
                counts[tier] += 1
                if bucket:
                    bucket_counts[bucket] += 1
                accepted += 1
    return seen, counts, bucket_counts, accepted


def open_append_handles(plan_dir: Path, workers: int) -> list[Any]:
    plan_dir.mkdir(parents=True, exist_ok=True)
    return [(plan_dir / f"worker_{idx:02d}.jsonl").open("a", encoding="utf-8") for idx in range(workers)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-records", type=int, default=5_000_000)
    parser.add_argument("--max-scan-rows", type=int, default=0)
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--existing-parquet-glob", action="append", default=[])
    parser.add_argument("--max-accessions-per-row", type=int, default=4)
    parser.add_argument("--max-sequence-chars", type=int, default=8192)
    parser.add_argument("--max-function-chars", type=int, default=4096)
    parser.add_argument("--bucket-mod", type=int, default=8192)
    parser.add_argument("--max-per-diversity-bucket", type=int, default=4096)
    args = parser.parse_args()

    inputs = source_paths(args.input)
    existing = load_existing_accessions(args.existing_parquet_glob)
    seen, counts, bucket_counts, accepted = read_existing_plan(args.plan_dir)
    seen.update(existing)
    initial_accepted = accepted
    skipped: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    scanned = 0
    handles = open_append_handles(args.plan_dir, args.workers)
    try:
        for path in inputs:
            if accepted >= int(args.target_records):
                break
            try:
                columns = parquet_columns(path)
                pf = pq.ParquetFile(path)
            except Exception as exc:
                skipped["unreadable_input_parquet"] += 1
                print(
                    f"skip_unreadable_input_parquet path={path} error={type(exc).__name__}:{exc}",
                    flush=True,
                )
                continue
            for batch in pf.iter_batches(batch_size=512, columns=columns):
                if accepted >= int(args.target_records):
                    break
                table = pa.Table.from_batches([batch])
                for row in table.to_pylist():
                    if accepted >= int(args.target_records):
                        break
                    if args.max_scan_rows > 0 and scanned >= args.max_scan_rows:
                        break
                    scanned += 1
                    seq = row_sequence(row, args.max_sequence_chars)
                    if not seq:
                        skipped["missing_sequence"] += 1
                        continue
                    accessions = row_accessions(row, max_accessions=args.max_accessions_per_row)
                    if not accessions:
                        skipped["missing_uniprot_accession"] += 1
                        continue
                    function_text = row_function(row, args.max_function_chars)
                    protein_name = row_protein_name(row)
                    profile = enzyme_profile(row, function_text, protein_name)
                    original_tier = str(profile["enzyme_tier"])
                    bucket = diversity_bucket(row, seq, function_text, "general_fill", args.bucket_mod)
                    if bucket_counts[bucket] >= int(args.max_per_diversity_bucket):
                        skipped["bucket_full_general_fill"] += 1
                        continue
                    for accession in accessions:
                        if accession in seen:
                            skipped["already_seen_accession"] += 1
                            continue
                        seen.add(accession)
                        payload = {
                            "schema": "toricblm.afdb_gcs_diverse_accession_plan.v1",
                            "accession": accession,
                            "entry_name": row_entry_name(row, accession),
                            "protein_name": protein_name,
                            "sequence": seq,
                            "function_text": function_text,
                            "source_parquet": str(path),
                            "source_row_index": scanned,
                            "selection": {
                                "tier": "general_fill",
                                "original_enzyme_tier": original_tier,
                                "diversity_bucket": bucket,
                                "selection_rank": accepted,
                                "enzyme_profile": profile,
                                "sequence_length": len(seq),
                                "length_bucket": length_bucket(seq),
                                "taxon": safe_text(row.get("common_taxon") or row.get("rep_organism"), 160),
                                "taxon_id": safe_text(row.get("common_taxon_id") or row.get("rep_organism_tax_id"), 64),
                                "member_count": int(row.get("member_count") or 0)
                                if str(row.get("member_count") or "").isdigit()
                                else 0,
                                "fill_policy": "enzyme coverage accepted as sufficient; remaining target filled with general diverse accessions",
                            },
                        }
                        handles[accepted % int(args.workers)].write(
                            json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n"
                        )
                        accepted += 1
                        counts["general_fill"] += 1
                        bucket_counts[bucket] += 1
                        source_counts[str(path)] += 1
                        if accepted % 10_000 == 0:
                            print(
                                f"filled accepted={accepted} added={accepted - initial_accepted} scanned={scanned}",
                                flush=True,
                            )
                        break
                if args.max_scan_rows > 0 and scanned >= args.max_scan_rows:
                    break
    finally:
        for handle in handles:
            handle.close()

    manifest = {
        "schema": "toricblm.afdb_gcs_diverse_accession_plan_manifest.v1",
        "target_records": int(args.target_records),
        "accepted_records": accepted,
        "initial_accepted_records": initial_accepted,
        "filled_records": accepted - initial_accepted,
        "scanned_rows_for_fill": scanned,
        "counts": dict(counts),
        "skipped": dict(skipped),
        "existing_accessions_loaded": len(existing),
        "worker_plan_dir": str(args.plan_dir),
        "worker_plan_files": [str(args.plan_dir / f"worker_{idx:02d}.jsonl") for idx in range(args.workers)],
        "source_counts": dict(source_counts),
        "policy": (
            "Existing 15%-cap enzyme-biased plan was retained. Enzyme coverage was deemed sufficient by operator; "
            "remaining slots were filled with general diverse UniProt accessions so AFDB coordinate curation can proceed."
        ),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if accepted < int(args.target_records):
        raise SystemExit(f"Only filled to {accepted}; target was {args.target_records}")


if __name__ == "__main__":
    main()
