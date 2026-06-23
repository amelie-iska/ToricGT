#!/usr/bin/env python3
"""Extract source anchors for handwritten UniProt FoT authoring.

This script does not generate reasoning records. It only collects compact,
auditable source-row summaries that a human/Codex author can inspect before
writing individual FoT artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


RAW_ROOT = Path("/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale")
OUTPUT_DIR = Path("/home/iska/Documents/amelie/bio/ToricGT/data/uniprot_fot/authored/source_anchors")
EC_RE = re.compile(r"\bEC\s+([0-9-]+(?:\.[0-9-]+){1,3})\b")


ANCHOR_SCHEMA = pa.schema(
    [
        ("anchor_id", pa.string()),
        ("dataset", pa.string()),
        ("source_file", pa.string()),
        ("source_row_index", pa.int64()),
        ("entry_id", pa.string()),
        ("title", pa.string()),
        ("domain_hint", pa.string()),
        ("sequence_length", pa.int64()),
        ("summary_json", pa.large_string()),
        ("source_fields_json", pa.large_string()),
    ]
)


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, prefix: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{prefix}.jsonl"
    parquet_path = output_dir / f"{prefix}.parquet"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    table = pa.Table.from_pylist(rows, schema=ANCHOR_SCHEMA)
    pq.write_table(table, parquet_path, compression="zstd", use_dictionary=True)
    counts = Counter(row["dataset"] for row in rows)
    manifest = {
        "jsonl": str(jsonl_path),
        "parquet": str(parquet_path),
        "records": len(rows),
        "counts_by_dataset": dict(counts),
        "note": "Source anchors only; no authored reasoning text is generated here.",
    }
    manifest_path = output_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest


def compact(text: Any, limit: int = 600) -> str:
    if text is None:
        return ""
    value = str(text)
    return value if len(value) <= limit else value[:limit] + "...<truncated>"


def rows_from_parquet(path: Path, columns: list[str], limit: int, predicate=None) -> Iterable[tuple[int, dict[str, Any]]]:
    produced = 0
    try:
        pf = pq.ParquetFile(path)
    except Exception:
        return
    for batch in pf.iter_batches(batch_size=4096, columns=columns):
        for row in batch.to_pylist():
            if predicate is not None and not predicate(row):
                continue
            yield produced, row
            produced += 1
            if produced >= limit:
                return


def make_anchor(
    dataset: str,
    source_file: Path,
    row_index: int,
    entry_id: str,
    title: str,
    domain_hint: str,
    sequence_length: int,
    summary: dict[str, Any],
    source_fields: dict[str, Any],
) -> dict[str, Any]:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", entry_id or title).strip("_")[:48]
    anchor_id = f"anchor_{dataset}_{row_index:06d}_{safe}"
    return {
        "anchor_id": anchor_id,
        "dataset": dataset,
        "source_file": str(source_file),
        "source_row_index": row_index,
        "entry_id": entry_id,
        "title": compact(title, 240),
        "domain_hint": domain_hint,
        "sequence_length": int(sequence_length or 0),
        "summary_json": json.dumps(summary, ensure_ascii=True),
        "source_fields_json": json.dumps(source_fields, ensure_ascii=True),
    }


def collect_anchors(raw_root: Path, limit_per_family: int) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []

    uniprot_path = raw_root / "uniprot_function_text_train/default/train/0000.parquet"
    for idx, row in rows_from_parquet(
        uniprot_path,
        ["entry", "entry_name", "protein_name", "function", "sequence"],
        limit_per_family,
    ):
        text = f"{row.get('protein_name', '')} {row.get('function', '')}"
        ecs = sorted(set(EC_RE.findall(text)))
        anchors.append(
            make_anchor(
                "uniprot_function_text_train",
                uniprot_path,
                idx,
                row.get("entry") or "",
                row.get("protein_name") or row.get("entry_name") or "",
                "protein_function_enzyme_annotation",
                len(row.get("sequence") or ""),
                {
                    "entry_name": row.get("entry_name"),
                    "protein_name": compact(row.get("protein_name"), 500),
                    "function": compact(row.get("function"), 900),
                    "ec_numbers": ecs,
                    "suggested_axes": ["substrate", "reaction", "product", "domain", "design_condition"],
                },
                {"entry": row.get("entry"), "entry_name": row.get("entry_name")},
            )
        )

    uniref_dir = raw_root / "uniprot_uniref50_sequence_train/default/train"
    uniref_count = 0
    for path in sorted(uniref_dir.glob("*.parquet")):
        if uniref_count >= limit_per_family:
            break
        for idx, row in rows_from_parquet(
            path,
            ["id", "name", "common_taxon", "go_mf", "go_bp", "go_cc", "rep_protein_name", "sequence_length"],
            limit_per_family - uniref_count,
            predicate=lambda r: bool(r.get("go_mf") or r.get("go_bp") or r.get("go_cc")),
        ):
            anchors.append(
                make_anchor(
                    "uniprot_uniref50_sequence_train",
                    path,
                    idx,
                    row.get("id") or "",
                    row.get("name") or row.get("rep_protein_name") or "",
                    "uniref_go_cluster_annotation",
                    row.get("sequence_length") or 0,
                    {
                        "name": row.get("name"),
                        "common_taxon": row.get("common_taxon"),
                        "rep_protein_name": row.get("rep_protein_name"),
                        "go_mf": row.get("go_mf") or [],
                        "go_bp": row.get("go_bp") or [],
                        "go_cc": row.get("go_cc") or [],
                        "suggested_axes": ["go_mf", "go_bp", "go_cc", "sequence_scale", "taxon"],
                    },
                    {"id": row.get("id")},
                )
            )
            uniref_count += 1
            if uniref_count >= limit_per_family:
                break

    rfam_path = raw_root / "rfam_sequence_train/default/train/0000.parquet"
    for idx, row in rows_from_parquet(rfam_path, ["id", "sequence", "family", "clan", "description"], limit_per_family):
        anchors.append(
            make_anchor(
                "rfam_sequence_train",
                rfam_path,
                idx,
                row.get("id") or "",
                row.get("family") or "",
                "rna_family_annotation",
                len(row.get("sequence") or ""),
                {
                    "family": row.get("family"),
                    "clan": row.get("clan"),
                    "description": compact(row.get("description"), 500),
                    "suggested_axes": ["rna_family", "secondary_structure", "organism_context"],
                },
                {"id": row.get("id"), "family": row.get("family")},
            )
        )

    rnacentral_path = raw_root / "rnacentral_8192_sequence_train/default/train/0000.parquet"
    for idx, row in rows_from_parquet(rnacentral_path, ["upi", "sequence", "type", "description"], limit_per_family):
        anchors.append(
            make_anchor(
                "rnacentral_8192_sequence_train",
                rnacentral_path,
                idx,
                row.get("upi") or "",
                row.get("type") or "",
                "rna_type_annotation",
                len(row.get("sequence") or ""),
                {
                    "upi": row.get("upi"),
                    "type": row.get("type"),
                    "description": compact(row.get("description"), 500),
                    "suggested_axes": ["rna_type", "length", "regulatory_hypothesis", "design_scope"],
                },
                {"upi": row.get("upi"), "type": row.get("type")},
            )
        )

    dna_path = raw_root / "dna_coding_regions_train/default/train/0000.parquet"
    for idx, row in rows_from_parquet(dna_path, ["accession", "organism", "sequence", "exons", "proteins"], limit_per_family):
        genes = sorted({feature.get("gene") for feature in (row.get("exons") or []) if feature.get("gene")})
        protein_lengths = [len(feature.get("sequence") or "") for feature in (row.get("proteins") or [])]
        anchors.append(
            make_anchor(
                "dna_coding_regions_train",
                dna_path,
                idx,
                row.get("accession") or "",
                ",".join(genes) or row.get("organism") or "",
                "coding_region_feature_graph",
                len(row.get("sequence") or ""),
                {
                    "organism": row.get("organism"),
                    "genes": genes,
                    "exon_count": len(row.get("exons") or []),
                    "protein_lengths": protein_lengths,
                    "suggested_axes": ["gene", "exon", "protein_product", "safety_boundary"],
                },
                {"accession": row.get("accession"), "organism": row.get("organism")},
            )
        )

    pubchem_path = raw_root / "pubchem10m_selfies_train/default/train/0000.parquet"
    for idx, row in rows_from_parquet(pubchem_path, ["SELFIES"], limit_per_family):
        selfies = row.get("SELFIES") or ""
        anchors.append(
            make_anchor(
                "pubchem10m_selfies_train",
                pubchem_path,
                idx,
                f"pubchem_selfies_row_{idx}",
                "SELFIES molecular string",
                "medicinal_chemistry_graph_string",
                0,
                {
                    "selfies_prefix": compact(selfies, 700),
                    "token_count_proxy": selfies.count("["),
                    "suggested_axes": ["atom_token", "ring", "branch", "heteroatom", "activity_uncertainty"],
                },
                {"row_index": idx},
            )
        )

    return anchors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=str(RAW_ROOT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--prefix", default="source_anchors_seed")
    parser.add_argument("--limit-per-family", type=int, default=24)
    args = parser.parse_args()

    rows = collect_anchors(Path(args.raw_root), args.limit_per_family)
    manifest = write_outputs(rows, Path(args.output_dir), args.prefix)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
