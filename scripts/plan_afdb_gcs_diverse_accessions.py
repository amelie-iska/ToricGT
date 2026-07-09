#!/usr/bin/env python3
"""Plan a capped, diversity-biased AFDB GCS coordinate corpus.

The planner does not mirror AlphaFold DB.  It selects a bounded set of UniProt
accessions from local UniProt/UniRef Parquet sources, preserving sequence and
function metadata needed by the downstream graph/FoT structure records.

Selection goals:
- cap the AFDB GCS pull to the requested target size;
- retain broad sequence length, taxonomy, and functional-label diversity;
- upweight enzymes;
- among enzyme-positive candidates, target 50% high-catalytic/kinetic evidence,
  25% medium evidence, and 25% low/weak evidence where available;
- skip records already present in existing coordinate Parquet outputs.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


EC_RE = re.compile(r"\b(?:EC\s*)?([1-7])\.(?:[0-9-]+)\.(?:[0-9-]+)\.(?:[0-9-]+)\b", re.I)
KINETIC_RE = re.compile(
    r"\b(kcat|k_cat|turnover|km|k_m|vmax|v_max|specific activity|catalytic efficiency|kcat\s*/\s*km)\b",
    re.I,
)
CATALYTIC_RE = re.compile(
    r"\b(cataly[sz]e|catalytic|enzyme|hydrolase|transferase|oxidoreductase|lyase|isomerase|ligase|translocase|"
    r"kinase|phosphatase|protease|peptidase|polymerase|synthase|synthetase|dehydrogenase|reductase|oxidase|"
    r"esterase|lipase|nuclease|glycosidase|glucosidase|methyltransferase|acyltransferase|aminotransferase)\b",
    re.I,
)
HIGH_ACTIVITY_RE = re.compile(
    r"\b(high(?:ly)? active|fast|efficient|high catalytic|high activity|rapid turnover|large kcat|low km|"
    r"high specific activity|high catalytic efficiency)\b",
    re.I,
)
LOW_ACTIVITY_RE = re.compile(
    r"\b(low activity|weak activity|poor activity|slow turnover|inactive|limited activity|trace activity|"
    r"low catalytic|low specific activity)\b",
    re.I,
)
UNIPROT_RE = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9](?:-[0-9]+)?|A0A[A-Z0-9]{7}(?:-[0-9]+)?)$")


def stable_hash(value: Any, n: int = 16) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=True) if not isinstance(value, str) else value
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=16).hexdigest()[:n]


def safe_text(value: Any, limit: int) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, list):
        text = ", ".join(str(item) for item in value)
    else:
        text = str(value)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def normalize_accession(value: Any) -> str:
    text = safe_text(value, 96)
    if not text:
        return ""
    text = text.split()[0]
    if "|" in text:
        parts = [part for part in text.split("|") if part]
        if len(parts) >= 2 and parts[0].lower() in {"sp", "tr", "uniprotkb"}:
            text = parts[1]
        else:
            text = parts[0]
    if "_" in text:
        left = text.split("_", 1)[0]
        if UNIPROT_RE.match(left):
            text = left
    text = re.sub(r"[^A-Za-z0-9-]", "", text).upper()
    return text if UNIPROT_RE.match(text) else ""


def row_accessions(row: dict[str, Any], *, max_accessions: int) -> list[str]:
    raw_values: list[Any] = []
    for key in ("entry", "accession", "uniprot_accession", "rep_member_id", "seed_id"):
        if row.get(key) not in (None, ""):
            raw_values.append(row.get(key))
    rep_accessions = row.get("rep_accessions")
    if isinstance(rep_accessions, list):
        raw_values.extend(rep_accessions)
    elif rep_accessions not in (None, ""):
        raw_values.append(rep_accessions)
    member_ids = row.get("member_ids")
    if isinstance(member_ids, list):
        raw_values.extend(member_ids[: max(0, int(max_accessions) * 2)])
    seen: set[str] = set()
    out: list[str] = []
    for raw in raw_values:
        accession = normalize_accession(raw)
        if accession and accession not in seen:
            seen.add(accession)
            out.append(accession)
            if len(out) >= max(1, int(max_accessions)):
                break
    return out


def row_sequence(row: dict[str, Any], max_chars: int) -> str:
    seq = safe_text(row.get("sequence"), max_chars)
    seq = re.sub(r"[^A-Za-z*]", "", seq).upper()
    if len(seq) < 16:
        return ""
    return seq[:max_chars]


def row_function(row: dict[str, Any], max_chars: int) -> str:
    parts: list[str] = []
    for key in (
        "function",
        "protein_name",
        "rep_protein_name",
        "name",
        "go_mf",
        "go_bp",
        "go_cc",
        "common_taxon",
        "rep_organism",
    ):
        value = row.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            parts.append(f"{key}: " + ", ".join(str(item) for item in value[:48]))
        else:
            parts.append(f"{key}: {value}")
    return safe_text("; ".join(parts), max_chars)


def row_entry_name(row: dict[str, Any], accession: str) -> str:
    return safe_text(row.get("entry_name") or row.get("id") or row.get("seed_id") or accession, 256)


def row_protein_name(row: dict[str, Any]) -> str:
    return safe_text(row.get("protein_name") or row.get("rep_protein_name") or row.get("name") or "protein", 1024)


def enzyme_profile(row: dict[str, Any], function_text: str, protein_name: str) -> dict[str, Any]:
    text = " ".join(
        [
            function_text,
            protein_name,
            safe_text(row.get("go_mf"), 1200),
            safe_text(row.get("go_bp"), 1200),
            safe_text(row.get("go_cc"), 1200),
        ]
    )
    ec_matches = EC_RE.findall(text)
    kinetic_terms = KINETIC_RE.findall(text)
    catalytic_terms = CATALYTIC_RE.findall(text)
    high_terms = HIGH_ACTIVITY_RE.findall(text)
    low_terms = LOW_ACTIVITY_RE.findall(text)
    go_catalytic = "GO:0003824" in text
    evidence_score = 0.0
    evidence_score += 3.0 * len(ec_matches)
    evidence_score += 2.5 * len(kinetic_terms)
    evidence_score += 1.0 * len(catalytic_terms)
    evidence_score += 3.0 * len(high_terms)
    evidence_score -= 2.0 * len(low_terms)
    evidence_score += 2.0 if go_catalytic else 0.0
    enzyme = evidence_score > 0.0
    if not enzyme:
        tier = "non_enzyme"
    elif len(high_terms) > 0 or len(kinetic_terms) >= 2 or evidence_score >= 8.0:
        tier = "enzyme_high"
    elif len(ec_matches) > 0 or len(kinetic_terms) == 1 or evidence_score >= 3.0:
        tier = "enzyme_mid"
    else:
        tier = "enzyme_low"
    return {
        "is_enzyme": enzyme,
        "enzyme_tier": tier,
        "enzyme_evidence_score": round(float(evidence_score), 4),
        "ec_evidence_count": len(ec_matches),
        "kinetic_evidence_count": len(kinetic_terms),
        "catalytic_keyword_count": len(catalytic_terms),
        "high_activity_evidence_count": len(high_terms),
        "low_activity_evidence_count": len(low_terms),
        "go_catalytic_activity": go_catalytic,
    }


def length_bucket(seq: str) -> str:
    length = max(1, len(seq))
    return f"len2^{min(16, max(4, int(math.log2(length))))}"


def diversity_bucket(row: dict[str, Any], seq: str, function_text: str, tier: str, bucket_mod: int) -> str:
    taxon = safe_text(row.get("common_taxon_id") or row.get("rep_organism_tax_id") or row.get("rep_organism") or "unknown", 64)
    function_basis = "|".join(EC_RE.findall(function_text)[:4]) or "|".join(re.findall(r"GO:[0-9]{7}", function_text)[:8])
    if not function_basis:
        function_basis = stable_hash(function_text[:512] or row_protein_name(row), 8)
    raw = stable_hash([taxon, length_bucket(seq), function_basis, tier], 12)
    bucket = int(raw, 16) % max(1, int(bucket_mod))
    return f"{tier}:{bucket:05d}:{length_bucket(seq)}"


def source_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for raw in sorted(glob.glob(pattern)):
            path = Path(raw)
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(path)
    if not paths:
        raise FileNotFoundError(f"no input parquet files matched: {patterns}")
    return paths


def parquet_columns(path: Path) -> list[str] | None:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    wanted = {
        "entry",
        "entry_name",
        "accession",
        "uniprot_accession",
        "protein_name",
        "sequence",
        "function",
        "id",
        "name",
        "seed_id",
        "rep_member_id",
        "rep_protein_name",
        "rep_accessions",
        "rep_organism",
        "rep_organism_tax_id",
        "member_ids",
        "member_count",
        "common_taxon",
        "common_taxon_id",
        "go_mf",
        "go_bp",
        "go_cc",
        "sequence_length",
        "sequence_crc64",
        "sequence_xxh128",
    }
    cols = [name for name in sorted(wanted) if name in names]
    return cols or None


def load_existing_accessions(patterns: list[str]) -> set[str]:
    out: set[str] = set()
    for pattern in patterns:
        for raw in sorted(glob.glob(pattern)):
            path = Path(raw)
            try:
                names = set(pq.ParquetFile(path).schema_arrow.names)
                col = "uniprot_accession" if "uniprot_accession" in names else "entry_id" if "entry_id" in names else ""
                if not col:
                    continue
                table = pq.read_table(path, columns=[col])
                out.update(str(v).upper() for v in table[col].to_pylist() if v)
            except Exception:
                continue
    return out


def writer_handles(out_dir: Path, workers: int) -> list[Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [(out_dir / f"worker_{idx:02d}.jsonl").open("w", encoding="utf-8") for idx in range(workers)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Input Parquet glob. May repeat.")
    parser.add_argument("--existing-parquet-glob", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target-records", type=int, default=5_000_000)
    parser.add_argument("--enzyme-target-fraction", type=float, default=0.40)
    parser.add_argument("--enzyme-high-fraction", type=float, default=0.50)
    parser.add_argument("--enzyme-mid-fraction", type=float, default=0.25)
    parser.add_argument("--enzyme-low-fraction", type=float, default=0.25)
    parser.add_argument("--max-scan-rows", type=int, default=0)
    parser.add_argument("--max-accessions-per-row", type=int, default=1)
    parser.add_argument("--max-sequence-chars", type=int, default=8192)
    parser.add_argument("--max-function-chars", type=int, default=2400)
    parser.add_argument("--bucket-mod", type=int, default=8192)
    parser.add_argument("--max-per-diversity-bucket", type=int, default=1024)
    parser.add_argument("--prefix", default="toricblm_afdb_gcs_diverse_plan")
    args = parser.parse_args()

    inputs = source_paths(args.input)
    existing = load_existing_accessions(args.existing_parquet_glob)
    plan_dir = args.output_dir / "worker_plans"
    manifest_dir = args.output_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)
    handles = writer_handles(plan_dir, args.workers)

    target = int(args.target_records)
    enzyme_target = int(round(target * max(0.0, min(1.0, args.enzyme_target_fraction))))
    quotas = {
        "enzyme_high": int(round(enzyme_target * args.enzyme_high_fraction)),
        "enzyme_mid": int(round(enzyme_target * args.enzyme_mid_fraction)),
        "enzyme_low": int(round(enzyme_target * args.enzyme_low_fraction)),
    }
    quotas["non_enzyme"] = max(0, target - sum(quotas.values()))
    counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    seen_accessions = set(existing)
    scanned = 0
    accepted = 0
    try:
        for path in inputs:
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
                for row in pa.Table.from_batches([batch]).to_pylist():
                    if args.max_scan_rows > 0 and scanned >= args.max_scan_rows:
                        break
                    if accepted >= target:
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
                    tier = str(profile["enzyme_tier"])
                    quota = quotas.get(tier, 0)
                    if counts[tier] >= quota:
                        skipped[f"quota_full_{tier}"] += 1
                        continue
                    bucket = diversity_bucket(row, seq, function_text, tier, args.bucket_mod)
                    if bucket_counts[bucket] >= int(args.max_per_diversity_bucket):
                        skipped[f"bucket_full_{tier}"] += 1
                        continue
                    for accession in accessions:
                        if accession in seen_accessions:
                            skipped["already_seen_accession"] += 1
                            continue
                        seen_accessions.add(accession)
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
                                "tier": tier,
                                "diversity_bucket": bucket,
                                "selection_rank": accepted,
                                "enzyme_profile": profile,
                                "sequence_length": len(seq),
                                "length_bucket": length_bucket(seq),
                                "taxon": safe_text(row.get("common_taxon") or row.get("rep_organism"), 160),
                                "taxon_id": safe_text(row.get("common_taxon_id") or row.get("rep_organism_tax_id"), 64),
                                "member_count": int(row.get("member_count") or 0) if str(row.get("member_count") or "").isdigit() else 0,
                            },
                        }
                        handles[accepted % int(args.workers)].write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
                        accepted += 1
                        counts[tier] += 1
                        bucket_counts[bucket] += 1
                        source_counts[str(path)] += 1
                        if accepted % 10000 == 0:
                            print(
                                f"accepted={accepted} scanned={scanned} "
                                f"high={counts['enzyme_high']} mid={counts['enzyme_mid']} low={counts['enzyme_low']} non={counts['non_enzyme']}",
                                flush=True,
                            )
                        break
                if accepted >= target or (args.max_scan_rows > 0 and scanned >= args.max_scan_rows):
                    break
            if accepted >= target or (args.max_scan_rows > 0 and scanned >= args.max_scan_rows):
                break
    finally:
        for handle in handles:
            handle.close()

    manifest = {
        "schema": "toricblm.afdb_gcs_diverse_accession_plan_manifest.v1",
        "target_records": target,
        "accepted_records": accepted,
        "scanned_rows": scanned,
        "quotas": quotas,
        "counts": dict(counts),
        "skipped": dict(skipped),
        "existing_accessions_loaded": len(existing),
        "worker_plan_dir": str(plan_dir),
        "worker_plan_files": [str(plan_dir / f"worker_{idx:02d}.jsonl") for idx in range(args.workers)],
        "source_counts": dict(source_counts),
        "policy": (
            "Capped AFDB GCS selection with enzyme upweighting. Enzyme-positive pool targets "
            "50% high catalytic/kinetic evidence, 25% medium evidence, and 25% low/weak evidence; "
            "non-enzyme records remain represented for broad protein coverage."
        ),
    }
    manifest_path = args.output_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
