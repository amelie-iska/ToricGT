#!/usr/bin/env python3
"""Extract PubChem CIDs from SELFIES shards by resolving decoded SMILES.

This is a real PubChem lookup path for PubChem10M SELFIES-style shards that do
not carry CIDs.  It does not generate coordinates.  The resulting CID file is
intended for `build_pubchem3d_structure_fot_dataset.py`, which downloads real
PubChem3D SDF conformers and rejects flat/non-coordinate records.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import selfies as sf


PUBCHEM_SMILES_TO_CIDS = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/cids/TXT"


def iter_selfies(patterns: list[str], *, max_rows: int) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path_str in sorted(glob.glob(pattern)):
            pf = pq.ParquetFile(path_str)
            for batch in pf.iter_batches(batch_size=1024, columns=["SELFIES"]):
                table = pa.Table.from_batches([batch])
                for row in table.to_pylist():
                    value = str(row.get("SELFIES") or "").strip()
                    if not value or value in seen:
                        continue
                    seen.add(value)
                    values.append(value)
                    if max_rows > 0 and len(values) >= max_rows:
                        return values
    return values


def cid_for_smiles(smiles: str, *, timeout: int) -> str | None:
    data = urllib.parse.urlencode({"smiles": smiles}).encode("utf-8")
    request = urllib.request.Request(
        PUBCHEM_SMILES_TO_CIDS,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "ToricGT-PubChemCID-resolver/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.isdigit():
            return line
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="SELFIES parquet glob; may repeat.")
    parser.add_argument("--output-cids", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--max-cids", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.12)
    args = parser.parse_args()

    selfies_values = iter_selfies(args.input, max_rows=args.max_rows)
    args.output_cids.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.output_report or args.output_cids.with_suffix(".report.json")

    cids: list[str] = []
    seen_cids: set[str] = set()
    decoded = 0
    failed_decode = 0
    failed_lookup = 0
    for value in selfies_values:
        if len(cids) >= args.max_cids:
            break
        try:
            smiles = sf.decoder(value)
            decoded += 1
        except Exception:
            failed_decode += 1
            continue
        cid = cid_for_smiles(smiles, timeout=args.timeout)
        if cid is None:
            failed_lookup += 1
        elif cid not in seen_cids:
            seen_cids.add(cid)
            cids.append(cid)
            print(f"cid {len(cids):05d}/{args.max_cids}: {cid}", flush=True)
        if args.sleep > 0:
            time.sleep(args.sleep)

    args.output_cids.write_text("\n".join(cids) + ("\n" if cids else ""), encoding="utf-8")
    report = {
        "schema": "toricgt.pubchem_cid_from_selfies.v1",
        "inputs": args.input,
        "selfies_scanned": len(selfies_values),
        "decoded": decoded,
        "failed_decode": failed_decode,
        "failed_lookup": failed_lookup,
        "unique_cids": len(cids),
        "cid_file": str(args.output_cids),
        "policy": "CIDs are resolved from decoded SELFIES through PubChem PUG; no coordinates are generated here.",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
