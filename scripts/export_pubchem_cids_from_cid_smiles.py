#!/usr/bin/env python3
"""Export deterministic PubChem CID lists from a CID-SMILES gzip table.

The NatureLM PubChem mirror stores rows as ``CID<TAB>SMILES``.  This script
streams that file and writes unique CIDs for downstream PubChem3D SDF
coordinate curation.  It does not decode or generate molecules.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5_000_000)
    parser.add_argument("--stride", type=int, default=1, help="Take every nth source row before uniqueness filtering.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many source rows before stride filtering.")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0
    scanned = 0
    with gzip.open(args.input, "rt", encoding="utf-8", errors="replace") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            if scanned < args.offset:
                scanned += 1
                continue
            if args.stride > 1 and ((scanned - args.offset) % args.stride) != 0:
                scanned += 1
                continue
            scanned += 1
            cid = line.split("\t", 1)[0].strip()
            if not cid or cid in seen:
                continue
            seen.add(cid)
            dst.write(cid + "\n")
            written += 1
            if args.limit > 0 and written >= args.limit:
                break
    print(f"wrote_cids:{written} scanned_rows:{scanned} output:{args.output}")


if __name__ == "__main__":
    main()
