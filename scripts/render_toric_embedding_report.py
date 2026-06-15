#!/usr/bin/env python3
"""Render tropical-attention-to-toric-embedding visualization reports.

This script consumes exact embedding CAS sidecar records and writes a dark-mode
HTML report visualizing the finite toric sidecar: Newton polytopes, initial
degeneration chambers, normal fans, orbit strata, toric ideals, resolutions,
Miller-Sturmfels staircases, and optional Klyachko vector-bundle certificates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toricgt.toric_embedding_visualization import (  # noqa: E402
    collect_sidecar_records,
    render_toric_embedding_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar-record-json",
        action="append",
        default=[],
        help="Exact *_cas_sidecar.json record. May be repeated.",
    )
    parser.add_argument(
        "--sidecar-dir",
        default="",
        help="Directory containing records/*_cas_sidecar.json, usually embedding_cas_sidecar.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vector-bundle-certificate-json", default="")
    parser.add_argument(
        "--build-vector-bundle-certificate",
        action="store_true",
        help="Build an exact Macaulay2 ToricVectorBundles certificate and include Klyachko/sheaf panels.",
    )
    parser.add_argument("--macaulay2-timeout-seconds", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_sidecar_records(
        sidecar_records=[Path(path) for path in args.sidecar_record_json],
        sidecar_dir=Path(args.sidecar_dir) if args.sidecar_dir else None,
    )
    manifest = render_toric_embedding_report(
        sidecar_records=records,
        output_dir=Path(args.output_dir),
        vector_bundle_certificate=Path(args.vector_bundle_certificate_json)
        if args.vector_bundle_certificate_json
        else None,
        build_vector_bundle_certificate=bool(args.build_vector_bundle_certificate),
        macaulay2_timeout_seconds=int(args.macaulay2_timeout_seconds),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
