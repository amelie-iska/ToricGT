#!/usr/bin/env python3
"""Small periodic CAS audit for training sidecars.

The audit records backend availability and validates that at least the exact
closed-form certificate path works.  Optional Sage/Macaulay2 checks are exact
and are skipped or failed explicitly depending on `--require-cas`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from toricgt.cas_oracles import discover_all_backends


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/periodic_cas_audit")
    parser.add_argument("--num-vertices", type=int, default=6)
    parser.add_argument("--sage-normal-fan", action="store_true")
    parser.add_argument("--macaulay2-smoke", action="store_true")
    parser.add_argument("--require-cas", action="store_true")
    parser.add_argument("--python-bin", default="python")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_cmd = [
        args.python_bin,
        "scripts/build_toric_tropical_certificates.py",
        "--output-dir",
        str(output_dir / "certificates"),
        "--num-vertices",
        str(args.num_vertices),
    ]
    if args.sage_normal_fan:
        build_cmd.append("--sage-normal-fan")
    if args.macaulay2_smoke:
        build_cmd.append("--macaulay2-smoke")
    if args.require_cas:
        build_cmd.append("--require-cas")
    proc = subprocess.run(build_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    statuses = discover_all_backends()
    report: dict[str, Any] = {
        "schema": "toricgt.periodic_cas_audit.v1",
        "output_dir": str(output_dir),
        "build_command": build_cmd,
        "build_returncode": proc.returncode,
        "backend_status": {name: info.to_dict() for name, info in statuses.items()},
        "stdout_tail": proc.stdout[-4000:],
    }
    (output_dir / "cas_audit_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
