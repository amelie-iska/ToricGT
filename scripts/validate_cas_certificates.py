#!/usr/bin/env python3
"""Validate exact CAS/closed-form certificate files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from toricgt.cas_certificates import validate_certificate_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Certificate JSON files or directories containing JSON certificates.")
    parser.add_argument("--allow-unavailable", action="store_true", help="Allow cas_unavailable status payloads.")
    return parser.parse_args()


def expand_paths(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(sorted(path.glob("**/*.json")))
        else:
            out.append(path)
    return out


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    failed = False
    for path in expand_paths(args.paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"path": str(path), "ok": False, "errors": [f"{type(exc).__name__}: {exc}"]})
            failed = True
            continue
        if "backend" in payload and "available" in payload and "kind" not in payload:
            ok = bool(args.allow_unavailable or payload.get("available"))
            errors = [] if ok else ["backend status is unavailable"]
        else:
            errors = validate_certificate_payload(payload)
            ok = not errors
        rows.append(
            {
                "path": str(path),
                "kind": payload.get("kind"),
                "provenance": payload.get("provenance"),
                "ok": ok,
                "errors": errors,
            }
        )
        failed = failed or not ok
    report = {
        "schema": "toricgt.cas_certificate_validation_report.v1",
        "checked": len(rows),
        "failed": sum(1 for row in rows if not row["ok"]),
        "rows": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
