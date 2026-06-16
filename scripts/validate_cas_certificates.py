#!/usr/bin/env python3
"""Validate exact CAS/closed-form certificate files."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from toricgt.cas_certificates import validate_certificate_payload
from toricgt.cas_oracles import discover_all_backends, discover_toric_toolchain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Certificate JSON files or directories containing JSON certificates.")
    parser.add_argument("--allow-unavailable", action="store_true", help="Allow cas_unavailable status payloads.")
    parser.add_argument("--preflight", action="store_true", help="Emit CAS/toolchain availability report.")
    parser.add_argument("--require-sage", action="store_true", help="Fail preflight if SageMath is unavailable.")
    parser.add_argument("--require-macaulay2", action="store_true", help="Fail preflight if Macaulay2 is unavailable.")
    parser.add_argument("--require-gudhi", action="store_true", help="Fail preflight if Python package gudhi is unavailable.")
    parser.add_argument(
        "--require-m2-package",
        action="append",
        default=[],
        help="Macaulay2 package required for preflight success. May be repeated.",
    )
    parser.add_argument(
        "--require-tool",
        action="append",
        default=[],
        help="External toric/tropical tool group required for preflight success, e.g. gfan or normaliz.",
    )
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


def gudhi_status() -> dict[str, Any]:
    try:
        module = importlib.import_module("gudhi")
    except Exception as exc:
        return {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "version": getattr(module, "__version__", "unknown"), "error": ""}


def preflight_report(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    backends = {name: info.to_dict() for name, info in discover_all_backends().items()}
    tools = discover_toric_toolchain()
    gudhi = gudhi_status()
    errors: list[str] = []
    if args.require_sage and not backends.get("sage", {}).get("available"):
        errors.append("SageMath is required but unavailable")
    if args.require_macaulay2 and not backends.get("macaulay2", {}).get("available"):
        errors.append("Macaulay2 is required but unavailable")
    if args.require_gudhi and not gudhi.get("available"):
        errors.append("GUDHI is required but unavailable")
    m2_packages = backends.get("macaulay2", {}).get("package_status", {}) or {}
    for package in args.require_m2_package:
        if not bool(m2_packages.get(str(package), False)):
            errors.append(f"Macaulay2 package {package!r} is required but unavailable")
    for tool in args.require_tool:
        if not bool(tools.get(str(tool), {}).get("available", False)):
            errors.append(f"tool group {tool!r} is required but unavailable")
    report = {
        "schema": "toricgt.cas_toolchain_preflight_report.v1",
        "ok": not errors,
        "errors": errors,
        "backends": backends,
        "python_packages": {"gudhi": gudhi},
        "tools": tools,
        "required": {
            "sage": bool(args.require_sage),
            "macaulay2": bool(args.require_macaulay2),
            "gudhi": bool(args.require_gudhi),
            "macaulay2_packages": list(args.require_m2_package),
            "tools": list(args.require_tool),
        },
    }
    return report, bool(errors)


def main() -> None:
    args = parse_args()
    if args.preflight:
        report, failed = preflight_report(args)
        print(json.dumps(report, indent=2, sort_keys=True))
        if failed:
            raise SystemExit(1)
        if not args.paths:
            return
    if not args.paths:
        raise SystemExit("no certificate paths supplied; use --preflight for toolchain discovery")
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
