#!/usr/bin/env python3
"""Build exact toric/tropical certificates.

This script never emits Torch surrogate certificates.  It can always build the
closed-form cyclic Stanley-Reisner toy certificate.  SageMath and Macaulay2
certificates are emitted only when the corresponding CAS backend is available
and returns a parseable exact payload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from toricgt.cas_certificates import CertificateCache
from toricgt.cas_oracles import (
    CASExecutionError,
    CASUnavailableError,
    Macaulay2TropicalOracle,
    SageToricOracle,
    cyclic_stanley_reisner_closed_form_certificate,
    discover_all_backends,
    discover_toric_toolchain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/cas_certificates")
    parser.add_argument("--num-vertices", type=int, default=6)
    parser.add_argument("--write-backend-status", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sage-normal-fan", action="store_true", help="Build a small exact Sage normal-fan certificate if Sage is available.")
    parser.add_argument("--macaulay2-smoke", action="store_true", help="Build a small exact Macaulay2 smoke certificate if M2 is available.")
    parser.add_argument("--macaulay2-toric-ideal", action="store_true", help="Build a real exact Macaulay2 toric-ideal certificate by elimination.")
    parser.add_argument("--macaulay2-vector-bundle", action="store_true", help="Build a real exact Macaulay2 toric vector-bundle certificate.")
    parser.add_argument("--macaulay2-koszul-resolution", action="store_true", help="Build a real exact Macaulay2 Koszul/free-resolution certificate.")
    parser.add_argument("--all-exact-cas", action="store_true", help="Enable all current exact Sage/Macaulay2 certificate builders.")
    parser.add_argument("--require-cas", action="store_true", help="Fail if requested Sage/Macaulay2 exact checks are unavailable.")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = CertificateCache(output_dir / "cache")
    report: dict[str, Any] = {
        "schema": "toricgt.cas_certificate_build_report.v1",
        "output_dir": str(output_dir),
        "certificates": [],
        "backend_status": {},
        "toolchain_status": {},
        "errors": [],
    }

    if args.write_backend_status:
        statuses = discover_all_backends()
        report["backend_status"] = {name: info.to_dict() for name, info in statuses.items()}
        write_json(output_dir / "backend_status.json", report["backend_status"])
        report["toolchain_status"] = discover_toric_toolchain()
        write_json(output_dir / "toolchain_status.json", report["toolchain_status"])

    closed_form = cyclic_stanley_reisner_closed_form_certificate(args.num_vertices)
    path = cache.write(closed_form)
    report["certificates"].append(
        {
            "kind": closed_form.kind,
            "provenance": closed_form.provenance,
            "path": str(path),
            "certificate_hash": closed_form.certificate_hash,
        }
    )

    if args.all_exact_cas:
        args.sage_normal_fan = True
        args.macaulay2_smoke = True
        args.macaulay2_toric_ideal = True
        args.macaulay2_vector_bundle = True
        args.macaulay2_koszul_resolution = True

    if args.sage_normal_fan:
        oracle = SageToricOracle()
        if not oracle.info.available:
            message = oracle.info.error or "SageMath backend unavailable"
            if args.require_cas:
                raise CASUnavailableError(message)
            report["errors"].append({"backend": "sage", "error": message, "required": False})
        else:
            # A tiny square Newton polytope gives a deterministic normal fan.
            exponent_matrix = [[0, 0], [1, 0], [0, 1], [1, 1]]
            try:
                cert = oracle.normal_fan_certificate(exponent_matrix)
                path = cache.write(cert)
                report["certificates"].append(
                    {
                        "kind": cert.kind,
                        "provenance": cert.provenance,
                        "path": str(path),
                        "certificate_hash": cert.certificate_hash,
                    }
                )
            except (CASExecutionError, CASUnavailableError) as exc:
                if args.require_cas:
                    raise
                report["errors"].append({"backend": "sage", "error": str(exc), "required": False})

    if args.macaulay2_smoke:
        oracle = Macaulay2TropicalOracle()
        if not oracle.info.available:
            message = oracle.info.error or "Macaulay2 backend unavailable"
            if args.require_cas:
                raise CASUnavailableError(message)
            report["errors"].append({"backend": "macaulay2", "error": message, "required": False})
        else:
            try:
                cert = oracle.smoke_certificate()
                path = cache.write(cert)
                report["certificates"].append(
                    {
                        "kind": cert.kind,
                        "provenance": cert.provenance,
                        "path": str(path),
                        "certificate_hash": cert.certificate_hash,
                    }
                )
            except (CASExecutionError, CASUnavailableError) as exc:
                if args.require_cas:
                    raise
                report["errors"].append({"backend": "macaulay2", "error": str(exc), "required": False})

    if args.macaulay2_toric_ideal:
        oracle = Macaulay2TropicalOracle()
        if not oracle.info.available:
            message = oracle.info.error or "Macaulay2 backend unavailable"
            if args.require_cas:
                raise CASUnavailableError(message)
            report["errors"].append({"backend": "macaulay2", "error": message, "required": False, "certificate": "toric_ideal"})
        else:
            try:
                # Columns are semigroup exponents in N^2. This nontrivial
                # example gives ideal(x_2^2-x_1*x_3, x_0*x_2-x_3, x_0*x_1-x_2).
                cert = oracle.toric_ideal_certificate([[1, 0, 1, 2], [0, 1, 1, 1]])
                path = cache.write(cert)
                report["certificates"].append(
                    {
                        "kind": cert.kind,
                        "provenance": cert.provenance,
                        "path": str(path),
                        "certificate_hash": cert.certificate_hash,
                    }
                )
            except (CASExecutionError, CASUnavailableError) as exc:
                if args.require_cas:
                    raise
                report["errors"].append({"backend": "macaulay2", "error": str(exc), "required": False, "certificate": "toric_ideal"})

    if args.macaulay2_vector_bundle:
        oracle = Macaulay2TropicalOracle()
        if not oracle.info.available:
            message = oracle.info.error or "Macaulay2 backend unavailable"
            if args.require_cas:
                raise CASUnavailableError(message)
            report["errors"].append(
                {"backend": "macaulay2", "error": message, "required": False, "certificate": "toric_vector_bundle"}
            )
        else:
            try:
                cert = oracle.vector_bundle_smoke_certificate()
                path = cache.write(cert)
                report["certificates"].append(
                    {
                        "kind": cert.kind,
                        "provenance": cert.provenance,
                        "path": str(path),
                        "certificate_hash": cert.certificate_hash,
                    }
                )
            except (CASExecutionError, CASUnavailableError) as exc:
                if args.require_cas:
                    raise
                report["errors"].append(
                    {"backend": "macaulay2", "error": str(exc), "required": False, "certificate": "toric_vector_bundle"}
                )

    if args.macaulay2_koszul_resolution:
        oracle = Macaulay2TropicalOracle()
        if not oracle.info.available:
            message = oracle.info.error or "Macaulay2 backend unavailable"
            if args.require_cas:
                raise CASUnavailableError(message)
            report["errors"].append(
                {"backend": "macaulay2", "error": message, "required": False, "certificate": "koszul_resolution"}
            )
        else:
            try:
                cert = oracle.koszul_resolution_certificate()
                path = cache.write(cert)
                report["certificates"].append(
                    {
                        "kind": cert.kind,
                        "provenance": cert.provenance,
                        "path": str(path),
                        "certificate_hash": cert.certificate_hash,
                    }
                )
            except (CASExecutionError, CASUnavailableError) as exc:
                if args.require_cas:
                    raise
                report["errors"].append(
                    {"backend": "macaulay2", "error": str(exc), "required": False, "certificate": "koszul_resolution"}
                )

    write_json(output_dir / "build_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
