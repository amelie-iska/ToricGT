"""Optional SageMath/Macaulay2 oracle wrappers for exact certificates.

The wrappers in this module are intentionally strict.  If a CAS executable is
missing, methods return backend status or raise when an exact computation was
explicitly requested.  They do not replace missing CAS output with Torch
surrogates.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .cas_certificates import (
    CASBackendInfo,
    CertificateCache,
    ToricTropicalCertificate,
    exact_closed_form_certificate,
    stable_hash,
    validate_certificate_payload,
)
from .symbolic_multigraded_resolution import (
    cyclic_flag_face_rows,
    cyclic_stanley_reisner_betti_rows,
    cyclic_stanley_reisner_generator_masks,
    cyclic_stanley_reisner_resolution_certificate,
    cyclic_stanley_reisner_resolution_metrics,
    cyclic_taylor_dg_differential_rows,
    cyclic_taylor_fitting_entry_ideal_rows,
    cyclic_taylor_fitting_summary_rows,
    cyclic_taylor_multidegree_counts,
    cyclic_taylor_rank_rows,
    squarefree_exponent_vector,
    squarefree_monomial,
)


class CASUnavailableError(RuntimeError):
    """Raised when an exact CAS computation is requested without a backend."""


class CASExecutionError(RuntimeError):
    """Raised when a CAS backend fails or returns an invalid payload."""


def _run(command: list[str], *, cwd: Path | None = None, timeout_seconds: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )


def discover_backend(name: str) -> CASBackendInfo:
    """Discover a supported CAS backend on PATH."""

    normalized = name.lower()
    if normalized in {"sage", "sagemath"}:
        executable = shutil.which("sage")
        if not executable:
            return CASBackendInfo(
                backend="sage",
                executable=None,
                version=None,
                packages=("sage.schemes.toric", "sage.geometry.polyhedron"),
                available=False,
                provenance="cas_unavailable",
                error="sage executable not found on PATH",
            )
        try:
            proc = _run([executable, "--version"], timeout_seconds=30)
            version = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else "unknown"
            ok = proc.returncode == 0
        except Exception as exc:
            return CASBackendInfo(
                backend="sage",
                executable=executable,
                version=None,
                packages=("sage.schemes.toric", "sage.geometry.polyhedron"),
                available=False,
                provenance="cas_unavailable",
                error=f"{type(exc).__name__}: {exc}",
            )
        return CASBackendInfo(
            backend="sage",
            executable=executable,
            version=version,
            packages=("sage.schemes.toric", "sage.geometry.polyhedron"),
            available=ok,
            provenance="exact_cas/sage" if ok else "cas_unavailable",
            error="" if ok else proc.stdout[-1000:],
        )

    if normalized in {"m2", "macaulay2", "macaulay"}:
        executable = shutil.which("M2") or shutil.which("macaulay2")
        if not executable:
            return CASBackendInfo(
                backend="macaulay2",
                executable=None,
                version=None,
                packages=("Tropical", "TropicalToric", "NormalToricVarieties", "gfanInterface"),
                available=False,
                provenance="cas_unavailable",
                error="M2 executable not found on PATH",
            )
        try:
            proc = _run([executable, "--version"], timeout_seconds=30)
            version = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else "unknown"
            ok = proc.returncode == 0
        except Exception as exc:
            return CASBackendInfo(
                backend="macaulay2",
                executable=executable,
                version=None,
                packages=("Tropical", "TropicalToric", "NormalToricVarieties", "gfanInterface"),
                available=False,
                provenance="cas_unavailable",
                error=f"{type(exc).__name__}: {exc}",
            )
        return CASBackendInfo(
            backend="macaulay2",
            executable=executable,
            version=version,
            packages=("Tropical", "TropicalToric", "NormalToricVarieties", "gfanInterface"),
            available=ok,
            provenance="exact_cas/macaulay2" if ok else "cas_unavailable",
            error="" if ok else proc.stdout[-1000:],
        )

    raise ValueError(f"unsupported CAS backend: {name}")


def discover_all_backends() -> dict[str, CASBackendInfo]:
    return {
        "sage": discover_backend("sage"),
        "macaulay2": discover_backend("macaulay2"),
    }


def cyclic_stanley_reisner_closed_form_certificate(num_vertices: int) -> ToricTropicalCertificate:
    """Exact finite certificate for the cyclic Stanley-Reisner toy family."""

    n = max(1, int(num_vertices))
    base = cyclic_stanley_reisner_resolution_certificate(n)
    metrics = cyclic_stanley_reisner_resolution_metrics(n)
    nonfaces = cyclic_stanley_reisner_generator_masks(n)
    source = {
        "family": "cyclic_one_skeleton_flag_fan",
        "num_vertices": n,
        "input_kind": "closed_form_toy_certificate",
    }
    toric = {
        "fan_kind": "cyclic_one_skeleton_flag_fan",
        "face_rows": [
            {"dimension": dim, "mask": mask, "cardinality": cardinality}
            for dim, mask, cardinality in cyclic_flag_face_rows(n)
        ],
        "rays": [[float(math.cos(2.0 * math.pi * idx / n)), float(math.sin(2.0 * math.pi * idx / n))] for idx in range(n)],
        "maximal_cones": [[idx, (idx + 1) % n] for idx in range(n)],
    }
    tropical = {
        "balanced": True,
        "multiplicities": [1 for _ in range(n)],
        "active_face_ids": list(range(n)),
        "tropical_convention": "max",
    }
    commutative_algebra = {
        "field": "F_2",
        "ring": base["ring"],
        "stanley_reisner_ideal": base["stanley_reisner_ideal"],
        "minimal_nonface_generators": base["minimal_nonface_generators"],
        "minimal_nonface_exponent_vectors": [
            squarefree_exponent_vector(mask, n) for mask in nonfaces
        ],
        "minimal_nonface_monomials": [squarefree_monomial(mask, n) for mask in nonfaces],
        "betti_table_by_homological_degree_and_support_size": base[
            "betti_table_by_homological_degree_and_support_size"
        ],
        "hochster_betti_rows": base["hochster_betti_rows"],
        "taylor_multidegree_counts": base["taylor_multidegree_counts"],
        "taylor_rank_rows": [
            {
                "homological_degree": degree,
                "rank": rank,
                "outgoing_boundary_terms": boundary_terms,
            }
            for degree, rank, boundary_terms in cyclic_taylor_rank_rows(n)
        ],
        "dg_differential_entries": [
            {
                "homological_degree": degree,
                "source_lcm_mask": source_lcm,
                "target_lcm_mask": target_lcm,
                "quotient_mask": quotient,
                "multiplicity": multiplicity,
            }
            for degree, source_lcm, target_lcm, quotient, multiplicity in cyclic_taylor_dg_differential_rows(n)
        ],
        "fitting_entry_ideal_generators": [
            {
                "homological_degree": degree,
                "quotient_mask": quotient,
                "multiplicity": multiplicity,
            }
            for degree, quotient, multiplicity in cyclic_taylor_fitting_entry_ideal_rows(n)
        ],
        "fitting_determinantal_summary": list(cyclic_taylor_fitting_summary_rows(n)),
        "dg_algebra": base["dg_algebra"],
    }
    diagnostics = {
        "symbolic_resolution_num_vertices": metrics.num_vertices,
        "symbolic_resolution_nonface_generator_count": metrics.nonface_generator_count,
        "symbolic_resolution_minimal_total_betti": metrics.minimal_total_betti,
        "symbolic_resolution_projective_dimension": metrics.projective_dimension,
        "symbolic_resolution_regularity": metrics.regularity,
        "symbolic_resolution_betti0": metrics.betti0,
        "symbolic_resolution_betti1": metrics.betti1,
        "symbolic_resolution_betti2": metrics.betti2,
        "exactness": bool(base["dg_algebra"]["d_squared_zero"] and base["dg_algebra"]["leibniz_rule"]),
    }
    return exact_closed_form_certificate(
        kind="cyclic_stanley_reisner_exact_certificate",
        source=source,
        toric=toric,
        tropical=tropical,
        commutative_algebra=commutative_algebra,
        diagnostics=diagnostics,
    )


class SageToricOracle:
    """Strict Sage wrapper for rational toric fan/polytope certificates."""

    def __init__(self, executable: str | None = None) -> None:
        self.info = discover_backend("sage")
        if executable is not None:
            self.info = CASBackendInfo(
                backend="sage",
                executable=executable,
                version=self.info.version,
                packages=self.info.packages,
                available=bool(executable),
                provenance="exact_cas/sage" if executable else "cas_unavailable",
                error="" if executable else "sage executable not provided",
            )

    def require_available(self) -> None:
        if not self.info.available or not self.info.executable:
            raise CASUnavailableError(self.info.error or "SageMath backend unavailable")

    @staticmethod
    def script_for_normal_fan(exponent_matrix: list[list[int | float]]) -> str:
        payload = json.dumps({"exponent_matrix": exponent_matrix}, sort_keys=True)
        return f"""
import json
from sage.all import *

payload = json.loads({payload!r})
points = [vector(QQ, row) for row in payload["exponent_matrix"]]
poly = Polyhedron(vertices=points, base_ring=QQ)
fan = poly.normal_fan()
out = {{
    "kind": "sage_normal_fan_certificate",
    "dimension": int(poly.dimension()),
    "num_vertices": int(len(poly.vertices())),
    "num_rays": int(len(fan.rays())),
    "fan_rays": [[int(x) if x in ZZ else str(x) for x in ray] for ray in fan.rays()],
    "maximal_cones": [list(map(int, cone.ambient_ray_indices())) for cone in fan],
    "face_vector": list(map(int, poly.f_vector())),
}}
print("TORICGT_JSON_BEGIN")
print(json.dumps(out, sort_keys=True))
print("TORICGT_JSON_END")
"""

    def normal_fan_certificate(
        self,
        exponent_matrix: list[list[int | float]],
        *,
        timeout_seconds: int = 120,
    ) -> ToricTropicalCertificate:
        self.require_available()
        assert self.info.executable is not None
        with tempfile.TemporaryDirectory(prefix="toricgt_sage_") as tmp:
            path = Path(tmp) / "normal_fan.sage"
            path.write_text(self.script_for_normal_fan(exponent_matrix), encoding="utf-8")
            proc = _run([self.info.executable, str(path)], cwd=Path(tmp), timeout_seconds=timeout_seconds)
        if proc.returncode != 0:
            raise CASExecutionError(proc.stdout[-4000:])
        payload = _extract_json_between_markers(proc.stdout)
        source = {"input_kind": "exponent_matrix", "input_hash": stable_hash(exponent_matrix)}
        cert = ToricTropicalCertificate(
            kind="sage_normal_fan_certificate",
            input_hash=stable_hash({"sage_normal_fan": exponent_matrix}),
            provenance="exact_cas/sage",
            source=source,
            cas=self.info.to_dict(),
            toric=payload,
            tropical={},
            commutative_algebra={},
            diagnostics={"raw_stdout_tail": proc.stdout[-1000:]},
        )
        _raise_if_invalid(cert)
        return cert


class Macaulay2TropicalOracle:
    """Strict Macaulay2 wrapper for toric ideal / resolution certificates."""

    def __init__(self, executable: str | None = None) -> None:
        self.info = discover_backend("macaulay2")
        if executable is not None:
            self.info = CASBackendInfo(
                backend="macaulay2",
                executable=executable,
                version=self.info.version,
                packages=self.info.packages,
                available=bool(executable),
                provenance="exact_cas/macaulay2" if executable else "cas_unavailable",
                error="" if executable else "M2 executable not provided",
            )

    def require_available(self) -> None:
        if not self.info.available or not self.info.executable:
            raise CASUnavailableError(self.info.error or "Macaulay2 backend unavailable")

    @staticmethod
    def script_for_backend_smoke() -> str:
        return """
needsPackage "JSON"
out = hashTable {
  "kind" => "macaulay2_smoke_certificate",
  "ring_ok" => true,
  "version" => version
}
print "TORICGT_JSON_BEGIN"
print toJSON out
print "TORICGT_JSON_END"
"""

    def smoke_certificate(self, *, timeout_seconds: int = 60) -> ToricTropicalCertificate:
        self.require_available()
        assert self.info.executable is not None
        with tempfile.TemporaryDirectory(prefix="toricgt_m2_") as tmp:
            path = Path(tmp) / "smoke.m2"
            path.write_text(self.script_for_backend_smoke(), encoding="utf-8")
            proc = _run([self.info.executable, "--script", str(path)], cwd=Path(tmp), timeout_seconds=timeout_seconds)
        if proc.returncode != 0:
            raise CASExecutionError(proc.stdout[-4000:])
        payload = _extract_json_between_markers(proc.stdout)
        cert = ToricTropicalCertificate(
            kind="macaulay2_smoke_certificate",
            input_hash=stable_hash({"macaulay2_smoke": self.info.version}),
            provenance="exact_cas/macaulay2",
            source={"input_kind": "backend_smoke"},
            cas=self.info.to_dict(),
            toric={},
            tropical={},
            commutative_algebra=payload,
            diagnostics={"raw_stdout_tail": proc.stdout[-1000:]},
        )
        _raise_if_invalid(cert)
        return cert


def _extract_json_between_markers(text: str) -> dict[str, Any]:
    start = text.find("TORICGT_JSON_BEGIN")
    end = text.find("TORICGT_JSON_END")
    if start < 0 or end < 0 or end <= start:
        raise CASExecutionError("CAS output did not contain TORICGT JSON markers")
    body = text[start + len("TORICGT_JSON_BEGIN") : end].strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise CASExecutionError(f"failed to parse CAS JSON payload: {exc}: {body[:1000]}") from exc


def _raise_if_invalid(certificate: ToricTropicalCertificate) -> None:
    errors = validate_certificate_payload(certificate.with_hash())
    if errors:
        raise CASExecutionError("; ".join(errors))


def write_closed_form_cyclic_certificate(
    num_vertices: int,
    output_dir: str | Path,
) -> Path:
    cache = CertificateCache(output_dir)
    return cache.write(cyclic_stanley_reisner_closed_form_certificate(num_vertices))
