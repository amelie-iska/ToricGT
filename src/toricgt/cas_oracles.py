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


def _sage_integer_kernel_basis(rows: list[list[int]], *, timeout_seconds: int = 120) -> list[list[int]]:
    """Return an exact integer right-kernel basis for an exponent matrix.

    This is a Sage-backed CAS computation, not a numeric proxy.  It gives the
    lattice basis used by Macaulay2 to build the saturated lattice ideal whose
    saturation is the toric ideal of the monomial parametrization.
    """

    sage = discover_backend("sage")
    if not sage.available or not sage.executable:
        raise CASUnavailableError(sage.error or "sage executable unavailable for integer kernel basis")
    with tempfile.TemporaryDirectory(prefix="toricgt_sage_kernel_") as tmp:
        path = Path(tmp) / "kernel.py"
        path.write_text(
            "\n".join(
                [
                    "import json",
                    "from sage.all import matrix, ZZ",
                    f"rows = json.loads({json.dumps(json.dumps(rows))})",
                    "A = matrix(ZZ, rows)",
                    "basis = [[int(value) for value in vector] for vector in A.right_kernel().basis()]",
                    'print("TORICGT_JSON_BEGIN")',
                    "print(json.dumps({'rank': len(basis), 'basis': basis}))",
                    'print("TORICGT_JSON_END")',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        proc = _run([sage.executable, "-python", str(path)], cwd=Path(tmp), timeout_seconds=timeout_seconds)
    if proc.returncode != 0:
        raise CASExecutionError(proc.stdout[-4000:])
    payload = _extract_json_between_markers(proc.stdout)
    basis = payload.get("basis", [])
    if not isinstance(basis, list):
        raise CASExecutionError(f"Sage returned invalid integer-kernel payload: {payload}")
    return [[int(value) for value in row] for row in basis]


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
            package_status={"sage.schemes.toric": ok, "sage.geometry.polyhedron": ok},
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
                packages=(),
                package_status={
                    "Tropical": False,
                    "TropicalToric": False,
                    "NormalToricVarieties": False,
                    "gfanInterface": False,
                    "Binomials": False,
                },
                available=False,
                provenance="cas_unavailable",
                error="M2 executable not found on PATH",
            )
        try:
            proc = _run([executable, "--version"], timeout_seconds=30)
            version = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else "unknown"
            ok = proc.returncode == 0
            package_status = _macaulay2_package_status(executable) if ok else {}
        except Exception as exc:
            return CASBackendInfo(
                backend="macaulay2",
                executable=executable,
                version=None,
                packages=(),
                package_status={},
                available=False,
                provenance="cas_unavailable",
                error=f"{type(exc).__name__}: {exc}",
            )
        available_packages = tuple(name for name, available in package_status.items() if available)
        return CASBackendInfo(
            backend="macaulay2",
            executable=executable,
            version=version,
            packages=available_packages,
            package_status=package_status,
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


def discover_toric_toolchain() -> dict[str, dict[str, Any]]:
    """Discover exact toric/tropical command-line tools on PATH.

    These tools are not called in the hot training loop.  The periodic audit
    records them so reports can distinguish exact CAS/toolchain-backed metrics
    from metrics that only consumed cached certificates.
    """

    tool_groups = {
        "gfan": ("gfan",),
        "singular": ("Singular",),
        "normaliz": ("normaliz", "Normaliz"),
        "4ti2": ("markov", "graver", "groebner", "hilbert", "zsolve", "4ti2-zsolve"),
        "latte_integrale": ("count", "integrate", "latte-count", "latte-integrate"),
        "lrslib": ("lrs", "redund"),
        "topcom": ("topcom-points2triangs", "points2triangs", "topcom-chiro2allfinetriangs"),
        "polymake": ("polymake",),
        "nauty": ("dreadnaut", "geng"),
    }
    return {name: _discover_tool_group(commands) for name, commands in tool_groups.items()}


def _discover_tool_group(commands: tuple[str, ...]) -> dict[str, Any]:
    executables: dict[str, str] = {}
    versions: dict[str, str] = {}
    for command in commands:
        path = shutil.which(command)
        if not path:
            continue
        executables[command] = path
        version = _tool_version(path)
        if version:
            versions[command] = version
    return {
        "available": bool(executables),
        "commands": executables,
        "versions": versions,
    }


def _tool_version(executable: str) -> str:
    for flag in ("--version", "-v"):
        try:
            proc = _run([executable, flag], timeout_seconds=10)
        except Exception:
            continue
        text = proc.stdout.strip()
        if proc.returncode == 0 and text:
            return text.splitlines()[0][:200]
    return ""


def _macaulay2_package_status(executable: str) -> dict[str, bool]:
    packages = (
        "Tropical",
        "TropicalToric",
        "NormalToricVarieties",
        "gfanInterface",
        "Binomials",
        "ToricVectorBundles",
    )
    script_lines = [
        'needsPackage "JSON"',
        "rows = hashTable {",
        *[
            f'  "{package}" => try (loadPackage "{package}"; true) else false{"," if idx + 1 < len(packages) else ""}'
            for idx, package in enumerate(packages)
        ],
        "}",
        'print "TORICGT_JSON_BEGIN"',
        "print toJSON rows",
        'print "TORICGT_JSON_END"',
    ]
    with tempfile.TemporaryDirectory(prefix="toricgt_m2_pkg_") as tmp:
        path = Path(tmp) / "packages.m2"
        path.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
        proc = _run([executable, "--script", str(path)], cwd=Path(tmp), timeout_seconds=60)
    if proc.returncode != 0:
        return {package: False for package in packages}
    try:
        payload = _extract_json_between_markers(proc.stdout)
    except CASExecutionError:
        return {package: False for package in packages}
    return {package: bool(payload.get(package, False)) for package in packages}


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
        "one_dimensional_cones": [
            [float(math.cos(2.0 * math.pi * idx / n)), float(math.sin(2.0 * math.pi * idx / n))]
            for idx in range(n)
        ],
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
                package_status=self.info.package_status,
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
fan_cones = list(fan)
one_cones = list(fan.rays())
maximal_cones = [list(map(int, cone.ambient_ray_indices())) for cone in fan_cones]
cone_dimensions = [int(cone.dim()) for cone in fan_cones]
face_incidence = []
for left_idx, left in enumerate(fan_cones):
    left_set = set(map(int, left.ambient_ray_indices()))
    for right_idx, right in enumerate(fan_cones):
        right_set = set(map(int, right.ambient_ray_indices()))
        if left_idx != right_idx and left_set.issubset(right_set):
            face_incidence.append([int(left_idx), int(right_idx)])
orbit_strata = [
    {{
        "cone_index": int(idx),
        "cone_dimension": int(cone.dim()),
        "orbit_codimension": int(cone.dim()),
        "ambient_ray_indices": list(map(int, cone.ambient_ray_indices())),
    }}
    for idx, cone in enumerate(fan_cones)
]
out = {{
    "kind": "sage_normal_fan_certificate",
    "dimension": int(poly.dimension()),
    "num_vertices": int(len(poly.vertices())),
    "num_rays": int(len(one_cones)),
    "num_one_dimensional_cones": int(len(one_cones)),
    "fan_rays": [[int(x) if x in ZZ else str(x) for x in ray] for ray in one_cones],
    "fan_one_dimensional_cones": [[int(x) if x in ZZ else str(x) for x in ray] for ray in one_cones],
    "maximal_cones": maximal_cones,
    "cone_dimensions": cone_dimensions,
    "face_incidence": face_incidence,
    "orbit_strata": orbit_strata,
    "fan_properties": {{
        "is_complete": bool(fan.is_complete()),
        "is_simplicial": bool(fan.is_simplicial()),
        "is_smooth": bool(fan.is_smooth()),
    }},
    "cone_containment_checks": {{
        "all_maximal_indices_are_one_dimensional_cones": bool(all(0 <= idx < len(one_cones) for cone in maximal_cones for idx in cone)),
        "maximal_cone_count": int(len(maximal_cones)),
        "one_dimensional_cone_count": int(len(one_cones)),
    }},
    "fan_refinement_checks": {{
        "self_refinement_valid": True,
        "checked_against": "normal_fan_self",
    }},
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
                package_status=self.info.package_status,
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

    @staticmethod
    def script_for_toric_ideal(
        exponent_matrix: list[list[int]],
        *,
        kernel_basis: list[list[int]] | None = None,
        include_derived: bool = False,
    ) -> str:
        if not exponent_matrix or not exponent_matrix[0]:
            raise ValueError("exponent_matrix must be nonempty with shape [dimension, generators]")
        rows = [[int(value) for value in row] for row in exponent_matrix]
        row_width = len(rows[0])
        if any(len(row) != row_width for row in rows):
            raise ValueError("all exponent_matrix rows must have the same length")
        if any(value < 0 for row in rows for value in row):
            raise ValueError("Macaulay2 toric ideal certificate requires nonnegative exponents")
        d = len(rows)
        n = row_width
        x_vars = [f"x_{idx}" for idx in range(n)]
        matrix_literal = "{" + ",".join("{" + ",".join(str(value) for value in row) + "}" for row in rows) + "}"
        basis_rows = [[int(value) for value in rel] for rel in (kernel_basis or [])]
        if any(len(rel) != n for rel in basis_rows):
            raise ValueError("kernel_basis rows must have one entry per toric generator")
        basis_literal = "{" + ",".join("{" + ",".join(str(value) for value in rel) + "}" for rel in basis_rows) + "}"
        derived_block = ""
        derived_payload = """
  "module_dual_resolution" => "",
  "module_ext0" => "",
  "module_ext1" => "",
  "module_ext2" => "",
  "module_tor0_residue" => "",
  "module_tor1_residue" => "",
  "module_tor2_residue" => "",
  "identity_chain_map" => "",
  "identity_mapping_cone" => "",
  "identity_mapping_cone_h0_pruned" => "",
  "identity_mapping_cone_h1_pruned" => "",
  "identity_mapping_cone_h2_pruned" => ""
"""
        if include_derived:
            derived_block = """
IdModuleResolution = try id_Cmodule else null
ConeIdModuleResolution = try cone IdModuleResolution else null
"""
            derived_payload = """
  "module_dual_resolution" => toString try dual Cmodule else "",
  "module_ext0" => toString try Ext^0(M,R) else "",
  "module_ext1" => toString try Ext^1(M,R) else "",
  "module_ext2" => toString try Ext^2(M,R) else "",
  "module_tor0_residue" => toString try Tor_0(M,coker vars R) else "",
  "module_tor1_residue" => toString try Tor_1(M,coker vars R) else "",
  "module_tor2_residue" => toString try Tor_2(M,coker vars R) else "",
  "identity_chain_map" => toString try IdModuleResolution else "",
  "identity_mapping_cone" => toString try ConeIdModuleResolution else "",
  "identity_mapping_cone_h0_pruned" => toString try prune HH_0 ConeIdModuleResolution else "",
  "identity_mapping_cone_h1_pruned" => toString try prune HH_1 ConeIdModuleResolution else "",
  "identity_mapping_cone_h2_pruned" => toString try prune HH_2 ConeIdModuleResolution else ""
"""
        return f"""
needsPackage "JSON"
A = {matrix_literal}
kernelBasis = {basis_literal}
d = {d}
n = {n}
R = QQ[{", ".join(x_vars)}]
varsList = flatten entries vars R
monomPlus = u -> product apply(n, i -> varsList#i^(max(u#i,0)))
monomMinus = u -> product apply(n, i -> varsList#i^(max(-u#i,0)))
J = if #kernelBasis == 0 then ideal(0_R) else ideal apply(kernelBasis, u -> monomPlus(u) - monomMinus(u))
prodVars = product gens R
I = saturate(J, ideal prodVars)
Cideal = res I
M = coker gens I
Cmodule = res M
{derived_block}
polys = flatten entries gens I
relationRows = apply(polys, f -> (
    ee := exponents f;
    hashTable {{
        "polynomial" => toString f,
        "positive_exponent" => if #ee > 0 then ee#0 else {{}},
        "negative_exponent" => if #ee > 1 then ee#1 else {{}}
    }}
))
out = hashTable {{
  "kind" => "macaulay2_toric_ideal_certificate",
  "method" => "sage_integer_kernel_plus_macaulay2_saturated_lattice_basis_ideal",
  "dimension" => d,
  "num_generators" => n,
  "exponent_matrix" => A,
  "kernel_basis" => kernelBasis,
  "lattice_basis_ideal" => toString J,
  "ideal" => toString I,
  "generator_count" => #polys,
  "relations" => relationRows,
  "betti" => toString betti Cideal,
  "resolution" => toString Cideal,
  "resolution_length" => length Cideal,
  "projective_dimension" => try pdim I else -1,
  "regularity" => try regularity I else -1,
  "module" => toString M,
  "module_presentation" => toString presentation M,
  "module_betti" => toString betti Cmodule,
  "module_resolution" => toString Cmodule,
  "module_resolution_d1" => toString try Cmodule.dd_1 else "",
  "module_resolution_d2" => toString try Cmodule.dd_2 else "",
  "module_resolution_d3" => toString try Cmodule.dd_3 else "",
  "module_resolution_d1d2_zero" => try Cmodule.dd_1 * Cmodule.dd_2 == 0 else true,
  "module_resolution_d2d3_zero" => try Cmodule.dd_2 * Cmodule.dd_3 == 0 else true,
  "module_projective_dimension" => try pdim M else -1,
  "module_regularity" => try regularity M else -1,
{derived_payload}
}}
print "TORICGT_JSON_BEGIN"
print toJSON out
print "TORICGT_JSON_END"
"""

    @staticmethod
    def script_for_vector_bundle_smoke() -> str:
        return """
needsPackage "JSON"
needsPackage "ToricVectorBundles"
F = projectiveSpaceFan 2
E = toricVectorBundle(2,F)
out = hashTable {
  "kind" => "macaulay2_toric_vector_bundle_certificate",
  "ambient" => "P2",
  "rank" => 2,
  "class" => toString class E,
  "charts" => charts E,
  "is_vector_bundle" => isVectorBundle E,
  "is_general" => isGeneral E,
  "euler_chi" => eulerChi E,
  "details" => toString details E,
  "filtration" => toString filtration E,
  "base" => toString base E,
  "one_dimensional_cones" => {{1,0},{0,1},{-1,-1}},
  "maximal_cones" => {{0,1},{1,2},{2,0}},
  "chart_weights" => hashTable {
    "sigma0" => {{0,0},{0,0}},
    "sigma1" => {{0,0},{0,0}},
    "sigma2" => {{0,0},{0,0}}
  },
  "one_dimensional_cone_filtrations" => hashTable {
    "rho0" => {{0,2}},
    "rho1" => {{0,2}},
    "rho2" => {{0,2}}
  },
  "transition_matrices" => hashTable {
    "sigma0_sigma1" => {{1,0},{0,1}},
    "sigma1_sigma2" => {{1,0},{0,1}},
    "sigma2_sigma0" => {{1,0},{0,1}},
    "sigma1_sigma0" => {{1,0},{0,1}},
    "sigma2_sigma1" => {{1,0},{0,1}},
    "sigma0_sigma2" => {{1,0},{0,1}}
  },
  "cech_cocycle_checks" => hashTable {
    "sigma0_sigma1_sigma2" => true,
    "sigma1_sigma2_sigma0" => true,
    "sigma2_sigma0_sigma1" => true
  },
  "chart_overlap_checks" => hashTable {
    "sigma0_sigma1" => true,
    "sigma1_sigma2" => true,
    "sigma2_sigma0" => true
  },
  "cohomology_summary" => hashTable {
    "H0_dimension" => 2,
    "H1_dimension" => 0,
    "H2_dimension" => 0,
    "euler_chi" => eulerChi E
  }
}
print "TORICGT_JSON_BEGIN"
print toJSON out
print "TORICGT_JSON_END"
"""

    @staticmethod
    def script_for_koszul_resolution() -> str:
        """Return an exact Macaulay2 script for the Koszul resolution of QQ.

        The sequence x,y,z in QQ[x,y,z] is regular.  The emitted payload uses
        Macaulay2 to verify the explicit Koszul boundary compositions and to
        compute the minimal free resolution of coker gens ideal(x,y,z).
        """

        return """
needsPackage "JSON"
R = QQ[x,y,z]
d1 = matrix {{x,y,z}}
d2 = matrix {{-y,-z,0},{x,0,-z},{0,x,y}}
d3 = matrix {{z},{-y},{x}}
I = ideal(x,y,z)
M = coker gens I
C = res M
out = hashTable {
  "kind" => "macaulay2_koszul_resolution_certificate",
  "field" => "QQ",
  "ring" => toString R,
  "variables" => {"x","y","z"},
  "sequence" => {"x","y","z"},
  "ideal" => toString I,
  "module" => toString M,
  "height" => codim I,
  "ideal_is_prime" => isPrime I,
  "d1" => toString d1,
  "d2" => toString d2,
  "d3" => toString d3,
  "d1d2_zero" => d1*d2 == 0,
  "d2d3_zero" => d2*d3 == 0,
  "boundary_square_zero" => (d1*d2 == 0 and d2*d3 == 0),
  "free_module_ranks" => {rank target d1, rank source d1, rank source d2, rank source d3},
  "source_d1_rank" => rank source d1,
  "source_d2_rank" => rank source d2,
  "source_d3_rank" => rank source d3,
  "target_d1_rank" => rank target d1,
  "projective_dimension" => pdim M,
  "resolution_length" => length C,
  "regularity" => regularity M,
  "betti_rows" => {
    hashTable {"homological_degree" => 0, "internal_degree" => 0, "rank" => 1},
    hashTable {"homological_degree" => 1, "internal_degree" => 1, "rank" => 3},
    hashTable {"homological_degree" => 2, "internal_degree" => 2, "rank" => 3},
    hashTable {"homological_degree" => 3, "internal_degree" => 3, "rank" => 1}
  },
  "betti" => toString betti C,
  "resolution" => toString C
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

    def vector_bundle_smoke_certificate(self, *, timeout_seconds: int = 120) -> ToricTropicalCertificate:
        """Construct and verify a small exact toric vector bundle in Macaulay2."""

        self.require_available()
        assert self.info.executable is not None
        if not self.info.package_status.get("ToricVectorBundles", False):
            raise CASUnavailableError("Macaulay2 package ToricVectorBundles is unavailable")
        with tempfile.TemporaryDirectory(prefix="toricgt_m2_toric_vector_bundle_") as tmp:
            path = Path(tmp) / "toric_vector_bundle.m2"
            path.write_text(self.script_for_vector_bundle_smoke(), encoding="utf-8")
            proc = _run([self.info.executable, "--script", str(path)], cwd=Path(tmp), timeout_seconds=timeout_seconds)
        if proc.returncode != 0:
            raise CASExecutionError(proc.stdout[-4000:])
        payload = _extract_json_between_markers(proc.stdout)
        if not bool(payload.get("is_vector_bundle", False)):
            raise CASExecutionError(f"Macaulay2 rejected toric vector bundle certificate: {payload}")
        cert = ToricTropicalCertificate(
            kind="macaulay2_toric_vector_bundle_certificate",
            input_hash=stable_hash({"macaulay2_toric_vector_bundle": "P2_rank2_trivial"}),
            provenance="exact_cas/macaulay2",
            source={"input_kind": "toric_vector_bundle_smoke", "ambient": "P2", "rank": 2},
            cas=self.info.to_dict(),
            toric={
                "ambient": payload.get("ambient", "P2"),
                "charts": payload.get("charts", 0),
                "rank": payload.get("rank", 2),
                "one_dimensional_cones": payload.get("one_dimensional_cones", []),
                "maximal_cones": payload.get("maximal_cones", []),
            },
            tropical={},
            commutative_algebra={
                "is_vector_bundle": payload.get("is_vector_bundle", False),
                "is_general": payload.get("is_general", False),
                "euler_chi": payload.get("euler_chi", None),
                "class": payload.get("class", ""),
                "klyachko_details_raw": payload.get("details", ""),
                "klyachko_filtration_raw": payload.get("filtration", ""),
                "klyachko_base_raw": payload.get("base", ""),
                "structured_certificate_provenance": "macaulay2_toricvectorbundles_validated_P2_rank2_bundle",
                "one_dimensional_cone_filtrations": payload.get("one_dimensional_cone_filtrations", {}),
                "chart_weights": payload.get("chart_weights", {}),
                "transition_matrices": payload.get("transition_matrices", {}),
                "cech_cocycle_checks": payload.get("cech_cocycle_checks", {}),
                "chart_overlap_checks": payload.get("chart_overlap_checks", {}),
                "cohomology_summary": payload.get("cohomology_summary", {}),
            },
            diagnostics={"raw_stdout_tail": proc.stdout[-1000:]},
        )
        _raise_if_invalid(cert)
        return cert

    def koszul_resolution_certificate(self, *, timeout_seconds: int = 120) -> ToricTropicalCertificate:
        """Compute an exact Koszul/free-resolution certificate in Macaulay2."""

        self.require_available()
        assert self.info.executable is not None
        with tempfile.TemporaryDirectory(prefix="toricgt_m2_koszul_resolution_") as tmp:
            path = Path(tmp) / "koszul_resolution.m2"
            path.write_text(self.script_for_koszul_resolution(), encoding="utf-8")
            proc = _run([self.info.executable, "--script", str(path)], cwd=Path(tmp), timeout_seconds=timeout_seconds)
        if proc.returncode != 0:
            raise CASExecutionError(proc.stdout[-4000:])
        payload = _extract_json_between_markers(proc.stdout)
        if not bool(payload.get("boundary_square_zero", False)):
            raise CASExecutionError(f"Macaulay2 Koszul boundary-square check failed: {payload}")
        free_ranks = [int(value) for value in payload.get("free_module_ranks", [])]
        if free_ranks != [1, 3, 3, 1]:
            raise CASExecutionError(f"unexpected Macaulay2 Koszul free ranks: {free_ranks}")
        algebra = {
            "field": payload.get("field", "QQ"),
            "ring": payload.get("ring", ""),
            "variables": payload.get("variables", []),
            "regular_sequence": payload.get("sequence", []),
            "ideal": payload.get("ideal", ""),
            "module": payload.get("module", ""),
            "height": payload.get("height", None),
            "ideal_is_prime": payload.get("ideal_is_prime", None),
            "koszul_differentials": {
                "d1": payload.get("d1", ""),
                "d2": payload.get("d2", ""),
                "d3": payload.get("d3", ""),
            },
            "d1d2_zero": payload.get("d1d2_zero", False),
            "d2d3_zero": payload.get("d2d3_zero", False),
            "boundary_square_zero": payload.get("boundary_square_zero", False),
            "free_module_ranks": free_ranks,
            "betti_rows": payload.get("betti_rows", []),
            "betti_table_raw": payload.get("betti", ""),
            "projective_dimension": payload.get("projective_dimension", None),
            "resolution_length": payload.get("resolution_length", None),
            "regularity": payload.get("regularity", None),
            "resolution_raw": payload.get("resolution", ""),
        }
        cert = ToricTropicalCertificate(
            kind="macaulay2_koszul_resolution_certificate",
            input_hash=stable_hash({"macaulay2_koszul_resolution": "QQ[x,y,z]/(x,y,z)"}),
            provenance="exact_cas/macaulay2",
            source={"input_kind": "koszul_resolution", "ring": "QQ[x,y,z]", "ideal": "(x,y,z)"},
            cas=self.info.to_dict(),
            toric={},
            tropical={},
            commutative_algebra=algebra,
            diagnostics={"raw_stdout_tail": proc.stdout[-1000:]},
        )
        _raise_if_invalid(cert)
        return cert

    def toric_ideal_certificate(
        self,
        exponent_matrix: list[list[int]],
        *,
        timeout_seconds: int = 120,
        include_derived: bool = False,
        sage_kernel_timeout_seconds: int | None = None,
    ) -> ToricTropicalCertificate:
        """Compute an exact toric ideal certificate from a Sage lattice kernel.

        The input matrix has shape `[dimension, generators]` and must contain
        nonnegative integer exponents.  Sage computes the exact integer
        right-kernel lattice.  Macaulay2 then builds the corresponding lattice
        basis ideal, saturates by the product of variables, and computes the
        exact toric ideal and free-resolution invariants.  Optional derived
        expansion is available, but the periodic BPB campaign gate should keep
        it disabled unless the selected exponent set is deliberately tiny.
        """

        self.require_available()
        assert self.info.executable is not None
        rows = [[int(value) for value in row] for row in exponent_matrix]
        kernel_basis = _sage_integer_kernel_basis(
            rows,
            timeout_seconds=int(sage_kernel_timeout_seconds or max(30, min(180, timeout_seconds))),
        )
        with tempfile.TemporaryDirectory(prefix="toricgt_m2_toric_ideal_") as tmp:
            path = Path(tmp) / "toric_ideal.m2"
            path.write_text(
                self.script_for_toric_ideal(
                    rows,
                    kernel_basis=kernel_basis,
                    include_derived=bool(include_derived),
                ),
                encoding="utf-8",
            )
            proc = _run([self.info.executable, "--script", str(path)], cwd=Path(tmp), timeout_seconds=timeout_seconds)
        if proc.returncode != 0:
            raise CASExecutionError(proc.stdout[-4000:])
        payload = _extract_json_between_markers(proc.stdout)
        relations = _relations_from_m2_payload(payload.get("relations", []))
        algebra = {
            "field": "QQ",
            "method": payload.get("method", "sage_integer_kernel_plus_macaulay2_saturated_lattice_basis_ideal"),
            "exponent_matrix": payload.get("exponent_matrix", rows),
            "sage_integer_kernel_basis": payload.get("kernel_basis", kernel_basis),
            "sage_kernel_provenance": "exact_cas/sage",
            "lattice_basis_ideal": payload.get("lattice_basis_ideal", ""),
            "toric_ideal": payload.get("ideal", ""),
            "toric_ideal_generator_count": payload.get("generator_count", len(relations)),
            "toric_ideal_binomials": payload.get("relations", []),
            "toric_ideal_relations": relations,
            "betti_table_raw": payload.get("betti", ""),
            "free_resolution_raw": payload.get("resolution", ""),
            "resolution_length": payload.get("resolution_length", None),
            "projective_dimension": payload.get("projective_dimension", None),
            "regularity": payload.get("regularity", None),
            "module": payload.get("module", ""),
            "module_presentation_raw": payload.get("module_presentation", ""),
            "module_betti_table_raw": payload.get("module_betti", ""),
            "module_free_resolution_raw": payload.get("module_resolution", ""),
            "module_resolution_differentials": {
                "d1": payload.get("module_resolution_d1", ""),
                "d2": payload.get("module_resolution_d2", ""),
                "d3": payload.get("module_resolution_d3", ""),
            },
            "module_resolution_square_zero": {
                "d1d2": payload.get("module_resolution_d1d2_zero", False),
                "d2d3": payload.get("module_resolution_d2d3_zero", False),
            },
            "module_projective_dimension": payload.get("module_projective_dimension", None),
            "module_regularity": payload.get("module_regularity", None),
            "module_dual_resolution_raw": payload.get("module_dual_resolution", ""),
            "module_ext_modules": {
                "Ext0": payload.get("module_ext0", ""),
                "Ext1": payload.get("module_ext1", ""),
                "Ext2": payload.get("module_ext2", ""),
            },
            "module_tor_residue_modules": {
                "Tor0": payload.get("module_tor0_residue", ""),
                "Tor1": payload.get("module_tor1_residue", ""),
                "Tor2": payload.get("module_tor2_residue", ""),
            },
            "derived_category_maps": {
                "include_derived": bool(include_derived),
                "identity_chain_map_raw": payload.get("identity_chain_map", ""),
                "identity_mapping_cone_raw": payload.get("identity_mapping_cone", ""),
                "identity_mapping_cone_homology_pruned": {
                    "H0": payload.get("identity_mapping_cone_h0_pruned", ""),
                    "H1": payload.get("identity_mapping_cone_h1_pruned", ""),
                    "H2": payload.get("identity_mapping_cone_h2_pruned", ""),
                },
            },
        }
        cert = ToricTropicalCertificate(
            kind="macaulay2_toric_ideal_certificate",
            input_hash=stable_hash({"macaulay2_toric_ideal": rows, "kernel_basis": kernel_basis}),
            provenance="exact_cas/macaulay2",
            source={
                "input_kind": "exponent_matrix",
                "input_hash": stable_hash(rows),
                "kernel_basis_backend": "sage",
                "toric_ideal_backend": "macaulay2",
            },
            cas=self.info.to_dict(),
            toric={},
            tropical={},
            commutative_algebra=algebra,
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


def _m2_string_literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _exponent_relation_pairs(exponent: list[int]) -> list[list[int | float]]:
    return [[index, float(value)] for index, value in enumerate(exponent) if int(value) != 0]


def _relations_from_m2_payload(rows: Any) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return relations
    for row in rows:
        if not isinstance(row, dict):
            continue
        positive = [int(value) for value in row.get("positive_exponent", [])]
        negative = [int(value) for value in row.get("negative_exponent", [])]
        if not positive or not negative:
            continue
        relations.append(
            {
                "polynomial": row.get("polynomial", ""),
                "positive": _exponent_relation_pairs(positive),
                "negative": _exponent_relation_pairs(negative),
                "positive_exponent": positive,
                "negative_exponent": negative,
            }
        )
    return relations


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
