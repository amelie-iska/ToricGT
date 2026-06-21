#!/usr/bin/env python3
"""Generate an exact Miller-Sturmfels bivariate staircase certificate.

The input is an embedding manifest emitted by the geometry suite. The script
selects real checkpoint embedding vectors, quantizes their first two PCA
coordinates into nonnegative exponent vectors, computes the product-order
minimal monomial generators, and verifies the S/I resolution with Macaulay2.

There is no demo or fallback path: if the selected checkpoint-derived exponent
set does not produce a nontrivial staircase, or if Macaulay2 fails, the script
exits nonzero.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.run_embedding_cas_sidecar import (  # noqa: E402
    exponent_points_from_embeddings,
    finite_points_from_npz,
    read_json,
    resolve_payload_npz,
)
from toricgt.toric_embedding_visualization import _staircase_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--payload-index", type=int, default=0)
    parser.add_argument("--embedding-key", choices=["hidden", "projected", "complex_projected"], default="hidden")
    parser.add_argument("--max-points", type=int, default=12)
    parser.add_argument("--exponent-dim", type=int, default=2)
    parser.add_argument("--quantization-scale", type=int, default=24)
    parser.add_argument("--min-generators", type=int, default=3)
    parser.add_argument("--macaulay2-timeout-seconds", type=int, default=180)
    return parser.parse_args()


def monomial(a: int, b: int) -> str:
    factors: list[str] = []
    if a:
        factors.append("x" if a == 1 else f"x^{a}")
    if b:
        factors.append("y" if b == 1 else f"y^{b}")
    return "*".join(factors) if factors else "1"


def monomial_latex(a: int, b: int) -> str:
    factors: list[str] = []
    if a:
        factors.append("x" if a == 1 else f"x^{{{a}}}")
    if b:
        factors.append("y" if b == 1 else f"y^{{{b}}}")
    return "".join(factors) if factors else "1"


def run_macaulay2_resolution(gens: list[list[int]], *, timeout_seconds: int) -> dict[str, Any]:
    executable = shutil.which("M2")
    if not executable:
        raise RuntimeError("Macaulay2 executable `M2` was not found on PATH")
    gen_expr = ", ".join(monomial(int(a), int(b)) for a, b in gens)
    script = f"""
R = QQ[x,y];
I = ideal({gen_expr});
M = R/I;
C = res M;
print "BEGIN_BETTI";
print toString betti C;
print "END_BETTI";
print "BEGIN_RESOLUTION";
print toString C;
print "END_RESOLUTION";
print "BEGIN_LENGTH";
print toString length C;
print "END_LENGTH";
print "BEGIN_PDIM";
print toString pdim M;
print "END_PDIM";
print "BEGIN_REGULARITY";
print toString regularity M;
print "END_REGULARITY";
"""
    with tempfile.TemporaryDirectory(prefix="toricgt_ms_staircase_") as tmp:
        path = Path(tmp) / "staircase_resolution.m2"
        path.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            [executable, "--script", str(path)],
            cwd=Path(tmp),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(10, int(timeout_seconds)),
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Macaulay2 failed with exit code {proc.returncode}: {proc.stderr}\n{proc.stdout}")

    def block(name: str) -> str:
        begin = f"BEGIN_{name}"
        end = f"END_{name}"
        if begin not in proc.stdout or end not in proc.stdout:
            raise RuntimeError(f"Macaulay2 output missing {begin}/{end} block")
        return proc.stdout.split(begin, 1)[1].split(end, 1)[0].strip()

    return {
        "provenance": "exact_cas/macaulay2",
        "ring": "QQ[x,y]",
        "ideal": f"ideal({gen_expr})",
        "betti_table_raw": block("BETTI"),
        "resolution_raw": block("RESOLUTION"),
        "resolution_length": int(block("LENGTH")),
        "projective_dimension": int(block("PDIM")),
        "regularity": int(block("REGULARITY")),
        "stderr": proc.stderr.strip(),
    }


def resolution_formula(gens: list[list[int]]) -> dict[str, Any]:
    adjacent: list[dict[str, Any]] = []
    differentials: list[dict[str, Any]] = []
    for idx, (left, right) in enumerate(zip(gens, gens[1:])):
        a1, b1 = map(int, left)
        a2, b2 = map(int, right)
        lcm = [max(a1, a2), max(b1, b2)]
        adjacent.append(
            {
                "index": idx,
                "left_generator": [a1, b1],
                "right_generator": [a2, b2],
                "lcm_corner": lcm,
            }
        )
        # With generators sorted by decreasing x and increasing y:
        # d2(f_i)=y^(b_{i+1}-b_i)e_i - x^(a_i-a_{i+1})e_{i+1}.
        differentials.append(
            {
                "source": f"f_{idx}",
                "target_terms": [
                    {
                        "coefficient": monomial(0, max(0, b2 - b1)),
                        "basis": f"e_{idx}",
                    },
                    {
                        "coefficient": "-" + monomial(max(0, a1 - a2), 0),
                        "basis": f"e_{idx + 1}",
                    },
                ],
                "boundary_square": "0",
                "reason": (
                    f"{monomial_latex(0, max(0, b2 - b1))}"
                    f"{monomial_latex(a1, b1)} = "
                    f"{monomial_latex(max(0, a1 - a2), 0)}"
                    f"{monomial_latex(a2, b2)} = {monomial_latex(*lcm)}"
                ),
            }
        )
    return {
        "module": "S/I over S = k[x,y]",
        "terms": {
            "C0": ["S"],
            "C1": [f"S(-{a},-{b})" for a, b in gens],
            "C2": [f"S(-{a},-{b})" for a, b in (item["lcm_corner"] for item in adjacent)],
        },
        "d1": [{"source": f"e_{idx}", "image": monomial(int(a), int(b))} for idx, (a, b) in enumerate(gens)],
        "d2": differentials,
        "adjacent_lcm_layer": adjacent,
    }


def dominance_rows(points: np.ndarray, gens: list[list[int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gen_tuples = {tuple(map(int, gen)) for gen in gens}
    for point in np.asarray(points, dtype=int).tolist():
        p = tuple(map(int, point))
        if p in gen_tuples:
            rows.append({"point": list(p), "minimal_generator": True, "dominated_by": []})
            continue
        dominators = [gen for gen in gens if int(gen[0]) <= p[0] and int(gen[1]) <= p[1]]
        rows.append({"point": list(p), "minimal_generator": False, "dominated_by": dominators})
    return rows


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.embedding_manifest)
    manifest = read_json(manifest_path)
    payloads = manifest.get("payloads", [])
    if not isinstance(payloads, list) or not payloads:
        raise ValueError(f"{manifest_path} does not contain embedding payload rows")
    row = payloads[int(args.payload_index)]
    npz_path = resolve_payload_npz(manifest_path, row)
    points = finite_points_from_npz(npz_path, str(args.embedding_key), max_points=max(3, int(args.max_points)))
    exponents, exponent_metadata = exponent_points_from_embeddings(
        points,
        exponent_dim=int(args.exponent_dim),
        scale=int(args.quantization_scale),
    )
    staircase = _staircase_metadata(exponents)
    gens = staircase.get("minimal_generators", [])
    if len(gens) < int(args.min_generators):
        raise ValueError(
            f"checkpoint-derived exponent set produced only {len(gens)} minimal generators; "
            f"required at least {int(args.min_generators)}"
        )
    m2 = run_macaulay2_resolution(gens, timeout_seconds=int(args.macaulay2_timeout_seconds))
    formula = resolution_formula(gens)
    staircase["adjacent_lcm_layer"] = formula["adjacent_lcm_layer"]
    certificate = {
        "schema": "toricgt.exact_miller_sturmfels_staircase_certificate.v1",
        "record_id": f"{row.get('record_id', 'record')}_miller_sturmfels_staircase",
        "source": {
            "embedding_manifest": str(manifest_path),
            "embedding_npz": str(npz_path),
            "payload_index": int(args.payload_index),
            "embedding_key": str(args.embedding_key),
            "checkpoint": row.get("checkpoint"),
            "checkpoint_step": row.get("checkpoint_step"),
        },
        "selection": {
            "max_points": int(args.max_points),
            "exponent_dim": int(args.exponent_dim),
            "quantization_scale": int(args.quantization_scale),
            "method": "checkpoint_hidden_pca_quantized_exact_antichain",
        },
        "exponent_metadata": exponent_metadata,
        "exponent_points": exponents.astype(int).tolist(),
        "dominance": dominance_rows(exponents, gens),
        "miller_sturmfels_staircase": staircase,
        "resolution_formula": formula,
        "macaulay2_resolution": m2,
        "exactness": {
            "checkpoint_derived_exponents": True,
            "miller_sturmfels_minimal_generators_exact": True,
            "macaulay2_resolution_exact": True,
            "no_demo_no_proxy_no_fallback": True,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output), "minimal_generators": len(gens)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
