from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from toricgt.toric_embedding_visualization import collect_sidecar_records, render_toric_embedding_report


def _fixture_sidecar() -> dict:
    return {
        "schema": "toricgt.embedding_cas_sidecar_record.v1",
        "record_id": "record_000",
        "embedding_npz": "fixture.npz",
        "embedding_key": "hidden",
        "exponent_points": [[0, 3], [1, 0], [2, 2], [3, 3], [4, 1]],
        "biases": [0, 1, 0, -1, 2],
        "sage_normal_fan": {
            "provenance": "exact_cas/sage",
            "toric": {
                "dimension": 2,
                "num_vertices": 5,
                "num_one_dimensional_cones": 4,
                "fan_one_dimensional_cones": [[1, 0], [0, 1], [-1, 0], [0, -1]],
                "maximal_cones": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "cone_dimensions": [2, 2, 2, 2],
                "face_incidence": [[0, 1], [1, 2], [2, 3]],
                "orbit_strata": [
                    {"cone_index": 0, "cone_dimension": 2, "orbit_codimension": 2},
                    {"cone_index": 1, "cone_dimension": 2, "orbit_codimension": 2},
                    {"cone_index": 2, "cone_dimension": 1, "orbit_codimension": 1},
                    {"cone_index": 3, "cone_dimension": 0, "orbit_codimension": 0},
                ],
                "fan_properties": {"is_complete": True, "is_simplicial": True, "is_smooth": True},
            },
        },
        "macaulay2_toric_ideal": {
            "provenance": "exact_cas/macaulay2",
            "commutative_algebra": {
                "field": "QQ",
                "toric_ideal": "ideal(x_0*x_3-x_1*x_2)",
                "toric_ideal_generator_count": 1,
                "toric_ideal_relations": [
                    {
                        "polynomial": "x_0*x_3-x_1*x_2",
                        "positive_exponent": [1, 0, 0, 1, 0],
                        "negative_exponent": [0, 1, 1, 0, 0],
                    }
                ],
                "free_resolution_raw": "0 <-- R^1 <-- R^1 <-- 0",
                "module_free_resolution_raw": "0 <-- cokernel | f | <-- R^1 <-- 0",
                "resolution_length": 1,
                "projective_dimension": 1,
                "regularity": 2,
                "module_resolution_square_zero": {"d1d2": True, "d2d3": True},
                "module_ext_modules": {"Ext0": "module", "Ext1": "module"},
                "module_tor_residue_modules": {"Tor0": "QQ", "Tor1": "QQ"},
                "derived_category_maps": {
                    "identity_chain_map_raw": "chainComplexMap",
                    "identity_mapping_cone_raw": "chainComplex",
                    "identity_mapping_cone_homology_pruned": {"H0": "R^0", "H1": "R^0"},
                },
            },
        },
    }


def _fixture_vector_bundle() -> dict:
    return {
        "kind": "macaulay2_toric_vector_bundle_certificate",
        "input_hash": "unit",
        "provenance": "exact_cas/macaulay2",
        "created_at_utc": "2026-01-01T00:00:00Z",
        "source": {"input_kind": "unit"},
        "cas": {"backend": "macaulay2", "available": True},
        "toric": {
            "ambient": "P2",
            "rank": 2,
            "charts": 3,
            "one_dimensional_cones": [[1, 0], [0, 1], [-1, -1]],
            "maximal_cones": [[0, 1], [1, 2], [2, 0]],
        },
        "commutative_algebra": {
            "is_vector_bundle": True,
            "euler_chi": 2,
            "structured_certificate_provenance": "unit",
            "one_dimensional_cone_filtrations": {"rho0": [[0, 2]], "rho1": [[0, 2]], "rho2": [[0, 2]]},
            "chart_weights": {"sigma0": [[0, 0], [0, 0]]},
            "transition_matrices": {"sigma0_sigma1": [[1, 0], [0, 1]]},
            "cech_cocycle_checks": {"sigma0_sigma1_sigma2": True},
            "chart_overlap_checks": {"sigma0_sigma1": True},
            "cohomology_summary": {"H0_dimension": 2, "H1_dimension": 0},
        },
    }


def _visible_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<script.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def test_render_toric_embedding_report_from_exact_fixture(tmp_path: Path) -> None:
    sidecar = tmp_path / "record_000_cas_sidecar.json"
    sidecar.write_text(json.dumps(_fixture_sidecar(), indent=2), encoding="utf-8")
    vb = tmp_path / "vector_bundle.json"
    vb.write_text(json.dumps(_fixture_vector_bundle(), indent=2), encoding="utf-8")

    manifest = render_toric_embedding_report(
        sidecar_records=[sidecar],
        output_dir=tmp_path / "report",
        vector_bundle_certificate=vb,
    )

    assert manifest["record_count"] == 1
    index = tmp_path / "report" / "index.html"
    page = tmp_path / "report" / "records" / "record_000_toric_embedding.html"
    vector_page = tmp_path / "report" / "vector_bundle" / "klyachko_vector_bundle.html"
    assert index.exists()
    assert page.exists()
    assert vector_page.exists()
    html = page.read_text(encoding="utf-8")
    for heading in [
        "Newton Polytope",
        "Lifted Newton Polytope",
        "Initial Degeneration And Active Chambers",
        "Normal Fan And One-Dimensional Cones",
        "Toric Orbit-Stratum Incidence",
        "Divisor, Balance, And Chow Audit State",
        "Toric Ideal Relations",
        "Free Resolution And Differentials",
        "Miller-Sturmfels Staircase",
        "Klyachko Vector-Bundle Filtrations",
        "Cox / Sheaf / Ext / Tor / Derived-Map Summary",
    ]:
        assert heading in html
    assert "Balance/Chow class not certified" in html
    visible = _visible_text(page) + _visible_text(index) + _visible_text(vector_page)
    assert re.search(r"\b(ray|rays)\b", visible, flags=re.IGNORECASE) is None


def test_render_toric_embedding_report_cli(tmp_path: Path) -> None:
    sidecar_dir = tmp_path / "embedding_cas_sidecar" / "records"
    sidecar_dir.mkdir(parents=True)
    sidecar = sidecar_dir / "record_000_cas_sidecar.json"
    sidecar.write_text(json.dumps(_fixture_sidecar(), indent=2), encoding="utf-8")

    out = tmp_path / "cli_report"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/render_toric_embedding_report.py",
            "--sidecar-dir",
            str(sidecar_dir.parent),
            "--output-dir",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["record_count"] == 1
    assert (out / "records" / "record_000_toric_embedding.html").exists()
    assert "no vector-bundle certificate requested" in (out / "index.html").read_text(encoding="utf-8")


def test_collect_sidecar_records_deduplicates(tmp_path: Path) -> None:
    records = tmp_path / "records"
    records.mkdir()
    sidecar = records / "record_000_cas_sidecar.json"
    sidecar.write_text("{}", encoding="utf-8")
    collected = collect_sidecar_records(sidecar_records=[sidecar], sidecar_dir=tmp_path)
    assert collected == [sidecar.resolve()]
