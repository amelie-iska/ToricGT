from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from toricgt.gudhi_persistence import (
    GudhiPersistenceConfig,
    audit_point_cloud,
    bigraded_chain_presentation,
    finite_field_chain_audit,
    torch_persistence_image,
    torch_persistence_landscape,
    vectorized_point_cloud_signature,
    xy_grid_module_summary,
)


def test_gudhi_macaulay2_bigraded_resolution_for_small_cloud() -> None:
    if shutil.which("M2") is None:
        pytest.skip("Macaulay2 is not installed")
    points = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.45, 0.55]],
        dtype=float,
    )
    cfg = GudhiPersistenceConfig(
        max_points=5,
        num_radii=3,
        num_levels=3,
        landscape_resolution=12,
        landscape_layers=2,
        image_resolution=5,
        macaulay2_timeout_seconds=60,
    )
    audit = audit_point_cloud(points, record_id="small_loop", cfg=cfg)

    assert audit["simplex_tree"]["num_simplices"] > 0
    assert audit["two_parameter_module"]["ring"] == "F2[x_level,y_radius]"
    assert audit["two_parameter_module"]["commutative_square_residual"] == 0.0
    assert audit["two_parameter_module"]["commutative_square_chain_residuals_by_dimension"] == {
        "0": 0,
        "1": 0,
        "2": 0,
    }
    assert audit["macaulay2_resolution"]["kind"] == "macaulay2_bigraded_persistence_resolution"
    assert audit["macaulay2_resolution"]["homogeneous_d1"] is True
    assert audit["macaulay2_resolution"]["homogeneous_d2"] is True
    assert audit["macaulay2_resolution"]["d_squared_zero"] is True
    assert "H0_betti" in audit["macaulay2_resolution"]["homology_and_resolutions"]
    derived = audit["macaulay2_resolution"]["derived_category_maps"]
    assert "ChainComplexMap" in derived["chain_identity_map"]
    assert derived["chain_identity_mapping_cone_homology_pruned"]["H0"] == "R^0"
    assert derived["chain_identity_mapping_cone_homology_pruned"]["H1"] == "R^0"
    assert "H0_Ext0" in derived["Ext_modules"]
    assert "H1_Tor1" in derived["Tor_residue_modules"]
    assert audit["xy_grid_module"]["kind"] == "miller_sturmfels_bivariate_grid_summary"
    assert audit["xy_grid_module"]["chain_degrees"]["1"]["generator_count"] > 0
    assert "adjacent_lcm_syzygies" in audit["xy_grid_module"]["chain_degrees"]["1"]
    assert audit["finite_field_chain_audit"]["field"] == "F2"
    assert audit["finite_field_chain_audit"]["d_squared_zero"] is True
    assert audit["two_parameter_module"]["structure_map_summary"]["mean_simplicial_map_valid_fraction"] == 1.0


def test_bigraded_chain_presentation_has_nonnegative_boundary_monomials() -> None:
    points = np.asarray([[0.0], [0.4], [1.0], [1.3]], dtype=float)
    radii = np.asarray([0.1, 0.5, 1.0], dtype=float)
    presentation = bigraded_chain_presentation(points, radii, max_dimension=2)
    for matrix in presentation["boundary_matrices"].values():
        for row in matrix:
            for entry in row:
                assert "^- " not in entry
                assert "x_level^-" not in entry
                assert "y_radius^-" not in entry


def test_xy_grid_summary_reports_inner_and_outer_corners() -> None:
    presentation = {
        "ring": "F2[x_level,y_radius]",
        "variables": ["x_level", "y_radius"],
        "generators": {
            "0": [{"degree": [0, 0]}, {"degree": [1, 0]}],
            "1": [{"degree": [0, 2]}, {"degree": [2, 0]}, {"degree": [2, 2]}],
            "2": [{"degree": [2, 2]}],
        },
    }
    summary = xy_grid_module_summary(presentation, {"hilbert_h0": [[1, 1], [1, 0]]})

    c1 = summary["chain_degrees"]["1"]
    assert c1["minimal_inner_corners"] == [[0, 2], [2, 0]]
    assert c1["adjacent_lcm_outer_corners"] == [[2, 2]]
    assert c1["adjacent_lcm_syzygies"][0]["left_multiplier"] == [2, 0]


def test_torch_persistence_vectorizers_are_differentiable() -> None:
    diagram = torch.tensor([[0.0, 0.4], [0.1, 0.8], [0.25, 0.9]], requires_grad=True)
    grid = torch.linspace(0.0, 1.0, 16)
    landscape = torch_persistence_landscape(diagram, grid, layers=3)
    image = torch_persistence_image(diagram, grid, grid, sigma=0.15)
    loss = landscape.pow(2).mean() + image.pow(2).mean()
    loss.backward()

    assert landscape.shape == (3, 16)
    assert image.shape == (16, 16)
    assert diagram.grad is not None
    assert torch.isfinite(diagram.grad).all()


def test_vectorized_point_cloud_signature_uses_exact_gudhi() -> None:
    points = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]],
        dtype=float,
    )
    cfg = GudhiPersistenceConfig(
        max_points=5,
        max_dimension=2,
        landscape_resolution=10,
        landscape_layers=2,
        image_resolution=5,
        macaulay2_resolutions=False,
    )
    signature, metrics = vectorized_point_cloud_signature(points, cfg)

    assert signature.ndim == 1
    assert signature.size > 0
    assert metrics["backend_gudhi"] == 1.0
    assert metrics["num_simplices"] > 0
    assert metrics["h0_interval_count"] >= 0.0
    assert metrics["h1_landscape_norm"] >= 0.0
    assert metrics["d_squared_residual"] == 0.0


def test_gudhi_audit_script_writes_dark_html_and_m2_script(tmp_path: Path) -> None:
    if shutil.which("M2") is None:
        pytest.skip("Macaulay2 is not installed")
    points = tmp_path / "points.json"
    points.write_text(
        json.dumps(
            {
                "point_clouds": [
                    {
                        "record_id": "audit_square",
                        "points": [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]],
                    },
                    {
                        "record_id": "audit_triangle",
                        "points": [[0, 0], [1, 0], [0.5, 0.86], [0.5, 0.4], [0.2, 0.2]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "audit"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_gudhi_persistence_audit.py",
            "--points-json",
            str(points),
            "--output-dir",
            str(out),
            "--records",
            "2",
            "--max-points",
            "5",
            "--num-radii",
            "3",
            "--num-levels",
            "3",
            "--landscape-resolution",
            "12",
            "--image-resolution",
            "5",
            "--macaulay2-timeout-seconds",
            "60",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    index = out / "index.html"
    assert index.exists()
    assert "F2[x_level,y_radius]" in index.read_text(encoding="utf-8")
    record_pages = list((out / "records").glob("*.html"))
    assert record_pages
    record_html = record_pages[0].read_text(encoding="utf-8")
    assert "F2[x_level,y_radius] Miller-Sturmfels Staircase Module View" in record_html
    assert "Miller-Sturmfels staircase boundary" in record_html
    assert "Pareto frontier" not in record_html
    assert "resolution-strip" in record_html
    assert "differential-heatmap" in record_html
    assert "Readable Resolution And Differential Summary" in record_html
    assert "Vectorized Persistent-Homology Feature Dashboard" in record_html
    assert "persistence landscapes" in record_html.lower()
    assert "persistence images" in record_html.lower()
    assert "silhouettes" in record_html.lower()
    assert "entropy vectors" in record_html.lower()
    assert "Betti curves" in record_html
    assert "M2 d1*d2=0" in record_html
    assert "Hilbert H0 grid" in record_html
    assert "Exact Chain And Simplicial-Map Audit" in record_html
    assert "Exact Module Maps And Derived Identity-Cones" in record_html
    assert "Analogical Memory PH Retrieval Panel" in record_html
    assert "Derived maps, identity mapping cones, Ext, and Tor modules" in record_html
    assert "exact simplicial-map score" in record_html
    assert "adjacent syzygies" in record_html
    assert "<details" in record_html
    assert list((out / "macaulay2").glob("*.m2"))
    assert list((out / "ph_features").glob("*_ph_features.json"))
    assert list((out / "ph_features").glob("*_ph_features.npz"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["mean_macaulay2_d_squared_zero"] == 1.0
    assert summary["mean_macaulay2_identity_cone_acyclic"] == 1.0
    assert summary["config"]["ph_feature_visualizations"] is True
    assert summary["ph_feature_artifacts"][0]["json"].startswith("ph_features/")
    feature_json = json.loads(next((out / "ph_features").glob("*_ph_features.json")).read_text(encoding="utf-8"))
    assert feature_json["provenance"] == "exact_gudhi_vectorizers"
    assert "landscape" in feature_json["vectorizations"]["1"]
    assert feature_json["ph_retrieval_signature"]["provenance"] == "exact_gudhi_vectorizers_and_exact_simplicial_map_audits"
    assert feature_json["ph_retrieval_candidates"]
    record_json = json.loads(next((out / "records").glob("*.json")).read_text(encoding="utf-8"))
    assert record_json["ph_retrieval_candidates"]
    assert "exact_internal_simplicial_map_score" in record_json["ph_retrieval_candidates"][0]
    assert record_json["staircase_visualization"]["schema"] == "toricgt.miller_sturmfels_staircase_visualization.v1"
    assert "quotient_lattice_points" in record_json["staircase_visualization"]
    validation = subprocess.run(
        [
            sys.executable,
            "scripts/validate_analysis_exactness.py",
            str(out),
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert validation.returncode == 0, validation.stdout
    validation_payload = json.loads(validation.stdout)
    assert validation_payload["ok"] is True
    inference_manifest = tmp_path / "inference_output.json"
    inference_manifest.write_text(
        json.dumps(
            {
                "mode": "tokengt_graph_inference_with_optional_geometry",
                "gudhi_persistence_enabled": True,
                "gudhi_persistence_output_dir": str(out),
                "ph_feature_output_dir": str(out / "ph_features"),
                "ph_feature_artifacts": summary["ph_feature_artifacts"],
                "embedding_cas_sidecar_enabled": False,
                "geometry_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    validation = subprocess.run(
        [
            sys.executable,
            "scripts/validate_analysis_exactness.py",
            str(inference_manifest),
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert validation.returncode == 0, validation.stdout
    assert json.loads(validation.stdout)["kind"] == "inference_manifest"


def test_gudhi_audit_script_can_disable_ph_feature_dashboard(tmp_path: Path) -> None:
    if shutil.which("M2") is None:
        pytest.skip("Macaulay2 is not installed")
    points = tmp_path / "points.json"
    points.write_text(
        json.dumps({"point_clouds": [{"record_id": "audit_square", "points": [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]}]}),
        encoding="utf-8",
    )
    out = tmp_path / "audit"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_gudhi_persistence_audit.py",
            "--points-json",
            str(points),
            "--output-dir",
            str(out),
            "--records",
            "1",
            "--max-points",
            "5",
            "--num-radii",
            "3",
            "--num-levels",
            "3",
            "--landscape-resolution",
            "12",
            "--image-resolution",
            "5",
            "--macaulay2-timeout-seconds",
            "60",
            "--no-emit-ph-feature-visualizations",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    record_html = next((out / "records").glob("*.html")).read_text(encoding="utf-8")
    assert "Visualization disabled by CLI flag" in record_html
    assert list((out / "ph_features").glob("*_ph_features.json"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["ph_feature_visualizations"] is False


def test_exactness_validator_fails_on_missing_ph_features(tmp_path: Path) -> None:
    out = tmp_path / "bad_gudhi"
    out.mkdir()
    (out / "summary.json").write_text(
        json.dumps(
            {
                "backend": "gudhi",
                "records": 1,
                "mean_finite_field_d_squared_zero": 1.0,
                "mean_finite_field_exact_at_c1": 1.0,
                "mean_simplicial_map_valid_fraction": 1.0,
                "mean_macaulay2_d_squared_zero": 1.0,
                "mean_macaulay2_homogeneous_d1": 1.0,
                "mean_macaulay2_homogeneous_d2": 1.0,
                "mean_macaulay2_identity_cone_acyclic": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (out / "records.json").write_text("[]", encoding="utf-8")
    (out / "index.html").write_text("<html></html>", encoding="utf-8")
    validation = subprocess.run(
        [
            sys.executable,
            "scripts/validate_analysis_exactness.py",
            str(out),
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert validation.returncode != 0
    payload = json.loads(validation.stdout)
    assert payload["ok"] is False
    assert any("PH feature" in item for item in payload["errors"])


def test_watcher_index_and_gudhi_args_are_available() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "watch_training_analysis.py"
    spec = importlib.util.spec_from_file_location("watch_training_analysis", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.parse_args(
        [
            "--start-step",
            "0",
            "--target-step",
            "250",
            "--gudhi-persistence-audit",
            "--gudhi-records",
            "2",
        ]
    )
    assert args.gudhi_persistence_audit is True
    assert args.gudhi_records == 2


def test_watcher_gudhi_wandb_payload_has_exact_metric_aliases() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "watch_training_analysis.py"
    spec = importlib.util.spec_from_file_location("watch_training_analysis", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    payload = module.gudhi_wandb_payload(
        {
            "records": 2,
            "mean_points": 5.0,
            "mean_num_simplices": 17.0,
            "mean_betti0": 1.0,
            "mean_betti1": 1.0,
            "mean_h0_landscape_norm": 0.5,
            "mean_h1_landscape_norm": 0.25,
            "mean_h1_persistence_image_norm": 0.125,
            "mean_h1_silhouette_norm": 0.0625,
            "mean_h1_entropy_vector_norm": 0.03125,
            "mean_h1_interval_count": 3.0,
            "mean_h1_total_persistence": 1.5,
            "mean_h1_persistence_entropy": 0.75,
            "mean_two_parameter_commutative_square_residual": 0.0,
            "mean_macaulay2_homogeneous_d1": 1.0,
            "mean_macaulay2_homogeneous_d2": 1.0,
            "mean_macaulay2_d_squared_zero": 1.0,
            "mean_macaulay2_identity_cone_acyclic": 1.0,
            "mean_finite_field_exact_at_c1": 1.0,
            "mean_be_rank_residual_c1": 0.0,
            "mean_simplicial_map_valid_fraction": 1.0,
        },
        step=250,
    )

    assert payload["trainer/step"] == 250.0
    assert payload["metrics_status/gudhi_persistence_audit_available"] == 1.0
    assert payload["gudhi_persistence/mean_h1_landscape_norm"] == 0.25
    assert payload["topology/exact_gudhi/mean_h1_persistence_image_norm"] == 0.125
    assert payload["gudhi_persistence/mean_h1_silhouette_norm"] == 0.0625
    assert payload["persistent_homology/h1/landscape_norm"] == 0.25
    assert payload["persistent_homology/h1/persistence_image_norm"] == 0.125
    assert payload["persistent_homology/h1/silhouette_norm"] == 0.0625
    assert payload["persistent_homology/h1/entropy_vector_norm"] == 0.03125
    assert payload["persistent_homology/h1/interval_count"] == 3.0
    assert payload["persistent_homology/h1/total_persistence"] == 1.5
    assert payload["persistent_homology/h1/persistence_entropy"] == 0.75
    assert payload["bgg_category_o/persistence/two_parameter_square_residual"] == 0.0
    assert payload["bgg_category_o/persistence/macaulay2_d_squared_zero"] == 1.0
    assert payload["bgg_category_o/persistence/macaulay2_identity_cone_acyclic"] == 1.0
    assert payload["topology/exact_gudhi/macaulay2_identity_cone_acyclic"] == 1.0
    assert payload["bgg_category_o/persistence/gf2_exact_at_c1"] == 1.0
    assert payload["bgg_category_o/persistence/be_rank_residual_c1"] == 0.0
    assert payload["topology/exact_gudhi/simplicial_map_valid_fraction"] == 1.0
    assert payload["analysis_control/exact_gudhi/f2_xy_module_available"] == 1.0
    assert payload["metrics_status/exact_ph_all_metrics_present"] == 1.0
    assert payload["metrics_status/exact_cas_all_metrics_present"] == 1.0
    assert payload["metrics_status/bgg_category_o_all_metrics_present"] == 1.0
    missing = module.missing_metric_alert_payload(
        payload,
        {"toric_vector_bundle": module.DEFAULT_REQUIRED_METRIC_GROUPS["toric_vector_bundle"]},
    )
    assert missing["metrics_status/toric_vector_bundle_all_metrics_present"] == 0.0
    assert missing["metrics_status/toric_vector_bundle_missing_metric_count"] > 0.0


def test_inference_cli_exposes_ph_feature_visualization_flags() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "infer_tokengt_with_geometry.py"
    spec = importlib.util.spec_from_file_location("infer_tokengt_with_geometry", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    parser_args = [
        "--checkpoint",
        "dummy.pt",
        "--no-emit-ph-feature-visualizations",
    ]
    old_argv = sys.argv
    try:
        sys.argv = ["infer_tokengt_with_geometry.py", *parser_args]
        args = module.parse_args()
    finally:
        sys.argv = old_argv
    assert args.emit_ph_feature_visualizations is False
