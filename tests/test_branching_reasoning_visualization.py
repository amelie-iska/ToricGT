from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from toricgt.branching_reasoning_visualization import (
    BranchingTrajectoryConfig,
    _long_branch_dag,
    build_branching_reasoning_payload,
    build_branching_reasoning_payload_from_embedding_payload,
    render_branching_reasoning_report,
)


def test_branching_reasoning_payload_has_required_topological_contract() -> None:
    assert BranchingTrajectoryConfig().analogy_step_threshold <= 0.45
    payload = build_branching_reasoning_payload(
        BranchingTrajectoryConfig(seed=321, embedding_dim=16, radius_levels=5)
    )

    assert payload["source_mode"] == "synthetic_long_branching_fixture"
    assert payload["source_metadata"]["trajectory_levels"] >= 18
    assert payload["source_metadata"]["branch_lanes"] >= 8
    assert payload["embedding_comparison_space"] == "original_high_dimensional_embeddings"
    assert payload["pca_display_only"] is True
    assert len(payload["nodes"]) >= 20
    assert any(node["is_branch"] for node in payload["nodes"])
    assert any(node["is_merge"] for node in payload["nodes"])
    assert payload["controls"]["full"] == ["radius_slider", "reasoning_level_slider"]
    assert payload["controls"]["step"] == ["step_radius_slider", "decoding_order_slider"]
    assert payload["controls"]["analogy"] == ["analogy_radius_slider", "analogy_reasoning_level_slider"]
    assert payload["distance_storage"] == "edge_births"
    assert payload["distances"] == []
    assert payload["edge_births"]
    assert payload["triangle_count_exact"] >= len(payload["triangles"])
    assert payload["steps"][0]["distance_storage"] == "edge_births"
    assert payload["steps"][0]["distances"] == []
    assert payload["steps"][0]["edge_births"]
    assert payload["steps"][0]["triangle_count_exact"] >= len(payload["steps"][0]["triangles"])
    analogy = payload["analogy"]
    assert payload["visible_edge_policy"] == "all_visible_edges"
    assert payload["distance_storage"] == "edge_births"
    assert payload["distances"] == []
    assert payload["edge_births"]
    assert "full_reasoning_trajectory_simplex_tree_map" in analogy
    assert "source_simplex_tree" in analogy
    assert "memory_simplex_tree" in analogy
    assert analogy["source_simplex_tree"]["edge_storage"] == "compact_edge_births"
    assert analogy["source_simplex_tree"]["triangle_storage"] == "compact_triangle_births"
    assert analogy["source_simplex_tree"]["edge_births"]
    assert analogy["source_simplex_tree"]["edges"] == []
    assert "candidate_map" in analogy
    assert analogy["source_simplex_tree"]["simplex_tree_backend"] == "gudhi.RipsComplex.create_simplex_tree"
    assert analogy["source_simplex_tree"]["visible_edge_policy"] == "all_visible_edges"
    assert analogy["source_simplex_tree"]["rendered_edge_count"] == analogy["source_simplex_tree"]["num_edges"]
    assert analogy["memory_simplex_tree"]["num_vertices"] == analogy["source_simplex_tree"]["num_vertices"]
    assert analogy["candidate_map"]["map_image_storage"] == "compact_arrays"
    assert "edge_images" in analogy["candidate_map"]
    assert "triangle_images" in analogy["candidate_map"]
    assert isinstance(analogy["candidate_map"]["edge_images"][0], list)
    assert "vectorized_persistence_comparisons" in analogy
    assert "vectorized_persistence_feature_family_summary" in analogy
    assert "persistence_diagram_distances" in analogy
    assert "by_dimension" in analogy["persistence_diagram_distances"]
    assert "bottleneck_distance" in analogy["persistence_diagram_distances"]["by_dimension"]["1"]
    assert analogy["analogy_status"] in {
        "strong_analogy",
        "weak_analogy",
        "candidate_analogy",
        "no_analogy",
    }
    assert "analogy_confidence_score" in analogy
    assert "map_confidence_summary" in analogy
    assert analogy["candidate_map"]["map_confidence_summary"] == analogy["map_confidence_summary"]
    assert 0.0 <= analogy["map_confidence_summary"]["combined_map_confidence"] <= 1.0
    assert "vertex_distance_quality_mean" in analogy["map_confidence_summary"]
    assert "analogy_passed_checks" in analogy
    assert "decision_summary" in analogy
    assert "narrative" in analogy["decision_summary"]
    assert {"strong", "weak", "candidate", "scoreboard"}.issubset(analogy["decision_summary"])
    assert any(
        item["label"].startswith("step simplex-map")
        for item in analogy["decision_summary"]["strong"]["conditions"]
    )
    assert any(
        item["role"] == "advisory" and item["required"] is False
        for item in analogy["decision_summary"]["strong"]["conditions"]
    )
    assert "ph_gate_threshold" in analogy["decision_rule"]["candidate"]
    assert analogy["decision_rule"]["strong"]["step_simplicial_map_is_advisory"] is True
    assert analogy["decision_rule"]["strong"]["step_simplicial_map_advisory_target"] <= 0.45
    assert analogy["decision_rule"]["strong"]["step_simplicial_map_minimum_threshold"] <= 0.15
    assert analogy["decision_rule"]["candidate"]["full_reasoning_trajectory_simplex_map_threshold"] <= 0.35
    assert analogy["decision_rule"]["candidate"]["vectorized_persistence_similarity_threshold"] <= 0.05
    assert len(analogy["candidate_map"]["vertex_map_distances"]) == len(analogy["candidate_map"]["vertex_map"])
    assert "ph_features" in analogy
    for side in ("source", "memory"):
        rendered = analogy["ph_features"][side]["rendered"]
        for dim in ("0", "1", "2"):
            assert "diagram" in rendered[dim]
            assert "landscape" in rendered[dim]
            assert "persistence_image" in rendered[dim]
            assert "silhouette" in rendered[dim]
            assert "entropy_vector" in rendered[dim]
            assert "betti_curve" in rendered[dim]
            assert "lifetime_histogram" in rendered[dim]
    assert {"landscape", "persistence_image", "silhouette", "entropy_vector"}.issubset(
        analogy["vectorized_persistence_comparisons"]
    )
    family_summary = analogy["vectorized_persistence_feature_family_summary"]
    assert family_summary["backend"] == "GUDHI vectorized_diagram_metrics"
    assert {"landscape", "persistence_image", "silhouette", "entropy_vector"}.issubset(
        family_summary["family_means"]
    )
    assert {"H0", "H1", "H2"}.issubset(family_summary["dimension_means"])
    assert len(family_summary["rows"]) == 12
    assert all("cosine_similarity" in row and "source_norm" in row and "memory_norm" in row for row in family_summary["rows"])
    assert "gflownet_flow" in payload


def test_strong_analogy_does_not_require_advisory_step_target() -> None:
    payload = build_branching_reasoning_payload(
        BranchingTrajectoryConfig(
            seed=123,
            embedding_dim=12,
            trajectory_levels=9,
            branch_lanes=4,
            side_branch_length=3,
            token_count_min=5,
            token_count_max=6,
            radius_levels=5,
            ph_landscape_resolution=16,
            ph_image_resolution=6,
            analogy_map_threshold=0.0,
            analogy_ph_threshold=0.0,
            analogy_step_threshold=0.99,
            analogy_strong_step_minimum=0.0,
        )
    )
    analogy = payload["analogy"]
    assert analogy["analogy_status"] == "strong_analogy"
    assert analogy["decision_summary"]["strong"]["passed"] is True
    assert analogy["decision_summary"]["strong"]["advisory_warning_count"] >= 1
    advisory_rows = [
        item for item in analogy["decision_summary"]["strong"]["conditions"] if item["role"] == "advisory"
    ]
    assert advisory_rows and advisory_rows[0]["passed"] is False


def test_long_branching_reasoning_payload_has_token_metadata() -> None:
    payload = build_branching_reasoning_payload(
        BranchingTrajectoryConfig(
            seed=444,
            embedding_dim=12,
            trajectory_levels=13,
            branch_lanes=6,
            token_count_min=5,
            token_count_max=7,
            radius_levels=5,
            ph_landscape_resolution=16,
            ph_image_resolution=6,
        )
    )

    assert len(payload["nodes"]) >= 90
    assert sum(1 for node in payload["nodes"] if node["is_branch"]) >= 10
    assert sum(1 for node in payload["nodes"] if node["is_merge"]) >= 10
    token = payload["steps"][0]["tokens"][0]
    for key in [
        "global_token_id",
        "text",
        "token_type",
        "source_step",
        "source_step_label",
        "decode_order",
        "nll",
        "logprob",
        "entropy",
        "rank",
    ]:
        assert key in token
    for step in payload["steps"][:5]:
        for token in step["tokens"][:3]:
            assert isinstance(token["text"], str) and token["text"]
            assert isinstance(token["token_type"], str) and token["token_type"]
            assert int(token["source_step"]) == int(step["step_index"])
            assert int(token["decode_order"]) >= 0
            assert np.isfinite(float(token["nll"]))
            assert np.isfinite(float(token["logprob"]))
            assert np.isfinite(float(token["entropy"]))


def test_larger_branching_fixture_exceeds_previous_report_scale() -> None:
    cfg = BranchingTrajectoryConfig(
        seed=555,
        embedding_dim=16,
        trajectory_levels=28,
        branch_lanes=12,
        side_branch_length=9,
        token_count_min=4,
        token_count_max=6,
        radius_levels=5,
        ph_landscape_resolution=16,
        ph_image_resolution=6,
        max_render_triangles=600,
        max_map_image_records=600,
    )
    levels, edges = _long_branch_dag(cfg)
    nodes = sum(len(level) for level in levels)

    assert nodes > 492
    assert len(edges) > 580


def test_embedding_payload_builder_uses_saved_hidden_vectors(tmp_path: Path) -> None:
    rng = np.random.default_rng(14)
    hidden = rng.normal(size=(10, 12)).astype(np.float32)
    hidden[1] += hidden[0] + 0.3
    hidden[2] += hidden[0] - 0.2
    nll = np.linspace(1.2, 2.4, num=10, dtype=np.float32)
    edges = np.asarray(
        [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [3, 4],
            [3, 5],
            [4, 6],
            [5, 6],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 9],
        ],
        dtype=np.int64,
    )
    npz = tmp_path / "record_000_embedding_payload.npz"
    np.savez_compressed(
        npz,
        hidden=hidden,
        projected=hidden[:, :3],
        nll=nll,
        energy=nll,
        edges=edges,
        complex_projected=hidden[:, :3],
        complex_nll=nll,
        complex_mass=np.ones((10,), dtype=np.float32),
        complex_edges=edges,
    )
    meta = tmp_path / "record_000_embedding_payload.json"
    meta.write_text(
        json.dumps(
            {
                "schema": "toricgt.embedding_payload.v1",
                "record_index": 0,
                "checkpoint_step": 42,
                "nll_source": "unit_test",
            }
        ),
        encoding="utf-8",
    )

    payload = build_branching_reasoning_payload_from_embedding_payload(
        npz,
        metadata_json_path=meta,
        cfg=BranchingTrajectoryConfig(
            seed=77,
            embedding_dim=12,
            token_count_min=4,
            token_count_max=6,
            radius_levels=5,
            ph_landscape_resolution=16,
            ph_image_resolution=6,
        ),
        embedding_max_nodes=6,
        embedding_node_offset=2,
    )

    assert payload["source_mode"] == "checkpoint_embedding_payload"
    assert payload["source_metadata"]["embedding_payload_schema"] == "toricgt.embedding_payload.v1"
    assert payload["source_metadata"]["checkpoint_step"] == 42
    assert payload["source_metadata"]["original_node_count"] == 10
    assert payload["source_metadata"]["selected_node_count"] == 6
    assert payload["source_metadata"]["selected_node_offset"] == 2
    assert payload["source_metadata"]["node_window_applied"] is True
    assert payload["source_metadata"]["visible_edge_policy_after_window"] == "all_visible_edges"
    assert payload["nodes"][0]["embedding"] == hidden[2].astype(float).round(6).tolist()
    assert payload["embedding_comparison_space"] == "original_high_dimensional_embeddings"
    assert payload["pca_display_only"] is True
    assert len(payload["nodes"]) == 6
    assert any(node["is_branch"] for node in payload["nodes"])
    assert any(node["is_merge"] for node in payload["nodes"])
    first_token = payload["steps"][0]["tokens"][0]
    assert first_token["token_type"] == "self"
    assert first_token["global_token_id"] == 0
    assert "source_simplex_tree" in payload["analogy"]
    assert payload["analogy"]["source_simplex_tree"]["visible_edge_policy"] == "all_visible_edges"

    manifest = render_branching_reasoning_report(payload, tmp_path / "rendered")
    assert manifest["source_mode"] == "checkpoint_embedding_payload"
    html = (tmp_path / "rendered" / "branching_reasoning_trajectory.html").read_text(encoding="utf-8")
    assert "checkpoint_embedding_payload" in html
    assert "Source mode" in html


def test_branching_reasoning_report_cli_accepts_embedding_payload_window(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    hidden = rng.normal(size=(12, 10)).astype(np.float32)
    nll = np.linspace(1.0, 2.2, num=12, dtype=np.float32)
    edges = np.asarray([[idx, idx + 1] for idx in range(11)] + [[2, 5], [3, 5], [5, 8]], dtype=np.int64)
    npz = tmp_path / "record_000_embedding_payload.npz"
    np.savez_compressed(npz, hidden=hidden, projected=hidden[:, :3], nll=nll, energy=nll, edges=edges)
    meta = tmp_path / "record_000_embedding_payload.json"
    meta.write_text(json.dumps({"schema": "toricgt.embedding_payload.v1", "checkpoint_step": 99}), encoding="utf-8")
    out = tmp_path / "cli_embedding_report"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/render_branching_reasoning_trajectory_report.py",
            "--output-dir",
            str(out),
            "--embedding-payload-npz",
            str(npz),
            "--embedding-payload-json",
            str(meta),
            "--embedding-max-nodes",
            "5",
            "--embedding-node-offset",
            "3",
            "--radius-levels",
            "5",
            "--ph-landscape-resolution",
            "16",
            "--ph-image-resolution",
            "6",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    payload = json.loads((out / "branching_reasoning_payload.json").read_text(encoding="utf-8"))
    assert manifest["source_mode"] == "checkpoint_embedding_payload"
    assert payload["source_metadata"]["selected_node_count"] == 5
    assert payload["source_metadata"]["selected_node_offset"] == 3
    assert payload["visible_edge_policy"] == "all_visible_edges"


def test_branching_reasoning_report_contains_dual_sliders_and_payloads(tmp_path: Path) -> None:
    payload = build_branching_reasoning_payload(
        BranchingTrajectoryConfig(seed=111, embedding_dim=16, radius_levels=5)
    )
    manifest = render_branching_reasoning_report(payload, tmp_path / "report")

    assert manifest["nodes"] >= 20
    html = (tmp_path / "report" / "branching_reasoning_trajectory.html").read_text(encoding="utf-8")
    required = [
        "trajectory_simplex_payload",
        "simplex_step_payload",
        "distance_storage",
        "edge_births",
        "edgeBirthPairs",
        "stepSimplicialEdges",
        "compact_edge_births",
        "compact_triangle_births",
        "compact_arrays",
        "mapImageSampleHTML",
        "step simplex-map advisory target",
        "radius_slider",
        "reasoning_level_slider",
        "step_radius_slider",
        "decoding_order_slider",
        "analogy_radius_slider",
        "analogy_reasoning_level_slider",
        "dotted_decode_edges",
        "half_arrow_payload",
        "gflownet_flow",
        "source_simplex_tree_payload",
        "memory_simplex_tree_payload",
        "strong_analogy",
        "weak_analogy",
        "candidate_analogy",
        "No analogy emitted",
        "full_triangle_toggle",
        "step_triangle_toggle",
        "analogy_triangle_toggle",
        "full_label_toggle",
        "step_label_toggle",
        "analogy_label_toggle",
        "full_label_state",
        "step_label_state",
        "analogy_label_state",
        "trajectory_state_caption",
        "step_state_caption",
        "analogy_state_caption",
        "analogy_decision_status_badges",
        "top_analogy_decision_status_badges",
        "top_analogy_threshold_table",
        "raw decision payload",
        "decisionThresholdTableHTML",
        "decisionStatusBadgesHTML",
        "status-badge",
        "Compact simplex-map summary",
        "Vectorized PH evidence near the map",
        "strong threshold",
        "weak threshold",
        "candidate threshold",
        "PH gate",
        "faces hidden",
        "labels hidden",
        "toricgtScreenshotAudit",
        "step_simplex_tree_table",
        "all visible one-dimensional simplices",
        "simplicial_map_validity_table",
        "analogy_vertex_map_table",
        "analogy_detail_panel",
        "ph_similarity_bar_plot",
        "ph_distance_table",
        "ph_difference_plot",
        "bottleneck_distance",
        "full_reasoning_trajectory_simplex_tree_map",
        "candidate nearest-neighbor simplex-tree map arrows",
        "source graph-of-thought branch/merge skeleton",
        "memory graph-of-thought branch/merge skeleton",
        "radius simplex-tree one-dimensional edges",
        "vectorized_persistence_comparisons",
        "persistence_diagram_plot",
        "persistence_landscape_plot",
        "persistence_image_heatmap",
        "silhouette_vector_plot",
        "entropy_vector_plot",
        "betti_curve_plot",
        "lifetime_histogram_plot",
        "graph-of-thought DAG branch/merge edges",
        "node NLL",
        "token NLL",
        "node_detail_panel",
        "token_detail_panel",
        "token_type",
        "logprob",
        "entropy",
        "rank",
    ]
    for needle in required:
        assert needle in html
    assert "Vectroized" not in html
    assert "analogy_decision_chips" not in html
    assert "decisionChipsHTML" not in html
    assert "chiprow" not in html


def test_branching_reasoning_report_cli(tmp_path: Path) -> None:
    out = tmp_path / "cli_report"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/render_branching_reasoning_trajectory_report.py",
            "--output-dir",
            str(out),
            "--embedding-dim",
            "16",
            "--radius-levels",
            "5",
            "--trajectory-levels",
            "9",
            "--branch-lanes",
            "4",
            "--side-branch-length",
            "3",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "toricgt.branching_reasoning_visual_report.v1"
    assert manifest["source_mode"] == "synthetic_long_branching_fixture"
    assert (out / "index.html").exists()
    assert (out / "branching_reasoning_trajectory.html").exists()


def test_branching_payload_static_screenshot_cli(tmp_path: Path) -> None:
    payload = build_branching_reasoning_payload(
        BranchingTrajectoryConfig(
            seed=551,
            embedding_dim=12,
            trajectory_levels=9,
            branch_lanes=4,
            side_branch_length=3,
            radius_levels=5,
            ph_landscape_resolution=8,
            ph_image_resolution=5,
            max_render_triangles=60,
            max_map_image_records=60,
        )
    )
    payload_path = tmp_path / "branching_reasoning_payload.json"
    payload_path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    out = tmp_path / "static_screenshots"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/render_branching_payload_static_screenshots.py",
            "--payload-json",
            str(payload_path),
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
    assert manifest["schema"] == "toricgt.branching_payload_static_screenshots.v1"
    assert manifest["nodes"] == len(payload["nodes"])
    assert manifest["selected_step_id"] >= 0
    for name in [
        "summary.png",
        "full_trajectory_filtered_complex.png",
        "selected_step_simplex_tree_tokens.png",
        "analogical_memory_simplex_tree_map.png",
        "vectorized_ph_features.png",
        "contact_sheet.png",
        "index.html",
    ]:
        assert (out / name).exists()
