from __future__ import annotations

import numpy as np
import pytest

from scripts import evaluate_tokengt_reasoning_geometry_suite as geometry
from toricgt.branching_reasoning_visualization import _simplicial_map_payload
from toricgt.gudhi_persistence import build_rips_simplex_tree, simplex_map_audit
from toricgt.topological_reasoning import _build_flag_complex_f2, _chain_map_matrices


SOURCE = np.array([[0.8], [0.9], [1.1]])
RADIUS = 0.5
VERTEX_MAP = {0: 0, 1: 0, 2: 1}


@pytest.fixture(params=[
    ([[0.0], [2.0]], False),
    ([[0.8], [1.2], [5.0]], True),
], ids=["missing-target-edge", "present-target-edge"])
def target_edge_case(request):
    points, has_edge = request.param
    return np.array(points), has_edge


def test_branching_simplex_map_checks_collapsed_triangle_image(target_edge_case) -> None:
    target, has_edge = target_edge_case
    audit = _simplicial_map_payload(SOURCE, target, {}, {}, RADIUS)

    assert audit["vertex_map"] == [0, 0, 1]
    assert audit["triangle_images"] == [[0, 1, 2, 0, 1, -1, "c" if has_edge else "i"]]
    assert audit["collapsed_triangles"] == int(has_edge)
    assert audit["invalid_triangles"] == int(not has_edge)
    assert audit["valid_fraction"] == (1.0 if has_edge else 0.25)


def test_full_trajectory_simplex_map_checks_collapsed_triangle_image(
    target_edge_case, monkeypatch,
) -> None:
    target, has_edge = target_edge_case
    monkeypatch.setattr(geometry, "_radius_schedule_from_pair", lambda *args, **kwargs: [RADIUS])
    audit = geometry._full_trajectory_simplicial_map_audit(SOURCE, target, levels=1)
    row, = audit["level_radius_rows"]

    assert row["source_triangles"] == 1
    assert row["target_edges"] == int(has_edge)
    assert row["edge_valid_fraction"] == (1.0 if has_edge else 1.0 / 3.0)
    assert row["triangle_valid_fraction"] == float(has_edge)


def test_flag_chain_map_checks_collapsed_triangle_image(target_edge_case) -> None:
    target, has_edge = target_edge_case

    def flag_complex(points):
        return _build_flag_complex_f2(np.abs(points - points.T) <= RADIUS)

    _, f1, edge_validity, triangle_validity = _chain_map_matrices(
        flag_complex(SOURCE), flag_complex(target), np.array(list(VERTEX_MAP.values())),
    )

    assert edge_validity == (1.0 if has_edge else 1.0 / 3.0)
    assert triangle_validity == float(has_edge)
    if has_edge:
        np.testing.assert_array_equal(f1, [[0, 1, 1]])


def test_gudhi_audit_counts_only_collapses_to_existing_faces(target_edge_case) -> None:
    target, has_edge = target_edge_case
    source_tree = build_rips_simplex_tree(SOURCE, RADIUS)
    target_tree = build_rips_simplex_tree(target, RADIUS)
    audit = simplex_map_audit(source_tree, target_tree, VERTEX_MAP)

    assert audit["simplicial_map_valid"] is has_edge
    assert audit["dim2_valid"] == int(has_edge)
    assert audit["dim2_collapsed"] == int(has_edge)


def test_gudhi_audit_rejects_missing_target_vertex_and_its_collapsed_images() -> None:
    source_tree = build_rips_simplex_tree(SOURCE, RADIUS)
    target_tree = build_rips_simplex_tree(np.array([[0.0]]), RADIUS)
    audit = simplex_map_audit(source_tree, target_tree, {vertex: 5 for vertex in VERTEX_MAP})

    assert audit["simplicial_map_valid"] is False
    assert audit["simplicial_map_valid_fraction"] == 0.0
    assert audit["dim0_valid_fraction"] == 0.0
    assert audit["dim1_collapsed"] == 0
    assert audit["dim2_collapsed"] == 0
    assert simplex_map_audit(source_tree, target_tree, max_dimension=0)["simplicial_map_valid"] is False
