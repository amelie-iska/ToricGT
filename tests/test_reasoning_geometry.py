import math

from toricgt.reasoning_geometry import (
    barycentric_to_cartesian,
    barycentric_weights,
    minmax_normalize,
    prim_mst_stats,
    rbf_interpolate,
    triangle_grid,
)


def test_barycentric_weights_sum_to_one():
    weights = barycentric_weights([0.0, 1.0, 3.0])
    assert math.isclose(sum(weights), 1.0)
    assert all(weight > 0 for weight in weights)


def test_minmax_normalize_can_invert_direction():
    assert minmax_normalize([10.0, 20.0, 30.0]) == [0.0, 0.5, 1.0]
    assert minmax_normalize([10.0, 20.0, 30.0], higher_is_better=False) == [1.0, 0.5, 0.0]
    assert minmax_normalize([5.0, 5.0]) == [0.5, 0.5]


def test_triangle_grid_and_rbf_interpolation_are_finite():
    grid = triangle_grid(resolution=6)
    assert len(grid) == 28
    value = rbf_interpolate((0.25, 0.25), [(0.0, 0.0), (1.0, 0.0)], [0.0, 1.0])
    assert 0.0 <= value <= 1.0


def test_prim_mst_stats_for_unit_square():
    stats = prim_mst_stats([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    assert stats["mst_nodes"] == 4.0
    assert stats["mst_edges"] == 3.0
    assert math.isclose(stats["mst_total_weight"], 3.0)
    assert 0.0 < stats["mst_efficiency"] < 1.0


def test_barycentric_projection_dimension_check():
    point = barycentric_to_cartesian([0.25, 0.25, 0.5], [(0, 0), (1, 0), (0, 1)])
    assert point == (0.25, 0.5)
