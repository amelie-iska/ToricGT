"""Reasoning simplex diagnostics for ToricGT evaluation.

The plots built from this module are diagnostic projections, not new objective
terms.  They map model-computed quantities such as reasoning budget, BPB,
compressor-based K proxies, GFlowNet diversity, and hidden-trajectory MST
geometry into triangle or tetrahedron coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


TRIANGLE_VERTICES: dict[str, tuple[float, float]] = {
    "left": (0.0, 0.0),
    "right": (1.0, 0.0),
    "top": (0.5, math.sqrt(3.0) / 2.0),
}

TETRAHEDRON_VERTICES: dict[str, tuple[float, float, float]] = {
    "a": (0.0, 0.0, 0.0),
    "b": (1.0, 0.0, 0.0),
    "c": (0.5, math.sqrt(3.0) / 2.0, 0.0),
    "d": (0.5, math.sqrt(3.0) / 6.0, math.sqrt(2.0 / 3.0)),
}


def minmax_normalize(values: Sequence[float], higher_is_better: bool = True) -> list[float]:
    """Normalize a metric series to [0, 1], with stable handling of ties."""

    if not values:
        return []
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return [0.5 for _ in values]
    lo = min(finite)
    hi = max(finite)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    out = []
    for value in values:
        val = float(value)
        if not math.isfinite(val):
            norm = 0.5
        else:
            norm = (val - lo) / (hi - lo)
        out.append(norm if higher_is_better else 1.0 - norm)
    return out


def barycentric_weights(scores: Sequence[float], floor: float = 0.08) -> list[float]:
    """Convert nonnegative scores into simplex weights that sum to one."""

    if not scores:
        raise ValueError("scores must be non-empty")
    cleaned = [max(float(floor), float(score)) if math.isfinite(float(score)) else float(floor) for score in scores]
    total = sum(cleaned)
    if total <= 0:
        return [1.0 / len(cleaned) for _ in cleaned]
    return [value / total for value in cleaned]


def barycentric_to_cartesian(
    weights: Sequence[float],
    vertices: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Project barycentric weights to 2D/3D Cartesian coordinates."""

    if len(weights) != len(vertices):
        raise ValueError("weights and vertices must have the same length")
    dim = len(vertices[0])
    if any(len(vertex) != dim for vertex in vertices):
        raise ValueError("all vertices must have the same dimension")
    coords = [0.0 for _ in range(dim)]
    for weight, vertex in zip(weights, vertices):
        for index, value in enumerate(vertex):
            coords[index] += float(weight) * float(value)
    return tuple(coords)


def triangle_grid(resolution: int = 90) -> list[tuple[float, float, tuple[float, float, float]]]:
    """Return Cartesian grid points inside the unit triangle and barycentrics."""

    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    vertices = [TRIANGLE_VERTICES["left"], TRIANGLE_VERTICES["right"], TRIANGLE_VERTICES["top"]]
    points = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            w_left = i / resolution
            w_right = j / resolution
            w_top = 1.0 - w_left - w_right
            x, y = barycentric_to_cartesian((w_left, w_right, w_top), vertices)
            points.append((x, y, (w_left, w_right, w_top)))
    return points


def rbf_interpolate(
    query_xy: tuple[float, float],
    points_xy: Sequence[tuple[float, float]],
    values: Sequence[float],
    bandwidth: float = 0.22,
) -> float:
    """Smoothly interpolate sparse simplex observations for heatmaps."""

    if len(points_xy) != len(values):
        raise ValueError("points and values must have the same length")
    if not points_xy:
        return 0.0
    bw2 = max(float(bandwidth) ** 2, 1e-8)
    numerator = 0.0
    denominator = 0.0
    qx, qy = query_xy
    for (px, py), value in zip(points_xy, values):
        dist2 = (qx - px) ** 2 + (qy - py) ** 2
        weight = math.exp(-0.5 * dist2 / bw2)
        numerator += weight * float(value)
        denominator += weight
    return numerator / max(denominator, 1e-12)


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def prim_mst_stats(points: Sequence[Sequence[float]]) -> dict[str, float]:
    """Compute MST length statistics for a complete graph over trajectory points."""

    n = len(points)
    if n <= 1:
        return {
            "mst_nodes": float(n),
            "mst_edges": 0.0,
            "mst_total_weight": 0.0,
            "mst_mean_edge_weight": 0.0,
            "mst_max_edge_weight": 0.0,
            "mst_efficiency": 1.0,
        }
    selected = [False] * n
    selected[0] = True
    min_dist = [_euclidean(points[0], points[index]) for index in range(n)]
    edge_weights: list[float] = []
    for _ in range(n - 1):
        best_index = -1
        best_dist = float("inf")
        for index in range(n):
            if not selected[index] and min_dist[index] < best_dist:
                best_index = index
                best_dist = min_dist[index]
        if best_index < 0:
            break
        selected[best_index] = True
        edge_weights.append(float(best_dist))
        for index in range(n):
            if not selected[index]:
                min_dist[index] = min(min_dist[index], _euclidean(points[best_index], points[index]))
    total = sum(edge_weights)
    mean_edge = total / max(1, len(edge_weights))
    return {
        "mst_nodes": float(n),
        "mst_edges": float(len(edge_weights)),
        "mst_total_weight": float(total),
        "mst_mean_edge_weight": float(mean_edge),
        "mst_max_edge_weight": float(max(edge_weights) if edge_weights else 0.0),
        "mst_efficiency": float(1.0 / (1.0 + mean_edge)),
    }


def attach_normalized_scores(
    records: list[dict[str, Any]],
    metric_specs: Mapping[str, tuple[str, bool]],
) -> list[dict[str, Any]]:
    """Attach normalized score fields to metric records.

    ``metric_specs`` maps an output score name to ``(record_key,
    higher_is_better)``.
    """

    for score_name, (record_key, higher_is_better) in metric_specs.items():
        normalized = minmax_normalize(
            [float(record.get(record_key, 0.0)) for record in records],
            higher_is_better=higher_is_better,
        )
        for record, value in zip(records, normalized):
            record[score_name] = float(value)
    return records


def simplex_record(
    record: Mapping[str, Any],
    score_keys: Sequence[str],
    labels: Sequence[str],
    floor: float = 0.08,
) -> dict[str, Any]:
    """Build a barycentric record from named normalized scores."""

    weights = barycentric_weights([float(record.get(key, 0.0)) for key in score_keys], floor=floor)
    return {
        "labels": list(labels),
        "score_keys": list(score_keys),
        "weights": weights,
    }


def mean_dict(records: Iterable[Mapping[str, float]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.items():
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(val):
                buckets.setdefault(key, []).append(val)
    return {key: sum(values) / len(values) for key, values in buckets.items() if values}
