"""Exact finite tropical hypersurface certificates for toric sidecars.

The routines here use integer lattice arithmetic for the finite tropical
polynomial

    psi(u) = max_i <a_i, u> + b_i

that is exported from tropical ring attention probes.  For two-dimensional
exponent supports, codimension-one tropical facets are pairwise active walls
whose lattice multiplicity is gcd(a_i-a_j).  Multiway active vertices are
audited by the dual Newton polygon: the weighted primitive edge normals sum to
zero exactly for a balanced local tropical cycle.

No floating surrogate is labeled exact.  Higher-dimensional supports are
reported as unsupported by this closed-form certificate rather than
approximated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Any

from .cas_certificates import exact_closed_form_certificate


@dataclass(frozen=True)
class _Interval:
    lo: Fraction | None = None
    hi: Fraction | None = None

    def intersect_ge(self, alpha: Fraction, beta: Fraction) -> "_Interval | None":
        """Intersect with alpha * t + beta >= 0."""

        if alpha == 0:
            return self if beta >= 0 else None
        threshold = -beta / alpha
        lo = self.lo
        hi = self.hi
        if alpha > 0:
            lo = threshold if lo is None else max(lo, threshold)
        else:
            hi = threshold if hi is None else min(hi, threshold)
        if lo is not None and hi is not None and lo > hi:
            return None
        return _Interval(lo, hi)

    def to_json(self) -> dict[str, str | None]:
        return {
            "lower": str(self.lo) if self.lo is not None else None,
            "upper": str(self.hi) if self.hi is not None else None,
            "kind": "bounded" if self.lo is not None and self.hi is not None else "ray_or_line",
        }


def _as_int_matrix(exponents: list[list[int | float]]) -> list[list[int]]:
    rows = [[int(value) for value in row] for row in exponents]
    if len(rows) < 3:
        raise ValueError("at least three exponent points are required")
    width = len(rows[0])
    if width != 2:
        raise ValueError("closed-form tropical hypersurface certificate currently requires exponent_dim == 2")
    if any(len(row) != width for row in rows):
        raise ValueError("all exponent rows must have the same width")
    return rows


def _as_int_biases(biases: list[int | float] | None, n: int) -> list[int]:
    if biases is None:
        return [0 for _ in range(n)]
    if len(biases) != n:
        raise ValueError("bias length must match exponent count")
    return [int(value) for value in biases]


def _primitive(vec: tuple[int, int]) -> tuple[list[int], int]:
    g = gcd(abs(int(vec[0])), abs(int(vec[1])))
    if g == 0:
        return [0, 0], 0
    return [int(vec[0] // g), int(vec[1] // g)], int(g)


def _line_point_and_direction(normal: tuple[int, int], constant: int) -> tuple[tuple[Fraction, Fraction], tuple[int, int]]:
    """Return p,d with normal . p + constant = 0 and normal . d = 0."""

    nx, ny = normal
    if nx == 0 and ny == 0:
        raise ValueError("duplicate exponent difference has zero normal")
    if nx != 0:
        point = (Fraction(-constant, nx), Fraction(0))
    else:
        point = (Fraction(0), Fraction(-constant, ny))
    direction = (-ny, nx)
    primitive_direction, _ = _primitive(direction)
    return point, (primitive_direction[0], primitive_direction[1])


def _wall_interval(
    exponents: list[list[int]],
    biases: list[int],
    i: int,
    j: int,
) -> _Interval | None:
    ai = exponents[i]
    aj = exponents[j]
    normal = (ai[0] - aj[0], ai[1] - aj[1])
    constant = biases[i] - biases[j]
    point, direction = _line_point_and_direction(normal, constant)
    interval = _Interval()
    for k, ak in enumerate(exponents):
        coeff = (ai[0] - ak[0], ai[1] - ak[1])
        beta = Fraction(coeff[0]) * point[0] + Fraction(coeff[1]) * point[1] + Fraction(biases[i] - biases[k])
        alpha = Fraction(coeff[0] * direction[0] + coeff[1] * direction[1])
        interval = interval.intersect_ge(alpha, beta)
        if interval is None:
            return None
    return interval


def _solve_triple_vertex(
    exponents: list[list[int]],
    biases: list[int],
    active: tuple[int, int, int],
) -> tuple[Fraction, Fraction] | None:
    i, j, k = active
    ai, aj, ak = exponents[i], exponents[j], exponents[k]
    c1 = biases[i] - biases[j]
    c2 = biases[i] - biases[k]
    a = ai[0] - aj[0]
    b = ai[1] - aj[1]
    c = ai[0] - ak[0]
    d = ai[1] - ak[1]
    det = a * d - b * c
    if det == 0:
        return None
    # a*x + b*y = -c1, c*x + d*y = -c2
    x = Fraction((-c1) * d - b * (-c2), det)
    y = Fraction(a * (-c2) - (-c1) * c, det)
    value = Fraction(ai[0]) * x + Fraction(ai[1]) * y + biases[i]
    for idx, row in enumerate(exponents):
        other = Fraction(row[0]) * x + Fraction(row[1]) * y + biases[idx]
        if other > value:
            return None
    return x, y


def _convex_hull_indices(points: list[list[int]], indices: list[int]) -> list[int]:
    """Return convex hull indices in CCW order for a small set of 2D points."""

    unique = sorted(set(indices), key=lambda idx: (points[idx][0], points[idx][1], idx))
    if len(unique) <= 2:
        return unique

    def cross(o: int, a: int, b: int) -> int:
        ox, oy = points[o]
        ax, ay = points[a]
        bx, by = points[b]
        return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)

    lower: list[int] = []
    for idx in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], idx) <= 0:
            lower.pop()
        lower.append(idx)
    upper: list[int] = []
    for idx in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], idx) <= 0:
            upper.pop()
        upper.append(idx)
    return lower[:-1] + upper[:-1]


def _active_indices_at_vertex(
    exponents: list[list[int]],
    biases: list[int],
    vertex: tuple[Fraction, Fraction],
) -> list[int]:
    values = [Fraction(row[0]) * vertex[0] + Fraction(row[1]) * vertex[1] + biases[idx] for idx, row in enumerate(exponents)]
    max_value = max(values)
    return [idx for idx, value in enumerate(values) if value == max_value]


def tropical_hypersurface_certificate(
    exponents: list[list[int | float]],
    biases: list[int | float] | None = None,
) -> dict[str, Any]:
    """Return an exact closed-form certificate for a 2D tropical hypersurface."""

    points = _as_int_matrix(exponents)
    coeffs = _as_int_biases(biases, len(points))
    facets: list[dict[str, Any]] = []
    facet_by_pair: dict[tuple[int, int], int] = {}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            normal = (points[i][0] - points[j][0], points[i][1] - points[j][1])
            primitive_normal, multiplicity = _primitive(normal)
            if multiplicity == 0:
                continue
            interval = _wall_interval(points, coeffs, i, j)
            if interval is None:
                continue
            point, direction = _line_point_and_direction(normal, coeffs[i] - coeffs[j])
            facet_index = len(facets)
            facet_by_pair[(i, j)] = facet_index
            facets.append(
                {
                    "facet_index": facet_index,
                    "active_pair": [i, j],
                    "normal": [normal[0], normal[1]],
                    "primitive_normal": primitive_normal,
                    "multiplicity": multiplicity,
                    "wall_constant": int(coeffs[i] - coeffs[j]),
                    "line_point": [str(point[0]), str(point[1])],
                    "primitive_direction": [int(direction[0]), int(direction[1])],
                    "parameter_interval": interval.to_json(),
                }
            )

    vertices: dict[tuple[str, str], dict[str, Any]] = {}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            for k in range(j + 1, len(points)):
                vertex = _solve_triple_vertex(points, coeffs, (i, j, k))
                if vertex is None:
                    continue
                key = (str(vertex[0]), str(vertex[1]))
                active = _active_indices_at_vertex(points, coeffs, vertex)
                if len(active) < 3:
                    continue
                vertices[key] = {
                    "vertex": [key[0], key[1]],
                    "active_indices": active,
                }

    balancing_stars: list[dict[str, Any]] = []
    max_residual_l1 = 0
    for star_index, row in enumerate(vertices.values()):
        active = row["active_indices"]
        hull = _convex_hull_indices(points, active)
        if len(hull) < 3:
            continue
        facet_indices: list[int] = []
        primitive_normals: list[list[int]] = []
        weights: list[int] = []
        weighted_normals: list[list[int]] = []
        residual = [0, 0]
        for offset, left in enumerate(hull):
            right = hull[(offset + 1) % len(hull)]
            edge = (points[right][0] - points[left][0], points[right][1] - points[left][1])
            normal = (edge[1], -edge[0])
            primitive, weight = _primitive(normal)
            pair = (left, right) if left < right else (right, left)
            facet_indices.append(facet_by_pair.get(pair, -1))
            primitive_normals.append(primitive)
            weights.append(weight)
            weighted = [weight * primitive[0], weight * primitive[1]]
            weighted_normals.append(weighted)
            residual[0] += weighted[0]
            residual[1] += weighted[1]
        residual_l1 = abs(residual[0]) + abs(residual[1])
        max_residual_l1 = max(max_residual_l1, residual_l1)
        balancing_stars.append(
            {
                "star_index": star_index,
                "vertex": row["vertex"],
                "active_indices": active,
                "dual_newton_cell_hull": hull,
                "facets": facet_indices,
                "primitive_normals": primitive_normals,
                "weights": weights,
                "weighted_primitive_normals": weighted_normals,
                "balance_residual": residual,
                "balance_residual_l1": residual_l1,
                "balanced": residual_l1 == 0,
            }
        )

    balanced = bool(balancing_stars) and all(star["balanced"] for star in balancing_stars)
    tropical = {
        "convention": "max_plus_tropical_attention; walls use ell_i=ell_j>=ell_k",
        "exponents": points,
        "biases": coeffs,
        "facets": facets,
        "multiplicities": [facet["multiplicity"] for facet in facets],
        "balancing_stars": balancing_stars,
        "balanced": balanced,
        "max_balance_residual_l1": int(max_residual_l1),
        "chow_minkowski_weight_certified": balanced,
        "chow_minkowski_weight_status": "certified_balanced_integer_weights" if balanced else "not_certified",
    }
    toric = {
        "minkowski_weight_facets": [
            {
                "facet_index": facet["facet_index"],
                "active_pair": facet["active_pair"],
                "weight": facet["multiplicity"],
                "primitive_normal": facet["primitive_normal"],
            }
            for facet in facets
        ],
        "chow_class_certified": balanced,
    }
    cert = exact_closed_form_certificate(
        kind="closed_form_tropical_hypersurface_multiplicity_balance_certificate",
        source={"input_kind": "two_dimensional_tropical_attention_support"},
        toric=toric,
        tropical=tropical,
        diagnostics={
            "facet_count": len(facets),
            "balancing_star_count": len(balancing_stars),
            "unsupported_dimensions": [] if len(points[0]) == 2 else [len(points[0])],
        },
    )
    return cert.with_hash()
