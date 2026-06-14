"""Exact GUDHI persistent-homology audits for ToricGT trajectories.

This module is intentionally separate from the differentiable training losses.
GUDHI gives exact finite Vietoris-Rips/simplex-tree persistence, vectorized
diagram representations, and simplex-tree operations.  The PyTorch helpers here
only vectorize already supplied birth/death pairs; they do not claim to
differentiate through GUDHI's persistence algorithm.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _require_gudhi() -> Any:
    try:
        import gudhi  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only on missing envs
        raise RuntimeError(
            "GUDHI is required for exact persistent-homology audits. "
            "Install it in the active environment, e.g. `pip install gudhi`."
        ) from exc
    return gudhi


@dataclass(frozen=True)
class GudhiPersistenceConfig:
    max_points: int = 32
    max_dimension: int = 2
    radius_quantile: float = 0.75
    num_radii: int = 5
    num_levels: int = 5
    landscape_resolution: int = 64
    landscape_layers: int = 5
    image_resolution: int = 16
    max_edge_length: float | None = None
    macaulay2_resolutions: bool = True
    macaulay2_timeout_seconds: int = 180


def standardize_points(points: np.ndarray, *, max_dim: int = 8) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"points must be a 2D array, got shape {x.shape}")
    if x.shape[0] < 2:
        raise ValueError("at least two points are required for a GUDHI PH audit")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    x = x / np.maximum(scale, 1e-8)
    if x.shape[1] > max_dim:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
        x = x @ vh[:max_dim].T
    return np.ascontiguousarray(x, dtype=np.float64)


def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    x = np.asarray(points)
    if x.shape[0] <= max_points:
        return x
    idx = np.linspace(0, x.shape[0] - 1, num=max_points).round().astype(int)
    return x[idx]


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))


def radius_grid(points: np.ndarray, cfg: GudhiPersistenceConfig) -> np.ndarray:
    dist = pairwise_distances(points)
    positive = dist[dist > 1e-10]
    if positive.size == 0:
        raise ValueError("point cloud has zero pairwise distances only")
    max_radius = (
        float(cfg.max_edge_length)
        if cfg.max_edge_length is not None and cfg.max_edge_length > 0
        else float(np.quantile(positive, min(max(cfg.radius_quantile, 0.05), 1.0)))
    )
    max_radius = max(max_radius, float(np.min(positive)))
    return np.linspace(max_radius / max(2, int(cfg.num_radii)), max_radius, num=max(2, int(cfg.num_radii)))


def build_rips_simplex_tree(points: np.ndarray, radius: float, max_dimension: int = 2) -> Any:
    gudhi = _require_gudhi()
    rips = gudhi.RipsComplex(points=np.asarray(points, dtype=float).tolist(), max_edge_length=float(radius))
    simplex_tree = rips.create_simplex_tree(max_dimension=int(max_dimension))
    simplex_tree.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    return simplex_tree


def simplex_sets(simplex_tree: Any, max_dimension: int = 2) -> dict[int, list[tuple[int, ...]]]:
    out: dict[int, list[tuple[int, ...]]] = {dim: [] for dim in range(max_dimension + 1)}
    for simplex, _filtration in simplex_tree.get_filtration():
        item = tuple(sorted(int(v) for v in simplex))
        dim = len(item) - 1
        if 0 <= dim <= max_dimension:
            out[dim].append(item)
    for dim in out:
        out[dim] = sorted(set(out[dim]))
    return out


def simplex_birth_bidegrees(
    points: np.ndarray,
    radii: np.ndarray,
    *,
    max_dimension: int = 2,
) -> dict[int, list[dict[str, Any]]]:
    """Return simplex generators with birth bidegrees.

    The first parameter is reasoning level: a simplex is born when all of its
    vertices have appeared in the trajectory prefix.  The second parameter is
    the Vietoris-Rips radius index.  This gives a finite free
    ``F2[x_level,y_radius]`` chain complex.
    """

    tree = build_rips_simplex_tree(points, float(radii[-1]), max_dimension=max_dimension)
    dist = pairwise_distances(points)
    out: dict[int, list[dict[str, Any]]] = {dim: [] for dim in range(max_dimension + 1)}
    for dim, simplices in simplex_sets(tree, max_dimension=max_dimension).items():
        for simplex in simplices:
            if dim == 0:
                diameter = 0.0
            else:
                diameter = max(float(dist[i, j]) for pos, i in enumerate(simplex) for j in simplex[pos + 1 :])
            radius_index = int(np.searchsorted(radii, diameter, side="left"))
            radius_index = min(max(radius_index, 0), len(radii) - 1)
            out[dim].append(
                {
                    "simplex": list(simplex),
                    "degree": [int(max(simplex)), int(radius_index)],
                    "diameter": float(diameter),
                }
            )
    return out


def _monomial_from_delta(delta: tuple[int, int]) -> str:
    x_power, y_power = int(delta[0]), int(delta[1])
    if x_power < 0 or y_power < 0:
        raise ValueError(f"negative bidegree delta {delta}")
    factors: list[str] = []
    if x_power == 1:
        factors.append("x_level")
    elif x_power > 1:
        factors.append(f"x_level^{x_power}")
    if y_power == 1:
        factors.append("y_radius")
    elif y_power > 1:
        factors.append(f"y_radius^{y_power}")
    return "*".join(factors) if factors else "1_R"


def _m2_free_module(name: str, generators: list[dict[str, Any]]) -> str:
    if not generators:
        return f"{name} = R^0"
    shifts = []
    for gen in generators:
        deg = gen["degree"]
        shifts.append("{" + f"{-int(deg[0])},{-int(deg[1])}" + "}")
    return f"{name} = R^{{{','.join(shifts)}}}"


def _m2_matrix_rows(
    target_generators: list[dict[str, Any]],
    source_generators: list[dict[str, Any]],
) -> list[list[str]]:
    target_index = {tuple(gen["simplex"]): idx for idx, gen in enumerate(target_generators)}
    rows = [["0_R" for _ in source_generators] for _ in target_generators]
    for col, source in enumerate(source_generators):
        simplex = tuple(int(v) for v in source["simplex"])
        source_degree = tuple(int(v) for v in source["degree"])
        for drop_idx in range(len(simplex)):
            face = tuple(v for idx, v in enumerate(simplex) if idx != drop_idx)
            row = target_index.get(tuple(sorted(face)))
            if row is None:
                continue
            face_degree = tuple(int(v) for v in target_generators[row]["degree"])
            delta = (source_degree[0] - face_degree[0], source_degree[1] - face_degree[1])
            rows[row][col] = _monomial_from_delta(delta)
    return rows


def _m2_matrix_literal(rows: list[list[str]]) -> str:
    return "matrix{" + ",".join("{" + ",".join(row) + "}" for row in rows) + "}"


def bigraded_chain_presentation(points: np.ndarray, radii: np.ndarray, *, max_dimension: int = 2) -> dict[str, Any]:
    generators = simplex_birth_bidegrees(points, radii, max_dimension=max_dimension)
    boundaries: dict[str, list[list[str]]] = {}
    for dim in range(1, max_dimension + 1):
        boundaries[str(dim)] = _m2_matrix_rows(generators.get(dim - 1, []), generators.get(dim, []))
    return {
        "ring": "F2[x_level,y_radius]",
        "variables": ["x_level", "y_radius"],
        "field": "GF(2)",
        "max_dimension": int(max_dimension),
        "radii": [float(v) for v in radii.tolist()],
        "generators": {str(dim): generators.get(dim, []) for dim in range(max_dimension + 1)},
        "boundary_matrices": boundaries,
    }


def _extract_json_between_markers(text: str) -> dict[str, Any]:
    begin = "TORICGT_JSON_BEGIN"
    end = "TORICGT_JSON_END"
    start = text.find(begin)
    stop = text.find(end)
    if start < 0 or stop < 0 or stop <= start:
        raise RuntimeError(f"CAS JSON markers not found in output tail:\n{text[-4000:]}")
    payload = text[start + len(begin) : stop].strip()
    return json.loads(payload)


def macaulay2_script_for_bigraded_chain(presentation: dict[str, Any]) -> str:
    generators = presentation["generators"]
    boundaries = presentation["boundary_matrices"]
    lines = [
        'needsPackage "JSON"',
        'R = GF(2)[x_level,y_radius, Degrees=>{{1,0},{0,1}}]',
        _m2_free_module("C0", generators.get("0", [])),
        _m2_free_module("C1", generators.get("1", [])),
        _m2_free_module("C2", generators.get("2", [])),
    ]
    if len(generators.get("0", [])) == 0 or len(generators.get("1", [])) == 0:
        lines.append("d1 = map(C0,C1,0)")
    else:
        lines.append(f"d1 = map(C0,C1,{_m2_matrix_literal(boundaries.get('1', []))})")
    if len(generators.get("1", [])) == 0 or len(generators.get("2", [])) == 0:
        lines.append("d2 = map(C1,C2,0)")
    else:
        lines.append(f"d2 = map(C1,C2,{_m2_matrix_literal(boundaries.get('2', []))})")
    lines.extend(
        [
            "C = chainComplex({d1,d2})",
            "H0 = HH_0 C",
            "H1 = HH_1 C",
            "H2 = HH_2 C",
            "resolutionSummary = hashTable {",
            '  "H0_module" => toString H0,',
            '  "H1_module" => toString H1,',
            '  "H2_module" => toString H2,',
            '  "H0_presentation" => toString presentation H0,',
            '  "H1_presentation" => toString presentation H1,',
            '  "H2_presentation" => toString presentation H2,',
            '  "H0_betti" => toString betti res H0,',
            '  "H1_betti" => toString betti res H1,',
            '  "H2_betti" => toString betti res H2,',
            '  "H0_resolution" => toString res H0,',
            '  "H1_resolution" => toString res H1,',
            '  "H2_resolution" => toString res H2',
            "}",
            "out = hashTable {",
            '  "kind" => "macaulay2_bigraded_persistence_resolution",',
            '  "ring" => "GF(2)[x_level,y_radius]",',
            '  "module_language" => "F2[x_level,y_radius]-graded chain complex",',
            '  "num_C0_generators" => rank C0,',
            '  "num_C1_generators" => rank C1,',
            '  "num_C2_generators" => rank C2,',
            '  "homogeneous_d1" => isHomogeneous d1,',
            '  "homogeneous_d2" => isHomogeneous d2,',
            '  "d_squared_zero" => d1*d2 == 0,',
            '  "chain_complex" => toString C,',
            '  "boundary_d1" => toString d1,',
            '  "boundary_d2" => toString d2,',
            '  "homology_and_resolutions" => resolutionSummary',
            "}",
            'print "TORICGT_JSON_BEGIN"',
            "print toJSON out",
            'print "TORICGT_JSON_END"',
        ]
    )
    return "\n".join(lines) + "\n"


def macaulay2_bigraded_resolution_certificate(
    presentation: dict[str, Any],
    *,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    executable = shutil.which("M2") or shutil.which("macaulay2")
    if not executable:
        raise RuntimeError("Macaulay2 executable `M2` was not found on PATH")
    script = macaulay2_script_for_bigraded_chain(presentation)
    with tempfile.TemporaryDirectory(prefix="toricgt_gudhi_m2_") as tmp:
        script_path = Path(tmp) / "bigraded_persistence.m2"
        script_path.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            [executable, "--script", str(script_path)],
            cwd=tmp,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(timeout_seconds),
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-5000:])
    payload = _extract_json_between_markers(proc.stdout)
    payload["macaulay2_stdout_tail"] = proc.stdout[-2000:]
    payload["script"] = script
    return payload


def persistence_diagrams(simplex_tree: Any, max_dimension: int = 2) -> dict[int, np.ndarray]:
    diagrams: dict[int, np.ndarray] = {}
    for dim in range(max_dimension + 1):
        diag = np.asarray(simplex_tree.persistence_intervals_in_dimension(dim), dtype=np.float64)
        if diag.size == 0:
            diag = np.zeros((0, 2), dtype=np.float64)
        diagrams[dim] = diag.reshape(-1, 2)
    return diagrams


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    diag = np.asarray(diagram, dtype=np.float64).reshape(-1, 2)
    if diag.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    mask = np.isfinite(diag).all(axis=1) & (diag[:, 1] > diag[:, 0])
    return np.ascontiguousarray(diag[mask], dtype=np.float64)


def _safe_gudhi_vectorizer(name: str, diagram: np.ndarray, cfg: GudhiPersistenceConfig) -> np.ndarray:
    from gudhi.representations import Entropy, Landscape, PersistenceImage, Silhouette  # type: ignore

    diag = finite_diagram(diagram)
    if diag.shape[0] == 0:
        if name == "image":
            return np.zeros((cfg.image_resolution * cfg.image_resolution,), dtype=np.float64)
        if name == "landscape":
            return np.zeros((cfg.landscape_layers * cfg.landscape_resolution,), dtype=np.float64)
        return np.zeros((cfg.landscape_resolution,), dtype=np.float64)
    if name == "landscape":
        obj = Landscape(num_landscapes=int(cfg.landscape_layers), resolution=int(cfg.landscape_resolution))
    elif name == "image":
        obj = PersistenceImage(
            bandwidth=0.1,
            resolution=[int(cfg.image_resolution), int(cfg.image_resolution)],
        )
    elif name == "silhouette":
        obj = Silhouette(resolution=int(cfg.landscape_resolution))
    elif name == "entropy":
        obj = Entropy(mode="vector", resolution=int(cfg.landscape_resolution))
    else:
        raise ValueError(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values = obj.fit_transform([diag])[0].astype(np.float64)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def vectorized_diagram_metrics(diagram: np.ndarray, cfg: GudhiPersistenceConfig) -> dict[str, Any]:
    diag = finite_diagram(diagram)
    persistence = diag[:, 1] - diag[:, 0] if diag.size else np.zeros((0,), dtype=np.float64)
    landscape = _safe_gudhi_vectorizer("landscape", diag, cfg)
    image = _safe_gudhi_vectorizer("image", diag, cfg)
    silhouette = _safe_gudhi_vectorizer("silhouette", diag, cfg)
    entropy_vector = _safe_gudhi_vectorizer("entropy", diag, cfg)
    return {
        "count": int(diag.shape[0]),
        "total_persistence": float(persistence.sum()) if persistence.size else 0.0,
        "max_persistence": float(persistence.max()) if persistence.size else 0.0,
        "mean_persistence": float(persistence.mean()) if persistence.size else 0.0,
        "landscape": landscape.tolist(),
        "landscape_norm": float(np.linalg.norm(landscape)),
        "persistence_image": image.tolist(),
        "persistence_image_norm": float(np.linalg.norm(image)),
        "silhouette": silhouette.tolist(),
        "silhouette_norm": float(np.linalg.norm(silhouette)),
        "entropy_vector": entropy_vector.tolist(),
        "entropy_vector_norm": float(np.linalg.norm(entropy_vector)),
    }


def gf2_rank(matrix: np.ndarray) -> int:
    mat = (np.asarray(matrix, dtype=np.uint8).copy() & 1)
    if mat.size == 0:
        return 0
    rows, cols = mat.shape
    rank = 0
    for col in range(cols):
        pivots = np.flatnonzero(mat[rank:, col])
        if pivots.size == 0:
            continue
        pivot = rank + int(pivots[0])
        if pivot != rank:
            mat[[rank, pivot]] = mat[[pivot, rank]]
        for row in range(rows):
            if row != rank and mat[row, col]:
                mat[row] ^= mat[rank]
        rank += 1
        if rank == rows:
            break
    return int(rank)


def gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
    mat = (np.asarray(matrix, dtype=np.uint8).copy() & 1)
    rows, cols = mat.shape
    if cols == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    rank = 0
    pivots: list[int] = []
    for col in range(cols):
        pivot_rows = np.flatnonzero(mat[rank:, col])
        if pivot_rows.size == 0:
            continue
        pivot = rank + int(pivot_rows[0])
        if pivot != rank:
            mat[[rank, pivot]] = mat[[pivot, rank]]
        for row in range(rows):
            if row != rank and mat[row, col]:
                mat[row] ^= mat[rank]
        pivots.append(col)
        rank += 1
        if rank == rows:
            break
    pivot_set = set(pivots)
    free_cols = [col for col in range(cols) if col not in pivot_set]
    if not free_cols:
        return np.zeros((cols, 0), dtype=np.uint8)
    basis = np.zeros((cols, len(free_cols)), dtype=np.uint8)
    for basis_col, free_col in enumerate(free_cols):
        vec = np.zeros((cols,), dtype=np.uint8)
        vec[free_col] = 1
        for row, pivot_col in enumerate(pivots):
            if mat[row, free_col]:
                vec[pivot_col] = 1
        basis[:, basis_col] = vec
    return basis


def boundary_matrix(simplices: dict[int, list[tuple[int, ...]]], dim: int) -> np.ndarray:
    if dim <= 0:
        cols = len(simplices.get(0, []))
        return np.zeros((0, cols), dtype=np.uint8)
    lower = simplices.get(dim - 1, [])
    current = simplices.get(dim, [])
    lower_index = {simplex: idx for idx, simplex in enumerate(lower)}
    mat = np.zeros((len(lower), len(current)), dtype=np.uint8)
    for col, simplex in enumerate(current):
        for drop_idx in range(len(simplex)):
            face = tuple(v for i, v in enumerate(simplex) if i != drop_idx)
            row = lower_index.get(tuple(sorted(face)))
            if row is not None:
                mat[row, col] ^= 1
    return mat


def chain_complex_summary(simplex_tree: Any, max_dimension: int = 2) -> dict[str, Any]:
    simplices = simplex_sets(simplex_tree, max_dimension=max_dimension)
    boundaries = {dim: boundary_matrix(simplices, dim) for dim in range(max_dimension + 1)}
    betti: dict[str, int] = {}
    boundary_ranks: dict[str, int] = {}
    for dim in range(max_dimension + 1):
        d_p = boundaries[dim]
        d_next = boundaries.get(dim + 1, np.zeros((len(simplices.get(dim, [])), 0), dtype=np.uint8))
        z_rank = d_p.shape[1] - gf2_rank(d_p)
        b_rank = gf2_rank(d_next)
        betti[str(dim)] = max(0, int(z_rank - b_rank))
        boundary_ranks[str(dim)] = gf2_rank(d_p)
    d1 = boundaries.get(1, np.zeros((0, 0), dtype=np.uint8))
    d2 = boundaries.get(2, np.zeros((0, 0), dtype=np.uint8))
    d2_residual = int(((d1 @ d2) & 1).sum()) if d1.size and d2.size else 0
    return {
        "num_vertices": len(simplices.get(0, [])),
        "num_edges": len(simplices.get(1, [])),
        "num_triangles": len(simplices.get(2, [])),
        "num_simplices": int(sum(len(v) for v in simplices.values())),
        "betti": betti,
        "boundary_ranks": boundary_ranks,
        "d_squared_residual": d2_residual,
        "_simplices": simplices,
        "_boundaries": boundaries,
    }


def simplex_map_audit(source: Any, target: Any, vertex_map: dict[int, int] | None = None, max_dimension: int = 2) -> dict[str, Any]:
    src = simplex_sets(source, max_dimension=max_dimension)
    tgt = simplex_sets(target, max_dimension=max_dimension)
    vertex_map = vertex_map or {v[0]: v[0] for v in src.get(0, [])}
    counts: dict[str, float] = {}
    total_valid = 0
    total = 0
    for dim in range(1, max_dimension + 1):
        dim_total = 0
        dim_valid = 0
        dim_collapsed = 0
        for simplex in src.get(dim, []):
            mapped = tuple(sorted(set(int(vertex_map.get(v, v)) for v in simplex)))
            dim_total += 1
            total += 1
            if len(mapped) <= 1:
                dim_valid += 1
                total_valid += 1
                dim_collapsed += 1
            elif target.find(list(mapped)):
                dim_valid += 1
                total_valid += 1
        label = f"dim{dim}"
        counts[f"{label}_simplices"] = float(dim_total)
        counts[f"{label}_valid"] = float(dim_valid)
        counts[f"{label}_valid_fraction"] = float(dim_valid / max(1, dim_total))
        counts[f"{label}_collapsed"] = float(dim_collapsed)
    counts["simplicial_map_valid_fraction"] = float(total_valid / max(1, total))
    counts["simplicial_map_valid"] = bool(total_valid == total)
    return counts


def inclusion_matrix(source_basis: list[tuple[int, ...]], target_basis: list[tuple[int, ...]]) -> np.ndarray:
    target_index = {simplex: idx for idx, simplex in enumerate(target_basis)}
    mat = np.zeros((len(target_basis), len(source_basis)), dtype=np.uint8)
    for col, simplex in enumerate(source_basis):
        row = target_index.get(simplex)
        if row is not None:
            mat[row, col] = 1
    return mat


def induced_homology_rank(source_summary: dict[str, Any], target_summary: dict[str, Any], dim: int) -> int:
    src_simplices = source_summary["_simplices"]
    tgt_simplices = target_summary["_simplices"]
    src_boundaries = source_summary["_boundaries"]
    tgt_boundaries = target_summary["_boundaries"]
    src_basis = src_simplices.get(dim, [])
    tgt_basis = tgt_simplices.get(dim, [])
    if not src_basis or not tgt_basis:
        return 0
    f_chain = inclusion_matrix(src_basis, tgt_basis)
    z_src = gf2_nullspace(src_boundaries.get(dim, np.zeros((0, len(src_basis)), dtype=np.uint8)))
    b_tgt = tgt_boundaries.get(dim + 1, np.zeros((len(tgt_basis), 0), dtype=np.uint8))
    image_cycles = (f_chain @ z_src) & 1 if z_src.size else np.zeros((len(tgt_basis), 0), dtype=np.uint8)
    combined = np.concatenate([b_tgt, image_cycles], axis=1) if b_tgt.size or image_cycles.size else np.zeros((len(tgt_basis), 0), dtype=np.uint8)
    return max(0, gf2_rank(combined) - gf2_rank(b_tgt))


def commutative_square_residual(
    summaries: dict[tuple[int, int], dict[str, Any]],
    *,
    num_levels: int,
    num_radii: int,
    max_dimension: int,
) -> dict[str, Any]:
    """Check x-level/y-radius inclusion squares on GF(2) chain groups."""

    by_dim = {str(dim): 0 for dim in range(max_dimension + 1)}
    checked = 0
    total = 0
    for li in range(max(0, int(num_levels) - 1)):
        for ri in range(max(0, int(num_radii) - 1)):
            checked += 1
            src = summaries[(li, ri)]["_simplices"]
            x_node = summaries[(li + 1, ri)]["_simplices"]
            y_node = summaries[(li, ri + 1)]["_simplices"]
            xy_node = summaries[(li + 1, ri + 1)]["_simplices"]
            for dim in range(max_dimension + 1):
                basis_src = src.get(dim, [])
                basis_x = x_node.get(dim, [])
                basis_y = y_node.get(dim, [])
                basis_xy = xy_node.get(dim, [])
                x_map = inclusion_matrix(basis_src, basis_x)
                y_after_x = inclusion_matrix(basis_x, basis_xy)
                y_map = inclusion_matrix(basis_src, basis_y)
                x_after_y = inclusion_matrix(basis_y, basis_xy)
                path_xy = (y_after_x @ x_map) & 1
                path_yx = (x_after_y @ y_map) & 1
                residual = int((path_xy ^ path_yx).sum())
                by_dim[str(dim)] += residual
                total += residual
    return {
        "commutative_squares_checked": int(checked),
        "commutative_square_chain_residuals_by_dimension": by_dim,
        "commutative_square_residual": float(total),
    }


def two_parameter_module(points: np.ndarray, cfg: GudhiPersistenceConfig) -> dict[str, Any]:
    radii = radius_grid(points, cfg)
    n = points.shape[0]
    levels = np.unique(np.linspace(2, n, num=max(2, int(cfg.num_levels))).round().astype(int))
    nodes: list[dict[str, Any]] = []
    summaries: dict[tuple[int, int], dict[str, Any]] = {}
    trees: dict[tuple[int, int], Any] = {}
    for li, count in enumerate(levels):
        subpoints = points[: int(count)]
        for ri, radius in enumerate(radii):
            tree = build_rips_simplex_tree(subpoints, float(radius), max_dimension=cfg.max_dimension)
            summary = chain_complex_summary(tree, max_dimension=cfg.max_dimension)
            summaries[(li, ri)] = summary
            trees[(li, ri)] = tree
            nodes.append(
                {
                    "x_level": int(li),
                    "x_prefix_points": int(count),
                    "y_radius_index": int(ri),
                    "y_radius": float(radius),
                    "num_simplices": int(summary["num_simplices"]),
                    "num_edges": int(summary["num_edges"]),
                    "num_triangles": int(summary["num_triangles"]),
                    "betti0": int(summary["betti"].get("0", 0)),
                    "betti1": int(summary["betti"].get("1", 0)),
                    "betti2": int(summary["betti"].get("2", 0)),
                    "d_squared_residual": int(summary["d_squared_residual"]),
                }
            )
    maps: list[dict[str, Any]] = []
    for li in range(len(levels)):
        for ri in range(len(radii)):
            src = summaries[(li, ri)]
            if li + 1 < len(levels):
                tgt = summaries[(li + 1, ri)]
                maps.append(
                    {
                        "from": [int(li), int(ri)],
                        "to": [int(li + 1), int(ri)],
                        "operator": "x_level",
                        "h0_rank": int(induced_homology_rank(src, tgt, 0)),
                        "h1_rank": int(induced_homology_rank(src, tgt, 1)),
                        "simplicial": simplex_map_audit(trees[(li, ri)], trees[(li + 1, ri)], max_dimension=cfg.max_dimension),
                    }
                )
            if ri + 1 < len(radii):
                tgt = summaries[(li, ri + 1)]
                maps.append(
                    {
                        "from": [int(li), int(ri)],
                        "to": [int(li), int(ri + 1)],
                        "operator": "y_radius",
                        "h0_rank": int(induced_homology_rank(src, tgt, 0)),
                        "h1_rank": int(induced_homology_rank(src, tgt, 1)),
                        "simplicial": simplex_map_audit(trees[(li, ri)], trees[(li, ri + 1)], max_dimension=cfg.max_dimension),
                    }
                )
    square_audit = commutative_square_residual(
        summaries,
        num_levels=len(levels),
        num_radii=len(radii),
        max_dimension=cfg.max_dimension,
    )
    return {
        "ring": "F2[x_level,y_radius]",
        "levels": [int(v) for v in levels.tolist()],
        "radii": [float(v) for v in radii.tolist()],
        "nodes": nodes,
        "structure_maps": maps,
        **square_audit,
        "hilbert_h0": [[int(summaries[(li, ri)]["betti"].get("0", 0)) for ri in range(len(radii))] for li in range(len(levels))],
        "hilbert_h1": [[int(summaries[(li, ri)]["betti"].get("1", 0)) for ri in range(len(radii))] for li in range(len(levels))],
        "chain_generators_c0": [[int(summaries[(li, ri)]["num_vertices"]) for ri in range(len(radii))] for li in range(len(levels))],
        "chain_generators_c1": [[int(summaries[(li, ri)]["num_edges"]) for ri in range(len(radii))] for li in range(len(levels))],
        "chain_generators_c2": [[int(summaries[(li, ri)]["num_triangles"]) for ri in range(len(radii))] for li in range(len(levels))],
    }


def torch_persistence_landscape(diagram: torch.Tensor, grid: torch.Tensor, *, layers: int = 5) -> torch.Tensor:
    """Differentiable persistence-landscape vectorization for birth/death pairs.

    ``diagram`` is ``[N, 2]`` and ``grid`` is ``[R]``.  The result is
    ``[layers, R]``.  Gradients flow to the supplied birth/death coordinates.
    """

    if diagram.ndim != 2 or diagram.shape[-1] != 2:
        raise ValueError("diagram must have shape [N, 2]")
    births = diagram[:, 0:1]
    deaths = diagram[:, 1:2]
    tents = torch.minimum(grid[None, :] - births, deaths - grid[None, :]).clamp_min(0.0)
    k = min(max(1, int(layers)), max(1, tents.shape[0]))
    values = torch.topk(tents, k=k, dim=0).values
    if k < layers:
        values = torch.cat([values, values.new_zeros((layers - k, values.shape[1]))], dim=0)
    return values


def torch_persistence_image(
    diagram: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    *,
    sigma: float = 0.05,
) -> torch.Tensor:
    """Differentiable persistence-image surface for birth/death pairs."""

    if diagram.ndim != 2 or diagram.shape[-1] != 2:
        raise ValueError("diagram must have shape [N, 2]")
    birth = diagram[:, 0]
    persistence = (diagram[:, 1] - diagram[:, 0]).clamp_min(0.0)
    xx, yy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    dx = xx[None, :, :] - birth[:, None, None]
    dy = yy[None, :, :] - persistence[:, None, None]
    weight = persistence[:, None, None]
    surface = weight * torch.exp(-(dx.pow(2) + dy.pow(2)) / max(2.0 * sigma * sigma, 1e-8))
    return surface.sum(dim=0)


def audit_point_cloud(points: np.ndarray, *, record_id: str, cfg: GudhiPersistenceConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GudhiPersistenceConfig()
    x = standardize_points(sample_points(points, int(cfg.max_points)))
    radii = radius_grid(x, cfg)
    max_radius = float(radii[-1])
    tree = build_rips_simplex_tree(x, max_radius, max_dimension=cfg.max_dimension)
    diagrams = persistence_diagrams(tree, max_dimension=cfg.max_dimension)
    chain = chain_complex_summary(tree, max_dimension=cfg.max_dimension)
    vectors = {
        str(dim): vectorized_diagram_metrics(diagrams[dim], cfg)
        for dim in range(cfg.max_dimension + 1)
    }
    module = two_parameter_module(x, cfg)
    bigraded_presentation = bigraded_chain_presentation(x, radii, max_dimension=cfg.max_dimension)
    m2_resolution = (
        macaulay2_bigraded_resolution_certificate(
            bigraded_presentation,
            timeout_seconds=int(cfg.macaulay2_timeout_seconds),
        )
        if bool(cfg.macaulay2_resolutions)
        else {}
    )
    return {
        "record_id": record_id,
        "points": int(x.shape[0]),
        "dimension": int(x.shape[1]),
        "max_radius": max_radius,
        "radii": [float(v) for v in radii.tolist()],
        "simplex_tree": {
            "num_vertices": int(tree.num_vertices()),
            "num_simplices": int(tree.num_simplices()),
            "dimension": int(tree.dimension()),
        },
        "chain_complex": {key: value for key, value in chain.items() if not key.startswith("_")},
        "diagrams": {str(dim): diagrams[dim].tolist() for dim in diagrams},
        "vectorizations": vectors,
        "two_parameter_module": module,
        "bigraded_chain_presentation": bigraded_presentation,
        "macaulay2_resolution": m2_resolution,
    }


def summarize_audits(records: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(path: tuple[str, ...]) -> float:
        values: list[float] = []
        for record in records:
            node: Any = record
            for key in path:
                if not isinstance(node, dict) or key not in node:
                    node = None
                    break
                node = node[key]
            if isinstance(node, (int, float)) and math.isfinite(float(node)):
                values.append(float(node))
        return float(np.mean(values)) if values else 0.0

    return {
        "schema": "toricgt.gudhi_persistence.summary.v1",
        "backend": "gudhi",
        "records": int(len(records)),
        "mean_points": mean(("points",)),
        "mean_num_simplices": mean(("simplex_tree", "num_simplices")),
        "mean_betti0": mean(("chain_complex", "betti", "0")),
        "mean_betti1": mean(("chain_complex", "betti", "1")),
        "mean_h0_landscape_norm": mean(("vectorizations", "0", "landscape_norm")),
        "mean_h1_landscape_norm": mean(("vectorizations", "1", "landscape_norm")),
        "mean_h1_persistence_image_norm": mean(("vectorizations", "1", "persistence_image_norm")),
        "mean_two_parameter_commutative_square_residual": mean(("two_parameter_module", "commutative_square_residual")),
        "mean_macaulay2_homogeneous_d1": mean(("macaulay2_resolution", "homogeneous_d1")),
        "mean_macaulay2_homogeneous_d2": mean(("macaulay2_resolution", "homogeneous_d2")),
        "mean_macaulay2_d_squared_zero": mean(("macaulay2_resolution", "d_squared_zero")),
        "records_detail": [
            {
                "record_id": item.get("record_id"),
                "points": item.get("points"),
                "max_radius": item.get("max_radius"),
                "betti": item.get("chain_complex", {}).get("betti", {}),
                "num_simplices": item.get("simplex_tree", {}).get("num_simplices"),
            }
            for item in records
        ],
    }


def load_points_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "point_clouds" in data:
        data = data["point_clouds"]
    if not isinstance(data, list):
        raise ValueError("points JSON must be a list or contain a `point_clouds` list")
    out = []
    for idx, item in enumerate(data):
        if isinstance(item, dict):
            points = item.get("points")
            record_id = str(item.get("record_id", f"points_{idx:02d}"))
        else:
            points = item
            record_id = f"points_{idx:02d}"
        out.append({"record_id": record_id, "points": np.asarray(points, dtype=np.float64)})
    return out


def sample_checkpoint_point_clouds(path: Path, *, records: int = 4, max_points: int = 32) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint {path} did not contain a tensor state dict")
    candidates: list[tuple[str, np.ndarray, float]] = []
    for key, value in state.items():
        if not torch.is_tensor(value) or value.ndim < 2:
            continue
        arr = value.detach().float().cpu().reshape(value.shape[0], -1).numpy()
        if arr.shape[0] < 4 or arr.shape[1] < 2:
            continue
        score = float(np.linalg.norm(arr))
        candidates.append((str(key), arr, score))
    candidates.sort(key=lambda item: item[2], reverse=True)
    out: list[dict[str, Any]] = []
    for key, arr, _score in candidates[: max(1, int(records))]:
        out.append({"record_id": key.replace(".", "_").replace("/", "_")[:80], "points": sample_points(arr, max_points)})
    if not out:
        raise ValueError(f"checkpoint {path} did not contain usable 2D tensor point clouds")
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
