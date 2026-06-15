#!/usr/bin/env python3
"""Run exact GUDHI + Macaulay2 persistence audits and write dark-mode HTML.

The audit treats reasoning step and Rips radius as the two parameters.  For
each point cloud it builds:

* exact GUDHI simplex trees and persistence diagrams;
* vectorized PH features: landscapes, persistence images, silhouettes, entropy;
* a finite F2[x_level,y_radius] module over the reasoning/radius grid;
* a Macaulay2 bigraded chain complex, homology modules, and free resolutions;
* a browser index linking all record pages and JSON/CAS artifacts.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toricgt.gudhi_persistence import (  # noqa: E402
    GudhiPersistenceConfig,
    audit_point_cloud,
    load_points_json,
    sample_checkpoint_point_clouds,
    summarize_audits,
    write_json,
)


CSS = """
:root {
  color-scheme: dark;
  --bg: #030712;
  --panel: #07111f;
  --panel2: #0b1728;
  --text: #e8fbff;
  --muted: #91a8b7;
  --cyan: #37e8ff;
  --magenta: #ff4fd8;
  --green: #8cff6a;
  --border: rgba(55, 232, 255, 0.28);
}
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: radial-gradient(circle at top left, #092238 0, var(--bg) 42rem);
  color: var(--text);
}
main { max-width: 1320px; margin: 0 auto; padding: 32px 24px 64px; }
a { color: var(--cyan); text-decoration: none; }
a:hover { text-decoration: underline; }
.hero, .card, .panel {
  background: linear-gradient(180deg, rgba(11, 23, 40, 0.94), rgba(5, 13, 25, 0.96));
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
}
.hero { padding: 24px; margin-bottom: 20px; }
h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }
h2 { margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }
p { color: var(--muted); line-height: 1.55; }
.grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
.card { padding: 16px; }
.metric { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; border-bottom: 1px solid rgba(145,168,183,0.14); padding: 7px 0; }
.metric span:first-child { color: var(--muted); }
.metric span:last-child { color: white; font-variant-numeric: tabular-nums; text-align: right; overflow-wrap: anywhere; }
.links { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.pill { border: 1px solid var(--border); border-radius: 999px; padding: 6px 10px; background: rgba(55,232,255,0.07); }
.badge { display: inline-block; border: 1px solid rgba(140,255,106,0.35); border-radius: 999px; color: #d9ffd2; background: rgba(140,255,106,0.11); padding: 4px 8px; font-size: 12px; margin: 2px 4px 2px 0; }
.badge.fail { border-color: rgba(255,79,216,0.45); color: #ffd8f7; background: rgba(255,79,216,0.12); }
.tablewrap { overflow-x: auto; margin: 12px 0; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { border: 1px solid rgba(145,168,183,0.17); padding: 6px 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; color: var(--muted); }
.xygrid { width: 100%; max-width: 820px; height: auto; display: block; background: #020713; border: 1px solid rgba(145,168,183,0.18); border-radius: 8px; }
.legend { color: var(--muted); font-size: 13px; line-height: 1.5; }
details { margin-top: 12px; }
summary { cursor: pointer; color: var(--cyan); }
pre {
  white-space: pre-wrap;
  overflow-x: auto;
  background: #020713;
  border: 1px solid rgba(145,168,183,0.18);
  border-radius: 8px;
  padding: 14px;
  color: #dff8ff;
}
.plot { margin: 18px 0; }
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", default="", help="Checkpoint whose tensors will be sampled as point clouds.")
    source.add_argument("--points-json", default="", help="JSON list of point clouds or {'point_clouds': [...]} payload.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--records", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=18)
    parser.add_argument("--max-dimension", type=int, default=2)
    parser.add_argument("--radius-quantile", type=float, default=0.62)
    parser.add_argument("--num-radii", type=int, default=5)
    parser.add_argument("--num-levels", type=int, default=5)
    parser.add_argument("--landscape-resolution", type=int, default=64)
    parser.add_argument("--landscape-layers", type=int, default=5)
    parser.add_argument("--image-resolution", type=int, default=16)
    parser.add_argument("--macaulay2-timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--emit-ph-feature-visualizations",
        dest="emit_ph_feature_visualizations",
        action="store_true",
        default=True,
        help="Render first-class dashboards for landscapes, persistence images, silhouettes, entropy vectors, Betti curves, and lifetime histograms.",
    )
    parser.add_argument(
        "--no-emit-ph-feature-visualizations",
        dest="emit_ph_feature_visualizations",
        action="store_false",
        help="Skip the optional PH feature dashboards while still writing exact GUDHI metrics.",
    )
    parser.add_argument(
        "--no-macaulay2-resolutions",
        action="store_true",
        help="Disable Macaulay2 free-resolution output. Periodic training should not use this.",
    )
    return parser.parse_args()


def safe_slug(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
    return out[:96] or "record"


def finite_matrix(payload: Any) -> np.ndarray:
    arr = np.asarray(payload, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)) and len(value) <= 4:
        return "(" + ", ".join(format_value(v) for v in value) + ")"
    return str(value)


def pass_badge(label: str, ok: Any) -> str:
    passed = bool(ok)
    cls = "badge" if passed else "badge fail"
    value = "PASS" if passed else "FAIL"
    return f'<span class="{cls}">{html.escape(label)}: {value}</span>'


def matrix_table(matrix: Any, *, row_label: str = "x_level") -> str:
    arr = finite_matrix(matrix)
    rows: list[str] = []
    header = "<tr><th>{}</th>{}</tr>".format(
        html.escape(row_label),
        "".join(f"<th>y={idx}</th>" for idx in range(arr.shape[1])),
    )
    for row_idx, row in enumerate(arr):
        rows.append(
            "<tr><td>x={}</td>{}</tr>".format(
                row_idx,
                "".join(f"<td>{html.escape(format_value(float(v)))}</td>" for v in row),
            )
        )
    return f'<div class="tablewrap"><table>{header}{"".join(rows)}</table></div>'


def generator_counts_by_degree(record: dict[str, Any]) -> dict[tuple[int, int], dict[str, int]]:
    presentation = record.get("bigraded_chain_presentation", {})
    generators = presentation.get("generators", {}) if isinstance(presentation, dict) else {}
    counts: dict[tuple[int, int], dict[str, int]] = {}
    for dim_name, items in generators.items():
        if not isinstance(items, list):
            continue
        key = f"C{dim_name}"
        for item in items:
            if not isinstance(item, dict):
                continue
            degree = item.get("degree", [0, 0])
            if not isinstance(degree, list) or len(degree) != 2:
                continue
            point = (int(degree[0]), int(degree[1]))
            counts.setdefault(point, {"C0": 0, "C1": 0, "C2": 0})
            counts[point][key] = counts[point].get(key, 0) + 1
    return counts


def generator_points_for_chain_degree(record: dict[str, Any], chain_degree: str) -> list[tuple[int, int]]:
    presentation = record.get("bigraded_chain_presentation", {})
    generators = presentation.get("generators", {}) if isinstance(presentation, dict) else {}
    raw = generators.get(chain_degree, [])
    points: list[tuple[int, int]] = []
    if not isinstance(raw, list):
        return points
    for item in raw:
        if not isinstance(item, dict):
            continue
        degree = item.get("degree", [0, 0])
        if isinstance(degree, list) and len(degree) == 2:
            points.append((int(degree[0]), int(degree[1])))
    return sorted(set(points))


def ph_signature_vector(record: dict[str, Any]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for dim_key in sorted(record.get("vectorizations", {}), key=lambda item: int(item) if str(item).isdigit() else 999):
        payload = record.get("vectorizations", {}).get(dim_key, {})
        if not isinstance(payload, dict):
            continue
        for key in ("landscape", "persistence_image", "silhouette", "entropy_vector"):
            parts.append(np.asarray(payload.get(key, []), dtype=float).reshape(-1))
        parts.append(
            np.asarray(
                [
                    float(payload.get("count", 0.0)),
                    float(payload.get("total_persistence", 0.0)),
                    float(payload.get("max_persistence", 0.0)),
                    float(payload.get("mean_persistence", 0.0)),
                    float(payload.get("persistence_entropy", 0.0)),
                    float(payload.get("landscape_norm", 0.0)),
                    float(payload.get("persistence_image_norm", 0.0)),
                    float(payload.get("silhouette_norm", 0.0)),
                    float(payload.get("entropy_vector_norm", 0.0)),
                ],
                dtype=float,
            )
        )
    return np.concatenate(parts).astype(float) if parts else np.zeros((0,), dtype=float)


def exact_internal_simplicial_valid_fraction(record: dict[str, Any]) -> float:
    value = (
        record.get("two_parameter_module", {})
        .get("structure_map_summary", {})
        .get("mean_simplicial_map_valid_fraction", 0.0)
    )
    try:
        return float(value)
    except Exception:
        return 0.0


def add_ph_retrieval_candidates(records: list[dict[str, Any]], *, top_k: int = 5) -> list[dict[str, Any]]:
    signatures = [ph_signature_vector(record) for record in records]
    norms = [float(np.linalg.norm(sig)) for sig in signatures]
    for i, record in enumerate(records):
        candidates: list[dict[str, Any]] = []
        source_valid = exact_internal_simplicial_valid_fraction(record)
        for j, other in enumerate(records):
            if i == j:
                continue
            if signatures[i].shape != signatures[j].shape or norms[i] <= 0.0 or norms[j] <= 0.0:
                cosine = 0.0
            else:
                cosine = float(np.dot(signatures[i], signatures[j]) / max(norms[i] * norms[j], 1e-12))
            target_valid = exact_internal_simplicial_valid_fraction(other)
            simplicial_score = float((source_valid + target_valid) / 2.0)
            score = float(0.72 * cosine + 0.28 * simplicial_score)
            candidates.append(
                {
                    "record_id": other.get("record_id"),
                    "ph_signature_cosine": cosine,
                    "exact_internal_simplicial_map_score": simplicial_score,
                    "retrieval_score": score,
                    "source_signature_norm": norms[i],
                    "candidate_signature_norm": norms[j],
                }
            )
        candidates.sort(key=lambda item: item["retrieval_score"], reverse=True)
        record["ph_retrieval_candidates"] = candidates[: max(1, int(top_k))]
        record["ph_retrieval_signature"] = {
            "schema": "toricgt.gudhi_ph_retrieval_signature.v1",
            "provenance": "exact_gudhi_vectorizers_and_exact_simplicial_map_audits",
            "signature_norm": norms[i],
            "signature_length": int(signatures[i].shape[0]),
            "exact_internal_simplicial_map_valid_fraction": source_valid,
            "top_k": int(max(1, top_k)),
        }
    return records


def staircase_inner_corners(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    corners: list[tuple[int, int]] = []
    for point in sorted(set(points)):
        x, y = point
        dominated = any((u <= x and v <= y and (u, v) != point) for u, v in points)
        if not dominated:
            corners.append(point)
    return corners


def adjacent_lcm_corners(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(set(points), key=lambda item: (item[0], -item[1]))
    return sorted({(max(a[0], b[0]), max(a[1], b[1])) for a, b in zip(ordered, ordered[1:])})


def staircase_visualization_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Persist the finite Miller-Sturmfels-style staircase data used by HTML."""

    c1_corners = staircase_inner_corners(generator_points_for_chain_degree(record, "1"))
    c2_corners = staircase_inner_corners(generator_points_for_chain_degree(record, "2"))
    lcms = adjacent_lcm_corners(c1_corners)
    module = record.get("two_parameter_module", {})
    h0 = finite_matrix(module.get("hilbert_h0", []))
    all_points = [*generator_counts_by_degree(record).keys(), *c1_corners, *c2_corners, *lcms]
    max_x = max([h0.shape[0] - 1, *[point[0] for point in all_points]] or [0])
    max_y = max([h0.shape[1] - 1, *[point[1] for point in all_points]] or [0])

    def in_c1_ideal(x_level: int, y_radius: int) -> bool:
        return any(a <= x_level and b <= y_radius for a, b in c1_corners)

    quotient_points = [
        [int(x), int(y)]
        for x in range(max_x + 1)
        for y in range(max_y + 1)
        if not in_c1_ideal(x, y)
    ]
    staircase_thresholds = []
    for x in range(max_x + 1):
        ys = [b for a, b in c1_corners if a <= x]
        staircase_thresholds.append([int(x), int(min(ys) if ys else max_y + 1)])
    return {
        "schema": "toricgt.miller_sturmfels_staircase_visualization.v1",
        "ring": "F2[x_level,y_radius]",
        "ideal_chain_degree": "C1",
        "inner_corners": [[int(a), int(b)] for a, b in c1_corners],
        "higher_syzygy_corners": [[int(a), int(b)] for a, b in c2_corners],
        "adjacent_lcm_outer_corners": [[int(a), int(b)] for a, b in lcms],
        "quotient_lattice_points": quotient_points,
        "staircase_thresholds_by_x_level": staircase_thresholds,
        "layer_height_offsets_px": {"C0": 0, "C1": -7, "C2": -14, "adjacent_lcm": -21, "syzygy_edge": -17},
        "bounded_window": {"x_level_max": int(max_x), "y_radius_max": int(max_y)},
    }


def xy_grid_svg(record: dict[str, Any]) -> str:
    """Render actual bigraded generator degrees as a Miller-Sturmfels staircase."""

    counts = generator_counts_by_degree(record)
    c1_corners = staircase_inner_corners(generator_points_for_chain_degree(record, "1"))
    c1_lcms = adjacent_lcm_corners(c1_corners)
    module = record.get("two_parameter_module", {})
    h0 = finite_matrix(module.get("hilbert_h0", []))
    h1 = finite_matrix(module.get("hilbert_h1", []))
    all_points = [*counts.keys(), *c1_corners, *c1_lcms]
    max_x = max([h0.shape[0] - 1, h1.shape[0] - 1, *[point[0] for point in all_points]] or [0])
    max_y = max([h0.shape[1] - 1, h1.shape[1] - 1, *[point[1] for point in all_points]] or [0])
    cell = 54
    pad_l = 68
    pad_b = 56
    pad_t = 24
    pad_r = 180
    width = pad_l + (max_y + 1) * cell + pad_r
    height = pad_t + (max_x + 1) * cell + pad_b
    max_h0 = float(np.max(h0)) if h0.size else 0.0

    def x_pos(y: int) -> float:
        return pad_l + y * cell + cell / 2

    def y_pos(x: int) -> float:
        return pad_t + (max_x - x) * cell + cell / 2

    def in_c1_ideal(x_level: int, y_radius: int) -> bool:
        return any(a <= x_level and b <= y_radius for a, b in c1_corners)

    parts = [
        f'<svg class="xygrid" viewBox="0 0 {width} {height}" role="img" aria-label="Miller-Sturmfels staircase module view">',
        '<defs><linearGradient id="orthantFill" x1="0" x2="1" y1="0" y2="1"><stop offset="0%" stop-color="#ffb86b" stop-opacity="0.22"/><stop offset="100%" stop-color="#ff4fd8" stop-opacity="0.12"/></linearGradient></defs>',
        '<rect x="0" y="0" width="100%" height="100%" fill="#020713"/>',
    ]
    for gen_idx, (a, b) in enumerate(c1_corners):
        rx = pad_l + b * cell
        ry = pad_t
        rw = (max_y - b + 1) * cell
        rh = (max_x - a + 1) * cell
        parts.append(
            f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" '
            f'fill="url(#orthantFill)" stroke="rgba(255,184,107,0.25)" stroke-width="1" '
            f'opacity="{max(0.18, 0.34 - 0.035 * gen_idx):.3f}"/>'
        )
    for x in range(max_x + 1):
        for y in range(max_y + 1):
            value = float(h0[x, y]) if x < h0.shape[0] and y < h0.shape[1] else 0.0
            alpha = 0.08 + (0.34 * value / max(max_h0, 1.0))
            px = pad_l + y * cell
            py = pad_t + (max_x - x) * cell
            parts.append(
                f'<rect x="{px:.1f}" y="{py:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
                f'fill="rgba(55,232,255,{alpha:.3f})" stroke="rgba(145,168,183,0.20)"/>'
            )
            if not in_c1_ideal(x, y):
                parts.append(
                    f'<circle cx="{x_pos(y):.1f}" cy="{y_pos(x):.1f}" r="3.8" '
                    'fill="#e8fbff" opacity="0.88"/>'
                )
            if value:
                parts.append(
                    f'<text x="{x_pos(y):.1f}" y="{y_pos(x)+4:.1f}" text-anchor="middle" '
                    f'font-size="12" fill="#e8fbff">H0 {format_value(value)}</text>'
                )
    if c1_corners:
        threshold_points = []
        for x in range(max_x + 1):
            ys = [b for a, b in c1_corners if a <= x]
            if not ys:
                continue
            threshold = min(ys)
            threshold_points.append((pad_l + threshold * cell, y_pos(x)))
        line_points = " ".join(f"{px:.1f},{py:.1f}" for px, py in threshold_points)
        parts.append(
            f'<polyline points="{line_points}" fill="none" stroke="#ffb86b" stroke-width="2.4" '
            'stroke-linejoin="round" opacity="0.96"/>'
        )
        for (a_x, a_y), (b_x, b_y) in zip(c1_corners, c1_corners[1:]):
            parts.append(
                f'<line x1="{x_pos(a_y):.1f}" y1="{y_pos(a_x):.1f}" '
                f'x2="{x_pos(b_y):.1f}" y2="{y_pos(b_x):.1f}" '
                'stroke="#ffd39b" stroke-width="1.3" opacity="0.72"/>'
            )
    for x, y in c1_lcms:
        cx, cy = x_pos(y), y_pos(x) - 21
        parts.append(
            f'<polygon points="{cx:.1f},{cy-9:.1f} {cx+9:.1f},{cy:.1f} {cx:.1f},{cy+9:.1f} {cx-9:.1f},{cy:.1f}" '
            'fill="none" stroke="#8cff6a" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{cy+23:.1f}" text-anchor="middle" font-size="10" fill="#8cff6a">lcm</text>'
        )
    for x in range(max_x + 1):
        parts.append(
            f'<text x="{pad_l-12}" y="{y_pos(x)+4:.1f}" text-anchor="end" font-size="12" fill="#91a8b7">x={x}</text>'
        )
    for y in range(max_y + 1):
        parts.append(
            f'<text x="{x_pos(y):.1f}" y="{height-24}" text-anchor="middle" font-size="12" fill="#91a8b7">y={y}</text>'
        )
    marker_specs = {
        "C0": ("#37e8ff", "circle"),
        "C1": ("#ffb86b", "rect"),
        "C2": ("#ff4fd8", "tri"),
    }
    for (x, y), dim_counts in sorted(counts.items()):
        cx, cy = x_pos(y), y_pos(x)
        offsets = {"C0": -12, "C1": 0, "C2": 12}
        heights = {"C0": 0, "C1": -7, "C2": -14}
        for name, count in dim_counts.items():
            if count <= 0:
                continue
            color, shape = marker_specs.get(name, ("#ffffff", "circle"))
            ox = offsets.get(name, 0)
            hy = heights.get(name, 0)
            parts.append(
                f'<line x1="{cx+ox:.1f}" y1="{cy:.1f}" x2="{cx+ox:.1f}" y2="{cy+hy:.1f}" '
                f'stroke="{color}" stroke-width="1" opacity="0.34"/>'
            )
            if shape == "circle":
                parts.append(f'<circle cx="{cx+ox:.1f}" cy="{cy+hy-14:.1f}" r="6" fill="{color}"/>')
            elif shape == "rect":
                parts.append(f'<rect x="{cx+ox-6:.1f}" y="{cy+hy-20:.1f}" width="12" height="12" fill="{color}"/>')
            else:
                pts = f"{cx+ox:.1f},{cy+hy-22:.1f} {cx+ox-7:.1f},{cy+hy-8:.1f} {cx+ox+7:.1f},{cy+hy-8:.1f}"
                parts.append(f'<polygon points="{pts}" fill="{color}"/>')
            parts.append(
                f'<text x="{cx+ox:.1f}" y="{cy+hy-25:.1f}" text-anchor="middle" font-size="10" fill="#e8fbff">{count}</text>'
            )
    legend_x = pad_l + (max_y + 1) * cell + 24
    legend = [
        ("C0 vertices", "#37e8ff", "circle"),
        ("C1 edges", "#ffb86b", "rect"),
        ("C2 triangles", "#ff4fd8", "tri"),
        ("quotient lattice points", "#e8fbff", "circle"),
        ("C1 shifted orthants", "#ffb86b", "rect"),
        ("staircase boundary", "#ffd39b", "line"),
        ("adjacent lcm corners", "#8cff6a", "diamond"),
    ]
    parts.append(f'<text x="{legend_x}" y="42" font-size="13" fill="#e8fbff">Bigraded generators</text>')
    for idx, (label, color, shape) in enumerate(legend):
        ly = 68 + idx * 24
        if shape == "circle":
            parts.append(f'<circle cx="{legend_x+8}" cy="{ly-4}" r="6" fill="{color}"/>')
        elif shape == "tri":
            parts.append(
                f'<polygon points="{legend_x+8},{ly-12} {legend_x+1},{ly+2} {legend_x+15},{ly+2}" fill="{color}"/>'
            )
        elif shape == "line":
            parts.append(f'<line x1="{legend_x+1}" y1="{ly-5}" x2="{legend_x+18}" y2="{ly-5}" stroke="{color}" stroke-width="2.4" stroke-dasharray="5 5"/>')
        elif shape == "diamond":
            parts.append(
                f'<polygon points="{legend_x+9},{ly-14} {legend_x+18},{ly-5} {legend_x+9},{ly+4} {legend_x},{ly-5}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
        else:
            parts.append(f'<rect x="{legend_x+2}" y="{ly-12}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+26}" y="{ly}" font-size="12" fill="#91a8b7">{html.escape(label)}</text>')
    parts.append(f'<text x="{pad_l + (max_y + 1) * cell / 2:.1f}" y="{height-6}" text-anchor="middle" font-size="12" fill="#91a8b7">y_radius degree</text>')
    parts.append(f'<text x="18" y="{pad_t + (max_x + 1) * cell / 2:.1f}" text-anchor="middle" font-size="12" fill="#91a8b7" transform="rotate(-90 18 {pad_t + (max_x + 1) * cell / 2:.1f})">x_level degree</text>')
    parts.append("</svg>")
    return "".join(parts)


def structure_map_svg(record: dict[str, Any]) -> str:
    module = record.get("two_parameter_module", {})
    levels = module.get("levels", [])
    radii = module.get("radii", [])
    maps = module.get("structure_maps", [])
    if not isinstance(levels, list) or not isinstance(radii, list) or not isinstance(maps, list):
        return ""
    rows = max(1, len(levels))
    cols = max(1, len(radii))
    cell = 74
    pad_l = 70
    pad_t = 42
    pad_r = 190
    pad_b = 54
    width = pad_l + cols * cell + pad_r
    height = pad_t + rows * cell + pad_b

    def x_pos(radius_idx: int) -> float:
        return pad_l + radius_idx * cell + cell / 2

    def y_pos(level_idx: int) -> float:
        return pad_t + (rows - 1 - level_idx) * cell + cell / 2

    h0 = finite_matrix(module.get("hilbert_h0", []))
    h1 = finite_matrix(module.get("hilbert_h1", []))
    max_h0 = float(np.max(h0)) if h0.size else 1.0
    parts = [
        f'<svg class="xygrid" viewBox="0 0 {width} {height}" role="img" aria-label="Exact two-parameter structure-map view">',
        '<defs><marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#37e8ff"/></marker></defs>',
        '<rect x="0" y="0" width="100%" height="100%" fill="#020713"/>',
    ]
    for li in range(rows):
        for ri in range(cols):
            betti0 = float(h0[li, ri]) if li < h0.shape[0] and ri < h0.shape[1] else 0.0
            betti1 = float(h1[li, ri]) if li < h1.shape[0] and ri < h1.shape[1] else 0.0
            cx, cy = x_pos(ri), y_pos(li)
            alpha = 0.10 + 0.32 * betti0 / max(max_h0, 1.0)
            parts.append(
                f'<rect x="{cx-22:.1f}" y="{cy-22:.1f}" width="44" height="44" rx="7" '
                f'fill="rgba(55,232,255,{alpha:.3f})" stroke="rgba(145,168,183,0.45)"/>'
            )
            parts.append(f'<text x="{cx:.1f}" y="{cy-2:.1f}" text-anchor="middle" font-size="11" fill="#e8fbff">H0 {format_value(betti0)}</text>')
            parts.append(f'<text x="{cx:.1f}" y="{cy+13:.1f}" text-anchor="middle" font-size="11" fill="#ffb7ed">H1 {format_value(betti1)}</text>')
    for item in maps:
        if not isinstance(item, dict):
            continue
        src = item.get("from", [0, 0])
        dst = item.get("to", [0, 0])
        if not (isinstance(src, list) and len(src) == 2 and isinstance(dst, list) and len(dst) == 2):
            continue
        sx, sy = x_pos(int(src[1])), y_pos(int(src[0]))
        tx, ty = x_pos(int(dst[1])), y_pos(int(dst[0]))
        valid = float(item.get("simplicial", {}).get("simplicial_map_valid_fraction", 0.0))
        color = "#37e8ff" if valid >= 0.999 else "#ff4fd8"
        parts.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" '
            f'stroke="{color}" stroke-width="2.2" opacity="0.85" marker-end="url(#arrowhead)"/>'
        )
    for li, level in enumerate(levels):
        parts.append(f'<text x="{pad_l-12}" y="{y_pos(li)+4:.1f}" text-anchor="end" font-size="12" fill="#91a8b7">level {html.escape(str(level))}</text>')
    for ri, _radius in enumerate(radii):
        parts.append(f'<text x="{x_pos(ri):.1f}" y="{height-22}" text-anchor="middle" font-size="12" fill="#91a8b7">r{ri}</text>')
    legend_x = pad_l + cols * cell + 22
    parts.append(f'<text x="{legend_x}" y="54" font-size="13" fill="#e8fbff">Exact maps</text>')
    for idx, (label, color) in enumerate((("valid simplicial map", "#37e8ff"), ("invalid map", "#ff4fd8"), ("node fill = H0 rank", "#1b6c7c"))):
        ly = 82 + idx * 26
        parts.append(f'<line x1="{legend_x}" y1="{ly-4}" x2="{legend_x+24}" y2="{ly-4}" stroke="{color}" stroke-width="2.2"/>')
        parts.append(f'<text x="{legend_x+32}" y="{ly}" font-size="12" fill="#91a8b7">{html.escape(label)}</text>')
    parts.append(f'<text x="{pad_l + cols * cell / 2:.1f}" y="{height-5}" text-anchor="middle" font-size="12" fill="#91a8b7">y_radius filtration index</text>')
    parts.append(f'<text x="18" y="{pad_t + rows * cell / 2:.1f}" text-anchor="middle" font-size="12" fill="#91a8b7" transform="rotate(-90 18 {pad_t + rows * cell / 2:.1f})">x_level filtration</text>')
    parts.append("</svg>")
    return "".join(parts)


def record_figure(record: dict[str, Any]) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Persistence diagrams",
            "H1 persistence landscape",
            "H1 persistence image",
            "2-parameter Hilbert H1 grid",
            "Chain generators C0/C1/C2",
            "Structure-map homology ranks",
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "xy"}],
        ],
        horizontal_spacing=0.09,
        vertical_spacing=0.11,
    )
    colors = {0: "#37e8ff", 1: "#ff4fd8", 2: "#8cff6a"}
    max_coord = 1.0
    for dim in (0, 1, 2):
        diag = np.asarray(record.get("diagrams", {}).get(str(dim), []), dtype=float).reshape(-1, 2)
        diag = diag[np.isfinite(diag).all(axis=1)] if diag.size else np.zeros((0, 2))
        if diag.size:
            max_coord = max(max_coord, float(np.max(diag)))
            fig.add_trace(
                go.Scatter(
                    x=diag[:, 0],
                    y=diag[:, 1],
                    mode="markers",
                    name=f"H{dim}",
                    marker={"color": colors[dim], "size": 8, "line": {"color": "white", "width": 0.5}},
                ),
                row=1,
                col=1,
            )
    fig.add_trace(
        go.Scatter(x=[0, max_coord], y=[0, max_coord], mode="lines", name="diagonal", line={"color": "#91a8b7", "dash": "dot"}),
        row=1,
        col=1,
    )

    landscape = np.asarray(record.get("vectorizations", {}).get("1", {}).get("landscape", []), dtype=float)
    layers = max(1, int(round(len(landscape) / max(1, len(record.get("vectorizations", {}).get("1", {}).get("silhouette", []))))))
    if landscape.size:
        resolution = max(1, landscape.size // max(1, layers))
        landscape = landscape[: layers * resolution].reshape(layers, resolution)
        x = np.linspace(0.0, 1.0, resolution)
        for idx, row in enumerate(landscape[: min(5, layers)]):
            fig.add_trace(go.Scatter(x=x, y=row, mode="lines", name=f"landscape {idx+1}"), row=1, col=2)

    image = np.asarray(record.get("vectorizations", {}).get("1", {}).get("persistence_image", []), dtype=float)
    if image.size:
        side = int(round(np.sqrt(image.size)))
        fig.add_trace(go.Heatmap(z=image[: side * side].reshape(side, side), colorscale="Viridis", name="H1 PI"), row=2, col=1)

    module = record.get("two_parameter_module", {})
    h1 = finite_matrix(module.get("hilbert_h1", [[0.0]]))
    fig.add_trace(go.Heatmap(z=h1, colorscale="Blues", name="Hilbert H1"), row=2, col=2)

    c0 = finite_matrix(module.get("chain_generators_c0", [[0.0]]))
    c1 = finite_matrix(module.get("chain_generators_c1", [[0.0]]))
    c2 = finite_matrix(module.get("chain_generators_c2", [[0.0]]))
    stacked = np.concatenate([c0, c1, c2], axis=0)
    fig.add_trace(go.Heatmap(z=stacked, colorscale="Turbo", name="chain generators"), row=3, col=1)

    maps = module.get("structure_maps", [])
    xs = list(range(len(maps)))
    fig.add_trace(go.Scatter(x=xs, y=[m.get("h0_rank", 0) for m in maps], mode="lines+markers", name="H0 map rank"), row=3, col=2)
    fig.add_trace(go.Scatter(x=xs, y=[m.get("h1_rank", 0) for m in maps], mode="lines+markers", name="H1 map rank"), row=3, col=2)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#030712",
        plot_bgcolor="#07111f",
        font={"color": "#e8fbff"},
        height=980,
        margin={"l": 50, "r": 30, "t": 90, "b": 40},
        legend={"orientation": "h", "y": -0.07},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#143344")
    fig.update_yaxes(showgrid=True, gridcolor="#143344")
    return fig


def metric_rows(record: dict[str, Any]) -> str:
    resolution = record.get("macaulay2_resolution", {})
    chain = record.get("chain_complex", {})
    finite_chain = record.get("finite_field_chain_audit", {})
    structure_maps = record.get("two_parameter_module", {}).get("structure_map_summary", {})
    metrics = {
        "points": record.get("points"),
        "ambient dimension": record.get("dimension"),
        "max radius": float(record.get("max_radius", 0.0)),
        "GUDHI simplices": record.get("simplex_tree", {}).get("num_simplices"),
        "Betti": chain.get("betti"),
        "GF(2) d^2=0": finite_chain.get("d_squared_zero"),
        "GF(2) exact at C1": finite_chain.get("exact_at_c1"),
        "BE rank residual C1": finite_chain.get("buchsbaum_eisenbud_rank_residual_c1"),
        "simplicial maps valid fraction": structure_maps.get("mean_simplicial_map_valid_fraction"),
        "M2 homogeneous d1": resolution.get("homogeneous_d1"),
        "M2 homogeneous d2": resolution.get("homogeneous_d2"),
        "M2 d^2=0": resolution.get("d_squared_zero"),
        "F2[x,y] C0/C1/C2": (
            resolution.get("num_C0_generators"),
            resolution.get("num_C1_generators"),
            resolution.get("num_C2_generators"),
        ),
    }
    return "\n".join(
        f'<div class="metric"><span>{html.escape(str(k))}</span><span>{html.escape(format_value(v))}</span></div>'
        for k, v in metrics.items()
    )


def module_check_badges(record: dict[str, Any]) -> str:
    resolution = record.get("macaulay2_resolution", {})
    module = record.get("two_parameter_module", {})
    finite_chain = record.get("finite_field_chain_audit", {})
    chain_residuals = module.get("commutative_square_chain_residuals_by_dimension", {})
    return "".join(
        [
            pass_badge("GF(2) d^2=0", finite_chain.get("d_squared_zero", False)),
            pass_badge("GF(2) exact at C1", finite_chain.get("exact_at_c1", False)),
            pass_badge("M2 homogeneous d1", resolution.get("homogeneous_d1", False)),
            pass_badge("M2 homogeneous d2", resolution.get("homogeneous_d2", False)),
            pass_badge("M2 d1*d2=0", resolution.get("d_squared_zero", False)),
            pass_badge("2-param square residual", float(module.get("commutative_square_residual", 1.0) or 0.0) == 0.0),
            pass_badge(
                "chain square residuals",
                isinstance(chain_residuals, dict) and all(int(value) == 0 for value in chain_residuals.values()),
            ),
        ]
    )


def module_tables(record: dict[str, Any]) -> str:
    module = record.get("two_parameter_module", {})
    table_specs = [
        ("Hilbert H0 grid", "hilbert_h0"),
        ("Hilbert H1 grid", "hilbert_h1"),
        ("Chain generators C0", "chain_generators_c0"),
        ("Chain generators C1", "chain_generators_c1"),
        ("Chain generators C2", "chain_generators_c2"),
    ]
    sections = []
    for title, key in table_specs:
        if key in module:
            sections.append(f"<h3>{html.escape(title)}</h3>{matrix_table(module[key])}")
    return "".join(sections)


def resolution_strip_svg(record: dict[str, Any]) -> str:
    """Render the chain complex before exposing raw Macaulay2 text."""

    presentation = record.get("bigraded_chain_presentation", {})
    generators = presentation.get("generators", {}) if isinstance(presentation, dict) else {}
    resolution = record.get("macaulay2_resolution", {})
    finite_chain = record.get("finite_field_chain_audit", {})
    ranks = {
        "C2": len(generators.get("2", [])) if isinstance(generators.get("2", []), list) else 0,
        "C1": len(generators.get("1", [])) if isinstance(generators.get("1", []), list) else 0,
        "C0": len(generators.get("0", [])) if isinstance(generators.get("0", []), list) else 0,
    }
    width, height = 980, 240
    nodes = {"C2": (170, 112), "C1": (490, 112), "C0": (810, 112)}
    colors = {"C2": "#ff4fd8", "C1": "#ffb86b", "C0": "#37e8ff"}

    def badge_text(ok: Any) -> tuple[str, str]:
        return ("PASS", "#8cff6a") if bool(ok) else ("FAIL", "#ff4fd8")

    checks = [
        ("GF(2) d1*d2", finite_chain.get("d_squared_zero", False)),
        ("exact at C1", finite_chain.get("exact_at_c1", False)),
        ("M2 homogeneous", bool(resolution.get("homogeneous_d1", False)) and bool(resolution.get("homogeneous_d2", False))),
        ("M2 d1*d2", resolution.get("d_squared_zero", False)),
    ]
    parts = [
        f'<svg class="resolution-strip" viewBox="0 0 {width} {height}" role="img" aria-label="Readable chain-complex resolution strip">',
        '<defs><marker id="resArrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="#91a8b7"/></marker></defs>',
        '<rect x="0" y="0" width="100%" height="100%" fill="#020713"/>',
        '<text x="28" y="32" fill="#e8fbff" font-size="16">Clean resolution view over F2[x_level,y_radius]</text>',
        '<text x="28" y="54" fill="#91a8b7" font-size="12">The arrows summarize the exact emitted boundary matrices before the raw Macaulay2 payload.</text>',
    ]
    for source, target, label in (("C2", "C1", "d2"), ("C1", "C0", "d1")):
        sx, sy = nodes[source]
        tx, ty = nodes[target]
        parts.append(
            f'<line x1="{sx+72}" y1="{sy}" x2="{tx-72}" y2="{ty}" stroke="#91a8b7" '
            'stroke-width="2.4" marker-end="url(#resArrow)"/>'
        )
        parts.append(
            f'<text x="{(sx+tx)/2:.1f}" y="{sy-16}" text-anchor="middle" fill="#e8fbff" font-size="13">{label}</text>'
        )
    for name, (cx, cy) in nodes.items():
        color = colors[name]
        parts.append(
            f'<rect x="{cx-70}" y="{cy-46}" width="140" height="92" rx="8" fill="rgba(11,23,40,.96)" '
            f'stroke="{color}" stroke-width="1.8"/>'
        )
        parts.append(f'<text x="{cx}" y="{cy-12}" text-anchor="middle" fill="{color}" font-size="22" font-weight="700">{name}</text>')
        parts.append(f'<text x="{cx}" y="{cy+14}" text-anchor="middle" fill="#e8fbff" font-size="13">rank {ranks[name]}</text>')
        parts.append(f'<text x="{cx}" y="{cy+34}" text-anchor="middle" fill="#91a8b7" font-size="11">bigraded free generators</text>')
    for idx, (label, ok) in enumerate(checks):
        text, color = badge_text(ok)
        bx = 50 + idx * 225
        by = 196
        parts.append(
            f'<rect x="{bx}" y="{by-20}" width="196" height="30" rx="15" fill="{color}" opacity="0.14" stroke="{color}" stroke-width="1"/>'
        )
        parts.append(f'<text x="{bx+14}" y="{by}" fill="#e8fbff" font-size="12">{html.escape(label)}</text>')
        parts.append(f'<text x="{bx+158}" y="{by}" fill="{color}" font-size="12" font-weight="700">{text}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _matrix_nonzero_weight(value: str) -> int:
    if value in {"", "0", "0_R"}:
        return 0
    x_exp = 0
    y_exp = 0
    for factor in str(value).split("*"):
        if factor == "x_level":
            x_exp += 1
        elif factor.startswith("x_level^"):
            x_exp += int(factor.split("^", 1)[1])
        elif factor == "y_radius":
            y_exp += 1
        elif factor.startswith("y_radius^"):
            y_exp += int(factor.split("^", 1)[1])
    return 1 + x_exp + y_exp


def differential_heatmap_svg(record: dict[str, Any]) -> str:
    """Render d1 and d2 monomial support over GF(2) as readable heatmaps."""

    presentation = record.get("bigraded_chain_presentation", {})
    matrices = presentation.get("boundary_matrices", {}) if isinstance(presentation, dict) else {}
    width, panel_h = 980, 260
    parts = [
        f'<svg class="differential-heatmap" viewBox="0 0 {width} {panel_h * 2}" role="img" aria-label="GF(2) differential matrix heatmaps">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#020713"/>',
    ]

    def render_panel(key: str, title: str, top: int, color: str) -> None:
        raw = matrices.get(key, [])
        rows = raw if isinstance(raw, list) else []
        row_count = max(1, len(rows))
        col_count = max(1, max((len(row) for row in rows if isinstance(row, list)), default=0))
        pad_l, pad_t, pad_r, pad_b = 74, top + 48, 24, 34
        cell = min(34.0, (width - pad_l - pad_r) / max(1, col_count), (panel_h - 88) / max(1, row_count))
        parts.append(f'<text x="28" y="{top+26}" fill="#e8fbff" font-size="15">{html.escape(title)}</text>')
        parts.append(
            f'<text x="28" y="{top+44}" fill="#91a8b7" font-size="11">Nonzero entries are the actual emitted monomial terms, reduced to GF(2) support and bidegree weight.</text>'
        )
        for r in range(row_count):
            parts.append(f'<text x="{pad_l-10}" y="{pad_t+r*cell+cell*.65:.1f}" text-anchor="end" fill="#91a8b7" font-size="10">r{r}</text>')
        for c in range(col_count):
            if c % max(1, math.ceil(col_count / 12)) == 0:
                parts.append(f'<text x="{pad_l+c*cell+cell/2:.1f}" y="{pad_t-8}" text-anchor="middle" fill="#91a8b7" font-size="10">c{c}</text>')
        for r, row in enumerate(rows):
            if not isinstance(row, list):
                continue
            for c in range(col_count):
                value = str(row[c]) if c < len(row) else "0_R"
                weight = _matrix_nonzero_weight(value)
                alpha = 0.08 if weight == 0 else min(0.90, 0.18 + 0.12 * weight)
                fill = f"rgba(145,168,183,{alpha:.3f})" if weight == 0 else color
                parts.append(
                    f'<rect x="{pad_l+c*cell:.1f}" y="{pad_t+r*cell:.1f}" width="{cell-1:.1f}" height="{cell-1:.1f}" '
                    f'fill="{fill}" opacity="{alpha:.3f}" stroke="rgba(145,168,183,0.14)"/>'
                )
                if weight > 0 and cell >= 20:
                    label = "1" if value == "1_R" else value.replace("x_level", "x").replace("y_radius", "y")
                    parts.append(
                        f'<text x="{pad_l+c*cell+cell/2:.1f}" y="{pad_t+r*cell+cell*.62:.1f}" '
                        f'text-anchor="middle" fill="#020713" font-size="{max(7, min(10, cell*.28)):.1f}" font-weight="700">{html.escape(label[:8])}</text>'
                    )
        nz = sum(1 for row in rows if isinstance(row, list) for value in row if _matrix_nonzero_weight(str(value)) > 0)
        parts.append(
            f'<text x="{width-28}" y="{top+26}" text-anchor="end" fill="{color}" font-size="13">{nz} nonzero terms</text>'
        )

    render_panel("1", "d1: C1 -> C0 differential support", 0, "#37e8ff")
    render_panel("2", "d2: C2 -> C1 differential support", panel_h, "#ff4fd8")
    parts.append("</svg>")
    return "".join(parts)


def ph_feature_norm_rows(record: dict[str, Any]) -> str:
    vectorizations = record.get("vectorizations", {})
    rows: list[str] = []
    for dim_key, payload in sorted(vectorizations.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 999):
        if not isinstance(payload, dict):
            continue
        rows.append(
            "<tr><td>H{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(str(dim_key)),
                html.escape(format_value(payload.get("count", 0))),
                html.escape(format_value(payload.get("total_persistence", 0.0))),
                html.escape(format_value(payload.get("landscape_norm", 0.0))),
                html.escape(format_value(payload.get("persistence_image_norm", 0.0))),
                html.escape(format_value(payload.get("silhouette_norm", 0.0))),
                html.escape(format_value(payload.get("entropy_vector_norm", 0.0))),
            )
        )
    if not rows:
        return ""
    return (
        '<div class="tablewrap"><table>'
        "<tr><th>homology</th><th>intervals</th><th>total persistence</th><th>landscape norm</th>"
        "<th>image norm</th><th>silhouette norm</th><th>entropy-vector norm</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )


def ph_feature_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Return the persisted vectorized PH feature payload for one record."""

    payload: dict[str, Any] = {
        "schema": "toricgt.gudhi_ph_features.v1",
        "record_id": record.get("record_id"),
        "backend": "gudhi",
        "provenance": "exact_gudhi_vectorizers",
        "radii": record.get("radii", []),
        "betti": record.get("chain_complex", {}).get("betti", {}),
        "vectorizations": record.get("vectorizations", {}),
    }
    module = record.get("two_parameter_module", {})
    if isinstance(module, dict):
        payload["betti_curves"] = {
            "h0_grid": module.get("hilbert_h0", []),
            "h1_grid": module.get("hilbert_h1", []),
            "structure_maps": module.get("structure_maps", []),
        }
    payload["ph_retrieval_signature"] = record.get("ph_retrieval_signature", {})
    payload["ph_retrieval_candidates"] = record.get("ph_retrieval_candidates", [])
    return payload


def write_ph_feature_artifacts(record: dict[str, Any], feature_dir: Path, slug: str) -> dict[str, str]:
    """Persist vectorized PH features as JSON and NPZ."""

    feature_dir.mkdir(parents=True, exist_ok=True)
    payload = ph_feature_payload(record)
    json_path = feature_dir / f"{slug}_ph_features.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    arrays: dict[str, np.ndarray] = {}
    for dim_key, vectorized in record.get("vectorizations", {}).items():
        if not isinstance(vectorized, dict):
            continue
        prefix = f"h{dim_key}"
        for name in ("landscape", "persistence_image", "silhouette", "entropy_vector"):
            arrays[f"{prefix}_{name}"] = np.asarray(vectorized.get(name, []), dtype=np.float32)
        arrays[f"{prefix}_stats"] = np.asarray(
            [
                float(vectorized.get("count", 0.0)),
                float(vectorized.get("total_persistence", 0.0)),
                float(vectorized.get("max_persistence", 0.0)),
                float(vectorized.get("mean_persistence", 0.0)),
                float(vectorized.get("persistence_entropy", 0.0)),
                float(vectorized.get("landscape_norm", 0.0)),
                float(vectorized.get("persistence_image_norm", 0.0)),
                float(vectorized.get("silhouette_norm", 0.0)),
                float(vectorized.get("entropy_vector_norm", 0.0)),
            ],
            dtype=np.float32,
        )
    module = record.get("two_parameter_module", {})
    if isinstance(module, dict):
        arrays["betti_h0_grid"] = np.asarray(module.get("hilbert_h0", []), dtype=np.float32)
        arrays["betti_h1_grid"] = np.asarray(module.get("hilbert_h1", []), dtype=np.float32)
    npz_path = feature_dir / f"{slug}_ph_features.npz"
    np.savez_compressed(npz_path, **arrays)
    return {"json": f"ph_features/{json_path.name}", "npz": f"ph_features/{npz_path.name}"}


def ph_feature_figure(record: dict[str, Any]) -> go.Figure:
    """Render all vectorized persistent-homology features in one dashboard."""

    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "Persistence landscapes",
            "Persistence image H1",
            "Silhouette functions",
            "Entropy vectors",
            "Betti curves over radius",
            "Interval lifetime histogram",
            "Vector feature norms",
            "Interval counts by homology",
        ),
        specs=[
            [{"type": "xy"}, {"type": "heatmap"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
        horizontal_spacing=0.09,
        vertical_spacing=0.10,
    )
    colors = {"0": "#37e8ff", "1": "#ff4fd8", "2": "#8cff6a"}
    vectorizations = record.get("vectorizations", {})
    for dim_key, payload in sorted(vectorizations.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 999):
        if not isinstance(payload, dict):
            continue
        color = colors.get(str(dim_key), "#ffffff")
        landscape = np.asarray(payload.get("landscape", []), dtype=float)
        silhouette = np.asarray(payload.get("silhouette", []), dtype=float)
        entropy_vec = np.asarray(payload.get("entropy_vector", []), dtype=float)
        if landscape.size and silhouette.size:
            resolution = int(silhouette.size)
            layers = max(1, landscape.size // max(1, resolution))
            landscape = landscape[: layers * resolution].reshape(layers, resolution)
            grid = np.linspace(0.0, 1.0, resolution)
            for layer_idx, row in enumerate(landscape[: min(3, layers)]):
                fig.add_trace(
                    go.Scatter(
                        x=grid,
                        y=row,
                        mode="lines",
                        name=f"H{dim_key} landscape {layer_idx + 1}",
                        line={"color": color, "width": max(1, 3 - layer_idx)},
                        opacity=max(0.35, 0.9 - 0.18 * layer_idx),
                    ),
                    row=1,
                    col=1,
                )
        if silhouette.size:
            grid = np.linspace(0.0, 1.0, silhouette.size)
            fig.add_trace(
                go.Scatter(x=grid, y=silhouette, mode="lines", name=f"H{dim_key} silhouette", line={"color": color}),
                row=2,
                col=1,
            )
        if entropy_vec.size:
            grid = np.linspace(0.0, 1.0, entropy_vec.size)
            fig.add_trace(
                go.Scatter(x=grid, y=entropy_vec, mode="lines", name=f"H{dim_key} entropy vector", line={"color": color}),
                row=2,
                col=2,
            )

    h1_image = np.asarray(vectorizations.get("1", {}).get("persistence_image", []), dtype=float)
    if h1_image.size:
        side = int(round(np.sqrt(h1_image.size)))
        if side * side == h1_image.size:
            fig.add_trace(go.Heatmap(z=h1_image.reshape(side, side), colorscale="Viridis", name="H1 persistence image"), row=1, col=2)

    module = record.get("two_parameter_module", {})
    for key, name, color in (("hilbert_h0", "H0 Betti/radius mean", "#37e8ff"), ("hilbert_h1", "H1 Betti/radius mean", "#ff4fd8")):
        grid = np.asarray(module.get(key, []), dtype=float)
        if grid.size:
            y = grid.mean(axis=0) if grid.ndim == 2 else grid.reshape(-1)
            fig.add_trace(go.Scatter(x=list(range(len(y))), y=y, mode="lines+markers", name=name, line={"color": color}), row=3, col=1)

    lifetimes: list[float] = []
    lifetimes_by_dim: dict[str, list[float]] = {}
    for dim_key, diag in record.get("diagrams", {}).items():
        arr = np.asarray(diag, dtype=float).reshape(-1, 2)
        arr = arr[np.isfinite(arr).all(axis=1)] if arr.size else np.zeros((0, 2))
        vals = [float(max(0.0, death - birth)) for birth, death in arr if death > birth]
        lifetimes.extend(vals)
        lifetimes_by_dim[str(dim_key)] = vals
    if lifetimes:
        fig.add_trace(go.Histogram(x=lifetimes, nbinsx=20, name="finite interval lifetimes", marker={"color": "#8cff6a"}), row=3, col=2)

    dims: list[str] = []
    for dim_key in sorted(vectorizations, key=lambda item: int(item) if str(item).isdigit() else 999):
        dims.append(str(dim_key))
    for norm_key, label, color in (
        ("landscape_norm", "landscape", "#37e8ff"),
        ("persistence_image_norm", "image", "#ff4fd8"),
        ("silhouette_norm", "silhouette", "#8cff6a"),
        ("entropy_vector_norm", "entropy vector", "#ffb86b"),
    ):
        fig.add_trace(
            go.Bar(
                x=[f"H{dim}" for dim in dims],
                y=[float(vectorizations.get(dim, {}).get(norm_key, 0.0)) for dim in dims],
                name=label,
                marker={"color": color},
            ),
            row=4,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=[f"H{dim}" for dim in dims],
            y=[float(vectorizations.get(dim, {}).get("count", 0.0)) for dim in dims],
            name="interval count",
            marker={"color": "#37e8ff"},
        ),
        row=4,
        col=2,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#030712",
        plot_bgcolor="#07111f",
        font={"color": "#e8fbff"},
        height=1180,
        margin={"l": 50, "r": 30, "t": 95, "b": 45},
        legend={"orientation": "h", "y": -0.06},
        barmode="group",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#143344")
    fig.update_yaxes(showgrid=True, gridcolor="#143344")
    return fig


def xy_grid_summary_html(record: dict[str, Any]) -> str:
    summary = record.get("xy_grid_module", {})
    chain_degrees = summary.get("chain_degrees", {}) if isinstance(summary, dict) else {}
    rows: list[str] = []
    for degree, payload in sorted(chain_degrees.items(), key=lambda item: item[0]):
        if not isinstance(payload, dict):
            continue
        rows.append(
            "<tr><td>C{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(str(degree)),
                html.escape(format_value(payload.get("generator_count", 0))),
                html.escape(format_value(payload.get("minimal_inner_corners", []))),
                html.escape(format_value(payload.get("adjacent_lcm_outer_corners", []))),
                html.escape(format_value(len(payload.get("adjacent_lcm_syzygies", [])))),
            )
        )
    if not rows:
        return ""
    return (
        '<div class="tablewrap"><table>'
        "<tr><th>chain</th><th>generators</th><th>minimal inner corners</th><th>outer lcm corners</th><th>adjacent syzygies</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )


def ph_retrieval_html(record: dict[str, Any]) -> str:
    signature = record.get("ph_retrieval_signature", {})
    candidates = record.get("ph_retrieval_candidates", [])
    rows: list[str] = []
    for idx, item in enumerate(candidates if isinstance(candidates, list) else []):
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                idx + 1,
                html.escape(str(item.get("record_id", ""))),
                html.escape(format_value(item.get("retrieval_score", 0.0))),
                html.escape(format_value(item.get("ph_signature_cosine", 0.0))),
                html.escape(format_value(item.get("exact_internal_simplicial_map_score", 0.0))),
            )
        )
    table = (
        '<div class="tablewrap"><table>'
        "<tr><th>rank</th><th>candidate</th><th>retrieval score</th><th>PH cosine</th><th>exact simplicial-map score</th></tr>"
        + "".join(rows)
        + "</table></div>"
        if rows
        else "<p>No retrieval candidates were available in this bundle.</p>"
    )
    return f"""
<section class="panel card">
  <h2>Analogical Memory PH Retrieval Panel</h2>
  <p>Candidate scores use persisted GUDHI vectorized PH features and exact internal simplicial-map validity from the reasoning/radius simplex-tree filtration.  No torch-only topology proxy is used in this panel.</p>
  <div class="grid">
    <div class="card">
      <h2>PH Retrieval Signature</h2>
      <div class="metric"><span>provenance</span><span>{html.escape(str(signature.get('provenance', 'n/a')))}</span></div>
      <div class="metric"><span>signature length</span><span>{html.escape(format_value(signature.get('signature_length', 0)))}</span></div>
      <div class="metric"><span>signature norm</span><span>{html.escape(format_value(signature.get('signature_norm', 0.0)))}</span></div>
      <div class="metric"><span>exact simplicial-map validity</span><span>{html.escape(format_value(signature.get('exact_internal_simplicial_map_valid_fraction', 0.0)))}</span></div>
    </div>
    <div class="card">
      <h2>Top Candidates</h2>
      {table}
    </div>
  </div>
</section>
"""


def derived_map_visual_html(record: dict[str, Any]) -> str:
    resolution = record.get("macaulay2_resolution", {})
    derived = resolution.get("derived_category_maps", {}) if isinstance(resolution.get("derived_category_maps", {}), dict) else {}
    homology = derived.get("chain_identity_mapping_cone_homology_pruned", {}) if isinstance(derived, dict) else {}
    resolution_cones = derived.get("homology_resolution_identity_cone_homology_pruned", {}) if isinstance(derived, dict) else {}
    cone_rows = []
    for key in ("H0", "H1", "H2"):
        value = str(homology.get(key, ""))
        cone_rows.append(
            f'<div class="metric"><span>chain identity cone {html.escape(key)}</span><span>{html.escape(value or "n/a")}</span></div>'
        )
    resolution_rows = []
    if isinstance(resolution_cones, dict):
        for hom_key, payload in sorted(resolution_cones.items()):
            resolution_rows.append(
                f'<div class="metric"><span>resolution identity cone {html.escape(str(hom_key))}</span><span>{html.escape(format_value(payload))}</span></div>'
            )
    return f"""
<section class="panel card">
  <h2>Exact Module Maps And Derived Identity-Cones</h2>
  <p>The arrows below are the actual two-parameter inclusion maps.  Macaulay2 then checks the bigraded differential and the identity mapping cone in the derived category.  Acyclic identity cones are the expected sanity check for the emitted complex and its homology resolutions.</p>
  {structure_map_svg(record)}
  <div class="grid">
    <div class="card">
      <h2>Identity Cone Acyclicity</h2>
      {''.join(cone_rows)}
    </div>
    <div class="card">
      <h2>Resolution-Level Identity Cones</h2>
      {''.join(resolution_rows) if resolution_rows else '<p>No resolution-level identity-cone payload.</p>'}
    </div>
  </div>
</section>
"""


def write_record_html(
    record: dict[str, Any],
    out: Path,
    *,
    rel_json: str,
    rel_m2: str,
    rel_ph_json: str = "",
    rel_ph_npz: str = "",
    emit_ph_feature_visualizations: bool = True,
) -> None:
    fig_html = record_figure(record).to_html(include_plotlyjs="cdn", full_html=False, div_id=f"plot_{safe_slug(record['record_id'])}")
    ph_fig_html = (
        ph_feature_figure(record).to_html(include_plotlyjs="cdn", full_html=False, div_id=f"ph_features_{safe_slug(record['record_id'])}")
        if bool(emit_ph_feature_visualizations)
        else ""
    )
    resolution = record.get("macaulay2_resolution", {})
    hom = resolution.get("homology_and_resolutions", {}) if isinstance(resolution.get("homology_and_resolutions", {}), dict) else {}
    derived = resolution.get("derived_category_maps", {}) if isinstance(resolution.get("derived_category_maps", {}), dict) else {}
    ph_links = ""
    if rel_ph_json:
        ph_links += f'<a class="pill" href="{html.escape(rel_ph_json)}">PH feature JSON</a>'
    if rel_ph_npz:
        ph_links += f'<a class="pill" href="{html.escape(rel_ph_npz)}">PH feature NPZ</a>'
    ph_section = (
        f"""
<section class="panel card">
  <h2>Vectorized Persistent-Homology Feature Dashboard</h2>
  <p>Exact GUDHI vectorizers for persistence landscapes, persistence images, silhouettes, entropy vectors, Betti curves, interval lifetimes, and feature norms.  These are persisted for training and analogical-memory retrieval; they are not torch topology fallbacks.</p>
  {ph_feature_norm_rows(record)}
  <div class="plot">{ph_fig_html}</div>
</section>
"""
        if bool(emit_ph_feature_visualizations)
        else """
<section class="panel card">
  <h2>Vectorized Persistent-Homology Features</h2>
  <p>Visualization disabled by CLI flag.  Exact feature JSON/NPZ artifacts may still be emitted when feature persistence is enabled.</p>
</section>
"""
    )
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(record['record_id'])} GUDHI persistence audit</title><style>{CSS}</style></head>
<body><main>
<section class="hero">
  <h1>{html.escape(record['record_id'])}</h1>
  <p>Exact GUDHI simplex-tree persistence, vectorized PH, F2[x_level,y_radius] module maps, and Macaulay2 free-resolution output for a reasoning/radius trajectory filtration.</p>
  <div class="links">
    <a class="pill" href="../index.html">index</a>
    <a class="pill" href="{html.escape(rel_json)}">record JSON</a>
    <a class="pill" href="{html.escape(rel_m2)}">Macaulay2 script</a>
    {ph_links}
  </div>
</section>
<section class="grid">
  <div class="card"><h2>Core Metrics</h2>{metric_rows(record)}</div>
  <div class="card"><h2>Module</h2>
    <div>{module_check_badges(record)}</div>
    <p>Ring: <code>{html.escape(str(record.get('two_parameter_module', {}).get('ring', 'F2[x_level,y_radius]')))}</code></p>
    <p>Commutative squares checked: <code>{html.escape(str(record.get('two_parameter_module', {}).get('commutative_squares_checked', 'n/a')))}</code></p>
    <p>Square residual: <code>{html.escape(str(record.get('two_parameter_module', {}).get('commutative_square_residual', 'n/a')))}</code></p>
    <p class="legend">The xy-grid below follows the bivariate monomial-ideal convention: x is reasoning level, y is radius degree, shaded regions are shifted positive orthants from C1 monomial generators, white dots are quotient lattice points, and the orange line is the Miller-Sturmfels staircase boundary.  Green corners are adjacent lcm corners computed from the emitted bidegrees.</p>
  </div>
</section>
<section class="panel card">
  <h2>F2[x_level,y_radius] Miller-Sturmfels Staircase Module View</h2>
  {xy_grid_svg(record)}
  {xy_grid_summary_html(record)}
  {module_tables(record)}
</section>
<section class="panel card">
  <h2>Readable Resolution And Differential Summary</h2>
  {resolution_strip_svg(record)}
  {differential_heatmap_svg(record)}
</section>
{derived_map_visual_html(record)}
{ph_section}
{ph_retrieval_html(record)}
<section class="panel card">
  <h2>Exact Chain And Simplicial-Map Audit</h2>
  <details open><summary>GF(2) chain exactness and Buchsbaum-Eisenbud-style rank check</summary>
  <pre>{html.escape(json.dumps(record.get('finite_field_chain_audit', {}), indent=2, sort_keys=True))}</pre>
  </details>
  <details open><summary>Reasoning/radius structure-map summary</summary>
  <pre>{html.escape(json.dumps(record.get('two_parameter_module', {}).get('structure_map_summary', {}), indent=2, sort_keys=True))}</pre>
  </details>
</section>
<section class="panel plot">{fig_html}</section>
<section class="panel card">
  <h2>Macaulay2 Homology Modules And Free Resolutions</h2>
  <details open><summary>Homology modules and free resolutions</summary>
  <pre>{html.escape(json.dumps(hom, indent=2, sort_keys=True))}</pre>
  </details>
  <details open><summary>Derived maps, identity mapping cones, Ext, and Tor modules</summary>
  <pre>{html.escape(json.dumps(derived, indent=2, sort_keys=True))}</pre>
  </details>
  <details><summary>Full Macaulay2 resolution payload</summary>
  <pre>{html.escape(json.dumps(resolution, indent=2, sort_keys=True))}</pre>
  </details>
</section>
</main></body></html>
"""
    out.write_text(body, encoding="utf-8")


def write_index(output_dir: Path, summary: dict[str, Any], record_pages: list[dict[str, Any]]) -> Path:
    cards = []
    for page in record_pages:
        badges = "".join(
            [
                pass_badge("M2 d1*d2=0", page.get("m2_d_squared_zero", False)),
                pass_badge("homogeneous", page.get("m2_homogeneous", False)),
                pass_badge("square residual", float(page.get("square_residual", 1.0) or 0.0) == 0.0),
            ]
        )
        metrics = "".join(
            [
                f'<div class="metric"><span>simplices</span><span>{html.escape(format_value(page.get("num_simplices", "n/a")))}</span></div>',
                f'<div class="metric"><span>Betti</span><span>{html.escape(format_value(page.get("betti", "n/a")))}</span></div>',
                f'<div class="metric"><span>H1 landscape norm</span><span>{html.escape(format_value(page.get("h1_landscape_norm", "n/a")))}</span></div>',
                f'<div class="metric"><span>H1 image norm</span><span>{html.escape(format_value(page.get("h1_persistence_image_norm", "n/a")))}</span></div>',
                f'<div class="metric"><span>H1 silhouette norm</span><span>{html.escape(format_value(page.get("h1_silhouette_norm", "n/a")))}</span></div>',
                f'<div class="metric"><span>H1 entropy-vector norm</span><span>{html.escape(format_value(page.get("h1_entropy_vector_norm", "n/a")))}</span></div>',
                f'<div class="metric"><span>GF(2) exact at C1</span><span>{html.escape(format_value(page.get("finite_field_exact_at_c1", "n/a")))}</span></div>',
                f'<div class="metric"><span>simplicial-map validity</span><span>{html.escape(format_value(page.get("simplicial_map_valid_fraction", "n/a")))}</span></div>',
            ]
        )
        feature_links = ""
        if page.get("ph_feature_json"):
            feature_links += f'<a class="pill" href="{html.escape(str(page["ph_feature_json"]))}">PH JSON</a>'
        if page.get("ph_feature_npz"):
            feature_links += f'<a class="pill" href="{html.escape(str(page["ph_feature_npz"]))}">PH NPZ</a>'
        cards.append(
            f"""<div class="card">
  <h2>{html.escape(page['record_id'])}</h2>
  <p>Exact GUDHI/Macaulay2 reasoning-radius persistence audit.</p>
  <div>{badges}</div>
  {metrics}
  <div class="links"><a class="pill" href="{html.escape(page['html'])}">open page</a><a class="pill" href="{html.escape(page['json'])}">JSON</a><a class="pill" href="{html.escape(page['m2'])}">M2 script</a>{feature_links}</div>
</div>"""
        )
    metrics = "\n".join(
        f'<div class="metric"><span>{html.escape(str(k))}</span><span>{html.escape(format_value(v))}</span></div>'
        for k, v in summary.items()
        if k != "records_detail"
    )
    index = output_dir / "index.html"
    index.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ToricGT GUDHI Persistence Audit</title><style>{CSS}</style></head>
<body><main>
<section class="hero">
  <h1>ToricGT GUDHI Persistence Audit</h1>
  <p>Exact simplex-tree persistent homology, vectorized PH metrics, simplicial maps by radius, and Macaulay2 F2[x_level,y_radius] resolutions for reasoning trajectories.</p>
  <div class="links"><a class="pill" href="summary.json">summary JSON</a><a class="pill" href="records.json">records JSON</a></div>
</section>
<section class="grid"><div class="card"><h2>Summary</h2>{metrics}</div>{''.join(cards)}</section>
</main></body></html>
""",
        encoding="utf-8",
    )
    return index


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    records_dir = output_dir / "records"
    scripts_dir = output_dir / "macaulay2"
    ph_features_dir = output_dir / "ph_features"
    records_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    ph_features_dir.mkdir(parents=True, exist_ok=True)

    cfg = GudhiPersistenceConfig(
        max_points=max(4, int(args.max_points)),
        max_dimension=max(1, int(args.max_dimension)),
        radius_quantile=float(args.radius_quantile),
        num_radii=max(2, int(args.num_radii)),
        num_levels=max(2, int(args.num_levels)),
        landscape_resolution=max(8, int(args.landscape_resolution)),
        landscape_layers=max(1, int(args.landscape_layers)),
        image_resolution=max(4, int(args.image_resolution)),
        macaulay2_resolutions=not bool(args.no_macaulay2_resolutions),
        macaulay2_timeout_seconds=max(30, int(args.macaulay2_timeout_seconds)),
    )

    if args.points_json:
        clouds = load_points_json(Path(args.points_json))[: max(1, int(args.records))]
    else:
        clouds = sample_checkpoint_point_clouds(Path(args.checkpoint), records=max(1, int(args.records)), max_points=cfg.max_points)

    audits: list[dict[str, Any]] = []
    for cloud in clouds:
        record_id = str(cloud["record_id"])
        audit = audit_point_cloud(np.asarray(cloud["points"], dtype=float), record_id=record_id, cfg=cfg)
        audits.append(audit)

    audits = add_ph_retrieval_candidates(audits, top_k=min(5, max(1, len(audits) - 1 if len(audits) > 1 else 1)))

    pages: list[dict[str, str]] = []
    for audit in audits:
        record_id = str(audit["record_id"])
        slug = safe_slug(record_id)
        record_json = records_dir / f"{slug}.json"
        record_html = records_dir / f"{slug}.html"
        m2_script = scripts_dir / f"{slug}.m2"
        ph_artifacts = write_ph_feature_artifacts(audit, ph_features_dir, slug)
        audit["staircase_visualization"] = staircase_visualization_metadata(audit)
        write_json(record_json, audit)
        m2_script.write_text(str(audit.get("macaulay2_resolution", {}).get("script", "")), encoding="utf-8")
        write_record_html(
            audit,
            record_html,
            rel_json=record_json.name,
            rel_m2=f"../macaulay2/{m2_script.name}",
            rel_ph_json=f"../{ph_artifacts['json']}",
            rel_ph_npz=f"../{ph_artifacts['npz']}",
            emit_ph_feature_visualizations=bool(args.emit_ph_feature_visualizations),
        )
        pages.append(
            {
                "record_id": record_id,
                "html": f"records/{record_html.name}",
                "json": f"records/{record_json.name}",
                "m2": f"macaulay2/{m2_script.name}",
                "ph_feature_json": ph_artifacts["json"],
                "ph_feature_npz": ph_artifacts["npz"],
                "num_simplices": audit.get("simplex_tree", {}).get("num_simplices"),
                "betti": audit.get("chain_complex", {}).get("betti"),
                "h1_landscape_norm": audit.get("vectorizations", {}).get("1", {}).get("landscape_norm"),
                "h1_persistence_image_norm": audit.get("vectorizations", {}).get("1", {}).get("persistence_image_norm"),
                "h1_silhouette_norm": audit.get("vectorizations", {}).get("1", {}).get("silhouette_norm"),
                "h1_entropy_vector_norm": audit.get("vectorizations", {}).get("1", {}).get("entropy_vector_norm"),
                "finite_field_exact_at_c1": audit.get("finite_field_chain_audit", {}).get("exact_at_c1", False),
                "simplicial_map_valid_fraction": audit.get("two_parameter_module", {})
                .get("structure_map_summary", {})
                .get("mean_simplicial_map_valid_fraction", "n/a"),
                "m2_d_squared_zero": audit.get("macaulay2_resolution", {}).get("d_squared_zero", False),
                "m2_homogeneous": bool(audit.get("macaulay2_resolution", {}).get("homogeneous_d1", False))
                and bool(audit.get("macaulay2_resolution", {}).get("homogeneous_d2", False)),
                "square_residual": audit.get("two_parameter_module", {}).get("commutative_square_residual", 1.0),
            }
        )

    summary = summarize_audits(audits)
    summary["config"] = {
        "max_points": cfg.max_points,
        "max_dimension": cfg.max_dimension,
        "radius_quantile": cfg.radius_quantile,
        "num_radii": cfg.num_radii,
        "num_levels": cfg.num_levels,
        "macaulay2_resolutions": cfg.macaulay2_resolutions,
        "ph_feature_visualizations": bool(args.emit_ph_feature_visualizations),
    }
    summary["ph_feature_artifacts"] = [
        {
            "record_id": page["record_id"],
            "json": page.get("ph_feature_json"),
            "npz": page.get("ph_feature_npz"),
        }
        for page in pages
    ]
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "records.json", audits)
    index = write_index(output_dir, summary, pages)
    print(json.dumps({"index_html": str(index), "summary_json": str(output_dir / "summary.json"), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
