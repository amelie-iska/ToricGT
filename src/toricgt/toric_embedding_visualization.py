"""HTML visualizations for tropical-attention toric embedding sidecars.

The renderer consumes exact sidecar records emitted by
``scripts/run_embedding_cas_sidecar.py``.  It visualizes the finite algebraic
objects used by the tropical-to-toric embedding program: exponent sets,
Newton/lifted Newton polytopes, initial-degeneration chamber decompositions,
normal fans, toric orbit incidence, toric ideals, free resolutions,
Miller-Sturmfels staircases, and optional Klyachko vector-bundle certificates.

The module is deliberately strict about language: if a panel needs certificate
data that are not present, it renders a visible "unavailable" explanation
rather than inventing a surrogate metric.
"""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .cas_certificates import validate_certificate_payload
from .cas_oracles import Macaulay2TropicalOracle
from .tropical_toric_certificates import tropical_hypersurface_certificate


CSS = """
:root{color-scheme:dark;--bg:#030712;--panel:#07111f;--panel2:#0b1728;--text:#e8fbff;--muted:#9fb1c9;--cyan:#37e8ff;--mag:#ff4fd8;--green:#8cff6a;--amber:#ffd166;--bad:#ff7b9c;--border:rgba(55,232,255,.28)}
body{margin:0;background:radial-gradient(circle at top left,#092238 0,#030712 44rem);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1320px;margin:0 auto;padding:32px 24px 72px}
a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}
.hero,.panel,.card{background:linear-gradient(180deg,rgba(11,23,40,.96),rgba(5,13,25,.98));border:1px solid var(--border);border-radius:8px;box-shadow:0 18px 50px rgba(0,0,0,.28)}
.hero{padding:24px;margin-bottom:18px}.panel,.card{padding:16px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
h1{margin:0 0 8px;font-size:29px;letter-spacing:0}h2{margin:0 0 10px;font-size:18px;letter-spacing:0}h3{margin:14px 0 8px;font-size:14px;color:#dff9ff}
p{color:var(--muted);line-height:1.55}
.metric{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid rgba(145,168,183,.14);padding:7px 0}
.metric span:first-child{color:var(--muted)}.metric span:last-child{text-align:right;color:white;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}
.pill{display:inline-block;border:1px solid var(--border);border-radius:999px;padding:6px 10px;margin:3px;background:rgba(55,232,255,.07);color:#dff9ff}
.ok{color:var(--green)}.bad{color:var(--bad)}.warn{color:var(--amber)}
pre{white-space:pre-wrap;overflow:auto;max-height:520px;background:#020713;border:1px solid rgba(145,168,183,.18);border-radius:8px;padding:14px;color:#dff8ff}
.plot{border:1px solid rgba(55,232,255,.14);border-radius:8px;overflow:hidden;background:#020713;margin:10px 0}
.unavailable{border:1px dashed rgba(255,209,102,.45);border-radius:8px;padding:12px;background:rgba(255,209,102,.06);color:#ffe8a8}
.diagram{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 0}
.box{border:1px solid rgba(55,232,255,.32);background:rgba(55,232,255,.07);border-radius:8px;padding:10px 12px;min-width:110px;text-align:center}
.arrow{color:var(--cyan);font-size:24px}
.small{font-size:12px;color:var(--muted)}
details{border:1px solid rgba(145,168,183,.18);border-radius:8px;margin:10px 0;background:#020713}
summary{cursor:pointer;color:var(--cyan);padding:10px 12px}
details pre{border:0;border-top:1px solid rgba(145,168,183,.18);border-radius:0;margin:0;max-height:520px}
.statusgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:12px 0}
.statuscard,.diffcard{border:1px solid rgba(55,232,255,.20);border-radius:8px;background:rgba(55,232,255,.045);padding:10px 12px}
.statuscard strong,.diffcard strong{display:block;color:#dff9ff;margin-bottom:4px}
.diffgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px;margin:12px 0}
.modulechain{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;margin:12px 0}
.modulebox{border:1px solid rgba(255,79,216,.28);background:rgba(255,79,216,.06);border-radius:8px;padding:10px 12px;min-width:116px;text-align:center}
"""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _plot_html(fig: go.Figure) -> str:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#020713",
        plot_bgcolor="#020713",
        font={"color": "#e8fbff"},
        margin={"l": 42, "r": 24, "t": 48, "b": 42},
        legend={"orientation": "h", "y": -0.18},
    )
    return '<div class="plot">' + pio.to_html(fig, include_plotlyjs=True, full_html=False) + "</div>"


def _metric(label: str, value: Any) -> str:
    return f'<div class="metric"><span>{html.escape(label)}</span><span>{html.escape(str(value))}</span></div>'


def _unavailable(title: str, reason: str) -> str:
    return (
        f'<section class="panel"><h2>{html.escape(title)}</h2>'
        f'<div class="unavailable"><strong>Unavailable:</strong> {html.escape(reason)}</div></section>'
    )


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "record"


def collect_sidecar_records(*, sidecar_records: list[Path] | None = None, sidecar_dir: Path | None = None) -> list[Path]:
    records: list[Path] = []
    if sidecar_records:
        records.extend(Path(path) for path in sidecar_records)
    if sidecar_dir:
        root = Path(sidecar_dir)
        if root.is_file() and root.name.endswith(".json"):
            records.append(root)
        else:
            records.extend(sorted(root.glob("records/*_cas_sidecar.json")))
            records.extend(sorted(root.glob("*_cas_sidecar.json")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in records:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        seen.add(resolved)
        unique.append(resolved)
    if not unique:
        raise ValueError("no sidecar records were provided")
    return unique


def _record_exactness(record: dict[str, Any]) -> dict[str, bool]:
    sage = record.get("sage_normal_fan", {})
    m2 = record.get("macaulay2_toric_ideal", {})
    return {
        "sage_normal_fan_exact": sage.get("provenance") == "exact_cas/sage",
        "macaulay2_toric_ideal_exact": m2.get("provenance") == "exact_cas/macaulay2",
    }


def _exponents(record: dict[str, Any]) -> np.ndarray:
    points = np.asarray(record.get("exponent_points", []), dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("record has no rank-2 exponent_points array")
    return points


def _biases(record: dict[str, Any], count: int) -> np.ndarray:
    raw = record.get("biases") or record.get("valuation_biases") or record.get("exponent_biases")
    if raw is None:
        return np.zeros((count,), dtype=float)
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size != count:
        raise ValueError(f"bias array length {arr.size} does not match exponent count {count}")
    return arr


def _exact_tropical_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Return the sidecar tropical certificate, or derive the exact 2D one.

    Older sidecars predate the explicit ``tropical_hypersurface`` JSON field.
    When their exponent support is two-dimensional, the same exact
    closed-form lattice certificate can be reconstructed from the stored
    exponents and valuation biases.  This is not a surrogate path: the
    generated payload is marked by its own certificate provenance and uses the
    same integer arithmetic as the sidecar writer.
    """

    existing = record.get("tropical_hypersurface")
    if isinstance(existing, dict) and existing.get("tropical"):
        return existing
    try:
        points = _exponents(record)
        if points.shape[1] != 2:
            return {
                "tropical": {},
                "diagnostics": {
                    "generated_from_record": False,
                    "reason": "closed-form tropical hypersurface certificate requires exponent_dim == 2",
                },
            }
        biases = _biases(record, points.shape[0])
        return tropical_hypersurface_certificate(points.astype(int).tolist(), biases.astype(int).tolist())
    except Exception as exc:
        return {
            "tropical": {},
            "diagnostics": {
                "generated_from_record": False,
                "reason": repr(exc),
            },
        }


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=float)[:, :2]})
    if len(pts) <= 1:
        return np.asarray(pts, dtype=float)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=float)


def _newton_polytope_plot(points: np.ndarray) -> str:
    if points.shape[1] < 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(points.shape[0]), y=points[:, 0], mode="markers+text", text=[str(i) for i in range(points.shape[0])], name="characters"))
        fig.update_layout(title="Exponent set in one coordinate", xaxis_title="generator", yaxis_title="coordinate")
        return _plot_html(fig)
    xy = points[:, :2]
    hull = _convex_hull(xy)
    fig = go.Figure()
    if hull.shape[0] >= 3:
        closed = np.vstack([hull, hull[0]])
        fig.add_trace(go.Scatter(x=closed[:, 0], y=closed[:, 1], fill="toself", mode="lines", name="Newton polytope", line={"color": "#37e8ff"}))
    fig.add_trace(
        go.Scatter(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers+text",
            text=[str(i) for i in range(points.shape[0])],
            textposition="top center",
            marker={"size": 10, "color": "#ff4fd8"},
            name="characters",
        )
    )
    fig.update_layout(title="Newton Polytope From Exponent Characters", xaxis_title="character coordinate 0", yaxis_title="character coordinate 1")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _plot_html(fig)


def _lifted_newton_plot(points: np.ndarray, biases: np.ndarray) -> str:
    if points.shape[1] == 1:
        x = points[:, 0]
        y = np.zeros_like(x)
    else:
        x = points[:, 0]
        y = points[:, 1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=biases,
            mode="markers+text",
            text=[str(i) for i in range(points.shape[0])],
            marker={"size": 5, "color": biases, "colorscale": "Turbo", "showscale": True, "colorbar": {"title": "bias"}},
            name="lifted characters",
        )
    )
    fig.update_layout(title="Lifted Newton Point Set", scene={"xaxis_title": "coord 0", "yaxis_title": "coord 1", "zaxis_title": "bias / valuation lift"})
    return _plot_html(fig)


def _active_chamber_plot(points: np.ndarray, biases: np.ndarray) -> str:
    if points.shape[1] < 2:
        return _unavailable("Initial Degeneration Chamber Plot", "At least two exponent coordinates are required for a chamber heatmap.")
    xy = points[:, :2]
    mins = xy.min(axis=0) - 1.0
    maxs = xy.max(axis=0) + 1.0
    xs = np.linspace(mins[0], maxs[0], 100)
    ys = np.linspace(mins[1], maxs[1], 100)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    scores = grid @ xy.T + biases.reshape(1, -1)
    active = np.argmax(scores, axis=1).reshape(xx.shape)
    sorted_scores = np.sort(scores, axis=1)
    margin = (sorted_scores[:, -1] - sorted_scores[:, -2]).reshape(xx.shape) if scores.shape[1] >= 2 else np.zeros_like(xx)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("unique-max active character", "top-two tropical margin"))
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=active, colorscale="Turbo", colorbar={"title": "active id"}), row=1, col=1)
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=margin, colorscale="Viridis", colorbar={"title": "margin"}), row=1, col=2)
    fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=[str(i) for i in range(points.shape[0])], marker={"color": "#ffffff", "size": 7}, name="characters"), row=1, col=1)
    fig.update_xaxes(title_text="u0", row=1, col=1)
    fig.update_yaxes(title_text="u1", row=1, col=1)
    fig.update_xaxes(title_text="u0", row=1, col=2)
    fig.update_yaxes(title_text="u1", row=1, col=2)
    fig.update_layout(title="Initial Degeneration / Tropical Attention Chamber Decomposition")
    return _plot_html(fig)


def _normal_fan_plot(record: dict[str, Any]) -> str:
    toric = record.get("sage_normal_fan", {}).get("toric", {})
    cones = toric.get("fan_one_dimensional_cones") or toric.get("fan_rays")
    maximal = toric.get("maximal_cones", [])
    if not cones:
        return _unavailable("Normal Fan And One-Dimensional Cones", "Sage normal-fan certificate has no one-dimensional-cone list.")
    arr = np.asarray(cones, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return _unavailable("Normal Fan And One-Dimensional Cones", "The current normal-fan plot requires at least two fan coordinates.")
    xy = arr[:, :2]
    scale = max(1.0, float(np.linalg.norm(xy, axis=1).max()))
    fig = go.Figure()
    for idx, vec in enumerate(xy):
        fig.add_trace(
            go.Scatter(
                x=[0, float(vec[0] / scale)],
                y=[0, float(vec[1] / scale)],
                mode="lines+markers+text",
                text=["", str(idx)],
                textposition="top center",
                line={"color": "#37e8ff", "width": 3},
                marker={"size": [1, 8], "color": "#37e8ff"},
                name=f"one-dimensional cone {idx}",
                showlegend=False,
            )
        )
    if maximal:
        for cone_idx, cone in enumerate(maximal[:24]):
            ids = [int(i) for i in cone if 0 <= int(i) < len(xy)]
            if len(ids) >= 2:
                poly = np.vstack([[0, 0], xy[ids[:2]] / scale, [0, 0]])
                fig.add_trace(go.Scatter(x=poly[:, 0], y=poly[:, 1], fill="toself", mode="lines", line={"width": 1, "color": "rgba(255,79,216,.35)"}, name=f"max cone {cone_idx}", showlegend=False))
    fig.update_layout(title="Normal Fan With One-Dimensional Cones", xaxis_title="fan coordinate 0", yaxis_title="fan coordinate 1")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _plot_html(fig)


def _orbit_incidence_plot(record: dict[str, Any]) -> str:
    toric = record.get("sage_normal_fan", {}).get("toric", {})
    strata = toric.get("orbit_strata", [])
    incidences = toric.get("face_incidence", [])
    if not strata:
        return _unavailable("Toric Orbit-Stratum Incidence", "Sage normal-fan certificate has no orbit_strata field.")
    dims = [int(row.get("cone_dimension", 0)) for row in strata]
    buckets: dict[int, int] = {}
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    for idx, dim in enumerate(dims):
        offset = buckets.get(dim, 0)
        buckets[dim] = offset + 1
        xs.append(float(dim))
        ys.append(float(offset))
        labels.append(f"O({idx})<br>codim {dim}")
    fig = go.Figure()
    for edge in incidences[:300]:
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        a, b = int(edge[0]), int(edge[1])
        if 0 <= a < len(xs) and 0 <= b < len(xs):
            fig.add_trace(go.Scatter(x=[xs[a], xs[b]], y=[ys[a], ys[b]], mode="lines", line={"color": "rgba(145,168,183,.35)", "width": 1}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=[str(i) for i in range(len(xs))], hovertext=labels, hoverinfo="text", marker={"size": 11, "color": dims, "colorscale": "Turbo", "showscale": True, "colorbar": {"title": "codim"}}, name="torus orbits"))
    fig.update_layout(title="Orbit-Cone Incidence Poset", xaxis_title="cone dimension / orbit codimension", yaxis_title="index within dimension")
    return _plot_html(fig)


def _staircase_generators(points: np.ndarray) -> np.ndarray:
    if points.shape[1] < 2:
        return np.empty((0, 2), dtype=int)
    pts = np.unique(np.maximum(0, np.rint(points[:, :2]).astype(int)), axis=0)
    minimal: list[np.ndarray] = []
    for p in pts:
        dominated = False
        for q in pts:
            if np.array_equal(p, q):
                continue
            if np.all(q <= p):
                dominated = True
                break
        if not dominated:
            minimal.append(p)
    if not minimal:
        return np.empty((0, 2), dtype=int)
    arr = np.asarray(minimal, dtype=int)
    order = np.lexsort((arr[:, 1], -arr[:, 0]))
    return arr[order]


def _staircase_metadata(points: np.ndarray) -> dict[str, Any]:
    gens = _staircase_generators(points)
    if gens.size == 0:
        return {}
    max_x = int(gens[:, 0].max() + 4)
    max_y = int(gens[:, 1].max() + 4)
    xs = np.arange(max_x + 1)
    ys = np.arange(max_y + 1)
    membership = np.zeros((len(ys), len(xs)), dtype=int)
    quotient_basis: list[list[int]] = []
    ideal_points: list[list[int]] = []
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            in_ideal = int(any(x >= int(g[0]) and y >= int(g[1]) for g in gens))
            membership[yi, xi] = in_ideal
            if in_ideal:
                ideal_points.append([int(x), int(y)])
            else:
                quotient_basis.append([int(x), int(y)])
    threshold_by_x: list[int] = []
    for x in xs:
        thresholds = [int(g[1]) for g in gens if x >= int(g[0])]
        threshold_by_x.append(min(thresholds) if thresholds else max_y + 1)
    adjacent_lcms: list[dict[str, Any]] = []
    for idx, (left, right) in enumerate(zip(gens, gens[1:])):
        lcm = [int(max(left[0], right[0])), int(max(left[1], right[1]))]
        adjacent_lcms.append(
            {
                "index": int(idx),
                "left_generator": [int(left[0]), int(left[1])],
                "right_generator": [int(right[0]), int(right[1])],
                "lcm_corner": lcm,
            }
        )
    return {
        "minimal_generators": gens.astype(int).tolist(),
        "window": {"x_max": int(max_x), "y_max": int(max_y)},
        "x_values": xs.astype(int).tolist(),
        "y_values": ys.astype(int).tolist(),
        "membership_grid": membership.astype(int).tolist(),
        "quotient_basis": quotient_basis,
        "ideal_points": ideal_points,
        "staircase_threshold_by_x": [int(v) for v in threshold_by_x],
        "adjacent_lcm_layer": adjacent_lcms,
    }


def _staircase_plot(points: np.ndarray) -> str:
    meta = _staircase_metadata(points)
    if not meta:
        return _unavailable("Miller-Sturmfels Staircase", "Need at least one two-variable monomial generator after integer quantization.")
    gens = np.asarray(meta["minimal_generators"], dtype=int)
    xs = np.asarray(meta["x_values"], dtype=int)
    ys = np.asarray(meta["y_values"], dtype=int)
    membership = np.asarray(meta["membership_grid"], dtype=int)
    basis = np.asarray(meta["quotient_basis"], dtype=int) if meta["quotient_basis"] else np.empty((0, 2), dtype=int)
    ideal = np.asarray(meta["ideal_points"], dtype=int) if meta["ideal_points"] else np.empty((0, 2), dtype=int)
    lcms = np.asarray([row["lcm_corner"] for row in meta["adjacent_lcm_layer"]], dtype=int) if meta["adjacent_lcm_layer"] else np.empty((0, 2), dtype=int)
    threshold = np.asarray(meta["staircase_threshold_by_x"], dtype=int)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("Miller-Sturmfels Overhead XY-Grid Staircase", "close z-height module layers"),
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=membership,
            colorscale=[[0, "#061124"], [0.49, "#061124"], [0.50, "#3f4650"], [1, "#3f4650"]],
            showscale=False,
            name="monomial ideal region I",
            hovertemplate="x=%{x}<br>y=%{y}<br>in I=%{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if basis.size:
        fig.add_trace(
            go.Scatter(
                x=basis[:, 0],
                y=basis[:, 1],
                mode="markers",
                marker={"size": 6, "color": "#e8fbff", "line": {"color": "#061124", "width": 0.7}},
                name="quotient basis S/I",
                hovertemplate="basis monomial x^%{x}y^%{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=np.minimum(threshold, int(ys.max())),
            mode="lines",
            line={"shape": "hv", "color": "#37e8ff", "width": 4},
            name="staircase boundary",
            hovertemplate="x=%{x}<br>first ideal y=%{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=gens[:, 0],
            y=gens[:, 1],
            mode="markers+text",
            text=[f"m{i}" for i in range(len(gens))],
            textposition="top center",
            marker={"size": 13, "color": "#ff4fd8", "line": {"color": "#ffffff", "width": 0.8}},
            name="minimal monomial generators",
            hovertemplate="m=%{text}<br>x^%{x}y^%{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if lcms.size:
        fig.add_trace(
            go.Scatter(
                x=lcms[:, 0],
                y=lcms[:, 1],
                mode="markers+text",
                text=[f"lcm{i}" for i in range(len(lcms))],
                textposition="bottom center",
                marker={"size": 10, "color": "#ffd166"},
                name="adjacent lcm corners",
            ),
            row=1,
            col=1,
        )

    if ideal.size:
        fig.add_trace(
            go.Scatter3d(
                x=ideal[:, 0],
                y=ideal[:, 1],
                z=np.full(len(ideal), 0.035),
                mode="markers",
                marker={"size": 2.6, "color": "rgba(145,168,183,.42)"},
                name="I lattice points layer z=0.035",
                hovertemplate="ideal monomial x^%{x}y^%{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    if basis.size:
        fig.add_trace(
            go.Scatter3d(
                x=basis[:, 0],
                y=basis[:, 1],
                z=np.zeros(len(basis)),
                mode="markers",
                marker={"size": 3.8, "color": "#e8fbff"},
                name="quotient basis S/I layer z=0.000",
                hovertemplate="S/I basis x^%{x}y^%{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter3d(
            x=gens[:, 0],
            y=gens[:, 1],
            z=np.full(len(gens), 0.070),
            mode="markers+text",
            text=[f"m{i}" for i in range(len(gens))],
            textposition="top center",
            marker={"size": 7, "color": "#ff4fd8"},
            name="minimal generator layer z=0.070",
        ),
        row=1,
        col=2,
    )
    if lcms.size:
        fig.add_trace(
            go.Scatter3d(
                x=lcms[:, 0],
                y=lcms[:, 1],
                z=np.full(len(lcms), 0.105),
                mode="markers+text",
                text=[f"lcm{i}" for i in range(len(lcms))],
                textposition="top center",
                marker={"size": 6, "color": "#ffd166"},
                name="adjacent lcm layer z=0.105",
            ),
            row=1,
            col=2,
        )
        for idx, row_meta in enumerate(meta["adjacent_lcm_layer"]):
            lcm = row_meta["lcm_corner"]
            for gen in (row_meta["left_generator"], row_meta["right_generator"]):
                fig.add_trace(
                    go.Scatter3d(
                        x=[gen[0], lcm[0]],
                        y=[gen[1], lcm[1]],
                        z=[0.070, 0.105],
                        mode="lines",
                        line={"color": "rgba(255,209,102,.55)", "width": 3},
                        name=f"adjacent lcm differential edge {idx}",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=2,
                )
    fig.update_xaxes(title_text="x exponent", row=1, col=1)
    fig.update_yaxes(title_text="y exponent", row=1, col=1)
    fig.update_layout(
        title="Miller-Sturmfels Monomial-Ideal Staircase With Close Z-Height Module Layers",
        scene={
            "xaxis": {"title": "x exponent"},
            "yaxis": {"title": "y exponent"},
            "zaxis": {"title": "module layer", "range": [-0.01, 0.14]},
            "camera": {"eye": {"x": 1.45, "y": 1.35, "z": 0.65}},
        },
    )
    meta_html = (
        "<details open><summary>Exact staircase metadata</summary><pre>"
        + html.escape(json.dumps({key: value for key, value in meta.items() if key != "membership_grid"}, indent=2, sort_keys=True))
        + "</pre></details>"
    )
    return _plot_html(fig) + meta_html


def _relation_graph(record: dict[str, Any]) -> str:
    algebra = record.get("macaulay2_toric_ideal", {}).get("commutative_algebra", {})
    relations = algebra.get("toric_ideal_relations", [])
    if not relations:
        return _unavailable("Toric Ideal Relation Graph", "Macaulay2 certificate contains no toric_ideal_relations.")
    traces: list[go.Scatter] = []
    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    for idx, rel in enumerate(relations[:20]):
        y = float(idx)
        node_x.extend([0.0, 1.0, 2.0])
        node_y.extend([y, y, y])
        pos = rel.get("positive_exponent", [])
        neg = rel.get("negative_exponent", [])
        node_text.extend([f"x^{pos}", f"relation {idx}", f"x^{neg}"])
        traces.append(go.Scatter(x=[0.0, 1.0, 2.0], y=[y, y, y], mode="lines", line={"color": "rgba(55,232,255,.35)", "width": 2}, showlegend=False, hoverinfo="skip"))
    fig = go.Figure(data=traces)
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="top center", marker={"size": 10, "color": "#ff4fd8"}, name="binomial relation terms"))
    fig.update_layout(title="Exact Macaulay2 Toric-Ideal Relations", xaxis={"visible": False}, yaxis_title="relation")
    return _plot_html(fig)


def _resolution_diagram(record: dict[str, Any]) -> str:
    algebra = record.get("macaulay2_toric_ideal", {}).get("commutative_algebra", {})
    length = algebra.get("resolution_length")
    pdim = algebra.get("projective_dimension")
    regularity = algebra.get("regularity")
    gen_count = algebra.get("toric_ideal_generator_count")
    try:
        length_int = max(0, int(length))
    except Exception:
        length_int = 0
    boxes = [("S/I", "quotient module")] + [
        (f"F_{idx}", "free module" if idx == 0 else f"d_{idx}: F_{idx}->F_{idx-1}")
        for idx in range(length_int + 1)
    ]
    diagram = '<div class="modulechain" aria-label="resolution module strip">'
    for idx, (title, sub) in enumerate(boxes):
        if idx:
            diagram += '<span class="arrow">&larr;</span>'
        diagram += f'<div class="modulebox"><strong>{html.escape(str(title))}</strong><br><span class="small">{html.escape(str(sub))}</span></div>'
    diagram += "</div>"
    status = _resolution_status_cards(algebra)
    diff_cards = _differential_cards(algebra)
    raw = html.escape(str(algebra.get("free_resolution_raw", "")))
    module_raw = html.escape(str(algebra.get("module_free_resolution_raw", "")))
    return f"""<section class="panel"><h2>Free Resolution And Differentials</h2>
{diagram}
<p>This panel renders exact Macaulay2 resolution data as a module strip plus differential cards.  The cards preserve the raw matrix strings and keep the derived checks adjacent to the maps they audit.</p>
{status}
{diff_cards}
<details><summary>Raw ideal free resolution</summary><pre>{raw}</pre></details>
<details><summary>Raw cokernel module free resolution</summary><pre>{module_raw}</pre></details>
</section>"""


def _resolution_status_cards(algebra: dict[str, Any]) -> str:
    square = algebra.get("module_resolution_square_zero", {})
    ext = algebra.get("module_ext_modules", {})
    tor = algebra.get("module_tor_residue_modules", {})
    derived = algebra.get("derived_category_maps", {})
    rows = [
        ("projective dimension", algebra.get("projective_dimension", "n/a")),
        ("regularity", algebra.get("regularity", "n/a")),
        ("resolution length", algebra.get("resolution_length", "n/a")),
        ("toric ideal generators", algebra.get("toric_ideal_generator_count", "n/a")),
        ("d^2=0 checks", square if square else "not supplied"),
        ("Ext modules", f"{len(ext)} supplied" if isinstance(ext, dict) else "not supplied"),
        ("Tor modules", f"{len(tor)} supplied" if isinstance(tor, dict) else "not supplied"),
        ("mapping cone", "supplied" if derived else "not supplied"),
    ]
    cards = []
    for label, value in rows:
        value_text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        cards.append(f"<div class='statuscard'><strong>{html.escape(label)}</strong><span>{html.escape(value_text)}</span></div>")
    return "<div class='statusgrid'>" + "".join(cards) + "</div>"


def _parse_macaulay2_matrix_string(text: str) -> list[list[str]] | None:
    stripped = str(text).strip()
    start = stripped.find("{{")
    stop = stripped.rfind("}}")
    if start < 0 or stop <= start:
        return None
    inner = stripped[start + 2 : stop]
    rows_raw = re.split(r"}\s*,\s*{", inner)
    rows: list[list[str]] = []
    for row in rows_raw:
        cells = [cell.strip() for cell in row.split(",")]
        if not cells:
            return None
        rows.append(cells)
    width = len(rows[0]) if rows else 0
    if width == 0 or any(len(row) != width for row in rows):
        return None
    return rows


def _matrix_table(rows: list[list[str]]) -> str:
    rendered_rows = []
    for row in rows:
        rendered_rows.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>")
    return "<div class='tablewrap'><table>" + "".join(rendered_rows) + "</table></div>"


def _differential_cards(algebra: dict[str, Any]) -> str:
    differentials = algebra.get("module_resolution_differentials", {})
    if not isinstance(differentials, dict) or not differentials:
        return "<div class='unavailable'><strong>Unavailable:</strong> Macaulay2 certificate contains no module_resolution_differentials.</div>"
    cards: list[str] = []
    for name in sorted(differentials):
        text = str(differentials.get(name, ""))
        matrix_like = html.escape(text[:260] + ("..." if len(text) > 260 else ""))
        raw = html.escape(text)
        parsed = _parse_macaulay2_matrix_string(text)
        parsed_html = ""
        if parsed is not None:
            parsed_html = (
                f"<div class='small'>parsed source rank {len(parsed[0])}, target rank {len(parsed)}</div>"
                + _matrix_table(parsed)
            )
        cards.append(
            "<div class='diffcard'>"
            f"<strong>{html.escape(str(name))}</strong>"
            f"<div class='small'>exact differential matrix/string</div>"
            f"<pre>{matrix_like}</pre>"
            f"{parsed_html}"
            f"<details><summary>full exact {html.escape(str(name))} string</summary><pre>{raw}</pre></details>"
            "</div>"
        )
    return "<h3>Exact Differential Cards</h3><div class='diffgrid'>" + "".join(cards) + "</div>"


def _divisor_chow_panel(record: dict[str, Any]) -> str:
    toric = record.get("sage_normal_fan", {}).get("toric", {})
    tropical_cert = _exact_tropical_payload(record)
    tropical = tropical_cert.get("tropical", {}) if isinstance(tropical_cert, dict) else {}
    if not tropical:
        tropical = record.get("sage_normal_fan", {}).get("tropical", {})
    has_mult = bool(tropical.get("multiplicities"))
    balanced = tropical.get("balanced", False)
    chow_certified = tropical.get("chow_minkowski_weight_certified", False)
    props = toric.get("fan_properties", {})
    metrics = [
        _metric("normal fan complete", props.get("is_complete", "unknown")),
        _metric("normal fan simplicial", props.get("is_simplicial", "unknown")),
        _metric("normal fan smooth", props.get("is_smooth", "unknown")),
        _metric("tropical multiplicities present", has_mult),
        _metric("balancing stars", len(tropical.get("balancing_stars", []) or [])),
        _metric("max balance residual L1", tropical.get("max_balance_residual_l1", "n/a")),
        _metric("balanced", balanced),
        _metric("Chow/Minkowski certified", chow_certified),
    ]
    if not has_mult:
        note = (
            '<div class="unavailable"><strong>Balance/Chow class not certified:</strong> '
            "this sidecar has a Sage normal fan but no tropical-cycle multiplicity certificate. "
            "The report therefore does not invent a Minkowski weight or Chow class.</div>"
        )
    elif not chow_certified:
        note = (
            '<div class="unavailable"><strong>Chow/Minkowski class not certified:</strong> '
            "integer tropical multiplicities are present, but the exact balancing audit did not certify all stars.</div>"
        )
    else:
        note = '<p class="ok">Exact integer multiplicities pass the balancing audit; the sidecar certifies a Minkowski-weight/Chow audit state for this finite tropical hypersurface.</p>'
    facet_table = ""
    facets = tropical.get("facets", []) or []
    if facets:
        rows = []
        for facet in facets[:32]:
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(facet.get('facet_index')))}</td>"
                f"<td>{html.escape(str(facet.get('active_pair')))}</td>"
                f"<td>{html.escape(str(facet.get('multiplicity')))}</td>"
                f"<td>{html.escape(str(facet.get('primitive_normal')))}</td>"
                f"<td>{html.escape(str(facet.get('parameter_interval', {}).get('kind')))}</td>"
                "</tr>"
            )
        facet_table = (
            "<h3>Exact Facet Multiplicities</h3><div class='tablewrap'><table>"
            "<tr><th>facet</th><th>active pair</th><th>weight</th><th>primitive normal</th><th>wall domain</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
    star_pre = ""
    if tropical.get("balancing_stars"):
        star_pre = "<h3>Balancing Stars</h3><pre>" + html.escape(json.dumps(tropical.get("balancing_stars"), indent=2, sort_keys=True)) + "</pre>"
    diagnostics = tropical_cert.get("diagnostics", {}) if isinstance(tropical_cert, dict) else {}
    diag_html = ""
    if diagnostics:
        diag_html = "<h3>Certificate Diagnostics</h3><pre>" + html.escape(json.dumps(diagnostics, indent=2, sort_keys=True)) + "</pre>"
    return f"""<section class="panel"><h2>Divisor, Balance, And Chow Audit State</h2>
{''.join(metrics)}
{note}
{facet_table}
{star_pre}
{diag_html}
<p>The Cartier-divisor bend audit requires cone-wise support-function slopes and codimension-one adjacency.  When those fields are certified, bends are displayed as toric invariant-curve intersection data.</p>
</section>"""


def _derived_summary(record: dict[str, Any]) -> str:
    algebra = record.get("macaulay2_toric_ideal", {}).get("commutative_algebra", {})
    square = algebra.get("module_resolution_square_zero", {})
    ext = algebra.get("module_ext_modules", {})
    tor = algebra.get("module_tor_residue_modules", {})
    derived = algebra.get("derived_category_maps", {})
    return f"""<section class="panel"><h2>Cox / Sheaf / Ext / Tor / Derived-Map Summary</h2>
<p>This page reads the Macaulay2 module associated to the toric ideal sidecar as the finite Cox-module/sheaf proxy used by the training audit.</p>
<h3>Differential square checks</h3><pre>{html.escape(json.dumps(square, indent=2, sort_keys=True))}</pre>
<h3>Ext modules</h3><pre>{html.escape(json.dumps(ext, indent=2, sort_keys=True))}</pre>
<h3>Tor modules against residue module</h3><pre>{html.escape(json.dumps(tor, indent=2, sort_keys=True))}</pre>
<h3>Identity chain map and mapping cone</h3><pre>{html.escape(json.dumps(derived, indent=2, sort_keys=True))}</pre>
</section>"""


def _klyachko_panel(vector_bundle_payload: dict[str, Any] | None, output_dir: Path) -> tuple[str, Path | None]:
    if vector_bundle_payload is None:
        return (
            _unavailable(
                "Klyachko Vector-Bundle Filtrations",
                "No exact toric vector-bundle certificate was supplied or requested.",
            ),
            None,
        )
    errors = validate_certificate_payload(vector_bundle_payload)
    if errors:
        return (
            _unavailable("Klyachko Vector-Bundle Filtrations", "Invalid vector-bundle certificate: " + "; ".join(errors)),
            None,
        )
    vector_dir = output_dir / "vector_bundle"
    vector_dir.mkdir(parents=True, exist_ok=True)
    cert_path = vector_dir / "toric_vector_bundle_certificate.json"
    _write_json(cert_path, vector_bundle_payload)
    toric = vector_bundle_payload.get("toric", {})
    algebra = vector_bundle_payload.get("commutative_algebra", {})
    cones = toric.get("one_dimensional_cones", [])
    if cones:
        arr = np.asarray(cones, dtype=float)
        fig = go.Figure()
        if arr.ndim == 2 and arr.shape[1] >= 2:
            for idx, row in enumerate(arr[:, :2]):
                fig.add_trace(go.Scatter(x=[0, row[0]], y=[0, row[1]], mode="lines+markers+text", text=["", str(idx)], line={"width": 3}, name=f"one-dimensional cone {idx}", showlegend=False))
            fig.update_layout(title="Klyachko Filtrations Indexed By One-Dimensional Cones", xaxis_title="coord 0", yaxis_title="coord 1")
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            cone_plot = _plot_html(fig)
        else:
            cone_plot = "<pre>" + html.escape(json.dumps(cones, indent=2, sort_keys=True)) + "</pre>"
    else:
        cone_plot = '<div class="unavailable">Certificate has no one-dimensional-cone coordinates.</div>'
    page = vector_dir / "klyachko_vector_bundle.html"
    page.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Klyachko Vector Bundle</title><style>{CSS}</style></head>
<body><main>
<section class="hero"><h1>Klyachko Vector-Bundle / Sheaf Certificate</h1>
<p>Exact finite certificate consumed by the tropical-to-toric embedding visualization report.</p>
<p><a class="pill" href="../index.html">report index</a><a class="pill" href="toric_vector_bundle_certificate.json">certificate JSON</a></p></section>
<section class="grid"><div class="card"><h2>Certificate</h2>
{_metric("provenance", vector_bundle_payload.get("provenance"))}
{_metric("ambient", toric.get("ambient"))}
{_metric("rank", toric.get("rank"))}
{_metric("is vector bundle", algebra.get("is_vector_bundle"))}
{_metric("Euler characteristic", algebra.get("euler_chi"))}
</div><div class="card"><h2>Cech Checks</h2><pre>{html.escape(json.dumps(algebra.get("cech_cocycle_checks", {}), indent=2, sort_keys=True))}</pre></div></section>
<section class="panel"><h2>One-Dimensional Cones</h2>{cone_plot}</section>
<section class="panel"><h2>Filtrations And Chart Weights</h2><pre>{html.escape(json.dumps({"one_dimensional_cone_filtrations": algebra.get("one_dimensional_cone_filtrations"), "chart_weights": algebra.get("chart_weights")}, indent=2, sort_keys=True))}</pre></section>
<section class="panel"><h2>Transitions, Overlaps, And Cohomology</h2><pre>{html.escape(json.dumps({"transition_matrices": algebra.get("transition_matrices"), "chart_overlap_checks": algebra.get("chart_overlap_checks"), "cohomology_summary": algebra.get("cohomology_summary")}, indent=2, sort_keys=True))}</pre></section>
</main></body></html>""",
        encoding="utf-8",
    )
    panel = f"""<section class="panel"><h2>Klyachko Vector-Bundle Filtrations</h2>
<p>Exact vector-bundle/sheaf certificate loaded.  The dedicated page visualizes one-dimensional-cone filtrations, chart weights, transition matrices, Cech cocycle checks, and cohomology summary.</p>
<p><a class="pill" href="vector_bundle/klyachko_vector_bundle.html">open vector-bundle page</a><a class="pill" href="vector_bundle/toric_vector_bundle_certificate.json">certificate JSON</a></p>
</section>"""
    return panel, page


def _record_page(record: dict[str, Any], output_dir: Path, vector_bundle_payload: dict[str, Any] | None) -> tuple[Path, dict[str, Any]]:
    if record.get("visualization_demo_only"):
        raise ValueError("toric embedding reports require exact sidecars; visualization_demo_only records are rejected")
    record_id = str(record.get("record_id", "record"))
    points = _exponents(record)
    biases = _biases(record, points.shape[0])
    exact = _record_exactness(record)
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    json_path = records_dir / f"{_safe_name(record_id)}_toric_embedding_summary.json"
    html_path = records_dir / f"{_safe_name(record_id)}_toric_embedding.html"
    summary = {
        "schema": "toricgt.toric_embedding_visual_record.v1",
        "record_id": record_id,
        "exactness": exact,
        "exponent_count": int(points.shape[0]),
        "exponent_dim": int(points.shape[1]),
        "bias_source": "record" if any(key in record for key in ("biases", "valuation_biases", "exponent_biases")) else "zero_valuation_default",
        "miller_sturmfels_staircase": {
            key: value
            for key, value in _staircase_metadata(points).items()
            if key in {"minimal_generators", "window", "quotient_basis", "staircase_threshold_by_x", "adjacent_lcm_layer"}
        },
    }
    _write_json(json_path, summary)
    klyachko_html, _ = _klyachko_panel(vector_bundle_payload, output_dir)
    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(record_id)} Toric Embedding</title><style>{CSS}</style></head>
<body><main>
<section class="hero"><h1>{html.escape(record_id)} Tropical-To-Toric Embedding Report</h1>
<p>This page visualizes the exact finite sidecar that embeds rationalized tropical ring attention into toric geometry.  It does not claim the full Transformer is toric.</p>
<p><a class="pill" href="../index.html">index</a><a class="pill" href="{html.escape(json_path.name)}">summary JSON</a></p></section>
<section class="grid">
<div class="card"><h2>Exactness And Conventions</h2>
{_metric("Sage normal fan exact", exact["sage_normal_fan_exact"])}
{_metric("Macaulay2 toric ideal exact", exact["macaulay2_toric_ideal_exact"])}
{_metric("attention convention", "max-plus")}
{_metric("CAS bridge", "min-plus initial form with in_{-u}(f)")}
{_metric("bias source", summary["bias_source"])}
</div>
<div class="card"><h2>Sidecar Size</h2>
{_metric("exponent count", points.shape[0])}
{_metric("exponent dimension", points.shape[1])}
{_metric("embedding key", record.get("embedding_key", ""))}
{_metric("embedding NPZ", record.get("embedding_npz", ""))}
</div>
</section>
<section class="panel"><h2>Newton Polytope</h2>{_newton_polytope_plot(points)}</section>
<section class="panel"><h2>Lifted Newton Polytope</h2>{_lifted_newton_plot(points, biases)}</section>
<section class="panel"><h2>Initial Degeneration And Active Chambers</h2>{_active_chamber_plot(points, biases)}</section>
<section class="panel"><h2>Normal Fan And One-Dimensional Cones</h2>{_normal_fan_plot(record)}</section>
<section class="panel"><h2>Toric Orbit-Stratum Incidence</h2>{_orbit_incidence_plot(record)}</section>
{_divisor_chow_panel(record)}
<section class="panel"><h2>Toric Ideal Relations</h2>{_relation_graph(record)}</section>
{_resolution_diagram(record)}
<section class="panel"><h2>Miller-Sturmfels Staircase</h2><p>This is the exact monomial-ideal staircase for the two-coordinate integer exponent generators in this sidecar.  It is not labeled as an initial ideal unless a separate initial-ideal certificate is present.  The 3D panel uses close z-height module layers: quotient basis S/I, ideal lattice region I, minimal generator layer, and adjacent lcm layer.</p>{_staircase_plot(points)}</section>
{klyachko_html}
{_derived_summary(record)}
</main></body></html>"""
    html_path.write_text(body, encoding="utf-8")
    summary["html"] = f"records/{html_path.name}"
    summary["json"] = f"records/{json_path.name}"
    _write_json(json_path, summary)
    return html_path, summary


def _load_or_build_vector_bundle(path: Path | None, build: bool, timeout_seconds: int) -> dict[str, Any] | None:
    if path is not None:
        return _read_json(path)
    if not build:
        return None
    oracle = Macaulay2TropicalOracle()
    cert = oracle.vector_bundle_smoke_certificate(timeout_seconds=max(30, int(timeout_seconds)))
    return cert.with_hash()


def render_toric_embedding_report(
    *,
    sidecar_records: list[Path],
    output_dir: Path,
    vector_bundle_certificate: Path | None = None,
    build_vector_bundle_certificate: bool = False,
    macaulay2_timeout_seconds: int = 120,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_payload = _load_or_build_vector_bundle(
        vector_bundle_certificate,
        bool(build_vector_bundle_certificate),
        int(macaulay2_timeout_seconds),
    )
    record_summaries: list[dict[str, Any]] = []
    for path in sidecar_records:
        record = _read_json(path)
        page, summary = _record_page(record, output_dir, vector_payload)
        summary["source_sidecar_json"] = str(Path(path))
        summary["html_path"] = str(page)
        record_summaries.append(summary)
    cards = []
    for summary in record_summaries:
        cards.append(
            f"""<section class="card"><h2>{html.escape(summary['record_id'])}</h2>
{_metric("exponents", summary['exponent_count'])}
{_metric("dimension", summary['exponent_dim'])}
{_metric("Sage exact", summary['exactness']['sage_normal_fan_exact'])}
{_metric("Macaulay2 exact", summary['exactness']['macaulay2_toric_ideal_exact'])}
<p><a class="pill" href="{html.escape(summary['html'])}">open report</a><a class="pill" href="{html.escape(summary['json'])}">summary JSON</a></p></section>"""
        )
    if vector_payload is not None:
        vector_link = '<a class="pill" href="vector_bundle/klyachko_vector_bundle.html">Klyachko vector-bundle page</a>'
    else:
        vector_link = '<span class="pill warn">no vector-bundle certificate requested</span>'
    index = output_dir / "index.html"
    index.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>ToricGT Tropical-To-Toric Embedding Visualizations</title><style>{CSS}</style></head>
<body><main>
<section class="hero"><h1>ToricGT Tropical-To-Toric Embedding Visualizations</h1>
<p>Visual audit pages for exact finite sidecars embedding tropical ring attention into toric geometry.  Panels show initial degenerations, Newton polytopes, normal fans, toric orbit strata, monomial staircases, toric ideals, free resolutions, and optional Klyachko vector-bundle/sheaf certificates.</p>
<p><a class="pill" href="manifest.json">manifest JSON</a>{vector_link}</p></section>
<section class="grid">{''.join(cards)}</section>
</main></body></html>""",
        encoding="utf-8",
    )
    manifest = {
        "schema": "toricgt.toric_embedding_visual_report.v1",
        "records": record_summaries,
        "record_count": len(record_summaries),
        "index_html": "index.html",
        "vector_bundle_certificate": bool(vector_payload is not None),
        "exactness_rule": "Panels marked exact only when exact CAS certificate provenance is present; missing exact data are rendered unavailable.",
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest
