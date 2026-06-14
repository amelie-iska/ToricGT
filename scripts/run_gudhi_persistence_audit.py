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
.metric { display: flex; justify-content: space-between; gap: 12px; border-bottom: 1px solid rgba(145,168,183,0.14); padding: 7px 0; }
.metric span:first-child { color: var(--muted); }
.metric span:last-child { color: white; font-variant-numeric: tabular-nums; text-align: right; }
.links { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.pill { border: 1px solid var(--border); border-radius: 999px; padding: 6px 10px; background: rgba(55,232,255,0.07); }
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
    metrics = {
        "points": record.get("points"),
        "ambient dimension": record.get("dimension"),
        "max radius": f"{float(record.get('max_radius', 0.0)):.5f}",
        "GUDHI simplices": record.get("simplex_tree", {}).get("num_simplices"),
        "Betti": chain.get("betti"),
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
        f'<div class="metric"><span>{html.escape(str(k))}</span><span>{html.escape(str(v))}</span></div>'
        for k, v in metrics.items()
    )


def write_record_html(record: dict[str, Any], out: Path, *, rel_json: str, rel_m2: str) -> None:
    fig_html = record_figure(record).to_html(include_plotlyjs="cdn", full_html=False, div_id=f"plot_{safe_slug(record['record_id'])}")
    resolution = record.get("macaulay2_resolution", {})
    hom = resolution.get("homology_and_resolutions", {}) if isinstance(resolution.get("homology_and_resolutions", {}), dict) else {}
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
  </div>
</section>
<section class="grid">
  <div class="card"><h2>Core Metrics</h2>{metric_rows(record)}</div>
  <div class="card"><h2>Module</h2>
    <p>Ring: <code>{html.escape(str(record.get('two_parameter_module', {}).get('ring', 'F2[x_level,y_radius]')))}</code></p>
    <p>Commutative squares checked: <code>{html.escape(str(record.get('two_parameter_module', {}).get('commutative_squares_checked', 'n/a')))}</code></p>
    <p>Square residual: <code>{html.escape(str(record.get('two_parameter_module', {}).get('commutative_square_residual', 'n/a')))}</code></p>
  </div>
</section>
<section class="panel plot">{fig_html}</section>
<section class="panel card">
  <h2>Macaulay2 Homology Modules And Free Resolutions</h2>
  <pre>{html.escape(json.dumps(hom, indent=2, sort_keys=True))}</pre>
</section>
</main></body></html>
"""
    out.write_text(body, encoding="utf-8")


def write_index(output_dir: Path, summary: dict[str, Any], record_pages: list[dict[str, str]]) -> Path:
    cards = []
    for page in record_pages:
        cards.append(
            f"""<div class="card">
  <h2>{html.escape(page['record_id'])}</h2>
  <p>Exact GUDHI/Macaulay2 reasoning-radius persistence audit.</p>
  <div class="links"><a class="pill" href="{html.escape(page['html'])}">open page</a><a class="pill" href="{html.escape(page['json'])}">JSON</a><a class="pill" href="{html.escape(page['m2'])}">M2 script</a></div>
</div>"""
        )
    metrics = "\n".join(
        f'<div class="metric"><span>{html.escape(str(k))}</span><span>{html.escape(str(v))}</span></div>'
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
    records_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

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
    pages: list[dict[str, str]] = []
    for cloud in clouds:
        record_id = str(cloud["record_id"])
        slug = safe_slug(record_id)
        audit = audit_point_cloud(np.asarray(cloud["points"], dtype=float), record_id=record_id, cfg=cfg)
        audits.append(audit)
        record_json = records_dir / f"{slug}.json"
        record_html = records_dir / f"{slug}.html"
        m2_script = scripts_dir / f"{slug}.m2"
        write_json(record_json, audit)
        m2_script.write_text(str(audit.get("macaulay2_resolution", {}).get("script", "")), encoding="utf-8")
        write_record_html(
            audit,
            record_html,
            rel_json=record_json.name,
            rel_m2=f"../macaulay2/{m2_script.name}",
        )
        pages.append(
            {
                "record_id": record_id,
                "html": f"records/{record_html.name}",
                "json": f"records/{record_json.name}",
                "m2": f"macaulay2/{m2_script.name}",
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
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "records.json", audits)
    index = write_index(output_dir, summary, pages)
    print(json.dumps({"index_html": str(index), "summary_json": str(output_dir / "summary.json"), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
