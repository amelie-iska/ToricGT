#!/usr/bin/env python3
"""Run TokenGT inference/evaluation samples with optional geometry outputs.

This is a thin CLI wrapper around the TokenGT reasoning-geometry evaluator.  It
keeps inference use simple while still allowing the full checkpoint-analysis
plot family bundle, including dark-mode Plotly HTML trajectory/energy/toric
geometry scenes, to be emitted for an individual model run.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
GEOMETRY_SCRIPT = ROOT / "scripts" / "evaluate_tokengt_reasoning_geometry_suite.py"
GUDHI_SCRIPT = ROOT / "scripts" / "run_gudhi_persistence_audit.py"
CAS_SIDECAR_SCRIPT = ROOT / "scripts" / "run_embedding_cas_sidecar.py"
BRANCHING_REASONING_SCRIPT = ROOT / "scripts" / "render_branching_reasoning_trajectory_report.py"
SCREENSHOT_SCRIPT = ROOT / "scripts" / "render_html_screenshots.py"

from toricgt.music import TorusMusicConfig, generate_wav  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="TokenGT .pt checkpoint to run.")
    parser.add_argument("--config", default="config/train.full_tokengt_got_fineweb_derived.yaml")
    parser.add_argument(
        "--data-glob",
        action="append",
        default=[],
        help="Inference/evaluation graph source. May be passed multiple times.",
    )
    parser.add_argument("--output-dir", default="outputs/tokengt_inference")
    parser.add_argument("--records", type=int, default=1, help="Number of inference graph records to run.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--parquet-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fineweb-tokenizer-path", default="")
    parser.add_argument("--fineweb-tokens-per-graph", type=int, default=0)
    parser.add_argument("--fineweb-stride-tokens", type=int, default=0)
    parser.add_argument("--derived-category-max-vertices", type=int, default=8)
    parser.add_argument("--topology-max-points", type=int, default=32)
    parser.add_argument("--topology-max-windows", type=int, default=4)
    parser.add_argument(
        "--emit-geometry",
        dest="emit_geometry",
        action="store_true",
        default=True,
        help="Emit trajectory, topology, CCA, toric, GraphCG, analogical, triangle, tetrahedron, static PNG, and Plotly HTML outputs.",
    )
    parser.add_argument(
        "--no-emit-geometry",
        dest="emit_geometry",
        action="store_false",
        help="Only write an inference manifest; do not run the geometry sidecar.",
    )
    parser.add_argument(
        "--rich-legacy-geometry",
        dest="rich_legacy_geometry",
        action="store_true",
        default=True,
        help="Include the R97-style rich geometry bundle in addition to TokenGT-native plots.",
    )
    parser.add_argument(
        "--no-rich-legacy-geometry",
        dest="rich_legacy_geometry",
        action="store_false",
        help="Emit only TokenGT-native trajectory/topology plots.",
    )
    parser.add_argument(
        "--geometry-output-dir",
        default="",
        help="Override where geometry files are written. Defaults to <output-dir>/geometry.",
    )
    parser.add_argument(
        "--emit-embedding-payloads",
        dest="emit_embedding_payloads",
        action="store_true",
        default=True,
        help="Persist exact hidden-state embedding payloads during geometry inference.",
    )
    parser.add_argument(
        "--no-emit-embedding-payloads",
        dest="emit_embedding_payloads",
        action="store_false",
        help="Skip embedding payload NPZ/JSON output.",
    )
    parser.add_argument(
        "--emit-gudhi-persistence",
        action="store_true",
        help="Also emit exact GUDHI/Macaulay2 persistent-homology HTML pages from checkpoint tensors.",
    )
    parser.add_argument("--gudhi-output-dir", default="")
    parser.add_argument("--gudhi-records", type=int, default=2)
    parser.add_argument("--gudhi-max-points", type=int, default=18)
    parser.add_argument("--gudhi-num-radii", type=int, default=5)
    parser.add_argument("--gudhi-num-levels", type=int, default=5)
    parser.add_argument(
        "--emit-ph-feature-visualizations",
        dest="emit_ph_feature_visualizations",
        action="store_true",
        default=True,
        help="When GUDHI persistence is enabled, render PH feature dashboards for landscapes, images, silhouettes, entropy vectors, Betti curves, and lifetimes.",
    )
    parser.add_argument(
        "--no-emit-ph-feature-visualizations",
        dest="emit_ph_feature_visualizations",
        action="store_false",
        help="When GUDHI persistence is enabled, skip PH feature dashboard rendering while keeping exact JSON/NPZ feature artifacts.",
    )
    parser.add_argument(
        "--emit-cas-sidecar",
        action="store_true",
        help="Run exact Sage/Macaulay2 toric audits from the saved embedding payloads.",
    )
    parser.add_argument("--cas-sidecar-output-dir", default="")
    parser.add_argument("--cas-sidecar-records", type=int, default=1)
    parser.add_argument("--cas-sidecar-max-points", type=int, default=12)
    parser.add_argument("--cas-sidecar-exponent-dim", type=int, default=2)
    parser.add_argument("--cas-sidecar-quantization-scale", type=int, default=12)
    parser.add_argument(
        "--emit-branching-reasoning-report",
        action="store_true",
        help="Render branch/merge simplex-tree and analogical-PH report from saved embedding payloads.",
    )
    parser.add_argument("--branching-reasoning-output-dir", default="")
    parser.add_argument("--branching-reasoning-max-nodes", type=int, default=160)
    parser.add_argument("--branching-reasoning-node-offset", type=int, default=0)
    parser.add_argument(
        "--branching-reasoning-screenshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture Playwright screenshots for the branch/simplex report.",
    )
    parser.add_argument(
        "--assert-branching-reasoning-report",
        dest="assert_branching_reasoning_report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require strict branching simplex-report DOM assertions when inference captures screenshots.",
    )
    parser.add_argument(
        "--emit-slepian-music",
        action="store_true",
        help="Export optional Slepian/Pollak prolate torus music WAV and JSON metadata for this inference bundle.",
    )
    parser.add_argument("--slepian-music-output-dir", default="")
    parser.add_argument("--slepian-music-seconds", type=float, default=6.0)
    parser.add_argument("--slepian-music-bpm", type=float, default=112.0)
    parser.add_argument("--slepian-music-sample-rate", type=int, default=22050)
    parser.add_argument("--slepian-music-modes", type=int, default=6)
    parser.add_argument("--slepian-music-bandwidth", type=float, default=0.075)
    return parser.parse_args()


def load_checkpoint_meta(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", {})
    parameter_count = 0
    if isinstance(state, dict):
        parameter_count = int(sum(int(tensor.numel()) for tensor in state.values() if hasattr(tensor, "numel")))
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("step", 0) or 0),
        "checkpoint_kind": str(payload.get("kind", "tokengt_graph")),
        "parameter_count": parameter_count,
        "has_lm_head": bool(
            isinstance(state, dict)
            and any(key.endswith("lm_head.weight") or "lm_head" in key or "lm_projection" in key for key in state)
        ),
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


INDEX_CSS = """
:root { color-scheme: dark; --bg:#030712; --panel:#07111f; --text:#e8fbff; --muted:#91a8b7; --cyan:#37e8ff; --border:rgba(55,232,255,.26); }
body { margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:radial-gradient(circle at top left,#092238 0,var(--bg) 42rem); color:var(--text); }
main { max-width:1180px; margin:0 auto; padding:32px 24px 64px; }
.hero,.card { background:linear-gradient(180deg,rgba(11,23,40,.94),rgba(5,13,25,.96)); border:1px solid var(--border); border-radius:8px; box-shadow:0 18px 50px rgba(0,0,0,.28); }
.hero { padding:24px; margin-bottom:18px; }
.grid { display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); }
.card { padding:16px; }
h1 { margin:0 0 8px; font-size:28px; letter-spacing:0; }
h2 { margin:0 0 10px; font-size:18px; letter-spacing:0; }
p { color:var(--muted); line-height:1.55; }
a { color:var(--cyan); text-decoration:none; }
a:hover { text-decoration:underline; }
.metric { display:flex; justify-content:space-between; gap:12px; border-bottom:1px solid rgba(145,168,183,.14); padding:7px 0; }
.metric span:first-child { color:var(--muted); }
.metric span:last-child { text-align:right; overflow-wrap:anywhere; }
.links { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
.pill { border:1px solid var(--border); border-radius:999px; padding:6px 10px; background:rgba(55,232,255,.07); }
.badge { display:inline-block; border:1px solid rgba(140,255,106,.35); border-radius:999px; color:#d9ffd2; background:rgba(140,255,106,.11); padding:4px 8px; font-size:12px; margin:2px 4px 2px 0; }
.badge.off { border-color:rgba(255,79,216,.45); color:#ffd8f7; background:rgba(255,79,216,.12); }
"""


def _rel_link(base: Path, target: str | Path | None) -> str:
    if not target:
        return ""
    path = Path(target)
    if not path.is_absolute():
        cwd_relative = path.resolve()
        if cwd_relative.exists():
            path = cwd_relative
        else:
            path = (base / path).resolve()
    try:
        return path.relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _metric(label: str, value: Any) -> str:
    return f'<div class="metric"><span>{html.escape(label)}</span><span>{html.escape(str(value))}</span></div>'


def _badge(label: str, enabled: bool) -> str:
    cls = "badge" if enabled else "badge off"
    value = "ON" if enabled else "OFF"
    return f'<span class="{cls}">{html.escape(label)}: {value}</span>'


def write_inference_index(output_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write a dark-mode root index linking every emitted inference artifact."""

    cards: list[str] = []

    def add_card(title: str, description: str, links: list[tuple[str, str]], metrics: list[tuple[str, Any]] | None = None) -> None:
        link_html = "".join(f'<a class="pill" href="{html.escape(href)}">{html.escape(label)}</a>' for label, href in links if href)
        metric_html = "".join(_metric(label, value) for label, value in (metrics or []))
        cards.append(
            f"""<div class="card"><h2>{html.escape(title)}</h2><p>{html.escape(description)}</p>{metric_html}<div class="links">{link_html}</div></div>"""
        )

    add_card(
        "Run Manifest",
        "Top-level inference manifest, checkpoint metadata, and enabled analysis sidecars.",
        [("inference_output.json", "inference_output.json")],
        [
            ("checkpoint step", manifest.get("checkpoint_step", "n/a")),
            ("parameter count", manifest.get("parameter_count", "n/a")),
            ("device", manifest.get("device", "n/a")),
            ("records", manifest.get("records", "n/a")),
        ],
    )
    if manifest.get("geometry_output_dir"):
        geometry_dir = _rel_link(output_dir, manifest.get("geometry_output_dir"))
        add_card(
            "Reasoning Geometry",
            "TokenGT trajectory, topology, toric/tropical, GraphCG, analogical, and energy-landscape HTML outputs.",
            [
                ("geometry directory", geometry_dir),
                ("plot manifest", f"{geometry_dir}/plot_family_manifest.json" if geometry_dir else ""),
                ("summary JSON", f"{geometry_dir}/reasoning_geometry_summary.json" if geometry_dir else ""),
            ],
            [
                ("embedding payloads", _rel_link(output_dir, manifest.get("embedding_payload_manifest")) or "not emitted"),
            ],
        )
    if manifest.get("gudhi_persistence_index_html"):
        add_card(
            "GUDHI / Macaulay2 Persistence",
            "Exact simplex-tree persistence, vectorized PH dashboards, F2[x_level,y_radius] modules, and Macaulay2 free resolutions.",
            [
                ("HTML index", _rel_link(output_dir, manifest.get("gudhi_persistence_index_html"))),
                ("summary JSON", _rel_link(output_dir, Path(str(manifest.get("gudhi_persistence_output_dir", ""))) / "summary.json")),
                ("PH features", _rel_link(output_dir, Path(str(manifest.get("gudhi_persistence_output_dir", ""))) / "ph_features")),
            ],
            [
                ("records", manifest.get("gudhi_persistence_summary", {}).get("records", "n/a")),
                ("mean H1 landscape norm", manifest.get("gudhi_persistence_summary", {}).get("mean_h1_landscape_norm", "n/a")),
                ("PH dashboards", "on" if manifest.get("ph_feature_visualizations_enabled") else "off"),
            ],
        )
    if manifest.get("embedding_cas_sidecar_index_html"):
        add_card(
            "Embedding CAS Sidecar",
            "Strict Sage normal-fan and Macaulay2 toric-ideal certificates derived from saved embedding payloads.",
            [
                ("HTML index", _rel_link(output_dir, manifest.get("embedding_cas_sidecar_index_html"))),
                ("manifest JSON", _rel_link(output_dir, Path(str(manifest.get("embedding_cas_sidecar_output_dir", ""))) / "manifest.json")),
            ],
            [
                ("records", manifest.get("embedding_cas_sidecar_manifest", {}).get("records", "n/a")),
                ("failures", manifest.get("embedding_cas_sidecar_manifest", {}).get("failures", "n/a")),
            ],
        )
    if manifest.get("branching_reasoning_report_html"):
        add_card(
            "Branching Reasoning Simplex Report",
            "Branch/merge graph-of-thought trajectory, per-step simplex trees, analogical simplex maps, and GUDHI vectorized PH comparisons rendered from saved hidden embeddings.",
            [
                ("trajectory HTML", _rel_link(output_dir, manifest.get("branching_reasoning_report_html"))),
                ("payload JSON", _rel_link(output_dir, Path(str(manifest.get("branching_reasoning_output_dir", ""))) / "branching_reasoning_payload.json")),
                ("screenshots", _rel_link(output_dir, manifest.get("branching_reasoning_screenshot_index_html"))),
            ],
            [
                ("source mode", manifest.get("branching_reasoning_manifest", {}).get("source_mode", "n/a")),
                ("nodes", manifest.get("branching_reasoning_manifest", {}).get("nodes", "n/a")),
                ("analogy emitted", manifest.get("branching_reasoning_manifest", {}).get("analogy_emitted", "n/a")),
            ],
        )
    if manifest.get("slepian_music_wav"):
        add_card(
            "Slepian/Pollak Music Export",
            "Optional inference-side WAV generated from the same finite Slepian/DPSS toric phase construction used by the visualization audits.",
            [
                ("WAV", _rel_link(output_dir, manifest.get("slepian_music_wav"))),
                ("metadata JSON", _rel_link(output_dir, manifest.get("slepian_music_metadata_json"))),
            ],
            [
                ("seconds", manifest.get("slepian_music_config", {}).get("seconds", "n/a")),
                ("sample rate", manifest.get("slepian_music_config", {}).get("sample_rate", "n/a")),
                ("modes", manifest.get("slepian_music_config", {}).get("slepian_modes", "n/a")),
            ],
        )

    badges = "".join(
        [
            _badge("geometry", bool(manifest.get("geometry_enabled"))),
            _badge("embedding payloads", bool(manifest.get("embedding_payloads_enabled"))),
            _badge("GUDHI persistence", bool(manifest.get("gudhi_persistence_enabled"))),
            _badge("PH feature dashboards", bool(manifest.get("ph_feature_visualizations_enabled"))),
            _badge("CAS sidecar", bool(manifest.get("embedding_cas_sidecar_enabled"))),
            _badge("Branching simplex report", bool(manifest.get("branching_reasoning_enabled"))),
            _badge("Slepian music", bool(manifest.get("slepian_music_enabled"))),
        ]
    )
    index = output_dir / "index.html"
    index.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ToricGT Inference Analysis Bundle</title><style>{INDEX_CSS}</style></head>
<body><main>
<section class="hero">
  <h1>ToricGT Inference Analysis Bundle</h1>
  <p>Single entry point for optional inference outputs: hidden-state embeddings, geometry plots, exact GUDHI/Macaulay2 persistence, and strict Sage/Macaulay2 toric sidecars.</p>
  <div>{badges}</div>
</section>
<section class="grid">{''.join(cards)}</section>
</main></body></html>
""",
        encoding="utf-8",
    )
    return index


def geometry_command(args: argparse.Namespace, geometry_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(GEOMETRY_SCRIPT),
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.config),
        "--output-dir",
        str(geometry_dir),
        "--records",
        str(max(1, int(args.records))),
        "--batch-size",
        str(max(1, int(args.batch_size))),
        "--parquet-batch-size",
        str(max(1, int(args.parquet_batch_size))),
        "--device",
        str(args.device),
        "--precision",
        str(args.precision),
        "--seed",
        str(int(args.seed)),
        "--derived-category-max-vertices",
        str(max(1, int(args.derived_category_max_vertices))),
        "--topology-max-points",
        str(max(4, int(args.topology_max_points))),
        "--topology-max-windows",
        str(max(1, int(args.topology_max_windows))),
    ]
    for data_glob in args.data_glob:
        cmd.extend(["--data-glob", str(data_glob)])
    if args.fineweb_tokenizer_path:
        cmd.extend(["--fineweb-tokenizer-path", str(args.fineweb_tokenizer_path)])
    if int(args.fineweb_tokens_per_graph or 0) > 0:
        cmd.extend(["--fineweb-tokens-per-graph", str(int(args.fineweb_tokens_per_graph))])
    if int(args.fineweb_stride_tokens or 0) > 0:
        cmd.extend(["--fineweb-stride-tokens", str(int(args.fineweb_stride_tokens))])
    if not bool(args.rich_legacy_geometry):
        cmd.append("--no-rich-legacy-geometry")
    if not bool(args.emit_embedding_payloads):
        cmd.append("--no-emit-embedding-payloads")
    return cmd


def gudhi_command(args: argparse.Namespace, gudhi_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(GUDHI_SCRIPT),
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(gudhi_dir),
        "--records",
        str(max(1, int(args.gudhi_records))),
        "--max-points",
        str(max(4, int(args.gudhi_max_points))),
        "--num-radii",
        str(max(2, int(args.gudhi_num_radii))),
        "--num-levels",
        str(max(2, int(args.gudhi_num_levels))),
    ]
    if not bool(args.emit_ph_feature_visualizations):
        cmd.append("--no-emit-ph-feature-visualizations")
    return cmd


def cas_sidecar_command(args: argparse.Namespace, embedding_manifest: Path, cas_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(CAS_SIDECAR_SCRIPT),
        "--embedding-manifest",
        str(embedding_manifest),
        "--output-dir",
        str(cas_dir),
        "--records",
        str(max(1, int(args.cas_sidecar_records))),
        "--max-points",
        str(max(3, int(args.cas_sidecar_max_points))),
        "--exponent-dim",
        str(max(1, int(args.cas_sidecar_exponent_dim))),
        "--quantization-scale",
        str(max(2, int(args.cas_sidecar_quantization_scale))),
    ]


def resolve_first_embedding_payload(embedding_manifest: Path) -> tuple[Path, Path | None]:
    manifest = read_json(embedding_manifest)
    rows = manifest.get("payloads", [])
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise FileNotFoundError(f"{embedding_manifest} does not contain embedding payload rows")
    row = rows[0]
    npz_candidates: list[Path] = []
    for key in ("npz", "relative_npz"):
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value))
        npz_candidates.extend([path, embedding_manifest.parent / path, embedding_manifest.parent.parent / path])
    npz = next((path for path in npz_candidates if path.exists()), None)
    if npz is None:
        raise FileNotFoundError(f"could not resolve first embedding payload NPZ from {embedding_manifest}")
    json_candidates: list[Path] = []
    for key in ("json", "relative_json"):
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value))
        json_candidates.extend([path, embedding_manifest.parent / path, embedding_manifest.parent.parent / path])
    json_path = next((path for path in json_candidates if path.exists()), None)
    return npz, json_path


def branching_reasoning_command(args: argparse.Namespace, npz: Path, metadata_json: Path | None, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(BRANCHING_REASONING_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--embedding-payload-npz",
        str(npz),
        "--embedding-max-nodes",
        str(max(0, int(args.branching_reasoning_max_nodes))),
        "--embedding-node-offset",
        str(max(0, int(args.branching_reasoning_node_offset))),
    ]
    if metadata_json is not None:
        cmd.extend(["--embedding-payload-json", str(metadata_json)])
    return cmd


def branching_screenshot_command(args: argparse.Namespace, branching_dir: Path, screenshot_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(SCREENSHOT_SCRIPT),
        "--source-dir",
        str(branching_dir),
        "--output-dir",
        str(screenshot_dir),
        "--no-full-page",
        "--viewport-slices",
        "8",
        "--interaction-audit",
        "--interaction-delay-ms",
        "700",
        "--wait-ms",
        "1800",
        "--timeout-ms",
        "90000",
    ]
    if bool(getattr(args, "assert_branching_reasoning_report", True)):
        cmd.append("--assert-branching-report")
    return cmd


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir = Path(args.geometry_output_dir) if args.geometry_output_dir else output_dir / "geometry"
    gudhi_dir = Path(args.gudhi_output_dir) if args.gudhi_output_dir else output_dir / "gudhi_persistence"
    cas_dir = Path(args.cas_sidecar_output_dir) if args.cas_sidecar_output_dir else output_dir / "embedding_cas_sidecar"
    branching_dir = Path(args.branching_reasoning_output_dir) if args.branching_reasoning_output_dir else output_dir / "branching_reasoning_report"
    music_dir = Path(args.slepian_music_output_dir) if args.slepian_music_output_dir else output_dir / "music"

    manifest: dict[str, Any] = {
        "schema": "toricgt.tokengt_inference.v1",
        "mode": "tokengt_graph_inference_with_optional_geometry",
        "config": str(args.config),
        "data_globs": [str(item) for item in args.data_glob],
        "records": int(args.records),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "precision": str(args.precision),
        "seed": int(args.seed),
        "geometry_enabled": bool(args.emit_geometry),
        "rich_legacy_geometry_enabled": bool(args.rich_legacy_geometry),
        "interactive_geometry_html_enabled": bool(args.emit_geometry),
        "embedding_payloads_enabled": bool(args.emit_embedding_payloads),
        "gudhi_persistence_enabled": bool(args.emit_gudhi_persistence),
        "ph_feature_visualizations_enabled": bool(args.emit_ph_feature_visualizations),
        "embedding_cas_sidecar_enabled": bool(args.emit_cas_sidecar),
        "branching_reasoning_enabled": bool(args.emit_branching_reasoning_report),
        "slepian_music_enabled": bool(args.emit_slepian_music),
        **load_checkpoint_meta(checkpoint),
    }

    if args.emit_geometry:
        geometry_dir.mkdir(parents=True, exist_ok=True)
        cmd = geometry_command(args, geometry_dir)
        subprocess.run(cmd, cwd=ROOT, check=True)
        manifest["geometry_output_dir"] = str(geometry_dir)
        manifest["geometry_summary"] = read_json(geometry_dir / "reasoning_geometry_summary.json")
        manifest["plot_family_manifest"] = read_json(geometry_dir / "plot_family_manifest.json")
        embedding_manifest = geometry_dir / "embeddings" / "manifest.json"
        if embedding_manifest.exists():
            manifest["embedding_payload_manifest"] = str(embedding_manifest)

    if args.emit_gudhi_persistence:
        gudhi_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(gudhi_command(args, gudhi_dir), cwd=ROOT, check=True)
        manifest["gudhi_persistence_output_dir"] = str(gudhi_dir)
        manifest["gudhi_persistence_index_html"] = str(gudhi_dir / "index.html")
        manifest["gudhi_persistence_summary"] = read_json(gudhi_dir / "summary.json")
        manifest["ph_feature_output_dir"] = str(gudhi_dir / "ph_features")
        manifest["ph_feature_artifacts"] = manifest["gudhi_persistence_summary"].get("ph_feature_artifacts", [])

    if args.emit_cas_sidecar:
        if not args.emit_geometry:
            raise ValueError("--emit-cas-sidecar requires --emit-geometry so embedding payloads can be generated")
        embedding_manifest = geometry_dir / "embeddings" / "manifest.json"
        if not embedding_manifest.exists():
            raise FileNotFoundError(
                f"embedding payload manifest not found: {embedding_manifest}. "
                "Run without --no-emit-embedding-payloads."
            )
        cas_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cas_sidecar_command(args, embedding_manifest, cas_dir), cwd=ROOT, check=True)
        manifest["embedding_cas_sidecar_output_dir"] = str(cas_dir)
        manifest["embedding_cas_sidecar_index_html"] = str(cas_dir / "index.html")
        manifest["embedding_cas_sidecar_manifest"] = read_json(cas_dir / "manifest.json")

    if args.emit_branching_reasoning_report:
        if not args.emit_geometry:
            raise ValueError("--emit-branching-reasoning-report requires --emit-geometry so embedding payloads can be generated")
        embedding_manifest = geometry_dir / "embeddings" / "manifest.json"
        if not embedding_manifest.exists():
            raise FileNotFoundError(
                f"embedding payload manifest not found: {embedding_manifest}. "
                "Run without --no-emit-embedding-payloads."
            )
        npz_path, json_path = resolve_first_embedding_payload(embedding_manifest)
        branching_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(branching_reasoning_command(args, npz_path, json_path, branching_dir), cwd=ROOT, check=True)
        manifest["branching_reasoning_output_dir"] = str(branching_dir)
        manifest["branching_reasoning_report_html"] = str(branching_dir / "branching_reasoning_trajectory.html")
        manifest["branching_reasoning_manifest"] = read_json(branching_dir / "manifest.json")
        manifest["branching_reasoning_embedding_payload_npz"] = str(npz_path)
        manifest["branching_reasoning_embedding_payload_json"] = str(json_path or "")
        if bool(args.branching_reasoning_screenshots):
            screenshot_dir = branching_dir / "html_screenshots"
            subprocess.run(branching_screenshot_command(args, branching_dir, screenshot_dir), cwd=ROOT, check=True)
            manifest["branching_reasoning_screenshot_index_html"] = str(screenshot_dir / "index.html")
            manifest["branching_reasoning_screenshot_manifest"] = read_json(screenshot_dir / "manifest.json")

    if args.emit_slepian_music:
        music_dir.mkdir(parents=True, exist_ok=True)
        wav_output = music_dir / "slepian_pollak_torus_music.wav"
        music_cfg = TorusMusicConfig(
            seconds=max(0.25, float(args.slepian_music_seconds)),
            bpm=max(24.0, float(args.slepian_music_bpm)),
            sample_rate=max(8000, int(args.slepian_music_sample_rate)),
            seed=int(args.seed),
            use_slepian=True,
            slepian_bandwidth=float(args.slepian_music_bandwidth),
            slepian_modes=max(1, int(args.slepian_music_modes)),
        )
        wav_path, metadata_path, events = generate_wav(wav_output, music_cfg)
        manifest["slepian_music_output_dir"] = str(music_dir)
        manifest["slepian_music_wav"] = str(wav_path)
        manifest["slepian_music_metadata_json"] = str(metadata_path)
        manifest["slepian_music_events"] = int(len(events))
        manifest["slepian_music_config"] = {
            "seconds": float(music_cfg.seconds),
            "bpm": float(music_cfg.bpm),
            "sample_rate": int(music_cfg.sample_rate),
            "slepian_modes": int(music_cfg.slepian_modes),
            "slepian_bandwidth": float(music_cfg.slepian_bandwidth),
        }

    output_path = output_dir / "inference_output.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    index = write_inference_index(output_dir, manifest)
    manifest["index_html"] = str(index)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
