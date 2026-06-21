#!/usr/bin/env python3
"""Build the ToricGT GitHub Pages site from current campaign artifacts."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_URL = "https://github.com/amelie-iska/ToricGT"
HF_URL = "https://huggingface.co/AmelieSchreiber/toricgt-checkpoints"
PARAMETER_GOLF_URL = "https://github.com/openai/parameter-golf"
LONG_PAPER_URL = (
    "https://github.com/amelie-iska/ToricGT/blob/oai-toricgt/"
    "assets/toricgt_toric_bgg_rewrite_amelie_schreiber.tex"
)
CONDENSED_PAPER_URL = (
    "https://github.com/amelie-iska/ToricGT/blob/oai-toricgt/"
    "assets/toricgt_neurips_condensed.tex"
)


IMAGE_SOURCES = {
    "logo": "assets/ToricGT.png",
    "architecture": "assets/toricgt_architecture_and_training_diagram.png",
    "torus": "assets/toricgt_torus_reasoning_dark.gif",
    "got": "assets/toricgt_pg_softmoe_figures/GoT-ToricGT.png",
    "tropical": "assets/toricgt_pg_softmoe_figures/fig_tropical_active_faces.png",
    "graphcg": "assets/toricgt_pg_softmoe_figures/fig_graphcg_topology_analogy_map.png",
    "branching": "assets/toricgt_pg_softmoe_figures/fig_graph_of_thought_branch_merge_dag.png",
    "protocol": "assets/toricgt_pg_softmoe_figures/fig_parameter_golf_protocol_clean.png",
    "dashboard": "assets/toricgt_pg_softmoe_figures/fig_visualization_dashboard.png",
    "fot_reference": "external/Forest-of-Thought/assets/fot.png",
    "trajectory": "outputs/branching_reasoning_trajectory_final_long_20260616T171742Z/static_screenshots/full_trajectory_filtered_complex.png",
    "analogy": "outputs/branching_reasoning_trajectory_final_long_20260616T171742Z/static_screenshots/analogical_memory_simplex_tree_map.png",
    "vectorized_ph": "outputs/branching_reasoning_trajectory_final_long_20260616T171742Z/static_screenshots/vectorized_ph_features.png",
    "cas": "outputs/smoke_oai_full_iteration_analysis_2/html_screenshots/screenshots/embedding_cas_sidecar__index__slice_00.png",
    "toric_embedding": "outputs/smoke_oai_full_iteration_analysis_2/html_screenshots/screenshots/toric_embedding_report__index__slice_00.png",
    "bgg": "outputs/smoke_oai_full_iteration_analysis_2/html_screenshots/screenshots/bgg_category_o_report__index__slice_00.png",
}

INTERACTIVE_REPORTS = {
    "branching_reasoning": {
        "source": "outputs/smoke_oai_full_iteration_analysis_2/branching_reasoning_report",
        "entry": "index.html",
        "title": "Simplex Trajectory Report",
        "summary": (
            "Potato-safe graph-of-thought trajectory report with exact summary metrics, "
            "static screenshots of the full and per-step filtered simplicial complexes, "
            "analogical memory maps, vectorized persistent-homology panels, and links to "
            "the generated analysis artifacts."
        ),
        "features": "static first · exact metrics · screenshots · 3D PCA · simplex maps · PH panels",
        "embed": False,
    },
    "gudhi_persistence": {
        "source": "outputs/smoke_oai_full_iteration_analysis_2/gudhi_persistence",
        "entry": "index.html",
        "title": "GUDHI Persistent Homology",
        "summary": (
            "Exact simplex-tree persistent homology, persistence landscapes/images/entropy, "
            "vectorized PH features, and Macaulay2 F2[x_level,y_radius] artifacts."
        ),
        "features": "PH records · landscapes/images · entropy · Macaulay2 links",
        "embed": False,
    },
    "toric_embedding": {
        "source": "outputs/smoke_oai_full_iteration_analysis_2/toric_embedding_report",
        "entry": "index.html",
        "title": "Toric Embedding and Staircases",
        "summary": (
            "Tropical ring-attention embeddings into toric charts, one-dimensional cones, "
            "Miller-Sturmfels staircase modules, staircase overlays, and fan diagnostics."
        ),
        "features": "toric charts · staircases · normal fans · one-dimensional cones",
        "embed": False,
    },
    "cas_sidecar": {
        "source": "outputs/smoke_oai_full_iteration_analysis_2/embedding_cas_sidecar",
        "entry": "index.html",
        "title": "CAS Algebra Sidecar",
        "summary": (
            "Sage/Macaulay2-backed module, resolution, syzygy, and derived-signature "
            "evidence generated for the embedding audits."
        ),
        "features": "resolutions · syzygies · CAS records · derived signatures",
        "embed": False,
    },
    "bgg_category_o": {
        "source": "outputs/smoke_oai_full_iteration_analysis_2/bgg_category_o_report",
        "entry": "index.html",
        "title": "Toric BGG Category O",
        "summary": (
            "Finite category-O certificates, standard-filtration checks, sparse "
            "differentials, Gale-dual signals, and homological consistency metrics."
        ),
        "features": "standard filtrations · differentials · Gale duality · category O",
        "embed": False,
    },
    "toric_vector_bundle": {
        "source": "outputs/smoke_oai_full_iteration_analysis_2/toric_vector_bundle_report",
        "entry": "index.html",
        "title": "Toric Vector-Bundle Sheaf Audit",
        "summary": (
            "Klyachko-style vector-bundle and sheaf compatibility evidence over "
            "one-dimensional cones and adjacent fan neighborhoods."
        ),
        "features": "1D-cone filtrations · sheaf CE · fan-neighborhood compatibility",
        "embed": False,
    },
}


LOSS_DESCRIPTIONS = {
    "graphcg_loss": "Full-rank GraphCG pressure: disentangles latent concept axes so graph and byte objectives can expose stable directions rather than collapsing into one entangled basis.",
    "analogy_loss": "Analogical-retrieval lattice loss: encourages maps between reasoning memories through simplex-tree and vectorized persistent-homology similarity gates.",
    "tokengt_graph_loss": "First-class TokenGT graph objective: trains causal node, edge, endpoint, distance, torus, and tokenization-DAG features used by the OAI FineWeb adapter.",
    "memory_loss": "Trajectory-memory retrieval loss: selects reusable reasoning traces and stabilizes retrieval-conditioned auxiliary routing.",
    "toric_geometry_loss": "Toric embedding and fan audit loss: keeps tropical ring attention active faces embedded into toric charts with interpretable cone/fan structure.",
    "toric_vector_bundle_1d_cone_ce_loss": "Toric vector-bundle one-dimensional-cone/sheaf CE: regularizes per-cone filtrations and sheaf-style compatibility across fan neighborhoods.",
    "toric_bgg_loss": "Toric BGG category-O supervision: keeps finite category-O certificates, standard filtrations, Gale-dual labels, and differential consistency visible.",
    "koszul_persistence_loss": "Koszul/persistence loss: checks filtered complexes, multigraded modules, and chain-complex consistency for topological reasoning traces.",
    "toric_cca_topology_loss": "Combinatorial toric commutative algebra/topology loss: tracks monomial staircases, syzygies, and CAS-backed toric module diagnostics.",
    "derived_signature_loss": "Derived-signature distillation: compresses resolution, chain-map, and derived-category evidence into low-rank signatures for training-time audits.",
    "oai_gflownet_loss": "Embedding-space GFlowNet loss: samples graph-of-thought continuations with reward tied to byte-likelihood, retrieval utility, and structural diversity.",
    "oai_fot_loss": "Embedding-space Forest-of-Thought loss: searches branching reasoning forests and uses BPB-delta reward to prioritize branches that improve compression.",
    "oai_mtp_loss": "Multi-token prediction loss: adds short-horizon targets that improve early byte likelihood while staying compatible with graph output flattening.",
    "graph_lm_loss": "Graph-LM primary loss: trains graph-structured records directly while FineWeb output flattening remains optional and BPB-focused.",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "pending"
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        if abs(float(value)) >= 10_000:
            return f"{float(value):,.0f}"
        return f"{float(value):.{digits}f}"
    return html.escape(str(value))


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def find_campaign_state(repo: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = repo / path
        return path if path.exists() else None
    candidates = sorted(
        repo.glob("training_notes/*/campaign_state.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def metric_bpb(metrics: dict[str, Any]) -> float:
    for key in ("final_int8_bpb", "val_bpb", "bpb", "train_bpb"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return float("inf")


def completed_rows(state: dict[str, Any], *, current_campaign_only: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    campaign_id = str(state.get("campaign_id", ""))
    for row in state.get("history", []):
        if not isinstance(row, dict):
            continue
        run_id = str(row.get("run_id", ""))
        if current_campaign_only and campaign_id and not run_id.startswith(campaign_id):
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        if not math.isfinite(metric_bpb(metrics)):
            continue
        if not metrics.get("checkpoint_path") and not metrics.get("final_int8_bpb"):
            continue
        rows.append(row)
    return rows


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: metric_bpb(row.get("metrics", {})))


def find_pr_url(repo: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    notes = sorted(repo.glob("training_notes/**/*PARAMETER-GOLF-PR*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in notes:
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"https://github\.com/openai/parameter-golf/pull/\d+", text)
        if match:
            return match.group(0)
    return ""


def find_analysis_output(repo: Path, explicit: str | None = None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = repo / path
        return path if path.exists() else None
    candidates = sorted(
        repo.glob("outputs/github_pages_best_checkpoint_long_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def copy_assets(repo: Path, docs: Path, analysis_dir: Path | None = None) -> dict[str, str]:
    assets_dir = docs / "page_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    sources = dict(IMAGE_SOURCES)
    if analysis_dir and analysis_dir.exists():
        dynamic_sources = {
            "trajectory": analysis_dir / "branching_reasoning_report" / "static_screenshots" / "full_trajectory_filtered_complex.png",
            "analogy": analysis_dir / "branching_reasoning_report" / "static_screenshots" / "analogical_memory_simplex_tree_map.png",
            "vectorized_ph": analysis_dir / "branching_reasoning_report" / "static_screenshots" / "vectorized_ph_features.png",
            "cas": analysis_dir / "html_screenshots" / "screenshots" / "embedding_cas_sidecar__index__slice_00.png",
            "toric_embedding": analysis_dir / "html_screenshots" / "screenshots" / "toric_embedding_report__index__slice_00.png",
            "bgg": analysis_dir / "html_screenshots" / "screenshots" / "bgg_category_o_report__index__slice_00.png",
        }
        for key, candidate in dynamic_sources.items():
            if candidate.exists():
                sources[key] = str(candidate)
    for key, rel in sources.items():
        src = repo / rel
        if Path(rel).is_absolute():
            src = Path(rel)
        if not src.exists():
            continue
        dst = assets_dir / f"{key}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        copied[key] = f"page_assets/{dst.name}"
    return copied


def copy_interactive_reports(repo: Path, docs: Path, analysis_dir: Path | None = None) -> dict[str, dict[str, str]]:
    out_dir = docs / "page_interactive"
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict[str, str]] = {}
    for key, spec in INTERACTIVE_REPORTS.items():
        src = repo / spec["source"]
        if analysis_dir and analysis_dir.exists():
            candidate = analysis_dir / Path(str(spec["source"])).name
            if (candidate / str(spec["entry"])).exists():
                src = candidate
        entry = str(spec["entry"])
        if not src.exists() or not (src / entry).exists():
            continue
        dst = out_dir / key
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        if key == "branching_reasoning":
            write_branching_reasoning_lite_report(dst)
        if key == "gudhi_persistence":
            enhance_gudhi_persistence_report(dst)
        if key == "toric_embedding":
            write_toric_embedding_lite_report(dst)
        copied[key] = {
            "href": f"page_interactive/{key}/{entry}",
            "title": str(spec["title"]),
            "summary": str(spec["summary"]),
            "features": str(spec["features"]),
            "embed": "1" if spec.get("embed") else "0",
        }
    scrub_public_report_paths(out_dir, repo)
    return copied


def enhance_gudhi_persistence_report(dst: Path) -> None:
    extra_css = """
/* ToricGT GitHub Pages low-resource layout polish. */
main{width:min(1420px,calc(100vw - 24px));max-width:1420px;padding-inline:12px}
.hero,.card,.panel{overflow:hidden}
.plot{overflow-x:auto;overflow-y:hidden;border:1px solid rgba(55,232,255,.18);border-radius:10px;background:#020713;padding:6px}
.plot>div{min-width:980px;max-width:none}
.tablewrap{max-width:100%;overflow:auto}
pre{max-height:min(72vh,720px);overflow:auto;font-size:12px;line-height:1.4}
details[open] pre{box-shadow:inset 0 0 0 1px rgba(55,232,255,.10)}
svg.xygrid{max-height:82vh;object-fit:contain}
@media(max-width:760px){main{width:100%;padding:10px}.plot>div{min-width:760px}h1{font-size:22px}.grid{grid-template-columns:1fr}}
"""
    for path in dst.glob("records/*.html"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "ToricGT GitHub Pages low-resource layout polish" not in text:
            text = text.replace("</style>", extra_css + "\n</style>", 1)
        text = text.replace("<div class=\"plot\"><div style=\"height:1180px; width:100%;\">", "<div class=\"plot\"><div style=\"height:980px; width:100%;\">")
        text = text.replace("<h2>Plotly Feature Panel</h2>", "<h2>Vectorized PH Feature Panel</h2>")
        path.write_text(text, encoding="utf-8")


def write_branching_reasoning_lite_report(dst: Path) -> None:
    original_index = dst / "index.html"
    payload_path = dst / "branching_reasoning_payload.json"
    compact_payload_written = False
    if payload_path.exists():
        try:
            payload = load_json(payload_path)
            compact_nodes = []
            for node in payload.get("nodes", []):
                pca = node.get("pca", [0, 0, 0])
                compact_nodes.append(
                    {
                        "id": int(node.get("id", len(compact_nodes))),
                        "label": str(node.get("label", node.get("id", len(compact_nodes)))),
                        "x": round(float(pca[0] if len(pca) > 0 else 0.0), 5),
                        "y": round(float(pca[1] if len(pca) > 1 else 0.0), 5),
                        "z": round(float(pca[2] if len(pca) > 2 else 0.0), 5),
                        "level": int(node.get("level", 0) or 0),
                        "nll": round(float(node.get("nll", 0.0) or 0.0), 4),
                        "branch": int(node.get("branch", 0) or 0),
                        "kind": str(node.get("node_kind", "node")),
                    }
                )

            def compact_pair_edges(rows: list[Any]) -> list[list[int]]:
                out: list[list[int]] = []
                for row in rows:
                    if isinstance(row, (list, tuple)) and len(row) >= 2:
                        out.append([int(row[0]), int(row[1])])
                return out

            radius_edges: list[list[float]] = []
            for row in payload.get("edge_births", []):
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    radius_edges.append([int(row[0]), int(row[1]), int(row[2]), round(float(row[3]), 4)])
            triangles: list[list[float]] = []
            for row in payload.get("triangles", []):
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    triangles.append([int(row[0]), int(row[1]), int(row[2]), round(float(row[3]), 4)])
            compact_steps: list[dict[str, Any]] = []
            for step in payload.get("steps", []):
                if not isinstance(step, dict):
                    continue
                step_tokens: list[dict[str, Any]] = []
                for token in step.get("tokens", []):
                    if not isinstance(token, dict):
                        continue
                    pca = token.get("pca", [0, 0, 0])
                    step_tokens.append(
                        {
                            "id": int(token.get("id", len(step_tokens)) or 0),
                            "global_token_id": int(token.get("global_token_id", -1) or -1),
                            "decode_order": int(token.get("decode_order", len(step_tokens)) or 0),
                            "text": str(token.get("text", "")),
                            "token_type": str(token.get("token_type", "")),
                            "x": round(float(pca[0] if len(pca) > 0 else 0.0), 5),
                            "y": round(float(pca[1] if len(pca) > 1 else 0.0), 5),
                            "z": round(float(pca[2] if len(pca) > 2 else 0.0), 5),
                            "nll": round(float(token.get("nll", 0.0) or 0.0), 4),
                            "entropy": round(float(token.get("entropy", 0.0) or 0.0), 4),
                            "rank": int(token.get("rank", 0) or 0),
                        }
                    )

                step_edges: list[list[float]] = []
                for row in step.get("edge_births", []):
                    if isinstance(row, (list, tuple)) and len(row) >= 4:
                        step_edges.append([int(row[0]), int(row[1]), int(row[2]), round(float(row[3]), 4)])
                step_triangles: list[list[float]] = []
                for row in step.get("triangles", []):
                    if isinstance(row, (list, tuple)) and len(row) >= 4:
                        step_triangles.append([int(row[0]), int(row[1]), int(row[2]), round(float(row[3]), 4)])
                step_decode_edges: list[list[int]] = []
                for row in step.get("decode_edges", []):
                    if isinstance(row, (list, tuple)) and len(row) >= 3:
                        step_decode_edges.append([int(row[0]), int(row[1]), int(row[2])])
                compact_steps.append(
                    {
                        "step_index": int(step.get("step_index", len(compact_steps)) or 0),
                        "level": int(step.get("level", 0) or 0),
                        "label": str(step.get("label", "")),
                        "tokens": step_tokens,
                        "decode_edges": step_decode_edges,
                        "edge_births": step_edges,
                        "triangles": step_triangles,
                        "radius_values": [round(float(v), 4) for v in step.get("radius_values", [])],
                        "triangle_count_exact": int(step.get("triangle_count_exact", len(step_triangles)) or len(step_triangles)),
                    }
                )
            compact_payload = {
                "schema": "toricgt.branching_reasoning_interactive_compact.v1",
                "source_mode": payload.get("source_mode", "unknown"),
                "nodes": compact_nodes,
                "steps": compact_steps,
                "dag_edges": compact_pair_edges(payload.get("dag_edges", [])),
                "reasoning_edges": compact_pair_edges(payload.get("reasoning_order_edges", [])),
                "radius_edges": radius_edges,
                "triangles": triangles,
                "radius_values": [round(float(v), 4) for v in payload.get("radius_values", [])],
                "analogy": {
                    "status": payload.get("analogy", {}).get("analogy_status"),
                    "confidence": payload.get("analogy", {}).get("analogy_confidence_score"),
                    "emitted": payload.get("analogy", {}).get("analogy_emitted"),
                    "decision_summary": payload.get("analogy", {}).get("decision_summary"),
                },
            }
            (dst / "branching_reasoning_interactive.json").write_text(
                json.dumps(compact_payload, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
            compact_payload_written = True
        except Exception as exc:
            (dst / "branching_reasoning_interactive_error.txt").write_text(str(exc), encoding="utf-8")
    text = original_index.read_text(encoding="utf-8", errors="replace") if original_index.exists() else ""
    metrics = re.findall(r"<div class='metric'><span>(.*?)</span><span>(.*?)</span></div>", text, flags=re.DOTALL)
    metric_rows = []
    for name, value in metrics:
        metric_rows.append(
            "<tr><th>"
            + html.escape(re.sub(r"<.*?>", "", name).strip())
            + "</th><td>"
            + html.escape(re.sub(r"<.*?>", "", value).strip())
            + "</td></tr>"
        )
    if not metric_rows:
        metric_rows = [
            "<tr><th>report mode</th><td>static low-resource summary</td></tr>",
            "<tr><th>source</th><td>generated branching reasoning analysis bundle</td></tr>",
        ]
    screenshot_specs = [
        ("summary.png", "Summary Dashboard", "Summary screenshot generated from the exact branching-reasoning payload."),
        (
            "full_trajectory_filtered_complex.png",
            "Full Trajectory Filtered Simplicial Complex",
            "3D PCA display of the full reasoning trajectory with one-dimensional simplex edges and reasoning-level structure.",
        ),
        (
            "selected_step_simplex_tree_tokens.png",
            "Selected Step Simplex Tree and Tokens",
            "Per-step simplex tree view with token metadata, NLL coloring, and local filtered-complex evidence.",
        ),
        (
            "analogical_memory_simplex_tree_map.png",
            "Analogical Memory Simplex-Tree Map",
            "Source and retrieved memory structures with simplex-map evidence and PH similarity gates.",
        ),
        (
            "vectorized_ph_features.png",
            "Vectorized Persistent-Homology Features",
            "Persistence landscapes, images, entropy, and vectorized comparisons used for analogy gating.",
        ),
    ]
    cards = []
    for filename, title, caption in screenshot_specs:
        path = dst / "static_screenshots" / filename
        if not path.exists():
            continue
        cards.append(
            f"""
            <figure class="shot">
              <a href="static_screenshots/{html.escape(filename)}"><img loading="lazy" decoding="async" src="static_screenshots/{html.escape(filename)}" alt="{html.escape(title)}"></a>
              <figcaption><strong>{html.escape(title)}</strong><span>{html.escape(caption)}</span></figcaption>
            </figure>
            """
        )
    for heavy in ("branching_reasoning_trajectory.html", "branching_reasoning_payload.json"):
        path = dst / heavy
        if path.exists():
            path.unlink()
    (dst / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "toricgt.branching_reasoning_lite.v1",
                "entry": "index.html",
                "mode": "low_resource_static",
                "removed_heavy_browser_payload_count": 2,
                "interactive_payload": "branching_reasoning_interactive.json" if compact_payload_written else None,
                "static_screenshots": [
                    f"static_screenshots/{filename}"
                    for filename, _, _ in screenshot_specs
                    if (dst / "static_screenshots" / filename).exists()
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    static_manifest = dst / "static_screenshots" / "manifest.json"
    if static_manifest.exists():
        static_data = load_json(static_manifest)
        static_data.pop("payload_json", None)
        static_data.pop("source_html", None)
        static_data["mode"] = "static_screenshot_evidence"
        static_manifest.write_text(json.dumps(static_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (dst / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Branching Reasoning Trajectory Report</title>
  <style>
    :root{{color-scheme:dark;--bg:#030712;--panel:#07111f;--line:rgba(55,232,255,.28);--text:#e8fbff;--muted:#9fb1c9;--cyan:#37e8ff;--gold:#ffd166}}
    *{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top left,#092238 0,#030712 42rem);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,sans-serif}}
    main{{max-width:1180px;margin:0 auto;padding:28px 18px 64px}}a{{color:var(--cyan)}}.hero,.panel,.shot{{background:linear-gradient(180deg,rgba(11,23,40,.96),rgba(5,13,25,.98));border:1px solid var(--line);border-radius:10px;box-shadow:0 18px 50px rgba(0,0,0,.28)}}
    .hero,.panel{{padding:18px;margin:14px 0}}h1{{margin:0 0 8px;font-size:clamp(2rem,5vw,3rem)}}h2{{margin:0 0 12px}}p{{color:var(--muted);line-height:1.55}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}}
    table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid rgba(159,177,201,.16);padding:9px;text-align:left;vertical-align:top}}th{{color:#bdeeff;width:42%}}td{{color:#fff}}
    .shot{{margin:0;overflow:hidden}}.shot img{{display:block;width:100%;height:auto;background:#020713}}.shot figcaption{{display:grid;gap:5px;padding:12px 14px}}.shot span{{color:var(--muted);line-height:1.45}}.pill{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:8px 12px;margin:4px 8px 4px 0;background:rgba(55,232,255,.08);font-weight:700}}
    .note{{border-left:3px solid var(--gold);padding:10px 12px;background:rgba(255,209,102,.08);border-radius:8px;color:#ffe8a8}}.controls{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:10px 0 14px}}.controls label{{display:grid;gap:4px;color:var(--muted);font-size:.9rem}}.controls input{{accent-color:var(--cyan)}}button,select{{border:1px solid var(--line);background:rgba(55,232,255,.08);color:var(--text);border-radius:999px;padding:8px 12px;font-weight:800}}canvas{{width:100%;height:auto;border:1px solid rgba(55,232,255,.18);border-radius:10px;background:#020713;display:block}}.detail{{border:1px solid rgba(55,232,255,.18);border-radius:10px;background:#020713;padding:12px;margin-top:10px;color:#dff8ff}}.split{{display:grid;grid-template-columns:1fr;gap:14px}}@media(min-width:1100px){{.split{{grid-template-columns:1.05fr .95fr}}}}
  </style>
</head>
<body><main>
  <section class="hero">
    <h1>Branching Reasoning Trajectory Report</h1>
    <p>This is the low-resource public entry point for the graph-of-thought simplex trajectory. It keeps the exact summary metrics and visual evidence while avoiding the large browser-side payload that can freeze older machines.</p>
    <p><a class="pill" href="static_screenshots/index.html">Static screenshot audit</a><a class="pill" href="static_screenshots/contact_sheet.png">Contact sheet</a><a class="pill" href="manifest.json">Manifest</a></p>
    <p class="note">The heavyweight Plotly/WebGL artifact is not the default GitHub Pages payload. This page restores the core interactions with compact native Canvas/SVG views so it can load on low-resource machines while keeping the exact metrics and screenshots.</p>
  </section>
  <section class="panel">
    <h2>Exact Summary Metrics</h2>
    <table><tbody>{''.join(metric_rows)}</tbody></table>
  </section>
  <section class="panel">
    <h2>Interactive Native 3D PCA Trajectory</h2>
    <p>This viewer restores rotation, pitch, zoom, reasoning-level filtering, radius filtering, one-dimensional simplex edges, and sampled 2-simplices without shipping hidden embeddings or the heavyweight plotting runtime.</p>
    <div class="controls">
      <label>reasoning level <input id="levelRange" type="range" min="0" max="1" value="1"></label>
      <label>radius <input id="radiusRange" type="range" min="0" max="1" value="0"></label>
      <label>yaw <input id="rotRange" type="range" min="-180" max="180" value="28"></label>
      <label>pitch <input id="pitchRange" type="range" min="-70" max="70" value="-14"></label>
      <label>zoom <input id="zoomRange" type="range" min="55" max="165" value="100"></label>
      <label><input id="radiusEdgesToggle" type="checkbox" checked> radius simplex edges</label>
      <label><input id="triToggle" type="checkbox"> 2-simplices</label>
    </div>
    <canvas id="trajectoryCanvas" width="1120" height="640"></canvas>
    <div id="trajectoryDetail" class="detail">Move over or click a point to inspect the reasoning node, NLL, branch, and level.</div>
  </section>
  <section class="panel">
    <h2>Reasoning-Step Filtered Simplicial Complex</h2>
    <p>Each reasoning step uses its real compact token PCA coordinates from the analysis payload. The radius slider adds one-dimensional simplex edges by distance; the decoding slider reveals token order and decode edges.</p>
    <div class="controls">
      <label>step <select id="stepSelect"></select></label>
      <label>radius <input id="stepRadiusRange" type="range" min="0" max="1" value="0"></label>
      <label>decoding order <input id="decodeRange" type="range" min="0" max="1" value="1"></label>
      <label>yaw <input id="stepYawRange" type="range" min="-180" max="180" value="36"></label>
      <label>pitch <input id="stepPitchRange" type="range" min="-70" max="70" value="-18"></label>
      <label>zoom <input id="stepZoomRange" type="range" min="65" max="190" value="115"></label>
      <label><input id="stepDecodeToggle" type="checkbox" checked> decode arrows</label>
      <label><input id="stepEdgeToggle" type="checkbox" checked> radius simplex edges</label>
      <label><input id="stepTriToggle" type="checkbox"> 2-simplices</label>
    </div>
    <canvas id="stepCanvas" width="1120" height="560"></canvas>
    <div id="stepDetail" class="detail">Move over or click a token point to inspect token text, type, NLL, entropy, rank, and source id.</div>
  </section>
  <section class="panel">
    <h2>Visualizations</h2>
    <div class="grid">{''.join(cards)}</div>
  </section>
  <script>
const canvas = document.getElementById('trajectoryCanvas');
const ctx = canvas.getContext('2d');
const levelRange = document.getElementById('levelRange');
const radiusRange = document.getElementById('radiusRange');
const rotRange = document.getElementById('rotRange');
const pitchRange = document.getElementById('pitchRange');
const zoomRange = document.getElementById('zoomRange');
const detail = document.getElementById('trajectoryDetail');
const edgeToggle = document.getElementById('radiusEdgesToggle');
const triToggle = document.getElementById('triToggle');
const stepCanvas = document.getElementById('stepCanvas');
const stepCtx = stepCanvas.getContext('2d');
const stepSelect = document.getElementById('stepSelect');
const stepRadiusRange = document.getElementById('stepRadiusRange');
const decodeRange = document.getElementById('decodeRange');
const stepYawRange = document.getElementById('stepYawRange');
const stepPitchRange = document.getElementById('stepPitchRange');
const stepZoomRange = document.getElementById('stepZoomRange');
const stepDecodeToggle = document.getElementById('stepDecodeToggle');
const stepEdgeToggle = document.getElementById('stepEdgeToggle');
const stepTriToggle = document.getElementById('stepTriToggle');
const stepDetail = document.getElementById('stepDetail');
let compact = null;
let projected = [];
let projectedById = new Map();
let stepProjected = [];
let stepProjectedById = new Map();
let stepVisibleIds = new Set();
function esc(value) {{ return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch])); }}
function colorForNll(nll, minNll, maxNll) {{
  const t = Math.max(0, Math.min(1, (nll - minNll) / Math.max(1e-6, maxNll - minNll)));
  const r = Math.round(70 + 185 * t), g = Math.round(230 - 120 * t), b = Math.round(255 - 200 * t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function rotated3(point, yawDeg, pitchDeg) {{
  const yaw = yawDeg * Math.PI / 180, pitch = pitchDeg * Math.PI / 180;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = point.x * cy - point.z * sy;
  const z1 = point.x * sy + point.z * cy;
  const y1 = point.y * cp - z1 * sp;
  const z2 = point.y * sp + z1 * cp;
  return {{x:x1, y:y1, z:z2}};
}}
function projectCollection(points, targetCanvas, yaw, pitch, zoom) {{
  if (!points.length) return [];
  const rotated = points.map(p => ({{...rotated3(p, yaw, pitch), node:p}}));
  const minX = Math.min(...rotated.map(p => p.x)), maxX = Math.max(...rotated.map(p => p.x));
  const minY = Math.min(...rotated.map(p => p.y)), maxY = Math.max(...rotated.map(p => p.y));
  const baseScale = Math.min(
    (targetCanvas.width - 150) / Math.max(1e-6, maxX - minX),
    (targetCanvas.height - 130) / Math.max(1e-6, maxY - minY)
  ) * (zoom / 100);
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  return rotated.map(p => {{
    const perspective = 780 / (780 + p.z * 42);
    return {{
      px: targetCanvas.width / 2 + (p.x - cx) * baseScale * perspective,
      py: targetCanvas.height / 2 - (p.y - cy) * baseScale * perspective,
      depth: p.z,
      node: p.node,
    }};
  }});
}}
function drawArrow(context, a, b, color, width) {{
  if (!a || !b) return;
  context.strokeStyle=color; context.fillStyle=color; context.lineWidth=width;
  context.beginPath(); context.moveTo(a.px, a.py); context.lineTo(b.px, b.py); context.stroke();
  const angle = Math.atan2(b.py - a.py, b.px - a.px);
  const len = 9;
  context.beginPath();
  context.moveTo(b.px, b.py);
  context.lineTo(b.px - len * Math.cos(angle - 0.42), b.py - len * Math.sin(angle - 0.42));
  context.lineTo(b.px - len * Math.cos(angle + 0.42), b.py - len * Math.sin(angle + 0.42));
  context.closePath(); context.fill();
}}
function bindDrag(targetCanvas, yawInput, pitchInput, render) {{
  let down = false, lastX = 0, lastY = 0;
  targetCanvas.addEventListener('pointerdown', event => {{ down = true; lastX = event.clientX; lastY = event.clientY; targetCanvas.setPointerCapture(event.pointerId); }});
  targetCanvas.addEventListener('pointermove', event => {{
    if (!down) return;
    const dx = event.clientX - lastX, dy = event.clientY - lastY;
    lastX = event.clientX; lastY = event.clientY;
    yawInput.value = Math.max(Number(yawInput.min), Math.min(Number(yawInput.max), Number(yawInput.value) + dx * 0.45));
    pitchInput.value = Math.max(Number(pitchInput.min), Math.min(Number(pitchInput.max), Number(pitchInput.value) - dy * 0.35));
    render();
  }});
  targetCanvas.addEventListener('pointerup', () => {{ down = false; }});
  targetCanvas.addEventListener('pointercancel', () => {{ down = false; }});
}}
function draw() {{
  if (!compact) return;
  const maxLevel = Number(levelRange.value);
  const radiusIndex = Number(radiusRange.value);
  const radius = compact.radius_values[Math.min(radiusIndex, compact.radius_values.length - 1)] ?? 0;
  const visible = compact.nodes.filter(n => n.level <= maxLevel);
  if (!visible.length) return;
  const visibleIds = new Set(visible.map(n => n.id));
  projected = projectCollection(compact.nodes, canvas, Number(rotRange.value), Number(pitchRange.value), Number(zoomRange.value));
  projectedById = new Map(projected.map(p => [p.node.id, p]));
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = '#020713'; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle = 'rgba(157,184,207,.18)'; ctx.lineWidth = 1;
  for (let i=0;i<7;i++) {{ const y=70+i*(canvas.height-140)/6; ctx.beginPath(); ctx.moveTo(60,y); ctx.lineTo(canvas.width-60,y); ctx.stroke(); }}
  function line(a,b,color,width) {{
    const pa = projectedById.get(Number(a)), pb = projectedById.get(Number(b));
    if (!pa || !pb || !visibleIds.has(pa.node.id) || !visibleIds.has(pb.node.id)) return;
    ctx.strokeStyle=color; ctx.lineWidth=width; ctx.beginPath(); ctx.moveTo(pa.px, pa.py); ctx.lineTo(pb.px, pb.py); ctx.stroke();
  }}
  for (const [a,b] of compact.reasoning_edges) line(a,b,'rgba(255,209,102,.52)',1.2);
  for (const [a,b] of compact.dag_edges) line(a,b,'rgba(70,231,255,.45)',1.1);
  if (edgeToggle.checked) for (const [a,b,l,birth] of compact.radius_edges) if (l <= maxLevel && birth <= radius) line(a,b,'rgba(136,255,134,.12)',0.65);
  if (triToggle.checked) {{
    ctx.fillStyle='rgba(167,139,250,.10)';
    for (const [a,b,c,birth] of compact.triangles) if (birth <= radius && visibleIds.has(a) && visibleIds.has(b) && visibleIds.has(c)) {{
      const pa=projectedById.get(Number(a)), pb=projectedById.get(Number(b)), pc=projectedById.get(Number(c)); if (!pa||!pb||!pc) continue;
      ctx.beginPath(); ctx.moveTo(pa.px,pa.py); ctx.lineTo(pb.px,pb.py); ctx.lineTo(pc.px,pc.py); ctx.closePath(); ctx.fill();
    }}
  }}
  const nlls = visible.map(n => n.nll), minNll = Math.min(...nlls), maxNll = Math.max(...nlls);
  for (const p of projected.filter(p => visibleIds.has(p.node.id)).sort((a,b)=>a.depth-b.depth)) {{
    ctx.fillStyle = colorForNll(p.node.nll, minNll, maxNll);
    ctx.strokeStyle = '#ecfbff'; ctx.lineWidth = 1.1;
    ctx.beginPath(); ctx.arc(p.px, p.py, 4.5, 0, Math.PI*2); ctx.fill(); ctx.stroke();
  }}
  ctx.fillStyle='#ecfbff'; ctx.font='16px system-ui'; ctx.fillText(`level ≤ ${{maxLevel}} · radius ${{radius.toFixed(3)}} · nodes ${{visible.length}} · yaw ${{rotRange.value}} · pitch ${{pitchRange.value}}`, 22, 34);
}}
function nearest(event) {{
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * canvas.width / rect.width;
  const y = (event.clientY - rect.top) * canvas.height / rect.height;
  let best=null, bd=Infinity;
  for (const p of projected) {{
    const d=(p.px-x)**2+(p.py-y)**2;
    if (d<bd) {{bd=d; best=p;}}
  }}
  if (best && bd < 400) detail.innerHTML = `<strong>${{esc(best.node.label)}} / id ${{best.node.id}}</strong><br>level ${{best.node.level}} · branch ${{best.node.branch}} · NLL ${{best.node.nll.toFixed(4)}} · kind ${{esc(best.node.kind)}}`;
}}
function currentStep() {{
  if (!compact?.steps?.length) return null;
  return compact.steps[Math.max(0, Math.min(compact.steps.length - 1, Number(stepSelect.value) || 0))];
}}
function drawStep() {{
  const step = currentStep();
  if (!step) return;
  const maxDecode = Number(decodeRange.value);
  const radiusIndex = Number(stepRadiusRange.value);
  const radius = step.radius_values[Math.min(radiusIndex, step.radius_values.length - 1)] ?? 0;
  const visible = step.tokens.filter(t => t.decode_order <= maxDecode);
  const visibleIds = new Set(visible.map(t => t.id));
  stepVisibleIds = visibleIds;
  stepProjected = projectCollection(step.tokens, stepCanvas, Number(stepYawRange.value), Number(stepPitchRange.value), Number(stepZoomRange.value));
  stepProjectedById = new Map(stepProjected.map(p => [p.node.id, p]));
  stepCtx.clearRect(0,0,stepCanvas.width,stepCanvas.height);
  stepCtx.fillStyle = '#020713'; stepCtx.fillRect(0,0,stepCanvas.width,stepCanvas.height);
  stepCtx.strokeStyle = 'rgba(157,184,207,.18)'; stepCtx.lineWidth = 1;
  for (let i=0;i<6;i++) {{ const y=58+i*(stepCanvas.height-116)/5; stepCtx.beginPath(); stepCtx.moveTo(60,y); stepCtx.lineTo(stepCanvas.width-60,y); stepCtx.stroke(); }}
  function stepLine(a,b,color,width,arrow=false) {{
    const pa = stepProjectedById.get(Number(a)), pb = stepProjectedById.get(Number(b));
    if (!pa || !pb || !visibleIds.has(pa.node.id) || !visibleIds.has(pb.node.id)) return;
    if (arrow) drawArrow(stepCtx, pa, pb, color, width);
    else {{ stepCtx.strokeStyle=color; stepCtx.lineWidth=width; stepCtx.beginPath(); stepCtx.moveTo(pa.px, pa.py); stepCtx.lineTo(pb.px, pb.py); stepCtx.stroke(); }}
  }}
  if (stepTriToggle.checked) {{
    stepCtx.fillStyle='rgba(167,139,250,.14)';
    for (const [a,b,c,birth] of step.triangles) if (birth <= radius && visibleIds.has(a) && visibleIds.has(b) && visibleIds.has(c)) {{
      const pa=stepProjectedById.get(Number(a)), pb=stepProjectedById.get(Number(b)), pc=stepProjectedById.get(Number(c)); if (!pa||!pb||!pc) continue;
      stepCtx.beginPath(); stepCtx.moveTo(pa.px,pa.py); stepCtx.lineTo(pb.px,pb.py); stepCtx.lineTo(pc.px,pc.py); stepCtx.closePath(); stepCtx.fill();
    }}
  }}
  if (stepEdgeToggle.checked) for (const [a,b,l,birth] of step.edge_births) if (birth <= radius) stepLine(a,b,'rgba(136,255,134,.22)',0.9,false);
  if (stepDecodeToggle.checked) for (const [a,b,order] of step.decode_edges) if (order <= maxDecode) stepLine(a,b,'rgba(255,209,102,.62)',1.6,true);
  const nlls = visible.map(t => t.nll), minNll = nlls.length ? Math.min(...nlls) : 0, maxNll = nlls.length ? Math.max(...nlls) : 1;
  for (const p of stepProjected.filter(p => visibleIds.has(p.node.id)).sort((a,b)=>a.depth-b.depth)) {{
    stepCtx.fillStyle = colorForNll(p.node.nll, minNll, maxNll);
    stepCtx.strokeStyle = '#ecfbff'; stepCtx.lineWidth = 1.1;
    stepCtx.beginPath(); stepCtx.arc(p.px, p.py, 6, 0, Math.PI*2); stepCtx.fill(); stepCtx.stroke();
    stepCtx.fillStyle = '#dff8ff'; stepCtx.font = '11px system-ui'; stepCtx.fillText(String(p.node.decode_order), p.px + 7, p.py - 6);
  }}
  stepCtx.fillStyle='#ecfbff'; stepCtx.font='16px system-ui'; stepCtx.fillText(`${{step.label || ('step '+step.step_index)}} · tokens ${{visible.length}}/${{step.tokens.length}} · radius ${{radius.toFixed(3)}} · decode ≤ ${{maxDecode}}`, 22, 32);
}}
function nearestStep(event) {{
  const rect = stepCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * stepCanvas.width / rect.width;
  const y = (event.clientY - rect.top) * stepCanvas.height / rect.height;
  let best=null, bd=Infinity;
  for (const p of stepProjected) {{
    if (!stepVisibleIds.has(p.node.id)) continue;
    const d=(p.px-x)**2+(p.py-y)**2;
    if (d<bd) {{bd=d; best=p;}}
  }}
  if (best && bd < 650) stepDetail.innerHTML = `<strong>${{esc(best.node.text || ('token '+best.node.id))}}</strong><br>type ${{esc(best.node.token_type)}} · local id ${{best.node.id}} · source node ${{best.node.global_token_id}} · decode ${{best.node.decode_order}}<br>NLL ${{best.node.nll.toFixed(4)}} · entropy ${{best.node.entropy.toFixed(4)}} · rank ${{best.node.rank}}`;
}}
fetch('branching_reasoning_interactive.json').then(r => r.json()).then(data => {{
  compact = data;
  const maxLevel = Math.max(...compact.nodes.map(n => n.level));
  levelRange.max = maxLevel; levelRange.value = maxLevel;
  radiusRange.max = Math.max(0, compact.radius_values.length - 1); radiusRange.value = Math.max(0, Math.floor((compact.radius_values.length - 1) / 3));
  [levelRange, radiusRange, rotRange, pitchRange, zoomRange, edgeToggle, triToggle].forEach(el => el.addEventListener('input', draw));
  canvas.addEventListener('mousemove', nearest); canvas.addEventListener('click', nearest);
  bindDrag(canvas, rotRange, pitchRange, draw);
  if (compact.steps?.length) {{
    stepSelect.innerHTML = compact.steps.map((s, i) => `<option value="${{i}}">${{esc(s.label || ('step '+s.step_index))}} · ${{s.tokens.length}} tokens</option>`).join('');
    stepSelect.value = String(Math.min(compact.steps.length - 1, Math.floor(compact.steps.length / 2)));
    function configureStepControls() {{
      const step = currentStep();
      if (!step) return;
      const maxDecode = Math.max(...step.tokens.map(t => t.decode_order));
      decodeRange.max = maxDecode; decodeRange.value = maxDecode;
      stepRadiusRange.max = Math.max(0, step.radius_values.length - 1);
      stepRadiusRange.value = Math.max(0, Math.floor((step.radius_values.length - 1) / 2));
      drawStep();
    }}
    stepSelect.addEventListener('change', configureStepControls);
    [stepRadiusRange, decodeRange, stepYawRange, stepPitchRange, stepZoomRange, stepDecodeToggle, stepEdgeToggle, stepTriToggle].forEach(el => el.addEventListener('input', drawStep));
    stepCanvas.addEventListener('mousemove', nearestStep); stepCanvas.addEventListener('click', nearestStep);
    bindDrag(stepCanvas, stepYawRange, stepPitchRange, drawStep);
    configureStepControls();
  }} else {{
    stepDetail.textContent = 'Per-step compact simplex payload is unavailable for this report.';
  }}
  draw();
}}).catch(error => {{ detail.textContent = 'Compact interactive payload unavailable: '+error; }});
  </script>
</main></body></html>
""",
        encoding="utf-8",
    )


def write_toric_embedding_lite_report(dst: Path) -> None:
    records_dir = dst / "records"
    summaries = sorted(records_dir.glob("*_summary.json")) if records_dir.exists() else []
    cards = []
    interactive_records: list[dict[str, Any]] = []
    for summary_path in summaries:
        summary = load_json(summary_path)
        record_id = str(summary.get("record_id", summary_path.stem.replace("_summary", "")))
        exactness = summary.get("exactness", {}) if isinstance(summary.get("exactness"), dict) else {}
        staircase = summary.get("miller_sturmfels_staircase", {}) if isinstance(summary.get("miller_sturmfels_staircase"), dict) else {}
        minimal_generators = staircase.get("minimal_generators", [])
        quotient_basis = staircase.get("quotient_basis", [])
        adjacent = staircase.get("adjacent_lcm_layer", [])
        interactive_records.append(
            {
                "record_id": record_id,
                "exponent_count": summary.get("exponent_count"),
                "exponent_dim": summary.get("exponent_dim"),
                "minimal_generators": minimal_generators,
                "quotient_basis": quotient_basis,
                "adjacent_lcm_layer": adjacent,
                "staircase_threshold_by_x": staircase.get("staircase_threshold_by_x", []),
                "window": staircase.get("window", {}),
                "exactness": exactness,
            }
        )
        cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(record_id)}</h2>
              <table>
                <tbody>
                  <tr><th>exponents</th><td>{fmt(summary.get("exponent_count"), 0)}</td></tr>
                  <tr><th>dimension</th><td>{fmt(summary.get("exponent_dim"), 0)}</td></tr>
                  <tr><th>Sage normal fan exact</th><td>{html.escape(str(exactness.get("sage_normal_fan_exact", "unknown")))}</td></tr>
                  <tr><th>Macaulay2 toric ideal exact</th><td>{html.escape(str(exactness.get("macaulay2_toric_ideal_exact", "unknown")))}</td></tr>
                  <tr><th>minimal generators</th><td><code>{html.escape(json.dumps(minimal_generators))}</code></td></tr>
                  <tr><th>quotient-basis lattice points</th><td>{fmt(len(quotient_basis), 0)}</td></tr>
                  <tr><th>adjacent LCM corners</th><td>{fmt(len(adjacent), 0)}</td></tr>
                </tbody>
              </table>
              <p><a class="pill" href="records/{html.escape(summary_path.name)}">summary JSON</a></p>
            </section>
            """
        )
    for html_record in records_dir.glob("*.html") if records_dir.exists() else []:
        html_record.unlink()
    if interactive_records:
        (dst / "toric_embedding_interactive.json").write_text(
            json.dumps(
                {
                    "schema": "toricgt.toric_embedding_interactive_compact.v1",
                    "records": interactive_records,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    if not cards:
        cards.append("<section class='card'><h2>No record summaries found</h2><p>The toric embedding manifest was copied, but no small summary JSON was available.</p></section>")
    (dst / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ToricGT Tropical-To-Toric Embedding</title>
  <style>
    :root{{color-scheme:dark;--bg:#030712;--panel:#07111f;--line:rgba(55,232,255,.28);--text:#e8fbff;--muted:#9fb1c9;--cyan:#37e8ff;--gold:#ffd166}}
    *{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top left,#092238 0,#030712 42rem);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,sans-serif}}
    main{{max-width:1180px;margin:0 auto;padding:28px 18px 64px}}a{{color:var(--cyan)}}.hero,.card,.panel{{background:linear-gradient(180deg,rgba(11,23,40,.96),rgba(5,13,25,.98));border:1px solid var(--line);border-radius:10px;box-shadow:0 18px 50px rgba(0,0,0,.28)}}
    .hero,.panel{{padding:18px;margin:14px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}}.card{{padding:16px}}h1{{margin:0 0 8px;font-size:clamp(2rem,5vw,3rem)}}p{{color:var(--muted);line-height:1.55}}
    table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid rgba(159,177,201,.16);padding:8px;text-align:left;vertical-align:top}}th{{color:#bdeeff;width:38%}}td{{color:#fff}}code{{color:#ffd166;white-space:pre-wrap;overflow-wrap:anywhere}}
    img{{display:block;width:100%;border:1px solid rgba(55,232,255,.20);border-radius:8px;background:#020713}}.pill{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:8px 12px;margin:4px 8px 4px 0;background:rgba(55,232,255,.08);font-weight:700}}.controls{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:10px 0 14px}}button,select{{border:1px solid var(--line);background:rgba(55,232,255,.08);color:var(--text);border-radius:999px;padding:8px 12px;font-weight:800}}svg{{width:100%;height:auto;border:1px solid rgba(55,232,255,.18);border-radius:10px;background:#020713}}.detail{{border:1px solid rgba(55,232,255,.18);border-radius:10px;background:#020713;padding:12px;margin-top:10px;color:#dff8ff}}
    .note{{border-left:3px solid var(--gold);padding:10px 12px;background:rgba(255,209,102,.08);border-radius:8px;color:#ffe8a8}}
  </style>
</head>
<body><main>
  <section class="hero">
    <h1>Tropical-To-Toric Embedding Report</h1>
    <p>This low-resource page preserves the exact finite toric-embedding record summaries: exponent counts, Sage normal-fan status, Macaulay2 toric-ideal status, Miller-Sturmfels staircase generators, quotient-basis counts, and adjacent LCM data. The original large Plotly record is omitted from GitHub Pages so older browsers do not parse tens of megabytes of JavaScript.</p>
    <p><a class="pill" href="manifest.json">manifest JSON</a></p>
    <p class="note">The visual screenshot below is generated from the original audit. The small summary JSON files keep the reproducible algebraic evidence without forcing a heavy interactive plot.</p>
  </section>
  <section class="panel">
    <h2>Rendered Audit Screenshot</h2>
    <img loading="lazy" decoding="async" src="../../page_assets/toric_embedding.png" alt="Toric embedding audit screenshot">
  </section>
  <section class="panel">
    <h2>Interactive Miller-Sturmfels Staircase</h2>
    <p>This compact viewer restores interaction from the large toric report without loading Plotly. The overhead mode shows the quotient-basis lattice points under the staircase; the projected 3D mode lifts adjacent module layers by small heights so the toric module geometry is visible without losing the familiar xy-grid view.</p>
    <div class="controls"><select id="recordSelect"></select><button id="view2d" type="button">Overhead xy Grid</button><button id="view3d" type="button">Projected 3D Layers</button><button id="idealToggle" type="button">Toggle Ideal Region</button></div>
    <svg id="staircaseSvg" viewBox="0 0 980 560" role="img" aria-label="Interactive Miller-Sturmfels monomial staircase"></svg>
    <div id="staircaseDetail" class="detail">Click a lattice point, generator, or LCM corner for exact monomial data.</div>
  </section>
  <section class="grid">{''.join(cards)}</section>
  <script>
const staircaseSvg = document.getElementById('staircaseSvg');
const recordSelect = document.getElementById('recordSelect');
const detail = document.getElementById('staircaseDetail');
let toricData = null, viewMode = '2d', showIdeal = true;
function esc(value) {{ return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch])); }}
function point2d(x,y,scale,margin,height) {{ return [margin + x*scale, height - margin - y*scale]; }}
function point3d(x,y,z,scale,margin,height) {{ return [margin + x*scale + z*18, height - margin - y*scale - z*16]; }}
function setDetail(html) {{ detail.innerHTML = html; }}
function drawStaircase() {{
  if (!toricData) return;
  const rec = toricData.records[Number(recordSelect.value || 0)];
  const basis = rec.quotient_basis || [];
  const gens = rec.minimal_generators || [];
  const adjacent = rec.adjacent_lcm_layer || [];
  const win = rec.window || {{}};
  const xMax = Math.max(win.x_max || 0, ...basis.map(p=>p[0]), ...gens.map(p=>p[0]), ...adjacent.map(a=>a.lcm_corner?.[0] || 0), 8) + 1;
  const yMax = Math.max(win.y_max || 0, ...basis.map(p=>p[1]), ...gens.map(p=>p[1]), ...adjacent.map(a=>a.lcm_corner?.[1] || 0), 8) + 1;
  const width=980, height=560, margin=66, scale=Math.min((width-2*margin)/xMax, (height-2*margin)/yMax);
  const project = viewMode === '3d' ? (x,y,z=0)=>point3d(x,y,z,scale,margin,height) : (x,y,z=0)=>point2d(x,y,scale,margin,height);
  let html = `<rect width="${{width}}" height="${{height}}" rx="14" fill="#020713"/>`;
  html += `<g stroke="rgba(157,184,207,.18)" stroke-width="1">`;
  for (let x=0; x<=xMax; x++) {{ const a=project(x,0,0), b=project(x,yMax,0); html += `<line x1="${{a[0]}}" y1="${{a[1]}}" x2="${{b[0]}}" y2="${{b[1]}}"/>`; }}
  for (let y=0; y<=yMax; y++) {{ const a=project(0,y,0), b=project(xMax,y,0); html += `<line x1="${{a[0]}}" y1="${{a[1]}}" x2="${{b[0]}}" y2="${{b[1]}}"/>`; }}
  html += `</g>`;
  if (showIdeal) {{
    for (const [gx,gy] of gens) {{
      const a=project(gx,gy,viewMode==='3d'?0.18:0), b=project(xMax,gy,viewMode==='3d'?0.18:0), c=project(xMax,yMax,viewMode==='3d'?0.18:0), d=project(gx,yMax,viewMode==='3d'?0.18:0);
      html += `<polygon points="${{a.join(',')}} ${{b.join(',')}} ${{c.join(',')}} ${{d.join(',')}}" fill="rgba(255,209,102,.11)" stroke="rgba(255,209,102,.28)"/>`;
    }}
  }}
  for (const [x,y] of basis) {{
    const z = viewMode === '3d' ? ((x + y) % 3) * 0.06 : 0;
    const p=project(x,y,z);
    html += `<circle class="basis-point" data-x="${{x}}" data-y="${{y}}" cx="${{p[0]}}" cy="${{p[1]}}" r="${{viewMode==='3d'?4.2:3.7}}" fill="#ecfbff" opacity=".88"><title>x^${{x}} y^${{y}} basis monomial</title></circle>`;
  }}
  for (const [i,g] of gens.entries()) {{
    const p=project(g[0],g[1],viewMode==='3d'?0.35:0);
    html += `<circle class="gen-point" data-index="${{i}}" cx="${{p[0]}}" cy="${{p[1]}}" r="9" fill="#ffd166" stroke="#ecfbff"><title>minimal generator x^${{g[0]}} y^${{g[1]}}</title></circle>`;
    html += `<text x="${{p[0]+12}}" y="${{p[1]-8}}" fill="#ffd166" font-size="15">m${{i+1}}</text>`;
  }}
  for (const a of adjacent) {{
    const c=a.lcm_corner || [0,0], p=project(c[0],c[1],viewMode==='3d'?0.55:0);
    html += `<rect class="lcm-point" data-index="${{a.index || 0}}" x="${{p[0]-8}}" y="${{p[1]-8}}" width="16" height="16" rx="3" fill="#ff5fa2" stroke="#ecfbff"><title>adjacent LCM corner x^${{c[0]}} y^${{c[1]}}</title></rect>`;
  }}
  html += `<text x="28" y="34" fill="#ecfbff" font-size="18">${{esc(rec.record_id)}} · ${{viewMode === '3d' ? 'projected 3D module layers' : 'overhead xy-grid staircase'}}</text>`;
  html += `<text x="28" y="528" fill="#9db8cf" font-size="14">white = monomial basis of S/I · gold = minimal generators · pink = adjacent LCM corners · shaded = ideal region</text>`;
  staircaseSvg.innerHTML = html;
  staircaseSvg.querySelectorAll('.basis-point').forEach(el => el.addEventListener('click', () => setDetail(`<strong>basis monomial</strong><br>x^${{el.dataset.x}} y^${{el.dataset.y}} lies outside the monomial ideal and represents a basis element of S/I.`)));
  staircaseSvg.querySelectorAll('.gen-point').forEach(el => {{
    el.addEventListener('click', () => {{
      const g = gens[Number(el.dataset.index)];
      setDetail(`<strong>minimal generator m${{Number(el.dataset.index)+1}}</strong><br>x^${{g[0]}} y^${{g[1]}} generates a shifted positive orthant inside the monomial ideal.`);
    }});
  }});
  staircaseSvg.querySelectorAll('.lcm-point').forEach(el => {{
    el.addEventListener('click', () => {{
      const a = adjacent[Number(el.dataset.index)] || adjacent[0];
      const c = a.lcm_corner || [0,0];
      setDetail(`<strong>adjacent LCM corner</strong><br>x^${{c[0]}} y^${{c[1]}} is the least common multiple corner for adjacent generators in the two-step Miller-Sturmfels inclusion-exclusion view.`);
    }});
  }});
}}
fetch('toric_embedding_interactive.json').then(r => r.json()).then(data => {{
  toricData = data;
  recordSelect.innerHTML = data.records.map((r,i)=>`<option value="${{i}}">${{esc(r.record_id)}}</option>`).join('');
  recordSelect.addEventListener('change', drawStaircase);
  document.getElementById('view2d').addEventListener('click', () => {{ viewMode='2d'; drawStaircase(); }});
  document.getElementById('view3d').addEventListener('click', () => {{ viewMode='3d'; drawStaircase(); }});
  document.getElementById('idealToggle').addEventListener('click', () => {{ showIdeal=!showIdeal; drawStaircase(); }});
  drawStaircase();
}}).catch(error => {{ detail.textContent = 'Compact toric interactive payload unavailable: '+error; }});
  </script>
</main></body></html>
""",
        encoding="utf-8",
    )


def scrub_public_report_paths(root: Path, repo: Path) -> None:
    text_suffixes = {".html", ".json", ".md", ".txt", ".m2", ".js", ".css"}
    repo_text = str(repo.resolve())
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        text = re.sub(
            r"\n?<!-- toricgt-html-screenshot-link:start -->.*?<!-- toricgt-html-screenshot-link:end -->\n?",
            "\n",
            text,
            flags=re.DOTALL,
        )
        text = text.replace(repo_text + "/", "ToricGT/")
        text = text.replace(repo_text, "ToricGT")
        text = re.sub(r"/tmp/toricgt_[^\"'<>\s]+", "local-screenshot-artifact", text)
        text = re.sub(r"/home/iska/miniconda3/envs/[^\"'<>\s]+", "local-cas-executable", text)
        text = re.sub(r"/home/iska/Documents/amelie/bio/[^\"'<>\s]+", "local-experiment-artifact", text)
        path.write_text(text, encoding="utf-8")


def tetra_metric(metrics: dict[str, Any], keys: list[str], *, invert: bool = False) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            value = float(value)
            return -value if invert else value
    return None


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


CAMPAIGN_METRIC_AXES: dict[str, dict[str, Any]] = {
    "low_bpb": {
        "label": "low BPB",
        "keys": ["__bpb__"],
        "direction": "low",
        "description": "Lower validation or final int8 BPB is the primary competition objective.",
    },
    "compact": {
        "label": "compact artifact",
        "keys": ["artifact_bytes"],
        "direction": "low",
        "description": "Smaller serialized artifact leaves more room under the 16MB competition budget.",
    },
    "train_bpb": {
        "label": "low train BPB",
        "keys": ["train_bpb"],
        "direction": "low",
        "description": "Training BPB is used when validation BPB is unavailable early in a run.",
    },
    "tokengt_graph": {
        "label": "TokenGT graph fit",
        "keys": ["tokengt_graph_loss"],
        "direction": "low",
        "description": "First-class graph tokenization pressure for node, edge, endpoint, distance, and ConvexTok-DAG structure.",
    },
    "graph_lm": {
        "label": "graph LM fit",
        "keys": ["graph_lm_loss"],
        "direction": "low",
        "description": "Graph-in/graph-out language-model objective before OAI FineWeb flattening for BPB scoring.",
    },
    "mtp": {
        "label": "MTP fit",
        "keys": ["oai_mtp_loss"],
        "direction": "low",
        "description": "Multi-token prediction pressure for early likelihood improvement.",
    },
    "fot": {
        "label": "FoT fit",
        "keys": ["oai_fot_loss"],
        "direction": "low",
        "description": "Embedding-space Forest-of-Thought branch objective.",
    },
    "fot_reward": {
        "label": "FoT reward",
        "keys": ["oai_fot_reward"],
        "direction": "high",
        "description": "Higher BPB-delta and diversity reward from Forest-of-Thought search.",
    },
    "gflownet": {
        "label": "GFlowNet fit",
        "keys": ["oai_gflownet_loss"],
        "direction": "low",
        "description": "Embedding-space GFlowNet trajectory-balance pressure.",
    },
    "gflownet_reward": {
        "label": "GFlowNet reward",
        "keys": ["oai_gflownet_reward"],
        "direction": "high",
        "description": "Higher reward for sampled graph-of-thought continuations.",
    },
    "graphcg": {
        "label": "GraphCG disentanglement",
        "keys": ["graphcg_loss"],
        "direction": "low",
        "description": "Full-rank concept-axis disentanglement pressure in embedding space.",
    },
    "analogy": {
        "label": "analogy retrieval fit",
        "keys": ["analogy_loss"],
        "direction": "low",
        "description": "Simplex-map and vectorized-persistent-homology analogy objective.",
    },
    "memory": {
        "label": "memory retrieval fit",
        "keys": ["memory_loss"],
        "direction": "low",
        "description": "Trajectory-memory retrieval pressure.",
    },
    "toric_geometry": {
        "label": "toric geometry fit",
        "keys": ["toric_geometry_loss"],
        "direction": "low",
        "description": "Tropical active-face embedding into toric charts and fan diagnostics.",
    },
    "toric_vector_bundle": {
        "label": "1D-cone sheaf fit",
        "keys": ["toric_vector_bundle_1d_cone_ce_loss"],
        "direction": "low",
        "description": "Toric vector-bundle/sheaf compatibility over one-dimensional cones.",
    },
    "toric_bgg": {
        "label": "Toric BGG fit",
        "keys": ["toric_bgg_loss"],
        "direction": "low",
        "description": "Finite category-O standard-filtration, differential, and Gale-dual supervision.",
    },
    "koszul": {
        "label": "Koszul persistence fit",
        "keys": ["koszul_persistence_loss"],
        "direction": "low",
        "description": "Koszul and filtered-persistence consistency pressure.",
    },
    "toric_cca": {
        "label": "toric CCA topology fit",
        "keys": ["toric_cca_topology_loss"],
        "direction": "low",
        "description": "Combinatorial commutative algebra, staircase, and topology diagnostics.",
    },
    "derived_signature": {
        "label": "derived signature fit",
        "keys": ["derived_signature_loss"],
        "direction": "low",
        "description": "Low-rank distilled signatures for resolutions, chain maps, and derived-category evidence.",
    },
}


CAMPAIGN_TETRAHEDRON_SPECS: list[dict[str, Any]] = [
    {
        "id": "bpb_compact_graph_toric",
        "title": "BPB, Compactness, Graph Fit, Toric Geometry",
        "axes": ["low_bpb", "compact", "tokengt_graph", "toric_geometry"],
    },
    {
        "id": "bpb_tokengt_fot_gflownet",
        "title": "BPB, TokenGT, FoT, GFlowNet",
        "axes": ["low_bpb", "tokengt_graph", "fot", "gflownet"],
    },
    {
        "id": "bpb_graphlm_mtp_compact",
        "title": "BPB, Graph-LM, MTP, Compactness",
        "axes": ["low_bpb", "graph_lm", "mtp", "compact"],
    },
    {
        "id": "bpb_graphcg_memory_analogy",
        "title": "BPB, GraphCG, Memory, Analogy",
        "axes": ["low_bpb", "graphcg", "memory", "analogy"],
    },
    {
        "id": "bpb_bgg_koszul_derived",
        "title": "BPB, Toric BGG, Koszul, Derived Signature",
        "axes": ["low_bpb", "toric_bgg", "koszul", "derived_signature"],
    },
    {
        "id": "bpb_toric_bundle_cca",
        "title": "BPB, Toric Geometry, 1D-Cone Sheaf, CCA",
        "axes": ["low_bpb", "toric_geometry", "toric_vector_bundle", "toric_cca"],
    },
    {
        "id": "bpb_rewards_fot_gfn",
        "title": "BPB, FoT Reward, GFlowNet Reward, Train BPB",
        "axes": ["low_bpb", "fot_reward", "gflownet_reward", "train_bpb"],
    },
]


CAMPAIGN_TRIANGLE_SPECS: list[dict[str, Any]] = [
    {
        "id": "bpb_tokengt_graphlm",
        "title": "Low BPB / TokenGT / Graph-LM",
        "axes": ["low_bpb", "tokengt_graph", "graph_lm"],
        "shade": "balanced tri-metric support",
    },
    {
        "id": "bpb_fot_gfn_rewards",
        "title": "Low BPB / FoT Reward / GFlowNet Reward",
        "axes": ["low_bpb", "fot_reward", "gflownet_reward"],
        "shade": "joint reward-support region",
    },
    {
        "id": "bpb_fot_gfn_losses",
        "title": "Low BPB / FoT Fit / GFlowNet Fit",
        "axes": ["low_bpb", "fot", "gflownet"],
        "shade": "balanced search-fit region",
    },
    {
        "id": "bpb_graphcg_analogy",
        "title": "Low BPB / GraphCG / Analogy",
        "axes": ["low_bpb", "graphcg", "analogy"],
        "shade": "concept-analogy balance",
    },
    {
        "id": "bpb_bgg_koszul",
        "title": "Low BPB / Toric BGG / Koszul",
        "axes": ["low_bpb", "toric_bgg", "koszul"],
        "shade": "homological support balance",
    },
    {
        "id": "bpb_toric_bundle_cca",
        "title": "Low BPB / 1D-Cone Sheaf / CCA",
        "axes": ["low_bpb", "toric_vector_bundle", "toric_cca"],
        "shade": "toric sheaf-topology balance",
    },
    {
        "id": "bpb_compact_train",
        "title": "Low BPB / Compactness / Train BPB",
        "axes": ["low_bpb", "compact", "train_bpb"],
        "shade": "score-size-stability balance",
    },
]


def campaign_axis_raw_value(metrics: dict[str, Any], axis: dict[str, Any]) -> float | None:
    for key in axis.get("keys", []):
        if key == "__bpb__":
            value = metric_bpb(metrics)
        else:
            value = metrics.get(str(key))
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def campaign_axis_score(metrics: dict[str, Any], axis: dict[str, Any]) -> float | None:
    value = campaign_axis_raw_value(metrics, axis)
    if value is None:
        return None
    return -value if axis.get("direction") == "low" else value


def campaign_record_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        bpb = metric_bpb(metrics)
        if not math.isfinite(bpb):
            continue
        raw_values: dict[str, float | None] = {}
        scores: dict[str, float | None] = {}
        for axis_id, axis in CAMPAIGN_METRIC_AXES.items():
            raw_values[axis_id] = campaign_axis_raw_value(metrics, axis)
            scores[axis_id] = campaign_axis_score(metrics, axis)
        records.append(
            {
                "index": index,
                "run": str(row.get("run_id", f"run-{index}"))[-64:],
                "profile": str(row.get("profile", f"run-{index}")),
                "bpb": bpb,
                "nll_loss": next(
                    (
                        float(metrics[key])
                        for key in ("val_loss", "final_int8_loss", "train_loss")
                        if isinstance(metrics.get(key), (int, float)) and math.isfinite(float(metrics[key]))
                    ),
                    None,
                ),
                "artifact": metrics.get("artifact_bytes"),
                "train_bpb": metrics.get("train_bpb"),
                "checkpoint_step": metrics.get("checkpoint_step", metrics.get("train_step")),
                "raw_values": raw_values,
                "scores": scores,
            }
        )
    return records


def normalize_axis_scores(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    normalized: dict[str, list[float]] = {}
    for axis_id in CAMPAIGN_METRIC_AXES:
        values = [record["scores"].get(axis_id) for record in records]
        finite = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
        scaled = normalize(finite)
        cursor = 0
        out: list[float] = []
        for value in values:
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                out.append(float(scaled[cursor]))
                cursor += 1
            else:
                out.append(0.0)
        normalized[axis_id] = out
    return normalized


def tetrahedron_view_payload(records: list[dict[str, Any]], normalized: dict[str, list[float]], spec: dict[str, Any]) -> dict[str, Any]:
    vertices = [
        [0.0, 0.0, 1.35],
        [-1.18, -0.82, -0.42],
        [1.18, -0.82, -0.42],
        [0.0, 1.28, -0.42],
    ]
    axes = [str(axis) for axis in spec["axes"]]
    labels = [str(CAMPAIGN_METRIC_AXES[axis]["label"]) for axis in axes]
    points: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        weights = [max(0.0, float(normalized.get(axis, [0.0] * len(records))[idx])) for axis in axes]
        total = sum(weights)
        bary = [0.25, 0.25, 0.25, 0.25] if total <= 0 else [value / total for value in weights]
        xyz = [sum(bary[i] * vertices[i][dim] for i in range(4)) for dim in range(3)]
        points.append({"record_index": record["index"], "weights": bary, "xyz": xyz})
    return {
        "id": str(spec["id"]),
        "title": str(spec["title"]),
        "axes": axes,
        "labels": labels,
        "vertices": vertices,
        "points": points,
    }


def triangle_view_payload(records: list[dict[str, Any]], normalized: dict[str, list[float]], spec: dict[str, Any]) -> dict[str, Any]:
    vertices = [[0.50, 0.92], [0.08, 0.12], [0.92, 0.12]]
    axes = [str(axis) for axis in spec["axes"]]
    labels = [str(CAMPAIGN_METRIC_AXES[axis]["label"]) for axis in axes]
    points: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        weights = [max(0.0, float(normalized.get(axis, [0.0] * len(records))[idx])) for axis in axes]
        total = sum(weights)
        bary = [1 / 3, 1 / 3, 1 / 3] if total <= 0 else [value / total for value in weights]
        xy = [sum(bary[i] * vertices[i][dim] for i in range(3)) for dim in range(2)]
        balance = 1.0 - min(1.0, (sum((value - 1 / 3) ** 2 for value in bary) ** 0.5) / ((2 / 3) ** 0.5))
        points.append({"record_index": record["index"], "weights": bary, "xy": xy, "balance": balance})
    grid: list[dict[str, Any]] = []
    resolution = 20
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            a = i / resolution
            b = j / resolution
            c = 1.0 - a - b
            bary = [a, b, c]
            xy = [sum(bary[k] * vertices[k][dim] for k in range(3)) for dim in range(2)]
            balance = 1.0 - min(1.0, (sum((value - 1 / 3) ** 2 for value in bary) ** 0.5) / ((2 / 3) ** 0.5))
            synergy = (max(a, 0.0) * max(b, 0.0) * max(c, 0.0)) ** (1 / 3) * 3.0
            grid.append({"x": xy[0], "y": xy[1], "shade": max(0.0, min(1.0, 0.55 * balance + 0.45 * synergy))})
    return {
        "id": str(spec["id"]),
        "title": str(spec["title"]),
        "axes": axes,
        "labels": labels,
        "vertices": vertices,
        "shade": str(spec.get("shade", "balanced metric support")),
        "grid": grid,
        "grid_resolution": resolution,
        "points": points,
    }


def write_campaign_tetrahedron_report(docs: Path, rows: list[dict[str, Any]]) -> dict[str, str] | None:
    records = campaign_record_payload(rows)
    if len(records) < 2:
        return None
    normalized = normalize_axis_scores(records)
    tetrahedra = [tetrahedron_view_payload(records, normalized, spec) for spec in CAMPAIGN_TETRAHEDRON_SPECS]
    triangles = [triangle_view_payload(records, normalized, spec) for spec in CAMPAIGN_TRIANGLE_SPECS]
    out_dir = docs / "page_interactive" / "campaign_tetrahedron"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "Campaign Metric Tetrahedra and Shaded Triangles",
        "axis_library": CAMPAIGN_METRIC_AXES,
        "records": records,
        "tetrahedra": tetrahedra,
        "triangles": triangles,
    }
    (out_dir / "campaign_tetrahedron.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    axis_ids = list(CAMPAIGN_METRIC_AXES)
    raw_header = "".join(f"<th>{html.escape(CAMPAIGN_METRIC_AXES[axis]['label'])}</th>" for axis in axis_ids)
    raw_rows = []
    for record in records:
        raw_rows.append(
            "<tr>"
            + f"<th>{record['index']}. {html.escape(record['profile'])}</th>"
            + "".join(f"<td>{fmt(record['raw_values'].get(axis))}</td>" for axis in axis_ids)
            + "</tr>"
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Campaign Metric Tetrahedra</title>
<style>
:root{{color-scheme:dark;--bg:#030712;--panel:#071421;--line:rgba(70,231,255,.28);--text:#ecfbff;--muted:#9db8cf;--cyan:#46e7ff;--gold:#ffd166}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 10% 10%,rgba(70,231,255,.16),transparent 30rem),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1280px;margin:0 auto;padding:24px}}.panel{{border:1px solid var(--line);border-radius:10px;background:rgba(7,20,33,.86);padding:16px;margin:14px 0}}
h1{{margin:0 0 8px}}p{{color:var(--muted);line-height:1.5}}.plot{{min-height:520px;border:1px solid rgba(70,231,255,.16);border-radius:10px;background:#020713;margin-top:12px;overflow:hidden}}
.plot.triangle{{min-height:420px}}.view-controls{{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:end;margin:8px 0 12px}}select{{width:100%;background:#020713;color:var(--text);border:1px solid rgba(70,231,255,.28);border-radius:8px;padding:10px}}label{{display:block;color:var(--muted);font-size:.9rem;margin-bottom:4px}}.metric-note{{font-size:.92rem;color:var(--muted);margin-top:8px}}code{{color:#7df5ff}}
input[type=range]{{accent-color:var(--cyan)}}.canvas-controls{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:8px 0 10px}}.canvas-controls label{{display:grid;gap:4px}}.canvas-controls input{{width:min(220px,60vw)}}canvas.tetra-canvas{{display:block;width:100%;height:auto;background:#020713}}
svg{{display:block;width:100%;height:auto}}.svg-label{{font-size:13px;fill:#ecfbff;paint-order:stroke;stroke:#020713;stroke-width:3px}}.svg-small{{font-size:11px;fill:#9db8cf}}.point-label{{font-size:11px;fill:#ecfbff;paint-order:stroke;stroke:#020713;stroke-width:3px}}.tetra-edge{{stroke:#46e7ff;stroke-width:2.5}}.tetra-face{{fill:rgba(70,231,255,.10);stroke:#46e7ff;stroke-width:1.5}}.triangle-cell{{stroke:none;opacity:.72}}.triangle-contour{{stroke:rgba(236,251,255,.18);stroke-width:.9;fill:none}}.run-point{{stroke:#fff;stroke-width:1.2;cursor:pointer}}.run-point:hover{{stroke:#ffd166;stroke-width:3px}}.details{{border:1px solid rgba(70,231,255,.16);border-radius:8px;background:#020713;padding:10px;margin-top:10px;color:#dff8ff;line-height:1.45}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid rgba(157,184,207,.16);padding:8px;text-align:left;vertical-align:top}}td.num{{text-align:right;color:var(--gold)}}.scroll{{overflow:auto;max-height:520px;border:1px solid rgba(70,231,255,.14);border-radius:8px}}.scroll table{{min-width:1100px}}.button{{border:1px solid var(--line);border-radius:999px;padding:8px 12px;background:rgba(70,231,255,.08);color:var(--text);font-weight:700;display:inline-block}}
</style></head><body><main>
<section class="panel"><h1>Campaign Metric Tetrahedra and Shaded Triangles</h1>
<p>This page uses compact native Canvas/SVG renderers instead of Plotly/WebGL. Each campaign metric tetrahedron is interactive in 3D with yaw, pitch, zoom, hover, and click details while remaining light enough for low-resource browsers.</p>
<p><a href="campaign_tetrahedron.json">payload JSON</a></p></section>
<section class="panel"><h2>Interactive 3D Tetrahedron Views</h2><div class="view-controls"><div><label for="tetra_select">metric tetrahedron</label><select id="tetra_select"><option>loading...</option></select></div><span id="tetra_count" class="metric-note"></span></div><div class="canvas-controls"><label>yaw <input id="tetra_yaw" type="range" min="-180" max="180" value="28"></label><label>pitch <input id="tetra_pitch" type="range" min="-70" max="70" value="-18"></label><label>zoom <input id="tetra_zoom" type="range" min="70" max="180" value="100"></label></div><div id="tetra_plot" class="plot"></div><p id="tetra_caption" class="metric-note"></p></section>
<section class="panel"><h2>Shaded Triangle Views</h2><p>Blue shading indicates the balanced-support region for the metric triple. Markers remain actual run observations; hover text reports profile, BPB, raw axis values, and barycentric weights.</p><div class="view-controls"><div><label for="triangle_select">metric triangle</label><select id="triangle_select"><option>loading...</option></select></div><span id="triangle_count" class="metric-note"></span></div><div id="triangle_plot" class="plot triangle"></div><p id="triangle_caption" class="metric-note"></p></section>
<section class="panel"><h2>Run Table</h2><table><thead><tr><th>profile</th><th>run</th><th>BPB</th><th>NLL/loss</th><th>artifact bytes</th><th>train BPB</th></tr></thead><tbody>
{''.join(f"<tr><td>{html.escape(p['profile'])}</td><td>{html.escape(p['run'])}</td><td class='num'>{p['bpb']:.6f}</td><td class='num'>{fmt(p.get('nll_loss'))}</td><td class='num'>{fmt(p.get('artifact'),0)}</td><td class='num'>{fmt(p.get('train_bpb'))}</td></tr>" for p in records)}
</tbody></table></section>
<section class="panel"><h2>Raw Metric Detail</h2><p>All axis values used by the tetrahedra and triangles are retained here so the low-resource view does not discard the underlying evidence.</p><div class="scroll"><table><thead><tr><th>run</th>{raw_header}</tr></thead><tbody>{''.join(raw_rows)}</tbody></table></div></section>
<script>
let payload = null;
let recordByIndex = new Map();
function esc(value) {{
  return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}
function fmtValue(value) {{
  if (typeof value === 'number' && Number.isFinite(value)) return Math.abs(value) >= 10000 ? Math.round(value).toLocaleString() : value.toFixed(4);
  return value ?? 'missing';
}}
function recordFor(point) {{
  return recordByIndex.get(point.record_index) || {{}};
}}
function axisRawText(point, axes) {{
  const rec = recordFor(point);
  return axes.map(axis => payload.axis_library[axis].label+': '+fmtValue(rec.raw_values?.[axis])).join('<br>');
}}
function pointTitle(point, axes) {{
  const rec = recordFor(point);
  return rec.profile+'\\nrun '+rec.run+'\\nBPB '+fmtValue(rec.bpb)+'\\nNLL/loss '+fmtValue(rec.nll_loss)+'\\nweights '+point.weights.map(w=>w.toFixed(3)).join(' / ')+'\\n'+axes.map(axis => payload.axis_library[axis].label+': '+fmtValue(rec.raw_values?.[axis])).join('\\n');
}}
function colorForBpb(value) {{
  const vals = payload.records.map(r => r.bpb).filter(v => typeof v === 'number');
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const t = hi <= lo ? 0.5 : (value - lo) / (hi - lo);
  const r = Math.round(255 * (0.18 + 0.72 * t));
  const g = Math.round(230 * (0.78 - 0.52 * t));
  const b = Math.round(115 * (0.28 + 0.38 * (1 - t)));
  return `rgb(${{r}},${{g}},${{b}})`;
}}
let tetraProjected = [];
function rotate3(v, yawDeg, pitchDeg) {{
  const yaw = yawDeg * Math.PI / 180, pitch = pitchDeg * Math.PI / 180;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = v[0] * cy - v[2] * sy;
  const z1 = v[0] * sy + v[2] * cy;
  const y1 = v[1] * cp - z1 * sp;
  const z2 = v[1] * sp + z1 * cp;
  return [x1, y1, z2];
}}
function project3Collection(points, width, height, yaw, pitch, zoom) {{
  const rotated = points.map(p => rotate3(p.xyz, yaw, pitch));
  const minX = Math.min(...rotated.map(p=>p[0])), maxX = Math.max(...rotated.map(p=>p[0]));
  const minY = Math.min(...rotated.map(p=>p[1])), maxY = Math.max(...rotated.map(p=>p[1]));
  const scale = Math.min((width - 210) / Math.max(1e-6, maxX - minX), (height - 190) / Math.max(1e-6, maxY - minY)) * zoom / 100;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  return points.map((p, i) => {{
    const r = rotated[i];
    const persp = 680 / (680 + r[2] * 90);
    return {{...p, rx:r[0], ry:r[1], rz:r[2], x: width / 2 + (r[0] - cx) * scale * persp, y: height / 2 - (r[1] - cy) * scale * persp}};
  }});
}}
function showDetails(id, point, axes) {{
  const rec = recordFor(point);
  const raw = axes.map(axis => `<tr><th>${{esc(payload.axis_library[axis].label)}}</th><td>${{esc(fmtValue(rec.raw_values?.[axis]))}}</td></tr>`).join('');
  document.getElementById(id).innerHTML = `<strong>${{esc(rec.profile)}}</strong><br>run ${{esc(rec.run)}} · BPB ${{esc(fmtValue(rec.bpb))}} · NLL/loss ${{esc(fmtValue(rec.nll_loss))}} · weights ${{esc(point.weights.map(w=>w.toFixed(3)).join(' / '))}}<table>${{raw}}</table>`;
}}
function populateSelect(select, views) {{
  select.innerHTML = '';
  views.forEach((view, idx) => {{
    const option = document.createElement('option');
    option.value = String(idx);
    option.textContent = view.title;
    select.appendChild(option);
  }});
}}
function renderTetra(view) {{
  document.getElementById('tetra_plot').innerHTML = `<canvas id="tetra_canvas" class="tetra-canvas" width="920" height="560" role="img" aria-label="${{esc(view.title)}}"></canvas><div id="tetra_details" class="details">Hover or click a run marker to inspect BPB, NLL/loss, artifact bytes, and raw axis values.</div>`;
  const tetraCanvas = document.getElementById('tetra_canvas');
  const tctx = tetraCanvas.getContext('2d');
  const yawInput = document.getElementById('tetra_yaw');
  const pitchInput = document.getElementById('tetra_pitch');
  const zoomInput = document.getElementById('tetra_zoom');
  const vertices = view.vertices.map((xyz, i) => ({{kind:'vertex', xyz, label:view.labels[i], vertex_index:i}}));
  const runPoints = view.points.map(point => ({{kind:'run', xyz:point.xyz, point}}));
  const edges = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]];
  const faces = [[0,1,2],[0,1,3],[0,2,3],[1,2,3]];
  function drawTetraCanvas() {{
    const allProjected = project3Collection([...vertices, ...runPoints], tetraCanvas.width, tetraCanvas.height, Number(yawInput.value), Number(pitchInput.value), Number(zoomInput.value));
    const projectedVertices = allProjected.slice(0, vertices.length);
    tetraProjected = allProjected.slice(vertices.length);
    tctx.clearRect(0,0,tetraCanvas.width,tetraCanvas.height);
    tctx.fillStyle = '#020713'; tctx.fillRect(0,0,tetraCanvas.width,tetraCanvas.height);
    tctx.strokeStyle = 'rgba(157,184,207,.18)'; tctx.lineWidth = 1;
    for (let i=0;i<6;i++) {{ const y=70+i*(tetraCanvas.height-140)/5; tctx.beginPath(); tctx.moveTo(60,y); tctx.lineTo(tetraCanvas.width-60,y); tctx.stroke(); }}
    const sortedFaces = faces.map(face => ({{face, depth: face.reduce((s,i)=>s+projectedVertices[i].rz,0)/3}})).sort((a,b)=>a.depth-b.depth);
    for (const item of sortedFaces) {{
      const pts = item.face.map(i => projectedVertices[i]);
      tctx.fillStyle = 'rgba(70,231,255,.075)'; tctx.strokeStyle = 'rgba(70,231,255,.32)'; tctx.lineWidth = 1.2;
      tctx.beginPath(); tctx.moveTo(pts[0].x, pts[0].y); tctx.lineTo(pts[1].x, pts[1].y); tctx.lineTo(pts[2].x, pts[2].y); tctx.closePath(); tctx.fill(); tctx.stroke();
    }}
    for (const [i,j] of edges) {{
      const a=projectedVertices[i], b=projectedVertices[j];
      tctx.strokeStyle = 'rgba(70,231,255,.78)'; tctx.lineWidth = 2.3;
      tctx.beginPath(); tctx.moveTo(a.x,a.y); tctx.lineTo(b.x,b.y); tctx.stroke();
    }}
    tctx.font = '13px system-ui'; tctx.textAlign = 'center';
    for (const v of projectedVertices) {{
      tctx.fillStyle = '#ffd166'; tctx.beginPath(); tctx.arc(v.x,v.y,5,0,Math.PI*2); tctx.fill();
      tctx.fillStyle = '#ecfbff'; tctx.fillText(v.label, v.x, v.y - 12);
    }}
    for (const p of tetraProjected.sort((a,b)=>a.rz-b.rz)) {{
      const rec = recordFor(p.point);
      tctx.fillStyle = colorForBpb(rec.bpb); tctx.strokeStyle = '#fff'; tctx.lineWidth = 1.2;
      tctx.beginPath(); tctx.arc(p.x, p.y, 8, 0, Math.PI*2); tctx.fill(); tctx.stroke();
      tctx.fillStyle = '#ecfbff'; tctx.font = '11px system-ui'; tctx.textAlign = 'left'; tctx.fillText(String(p.point.record_index), p.x + 10, p.y + 4);
    }}
    tctx.fillStyle='#ecfbff'; tctx.font='15px system-ui'; tctx.textAlign='left';
    tctx.fillText(`drag to rotate · yaw ${{yawInput.value}} · pitch ${{pitchInput.value}} · zoom ${{zoomInput.value}} · lower BPB is greener`, 22, 34);
  }}
  function hitTetra(event) {{
    const rect = tetraCanvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * tetraCanvas.width / rect.width;
    const y = (event.clientY - rect.top) * tetraCanvas.height / rect.height;
    let best = null, bd = Infinity;
    for (const p of tetraProjected) {{
      const d = (p.x-x)**2 + (p.y-y)**2;
      if (d < bd) {{ bd = d; best = p; }}
    }}
    if (best && bd < 360) showDetails('tetra_details', best.point, view.axes);
  }}
  let down=false, lastX=0, lastY=0;
  tetraCanvas.addEventListener('pointerdown', event => {{ down=true; lastX=event.clientX; lastY=event.clientY; tetraCanvas.setPointerCapture(event.pointerId); }});
  tetraCanvas.addEventListener('pointermove', event => {{
    if (!down) {{ hitTetra(event); return; }}
    const dx=event.clientX-lastX, dy=event.clientY-lastY; lastX=event.clientX; lastY=event.clientY;
    yawInput.value = Math.max(Number(yawInput.min), Math.min(Number(yawInput.max), Number(yawInput.value) + dx * 0.45));
    pitchInput.value = Math.max(Number(pitchInput.min), Math.min(Number(pitchInput.max), Number(pitchInput.value) - dy * 0.35));
    drawTetraCanvas();
  }});
  tetraCanvas.addEventListener('pointerup', () => {{ down=false; }});
  tetraCanvas.addEventListener('pointercancel', () => {{ down=false; }});
  tetraCanvas.addEventListener('click', hitTetra);
  [yawInput, pitchInput, zoomInput].forEach(el => el.oninput = drawTetraCanvas);
  drawTetraCanvas();
  document.getElementById('tetra_caption').innerHTML = 'Axes: '+view.axes.map(axis => payload.axis_library[axis].label).join(' · ');
}}
function renderTriangle(view) {{
  const boundary = [...view.vertices, view.vertices[0]];
  const map2 = p => [70 + p[0] * 580, 370 - p[1] * 330];
  const res = view.grid_resolution || 20;
  function shadeFromBary(a,b,c) {{
    const balance = 1.0 - Math.min(1.0, Math.sqrt((a-1/3)**2 + (b-1/3)**2 + (c-1/3)**2) / Math.sqrt(2/3));
    const synergy = 3 * Math.cbrt(Math.max(0, a*b*c));
    return Math.max(0, Math.min(1, 0.55 * balance + 0.45 * synergy));
  }}
  function shadeColor(shade) {{
    const r = Math.round(7 + 18 * shade);
    const g = Math.round(45 + 118 * shade);
    const b = Math.round(88 + 154 * shade);
    return `rgb(${{r}},${{g}},${{b}})`;
  }}
  function baryToXY(a,b,c) {{
    return map2([
      a * view.vertices[0][0] + b * view.vertices[1][0] + c * view.vertices[2][0],
      a * view.vertices[0][1] + b * view.vertices[1][1] + c * view.vertices[2][1],
    ]);
  }}
  const cells = [];
  for (let i = 0; i < res; i++) {{
    for (let j = 0; j < res - i; j++) {{
      const a = i / res, b0 = j / res, c = 1 - a - b0;
      const a1 = (i + 1) / res, b1 = j / res, c1 = 1 - a1 - b1;
      const a2 = i / res, b2 = (j + 1) / res, c2 = 1 - a2 - b2;
      const p0 = baryToXY(a,b0,c), p1 = baryToXY(a1,b1,c1), p2 = baryToXY(a2,b2,c2);
      const s = shadeFromBary((a+a1+a2)/3, (b0+b1+b2)/3, (c+c1+c2)/3);
      cells.push(`<polygon class="triangle-cell" points="${{p0.join(',')}} ${{p1.join(',')}} ${{p2.join(',')}}" fill="${{shadeColor(s)}}"><title>balanced support shade ${{s.toFixed(3)}}</title></polygon>`);
      if (i + j < res - 1) {{
        const a3 = (i + 1) / res, b3 = (j + 1) / res, c3 = 1 - a3 - b3;
        const p3 = baryToXY(a3,b3,c3);
        const s2 = shadeFromBary((a1+a2+a3)/3, (b1+b2+b3)/3, (c1+c2+c3)/3);
        cells.push(`<polygon class="triangle-cell" points="${{p1.join(',')}} ${{p3.join(',')}} ${{p2.join(',')}}" fill="${{shadeColor(s2)}}"><title>balanced support shade ${{s2.toFixed(3)}}</title></polygon>`);
      }}
    }}
  }}
  const grid = cells.join('');
  const b = boundary.map(map2);
  const boundarySvg = `<polyline points="${{b.map(p=>p.join(',')).join(' ')}}" fill="none" stroke="#46e7ff" stroke-width="3"/>`;
  const labels = view.vertices.map((v,i) => {{ const [x,y] = map2(v); return `<text class="svg-label" text-anchor="middle" x="${{x}}" y="${{y-12}}">${{esc(view.labels[i])}}</text>`; }}).join('');
  const contours = [0.25, 0.50, 0.75].map(t => {{
    const pA = baryToXY(t,(1-t)/2,(1-t)/2);
    const pB = baryToXY((1-t)/2,t,(1-t)/2);
    const pC = baryToXY((1-t)/2,(1-t)/2,t);
    return `<polyline class="triangle-contour" points="${{pA.join(',')}} ${{pB.join(',')}} ${{pC.join(',')}} ${{pA.join(',')}}" />`;
  }}).join('');
  const points = view.points.map(point => {{
    const rec = recordFor(point);
    const [x,y] = map2(point.xy);
    return `<g><circle class="run-point" cx="${{x}}" cy="${{y}}" r="9" fill="${{colorForBpb(rec.bpb)}}" onclick="showDetails('triangle_details', payload.triangles[document.getElementById('triangle_select').value].points.find(p=>p.record_index===${{point.record_index}}), payload.triangles[document.getElementById('triangle_select').value].axes)"><title>${{esc(pointTitle(point, view.axes))}}</title></circle><text class="point-label" x="${{x+11}}" y="${{y+4}}">${{point.record_index}}</text></g>`;
  }}).join('');
  document.getElementById('triangle_plot').innerHTML = `<svg viewBox="0 0 720 420" role="img" aria-label="${{esc(view.title)}}">${{grid}}${{contours}}${{boundarySvg}}${{labels}}${{points}}</svg><div id="triangle_details" class="details">Hover a run marker for BPB/NLL details; click it to pin exact raw axis values. NLL/loss stays in this panel and in the table so it does not overlap the metric geometry.</div>`;
  document.getElementById('triangle_caption').innerHTML = 'Axes: '+view.axes.map(axis => payload.axis_library[axis].label).join(' · ')+' · shade: '+view.shade;
}}
function initCampaignViews(data) {{
  payload = data;
  recordByIndex = new Map(payload.records.map(record => [record.index, record]));
  const tetraSelect = document.getElementById('tetra_select');
  const triangleSelect = document.getElementById('triangle_select');
  populateSelect(tetraSelect, payload.tetrahedra);
  populateSelect(triangleSelect, payload.triangles);
  document.getElementById('tetra_count').textContent = payload.tetrahedra.length+' views; one rendered at a time';
  document.getElementById('triangle_count').textContent = payload.triangles.length+' views; one rendered at a time';
  tetraSelect.addEventListener('change', () => renderTetra(payload.tetrahedra[Number(tetraSelect.value)]));
  triangleSelect.addEventListener('change', () => renderTriangle(payload.triangles[Number(triangleSelect.value)]));
  renderTetra(payload.tetrahedra[0]);
  renderTriangle(payload.triangles[0]);
}}
fetch('campaign_tetrahedron.json')
  .then(response => response.json())
  .then(initCampaignViews)
  .catch(error => {{
    document.getElementById('tetra_caption').textContent = 'Could not load campaign_tetrahedron.json: '+error;
    document.getElementById('triangle_caption').textContent = 'Could not load campaign_tetrahedron.json: '+error;
  }});
</script></main></body></html>
"""
    page = "\n".join(line.rstrip() for line in page.splitlines()) + "\n"
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    return {
        "href": "page_interactive/campaign_tetrahedron/index.html",
        "title": "Campaign Metric Tetrahedra and Triangles",
        "summary": "Interactive gallery over completed run metrics: multiple tetrahedra plus blue-shaded triangle views for BPB, graph/FoT, toric, topology, BGG, memory, and compactness metrics.",
        "features": "metric tetrahedra · shaded triangles · run labels · BPB color scale · hover raw metrics",
        "embed": "0",
    }


def table_rows(metrics: dict[str, Any], keys: list[str]) -> str:
    rows = []
    for key in keys:
        rows.append(
            f"<tr><th>{html.escape(key)}</th><td>{fmt(metrics.get(key), 6 if 'loss' in key else 4)}</td></tr>"
        )
    return "\n".join(rows)


def markdown_links_to_html(text: str, *, base_url: str = PARAMETER_GOLF_URL) -> str:
    def replace(match: re.Match[str]) -> str:
        label = html.escape(match.group(1))
        href = match.group(2)
        if href.startswith("records/"):
            href = f"{base_url}/tree/main/{href}"
        return f'<a href="{html.escape(href)}">{label}</a>'

    escaped_chunks: list[str] = []
    last = 0
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        escaped_chunks.append(html.escape(text[last : match.start()]))
        escaped_chunks.append(replace(match))
        last = match.end()
    escaped_chunks.append(html.escape(text[last:]))
    return "".join(escaped_chunks)


def parse_parameter_golf_top5(repo: Path) -> list[dict[str, str]]:
    readme = repo / "amelie-iska" / "parameter-golf" / "README.md"
    if not readme.exists():
        return []
    rows: list[dict[str, str]] = []
    in_leaderboard = False
    for line in readme.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip() == "## Leaderboard":
            in_leaderboard = True
            continue
        if in_leaderboard and line.startswith("#### "):
            break
        if not in_leaderboard or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 6 or cells[0] == "Run" or set(cells[0]) <= {"-"}:
            continue
        rows.append(
            {
                "run": cells[0],
                "score": cells[1],
                "author": cells[2],
                "summary": cells[3],
                "date": cells[4],
                "info": cells[5],
            }
        )
        if len(rows) >= 5:
            break
    return rows


def competition_table(repo: Path, best_metrics: dict[str, Any], pr_href: str) -> str:
    official = parse_parameter_golf_top5(repo)
    toricgt_score = fmt(best_metrics.get("final_int8_bpb"))
    toricgt_val = fmt(best_metrics.get("val_bpb"))
    toricgt_summary = (
        "ToricGT ConvexTok-2048 with first-class TokenGT graphification, "
        "OAI-only flattening, FoT/GFlowNet heads, MTP, GraphCG, toric/BGG/"
        "Koszul/topological audits. Pending Parameter Golf PR review and "
        "independent verification; local capped validation BPB "
        f"{toricgt_val}."
    )
    rows = [
        {
            "rank": "pending",
            "run": "ToricGT ConvexTok-2048 Graphified FoT",
            "score": toricgt_score,
            "author": "Amelie Schreiber",
            "summary": toricgt_summary,
            "date": "2026-06-21",
            "info": f'<a href="{html.escape(HF_URL)}">checkpoint</a> · <a href="{html.escape(pr_href)}">PR/status</a>',
            "class": "toricgt-row",
        }
    ]
    for idx, row in enumerate(official, start=1):
        rows.append(
            {
                "rank": str(idx),
                "run": row["run"],
                "score": row["score"],
                "author": row["author"],
                "summary": row["summary"],
                "date": row["date"],
                "info": markdown_links_to_html(row["info"]),
                "class": "",
            }
        )
    body = []
    for row in rows:
        body.append(
            f"""
            <tr class="{html.escape(row['class'])}">
              <td>{html.escape(row['rank'])}</td>
              <th>{html.escape(row['run'])}</th>
              <td class="score-cell">{html.escape(row['score'])}</td>
              <td>{html.escape(row['author'])}</td>
              <td>{html.escape(row['summary'])}</td>
              <td>{html.escape(row['date'])}</td>
              <td>{row['info']}</td>
            </tr>
            """
        )
    return "\n".join(body)


def run_history_json(rows: list[dict[str, Any]]) -> str:
    data = []
    for row in rows:
        metrics = row.get("metrics", {})
        data.append(
            {
                "run": str(row.get("run_id", ""))[-48:],
                "profile": row.get("profile"),
                "bpb": metric_bpb(metrics if isinstance(metrics, dict) else {}),
                "artifact": metrics.get("artifact_bytes") if isinstance(metrics, dict) else None,
                "train_bpb": metrics.get("train_bpb") if isinstance(metrics, dict) else None,
            }
        )
    return json.dumps(data, sort_keys=True)


METHOD_CSS = """
:root{color-scheme:dark;--bg:#030712;--panel:#071421;--panel2:#0b1b2b;--line:rgba(70,231,255,.28);--text:#ecfbff;--muted:#9db8cf;--cyan:#46e7ff;--gold:#ffd166;--pink:#ff5fa2;--green:#88ff86;--violet:#a78bfa}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 12% 10%,rgba(70,231,255,.18),transparent 30rem),radial-gradient(circle at 86% 6%,rgba(255,95,162,.13),transparent 24rem),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1240px;margin:0 auto;padding:28px 20px 64px}a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}
.hero,.panel,.card{border:1px solid var(--line);border-radius:18px;background:linear-gradient(180deg,rgba(11,27,43,.92),rgba(7,20,33,.88));box-shadow:0 24px 80px rgba(0,0,0,.38)}
.hero{padding:24px;margin-bottom:18px}.panel{padding:18px;margin:18px 0}.grid{display:grid;gap:16px}.two{grid-template-columns:1fr 1fr}.three{grid-template-columns:repeat(3,minmax(0,1fr))}
h1{font-size:clamp(2.1rem,5vw,4.5rem);line-height:.95;margin:0 0 12px;letter-spacing:0}h2{font-size:clamp(1.35rem,3vw,2.2rem);margin:0 0 12px}h3{margin:0 0 8px}.lead,p,li{color:#bdd6e8;line-height:1.62}.eyebrow{color:var(--gold);text-transform:uppercase;letter-spacing:.15em;font-weight:800;font-size:.78rem}.equation{font-family:"STIX Two Text",Cambria,Georgia,serif;color:#fff;background:#020713;border:1px solid rgba(157,184,207,.18);border-radius:12px;padding:12px 14px;overflow:auto}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:6px 10px;margin:3px;background:rgba(70,231,255,.08);font-size:.9rem}.metric-note{font-size:.94rem;color:var(--muted);line-height:1.5}.card{padding:16px;min-height:0;overflow-wrap:anywhere}.card p{margin-bottom:0}svg{max-width:100%}.plot{height:560px;border:1px solid rgba(70,231,255,.18);border-radius:14px;background:#020713}.source-img{width:100%;max-height:360px;object-fit:contain;border:1px solid rgba(70,231,255,.18);border-radius:14px;background:#020713;padding:10px}.mini-table{width:100%;border-collapse:collapse}.mini-table th,.mini-table td{border-bottom:1px solid rgba(157,184,207,.16);padding:9px;text-align:left;vertical-align:top}.mini-table th{color:#9fdcff}.accent{color:var(--gold);font-weight:800}.callout{border-left:3px solid var(--gold);background:rgba(255,209,102,.08);border-radius:12px;padding:12px 14px;color:#ffe8a8}.detail-panel{border:1px solid rgba(70,231,255,.18);border-radius:14px;background:#020713;padding:14px;margin-top:12px;min-height:180px}.detail-panel strong{color:#fff}.svg-button,.clickable{cursor:pointer}.svg-button:hover,.clickable:hover{filter:drop-shadow(0 0 10px rgba(255,209,102,.75))}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0 14px}.controls button,.controls select{border:1px solid var(--line);background:rgba(70,231,255,.08);color:var(--text);border-radius:999px;padding:8px 12px;font-weight:800}.controls label{display:grid;gap:4px;color:var(--muted);font-size:.86rem}.controls input[type=range]{width:min(320px,72vw);accent-color:var(--cyan)}.hidden{display:none!important}.method-list{margin:0;padding-left:1.1rem}.method-list li{margin:0 0 .65rem}@media(max-width:900px){.two,.three{grid-template-columns:1fr}.plot{height:420px}}
"""


def method_report_shell(title: str, eyebrow: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{METHOD_CSS}</style></head><body><main>
<section class="hero"><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1></section>
{body}
</main></body></html>
"""


def write_fot_method_report(docs: Path, copied: dict[str, str]) -> dict[str, str]:
    out = docs / "page_interactive" / "fot_method"
    out.mkdir(parents=True, exist_ok=True)
    image = "../../" + copied.get("fot_reference", "")
    fot_nodes = {
        "root": (68, 210, "#ffd166", "prompt root; initializes the embedding-space forest"),
        "t0a": (178, 112, "#46e7ff", "tree 0 branch; high fan margin, medium BPB delta"),
        "t0b": (178, 296, "#46e7ff", "tree 0 branch; exploratory graph edit"),
        "t1a": (328, 66, "#a78bfa", "tree 1 branch; retrieved analogy and topological stability"),
        "t1b": (332, 202, "#a78bfa", "tree 1 branch; correction branch"),
        "t2a": (470, 292, "#ff5fa2", "tree 2 branch; diversity-preserving candidate"),
        "vote": (610, 178, "#88ff86", "forest consensus; candidate selected by BPB-delta reward"),
    }
    fot_edges = [
        ("root", "t0a", "spawn tree branch"),
        ("root", "t0b", "spawn exploratory branch"),
        ("t0a", "t1a", "expand promising branch"),
        ("t0a", "t1b", "self-correction branch"),
        ("t0b", "t2a", "independent tree expansion"),
        ("t1a", "vote", "consensus edge"),
        ("t1b", "vote", "consensus edge"),
        ("t2a", "vote", "minority but diverse vote"),
    ]
    edge_svg = []
    for a, b, label in fot_edges:
        x1, y1, _, _ = fot_nodes[a]
        x2, y2, _, _ = fot_nodes[b]
        edge_svg.append(
            f'<path d="M{x1} {y1} C{(x1+x2)//2} {y1-42}, {(x1+x2)//2} {y2+42}, {x2} {y2}" '
            f'fill="none" stroke="#46e7ff" stroke-width="4" stroke-opacity=".72"><title>{html.escape(label)}</title></path>'
        )
    node_svg = []
    for node_id, (x, y, color, label) in fot_nodes.items():
        node_svg.append(
            f'<circle id="fot-node-{html.escape(node_id)}" data-node="{html.escape(node_id)}" class="svg-button" '
            f'cx="{x}" cy="{y}" r="18" fill="{color}" stroke="#ecfbff" stroke-width="1.5" role="button" '
            f'aria-label="Reveal FoT tree for {html.escape(label)}" '
            f'tabindex="0">'
            f'<title>{html.escape(label)}</title></circle>'
            f'<text x="{x}" y="{y+40}" fill="#ecfbff" font-size="16" text-anchor="middle">{html.escape(node_id)}</text>'
        )
    fot_detail_payload = {
        "root": {
            "title": "Prompt Root",
            "role": "Initial graph state before the forest is expanded.",
            "tree": [["prompt", "token graph"], ["prompt", "retrieval query"], ["token graph", "byte flattening"]],
            "metrics": {"reward seed": "BPB prior + graph validity", "fan": "no wall crossing yet", "memory": "query only"},
        },
        "t0a": {
            "title": "High-Margin Branch",
            "role": "First tree branch with the best early tropical fan margin.",
            "tree": [["t0a", "local edit"], ["t0a", "analogy probe"], ["local edit", "candidate answer"], ["analogy probe", "memory support"]],
            "metrics": {"reward": "positive BPB-delta", "fan margin": "high", "PH gate": "medium-high"},
        },
        "t0b": {
            "title": "Exploratory Branch",
            "role": "Diversity branch retained to avoid early collapse into one reasoning path.",
            "tree": [["t0b", "alternate graph edge"], ["t0b", "low-rank toric probe"], ["alternate graph edge", "candidate answer"]],
            "metrics": {"reward": "uncertain", "diversity": "high", "correction need": "medium"},
        },
        "t1a": {
            "title": "Retrieved-Analogy Branch",
            "role": "Branch that uses memory retrieval after simplex-map and PH similarity checks.",
            "tree": [["t1a", "simplex map"], ["t1a", "PH vector gate"], ["simplex map", "memory trajectory"], ["PH vector gate", "retrieval score"]],
            "metrics": {"full map": "strong", "PH cosine": "high", "memory": "accepted"},
        },
        "t1b": {
            "title": "Self-Correction Branch",
            "role": "Correction branch for a candidate whose byte likelihood improved but certificate pressure was weak.",
            "tree": [["t1b", "certificate check"], ["t1b", "graph repair"], ["certificate check", "BGG/Koszul probe"], ["graph repair", "candidate answer"]],
            "metrics": {"reward": "mixed", "certificate": "improved", "BPB": "small positive"},
        },
        "t2a": {
            "title": "Independent Tree Expansion",
            "role": "Separate thought tree that preserves forest-level diversity and can outvote brittle local branches.",
            "tree": [["t2a", "alternate token DAG"], ["t2a", "toric wall crossing"], ["alternate token DAG", "flattened output"], ["toric wall crossing", "fan audit"]],
            "metrics": {"diversity": "high", "fan crossing": "controlled", "reward": "candidate"},
        },
        "vote": {
            "title": "Forest Consensus",
            "role": "Consensus node that chooses a compact output path from branch rewards and structural gates.",
            "tree": [["vote", "best BPB branch"], ["vote", "certificate-safe branch"], ["vote", "diversity backup"], ["best BPB branch", "final graph"], ["certificate-safe branch", "final graph"]],
            "metrics": {"selection": "BPB-delta first", "fallback": "structural gates", "output": "graph plus optional flattening"},
        },
    }
    forest_svg = f"""
<svg viewBox="0 0 700 390" role="img" aria-label="Forest-of-Thought embedding-space forest diagram">
  <rect x="8" y="8" width="684" height="374" rx="18" fill="#020713" stroke="rgba(70,231,255,.30)"/>
  <defs><marker id="fotArrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#46e7ff"/></marker></defs>
  <g marker-end="url(#fotArrow)">{''.join(edge_svg)}</g>
  <g>{''.join(node_svg)}</g>
</svg>
"""
    body = f"""
<section class="panel">
  <p class="lead">Forest-of-Thought (FoT) scales reasoning by running several thought trees, activating promising branches sparsely, correcting weak branches, and using consensus over the forest. In ToricGT the same idea is moved from prompt strings into embedding space: each branch is a graph-valued hidden trajectory, each edge is a graph edit or continuous displacement, and branch rewards are tied to BPB delta, verifier score, topological stability, and toric fan margins.</p>
  <p><a href="https://arxiv.org/abs/2412.09078">FoT arXiv:2412.09078</a> · <a href="https://github.com/iamhankai/Forest-of-Thought">upstream code</a> · <a href="https://huggingface.co/blog/AmelieSchreiber/toricblm">ToricBLM continuation</a></p>
</section>
<section class="grid two">
  <article class="panel"><h2>FoT Reference Diagram</h2>{f'<img class="source-img" src="{html.escape(image)}" alt="Forest-of-Thought paper diagram">' if image != "../../" else '<p>Local FoT paper image not found.</p>'}</article>
  <article class="panel"><h2>Embedding-Space Adaptation</h2><p class="metric-note">Click or keyboard-select a vertex to reveal its local thought tree and branch metrics. The SVG is a lightweight projection; the training controller works in the original embedding coordinates.</p>{forest_svg}<div id="fotDetail" class="detail-panel">Click a vertex to reveal the local thought tree, branch role, and training signals for that node.</div></article>
</section>
<section class="grid three">
  <article class="card"><h3>State</h3><p>Each state is <span class="accent">s=(G,H,Σ,μ,M)</span>: a reasoning graph, hidden table, toric/fan chart, moment coordinate, and memory pointer.</p></article>
  <article class="card"><h3>Action</h3><p>Actions add/merge thought nodes, move inside a normal cone, cross a wall, retrieve an analogy, or flatten a graph output for BPB scoring.</p></article>
  <article class="card"><h3>Reward</h3><p>Reward combines byte likelihood improvement with structural diversity, GUDHI persistent-homology stability, and toric/BGG certificate consistency.</p></article>
</section>
<section class="panel"><h2>Training Equation</h2><div class="equation">L<sub>FoT</sub> = (log Z + Σ<sub>t</sub> log p<sub>F</sub>(s<sub>t+1</sub>|s<sub>t</sub>) − log R(x) − Σ<sub>t</sub> log p<sub>B</sub>(s<sub>t</sub>|s<sub>t+1</sub>))<sup>2</sup>, with R(x)=exp((ΔBPB + α·diversity + β·fan_margin − γ·certificate_error)/τ).</div></section>
<script>
const fotDetails = {json.dumps(fot_detail_payload)};
function escapeHtml(value) {{
  return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}
function renderMiniTree(edges) {{
  const nodes = Array.from(new Set(edges.flat()));
  const levels = new Map();
  levels.set(nodes[0], 0);
  for (const [a,b] of edges) levels.set(b, Math.max(levels.get(b) || 0, (levels.get(a) || 0) + 1));
  const byLevel = {{}};
  for (const node of nodes) {{
    const level = levels.get(node) || 0;
    (byLevel[level] ||= []).push(node);
  }}
  const positions = {{}};
  for (const [levelText, items] of Object.entries(byLevel)) {{
    const level = Number(levelText);
    items.forEach((node, idx) => {{
      positions[node] = [72 + level * 185, 72 + (idx - (items.length - 1) / 2) * 86];
    }});
  }}
  const edgeSvg = edges.map(([a,b]) => {{
    const p = positions[a], q = positions[b];
    return `<line x1="${{p[0]}}" y1="${{p[1]}}" x2="${{q[0]}}" y2="${{q[1]}}" stroke="#46e7ff" stroke-width="3" stroke-opacity=".72" />`;
  }}).join('');
  const nodeSvg = nodes.map(node => {{
    const p = positions[node];
    return `<g><circle cx="${{p[0]}}" cy="${{p[1]}}" r="15" fill="#ffd166" stroke="#ecfbff"/><text x="${{p[0]}}" y="${{p[1]+34}}" text-anchor="middle" fill="#ecfbff" font-size="13">${{escapeHtml(node)}}</text></g>`;
  }}).join('');
  return `<svg viewBox="0 0 560 260" role="img" aria-label="Local FoT thought tree"><rect x="8" y="8" width="544" height="244" rx="14" fill="#071421" stroke="rgba(70,231,255,.20)"/>${{edgeSvg}}${{nodeSvg}}</svg>`;
}}
function renderFotDetail(id) {{
  const detail = fotDetails[id];
  if (!detail) return;
  const metrics = Object.entries(detail.metrics || {{}}).map(([k,v]) => `<tr><th>${{escapeHtml(k)}}</th><td>${{escapeHtml(v)}}</td></tr>`).join('');
  document.getElementById('fotDetail').innerHTML = `<strong>${{escapeHtml(detail.title)}}</strong><p>${{escapeHtml(detail.role)}}</p>${{renderMiniTree(detail.tree || [])}}<table class="mini-table"><tbody>${{metrics}}</tbody></table>`;
}}
document.querySelectorAll('[data-node]').forEach(node => {{
  node.addEventListener('click', () => renderFotDetail(node.dataset.node));
  node.addEventListener('keydown', event => {{
    if (event.key === 'Enter' || event.key === ' ') {{
      event.preventDefault();
      renderFotDetail(node.dataset.node);
    }}
  }});
}});
renderFotDetail('root');
</script>
"""
    (out / "index.html").write_text(method_report_shell("Forest-of-Thought Reasoning", "Embedding-space reasoning forest", body), encoding="utf-8")
    return {
        "href": "page_interactive/fot_method/index.html",
        "title": "Forest-of-Thought Method",
        "summary": "Dark-mode FoT explanation and interactive embedding-space forest diagram adapted to ToricGT’s graph-valued hidden trajectories.",
        "features": "multiple thought trees · sparse activation · self-correction · BPB-delta reward",
        "embed": "0",
    }


def write_convextok_method_report(docs: Path) -> dict[str, str]:
    out = docs / "page_interactive" / "convextok_method"
    out.mkdir(parents=True, exist_ok=True)
    sample = "token graph"
    edges = [
        *[
            {
                "i": idx,
                "j": idx + 1,
                "token": sample[idx : idx + 1],
                "kind": "free byte fallback",
                "cost": 1.0,
                "lp": 0.0,
                "selected": False,
                "lane": -1,
            }
            for idx in range(len(sample))
        ],
        {"i": 0, "j": 5, "token": "token", "kind": "selected rounded token", "cost": 0.27, "lp": 0.82, "selected": True, "lane": 2},
        {"i": 0, "j": 6, "token": "token ", "kind": "priced vocabulary token", "cost": 0.35, "lp": 0.48, "selected": False, "lane": 1},
        {"i": 6, "j": 11, "token": "graph", "kind": "selected rounded token", "cost": 0.21, "lp": 0.77, "selected": True, "lane": 2},
        {"i": 6, "j": 10, "token": "grap", "kind": "priced vocabulary token", "cost": 0.29, "lp": 0.32, "selected": False, "lane": 1},
        {"i": 0, "j": 11, "token": "token graph", "kind": "LP-relaxation long-span candidate", "cost": 0.49, "lp": 0.41, "selected": False, "lane": 4},
    ]
    width, height = 900, 500
    margin_x, baseline = 62, 340
    step_x = (width - 2 * margin_x) / len(sample)

    def x_at(idx: int) -> float:
        return margin_x + step_x * idx

    def edge_color(edge: dict[str, Any]) -> str:
        if edge["selected"]:
            return "#88ff86"
        if "LP-relaxation" in edge["kind"]:
            return "#ffd166"
        if "fallback" in edge["kind"]:
            return "rgba(70,231,255,.58)"
        return "#a78bfa"

    edge_paths: list[str] = []
    edge_paths_3d: list[str] = []
    for edge_index, edge in enumerate(edges):
        x1, x2 = x_at(edge["i"]), x_at(edge["j"])
        mid = (x1 + x2) / 2
        span = edge["j"] - edge["i"]
        if "fallback" in edge["kind"]:
            control_y = baseline + 34
            label_y = baseline + 62
            stroke_width = 2
            dash = ""
            label = ""
        else:
            control_y = baseline - (46 + 12 * span + 22 * edge["lane"])
            stroke_width = 7 if edge["selected"] else 4
            dash = ' stroke-dasharray="8 8"' if "LP-relaxation" in edge["kind"] else ""
        selected = "yes" if edge["selected"] else "no"
        title = (
            f"token {edge['token']!r}; span [{edge['i']},{edge['j']}]; "
            f"{edge['kind']}; cost {edge['cost']:.3f}; LP {edge['lp']:.2f}; selected {selected}"
        )
        edge_paths.append(
            f'<path d="M{x1:.1f} {baseline:.1f} Q{mid:.1f} {control_y:.1f} {x2:.1f} {baseline:.1f}" '
            f'class="clickable dag-edge" data-edge="{edge_index}" fill="none" stroke="{edge_color(edge)}" stroke-width="{stroke_width}" stroke-linecap="round"{dash}>'
            f'<title>{html.escape(title)}</title></path>'
        )
        z1 = 42 * edge["lane"] if edge["lane"] > 0 else -22
        z2 = z1
        x1p, y1p = x1 + z1 * 0.45, baseline - z1 * 0.55
        x2p, y2p = x2 + z2 * 0.45, baseline - z2 * 0.55
        control_y3 = min(y1p, y2p) - (30 + 14 * span)
        edge_paths_3d.append(
            f'<path d="M{x1p:.1f} {y1p:.1f} Q{(x1p+x2p)/2:.1f} {control_y3:.1f} {x2p:.1f} {y2p:.1f}" '
            f'class="clickable dag-edge" data-edge="{edge_index}" fill="none" stroke="{edge_color(edge)}" stroke-width="{stroke_width}" stroke-linecap="round"{dash}>'
            f'<title>{html.escape(title)}</title></path>'
        )
    node_svg = []
    node_svg_3d = []
    for idx in range(len(sample) + 1):
        node_svg.append(
            f'<circle class="clickable dag-node" data-node="{idx}" cx="{x_at(idx):.1f}" cy="{baseline}" r="9" fill="#46e7ff" stroke="#ecfbff" stroke-width="1.2">'
            f'<title>byte boundary {idx}; prefix {html.escape(repr(sample[:idx]))}</title></circle>'
            f'<text x="{x_at(idx):.1f}" y="{baseline + 28}" fill="#9db8cf" font-size="13" text-anchor="middle">{idx}</text>'
        )
        xp, yp = x_at(idx) - 10, baseline + 12
        node_svg_3d.append(
            f'<circle class="clickable dag-node" data-node="{idx}" cx="{xp:.1f}" cy="{yp:.1f}" r="8" fill="#46e7ff" stroke="#ecfbff" stroke-width="1.2">'
            f'<title>byte boundary {idx}; prefix {html.escape(repr(sample[:idx]))}</title></circle>'
            f'<text x="{xp:.1f}" y="{yp + 24:.1f}" fill="#9db8cf" font-size="12" text-anchor="middle">{idx}</text>'
        )
    char_svg = []
    for idx, ch in enumerate(sample):
        char_label = "space" if ch == " " else ch
        char_svg.append(
            f'<text x="{(x_at(idx)+x_at(idx+1))/2:.1f}" y="{baseline + 48}" fill="#ecfbff" font-size="13" text-anchor="middle">'
            f'{html.escape(char_label)}</text>'
        )
    edge_rows = "\n".join(
        "<tr>"
        f"<td>{edge['i']}→{edge['j']}</td>"
        f"<td><code>{html.escape(edge['token'])}</code></td>"
        f"<td>{html.escape(edge['kind'])}</td>"
        f"<td>{edge['cost']:.3f}</td>"
        f"<td>{edge['lp']:.2f}</td>"
        f"<td>{'yes' if edge['selected'] else 'no'}</td>"
        "</tr>"
        for edge in edges
    )
    dag_svg = f"""
<svg id="dag2d" viewBox="0 0 {width} {height}" role="img" aria-label="ConvexTok byte-boundary tokenization DAG">
  <rect x="8" y="8" width="{width-16}" height="{height-16}" rx="18" fill="#020713" stroke="rgba(70,231,255,.30)"/>
  <text x="32" y="42" fill="#ffd166" font-size="20">sample string: {html.escape(sample)}</text>
  <text x="32" y="70" fill="#9db8cf" font-size="15">Vertices are byte boundaries; arcs are candidate tokens. Click arcs and vertices for exact edge metadata.</text>
  <line x1="{margin_x}" y1="{baseline}" x2="{width-margin_x}" y2="{baseline}" stroke="rgba(236,251,255,.32)" stroke-width="2"/>
  <g>{''.join(edge_paths)}</g>
  <g>{''.join(node_svg)}</g>
  <g>{''.join(char_svg)}</g>
  <g font-size="14">
    <rect x="32" y="392" width="16" height="8" fill="#88ff86"/><text x="56" y="401" fill="#bdd6e8">selected rounded path</text>
    <rect x="244" y="392" width="16" height="8" fill="#a78bfa"/><text x="268" y="401" fill="#bdd6e8">priced vocabulary token</text>
    <rect x="496" y="392" width="16" height="8" fill="#ffd166"/><text x="520" y="401" fill="#bdd6e8">LP-relaxation support</text>
    <rect x="32" y="422" width="16" height="8" fill="#46e7ff"/><text x="56" y="431" fill="#bdd6e8">free byte fallback</text>
  </g>
</svg>
<svg id="dag3d" class="hidden" viewBox="0 0 {width} {height}" role="img" aria-label="ConvexTok projected 3D tokenization cost layers">
  <rect x="8" y="8" width="{width-16}" height="{height-16}" rx="18" fill="#020713" stroke="rgba(70,231,255,.30)"/>
  <text x="32" y="42" fill="#ffd166" font-size="20">projected 3D cost layers</text>
  <text x="32" y="70" fill="#9db8cf" font-size="15">x = byte boundary, height = token span/layer, color = selected, priced, LP, or fallback edge.</text>
  <g stroke="rgba(157,184,207,.20)" stroke-width="1">
    <line x1="{margin_x-10}" y1="{baseline+12}" x2="{width-margin_x}" y2="{baseline+12}"/>
    <line x1="{margin_x-10}" y1="{baseline+12}" x2="{margin_x+96}" y2="{baseline-116}"/>
    <line x1="{margin_x-10}" y1="{baseline+12}" x2="{margin_x-10}" y2="{baseline-150}"/>
  </g>
  <g>{''.join(edge_paths_3d)}</g>
  <g>{''.join(node_svg_3d)}</g>
  <g font-size="14">
    <rect x="32" y="392" width="16" height="8" fill="#88ff86"/><text x="56" y="401" fill="#bdd6e8">selected path</text>
    <rect x="210" y="392" width="16" height="8" fill="#a78bfa"/><text x="234" y="401" fill="#bdd6e8">priced candidates</text>
    <rect x="410" y="392" width="16" height="8" fill="#ffd166"/><text x="434" y="401" fill="#bdd6e8">LP-relaxation edge</text>
    <rect x="630" y="392" width="16" height="8" fill="#46e7ff"/><text x="654" y="401" fill="#bdd6e8">fallback bytes</text>
  </g>
</svg>
"""
    body = f"""
<section class="panel">
  <p class="lead">ConvexTok turns tokenizer construction into a shortest-path and sparse linear-programming problem over a byte-boundary DAG. ToricGT uses that DAG as real model input: byte boundaries are nodes, candidate substrings are edges, LP relaxation scores are edge features, and the final segmentation is the selected path used for BPB scoring.</p>
  <p><a href="https://arxiv.org/abs/2605.22821">ConvexTok preprint</a> · <a href="https://github.com/openai/parameter-golf">OpenAI Parameter Golf</a></p>
  <p class="callout">The key point is not merely “a different tokenizer.” The tokenization computation is itself a graph-structured dynamic program, so it matches TokenGT-style graphification instead of being hidden preprocessing.</p>
</section>
<section class="panel"><h2>Tokenization DAG</h2>
  <div class="controls"><button type="button" id="showDag2d">Layered DAG</button><button type="button" id="showDag3d">Projected 3D Cost View</button></div>
  {dag_svg}
  <div id="dagDetail" class="detail-panel"><strong>Click an arc or boundary vertex.</strong><p>Edge metadata will appear here without loading any external plotting runtime.</p></div>
  <p class="metric-note">Curved arcs span byte intervals. Cyan bottom arcs are fallback byte edges, violet arcs are priced candidate tokens, gold arcs are LP-relaxation support, and green arcs are the selected rounded token path. The model receives the same structure as TokenGT-style node/edge features; BPB is still scored on the flattened selected path.</p>
</section>
<section class="grid two">
  <article class="panel"><h2>Equations</h2>
    <div class="equation">D[j] = min<sub>(i,j,t)∈E</sub> D[i] + w<sub>t</sub></div>
    <div class="equation">min Σ<sub>e</sub> c<sub>e</sub>x<sub>e</sub> subject to flow conservation and x<sub>e</sub> ≤ y<sub>token(e)</sub>, Σ<sub>t</sub> y<sub>t</sub> ≤ B.</div>
    <p>The min-plus recurrence is tropical dynamic programming. The LP lower bound gives a tokenizer-regret metric: path length minus relaxed optimum. ToricGT can log this gap to decide whether BPB is tokenizer-bound or model-bound.</p>
    <table class="mini-table"><tbody>
      <tr><th>Boundary node</th><td>Byte offset in the source string; causal topological order is left to right.</td></tr>
      <tr><th>Candidate edge</th><td>A token proposal from offset <code>i</code> to <code>j</code>, carrying byte length, token rank, price, LP score, and selected-path flags.</td></tr>
      <tr><th>Tropical path</th><td>The rounded segmentation is a min-plus path through the DAG; active edges form a tropical curve inside the tokenizer graph.</td></tr>
    </tbody></table>
  </article>
  <article class="panel"><h2>Candidate Edge Metadata</h2><table class="mini-table"><thead><tr><th>span</th><th>token</th><th>kind</th><th>cost</th><th>LP</th><th>selected</th></tr></thead><tbody>{edge_rows}</tbody></table></article>
</section>
<section class="grid three">
  <article class="card"><h3>1. BPB Use</h3><p>FineWeb is graphified, but the OAI scoring path still flattens the selected token path. This keeps the byte objective primary while letting the model learn node/edge structure around the tokenizer.</p></article>
  <article class="card"><h3>2. Tropical Use</h3><p>The dynamic program <span class="accent">D[j]=min(D[i]+w)</span> is min-plus computation. Active edges, margins, and path regret become tropical diagnostics instead of opaque tokenizer side effects.</p></article>
  <article class="card"><h3>3. Toric Use</h3><p>Candidate-token exponents and LP scores define active faces of a tokenization polytope. Those faces are embedded into toric charts for fan, one-dimensional-cone, divisor, and sheaf-style audits.</p></article>
  <article class="card"><h3>4. Graph Use</h3><p>Boundary nodes, candidate-token edges, endpoint offsets, edge ranks, selected-path flags, and LP scores are TokenGT-style features. The graph is causal left-to-right for FineWeb.</p></article>
  <article class="card"><h3>5. Regret Use</h3><p>The LP lower bound gives a tokenizer-regret metric: if rounded path length is close to the LP optimum, BPB problems are more likely model-bound; otherwise tokenizer vocabulary or rounding is suspect.</p></article>
  <article class="card"><h3>6. OOD Use</h3><p>Because substring choices are graph paths, the model sees reusable local graph grammar rather than isolated token IDs. That is the bridge to graph-structured biological and 3D tokenizers later.</p></article>
</section>
<script>
const dagEdges = {json.dumps(edges)};
const sampleString = {json.dumps(sample)};
function esc(value) {{
  return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}
function showDag(which) {{
  document.getElementById('dag2d').classList.toggle('hidden', which !== '2d');
  document.getElementById('dag3d').classList.toggle('hidden', which !== '3d');
}}
function showEdgeDetail(index) {{
  const e = dagEdges[Number(index)];
  if (!e) return;
  document.getElementById('dagDetail').innerHTML =
    `<strong>${{esc(e.kind)}}</strong><table class="mini-table"><tbody>`+
    `<tr><th>token</th><td><code>${{esc(e.token)}}</code></td></tr>`+
    `<tr><th>span</th><td>${{e.i}} → ${{e.j}}</td></tr>`+
    `<tr><th>cost</th><td>${{Number(e.cost).toFixed(3)}}</td></tr>`+
    `<tr><th>LP mass</th><td>${{Number(e.lp).toFixed(2)}}</td></tr>`+
    `<tr><th>selected rounded path</th><td>${{e.selected ? 'yes' : 'no'}}</td></tr>`+
    `</tbody></table>`;
}}
function showNodeDetail(index) {{
  const i = Number(index);
  document.getElementById('dagDetail').innerHTML =
    `<strong>byte boundary ${{i}}</strong><p>Prefix: <code>${{esc(sampleString.slice(0, i))}}</code></p>`+
    `<p>Candidate token edges enter or leave this boundary in the tokenizer DAG. FineWeb BPB is scored after the selected path is flattened back to a sequence.</p>`;
}}
document.getElementById('showDag2d').addEventListener('click', () => showDag('2d'));
document.getElementById('showDag3d').addEventListener('click', () => showDag('3d'));
document.querySelectorAll('.dag-edge').forEach(el => el.addEventListener('click', () => showEdgeDetail(el.dataset.edge)));
document.querySelectorAll('.dag-node').forEach(el => el.addEventListener('click', () => showNodeDetail(el.dataset.node)));
showEdgeDetail(dagEdges.findIndex(e => e.selected));
</script>
"""
    (out / "index.html").write_text(method_report_shell("ConvexTok Tokenization Geometry", "Tokenizer as tropical DAG and toric chart", body), encoding="utf-8")
    return {
        "href": "page_interactive/convextok_method/index.html",
        "title": "ConvexTok Tokenization Geometry",
        "summary": "Interactive byte-boundary DAG, min-plus equations, LP tokenizer-regret interpretation, and ToricGT TokenGT graphification links.",
        "features": "LP lower bound · min-plus DP · tokenization DAG · tokenizer regret",
        "embed": "0",
    }


def write_bgg_method_report(docs: Path, best_metrics: dict[str, Any]) -> dict[str, str]:
    out = docs / "page_interactive" / "bgg_method"
    out.mkdir(parents=True, exist_ok=True)
    copied_out = docs / "page_interactive" / "bgg_category_o"
    copied_out.mkdir(parents=True, exist_ok=True)
    copied_report = load_json(copied_out / "bgg_category_o_report.json")
    copied_metrics = copied_report.get("metrics", {}) if isinstance(copied_report.get("metrics"), dict) else {}
    copied_provenance = copied_report.get("metric_provenance", {}) if isinstance(copied_report.get("metric_provenance"), dict) else {}
    copied_gates = copied_report.get("gates", {}) if isinstance(copied_report.get("gates"), dict) else {}
    unavailable = copied_report.get("unavailable", []) if isinstance(copied_report.get("unavailable"), list) else []
    metrics = {
        "toric_bgg_loss": best_metrics.get("toric_bgg_loss"),
        "koszul_persistence_loss": best_metrics.get("koszul_persistence_loss"),
        "derived_signature_loss": best_metrics.get("derived_signature_loss"),
        "toric_cca_topology_loss": best_metrics.get("toric_cca_topology_loss"),
    }
    table = "".join(f"<tr><th>{html.escape(k)}</th><td>{fmt(v,6)}</td></tr>" for k, v in metrics.items())
    exact_rows = []
    exact_keys = [
        ("toric_bgg_d2_residual", "Boundary-square residual for sparse finite differentials. Low means the predicted differential behaves like a chain-complex map with d²=0."),
        ("toric_bgg_standard_leakage", "Attention mass outside the standard-filtration mask. Low means hidden standard-object routing respects the poset."),
        ("toric_bgg_koszul_linearity_residual", "Deviation from the expected Koszul degree profile. Low means the finite BGG/Koszul proxy is degree-consistent."),
        ("toric_bgg_gale_dual_consistency", "Agreement between an arrangement certificate and its Gale-dual signature. High consistency supports dual-curriculum transfer."),
        ("toric_bgg_signature_smoothness", "Variation of the BGG signature along the reasoning trajectory. Low values indicate stable categorical coordinates."),
        ("toric_bgg_standard_entropy", "Entropy of standard-label assignments. It detects collapse to one standard object or diffuse unusable labels."),
    ]
    for key, meaning in exact_keys:
        value = copied_metrics.get(key)
        status = "unavailable in copied run" if key in unavailable or value is None else fmt(value, 6)
        exact_rows.append(
            f"<tr><th>{html.escape(key)}</th><td>{html.escape(status)}</td><td>{html.escape(copied_provenance.get(key, 'not recorded'))}</td><td>{html.escape(meaning)}</td></tr>"
        )
    gate_rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{fmt(v, 4)}</td></tr>" for k, v in copied_gates.items())
    body = f"""
<section class="panel">
  <p class="lead">The Toric BGG layer is a finite certificate system: it never claims the model contains a literal abelian category. It attaches small sign-vector posets, standard-filtration masks, sparse differentials, Koszul degree profiles, Gale-dual labels, and derived signatures to training records. Late-phase weights remain small so BPB stays primary.</p>
  <p class="callout">This page combines the hand-written method explainer with the generated Category O audit. If a metric is listed as unavailable, that does not mean the method is absent; it means the copied public run did not emit the exact finite certificate required to compute that metric. The late gate is intentionally conservative.</p>
</section>
<section class="grid two">
  <article class="panel"><h2>Finite Category-O Skeleton</h2>
  <svg viewBox="0 0 700 460" aria-label="BGG poset and chain complex">
    <defs><marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#46e7ff"/></marker></defs>
    <rect x="20" y="20" width="660" height="420" rx="14" fill="#020713" stroke="rgba(70,231,255,.32)"/>
    <g stroke="#46e7ff" stroke-width="3" marker-end="url(#arr)" fill="none">
      <path d="M110 360 L225 270 L340 360"/><path d="M225 270 L350 170 L475 270"/><path d="M340 360 L475 270 L590 360"/><path d="M350 170 L475 80"/>
    </g>
    <g fill="#ffd166" stroke="#fff" stroke-width="1.5">
      <circle cx="110" cy="360" r="20"/><circle cx="340" cy="360" r="20"/><circle cx="590" cy="360" r="20"/><circle cx="225" cy="270" r="20"/><circle cx="475" cy="270" r="20"/><circle cx="350" cy="170" r="20"/><circle cx="475" cy="80" r="20"/>
    </g>
    <g fill="#ecfbff" font-size="18" text-anchor="middle"><text x="110" y="366">Lα</text><text x="340" y="366">Lβ</text><text x="590" y="366">Lγ</text><text x="225" y="276">Δα</text><text x="475" y="276">Δβ</text><text x="350" y="176">P</text><text x="475" y="86">C</text></g>
    <text x="52" y="58" fill="#9db8cf" font-size="18">highest-weight finite skeleton: simples L, standards Δ, projectives P, complex C</text>
    <text x="52" y="398" fill="#88ff86" font-size="16">checks: d²=0 · standard masks · Gale dual</text>
    <text x="52" y="420" fill="#88ff86" font-size="16">Koszul degree profile · signature smoothness</text>
  </svg></article>
  <article class="panel"><h2>Best-Run BGG-Adjacent Losses</h2><table class="mini-table"><tbody>{table}</tbody></table><div class="equation">d(m⊗ξ)=Σᵢ xᵢm⊗eᵢ∧ξ. The square vanishes because xᵢxⱼ=xⱼxᵢ while eᵢ∧eⱼ=-eⱼ∧eᵢ.</div></article>
</section>
<section class="grid two">
  <article class="panel"><h2>Generated Audit Gates</h2><table class="mini-table"><tbody>{gate_rows or '<tr><th>gates</th><td>not emitted in copied run</td></tr>'}</tbody></table><p>A late gate of 1 means exact certificate metrics are expected only in late-phase or certificate-bearing records, avoiding noisy penalties during BPB-first early training.</p></article>
  <article class="panel"><h2>How It Affects Training</h2><ul class="method-list"><li><strong>Weight up carefully:</strong> improves standard-object routing and homological consistency when records carry exact certificates.</li><li><strong>Weight down early:</strong> prevents sparse certificate losses from dominating byte likelihood before BPB stabilizes.</li><li><strong>Use as audit:</strong> unavailable exact metrics should trigger data/certificate instrumentation review, not a blind loss increase.</li></ul></article>
</section>
<section class="panel"><h2>Resolution and Standard-Filtration Metrics</h2><table class="mini-table"><thead><tr><th>metric</th><th>value/status</th><th>provenance</th><th>meaning</th></tr></thead><tbody>{''.join(exact_rows)}</tbody></table>
</section>
"""
    content = method_report_shell("Toric BGG Category O", "Finite homological certificates", body)
    (out / "index.html").write_text(content, encoding="utf-8")
    (copied_out / "index.html").write_text(content, encoding="utf-8")
    return {
        "href": "page_interactive/bgg_category_o/index.html",
        "title": "Toric BGG Category O",
        "summary": "Combined Category O explainer and generated audit table for finite highest-weight skeletons, standard masks, BGG differentials, Gale duality, and Koszul checks.",
        "features": "category O skeleton · d² residual · standard masks · Gale dual · Koszul profile",
        "embed": "0",
    }


def write_theory_gallery_report(docs: Path) -> dict[str, str]:
    out = docs / "page_interactive" / "theory_gallery"
    out.mkdir(parents=True, exist_ok=True)
    body = """
<section class="panel">
  <p class="lead">This gallery is the conceptual map for the ToricGT training stack. It connects tropical dynamic programming, toric embeddings, tableau combinatorics, tensor operations, multiparameter persistence, intersection-style certificates, and control of reasoning trajectories. Every diagram is a lightweight inline SVG with accompanying text so the page stays readable on older hardware.</p>
  <p><a href="../../assets/2210.11433v1.pdf">Multiparameter persistence PDF</a> · <a href="../../assets/ergodic-theory.pdf">Ergodic theory reference</a> · <a href="https://github.com/amelie-iska/Tropical_Quivers_of_Archs/blob/main/tropical_quiver_research_program.tex">Tropical Quivers of Archs</a></p>
</section>
<section class="grid two">
  <article class="card">
    <h3>Tropical Ring Attention Embedded In A Toric Fan</h3>
    <svg viewBox="0 0 620 340"><rect width="620" height="340" rx="14" fill="#020713" stroke="rgba(70,231,255,.28)"/><g transform="translate(160 178)" stroke="#46e7ff" stroke-width="3"><line x1="0" y1="0" x2="120" y2="-88"/><line x1="0" y1="0" x2="128" y2="78"/><line x1="0" y1="0" x2="-108" y2="92"/><line x1="0" y1="0" x2="-96" y2="-95"/><line x1="0" y1="0" x2="8" y2="-130"/></g><polyline points="325,250 390,125 462,160 538,72" fill="none" stroke="#ffd166" stroke-width="7"/><circle cx="390" cy="125" r="8" fill="#88ff86"/><text x="30" y="44" fill="#ecfbff" font-size="22">Yᵢc = maxⱼ(Sᵢⱼ + Vⱼc)</text><text x="310" y="294" fill="#9db8cf" font-size="16">active affine candidate → Newton face → normal cone chart</text></svg>
    <p>Tropical attention is piecewise-linear. The active candidate set is a face of a lifted Newton polytope, and the corresponding normal cone is the toric chart used by the audits.</p>
    <p><strong>Training use:</strong> fan-margin, active-face stability, toric embedding metrics, and BPB-safe routing diagnostics.</p>
  </article>
  <article class="card">
    <h3>Young Tableaux And Schur-Style Coordinates</h3>
    <svg viewBox="0 0 620 340"><rect width="620" height="340" rx="14" fill="#020713" stroke="rgba(70,231,255,.28)"/><g transform="translate(48 62)" fill="rgba(70,231,255,.18)" stroke="#46e7ff" stroke-width="2"><rect x="0" y="0" width="54" height="54"/><rect x="54" y="0" width="54" height="54"/><rect x="108" y="0" width="54" height="54"/><rect x="0" y="54" width="54" height="54"/><rect x="54" y="54" width="54" height="54"/><rect x="0" y="108" width="54" height="54"/></g><g fill="#ecfbff" font-size="25" text-anchor="middle"><text x="75" y="98">1</text><text x="129" y="98">2</text><text x="183" y="98">4</text><text x="75" y="152">2</text><text x="129" y="152">3</text><text x="75" y="206">4</text></g><text x="260" y="93" fill="#ffd166" font-size="24">LλE from symmetrizers</text><text x="260" y="132" fill="#ecfbff" font-size="18">row symmetries + column antisymmetries</text><text x="260" y="176" fill="#9db8cf" font-size="17">tensor channels · wedge features</text><text x="260" y="202" fill="#9db8cf" font-size="17">standard monomial bases</text></svg>
    <p>Tableaux encode structured bases for polynomial and exterior constructions. They are useful when graph-token features behave like tensor products but need symmetry constraints.</p>
    <p><strong>Training use:</strong> candidate future losses for symmetrizer/shuffle consistency and combinatorial certificate compression.</p>
  </article>
  <article class="card">
    <h3>Tensor, Wedge, Symmetric, And Shuffle Relations</h3>
    <svg viewBox="0 0 620 340"><rect width="620" height="340" rx="14" fill="#020713" stroke="rgba(70,231,255,.28)"/><text x="42" y="74" fill="#ecfbff" font-size="27">V ⊗ W</text><text x="42" y="136" fill="#88ff86" font-size="25">∧²V: v⊗w − w⊗v</text><text x="42" y="198" fill="#ffd166" font-size="25">Sym²V: v⊗w + w⊗v</text><path d="M370 72 C450 30 480 128 560 86" fill="none" stroke="#ff5fa2" stroke-width="5"/><path d="M370 178 C452 236 492 134 560 194" fill="none" stroke="#46e7ff" stroke-width="5" stroke-dasharray="9 8"/><text x="42" y="266" fill="#9db8cf" font-size="17">Shuffle maps tell the model which feature products should commute, anticommute, or decompose.</text></svg>
    <p>Graph-token reasoning creates many pairwise and higher-order feature products. Algebraic symmetry gives a principled way to constrain which products should be identified or separated.</p>
    <p><strong>Training use:</strong> low-rank probes for wedge/symmetric consistency and future Schur-functor-inspired regularizers.</p>
  </article>
  <article class="card">
    <h3>F₂[x_level,y_radius] Persistence Modules</h3>
    <svg viewBox="0 0 620 340"><rect width="620" height="340" rx="14" fill="#020713" stroke="rgba(70,231,255,.28)"/><g stroke="rgba(157,184,207,.28)"><path d="M82 270 H540"/><path d="M82 225 H540"/><path d="M82 180 H540"/><path d="M82 135 H540"/><path d="M82 90 H540"/><path d="M112 62 V286"/><path d="M184 62 V286"/><path d="M256 62 V286"/><path d="M328 62 V286"/><path d="M400 62 V286"/><path d="M472 62 V286"/></g><path d="M112 270 L112 135 L184 135 L184 90 L328 90 L328 62 L540 62 L540 270 Z" fill="rgba(255,209,102,.25)" stroke="#ffd166" stroke-width="4"/><circle cx="184" cy="135" r="8" fill="#46e7ff"/><circle cx="328" cy="90" r="8" fill="#ff5fa2"/><text x="82" y="312" fill="#9db8cf" font-size="17">reasoning level</text><text x="18" y="78" fill="#9db8cf" font-size="17">radius</text></svg>
    <p>The full reasoning trajectory can be filtered by reasoning level and embedding radius. Algebraically, this is a two-parameter persistence module over F₂[x_level,y_radius].</p>
    <p><strong>Training use:</strong> GUDHI landscapes/images/entropy, simplex-map gates for analogical memory, and Macaulay2 resolution checks.</p>
  </article>
  <article class="card">
    <h3>Thought Alcove Control</h3>
    <svg viewBox="0 0 620 340"><rect width="620" height="340" rx="14" fill="#020713" stroke="rgba(70,231,255,.28)"/><g stroke="#46e7ff" stroke-width="2"><path d="M100 280 L310 45 L520 280"/><path d="M100 280 H520"/><path d="M205 162 H410"/></g><path d="M132 250 C220 185 260 235 330 142 S450 130 500 84" fill="none" stroke="#88ff86" stroke-width="7"/><circle cx="330" cy="142" r="10" fill="#ffd166"/><text x="46" y="40" fill="#ecfbff" font-size="20">control u(t) keeps trajectories inside stable fan alcoves</text><text x="78" y="318" fill="#9db8cf" font-size="16">Lyapunov, ergodic, and optimal-control diagnostics</text></svg>
    <p>Reasoning paths move through fan cells. A stable path should cross walls for reasons tied to the task, not because the latent state is noisy.</p>
    <p><strong>Training use:</strong> fan-margin scheduling, branch pruning, FoT reward shaping, and future control-theoretic audits.</p>
  </article>
  <article class="card">
    <h3>Intersection And Divisor Certificates</h3>
    <svg viewBox="0 0 620 340"><rect width="620" height="340" rx="14" fill="#020713" stroke="rgba(70,231,255,.28)"/><circle cx="220" cy="170" r="92" fill="rgba(70,231,255,.16)" stroke="#46e7ff" stroke-width="4"/><circle cx="360" cy="170" r="92" fill="rgba(255,95,162,.16)" stroke="#ff5fa2" stroke-width="4"/><circle cx="290" cy="108" r="92" fill="rgba(255,209,102,.13)" stroke="#ffd166" stroke-width="4"/><text x="222" y="174" fill="#ecfbff" text-anchor="middle" font-size="22">σλ</text><text x="363" y="174" fill="#ecfbff" text-anchor="middle" font-size="22">σμ</text><text x="290" y="108" fill="#ecfbff" text-anchor="middle" font-size="22">Dρ</text><text x="92" y="286" fill="#9db8cf" font-size="17">intersection-style numbers become compact certificate features</text></svg>
    <p>Toric divisors, one-dimensional cones, and intersection numbers provide compact summaries of how a local piecewise-linear decision bends across walls.</p>
    <p><strong>Training use:</strong> toric vector-bundle one-dimensional-cone CE, sheaf compatibility, and divisor/fan audit metrics.</p>
  </article>
</section>
"""
    (out / "index.html").write_text(method_report_shell("Advanced Geometry Gallery", "Combinatorics, topology, and control", body), encoding="utf-8")
    return {
        "href": "page_interactive/theory_gallery/index.html",
        "title": "Advanced Geometry Gallery",
        "summary": "Dark-mode diagrams for tropical ring attention, toric fans, Young tableaux, tensor products, persistence modules, and control theory.",
        "features": "Young tableaux · Schur functors · F2[x,y] modules · thought alcoves",
        "embed": "0",
    }


def write_method_reports(docs: Path, copied: dict[str, str], best_metrics: dict[str, Any]) -> dict[str, dict[str, str]]:
    reports = {
        "fot_method": write_fot_method_report(docs, copied),
        "convextok_method": write_convextok_method_report(docs),
        "bgg_category_o": write_bgg_method_report(docs, best_metrics),
        "theory_gallery": write_theory_gallery_report(docs),
    }
    return reports


def card(title: str, value: str, note: str) -> str:
    return f"""
    <article class="metric-card">
      <span>{html.escape(title)}</span>
      <strong>{html.escape(value)}</strong>
      <p>{html.escape(note)}</p>
    </article>
    """


def build_html(
    repo: Path,
    state_path: Path | None,
    state: dict[str, Any],
    pr_url: str,
    copied: dict[str, str],
    interactive: dict[str, dict[str, str]],
) -> str:
    rows = completed_rows(state, current_campaign_only=True)
    best = best_row(rows)
    best_metrics = best.get("metrics", {}) if best else {}
    best_profile = str(best.get("profile", "pending")) if best else "pending"
    best_run_id = str(best.get("run_id", "pending")) if best else "pending"
    campaign_id = str(state.get("campaign_id", state_path.parent.name if state_path else "pending"))
    pr_label = pr_url or "pending after best-of-10 + 900-step candidate"
    pr_href = pr_url or PARAMETER_GOLF_URL
    hero = copied.get("logo") or copied.get("architecture") or ""
    history = run_history_json(rows)
    competition_rows = competition_table(repo, best_metrics, pr_href)
    loss_items = []
    for key, desc in LOSS_DESCRIPTIONS.items():
        if key in best_metrics:
            loss_items.append(
                f"<li><span>{html.escape(key)}</span><strong>{fmt(best_metrics.get(key), 6)}</strong><p>{html.escape(desc)}</p></li>"
            )
    image_cards = [
        ("Architecture", "architecture", "Graph-token, tropical, toric, FoT, and BPB paths in one training system."),
        ("Tropical Active Faces", "tropical", "Max-plus attention exposes active predecessors, margins, and normal-fan cells."),
        ("GraphCG + Topology", "graphcg", "Disentangled concept axes meet vectorized persistent homology and retrieval."),
        ("Branching Reasoning", "branching", "Forest/graph-of-thought branches merge through retrieval and BPB-delta rewards."),
        ("Trajectory Complex", "trajectory", "Filtered simplicial complexes reveal token-level reasoning structure."),
        ("Analogy Maps", "analogy", "Memory retrieval is gated by simplex-tree maps and PH feature similarity."),
        ("Vectorized PH", "vectorized_ph", "Landscapes, images, entropy, and persistence vectors become auditable features."),
        ("CAS Sidecar", "cas", "Sage/Macaulay2-backed algebraic certificates validate toric and module data."),
        ("Toric Embedding", "toric_embedding", "Tropical ring attention is embedded into toric charts for fan/cone audits."),
        ("BGG Category O", "bgg", "Finite Toric BGG certificates track standards, differentials, and homological consistency."),
    ]
    gallery = []
    for title, key, caption in image_cards:
        if key in copied:
            gallery.append(
                f"""
                <figure class="gallery-card">
                  <img loading="lazy" decoding="async" src="{html.escape(copied[key])}" alt="{html.escape(title)}">
                  <figcaption><strong>{html.escape(title)}</strong><span>{html.escape(caption)}</span></figcaption>
                </figure>
                """
            )
    interactive_cards = []
    embedded_reports = []
    for key, report in interactive.items():
        href = report["href"]
        title = report["title"]
        summary = report["summary"]
        features = report["features"]
        interactive_cards.append(
            f"""
            <article class="interactive-card">
              <div>
                <h3>{html.escape(title)}</h3>
                <p>{html.escape(summary)}</p>
                <span>{html.escape(features)}</span>
              </div>
              <a class="button" href="{html.escape(href)}">Open report</a>
            </article>
            """
        )
        if report.get("embed") == "1":
            embedded_reports.append(
                f"""
                <article class="panel interactive-embed">
                  <h3>{html.escape(title)} Live Preview</h3>
                  <p>Embedded directly from the generated HTML report; open it separately for full-screen slider and hover/click use.</p>
                  <iframe src="{html.escape(href)}" title="{html.escape(title)}"></iframe>
                </article>
                """
            )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ToricGT</title>
  <meta name="description" content="ToricGT: graph-token transformers with tropical attention embedded into toric geometry, topological audits, Toric BGG supervision, and Parameter Golf BPB training.">
  <style>
    :root {{
      color-scheme: dark;
      --bg: #030811;
      --panel: #071421;
      --panel2: #0b1b2b;
      --line: rgba(79, 223, 255, .28);
      --text: #ecfbff;
      --muted: #9db8cf;
      --cyan: #46e7ff;
      --gold: #ffd166;
      --pink: #ff5fa2;
      --green: #88ff86;
      --violet: #a78bfa;
      --shadow: 0 24px 80px rgba(0,0,0,.42);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 14% 10%, rgba(70,231,255,.20), transparent 28rem),
        radial-gradient(circle at 82% 4%, rgba(255,95,162,.15), transparent 24rem),
        linear-gradient(135deg, #030811 0%, #06111c 52%, #03101b 100%);
      color: var(--text);
    }}
    a {{ color: var(--cyan); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .wrap {{ width: min(1180px, calc(100vw - 40px)); margin: 0 auto; }}
    header {{
      position: sticky; top: 0; z-index: 20;
      backdrop-filter: blur(18px);
      background: rgba(3,8,17,.78);
      border-bottom: 1px solid var(--line);
    }}
    nav {{ display:flex; align-items:center; justify-content:space-between; min-height:72px; gap:24px; }}
    nav .brand {{ font-weight: 800; letter-spacing: .02em; font-size: 1.05rem; }}
    nav .links {{ display:flex; flex-wrap:wrap; gap: 14px; font-size:.92rem; }}
    .hero {{ padding: 74px 0 48px; display:grid; grid-template-columns: 1.05fr .95fr; gap: 38px; align-items:center; }}
    .eyebrow {{ color: var(--gold); text-transform: uppercase; letter-spacing:.16em; font-size:.77rem; font-weight:700; }}
    h1 {{ margin: 14px 0 18px; font-size: clamp(2.9rem, 7vw, 6.8rem); line-height:.88; letter-spacing:0; }}
    .lead {{ color: #c8e0f0; font-size: clamp(1.05rem, 2vw, 1.35rem); line-height:1.65; max-width: 62ch; }}
    .hero-actions {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:28px; }}
    .button {{ border:1px solid var(--line); background:rgba(70,231,255,.08); color:var(--text); border-radius:999px; padding:11px 15px; font-weight:700; }}
    .button.primary {{ background:linear-gradient(135deg, rgba(70,231,255,.95), rgba(136,255,134,.85)); color:#041017; border:0; }}
    .hero-visual {{ position:relative; min-height: 440px; border:1px solid var(--line); background:rgba(7,20,33,.72); border-radius:28px; overflow:hidden; box-shadow:var(--shadow); }}
    .hero-visual img {{ width:100%; height:100%; object-fit:cover; opacity:.72; position:absolute; inset:0; }}
    .fan {{
      position:absolute; inset:0; display:grid; place-items:center;
      background: radial-gradient(circle, rgba(70,231,255,.15), transparent 42%);
    }}
    .fan svg {{ width:min(88%,560px); filter:drop-shadow(0 0 28px rgba(70,231,255,.35)); }}
    .ray {{ stroke-dasharray: 7 10; animation: dash 5s linear infinite; }}
    .poly {{ animation: pulse 3.6s ease-in-out infinite; transform-origin:center; }}
    @keyframes dash {{ to {{ stroke-dashoffset:-90; }} }}
    @keyframes pulse {{ 50% {{ transform:scale(1.035); opacity:.82; }} }}
    section {{ padding: 38px 0; }}
    .section-head {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-end; margin-bottom:18px; }}
    h2 {{ font-size: clamp(1.7rem, 3vw, 2.6rem); margin:0; letter-spacing:0; }}
    .section-head p {{ margin:0; color:var(--muted); max-width:62ch; line-height:1.55; }}
    .grid {{ display:grid; gap:16px; }}
    .metrics {{ grid-template-columns: repeat(4, minmax(0,1fr)); }}
    .metric-card, .panel, .gallery-card {{
      border:1px solid var(--line); background:linear-gradient(180deg, rgba(11,27,43,.92), rgba(7,20,33,.86));
      border-radius:20px; box-shadow:var(--shadow);
    }}
    .metric-card {{ padding:18px; min-height:142px; }}
    .metric-card span {{ color:var(--muted); font-size:.84rem; text-transform:uppercase; letter-spacing:.08em; }}
    .metric-card strong {{ display:block; font-size:clamp(1.6rem, 4vw, 2.7rem); margin:10px 0 8px; }}
    .metric-card p {{ color:#b4ccdd; margin:0; line-height:1.45; }}
    .panels {{ grid-template-columns: 1.2fr .8fr; }}
    .panel {{ padding:22px; overflow:hidden; }}
    .panel h3 {{ margin:0 0 14px; font-size:1.22rem; }}
    .metric-table {{ width:100%; border-collapse:collapse; }}
    .metric-table th,.metric-table td {{ border-bottom:1px solid rgba(157,184,207,.16); padding:10px 8px; text-align:left; }}
    .metric-table th {{ color:#b8d4e8; font-weight:700; }}
    .competition-table {{ width:100%; border-collapse:collapse; font-size:.92rem; }}
    .competition-table th,.competition-table td {{ border-bottom:1px solid rgba(157,184,207,.16); padding:12px 10px; text-align:left; vertical-align:top; }}
    .competition-table thead th {{ color:#9fdcff; text-transform:uppercase; letter-spacing:.07em; font-size:.76rem; }}
    .competition-table tbody th {{ color:#e9fbff; min-width:210px; }}
    .competition-table .score-cell {{ color:var(--gold); font-weight:900; white-space:nowrap; }}
    .competition-table .toricgt-row {{ background:linear-gradient(90deg, rgba(70,231,255,.16), rgba(136,255,134,.08)); }}
    .competition-note {{ margin:10px 0 0; color:var(--muted); line-height:1.5; }}
    .loss-list {{ list-style:none; padding:0; margin:0; display:grid; gap:10px; }}
    .loss-list li {{ border:1px solid rgba(70,231,255,.18); border-radius:14px; padding:12px; background:rgba(3,8,17,.34); }}
    .loss-list span {{ color:var(--cyan); font-weight:800; }}
    .loss-list strong {{ float:right; color:var(--gold); }}
    .loss-list p {{ clear:both; margin:7px 0 0; color:#b7cce0; line-height:1.45; }}
    .chart {{ height:260px; display:flex; align-items:flex-end; gap:10px; padding:34px 6px 6px; border-bottom:1px solid rgba(157,184,207,.25); }}
    .bar {{ flex:1; min-width:24px; border-radius:10px 10px 0 0; background:linear-gradient(180deg, var(--cyan), var(--violet)); position:relative; }}
    .bar.best {{ background:linear-gradient(180deg, var(--green), var(--gold)); }}
    .bar span {{ position:absolute; inset:auto 0 calc(100% + 8px); text-align:center; font-size:.75rem; color:#dff9ff; }}
    .chart-note {{ color:var(--muted); font-size:.9rem; margin:10px 0 0; }}
    .gallery {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
    .gallery-card {{ margin:0; overflow:hidden; }}
    .gallery-card img {{ display:block; width:100%; aspect-ratio: 16 / 9; object-fit:cover; background:#020711; }}
    .gallery-card figcaption {{ padding:14px 16px 16px; display:grid; gap:5px; }}
    .gallery-card figcaption span {{ color:var(--muted); line-height:1.45; }}
    .interactive-grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
    .interactive-card {{
      border:1px solid var(--line); background:linear-gradient(180deg, rgba(11,27,43,.92), rgba(7,20,33,.86));
      border-radius:20px; padding:18px; display:flex; gap:16px; justify-content:space-between; align-items:flex-start; min-height:190px;
    }}
    .interactive-card h3 {{ margin:0 0 10px; }}
    .interactive-card p {{ color:#b8d1e2; line-height:1.5; margin:0 0 12px; }}
    .interactive-card span {{ color:var(--gold); font-size:.86rem; }}
    .interactive-card .button {{ white-space:nowrap; }}
    .interactive-embed iframe {{ width:100%; height:min(72vh, 760px); border:1px solid rgba(70,231,255,.18); border-radius:14px; background:#020713; }}
    .concepts {{ grid-template-columns: repeat(4, minmax(0,1fr)); }}
    .concept {{ padding:18px; border:1px solid rgba(70,231,255,.22); background:rgba(7,20,33,.72); border-radius:18px; min-height:220px; position:relative; overflow:hidden; }}
    .concept h3 {{ margin:0 0 10px; }}
    .concept p {{ color:#b8d1e2; line-height:1.5; }}
    .mini-svg {{ height:82px; margin-top:12px; }}
    .moving-dot {{ animation: glide 3.2s ease-in-out infinite alternate; }}
    @keyframes glide {{ from {{ transform:translateX(0); }} to {{ transform:translateX(64px); }} }}
    footer {{ padding: 44px 0 56px; color:var(--muted); border-top:1px solid var(--line); margin-top:42px; }}
    @media (max-width: 860px) {{
      .hero, .panels {{ grid-template-columns:1fr; }}
      .metrics, .concepts, .gallery {{ grid-template-columns:1fr; }}
      .interactive-grid {{ grid-template-columns:1fr; }}
      .interactive-card {{ flex-direction:column; }}
      nav {{ align-items:flex-start; flex-direction:column; padding:18px 0; }}
      .hero-visual {{ min-height: 320px; }}
    }}
  </style>
</head>
<body>
  <header>
    <nav class="wrap">
      <a class="brand" href="#">ToricGT</a>
      <div class="links">
        <a href="{REPO_URL}">Code</a>
        <a href="{HF_URL}">Hugging Face</a>
        <a href="{PARAMETER_GOLF_URL}">Parameter Golf</a>
        <a href="{html.escape(pr_href)}">Submission PR</a>
        <a href="{LONG_PAPER_URL}">Long Paper</a>
        <a href="{CONDENSED_PAPER_URL}">Condensed Paper</a>
      </div>
    </nav>
  </header>

  <main>
    <section class="hero wrap">
      <div>
        <div class="eyebrow">Graph-token reasoning under a BPB artifact budget</div>
        <h1>ToricGT</h1>
        <p class="lead">ToricGT embeds tropical ring attention into toric geometry, then uses graph tokens, Forest-of-Thought search, persistent homology, Toric BGG certificates, vector-bundle one-dimensional-cone sheaf audits, and BPB-first training to pressure compact language models toward structured reasoning.</p>
        <div class="hero-actions">
          <a class="button primary" href="{REPO_URL}">Explore the codebase</a>
          <a class="button" href="{HF_URL}">Best checkpoints</a>
          <a class="button" href="{html.escape(pr_href)}">Parameter Golf PR: {html.escape(pr_label)}</a>
        </div>
      </div>
      <div class="hero-visual" aria-label="Animated toric fan visualization">
        {f'<img loading="eager" decoding="async" src="{html.escape(hero)}" alt="ToricGT visual backdrop">' if hero else ''}
        <div class="fan">
          <svg viewBox="0 0 520 380" role="img" aria-label="Tropical active faces embedded in a toric fan">
            <defs>
              <linearGradient id="g" x1="0" x2="1"><stop offset="0%" stop-color="#46e7ff"/><stop offset="100%" stop-color="#ffd166"/></linearGradient>
            </defs>
            <g transform="translate(260 190)">
              <circle r="128" fill="rgba(70,231,255,.05)" stroke="rgba(70,231,255,.25)"/>
              <g stroke="rgba(236,251,255,.55)" stroke-width="2">
                <line class="ray" x1="0" y1="0" x2="178" y2="-84"/>
                <line class="ray" x1="0" y1="0" x2="124" y2="134"/>
                <line class="ray" x1="0" y1="0" x2="-168" y2="92"/>
                <line class="ray" x1="0" y1="0" x2="-120" y2="-142"/>
                <line class="ray" x1="0" y1="0" x2="18" y2="-188"/>
              </g>
              <polygon class="poly" points="-88,-64 8,-128 112,-36 92,92 -34,126 -126,24" fill="rgba(70,231,255,.16)" stroke="url(#g)" stroke-width="4"/>
              <path d="M -110 70 C -48 -20, 60 118, 132 -50" fill="none" stroke="#ff5fa2" stroke-width="5" stroke-linecap="round"/>
              <circle class="moving-dot" cx="-108" cy="70" r="8" fill="#88ff86"/>
            </g>
          </svg>
        </div>
      </div>
    </section>

    <section class="wrap">
      <div class="section-head">
        <h2>Best Completed Run</h2>
      </div>
      <div class="grid metrics">
        {card("Best int8 BPB", fmt(best_metrics.get("final_int8_bpb")), "final int8+zlib round-trip BPB")}
        {card("Validation BPB", fmt(best_metrics.get("val_bpb")), "score on the OAI FineWeb validation stream")}
        {card("Train BPB", fmt(best_metrics.get("train_bpb")), "latest train-side BPB in the selected run")}
        {card("Compressed bytes", fmt(best_metrics.get("artifact_bytes"), 0), "model artifact before code/dependency accounting")}
      </div>
    </section>

    <section class="wrap grid panels">
      <article class="panel">
        <h3>Campaign BPB Trace</h3>
        <div id="chart" class="chart" data-history='{html.escape(history)}'></div>
        <p class="chart-note">Lower bars are better: BPB is a cost, so the best run is highlighted in green and should sit closest to the baseline.</p>
        <p class="lead">Campaign <code>{html.escape(campaign_id)}</code> restarts from step 0 each attempt. Current best profile: <code>{html.escape(best_profile)}</code>. Best run id: <code>{html.escape(best_run_id)}</code>.</p>
      </article>
      <article class="panel">
        <h3>Core Metrics</h3>
        <table class="metric-table">
          <tbody>
          {table_rows(best_metrics, ["final_int8_loss", "final_int8_bpb", "val_loss", "val_bpb", "train_loss", "train_bpb", "graph_lm_bpb", "checkpoint_step"])}
          </tbody>
        </table>
      </article>
    </section>

    <section class="wrap">
      <div class="section-head">
        <h2>Competition Context</h2>
        <p>Lower BPB is better. ToricGT is shown with the current local best exported run, followed by the official top five entries listed in the local OpenAI Parameter Golf clone.</p>
      </div>
      <article class="panel">
        <table class="competition-table">
          <thead>
            <tr>
              <th>Status</th>
              <th>Run</th>
              <th>BPB</th>
              <th>Author</th>
              <th>Summary</th>
              <th>Date</th>
              <th>Info</th>
            </tr>
          </thead>
          <tbody>
            {competition_rows}
          </tbody>
        </table>
        <p class="competition-note">The ToricGT row is not claiming accepted leaderboard status. It is a pending experimental/non-record candidate until the Parameter Golf PR, tokenizer accounting, reproducibility checks, and official review are complete.</p>
      </article>
    </section>

    <section class="wrap">
      <div class="section-head">
        <h2>What Is Being Optimized</h2>
        <p>BPB stays primary. The advanced losses are small, evidence-weighted pressures that shape hidden graph structure, retrieval, toric/tropical geometry, and algebraic consistency without letting those objectives dominate byte likelihood.</p>
      </div>
      <article class="panel">
        <ul class="loss-list">
          {''.join(loss_items)}
        </ul>
      </article>
    </section>

    <section class="wrap">
      <div class="section-head">
        <h2>Conceptual Helpers</h2>
        <p>These small animated sketches summarize the operational math: tropical dynamic programming, toric embedding, reasoning forests, and persistence-vector retrieval.</p>
      </div>
      <div class="grid concepts">
        <div class="concept">
          <h3>Tropical Path</h3>
          <p>ConvexTok and tropical attention both expose min-plus or max-plus active paths with margins.</p>
          <svg class="mini-svg" viewBox="0 0 180 80"><path d="M10 62 L55 22 L100 52 L158 16" fill="none" stroke="#46e7ff" stroke-width="4"/><circle class="moving-dot" cx="20" cy="56" r="6" fill="#ffd166"/></svg>
        </div>
        <div class="concept">
          <h3>Toric Fan</h3>
          <p>Active tropical faces are embedded into toric charts so cone, divisor, and one-dimensional-cone audits become meaningful.</p>
          <svg class="mini-svg" viewBox="0 0 180 80"><g transform="translate(90 42)" stroke="#88ff86" stroke-width="2"><line class="ray" x1="0" y1="0" x2="70" y2="-22"/><line class="ray" x1="0" y1="0" x2="-60" y2="-35"/><line class="ray" x1="0" y1="0" x2="-52" y2="34"/><line class="ray" x1="0" y1="0" x2="50" y2="32"/></g></svg>
        </div>
        <div class="concept">
          <h3>Forest Search</h3>
          <p>Embedding-space FoT/GFlowNet branches are rewarded by BPB-delta and structural diversity.</p>
          <svg class="mini-svg" viewBox="0 0 180 80"><g stroke="#ff5fa2" stroke-width="3" fill="none"><path d="M20 65 C50 55 55 28 88 18"/><path d="M20 65 C56 60 80 62 126 46"/><path d="M88 18 C118 20 136 16 160 8"/><path d="M88 18 C114 34 132 36 160 30"/></g></svg>
        </div>
        <div class="concept">
          <h3>PH Retrieval</h3>
          <p>Analogies require simplex-map evidence and vectorized persistent-homology agreement.</p>
          <svg class="mini-svg" viewBox="0 0 180 80"><rect x="18" y="18" width="44" height="44" fill="rgba(70,231,255,.22)" stroke="#46e7ff"/><rect x="118" y="18" width="44" height="44" fill="rgba(255,209,102,.22)" stroke="#ffd166"/><path d="M66 40 L112 40" stroke="#88ff86" stroke-width="4" stroke-dasharray="6 7"/></svg>
        </div>
      </div>
    </section>

    <section class="wrap">
      <div class="section-head">
        <h2>Visualizations</h2>
      </div>
      <div class="grid gallery">
        {''.join(gallery)}
      </div>
    </section>

    <section class="wrap">
      <div class="section-head">
        <h2>Interactive Analysis Lab</h2>
        <p>The generated reports use compact native Canvas/SVG interactivity wherever it matters: 3D campaign tetrahedra, 3D reasoning trajectories, filtered reasoning-step simplicial complexes, CAS links, simplex-map summaries, vectorized PH panels, and raw metric tables without forcing giant plotting payloads on first load.</p>
      </div>
      <div class="grid interactive-grid">
        {''.join(interactive_cards)}
      </div>
    </section>

    {''.join(embedded_reports)}

    <section class="wrap grid panels">
      <article class="panel">
        <h3>Central Construction</h3>
        <p class="lead">ToricGT treats graph-structured data as graph-in/graph-out by default, but gives OAI FineWeb an optional BPB-only flattening path. Tokenization DAGs are graphified, edge tokens are first-class, and tropical dynamic-programming paths are embedded into toric varieties so the model can audit active faces using fans, cones, one-dimensional cones, divisors, sheaf/vector-bundle signals, and commutative-algebra certificates.</p>
      </article>
      <article class="panel">
        <h3>Links</h3>
        <table class="metric-table">
          <tbody>
            <tr><th>Code</th><td><a href="{REPO_URL}">{REPO_URL}</a></td></tr>
            <tr><th>Hugging Face</th><td><a href="{HF_URL}">{HF_URL}</a></td></tr>
            <tr><th>OpenAI Parameter Golf</th><td><a href="{PARAMETER_GOLF_URL}">{PARAMETER_GOLF_URL}</a></td></tr>
            <tr><th>Submission PR</th><td><a href="{html.escape(pr_href)}">{html.escape(pr_label)}</a></td></tr>
            <tr><th>Long Paper</th><td><a href="{LONG_PAPER_URL}">assets/toricgt_toric_bgg_rewrite_amelie_schreiber.tex</a></td></tr>
            <tr><th>Condensed Paper</th><td><a href="{CONDENSED_PAPER_URL}">assets/toricgt_neurips_condensed.tex</a></td></tr>
          </tbody>
        </table>
      </article>
    </section>

    <section class="wrap">
      <div class="section-head">
        <h2>Next Steps</h2>
        <p>The next phase turns ToricGT from a compact BPB competitor into a general graph-to-graph reasoning substrate for scientific and multimodal agents.</p>
      </div>
      <article class="panel">
        <h3>Continuous Structure Tokenization</h3>
        <p class="lead">ConvexTok suggests a practical recipe: build a candidate graph, solve or relax a global path/cover objective, then round into a compact token inventory. The same idea can be lifted from byte strings to continuous 3D structures by replacing byte-boundary edges with geometric motifs: protein backbone fragments, residue-neighborhood contact patches, ligand pharmacophore neighborhoods, mesh cells, point-cloud patches, and local dynamical states. The tokenization objective should remain tropical-toric: dynamic programming supplies active paths, LP or optimal-transport relaxations supply lower bounds and regret metrics, and toric charts organize the active motif complex into cones, divisors, and sheaf-compatible local neighborhoods.</p>
        <h3>ToricBLM Continuation</h3>
        <p class="lead">The ToricBLM direction builds on ToricGT by continuing training toward a universal-modality biomedical reasoning model: sequences, atom graphs, protein contact graphs, RNA/DNA structures, complexes, assay facts, trajectories, and scientific claims all become typed graphs with optional continuous coordinates. The mathematical foundation is the bounded-domain universal equivariant graph-to-graph approximation theorem: within fixed graph budgets, a TokenGT-style model can approximate continuous equivariant graph maps, while tropical heads add dynamic-programming and Boolean-circuit-like active support, and ConvexTok/TokenGT positional encodings expose graph grammar structure that ordinary token-only transformers must infer indirectly.</p>
        <h3>Reasoning, Memory, and Control</h3>
        <p class="lead">Embedding-space Forest-of-Thought and GFlowNet training make reasoning a controlled search process, not a single chain. GraphCG full-rank concept axes make the search directions inspectable. Persistent homology and Toric BGG certificates decide when a retrieved memory is an analogy rather than a superficial nearest neighbor. The proposed thought-alcove control layer then treats reasoning trajectories as dynamical systems inside tropical-toric cells: fan walls are decision boundaries, Lyapunov-style energies discourage unstable exits, ergodic averages diagnose repeated itinerary behavior, and optimal-control objectives choose interventions that keep the model inside productive alcoves while preserving BPB and task reward.</p>
        <h3>Pinched-Embedding Disambiguation</h3>
        <p class="lead">Jakubowski, Gašić, and Zibrowius argue that static word vectors are better modeled as a pinched manifold: a quotient of a meaning manifold in which several meaning sheets are identified at one word vector. Their diagnostic is operationally useful for ToricGT because our hidden states also identify many roles at one surface token, graph node, ConvexTok edge, or memory key. The proposed extension is to puncture the local neighborhood of an ambiguous token/edge/state, normalize the outgoing directions, compute degree-zero persistent homology with GUDHI, and use the resulting component structure as a sense, role, or analogy gate.</p>
        <table class="metric-table">
          <tbody>
            <tr><th>Finite theorem</th><td>If a local embedding neighborhood is the quotient of k disjoint meaning sheets glued at a target point z, then every sufficiently small punctured neighborhood, U with z removed, has k connected components. A Vietoris-Rips filtration sampled densely enough from that punctured neighborhood has k long-lived H0 branches, up to the usual stability error under Hausdorff perturbation.</td></tr>
            <tr><th>Proof sketch</th><td>Before quotienting, choose disjoint local balls around the k preimages of z. Removing z removes the shared glued point, so the images of those balls no longer touch. Thus the punctured quotient neighborhood splits into k components. Persistent H0 detects the merge radii of sampled components; stability of persistence diagrams bounds the change when hidden vectors or approximate nearest neighbors move slightly.</td></tr>
            <tr><th>ToricGT implementation</th><td>For a target token, graph vertex, graph edge, ConvexTok DAG edge, or retrieved memory state, gather k nearest original embedding vectors, remove the target, map neighbors to (v-z)/||v-z||, compute H0 persistence and vectorized summaries, cluster components, then train a small disambiguation head only when the PH gate is confident and BPB loss is not harmed.</td></tr>
            <tr><th>Training losses</th><td>Use a BPB-gated component cross-entropy, a component-margin loss between selected and nonselected local sheets, a PH landscape/image similarity term for analogical retrieval, and a toric fan-refinement term that aligns component boundaries with tropical active-face walls. Keep all weights adaptive, uncertainty-controlled, and subordinate to byte likelihood.</td></tr>
            <tr><th>Gotchas</th><td>Neighborhood size matters: too small misses senses, too large measures global density. Contextual hidden states may already split some senses, so the gate should detect residual pinches rather than force artificial clusters. Comparisons must use original hidden coordinates, not PCA. Synonymy is not the same obstruction as polysemy, and PH pseudo-labels must not leak validation text or override BPB.</td></tr>
          </tbody>
        </table>
        <p class="lead">This gives a rigorous path from topological word-sense induction to ToricGT: ConvexTok supplies tokenization DAG neighborhoods; TokenGT supplies equivariant vertex/edge neighborhoods; tropical attention supplies active dynamic-programming sheets; toric charts turn those sheets into cones, walls, divisors, and one-dimensional-cone audits; GraphCG supplies interpretable coordinates; analogical memory requires simplex-map and PH agreement inside the same component; and Toric BGG certificates prevent standard-object leakage across incompatible local sheets. The expected benefit is fewer sense-collisions during retrieval and graph flattening, cleaner memory selection, and lower BPB when ambiguous token choices were causing early likelihood spikes.</p>
        <h3>Thought Fluid Dynamics</h3>
        <p class="lead">The Tropical Quivers of Archs program frames learned graph-to-graph functions as composable operators with tropical/polyhedral local charts. For ToricGT, that suggests a fluid view of reasoning: hidden trajectories have divergence, circulation, vorticity, boundary flux, and energy. Navier-Stokes-inspired regularizers should be used conservatively, as audits and small penalties, but they give a precise vocabulary for branch merging, turbulence in unstable reasoning zones, and dissipative correction when FoT branches drift. The long-run objective is optimal control over a learned, graph-valued, tropical-toric dynamical system.</p>
        <p><a href="https://arxiv.org/abs/2011.09413">Topology of Word Embeddings</a> · <a href="https://huggingface.co/blog/AmelieSchreiber/toricblm">ToricBLM blog</a> · <a href="https://github.com/amelie-iska/Tropical_Quivers_of_Archs/blob/main/tropical_quiver_research_program.tex">Tropical Quivers of Archs</a> · <a href="https://zitniklab.hms.harvard.edu/projects/GeoBPE/">GeoBPE reference</a> · <a href="https://openreview.net/forum?id=o4ANDWaomX">protein structure tokenization benchmark</a></p>
      </article>
    </section>
  </main>

  <footer>
    <div class="wrap">
      <p>ToricGT page generated from local experiment artifacts. Competition scores should be interpreted with the record/non-record status and tokenizer/BPB validation caveats documented in the associated Parameter Golf submission.</p>
    </div>
  </footer>
  <script>
    const chart = document.getElementById('chart');
    const history = JSON.parse(chart.dataset.history || '[]').filter(d => Number.isFinite(d.bpb));
    if (history.length) {{
      const best = Math.min(...history.map(d => d.bpb));
      const worst = Math.max(...history.map(d => d.bpb));
      for (const [idx, d] of history.entries()) {{
        const bar = document.createElement('div');
        const t = worst === best ? 0 : (d.bpb - best) / (worst - best);
        bar.className = 'bar' + (d.bpb === best ? ' best' : '');
        bar.style.height = `${{Math.max(26, 42 + t * 182)}}px`;
        bar.title = `${{idx + 1}} · ${{d.profile || 'profile'}} · BPB ${{d.bpb.toFixed(4)}} · lower is better`;
        const label = document.createElement('span');
        label.textContent = d.bpb.toFixed(3);
        bar.appendChild(label);
        chart.appendChild(bar);
      }}
    }} else {{
      chart.textContent = 'No completed run metrics found yet.';
    }}
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--campaign-state")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--parameter-golf-pr-url", default="")
    parser.add_argument("--analysis-output-dir", default="")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    docs = Path(args.docs_dir)
    if not docs.is_absolute():
        docs = repo / docs
    docs.mkdir(parents=True, exist_ok=True)

    state_path = find_campaign_state(repo, args.campaign_state)
    state = load_json(state_path) if state_path else {}
    pr_url = find_pr_url(repo, args.parameter_golf_pr_url)
    analysis_dir = find_analysis_output(repo, args.analysis_output_dir)
    copied = copy_assets(repo, docs, analysis_dir=analysis_dir)
    interactive = copy_interactive_reports(repo, docs, analysis_dir=analysis_dir)
    rows = completed_rows(state, current_campaign_only=True)
    best = best_row(rows)
    best_metrics = best.get("metrics", {}) if best else {}
    interactive = {**interactive, **write_method_reports(docs, copied, best_metrics)}
    tetra = write_campaign_tetrahedron_report(docs, rows)
    if tetra:
        interactive = {"campaign_tetrahedron": tetra, **interactive}
    html_text = build_html(repo, state_path, state, pr_url, copied, interactive)
    html_text = "\n".join(line.rstrip() for line in html_text.splitlines()) + "\n"
    (docs / "index.html").write_text(html_text, encoding="utf-8")
    (docs / ".nojekyll").write_text("", encoding="utf-8")
    manifest = {
        "generated_utc": utc_now(),
        "campaign_state": (
            str(state_path.resolve().relative_to(repo.resolve()))
            if state_path and state_path.resolve().is_relative_to(repo.resolve())
            else (state_path.name if state_path else None)
        ),
        "parameter_golf_pr_url": pr_url or None,
        "analysis_output_dir": (
            str(analysis_dir.resolve().relative_to(repo.resolve()))
            if analysis_dir and analysis_dir.resolve().is_relative_to(repo.resolve())
            else (str(analysis_dir) if analysis_dir else None)
        ),
        "copied_assets": copied,
        "interactive_reports": interactive,
    }
    (docs / "page_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {docs / 'index.html'}")
    print(f"copied {len(copied)} assets")
    print(f"copied {len(interactive)} interactive reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
