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
    "trajectory": "outputs/branching_reasoning_trajectory_final_long_20260616T171742Z/static_screenshots/full_trajectory_filtered_complex.png",
    "analogy": "outputs/branching_reasoning_trajectory_final_long_20260616T171742Z/static_screenshots/analogical_memory_simplex_tree_map.png",
    "vectorized_ph": "outputs/branching_reasoning_trajectory_final_long_20260616T171742Z/static_screenshots/vectorized_ph_features.png",
    "cas": "outputs/smoke_oai_full_iteration_analysis_2/html_screenshots/screenshots/embedding_cas_sidecar__index__slice_00.png",
    "toric_embedding": "outputs/smoke_oai_full_iteration_analysis_2/html_screenshots/screenshots/toric_embedding_report__index__slice_00.png",
    "bgg": "outputs/smoke_oai_full_iteration_analysis_2/html_screenshots/screenshots/bgg_category_o_report__index__slice_00.png",
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


def completed_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in state.get("history", []):
        if not isinstance(row, dict):
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


def copy_assets(repo: Path, docs: Path) -> dict[str, str]:
    assets_dir = docs / "page_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for key, rel in IMAGE_SOURCES.items():
        src = repo / rel
        if not src.exists():
            continue
        dst = assets_dir / f"{key}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        copied[key] = f"page_assets/{dst.name}"
    return copied


def table_rows(metrics: dict[str, Any], keys: list[str]) -> str:
    rows = []
    for key in keys:
        rows.append(
            f"<tr><th>{html.escape(key)}</th><td>{fmt(metrics.get(key), 6 if 'loss' in key else 4)}</td></tr>"
        )
    return "\n".join(rows)


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


def card(title: str, value: str, note: str) -> str:
    return f"""
    <article class="metric-card">
      <span>{html.escape(title)}</span>
      <strong>{html.escape(value)}</strong>
      <p>{html.escape(note)}</p>
    </article>
    """


def build_html(repo: Path, state_path: Path | None, state: dict[str, Any], pr_url: str, copied: dict[str, str]) -> str:
    rows = completed_rows(state)
    best = best_row(rows)
    best_metrics = best.get("metrics", {}) if best else {}
    best_profile = str(best.get("profile", "pending")) if best else "pending"
    best_run_id = str(best.get("run_id", "pending")) if best else "pending"
    campaign_id = str(state.get("campaign_id", state_path.parent.name if state_path else "pending"))
    if state_path:
        try:
            state_label = str(state_path.resolve().relative_to(repo.resolve()))
        except ValueError:
            state_label = state_path.name
    else:
        state_label = "no campaign state found"
    pr_label = pr_url or "pending after best-of-10 + 900-step candidate"
    pr_href = pr_url or PARAMETER_GOLF_URL
    hero = copied.get("logo") or copied.get("architecture") or ""
    history = run_history_json(rows)
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
                  <img src="{html.escape(copied[key])}" alt="{html.escape(title)}">
                  <figcaption><strong>{html.escape(title)}</strong><span>{html.escape(caption)}</span></figcaption>
                </figure>
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
    .loss-list {{ list-style:none; padding:0; margin:0; display:grid; gap:10px; }}
    .loss-list li {{ border:1px solid rgba(70,231,255,.18); border-radius:14px; padding:12px; background:rgba(3,8,17,.34); }}
    .loss-list span {{ color:var(--cyan); font-weight:800; }}
    .loss-list strong {{ float:right; color:var(--gold); }}
    .loss-list p {{ clear:both; margin:7px 0 0; color:#b7cce0; line-height:1.45; }}
    .chart {{ height:260px; display:flex; align-items:flex-end; gap:10px; padding:18px 6px 6px; border-bottom:1px solid rgba(157,184,207,.25); }}
    .bar {{ flex:1; min-width:24px; border-radius:10px 10px 0 0; background:linear-gradient(180deg, var(--cyan), var(--violet)); position:relative; }}
    .bar.best {{ background:linear-gradient(180deg, var(--green), var(--gold)); }}
    .bar span {{ position:absolute; inset:auto 0 calc(100% + 8px); text-align:center; font-size:.75rem; color:#dff9ff; }}
    .gallery {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
    .gallery-card {{ margin:0; overflow:hidden; }}
    .gallery-card img {{ display:block; width:100%; aspect-ratio: 16 / 9; object-fit:cover; background:#020711; }}
    .gallery-card figcaption {{ padding:14px 16px 16px; display:grid; gap:5px; }}
    .gallery-card figcaption span {{ color:var(--muted); line-height:1.45; }}
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
        {f'<img src="{html.escape(hero)}" alt="ToricGT visual backdrop">' if hero else ''}
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
        <p>Generated {html.escape(utc_now())} from <code>{html.escape(state_label)}</code>. The page should be regenerated after the 10-run sweep, 900-step candidate, HF upload, and PR creation finish.</p>
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
        <h3>Campaign Trace</h3>
        <div id="chart" class="chart" data-history='{html.escape(history)}'></div>
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
        <h2>Visual Evidence</h2>
        <p>These are copied from the project’s generated diagrams and audit screenshots so the public page reflects real outputs from the codebase.</p>
      </div>
      <div class="grid gallery">
        {''.join(gallery)}
      </div>
    </section>

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
        const t = worst === best ? 1 : (worst - d.bpb) / (worst - best);
        bar.className = 'bar' + (d.bpb === best ? ' best' : '');
        bar.style.height = `${{Math.max(24, 54 + t * 180)}}px`;
        bar.title = `${{idx + 1}} · ${{d.profile || 'profile'}} · BPB ${{d.bpb.toFixed(4)}}`;
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
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    docs = Path(args.docs_dir)
    if not docs.is_absolute():
        docs = repo / docs
    docs.mkdir(parents=True, exist_ok=True)

    state_path = find_campaign_state(repo, args.campaign_state)
    state = load_json(state_path) if state_path else {}
    pr_url = find_pr_url(repo, args.parameter_golf_pr_url)
    copied = copy_assets(repo, docs)
    html_text = build_html(repo, state_path, state, pr_url, copied)
    (docs / "index.html").write_text(html_text, encoding="utf-8")
    (docs / ".nojekyll").write_text("", encoding="utf-8")
    manifest = {
        "generated_utc": utc_now(),
        "campaign_state": str(state_path) if state_path else None,
        "parameter_golf_pr_url": pr_url or None,
        "copied_assets": copied,
    }
    (docs / "page_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {docs / 'index.html'}")
    print(f"copied {len(copied)} assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
