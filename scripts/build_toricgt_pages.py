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
        "entry": "branching_reasoning_trajectory.html",
        "title": "Interactive Simplex Trajectory",
        "summary": (
            "Full graph-of-thought reasoning trajectory with 3D PCA display coordinates, "
            "radius and reasoning-level sliders, token hover/click panels, one-dimensional "
            "simplex edges, optional filled 2-simplices, and analogy gates."
        ),
        "features": "radius slider · reasoning-level slider · token hover/click · 3D PCA · simplex maps",
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
        copied[key] = {
            "href": f"page_interactive/{key}/{entry}",
            "title": str(spec["title"]),
            "summary": str(spec["summary"]),
            "features": str(spec["features"]),
            "embed": "1" if spec.get("embed") else "0",
        }
    scrub_public_report_paths(out_dir, repo)
    return copied


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
        points.append({**record, "weights": bary, "xyz": xyz})
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
        points.append({**record, "weights": bary, "xy": xy, "balance": balance})
    grid: list[dict[str, Any]] = []
    resolution = 32
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
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Campaign Metric Tetrahedra</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root{{color-scheme:dark;--bg:#030712;--panel:#071421;--line:rgba(70,231,255,.28);--text:#ecfbff;--muted:#9db8cf;--cyan:#46e7ff;--gold:#ffd166}}
body{{margin:0;background:radial-gradient(circle at 10% 10%,rgba(70,231,255,.16),transparent 30rem),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1280px;margin:0 auto;padding:24px}}.panel{{border:1px solid var(--line);border-radius:10px;background:rgba(7,20,33,.86);padding:16px;margin:14px 0}}
h1{{margin:0 0 8px}}p{{color:var(--muted);line-height:1.5}}.plot{{height:620px;border:1px solid rgba(70,231,255,.16);border-radius:10px;background:#020713;margin-top:12px}}
.plot.triangle{{height:500px}}.view-controls{{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:end;margin:8px 0 12px}}select{{width:100%;background:#020713;color:var(--text);border:1px solid rgba(70,231,255,.28);border-radius:8px;padding:10px}}label{{display:block;color:var(--muted);font-size:.9rem;margin-bottom:4px}}.metric-note{{font-size:.92rem;color:var(--muted);margin-top:8px}}code{{color:#7df5ff}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid rgba(157,184,207,.16);padding:8px;text-align:left}}td.num{{text-align:right;color:var(--gold)}}
</style></head><body><main>
<section class="panel"><h1>Campaign Metric Tetrahedra and Shaded Triangles</h1>
<p>This page renders interactive tetrahedra and shaded metric triangles over completed campaign runs without mounting every Plotly view at once. Each tetrahedron uses four normalized axes. Each triangle uses three normalized axes; the blue background shading marks regions where the three metrics are jointly balanced, while run markers are colored by BPB and labeled by run index. Lower BPB is always better.</p>
<p><a href="campaign_tetrahedron.json">payload JSON</a></p></section>
<section class="panel"><h2>Tetrahedron Views</h2><div class="view-controls"><div><label for="tetra_select">metric tetrahedron</label><select id="tetra_select"><option>loading...</option></select></div><span id="tetra_count" class="metric-note"></span></div><div id="tetra_plot" class="plot"></div><p id="tetra_caption" class="metric-note"></p></section>
<section class="panel"><h2>Shaded Triangle Views</h2><p>Blue shading indicates the balanced-support region for the metric triple. Markers remain actual run observations; hover text reports profile, BPB, raw axis values, and barycentric weights.</p><div class="view-controls"><div><label for="triangle_select">metric triangle</label><select id="triangle_select"><option>loading...</option></select></div><span id="triangle_count" class="metric-note"></span></div><div id="triangle_plot" class="plot triangle"></div><p id="triangle_caption" class="metric-note"></p></section>
<section class="panel"><h2>Run Table</h2><table><thead><tr><th>profile</th><th>run</th><th>BPB</th><th>artifact bytes</th><th>train BPB</th></tr></thead><tbody>
{''.join(f"<tr><td>{html.escape(p['profile'])}</td><td>{html.escape(p['run'])}</td><td class='num'>{p['bpb']:.6f}</td><td class='num'>{fmt(p.get('artifact'),0)}</td><td class='num'>{fmt(p.get('train_bpb'))}</td></tr>" for p in records)}
</tbody></table></section>
<script>
let payload = null;
function axisRawText(point, axes) {{
  return axes.map(axis => payload.axis_library[axis].label+': '+(point.raw_values[axis] ?? 'missing')).join('<br>');
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
  const vertices = view.vertices;
  const edges = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]];
  const traces = [];
  for (const [i,j] of edges) {{
    traces.push({{type:'scatter3d', mode:'lines', x:[vertices[i][0],vertices[j][0]], y:[vertices[i][1],vertices[j][1]], z:[vertices[i][2],vertices[j][2]], line:{{color:'#46e7ff',width:5}}, hoverinfo:'skip', showlegend:false}});
  }}
  traces.push({{type:'mesh3d', x:vertices.map(v=>v[0]), y:vertices.map(v=>v[1]), z:vertices.map(v=>v[2]), i:[0,0,0,1], j:[1,1,2,2], k:[2,3,3,3], opacity:0.09, color:'#46e7ff', hoverinfo:'skip', name:'tetrahedron faces'}});
  traces.push({{type:'scatter3d', mode:'markers+text', name:'campaign runs', showlegend:false, x:view.points.map(p=>p.xyz[0]), y:view.points.map(p=>p.xyz[1]), z:view.points.map(p=>p.xyz[2]), text:view.points.map(p=>String(p.index)), customdata:view.points.map(p=>[p.profile,p.run,p.bpb,p.artifact,p.train_bpb,p.weights.map(w=>w.toFixed(3)).join(' / '),axisRawText(p, view.axes)]), marker:{{size:8,color:view.points.map(p=>p.bpb),colorscale:'Magma',reversescale:true,colorbar:{{title:'BPB'}},line:{{color:'white',width:1}}}}, hovertemplate:'%{{customdata[0]}}<br>run %{{customdata[1]}}<br>BPB %{{customdata[2]:.6f}}<br>artifact %{{customdata[3]}}<br>train BPB %{{customdata[4]}}<br>weights %{{customdata[5]}}<br>%{{customdata[6]}}<extra></extra>'}});
  traces.push({{type:'scatter3d', mode:'text', x:vertices.map(v=>v[0]), y:vertices.map(v=>v[1]), z:vertices.map(v=>v[2]+0.08), text:view.labels, textfont:{{color:'#ecfbff',size:14}}, hoverinfo:'skip', showlegend:false}});
  Plotly.react('tetra_plot', traces, {{template:'plotly_dark', title:{{text:view.title,font:{{color:'#ecfbff'}}}}, paper_bgcolor:'#020713', plot_bgcolor:'#020713', scene:{{aspectmode:'data', camera:{{eye:{{x:1.55,y:1.35,z:1.15}}}}, xaxis:{{visible:false}}, yaxis:{{visible:false}}, zaxis:{{visible:false}}, bgcolor:'#020713'}}, margin:{{l:0,r:0,t:42,b:0}}}}, {{responsive:true}});
  document.getElementById('tetra_caption').innerHTML = 'Axes: '+view.axes.map(axis => payload.axis_library[axis].label).join(' · ');
}}
function renderTriangle(view) {{
  const boundary = [...view.vertices, view.vertices[0]];
  const traces = [
    {{type:'scatter', mode:'markers', x:view.grid.map(g=>g.x), y:view.grid.map(g=>g.y), marker:{{size:9,color:view.grid.map(g=>g.shade),colorscale:'Blues',opacity:0.68,colorbar:{{title:'blue balance'}}}}, hoverinfo:'skip', showlegend:false}},
    {{type:'scatter', mode:'lines', x:boundary.map(v=>v[0]), y:boundary.map(v=>v[1]), line:{{color:'#46e7ff',width:3}}, hoverinfo:'skip', showlegend:false}},
    {{type:'scatter', mode:'markers+text', name:'campaign runs', showlegend:false, x:view.points.map(p=>p.xy[0]), y:view.points.map(p=>p.xy[1]), text:view.points.map(p=>String(p.index)), textposition:'top center', customdata:view.points.map(p=>[p.profile,p.run,p.bpb,p.weights.map(w=>w.toFixed(3)).join(' / '),p.balance.toFixed(3),axisRawText(p, view.axes)]), marker:{{size:13,color:view.points.map(p=>p.bpb),colorscale:'Magma',reversescale:true,line:{{color:'#ecfbff',width:1}},colorbar:{{title:'BPB'}}}}, hovertemplate:'%{{customdata[0]}}<br>run %{{customdata[1]}}<br>BPB %{{customdata[2]:.6f}}<br>weights %{{customdata[3]}}<br>balance %{{customdata[4]}}<br>%{{customdata[5]}}<extra></extra>'}},
    {{type:'scatter', mode:'text', x:view.vertices.map(v=>v[0]), y:view.vertices.map(v=>v[1]), text:view.labels, textfont:{{color:'#ecfbff',size:13}}, hoverinfo:'skip', showlegend:false}},
  ];
  Plotly.react('triangle_plot', traces, {{template:'plotly_dark', title:{{text:view.title+' · '+view.shade,font:{{color:'#ecfbff'}}}}, paper_bgcolor:'#020713', plot_bgcolor:'#020713', xaxis:{{visible:false,range:[0,1]}}, yaxis:{{visible:false,range:[0,1]}}, margin:{{l:12,r:12,t:42,b:12}}}}, {{responsive:true}});
  document.getElementById('triangle_caption').innerHTML = 'Axes: '+view.axes.map(axis => payload.axis_library[axis].label).join(' · ')+' · shade: '+view.shade;
}}
function initCampaignViews(data) {{
  payload = data;
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
h1{font-size:clamp(2.1rem,5vw,4.5rem);line-height:.95;margin:0 0 12px;letter-spacing:0}h2{font-size:clamp(1.35rem,3vw,2.2rem);margin:0 0 12px}h3{margin:0 0 8px}.lead,p,li{color:#bdd6e8;line-height:1.62}.eyebrow{color:var(--gold);text-transform:uppercase;letter-spacing:.15em;font-weight:800;font-size:.78rem}.equation{font-family:"STIX Two Text",Cambria,Georgia,serif;color:#fff;background:#020713;border:1px solid rgba(157,184,207,.18);border-radius:12px;padding:12px 14px;overflow:auto}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:6px 10px;margin:3px;background:rgba(70,231,255,.08);font-size:.9rem}.card{padding:16px;min-height:220px}svg{max-width:100%}.plot{height:560px;border:1px solid rgba(70,231,255,.18);border-radius:14px;background:#020713}.source-img{width:100%;max-height:360px;object-fit:contain;border:1px solid rgba(70,231,255,.18);border-radius:14px;background:#020713;padding:10px}.mini-table{width:100%;border-collapse:collapse}.mini-table th,.mini-table td{border-bottom:1px solid rgba(157,184,207,.16);padding:9px;text-align:left;vertical-align:top}.mini-table th{color:#9fdcff}.accent{color:var(--gold);font-weight:800}@media(max-width:900px){.two,.three{grid-template-columns:1fr}.plot{height:420px}}
"""


def method_report_shell(title: str, eyebrow: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>{METHOD_CSS}</style></head><body><main>
<section class="hero"><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1></section>
{body}
</main></body></html>
"""


def write_fot_method_report(docs: Path, copied: dict[str, str]) -> dict[str, str]:
    out = docs / "page_interactive" / "fot_method"
    out.mkdir(parents=True, exist_ok=True)
    image = "../../" + copied.get("fot_reference", "")
    payload = {
        "nodes": [
            {"id": "root", "tree": 0, "x": 0, "y": 0, "z": 0, "kind": "prompt"},
            {"id": "t0a", "tree": 0, "x": 1, "y": 1.2, "z": .2, "kind": "branch"},
            {"id": "t0b", "tree": 0, "x": 1, "y": -.9, "z": -.15, "kind": "branch"},
            {"id": "t1a", "tree": 1, "x": 2.1, "y": 1.7, "z": .7, "kind": "tree"},
            {"id": "t1b", "tree": 1, "x": 2.3, "y": .1, "z": -.25, "kind": "tree"},
            {"id": "t2a", "tree": 2, "x": 3.4, "y": -.4, "z": .6, "kind": "tree"},
            {"id": "vote", "tree": 3, "x": 4.6, "y": .25, "z": .1, "kind": "consensus"},
        ],
        "edges": [["root", "t0a"], ["root", "t0b"], ["t0a", "t1a"], ["t0a", "t1b"], ["t0b", "t2a"], ["t1a", "vote"], ["t1b", "vote"], ["t2a", "vote"]],
    }
    body = f"""
<section class="panel">
  <p class="lead">Forest-of-Thought (FoT) scales reasoning by running several thought trees, activating promising branches sparsely, correcting weak branches, and using consensus over the forest. In ToricGT the same idea is moved from prompt strings into embedding space: each branch is a graph-valued hidden trajectory, each edge is a graph edit or continuous displacement, and branch rewards are tied to BPB delta, verifier score, topological stability, and toric fan margins.</p>
  <p><a href="https://arxiv.org/abs/2412.09078">FoT arXiv:2412.09078</a> · <a href="https://github.com/iamhankai/Forest-of-Thought">upstream code</a> · <a href="https://huggingface.co/blog/AmelieSchreiber/toricblm">ToricBLM continuation</a></p>
</section>
<section class="grid two">
  <article class="panel"><h2>FoT Reference Diagram</h2>{f'<img class="source-img" src="{html.escape(image)}" alt="Forest-of-Thought paper diagram">' if image != "../../" else '<p>Local FoT paper image not found.</p>'}</article>
  <article class="panel"><h2>Embedding-Space Adaptation</h2><div id="fotPlot" class="plot"></div></article>
</section>
<section class="grid three">
  <article class="card"><h3>State</h3><p>Each state is <span class="accent">s=(G,H,Σ,μ,M)</span>: a reasoning graph, hidden table, toric/fan chart, moment coordinate, and memory pointer.</p></article>
  <article class="card"><h3>Action</h3><p>Actions add/merge thought nodes, move inside a normal cone, cross a wall, retrieve an analogy, or flatten a graph output for BPB scoring.</p></article>
  <article class="card"><h3>Reward</h3><p>Reward combines byte likelihood improvement with structural diversity, GUDHI persistent-homology stability, and toric/BGG certificate consistency.</p></article>
</section>
<section class="panel"><h2>Training Equation</h2><div class="equation">L<sub>FoT</sub> = (log Z + Σ<sub>t</sub> log p<sub>F</sub>(s<sub>t+1</sub>|s<sub>t</sub>) − log R(x) − Σ<sub>t</sub> log p<sub>B</sub>(s<sub>t</sub>|s<sub>t+1</sub>))<sup>2</sup>, with R(x)=exp((ΔBPB + α·diversity + β·fan_margin − γ·certificate_error)/τ).</div></section>
<script>
const payload = {json.dumps(payload)};
const byId = Object.fromEntries(payload.nodes.map(n => [n.id, n]));
const edgeTraces = payload.edges.map(e => {{
  const a = byId[e[0]], b = byId[e[1]];
  return {{type:'scatter3d',mode:'lines',x:[a.x,b.x],y:[a.y,b.y],z:[a.z,b.z],line:{{color:'#46e7ff',width:5}},hoverinfo:'skip',showlegend:false}};
}});
const nodeTrace = {{type:'scatter3d',mode:'markers+text',x:payload.nodes.map(n=>n.x),y:payload.nodes.map(n=>n.y),z:payload.nodes.map(n=>n.z),text:payload.nodes.map(n=>n.id),customdata:payload.nodes.map(n=>[n.kind,n.tree]),marker:{{size:9,color:payload.nodes.map(n=>n.tree),colorscale:'Viridis',line:{{color:'#fff',width:1}}}},hovertemplate:'%{{text}}<br>kind %{{customdata[0]}}<br>tree %{{customdata[1]}}<extra></extra>'}};
Plotly.newPlot('fotPlot',[...edgeTraces,nodeTrace],{{template:'plotly_dark',paper_bgcolor:'#020713',plot_bgcolor:'#020713',scene:{{bgcolor:'#020713'}},margin:{{l:0,r:0,t:10,b:0}}}},{{responsive:true}});
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
    sample = "byte graphs!"
    nodes = [{"id": idx, "prefix": sample[:idx]} for idx in range(len(sample) + 1)]
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
        {"i": 0, "j": 4, "token": "byte", "kind": "priced vocabulary token", "cost": 0.31, "lp": 0.66, "selected": False, "lane": 1},
        {"i": 0, "j": 5, "token": "byte ", "kind": "selected rounded token", "cost": 0.29, "lp": 0.78, "selected": True, "lane": 2},
        {"i": 5, "j": 10, "token": "graph", "kind": "priced vocabulary token", "cost": 0.22, "lp": 0.54, "selected": False, "lane": 1},
        {"i": 5, "j": 11, "token": "graphs", "kind": "priced vocabulary token", "cost": 0.20, "lp": 0.47, "selected": False, "lane": 2},
        {"i": 5, "j": 12, "token": "graphs!", "kind": "selected rounded token", "cost": 0.18, "lp": 0.82, "selected": True, "lane": 3},
        {"i": 0, "j": 12, "token": "byte graphs!", "kind": "LP-relaxation long-span candidate", "cost": 0.53, "lp": 0.35, "selected": False, "lane": 4},
    ]
    payload = {
        "sample_text": sample,
        "nodes": nodes,
        "edges": edges,
        "selected_path": [[0, 5], [5, 12]],
        "note": (
            "Vertices are byte boundaries. Edges are possible tokens spanning byte intervals. "
            "The highlighted selected path is the rounded tokenizer segmentation used for BPB scoring; "
            "LP scores are retained as graph edge features and lower-bound/regret diagnostics."
        ),
    }
    body = f"""
<section class="panel">
  <p class="lead">ConvexTok turns tokenizer construction into a shortest-path and sparse linear-programming problem over a byte-boundary DAG. ToricGT uses the tokenization DAG as first-class graph structure: boundary vertices are graph nodes, candidate substrings are priced edges, LP scores become edge features, and the selected token path is a tropical min-plus dynamic program.</p>
  <p><a href="https://arxiv.org/abs/2605.22821">ConvexTok preprint requested in the training notes</a> · <a href="https://github.com/openai/parameter-golf">OpenAI Parameter Golf</a></p>
</section>
<section class="panel"><h2>Tokenization DAG</h2><div id="dagPlot" class="plot" style="height:680px"></div><p class="metric-note">Curved arcs span byte intervals. Cyan bottom arcs are fallback byte edges, violet arcs are priced candidate tokens, gold arcs are LP-relaxation support, and green arcs are the selected rounded token path. The model receives the same structure as TokenGT-style node/edge features; BPB is still scored on the flattened selected path.</p></section>
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
</section>
<section class="grid three">
  <article class="card"><h3>BPB Use</h3><p>FineWeb is graphified, but OAI scoring still uses the optional flattening path so the byte objective remains primary.</p></article>
  <article class="card"><h3>Toric Use</h3><p>Candidate-token exponent and LP-score features define active faces; the selected path is audited as a tropical curve embedded into a toric chart.</p></article>
  <article class="card"><h3>OOD Use</h3><p>Because substring choices are graph paths, the model sees reusable local graph grammar rather than opaque token IDs only.</p></article>
</section>
<script>
const payload = {json.dumps(payload)};
function arcPoints(edge) {{
  const n = 42;
  const span = edge.j - edge.i;
  const base = edge.kind.includes('fallback') ? -0.15 : 0.02;
  const height = edge.kind.includes('fallback') ? -0.16 : (0.38 + 0.10 * span + 0.13 * edge.lane);
  const xs = [], ys = [];
  for (let k = 0; k < n; k++) {{
    const t = k / (n - 1);
    xs.push(edge.i + span * t);
    ys.push(base + height * Math.sin(Math.PI * t));
  }}
  return [xs, ys];
}}
function edgeStyle(edge) {{
  if (edge.selected) return {{color:'#88ff86', width:7, dash:'solid'}};
  if (edge.kind.includes('LP-relaxation')) return {{color:'#ffd166', width:4, dash:'dash'}};
  if (edge.kind.includes('fallback')) return {{color:'rgba(70,231,255,.58)', width:2.5, dash:'solid'}};
  return {{color:'#a78bfa', width:4, dash:'solid'}};
}}
function edgeTrace(edge) {{
  const [x, y] = arcPoints(edge);
  const style = edgeStyle(edge);
  return {{type:'scatter',mode:'lines',x,y,line:style,name:edge.kind,legendgroup:edge.kind,showlegend:false,customdata:x.map(() => [edge.token, edge.kind, edge.cost, edge.lp, edge.i, edge.j, edge.selected]),hovertemplate:'token <b>%{{customdata[0]}}</b><br>%{{customdata[1]}}<br>span [%{{customdata[4]}}, %{{customdata[5]}}]<br>cost %{{customdata[2]:.3f}} · LP %{{customdata[3]:.2f}}<br>selected %{{customdata[6]}}<extra></extra>'}};
}}
function edgeLabel(edge) {{
  const span = edge.j - edge.i;
  const x = (edge.i + edge.j) / 2;
  const y = edge.kind.includes('fallback') ? -0.36 : (0.20 + 0.10 * span + 0.13 * edge.lane);
  return {{type:'scatter',mode:'text',x:[x],y:[y],text:[edge.token === ' ' ? 'space' : edge.token],textfont:{{color:edge.selected?'#88ff86':(edge.kind.includes('LP-relaxation')?'#ffd166':'#ecfbff'),size:edge.selected?15:11}},hoverinfo:'skip',showlegend:false}};
}}
const traces = [];
for (const edge of payload.edges) traces.push(edgeTrace(edge));
for (const edge of payload.edges.filter(e => !e.kind.includes('fallback'))) traces.push(edgeLabel(edge));
traces.push({{type:'scatter',mode:'lines',x:payload.nodes.map(n=>n.id),y:payload.nodes.map(_=>0),line:{{color:'rgba(236,251,255,.32)',width:2}},hoverinfo:'skip',showlegend:false}});
traces.push({{type:'scatter',mode:'markers+text',x:payload.nodes.map(n=>n.id),y:payload.nodes.map(_=>0),text:payload.nodes.map(n=>String(n.id)),textposition:'bottom center',customdata:payload.nodes.map(n=>[n.id,n.prefix]),marker:{{size:11,color:'#46e7ff',line:{{color:'#ecfbff',width:1}}}},hovertemplate:'byte boundary %{{customdata[0]}}<br>prefix “%{{customdata[1]}}”<extra></extra>',name:'byte boundaries'}});
traces.push({{type:'scatter',mode:'markers',x:payload.edges.filter(e=>e.selected).map(e=>e.j),y:payload.edges.filter(e=>e.selected).map(_=>0.02),marker:{{symbol:'triangle-right',size:16,color:'#88ff86'}},hovertemplate:'selected path boundary<extra></extra>',name:'selected path arrows'}});
Plotly.newPlot('dagPlot', traces, {{template:'plotly_dark',paper_bgcolor:'#020713',plot_bgcolor:'#020713',xaxis:{{title:'byte boundary index',zeroline:false,gridcolor:'rgba(157,184,207,.16)',range:[-0.4,12.4]}},yaxis:{{title:'candidate-token arcs',zeroline:false,gridcolor:'rgba(157,184,207,.12)',range:[-0.55,2.35]}},margin:{{l:54,r:18,t:34,b:52}},annotations:[{{xref:'paper',yref:'paper',x:0.015,y:0.99,text:payload.sample_text,showarrow:false,font:{{color:'#ffd166',size:16}}}}]}}, {{responsive:true}});
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
    metrics = {
        "toric_bgg_loss": best_metrics.get("toric_bgg_loss"),
        "koszul_persistence_loss": best_metrics.get("koszul_persistence_loss"),
        "derived_signature_loss": best_metrics.get("derived_signature_loss"),
        "toric_cca_topology_loss": best_metrics.get("toric_cca_topology_loss"),
    }
    table = "".join(f"<tr><th>{html.escape(k)}</th><td>{fmt(v,6)}</td></tr>" for k, v in metrics.items())
    body = f"""
<section class="panel">
  <p class="lead">The Toric BGG layer is a finite certificate system: it never claims the model contains a literal abelian category. It attaches small sign-vector posets, standard-filtration masks, sparse differentials, Koszul degree profiles, Gale-dual labels, and derived signatures to training records. Late-phase weights remain small so BPB stays primary.</p>
</section>
<section class="grid two">
  <article class="panel"><h2>Finite Category-O Skeleton</h2>
  <svg viewBox="0 0 640 420" aria-label="BGG poset and chain complex">
    <defs><marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#46e7ff"/></marker></defs>
    <rect x="20" y="20" width="600" height="380" rx="14" fill="#020713" stroke="rgba(70,231,255,.32)"/>
    <g stroke="#46e7ff" stroke-width="3" marker-end="url(#arr)" fill="none">
      <path d="M110 340 L210 250 L310 340"/><path d="M210 250 L310 160 L410 250"/><path d="M310 340 L410 250 L510 340"/><path d="M310 160 L410 70"/>
    </g>
    <g fill="#ffd166" stroke="#fff" stroke-width="1.5">
      <circle cx="110" cy="340" r="18"/><circle cx="310" cy="340" r="18"/><circle cx="510" cy="340" r="18"/><circle cx="210" cy="250" r="18"/><circle cx="410" cy="250" r="18"/><circle cx="310" cy="160" r="18"/><circle cx="410" cy="70" r="18"/>
    </g>
    <g fill="#ecfbff" font-size="18" text-anchor="middle"><text x="110" y="346">Lα</text><text x="310" y="346">Lβ</text><text x="510" y="346">Lγ</text><text x="210" y="256">Δα</text><text x="410" y="256">Δβ</text><text x="310" y="166">P</text><text x="410" y="76">C</text></g>
    <text x="52" y="58" fill="#9db8cf" font-size="18">standard objects, covers, and differentials</text>
    <text x="52" y="382" fill="#88ff86" font-size="18">losses: d²≈0 · standard leakage · Gale dual · Koszul profile</text>
  </svg></article>
  <article class="panel"><h2>Best-Run BGG Metrics</h2><table class="mini-table"><tbody>{table}</tbody></table><div class="equation">d(m⊗ξ)=Σᵢ xᵢm⊗eᵢ∧ξ, so d²=0 by commutativity of xᵢ and anti-commutativity of eᵢ.</div></article>
</section>
"""
    (out / "index.html").write_text(method_report_shell("Toric BGG Category O", "Finite homological certificates", body), encoding="utf-8")
    return {
        "href": "page_interactive/bgg_method/index.html",
        "title": "Toric BGG Category O Method",
        "summary": "Finite highest-weight skeletons, standard masks, BGG differentials, Gale duality, Koszul checks, and best-run metric table.",
        "features": "category O skeleton · d² residual · Gale dual · Koszul profile",
        "embed": "0",
    }


def write_theory_gallery_report(docs: Path) -> dict[str, str]:
    out = docs / "page_interactive" / "theory_gallery"
    out.mkdir(parents=True, exist_ok=True)
    body = """
<section class="panel">
  <p class="lead">This gallery connects the visual mathematics behind ToricGT: tropical ring attention, toric fans, Young tableaux and Schur functors, tensor/wedge/symmetric products, multiparameter persistence modules, and thought-control dynamics. The diagrams are didactic illustrations generated for this page, not copied from textbooks.</p>
  <p><a href="../../assets/2210.11433v1.pdf">Multiparameter persistence PDF</a> · <a href="../../assets/ergodic-theory.pdf">Ergodic theory reference</a> · <a href="https://github.com/amelie-iska/Tropical_Quivers_of_Archs/blob/main/tropical_quiver_research_program.tex">Tropical Quivers of Archs</a></p>
</section>
<section class="grid two">
  <article class="card"><h3>Tropical Ring Attention → Toric Fan</h3><svg viewBox="0 0 520 300"><rect width="520" height="300" rx="14" fill="#020713" stroke="rgba(70,231,255,.28)"/><g transform="translate(128 150)" stroke="#46e7ff" stroke-width="3"><line x1="0" y1="0" x2="100" y2="-75"/><line x1="0" y1="0" x2="108" y2="70"/><line x1="0" y1="0" x2="-92" y2="80"/><line x1="0" y1="0" x2="-80" y2="-82"/></g><polyline points="255,210 315,112 372,148 430,64" fill="none" stroke="#ffd166" stroke-width="6"/><text x="30" y="36" fill="#ecfbff">Yᵢc=maxⱼ(Sᵢⱼ+Vⱼc)</text><text x="255" y="248" fill="#9db8cf">active face gives cone chart</text></svg></article>
  <article class="card"><h3>Young Tableaux, Schur Functors</h3><svg viewBox="0 0 520 300"><rect width="520" height="300" rx="14" fill="#020713" stroke="rgba(70,231,255,.28)"/><g transform="translate(42 60)" fill="rgba(70,231,255,.18)" stroke="#46e7ff"><rect x="0" y="0" width="48" height="48"/><rect x="48" y="0" width="48" height="48"/><rect x="96" y="0" width="48" height="48"/><rect x="0" y="48" width="48" height="48"/><rect x="48" y="48" width="48" height="48"/><rect x="0" y="96" width="48" height="48"/></g><g fill="#ecfbff" font-size="24" text-anchor="middle"><text x="66" y="92">1</text><text x="114" y="92">2</text><text x="162" y="92">4</text><text x="66" y="140">2</text><text x="114" y="140">3</text><text x="66" y="188">4</text></g><text x="245" y="96" fill="#ffd166" font-size="24">LλE = (∧λ₁E ⊗ ⋯ ⊗ ∧λₛE)/R</text><text x="245" y="145" fill="#9db8cf" font-size="18">standard monomial bases for<br/>coordinate rings of complexes</text></svg></article>
  <article class="card"><h3>Tensor, Wedge, Symmetric, Shuffle</h3><svg viewBox="0 0 520 300"><rect width="520" height="300" rx="14" fill="#020713" stroke="rgba(70,231,255,.28)"/><text x="44" y="70" fill="#ecfbff" font-size="26">V⊗W</text><text x="44" y="132" fill="#88ff86" font-size="26">∧²V: v⊗w − w⊗v</text><text x="44" y="194" fill="#ffd166" font-size="26">Sym²V: v⊗w + w⊗v</text><path d="M315 62 C390 30 410 120 470 84" fill="none" stroke="#ff5fa2" stroke-width="5"/><path d="M315 162 C390 210 420 120 470 182" fill="none" stroke="#46e7ff" stroke-width="5" stroke-dasharray="8 8"/><text x="308" y="244" fill="#9db8cf">shuffle relations organize representation channels</text></svg></article>
  <article class="card"><h3>F₂[x_level,y_radius] Persistence Module</h3><svg viewBox="0 0 520 300"><rect width="520" height="300" rx="14" fill="#020713" stroke="rgba(70,231,255,.28)"/><g stroke="rgba(157,184,207,.28)"><path d="M70 240 H450"/><path d="M70 200 H450"/><path d="M70 160 H450"/><path d="M70 120 H450"/><path d="M70 80 H450"/><path d="M90 60 V255"/><path d="M150 60 V255"/><path d="M210 60 V255"/><path d="M270 60 V255"/><path d="M330 60 V255"/><path d="M390 60 V255"/></g><path d="M90 240 L90 120 L150 120 L150 80 L270 80 L270 60 L450 60 L450 240 Z" fill="rgba(255,209,102,.25)" stroke="#ffd166" stroke-width="4"/><circle cx="150" cy="120" r="8" fill="#46e7ff"/><circle cx="270" cy="80" r="8" fill="#ff5fa2"/><text x="70" y="278" fill="#9db8cf">level</text><text x="20" y="72" fill="#9db8cf">radius</text></svg></article>
  <article class="card"><h3>Thought Alcove Control</h3><svg viewBox="0 0 520 300"><rect width="520" height="300" rx="14" fill="#020713" stroke="rgba(70,231,255,.28)"/><g stroke="#46e7ff" stroke-width="2"><path d="M80 250 L260 40 L450 250"/><path d="M80 250 H450"/><path d="M170 145 H355"/></g><path d="M110 222 C180 170 230 212 280 126 S375 124 420 82" fill="none" stroke="#88ff86" stroke-width="6"/><circle cx="280" cy="126" r="9" fill="#ffd166"/><text x="56" y="36" fill="#ecfbff">control u(t) keeps trajectory inside stable fan alcoves</text><text x="84" y="282" fill="#9db8cf">Lyapunov / ergodic / optimal-control audits</text></svg></article>
  <article class="card"><h3>Schubert/Toric Intersection Heuristic</h3><svg viewBox="0 0 520 300"><rect width="520" height="300" rx="14" fill="#020713" stroke="rgba(70,231,255,.28)"/><circle cx="180" cy="150" r="88" fill="rgba(70,231,255,.16)" stroke="#46e7ff" stroke-width="4"/><circle cx="300" cy="150" r="88" fill="rgba(255,95,162,.16)" stroke="#ff5fa2" stroke-width="4"/><circle cx="240" cy="95" r="88" fill="rgba(255,209,102,.13)" stroke="#ffd166" stroke-width="4"/><text x="182" y="154" fill="#ecfbff" text-anchor="middle">σλ</text><text x="303" y="154" fill="#ecfbff" text-anchor="middle">σμ</text><text x="240" y="96" fill="#ecfbff" text-anchor="middle">Dρ</text><text x="112" y="260" fill="#9db8cf">intersection numbers become certificate features</text></svg></article>
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
        "bgg_method": write_bgg_method_report(docs, best_metrics),
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
                  <img src="{html.escape(copied[key])}" alt="{html.escape(title)}">
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
              <a class="button" href="{html.escape(href)}">Open interactive report</a>
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
        <h2>Visual Evidence</h2>
      </div>
      <div class="grid gallery">
        {''.join(gallery)}
      </div>
    </section>

    <section class="wrap">
      <div class="section-head">
        <h2>Interactive Analysis Lab</h2>
        <p>The full generated reports are bundled into the page, not flattened to screenshots. They retain Plotly rotation, hover/click token details, radius sliders, reasoning-level sliders, simplex-tree maps, vectorized PH panels, CAS links, and tetrahedron views.</p>
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
        <h3>Thought Fluid Dynamics</h3>
        <p class="lead">The Tropical Quivers of Archs program frames learned graph-to-graph functions as composable operators with tropical/polyhedral local charts. For ToricGT, that suggests a fluid view of reasoning: hidden trajectories have divergence, circulation, vorticity, boundary flux, and energy. Navier-Stokes-inspired regularizers should be used conservatively, as audits and small penalties, but they give a precise vocabulary for branch merging, turbulence in unstable reasoning zones, and dissipative correction when FoT branches drift. The long-run objective is optimal control over a learned, graph-valued, tropical-toric dynamical system.</p>
        <p><a href="https://huggingface.co/blog/AmelieSchreiber/toricblm">ToricBLM blog</a> · <a href="https://github.com/amelie-iska/Tropical_Quivers_of_Archs/blob/main/tropical_quiver_research_program.tex">Tropical Quivers of Archs</a> · <a href="https://zitniklab.hms.harvard.edu/projects/GeoBPE/">GeoBPE reference</a> · <a href="https://openreview.net/forum?id=o4ANDWaomX">protein structure tokenization benchmark</a></p>
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
    interactive = {**write_method_reports(docs, copied, best_metrics), **interactive}
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
