#!/usr/bin/env python3
"""Render an exact embedding-space Forest-of-Thought report for one checkpoint.

The report uses the real OAI checkpoint FoT head saved by ``train_gpt.py`` and
the real hidden-state payload emitted by ``extract_oai_sidecar_embeddings.py``.
It does not synthesize hidden states or forest scores.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toricgt.embedding_forest_of_thought import EmbeddingFoTConfig, EmbeddingForestOfThoughtHead  # noqa: E402


CSS = """
:root { color-scheme: dark; --bg:#06101c; --panel:#0c1928; --ink:#eef8ff; --muted:#a9c5da; --cyan:#31e7ff; --line:#21445c; --gold:#ffd166; --pink:#ff4fd8; }
body { margin:0; background:radial-gradient(circle at top left,#10263d,#06101c 55%); color:var(--ink); font:15px/1.5 system-ui,Segoe UI,sans-serif; }
main { max-width:1280px; margin:0 auto; padding:28px; }
.card { background:rgba(12,25,40,.94); border:1px solid #15516c; border-radius:8px; padding:18px; margin:16px 0; box-shadow:0 12px 40px rgba(0,0,0,.28); }
h1,h2 { margin:0 0 10px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }
.metric { display:flex; justify-content:space-between; gap:12px; border-bottom:1px solid rgba(160,210,240,.12); padding:7px 0; }
.metric span:first-child { color:var(--muted); }
.metric span:last-child { font-variant-numeric:tabular-nums; text-align:right; }
svg { width:100%; height:auto; background:#030a14; border:1px solid #14374d; border-radius:8px; }
.legend { display:flex; flex-wrap:wrap; gap:12px; color:var(--muted); font-size:13px; }
.swatch { display:inline-block; width:12px; height:12px; border-radius:50%; vertical-align:-2px; margin-right:5px; }
code,pre { background:#030a14; border:1px solid #14374d; border-radius:6px; }
pre { overflow:auto; padding:12px; }
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding-npz", required=True)
    parser.add_argument("--embedding-json", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-trees", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-positions", type=int, default=192)
    parser.add_argument("--topk-trees", type=int, default=2)
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def checkpoint_fot_state(path: Path) -> dict[str, torch.Tensor] | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("oai_fot"), dict):
        return None
    state = payload["oai_fot"]
    if not state:
        return None
    return state


def infer_config(state: dict[str, torch.Tensor], dim: int, args: argparse.Namespace) -> EmbeddingFoTConfig:
    hidden_dim = int(state.get("activation_head.0.weight", torch.empty(192, dim)).shape[0])
    branching = int(state.get("forward_policy.2.weight", torch.empty(4, hidden_dim)).shape[0])
    consensus_buckets = int(state.get("consensus_head.2.weight", torch.empty(64, hidden_dim)).shape[0])
    return EmbeddingFoTConfig(
        dim=int(dim),
        num_trees=max(1, int(args.num_trees)),
        max_depth=max(1, int(args.max_depth)),
        branching=max(2, branching),
        topk_trees=max(1, int(args.topk_trees)),
        hidden_dim=max(1, hidden_dim),
        max_positions=max(2, int(args.max_positions)),
        consensus_buckets=max(2, consensus_buckets),
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_xy(points: np.ndarray, width: int, height: int, pad: int = 45) -> np.ndarray:
    xy = np.asarray(points[:, :2], dtype=np.float64)
    if xy.shape[0] == 0:
        return xy
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1.0e-8)
    out = (xy - lo) / span
    out[:, 0] = pad + out[:, 0] * (width - 2 * pad)
    out[:, 1] = height - pad - out[:, 1] * (height - 2 * pad)
    return out


def color_for(value: float, lo: float, hi: float) -> str:
    if hi <= lo:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    r = int(49 + 206 * t)
    g = int(231 - 110 * t)
    b = int(255 - 220 * t)
    return f"rgb({r},{g},{b})"


def svg_forest(trace: dict[str, Any], projected: np.ndarray) -> str:
    nodes = trace.get("nodes", [])
    edges = trace.get("edges", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    width, height = 1120, 680
    xy = normalize_xy(projected[: len(nodes)], width, height)
    nlls = [finite_float(node.get("nll")) for node in nodes if isinstance(node, dict)]
    lo = min(nlls) if nlls else 0.0
    hi = max(nlls) if nlls else 1.0
    tree_colors = ["#31e7ff", "#ff4fd8", "#75ff6a", "#ffd166", "#a78bfa", "#ff8c42", "#6fffe9", "#ff6b91"]
    parts: list[str] = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Embedding FoT forest">']
    parts.append('<rect width="100%" height="100%" fill="#030a14"/>')
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = int(edge.get("source", 0))
        target = int(edge.get("target", 0))
        tree_id = int(edge.get("tree_id", 0))
        if source >= len(xy) or target >= len(xy):
            continue
        x1, y1 = xy[source]
        x2, y2 = xy[target]
        color = tree_colors[tree_id % len(tree_colors)]
        parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="1.5" opacity="0.46"/>'
        )
    # Consensus guide lines connect tree leaves to the final sampled node.
    by_tree: dict[int, int] = {}
    for idx, node in enumerate(nodes):
        if isinstance(node, dict):
            by_tree[int(node.get("tree_id", 0))] = idx
    final_idx = len(nodes) - 1
    for tree_id, source in by_tree.items():
        if source == final_idx or source >= len(xy) or final_idx >= len(xy):
            continue
        x1, y1 = xy[source]
        x2, y2 = xy[final_idx]
        parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            'stroke="#ffd166" stroke-width="1.2" stroke-dasharray="6 7" opacity="0.58"/>'
        )
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict) or idx >= len(xy):
            continue
        x, y = xy[idx]
        tree_id = int(node.get("tree_id", 0))
        nll = finite_float(node.get("nll"))
        fill = color_for(nll, lo, hi)
        stroke = tree_colors[tree_id % len(tree_colors)]
        title = html.escape(
            f"node {idx} tree {tree_id} pos {node.get('position')} token {node.get('target_id')} "
            f"nll {nll:.4f} activation {finite_float(node.get('activation')):.4f} value {finite_float(node.get('value')):.4f}"
        )
        parts.append(f'<g><title>{title}</title><circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        if idx % max(1, len(nodes) // 32) == 0 or idx == final_idx:
            parts.append(f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" fill="#dff8ff" font-size="11">{idx}</text>')
        parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


def metric_rows(metrics: dict[str, Any], trace_metrics: dict[str, Any]) -> str:
    latest = metrics.get("latest_train", {}) if isinstance(metrics.get("latest_train"), dict) else {}
    wanted = {
        "train_bpb": latest.get("train_bpb"),
        "val_or_final_bpb": metrics.get("selected_bpb"),
        "oai_fot_loss": latest.get("oai_fot_loss"),
        "oai_fot_entropy": latest.get("oai_fot_entropy"),
        "oai_fot_reward": latest.get("oai_fot_reward"),
        "oai_fot_diversity": latest.get("oai_fot_diversity"),
        "trace_oai_fot_loss": trace_metrics.get("oai_fot_loss"),
        "trace_sparse_activation_loss": trace_metrics.get("oai_fot_sparse_activation_loss"),
        "trace_ucb_loss": trace_metrics.get("oai_fot_ucb_loss"),
        "trace_correction_loss": trace_metrics.get("oai_fot_self_correction_loss"),
        "trace_consensus_loss": trace_metrics.get("oai_fot_consensus_loss"),
        "trace_tb_residual": trace_metrics.get("oai_fot_tb_residual"),
        "trace_tree_diversity": trace_metrics.get("oai_fot_tree_diversity"),
        "trace_consensus_margin": trace_metrics.get("oai_fot_consensus_margin"),
    }
    rows = []
    for key, value in wanted.items():
        if isinstance(value, torch.Tensor):
            value = float(value.detach().float().item())
        display = f"{float(value):.6g}" if isinstance(value, (int, float)) and math.isfinite(float(value)) else str(value)
        rows.append(f'<div class="metric"><span>{html.escape(key)}</span><span>{html.escape(display)}</span></div>')
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = resolve(args.embedding_npz)
    json_path = resolve(args.embedding_json)
    checkpoint_path = resolve(args.checkpoint)
    metrics = load_json(resolve(args.metrics_json))
    metadata = load_json(json_path)
    arrays = np.load(npz_path)
    hidden = torch.from_numpy(np.asarray(arrays["hidden"], dtype=np.float32)).unsqueeze(0)
    token_ids = torch.from_numpy(np.asarray(arrays["token_ids"], dtype=np.int64)).unsqueeze(0)
    nll = torch.from_numpy(np.asarray(arrays["nll"], dtype=np.float32)).unsqueeze(0)
    projected = np.asarray(arrays["projected"], dtype=np.float32)
    state = checkpoint_fot_state(checkpoint_path)
    latest = metrics.get("latest_train", {}) if isinstance(metrics.get("latest_train"), dict) else {}
    fot_was_logged = isinstance(latest, dict) and latest.get("oai_fot_loss") is not None
    if state is None:
        if fot_was_logged:
            raise RuntimeError(
                f"train log contains FoT metrics, but checkpoint {checkpoint_path} has no `oai_fot` state"
            )
        absent = {
            "schema": "toricgt.embedding_forest_of_thought.absent.v1",
            "checkpoint": str(checkpoint_path),
            "reason": "The checkpoint predates FoT or was trained with OAI_EMBEDDING_FOT=0; no FoT forest is rendered.",
        }
        (output_dir / "embedding_fot_trace.json").write_text(
            json.dumps(absent, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Embedding-Space Forest-of-Thought Audit</title><style>{CSS}</style></head>
<body><main><section class="card"><h1>Embedding-Space Forest-of-Thought Audit</h1>
<p>No FoT state was present in this checkpoint, and no FoT metrics were present in the train log. This is an absence audit, not a proxy visualization.</p>
<pre>{html.escape(json.dumps(absent, indent=2, sort_keys=True))}</pre>
</section></main></body></html>"""
        (output_dir / "index.html").write_text(html_doc, encoding="utf-8")
        print(json.dumps({"index": str(output_dir / "index.html"), "trace": str(output_dir / "embedding_fot_trace.json")}, indent=2))
        return
    config = infer_config(state, int(hidden.shape[-1]), args)
    head = EmbeddingForestOfThoughtHead(config)
    head.load_state_dict(state, strict=True)
    head.eval()
    with torch.no_grad():
        out = head(hidden, token_ids, nll)
        trace = head.trace_payload(hidden, token_ids, nll)
    trace_metrics: dict[str, Any] = {}
    for key, value in out.items():
        if isinstance(value, torch.Tensor):
            trace_metrics[key] = float(value.detach().float().item())
        else:
            trace_metrics[key] = value
    trace["trace_metrics"] = trace_metrics
    trace["embedding_payload"] = str(npz_path)
    trace["embedding_metadata"] = metadata
    write_path = output_dir / "embedding_fot_trace.json"
    write_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    title = "Embedding-Space Forest-of-Thought Audit"
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head>
<body><main>
<section class="card"><h1>{title}</h1>
<p>This report loads the checkpoint's saved FoT head and applies it to the real hidden-state embedding payload from the OAI baseline extractor. Node fill color is token NLL; edge color is tree id; dashed gold lines show leaf-to-consensus guide edges.</p>
<p><a href="embedding_fot_trace.json">trace JSON</a></p>
</section>
<section class="card"><h2>FoT Metrics</h2><div class="grid">{metric_rows(metrics, trace_metrics)}</div></section>
<section class="card"><h2>Hidden Forest</h2>
<div class="legend"><span><i class="swatch" style="background:#31e7ff"></i>tree edges</span><span><i class="swatch" style="background:#ffd166"></i>leaf consensus guide</span><span>node fill: NLL</span></div>
{svg_forest(trace, projected)}
</section>
<section class="card"><h2>Configuration</h2><pre>{html.escape(json.dumps(config.__dict__, indent=2, sort_keys=True))}</pre></section>
</main></body></html>"""
    (output_dir / "index.html").write_text(html_doc, encoding="utf-8")
    print(json.dumps({"index": str(output_dir / "index.html"), "trace": str(write_path)}, indent=2))


if __name__ == "__main__":
    main()
