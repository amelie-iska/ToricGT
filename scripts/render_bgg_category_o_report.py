#!/usr/bin/env python3
"""Render a strict Toric BGG / Category O metric report.

The report is intentionally finite.  It either consumes a metrics JSON emitted
by training/analysis or, when no metrics JSON is supplied, renders the exact
toy BGG certificate used as the sanity check.  It does not fabricate unavailable
training metrics: missing requested metrics are listed as unavailable with
provenance rather than substituted by proxy values.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toricgt.toric_bgg import (  # noqa: E402
    ToricBGGProbe,
    boundary_square_residual,
    standard_filtration_leakage,
    toric_bgg_metric_provenance,
    toy_bgg_certificate,
)


CSS = """
:root { color-scheme: dark; --bg:#030712; --panel:#07111f; --text:#e8fbff; --muted:#91a8b7; --cyan:#37e8ff; --magenta:#ff4fd8; --green:#8cff6a; --border:rgba(55,232,255,.28); }
body { margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:radial-gradient(circle at top left,#092238 0,var(--bg) 42rem); color:var(--text); }
main { max-width:1180px; margin:0 auto; padding:32px 24px 64px; }
.hero,.card,.panel { background:linear-gradient(180deg,rgba(11,23,40,.94),rgba(5,13,25,.96)); border:1px solid var(--border); border-radius:8px; box-shadow:0 18px 50px rgba(0,0,0,.28); }
.hero,.card,.panel { padding:18px; margin-bottom:16px; }
.grid { display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(270px,1fr)); }
h1 { margin:0 0 8px; font-size:28px; letter-spacing:0; } h2 { margin:0 0 10px; font-size:18px; letter-spacing:0; }
p { color:var(--muted); line-height:1.55; }
a { color:var(--cyan); text-decoration:none; } a:hover { text-decoration:underline; }
.metric { display:flex; justify-content:space-between; gap:12px; border-bottom:1px solid rgba(145,168,183,.14); padding:7px 0; }
.metric span:first-child { color:var(--muted); }
.metric span:last-child { color:white; overflow-wrap:anywhere; text-align:right; font-variant-numeric:tabular-nums; }
.badge { display:inline-block; border:1px solid rgba(140,255,106,.35); border-radius:999px; color:#d9ffd2; background:rgba(140,255,106,.11); padding:4px 8px; font-size:12px; margin:2px 4px 2px 0; }
.badge.off { border-color:rgba(255,79,216,.45); color:#ffd8f7; background:rgba(255,79,216,.12); }
pre { white-space:pre-wrap; overflow-x:auto; background:#020713; border:1px solid rgba(145,168,183,.18); border-radius:8px; padding:14px; color:#dff8ff; }
.links { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
.pill { border:1px solid var(--border); border-radius:999px; padding:6px 10px; background:rgba(55,232,255,.07); }
"""


EXPECTED_METRICS = (
    "toric_bgg_d2_residual",
    "toric_bgg_standard_leakage",
    "toric_bgg_koszul_linearity_residual",
    "toric_bgg_gale_dual_consistency",
    "toric_bgg_signature_smoothness",
    "toric_bgg_standard_entropy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", default="", help="Optional training/analysis metrics JSON.")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint containing the trained Toric BGG sidecar probe.",
    )
    parser.add_argument(
        "--embedding-manifest",
        default="",
        help="Optional manifest emitted by extract_oai_sidecar_embeddings.py.",
    )
    parser.add_argument(
        "--embedding-npz",
        default="",
        help="Optional direct hidden-state npz. Overrides --embedding-manifest.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _manifest_first_npz(path: Path) -> Path:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent.parent if path.name == "manifest.json" and path.parent.name == "embeddings" else path.parent
    for section in ("payloads", "records"):
        for record in manifest.get(section, []) or []:
            rel = record.get("relative_npz") or record.get("npz") or record.get("npz_path")
            if rel:
                candidate = Path(rel)
                if not candidate.is_absolute():
                    candidate = base / candidate
                if candidate.exists():
                    return candidate
    raise FileNotFoundError(f"no embedding npz found in {path}")


def checkpoint_bgg_metrics(checkpoint: Path, embedding_npz: Path) -> dict[str, Any]:
    """Evaluate the trained finite Toric BGG probe on extracted hidden states."""

    ckpt = torch.load(checkpoint, map_location="cpu")
    sidecar = ckpt.get("sidecar") if isinstance(ckpt, dict) else None
    if not isinstance(sidecar, dict):
        raise ValueError(f"{checkpoint} does not contain a sidecar state dict")
    bgg_state = {
        key.removeprefix("toric_bgg."): value
        for key, value in sidecar.items()
        if key.startswith("toric_bgg.")
    }
    if not bgg_state:
        raise ValueError(f"{checkpoint} sidecar has no toric_bgg.* weights")
    hidden_npz = np.load(embedding_npz, allow_pickle=False)
    if "hidden" not in hidden_npz:
        raise ValueError(f"{embedding_npz} does not contain a hidden array")
    hidden = torch.from_numpy(hidden_npz["hidden"]).float().unsqueeze(0)
    token_ids = None
    if "token_ids" in hidden_npz:
        token_ids = torch.from_numpy(hidden_npz["token_ids"]).long().unsqueeze(0)
    probe = ToricBGGProbe(d_model=int(hidden.shape[-1]))
    missing, unexpected = probe.load_state_dict(bgg_state, strict=False)
    if unexpected:
        raise ValueError(f"unexpected Toric BGG probe keys: {unexpected}")
    probe.eval()
    with torch.no_grad():
        out = probe(hidden, target_tokens=token_ids)
    metrics: dict[str, Any] = {
        "source": "trained_checkpoint_toric_bgg_probe_on_hidden_states",
        "checkpoint": str(checkpoint),
        "embedding_npz": str(embedding_npz),
        "probe_missing_state_keys": list(missing),
    }
    for key in EXPECTED_METRICS:
        value = out.get(key)
        if value is None:
            continue
        metrics[key] = float(value.detach().cpu().item())
    for key in (
        "toric_bgg_loss",
        "toric_bgg_resolution_consistency",
        "toric_bgg_standard_allowed_mass",
        "toric_bgg_exact_certificate_available",
        "toric_bgg_provenance_exact_finite_chain",
        "toric_bgg_late_gate_required",
    ):
        value = out.get(key)
        if value is not None:
            metrics[key] = float(value.detach().cpu().item())
    return metrics


def read_metrics(
    path: Path | None,
    *,
    checkpoint: Path | None = None,
    embedding_manifest: Path | None = None,
    embedding_npz: Path | None = None,
) -> dict[str, Any]:
    if path is None:
        cert = toy_bgg_certificate()
        probs = torch.eye(3)
        labels = torch.arange(3)
        return {
            "source": "exact_toy_bgg_certificate",
            "toric_bgg_d2_residual": float(boundary_square_residual(cert.boundaries).item()),
            "toric_bgg_standard_leakage": float(standard_filtration_leakage(probs, labels, cert.standard_mask).item()),
            "toric_bgg_koszul_linearity_residual": 0.0,
            "toric_bgg_gale_dual_consistency": 0.0,
            "toric_bgg_signature_smoothness": 0.0,
            "toric_bgg_standard_entropy": 0.0,
            "toric_bgg_exact_certificate_available": 1.0,
            "toric_bgg_provenance_exact_finite_chain": 1.0,
            "toric_bgg_late_gate_required": 1.0,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    missing = [key for key in EXPECTED_METRICS if payload.get(key) is None]
    if missing and checkpoint is not None and (embedding_manifest is not None or embedding_npz is not None):
        if embedding_npz is None:
            assert embedding_manifest is not None
            embedding_npz = _manifest_first_npz(embedding_manifest)
        computed = checkpoint_bgg_metrics(checkpoint, embedding_npz)
        merged = dict(payload)
        for key, value in computed.items():
            if key == "source":
                continue
            merged[key] = value
        merged["source"] = f"{payload.get('source', 'metrics_json')}+trained_checkpoint_toric_bgg_probe"
        merged["toric_bgg_probe_source"] = computed
        return merged
    return payload


def metric_value(metrics: dict[str, Any], key: str) -> tuple[str, bool]:
    raw = metrics.get(key)
    if raw is None:
        return "unavailable", False
    if isinstance(raw, (int, float)):
        return f"{float(raw):.6g}", True
    return str(raw), True


def metric_card(title: str, keys: tuple[str, ...], metrics: dict[str, Any], provenance: dict[str, str]) -> str:
    rows = []
    for key in keys:
        value, available = metric_value(metrics, key)
        badge_cls = "badge" if available else "badge off"
        badge_text = "available" if available else "unavailable"
        rows.append(
            f'<div class="metric"><span>{html.escape(key)}</span><span>{html.escape(value)} '
            f'<span class="{badge_cls}">{badge_text}</span><br><small>{html.escape(provenance.get(key, "no provenance registered"))}</small></span></div>'
        )
    return f'<div class="card"><h2>{html.escape(title)}</h2>{"".join(rows)}</div>'


def write_report(metrics: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = toric_bgg_metric_provenance()
    report = {
        "schema": "toricgt.bgg_category_o_report.v1",
        "source": metrics.get("source", "metrics_json"),
        "metric_provenance": provenance,
        "metrics": {key: metrics.get(key, None) for key in EXPECTED_METRICS},
        "unavailable": [key for key in EXPECTED_METRICS if key not in metrics],
        "gates": {
            "exact_certificate_available": metrics.get("toric_bgg_exact_certificate_available", 0.0),
            "exact_finite_chain_provenance": metrics.get("toric_bgg_provenance_exact_finite_chain", 0.0),
            "late_gate_required": metrics.get("toric_bgg_late_gate_required", 1.0),
        },
    }
    report_json = output_dir / "bgg_category_o_report.json"
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gates = "".join(
        f'<span class="badge{" off" if not bool(value) else ""}">{html.escape(label)}: {html.escape(str(value))}</span>'
        for label, value in report["gates"].items()
    )
    html_path = output_dir / "index.html"
    html_path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Toric BGG Category O Report</title><style>{CSS}</style></head>
<body><main>
<section class="hero">
  <h1>Toric BGG Category O Report</h1>
  <p>Finite Category O/BGG diagnostics with explicit certificate provenance.  When a checkpoint sidecar and hidden-state payload are supplied, the trained Toric BGG probe is evaluated directly on that payload.</p>
  <div>{gates}</div>
  <div class="links"><a class="pill" href="bgg_category_o_report.json">report JSON</a></div>
</section>
<section class="grid">
  {metric_card("Resolution And Standard Filtration", ("toric_bgg_d2_residual", "toric_bgg_standard_leakage"), metrics, provenance)}
  {metric_card("Koszul, Gale, And Signature", ("toric_bgg_koszul_linearity_residual", "toric_bgg_gale_dual_consistency", "toric_bgg_signature_smoothness", "toric_bgg_standard_entropy"), metrics, provenance)}
</section>
<section class="panel"><h2>Raw Metrics</h2><pre>{html.escape(json.dumps(metrics, indent=2, sort_keys=True))}</pre></section>
</main></body></html>
""",
        encoding="utf-8",
    )
    return html_path, report_json


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_json) if args.metrics_json else None
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    embedding_manifest = Path(args.embedding_manifest) if args.embedding_manifest else None
    embedding_npz = Path(args.embedding_npz) if args.embedding_npz else None
    metrics = read_metrics(
        metrics_path,
        checkpoint=checkpoint,
        embedding_manifest=embedding_manifest,
        embedding_npz=embedding_npz,
    )
    html_path, report_json = write_report(metrics, Path(args.output_dir))
    print(json.dumps({"index_html": str(html_path), "report_json": str(report_json)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
