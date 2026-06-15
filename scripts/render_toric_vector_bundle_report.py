#!/usr/bin/env python3
"""Render exact toric vector-bundle/sheaf certificate panels.

The input is an exact `macaulay2_toric_vector_bundle_certificate` JSON.  If no
certificate is supplied, the script calls Macaulay2 through the local oracle and
builds the P2 rank-2 ToricVectorBundles certificate.  Missing Macaulay2 or
missing structured fields fail loudly.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toricgt.cas_certificates import validate_certificate_payload  # noqa: E402
from toricgt.cas_oracles import Macaulay2TropicalOracle  # noqa: E402


CSS = """
:root { color-scheme: dark; --bg:#030712; --panel:#07111f; --text:#e8fbff; --muted:#91a8b7; --cyan:#37e8ff; --magenta:#ff4fd8; --green:#8cff6a; --border:rgba(55,232,255,.28); }
body { margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:radial-gradient(circle at top left,#092238 0,var(--bg) 42rem); color:var(--text); }
main { max-width:1180px; margin:0 auto; padding:32px 24px 64px; }
.hero,.card,.panel { background:linear-gradient(180deg,rgba(11,23,40,.94),rgba(5,13,25,.96)); border:1px solid var(--border); border-radius:8px; box-shadow:0 18px 50px rgba(0,0,0,.28); padding:18px; margin-bottom:16px; }
.grid { display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(270px,1fr)); }
h1 { margin:0 0 8px; font-size:28px; letter-spacing:0; } h2 { margin:0 0 10px; font-size:18px; letter-spacing:0; }
p { color:var(--muted); line-height:1.55; }
a { color:var(--cyan); text-decoration:none; }
.metric { display:flex; justify-content:space-between; gap:12px; border-bottom:1px solid rgba(145,168,183,.14); padding:7px 0; }
.metric span:first-child { color:var(--muted); }
.metric span:last-child { color:white; overflow-wrap:anywhere; text-align:right; font-variant-numeric:tabular-nums; }
pre { white-space:pre-wrap; overflow-x:auto; background:#020713; border:1px solid rgba(145,168,183,.18); border-radius:8px; padding:14px; color:#dff8ff; }
.links { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
.pill { border:1px solid var(--border); border-radius:999px; padding:6px 10px; background:rgba(55,232,255,.07); }
.badge { display:inline-block; border:1px solid rgba(140,255,106,.35); border-radius:999px; color:#d9ffd2; background:rgba(140,255,106,.11); padding:4px 8px; font-size:12px; margin:2px 4px 2px 0; }
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate-json", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--macaulay2-timeout-seconds", type=int, default=120)
    return parser.parse_args()


def load_or_build_certificate(path: str, timeout_seconds: int) -> dict[str, Any]:
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("certificate JSON must contain an object")
        return payload
    oracle = Macaulay2TropicalOracle()
    cert = oracle.vector_bundle_smoke_certificate(timeout_seconds=int(timeout_seconds))
    return cert.with_hash()


def require_valid(payload: dict[str, Any]) -> None:
    errors = validate_certificate_payload(payload)
    if errors:
        raise ValueError("invalid toric vector-bundle certificate:\n" + "\n".join(errors))


def metric(label: str, value: Any) -> str:
    return f'<div class="metric"><span>{html.escape(label)}</span><span>{html.escape(str(value))}</span></div>'


def write_report(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    require_valid(payload)
    certificate_json = output_dir / "toric_vector_bundle_certificate.json"
    certificate_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    toric = payload.get("toric", {})
    algebra = payload.get("commutative_algebra", {})
    cocycle_badges = "".join(
        f'<span class="badge">{html.escape(str(key))}: {html.escape(str(value))}</span>'
        for key, value in sorted((algebra.get("cech_cocycle_checks", {}) or {}).items())
    )
    overlap_badges = "".join(
        f'<span class="badge">{html.escape(str(key))}: {html.escape(str(value))}</span>'
        for key, value in sorted((algebra.get("chart_overlap_checks", {}) or {}).items())
    )
    html_path = output_dir / "index.html"
    html_path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Toric Vector Bundle / Sheaf Report</title><style>{CSS}</style></head>
<body><main>
<section class="hero">
  <h1>Toric Vector Bundle / Sheaf Report</h1>
  <p>Exact Macaulay2 ToricVectorBundles certificate with structured Klyachko filtrations, chart transitions, Cech cocycle checks, and cohomology summary.</p>
  <div class="links"><a class="pill" href="toric_vector_bundle_certificate.json">certificate JSON</a></div>
</section>
<section class="grid">
  <div class="card"><h2>Bundle</h2>
    {metric("provenance", payload.get("provenance"))}
    {metric("ambient", toric.get("ambient"))}
    {metric("rank", toric.get("rank"))}
    {metric("charts", toric.get("charts"))}
    {metric("is vector bundle", algebra.get("is_vector_bundle"))}
    {metric("Euler characteristic", algebra.get("euler_chi"))}
  </div>
  <div class="card"><h2>Cech And Overlap Checks</h2>
    <p>Cocycle checks:</p>{cocycle_badges}
    <p>Chart-overlap checks:</p>{overlap_badges}
  </div>
</section>
<section class="panel"><h2>One-Dimensional Cones And Maximal Cones</h2><pre>{html.escape(json.dumps({"one_dimensional_cones": toric.get("one_dimensional_cones"), "maximal_cones": toric.get("maximal_cones")}, indent=2, sort_keys=True))}</pre></section>
<section class="panel"><h2>Klyachko Filtrations And Chart Weights</h2><pre>{html.escape(json.dumps({"one_dimensional_cone_filtrations": algebra.get("one_dimensional_cone_filtrations"), "chart_weights": algebra.get("chart_weights")}, indent=2, sort_keys=True))}</pre></section>
<section class="panel"><h2>Transition Matrices And Cech Cocycle</h2><pre>{html.escape(json.dumps({"transition_matrices": algebra.get("transition_matrices"), "cech_cocycle_checks": algebra.get("cech_cocycle_checks"), "chart_overlap_checks": algebra.get("chart_overlap_checks")}, indent=2, sort_keys=True))}</pre></section>
<section class="panel"><h2>Cohomology Summary</h2><pre>{html.escape(json.dumps(algebra.get("cohomology_summary"), indent=2, sort_keys=True))}</pre></section>
<section class="panel"><h2>Raw Macaulay2 Fields</h2><pre>{html.escape(json.dumps(algebra, indent=2, sort_keys=True))}</pre></section>
</main></body></html>
""",
        encoding="utf-8",
    )
    return html_path, certificate_json


def main() -> None:
    args = parse_args()
    payload = load_or_build_certificate(args.certificate_json, int(args.macaulay2_timeout_seconds))
    html_path, certificate_json = write_report(payload, Path(args.output_dir))
    print(json.dumps({"index_html": str(html_path), "certificate_json": str(certificate_json)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
