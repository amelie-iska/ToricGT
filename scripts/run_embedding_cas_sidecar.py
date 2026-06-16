#!/usr/bin/env python3
"""Run exact Sage/Macaulay2 toric audits from saved embedding payloads.

The input is the ``embeddings/manifest.json`` written by
``evaluate_tokengt_reasoning_geometry_suite.py``.  For each saved hidden-state
trajectory this script builds a finite nonnegative integer exponent set from
the actual embedding vectors, then runs:

* SageMath normal-fan computation for the exponent polytope;
* Macaulay2 elimination for the toric ideal of the monomial parametrization.

Requested CAS computations are strict: missing or failing backends raise an
error rather than producing a substitute value.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toricgt.cas_oracles import Macaulay2TropicalOracle, SageToricOracle  # noqa: E402
from toricgt.tropical_toric_certificates import tropical_hypersurface_certificate  # noqa: E402


CSS = """
:root { color-scheme: dark; --bg:#030712; --panel:#07111f; --text:#e8fbff; --muted:#91a8b7; --cyan:#37e8ff; --border:rgba(55,232,255,.28); }
body { margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:radial-gradient(circle at top left,#092238 0,var(--bg) 42rem); color:var(--text); }
main { max-width:1240px; margin:0 auto; padding:32px 24px 64px; }
a { color:var(--cyan); text-decoration:none; } a:hover { text-decoration:underline; }
.hero,.card,.panel { background:linear-gradient(180deg,rgba(11,23,40,.94),rgba(5,13,25,.96)); border:1px solid var(--border); border-radius:8px; box-shadow:0 18px 50px rgba(0,0,0,.28); }
.hero { padding:24px; margin-bottom:20px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; }
.card,.panel { padding:16px; margin-bottom:16px; }
h1 { margin:0 0 8px; font-size:28px; letter-spacing:0; } h2 { margin:0 0 12px; font-size:18px; letter-spacing:0; }
p { color:var(--muted); line-height:1.55; }
.metric { display:flex; justify-content:space-between; gap:14px; border-bottom:1px solid rgba(145,168,183,.14); padding:7px 0; }
.metric span:first-child { color:var(--muted); } .metric span:last-child { color:white; text-align:right; overflow-wrap:anywhere; font-variant-numeric:tabular-nums; }
.links { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
.pill { border:1px solid var(--border); border-radius:999px; padding:6px 10px; background:rgba(55,232,255,.07); }
.tablewrap { overflow-x:auto; margin-top:12px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th,td { border:1px solid rgba(145,168,183,.17); padding:6px 8px; text-align:right; }
th:first-child,td:first-child { text-align:left; color:var(--muted); }
pre { white-space:pre-wrap; overflow-x:auto; background:#020713; border:1px solid rgba(145,168,183,.18); border-radius:8px; padding:14px; color:#dff8ff; }
summary { cursor:pointer; color:var(--cyan); }
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--records", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=18)
    parser.add_argument("--embedding-key", choices=["hidden", "projected", "complex_projected"], default="hidden")
    parser.add_argument("--exponent-dim", type=int, default=2)
    parser.add_argument("--quantization-scale", type=int, default=12)
    parser.add_argument("--sage-timeout-seconds", type=int, default=180)
    parser.add_argument("--macaulay2-timeout-seconds", type=int, default=180)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_payload_npz(manifest_path: Path, row: dict[str, Any]) -> Path:
    for key in ("npz", "relative_npz"):
        raw = row.get(key)
        if not raw:
            continue
        candidate = Path(str(raw))
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if key == "relative_npz":
            relative = manifest_path.parent.parent / candidate
        else:
            relative = manifest_path.parent / candidate
        if relative.exists():
            return relative
    raise FileNotFoundError(f"no NPZ payload path resolved for manifest row: {row}")


def finite_points_from_npz(path: Path, key: str, *, max_points: int) -> np.ndarray:
    with np.load(path) as data:
        if key not in data.files:
            raise KeyError(f"{path} does not contain array {key!r}; available arrays: {sorted(data.files)}")
        points = np.asarray(data[key], dtype=float)
    if points.ndim != 2:
        raise ValueError(f"{path}:{key} must be rank-2, got shape {points.shape}")
    if points.shape[0] < 3:
        raise ValueError(f"{path}:{key} must contain at least 3 points for a toric embedding audit")
    points = points[: max(3, int(max_points))]
    if not np.isfinite(points).all():
        raise ValueError(f"{path}:{key} contains non-finite values")
    return points


def exponent_points_from_embeddings(points: np.ndarray, *, exponent_dim: int, scale: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert real embedding vectors to a finite exponent set over Z_{\ge 0}^d."""

    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError("embedding points must have shape [num_points, hidden_dim] with at least 3 rows")
    exponent_dim = max(1, min(int(exponent_dim), int(x.shape[0] - 1), int(x.shape[1])))
    centered = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:exponent_dim].T
    mins = projected.min(axis=0, keepdims=True)
    ranges = projected.max(axis=0, keepdims=True) - mins
    if np.any(ranges <= 1e-12):
        raise ValueError("embedding trajectory is rank-deficient after PCA; exact toric exponent audit requires nonzero spread")
    normalized = (projected - mins) / ranges
    exponents = np.rint(normalized * max(2, int(scale))).astype(int)
    unique = np.unique(exponents, axis=0)
    if unique.shape[0] < exponent_dim + 1:
        raise ValueError(
            f"quantized exponent set has only {unique.shape[0]} unique points; need at least {exponent_dim + 1}"
        )
    metadata = {
        "method": "actual_hidden_pca_rank_quantized_nonnegative_exponents",
        "input_shape": list(x.shape),
        "exponent_dim": int(exponent_dim),
        "quantization_scale": int(scale),
        "singular_values": singular_values[:exponent_dim].astype(float).tolist(),
        "unique_exponent_count": int(unique.shape[0]),
        "duplicate_count_removed": int(exponents.shape[0] - unique.shape[0]),
    }
    return unique.astype(int), metadata


def matrix_table(matrix: np.ndarray) -> str:
    rows = []
    for idx, row in enumerate(np.asarray(matrix, dtype=int)):
        rows.append(
            "<tr><td>{}</td>{}</tr>".format(
                idx,
                "".join(f"<td>{int(value)}</td>" for value in row),
            )
        )
    header = "<tr><th>generator</th>{}</tr>".format(
        "".join(f"<th>coord {idx}</th>" for idx in range(matrix.shape[1]))
    )
    return f'<div class="tablewrap"><table>{header}{"".join(rows)}</table></div>'


def metric_rows(payload: dict[str, Any]) -> str:
    derived = payload["macaulay2_toric_ideal"]["commutative_algebra"].get("derived_category_maps", {})
    cone_homology = derived.get("identity_mapping_cone_homology_pruned", {}) if isinstance(derived, dict) else {}
    rows = {
        "record": payload["record_id"],
        "embedding NPZ": payload["embedding_npz"],
        "exponent method": payload["exponent_metadata"]["method"],
        "unique exponents": payload["exponent_metadata"]["unique_exponent_count"],
        "Sage fan dimension": payload["sage_normal_fan"]["toric"].get("dimension"),
        "Sage fan one-dimensional cones": payload["sage_normal_fan"]["toric"].get(
            "num_one_dimensional_cones",
            payload["sage_normal_fan"]["toric"].get("num_rays"),
        ),
        "Macaulay2 toric generators": payload["macaulay2_toric_ideal"]["commutative_algebra"].get(
            "toric_ideal_generator_count"
        ),
        "Macaulay2 resolution length": payload["macaulay2_toric_ideal"]["commutative_algebra"].get(
            "resolution_length"
        ),
        "Macaulay2 projective dimension": payload["macaulay2_toric_ideal"]["commutative_algebra"].get(
            "projective_dimension"
        ),
        "Macaulay2 regularity": payload["macaulay2_toric_ideal"]["commutative_algebra"].get("regularity"),
        "Tropical facets": len(payload.get("tropical_hypersurface", {}).get("tropical", {}).get("facets", [])),
        "Tropical balanced": payload.get("tropical_hypersurface", {}).get("tropical", {}).get("balanced", "n/a"),
        "Chow/Minkowski certified": payload.get("tropical_hypersurface", {}).get("tropical", {}).get(
            "chow_minkowski_weight_certified",
            "n/a",
        ),
        "identity cone H0": cone_homology.get("H0", "n/a"),
        "identity cone H1": cone_homology.get("H1", "n/a"),
        "identity cone H2": cone_homology.get("H2", "n/a"),
    }
    return "".join(
        f'<div class="metric"><span>{html.escape(str(key))}</span><span>{html.escape(str(value))}</span></div>'
        for key, value in rows.items()
    )


def write_record_html(record: dict[str, Any], out: Path) -> None:
    algebra = record["macaulay2_toric_ideal"]["commutative_algebra"]
    derived = algebra.get("derived_category_maps", {})
    ext_modules = algebra.get("module_ext_modules", {})
    tor_modules = algebra.get("module_tor_residue_modules", {})
    differentials = algebra.get("module_resolution_differentials", {})
    square_zero = algebra.get("module_resolution_square_zero", {})
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(record['record_id'])} CAS sidecar</title><style>{CSS}</style></head>
<body><main>
<section class="hero">
  <h1>{html.escape(record['record_id'])}</h1>
  <p>Sage normal fan and Macaulay2 toric ideal computed from the saved embedding payload exponent set.</p>
  <div class="links"><a class="pill" href="../index.html">index</a><a class="pill" href="{html.escape(Path(record['json']).name)}">record JSON</a></div>
</section>
<section class="grid">
  <div class="card"><h2>Exact CAS Metrics</h2>{metric_rows(record)}</div>
  <div class="card"><h2>Exponent Matrix</h2>{matrix_table(np.asarray(record['exponent_points'], dtype=int))}</div>
</section>
<section class="panel">
  <h2>Exact Tropical Multiplicity, Balance, And Chow/Minkowski Audit</h2>
  <pre>{html.escape(json.dumps(record.get('tropical_hypersurface', {}), indent=2, sort_keys=True))}</pre>
</section>
<section class="panel">
  <h2>Sage Normal Fan</h2>
  <pre>{html.escape(json.dumps(record['sage_normal_fan']['toric'], indent=2, sort_keys=True))}</pre>
</section>
<section class="panel">
  <h2>Macaulay2 Toric Ideal</h2>
  <pre>{html.escape(json.dumps(record['macaulay2_toric_ideal']['commutative_algebra'], indent=2, sort_keys=True))}</pre>
</section>
<section class="panel">
  <h2>Macaulay2 Free Resolution</h2>
  <p>The resolution, projective dimension, and regularity below are computed by Macaulay2 from the toric ideal/module generated by the saved embedding exponent set.</p>
  <h3>Ideal resolution</h3>
  <pre>{html.escape(str(algebra.get('free_resolution_raw', '')))}</pre>
  <h3>Cokernel module resolution</h3>
  <pre>{html.escape(str(algebra.get('module_free_resolution_raw', '')))}</pre>
</section>
<section class="panel">
  <h2>Macaulay2 Derived Maps And Modules</h2>
  <p>This section is computed by Macaulay2 from the same free resolution.  It records the resolution differentials, their square-zero checks, the dual complex, Ext/Tor modules, and the mapping cone of the identity chain map as an exact sanity check in the derived category.</p>
  <h3>Resolution differential square checks</h3>
  <pre>{html.escape(json.dumps(square_zero, indent=2, sort_keys=True))}</pre>
  <h3>Resolution differentials</h3>
  <pre>{html.escape(json.dumps(differentials, indent=2, sort_keys=True))}</pre>
  <h3>Dual resolution</h3>
  <pre>{html.escape(str(algebra.get('module_dual_resolution_raw', '')))}</pre>
  <h3>Ext modules</h3>
  <pre>{html.escape(json.dumps(ext_modules, indent=2, sort_keys=True))}</pre>
  <h3>Tor against residue module</h3>
  <pre>{html.escape(json.dumps(tor_modules, indent=2, sort_keys=True))}</pre>
  <h3>Identity chain map and mapping cone</h3>
  <pre>{html.escape(json.dumps(derived, indent=2, sort_keys=True))}</pre>
</section>
</main></body></html>
"""
    out.write_text(body, encoding="utf-8")


def write_index(records: list[dict[str, Any]], output_dir: Path) -> Path:
    cards = []
    for record in records:
        cards.append(
            f"""<div class="card">
  <h2>{html.escape(record['record_id'])}</h2>
  {metric_rows(record)}
  <div class="links"><a class="pill" href="{html.escape(record['html'])}">open page</a><a class="pill" href="{html.escape(record['json'])}">JSON</a></div>
</div>"""
        )
    index = output_dir / "index.html"
    index.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ToricGT Embedding CAS Sidecar</title><style>{CSS}</style></head>
<body><main>
<section class="hero">
  <h1>ToricGT Embedding CAS Sidecar</h1>
  <p>Exact SageMath and Macaulay2 toric computations from saved TokenGT embedding payloads.</p>
  <div class="links"><a class="pill" href="manifest.json">manifest JSON</a></div>
</section>
<section class="grid">{''.join(cards)}</section>
</main></body></html>
""",
        encoding="utf-8",
    )
    return index


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.embedding_manifest)
    manifest = read_json(manifest_path)
    payloads = manifest.get("payloads", [])
    if not isinstance(payloads, list) or not payloads:
        raise ValueError(f"{manifest_path} does not contain embedding payload rows")
    output_dir = Path(args.output_dir)
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    sage = SageToricOracle()
    m2 = Macaulay2TropicalOracle()
    sage.require_available()
    m2.require_available()

    records: list[dict[str, Any]] = []
    for row in payloads[: max(1, int(args.records))]:
        if not isinstance(row, dict):
            raise ValueError(f"invalid manifest row: {row}")
        npz_path = resolve_payload_npz(manifest_path, row)
        points = finite_points_from_npz(npz_path, str(args.embedding_key), max_points=int(args.max_points))
        exponents, exponent_metadata = exponent_points_from_embeddings(
            points,
            exponent_dim=int(args.exponent_dim),
            scale=int(args.quantization_scale),
        )
        record_id = f"record_{int(row.get('record_index', len(records))):03d}"
        biases = [0 for _ in range(int(exponents.shape[0]))]
        tropical_cert = tropical_hypersurface_certificate(
            exponents.astype(int).tolist(),
            biases,
        )
        sage_cert = sage.normal_fan_certificate(
            exponents.astype(int).tolist(),
            timeout_seconds=max(30, int(args.sage_timeout_seconds)),
        ).with_hash()
        m2_cert = m2.toric_ideal_certificate(
            exponents.T.astype(int).tolist(),
            timeout_seconds=max(30, int(args.macaulay2_timeout_seconds)),
        ).with_hash()
        record_json = records_dir / f"{record_id}_cas_sidecar.json"
        record_html = records_dir / f"{record_id}_cas_sidecar.html"
        record = {
            "schema": "toricgt.embedding_cas_sidecar_record.v1",
            "record_id": record_id,
            "embedding_npz": str(npz_path),
            "embedding_key": str(args.embedding_key),
            "exponent_points": exponents.astype(int).tolist(),
            "exponent_matrix_for_macaulay2": exponents.T.astype(int).tolist(),
            "exponent_metadata": exponent_metadata,
            "biases": biases,
            "tropical_hypersurface": tropical_cert,
            "sage_normal_fan": sage_cert,
            "macaulay2_toric_ideal": m2_cert,
            "json": str(record_json),
            "html": f"records/{record_html.name}",
        }
        write_json(record_json, record)
        write_record_html(record, record_html)
        records.append(record)

    summary = {
        "schema": "toricgt.embedding_cas_sidecar_manifest.v1",
        "embedding_manifest": str(manifest_path),
        "records": len(records),
        "sage_backend": sage.info.to_dict(),
        "macaulay2_backend": m2.info.to_dict(),
        "record_pages": [{"record_id": rec["record_id"], "html": rec["html"], "json": f"records/{Path(rec['json']).name}"} for rec in records],
    }
    write_json(output_dir / "manifest.json", summary)
    index = write_index(records, output_dir)
    print(json.dumps({"index_html": str(index), "manifest_json": str(output_dir / "manifest.json"), "records": len(records)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
