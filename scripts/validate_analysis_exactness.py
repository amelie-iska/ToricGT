#!/usr/bin/env python3
"""Validate exactness/provenance for ToricGT analysis bundles.

This script is intentionally strict.  If an analysis manifest claims that an
exact sidecar was emitted, the corresponding exact records, vectorized PH
features, CAS payloads, and browser artifacts must exist.  Missing exact data
is reported as an error rather than treated as a proxy or fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_PH_FEATURES = ("landscape", "persistence_image", "silhouette", "entropy_vector")
REQUIRED_PH_NORMS = ("landscape_norm", "persistence_image_norm", "silhouette_norm", "entropy_vector_norm")
REQUIRED_SAGE_NORMAL_FAN_FIELDS = (
    "num_one_dimensional_cones",
    "fan_one_dimensional_cones",
    "maximal_cones",
    "cone_dimensions",
    "face_incidence",
    "orbit_strata",
    "fan_properties",
    "cone_containment_checks",
    "fan_refinement_checks",
)

REQUIRED_TOKENGT_GRAPH_WANDB_KEYS = (
    "tokengt_graph_data/node_mask_valid_fraction",
    "tokengt_graph_data/edge_mask_valid_fraction",
    "tokengt_graph_data/edge_endpoint_valid_fraction",
    "tokengt_graph_data/causal_directed_edge_count",
    "tokengt_graph_data/causal_edge_fraction",
    "tokengt_graph_data/graph_token_utilization",
    "tokengt_graph_data/topological_dag_fraction",
    "tokengt_graph_data/sequential_chain_fraction",
    "tokengt_graph_data/branch_merge_record_fraction",
    "tokengt_graph_data/invalid_record_count",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (base / path).resolve()


def validate_gudhi_dir(gudhi_dir: Path, *, require_macaulay2: bool = True) -> list[str]:
    errors: list[str] = []
    summary_path = gudhi_dir / "summary.json"
    records_path = gudhi_dir / "records.json"
    index_path = gudhi_dir / "index.html"
    if not summary_path.exists():
        return [f"missing GUDHI summary: {summary_path}"]
    if not records_path.exists():
        errors.append(f"missing GUDHI records JSON: {records_path}")
    if not index_path.exists():
        errors.append(f"missing GUDHI HTML index: {index_path}")
    summary = read_json(summary_path)
    if summary.get("backend") != "gudhi":
        errors.append(f"GUDHI summary backend is not exact gudhi: {summary.get('backend')!r}")
    if int(summary.get("records", 0) or 0) <= 0:
        errors.append("GUDHI summary has no records")
    for key in (
        "mean_finite_field_d_squared_zero",
        "mean_finite_field_exact_at_c1",
        "mean_simplicial_map_valid_fraction",
    ):
        if key not in summary:
            errors.append(f"GUDHI summary missing exact metric {key}")
    if require_macaulay2:
        for key in (
            "mean_macaulay2_d_squared_zero",
            "mean_macaulay2_homogeneous_d1",
            "mean_macaulay2_homogeneous_d2",
            "mean_macaulay2_identity_cone_acyclic",
        ):
            if key not in summary:
                errors.append(f"GUDHI summary missing Macaulay2 metric {key}")
            elif float(summary.get(key, 0.0) or 0.0) < 1.0:
                errors.append(f"Macaulay2 exact metric did not pass: {key}={summary.get(key)!r}")
    if records_path.exists():
        records = read_json(records_path)
        if not isinstance(records, list):
            errors.append("GUDHI records JSON is not a list")
            records = []
        for idx, record in enumerate(records):
            rid = str(record.get("record_id", f"record_{idx}"))
            if record.get("finite_field_chain_audit", {}).get("d_squared_zero") is not True:
                errors.append(f"{rid}: finite-field d^2 check failed or missing")
            if "two_parameter_module" not in record:
                errors.append(f"{rid}: missing two_parameter_module")
            else:
                module = record.get("two_parameter_module", {})
                if not isinstance(module, dict):
                    errors.append(f"{rid}: two_parameter_module is not an object")
                else:
                    structure_summary = module.get("structure_map_summary", {})
                    if not isinstance(structure_summary, dict) or "mean_simplicial_map_valid_fraction" not in structure_summary:
                        errors.append(f"{rid}: missing exact simplicial-map summary")
            if "bigraded_chain_presentation" not in record:
                errors.append(f"{rid}: missing bigraded_chain_presentation")
            retrieval = record.get("ph_retrieval_signature", {})
            if not isinstance(retrieval, dict) or retrieval.get("provenance") != "exact_gudhi_vectorizers_and_exact_simplicial_map_audits":
                errors.append(f"{rid}: missing exact PH retrieval signature provenance")
            if "ph_retrieval_candidates" not in record:
                errors.append(f"{rid}: missing PH retrieval candidates")
            vectorizations = record.get("vectorizations", {})
            if not isinstance(vectorizations, dict) or not vectorizations:
                errors.append(f"{rid}: missing vectorized PH features")
            for dim_key, payload in vectorizations.items():
                if not isinstance(payload, dict):
                    errors.append(f"{rid}: H{dim_key} vectorization payload is not an object")
                    continue
                for key in REQUIRED_PH_FEATURES:
                    if key not in payload:
                        errors.append(f"{rid}: H{dim_key} missing PH feature {key}")
                for key in REQUIRED_PH_NORMS:
                    if key not in payload:
                        errors.append(f"{rid}: H{dim_key} missing PH norm {key}")
            if require_macaulay2:
                m2 = record.get("macaulay2_resolution", {})
                if m2.get("kind") != "macaulay2_bigraded_persistence_resolution":
                    errors.append(f"{rid}: missing exact Macaulay2 bigraded resolution")
                if m2.get("d_squared_zero") is not True:
                    errors.append(f"{rid}: Macaulay2 d^2 check failed or missing")
                if "derived_category_maps" not in m2:
                    errors.append(f"{rid}: missing Macaulay2 derived_category_maps")
    feature_dir = gudhi_dir / "ph_features"
    if not feature_dir.exists():
        errors.append(f"missing PH feature artifact directory: {feature_dir}")
    else:
        json_files = sorted(feature_dir.glob("*_ph_features.json"))
        npz_files = sorted(feature_dir.glob("*_ph_features.npz"))
        if not json_files:
            errors.append(f"missing PH feature JSON artifacts in {feature_dir}")
        if not npz_files:
            errors.append(f"missing PH feature NPZ artifacts in {feature_dir}")
        for path in json_files:
            payload = read_json(path)
            if payload.get("provenance") != "exact_gudhi_vectorizers":
                errors.append(f"{path}: PH feature provenance is not exact_gudhi_vectorizers")
            if "vectorizations" not in payload:
                errors.append(f"{path}: missing vectorizations")
    return errors


def validate_embedding_cas_dir(cas_dir: Path) -> list[str]:
    errors: list[str] = []
    cas_manifest = cas_dir / "manifest.json"
    cas_index = cas_dir / "index.html"
    if not cas_manifest.exists():
        return [f"missing embedding CAS manifest: {cas_manifest}"]
    if not cas_index.exists():
        errors.append(f"missing embedding CAS index: {cas_index}")
    payload = read_json(cas_manifest)
    if int(payload.get("records", 0) or 0) <= 0:
        errors.append("embedding CAS sidecar has no records")
    if int(payload.get("failures", 0) or 0) != 0:
        errors.append(f"embedding CAS sidecar reports failures={payload.get('failures')!r}")
    for backend_key, expected in (("sage_backend", "exact_cas/sage"), ("macaulay2_backend", "exact_cas/macaulay2")):
        backend = payload.get(backend_key, {})
        if not isinstance(backend, dict) or backend.get("provenance") != expected or backend.get("available") is not True:
            errors.append(f"embedding CAS {backend_key} is not available exact backend: {backend!r}")
    for idx, row in enumerate(payload.get("record_pages", []) or []):
        if not isinstance(row, dict):
            errors.append(f"embedding CAS record_pages[{idx}] is not an object")
            continue
        record_json = _resolve(cas_dir, row.get("json"))
        record_html = _resolve(cas_dir, row.get("html"))
        rid = str(row.get("record_id", f"record_{idx}"))
        if record_json is None or not record_json.exists():
            errors.append(f"{rid}: missing embedding CAS record JSON")
            continue
        if record_html is None or not record_html.exists():
            errors.append(f"{rid}: missing embedding CAS record HTML")
        record = read_json(record_json)
        sage = record.get("sage_normal_fan", {})
        if sage.get("provenance") != "exact_cas/sage":
            errors.append(f"{rid}: Sage normal fan provenance is not exact_cas/sage")
        toric = sage.get("toric", {}) if isinstance(sage, dict) else {}
        if not isinstance(toric, dict):
            errors.append(f"{rid}: Sage normal fan toric payload is not an object")
            toric = {}
        for key in REQUIRED_SAGE_NORMAL_FAN_FIELDS:
            if key not in toric:
                errors.append(f"{rid}: Sage normal fan missing {key}")
        if toric.get("num_one_dimensional_cones") != toric.get("num_rays"):
            errors.append(f"{rid}: Sage one-dimensional cone count differs from compatibility ray count")
        if toric.get("fan_one_dimensional_cones") != toric.get("fan_rays"):
            errors.append(f"{rid}: Sage one-dimensional cone list differs from compatibility ray list")
        if toric.get("cone_containment_checks", {}).get("all_maximal_indices_are_one_dimensional_cones") is not True:
            errors.append(f"{rid}: Sage cone containment check failed")
        if toric.get("fan_properties", {}).get("is_complete") is not True:
            errors.append(f"{rid}: Sage normal fan completeness check failed")
        m2 = record.get("macaulay2_toric_ideal", {})
        if m2.get("provenance") != "exact_cas/macaulay2":
            errors.append(f"{rid}: Macaulay2 toric ideal provenance is not exact_cas/macaulay2")
        algebra = m2.get("commutative_algebra", {}) if isinstance(m2, dict) else {}
        if not isinstance(algebra, dict) or not algebra.get("free_resolution_raw"):
            errors.append(f"{rid}: missing Macaulay2 free resolution")
        if isinstance(algebra, dict) and "derived_category_maps" not in algebra:
            errors.append(f"{rid}: missing Macaulay2 derived category maps")
    return errors


def validate_tokengt_graph_data_dir(graph_dir: Path) -> list[str]:
    errors: list[str] = []
    summary_path = graph_dir / "summary.json"
    records_path = graph_dir / "records.json"
    index_path = graph_dir / "index.html"
    invalid_path = graph_dir / "invalid_records.json"
    if not summary_path.exists():
        return [f"missing TokenGT graph-data summary: {summary_path}"]
    if not records_path.exists():
        errors.append(f"missing TokenGT graph-data records JSON: {records_path}")
    if not index_path.exists():
        errors.append(f"missing TokenGT graph-data HTML index: {index_path}")
    if not invalid_path.exists():
        errors.append(f"missing TokenGT graph-data invalid-records JSON: {invalid_path}")
    summary = read_json(summary_path)
    if summary.get("schema") != "toricgt.tokengt_graph_data_validation.v1":
        errors.append(f"TokenGT graph-data summary has unexpected schema: {summary.get('schema')!r}")
    if int(summary.get("records", 0) or 0) <= 0:
        errors.append("TokenGT graph-data validation has no valid records")
    if int(summary.get("invalid_records", 0) or 0) != 0:
        errors.append(f"TokenGT graph-data validation has invalid_records={summary.get('invalid_records')!r}")
    for key in (
        "mean_node_mask_valid",
        "mean_edge_mask_valid",
        "mean_edge_endpoint_valid_fraction",
        "mean_graph_token_utilization",
        "mean_causal_directed_edge_count",
        "mean_causal_edge_fraction",
    ):
        if key not in summary:
            errors.append(f"TokenGT graph-data summary missing metric {key}")
    if summary.get("all_masks_and_endpoints_valid") is not True:
        errors.append("TokenGT graph-data masks/endpoints are not all valid")
    wandb = summary.get("wandb_metrics", {})
    if not isinstance(wandb, dict):
        errors.append("TokenGT graph-data summary missing wandb_metrics object")
        wandb = {}
    for key in REQUIRED_TOKENGT_GRAPH_WANDB_KEYS:
        if key not in wandb:
            errors.append(f"TokenGT graph-data W&B payload missing {key}")
    if records_path.exists():
        records = read_json(records_path)
        if not isinstance(records, list):
            errors.append("TokenGT graph-data records JSON is not a list")
            records = []
        for idx, record in enumerate(records):
            rid = str(record.get("record_id", f"record_{idx}"))
            if record.get("schema") != "toricgt.tokengt_graph_data_record.v1":
                errors.append(f"{rid}: unexpected TokenGT graph-data record schema")
            if record.get("node_mask_valid") is not True:
                errors.append(f"{rid}: node mask invalid")
            if record.get("edge_mask_valid") is not True:
                errors.append(f"{rid}: edge mask invalid")
            if float(record.get("edge_endpoint_valid_fraction", 0.0) or 0.0) < 1.0:
                errors.append(f"{rid}: edge endpoint validity below 1")
            if float(record.get("graph_token_utilization", 0.0) or 0.0) <= 0.0:
                errors.append(f"{rid}: graph token utilization is zero")
            if "causal_rank_kind" not in record:
                errors.append(f"{rid}: missing causal_rank_kind")
    return errors


def validate_inference_manifest(path: Path, *, require_macaulay2: bool = True) -> list[str]:
    errors: list[str] = []
    manifest = read_json(path)
    base = path.parent.resolve()
    if manifest.get("mode") != "tokengt_graph_inference_with_optional_geometry":
        errors.append(f"unexpected inference mode: {manifest.get('mode')!r}")
    if manifest.get("gudhi_persistence_enabled"):
        gudhi_dir = _resolve(base, manifest.get("gudhi_persistence_output_dir"))
        if gudhi_dir is None:
            errors.append("GUDHI persistence enabled but output dir missing")
        else:
            errors.extend(validate_gudhi_dir(gudhi_dir, require_macaulay2=require_macaulay2))
            ph_dir = _resolve(base, manifest.get("ph_feature_output_dir"))
            if ph_dir is None:
                errors.append("GUDHI persistence enabled but PH feature output dir missing from manifest")
            elif not ph_dir.exists():
                errors.append(f"PH feature output dir does not exist: {ph_dir}")
            if "ph_feature_artifacts" not in manifest:
                errors.append("GUDHI persistence enabled but manifest missing ph_feature_artifacts")
    if manifest.get("embedding_cas_sidecar_enabled"):
        cas_dir = _resolve(base, manifest.get("embedding_cas_sidecar_output_dir"))
        if cas_dir is None:
            errors.append("embedding CAS sidecar enabled but output dir missing")
        else:
            errors.extend(validate_embedding_cas_dir(cas_dir))
    if manifest.get("geometry_enabled"):
        geom = _resolve(base, manifest.get("geometry_output_dir"))
        if geom is None:
            errors.append("geometry enabled but output dir missing")
        else:
            if not (geom / "plot_family_manifest.json").exists():
                errors.append(f"missing geometry plot family manifest: {geom / 'plot_family_manifest.json'}")
            if manifest.get("embedding_payloads_enabled") and not (geom / "embeddings" / "manifest.json").exists():
                errors.append(f"missing embedding payload manifest: {geom / 'embeddings' / 'manifest.json'}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Inference output JSON or GUDHI output directory.")
    parser.add_argument("--allow-missing-macaulay2", action="store_true", help="Do not require Macaulay2-derived resolution fields.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    require_macaulay2 = not bool(args.allow_missing_macaulay2)
    if path.is_dir():
        manifest = path / "manifest.json"
        summary = path / "summary.json"
        if manifest.exists():
            try:
                manifest_payload = read_json(manifest)
            except Exception:
                manifest_payload = {}
            if manifest_payload.get("schema") == "toricgt.embedding_cas_sidecar_manifest.v1":
                errors = validate_embedding_cas_dir(path)
                kind = "embedding_cas_dir"
            else:
                errors = validate_gudhi_dir(path, require_macaulay2=require_macaulay2)
                kind = "gudhi_dir"
        elif summary.exists():
            try:
                summary_payload = read_json(summary)
            except Exception:
                summary_payload = {}
            if summary_payload.get("schema") == "toricgt.tokengt_graph_data_validation.v1":
                errors = validate_tokengt_graph_data_dir(path)
                kind = "tokengt_graph_data_dir"
            else:
                errors = validate_gudhi_dir(path, require_macaulay2=require_macaulay2)
                kind = "gudhi_dir"
        else:
            errors = [f"directory is neither GUDHI nor embedding-CAS analysis output: {path}"]
            kind = "unknown_dir"
    else:
        errors = validate_inference_manifest(path, require_macaulay2=require_macaulay2)
        kind = "inference_manifest"
    payload = {
        "schema": "toricgt.analysis_exactness_validation.v1",
        "kind": kind,
        "path": str(path),
        "ok": not errors,
        "errors": errors,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif errors:
        for error in errors:
            print(f"ERROR: {error}")
    else:
        print(f"OK: exact analysis validation passed for {path}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
