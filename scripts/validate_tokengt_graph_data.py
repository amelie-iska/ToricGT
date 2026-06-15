#!/usr/bin/env python3
"""Validate TokenGT graph-structured data before training or analysis.

The report is intentionally about the actual graph-token path used by
``CuratedGraphIterableDataset``: records are converted through
``record_to_graph_json``/``fineweb_tokens_to_graph_json`` and then padded by
``graph_json_to_item``.  It reports causal directed structure, mask validity,
edge endpoint validity, and graph-token utilization.  Invalid records are a
hard failure unless ``--allow-invalid-records`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

from toricgt.config import ModelConfig
from toricgt.got_trajectory import got_dag_summary_np
from toricgt.graph_dataset import (
    TEXT_COLUMNS,
    expand_dataset_paths,
    fineweb_tokens_to_graph_json,
    graph_json_to_item,
    load_competition_token_memmap,
    record_to_graph_json,
)


SCHEMA = "toricgt.tokengt_graph_data_validation.v1"


def _read_jsonl_records(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and emitted >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            record.setdefault("_source_path", str(path))
            record.setdefault("_source_row", line_number - 1)
            emitted += 1
            yield record


def _read_parquet_records(path: Path, limit: int | None, batch_size: int) -> Iterable[dict[str, Any]]:
    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    columns = [column for column in TEXT_COLUMNS if column in available]
    identity_columns = [column for column in ("record_id", "dataset", "task_family", "source", "content_hash") if column in available]
    columns = sorted(set(columns + identity_columns))
    if not columns:
        columns = list(parquet_file.schema_arrow.names)
    emitted = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        if limit is not None and emitted >= limit:
            break
        batch_dict = batch.to_pydict()
        row_count = len(next(iter(batch_dict.values()))) if batch_dict else 0
        for row_idx in range(row_count):
            if limit is not None and emitted >= limit:
                break
            record = {column: batch_dict[column][row_idx] for column in batch_dict}
            record.setdefault("_source_path", str(path))
            record.setdefault("_source_row", emitted)
            emitted += 1
            yield record


def _iter_fineweb_graphs(
    path: Path,
    *,
    cfg: ModelConfig,
    tokens_per_graph: int,
    stride_tokens: int,
    limit: int | None,
) -> Iterable[tuple[str, np.ndarray, dict[str, Any]]]:
    tokens = load_competition_token_memmap(path)
    chunk = max(2, int(tokens_per_graph))
    stride = max(1, int(stride_tokens))
    if int(tokens.shape[0]) < chunk:
        starts: Iterable[int] = [0]
    else:
        starts = range(0, int(tokens.shape[0]) - chunk + 1, stride)
    emitted = 0
    for start in starts:
        if limit is not None and emitted >= limit:
            break
        end = min(int(start) + chunk, int(tokens.shape[0]))
        token_slice = np.asarray(tokens[int(start) : end], dtype=np.int32)
        record_id = f"{path.stem}:{int(start)}:{int(end)}"
        graph_json = fineweb_tokens_to_graph_json(
            token_slice,
            dataset="fineweb10B_sp1024",
            record_id=record_id,
            max_nodes=cfg.max_nodes,
        )
        metadata = {
            "_source_path": str(path),
            "_source_row": emitted,
            "record_id": record_id,
            "dataset": "fineweb10B_sp1024",
            "task_family": "fineweb_language_modeling_graph",
            "conversion_source": "fineweb_tokens_to_graph_json",
        }
        emitted += 1
        yield graph_json, token_slice, metadata


def _valid_edge_array(payload: dict[str, Any], max_nodes: int) -> tuple[np.ndarray, int, int]:
    nodes = list(payload.get("nodes") or [])[:max_nodes]
    raw_edges = list(payload.get("edges") or [])
    node_ids = {str(node.get("id", idx)): idx for idx, node in enumerate(nodes)}
    valid: list[tuple[int, int]] = []
    invalid = 0
    for edge in raw_edges:
        src = node_ids.get(str(edge.get("source")))
        dst = node_ids.get(str(edge.get("target")))
        if src is None or dst is None or src == dst:
            invalid += 1
            continue
        valid.append((int(src), int(dst)))
    return np.asarray(valid, dtype=np.int64).reshape(-1, 2), len(raw_edges), invalid


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(out):
        return 0.0
    return out


def audit_graph_json(
    graph_json: str,
    *,
    cfg: ModelConfig,
    lm_tokens: np.ndarray | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(graph_json or "{}")
    item = graph_json_to_item(graph_json, cfg, lm_tokens=lm_tokens)
    valid_edges, raw_edge_count, invalid_edge_count = _valid_edge_array(payload, cfg.max_nodes)
    node_count = int(min(len(payload.get("nodes") or []), cfg.max_nodes))
    raw_node_count = int(len(payload.get("nodes") or []))
    node_mask_count = int(item.graph.node_mask.sum().item())
    edge_mask_count = int(item.graph.edge_mask.sum().item())
    expected_node_mask_count = max(1, node_count)
    expected_edge_mask_count = max(1, min(int(valid_edges.shape[0]), cfg.max_edges))
    edge_index = item.graph.edge_index[item.graph.edge_mask].cpu().numpy().astype(np.int64).reshape(-1, 2)
    endpoint_valid = (
        (edge_index[:, 0] >= 0)
        & (edge_index[:, 1] >= 0)
        & (edge_index[:, 0] < max(1, node_mask_count))
        & (edge_index[:, 1] < max(1, node_mask_count))
    )
    endpoint_valid_fraction = float(endpoint_valid.mean()) if endpoint_valid.size else 1.0
    ranks = item.graph.node_causal_rank[: max(1, node_mask_count)].cpu().numpy().astype(np.int64)
    causal_edges = 0
    for src, dst in edge_index:
        if int(src) < ranks.shape[0] and int(dst) < ranks.shape[0] and int(ranks[int(src)]) < int(ranks[int(dst)]):
            causal_edges += 1
    causal_edge_fraction = float(causal_edges / max(1, edge_index.shape[0]))
    dag = got_dag_summary_np(max(1, node_count), valid_edges if valid_edges.size else None)
    mask_ok = (
        node_mask_count == expected_node_mask_count
        and edge_mask_count == expected_edge_mask_count
        and bool(np.all(np.isfinite(item.graph.node_features.cpu().numpy())))
        and bool(np.all(np.isfinite(item.graph.edge_features.cpu().numpy())))
        and endpoint_valid_fraction == 1.0
    )
    graph_token_utilization = (node_mask_count + edge_mask_count) / max(1, int(cfg.max_nodes) + int(cfg.max_edges))
    record_id = str(payload.get("record_id") or metadata.get("record_id") or metadata.get("_source_row") or "")
    dataset = str(payload.get("dataset") or metadata.get("dataset") or "")
    task_family = str(payload.get("task_family") or metadata.get("task_family") or "")
    causal_rank_kind = str((item.metadata or {}).get("causal_rank_kind") or "")
    return {
        "schema": "toricgt.tokengt_graph_data_record.v1",
        "record_id": record_id,
        "dataset": dataset,
        "task_family": task_family,
        "source_path": metadata.get("_source_path", ""),
        "source_row": metadata.get("_source_row", ""),
        "conversion_source": metadata.get("conversion_source", "record_to_graph_json"),
        "trajectory_kind": str(payload.get("trajectory_kind") or ""),
        "causal_rank_kind": causal_rank_kind,
        "raw_node_count": raw_node_count,
        "node_count": node_count,
        "raw_edge_count": raw_edge_count,
        "valid_raw_edge_count": int(valid_edges.shape[0]),
        "invalid_raw_edge_count": int(invalid_edge_count),
        "node_mask_count": node_mask_count,
        "edge_mask_count": edge_mask_count,
        "node_mask_valid": node_mask_count == expected_node_mask_count,
        "edge_mask_valid": edge_mask_count == expected_edge_mask_count,
        "edge_endpoint_valid_fraction": endpoint_valid_fraction,
        "causal_directed_edge_count": int(causal_edges),
        "causal_edge_fraction": causal_edge_fraction,
        "graph_token_utilization": float(graph_token_utilization),
        "branch_count": _safe_float(dag.get("branch_count")),
        "merge_count": _safe_float(dag.get("merge_count")),
        "linear_chain_fraction": _safe_float(dag.get("linear_chain_fraction")),
        "branch_merge_edge_fraction": _safe_float(dag.get("branch_merge_edge_fraction")),
        "simplex_edge_density": _safe_float(dag.get("simplex_edge_density")),
        "mask_and_endpoint_valid": bool(mask_ok),
    }


def summarize_records(records: list[dict[str, Any]], invalid_records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)

    def mean(key: str) -> float:
        if not records:
            return 0.0
        return float(np.mean([_safe_float(record.get(key)) for record in records]))

    def frac_equal(key: str, value: str) -> float:
        if not records:
            return 0.0
        return float(np.mean([str(record.get(key) or "") == value for record in records]))

    summary = {
        "schema": SCHEMA,
        "records": total,
        "invalid_records": len(invalid_records),
        "datasets": sorted({str(record.get("dataset") or "") for record in records}),
        "task_families": sorted({str(record.get("task_family") or "") for record in records}),
        "mean_node_count": mean("node_count"),
        "mean_edge_count": mean("edge_mask_count"),
        "mean_raw_edge_count": mean("raw_edge_count"),
        "mean_valid_raw_edge_count": mean("valid_raw_edge_count"),
        "mean_invalid_raw_edge_count": mean("invalid_raw_edge_count"),
        "mean_graph_token_utilization": mean("graph_token_utilization"),
        "mean_node_mask_valid": mean("node_mask_valid"),
        "mean_edge_mask_valid": mean("edge_mask_valid"),
        "mean_edge_endpoint_valid_fraction": mean("edge_endpoint_valid_fraction"),
        "mean_causal_directed_edge_count": mean("causal_directed_edge_count"),
        "mean_causal_edge_fraction": mean("causal_edge_fraction"),
        "topological_dag_fraction": frac_equal("causal_rank_kind", "topological_dag"),
        "sequential_chain_fraction": frac_equal("causal_rank_kind", "linear_causal"),
        "random_cycle_fraction": frac_equal("causal_rank_kind", "random_cycle"),
        "random_no_edges_fraction": frac_equal("causal_rank_kind", "random_no_edges"),
        "branch_merge_record_fraction": float(np.mean([_safe_float(record.get("branch_merge_edge_fraction")) > 0.0 for record in records]))
        if records
        else 0.0,
        "all_masks_and_endpoints_valid": bool(records and not invalid_records and all(record.get("mask_and_endpoint_valid") for record in records)),
    }
    summary["wandb_metrics"] = {
        "tokengt_graph_data/node_mask_valid_fraction": summary["mean_node_mask_valid"],
        "tokengt_graph_data/edge_mask_valid_fraction": summary["mean_edge_mask_valid"],
        "tokengt_graph_data/edge_endpoint_valid_fraction": summary["mean_edge_endpoint_valid_fraction"],
        "tokengt_graph_data/causal_directed_edge_count": summary["mean_causal_directed_edge_count"],
        "tokengt_graph_data/causal_edge_fraction": summary["mean_causal_edge_fraction"],
        "tokengt_graph_data/graph_token_utilization": summary["mean_graph_token_utilization"],
        "tokengt_graph_data/topological_dag_fraction": summary["topological_dag_fraction"],
        "tokengt_graph_data/sequential_chain_fraction": summary["sequential_chain_fraction"],
        "tokengt_graph_data/branch_merge_record_fraction": summary["branch_merge_record_fraction"],
        "tokengt_graph_data/invalid_record_count": float(summary["invalid_records"]),
    }
    return summary


def write_html(output_dir: Path, summary: dict[str, Any], records: list[dict[str, Any]], invalid: list[dict[str, Any]]) -> None:
    css = """
    body{margin:0;background:#0b1020;color:#e5edf7;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    main{max-width:1240px;margin:0 auto;padding:28px}
    h1{font-size:30px;margin:0 0 8px} h2{font-size:20px;margin-top:28px}
    .sub{color:#9fb1c9;margin-bottom:22px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
    .card{border:1px solid #26364f;background:#111a2c;border-radius:8px;padding:14px}
    .label{color:#9fb1c9;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
    .value{font-size:24px;font-weight:700;margin-top:5px}
    table{width:100%;border-collapse:collapse;background:#0f1729;border:1px solid #26364f;border-radius:8px;overflow:hidden}
    th,td{border-bottom:1px solid #22324a;padding:8px 10px;text-align:left;font-size:13px;vertical-align:top}
    th{background:#172238;color:#b9c8dd;font-weight:700}
    tr:last-child td{border-bottom:0}
    code{color:#9ee7ff}.ok{color:#73e6a5}.bad{color:#ff8b8b}
    a{color:#8dd6ff}
    """
    metric_cards = [
        ("Records", summary.get("records")),
        ("Invalid", summary.get("invalid_records")),
        ("Node Mask Valid", f"{100.0 * _safe_float(summary.get('mean_node_mask_valid')):.1f}%"),
        ("Edge Mask Valid", f"{100.0 * _safe_float(summary.get('mean_edge_mask_valid')):.1f}%"),
        ("Endpoint Valid", f"{100.0 * _safe_float(summary.get('mean_edge_endpoint_valid_fraction')):.1f}%"),
        ("Graph Token Util.", f"{100.0 * _safe_float(summary.get('mean_graph_token_utilization')):.1f}%"),
        ("Causal Edge Frac.", f"{100.0 * _safe_float(summary.get('mean_causal_edge_fraction')):.1f}%"),
        ("Branch/Merge Rows", f"{100.0 * _safe_float(summary.get('branch_merge_record_fraction')):.1f}%"),
    ]
    rows = []
    for record in records[:200]:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(record.get('record_id')))}</code></td>"
            f"<td>{html.escape(str(record.get('dataset')))}</td>"
            f"<td>{html.escape(str(record.get('trajectory_kind')))}</td>"
            f"<td>{html.escape(str(record.get('causal_rank_kind')))}</td>"
            f"<td>{int(record.get('node_mask_count', 0))}/{int(record.get('edge_mask_count', 0))}</td>"
            f"<td>{_safe_float(record.get('edge_endpoint_valid_fraction')):.3f}</td>"
            f"<td>{_safe_float(record.get('causal_edge_fraction')):.3f}</td>"
            f"<td>{_safe_float(record.get('graph_token_utilization')):.3f}</td>"
            f"<td>{'yes' if record.get('mask_and_endpoint_valid') else 'no'}</td>"
            "</tr>"
        )
    invalid_rows = []
    for row in invalid[:100]:
        invalid_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('source_path')))}</td>"
            f"<td>{html.escape(str(row.get('source_row')))}</td>"
            f"<td class='bad'>{html.escape(str(row.get('error')))}</td>"
            "</tr>"
        )
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>TokenGT Graph Data Validation</title><style>{css}</style></head>
<body><main>
<h1>TokenGT Graph Data Validation</h1>
<div class="sub">Schema <code>{SCHEMA}</code>. This report audits the actual graph-token conversion, padding masks, causal ranks, and directed edge structure before training.</div>
<section class="grid">
{''.join(f"<div class='card'><div class='label'>{html.escape(str(label))}</div><div class='value'>{html.escape(str(value))}</div></div>" for label, value in metric_cards)}
</section>
<h2>W&amp;B Metric Payload</h2>
<pre class="card">{html.escape(json.dumps(summary.get('wandb_metrics', {}), indent=2, sort_keys=True))}</pre>
<h2>Records</h2>
<table><thead><tr><th>Record</th><th>Dataset</th><th>Trajectory</th><th>Causal Rank</th><th>Node/Edge Masks</th><th>Endpoint Valid</th><th>Causal Edge</th><th>Utilization</th><th>Valid</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table>
<h2>Invalid Records</h2>
<table><thead><tr><th>Source</th><th>Row</th><th>Error</th></tr></thead><tbody>
{''.join(invalid_rows) if invalid_rows else "<tr><td colspan='3' class='ok'>No invalid records.</td></tr>"}
</tbody></table>
</main></body></html>"""
    (output_dir / "index.html").write_text(body, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Input JSONL/Parquet/FineWeb .bin path, directory, or glob. Repeatable.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary.json, records.json, and index.html.")
    parser.add_argument("--max-records", type=int, default=256, help="Maximum records per entire validation run.")
    parser.add_argument("--parquet-batch-size", type=int, default=512)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--max-edges", type=int, default=1024)
    parser.add_argument("--fineweb-tokens-per-graph", type=int, default=1024)
    parser.add_argument("--fineweb-stride-tokens", type=int, default=1024)
    parser.add_argument("--allow-invalid-records", action="store_true", help="Write the report even if invalid records are encountered.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig(max_nodes=int(args.max_nodes), max_edges=int(args.max_edges))
    paths = expand_dataset_paths(args.input)
    if not paths:
        raise SystemExit("no input paths matched")
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    remaining = int(args.max_records) if int(args.max_records) > 0 else None

    for path in paths:
        if remaining is not None and remaining <= 0:
            break
        per_path_limit = remaining
        try:
            if path.suffix == ".jsonl":
                iterator = (
                    (record_to_graph_json(record, cfg), None, {**record, "conversion_source": "graph_json" if record.get("graph_json") else "record_to_graph_json"})
                    for record in _read_jsonl_records(path, per_path_limit)
                )
            elif path.suffix == ".bin":
                iterator = _iter_fineweb_graphs(
                    path,
                    cfg=cfg,
                    tokens_per_graph=int(args.fineweb_tokens_per_graph),
                    stride_tokens=int(args.fineweb_stride_tokens),
                    limit=per_path_limit,
                )
            else:
                iterator = (
                    (record_to_graph_json(record, cfg), None, {**record, "conversion_source": "graph_json" if record.get("graph_json") else "record_to_graph_json"})
                    for record in _read_parquet_records(path, per_path_limit, int(args.parquet_batch_size))
                )
            for graph_json, lm_tokens, metadata in iterator:
                try:
                    records.append(audit_graph_json(graph_json, cfg=cfg, lm_tokens=lm_tokens, metadata=metadata))
                except Exception as exc:
                    invalid.append(
                        {
                            "source_path": str(metadata.get("_source_path") or path),
                            "source_row": metadata.get("_source_row", ""),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        break
        except Exception as exc:
            invalid.append({"source_path": str(path), "source_row": "", "error": f"{type(exc).__name__}: {exc}"})

    summary = summarize_records(records, invalid)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "records.json").write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "invalid_records.json").write_text(json.dumps(invalid, indent=2, sort_keys=True), encoding="utf-8")
    write_html(output_dir, summary, records, invalid)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if invalid and not args.allow_invalid_records:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
