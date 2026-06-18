#!/usr/bin/env python3
"""Run the strict full-analysis pass after one OAI-sidecar gate attempt."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adaptive_bpb_annealing import build_adaptive_decision


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

VAL_RE = re.compile(r"step:(?P<step>\d+)/(?P<total>\d+) val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+)")
TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+) train_loss:(?P<loss>[0-9.]+)"
    r"(?: train_bpb:(?P<train_bpb>[0-9.eE+-]+) train_bpt:(?P<train_bpt>[0-9.eE+-]+))?.*?"
    r"(?:graph_lm_loss:(?P<graph_lm_loss>[0-9.eE+-]+) graph_lm_bpb:(?P<graph_lm_bpb>[0-9.eE+-]+) "
    r"graph_lm_w:(?P<graph_lm_weight>[0-9.eE+-]+) )?.*?"
    r"(?:oai_gfn:(?P<oai_gflownet_loss>[0-9.eE+-]+) gfn_H:(?P<oai_gflownet_entropy>[0-9.eE+-]+) "
    r"gfn_R:(?P<oai_gflownet_reward>[0-9.eE+-]+) )?.*?"
    r"(?:mtp:(?P<oai_mtp_loss>[0-9.eE+-]+) mtp_w:(?P<oai_mtp_weight>[0-9.eE+-]+) )?.*?"
    r"sidecar_loss:(?P<sidecar>[0-9.eE+-]+).*?"
    r"graphcg:(?P<graphcg>[0-9.eE+-]+).*?"
    r"analogy:(?P<analogy>[0-9.eE+-]+).*?"
    r"tokengt_graph:(?P<tokengt>[0-9.eE+-]+).*?"
    r"memory:(?P<memory>[0-9.eE+-]+)"
    r"(?:.*?toric:(?P<toric>[0-9.eE+-]+).*?(?:vb1d|vb):(?P<vector_bundle_1d_cone>[0-9.eE+-]+).*?"
    r"bgg:(?P<bgg>[0-9.eE+-]+).*?koszul:(?P<koszul>[0-9.eE+-]+).*?cca:(?P<cca>[0-9.eE+-]+))?"
)
FINAL_RE = re.compile(r"final_int8_zlib_roundtrip_exact val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+)")
SIZE_RE = re.compile(r"Total submission size int8\+zlib: (?P<size>\d+) bytes")
CHECKPOINT_RE = re.compile(r"checkpoint_saved:(?P<path>.*?) step:(?P<step>\d+) val_bpb:(?P<bpb>[-+0-9.eE]+|None)")
OAI_GFLOWNET_RE = re.compile(
    r"oai_gfn:(?P<oai_gflownet_loss>[0-9.eE+-]+)\s+"
    r"gfn_H:(?P<oai_gflownet_entropy>[0-9.eE+-]+)\s+"
    r"gfn_R:(?P<oai_gflownet_reward>[0-9.eE+-]+)"
)
OAI_MTP_RE = re.compile(
    r"mtp:(?P<oai_mtp_loss>[0-9.eE+-]+)\s+mtp_w:(?P<oai_mtp_weight>[0-9.eE+-]+)"
)


@dataclass
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    stdout_path: str
    stderr_path: str


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-path", default="", help="W&B path entity/project/run_id.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path", default="amelie-iska/parameter-golf/data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument(
        "--graph-train-glob",
        default="/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/curated_hf_shards/train/*.parquet",
    )
    parser.add_argument("--records", type=int, default=3)
    parser.add_argument("--embedding-records", type=int, default=3)
    parser.add_argument("--embedding-seq-len", type=int, default=256)
    parser.add_argument(
        "--embedding-device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for OAI hidden-state extraction. Use cpu when training is already occupying the GPU.",
    )
    parser.add_argument("--branching-max-nodes", type=int, default=260)
    parser.add_argument("--gudhi-max-points", type=int, default=18)
    parser.add_argument(
        "--cas-max-points",
        type=int,
        default=10,
        help="Bound the exact Sage/Macaulay2 toric embedding audit. This remains exact on the selected embedding subset.",
    )
    parser.add_argument("--cas-macaulay2-timeout-seconds", type=int, default=900)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quick", action="store_true", help="Use tiny settings for smoke tests.")
    return parser.parse_args()


def safe_float(value: str | None) -> float | None:
    if value is None or value == "None":
        return None
    try:
        return float(value)
    except Exception:
        return None


def parse_log(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "train_rows": [],
        "val_rows": [],
        "checkpoint_paths": [],
        "artifact_bytes": None,
        "final_int8_loss": None,
        "final_int8_bpb": None,
    }
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    for line in text.splitlines():
        if match := TRAIN_RE.search(line):
            row = {}
            for key, raw in match.groupdict().items():
                if raw is None:
                    continue
                row[key] = safe_float(raw)
            if gfn_match := OAI_GFLOWNET_RE.search(line):
                for key, raw in gfn_match.groupdict().items():
                    row[key] = safe_float(raw)
            if mtp_match := OAI_MTP_RE.search(line):
                for key, raw in mtp_match.groupdict().items():
                    row[key] = safe_float(raw)
            metrics["train_rows"].append(row)
        if match := VAL_RE.search(line):
            metrics["val_rows"].append({key: safe_float(value) for key, value in match.groupdict().items()})
        if match := FINAL_RE.search(line):
            metrics["final_int8_loss"] = safe_float(match.group("loss"))
            metrics["final_int8_bpb"] = safe_float(match.group("bpb"))
        if match := SIZE_RE.search(line):
            metrics["artifact_bytes"] = int(match.group("size"))
        if match := CHECKPOINT_RE.search(line):
            metrics["checkpoint_paths"].append(
                {"path": match.group("path"), "step": int(match.group("step")), "val_bpb": safe_float(match.group("bpb"))}
            )
    latest_train = metrics["train_rows"][-1] if metrics["train_rows"] else {}
    latest_val = metrics["val_rows"][-1] if metrics["val_rows"] else {}
    metrics["latest_train"] = latest_train
    metrics["latest_val"] = latest_val
    metrics["selected_bpb"] = metrics.get("final_int8_bpb") or latest_val.get("bpb")
    metrics["train_log"] = str(path)
    return metrics


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = []
    for row in metrics.get("train_rows", []):
        out = {"kind": "train", **row}
        rows.append(out)
    for row in metrics.get("val_rows", []):
        out = {"kind": "validation", **row}
        rows.append(out)
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        for idx in order[start:end]:
            ranks[idx] = rank
        start = end
    return ranks


def pearson_corr(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    dx = [value - x_mean for value in x]
    dy = [value - y_mean for value in y]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denom <= 1e-12:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def spearman_corr(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    return pearson_corr(rank_values(x), rank_values(y))


def compute_metric_correlations(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Rank correlations among logged train metrics after one gate attempt."""

    train_rows = [row for row in metrics.get("train_rows", []) if isinstance(row, dict)]
    numeric_keys = sorted(
        key
        for key in {key for row in train_rows for key in row}
        if key not in {"step", "total"}
        and sum(finite_number(row.get(key)) is not None for row in train_rows) >= 3
    )
    targets = [target for target in ("train_bpb", "loss", "sidecar") if target in numeric_keys]
    rows: list[dict[str, Any]] = []
    for target in targets:
        for metric in numeric_keys:
            if metric == target:
                continue
            x: list[float] = []
            y: list[float] = []
            steps: list[float] = []
            for row in train_rows:
                xv = finite_number(row.get(metric))
                yv = finite_number(row.get(target))
                step = finite_number(row.get("step"))
                if xv is None or yv is None:
                    continue
                x.append(xv)
                y.append(yv)
                if step is not None:
                    steps.append(step)
            if len(x) < 3:
                continue
            pearson = pearson_corr(x, y)
            spearman = spearman_corr(x, y)
            if pearson is None and spearman is None:
                continue
            rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "n": len(x),
                    "first_step": steps[0] if steps else None,
                    "last_step": steps[-1] if steps else None,
                    "metric_first": x[0],
                    "metric_last": x[-1],
                    "target_first": y[0],
                    "target_last": y[-1],
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_spearman": abs(spearman) if spearman is not None else None,
                    "direction": (
                        "higher_metric_tracks_higher_target"
                        if (spearman or 0.0) > 0
                        else "higher_metric_tracks_lower_target"
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda row: (str(row["target"]), -(row["abs_spearman"] or 0.0), str(row["metric"])),
    )


def write_metric_correlation_artifacts(output_dir: Path, metrics: dict[str, Any]) -> dict[str, str]:
    rows = compute_metric_correlations(metrics)
    json_path = output_dir / "metric_correlations.json"
    csv_path = output_dir / "metric_correlations.csv"
    md_path = output_dir / "metric_correlations.md"
    write_json(json_path, rows)
    keys = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Metric/Loss Correlations",
        "",
        "Computed from the per-step training log for this gate attempt. `train_bpb` is actual byte-normalized train BPB when the trainer emits it; older logs only support train-loss/sidecar correlations.",
        "",
    ]
    if not rows:
        lines.append("No correlation rows were available. This usually means the run is old enough to lack `train_bpb`, or fewer than three train rows were logged.")
    for target in ("train_bpb", "loss", "sidecar"):
        subset = [row for row in rows if row["target"] == target][:12]
        if not subset:
            continue
        lines.extend(
            [
                f"## Target: `{target}`",
                "",
                "| metric | n | spearman | pearson | direction |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in subset:
            lines.append(
                "| {metric} | {n} | {spearman} | {pearson} | {direction} |".format(
                    metric=row["metric"],
                    n=row["n"],
                    spearman=f"{row['spearman']:.4f}" if isinstance(row.get("spearman"), (int, float)) else "n/a",
                    pearson=f"{row['pearson']:.4f}" if isinstance(row.get("pearson"), (int, float)) else "n/a",
                    direction=row["direction"],
                )
            )
        lines.append("")
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {
        "metric_correlations_json": str(json_path),
        "metric_correlations_csv": str(csv_path),
        "metric_correlations_md": str(md_path),
    }


def run_command(name: str, command: list[str], log_dir: Path, *, strict: bool) -> CommandResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, text=True)
    result = CommandResult(name, command, int(proc.returncode), str(stdout_path), str(stderr_path))
    if strict and result.returncode != 0:
        raise RuntimeError(
            f"{name} failed with return code {result.returncode}; stdout={stdout_path}; stderr={stderr_path}"
        )
    return result


def find_checkpoint(args: argparse.Namespace, metrics: dict[str, Any]) -> Path:
    if args.checkpoint:
        path = Path(args.checkpoint)
        return path if path.is_absolute() else ROOT / path
    ckpts = metrics.get("checkpoint_paths", [])
    if ckpts:
        path = Path(str(ckpts[-1]["path"]))
        return path if path.is_absolute() else ROOT / path
    raise FileNotFoundError("No checkpoint path was supplied or found in the train log")


def first_embedding_payload(manifest_path: Path) -> tuple[Path, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    if not records:
        raise ValueError(f"embedding manifest contains no records: {manifest_path}")
    row = records[0]
    npz = ROOT / str(row["relative_npz"])
    js = ROOT / str(row["relative_json"])
    if not npz.exists():
        npz = manifest_path.parent.parent / str(row["relative_npz"])
    if not js.exists():
        js = manifest_path.parent.parent / str(row["relative_json"])
    return npz, js


def decide_next_profile(
    metrics: dict[str, Any],
    validation: dict[str, Any],
    sidecar_review: dict[str, Any] | None = None,
    *,
    run_id: str = "",
    target_bpb: float = 1.19,
) -> dict[str, Any]:
    adaptive = build_adaptive_decision(
        metrics,
        validation,
        sidecar_review or {},
        run_id=run_id,
        target_bpb=target_bpb,
    )
    bpb = metrics.get("selected_bpb")
    bpb = float(bpb) if isinstance(bpb, (int, float)) and math.isfinite(float(bpb)) else float("inf")
    train = metrics.get("latest_train", {})
    sidecar = float(train.get("sidecar") or 0.0) if isinstance(train, dict) else 0.0
    train_bpb = float(train.get("train_bpb") or float("inf")) if isinstance(train, dict) else float("inf")
    graph_lm_bpb = float(train.get("graph_lm_bpb") or float("inf")) if isinstance(train, dict) else float("inf")
    graph_lm_weight = float(train.get("graph_lm_weight") or 0.0) if isinstance(train, dict) else 0.0
    artifact = int(metrics.get("artifact_bytes") or 0)
    exact_ok = bool(validation.get("strict_validation_passed", False))
    sidecar_review = sidecar_review or {}
    sidecar_count = int(sidecar_review.get("observed_metric_count") or 0)
    sidecar_risk = sidecar_review.get("bpb_risk_metrics_top") or []
    if not exact_ok:
        adaptive.update(
            {
                "next_profile_hint": "analysis_blocked_repeat_only_after_fix",
                "reason": "strict exactness or screenshot analysis failed; do not launch the next hyperparameter profile blindly",
                "sidecar_metric_review": sidecar_review,
            }
        )
        return adaptive
    if artifact and artifact > 15_850_000:
        profile = "artifact_margin_conservative_aux"
        reason = "Artifact size is close to the 16,000,000-byte cap; avoid shape changes and keep auxiliary weights light."
    elif bpb > 1.30 or train_bpb > 1.45:
        profile = "gate1500_fast_main_lr_light_graphcg"
        reason = (
            "BPB/train BPB remain high by the gate, so the next run should emphasize faster main FineWeb optimization, "
            "lower early graph-LM pressure through the new schedule, and light GraphCG/advanced pressure with conflict-aware gradient routing."
        )
    elif math.isfinite(graph_lm_bpb) and graph_lm_bpb < 0.25 and graph_lm_weight >= 0.10 and bpb > 1.19:
        profile = "gate1500_fast_main_lr_light_graphcg"
        reason = (
            "The graph-LM stream is already much easier than FineWeb while its fixed weight is nontrivial; use the new graph-LM curriculum "
            "and routing so graph structure remains active without overcompeting with BPB early."
        )
    elif bpb > 1.24 and sidecar < 1e-3:
        profile = "gate1500_high_batch_toric_bgg_memory"
        reason = (
            "Validation BPB is above target but sidecar pressure is tiny; keep high batch utilization and use routed toric/BGG/memory pressure "
            "so useful structure can help uncertain tokens without dominating the primary BPB gradient."
        )
    elif bpb > 1.20:
        profile = "gate1500_structural_toric_heavy"
        reason = (
            "BPB is in the near-target plateau band; test stronger first-class TokenGT graphification and toric/tropical active-face structure "
            "while relying on uncertainty localization and PCGrad routing to protect the byte objective."
        )
    else:
        profile = "large_batch_all_advanced_low_weight"
        reason = "BPB is close to target; test a broad advanced regularizer subset at low weights while preserving all metric observability."
    if sidecar_count:
        reason = (
            f"{reason} Sidecar review observed {sidecar_count} detailed W&B metrics; "
            f"{len(sidecar_risk) if isinstance(sidecar_risk, list) else 0} top BPB-risk metrics were exported for Codex review."
        )
    else:
        reason = f"{reason} Sidecar W&B metric review did not observe detailed metrics and should be inspected."
    adaptive.update(
        {
            "next_profile_hint": profile,
            "reason": f"{reason} {adaptive.get('reason', '')}",
            "score_context": {
                "selected_bpb": bpb,
                "train_bpb": train_bpb,
                "graph_lm_bpb": graph_lm_bpb,
                "graph_lm_weight": graph_lm_weight,
                "sidecar_loss": sidecar,
                "artifact_bytes": artifact,
            },
            "sidecar_metric_review": sidecar_review,
        }
    )
    return adaptive


def write_index(output_dir: Path, results: list[CommandResult], paths: dict[str, str], metrics: dict[str, Any], decision: dict[str, Any]) -> None:
    links = []
    for label, raw in paths.items():
        path = Path(raw)
        if path.exists():
            rel = path.relative_to(output_dir) if path.is_relative_to(output_dir) else path
            links.append(f'<a class="pill" href="{rel.as_posix()}">{label}</a>')
    rows = "\n".join(
        f"<tr><td>{result.name}</td><td>{result.returncode}</td><td><code>{' '.join(result.command)}</code></td></tr>"
        for result in results
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ToricGT OAI Full Iteration Analysis</title>
<style>
body{{margin:0;background:#030712;color:#e8fbff;font-family:Inter,ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1280px;margin:0 auto;padding:28px 22px 64px}}
.hero,.card{{background:#07111f;border:1px solid rgba(55,232,255,.28);border-radius:8px;padding:18px;margin:14px 0}}
a{{color:#37e8ff}}.pill{{display:inline-block;border:1px solid rgba(55,232,255,.28);border-radius:999px;padding:7px 11px;margin:4px;background:rgba(55,232,255,.08)}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid rgba(145,168,183,.17);padding:7px;text-align:left;vertical-align:top}}code,pre{{white-space:pre-wrap;color:#dff8ff}}
</style></head><body><main>
<section class="hero"><h1>ToricGT OAI Full Iteration Analysis</h1>
<p>Generated {utc_iso()}. This page links the W&B metric export, exact PH/CAS reports, toric embedding reports, branching reasoning visualizations, screenshots, and the next-profile decision.</p>
<div>{''.join(links)}</div></section>
<section class="card"><h2>Selected Metrics</h2><pre>{json.dumps(metrics, indent=2, sort_keys=True)}</pre></section>
<section class="card"><h2>Next Profile Decision</h2><pre>{json.dumps(decision, indent=2, sort_keys=True)}</pre></section>
<section class="card"><h2>Commands</h2><table><tr><th>name</th><th>returncode</th><th>command</th></tr>{rows}</table></section>
</main></body></html>"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    if args.quick:
        args.records = min(args.records, 1)
        args.embedding_records = min(args.embedding_records, 1)
        args.embedding_seq_len = min(args.embedding_seq_len, 96)
        args.branching_max_nodes = min(args.branching_max_nodes, 96)
        args.gudhi_max_points = min(args.gudhi_max_points, 10)
        args.cas_max_points = min(args.cas_max_points, 8)
    log_dir = output_dir / "command_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_log = Path(args.train_log)
    train_log = train_log if train_log.is_absolute() else ROOT / train_log
    metrics = parse_log(train_log)
    checkpoint = find_checkpoint(args, metrics)
    metrics["checkpoint"] = str(checkpoint)
    write_json(output_dir / "metrics.json", metrics)
    write_metrics_csv(output_dir / "metrics.csv", metrics)
    correlation_paths = write_metric_correlation_artifacts(output_dir, metrics)

    results: list[CommandResult] = []
    paths: dict[str, str] = {
        "metrics_json": str(output_dir / "metrics.json"),
        "metrics_csv": str(output_dir / "metrics.csv"),
        **correlation_paths,
    }
    strict = bool(args.strict)

    if args.run_path:
        paths["wandb_metrics"] = str(output_dir / "wandb_metrics")
        results.append(
            run_command(
                "wandb_metrics",
                [
                    PYTHON,
                    "scripts/analyze_wandb_metrics.py",
                    "--run-path",
                    args.run_path,
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(output_dir / "wandb_metrics"),
                ],
                log_dir,
                strict=strict,
            )
        )
        paths["sidecar_metric_review"] = str(output_dir / "sidecar_metric_review" / "SIDECAR-METRIC-REVIEW.md")
        results.append(
            run_command(
                "sidecar_metric_review",
                [
                    PYTHON,
                    "scripts/analyze_toricgt_sidecar_wandb_metrics.py",
                    "--run-path",
                    args.run_path,
                    "--output-dir",
                    str(output_dir / "sidecar_metric_review"),
                ],
                log_dir,
                strict=strict,
            )
        )

    embedding_root = output_dir / "geometry"
    results.append(
        run_command(
            "extract_oai_embeddings",
            [
                PYTHON,
                "scripts/extract_oai_sidecar_embeddings.py",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(embedding_root),
                "--tokenizer-path",
                args.tokenizer_path,
                "--graph-train-glob",
                args.graph_train_glob,
                "--records",
                str(args.embedding_records),
                "--seq-len",
                str(args.embedding_seq_len),
                "--device",
                str(args.embedding_device),
            ],
            log_dir,
            strict=strict,
        )
    )
    manifest = embedding_root / "embeddings" / "manifest.json"
    paths["embedding_manifest"] = str(manifest)
    first_npz, first_json = first_embedding_payload(manifest)

    paths["gudhi_persistence"] = str(output_dir / "gudhi_persistence" / "index.html")
    results.append(
        run_command(
            "gudhi_persistence",
            [
                PYTHON,
                "scripts/run_gudhi_persistence_audit.py",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir / "gudhi_persistence"),
                "--records",
                str(args.records),
                "--max-points",
                str(args.gudhi_max_points),
                "--emit-ph-feature-visualizations",
            ],
            log_dir,
            strict=strict,
        )
    )

    paths["embedding_cas_sidecar"] = str(output_dir / "embedding_cas_sidecar" / "index.html")
    results.append(
        run_command(
            "embedding_cas_sidecar",
            [
                PYTHON,
                "scripts/run_embedding_cas_sidecar.py",
                "--embedding-manifest",
                str(manifest),
                "--output-dir",
                str(output_dir / "embedding_cas_sidecar"),
                "--records",
                str(args.records),
                "--max-points",
                str(args.cas_max_points),
                "--exponent-dim",
                "2",
                "--macaulay2-timeout-seconds",
                str(args.cas_macaulay2_timeout_seconds),
            ],
            log_dir,
            strict=strict,
        )
    )

    paths["toric_embedding_report"] = str(output_dir / "toric_embedding_report" / "index.html")
    results.append(
        run_command(
            "toric_embedding_report",
            [
                PYTHON,
                "scripts/render_toric_embedding_report.py",
                "--sidecar-dir",
                str(output_dir / "embedding_cas_sidecar"),
                "--output-dir",
                str(output_dir / "toric_embedding_report"),
            ],
            log_dir,
            strict=strict,
        )
    )

    paths["bgg_category_o_report"] = str(output_dir / "bgg_category_o_report" / "index.html")
    results.append(
        run_command(
            "bgg_category_o_report",
            [
                PYTHON,
                "scripts/render_bgg_category_o_report.py",
                "--metrics-json",
                str(output_dir / "metrics.json"),
                "--output-dir",
                str(output_dir / "bgg_category_o_report"),
            ],
            log_dir,
            strict=strict,
        )
    )

    paths["branching_reasoning_report"] = str(output_dir / "branching_reasoning_report" / "index.html")
    results.append(
        run_command(
            "branching_reasoning_report",
            [
                PYTHON,
                "scripts/render_branching_reasoning_trajectory_report.py",
                "--output-dir",
                str(output_dir / "branching_reasoning_report"),
                "--embedding-payload-npz",
                str(first_npz),
                "--embedding-payload-json",
                str(first_json),
                "--embedding-max-nodes",
                str(args.branching_max_nodes),
                "--radius-levels",
                "7",
                "--ph-landscape-layers",
                "3",
                "--ph-landscape-resolution",
                "32",
                "--ph-image-resolution",
                "10",
            ],
            log_dir,
            strict=strict,
        )
    )

    paths["branching_static_screenshots"] = str(output_dir / "branching_reasoning_report" / "static_screenshots")
    results.append(
        run_command(
            "branching_static_screenshots",
            [
                PYTHON,
                "scripts/render_branching_payload_static_screenshots.py",
                "--payload-json",
                str(output_dir / "branching_reasoning_report" / "branching_reasoning_payload.json"),
                "--output-dir",
                str(output_dir / "branching_reasoning_report" / "static_screenshots"),
            ],
            log_dir,
            strict=strict,
        )
    )

    paths["screenshots"] = str(output_dir / "html_screenshots" / "index.html")
    results.append(
        run_command(
            "html_screenshots",
            [
                PYTHON,
                "scripts/render_html_screenshots.py",
                "--source-dir",
                str(output_dir),
                "--output-dir",
                str(output_dir / "html_screenshots"),
                "--interaction-audit",
                "--assert-branching-report",
                "--no-full-page",
                "--viewport-slices",
                "3",
            ],
            log_dir,
            strict=strict,
        )
    )

    validation: dict[str, Any] = {"strict_validation_passed": True, "errors": []}
    try:
        results.append(
            run_command(
                "validate_gudhi_exactness",
                [
                    PYTHON,
                    "scripts/validate_analysis_exactness.py",
                    str(output_dir / "gudhi_persistence"),
                ],
                log_dir,
                strict=True,
            )
        )
    except Exception as exc:
        validation = {"strict_validation_passed": False, "errors": [str(exc)]}
        if strict:
            write_json(output_dir / "validation.json", validation)
            raise
    write_json(output_dir / "validation.json", validation)
    sidecar_review_summary_path = output_dir / "sidecar_metric_review" / "sidecar_metric_summary.json"
    sidecar_review: dict[str, Any] = {}
    if sidecar_review_summary_path.exists():
        sidecar_review = json.loads(sidecar_review_summary_path.read_text(encoding="utf-8"))
    decision = decide_next_profile(metrics, validation, sidecar_review, run_id=args.run_id, target_bpb=1.19)
    write_json(output_dir / "next_profile_decision.json", decision)

    write_index(output_dir, results, paths, metrics, decision)
    report_lines = [
        "# ToricGT OAI Full Iteration Analysis",
        "",
        f"- generated UTC: `{utc_iso()}`",
        f"- run id: `{args.run_id}`",
        f"- checkpoint: `{checkpoint}`",
        f"- selected BPB: `{metrics.get('selected_bpb')}`",
        f"- strict validation passed: `{validation.get('strict_validation_passed')}`",
        f"- next profile hint: `{decision.get('next_profile_hint')}`",
        f"- reason: {decision.get('reason')}",
        "",
        "## Artifacts",
        "",
    ]
    for label, raw in sorted(paths.items()):
        report_lines.append(f"- {label}: `{raw}`")
    sidecar_summary = decision.get("sidecar_metric_review", {}) if isinstance(decision, dict) else {}
    if isinstance(sidecar_summary, dict):
        report_lines.extend(
            [
                "",
                "## Sidecar Metric Review Summary",
                "",
                f"- observed detailed sidecar/related metrics: `{sidecar_summary.get('observed_metric_count', 0)}`",
                f"- raw `toricgt_sidecar/*` metrics: `{sidecar_summary.get('raw_sidecar_metric_count', 0)}`",
                f"- related alias metrics: `{sidecar_summary.get('related_alias_metric_count', 0)}`",
                f"- review markdown: `{sidecar_summary.get('review_markdown', '')}`",
                f"- review JSON: `{sidecar_summary.get('review_json', '')}`",
                f"- correlation plot: `{sidecar_summary.get('correlation_plot', '')}`",
                "",
                "Codex review requirement: read the sidecar markdown and JSON before selecting the next restart profile. Every observed raw `toricgt_sidecar/*` metric must be considered with its description, trend, train-BPB correlation, and family weight guidance.",
                "",
                "Top BPB-risk sidecar metrics are embedded in `next_profile_decision.json` so the campaign Codex review can account for detailed mathematical behavior before the next restart profile is selected.",
                "",
            ]
        )
    report_lines.extend(["", "## Command Results", ""])
    for result in results:
        report_lines.append(f"- `{result.name}`: returncode `{result.returncode}`, stdout `{result.stdout_path}`, stderr `{result.stderr_path}`")
    (output_dir / "FULL-ITERATION-REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "decision": decision, "report": str(output_dir / "FULL-ITERATION-REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
