#!/usr/bin/env python3
# Example:
# conda run --no-capture-output -n tokengt env PYTHONPATH=src \
#   python scripts/evaluate_reasoning_simplex.py \
#   --checkpoint checkpoints/parameter_golf_oai_dense/best.pt \
#   --data-glob 'data/curated_hf_shards/validation/*.parquet' \
#   --samples 8 --budgets 1 2 4 --output-dir outputs/reasoning_simplex/oai-best
"""Evaluate ToricGT reasoning-simplex and tetrahedron diagnostics.

The script runs a checkpoint under several inference-time reasoning budgets,
computes BPB/loss, Kolmogorov-style compression proxies, GFlowNet action-trace
complexity, and hidden-trajectory minimum-spanning-tree geometry, then plots
data-driven triangle and tetrahedron projections.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from toricgt.complexity import random_order_complexity_metrics
from toricgt.random_order_lm import DenseRandomOrderToricLM, RandomOrderLMConfig, random_order_batch
from toricgt.reasoning_geometry import (
    TETRAHEDRON_VERTICES,
    TRIANGLE_VERTICES,
    attach_normalized_scores,
    barycentric_to_cartesian,
    mean_dict,
    prim_mst_stats,
    rbf_interpolate,
    simplex_record,
    triangle_grid,
)

from train_parameter_golf_random_order import build_loader, config_get, read_yaml


TRIANGLE_SPECS = {
    "reasoning_k_bpb": {
        "labels": ["reasoning budget", "K(x)", "low BPB"],
        "scores": ["score/reasoning", "score/k", "score/bpb_quality"],
        "title": "Reasoning/K/BPB simplex",
    },
    "robustness": {
        "labels": ["low loss", "order robustness", "GFlowNet diversity"],
        "scores": ["score/loss_quality", "score/robustness", "score/gflownet_diversity"],
        "title": "Robustness simplex",
    },
    "efficiency": {
        "labels": ["low BPB", "MST efficiency", "K(x)"],
        "scores": ["score/bpb_quality", "score/mst_efficiency", "score/k"],
        "title": "Compression-efficiency simplex",
    },
}

TETRAHEDRON_SPECS = {
    "reasoning_k_bpb_mst": {
        "labels": ["reasoning budget", "K(x)", "low BPB", "MST efficiency"],
        "scores": ["score/reasoning", "score/k", "score/bpb_quality", "score/mst_efficiency"],
        "title": "Reasoning/K/BPB/MST tetrahedron",
    },
    "reasoning_k_bpb_diversity": {
        "labels": ["reasoning budget", "K(x)", "low BPB", "GFlowNet diversity"],
        "scores": ["score/reasoning", "score/k", "score/bpb_quality", "score/gflownet_diversity"],
        "title": "Reasoning/K/BPB/GFlowNet tetrahedron",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/parameter_golf_oai_dense/best.pt")
    parser.add_argument("--config", default="config/train.parameter_golf_random_order_dense.yaml")
    parser.add_argument("--data-glob", default="data/curated_hf_shards/validation/*.parquet")
    parser.add_argument("--output-dir", default="outputs/reasoning_simplex")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--rows-per-batch", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-estimated-tokens", type=int, default=0)
    parser.add_argument("--max-estimated-tokens", type=int, default=0)
    parser.add_argument("--task-family-keywords", nargs="+", default=[])
    parser.add_argument("--dataset-keywords", nargs="+", default=[])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--max-order-samples", type=int, default=4)
    parser.add_argument("--max-gflownet-samples", type=int, default=4)
    parser.add_argument("--max-recurrent-passes", type=int, default=4)
    parser.add_argument("--max-mst-nodes", type=int, default=96)
    parser.add_argument("--complexity-samples", type=int, default=2)
    parser.add_argument("--compressors", nargs="+", default=["zlib", "lzma"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--random-init", action="store_true", help="Use config weights when checkpoint is unavailable.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="toricgt-parameter-golf")
    parser.add_argument("--wandb-run-name", default="reasoning-simplex-eval")
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> DenseRandomOrderToricLM:
    payload: dict[str, Any] | None = None
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists() and not args.random_init:
        payload = torch.load(checkpoint, map_location="cpu")
        cfg = RandomOrderLMConfig(**payload["config"])
    else:
        file_config = read_yaml(args.config)
        cfg = RandomOrderLMConfig(
            vocab_size=config_get(file_config, "model", "vocab_size", 260),
            max_seq_len=config_get(file_config, "model", "max_seq_len", 1024),
            d_model=config_get(file_config, "model", "d_model", 384),
            num_heads=config_get(file_config, "model", "num_heads", 6),
            num_layers=config_get(file_config, "model", "num_layers", 7),
            recurrent_passes=config_get(file_config, "model", "recurrent_passes", 2),
            ffn_multiplier=config_get(file_config, "model", "ffn_multiplier", 4),
            dropout=config_get(file_config, "model", "dropout", 0.1),
            attention=config_get(file_config, "model", "attention", "hybrid"),
            ring_block_size=config_get(file_config, "model", "ring_block_size", 256),
            seed=config_get(file_config, "training", "seed", 17),
            polarquant_kv_bits=config_get(file_config, "model", "polarquant_kv_bits", 8),
            polarquant_train=config_get(file_config, "model", "polarquant_train", False),
            use_gflownet_policy=config_get(file_config, "model", "use_gflownet_policy", True),
            gflownet_num_actions=config_get(file_config, "model", "gflownet_num_actions", 16),
            gflownet_hidden_dim=config_get(file_config, "model", "gflownet_hidden_dim", 192),
            gflownet_action_scale=config_get(file_config, "model", "gflownet_action_scale", 0.06),
            use_bigram_hash=config_get(file_config, "model", "use_bigram_hash", True),
            bigram_hash_buckets=config_get(file_config, "model", "bigram_hash_buckets", 4096),
            bigram_hash_weight=config_get(file_config, "model", "bigram_hash_weight", 0.35),
            use_caseops_features=config_get(file_config, "model", "use_caseops_features", True),
            caseops_weight=config_get(file_config, "model", "caseops_weight", 0.35),
            use_smear_gate=config_get(file_config, "model", "use_smear_gate", True),
            smear_temperature_min=config_get(file_config, "model", "smear_temperature_min", 0.55),
            smear_temperature_max=config_get(file_config, "model", "smear_temperature_max", 1.75),
            use_toric_memory=config_get(file_config, "model", "use_toric_memory", True),
            toric_memory_slots=config_get(file_config, "model", "toric_memory_slots", 32),
            toric_memory_weight=config_get(file_config, "model", "toric_memory_weight", 0.08),
            aux_mtp_offsets=config_get(file_config, "model", "aux_mtp_offsets", 2),
            contrastive_temperature=config_get(file_config, "model", "contrastive_temperature", 0.2),
            trajectory_flow_viscosity=config_get(file_config, "model", "trajectory_flow_viscosity", 0.05),
            target_artifact_bytes=config_get(file_config, "model", "target_artifact_bytes", 15_600_000),
        )
    model = DenseRandomOrderToricLM(cfg)
    if payload is not None:
        model.load_state_dict(payload["model"])
    return model


def hidden_mst_metrics(model: DenseRandomOrderToricLM, out: dict[str, torch.Tensor], max_nodes: int) -> dict[str, float]:
    aux = model.forward_from_previous(
        out["previous_tokens"],
        out["target_positions"],
        sample_gflownet=False,
        return_aux=True,
    )
    if not isinstance(aux, dict) or "hidden" not in aux:
        return {}
    hidden = aux["hidden"].detach().float().cpu()
    records = []
    for sample in range(hidden.shape[0]):
        length = hidden.shape[1]
        if length > max_nodes:
            indices = torch.linspace(0, length - 1, steps=max_nodes).round().long()
            points = hidden[sample, indices]
        else:
            points = hidden[sample]
        records.append(prim_mst_stats(points.tolist()))
    return {f"mst/{key}": value for key, value in mean_dict(records).items()}


@torch.no_grad()
def evaluate_budget(
    model: DenseRandomOrderToricLM,
    loader,
    args: argparse.Namespace,
    budget: int,
) -> dict[str, float]:
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    original_recurrent = model.config.recurrent_passes
    recurrent_passes = max(1, min(args.max_recurrent_passes, original_recurrent + max(0, int(math.log2(max(1, budget))))))
    model.config = replace(model.config, recurrent_passes=recurrent_passes)
    order_samples = max(1, min(args.max_order_samples, budget))
    gflownet_samples = max(1, min(args.max_gflownet_samples, budget))
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    use_amp = device.type == "cuda" and args.precision in {"bf16", "fp16"}
    iterator = iter(loader)
    per_pass_records: list[dict[str, float]] = []
    batches = max(1, math.ceil(args.samples / args.batch_size))
    seen = 0
    start = time.perf_counter()
    for batch_index in range(batches):
        batch = next(iterator)
        tokens = batch["tokens"].to(device)
        sample_ids = batch["sample_ids"].to(device)
        if seen >= args.samples:
            break
        keep = min(tokens.shape[0], args.samples - seen)
        tokens = tokens[:keep]
        sample_ids = sample_ids[:keep]
        seen += keep
        for order_sample in range(order_samples):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(
                    tokens,
                    sample_ids=sample_ids,
                    pass_id=args.seed + budget * 100_003 + batch_index * 1_009 + order_sample,
                    sample_gflownet=gflownet_samples > 1,
                    gflownet_samples=gflownet_samples,
                    return_order=True,
                )
            complexity = random_order_complexity_metrics(
                tokens=tokens,
                previous_tokens=out["previous_tokens"],
                target_tokens=out["target_tokens"],
                permutation=out["permutation"],
                logits=out["logits"],
                action_ids=out.get("gflownet_action_ids"),
                byte_offset=model.config.byte_offset,
                prefix="complexity",
                compressors=tuple(args.compressors),
                max_samples=args.complexity_samples,
            )
            record = {
                "loss": float(out["loss"].detach().cpu()),
                "bpb": float(out["bpb"].detach().cpu()),
                "gflownet_entropy": float(out.get("gflownet_entropy", torch.zeros(())).detach().cpu()),
                "gflownet_diversity": float(out.get("gflownet_action_diversity", torch.zeros(())).detach().cpu()),
                "trajectory_tokens": float(tokens.shape[1] * order_samples * gflownet_samples * recurrent_passes),
            }
            for key, value in complexity.items():
                if key.endswith("_mean") or key in {"complexity/argmax_byte_accuracy", "complexity/samples"}:
                    record[key] = float(value)
            record.update(hidden_mst_metrics(model, out, max_nodes=args.max_mst_nodes))
            per_pass_records.append(record)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    model.config = replace(model.config, recurrent_passes=original_recurrent)
    aggregate = mean_dict(per_pass_records)
    aggregate.update(
        {
            "budget": float(budget),
            "order_samples": float(order_samples),
            "gflownet_samples": float(gflownet_samples),
            "recurrent_passes": float(recurrent_passes),
            "wall_ms": float(elapsed_ms),
            "samples": float(seen),
        }
    )
    if len(per_pass_records) > 1:
        bpb_values = [item["bpb"] for item in per_pass_records]
        aggregate["bpb_std"] = float(np.std(bpb_values))
    else:
        aggregate["bpb_std"] = 0.0
    aggregate["k_proxy"] = float(
        aggregate.get("complexity/target_cond_k_lzma_mean", aggregate.get("complexity/target_cond_k_zlib_mean", 0.0))
        + 0.25
        * aggregate.get(
            "complexity/gflownet_action_trace_k_lzma_mean",
            aggregate.get("complexity/order_program_k_lzma_mean", 0.0),
        )
    )
    aggregate["reasoning_budget"] = float(
        math.log1p(aggregate["trajectory_tokens"]) + 0.001 * aggregate["wall_ms"]
    )
    aggregate["mst_efficiency"] = float(aggregate.get("mst/mst_efficiency", 0.5))
    return aggregate


def enrich_scores(records: list[dict[str, float]]) -> list[dict[str, float]]:
    attach_normalized_scores(
        records,
        {
            "score/reasoning": ("reasoning_budget", True),
            "score/k": ("k_proxy", True),
            "score/bpb_quality": ("bpb", False),
            "score/loss_quality": ("loss", False),
            "score/robustness": ("bpb_std", False),
            "score/gflownet_diversity": ("gflownet_diversity", True),
            "score/mst_efficiency": ("mst_efficiency", True),
        },
    )
    return records


def blue_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "toricgt_reasoning_blues",
        ["#06112a", "#082f63", "#0b5ea8", "#39b8ff", "#dff8ff"],
    )


def plot_triangle(records: list[dict[str, float]], spec: dict[str, Any], output_path: Path) -> None:
    vertices = [TRIANGLE_VERTICES["left"], TRIANGLE_VERTICES["right"], TRIANGLE_VERTICES["top"]]
    labels = spec["labels"]
    points = []
    intensities = []
    for record in records:
        simplex = simplex_record(record, spec["scores"], labels)
        x, y = barycentric_to_cartesian(simplex["weights"], vertices)
        points.append((x, y))
        intensities.append(float(record["score/reasoning"]))
    grid = triangle_grid(resolution=100)
    xs = [item[0] for item in grid]
    ys = [item[1] for item in grid]
    zs = [rbf_interpolate((x, y), points, intensities, bandwidth=0.18) for x, y, _ in grid]
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#030712")
    ax.set_facecolor("#030712")
    tri = mtri.Triangulation(xs, ys)
    contour = ax.tricontourf(tri, zs, levels=18, cmap=blue_colormap(), alpha=0.95)
    ax.tricontour(tri, zs, levels=8, colors="#9eeeff", linewidths=0.35, alpha=0.45)
    boundary_x = [vertices[0][0], vertices[1][0], vertices[2][0], vertices[0][0]]
    boundary_y = [vertices[0][1], vertices[1][1], vertices[2][1], vertices[0][1]]
    ax.plot(boundary_x, boundary_y, color="#6df6ff", linewidth=1.8)
    scatter = ax.scatter(
        [p[0] for p in points],
        [p[1] for p in points],
        c=[record["bpb"] for record in records],
        cmap="magma_r",
        s=90,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    for record, (x, y) in zip(records, points):
        ax.text(x, y + 0.025, f"B{int(record['budget'])}", color="white", fontsize=8, ha="center")
    label_offsets = [(-0.06, -0.055), (0.06, -0.055), (0.0, 0.045)]
    for label, vertex, offset in zip(labels, vertices, label_offsets):
        ax.text(vertex[0] + offset[0], vertex[1] + offset[1], label, color="#e8fbff", fontsize=10, ha="center")
    ax.set_title(spec["title"], color="white", fontsize=14)
    ax.set_aspect("equal")
    ax.set_axis_off()
    cbar = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("reasoning intensity", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    sb = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.09)
    sb.set_label("BPB", color="white")
    sb.ax.yaxis.set_tick_params(color="white")
    plt.setp(sb.ax.get_yticklabels(), color="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_tetrahedron(records: list[dict[str, float]], spec: dict[str, Any], output_path: Path) -> None:
    vertices = [TETRAHEDRON_VERTICES[key] for key in ("a", "b", "c", "d")]
    labels = spec["labels"]
    coords = []
    for record in records:
        simplex = simplex_record(record, spec["scores"], labels)
        coords.append(barycentric_to_cartesian(simplex["weights"], vertices))
    fig = plt.figure(figsize=(8, 7), facecolor="#030712")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#030712")
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for left, right in edges:
        ax.plot(
            [vertices[left][0], vertices[right][0]],
            [vertices[left][1], vertices[right][1]],
            [vertices[left][2], vertices[right][2]],
            color="#6df6ff",
            linewidth=1.5,
        )
    sc = ax.scatter(
        [coord[0] for coord in coords],
        [coord[1] for coord in coords],
        [coord[2] for coord in coords],
        c=[record["bpb"] for record in records],
        cmap="magma_r",
        s=85,
        edgecolor="white",
        linewidth=0.7,
    )
    for record, coord in zip(records, coords):
        ax.text(coord[0], coord[1], coord[2] + 0.025, f"B{int(record['budget'])}", color="white", fontsize=8)
    for label, vertex in zip(labels, vertices):
        ax.text(vertex[0], vertex[1], vertex[2] + 0.04, label, color="#e8fbff", fontsize=9, ha="center")
    ax.set_title(spec["title"], color="white")
    ax.set_axis_off()
    ax.view_init(elev=22, azim=38)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("BPB", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_interactive_tetrahedron(records: list[dict[str, float]], spec: dict[str, Any], output_path: Path) -> None:
    vertices = [TETRAHEDRON_VERTICES[key] for key in ("a", "b", "c", "d")]
    labels = spec["labels"]
    coords = []
    weights = []
    hover = []
    for record in records:
        simplex = simplex_record(record, spec["scores"], labels)
        weights.append(simplex["weights"])
        coords.append(barycentric_to_cartesian(simplex["weights"], vertices))
        hover.append(
            f"budget={int(record['budget'])}<br>BPB={record['bpb']:.4f}<br>"
            f"K proxy={record['k_proxy']:.2f}<br>MST efficiency={record['mst_efficiency']:.4f}"
        )
    payload = {
        "title": spec["title"],
        "labels": labels,
        "vertices": vertices,
        "coords": coords,
        "bpb": [record["bpb"] for record in records],
        "hover": hover,
        "weights": weights,
    }
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>{spec['title']}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script></head>
<body style="margin:0;background:#030712;color:white;font-family:system-ui">
<div id="plot" style="width:100vw;height:100vh"></div>
<script>
const payload = {json.dumps(payload)};
const edges = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]];
const traces = edges.map(([i,j]) => ({{
  type:'scatter3d', mode:'lines',
  x:[payload.vertices[i][0], payload.vertices[j][0]],
  y:[payload.vertices[i][1], payload.vertices[j][1]],
  z:[payload.vertices[i][2], payload.vertices[j][2]],
  line:{{color:'#6df6ff', width:4}}, hoverinfo:'skip', showlegend:false
}}));
traces.push({{
  type:'scatter3d', mode:'markers+text',
  x:payload.coords.map(p => p[0]), y:payload.coords.map(p => p[1]), z:payload.coords.map(p => p[2]),
  text:payload.coords.map((_, i) => 'B' + (i + 1)),
  hovertext:payload.hover, hoverinfo:'text',
  marker:{{size:7, color:payload.bpb, colorscale:'Magma', reversescale:true, colorbar:{{title:'BPB'}}, line:{{color:'white', width:1}}}},
  showlegend:false
}});
traces.push({{
  type:'scatter3d', mode:'text',
  x:payload.vertices.map(p => p[0]), y:payload.vertices.map(p => p[1]), z:payload.vertices.map(p => p[2] + 0.05),
  text:payload.labels, textfont:{{color:'#e8fbff', size:14}}, hoverinfo:'skip', showlegend:false
}});
Plotly.newPlot('plot', traces, {{
  title:{{text:payload.title, font:{{color:'white'}}}},
  paper_bgcolor:'#030712', plot_bgcolor:'#030712',
  scene:{{xaxis:{{visible:false}}, yaxis:{{visible:false}}, zaxis:{{visible:false}}, bgcolor:'#030712'}},
  margin:{{l:0,r:0,b:0,t:48}}
}});
</script></body></html>
"""
    output_path.write_text(html, encoding="utf-8")


def write_records(records: list[dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reasoning_simplex_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    keys = sorted({key for record in records for key in record})
    with (output_dir / "reasoning_simplex_records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    model = load_model(args)
    seq_len = min(args.seq_len or model.config.max_seq_len, model.config.max_seq_len)
    loader = build_loader(
        parquet_glob=args.data_glob,
        batch_size=args.batch_size,
        seq_len=seq_len,
        byte_offset=model.config.byte_offset,
        rows_per_batch=args.rows_per_batch or 128,
        seed=args.seed,
        workers=args.num_workers,
        repeat=True,
        synthetic=args.synthetic,
        vocab_size=model.config.vocab_size,
        include_graph_projection=True,
        graph_projection_max_chars=4096,
        coprime_row_stride=True,
        document_separator="\n\n",
        min_estimated_tokens=args.min_estimated_tokens,
        max_estimated_tokens=args.max_estimated_tokens,
        task_family_keywords=tuple(args.task_family_keywords),
        dataset_keywords=tuple(args.dataset_keywords),
    )
    records = []
    for budget in args.budgets:
        print(f"evaluating budget={budget}", flush=True)
        records.append(evaluate_budget(model, loader, args, int(budget)))
    enrich_scores(records)
    out_dir = Path(args.output_dir)
    write_records(records, out_dir)
    for name, spec in TRIANGLE_SPECS.items():
        plot_triangle(records, spec, out_dir / f"{name}_triangle.png")
    for name, spec in TETRAHEDRON_SPECS.items():
        plot_tetrahedron(records, spec, out_dir / f"{name}_tetrahedron.png")
        write_interactive_tetrahedron(records, spec, out_dir / f"{name}_tetrahedron.html")
    summary = {
        "checkpoint": args.checkpoint,
        "records": len(records),
        "outputs": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    }
    (out_dir / "reasoning_simplex_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        for record in records:
            run.log({f"reasoning_simplex/{key}": value for key, value in record.items()}, step=int(record["budget"]))
        run.finish()


if __name__ == "__main__":
    main()
