#!/usr/bin/env python3
"""Run TokenGT inference/evaluation samples with optional geometry outputs.

This is a thin CLI wrapper around the TokenGT reasoning-geometry evaluator.  It
keeps inference use simple while still allowing the full checkpoint-analysis
plot family bundle, including dark-mode Plotly HTML trajectory/energy/toric
geometry scenes, to be emitted for an individual model run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_SCRIPT = ROOT / "scripts" / "evaluate_tokengt_reasoning_geometry_suite.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="TokenGT .pt checkpoint to run.")
    parser.add_argument("--config", default="config/train.full_tokengt_got_fineweb_derived.yaml")
    parser.add_argument(
        "--data-glob",
        action="append",
        default=[],
        help="Inference/evaluation graph source. May be passed multiple times.",
    )
    parser.add_argument("--output-dir", default="outputs/tokengt_inference")
    parser.add_argument("--records", type=int, default=1, help="Number of inference graph records to run.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--parquet-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fineweb-tokenizer-path", default="")
    parser.add_argument("--fineweb-tokens-per-graph", type=int, default=0)
    parser.add_argument("--fineweb-stride-tokens", type=int, default=0)
    parser.add_argument("--derived-category-max-vertices", type=int, default=8)
    parser.add_argument("--topology-max-points", type=int, default=32)
    parser.add_argument("--topology-max-windows", type=int, default=4)
    parser.add_argument(
        "--emit-geometry",
        dest="emit_geometry",
        action="store_true",
        default=True,
        help="Emit trajectory, topology, CCA, toric, GraphCG, analogical, triangle, tetrahedron, static PNG, and Plotly HTML outputs.",
    )
    parser.add_argument(
        "--no-emit-geometry",
        dest="emit_geometry",
        action="store_false",
        help="Only write an inference manifest; do not run the geometry sidecar.",
    )
    parser.add_argument(
        "--rich-legacy-geometry",
        dest="rich_legacy_geometry",
        action="store_true",
        default=True,
        help="Include the R97-style rich geometry bundle in addition to TokenGT-native plots.",
    )
    parser.add_argument(
        "--no-rich-legacy-geometry",
        dest="rich_legacy_geometry",
        action="store_false",
        help="Emit only TokenGT-native trajectory/topology plots.",
    )
    parser.add_argument(
        "--geometry-output-dir",
        default="",
        help="Override where geometry files are written. Defaults to <output-dir>/geometry.",
    )
    return parser.parse_args()


def load_checkpoint_meta(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", {})
    parameter_count = 0
    if isinstance(state, dict):
        parameter_count = int(sum(int(tensor.numel()) for tensor in state.values() if hasattr(tensor, "numel")))
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("step", 0) or 0),
        "checkpoint_kind": str(payload.get("kind", "tokengt_graph")),
        "parameter_count": parameter_count,
        "has_lm_head": bool(
            isinstance(state, dict)
            and any(key.endswith("lm_head.weight") or "lm_head" in key or "lm_projection" in key for key in state)
        ),
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def geometry_command(args: argparse.Namespace, geometry_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(GEOMETRY_SCRIPT),
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.config),
        "--output-dir",
        str(geometry_dir),
        "--records",
        str(max(1, int(args.records))),
        "--batch-size",
        str(max(1, int(args.batch_size))),
        "--parquet-batch-size",
        str(max(1, int(args.parquet_batch_size))),
        "--device",
        str(args.device),
        "--precision",
        str(args.precision),
        "--seed",
        str(int(args.seed)),
        "--derived-category-max-vertices",
        str(max(1, int(args.derived_category_max_vertices))),
        "--topology-max-points",
        str(max(4, int(args.topology_max_points))),
        "--topology-max-windows",
        str(max(1, int(args.topology_max_windows))),
    ]
    for data_glob in args.data_glob:
        cmd.extend(["--data-glob", str(data_glob)])
    if args.fineweb_tokenizer_path:
        cmd.extend(["--fineweb-tokenizer-path", str(args.fineweb_tokenizer_path)])
    if int(args.fineweb_tokens_per_graph or 0) > 0:
        cmd.extend(["--fineweb-tokens-per-graph", str(int(args.fineweb_tokens_per_graph))])
    if int(args.fineweb_stride_tokens or 0) > 0:
        cmd.extend(["--fineweb-stride-tokens", str(int(args.fineweb_stride_tokens))])
    if not bool(args.rich_legacy_geometry):
        cmd.append("--no-rich-legacy-geometry")
    return cmd


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir = Path(args.geometry_output_dir) if args.geometry_output_dir else output_dir / "geometry"

    manifest: dict[str, Any] = {
        "schema": "toricgt.tokengt_inference.v1",
        "mode": "tokengt_graph_inference_with_optional_geometry",
        "config": str(args.config),
        "data_globs": [str(item) for item in args.data_glob],
        "records": int(args.records),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "precision": str(args.precision),
        "seed": int(args.seed),
        "geometry_enabled": bool(args.emit_geometry),
        "rich_legacy_geometry_enabled": bool(args.rich_legacy_geometry),
        "interactive_geometry_html_enabled": bool(args.emit_geometry),
        **load_checkpoint_meta(checkpoint),
    }

    if args.emit_geometry:
        geometry_dir.mkdir(parents=True, exist_ok=True)
        cmd = geometry_command(args, geometry_dir)
        subprocess.run(cmd, cwd=ROOT, check=True)
        manifest["geometry_output_dir"] = str(geometry_dir)
        manifest["geometry_summary"] = read_json(geometry_dir / "reasoning_geometry_summary.json")
        manifest["plot_family_manifest"] = read_json(geometry_dir / "plot_family_manifest.json")

    output_path = output_dir / "inference_output.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
