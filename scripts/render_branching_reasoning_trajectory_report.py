#!/usr/bin/env python3
"""Render a branch/merge graph-of-thought simplex visualization bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toricgt.branching_reasoning_visualization import (  # noqa: E402
    BranchingTrajectoryConfig,
    build_branching_reasoning_payload_from_embedding_payload,
    render_branching_reasoning_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--embedding-payload-npz",
        required=True,
        help="Required toricgt.embedding_payload.v1 NPZ extracted from a real checkpoint/data batch.",
    )
    parser.add_argument(
        "--embedding-payload-json",
        default="",
        help="Optional JSON metadata companion for --embedding-payload-npz.",
    )
    parser.add_argument(
        "--embedding-max-nodes",
        type=int,
        default=0,
        help="Optional explicit node-window size for large embedding payloads. No cap is applied inside the selected window.",
    )
    parser.add_argument(
        "--embedding-node-offset",
        type=int,
        default=0,
        help="First original node id for --embedding-max-nodes window selection.",
    )
    parser.add_argument("--seed", type=int, default=117)
    parser.add_argument("--embedding-dim", type=int, default=24)
    parser.add_argument("--trajectory-levels", type=int, default=18)
    parser.add_argument("--branch-lanes", type=int, default=8)
    parser.add_argument("--side-branch-length", type=int, default=5)
    parser.add_argument("--token-count-min", type=int, default=8)
    parser.add_argument("--token-count-max", type=int, default=12)
    parser.add_argument(
        "--max-render-edges",
        type=int,
        default=0,
        help="Deprecated compatibility flag; visible one-dimensional simplex edges are always rendered.",
    )
    parser.add_argument("--max-render-triangles", type=int, default=900)
    parser.add_argument("--max-map-image-records", type=int, default=900)
    parser.add_argument("--radius-levels", type=int, default=7)
    parser.add_argument("--ph-landscape-layers", type=int, default=3)
    parser.add_argument("--ph-landscape-resolution", type=int, default=32)
    parser.add_argument("--ph-image-resolution", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BranchingTrajectoryConfig(
        seed=int(args.seed),
        embedding_dim=int(args.embedding_dim),
        trajectory_levels=int(args.trajectory_levels),
        branch_lanes=int(args.branch_lanes),
        side_branch_length=int(args.side_branch_length),
        token_count_min=int(args.token_count_min),
        token_count_max=int(args.token_count_max),
        max_render_edges=int(args.max_render_edges),
        max_render_triangles=int(args.max_render_triangles),
        max_map_image_records=int(args.max_map_image_records),
        radius_levels=int(args.radius_levels),
        ph_landscape_layers=int(args.ph_landscape_layers),
        ph_landscape_resolution=int(args.ph_landscape_resolution),
        ph_image_resolution=int(args.ph_image_resolution),
    )
    payload = build_branching_reasoning_payload_from_embedding_payload(
        Path(args.embedding_payload_npz),
        metadata_json_path=Path(args.embedding_payload_json) if args.embedding_payload_json else None,
        cfg=cfg,
        embedding_max_nodes=int(args.embedding_max_nodes) if int(args.embedding_max_nodes) > 0 else None,
        embedding_node_offset=int(args.embedding_node_offset),
    )
    manifest = render_branching_reasoning_report(payload, Path(args.output_dir))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
