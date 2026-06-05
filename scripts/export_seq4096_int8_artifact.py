#!/usr/bin/env python3
"""Export compact Seq4096 Parameter-Golf checkpoints as int8+zlib artifacts.

The raw training checkpoints are intentionally large because they include
optimizer and RNG state.  The OpenAI Parameter Golf counted artifact is the
compressed model payload plus the clean training script bytes, which must fit
under 16,000,000 bytes.  This script regenerates the same pruned int8+zlib
payload format used by the Seq4096 clean-script exporter.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPACT_SCRIPT = (
    ROOT
    / "amelie-iska"
    / "parameter-golf"
    / "records"
    / "track_10min_16mb"
    / "2026-03-19_TrainingOptSeq4096"
    / "train_gpt.py"
)
BYTE_LIMIT = 16_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Raw Seq4096 .pt checkpoint containing checkpoint['model'].")
    parser.add_argument("--output", required=True, help="Output .ptz artifact path.")
    parser.add_argument("--compact-script", default=str(DEFAULT_COMPACT_SCRIPT))
    parser.add_argument("--export-prune-fraction", type=float, default=0.0)
    parser.add_argument("--byte-limit", type=int, default=BYTE_LIMIT)
    parser.add_argument("--train-loss", type=float, default=float("nan"))
    parser.add_argument("--train-bpb", type=float, default=float("nan"))
    parser.add_argument("--val-loss", type=float, default=float("nan"))
    parser.add_argument("--val-bpb", type=float, default=float("nan"))
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def load_export_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("seq4096_export_train_gpt", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import compact script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    compact_script = Path(args.compact_script)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module = load_export_module(compact_script)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload.get("model"), dict) or "tok_emb.weight" not in payload["model"]:
        raise ValueError(f"not a compact Seq4096 checkpoint: {checkpoint_path}")

    code_bytes = len(compact_script.read_text(encoding="utf-8").encode("utf-8"))
    blob, _export_state, _quant_obj, metrics = module.quantized_export_blob_for_limit(
        payload["model"],
        code_bytes=code_bytes,
        artifact_size_limit_bytes=int(args.byte_limit),
        export_prune_fraction=float(args.export_prune_fraction),
    )
    output_path.write_bytes(blob)
    total_submission_bytes = output_path.stat().st_size + code_bytes
    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "artifact_path": str(output_path),
        "artifact_bytes": output_path.stat().st_size,
        "compact_script": str(compact_script),
        "code_bytes": code_bytes,
        "total_submission_bytes": total_submission_bytes,
        "byte_limit": int(args.byte_limit),
        "under_16mb_limit": bool(total_submission_bytes <= int(args.byte_limit)),
        "quant_format": "int8_clean_per_row_v1 + zlib level 9 + export_prune",
        "export_metrics": metrics,
        "step": int(payload.get("step", 0)),
        "best_val_bpb": payload.get("best_val_bpb"),
        "initial_val_bpb": payload.get("initial_val_bpb"),
        "run_id": args.run_id,
        "train_loss": args.train_loss,
        "train_bpb": args.train_bpb,
        "val_loss": args.val_loss,
        "val_bpb": args.val_bpb,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
