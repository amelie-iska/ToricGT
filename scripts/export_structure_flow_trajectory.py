#!/usr/bin/env python3
"""Export ToricBLM structure flow-matching frames as trajectory artifacts.

The exporter accepts either:

1. a model-produced frame tensor, shaped ``[frames, atoms, 3]`` or
   ``[batch, frames, atoms, 3]``; or
2. model-produced ``noisy_coords`` and ``pred_velocity`` tensors, shaped
   ``[atoms, 3]`` or ``[batch, atoms, 3]``.

It writes a multi-MODEL PDB file.  This is a denoising/generation trajectory
for the flow-matching head, not a molecular-dynamics trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toricgt.structure_flow_matching import rectified_flow_trajectory_frames, write_multimodel_pdb


def load_array(path: Path) -> Any:
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu")
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        return {key: payload[key] for key in payload.files}
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    raise ValueError(f"Unsupported array path suffix: {path}")


def as_tensor(value: Any, key: str | None = None) -> torch.Tensor:
    if isinstance(value, dict):
        if key is None or key not in value:
            raise KeyError(f"Array payload does not contain key {key!r}; available keys={sorted(value)}")
        value = value[key]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().float()
    else:
        tensor = torch.tensor(value, dtype=torch.float32)
    return tensor


def first_batch(tensor: torch.Tensor, expected_dims: int) -> torch.Tensor:
    if tensor.ndim == expected_dims + 1:
        return tensor[0]
    if tensor.ndim == expected_dims:
        return tensor
    raise ValueError(f"Expected tensor with {expected_dims} or {expected_dims + 1} dimensions, got {tuple(tensor.shape)}")


def load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frames", type=Path, help=".npy/.npz/.pt frame tensor with key 'flow_trajectory' for npz/dict payloads")
    group.add_argument("--trajectory-npz", type=Path, help=".npz containing noisy_coords and pred_velocity, optionally mask")
    parser.add_argument("--noisy-coords", type=Path, default=None)
    parser.add_argument("--pred-velocity", type=Path, default=None)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--output-pdb", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()

    metadata = load_metadata(args.metadata_json)
    mask = None
    if args.frames:
        payload = load_array(args.frames)
        if isinstance(payload, dict):
            key = "flow_trajectory" if "flow_trajectory" in payload else "frames"
            frames = as_tensor(payload, key)
            if "mask" in payload:
                mask = as_tensor(payload, "mask")
        else:
            frames = as_tensor(payload)
        frames = first_batch(frames, 3)
    else:
        payload = load_array(args.trajectory_npz)
        noisy = as_tensor(payload, "noisy_coords")
        velocity = as_tensor(payload, "pred_velocity")
        mask = as_tensor(payload, "mask") if isinstance(payload, dict) and "mask" in payload else None
        noisy = first_batch(noisy, 2).unsqueeze(0)
        velocity = first_batch(velocity, 2).unsqueeze(0)
        if mask is None:
            mask = torch.ones(noisy.shape[:2], dtype=torch.bool)
        else:
            mask = first_batch(mask.bool(), 1).unsqueeze(0)
        frames = rectified_flow_trajectory_frames(noisy, velocity, mask, steps=args.steps)[0]
    if args.noisy_coords is not None or args.pred_velocity is not None:
        if args.noisy_coords is None or args.pred_velocity is None:
            raise ValueError("--noisy-coords and --pred-velocity must be supplied together")
        noisy = first_batch(as_tensor(load_array(args.noisy_coords)), 2).unsqueeze(0)
        velocity = first_batch(as_tensor(load_array(args.pred_velocity)), 2).unsqueeze(0)
        if args.mask is not None:
            mask = first_batch(as_tensor(load_array(args.mask)).bool(), 1).unsqueeze(0)
        else:
            mask = torch.ones(noisy.shape[:2], dtype=torch.bool)
        frames = rectified_flow_trajectory_frames(noisy, velocity, mask, steps=args.steps)[0]
    if args.mask is not None and args.noisy_coords is None:
        mask = first_batch(as_tensor(load_array(args.mask)).bool(), 1)

    output = write_multimodel_pdb(
        frames,
        args.output_pdb,
        mask=mask,
        residue_names=metadata.get("residue_names"),
        chain_ids=metadata.get("chain_ids"),
        residue_indices=metadata.get("residue_indices"),
        atom_modalities=metadata.get("atom_modalities"),
        b_factors=metadata.get("b_factors") or metadata.get("plddt"),
        remarks=[
            "Exported from ToricBLM structure flow-matching frames.",
            "Frames are denoising states, not physical time integration.",
        ],
    )
    print(json.dumps({"output_pdb": str(output), "frames": int(frames.shape[0]), "atoms": int(frames.shape[1])}, indent=2))


if __name__ == "__main__":
    main()
