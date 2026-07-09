"""Coordinate-native structure losses for ToricBLM training phases.

The current UniProt/FoT curation may contain only structure hooks such as PDB
or AlphaFold identifiers.  These functions are intentionally coordinate-native:
they compute real geometry losses when coordinate tensors are present, and
callers should keep them disabled for hook-only records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class StructureFlowMetrics:
    """Scalar diagnostics returned with structure-flow losses."""

    loss: Tensor
    flow_matching_loss: Tensor
    contact_bce_loss: Tensor
    distogram_loss: Tensor
    rmsd: Tensor
    coordinate_count: Tensor


def masked_mean(value: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    weighted = value * mask.to(dtype=value.dtype)
    return weighted.sum() / mask.to(dtype=value.dtype).sum().clamp_min(eps)


def pairwise_distances(coords: Tensor, mask: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    """Return pairwise Euclidean distances for batched 3D coordinates."""

    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape [batch, atoms, 3]")
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist = diff.square().sum(dim=-1).clamp_min(eps).sqrt()
    if mask is None:
        return dist
    if mask.ndim != 2 or mask.shape != coords.shape[:2]:
        raise ValueError("mask must have shape [batch, atoms]")
    pair_mask = mask[:, :, None] & mask[:, None, :]
    return dist * pair_mask.to(dtype=dist.dtype)


def contact_map(coords: Tensor, mask: Tensor | None = None, cutoff: float = 8.0) -> Tensor:
    contacts = (pairwise_distances(coords, mask) <= float(cutoff)).to(dtype=coords.dtype)
    eye = torch.eye(coords.shape[1], device=coords.device, dtype=torch.bool).unsqueeze(0)
    contacts = contacts.masked_fill(eye, 0.0)
    if mask is not None:
        pair_mask = mask[:, :, None] & mask[:, None, :]
        contacts = contacts * pair_mask.to(dtype=contacts.dtype)
    return contacts


def contact_bce_from_logits(
    pred_contact_logits: Tensor,
    target_coords: Tensor,
    mask: Tensor,
    cutoff: float = 8.0,
) -> Tensor:
    if pred_contact_logits.shape != (*target_coords.shape[:2], target_coords.shape[1]):
        raise ValueError("pred_contact_logits must have shape [batch, atoms, atoms]")
    target = contact_map(target_coords, mask, cutoff=cutoff)
    pair_mask = (mask[:, :, None] & mask[:, None, :]).to(dtype=pred_contact_logits.dtype)
    eye = torch.eye(target_coords.shape[1], device=target_coords.device, dtype=torch.bool).unsqueeze(0)
    pair_mask = pair_mask.masked_fill(eye, 0.0)
    loss = F.binary_cross_entropy_with_logits(pred_contact_logits.float(), target.float(), reduction="none")
    return masked_mean(loss, pair_mask)


def distogram_targets(target_coords: Tensor, mask: Tensor, bin_edges: Tensor) -> Tensor:
    if bin_edges.ndim != 1 or bin_edges.numel() < 2:
        raise ValueError("bin_edges must be a one-dimensional tensor with at least two edges")
    dist = pairwise_distances(target_coords, mask)
    return torch.bucketize(dist, bin_edges.to(device=target_coords.device, dtype=dist.dtype)[1:-1])


def distogram_cross_entropy(pred_logits: Tensor, target_coords: Tensor, mask: Tensor, bin_edges: Tensor) -> Tensor:
    if pred_logits.ndim != 4:
        raise ValueError("pred_logits must have shape [batch, atoms, atoms, bins]")
    if pred_logits.shape[:3] != (*target_coords.shape[:2], target_coords.shape[1]):
        raise ValueError("pred_logits leading dimensions must match [batch, atoms, atoms]")
    target = distogram_targets(target_coords, mask, bin_edges)
    pair_mask = (mask[:, :, None] & mask[:, None, :]).to(dtype=pred_logits.dtype)
    eye = torch.eye(target_coords.shape[1], device=target_coords.device, dtype=torch.bool).unsqueeze(0)
    pair_mask = pair_mask.masked_fill(eye, 0.0)
    loss = F.cross_entropy(pred_logits.float().permute(0, 3, 1, 2), target.long(), reduction="none")
    return masked_mean(loss, pair_mask)


def flow_matching_mse(pred_velocity: Tensor, noisy_coords: Tensor, target_coords: Tensor, mask: Tensor) -> Tensor:
    """Conditional flow-matching MSE against the rectified-flow velocity."""

    if pred_velocity.shape != target_coords.shape or noisy_coords.shape != target_coords.shape:
        raise ValueError("pred_velocity, noisy_coords, and target_coords must have identical shapes")
    target_velocity = target_coords - noisy_coords
    atom_mask = mask.unsqueeze(-1).to(dtype=pred_velocity.dtype)
    return masked_mean((pred_velocity.float() - target_velocity.float()).square(), atom_mask)


def centered_rmsd(pred_coords: Tensor, target_coords: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    if pred_coords.shape != target_coords.shape:
        raise ValueError("pred_coords and target_coords must have identical shapes")
    atom_mask = mask.unsqueeze(-1).to(dtype=pred_coords.dtype)
    denom = atom_mask.sum(dim=1, keepdim=True).clamp_min(eps)
    pred_center = (pred_coords * atom_mask).sum(dim=1, keepdim=True) / denom
    target_center = (target_coords * atom_mask).sum(dim=1, keepdim=True) / denom
    err = ((pred_coords - pred_center) - (target_coords - target_center)).square().sum(dim=-1)
    return masked_mean(err.clamp_min(0.0).sqrt(), mask)


def structure_flow_loss(
    *,
    pred_velocity: Tensor,
    noisy_coords: Tensor,
    target_coords: Tensor,
    mask: Tensor,
    pred_contact_logits: Tensor | None = None,
    pred_distogram_logits: Tensor | None = None,
    distogram_bin_edges: Tensor | None = None,
    flow_weight: float = 1.0,
    contact_weight: float = 0.0,
    distogram_weight: float = 0.0,
) -> StructureFlowMetrics:
    flow_loss = flow_matching_mse(pred_velocity, noisy_coords, target_coords, mask)
    zero = flow_loss.new_zeros(())
    contact_loss = zero
    if pred_contact_logits is not None and contact_weight > 0.0:
        contact_loss = contact_bce_from_logits(pred_contact_logits, target_coords, mask)
    dist_loss = zero
    if pred_distogram_logits is not None and distogram_bin_edges is not None and distogram_weight > 0.0:
        dist_loss = distogram_cross_entropy(pred_distogram_logits, target_coords, mask, distogram_bin_edges)
    pred_coords = noisy_coords + pred_velocity
    rmsd = centered_rmsd(pred_coords, target_coords, mask)
    total = float(flow_weight) * flow_loss + float(contact_weight) * contact_loss + float(distogram_weight) * dist_loss
    return StructureFlowMetrics(
        loss=total,
        flow_matching_loss=flow_loss,
        contact_bce_loss=contact_loss,
        distogram_loss=dist_loss,
        rmsd=rmsd,
        coordinate_count=mask.to(dtype=flow_loss.dtype).sum(),
    )


def rectified_flow_trajectory_frames(
    noisy_coords: Tensor,
    pred_velocity: Tensor,
    mask: Tensor,
    *,
    steps: int = 16,
) -> Tensor:
    """Return denoising frames for the current rectified-flow structure head.

    The present ToricBLM structure head predicts a rectified-flow velocity from
    a noisy coordinate cloud.  With that parameterization, the optional
    generation trajectory is the model-implied path

    ``x(t) = x_noisy + t v_theta(x_noisy, context)``, ``t in [0, 1]``.

    This is a denoising/generation trajectory for the flow-matching head, not a
    physical molecular-dynamics trajectory.  Later dynamics training should use
    a separate trajectory schema with time units, energies, forces, ensembles,
    and integrator metadata.
    """

    if noisy_coords.shape != pred_velocity.shape:
        raise ValueError("noisy_coords and pred_velocity must have identical shape")
    if noisy_coords.ndim != 3 or noisy_coords.shape[-1] != 3:
        raise ValueError("noisy_coords must have shape [batch, atoms, 3]")
    if mask.shape != noisy_coords.shape[:2]:
        raise ValueError("mask must have shape [batch, atoms]")
    num_steps = max(2, int(steps))
    t = torch.linspace(0.0, 1.0, num_steps, device=noisy_coords.device, dtype=noisy_coords.dtype)
    frames = noisy_coords[:, None, :, :] + t[None, :, None, None] * pred_velocity[:, None, :, :]
    return frames * mask[:, None, :, None].to(dtype=frames.dtype)


def _as_python_frames(frames: Tensor | Any) -> list[list[list[float]]]:
    if isinstance(frames, Tensor):
        data = frames.detach().cpu().float().tolist()
    elif hasattr(frames, "tolist"):
        data = frames.tolist()
    else:
        data = frames
    if not isinstance(data, list) or not data:
        raise ValueError("frames must be a nonempty [frames, atoms, 3] array")
    out: list[list[list[float]]] = []
    for frame in data:
        if not isinstance(frame, list):
            raise ValueError("each frame must be a list of atom coordinates")
        frame_out: list[list[float]] = []
        for xyz in frame:
            if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
                raise ValueError("each atom coordinate must have at least three values")
            frame_out.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
        out.append(frame_out)
    return out


def _mask_to_list(mask: Tensor | Any | None, atoms: int) -> list[bool]:
    if mask is None:
        return [True] * atoms
    if isinstance(mask, Tensor):
        data = mask.detach().cpu().bool().tolist()
    elif hasattr(mask, "tolist"):
        data = mask.tolist()
    else:
        data = mask
    if data and isinstance(data[0], list):
        data = data[0]
    if not isinstance(data, list):
        raise ValueError("mask must be a one-dimensional boolean array or a [1, atoms] array")
    return [bool(x) for x in data[:atoms]] + [False] * max(0, atoms - len(data))


def _pdb_atom_line(
    *,
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_index: int,
    x: float,
    y: float,
    z: float,
    b_factor: float,
) -> str:
    element = "".join(ch for ch in atom_name.strip() if ch.isalpha())[:1] or "C"
    return (
        f"ATOM  {serial:5d} {atom_name[:4]:>4s} {residue_name[:3]:>3s} {chain_id[:1] or 'A'}"
        f"{residue_index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{b_factor:6.2f}"
        f"          {element:>2s}\n"
    )


def write_multimodel_pdb(
    frames: Tensor | Any,
    path: str | Path,
    *,
    mask: Tensor | Any | None = None,
    residue_names: list[str] | None = None,
    chain_ids: list[str] | None = None,
    residue_indices: list[int] | None = None,
    atom_modalities: list[str] | None = None,
    b_factors: list[float] | None = None,
    atom_name: str = "CA",
    remarks: list[str] | None = None,
) -> Path:
    """Write model-produced denoising frames as a multi-``MODEL`` PDB file.

    The function writes exactly the supplied frames.  It does not generate
    conformers, infer missing coordinates, or simulate physical dynamics.
    """

    frame_list = _as_python_frames(frames)
    atoms = len(frame_list[0])
    if atoms == 0:
        raise ValueError("cannot write a trajectory with zero atoms")
    for frame in frame_list:
        if len(frame) != atoms:
            raise ValueError("all frames must have the same atom count")
    active = _mask_to_list(mask, atoms)
    residue_names = residue_names or ["GLY"] * atoms
    chain_ids = chain_ids or ["A"] * atoms
    residue_indices = residue_indices or list(range(1, atoms + 1))
    b_factors = b_factors or [0.0] * atoms
    atom_modalities = atom_modalities or ["protein"] * atoms
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="ascii") as handle:
        handle.write("REMARK   1 ToricBLM flow-matching denoising trajectory\n")
        handle.write("REMARK   2 This is not a molecular dynamics trajectory.\n")
        for remark in remarks or []:
            handle.write(f"REMARK   3 {str(remark)[:66]}\n")
        serial_limit = 99999
        for model_index, frame in enumerate(frame_list, start=1):
            handle.write(f"MODEL     {model_index:4d}\n")
            serial = 1
            for atom_index, xyz in enumerate(frame):
                if not active[atom_index]:
                    continue
                modality = atom_modalities[atom_index] if atom_index < len(atom_modalities) else "protein"
                inferred_atom_name = atom_name
                if atom_name == "CA" and modality in {"rna", "dna"}:
                    inferred_atom_name = "P"
                handle.write(
                    _pdb_atom_line(
                        serial=((serial - 1) % serial_limit) + 1,
                        atom_name=inferred_atom_name,
                        residue_name=residue_names[atom_index] if atom_index < len(residue_names) else "GLY",
                        chain_id=chain_ids[atom_index] if atom_index < len(chain_ids) else "A",
                        residue_index=int(residue_indices[atom_index]) if atom_index < len(residue_indices) else atom_index + 1,
                        x=float(xyz[0]),
                        y=float(xyz[1]),
                        z=float(xyz[2]),
                        b_factor=float(b_factors[atom_index]) if atom_index < len(b_factors) else 0.0,
                    )
                )
                serial += 1
            handle.write("ENDMDL\n")
        handle.write("END\n")
    return out_path
