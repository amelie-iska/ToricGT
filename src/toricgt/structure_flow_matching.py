"""Coordinate-native structure losses for ToricBLM training phases.

The current UniProt/FoT curation may contain only structure hooks such as PDB
or AlphaFold identifiers.  These functions are intentionally coordinate-native:
they compute real geometry losses when coordinate tensors are present, and
callers should keep them disabled for hook-only records.
"""

from __future__ import annotations

from dataclasses import dataclass

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
