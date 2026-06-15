"""Finite toric vector-bundle and sheaf probes for ToricGT training.

This module implements the trainable part of the Klyachko/sheaf story in a
bounded form.  A toric vector bundle on a toric variety is represented by a
shared fiber together with compatible filtrations indexed by fan
one-dimensional cones.  The
probe below uses an exact finite certificate for those filtrations and trains
hidden states to expose:

* boundary one-dimensional-cone labels for tropical/toric compactification strata,
* membership in Klyachko filtration subspaces,
* local splitting over affine toric charts, and
* Cech-style gluing of local sheaf sections across chart overlaps.

The certificate is deliberately small and deterministic.  It is a training and
metric layer, not a replacement for Sage/Macaulay2 computations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ToricVectorBundleConfig:
    """Configuration for the finite Klyachko/sheaf probe."""

    rank: int = 8
    num_rays: int = 8
    num_cones: int = 8
    filtration_levels: int = 3
    max_positions: int = 128
    temperature: float = 0.25
    ray_weight: float = 0.20
    filtration_weight: float = 1.00
    splitting_weight: float = 0.35
    cech_weight: float = 0.25
    cocycle_weight: float = 0.05


@dataclass(frozen=True)
class KlyachkoBundleCertificate:
    """Finite toric vector-bundle certificate.

    ``filtration_masks[r, l]`` is the basis mask for the level ``l`` subspace
    of the decreasing filtration attached to the one-dimensional cone indexed
    by ``r``.  ``chart_frames`` are orthogonal bases for affine charts.
    Transition matrices derived from these frames satisfy the Cech cocycle
    identity exactly up to floating-point round off.
    """

    rays: torch.Tensor
    cone_ray_mask: torch.Tensor
    filtration_masks: torch.Tensor
    chart_frames: torch.Tensor
    cone_adjacency: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.filtration_masks.shape[-1])

    @property
    def num_rays(self) -> int:
        return int(self.filtration_masks.shape[0])

    @property
    def num_cones(self) -> int:
        return int(self.cone_ray_mask.shape[0])

    @property
    def filtration_levels(self) -> int:
        return int(self.filtration_masks.shape[1])


def default_klyachko_certificate(
    *,
    rank: int = 8,
    num_rays: int = 8,
    num_cones: int | None = None,
    filtration_levels: int = 3,
    device: torch.device | None = None,
) -> KlyachkoBundleCertificate:
    """Construct a small exact Klyachko-style certificate.

    The fan is the cyclic two-dimensional fan whose maximal cones are spanned
    by adjacent one-dimensional cones.  The filtration masks are nested
    coordinate subspaces; this makes the certificate itself exactly compatible
    while still giving hidden states a nontrivial membership and
    chart-splitting target.
    """

    rank = max(2, int(rank))
    num_rays = max(2, int(num_rays))
    num_cones = max(1, int(num_rays if num_cones is None else num_cones))
    filtration_levels = max(2, int(filtration_levels))
    device = device or torch.device("cpu")

    angles = torch.arange(num_rays, device=device, dtype=torch.float32) * (2.0 * math.pi / float(num_rays))
    rays = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    cone_ray_mask = torch.zeros(num_cones, num_rays, dtype=torch.bool, device=device)
    for cone in range(num_cones):
        left = cone % num_rays
        right = (cone + 1) % num_rays
        cone_ray_mask[cone, left] = True
        cone_ray_mask[cone, right] = True

    coords = torch.arange(rank, device=device)
    filtration_masks = torch.zeros(num_rays, filtration_levels, rank, dtype=torch.float32, device=device)
    for ray in range(num_rays):
        classes = torch.remainder(coords + ray, filtration_levels)
        for level in range(filtration_levels):
            filtration_masks[ray, level] = (classes >= level).to(torch.float32)

    eye = torch.eye(rank, device=device)
    frames = []
    for cone in range(num_cones):
        frame = torch.roll(eye, shifts=cone % rank, dims=0)
        signs = torch.where(
            torch.remainder(torch.arange(rank, device=device) + cone, 2) == 0,
            torch.ones(rank, device=device),
            -torch.ones(rank, device=device),
        )
        frames.append(frame * signs[:, None])
    chart_frames = torch.stack(frames, dim=0)

    cone_adjacency = torch.zeros(num_cones, num_cones, dtype=torch.bool, device=device)
    for cone in range(num_cones):
        cone_adjacency[cone, (cone - 1) % num_cones] = True
        cone_adjacency[cone, (cone + 1) % num_cones] = True
    cone_adjacency.fill_diagonal_(False)

    return KlyachkoBundleCertificate(
        rays=rays,
        cone_ray_mask=cone_ray_mask,
        filtration_masks=filtration_masks,
        chart_frames=chart_frames,
        cone_adjacency=cone_adjacency,
    )


def klyachko_nesting_residual(filtration_masks: torch.Tensor) -> torch.Tensor:
    """Return exact finite residual for decreasing filtrations."""

    if filtration_masks.ndim != 3 or filtration_masks.shape[1] < 2:
        return filtration_masks.float().sum() * 0.0
    coarse = filtration_masks[:, :-1, :].float()
    fine = filtration_masks[:, 1:, :].float()
    return F.relu(fine - coarse).pow(2).mean()


def cech_cocycle_residual(chart_frames: torch.Tensor, cone_adjacency: torch.Tensor) -> torch.Tensor:
    """Check the chart transition cocycle ``T_ac = T_bc T_ab``."""

    if chart_frames.ndim != 3 or chart_frames.shape[0] < 3:
        return chart_frames.float().sum() * 0.0
    frames = chart_frames.float()
    adjacency = cone_adjacency.to(device=frames.device)
    terms = []
    count = frames.shape[0]
    for a in range(count):
        for b in range(count):
            if not bool(adjacency[a, b]):
                continue
            for c in range(count):
                if bool(adjacency[b, c]) and bool(adjacency[a, c]):
                    t_ab = frames[b] @ frames[a].transpose(0, 1)
                    t_bc = frames[c] @ frames[b].transpose(0, 1)
                    t_ac = frames[c] @ frames[a].transpose(0, 1)
                    terms.append((t_bc @ t_ab - t_ac).pow(2).mean())
    if not terms:
        return frames.sum() * 0.0
    return torch.stack(terms).mean()


class ToricVectorBundleProbe(nn.Module):
    """Training-only Klyachko vector-bundle/sheaf probe."""

    def __init__(self, d_model: int, config: ToricVectorBundleConfig | None = None) -> None:
        super().__init__()
        self.config = config or ToricVectorBundleConfig()
        rank = max(2, int(self.config.rank))
        rays = max(2, int(self.config.num_rays))
        levels = max(2, int(self.config.filtration_levels))
        self.fiber = nn.Linear(d_model, rank)
        self.ray_head = nn.Linear(rank, rays)
        self.level_head = nn.Linear(rank, levels)
        cert = default_klyachko_certificate(
            rank=rank,
            num_rays=rays,
            num_cones=int(self.config.num_cones),
            filtration_levels=levels,
        )
        self.register_buffer("rays", cert.rays, persistent=False)
        self.register_buffer("cone_ray_mask", cert.cone_ray_mask, persistent=False)
        self.register_buffer("filtration_masks", cert.filtration_masks, persistent=False)
        self.register_buffer("chart_frames", cert.chart_frames, persistent=False)
        self.register_buffer("cone_adjacency", cert.cone_adjacency, persistent=False)

    def forward(
        self,
        hidden: torch.Tensor,
        target_positions: torch.Tensor | None = None,
        target_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hidden.ndim != 3 or hidden.shape[1] < 2:
            zero = hidden.float().sum() * 0.0
            return self._zero_like(zero)
        max_positions = max(2, int(self.config.max_positions))
        x = hidden.float()
        if x.shape[1] > max_positions:
            index = torch.linspace(0, x.shape[1] - 1, steps=max_positions, device=x.device).round().long()
            x = x.index_select(1, index)
            if target_positions is not None:
                target_positions = target_positions.index_select(1, index)
            if target_tokens is not None:
                target_tokens = target_tokens.index_select(1, index)

        fiber = torch.tanh(self.fiber(x))
        flat_fiber = fiber.reshape(-1, fiber.shape[-1])
        ray_logits = self.ray_head(flat_fiber)
        level_logits = self.level_head(flat_fiber)
        ray_probs = F.softmax(ray_logits / max(float(self.config.temperature), 1e-4), dim=-1)
        level_probs = F.softmax(level_logits / max(float(self.config.temperature), 1e-4), dim=-1)

        ray_labels = self._ray_labels(ray_probs, target_positions)
        level_labels = self._level_labels(level_probs, target_positions, target_tokens)
        ray_ce = F.cross_entropy(ray_logits, ray_labels)
        level_ce = F.cross_entropy(level_logits, level_labels)
        filtration = self._filtration_membership(flat_fiber, ray_labels, level_labels)
        splitting = self._cone_splitting_residual(fiber, ray_probs.view(fiber.shape[0], fiber.shape[1], -1))
        gluing = self._cech_gluing_residual(fiber, ray_probs.view(fiber.shape[0], fiber.shape[1], -1))
        nesting = klyachko_nesting_residual(self.filtration_masks)
        cocycle = cech_cocycle_residual(self.chart_frames, self.cone_adjacency)
        total = (
            float(self.config.ray_weight) * (ray_ce + 0.25 * level_ce)
            + float(self.config.filtration_weight) * filtration
            + float(self.config.splitting_weight) * splitting
            + float(self.config.cech_weight) * gluing
            + float(self.config.cocycle_weight) * (nesting + cocycle)
        )
        ray_entropy = -(ray_probs.clamp_min(1e-8) * ray_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        ray_entropy = ray_entropy / math.log(max(2, int(self.rays.shape[0])))
        active_ray_mass = ray_probs.max(dim=-1).values.mean()
        return {
            "toric_vector_bundle_loss": total,
            "toric_vector_bundle_ray_ce": ray_ce.detach(),
            "toric_vector_bundle_level_ce": level_ce.detach(),
            "toric_vector_bundle_filtration_residual": filtration.detach(),
            "toric_vector_bundle_klyachko_nesting_residual": nesting.detach(),
            "toric_vector_bundle_cone_splitting_residual": splitting.detach(),
            "toric_vector_bundle_cech_gluing_residual": gluing.detach(),
            "toric_sheaf_chart_gluing_residual": gluing.detach(),
            "toric_sheaf_cocycle_residual": cocycle.detach(),
            "toric_vector_bundle_ray_entropy": ray_entropy.detach(),
            "toric_vector_bundle_active_ray_mass": active_ray_mass.detach(),
            "toric_vector_bundle_rank": torch.as_tensor(float(self.filtration_masks.shape[-1]), device=hidden.device),
            "toric_vector_bundle_num_rays": torch.as_tensor(float(self.rays.shape[0]), device=hidden.device),
            "toric_vector_bundle_filtration_levels": torch.as_tensor(
                float(self.filtration_masks.shape[1]),
                device=hidden.device,
            ),
        }

    def _ray_labels(self, probs: torch.Tensor, target_positions: torch.Tensor | None) -> torch.Tensor:
        rays = int(self.rays.shape[0])
        if target_positions is not None:
            return torch.remainder(target_positions.reshape(-1).to(device=probs.device, dtype=torch.long), rays)
        return probs.detach().argmax(dim=-1)

    def _level_labels(
        self,
        probs: torch.Tensor,
        target_positions: torch.Tensor | None,
        target_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        levels = int(self.filtration_masks.shape[1])
        if target_tokens is not None:
            return torch.remainder(target_tokens.reshape(-1).to(device=probs.device, dtype=torch.long), levels)
        if target_positions is not None:
            return torch.remainder(target_positions.reshape(-1).to(device=probs.device, dtype=torch.long), levels)
        return probs.detach().argmax(dim=-1)

    def _filtration_membership(
        self,
        flat_fiber: torch.Tensor,
        ray_labels: torch.Tensor,
        level_labels: torch.Tensor,
    ) -> torch.Tensor:
        masks = self.filtration_masks.to(device=flat_fiber.device, dtype=flat_fiber.dtype)
        selected = masks[ray_labels, level_labels]
        residual = flat_fiber * (1.0 - selected)
        denom = flat_fiber.pow(2).mean(dim=-1).clamp_min(1e-6)
        return (residual.pow(2).mean(dim=-1) / denom).mean()

    def _cone_weights(self, ray_probs: torch.Tensor) -> torch.Tensor:
        cone_mask = self.cone_ray_mask.to(device=ray_probs.device, dtype=ray_probs.dtype)
        weights = ray_probs @ cone_mask.transpose(0, 1)
        return weights / cone_mask.sum(dim=-1).clamp_min(1.0)

    def _cone_splitting_residual(self, fiber: torch.Tensor, ray_probs: torch.Tensor) -> torch.Tensor:
        cone_weights = self._cone_weights(ray_probs)
        frames = self.chart_frames.to(device=fiber.device, dtype=fiber.dtype)
        terms = []
        flat = fiber.reshape(-1, fiber.shape[-1])
        flat_weights = cone_weights.reshape(-1, cone_weights.shape[-1])
        for cone in range(frames.shape[0]):
            weights = flat_weights[:, cone]
            if weights.detach().sum() <= 1e-8:
                continue
            local = flat @ frames[cone].transpose(0, 1)
            centered = local - (weights[:, None] * local).sum(dim=0, keepdim=True) / weights.sum().clamp_min(1e-6)
            cov = (centered * weights[:, None]).transpose(0, 1) @ centered / weights.sum().clamp_min(1e-6)
            off_diag = cov - torch.diag(torch.diagonal(cov))
            terms.append(off_diag.pow(2).mean())
        if not terms:
            return fiber.float().sum() * 0.0
        return torch.stack(terms).mean()

    def _cech_gluing_residual(self, fiber: torch.Tensor, ray_probs: torch.Tensor) -> torch.Tensor:
        cone_weights = self._cone_weights(ray_probs)
        frames = self.chart_frames.to(device=fiber.device, dtype=fiber.dtype)
        adjacency = self.cone_adjacency.to(device=fiber.device)
        flat = fiber.reshape(-1, fiber.shape[-1])
        flat_weights = cone_weights.reshape(-1, cone_weights.shape[-1])
        local_means = []
        valid = []
        for cone in range(frames.shape[0]):
            weights = flat_weights[:, cone]
            mass = weights.sum()
            if mass.detach() <= 1e-8:
                local_means.append(torch.zeros(frames.shape[-1], device=fiber.device, dtype=fiber.dtype))
                valid.append(False)
                continue
            local = flat @ frames[cone].transpose(0, 1)
            local_means.append((weights[:, None] * local).sum(dim=0) / mass.clamp_min(1e-6))
            valid.append(True)
        terms = []
        for left in range(frames.shape[0]):
            if not valid[left]:
                continue
            for right in range(frames.shape[0]):
                if not valid[right] or not bool(adjacency[left, right]):
                    continue
                transition = frames[right] @ frames[left].transpose(0, 1)
                transported = local_means[left] @ transition.transpose(0, 1)
                terms.append((transported - local_means[right]).pow(2).mean())
        if not terms:
            return fiber.float().sum() * 0.0
        return torch.stack(terms).mean()

    @staticmethod
    def _zero_like(zero: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "toric_vector_bundle_loss": zero,
            "toric_vector_bundle_ray_ce": zero.detach(),
            "toric_vector_bundle_level_ce": zero.detach(),
            "toric_vector_bundle_filtration_residual": zero.detach(),
            "toric_vector_bundle_klyachko_nesting_residual": zero.detach(),
            "toric_vector_bundle_cone_splitting_residual": zero.detach(),
            "toric_vector_bundle_cech_gluing_residual": zero.detach(),
            "toric_sheaf_chart_gluing_residual": zero.detach(),
            "toric_sheaf_cocycle_residual": zero.detach(),
            "toric_vector_bundle_ray_entropy": zero.detach(),
            "toric_vector_bundle_active_ray_mass": zero.detach(),
            "toric_vector_bundle_rank": zero.detach(),
            "toric_vector_bundle_num_rays": zero.detach(),
            "toric_vector_bundle_filtration_levels": zero.detach(),
        }
