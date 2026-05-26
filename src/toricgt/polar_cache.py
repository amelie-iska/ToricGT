"""Polar-coordinate utilities for optional long-context KV-cache compression."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PolarCache:
    """Exact recursive polar representation before quantized packing."""

    radius: torch.Tensor
    angles: tuple[torch.Tensor, ...]
    signs: torch.Tensor
    original_dim: int


def recursive_polar_encode(x: torch.Tensor, eps: float = 1e-12) -> PolarCache:
    """Encode the last dimension of ``x`` by recursive radii and angles.

    The dimension must be a power of two. This exact representation is a
    building block for later angle quantization and bit packing.
    """

    dim = x.shape[-1]
    if dim < 2 or dim & (dim - 1):
        raise ValueError("last dimension must be a power of two and at least 2")
    signs = torch.sign(x)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    current = x.abs()
    angles: list[torch.Tensor] = []
    while current.shape[-1] > 1:
        pairs = current.reshape(*current.shape[:-1], current.shape[-1] // 2, 2)
        left = pairs[..., 0]
        right = pairs[..., 1]
        radius = torch.sqrt(left.square() + right.square()).clamp_min(eps)
        angles.append(torch.atan2(right, left))
        current = radius
    return PolarCache(radius=current.squeeze(-1), angles=tuple(angles), signs=signs, original_dim=dim)


def recursive_polar_decode(cache: PolarCache) -> torch.Tensor:
    current = cache.radius.unsqueeze(-1)
    for angle in reversed(cache.angles):
        left = current * torch.cos(angle)
        right = current * torch.sin(angle)
        current = torch.stack([left, right], dim=-1).flatten(-2)
    return current * cache.signs


def uniform_quantize_angles(cache: PolarCache, bits: int = 8) -> PolarCache:
    """Return a copy with angles quantized uniformly on ``[0, pi/2]``."""

    if bits < 1:
        raise ValueError("bits must be positive")
    levels = 2**bits - 1
    quantized = []
    for angle in cache.angles:
        clipped = angle.clamp(0, torch.pi / 2)
        idx = torch.round(clipped / (torch.pi / 2) * levels)
        quantized.append(idx / levels * (torch.pi / 2))
    return PolarCache(radius=cache.radius, angles=tuple(quantized), signs=cache.signs, original_dim=cache.original_dim)


def score_perturbation_bound(query_norm: float, key_error: float, head_dim: int) -> float:
    return query_norm * key_error / (head_dim**0.5)
