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


@dataclass(frozen=True)
class PolarQuantMemoryEstimate:
    """Packed PolarQuant KV-cache memory estimate for a tensor shape."""

    fp_bytes: int
    polarquant_bytes: int
    angle_bits: int
    radius_bits: int
    sign_bits: int
    vectors: int
    values_per_vector: int

    @property
    def compression_ratio(self) -> float:
        return float(self.fp_bytes) / max(float(self.polarquant_bytes), 1.0)

    @property
    def saved_bytes(self) -> int:
        return max(int(self.fp_bytes) - int(self.polarquant_bytes), 0)


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


def hadamard_precondition(x: torch.Tensor, signs: torch.Tensor | None = None) -> torch.Tensor:
    """Apply a normalized Walsh-Hadamard transform with optional random signs.

    PolarQuant uses random preconditioning before the recursive polar transform.
    A sign-flipped Hadamard transform is an orthogonal, cheap preconditioner
    when the head dimension is a power of two; applying the same transform again
    and then the sign flip inverts it.
    """

    dim = x.shape[-1]
    if dim < 2 or dim & (dim - 1):
        raise ValueError("last dimension must be a power of two and at least 2")
    y = x if signs is None else x * signs.to(device=x.device, dtype=x.dtype)
    y = y.reshape(*y.shape[:-1], 1, dim)
    size = dim
    while size > 1:
        y = y.reshape(*y.shape[:-2], -1, 2, size // 2)
        left = y[..., 0, :]
        right = y[..., 1, :]
        y = torch.cat((left + right, left - right), dim=-1)
        size //= 2
    return y.reshape(*x.shape) / (float(dim) ** 0.5)


def inverse_hadamard_precondition(x: torch.Tensor, signs: torch.Tensor | None = None) -> torch.Tensor:
    """Invert ``hadamard_precondition`` for an orthonormal Hadamard matrix."""

    y = hadamard_precondition(x, signs=None)
    return y if signs is None else y * signs.to(device=x.device, dtype=x.dtype)


def deterministic_signs(dim: int, seed: int = 271828, *, device: torch.device | None = None) -> torch.Tensor:
    """Return deterministic +/-1 signs for PolarQuant preconditioning."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1_000_003 * int(dim))
    signs = torch.randint(0, 2, (int(dim),), generator=generator, dtype=torch.int8)
    signs = signs.to(dtype=torch.float32).mul_(2.0).sub_(1.0)
    return signs if device is None else signs.to(device=device)


def polarquant_memory_estimate(
    shape: tuple[int, ...] | torch.Size,
    *,
    angle_bits: int,
    radius_bits: int = 16,
    dtype_bits: int = 16,
    include_sign_bits: bool = True,
) -> PolarQuantMemoryEstimate:
    """Estimate packed PolarQuant storage for a KV tensor.

    The estimate follows the paper's representation: one radius plus ``d - 1``
    quantized angles per vector.  We also account for sign bits because this
    implementation stores signs separately instead of a first-level [0, 2pi)
    angle.  No per-block scale/zero-point overhead is included, matching the
    PolarQuant motivation of avoiding normalization metadata.
    """

    if len(shape) < 1:
        raise ValueError("shape must include a vector dimension")
    dim = int(shape[-1])
    if dim < 2 or dim & (dim - 1):
        raise ValueError("last dimension must be a power of two and at least 2")
    vectors = 1
    for value in tuple(shape[:-1]):
        vectors *= int(value)
    sign_bits = dim if include_sign_bits else 0
    bits_per_vector = int(radius_bits) + max(dim - 1, 0) * int(angle_bits) + sign_bits
    fp_bits = vectors * dim * int(dtype_bits)
    pq_bits = vectors * bits_per_vector
    return PolarQuantMemoryEstimate(
        fp_bytes=(fp_bits + 7) // 8,
        polarquant_bytes=(pq_bits + 7) // 8,
        angle_bits=int(angle_bits),
        radius_bits=int(radius_bits),
        sign_bits=int(sign_bits),
        vectors=int(vectors),
        values_per_vector=dim,
    )
