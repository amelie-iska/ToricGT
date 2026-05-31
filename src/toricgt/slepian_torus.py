"""Slepian/DPSS probes for toric phase trajectories.

Classical prolate spheroidal wave functions and their discrete Slepian/DPSS
counterparts are optimal concentration functions: they maximize energy inside a
time window subject to a bandlimit.  On ToricGT's finite torus projection, the
same operator gives a compact audit of whether a reasoning trajectory has
coherent, localized phase content or diffuse spectral leakage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ToricSlepianConfig:
    """Configuration for a finite toric Slepian audit."""

    half_bandwidth: float = 0.075
    modes: int = 6
    theta: float = (math.sqrt(5.0) - 1.0) / 2.0
    beta: float = math.sqrt(2.0) - 1.0
    min_length: int = 8


def dpss_kernel(length: int, half_bandwidth: float) -> np.ndarray:
    """Return the discrete time-band limiting kernel.

    ``half_bandwidth`` is normalized cycles/sample and should lie in ``(0, .5)``.
    The kernel has entries ``sin(2*pi*W*(m-n))/(pi*(m-n))`` off diagonal and
    ``2W`` on the diagonal.  Its eigenvectors are discrete Slepian sequences.
    """

    n = int(length)
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    w = float(np.clip(half_bandwidth, 1.0 / max(8, 4 * n), 0.49))
    idx = np.arange(n, dtype=np.float64)
    diff = idx[:, None] - idx[None, :]
    kernel = np.empty((n, n), dtype=np.float64)
    np.fill_diagonal(kernel, 2.0 * w)
    mask = diff != 0
    kernel[mask] = np.sin(2.0 * np.pi * w * diff[mask]) / (np.pi * diff[mask])
    return 0.5 * (kernel + kernel.T)


def dpss_basis(length: int, half_bandwidth: float, modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return leading concentration eigenvalues and DPSS vectors."""

    kernel = dpss_kernel(length, half_bandwidth)
    if kernel.size == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    values, vectors = np.linalg.eigh(kernel)
    order = np.argsort(values)[::-1]
    keep = order[: max(1, min(int(modes), length))]
    eig = np.clip(values[keep], 0.0, 1.0)
    vec = vectors[:, keep]
    # Fix signs for deterministic metadata and music envelopes.
    for col in range(vec.shape[1]):
        pivot = int(np.argmax(np.abs(vec[:, col])))
        if vec[pivot, col] < 0:
            vec[:, col] *= -1.0
    return eig.astype(np.float64), vec.astype(np.float64)


def toric_phase_signal(
    phase_u: np.ndarray,
    phase_v: np.ndarray,
    *,
    energy: np.ndarray | None = None,
) -> np.ndarray:
    """Map a T^2 phase path to a real probe signal.

    The signal mixes two circle characters and the cocycle-like difference
    character.  If local NLL energy is supplied, the signal is gently weighted
    so high-energy reasoning segments contribute to the leakage audit.
    """

    u = np.asarray(phase_u, dtype=np.float64).reshape(-1)
    v = np.asarray(phase_v, dtype=np.float64).reshape(-1)
    if u.shape != v.shape:
        raise ValueError("phase_u and phase_v must have the same shape")
    signal = (
        np.cos(2.0 * np.pi * u)
        + 0.75 * np.sin(2.0 * np.pi * v)
        + 0.45 * np.cos(2.0 * np.pi * (u - v))
    )
    if energy is not None:
        e = np.asarray(energy, dtype=np.float64).reshape(-1)
        if e.shape == signal.shape and e.size > 0:
            e = e - np.nanmin(e)
            denom = float(np.nanmax(e)) or 1.0
            signal = signal * (0.75 + 0.50 * (e / denom))
    signal = signal - float(np.nanmean(signal)) if signal.size else signal
    norm = float(np.linalg.norm(signal))
    return signal / norm if norm > 1e-12 else signal


def toric_slepian_audit(
    phase_u: np.ndarray,
    phase_v: np.ndarray,
    *,
    energy: np.ndarray | None = None,
    config: ToricSlepianConfig | None = None,
) -> dict[str, Any]:
    """Compute Slepian concentration metrics for a toric phase path."""

    cfg = config or ToricSlepianConfig()
    signal = toric_phase_signal(phase_u, phase_v, energy=energy)
    n = int(signal.shape[0])
    if n < int(cfg.min_length):
        return {
            "slepian_eigenvalues": np.zeros((0,), dtype=np.float64),
            "slepian_coefficients": np.zeros((0,), dtype=np.float64),
            "slepian_reconstruction": signal,
            "slepian_envelope": np.ones((n,), dtype=np.float64),
            "slepian_concentration": 0.0,
            "slepian_leakage": 1.0,
            "slepian_mode_entropy": 0.0,
            "slepian_effective_modes": 0.0,
            "slepian_bandwidth": float(cfg.half_bandwidth),
        }
    eig, basis = dpss_basis(n, float(cfg.half_bandwidth), int(cfg.modes))
    coeff = basis.T @ signal
    power = coeff**2
    total = float(power.sum())
    if total <= 1e-12:
        concentration = 0.0
        entropy = 0.0
        effective = 0.0
    else:
        concentration = float(np.sum(power * eig) / total)
        probs = power / total
        entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))) / math.log(max(2, len(probs))))
        effective = float(np.exp(-np.sum(probs * np.log(np.maximum(probs, 1e-12)))))
    reconstruction = basis @ coeff
    envelope = np.abs(basis[:, 0])
    if basis.shape[1] > 1:
        envelope = envelope + 0.45 * np.abs(basis[:, 1])
    envelope = envelope / max(float(np.max(envelope)), 1e-12)
    return {
        "slepian_eigenvalues": eig,
        "slepian_coefficients": coeff,
        "slepian_reconstruction": reconstruction,
        "slepian_envelope": envelope,
        "slepian_concentration": concentration,
        "slepian_leakage": float(1.0 - concentration),
        "slepian_mode_entropy": entropy,
        "slepian_effective_modes": effective,
        "slepian_bandwidth": float(cfg.half_bandwidth),
    }


def toric_slepian_envelope(
    length: int,
    *,
    half_bandwidth: float = 0.075,
    modes: int = 3,
) -> np.ndarray:
    """Return a normalized Slepian envelope for deterministic music dynamics."""

    if length <= 0:
        return np.zeros((0,), dtype=np.float64)
    eig, basis = dpss_basis(length, half_bandwidth, modes)
    if basis.size == 0:
        return np.ones((length,), dtype=np.float64)
    weights = eig[: basis.shape[1]]
    envelope = np.abs(basis[:, : len(weights)] @ weights)
    envelope = envelope - float(envelope.min())
    denom = float(envelope.max()) or 1.0
    return 0.35 + 0.65 * (envelope / denom)
