"""Finite approximants to irrational rotation and noncommutative torus algebras."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def continued_fraction_rational(theta: float, max_denominator: int) -> tuple[int, int]:
    """Return a good rational approximant p/q to ``theta`` with q bounded."""

    if max_denominator < 1:
        raise ValueError("max_denominator must be positive")
    x = theta
    a0 = math.floor(x)
    p_nm2, p_nm1 = 0, 1
    q_nm2, q_nm1 = 1, 0
    best = (a0, 1)
    while True:
        a = math.floor(x)
        p = a * p_nm1 + p_nm2
        q = a * q_nm1 + q_nm2
        if q > max_denominator:
            break
        best = (p, q)
        frac = x - a
        if abs(frac) < 1e-15:
            break
        x = 1.0 / frac
        p_nm2, p_nm1 = p_nm1, p
        q_nm2, q_nm1 = q_nm1, q
    return best


def clock_matrix(q: int, phase: float = 1.0) -> np.ndarray:
    """Return the q by q clock matrix C with C_jj = exp(2 pi i phase j/q)."""

    j = np.arange(q, dtype=np.float64)
    return np.diag(np.exp(2j * np.pi * phase * j / q))


def shift_matrix(q: int, twist: complex = 1.0 + 0.0j) -> np.ndarray:
    """Return a periodic shift matrix with optional boundary twist.

    The convention is S e_j = e_{j-1 mod q}, so S C = omega C S for
    C_jj = omega^j.
    """

    s = np.zeros((q, q), dtype=np.complex128)
    for j in range(1, q):
        s[j - 1, j] = 1.0
    s[q - 1, 0] = twist
    return s


@dataclass(frozen=True)
class RotationAlgebraApproximant:
    """Finite q-dimensional Weyl pair approximating A_theta.

    For rational theta = p/q the matrices satisfy V U = omega U V with
    omega = exp(2 pi i p / q). For irrational theta, p/q is chosen by a
    continued-fraction bound.
    """

    theta: float
    max_denominator: int = 257
    twist: complex = 1.0 + 0.0j

    def pq(self) -> tuple[int, int]:
        return continued_fraction_rational(self.theta, self.max_denominator)

    def generators(self) -> tuple[np.ndarray, np.ndarray]:
        p, q = self.pq()
        u = clock_matrix(q, p)
        v = shift_matrix(q, self.twist)
        return u, v

    def commutator_error(self) -> float:
        p, q = self.pq()
        u, v = self.generators()
        omega = np.exp(2j * np.pi * p / q)
        return float(np.linalg.norm(v @ u - omega * u @ v, ord="fro"))


def skew_symmetric_theta(rank: int, theta: float) -> np.ndarray:
    """Build a simple rank x rank skew matrix for higher torus experiments."""

    mat = np.zeros((rank, rank), dtype=np.float64)
    for i in range(rank):
        for j in range(i + 1, rank):
            mat[i, j] = theta / (1 + j - i)
            mat[j, i] = -mat[i, j]
    return mat


def projective_phase(theta_matrix: np.ndarray, a: np.ndarray, b: np.ndarray) -> complex:
    """Return exp(2 pi i a^T Theta b) for lattice vectors a,b."""

    return complex(np.exp(2j * np.pi * float(a @ theta_matrix @ b)))


def braid_generator_points(points: np.ndarray, i: int, t: float) -> np.ndarray:
    """Move marked unit-circle points through the i-th Artin half-twist.

    This is a geometric data generator, not a symbolic braid package. The two
    selected points rotate around their midpoint in the complex plane.
    """

    if not (0 <= i < len(points) - 1):
        raise IndexError("braid index out of range")
    z = np.asarray(points, dtype=np.complex128).copy()
    a, b = z[i], z[i + 1]
    center = 0.5 * (a + b)
    rel_a = a - center
    rel_b = b - center
    angle = np.pi * t
    z[i] = center + rel_a * np.exp(1j * angle)
    z[i + 1] = center + rel_b * np.exp(1j * angle)
    return z / np.maximum(np.abs(z), 1e-12)


def affine_type_a_simple_reflection(x: np.ndarray, i: int) -> np.ndarray:
    """Simple reflections for affine type A on cyclic coordinates.

    For i < n-1 this swaps adjacent coordinates. The affine reflection swaps
    x_{n-1} and x_0 with a unit translation that preserves the sum modulo n.
    """

    y = np.asarray(x, dtype=np.float64).copy()
    n = len(y)
    if not (0 <= i < n):
        raise IndexError("reflection index out of range")
    if i < n - 1:
        y[i], y[i + 1] = y[i + 1], y[i]
    else:
        y[0], y[-1] = y[-1] + 1.0, y[0] - 1.0
    return y
