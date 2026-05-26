import numpy as np

from toricgt.toric_algebra import RotationAlgebraApproximant


def test_rational_clock_shift_relation():
    approx = RotationAlgebraApproximant(theta=2 / 5, max_denominator=5)
    assert approx.pq() == (2, 5)
    assert approx.commutator_error() < 1e-10


def test_irrational_approximant_is_bounded():
    approx = RotationAlgebraApproximant(theta=(np.sqrt(5) - 1) / 2, max_denominator=32)
    _, q = approx.pq()
    assert q <= 32

