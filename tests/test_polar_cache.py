import torch

from toricgt.polar_cache import (
    deterministic_signs,
    hadamard_precondition,
    inverse_hadamard_precondition,
    polarquant_memory_estimate,
    recursive_polar_decode,
    recursive_polar_encode,
    score_perturbation_bound,
    uniform_quantize_angles,
)


def test_recursive_polar_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(3, 8)
    cache = recursive_polar_encode(x)
    y = recursive_polar_decode(cache)
    assert torch.allclose(x, y, atol=1e-5, rtol=1e-5)


def test_quantized_angles_are_finite_and_bounded():
    x = torch.randn(2, 16)
    cache = uniform_quantize_angles(recursive_polar_encode(x), bits=4)
    y = recursive_polar_decode(cache)
    assert torch.isfinite(y).all()
    assert score_perturbation_bound(query_norm=2.0, key_error=0.5, head_dim=16) == 0.25


def test_hadamard_precondition_is_orthonormal_and_invertible():
    torch.manual_seed(3)
    x = torch.randn(5, 8)
    signs = deterministic_signs(8, seed=123)

    y = hadamard_precondition(x, signs)
    z = inverse_hadamard_precondition(y, signs)

    assert torch.allclose(x, z, atol=1e-6, rtol=1e-6)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(x @ x.T, y @ y.T, atol=1e-5, rtol=1e-5)


def test_polarquant_memory_estimate_tracks_packed_cache_savings():
    estimate = polarquant_memory_estimate(
        (2, 7, 6, 2048, 64),
        angle_bits=4,
        radius_bits=16,
        dtype_bits=16,
        include_sign_bits=True,
    )

    assert estimate.values_per_vector == 64
    assert estimate.fp_bytes > estimate.polarquant_bytes
    assert estimate.compression_ratio > 2.0
    assert estimate.saved_bytes == estimate.fp_bytes - estimate.polarquant_bytes
