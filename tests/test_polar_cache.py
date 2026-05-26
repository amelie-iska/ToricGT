import torch

from toricgt.polar_cache import recursive_polar_decode, recursive_polar_encode, score_perturbation_bound, uniform_quantize_angles


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
