import math

import torch

from toricgt.tropical_attention import MultiHeadTropicalAttention, tropical_attention, tropical_ring_attention


def test_tropical_ring_matches_full_attention():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 9, 4)
    k = torch.randn(2, 3, 9, 4)
    v = torch.randn(2, 3, 9, 5)
    full = tropical_attention(q, k, v)
    ring = tropical_ring_attention(q, k, v, block_size=4)
    assert torch.allclose(full, ring, atol=1e-6)


def test_sampled_polarquant_only_perturbs_sampled_tokens_in_training():
    torch.manual_seed(0)
    attn = MultiHeadTropicalAttention(
        d_model=8,
        num_heads=2,
        mode="softmax",
        polarquant_kv_bits=3,
        polarquant_train=True,
        polarquant_train_sample_tokens=3,
    )
    attn.train()
    x = torch.linspace(-1.7, 2.3, steps=1 * 2 * 8 * 4).reshape(1, 2, 8, 4)

    out = attn._maybe_polarquant(x)

    stride = max(1, math.ceil(x.shape[-2] / 3))
    sampled = torch.arange(0, x.shape[-2], stride)[:3]
    mask = torch.ones(x.shape[-2], dtype=torch.bool)
    mask[sampled] = False
    assert torch.allclose(out[..., mask, :], x[..., mask, :])
    assert not torch.allclose(out[..., sampled, :], x[..., sampled, :])


def test_sampled_polarquant_attention_preserves_shape_and_gradients():
    torch.manual_seed(1)
    attn = MultiHeadTropicalAttention(
        d_model=8,
        num_heads=2,
        mode="softmax",
        polarquant_kv_bits=4,
        polarquant_train=True,
        polarquant_train_sample_tokens=4,
    )
    attn.train()
    x = torch.randn(2, 9, 8, requires_grad=True)

    result = attn(x).output
    loss = result.square().mean()
    loss.backward()

    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_preconditioned_polarquant_attention_preserves_shape_and_gradients():
    torch.manual_seed(2)
    attn = MultiHeadTropicalAttention(
        d_model=16,
        num_heads=2,
        mode="softmax",
        polarquant_kv_bits=4,
        polarquant_train=True,
        polarquant_train_sample_tokens=5,
        polarquant_precondition=True,
        polarquant_radius_bits=16,
        polarquant_seed=42,
    )
    attn.train()
    x = torch.randn(2, 11, 16, requires_grad=True)

    result = attn(x).output
    loss = result.square().mean()
    loss.backward()

    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
