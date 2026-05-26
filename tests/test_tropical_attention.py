import torch

from toricgt.tropical_attention import tropical_attention, tropical_ring_attention


def test_tropical_ring_matches_full_attention():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 9, 4)
    k = torch.randn(2, 3, 9, 4)
    v = torch.randn(2, 3, 9, 5)
    full = tropical_attention(q, k, v)
    ring = tropical_ring_attention(q, k, v, block_size=4)
    assert torch.allclose(full, ring, atol=1e-6)

