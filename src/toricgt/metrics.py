"""Metrics for graph-to-graph and geometric reasoning runs."""

from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).square()
    while mask.ndim < diff.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.to(diff.dtype).sum().clamp_min(1.0)
    return (diff * mask.to(diff.dtype)).sum() / denom


def equivariance_error(y: torch.Tensor, y_permuted: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """Mean error after undoing a node/token permutation."""

    inv = torch.argsort(permutation, dim=-1)
    restored = torch.gather(y_permuted, dim=1, index=inv.unsqueeze(-1).expand_as(y_permuted))
    return torch.mean(torch.linalg.vector_norm(y - restored, dim=-1))


def covered_target_bytes(lengths: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor | None:
    """Return the byte sum only if every scored target has a valid byte length."""
    if lengths is None:
        return None
    selected = lengths.to(device=mask.device, dtype=torch.float32)[mask]
    if bool((~torch.isfinite(selected) | (selected < 0)).any()):
        return None
    return selected.sum()


def tropical_margin(scores: torch.Tensor) -> torch.Tensor:
    """Average max-minus-second-max margin of tropical attention scores."""

    top2 = torch.topk(scores, k=2, dim=-1).values
    return (top2[..., 0] - top2[..., 1]).mean()

