"""Softmax, tropical, and blockwise-ring attention kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .polar_cache import recursive_polar_decode, recursive_polar_encode, uniform_quantize_angles
from .soft_moe import GraphTokenSoftMoE


def masked_fill(scores: torch.Tensor, mask: torch.Tensor | None, value: float) -> torch.Tensor:
    if mask is None:
        return scores
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    return scores.masked_fill(~mask, value)


def maxplus_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute C_ij = max_k a_ik + b_kj with broadcasting support."""

    return torch.amax(a.unsqueeze(-1) + b.unsqueeze(-3), dim=-2)


def tropical_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
    return_scores: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Tropical attention using max-plus scores and max-plus aggregation.

    Shapes follow PyTorch attention convention after projection:
    q,k,v are ``[batch, heads, tokens, dim]``. The output has the same leading
    dimensions as q with value dimension ``v.shape[-1]``.
    """

    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = masked_fill(scores, mask, torch.finfo(scores.dtype).min)
    out = torch.amax(scores.unsqueeze(-1) + v.unsqueeze(-3), dim=-2)
    if mask is not None:
        valid_query = mask.any(dim=-1, keepdim=True)
        out = torch.where(valid_query, out, torch.zeros_like(out))
    if return_scores:
        return out, scores
    return out


def softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
    return_scores: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = masked_fill(scores, mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if mask is not None:
        valid_query = mask.any(dim=-1, keepdim=True)
        weights = torch.where(valid_query, weights, torch.zeros_like(weights))
    out = torch.matmul(weights, v)
    if return_scores:
        return out, scores
    return out


def tropical_ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    block_size: int = 256,
    scale: float | None = None,
    return_scores: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """Exact single-process simulation of ring/block tropical attention.

    Ring attention distributes key/value blocks across devices. This function
    implements the same semiring reduction locally: each query block visits all
    key/value blocks and combines partial max-plus contexts by a max reduction.
    It is exact for tropical attention because max-plus addition is associative.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    batch, heads, n_query, _ = q.shape
    value_dim = v.shape[-1]
    out = q.new_full((batch, heads, n_query, value_dim), torch.finfo(q.dtype).min)
    saved_scores = [] if return_scores else None

    for q_start in range(0, n_query, block_size):
        q_end = min(q_start + block_size, n_query)
        q_block = q[:, :, q_start:q_end, :]
        partial = q.new_full((batch, heads, q_end - q_start, value_dim), torch.finfo(q.dtype).min)
        score_parts = [] if return_scores else None
        for k_start in range(0, k.shape[-2], block_size):
            k_end = min(k_start + block_size, k.shape[-2])
            scores = torch.matmul(q_block, k[:, :, k_start:k_end, :].transpose(-1, -2)) * scale
            block_mask = None
            if mask is not None:
                block_mask = mask[..., q_start:q_end, k_start:k_end]
            scores = masked_fill(scores, block_mask, torch.finfo(scores.dtype).min)
            block_context = torch.amax(scores.unsqueeze(-1) + v[:, :, k_start:k_end, :].unsqueeze(-3), dim=-2)
            partial = torch.maximum(partial, block_context)
            if return_scores:
                score_parts.append(scores)
        out[:, :, q_start:q_end, :] = partial
        if mask is not None:
            valid_query = mask[..., q_start:q_end, :].any(dim=-1, keepdim=True)
            out[:, :, q_start:q_end, :] = torch.where(
                valid_query,
                out[:, :, q_start:q_end, :],
                torch.zeros_like(out[:, :, q_start:q_end, :]),
            )
        if return_scores:
            saved_scores.append(torch.cat(score_parts, dim=-1))
    if return_scores:
        return out, torch.cat(saved_scores, dim=-2)
    return out


@dataclass(frozen=True)
class AttentionResult:
    output: torch.Tensor
    scores: torch.Tensor | None = None


class MultiHeadTropicalAttention(nn.Module):
    """Multi-head attention module with softmax, tropical, or tropical-ring mode."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mode: str = "tropical_ring",
        dropout: float = 0.0,
        ring_block_size: int = 256,
        polarquant_kv_bits: int = 0,
        polarquant_train: bool = False,
        polarquant_train_sample_tokens: int = 0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.mode = mode
        self.ring_block_size = ring_block_size
        self.polarquant_kv_bits = polarquant_kv_bits
        self.polarquant_train = polarquant_train
        self.polarquant_train_sample_tokens = int(polarquant_train_sample_tokens)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_gate = nn.Parameter(torch.full((num_heads, 1, 1), -2.2))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, heads, seq_len, dim = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, heads * dim)

    @staticmethod
    def _sample_token_indices(length: int, max_tokens: int, device: torch.device) -> torch.Tensor:
        if max_tokens <= 0 or length <= max_tokens:
            return torch.arange(length, device=device)
        stride = max(1, math.ceil(float(length) / float(max_tokens)))
        return torch.arange(0, length, stride, device=device)[:max_tokens]

    def _polarquant_full(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        encoded = recursive_polar_encode(x.float())
        quantized = uniform_quantize_angles(encoded, bits=self.polarquant_kv_bits)
        return recursive_polar_decode(quantized).to(dtype=original_dtype)

    def _maybe_polarquant(self, x: torch.Tensor) -> torch.Tensor:
        if self.polarquant_kv_bits <= 0:
            return x
        if self.training and not self.polarquant_train:
            return x
        if self.head_dim < 2 or self.head_dim & (self.head_dim - 1):
            return x
        if self.training and self.polarquant_train_sample_tokens > 0 and x.shape[-2] > self.polarquant_train_sample_tokens:
            index = self._sample_token_indices(
                int(x.shape[-2]),
                int(self.polarquant_train_sample_tokens),
                x.device,
            )
            sampled = x.index_select(-2, index)
            perturbed_sample = self._polarquant_full(sampled)
            out = x.clone()
            return out.index_copy(-2, index, perturbed_sample)
        return self._polarquant_full(x)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_scores: bool = False,
    ) -> AttentionResult:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        # Projective normalization stabilizes max-plus scores without changing
        # tropical projective classes.
        q = q - q.amax(dim=-1, keepdim=True)
        k = k - k.amax(dim=-1, keepdim=True)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        k = self._maybe_polarquant(k)
        v = self._maybe_polarquant(v)
        if self.mode == "softmax":
            result = softmax_attention(q, k, v, mask, return_scores=return_scores)
        elif self.mode == "tropical":
            result = tropical_attention(q, k, v, mask, return_scores=return_scores)
        elif self.mode == "tropical_ring":
            result = tropical_ring_attention(
                q,
                k,
                v,
                mask=mask,
                block_size=self.ring_block_size,
                return_scores=return_scores,
            )
        else:
            raise ValueError(f"unknown attention mode: {self.mode}")
        if return_scores:
            y, scores = result
        else:
            y, scores = result, None

        y = y * torch.sigmoid(self.norm_gate).view(1, self.num_heads, 1, 1)
        y = self._merge_heads(y)
        y = self.out(self.dropout(y))
        return AttentionResult(output=y, scores=scores if return_scores else None)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block using the selected attention kernel."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_multiplier: int,
        attention: str,
        dropout: float,
        ring_block_size: int,
        use_soft_moe: bool = False,
        soft_moe_num_experts: int = 4,
        soft_moe_slots_per_expert: int = 2,
        soft_moe_residual_scale: float = 0.1,
        polarquant_kv_bits: int = 0,
        polarquant_train: bool = False,
        polarquant_train_sample_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadTropicalAttention(
            d_model=d_model,
            num_heads=num_heads,
            mode=attention,
            dropout=dropout,
            ring_block_size=ring_block_size,
            polarquant_kv_bits=polarquant_kv_bits,
            polarquant_train=polarquant_train,
            polarquant_train_sample_tokens=polarquant_train_sample_tokens,
        )
        self.ln2 = nn.LayerNorm(d_model)
        hidden = ffn_multiplier * d_model
        self.uses_soft_moe = use_soft_moe
        if use_soft_moe:
            self.ffn = GraphTokenSoftMoE(
                d_model=d_model,
                hidden_dim=hidden,
                num_experts=soft_moe_num_experts,
                slots_per_expert=soft_moe_slots_per_expert,
                dropout=dropout,
                residual_scale=soft_moe_residual_scale,
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask=mask).output
        if self.uses_soft_moe:
            x = x + self.ffn(self.ln2(x), token_mask=token_mask)
        else:
            x = x + self.ffn(self.ln2(x))
        return x

    def set_active_experts(self, expert_ids: list[int] | tuple[int, ...] | None) -> None:
        if self.uses_soft_moe:
            self.ffn.set_active_experts(expert_ids)
