"""Soft mixture-of-experts layers for graph tokens and causal byte models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SoftMoEDiagnostics:
    """Detached routing statistics useful for training monitors."""

    expert_mass: torch.Tensor
    dispatch_entropy: torch.Tensor
    combine_entropy: torch.Tensor


def _masked_exp_normalize_over_tokens(
    logits: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize ``logits`` over token rows while tolerating all-pad examples."""

    mask = token_mask[:, :, None, None].to(dtype=logits.dtype)
    masked_logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
    max_logits = masked_logits.amax(dim=1, keepdim=True)
    finite = torch.isfinite(max_logits)
    shifted = torch.where(finite, masked_logits - max_logits, torch.zeros_like(masked_logits))
    weights = torch.exp(shifted) * mask
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)


class ExpertMLP(nn.Module):
    """Small expert MLP used by Soft-MoE slots."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphTokenSoftMoE(nn.Module):
    """Permutation-equivariant Soft-MoE feed-forward replacement for graph tokens.

    The layer dispatches every expert slot to a weighted average over valid graph
    tokens, applies expert MLPs to those slots, and combines the expert-slot
    outputs back into the original token rows. Routing parameters are shared
    across rows, so token relabeling only relabels outputs.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int = 4,
        slots_per_expert: int = 2,
        dropout: float = 0.0,
        residual_scale: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if slots_per_expert < 1:
            raise ValueError("slots_per_expert must be positive")
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.eps = eps
        self.slot_keys = nn.Parameter(torch.empty(num_experts, slots_per_expert, d_model))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0)))
        self.output_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.experts = nn.ModuleList([ExpertMLP(d_model, hidden_dim, dropout) for _ in range(num_experts)])
        self._diagnostics: SoftMoEDiagnostics | None = None
        self._active_experts: tuple[int, ...] | None = None
        nn.init.normal_(self.slot_keys, std=d_model**-0.5)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=-1, eps=self.eps)
        slot_norm = F.normalize(self.slot_keys, p=2, dim=-1, eps=self.eps)
        return self.logit_scale.exp() * torch.einsum("bld,esd->bles", x_norm, slot_norm)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        if token_mask is None:
            token_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        token_mask = token_mask.to(dtype=torch.bool, device=x.device)

        logits = self._logits(x)
        slot_active = None
        if self._active_experts is not None:
            slot_active = torch.zeros(
                self.num_experts,
                self.slots_per_expert,
                dtype=torch.bool,
                device=x.device,
            )
            slot_active[list(self._active_experts), :] = True
            logits = logits.masked_fill(~slot_active[None, None, :, :], torch.finfo(logits.dtype).min)
        dispatch = _masked_exp_normalize_over_tokens(logits, token_mask, self.eps)
        if slot_active is not None:
            dispatch = dispatch * slot_active[None, None, :, :].to(dtype=dispatch.dtype)
        slots = torch.einsum("bles,bld->besd", dispatch, x)

        expert_outputs = []
        for expert_idx, expert in enumerate(self.experts):
            expert_outputs.append(expert(slots[:, expert_idx, :, :]))
        out_slots = torch.stack(expert_outputs, dim=1)

        combine = torch.softmax(logits.flatten(2), dim=-1).view_as(logits)
        combine = combine * token_mask[:, :, None, None].to(dtype=combine.dtype)
        y = torch.einsum("bles,besd->bld", combine, out_slots)
        y = y * token_mask.unsqueeze(-1).to(dtype=y.dtype)

        with torch.no_grad():
            expert_mass = combine.sum(dim=(0, 1, 3)) / token_mask.sum().clamp_min(1).to(combine.dtype)
            dispatch_safe = dispatch.clamp_min(self.eps)
            combine_safe = combine.flatten(2).clamp_min(self.eps)
            dispatch_entropy = -(dispatch_safe * dispatch_safe.log()).sum(dim=1).mean(dim=(0, 2))
            valid_combine = combine_safe[token_mask]
            if valid_combine.numel() == 0:
                combine_entropy = x.new_zeros(())
            else:
                combine_entropy = -(valid_combine * valid_combine.log()).sum(dim=-1).mean()
            self._diagnostics = SoftMoEDiagnostics(
                expert_mass=expert_mass.detach().cpu(),
                dispatch_entropy=dispatch_entropy.detach().cpu(),
                combine_entropy=combine_entropy.detach().cpu(),
            )
        return self.output_scale * y

    def diagnostics(self) -> SoftMoEDiagnostics | None:
        return self._diagnostics

    def set_active_experts(self, expert_ids: Iterable[int] | None) -> None:
        """Restrict routing to a subset of experts until reset with ``None``."""

        if expert_ids is None:
            self._active_experts = None
            return
        ids = tuple(sorted(set(int(expert_id) for expert_id in expert_ids)))
        if not ids:
            raise ValueError("active expert set cannot be empty")
        if ids[0] < 0 or ids[-1] >= self.num_experts:
            raise IndexError("active expert index out of range")
        self._active_experts = ids


class PrefixCausalSoftMoE(nn.Module):
    """Prefix-causal Soft-MoE for autoregressive Parameter-Golf experiments."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int = 4,
        slots_per_expert: int = 1,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.eps = eps
        self.slot_keys = nn.Parameter(torch.empty(num_experts, slots_per_expert, d_model))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0)))
        self.experts = nn.ModuleList([ExpertMLP(d_model, hidden_dim, dropout) for _ in range(num_experts)])
        nn.init.normal_(self.slot_keys, std=d_model**-0.5)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=-1, eps=self.eps)
        slot_norm = F.normalize(self.slot_keys, p=2, dim=-1, eps=self.eps)
        return self.logit_scale.exp() * torch.einsum("btd,esd->btes", x_norm, slot_norm)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        if token_mask is None:
            token_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        mask = token_mask.to(dtype=x.dtype, device=x.device)
        bsz, time, d_model = x.shape
        logits = self._logits(x).flatten(2)
        weights = torch.exp(logits - logits.amax(dim=-1, keepdim=True)).clamp_max(1e4) * mask[:, :, None]

        prefix_num = torch.cumsum(weights[:, :, :, None] * x[:, :, None, :], dim=1)
        prefix_den = torch.cumsum(weights, dim=1)
        zero_num = torch.zeros(bsz, 1, self.num_experts * self.slots_per_expert, d_model, device=x.device, dtype=x.dtype)
        zero_den = torch.zeros(bsz, 1, self.num_experts * self.slots_per_expert, device=x.device, dtype=x.dtype)
        strict_num = torch.cat([zero_num, prefix_num[:, :-1]], dim=1)
        strict_den = torch.cat([zero_den, prefix_den[:, :-1]], dim=1)
        slots = strict_num / strict_den[:, :, :, None].clamp_min(self.eps)
        slots = slots.view(bsz, time, self.num_experts, self.slots_per_expert, d_model)

        expert_outputs = []
        for expert_idx, expert in enumerate(self.experts):
            expert_input = slots[:, :, expert_idx, :, :]
            expert_outputs.append(expert(expert_input))
        out_slots = torch.stack(expert_outputs, dim=2).view(
            bsz,
            time,
            self.num_experts * self.slots_per_expert,
            d_model,
        )
        combine = torch.softmax(logits, dim=-1)
        y = torch.einsum("bts,btsd->btd", combine, out_slots)
        return y * mask[:, :, None]
