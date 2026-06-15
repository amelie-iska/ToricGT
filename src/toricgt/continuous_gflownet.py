"""Continuous embedding-space GFlowNet helpers.

The discrete ``EmbeddingPolicy`` in :mod:`toricgt.gflownet` is still useful for
symbolic refinement actions.  This module covers the continuous case required
for graph-of-thought moves in hidden embedding space: policies produce density
log-probabilities for vector increments, and trajectory balance sums those
log-densities instead of categorical action log-probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ContinuousTrajectoryBatch:
    """A batch of continuous embedding-space GFlowNet trajectories.

    ``states`` has shape ``[batch, time, dim]``.  ``deltas`` has shape
    ``[batch, time - 1, dim]`` and represents the actual forward move
    ``states[:, t + 1] - states[:, t]`` after any environment projection.  The
    backward move is therefore ``-deltas`` for reversible local embedding edits.
    """

    states: torch.Tensor
    deltas: torch.Tensor
    terminal_rewards: torch.Tensor
    lengths: torch.Tensor


class GaussianEmbeddingFlowPolicy(nn.Module):
    """Diagonal-Gaussian forward/backward density model over embedding deltas."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        *,
        min_log_std: float = -6.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.net = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.forward_mean = nn.Linear(hidden_dim, self.embedding_dim)
        self.forward_log_std = nn.Linear(hidden_dim, self.embedding_dim)
        self.backward_mean = nn.Linear(hidden_dim, self.embedding_dim)
        self.backward_log_std = nn.Linear(hidden_dim, self.embedding_dim)
        self.log_z = nn.Parameter(torch.zeros(()))

    def _heads(self, states: torch.Tensor, *, backward: bool) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(states)
        if backward:
            mean = self.backward_mean(h)
            raw_log_std = self.backward_log_std(h)
        else:
            mean = self.forward_mean(h)
            raw_log_std = self.forward_log_std(h)
        log_std = self.min_log_std + (self.max_log_std - self.min_log_std) * torch.sigmoid(raw_log_std)
        return mean, log_std

    def forward_distribution(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._heads(states, backward=False)

    def backward_distribution(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._heads(states, backward=True)


def diagonal_gaussian_log_prob(delta: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Return summed diagonal Gaussian log-density for each state/action pair."""

    var_term = ((delta - mean) / torch.exp(log_std).clamp_min(1e-12)).square()
    log_prob = -0.5 * (var_term + 2.0 * log_std + math.log(2.0 * math.pi))
    return log_prob.sum(dim=-1)


def continuous_trajectory_balance_loss(
    policy: GaussianEmbeddingFlowPolicy,
    batch: ContinuousTrajectoryBatch,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Trajectory-balance loss for continuous embedding moves.

    Forward terms use ``p_F(delta_t | s_t)``.  Backward terms use
    ``p_B(-delta_t | s_{t+1})``.  Padding beyond ``lengths`` is ignored.
    """

    states = batch.states
    deltas = batch.deltas
    if states.ndim != 3 or deltas.ndim != 3:
        raise ValueError("states and deltas must have shape [batch, time, dim] and [batch, time - 1, dim]")
    if states.shape[1] != deltas.shape[1] + 1:
        raise ValueError("states time dimension must be one larger than deltas time dimension")
    if states.shape[2] != deltas.shape[2]:
        raise ValueError("states and deltas must share embedding dimension")

    f_mean, f_log_std = policy.forward_distribution(states[:, :-1, :])
    b_mean, b_log_std = policy.backward_distribution(states[:, 1:, :])
    log_pf = diagonal_gaussian_log_prob(deltas, f_mean, f_log_std)
    log_pb = diagonal_gaussian_log_prob(-deltas, b_mean, b_log_std)

    steps = torch.arange(deltas.shape[1], device=states.device).unsqueeze(0)
    valid = steps < (batch.lengths.to(states.device) - 1).unsqueeze(1)
    log_pf_sum = (log_pf * valid).sum(dim=1)
    log_pb_sum = (log_pb * valid).sum(dim=1)
    log_reward = torch.log(batch.terminal_rewards.to(states.device).clamp_min(eps))
    residual = policy.log_z + log_pf_sum - log_reward - log_pb_sum
    return torch.mean(residual.square())


def projected_embedding_reward(
    correctness: torch.Tensor,
    novelty: torch.Tensor,
    fan_margin: torch.Tensor,
    bend_residual: torch.Tensor,
    energy: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Positive reward for continuous toric/graph-of-thought states."""

    temp = max(float(temperature), 1e-6)
    score = 2.0 * correctness + 0.5 * novelty + 0.25 * fan_margin - 0.35 * bend_residual - 0.1 * energy
    return torch.exp(score / temp).clamp_min(1e-8)


def continuous_embedding_step(
    state: torch.Tensor,
    delta: torch.Tensor,
    *,
    max_norm: float | None = None,
) -> torch.Tensor:
    """Apply a continuous embedding move with optional norm projection."""

    next_state = state + delta
    if max_norm is None:
        return next_state
    norm = torch.linalg.vector_norm(next_state, dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(float(max_norm) / norm, max=1.0)
    return next_state * scale


__all__ = [
    "ContinuousTrajectoryBatch",
    "GaussianEmbeddingFlowPolicy",
    "continuous_embedding_step",
    "continuous_trajectory_balance_loss",
    "diagonal_gaussian_log_prob",
    "projected_embedding_reward",
]
