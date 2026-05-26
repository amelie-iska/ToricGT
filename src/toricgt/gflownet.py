"""Embedding-space GFlowNet objectives for graph-structured reasoning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TrajectoryBatch:
    """A batch of GFlowNet trajectories in embedding space."""

    states: torch.Tensor
    actions: torch.Tensor
    terminal_rewards: torch.Tensor
    lengths: torch.Tensor


class EmbeddingPolicy(nn.Module):
    """Forward/backward policy over discrete refinement actions from embeddings."""

    def __init__(self, embedding_dim: int, num_actions: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.forward_head = nn.Linear(hidden_dim, num_actions)
        self.backward_head = nn.Linear(hidden_dim, num_actions)
        self.log_z = nn.Parameter(torch.zeros(()))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(states)
        return self.forward_head(h), self.backward_head(h)


def trajectory_balance_loss(
    policy: EmbeddingPolicy,
    batch: TrajectoryBatch,
    action_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Trajectory-balance loss.

    The input states are ``[batch, time, dim]`` and actions are ``[batch, time-1]``.
    Terminal rewards are positive unnormalized rewards for completed graph
    embeddings. Padding beyond ``lengths`` is ignored.
    """

    bsz, time, _ = batch.states.shape
    f_logits, b_logits = policy(batch.states[:, :-1, :])
    if action_mask is not None:
        f_logits = f_logits.masked_fill(~action_mask[:, :-1, :], torch.finfo(f_logits.dtype).min)
        b_logits = b_logits.masked_fill(~action_mask[:, :-1, :], torch.finfo(b_logits.dtype).min)
    log_pf = F.log_softmax(f_logits, dim=-1).gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
    log_pb = F.log_softmax(b_logits, dim=-1).gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)

    steps = torch.arange(time - 1, device=batch.states.device).unsqueeze(0)
    valid = steps < (batch.lengths - 1).unsqueeze(1)
    log_pf_sum = (log_pf * valid).sum(dim=1)
    log_pb_sum = (log_pb * valid).sum(dim=1)
    log_reward = torch.log(batch.terminal_rewards.clamp_min(eps))
    residual = policy.log_z + log_pf_sum - log_reward - log_pb_sum
    return torch.mean(residual.square())


def reward_from_metrics(
    correctness: torch.Tensor,
    novelty: torch.Tensor,
    equivariance_error: torch.Tensor,
    energy: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Positive reward for reasoning trajectories and geometric graph states."""

    score = 2.0 * correctness + 0.5 * novelty - equivariance_error - 0.1 * energy
    return torch.exp(score / temperature).clamp_min(1e-8)

