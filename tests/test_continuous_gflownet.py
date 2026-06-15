import torch

from toricgt.continuous_gflownet import (
    ContinuousTrajectoryBatch,
    GaussianEmbeddingFlowPolicy,
    continuous_embedding_step,
    continuous_trajectory_balance_loss,
    diagonal_gaussian_log_prob,
    projected_embedding_reward,
)


def test_diagonal_gaussian_log_prob_shape_and_finiteness():
    delta = torch.zeros(3, 4, 5)
    mean = torch.zeros_like(delta)
    log_std = torch.zeros_like(delta)
    log_prob = diagonal_gaussian_log_prob(delta, mean, log_std)
    assert log_prob.shape == (3, 4)
    assert torch.isfinite(log_prob).all()


def test_continuous_trajectory_balance_loss_backpropagates():
    torch.manual_seed(7)
    policy = GaussianEmbeddingFlowPolicy(embedding_dim=6, hidden_dim=16)
    states = torch.randn(4, 5, 6)
    deltas = states[:, 1:, :] - states[:, :-1, :]
    batch = ContinuousTrajectoryBatch(
        states=states,
        deltas=deltas,
        terminal_rewards=torch.ones(4) * 1.5,
        lengths=torch.tensor([5, 4, 3, 5]),
    )
    loss = continuous_trajectory_balance_loss(policy, batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    grad_norm = sum(float(p.grad.detach().abs().sum()) for p in policy.parameters() if p.grad is not None)
    assert grad_norm > 0.0


def test_projected_embedding_reward_and_step_projection():
    state = torch.tensor([[3.0, 4.0]])
    delta = torch.tensor([[3.0, 0.0]])
    projected = continuous_embedding_step(state, delta, max_norm=2.0)
    assert torch.linalg.vector_norm(projected, dim=-1).item() <= 2.0 + 1e-6
    reward = projected_embedding_reward(
        correctness=torch.tensor([1.0]),
        novelty=torch.tensor([0.5]),
        fan_margin=torch.tensor([0.25]),
        bend_residual=torch.tensor([0.1]),
        energy=torch.tensor([0.2]),
    )
    assert reward.shape == (1,)
    assert torch.isfinite(reward).all()
    assert float(reward.item()) > 0.0
