import torch

from toricgt.gflownet import EmbeddingPolicy, TrajectoryBatch, trajectory_balance_loss


def test_trajectory_balance_loss_backward():
    torch.manual_seed(0)
    policy = EmbeddingPolicy(embedding_dim=8, num_actions=5, hidden_dim=16)
    batch = TrajectoryBatch(
        states=torch.randn(3, 4, 8),
        actions=torch.randint(0, 5, (3, 3)),
        terminal_rewards=torch.rand(3) + 0.1,
        lengths=torch.tensor([4, 3, 4]),
    )
    loss = trajectory_balance_loss(policy, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert policy.log_z.grad is not None

