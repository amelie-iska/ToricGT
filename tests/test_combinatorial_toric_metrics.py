import torch

from toricgt.combinatorial_toric_metrics import (
    CombinatorialToricConfig,
    combinatorial_toric_cca_topology_loss,
)


def test_combinatorial_toric_metrics_are_finite_and_differentiable():
    torch.manual_seed(7)
    hidden = torch.randn(1, 40, 24, requires_grad=True)
    positions = torch.arange(hidden.shape[1]).view(1, -1)

    out = combinatorial_toric_cca_topology_loss(
        hidden,
        positions,
        config=CombinatorialToricConfig(
            max_points=10,
            max_windows=1,
            window_size=24,
            step_stride=16,
            num_chambers=6,
            max_relations=8,
        ),
    )

    assert out["toric_cca_windows"].item() == 1.0
    assert torch.isfinite(out["toric_cca_topology_loss"])
    assert torch.isfinite(out["toric_cca_binomial_residual"])
    assert torch.isfinite(out["toric_cca_stanley_reisner_nonface_mass"])
    assert 0.0 <= out["toric_cca_chamber_coverage"].item() <= 1.0

    out["toric_cca_topology_loss"].backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert hidden.grad.abs().sum().item() > 0.0


def test_combinatorial_toric_metrics_zero_for_too_short_windows():
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    out = combinatorial_toric_cca_topology_loss(hidden)

    assert out["toric_cca_topology_loss"].item() == 0.0
    assert out["toric_cca_windows"].item() == 0.0
