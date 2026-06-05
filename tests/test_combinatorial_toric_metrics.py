import torch

from toricgt.combinatorial_toric_metrics import (
    CombinatorialToricConfig,
    combinatorial_toric_cca_topology_loss,
)
from toricgt.symbolic_multigraded_resolution import (
    cyclic_stanley_reisner_generator_masks,
    cyclic_stanley_reisner_resolution_dict,
    cyclic_stanley_reisner_resolution_metrics,
    cyclic_taylor_multidegree_counts,
)


def test_symbolic_multigraded_resolution_for_cyclic_fan_is_exact_and_finite():
    metrics = cyclic_stanley_reisner_resolution_metrics(6)

    assert metrics.num_vertices == 6
    assert metrics.nonface_generator_count > 0
    assert metrics.minimal_total_betti >= metrics.betti0
    assert metrics.projective_dimension > 0
    assert metrics.regularity >= 1
    assert metrics.taylor_total_rank_log2 >= metrics.taylor_positive_rank_log2

    generators = cyclic_stanley_reisner_generator_masks(6)
    taylor_rows = cyclic_taylor_multidegree_counts(6)
    assert len(generators) == metrics.nonface_generator_count
    assert taylor_rows
    assert sum(count for _degree, _mask, count in taylor_rows) == (2 ** len(generators)) - 1

    payload = cyclic_stanley_reisner_resolution_dict(6)
    assert payload["symbolic_resolution_minimal_total_betti"] == float(metrics.minimal_total_betti)
    assert payload["symbolic_resolution_projective_dimension"] == float(metrics.projective_dimension)


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
    assert torch.isfinite(out["toric_cca_symbolic_resolution_loss"])
    assert torch.isfinite(out["toric_cca_symbolic_sr_monomial_generator_mass"])
    assert torch.isfinite(out["toric_cca_symbolic_taylor_lcm_syzygy_mass"])
    assert torch.isfinite(out["toric_cca_symbolic_taylor_full_resolution_mass"])
    assert torch.isfinite(out["toric_cca_symbolic_hilbert_betti_pressure"])
    assert torch.isfinite(out["toric_cca_symbolic_dg_augmentation_ideal_mass"])
    assert out["toric_cca_symbolic_dg_d_squared_residual"].item() == 0.0
    assert out["toric_cca_symbolic_dg_leibniz_residual"].item() == 0.0
    assert out["toric_cca_symbolic_resolution_num_vertices"].item() == 6.0
    assert out["toric_cca_symbolic_resolution_minimal_total_betti"].item() > 0.0
    assert out["toric_cca_symbolic_resolution_projective_dimension"].item() > 0.0
    assert torch.isfinite(out["toric_cca_koszul_exactness_residual"])
    assert torch.isfinite(out["toric_cca_koszul_syzygy_residual"])
    assert torch.isfinite(out["toric_cca_koszul_fitting_rank_residual"])
    assert torch.isfinite(out["toric_cca_koszul_buchsbaum_eisenbud_rank_residual"])
    assert torch.isfinite(out["toric_cca_koszul_buchsbaum_eisenbud_multiplier_residual"])
    assert torch.isfinite(out["toric_cca_koszul_multigraded_betti_mass"])
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


def test_combinatorial_toric_metrics_sanitize_nonfinite_hidden():
    hidden = torch.randn(1, 24, 16, requires_grad=True)
    with torch.no_grad():
        hidden[0, 4, 3] = float("nan")
        hidden[0, 8, 5] = float("inf")
    positions = torch.arange(hidden.shape[1]).view(1, -1)

    out = combinatorial_toric_cca_topology_loss(
        hidden,
        positions,
        config=CombinatorialToricConfig(
            max_points=8,
            max_windows=1,
            window_size=16,
            step_stride=12,
            num_chambers=6,
            max_relations=8,
        ),
    )

    for key, value in out.items():
        assert torch.isfinite(value).all(), key
