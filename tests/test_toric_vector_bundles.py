import torch

from toricgt.parameter_golf_export import quantized_state_dict
from toricgt.random_order_lm import DenseRandomOrderToricLM, RandomOrderLMConfig
from toricgt.toric_vector_bundles import (
    ToricVectorBundleConfig,
    ToricVectorBundleProbe,
    cech_cocycle_residual,
    default_klyachko_certificate,
    klyachko_nesting_residual,
)
from toricgt.wandb_organization import organize_wandb_payload


def test_default_klyachko_certificate_is_nested_and_cocycle_consistent() -> None:
    cert = default_klyachko_certificate(rank=6, num_rays=5, num_cones=5, filtration_levels=3)
    assert cert.rays.shape == (5, 2)
    assert cert.filtration_masks.shape == (5, 3, 6)
    assert cert.chart_frames.shape == (5, 6, 6)
    assert torch.allclose(klyachko_nesting_residual(cert.filtration_masks), torch.zeros(()))
    assert cech_cocycle_residual(cert.chart_frames, cert.cone_adjacency).item() < 1e-10


def test_toric_vector_bundle_probe_outputs_finite_trainable_metrics() -> None:
    probe = ToricVectorBundleProbe(
        16,
        ToricVectorBundleConfig(
            rank=4,
            num_rays=4,
            num_cones=4,
            filtration_levels=3,
            max_positions=10,
        ),
    )
    hidden = torch.randn(2, 12, 16, requires_grad=True)
    positions = torch.arange(12).view(1, 12).repeat(2, 1)
    tokens = torch.randint(4, 48, (2, 12))
    out = probe(hidden, positions, tokens)
    assert out["toric_vector_bundle_loss"].isfinite()
    assert out["toric_vector_bundle_filtration_residual"].isfinite()
    assert out["toric_sheaf_cocycle_residual"].item() < 1e-10
    out["toric_vector_bundle_loss"].backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_random_order_lm_emits_vector_bundle_metrics_and_excludes_probe_from_export() -> None:
    cfg = RandomOrderLMConfig(
        vocab_size=48,
        max_seq_len=12,
        d_model=24,
        num_heads=4,
        num_layers=1,
        recurrent_passes=1,
        ffn_multiplier=2,
        dropout=0.0,
        attention="hybrid",
        ring_block_size=6,
        byte_offset=4,
        use_toric_vector_bundle=True,
        toric_vector_bundle_rank=4,
        toric_vector_bundle_num_rays=4,
        toric_vector_bundle_num_cones=4,
        toric_vector_bundle_max_positions=10,
        use_toric_geometry_tasks=False,
        use_toric_bgg=False,
        use_koszul_persistence=False,
        use_gflownet_policy=False,
    )
    model = DenseRandomOrderToricLM(cfg)
    tokens = torch.randint(4, cfg.vocab_size, (2, 12))
    out = model(tokens, sample_ids=torch.arange(2))
    assert out["loss"].isfinite()
    assert out["toric_vector_bundle_loss"].isfinite()
    packed = quantized_state_dict(model, bits=8)
    assert any(name.startswith("toric_vector_bundle_probe.") for name in packed["excluded"])
    assert not any(name.startswith("toric_vector_bundle_probe.") for name in packed["tensors"])


def test_wandb_routes_vector_bundle_and_sheaf_metrics() -> None:
    organized = organize_wandb_payload(
        {
            "trainer/step": 1,
            "train/toric_vector_bundle_loss": 0.2,
            "toric_vector_bundle/filtration_residual": 0.1,
            "toric_sheaf/cocycle_residual": 0.0,
        }
    )
    assert organized["00_primary/toric_vector_bundle_loss"] == 0.2
    assert organized["08_toric_tropical_bgg/toric_vector_bundle/filtration_residual"] == 0.1
    assert organized["08_toric_tropical_bgg/toric_sheaf/cocycle_residual"] == 0.0
