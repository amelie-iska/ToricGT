import torch

from toricgt.koszul_persistence import KoszulPersistenceConfig, koszul_persistence_loss
from toricgt.random_order_lm import (
    DenseRandomOrderToricLM,
    RandomOrderLMConfig,
    byte_decode,
    byte_encode,
    random_order_batch,
    random_order_permutation,
)
from toricgt.topological_reasoning import (
    ReasoningTopologyConfig,
    directed_step_filtration_stats_np,
    reasoning_step_topology_loss,
)


def tiny_config(**overrides):
    values = {
        "vocab_size": 32,
        "max_seq_len": 16,
        "d_model": 32,
        "num_heads": 4,
        "num_layers": 2,
        "recurrent_passes": 1,
        "ffn_multiplier": 2,
        "dropout": 0.0,
        "attention": "hybrid",
        "ring_block_size": 8,
        "seed": 11,
        "byte_offset": 4,
        "bigram_hash_buckets": 64,
        "aux_mtp_offsets": 1,
    }
    values.update(overrides)
    return RandomOrderLMConfig(**values)


def test_random_order_is_content_independent_and_seeded():
    tokens_a = torch.arange(8, dtype=torch.long).view(1, 8) + 4
    tokens_b = torch.flip(tokens_a, dims=[1])
    sample_ids = torch.tensor([123])
    batch_a = random_order_batch(tokens_a, seed=17, sample_ids=sample_ids, bos_token_id=40)
    batch_b = random_order_batch(tokens_b, seed=17, sample_ids=sample_ids, bos_token_id=40)
    assert torch.equal(batch_a.permutation, batch_b.permutation)
    assert not torch.equal(
        random_order_permutation(8, seed=17, sample_id=123),
        random_order_permutation(8, seed=17, sample_id=124),
    )


def test_score_before_update_future_tokens_do_not_affect_current_logit():
    cfg = tiny_config()
    model = DenseRandomOrderToricLM(cfg).eval()
    permutation = torch.arange(8, dtype=torch.long).view(1, 8)
    tokens_a = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    tokens_b = torch.tensor([[4, 5, 6, 7, 20, 21, 22, 23]])
    out_a = model(tokens_a, permutation=permutation, return_order=True)
    out_b = model(tokens_b, permutation=permutation, return_order=True)
    assert torch.allclose(out_a["logits"][:, 4], out_b["logits"][:, 4], atol=1e-6)


def test_dense_random_order_lm_loss_and_generation_shapes():
    cfg = tiny_config(vocab_size=48)
    model = DenseRandomOrderToricLM(cfg)
    tokens = torch.randint(4, cfg.vocab_size, (3, 12))
    out = model(tokens, sample_ids=torch.arange(3), return_order=True)
    assert out["logits"].shape == (3, 12, cfg.vocab_size)
    assert out["loss"].isfinite()
    assert out["mtp_loss"].isfinite()
    assert out["smear_temperature"].isfinite()
    assert out["toric_memory_entropy"].isfinite()
    assert out["contrastive_loss"].isfinite()
    assert out["trajectory_flow_loss"].isfinite()
    assert out["trajectory_kinetic_energy"].isfinite()
    assert out["koszul_persistence_loss"].isfinite()
    assert out["koszul_exactness_residual"].isfinite()
    assert out["koszul_buchsbaum_eisenbud_multiplier_residual"].isfinite()
    assert out["permutation"].shape == tokens.shape
    generated = model.eval().sample_random_order(length=10, seed=99, sample_id=0, top_k=8)
    assert generated.shape == (10,)


def test_gflownet_adapter_losses_and_multi_sample_scaling():
    cfg = tiny_config(vocab_size=48, use_gflownet_policy=True, gflownet_num_actions=4, gflownet_hidden_dim=16)
    model = DenseRandomOrderToricLM(cfg)
    tokens = torch.randint(4, cfg.vocab_size, (2, 10))
    out = model(tokens, sample_ids=torch.arange(2), sample_gflownet=True)
    assert out["loss"].isfinite()
    assert out["gflownet_loss"].isfinite()
    assert out["gflownet_entropy"].isfinite()
    assert 0.0 <= float(out["gflownet_action_diversity"]) <= 1.0

    scaled = model(tokens, sample_ids=torch.arange(2), gflownet_samples=3)
    assert scaled["logits"].shape == (2, 10, cfg.vocab_size)
    assert scaled["loss"].isfinite()
    assert scaled["single_sample_loss"].isfinite()


def test_analogy_hdbscan_surrogate_metrics_are_finite():
    cfg = tiny_config(
        vocab_size=48,
        use_graphcg=True,
        graphcg_num_directions=8,
        use_analogy_lattice=True,
        analogy_lattice_max_pairs=128,
        analogy_topology_max_points_per_group=12,
        analogy_topology_max_groups=8,
        analogy_hdbscan_enabled=True,
    )
    model = DenseRandomOrderToricLM(cfg)
    base = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    tokens = torch.stack([base, base.roll(1), base.roll(2)], dim=0)
    out = model(tokens, sample_ids=torch.arange(3))
    assert out["analogy_lattice_loss"].isfinite()
    assert out["analogy_hdbscan_loss"].isfinite()
    assert out["analogy_hdbscan_stability"].isfinite()
    assert out["analogy_hdbscan_persistent_edge_density"].isfinite()
    assert out["analogy_hdbscan_outlier_score"].isfinite()
    assert out["analogy_hdbscan_core_radius"].isfinite()
    assert out["analogy_step_topology_loss"].isfinite()
    assert out["analogy_step_dirichlet_energy"].isfinite()
    assert out["analogy_step_dec_conservation_loss"].isfinite()
    assert out["analogy_step_dec_mass_residual"].isfinite()
    assert out["analogy_step_dec_vorticity_drift"].isfinite()
    assert out["analogy_step_dec_kinetic_energy"].isfinite()
    assert out["analogy_step_dec_hodge_balance"].isfinite()
    assert out["analogy_step_dec_wedge_interior_residual"].isfinite()
    assert out["analogy_step_hdbscan_stability"].isfinite()
    assert out["analogy_step_cycle_rank"].isfinite()
    assert out["analogy_step_analogical_map_loss"].isfinite()
    assert out["analogy_step_directed_map_loss"].isfinite()
    assert out["analogy_graphcg_chart_dim"].item() >= 1


def test_toric_geometry_training_signals_are_finite():
    cfg = tiny_config(
        vocab_size=48,
        use_toric_geometry_tasks=True,
        toric_geometry_num_exponents=8,
        toric_geometry_exponent_dim=4,
        toric_geometry_probe_rank=6,
        toric_geometry_max_positions=12,
    )
    model = DenseRandomOrderToricLM(cfg)
    tokens = torch.randint(4, cfg.vocab_size, (2, 12))
    out = model(tokens, sample_ids=torch.arange(2))
    assert out["loss"].isfinite()
    assert out["toric_geometry_loss"].isfinite()
    assert out["toric_fan_loss"].isfinite()
    assert out["toric_active_face_margin"].isfinite()
    assert out["toric_bend_magnitude"].isfinite()
    assert out["toric_binomial_residual"].isfinite()
    assert out["toric_affine_wall_distance"].isfinite()
    assert out["toric_leaf_residual"].isfinite()


def test_reasoning_step_topology_builds_nested_directed_complexes():
    t = torch.linspace(0.0, 1.0, steps=18)
    hidden = torch.stack(
        [
            torch.cos(2 * torch.pi * t),
            torch.sin(2 * torch.pi * t),
            t,
            t.square(),
            torch.cos(4 * torch.pi * t),
            torch.sin(4 * torch.pi * t),
            1.0 - t,
            torch.ones_like(t),
        ],
        dim=-1,
    ).unsqueeze(0)
    cfg = ReasoningTopologyConfig(max_points=12, max_windows=3, window_size=10, levels=4)
    losses = reasoning_step_topology_loss(hidden, config=cfg)
    assert losses["reasoning_step_topology_loss"].isfinite()
    assert losses["reasoning_step_windows"].item() >= 1
    assert losses["reasoning_step_directed_asymmetry"].item() > 0
    assert losses["reasoning_step_analogical_map_loss"].isfinite()
    assert losses["reasoning_step_directed_map_loss"].isfinite()
    assert losses["reasoning_step_dec_conservation_loss"].isfinite()
    assert losses["reasoning_step_dec_mass_residual"].isfinite()
    assert losses["reasoning_step_dec_vorticity_drift"].isfinite()
    assert losses["reasoning_step_dec_kinetic_energy"].isfinite()
    assert losses["reasoning_step_dec_hodge_balance"].isfinite()
    assert losses["reasoning_step_dec_wedge_interior_residual"].isfinite()

    stats = directed_step_filtration_stats_np(hidden.squeeze(0).numpy(), config=cfg)
    assert "simplex_tree_summary" in stats
    assert "analogical_map_loss" in stats
    assert "directed_map_loss" in stats
    assert "dec_conservation_loss" in stats
    assert "variety_complex_residual_mean" in stats
    assert "fitting_minor_rank_residual_mean" in stats
    assert "buchsbaum_eisenbud_multiplier_residual_mean" in stats
    assert "multigraded_betti_mass_mean" in stats
    assert torch.isfinite(torch.tensor(stats["dec_conservation_loss"])).all().item()
    assert torch.isfinite(torch.tensor(stats["dec_mass_residual"])).all().item()
    assert torch.isfinite(torch.tensor(stats["dec_kinetic_energy"])).all().item()
    assert len(stats["simplex_tree_summary"]) >= cfg.levels
    edge_density = stats["edge_density"]
    assert all(edge_density[i] <= edge_density[i + 1] + 1e-8 for i in range(len(edge_density) - 1))


def test_koszul_persistence_loss_reports_affine_audit_metrics():
    hidden = torch.randn(2, 24, 16)
    positions = torch.arange(24).expand(2, 24)
    out = koszul_persistence_loss(
        hidden,
        positions,
        config=KoszulPersistenceConfig(max_points=8, max_windows=2, window_size=12),
    )
    assert out["koszul_persistence_loss"].isfinite()
    assert out["koszul_exactness_residual"].isfinite()
    assert out["koszul_syzygy_residual"].isfinite()
    assert out["koszul_fitting_rank_residual"].isfinite()
    assert out["koszul_buchsbaum_eisenbud_rank_residual"].isfinite()
    assert out["koszul_buchsbaum_eisenbud_multiplier_residual"].isfinite()
    assert out["koszul_multigraded_betti_mass"].isfinite()
    assert out["koszul_windows"].item() >= 1


def test_gflownet_adapter_preserves_score_before_update():
    cfg = tiny_config(use_gflownet_policy=True, gflownet_num_actions=4, gflownet_hidden_dim=16)
    model = DenseRandomOrderToricLM(cfg).eval()
    permutation = torch.arange(8, dtype=torch.long).view(1, 8)
    tokens_a = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    tokens_b = torch.tensor([[4, 5, 6, 7, 20, 21, 22, 23]])
    out_a = model(tokens_a, permutation=permutation, return_order=True)
    out_b = model(tokens_b, permutation=permutation, return_order=True)
    assert torch.allclose(out_a["logits"][:, 4], out_b["logits"][:, 4], atol=1e-6)


def test_score_first_bias_adaptation_is_finite_and_prefix_safe():
    cfg = tiny_config(vocab_size=48, use_gflownet_policy=True, gflownet_num_actions=4, gflownet_hidden_dim=16)
    model = DenseRandomOrderToricLM(cfg).eval()
    permutation = torch.arange(8, dtype=torch.long).view(1, 8)
    tokens = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    out = model.score_with_bias_adaptation(tokens, permutation=permutation, lr=0.05, gflownet_samples=2)
    assert out["loss"].isfinite()
    assert out["bias_norm"].isfinite()
    assert model.causal_future_permutation_error(tokens, permutation=permutation) < 1e-5


def test_byte_roundtrip():
    text = "ToricGT \u05db\u05bc\u05b8\u05ea\u05b7\u05d1"
    tokens = byte_encode(text, byte_offset=4)
    assert byte_decode(tokens, byte_offset=4) == text
