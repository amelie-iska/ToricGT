from __future__ import annotations

from toricgt.wandb_organization import organize_wandb_payload, primary_metric_aliases


def test_primary_aliases_promote_oai_and_training_scorecard() -> None:
    payload = {
        "trainer/step": 250.0,
        "oai_competition/bpb": 1.187,
        "oai_competition/best_bpb": 1.187,
        "oai_competition/test_time_scaled_bpb": 1.181,
        "train/bpb": 1.24,
        "val/bpb": 1.29,
        "train/loss": 0.86,
        "train/lr": 3.0e-5,
        "data/full_curated_train_split_active": 1.0,
        "artifact/initial_bytes": 11_699_487,
    }

    organized = organize_wandb_payload(payload)

    assert list(organized)[:5] == [
        "trainer/step",
        "00_primary/oai_bpb",
        "00_primary/oai_best_bpb",
        "00_primary/oai_test_time_scaled_bpb",
        "00_primary/train_bpb",
    ]
    assert organized["00_primary/oai_bpb"] == 1.187
    assert organized["00_primary/full_dataset_active"] == 1.0
    assert organized["00_primary/artifact_mb"] == 11.699487
    assert organized["01_oai/val_bpb"] == 1.187
    assert organized["02_train/bpb"] == 1.24
    assert organized["03_validation/bpb"] == 1.29
    assert organized["10_data_curriculum/full_curated_train_split_active"] == 1.0
    assert "oai_competition/bpb" not in organized


def test_category_aliases_route_advanced_losses_once() -> None:
    payload = {
        "trainer/step": 10,
        "train/gflownet_loss": 0.2,
        "train/graphcg_loss": 0.03,
        "train/analogy_topology_loss": 0.04,
        "train/toric_bgg_loss": 0.005,
        "complexity/val/prediction_target_ncd_lzma_mean": 0.41,
        "analysis/images/trajectory_3d": object(),
        "metrics_status/hessian_enabled": 1.0,
    }

    organized = organize_wandb_payload(payload)

    assert organized["05_gflownet/train/gflownet_loss"] == 0.2
    assert organized["06_graphcg/train/graphcg_loss"] == 0.03
    assert organized["07_topology_geometry/train/analogy_topology_loss"] == 0.04
    assert organized["08_toric_tropical_bgg/train/toric_bgg_loss"] == 0.005
    assert organized["09_complexity/val/prediction_target_ncd_lzma_mean"] == 0.41
    assert "15_analysis_media/analysis/images/trajectory_3d" in organized
    assert organized["16_status/metrics_status/hessian_enabled"] == 1.0


def test_symbolic_cca_resolution_metrics_are_primary_and_grouped() -> None:
    payload = {
        "trainer/step": 16,
        "advanced/toric_cca_topology_loss": 0.3,
        "advanced/toric_cca_symbolic_resolution_loss": 0.07,
        "advanced/toric_cca_symbolic_hilbert_betti_pressure": 0.09,
        "advanced/toric_cca_symbolic_taylor_full_resolution_mass": 0.11,
        "advanced/toric_cca_koszul_buchsbaum_eisenbud_multiplier_residual": 0.02,
    }

    organized = organize_wandb_payload(payload)

    assert organized["00_primary/toric_cca_topology_loss"] == 0.3
    assert organized["00_primary/cca_symbolic_resolution_loss"] == 0.07
    assert organized["00_primary/cca_buchsbaum_eisenbud_multiplier_residual"] == 0.02
    assert organized["08_toric_tropical_bgg/advanced/toric_cca_symbolic_hilbert_betti_pressure"] == 0.09
    assert organized["08_toric_tropical_bgg/advanced/toric_cca_symbolic_taylor_full_resolution_mass"] == 0.11


def test_polarquant_metrics_promote_to_primary_and_artifact_size() -> None:
    payload = {
        "trainer/step": 12,
        "polarquant/estimated_compression_ratio": 1.75,
        "polarquant/estimated_polarquant_kv_cache_mb": 92.4,
        "polarquant/estimated_saved_mb": 69.3,
    }

    organized = organize_wandb_payload(payload)

    assert organized["00_primary/polarquant_kv_cache_compression_ratio"] == 1.75
    assert organized["00_primary/polarquant_kv_cache_mb"] == 92.4
    assert organized["00_primary/polarquant_saved_mb"] == 69.3
    assert organized["11_artifact_size/polarquant/estimated_compression_ratio"] == 1.75
    assert organized["11_artifact_size/polarquant/estimated_polarquant_kv_cache_mb"] == 92.4


def test_primary_aliases_do_not_emit_missing_metrics() -> None:
    aliases = primary_metric_aliases({"trainer/step": 1})

    assert aliases == {"trainer/step": 1}


def test_raw_aliases_can_be_prefixed_for_legacy_compatibility(monkeypatch) -> None:
    monkeypatch.setenv("TORICGT_WANDB_RAW_MODE", "legacy")

    organized = organize_wandb_payload({"trainer/step": 1, "bpb": 1.2, "train_bpb": 1.3})

    assert organized["01_oai/val_bpb"] == 1.2
    assert organized["02_train/bpb"] == 1.3
    assert organized["99_legacy/bpb"] == 1.2
    assert organized["99_legacy/train_bpb"] == 1.3


def test_artifact_roundtrip_metrics_are_primary() -> None:
    organized = organize_wandb_payload(
        {
            "trainer/step": 20_000,
            "final/int8_zlib_roundtrip_bpb": 1.2906,
            "artifact/int8_zlib_total_bytes": 15_885_926,
            "artifact/under_size_limit": 1.0,
        }
    )

    assert organized["00_primary/int8_roundtrip_bpb"] == 1.2906
    assert organized["00_primary/artifact_int8_zlib_mb"] == 15.885926
    assert organized["00_primary/artifact_within_limit"] == 1.0
    assert organized["11_artifact_size/int8_roundtrip_bpb"] == 1.2906


def test_analysis_control_exact_cca_routes_to_analysis_category() -> None:
    organized = organize_wandb_payload(
        {
            "trainer/step": 750,
            "analysis_control/exact_cca/toric_cca_exact_sr_nonface_edge_fraction": 0.42,
            "analysis_control/family_pressure/combinatorial_cca_topology": 0.91,
        }
    )

    assert (
        organized["15_analysis_media/control/exact_cca/toric_cca_exact_sr_nonface_edge_fraction"]
        == 0.42
    )
    assert organized["15_analysis_media/control/family_pressure/combinatorial_cca_topology"] == 0.91
    assert "analysis_control/exact_cca/toric_cca_exact_sr_nonface_edge_fraction" not in organized
