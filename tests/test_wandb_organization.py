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
    assert organized["01_oai/deterministic_bpb"] == 1.187
    assert organized["02_train/bpb"] == 1.24
    assert organized["03_validation/bpb"] == 1.29
    assert organized["10_data_curriculum/full_curated_train_split_active"] == 1.0
    assert organized["oai_competition/bpb"] == 1.187


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
