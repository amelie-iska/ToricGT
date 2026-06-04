import pytest

from toricgt.bpb_transfer_controller import bpb_transfer_control_report


def test_controller_allows_guarded_supportive_family_before_threshold():
    report = bpb_transfer_control_report(
        current_report={
            "best_val_bpb": 1.228,
            "target_bpb": 1.2,
            "bpb_velocity_shortfall_pressure": 0.72,
            "structural_family_pressures": {
                "topology_directed": 0.34,
                "toric_slepian": 0.09,
            },
        },
        evidence_report={
            "family_evidence": {
                "topology_directed": {"evidence_action": "supports_guarded_velocity"},
                "toric_slepian": {"evidence_action": "sidecar_or_damp"},
            }
        },
        artifact_report={
            "artifact/int8_zlib_total_bytes": 15_700_000,
            "artifact/under_size_limit": 1.0,
        },
        target_bpb=1.2,
        gate_step=4000,
        artifact_size_limit_bytes=16_000_000,
    )

    topology = report["family_controls"]["topology_directed"]
    toric = report["family_controls"]["toric_slepian"]

    assert report["competition_phase_policy"] == "pre_threshold_guarded_bpb_transfer"
    assert report["artifact_size_policy"] == "under_limit"
    assert topology["mode"] == "guarded_aux_loss"
    assert 0.0 < topology["recommended_loss_scale"] <= 0.30
    assert toric["mode"] == "sidecar_or_damp"
    assert toric["recommended_loss_scale"] == 0.0


def test_controller_forces_sidecars_when_artifact_is_over_limit():
    report = bpb_transfer_control_report(
        current_report={
            "best_val_bpb": 1.229,
            "target_bpb": 1.2,
            "bpb_velocity_shortfall_pressure": 1.0,
            "structural_family_pressures": {"topology_directed": 0.50},
        },
        evidence_report={
            "family_evidence": {
                "topology_directed": {"evidence_action": "supports_guarded_velocity"},
            }
        },
        artifact_report={
            "artifact/int8_zlib_total_bytes": 16_120_000,
            "artifact/under_size_limit": 0.0,
        },
        artifact_size_limit_bytes=16_000_000,
    )

    topology = report["family_controls"]["topology_directed"]

    assert report["artifact_size_policy"] == "over_limit_export_guard_required"
    assert report["competition_phase_policy"] == "export_guard_before_auxiliary_promotion"
    assert topology["mode"] == "artifact_guard_sidecar_only"
    assert topology["recommended_loss_scale"] == 0.0


def test_controller_promotes_advanced_phase_after_threshold():
    report = bpb_transfer_control_report(
        current_report={
            "best_val_bpb": 1.197,
            "target_bpb": 1.2,
            "bpb_velocity_shortfall_pressure": 0.0,
            "structural_family_pressures": {
                "bgg_koszul": 0.42,
                "tropical_complexity": 0.18,
            },
        },
        evidence_report={
            "family_evidence": {
                "bgg_koszul": {"evidence_action": "monitor_as_context"},
                "tropical_complexity": {"evidence_action": "insufficient_history"},
            }
        },
        artifact_report={
            "artifact/int8_zlib_total_bytes": 15_920_000,
            "artifact/under_size_limit": 1.0,
        },
        target_bpb=1.2,
    )

    assert report["target_checkpoint_state"] == "threshold_reached"
    assert report["competition_phase_policy"] == "preserve_threshold_checkpoint_then_advanced_phases"
    assert report["family_controls"]["bgg_koszul"]["mode"] == "post_threshold_advanced_phase"
    assert report["family_controls"]["bgg_koszul"]["recommended_loss_scale"] > 0.0
    assert report["family_controls"]["tropical_complexity"]["recommended_loss_scale"] > 0.0


def test_controller_marks_tight_artifact_margin_under_limit():
    report = bpb_transfer_control_report(
        current_report={
            "best_val_bpb": 1.23,
            "structural_family_pressures": {"topology_directed": 0.1},
        },
        evidence_report={},
        artifact_report={
            "artifact/int8_zlib_total_bytes": 15_930_000,
            "artifact/under_size_limit": 1.0,
        },
        artifact_size_limit_bytes=16_000_000,
        min_artifact_margin_bytes=100_000,
    )

    assert report["artifact_size_policy"] == "under_limit_tight_margin"
    assert report["artifact_size_margin_bytes"] == pytest.approx(70_000)
