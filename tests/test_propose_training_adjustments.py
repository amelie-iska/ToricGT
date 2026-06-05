import math

from scripts.propose_training_adjustments import compute_bpb_gate, compute_structural_pressures, wandb_payload


def test_sampled_oai_and_train_bpb_do_not_satisfy_gate():
    gate = compute_bpb_gate(
        stats_by_name={
            "train/bpb": {
                "last_value": 1.1936,
                "best_value": 1.1936,
                "recent_slope_per_1k": -7.0,
            }
        },
        bpb_report={
            "latest_val_bpb": 1.2131,
            "best_val_bpb": 1.2131,
            "projected_target_step_from_val": float("nan"),
            "val_bpb_recent_slope_per_100_steps": float("nan"),
        },
        oai={
            "oai_competition/bpb": 1.1972,
            "oai_competition/eval_scope": "sampled",
            "oai_competition/val_max_sequences": 1,
        },
        geometry={},
        step=3750,
        target_bpb=1.2,
        gate_step=4000,
    )

    assert gate["current_primary_bpb"] == 1.2131
    assert gate["best_gate_bpb"] == 1.2131
    assert gate["best_observed_bpb"] < 1.2
    assert math.isnan(gate["authoritative_oai_bpb"])
    assert gate["target_reached"] is False
    assert gate["on_track"] is False
    assert gate["status"] == "off_track"


def test_authoritative_oai_can_satisfy_gate():
    gate = compute_bpb_gate(
        stats_by_name={},
        bpb_report={"latest_val_bpb": 1.203, "best_val_bpb": 1.203},
        oai={"oai_competition/bpb": 1.198, "oai_competition/eval_scope": "full"},
        geometry={},
        step=4000,
        target_bpb=1.2,
        gate_step=4000,
    )

    assert gate["authoritative_oai_bpb"] == 1.198
    assert gate["best_gate_bpb"] == 1.198
    assert gate["target_reached"] is True
    assert gate["status"] == "target_reached"


def test_nested_geometry_means_drive_exact_cca_pressure_and_payload():
    structural = compute_structural_pressures(
        stats_by_name={},
        fineweb_diag={},
        geometry={
            "means": {
                "toric_cca_exact_sr_nonface_edge_fraction": 0.42,
                "toric_cca_exact_betti_mismatch": 3.0,
                "toric_cca_exact_audit_backed_score": 0.2,
                "toric_cca_exact_relation_pass_rate": 0.5,
                "toric_cca_topology_loss": 0.73,
                "toric_cca_binomial_residual": 2.8,
                "toric_cca_betti1_proxy": 21.0,
                "toric_cca_chamber_coverage": 1.0,
                "toric_mean_bend": 1.25,
                "slepian_leakage": 0.4,
            }
        },
    )

    assert structural["family_pressures"]["combinatorial_cca_topology"] > 0.7
    assert structural["raw"]["toric_cca_exact_sr_nonface_edge_fraction"] == 0.42
    assert structural["raw"]["toric_cca_exact_betti_mismatch"] == 3.0
    assert structural["raw"]["toric_cca_topology_loss"] == 0.73
    assert structural["raw"]["toric_cca_binomial_residual"] == 2.8
    assert structural["raw"]["toric_cca_betti1_proxy"] == 21.0

    payload = wandb_payload(
        {
            "bpb_gate": {
                "step": 750,
                "status": "warming_up_unestablished",
                "current_primary_bpb": 1.4,
                "best_observed_bpb": 1.4,
                "gap_to_target": 0.31,
                "recent_drop_per_100_steps": 0.0,
                "required_drop_per_100_steps": 0.0,
                "velocity_shortfall_per_100_steps": 0.0,
                "projected_target_step": 0.0,
                "target_reached": False,
                "on_track": False,
            },
            "structural_diagnostics": structural,
            "decision": {"primary_action": "continue_until_first_gate_signal", "loss_policy": {}},
        }
    )

    assert payload["analysis_control/exact_cca/toric_cca_exact_sr_nonface_edge_fraction"] == 0.42
    assert payload["analysis_control/exact_cca/toric_cca_exact_betti_mismatch"] == 3.0
