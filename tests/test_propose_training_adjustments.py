import math

from scripts.propose_training_adjustments import compute_bpb_gate


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
