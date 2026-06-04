from __future__ import annotations

import importlib.util
from pathlib import Path
import json
import sys

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "watch_seq4096_analysis.py"
FULL_DIAG_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mirror_fineweb_full_diagnostics_to_wandb.py"
LOG_MIRROR_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mirror_fineweb_log_to_wandb.py"


def load_module():
    assert SCRIPT_PATH.exists(), "Seq4096 watcher script should exist"
    spec = importlib.util.spec_from_file_location("watch_seq4096_analysis", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_full_diag_module():
    assert FULL_DIAG_SCRIPT_PATH.exists(), "FineWeb full diagnostics script should exist"
    src_path = str(FULL_DIAG_SCRIPT_PATH.resolve().parents[1] / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    spec = importlib.util.spec_from_file_location("mirror_fineweb_full_diagnostics_to_wandb", FULL_DIAG_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_log_mirror_module():
    assert LOG_MIRROR_SCRIPT_PATH.exists(), "FineWeb W&B mirror script should exist"
    spec = importlib.util.spec_from_file_location("mirror_fineweb_log_to_wandb", LOG_MIRROR_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_sample_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "step:0/20000 val_loss:6.9357 val_bpb:4.1077 train_time:0ms step_avg:0.01ms",
                "step:1/20000 train_loss:6.9366 train_time:1060ms step_avg:1060.00ms train_bpb:4.0876",
                "step:100/20000 train_loss:3.4356 train_time:137848ms step_avg:1378.48ms train_bpb:2.0053",
                "step:200/20000 train_loss:2.7897 train_time:277396ms step_avg:1386.98ms train_bpb:1.6322",
                "step:500/20000 val_loss:2.3033 val_bpb:1.3641 train_time:695011ms step_avg:1390.02ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_transfer_lag_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "step:3000/20000 train_loss:1.9942 train_time:7600000ms step_avg:2533.33ms train_bpb:1.1432",
                "step:3000/20000 val_loss:2.1028 val_bpb:1.2454 train_time:7600100ms step_avg:2533.36ms",
                "step:3250/20000 train_loss:1.9520 train_time:8250000ms step_avg:2538.46ms train_bpb:1.1200",
                "step:3250/20000 val_loss:2.0922 val_bpb:1.2391 train_time:8250100ms step_avg:2538.49ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_parse_seq4096_log_captures_train_val_and_train_bpb(tmp_path: Path) -> None:
    module = load_module()
    log_path = tmp_path / "train.log"
    write_sample_log(log_path)

    parsed = module.parse_seq4096_log(log_path)
    frame = module.training_dataframe(parsed)

    assert parsed.latest_step == 500
    assert parsed.total_steps == 20000
    assert parsed.train_rows[-1].train_bpb == pytest.approx(1.6322)
    assert parsed.val_rows[-1].val_bpb == pytest.approx(1.3641)
    assert {"step", "train_bpb", "val_bpb", "target_gap"}.issubset(frame.columns)
    assert frame.loc[frame["step"] == 500, "val_bpb"].iloc[0] == pytest.approx(1.3641)


def test_checkpoint_step_detection_uses_seq4096_filename_pattern(tmp_path: Path) -> None:
    module = load_module()
    checkpoint = tmp_path / "toricgt_seq4096_run_step_000500.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "toricgt_seq4096_run_step_000250.pt").write_bytes(b"old")
    (tmp_path / "random_order_step_000500.pt").write_bytes(b"native")

    assert module.extract_checkpoint_step(checkpoint) == 500
    assert module.find_checkpoint_at_step(tmp_path, 500) == checkpoint
    assert module.find_latest_checkpoint(tmp_path).name == checkpoint.name


def test_bpb_acceleration_report_marks_near_target_descent(tmp_path: Path) -> None:
    module = load_module()
    log_path = tmp_path / "train.log"
    write_sample_log(log_path)
    parsed = module.parse_seq4096_log(log_path)
    frame = module.training_dataframe(parsed, target_bpb=1.2)

    report = module.bpb_acceleration_report(frame, target_bpb=1.2, checkpoint_step=500)

    assert report["best_val_bpb"] == pytest.approx(1.3641)
    assert report["target_gap"] == pytest.approx(0.1641)
    assert report["latest_train_bpb"] == pytest.approx(1.6322)
    assert report["train_bpb_recent_slope_per_100_steps"] < 0
    assert report["state"] == "near_target"
    assert any("validation" in item.lower() or "checkpoint" in item.lower() for item in report["recommendations"])


def test_bpb_acceleration_report_marks_near_target_unreachable_by_gate(tmp_path: Path) -> None:
    module = load_module()
    log_path = tmp_path / "flat_validation.log"
    log_path.write_text(
        "\n".join(
            [
                "step:3250/20000 train_loss:2.1083 train_time:1ms step_avg:1ms train_bpb:1.2377",
                "step:3250/20000 val_loss:2.0833 val_bpb:1.2338 train_time:1ms step_avg:1ms",
                "step:3500/20000 train_loss:2.1093 train_time:1ms step_avg:1ms train_bpb:1.2383",
                "step:3500/20000 val_loss:2.0834 val_bpb:1.2339 train_time:1ms step_avg:1ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = module.parse_seq4096_log(log_path)
    frame = module.training_dataframe(parsed, target_bpb=1.2)

    report = module.bpb_acceleration_report(
        frame,
        target_bpb=1.2,
        checkpoint_step=3500,
        gate_step=4000,
    )

    assert report["state"] == "off_track_unreachable"
    assert report["required_val_velocity_to_gate_per_100_steps"] > 0.006
    assert report["val_bpb_velocity_recent_per_100_steps"] == pytest.approx(0.0)
    assert report["bpb_velocity_shortfall_pressure"] == pytest.approx(1.0)
    assert any("advanced metric" in item.lower() for item in report["recommendations"])


def test_transfer_efficiency_report_identifies_train_descent_validation_lag(tmp_path: Path) -> None:
    module = load_module()
    log_path = tmp_path / "transfer_lag.log"
    write_transfer_lag_log(log_path)
    parsed = module.parse_seq4096_log(log_path)
    frame = module.training_dataframe(parsed, target_bpb=1.2)

    report = module.transfer_efficiency_report(frame, target_bpb=1.2)

    assert report["transfer_pairs"] == 2
    assert report["transfer_regime"] == "train_descent_validation_lag"
    assert report["recent_transfer_efficiency"] == pytest.approx((1.2454 - 1.2391) / (1.1432 - 1.1200))
    assert report["latest_generalization_gap_bpb"] == pytest.approx(1.2391 - 1.1200)
    assert report["generalization_gap_slope_per_100_steps"] > 0.0
    assert any("validation bpb" in item.lower() for item in report["transfer_recommendations"])


def test_write_bpb_artifacts_creates_actionable_plots_and_synopsis(tmp_path: Path) -> None:
    module = load_module()
    log_path = tmp_path / "train.log"
    write_sample_log(log_path)
    parsed = module.parse_seq4096_log(log_path)
    checkpoint = tmp_path / "toricgt_seq4096_run_step_000500.pt"
    checkpoint.write_bytes(b"checkpoint")

    report = module.write_bpb_analysis_artifacts(
        parsed,
        tmp_path / "analysis",
        target_bpb=1.2,
        checkpoint_path=checkpoint,
        run_path="entity/project/run-id",
        diagnostic_payload={
            "diagnostics/families/topology_available": 1.0,
            "diagnostics/families/toric_available": 1.0,
            "diagnostics/families/slepian_pollak_prolate_available": 1.0,
            "topology/topology_loss": 1.20,
            "topology/directed_topology_loss": 0.18,
            "toric/shadow_fan_cell_entropy": 0.84,
            "toric/shadow_mean_bend": 1.80,
            "toric/active_face_margin": -1.20,
            "toric/slepian_leakage": 1.0,
            "tropical/bpb_recent_slope": -0.09,
            "bgg_category_o/standard_leakage": 0.40,
            "complexity/recent_full_log_ncd_lzma": 0.86,
        },
        artifact_report={
            "artifact_probe_status": "ok",
            "artifact/int8_zlib_total_bytes": 15_900_000,
            "artifact/under_size_limit": 1,
            "artifact/size_limit_bytes": 16_000_000,
        },
    )

    assert report["state"] == "near_target"
    assert report["structural_recapture_score"] > 0.55
    assert report["structural_recapture_band"] in {"guarded", "high"}
    assert report["dominant_structural_pressure_family"] in {
        "topology_directed",
        "toric_slepian",
        "bgg_koszul",
        "tropical_complexity",
        "bpb_gap",
    }
    assert report["structural_family_pressures"]["toric_slepian"] > 0.0
    assert (tmp_path / "analysis" / "bpb" / "bpb_descent_timeseries.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_descent_simplex.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_velocity.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_target_zone.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_drop_waterfall.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_eta_to_target.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_rockfall_dashboard.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_structural_recapture_map.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "advanced_metric_control_map.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "structural_recapture_report.json").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_transfer_efficiency.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_transfer_efficiency_report.json").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_gate_velocity_requirement.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "advanced_metric_evidence_map.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "advanced_metric_evidence_report.json").exists()
    assert (tmp_path / "analysis" / "bpb" / "artifact_export_probe.json").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_transfer_controller_map.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_transfer_controller_report.json").exists()
    assert report["bpb_transfer_competition_phase_policy"] in {
        "pre_threshold_primary_bpb_clean",
        "pre_threshold_guarded_bpb_transfer",
        "export_guard_before_auxiliary_promotion",
        "preserve_threshold_checkpoint_then_advanced_phases",
    }
    assert "topology_directed" in report["bpb_transfer_recommended_loss_scales"]
    assert "toric_slepian" in report["bpb_transfer_recommended_modes"]
    assert report["bpb_transfer_controller"]["artifact_size_policy"] == "under_limit_tight_margin"
    assert report["bpb_transfer_controller"]["artifact_size_margin_bytes"] == pytest.approx(100_000)
    assert "required_val_velocity_to_gate_per_100_steps" in report
    assert "bpb_velocity_shortfall_pressure" in report
    phase_plan_path = tmp_path / "analysis" / "bpb" / "phase_bpb_breakdown_plan.json"
    assert phase_plan_path.exists()
    phase_plan = json.loads(phase_plan_path.read_text(encoding="utf-8"))
    assert "competition_fineweb" in phase_plan["phases"]
    assert "reasoning_got_tot_cot" in phase_plan["phases"]
    assert "memory_retrieval" in phase_plan["phases"]
    assert "analogical_transfer" in phase_plan["phases"]
    synopsis = (tmp_path / "analysis" / "SYNOPSIS.md").read_text(encoding="utf-8")
    assert "BPB Descent Simplex" in synopsis
    assert "entity/project/run-id" in synopsis
    assert "graph-structured reasoning" in synopsis
    assert "memory boundary tokens" in synopsis
    assert "long-context tropical ring attention" in synopsis
    assert "embedding-space GFlowNet" in synopsis
    assert "GoT/ToT/CoT" in synopsis
    assert "memory-retrieval BPB" in synopsis
    assert "analogical-transfer BPB" in synopsis
    assert "structural recapture score" in synopsis
    assert "BPB Structural Recapture Map" in synopsis
    assert "Train-To-Validation BPB Transfer Efficiency" in synopsis
    assert "Advanced Metric Control Map" in synopsis
    assert "Gate Velocity Requirement" in synopsis
    assert "Advanced Metric Evidence Map" in synopsis
    assert "BPB Transfer Controller Map" in synopsis


def test_advanced_metric_evidence_report_scores_next_validation_transfer(tmp_path: Path) -> None:
    module = load_module()
    history_root = tmp_path / "post_resume_analysis"

    def write_status(run: str, step: int, val_bpb: float, family: dict[str, float]) -> None:
        step_dir = history_root / run / f"step-{step:08d}"
        step_dir.mkdir(parents=True)
        (step_dir / "analysis_status.json").write_text(
            json.dumps(
                {
                    "checkpoint_step": step,
                    "latest_val_bpb": val_bpb,
                    "best_val_bpb": val_bpb,
                    "gate_step": 4000,
                    "bpb_velocity_shortfall_pressure": 0.5,
                    "structural_family_pressures": family,
                    "dominant_structural_pressure_family": max(family, key=family.get),
                }
            )
            + "\n",
            encoding="utf-8",
        )

    write_status("run_good_a", 3250, 1.2340, {"topology_directed": 0.31, "toric_slepian": 0.08})
    write_status("run_good_a", 3500, 1.2280, {"topology_directed": 0.20, "toric_slepian": 0.10})
    write_status("run_good_b", 3250, 1.2350, {"topology_directed": 0.28, "toric_slepian": 0.05})
    write_status("run_good_b", 3500, 1.2300, {"topology_directed": 0.18, "toric_slepian": 0.10})
    write_status("run_bad_a", 3250, 1.2338, {"topology_directed": 0.05, "toric_slepian": 0.36})
    write_status("run_bad_a", 3500, 1.2336, {"topology_directed": 0.04, "toric_slepian": 0.34})

    report = module.advanced_metric_evidence_report(
        history_root,
        current_report={
            "dominant_structural_pressure_family": "topology_directed",
            "structural_family_pressures": {"topology_directed": 0.30, "toric_slepian": 0.07},
            "bpb_velocity_shortfall_pressure": 0.7,
        },
    )

    topology = report["family_evidence"]["topology_directed"]
    toric = report["family_evidence"]["toric_slepian"]

    assert report["evidence_observations"] == 3
    assert topology["active_mean_next_val_drop_bpb"] > topology["inactive_mean_next_val_drop_bpb"]
    assert topology["evidence_action"] == "supports_guarded_velocity"
    assert toric["active_mean_next_val_drop_bpb"] < toric["inactive_mean_next_val_drop_bpb"]
    assert toric["evidence_action"] == "sidecar_or_damp"
    assert report["current_evidence_policy"] == "metric_supported_velocity_push"


def test_full_diagnostics_can_write_json_without_wandb(tmp_path: Path) -> None:
    module = load_full_diag_module()
    log_path = tmp_path / "train.log"
    write_sample_log(log_path)
    output_json = tmp_path / "diagnostics.json"

    payload = module.run_once_without_wandb(
        log_path=log_path,
        output_json=output_json,
        target_bpb=1.2,
        max_points=16,
    )

    assert output_json.exists()
    assert payload["trainer/step"] == 500
    assert payload["fineweb_curve/latest_train_bpb"] == pytest.approx(1.6322)
    assert payload["fineweb_curve/latest_val_bpb"] == pytest.approx(1.3641)
    assert payload["diagnostics/latest/train_bpb"] == pytest.approx(1.6322)
    assert payload["diagnostics/latest/val_bpb"] == pytest.approx(1.3641)
    assert payload["diagnostics/latest/topology_loss"] == pytest.approx(payload["topology/topology_loss"])
    assert payload["diagnostics/latest/bgg_d2_residual"] == pytest.approx(payload["bgg_category_o/d2_residual"])
    assert payload["diagnostics/latest/pollak_prolate_slepian_concentration"] == pytest.approx(
        payload["toric/slepian_concentration"]
    )
    assert payload["diagnostics/latest/structural_recapture_score"] > 0.0
    assert payload["diagnostics/structural_recapture_score"] == pytest.approx(
        payload["diagnostics/latest/structural_recapture_score"]
    )
    assert "diagnostics/structural_recapture_components/topology_loss" in payload
    assert "diagnostics/structural_recapture_components/slepian_leakage" in payload
    assert "diagnostics/structural_family_pressure/topology_directed" in payload
    assert "diagnostics/latest/structural_family_pressure_toric_slepian" in payload
    assert payload["diagnostics/latest/dominant_structural_pressure_id"] > 0.0
    assert payload["diagnostics/latest/dominant_structural_pressure_value"] > 0.0
    assert payload["diagnostics/families/structural_recapture_available"] == 1.0
    assert payload["diagnostics/families/topology_available"] == 1.0
    assert payload["diagnostics/families/category_o_bgg_available"] == 1.0
    assert payload["diagnostics/families/koszul_persistence_available"] == 1.0
    assert payload["diagnostics/families/slepian_pollak_prolate_available"] == 1.0
    assert payload["tropical/bpb_best_so_far"] == pytest.approx(1.3641)
    assert payload["metrics_status/fineweb_curve_diagnostics_available"] == 1.0


def test_full_diagnostics_reports_transfer_efficiency_aliases(tmp_path: Path) -> None:
    module = load_full_diag_module()
    log_path = tmp_path / "transfer_lag.log"
    write_transfer_lag_log(log_path)
    output_json = tmp_path / "diagnostics.json"

    payload = module.run_once_without_wandb(
        log_path=log_path,
        output_json=output_json,
        target_bpb=1.2,
        max_points=16,
    )

    expected_efficiency = (1.2454 - 1.2391) / (1.1432 - 1.1200)
    assert payload["fineweb_curve/transfer_pairs"] == pytest.approx(2.0)
    assert payload["fineweb_curve/transfer_efficiency_recent"] == pytest.approx(expected_efficiency)
    assert payload["diagnostics/latest/transfer_efficiency_recent"] == pytest.approx(expected_efficiency)
    assert payload["diagnostics/latest/generalization_gap_slope_per_100_steps"] > 0.0
    assert payload["diagnostics/latest/validation_transfer_pressure"] > 0.0


def test_full_diagnostics_reports_gate_velocity_shortfall_aliases(tmp_path: Path) -> None:
    module = load_full_diag_module()
    log_path = tmp_path / "gate_velocity.log"
    log_path.write_text(
        "\n".join(
            [
                "step:3000/20000 train_loss:2.1200 train_time:7600000ms step_avg:2533.33ms train_bpb:1.2450",
                "step:3000/20000 val_loss:2.1028 val_bpb:1.2454 train_time:7600001ms step_avg:2533.33ms",
                "step:3250/20000 train_loss:2.1083 train_time:8250000ms step_avg:2538.46ms train_bpb:1.2377",
                "step:3250/20000 val_loss:2.0836 val_bpb:1.2340 train_time:8250001ms step_avg:2538.46ms",
                "step:3500/20000 train_loss:2.0939 train_time:8900000ms step_avg:2542.85ms train_bpb:1.2550",
                "step:3500/20000 val_loss:2.0751 val_bpb:1.2290 train_time:8900001ms step_avg:2542.85ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = module.run_once_without_wandb(
        log_path=log_path,
        output_json=tmp_path / "diagnostics.json",
        target_bpb=1.2,
        max_points=16,
        gate_step=4000,
    )

    assert payload["fineweb_curve/val_bpb_velocity_recent_per_100_steps"] == pytest.approx(0.002)
    assert payload["fineweb_curve/required_val_velocity_to_gate_per_100_steps"] == pytest.approx(0.0058)
    assert payload["fineweb_curve/val_velocity_shortfall_to_gate_per_100_steps"] == pytest.approx(0.0038)
    assert payload["diagnostics/latest/val_velocity_shortfall_to_gate_per_100_steps"] == pytest.approx(0.0038)
    assert payload["diagnostics/latest/bpb_velocity_shortfall_pressure"] > 0.5


def test_full_diagnostics_signature_changes_when_validation_arrives_after_train(tmp_path: Path) -> None:
    module = load_full_diag_module()
    log_path = tmp_path / "late_val.log"
    log_path.write_text(
        "\n".join(
            [
                "step:3000/20000 val_loss:2.1028 val_bpb:1.2454 train_time:7600000ms step_avg:2533.33ms",
                "step:3250/20000 train_loss:2.1083 train_time:8627503ms step_avg:2654.62ms train_bpb:1.2377",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train_only = module.parse_log(log_path)
    train_only_signature = module.diagnostic_log_signature(train_only)
    train_only_payload = module.run_once_without_wandb(
        log_path=log_path,
        output_json=tmp_path / "train_only.json",
        target_bpb=1.2,
        max_points=16,
    )

    assert train_only_payload["trainer/step"] == 3250
    assert train_only_payload["fineweb_curve/latest_val_bpb"] == pytest.approx(1.2454)
    assert train_only_payload["fineweb_curve/transfer_pairs"] == pytest.approx(0.0)

    log_path.write_text(
        log_path.read_text(encoding="utf-8")
        + "step:3250/20000 val_loss:2.0836 val_bpb:1.2340 train_time:8627508ms step_avg:2654.62ms\n",
        encoding="utf-8",
    )
    with_validation = module.parse_log(log_path)
    validation_signature = module.diagnostic_log_signature(with_validation)
    validation_payload = module.run_once_without_wandb(
        log_path=log_path,
        output_json=tmp_path / "with_validation.json",
        target_bpb=1.2,
        max_points=16,
    )

    assert validation_signature != train_only_signature
    assert validation_payload["trainer/step"] == 3250
    assert validation_payload["fineweb_curve/latest_val_bpb"] == pytest.approx(1.2340)
    assert validation_payload["fineweb_curve/transfer_pairs"] == pytest.approx(1.0)
    assert validation_payload["diagnostics/latest/val_bpb_for_transfer"] == pytest.approx(1.2340)


def test_dense_wandb_mirror_loads_train_bpb_and_diagnostic_aliases(tmp_path: Path) -> None:
    module = load_log_mirror_module()
    train_match = module.TRAIN_RE.search(
        "step:2200/20000 train_loss:2.2024 train_time:3089227ms step_avg:1404.19ms train_bpb:1.2833"
    )
    assert train_match is not None
    assert float(train_match.group("bpb")) == pytest.approx(1.2833)

    diagnostics_json = tmp_path / "latest.json"
    diagnostics_json.write_text(
        json.dumps(
            {
                "trainer/step": 2100,
                "diagnostics/latest/topology_loss": 0.8,
                "diagnostics/latest/bpb_intervention_pressure": 0.12,
                "diagnostics/families/slepian_pollak_prolate_available": 1.0,
                "diagnostics/structural_recapture_score": 0.71,
                "diagnostics/structural_recapture_components/topology_loss": 0.19,
                "fineweb_curve/latest_train_bpb": 1.2821,
                "toric/shadow_fan_cell_entropy": 0.75,
            }
        ),
        encoding="utf-8",
    )

    aliases = module.latest_diagnostics_aliases(diagnostics_json, step=2200)

    assert aliases["diagnostics/latest/topology_loss"] == pytest.approx(0.8)
    assert aliases["diagnostics/latest/bpb_intervention_pressure"] == pytest.approx(0.12)
    assert aliases["diagnostics/families/slepian_pollak_prolate_available"] == 1.0
    assert aliases["diagnostics/structural_recapture_score"] == pytest.approx(0.71)
    assert aliases["diagnostics/structural_recapture_components/topology_loss"] == pytest.approx(0.19)
    assert aliases["fineweb_curve/latest_train_bpb"] == pytest.approx(1.2821)
    assert aliases["diagnostics/latest_full_metrics_step"] == pytest.approx(2100)
    assert aliases["diagnostics/latest/staleness_steps"] == pytest.approx(100)
    assert "toric/shadow_fan_cell_entropy" not in aliases


def test_dense_wandb_mirror_builds_diagnostics_summary_pin_payload(tmp_path: Path) -> None:
    module = load_log_mirror_module()
    diagnostics_json = tmp_path / "latest.json"
    diagnostics_json.write_text(
        json.dumps(
            {
                "trainer/step": 3200,
                "diagnostics/latest/structural_recapture_score": 0.7814,
                "diagnostics/latest/structural_pressure_high": 1.0,
                "diagnostics/latest/topology_loss": 1.2687,
                "diagnostics/families/structural_recapture_available": 1.0,
                "diagnostics/structural_recapture_score": 0.7814,
                "diagnostics/structural_recapture_components/topology_loss": 0.154,
                "fineweb_curve/latest_train_bpb": 1.1327,
            }
        ),
        encoding="utf-8",
    )

    payload = module.latest_diagnostics_summary_pin_payload(diagnostics_json, step=3229)
    fingerprint = module.summary_payload_fingerprint(payload)

    assert payload["diagnostics/latest/structural_recapture_score"] == pytest.approx(0.7814)
    assert payload["diagnostics/latest/structural_pressure_high"] == 1.0
    assert payload["diagnostics/families/structural_recapture_available"] == 1.0
    assert payload["diagnostics/structural_recapture_score"] == pytest.approx(0.7814)
    assert payload["diagnostics/latest/staleness_steps"] == pytest.approx(29)
    assert payload["diagnostics/summary_pin_active"] == 1.0
    assert fingerprint == module.summary_payload_fingerprint(dict(reversed(list(payload.items()))))
