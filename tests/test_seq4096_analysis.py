from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "watch_seq4096_analysis.py"


def load_module():
    assert SCRIPT_PATH.exists(), "Seq4096 watcher script should exist"
    spec = importlib.util.spec_from_file_location("watch_seq4096_analysis", SCRIPT_PATH)
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
            "topology/topology_loss": 0.12,
            "toric/shadow_fan_cell_entropy": 0.84,
            "tropical/bpb_recent_slope": -0.09,
        },
    )

    assert report["state"] == "near_target"
    assert (tmp_path / "analysis" / "bpb" / "bpb_descent_timeseries.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_descent_simplex.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_velocity.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_target_zone.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_drop_waterfall.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_eta_to_target.png").exists()
    assert (tmp_path / "analysis" / "bpb" / "bpb_rockfall_dashboard.png").exists()
    synopsis = (tmp_path / "analysis" / "SYNOPSIS.md").read_text(encoding="utf-8")
    assert "BPB Descent Simplex" in synopsis
    assert "entity/project/run-id" in synopsis
    assert "graph-structured reasoning" in synopsis
    assert "memory boundary tokens" in synopsis
    assert "long-context tropical ring attention" in synopsis
