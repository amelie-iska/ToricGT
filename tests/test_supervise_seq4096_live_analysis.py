from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "supervise_seq4096_live_analysis.py"


def load_module():
    spec = importlib.util.spec_from_file_location("supervise_seq4096_live_analysis", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_latest_active_run_filters_sidecars_and_picks_newest_training_session() -> None:
    module = load_module()
    sessions = [
        "toricgt_seq4096_4k_recovery_r54_20260604T134941Z",
        "toricgt_seq4096_4k_analysis_r55_20260604T140841Z",
        "toricgt_seq4096_4k_recovery_r55_20260604T140841Z",
        "toricgt_seq4096_live_full_analysis_toricgt_seq4096_4k_recovery_r55_20260604T140841Z",
        "toricgt_seq4096_4k_gate_r55_20260604T140841Z",
        "other_session",
    ]

    assert module.latest_active_run(sessions) == "toricgt_seq4096_4k_recovery_r55_20260604T140841Z"


def test_start_step_for_run_prefers_latest_checkpoint_minus_interval(tmp_path: Path) -> None:
    module = load_module()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "run_step_003500.pt").write_bytes(b"checkpoint")
    (checkpoint_dir / "run_step_003250.pt").write_bytes(b"checkpoint")
    log_path = tmp_path / "train.log"
    log_path.write_text("checkpoint_resumed:/x step:3000 train_time:1ms\n", encoding="utf-8")

    assert module.start_step_for_run(checkpoint_dir, log_path, 250) == 3250


def test_start_step_for_run_falls_back_to_log_interval_floor(tmp_path: Path) -> None:
    module = load_module()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\n".join(
            [
                "checkpoint_resumed:/x step:3000 train_time:1ms",
                "step:3561/20000 train_loss:2.1 train_time:1ms step_avg:1ms train_bpb:1.22",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert module.start_step_for_run(checkpoint_dir, log_path, 250) == 3500


def test_start_step_for_run_uses_resume_checkpoint_path_before_resume_event(tmp_path: Path) -> None:
    module = load_module()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "resume_checkpoint:/checkpoints/old_run/old_run_step_003750.pt\n",
        encoding="utf-8",
    )

    assert module.resume_checkpoint_path_step(log_path) == 3750
    assert module.start_step_for_run(checkpoint_dir, log_path, 250) == 3750


def test_build_watcher_launch_renders_seq4096_analysis_command(tmp_path: Path) -> None:
    module = load_module()
    launch = module.build_watcher_launch(
        root=tmp_path / "ToricGT",
        python_bin=Path("/opt/python"),
        live_root=tmp_path / "live",
        run_id="toricgt_seq4096_4k_recovery_r55_20260604T140841Z",
        start_step=3250,
        interval_steps=250,
        watcher_poll_seconds=30,
        target_bpb=1.2,
        gate_step=4000,
        project="toricgt-parameter-golf",
        entity="amelie-iska-math",
        tmux_prefix="toricgt_seq4096_live_full_analysis_",
        codex_review_hook="scripts/codex_training_review_resume.sh",
        codex_review_tmux_prefix="toricgt_codex_review_seq4096_live",
    )

    assert launch.start_step == 3250
    assert "scripts/watch_seq4096_analysis.py" in launch.shell_text
    assert "--start-step 3250" in launch.shell_text
    assert "--analyze-start-step" in launch.shell_text
    assert "--gate-step 4000" in launch.shell_text
    assert "--codex-review-hook" in launch.shell_text
    assert "scripts/codex_training_review_resume.sh" in launch.shell_text
    assert "--codex-review-tmux-prefix toricgt_codex_review_seq4096_live" in launch.shell_text
    assert "amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r55_20260604T140841Z" in launch.shell_text
    assert launch.command_file.name == "live_periodic_full_analysis_command.sh"
    assert launch.output_root.name == "toricgt_seq4096_4k_recovery_r55_20260604T140841Z"
