from pathlib import Path

from scripts.watch_seq4096_4k_recovery import (
    build_recovery_launch,
    checkpoint_for_step,
    parse_seq4096_log,
    select_recovery_validation,
    select_best_validation,
)


def test_4k_recovery_selects_best_checkpoint_and_sets_resume_env(tmp_path: Path):
    run_id = "unit_run"
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "step:500/20000 val_loss:2.4389 val_bpb:1.4445 train_time:1ms step_avg:1ms",
                "step:2500/20000 val_loss:2.1729 val_bpb:1.2869 train_time:1ms step_avg:1ms",
                "step:4000/20000 val_loss:2.2100 val_bpb:1.3000 train_time:1ms step_avg:1ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    best_checkpoint = checkpoint_dir / f"{run_id}_step_002500.pt"
    best_checkpoint.write_bytes(b"checkpoint")
    (checkpoint_dir / f"{run_id}_step_004000.pt").write_bytes(b"checkpoint")

    parsed = parse_seq4096_log(log)
    best = select_best_validation(parsed.val_rows, gate_step=4000)
    assert best is not None
    assert best.step == 2500
    assert checkpoint_for_step(checkpoint_dir, run_id, best.step) == best_checkpoint

    launch = build_recovery_launch(
        repo_root=tmp_path,
        parameter_golf_root=tmp_path / "parameter-golf",
        run_id="unit_recovery",
        checkpoint_dir=tmp_path / "recovery_checkpoints",
        log_path=tmp_path / "unit_recovery.log",
        resume_checkpoint=best_checkpoint,
        seed=7331,
        target_bpb=1.2,
    )
    rendered = " ".join(launch.training_command)
    assert "RESUME_CHECKPOINT=" in rendered
    assert str(best_checkpoint) in rendered
    assert "RUN_ID=unit_recovery" in rendered
    assert "CHECKPOINT_DIR=" in rendered
    assert "RESET_OPTIMIZER_ON_RESUME=1" in rendered
    assert "RESET_RNG_ON_RESUME=1" in rendered
    assert "RESET_LOADER_ON_RESUME=1" in rendered


def test_4k_recovery_uses_best_pre_gate_checkpoint_when_gate_step_is_best_miss(tmp_path: Path):
    run_id = "unit_run"
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "step:3000/20000 val_loss:2.1498 val_bpb:1.2732 train_time:1ms step_avg:1ms",
                "step:3500/20000 val_loss:2.1343 val_bpb:1.2641 train_time:1ms step_avg:1ms",
                "step:4000/20000 val_loss:2.1223 val_bpb:1.2569 train_time:1ms step_avg:1ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for step in (3000, 3500, 4000):
        (checkpoint_dir / f"{run_id}_step_{step:06d}.pt").write_bytes(b"checkpoint")

    parsed = parse_seq4096_log(log)
    best = select_best_validation(parsed.val_rows, gate_step=4000)
    recovery = select_recovery_validation(parsed.val_rows, gate_step=4000)

    assert best is not None
    assert best.step == 4000
    assert recovery is not None
    assert recovery.step == 3500
    recovery_checkpoint = checkpoint_for_step(checkpoint_dir, run_id, recovery.step)
    assert recovery_checkpoint is not None
    assert recovery_checkpoint.is_absolute()


def test_4k_recovery_can_hold_back_best_checkpoint_for_minimum_runway(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "step:3000/20000 val_loss:2.1498 val_bpb:1.2732 train_time:1ms step_avg:1ms",
                "step:3500/20000 val_loss:2.1343 val_bpb:1.2641 train_time:1ms step_avg:1ms",
                "step:3750/20000 val_loss:2.1179 val_bpb:1.2543 train_time:1ms step_avg:1ms",
                "step:4000/20000 val_loss:2.1150 val_bpb:1.2528 train_time:1ms step_avg:1ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_seq4096_log(log)
    recovery = select_recovery_validation(
        parsed.val_rows,
        gate_step=4000,
        min_recovery_runway_steps=500,
    )

    assert recovery is not None
    assert recovery.step == 3500

    longer_runway_recovery = select_recovery_validation(
        parsed.val_rows,
        gate_step=4000,
        min_recovery_runway_steps=1000,
    )

    assert longer_runway_recovery is not None
    assert longer_runway_recovery.step == 3000
