from pathlib import Path

from scripts.watch_seq4096_4k_recovery import (
    RecoveryControls,
    build_recovery_launch,
    build_recovery_run_id,
    checkpoint_for_step,
    load_advanced_diagnostics,
    load_preemptive_gate_risk,
    plan_metric_driven_recovery_controls,
    parse_seq4096_log,
    should_preempt_for_gate_risk,
    select_recovery_validation,
    summarize_advanced_diagnostics,
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


def test_recovery_run_id_stays_compact_for_wandb_resume():
    parent = (
        "toricgt_seq4096_fresh_metrics_seed1337_20260603T1818Z"
        "_4k_recovery_r7_20260603T212821Z"
        "_4k_recovery_r8_20260603T230843Z"
    )

    run_id = build_recovery_run_id(parent, restart_index=9, stamp="20260603T233419Z")

    assert run_id == "toricgt_seq4096_4k_recovery_r9_20260603T233419Z"
    assert len(run_id) < 96
    assert "_4k_recovery_r7_" not in run_id
    assert "_4k_recovery_r8_" not in run_id


def test_preemptive_gate_risk_requires_repeated_projected_miss(tmp_path: Path):
    analysis_root = tmp_path / "analysis"
    for step, projected in ((2500, 4136.6), (2750, 4188.2)):
        step_dir = analysis_root / f"step-{step:08d}"
        step_dir.mkdir(parents=True)
        (step_dir / "analysis_status.json").write_text(
            (
                "{"
                f"\"checkpoint_step\": {step},"
                "\"state\": \"near_target\","
                "\"best_val_bpb\": 1.25,"
                "\"target_bpb\": 1.2,"
                "\"val_bpb_recent_slope_per_100_steps\": -0.003,"
                f"\"projected_target_step_from_val\": {projected}"
                "}"
            ),
            encoding="utf-8",
        )

    risk = load_preemptive_gate_risk(
        analysis_root,
        gate_step=4000,
        target_bpb=1.2,
        min_step=2500,
        patience=2,
    )

    assert risk is not None
    assert risk.latest_analysis_step == 2750
    assert risk.missed_projection_count == 2
    assert should_preempt_for_gate_risk(risk)


def test_preemptive_gate_risk_ignores_single_noisy_projected_miss(tmp_path: Path):
    analysis_root = tmp_path / "analysis"
    step_dir = analysis_root / "step-00002500"
    step_dir.mkdir(parents=True)
    (step_dir / "analysis_status.json").write_text(
        (
            "{"
            "\"checkpoint_step\": 2500,"
            "\"state\": \"near_target\","
            "\"best_val_bpb\": 1.2563,"
            "\"target_bpb\": 1.2,"
            "\"val_bpb_recent_slope_per_100_steps\": -0.0034,"
            "\"projected_target_step_from_val\": 4136.6"
            "}"
        ),
        encoding="utf-8",
    )

    risk = load_preemptive_gate_risk(
        analysis_root,
        gate_step=4000,
        target_bpb=1.2,
        min_step=2500,
        patience=2,
    )

    assert risk is not None
    assert risk.missed_projection_count == 1
    assert not should_preempt_for_gate_risk(risk)


def test_seq4096_log_parses_train_bpb_for_validation_gap_controls(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "step:3000/20000 train_loss:1.9942 train_time:1ms step_avg:1ms train_bpb:1.1432",
                "step:3000/20000 val_loss:2.1028 val_bpb:1.2454 train_time:1ms step_avg:1ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_seq4096_log(log)

    assert parsed.train_rows[-1].train_bpb == 1.1432


def test_metric_controls_recapture_validation_gap_without_pushing_lr(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "step:2750/20000 val_loss:2.1128 val_bpb:1.2513 train_time:1ms step_avg:1ms",
                "step:3000/20000 train_loss:1.9942 train_time:1ms step_avg:1ms train_bpb:1.1432",
                "step:3000/20000 val_loss:2.1028 val_bpb:1.2454 train_time:1ms step_avg:1ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_seq4096_log(log)
    base = RecoveryControls(
        train_batch_tokens=917_504,
        tied_embed_lr=0.0352,
        matrix_lr=0.020,
        scalar_lr=0.020,
        muon_momentum=0.985,
        muon_momentum_warmup_steps=400,
        muon_momentum_warmup_start=0.90,
        grad_clip_norm=1.0,
    )

    planned = plan_metric_driven_recovery_controls(
        base,
        parsed=parsed,
        target_bpb=1.2,
        gate_step=4000,
        projected_target_step=4923.7,
        max_train_batch_tokens=983_040,
    )

    assert planned.policy == "validation_gap_recapture"
    assert planned.train_batch_tokens == 983_040
    assert planned.tied_embed_lr < base.tied_embed_lr
    assert planned.matrix_lr < base.matrix_lr
    assert planned.scalar_lr < base.scalar_lr
    assert planned.advanced_metric_policy == "graphcg_slepian_sidecar_primary_bpb_clean"


def test_metric_controls_accelerate_projected_miss_without_heavy_structural_losses(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "step:3000/20000 train_loss:2.0296 train_time:1ms step_avg:1ms train_bpb:1.2206",
                "step:3000/20000 val_loss:2.1028 val_bpb:1.2454 train_time:1ms step_avg:1ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_seq4096_log(log)
    base = RecoveryControls(
        train_batch_tokens=917_504,
        tied_embed_lr=0.0352,
        matrix_lr=0.020,
        scalar_lr=0.020,
        muon_momentum=0.985,
        muon_momentum_warmup_steps=400,
        muon_momentum_warmup_start=0.90,
        grad_clip_norm=1.0,
    )

    planned = plan_metric_driven_recovery_controls(
        base,
        parsed=parsed,
        target_bpb=1.2,
        gate_step=4000,
        projected_target_step=4923.7,
        max_train_batch_tokens=983_040,
    )

    assert planned.policy == "bpb_velocity_recapture"
    assert planned.train_batch_tokens == 983_040
    assert planned.tied_embed_lr > base.tied_embed_lr
    assert planned.matrix_lr == base.matrix_lr
    assert planned.scalar_lr == base.scalar_lr
    assert planned.advanced_metric_policy == "proposal_guided_bpb_recapture_structural_sidecars"


def test_metric_controls_turn_on_bigram_bias_when_tied_lr_is_capped(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "step:3000/20000 train_loss:2.0296 train_time:1ms step_avg:1ms train_bpb:1.2206",
                "step:3000/20000 val_loss:2.1028 val_bpb:1.2454 train_time:1ms step_avg:1ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_seq4096_log(log)
    base = RecoveryControls(
        train_batch_tokens=983_040,
        tied_embed_lr=0.040,
        matrix_lr=0.020,
        scalar_lr=0.020,
        muon_momentum=0.985,
        muon_momentum_warmup_steps=400,
        muon_momentum_warmup_start=0.90,
        grad_clip_norm=1.0,
    )

    planned = plan_metric_driven_recovery_controls(
        base,
        parsed=parsed,
        target_bpb=1.2,
        gate_step=4000,
        projected_target_step=6500.0,
        max_train_batch_tokens=983_040,
    )

    assert planned.policy == "lexical_transition_bias_recapture"
    assert planned.bigram_bias is True
    assert planned.bigram_bias_init_from_data is False
    assert planned.bigram_bias_lr == 0.05
    assert planned.tied_embed_lr == base.tied_embed_lr
    assert planned.advanced_metric_policy == "bigram_bias_primary_bpb_clean_structural_sidecars"


def test_advanced_diagnostics_loader_summarizes_structural_pressure(tmp_path: Path):
    diagnostics = tmp_path / "logs" / "unit_run.full_diag.latest.json"
    diagnostics.parent.mkdir()
    diagnostics.write_text(
        "{"
        '"diagnostics/families/topology_available": 1,'
        '"diagnostics/families/toric_available": 1,'
        '"diagnostics/families/slepian_pollak_prolate_available": 1,'
        '"diagnostics/latest/bpb_intervention_pressure": 0.12,'
        '"diagnostics/latest/topology_loss": 1.10,'
        '"diagnostics/latest/slepian_leakage": 1.0,'
        '"diagnostics/latest/toric_active_face_margin": -1.7,'
        '"tropical/bpb_plateau_pressure": 0.02'
        "}",
        encoding="utf-8",
    )

    loaded = load_advanced_diagnostics(tmp_path, "unit_run")
    summary = summarize_advanced_diagnostics(loaded)

    assert summary["available"] is True
    assert summary["structural_pressure_high"] is True
    assert summary["structural_recapture_score"] >= 0.50
    assert summary["structural_recapture_band"] in {"guarded", "high"}
    assert summary["bpb_intervention_pressure"] == 0.12
    assert summary["topology_loss"] == 1.10
    assert summary["slepian_leakage"] == 1.0


def test_metric_controls_use_structural_pressure_after_bigram_is_already_enabled(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "step:3000/20000 train_loss:2.0296 train_time:1ms step_avg:1ms train_bpb:1.2206",
                "step:3000/20000 val_loss:2.1028 val_bpb:1.2454 train_time:1ms step_avg:1ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_seq4096_log(log)
    base = RecoveryControls(
        train_batch_tokens=983_040,
        tied_embed_lr=0.040,
        matrix_lr=0.020,
        scalar_lr=0.020,
        muon_momentum=0.985,
        muon_momentum_warmup_steps=400,
        muon_momentum_warmup_start=0.90,
        grad_clip_norm=1.0,
        bigram_bias=True,
        bigram_bias_lr=0.05,
    )

    planned = plan_metric_driven_recovery_controls(
        base,
        parsed=parsed,
        target_bpb=1.2,
        gate_step=4000,
        projected_target_step=6500.0,
        max_train_batch_tokens=983_040,
        advanced_diagnostics={
            "diagnostics/families/topology_available": 1,
            "diagnostics/families/toric_available": 1,
            "diagnostics/families/slepian_pollak_prolate_available": 1,
            "diagnostics/latest/bpb_intervention_pressure": 0.12,
            "diagnostics/latest/topology_loss": 1.10,
            "diagnostics/latest/slepian_leakage": 1.0,
            "diagnostics/latest/toric_active_face_margin": -1.7,
        },
    )

    assert planned.policy == "structural_pressure_recapture"
    assert planned.bigram_bias is True
    assert planned.bigram_bias_lr < base.bigram_bias_lr
    assert planned.tied_embed_lr < base.tied_embed_lr
    assert planned.matrix_lr <= base.matrix_lr
    assert planned.scalar_lr <= base.scalar_lr
    assert planned.muon_momentum_warmup_steps > base.muon_momentum_warmup_steps
    assert planned.advanced_metric_policy == "toric_topology_slepian_guarded_bpb_recapture"
    assert any("structural recapture score" in item for item in planned.rationale)


def test_structural_pressure_recapture_keeps_muon_warmup_active_after_resume_step(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "step:3000/20000 train_loss:2.0296 train_time:1ms step_avg:1ms train_bpb:1.2206",
                "step:3000/20000 val_loss:2.1028 val_bpb:1.2454 train_time:1ms step_avg:1ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_seq4096_log(log)
    base = RecoveryControls(
        train_batch_tokens=983_040,
        tied_embed_lr=0.034,
        matrix_lr=0.018,
        scalar_lr=0.018,
        muon_momentum=0.985,
        muon_momentum_warmup_steps=650,
        muon_momentum_warmup_start=0.90,
        grad_clip_norm=1.0,
        bigram_bias=True,
        bigram_bias_lr=0.02,
    )

    planned = plan_metric_driven_recovery_controls(
        base,
        parsed=parsed,
        target_bpb=1.2,
        gate_step=4000,
        projected_target_step=4830.6,
        max_train_batch_tokens=983_040,
        advanced_diagnostics={
            "diagnostics/families/topology_available": 1,
            "diagnostics/families/toric_available": 1,
            "diagnostics/families/slepian_pollak_prolate_available": 1,
            "diagnostics/latest/bpb_intervention_pressure": 0.12,
            "diagnostics/latest/topology_loss": 1.10,
            "diagnostics/latest/slepian_leakage": 1.0,
            "diagnostics/latest/toric_active_face_margin": -1.7,
        },
    )

    assert planned.policy == "structural_pressure_recapture"
    assert planned.muon_momentum_warmup_steps > parsed.val_rows[-1].step
    assert any("resume-step-aware Muon warmup" in item for item in planned.rationale)


def test_metric_controls_use_velocity_when_structural_pressure_eases_and_validation_transfers(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "step:3250/20000 train_loss:2.1083 train_time:1ms step_avg:1ms train_bpb:1.2377",
                "step:3250/20000 val_loss:2.0836 val_bpb:1.2340 train_time:1ms step_avg:1ms",
                "step:3500/20000 train_loss:2.0939 train_time:1ms step_avg:1ms train_bpb:1.2550",
                "step:3500/20000 val_loss:2.0751 val_bpb:1.2290 train_time:1ms step_avg:1ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_seq4096_log(log)
    base = RecoveryControls(
        train_batch_tokens=983_040,
        tied_embed_lr=0.034,
        matrix_lr=0.018,
        scalar_lr=0.018,
        muon_momentum=0.985,
        muon_momentum_warmup_steps=4250,
        muon_momentum_warmup_start=0.90,
        grad_clip_norm=1.0,
        bigram_bias=True,
        bigram_bias_lr=0.02,
    )

    planned = plan_metric_driven_recovery_controls(
        base,
        parsed=parsed,
        target_bpb=1.2,
        gate_step=4000,
        projected_target_step=4384.1,
        max_train_batch_tokens=983_040,
        advanced_diagnostics={
            "diagnostics/families/topology_available": 1,
            "diagnostics/families/toric_available": 1,
            "diagnostics/families/slepian_pollak_prolate_available": 1,
            "diagnostics/families/category_o_bgg_available": 1,
            "diagnostics/latest/bpb_intervention_pressure": 0.104,
            "diagnostics/latest/topology_loss": 1.156,
            "diagnostics/latest/directed_topology_loss": 0.1205,
            "diagnostics/latest/slepian_leakage": 0.862,
            "diagnostics/latest/toric_active_face_margin": -2.03,
            "diagnostics/latest/toric_shadow_mean_bend": 0.861,
            "diagnostics/latest/bgg_standard_leakage": 0.544,
            "diagnostics/latest/bgg_d2_residual": 0.020,
            "diagnostics/latest/complexity_recent_full_log_ncd_lzma": 0.874,
        },
    )

    assert planned.policy == "structural_relief_velocity_recapture"
    assert planned.advanced_metric_policy == "guarded_transfer_relief_bpb_velocity"
    assert planned.tied_embed_lr > base.tied_embed_lr
    assert planned.bigram_bias_lr > base.bigram_bias_lr
    assert planned.matrix_lr == base.matrix_lr
    assert planned.scalar_lr == base.scalar_lr
    assert planned.muon_momentum_warmup_steps == base.muon_momentum_warmup_steps
    assert any("no longer lagging train" in item for item in planned.rationale)


def test_recovery_launch_exports_bigram_bias_env_when_enabled(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    launch = build_recovery_launch(
        repo_root=tmp_path,
        parameter_golf_root=tmp_path / "parameter-golf",
        run_id="unit_bigram_recovery",
        checkpoint_dir=tmp_path / "recovery_checkpoints",
        log_path=tmp_path / "unit_recovery.log",
        resume_checkpoint=checkpoint,
        seed=7331,
        target_bpb=1.2,
        bigram_bias=True,
        bigram_bias_lr=0.05,
        bigram_bias_init_from_data=True,
        bigram_bias_init_tokens=12_345,
        bigram_bias_init_alpha=0.2,
        bigram_bias_init_strength=0.4,
    )
    rendered = " ".join(launch.training_command)

    assert "BIGRAM_BIAS=1" in rendered
    assert "BIGRAM_BIAS_LR=0.05" in rendered
    assert "BIGRAM_BIAS_INIT_FROM_DATA=1" in rendered
    assert "BIGRAM_BIAS_INIT_TOKENS=12345" in rendered
    assert "BIGRAM_BIAS_INIT_ALPHA=0.2" in rendered
    assert "BIGRAM_BIAS_INIT_STRENGTH=0.4" in rendered


def test_metric_controls_hold_when_projection_is_on_track(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "step:3000/20000 train_loss:2.2500 train_time:1ms step_avg:1ms train_bpb:1.3100",
                "step:3000/20000 val_loss:2.3000 val_bpb:1.3300 train_time:1ms step_avg:1ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_seq4096_log(log)
    base = RecoveryControls(
        train_batch_tokens=917_504,
        tied_embed_lr=0.0352,
        matrix_lr=0.020,
        scalar_lr=0.020,
        muon_momentum=0.985,
        muon_momentum_warmup_steps=400,
        muon_momentum_warmup_start=0.90,
        grad_clip_norm=1.0,
    )

    planned = plan_metric_driven_recovery_controls(
        base,
        parsed=parsed,
        target_bpb=1.2,
        gate_step=4000,
        projected_target_step=3900.0,
        max_train_batch_tokens=983_040,
    )

    assert planned.policy == "base_controls"
    assert planned.train_batch_tokens == base.train_batch_tokens
    assert planned.tied_embed_lr == base.tied_embed_lr
