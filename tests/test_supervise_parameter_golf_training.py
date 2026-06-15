from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "supervise_parameter_golf_training.py"
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("supervise_parameter_golf_training", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_start_training_passes_run_specific_checkpoint_dir(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    launched: dict[str, str] = {}

    monkeypatch.setattr(module, "tmux_has", lambda session: False)
    monkeypatch.setattr(module, "tmux_kill", lambda session: None)

    def fake_tmux_start(session: str, command: str, dry_run: bool) -> None:
        launched["session"] = session
        launched["command"] = command

    monkeypatch.setattr(module, "tmux_start", fake_tmux_start)

    args = Namespace(
        max_restarts=10,
        train_session="toricgt_test_train",
        dry_run=False,
        conda_bin="conda",
        conda_env="tokengt",
        wandb_entity="entity",
        wandb_project="project",
        run_id="run123",
        run_name="run-name",
        log_root=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "checkpoints" / "run123"),
        config="config/train.parameter_golf_all_phases.yaml",
        allow_training_start=True,
        allow_training_restart=True,
    )
    state = {"restarts": 0, "events": []}

    module.start_training(args, state, checkpoint=None)

    assert launched["session"] == "toricgt_test_train"
    assert "--checkpoint-dir" in launched["command"]
    assert str(tmp_path / "checkpoints" / "run123") in launched["command"]
    assert (tmp_path / "logs").exists()


def test_start_training_blocks_without_explicit_initial_allow(tmp_path: Path, monkeypatch) -> None:
    module = load_module()

    monkeypatch.setattr(module, "tmux_has", lambda session: False)

    def fail_tmux_start(session: str, command: str, dry_run: bool) -> None:
        raise AssertionError("tmux_start must not be called without explicit start allow")

    monkeypatch.setattr(module, "tmux_start", fail_tmux_start)

    args = Namespace(
        max_restarts=10,
        train_session="toricgt_test_train",
        dry_run=False,
        conda_bin="conda",
        conda_env="tokengt",
        wandb_entity="entity",
        wandb_project="project",
        run_id="run123",
        run_name="run-name",
        log_root=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "checkpoints" / "run123"),
        config="config/train.parameter_golf_all_phases.yaml",
        allow_training_start=False,
        allow_training_restart=False,
    )
    state = {"restarts": 0, "events": []}

    module.start_training(args, state, checkpoint=None)

    assert state["restarts"] == 0
    assert state["events"][-1]["event"] == "training_start_blocked"
    assert state["events"][-1]["required_flag"] == "--allow-training-start"


def test_all_phases_launcher_requires_explicit_training_start() -> None:
    launcher = REPO_ROOT / "scripts" / "launch_parameter_golf_all_phases.sh"
    env = os.environ.copy()
    env.pop("TORICGT_ALLOW_TRAINING_START", None)
    env.pop("TORICGT_ALLOW_TRAINING_RESTART", None)
    result = subprocess.run(
        [str(launcher)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "refusing to start training" in result.stderr
    assert "--allow-training-start" in result.stderr


def test_start_training_blocks_restart_without_explicit_restart_allow(tmp_path: Path, monkeypatch) -> None:
    module = load_module()

    monkeypatch.setattr(module, "tmux_has", lambda session: False)

    def fail_tmux_start(session: str, command: str, dry_run: bool) -> None:
        raise AssertionError("tmux_start must not be called without explicit restart allow")

    monkeypatch.setattr(module, "tmux_start", fail_tmux_start)

    checkpoint = tmp_path / "checkpoints" / "run123" / "random_order_step_00000250.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    args = Namespace(
        max_restarts=10,
        train_session="toricgt_test_train",
        dry_run=False,
        conda_bin="conda",
        conda_env="tokengt",
        wandb_entity="entity",
        wandb_project="project",
        run_id="run123",
        run_name="run-name",
        log_root=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "checkpoints" / "run123"),
        config="config/train.parameter_golf_all_phases.yaml",
        allow_training_start=True,
        allow_training_restart=False,
    )
    state = {"restarts": 1, "events": []}

    module.start_training(args, state, checkpoint=checkpoint)

    assert state["restarts"] == 1
    assert state["events"][-1]["event"] == "training_restart_blocked"
    assert state["events"][-1]["required_flag"] == "--allow-training-restart"
