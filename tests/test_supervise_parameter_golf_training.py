from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "supervise_parameter_golf_training.py"


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
    )
    state = {"restarts": 0, "events": []}

    module.start_training(args, state, checkpoint=None)

    assert launched["session"] == "toricgt_test_train"
    assert "--checkpoint-dir" in launched["command"]
    assert str(tmp_path / "checkpoints" / "run123") in launched["command"]
    assert (tmp_path / "logs").exists()
