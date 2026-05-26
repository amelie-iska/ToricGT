from argparse import ArgumentParser, BooleanOptionalAction

from toricgt.cli_config import apply_yaml_defaults, flatten_cli_config


def test_flatten_cli_config_ignores_metadata_and_phase_plan():
    payload = {
        "metadata": {"purpose": "document only"},
        "phase_plan": {"optimizer_steps": 10},
        "model": {"d-model": 384},
        "training": {"wandb": True},
    }
    assert flatten_cli_config(payload) == {"d_model": 384, "wandb": True}


def test_apply_yaml_defaults_keeps_cli_override_surface(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
training:
  steps: 2000
  wandb: true
data:
  data_path:
    - data/curated/train.parquet
""",
        encoding="utf-8",
    )
    parser = ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--wandb", action=BooleanOptionalAction, default=False)
    parser.add_argument("--data-path", action="append", default=[])
    apply_yaml_defaults(parser, config_path)
    args = parser.parse_args(["--steps", "10", "--no-wandb"])
    assert args.steps == 10
    assert args.wandb is False
    assert args.data_path == ["data/curated/train.parquet"]
