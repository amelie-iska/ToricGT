"""YAML-backed argparse defaults for ToricGT scripts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


CONFIG_SECTIONS = {
    "data",
    "model",
    "training",
    "curriculum",
    "inference",
    "curation",
    "runtime",
}

IGNORED_CONFIG_SECTIONS = {
    "metadata",
    "notes",
    "phase_plan",
}


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config file, returning an empty mapping for ``None``."""

    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML config must contain a mapping at the top level: {config_path}")
    return dict(payload)


def flatten_cli_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten supported config sections into argparse destination names.

    The scripts keep their command-line surface as the source of truth.  YAML
    files may either use flat argparse destination names or group those names
    under known sections such as ``model`` or ``training``.
    """

    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        normalized_key = str(key).replace("-", "_")
        if normalized_key in IGNORED_CONFIG_SECTIONS:
            continue
        if normalized_key in CONFIG_SECTIONS:
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise ValueError(f"Config section {key!r} must be a mapping")
            for inner_key, inner_value in value.items():
                flattened[str(inner_key).replace("-", "_")] = inner_value
        else:
            flattened[normalized_key] = value
    return flattened


def apply_yaml_defaults(parser: argparse.ArgumentParser, path: str | Path | None) -> None:
    """Set parser defaults from a YAML config file.

    Unknown keys are treated as errors so stale configuration does not silently
    drift away from the scripts it is meant to drive.
    """

    payload = flatten_cli_config(load_yaml_config(path))
    if not payload:
        return
    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    unknown = sorted(key for key in payload if key not in actions)
    if unknown:
        raise ValueError(f"Unknown YAML config key(s) for {parser.prog}: {', '.join(unknown)}")
    parser.set_defaults(**payload)


def parse_config_path(argv: list[str] | None = None) -> str | None:
    """Read only ``--config`` from argv before building the full parser."""

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    args, _ = config_parser.parse_known_args(argv)
    return args.config
