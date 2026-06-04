from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_reasoning_geometry_suite.py"


def load_module():
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("evaluate_reasoning_geometry_suite", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_geometry_energy_landscape_uses_static_3d_axis(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    projections: list[str | None] = []
    original_add_subplot = Figure.add_subplot

    def capture_add_subplot(self, *args, **kwargs):
        projections.append(kwargs.get("projection"))
        return original_add_subplot(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "add_subplot", capture_add_subplot)
    t = np.linspace(0.0, 2.0 * np.pi, 24)
    branches = [
        {
            "projected_path": np.column_stack([np.cos(t), np.sin(t), t / np.pi]),
            "per_token_nll": 1.0 + 0.2 * np.cos(t),
        },
        {
            "projected_path": np.column_stack([0.7 * np.cos(t + 0.2), 0.7 * np.sin(t + 0.2), 0.2 + t / np.pi]),
            "per_token_nll": 1.1 + 0.15 * np.sin(t),
        },
    ]

    output = tmp_path / "full_energy_3d.png"
    module.plot_energy_landscape({"record_index": 0}, branches, output)

    assert output.exists()
    assert "3d" in projections
