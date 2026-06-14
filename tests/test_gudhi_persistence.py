from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from toricgt.gudhi_persistence import (
    GudhiPersistenceConfig,
    audit_point_cloud,
    bigraded_chain_presentation,
    torch_persistence_image,
    torch_persistence_landscape,
)


def test_gudhi_macaulay2_bigraded_resolution_for_small_cloud() -> None:
    if shutil.which("M2") is None:
        pytest.skip("Macaulay2 is not installed")
    points = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.45, 0.55]],
        dtype=float,
    )
    cfg = GudhiPersistenceConfig(
        max_points=5,
        num_radii=3,
        num_levels=3,
        landscape_resolution=12,
        landscape_layers=2,
        image_resolution=5,
        macaulay2_timeout_seconds=60,
    )
    audit = audit_point_cloud(points, record_id="small_loop", cfg=cfg)

    assert audit["simplex_tree"]["num_simplices"] > 0
    assert audit["two_parameter_module"]["ring"] == "F2[x_level,y_radius]"
    assert audit["two_parameter_module"]["commutative_square_residual"] == 0.0
    assert audit["two_parameter_module"]["commutative_square_chain_residuals_by_dimension"] == {
        "0": 0,
        "1": 0,
        "2": 0,
    }
    assert audit["macaulay2_resolution"]["kind"] == "macaulay2_bigraded_persistence_resolution"
    assert audit["macaulay2_resolution"]["homogeneous_d1"] is True
    assert audit["macaulay2_resolution"]["homogeneous_d2"] is True
    assert audit["macaulay2_resolution"]["d_squared_zero"] is True
    assert "H0_betti" in audit["macaulay2_resolution"]["homology_and_resolutions"]


def test_bigraded_chain_presentation_has_nonnegative_boundary_monomials() -> None:
    points = np.asarray([[0.0], [0.4], [1.0], [1.3]], dtype=float)
    radii = np.asarray([0.1, 0.5, 1.0], dtype=float)
    presentation = bigraded_chain_presentation(points, radii, max_dimension=2)
    for matrix in presentation["boundary_matrices"].values():
        for row in matrix:
            for entry in row:
                assert "^- " not in entry
                assert "x_level^-" not in entry
                assert "y_radius^-" not in entry


def test_torch_persistence_vectorizers_are_differentiable() -> None:
    diagram = torch.tensor([[0.0, 0.4], [0.1, 0.8], [0.25, 0.9]], requires_grad=True)
    grid = torch.linspace(0.0, 1.0, 16)
    landscape = torch_persistence_landscape(diagram, grid, layers=3)
    image = torch_persistence_image(diagram, grid, grid, sigma=0.15)
    loss = landscape.pow(2).mean() + image.pow(2).mean()
    loss.backward()

    assert landscape.shape == (3, 16)
    assert image.shape == (16, 16)
    assert diagram.grad is not None
    assert torch.isfinite(diagram.grad).all()


def test_gudhi_audit_script_writes_dark_html_and_m2_script(tmp_path: Path) -> None:
    if shutil.which("M2") is None:
        pytest.skip("Macaulay2 is not installed")
    points = tmp_path / "points.json"
    points.write_text(
        json.dumps(
            {
                "point_clouds": [
                    {
                        "record_id": "audit_square",
                        "points": [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "audit"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_gudhi_persistence_audit.py",
            "--points-json",
            str(points),
            "--output-dir",
            str(out),
            "--records",
            "1",
            "--max-points",
            "5",
            "--num-radii",
            "3",
            "--num-levels",
            "3",
            "--landscape-resolution",
            "12",
            "--image-resolution",
            "5",
            "--macaulay2-timeout-seconds",
            "60",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    index = out / "index.html"
    assert index.exists()
    assert "F2[x_level,y_radius]" in index.read_text(encoding="utf-8")
    assert list((out / "records").glob("*.html"))
    assert list((out / "macaulay2").glob("*.m2"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["mean_macaulay2_d_squared_zero"] == 1.0


def test_watcher_index_and_gudhi_args_are_available() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "watch_training_analysis.py"
    spec = importlib.util.spec_from_file_location("watch_training_analysis", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.parse_args(
        [
            "--start-step",
            "0",
            "--target-step",
            "250",
            "--gudhi-persistence-audit",
            "--gudhi-records",
            "2",
        ]
    )
    assert args.gudhi_persistence_audit is True
    assert args.gudhi_records == 2
