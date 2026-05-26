import numpy as np

from toricgt.visualization import (
    plot_energy_landscape,
    plot_ramachandran_style_reasoning,
    plot_reasoning_trajectory_3d,
    reasoning_trajectory,
    write_interactive_reasoning_plot,
)


def test_reasoning_trajectory_visualizations(tmp_path):
    path, energy = reasoning_trajectory(seed=23, steps=24)
    assert path.shape == (24, 3)
    assert energy.shape == (24,)
    assert np.isfinite(path).all()
    assert np.isfinite(energy).all()

    outputs = [
        tmp_path / "trajectory.png",
        tmp_path / "torsion.png",
        tmp_path / "landscape.png",
        tmp_path / "trajectory.html",
    ]
    plot_reasoning_trajectory_3d(path, energy, outputs[0])
    plot_ramachandran_style_reasoning(path, energy, outputs[1])
    plot_energy_landscape(path, energy, outputs[2])
    write_interactive_reasoning_plot(path, energy, outputs[3])

    for output in outputs:
        assert output.exists()
        assert output.stat().st_size > 0
