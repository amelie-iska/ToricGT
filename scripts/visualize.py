#!/usr/bin/env python3
"""Generate toric/tropical visual diagnostics."""

from __future__ import annotations

import argparse

import numpy as np
from tqdm.auto import tqdm

from toricgt.toric_algebra import braid_generator_points
from toricgt.visualization import plot_tropical_decision_boundary, plot_unit_circle_braid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/visualizations")
    args = parser.parse_args()

    base = np.exp(2j * np.pi * np.linspace(0, 1, 12, endpoint=False))
    for frame, t in enumerate(tqdm(np.linspace(0, 1, 16), desc="braid frames")):
        points = braid_generator_points(base, 3, float(t))
        plot_unit_circle_braid(points, f"{args.output_dir}/braid_{frame:03d}.png")

    rng = np.random.default_rng(17)
    points = rng.normal(size=(1000, 2))
    labels = np.argmax(np.stack([points[:, 0], points[:, 1], -points[:, 0] - points[:, 1]], axis=1), axis=1)
    plot_tropical_decision_boundary(points, labels, f"{args.output_dir}/tropical_boundary.png")


if __name__ == "__main__":
    main()

