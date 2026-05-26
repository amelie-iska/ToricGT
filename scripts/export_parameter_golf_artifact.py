#!/usr/bin/env python3
"""Export a compressed experimental Parameter-Golf artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from toricgt.config import ModelConfig
from toricgt.model import ToricTokenGT
from toricgt.parameter_golf_export import write_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/parameter_golf/toricgt_artifact.zip")
    parser.add_argument("--bits", type=int, choices=[4, 6, 8], default=8)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    cfg = ModelConfig(**payload["config"])
    model = ToricTokenGT(cfg)
    model.load_state_dict(payload["model"])
    report = write_artifact(model, Path(args.output), config=payload["config"], bits=args.bits)
    print(report)


if __name__ == "__main__":
    main()
