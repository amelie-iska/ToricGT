"""Byte accounting helpers for Parameter-Golf-style artifacts."""

from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


PARAMETER_GOLF_BYTE_LIMIT = 16_000_000


@dataclass(frozen=True)
class ArtifactReport:
    path: str
    bytes_total: int
    bytes_limit: int
    within_limit: bool
    tensors: int
    parameters: int


def model_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def quantized_state_dict(model: nn.Module, bits: int = 8) -> dict[str, Any]:
    """Simple symmetric per-tensor quantization for size experiments."""

    if bits not in {4, 6, 8}:
        raise ValueError("bits must be one of 4, 6, or 8")
    qmax = 2 ** (bits - 1) - 1
    state: dict[str, Any] = {"__bits__": bits, "tensors": {}}
    for name, tensor in model.state_dict().items():
        if not torch.is_floating_point(tensor):
            state["tensors"][name] = {"dtype": str(tensor.dtype), "shape": tuple(tensor.shape), "data": tensor.cpu()}
            continue
        cpu = tensor.detach().cpu().float()
        scale = cpu.abs().max().clamp_min(1e-8) / qmax
        q = torch.round(cpu / scale).clamp(-qmax, qmax).to(torch.int8)
        state["tensors"][name] = {"shape": tuple(cpu.shape), "scale": float(scale), "data": q}
    return state


def write_artifact(model: nn.Module, path: str | Path, config: dict[str, Any] | None = None, bits: int = 8) -> ArtifactReport:
    """Write a compressed experimental artifact and return byte accounting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = quantized_state_dict(model, bits=bits)
    metadata = {"config": config or {}, "bits": bits, "parameters": model_parameter_count(model)}
    with tempfile.TemporaryDirectory(prefix="toricgt_pg_") as tmp:
        tmp_path = Path(tmp)
        torch.save(payload, tmp_path / "weights.pt")
        (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.write(tmp_path / "weights.pt", arcname="weights.pt")
            archive.write(tmp_path / "metadata.json", arcname="metadata.json")
    report = ArtifactReport(
        path=str(out),
        bytes_total=out.stat().st_size,
        bytes_limit=PARAMETER_GOLF_BYTE_LIMIT,
        within_limit=out.stat().st_size <= PARAMETER_GOLF_BYTE_LIMIT,
        tensors=len(model.state_dict()),
        parameters=model_parameter_count(model),
    )
    (out.with_suffix(out.suffix + ".json")).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report
