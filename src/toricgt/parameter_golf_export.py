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
    deployment_parameters: int
    excluded_tensors: int
    compression: str


def _excluded(name: str, exclude_prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in exclude_prefixes)


def model_parameter_count(model: nn.Module, exclude_prefixes: tuple[str, ...] = ()) -> int:
    named = dict(model.named_parameters())
    return sum(param.numel() for name, param in named.items() if not _excluded(name, exclude_prefixes))


def _pack_unsigned(values: torch.Tensor, bits: int) -> bytes:
    """Pack low-bit unsigned integer tensor values into bytes."""

    if bits == 8:
        return bytes(values.to(torch.uint8).flatten().tolist())
    flat = values.to(torch.int64).flatten().tolist()
    out = bytearray()
    accumulator = 0
    filled = 0
    mask = (1 << bits) - 1
    for value in flat:
        accumulator |= (int(value) & mask) << filled
        filled += bits
        while filled >= 8:
            out.append(accumulator & 0xFF)
            accumulator >>= 8
            filled -= 8
    if filled:
        out.append(accumulator & 0xFF)
    return bytes(out)


def _quantize_tensor(cpu: torch.Tensor, bits: int, mode: str) -> dict[str, Any]:
    if mode not in {"tensor", "row"}:
        raise ValueError("quantization mode must be 'tensor' or 'row'")
    qmax = 2 ** (bits - 1) - 1
    if mode == "row" and cpu.ndim >= 2:
        flat = cpu.reshape(cpu.shape[0], -1)
        scales = flat.abs().amax(dim=1).clamp_min(1e-8) / qmax
        q = torch.round(flat / scales[:, None]).clamp(-qmax, qmax).to(torch.int16)
        q = q.reshape(cpu.shape)
        scale_payload: float | list[float] = scales.tolist()
    else:
        scale = cpu.abs().max().clamp_min(1e-8) / qmax
        q = torch.round(cpu / scale).clamp(-qmax, qmax).to(torch.int16)
        scale_payload = float(scale)
    if bits < 8:
        unsigned = (q + qmax).clamp(0, (1 << bits) - 1).to(torch.uint8)
        data: Any = _pack_unsigned(unsigned, bits)
        packed = True
    else:
        data = q.to(torch.int8)
        packed = False
    return {
        "shape": tuple(cpu.shape),
        "scale": scale_payload,
        "offset": qmax,
        "data": data,
        "packed": packed,
        "mode": mode,
    }


def quantized_state_dict(
    model: nn.Module,
    bits: int = 8,
    mode: str = "tensor",
    exclude_prefixes: tuple[str, ...] = ("aux_", "graphcg_direction_basis", "toric_geometry_probe."),
) -> dict[str, Any]:
    """Symmetric quantization for artifact accounting and export tests.

    Training-only modules such as auxiliary prediction heads are excluded by
    default.  For 4-bit and 6-bit exports, quantized values are bit-packed
    instead of stored as int8 tensors, which makes artifact-size estimates
    closer to the real competition constraint.
    """

    if bits not in {4, 6, 8}:
        raise ValueError("bits must be one of 4, 6, or 8")
    state: dict[str, Any] = {"__bits__": bits, "__mode__": mode, "tensors": {}, "excluded": []}
    for name, tensor in model.state_dict().items():
        if _excluded(name, exclude_prefixes):
            state["excluded"].append(name)
            continue
        if not torch.is_floating_point(tensor):
            state["tensors"][name] = {"dtype": str(tensor.dtype), "shape": tuple(tensor.shape), "data": tensor.cpu()}
            continue
        cpu = tensor.detach().cpu().float()
        state["tensors"][name] = _quantize_tensor(cpu, bits=bits, mode=mode)
    return state


def _zip_compression(name: str) -> int:
    if name == "deflated":
        return zipfile.ZIP_DEFLATED
    if name == "bzip2":
        return zipfile.ZIP_BZIP2
    if name == "lzma":
        return zipfile.ZIP_LZMA
    raise ValueError("compression must be one of: deflated, bzip2, lzma")


def write_artifact(
    model: nn.Module,
    path: str | Path,
    config: dict[str, Any] | None = None,
    bits: int = 8,
    quantization_mode: str = "tensor",
    compression: str = "deflated",
    exclude_prefixes: tuple[str, ...] = ("aux_", "graphcg_direction_basis", "toric_geometry_probe."),
) -> ArtifactReport:
    """Write a compressed experimental artifact and return byte accounting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = quantized_state_dict(model, bits=bits, mode=quantization_mode, exclude_prefixes=exclude_prefixes)
    total_parameters = model_parameter_count(model)
    deployment_parameters = model_parameter_count(model, exclude_prefixes=exclude_prefixes)
    metadata = {
        "config": config or {},
        "bits": bits,
        "quantization_mode": quantization_mode,
        "compression": compression,
        "parameters": total_parameters,
        "deployment_parameters": deployment_parameters,
        "excluded_tensors": len(payload["excluded"]),
    }
    with tempfile.TemporaryDirectory(prefix="toricgt_pg_") as tmp:
        tmp_path = Path(tmp)
        torch.save(payload, tmp_path / "weights.pt")
        (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        compression_id = _zip_compression(compression)
        zip_kwargs: dict[str, Any] = {"compression": compression_id}
        if compression in {"deflated", "bzip2"}:
            zip_kwargs["compresslevel"] = 9
        with zipfile.ZipFile(out, "w", **zip_kwargs) as archive:
            archive.write(tmp_path / "weights.pt", arcname="weights.pt")
            archive.write(tmp_path / "metadata.json", arcname="metadata.json")
    report = ArtifactReport(
        path=str(out),
        bytes_total=out.stat().st_size,
        bytes_limit=PARAMETER_GOLF_BYTE_LIMIT,
        within_limit=out.stat().st_size <= PARAMETER_GOLF_BYTE_LIMIT,
        tensors=len(payload["tensors"]),
        parameters=total_parameters,
        deployment_parameters=deployment_parameters,
        excluded_tensors=len(payload["excluded"]),
        compression=compression,
    )
    (out.with_suffix(out.suffix + ".json")).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report
