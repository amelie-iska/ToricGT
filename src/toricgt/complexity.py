"""Kolmogorov-style complexity diagnostics for ToricGT.

True Kolmogorov complexity is uncomputable.  This module provides explicit
estimator-tagged proxies that are useful for comparing reasoning traces,
random-order byte programs, graph serializations, and GFlowNet action paths.
All metrics are diagnostic by default; they should not be treated as a single
canonical "K" value.
"""

from __future__ import annotations

import json
import lzma
import math
import zlib
from collections.abc import Iterable, Mapping
from typing import Any

import torch


def _zstd_compress(data: bytes) -> bytes | None:
    try:
        import zstandard as zstd  # type: ignore
    except Exception:
        return None
    compressor = zstd.ZstdCompressor(level=3)
    return compressor.compress(data)


def compress_len(data: bytes, compressor: str = "lzma") -> int:
    """Return compressed length in bytes for a named estimator."""

    if compressor == "raw":
        return len(data)
    if compressor == "zlib":
        return len(zlib.compress(data, level=9))
    if compressor == "lzma":
        return len(lzma.compress(data, preset=6))
    if compressor == "zstd":
        payload = _zstd_compress(data)
        if payload is None:
            return len(lzma.compress(data, preset=6))
        return len(payload)
    raise ValueError(f"unknown compressor: {compressor}")


def conditional_compress_len(context: bytes, payload: bytes, compressor: str = "lzma") -> int:
    """Approximate K(payload | context) by a two-part compressor code."""

    joint = context + b"\x00<ToricGT-KCOND>\x00" + payload
    return max(0, compress_len(joint, compressor) - compress_len(context, compressor))


def shortest_known_program_len(
    payload: bytes,
    *,
    compressor: str = "lzma",
    references: Iterable[bytes] = (),
) -> int:
    """Return the shortest compressor program currently known for ``payload``.

    This is a computable proxy, not true Kolmogorov complexity.  The candidates
    are the unconditional compressed payload and conditional codes against the
    supplied references.  Adding more references can only decrease the proxy, so
    every metric that uses this function must report the estimator and context.
    """

    candidates = [compress_len(payload, compressor)]
    for reference in references:
        if reference:
            candidates.append(conditional_compress_len(reference, payload, compressor))
    return int(min(candidates))


def relative_complexity(complexity_value: float, shortest_known: float) -> float:
    """Signed complexity relative to the best known program length.

    Negative values mean the candidate representation is shorter than the
    reference program.  Positive values mean it is longer.
    """

    return float(complexity_value) - float(shortest_known)


def relative_complexity_reward(
    complexity_value: float,
    shortest_known: float,
    *,
    clip: float = 128.0,
) -> float:
    """Clipped positive reward for beating a known program length."""

    return float(max(0.0, min(float(clip), float(shortest_known) - float(complexity_value))))


def normalized_compression_distance(a: bytes, b: bytes, compressor: str = "lzma") -> float:
    """Normalized compression distance proxy in [0, inf), usually near [0, 1]."""

    ka = compress_len(a, compressor)
    kb = compress_len(b, compressor)
    denom = max(ka, kb, 1)
    kab = compress_len(a + b"\x00<NCD>\x00" + b, compressor)
    return max(0.0, float(kab - min(ka, kb)) / float(denom))


def token_bytes(tokens: torch.Tensor | Iterable[int], byte_offset: int = 4) -> bytes:
    """Convert byte-offset vocabulary ids back to raw bytes, dropping specials."""

    if isinstance(tokens, torch.Tensor):
        values = tokens.detach().cpu().reshape(-1).tolist()
    else:
        values = list(tokens)
    return bytes(
        max(0, min(255, int(value) - byte_offset))
        for value in values
        if byte_offset <= int(value) <= byte_offset + 255
    )


def int_sequence_bytes(values: torch.Tensor | Iterable[int], width: int = 4) -> bytes:
    """Serialize an integer sequence to little-endian signed differences."""

    if isinstance(values, torch.Tensor):
        seq = [int(item) for item in values.detach().cpu().reshape(-1).tolist()]
    else:
        seq = [int(item) for item in values]
    if not seq:
        return b""
    diffs = [seq[0], *[seq[index] - seq[index - 1] for index in range(1, len(seq))]]
    lo = -(2 ** (8 * width - 1))
    hi = 2 ** (8 * width - 1) - 1
    return b"".join(max(lo, min(hi, value)).to_bytes(width, "little", signed=True) for value in diffs)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def _metric_stats(prefix: str, name: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        f"{prefix}/{name}_mean": _mean(values),
        f"{prefix}/{name}_std": _std(values),
        f"{prefix}/{name}_min": float(min(values)),
        f"{prefix}/{name}_max": float(max(values)),
    }


def random_order_complexity_metrics(
    tokens: torch.Tensor,
    previous_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    permutation: torch.Tensor,
    logits: torch.Tensor | None = None,
    action_ids: torch.Tensor | None = None,
    byte_offset: int = 4,
    prefix: str = "complexity/train",
    compressors: tuple[str, ...] = ("zlib", "lzma"),
    max_samples: int = 4,
    helper_payloads: list[Mapping[str, bytes | str]] | None = None,
) -> dict[str, float]:
    """Compute complexity proxies for a random-order byte batch.

    The metrics are intentionally small-sample and CPU-side.  They are intended
    for periodic W&B diagnostics, not every microbatch on every step.
    """

    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, length]")
    batch = min(int(tokens.shape[0]), max(1, int(max_samples)))
    metrics: dict[str, float] = {f"{prefix}/samples": float(batch)}
    pred_tokens = logits.argmax(dim=-1) if logits is not None else None
    if pred_tokens is not None:
        accuracy = (pred_tokens[:batch] == target_tokens[:batch]).float().mean().item()
        metrics[f"{prefix}/argmax_byte_accuracy"] = float(accuracy)

    for compressor in compressors:
        input_k: list[float] = []
        target_k: list[float] = []
        target_cond_k: list[float] = []
        order_k: list[float] = []
        pred_k: list[float] = []
        pred_cond_k: list[float] = []
        action_k: list[float] = []
        target_rel_raw_k: list[float] = []
        target_known_k: list[float] = []
        target_helper_cond_k: list[float] = []
        information_symmetry_gap_k: list[float] = []
        pred_known_k: list[float] = []
        pred_relative_known_k: list[float] = []
        pred_relative_raw_reward: list[float] = []
        pred_relative_reward: list[float] = []
        pred_relative_gate: list[float] = []
        analogy_transfer_k: list[float] = []
        analogy_transfer_relative_k: list[float] = []
        analogy_transfer_gain_k: list[float] = []
        analogy_transfer_reward: list[float] = []
        pred_analogy_transfer_k: list[float] = []
        pred_analogy_transfer_relative_k: list[float] = []
        pred_analogy_transfer_gain_k: list[float] = []
        input_target_ncd: list[float] = []
        pred_target_ncd: list[float] = []
        sample_contexts: list[bytes] = []
        sample_previous: list[bytes] = []
        sample_direct_helpers: list[list[bytes]] = []
        sample_targets: list[bytes] = []
        sample_predictions: list[bytes] = []
        for index in range(batch):
            input_bytes = token_bytes(tokens[index], byte_offset=byte_offset)
            previous_bytes = token_bytes(previous_tokens[index], byte_offset=byte_offset)
            target_bytes = token_bytes(target_tokens[index], byte_offset=byte_offset)
            order_bytes = int_sequence_bytes(permutation[index], width=4)
            action_bytes = (
                int_sequence_bytes(action_ids[index], width=2)
                if action_ids is not None
                else b""
            )
            extra_helpers: list[bytes] = []
            if helper_payloads is not None and index < len(helper_payloads):
                for helper_name, helper_payload in sorted(helper_payloads[index].items()):
                    if isinstance(helper_payload, bytes):
                        payload_bytes = helper_payload
                    else:
                        payload_bytes = str(helper_payload).encode("utf-8", errors="replace")
                    if payload_bytes:
                        extra_helpers.append(
                            b"\x00<ToricGT-HELPER:"
                            + str(helper_name).encode("utf-8", errors="replace")
                            + b">\x00"
                            + payload_bytes
                        )
            direct_helpers = [previous_bytes, input_bytes, order_bytes, action_bytes, *extra_helpers]
            input_len = max(1, len(input_bytes))
            target_len = max(1, len(target_bytes))
            input_value = compress_len(input_bytes, compressor)
            target_value = compress_len(target_bytes, compressor)
            target_cond_value = conditional_compress_len(previous_bytes, target_bytes, compressor)
            known_target = shortest_known_program_len(
                target_bytes,
                compressor=compressor,
                references=direct_helpers,
            )
            target_helper_value = conditional_compress_len(
                b"\x00<ToricGT-DIRECT-HELPERS>\x00".join(helper for helper in direct_helpers if helper),
                target_bytes,
                compressor,
            )
            info_xy = float(target_value - conditional_compress_len(input_bytes, target_bytes, compressor))
            info_yx = float(input_value - conditional_compress_len(target_bytes, input_bytes, compressor))
            input_k.append(float(input_value))
            target_k.append(float(target_value))
            target_cond_k.append(float(target_cond_value))
            target_rel_raw_k.append(relative_complexity(target_cond_value, target_value))
            target_known_k.append(float(known_target))
            target_helper_cond_k.append(float(target_helper_value))
            information_symmetry_gap_k.append(abs(info_xy - info_yx))
            order_k.append(float(compress_len(order_bytes, compressor)))
            input_target_ncd.append(normalized_compression_distance(input_bytes, target_bytes, compressor))
            sample_previous.append(previous_bytes)
            sample_targets.append(target_bytes)
            sample_direct_helpers.append([helper for helper in direct_helpers if helper])
            sample_contexts.append(
                b"\x00<ToricGT-ANALOGY-SOURCE-X>\x00"
                + previous_bytes
                + b"\x00<ToricGT-ANALOGY-SOURCE-Y>\x00"
                + target_bytes
                + b"\x00<ToricGT-ANALOGY-SOURCE-INPUT>\x00"
                + input_bytes
                + b"\x00<ToricGT-ANALOGY-SOURCE-ORDER>\x00"
                + order_bytes
                + b"\x00<ToricGT-ANALOGY-SOURCE-ACTIONS>\x00"
                + action_bytes
                + b"".join(extra_helpers)
            )
            metrics[f"{prefix}/input_k_{compressor}_per_byte_sample_{index}"] = float(input_value) / input_len
            metrics[f"{prefix}/target_k_{compressor}_per_byte_sample_{index}"] = float(target_value) / target_len
            if pred_tokens is not None:
                pred_bytes = token_bytes(pred_tokens[index], byte_offset=byte_offset)
                pred_value = compress_len(pred_bytes, compressor)
                pred_cond_value = conditional_compress_len(previous_bytes, pred_bytes, compressor)
                known_pred = shortest_known_program_len(
                    pred_bytes,
                    compressor=compressor,
                    references=direct_helpers,
                )
                correctness_gate = float((pred_tokens[index] == target_tokens[index]).float().mean().item())
                pred_k.append(float(pred_value))
                pred_cond_k.append(float(pred_cond_value))
                pred_known_k.append(float(known_pred))
                pred_relative_known_k.append(relative_complexity(known_pred, known_target))
                raw_reward = relative_complexity_reward(known_pred, known_target)
                pred_relative_raw_reward.append(raw_reward)
                pred_relative_reward.append(raw_reward * correctness_gate)
                pred_relative_gate.append(correctness_gate)
                pred_target_ncd.append(normalized_compression_distance(pred_bytes, target_bytes, compressor))
                sample_predictions.append(pred_bytes)
            if action_ids is not None:
                action_k.append(float(compress_len(action_bytes, compressor)))
        for index in range(1, batch):
            analogy_context = (
                b"\x00<ToricGT-DIRECT-HELPERS>\x00".join(sample_direct_helpers[index])
                + b"\x00<ToricGT-ANALOGY-SOURCE>\x00"
                + sample_contexts[index - 1]
            )
            transfer_value = shortest_known_program_len(
                sample_targets[index],
                compressor=compressor,
                references=(*sample_direct_helpers[index], analogy_context),
            )
            direct_value = shortest_known_program_len(
                sample_targets[index],
                compressor=compressor,
                references=sample_direct_helpers[index],
            )
            transfer_gain = float(direct_value - transfer_value)
            analogy_transfer_k.append(float(transfer_value))
            analogy_transfer_relative_k.append(relative_complexity(transfer_value, direct_value))
            analogy_transfer_gain_k.append(transfer_gain)
            analogy_transfer_reward.append(max(0.0, min(128.0, transfer_gain)))
            if pred_tokens is not None and index < len(sample_predictions):
                pred_transfer_value = shortest_known_program_len(
                    sample_predictions[index],
                    compressor=compressor,
                    references=(*sample_direct_helpers[index], analogy_context),
                )
                pred_direct_value = shortest_known_program_len(
                    sample_predictions[index],
                    compressor=compressor,
                    references=sample_direct_helpers[index],
                )
                pred_analogy_transfer_k.append(float(pred_transfer_value))
                pred_analogy_transfer_relative_k.append(relative_complexity(pred_transfer_value, pred_direct_value))
                pred_analogy_transfer_gain_k.append(float(pred_direct_value - pred_transfer_value))
        metrics.update(_metric_stats(prefix, f"input_k_{compressor}", input_k))
        metrics.update(_metric_stats(prefix, f"target_k_{compressor}", target_k))
        metrics.update(_metric_stats(prefix, f"target_cond_k_{compressor}", target_cond_k))
        metrics.update(_metric_stats(prefix, f"target_relative_to_raw_k_{compressor}", target_rel_raw_k))
        metrics.update(_metric_stats(prefix, f"target_shortest_known_program_k_{compressor}", target_known_k))
        metrics.update(_metric_stats(prefix, f"target_helper_cond_k_{compressor}", target_helper_cond_k))
        metrics.update(_metric_stats(prefix, f"information_symmetry_gap_k_{compressor}", information_symmetry_gap_k))
        metrics.update(_metric_stats(prefix, f"order_program_k_{compressor}", order_k))
        metrics.update(_metric_stats(prefix, f"input_target_ncd_{compressor}", input_target_ncd))
        if pred_k:
            metrics.update(_metric_stats(prefix, f"prediction_k_{compressor}", pred_k))
            metrics.update(_metric_stats(prefix, f"prediction_cond_k_{compressor}", pred_cond_k))
            metrics.update(_metric_stats(prefix, f"prediction_shortest_known_program_k_{compressor}", pred_known_k))
            metrics.update(
                _metric_stats(prefix, f"prediction_relative_to_target_known_k_{compressor}", pred_relative_known_k)
            )
            metrics.update(
                _metric_stats(prefix, f"prediction_relative_k_raw_reward_{compressor}", pred_relative_raw_reward)
            )
            metrics.update(_metric_stats(prefix, f"prediction_relative_k_reward_{compressor}", pred_relative_reward))
            metrics.update(_metric_stats(prefix, f"prediction_relative_k_correctness_gate_{compressor}", pred_relative_gate))
            metrics.update(_metric_stats(prefix, f"prediction_target_ncd_{compressor}", pred_target_ncd))
        if action_k:
            metrics.update(_metric_stats(prefix, f"gflownet_action_trace_k_{compressor}", action_k))
        if analogy_transfer_k:
            metrics.update(_metric_stats(prefix, f"analogical_transfer_target_k_{compressor}", analogy_transfer_k))
            metrics.update(
                _metric_stats(prefix, f"analogical_transfer_relative_k_{compressor}", analogy_transfer_relative_k)
            )
            metrics.update(_metric_stats(prefix, f"analogical_transfer_gain_k_{compressor}", analogy_transfer_gain_k))
            metrics.update(_metric_stats(prefix, f"analogical_transfer_reward_{compressor}", analogy_transfer_reward))
        if pred_analogy_transfer_k:
            metrics.update(
                _metric_stats(prefix, f"prediction_analogical_transfer_k_{compressor}", pred_analogy_transfer_k)
            )
            metrics.update(
                _metric_stats(
                    prefix,
                    f"prediction_analogical_transfer_relative_k_{compressor}",
                    pred_analogy_transfer_relative_k,
                )
            )
            metrics.update(
                _metric_stats(
                    prefix,
                    f"prediction_analogical_transfer_gain_k_{compressor}",
                    pred_analogy_transfer_gain_k,
                )
            )
    return metrics


def _canonicalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        items = {}
        for key, item in value.items():
            if key in {"uuid", "created_at", "updated_at", "source_index"}:
                continue
            items[str(key)] = _canonicalize_json(item)
        return {key: items[key] for key in sorted(items)}
    if isinstance(value, list):
        canonical_items = [_canonicalize_json(item) for item in value]
        return sorted(canonical_items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    if isinstance(value, float):
        return round(value, 8)
    return value


def canonical_graph_bytes(graph_json: str | Mapping[str, Any]) -> bytes:
    """Return a deterministic graph serialization for MDL-style diagnostics."""

    if isinstance(graph_json, str):
        try:
            payload = json.loads(graph_json)
        except json.JSONDecodeError:
            payload = {"raw": graph_json}
    else:
        payload = dict(graph_json)
    canonical = _canonicalize_json(payload)
    return json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def graph_mdl_metrics(
    graph_json: str | Mapping[str, Any],
    prefix: str = "complexity/graph",
    compressors: tuple[str, ...] = ("zlib", "lzma"),
) -> dict[str, float]:
    """Compute simple graph-MDL metrics for one graph JSON payload."""

    if isinstance(graph_json, str):
        try:
            payload = json.loads(graph_json)
        except json.JSONDecodeError:
            payload = {"raw": graph_json}
    else:
        payload = dict(graph_json)
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    targets = payload.get("targets", {})
    node_count = len(nodes) if isinstance(nodes, list) else 0
    edge_count = len(edges) if isinstance(edges, list) else 0
    structure = {
        "node_types": sorted({str(node.get("type", "")) for node in nodes if isinstance(node, Mapping)}),
        "edge_types": sorted({str(edge.get("type", edge.get("label", ""))) for edge in edges if isinstance(edge, Mapping)}),
        "node_count": node_count,
        "edge_count": edge_count,
    }
    payload_bytes = canonical_graph_bytes(payload)
    structure_bytes = json.dumps(structure, sort_keys=True, separators=(",", ":")).encode("utf-8")
    target_bytes = json.dumps(_canonicalize_json(targets), sort_keys=True, ensure_ascii=False).encode("utf-8")
    metrics = {
        f"{prefix}/nodes": float(node_count),
        f"{prefix}/edges": float(edge_count),
        f"{prefix}/payload_bytes": float(len(payload_bytes)),
    }
    for compressor in compressors:
        metrics[f"{prefix}/mdl_total_{compressor}"] = float(compress_len(payload_bytes, compressor))
        metrics[f"{prefix}/mdl_structure_{compressor}"] = float(compress_len(structure_bytes, compressor))
        metrics[f"{prefix}/mdl_target_{compressor}"] = float(compress_len(target_bytes, compressor))
    return metrics
