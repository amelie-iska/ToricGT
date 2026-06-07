"""Canonical W&B metric organization for ToricGT training and analyses.

The training stack emits many historical aliases because older watchdogs,
analysis scripts, and notebooks consume them.  This module adds a smaller
ordered dashboard surface and suppresses those raw aliases by default:

* ``00_primary/*``: the scorecard a human should inspect first.
* ``01_oai/*`` through ``16_status/*``: stable, visible categories.
* raw historical namespaces can be re-enabled through
  ``TORICGT_WANDB_RAW_MODE=raw`` or ``TORICGT_WANDB_RAW_MODE=legacy``.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Any, Iterable, Mapping


STEP_METRIC = "trainer/step"

VISIBLE_METRIC_PREFIXES = (
    "00_primary",
    "01_oai",
    "02_train",
    "03_validation",
    "04_losses",
    "05_gflownet",
    "06_graphcg",
    "07_topology_geometry",
    "08_toric_tropical_bgg",
    "09_complexity",
    "10_data_curriculum",
    "11_artifact_size",
    "12_optimization",
    "13_system",
    "14_phase_adaptive",
    "15_analysis_media",
    "16_status",
)

RAW_MODE_ENV = "TORICGT_WANDB_RAW_MODE"
RAW_MODE_DEFAULT = "off"

RAW_HIDDEN_PATTERNS = (
    "99_legacy/*",
    "train/*",
    "val/*",
    "bpb",
    "val_bpb",
    "train_bpb",
    "oai_competition/*",
    "fineweb/*",
    "fineweb_calibration/*",
    "openai_parameter_golf/*",
    "competition/*",
    "bpb/*",
    "complexity/*",
    "hessian/*",
    "data/*",
    "phase/*",
    "adaptive/*",
    "advanced_control/*",
    "artifact/*",
    "model/*",
    "polarquant/*",
    "metrics_status/*",
    "system/*",
    "analysis/*",
    "reasoning_simplex/*",
    "topology/*",
    "toric/*",
    "tropical/*",
    "bgg_category_o/*",
    "category_o/*",
    "koszul/*",
    "slepian_pollak/*",
    "checkpoint/*",
    "trigger/*",
    "progress/*",
    "time/*",
    "eval/*",
    "audit/*",
    "publish/*",
    "diagnostics/*",
)

MINIMIZE_PATTERNS = (
    "00_primary/*bpb*",
    "01_oai/*bpb*",
    "02_train/*loss*",
    "03_validation/*loss*",
    "04_losses/*",
    "05_gflownet/*loss*",
    "06_graphcg/*loss*",
    "07_topology_geometry/*loss*",
    "08_toric_tropical_bgg/*loss*",
    "09_complexity/*ncd*",
    "11_artifact_size/polarquant/*cache_mb*",
    "12_optimization/*grad_norm*",
)

MAXIMIZE_PATTERNS = (
    "00_primary/*target_reached*",
    "00_primary/*full_dataset*",
    "05_gflownet/*diversity*",
    "06_graphcg/*basis*",
    "07_topology_geometry/*stability*",
    "08_toric_tropical_bgg/*entropy*",
    "11_artifact_size/polarquant/*compression_ratio*",
    "10_data_curriculum/*active*",
    "11_artifact_size/*within_limit*",
)


def _raw_mode_from_env() -> str:
    mode = os.environ.get(RAW_MODE_ENV, RAW_MODE_DEFAULT).strip().lower()
    if mode in {"0", "false", "no", "off", "disabled"}:
        return "off"
    if mode in {"1", "true", "yes", "on", "raw"}:
        return "raw"
    if mode in {"legacy", "prefixed", "compat"}:
        return "legacy"
    return RAW_MODE_DEFAULT


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _first_numeric(payload: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if _is_number(value):
            return float(value)
    return None


def _add_first(out: OrderedDict[str, Any], payload: Mapping[str, Any], target: str, keys: Iterable[str]) -> None:
    value = _first_numeric(payload, keys)
    if value is not None:
        out[target] = value


def _clean_name(name: str) -> str:
    return name.replace("//", "/").strip("/")


def _category_alias(key: str) -> str | None:
    if key == STEP_METRIC or key.startswith(tuple(prefix + "/" for prefix in VISIBLE_METRIC_PREFIXES)):
        return None

    if key == "bpb" or key == "val_bpb":
        return "01_oai/val_bpb"
    if key == "train_bpb":
        return "02_train/bpb"

    if key.startswith("oai_competition/"):
        rest = key.split("/", 1)[1]
        rest = {
            "bpb": "val_bpb",
            "loss": "val_loss",
            "best_bpb": "best_val_bpb",
        }.get(rest, rest)
        return f"01_oai/{rest}"
    if key.startswith("openai_parameter_golf/"):
        rest = key.split("/", 1)[1]
        if rest == "bpb":
            return "01_oai/val_bpb"
        if rest == "train_bpb":
            return "02_train/bpb"
        if rest == "best_bpb":
            return "01_oai/best_val_bpb"
        if rest == "target_bpb":
            return "01_oai/target_bpb"
        return f"01_oai/openai_parameter_golf/{rest}"
    if key.startswith("competition/"):
        return f"01_oai/competition/{key.split('/', 1)[1]}"
    if key.startswith("fineweb/"):
        rest = key.split("/", 1)[1]
        if rest == "val_bpb":
            return "01_oai/val_bpb"
        if rest == "best_val_bpb":
            return "01_oai/best_val_bpb"
        if rest == "target_bpb":
            return "01_oai/target_bpb"
        if rest == "target_gap_bpb":
            return "01_oai/gap_to_target"
        if rest == "target_reached":
            return "01_oai/target_reached"
        if rest == "train_bpb":
            return "02_train/bpb"
        if rest == "train_loss":
            return "02_train/loss"
        if rest == "val_loss":
            return "03_validation/loss"
        return f"01_oai/fineweb/{rest}"
    if key.startswith("fineweb_calibration/"):
        return f"01_oai/fineweb_calibration/{key.split('/', 1)[1]}"
    if key.startswith("bpb/"):
        rest = key.split("/", 1)[1]
        if rest == "val":
            return "01_oai/val_bpb"
        if rest == "best":
            return "01_oai/best_val_bpb"
        if rest == "train":
            return "02_train/bpb"
        if rest == "target":
            return "01_oai/target_bpb"
        if rest == "gap_to_target":
            return "01_oai/gap_to_target"
        if rest == "target_reached":
            return "01_oai/target_reached"
        if rest == "improvement_from_initial":
            return "01_oai/val_delta_from_checkpoint_initial"
        return f"01_oai/bpb_diagnostic/{rest}"

    if key.startswith("val/"):
        return f"03_validation/{key.split('/', 1)[1]}"

    if key.startswith("train/"):
        rest = key.split("/", 1)[1]
        lower = rest.lower()
        if "gflownet" in lower:
            return f"05_gflownet/train/{rest}"
        if "graphcg" in lower:
            return f"06_graphcg/train/{rest}"
        if any(
            token in lower
            for token in ("derived_category", "mapping_cone", "chain_map", "projective_resolution", "betti_transport", "boundary_2")
        ):
            return f"08_toric_tropical_bgg/train/{rest}"
        if any(token in lower for token in ("analogy", "topology", "hdbscan", "simplex", "trajectory", "got_dag", "branch", "merge")):
            return f"07_topology_geometry/train/{rest}"
        if any(token in lower for token in ("toric", "tropical", "bgg", "category_o", "koszul", "slepian", "pollak")):
            return f"08_toric_tropical_bgg/train/{rest}"
        if any(token in lower for token in ("lr", "grad", "shock", "guard", "update", "nonfinite", "skip")):
            return f"12_optimization/train/{rest}"
        if any(token in lower for token in ("loss", "entropy", "mtp", "contrastive")):
            return f"04_losses/train/{rest}"
        return f"02_train/{rest}"

    if key.startswith("advanced/"):
        rest = key.split("/", 1)[1]
        lower = rest.lower()
        if "graphcg" in lower:
            return f"06_graphcg/advanced/{rest}"
        if any(
            token in lower
            for token in (
                "toric",
                "tropical",
                "bgg",
                "category_o",
                "koszul",
                "slepian",
                "pollak",
                "cca",
                "derived_category",
                "mapping_cone",
                "chain_map",
                "projective_resolution",
            )
        ):
            return f"08_toric_tropical_bgg/advanced/{rest}"
        if any(token in lower for token in ("topology", "hdbscan", "simplex", "trajectory", "analogy", "got_dag", "branch", "merge")):
            return f"07_topology_geometry/advanced/{rest}"
        if any(token in lower for token in ("nonfinite", "clamp", "runtime", "aux")):
            return f"12_optimization/advanced/{rest}"
        return f"04_losses/advanced/{rest}"

    if key.startswith("complexity/"):
        return f"09_complexity/{key.split('/', 1)[1]}"
    if key.startswith("hessian/"):
        return f"12_optimization/hessian/{key.split('/', 1)[1]}"
    if key.startswith("data/"):
        return f"10_data_curriculum/{key.split('/', 1)[1]}"
    if key.startswith("polarquant/"):
        return f"11_artifact_size/polarquant/{key.split('/', 1)[1]}"
    if key.startswith("optim/"):
        return f"12_optimization/optim/{key.split('/', 1)[1]}"
    if key.startswith("artifact/"):
        rest = key.split("/", 1)[1]
        if rest in {"under_size_limit", "within_limit"}:
            return "11_artifact_size/within_limit"
        if rest in {"quantized_total_bytes", "compressed_total_bytes", "probe_total_bytes"}:
            return f"11_artifact_size/{rest}"
        if rest == "size_margin_bytes":
            return "11_artifact_size/size_margin_bytes"
        if rest == "size_limit_bytes":
            return "11_artifact_size/size_limit_bytes"
        if rest == "int8_zlib_total_bytes":
            return "11_artifact_size/int8_zlib_total_bytes"
        if rest == "int8_zlib_model_bytes":
            return "11_artifact_size/int8_zlib_model_bytes"
        if rest == "raw_total_bytes":
            return "11_artifact_size/raw_total_bytes"
        if rest == "raw_model_bytes":
            return "11_artifact_size/raw_model_bytes"
        return f"11_artifact_size/artifact/{rest}"
    if key.startswith("model/"):
        return f"11_artifact_size/{key}"
    if key.startswith("final/"):
        rest = key.split("/", 1)[1]
        if rest == "int8_zlib_roundtrip_bpb":
            return "11_artifact_size/int8_roundtrip_bpb"
        if rest == "int8_zlib_roundtrip_loss":
            return "11_artifact_size/int8_roundtrip_loss"
        if rest == "int8_zlib_roundtrip_eval_ms":
            return "11_artifact_size/int8_roundtrip_eval_ms"
        return f"11_artifact_size/final/{rest}"
    if key.startswith(("phase/", "adaptive/", "advanced_control/", "eval/")):
        return f"14_phase_adaptive/{key}"
    if key.startswith(("time/", "progress/", "trainer/")):
        return f"12_optimization/{key}"
    if key.startswith("system/"):
        return f"13_system/{key.split('/', 1)[1]}"
    if key.startswith("analysis_control/"):
        return f"15_analysis_media/control/{key.split('/', 1)[1]}"
    if key.startswith(("analysis/", "reasoning_simplex/")):
        return f"15_analysis_media/{key}"
    if key.startswith(("metrics_status/", "diagnostics/")):
        return f"16_status/{key}"
    if key.startswith(("topology/",)):
        return f"07_topology_geometry/{key}"
    if key.startswith(("toric/", "tropical/", "bgg_category_o/", "category_o/", "koszul/", "slepian_pollak/")):
        return f"08_toric_tropical_bgg/{key}"
    if key.startswith(("checkpoint/", "trigger/", "audit/", "publish/")):
        return f"12_optimization/{key}"
    return None


def primary_metric_aliases(payload: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Return the compact scorecard aliases for a W&B payload."""

    out: OrderedDict[str, Any] = OrderedDict()
    if STEP_METRIC in payload:
        out[STEP_METRIC] = payload[STEP_METRIC]

    _add_first(
        out,
        payload,
        "00_primary/oai_bpb",
        ("oai_competition/bpb", "openai_parameter_golf/bpb", "fineweb/val_bpb", "bpb/oai_competition"),
    )
    _add_first(
        out,
        payload,
        "00_primary/oai_best_bpb",
        ("oai_competition/best_bpb", "openai_parameter_golf/best_bpb", "fineweb/best_val_bpb", "bpb/best"),
    )
    _add_first(
        out,
        payload,
        "00_primary/oai_test_time_scaled_bpb",
        ("oai_competition/test_time_scaled_bpb", "competition/oai_test_time_scaled_bpb", "bpb/oai_competition_test_time_scaled"),
    )
    _add_first(out, payload, "00_primary/train_bpb", ("train/bpb", "fineweb/train_bpb", "bpb/train"))
    _add_first(out, payload, "00_primary/validation_bpb", ("val/bpb", "fineweb/val_bpb", "bpb/val"))
    _add_first(out, payload, "00_primary/train_loss", ("train/loss", "fineweb/train_loss"))
    _add_first(out, payload, "00_primary/validation_loss", ("val/loss", "fineweb/val_loss"))
    _add_first(out, payload, "00_primary/int8_roundtrip_bpb", ("final/int8_zlib_roundtrip_bpb",))
    _add_first(out, payload, "00_primary/learning_rate", ("train/lr", "optim/token_lr", "optim/matrix_lr"))
    _add_first(out, payload, "00_primary/grad_norm", ("train/grad_norm", "optim/grad_norm"))
    _add_first(out, payload, "00_primary/nonfinite_update_skip", ("train/nonfinite_update_skip",))
    _add_first(out, payload, "00_primary/nonfinite_microbatch_skip_count", ("train/nonfinite_microbatch_skip_count",))
    _add_first(out, payload, "00_primary/nonfinite_ce_loss_count", ("train/nonfinite_ce_loss_count",))
    _add_first(out, payload, "00_primary/nonfinite_aux_loss_count", ("train/nonfinite_aux_loss_count",))
    _add_first(out, payload, "00_primary/nonfinite_param_count", ("train/nonfinite_param_count",))
    _add_first(out, payload, "00_primary/nonfinite_grad_count", ("train/nonfinite_grad_count",))
    _add_first(out, payload, "00_primary/max_abs_param", ("optim/max_abs_param",))
    _add_first(out, payload, "00_primary/max_abs_grad", ("optim/max_abs_grad",))
    _add_first(out, payload, "00_primary/full_dataset_active", ("data/full_curated_train_split_active",))
    _add_first(out, payload, "00_primary/fineweb_mix_ratio", ("data/fineweb_mix_ratio", "fineweb_calibration/mix_ratio"))
    _add_first(out, payload, "00_primary/gflownet_loss", ("train/gflownet_loss",))
    _add_first(out, payload, "00_primary/graphcg_loss", ("train/graphcg_loss",))
    _add_first(out, payload, "00_primary/toric_bgg_loss", ("train/toric_bgg_loss",))
    _add_first(out, payload, "00_primary/koszul_persistence_loss", ("train/koszul_persistence_loss",))
    _add_first(out, payload, "00_primary/toric_cca_topology_loss", ("advanced/toric_cca_topology_loss",))
    _add_first(
        out,
        payload,
        "00_primary/cca_symbolic_resolution_loss",
        ("advanced/toric_cca_symbolic_resolution_loss",),
    )
    _add_first(
        out,
        payload,
        "00_primary/cca_buchsbaum_eisenbud_multiplier_residual",
        ("advanced/toric_cca_koszul_buchsbaum_eisenbud_multiplier_residual",),
    )
    _add_first(out, payload, "00_primary/slepian_pollak_loss", ("train/slepian_pollak_loss",))
    _add_first(out, payload, "00_primary/trajectory_memory_loss", ("train/trajectory_memory_loss",))
    _add_first(out, payload, "00_primary/got_dag_loss", ("train/got_dag_loss",))
    _add_first(out, payload, "00_primary/got_dag_branch_count", ("train/got_dag_branch_count",))
    _add_first(out, payload, "00_primary/got_dag_merge_count", ("train/got_dag_merge_count",))
    _add_first(out, payload, "00_primary/trajectory_memory_dag_similarity", ("train/trajectory_memory_dag_similarity",))
    _add_first(out, payload, "00_primary/derived_category_loss", ("train/derived_category_loss",))
    _add_first(out, payload, "00_primary/derived_category_chain_map_residual", ("train/derived_category_chain_map_residual",))
    _add_first(out, payload, "00_primary/derived_category_mapping_cone_residual", ("train/derived_category_mapping_cone_residual",))
    _add_first(out, payload, "00_primary/trajectory_memory_derived_similarity", ("train/trajectory_memory_derived_similarity",))
    _add_first(out, payload, "00_primary/complexity_prediction_target_ncd", ("complexity/val/prediction_target_ncd_lzma_mean",))
    _add_first(out, payload, "00_primary/vram_allocated_gb", ("system/vram_allocated_gb",))
    _add_first(out, payload, "00_primary/artifact_within_limit", ("artifact/within_limit", "artifact/under_size_limit"))
    _add_first(out, payload, "00_primary/artifact_probe_failed", ("artifact/probe_failed",))
    _add_first(
        out,
        payload,
        "00_primary/polarquant_kv_cache_compression_ratio",
        ("polarquant/estimated_compression_ratio",),
    )
    _add_first(
        out,
        payload,
        "00_primary/polarquant_kv_cache_mb",
        ("polarquant/estimated_polarquant_kv_cache_mb",),
    )
    _add_first(out, payload, "00_primary/polarquant_saved_mb", ("polarquant/estimated_saved_mb",))

    bytes_value = _first_numeric(payload, ("artifact/initial_bytes", "artifact/final_bytes", "artifact/raw_total_bytes"))
    if bytes_value is not None:
        out["00_primary/artifact_mb"] = bytes_value / 1_000_000.0
    quantized_bytes_value = _first_numeric(
        payload,
        ("artifact/quantized_total_bytes", "artifact/compressed_total_bytes", "artifact/probe_total_bytes"),
    )
    if quantized_bytes_value is not None:
        out["00_primary/artifact_quantized_mb"] = quantized_bytes_value / 1_000_000.0
    artifact_margin = _first_numeric(payload, ("artifact/size_margin_bytes",))
    if artifact_margin is not None:
        out["00_primary/artifact_margin_mb"] = artifact_margin / 1_000_000.0
    int8_bytes_value = _first_numeric(payload, ("artifact/int8_zlib_total_bytes",))
    if int8_bytes_value is not None:
        out["00_primary/artifact_int8_zlib_mb"] = int8_bytes_value / 1_000_000.0

    target = _first_numeric(payload, ("fineweb/target_bpb", "openai_parameter_golf/target_bpb", "bpb/target"))
    oai_bpb = out.get("00_primary/oai_bpb")
    if target is not None:
        out["00_primary/target_bpb"] = target
        if _is_number(oai_bpb):
            out["00_primary/oai_gap_to_target"] = float(oai_bpb) - target
            out["00_primary/oai_target_reached"] = float(float(oai_bpb) <= target)
    return out


def organize_wandb_payload(payload: Mapping[str, Any], *, include_raw: bool | str | None = None) -> OrderedDict[str, Any]:
    """Add visible ordered aliases to a raw W&B payload.

    By default only canonical aliases are emitted.  Set
    ``TORICGT_WANDB_RAW_MODE=raw`` to retain old metric names, or
    ``TORICGT_WANDB_RAW_MODE=legacy`` to log them under ``99_legacy/*``.
    """

    out = primary_metric_aliases(payload)
    for key, value in payload.items():
        alias = _category_alias(str(key))
        if alias:
            out.setdefault(_clean_name(alias), value)
    if include_raw is None:
        raw_mode = _raw_mode_from_env()
    elif isinstance(include_raw, str):
        raw_mode = include_raw.strip().lower()
    else:
        raw_mode = "raw" if include_raw else "off"
    if raw_mode == "raw":
        for key, value in payload.items():
            out.setdefault(str(key), value)
    elif raw_mode == "legacy":
        for key, value in payload.items():
            out.setdefault(_clean_name(f"99_legacy/{key}"), value)
    return out


def configure_wandb_metrics(wandb_module: Any, *, step_metric: str = STEP_METRIC) -> None:
    """Define W&B metric visibility and summaries without failing training."""

    def define(name: str, **kwargs: Any) -> None:
        try:
            wandb_module.define_metric(name, **kwargs)
        except Exception:
            pass

    define(step_metric)
    define("*", step_metric=step_metric, hidden=True)
    for prefix in VISIBLE_METRIC_PREFIXES:
        define(f"{prefix}/*", step_metric=step_metric, hidden=False)
    for pattern in RAW_HIDDEN_PATTERNS:
        define(pattern, step_metric=step_metric, hidden=True)
    for pattern in MINIMIZE_PATTERNS:
        define(pattern, step_metric=step_metric, hidden=False, summary="min", goal="minimize")
    for pattern in MAXIMIZE_PATTERNS:
        define(pattern, step_metric=step_metric, hidden=False, summary="max", goal="maximize")


def update_wandb_summary(run: Any, payload: Mapping[str, Any]) -> None:
    """Update a W&B summary with canonical aliases plus raw compatibility keys."""

    try:
        run.summary.update(organize_wandb_payload(payload))
    except Exception:
        try:
            run.summary.update(dict(payload))
        except Exception:
            pass
