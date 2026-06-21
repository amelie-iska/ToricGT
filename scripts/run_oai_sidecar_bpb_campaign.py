#!/usr/bin/env python3
"""Run threshold-gated OAI-baseline ToricGT sidecar BPB campaigns.

This supervisor intentionally restarts every attempt from step 0.  The loop is:

1. launch the OAI Parameter-Golf baseline with the ToricGT sidecar active;
2. stop naturally at the configured analysis cadence, 1.5K steps by default;
3. parse BPB, artifact size, and sidecar metrics;
4. write a report under training_notes;
5. choose a changed hyperparameter profile for the next fresh attempt.

It does not resume checkpoints between attempts.  Checkpoints are only evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from adaptive_bpb_annealing import build_adaptive_decision


VAL_RE = re.compile(r"step:(?P<step>\d+)/(?P<total>\d+) val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+)")
TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+) train_loss:(?P<loss>[0-9.]+)"
    r"(?: train_bpb:(?P<train_bpb>[0-9.eE+-]+) train_bpt:(?P<train_bpt>[0-9.eE+-]+))?.*?"
    r"(?: graph_lm_loss:(?P<graph_lm_loss>[0-9.eE+-]+) graph_lm_bpb:(?P<graph_lm_bpb>[0-9.eE+-]+) "
    r"graph_lm_w:(?P<graph_lm_weight>[0-9.eE+-]+))?.*?"
    r"(?: oai_gfn:(?P<oai_gflownet_loss>[0-9.eE+-]+) gfn_H:(?P<oai_gflownet_entropy>[0-9.eE+-]+) "
    r"gfn_R:(?P<oai_gflownet_reward>[0-9.eE+-]+))?.*?"
    r"(?: oai_fot:(?P<oai_fot_loss>[0-9.eE+-]+) fot_H:(?P<oai_fot_entropy>[0-9.eE+-]+) "
    r"fot_R:(?P<oai_fot_reward>[0-9.eE+-]+) fot_div:(?P<oai_fot_diversity>[0-9.eE+-]+))?.*?"
    r"(?: mtp:(?P<oai_mtp_loss>[0-9.eE+-]+) mtp_w:(?P<oai_mtp_weight>[0-9.eE+-]+))?.*?"
    r"sidecar_loss:(?P<sidecar>[0-9.eE+-]+) graphcg:(?P<graphcg>[0-9.eE+-]+) "
    r"analogy:(?P<analogy>[0-9.eE+-]+) tokengt_graph:(?P<tokengt>[0-9.eE+-]+) "
    r"memory:(?P<memory>[0-9.eE+-]+)"
    r"(?: toric:(?P<toric>[0-9.eE+-]+) (?:vb1d|vb):(?P<vector_bundle_1d_cone>[0-9.eE+-]+) "
    r"bgg:(?P<bgg>[0-9.eE+-]+) koszul:(?P<koszul>[0-9.eE+-]+) "
    r"cca:(?P<combinatorial_toric>[0-9.eE+-]+)(?: derived:(?P<derived_signature>[0-9.eE+-]+))?)?"
)
FINAL_RE = re.compile(r"final_int8_zlib_roundtrip_exact val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+)")
SIZE_RE = re.compile(r"Total submission size int8\+zlib: (?P<size>\d+) bytes")
CHECKPOINT_RE = re.compile(r"checkpoint_saved:(?P<path>.*?) step:(?P<step>\d+) val_bpb:(?P<bpb>[-+0-9.eE]+|None)")
OAI_GFLOWNET_RE = re.compile(
    r"oai_gfn:(?P<oai_gflownet_loss>[0-9.eE+-]+)\s+"
    r"gfn_H:(?P<oai_gflownet_entropy>[0-9.eE+-]+)\s+"
    r"gfn_R:(?P<oai_gflownet_reward>[0-9.eE+-]+)"
)
OAI_MTP_RE = re.compile(
    r"mtp:(?P<oai_mtp_loss>[0-9.eE+-]+)\s+mtp_w:(?P<oai_mtp_weight>[0-9.eE+-]+)"
)
OAI_FOT_RE = re.compile(
    r"oai_fot:(?P<oai_fot_loss>[0-9.eE+-]+)\s+"
    r"fot_H:(?P<oai_fot_entropy>[0-9.eE+-]+)\s+"
    r"fot_R:(?P<oai_fot_reward>[0-9.eE+-]+)\s+"
    r"fot_div:(?P<oai_fot_diversity>[0-9.eE+-]+)"
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_run_id(campaign_id: str, run_index: int, profile_name: str, *, max_len: int = 120) -> str:
    """Build a W&B-safe run id/name.

    W&B rejects run names longer than 128 characters.  Keep a little margin
    because downstream wrappers may append suffixes in logs or UI labels.
    """

    stamp = utc_stamp()
    raw = f"{campaign_id}-r{run_index:03d}-{profile_name}-{stamp}"
    if len(raw) <= max_len:
        return raw
    campaign_budget = max(16, min(44, max_len // 3))
    profile_budget = max(24, max_len - campaign_budget - len(stamp) - len("-r000--") - 1)
    campaign_part = campaign_id[:campaign_budget].rstrip("-_")
    profile_part = profile_name[:profile_budget].rstrip("-_")
    shortened = f"{campaign_part}-r{run_index:03d}-{profile_part}-{stamp}"
    return shortened[:max_len].rstrip("-_")


def env_truthy(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Profile:
    name: str
    env: dict[str, str]
    rationale: str


PROFILES: list[Profile] = [
    Profile(
        "convextok2048_det_tropical_toric_bpb",
        {
            "TRAIN_BATCH_TOKENS": "786432",
            "MATRIX_LR": "0.044",
            "SCALAR_LR": "0.044",
            "TIED_EMBED_LR": "0.054",
            "WARMDOWN_ITERS": "560",
            "WARMUP_STEPS": "10",
            "TOKENGT_FIRST_CLASS_LR": "2.1e-4",
            "TOKENGT_GRAPH_RADIUS": "4",
            "TOKENGT_TOKEN_CLASS_BUCKETS": "128",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.052",
            "TOKENGT_EDGE_WEIGHT": "0.038",
            "TOKENGT_TORUS_WEIGHT": "0.014",
            "CONVEXTOK_DAG_FEATURES": "1",
            "CONVEXTOK_DAG_FEATURE_WEIGHT": "0.026",
            "CONVEXTOK_TORIC_REG_WEIGHT": "8.0e-6",
            "CONVEXTOK_TORIC_REG_TOPK": "128",
            "GRAPH_OUTPUT_FLATTENING_LR": "1.9e-4",
            "GRAPH_OUTPUT_EDGE_RADIUS": "4",
            "GRAPH_OUTPUT_NODE_WEIGHT": "0.060",
            "GRAPH_OUTPUT_EDGE_WEIGHT": "0.062",
            "GRAPH_OUTPUT_EDGE_TOKEN_WEIGHT": "0.040",
            "GRAPH_OUTPUT_SCORE_CORRECTION_WEIGHT": "0.026",
            "GRAPH_LM_LOSS_WEIGHT": "0.10",
            "GRAPH_LM_BATCH_SIZE": "2",
            "GRAPH_LM_EVERY": "4",
            "GRAPHCG_LOSS_WEIGHT": "2e-6",
            "ANALOGY_LOSS_WEIGHT": "2.0e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "4.5e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "2.0e-5",
            "TORICGT_SIDECAR_BATCH_SIZE": "4",
            "TORICGT_SIDECAR_EVERY": "4",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "4.0e-6",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "1.6e-6",
            "TORIC_BGG_LOSS_WEIGHT": "2.0e-6",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "8.0e-7",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "1.0e-6",
            "OAI_GFLOWNET_EVERY": "4",
            "OAI_GFLOWNET_MAX_SEQUENCES": "1",
            "OAI_GFLOWNET_MAX_POSITIONS": "128",
            "OAI_FOT_EVERY": "4",
            "OAI_FOT_NUM_TREES": "3",
            "OAI_FOT_MAX_DEPTH": "4",
            "OAI_FOT_BRANCHING": "3",
            "OAI_FOT_MAX_SEQUENCES": "1",
            "OAI_FOT_MAX_POSITIONS": "128",
            "OAI_MTP_EVERY": "2",
        },
        "ConvexTok-2048 Det first run: expose LP/tokenisation-DAG metadata to first-class TokenGT, use min-plus/tropical shortest-path structure as a primary graph feature, and keep BPB-dominant optimization with low nonzero toric/tropical/BGG pressure.",
    ),
    Profile(
        "convextok2048_bias_ood_probe",
        {
            "TRAIN_BATCH_TOKENS": "1048576",
            "MATRIX_LR": "0.040",
            "SCALAR_LR": "0.040",
            "TIED_EMBED_LR": "0.050",
            "WARMDOWN_ITERS": "640",
            "WARMUP_STEPS": "10",
            "TOKENGT_FIRST_CLASS_LR": "1.9e-4",
            "TOKENGT_GRAPH_RADIUS": "4",
            "TOKENGT_TOKEN_CLASS_BUCKETS": "128",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.048",
            "TOKENGT_EDGE_WEIGHT": "0.034",
            "TOKENGT_TORUS_WEIGHT": "0.014",
            "CONVEXTOK_DAG_FEATURES": "1",
            "CONVEXTOK_DAG_FEATURE_WEIGHT": "0.024",
            "CONVEXTOK_TORIC_REG_WEIGHT": "6.0e-6",
            "CONVEXTOK_TORIC_REG_TOPK": "128",
            "GRAPH_OUTPUT_FLATTENING_LR": "1.7e-4",
            "GRAPH_OUTPUT_EDGE_RADIUS": "4",
            "GRAPH_OUTPUT_NODE_WEIGHT": "0.055",
            "GRAPH_OUTPUT_EDGE_WEIGHT": "0.058",
            "GRAPH_OUTPUT_EDGE_TOKEN_WEIGHT": "0.038",
            "GRAPH_OUTPUT_SCORE_CORRECTION_WEIGHT": "0.024",
            "GRAPH_LM_LOSS_WEIGHT": "0.09",
            "GRAPHCG_LOSS_WEIGHT": "2e-6",
            "ANALOGY_LOSS_WEIGHT": "2.2e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "4.0e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "2.4e-5",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "3.5e-6",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "1.5e-6",
            "TORIC_BGG_LOSS_WEIGHT": "2.2e-6",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "8.0e-7",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "1.0e-6",
        },
        "ConvexTok-2048 Bias profile: test the more length-normalized tokenizer rounding as an OOD-stability/early-BPB probe while keeping the same first-class tropical tokenisation-DAG feature path.",
    ),
    Profile(
        "gate1500_high_batch_toric_bgg_memory",
        {
            "TRAIN_BATCH_TOKENS": "1048576",
            "MATRIX_LR": "0.042",
            "SCALAR_LR": "0.042",
            "TIED_EMBED_LR": "0.052",
            "WARMDOWN_ITERS": "650",
            "WARMUP_STEPS": "10",
            "TOKENGT_FIRST_CLASS_LR": "2.0e-4",
            "TOKENGT_GRAPH_RADIUS": "4",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.050",
            "TOKENGT_EDGE_WEIGHT": "0.036",
            "TOKENGT_TORUS_WEIGHT": "0.012",
            "GRAPH_OUTPUT_FLATTENING_LR": "1.8e-4",
            "GRAPH_OUTPUT_EDGE_RADIUS": "4",
            "GRAPH_OUTPUT_NODE_WEIGHT": "0.060",
            "GRAPH_OUTPUT_EDGE_WEIGHT": "0.060",
            "TORICGT_SIDECAR_LR": "1.8e-4",
            "GRAPH_LM_LOSS_WEIGHT": "0.12",
            "GRAPHCG_LOSS_WEIGHT": "3e-6",
            "ANALOGY_LOSS_WEIGHT": "2.2e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "4.5e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "2.5e-5",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "3.0e-6",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "1.6e-6",
            "TORIC_BGG_LOSS_WEIGHT": "2.5e-6",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "1.0e-6",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "1.0e-6",
        },
        "1.5K experimental profile: raise training-token microbatch toward ~20GB VRAM, increase early main-model BPB pressure, keep GraphCG full-rank but light, and emphasize TokenGT graphification, toric/tropical geometry, Toric BGG, analogy, and trajectory memory based on family-specific evidence.",
    ),
    Profile(
        "gate1500_fast_main_lr_light_graphcg",
        {
            "TRAIN_BATCH_TOKENS": "917504",
            "MATRIX_LR": "0.048",
            "SCALAR_LR": "0.048",
            "TIED_EMBED_LR": "0.058",
            "WARMDOWN_ITERS": "550",
            "WARMUP_STEPS": "10",
            "TOKENGT_FIRST_CLASS_LR": "1.8e-4",
            "TOKENGT_GRAPH_RADIUS": "4",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.040",
            "TOKENGT_EDGE_WEIGHT": "0.028",
            "TOKENGT_TORUS_WEIGHT": "0.008",
            "GRAPH_OUTPUT_FLATTENING_LR": "1.5e-4",
            "GRAPH_OUTPUT_EDGE_RADIUS": "4",
            "GRAPH_OUTPUT_NODE_WEIGHT": "0.040",
            "GRAPH_OUTPUT_EDGE_WEIGHT": "0.040",
            "TORICGT_SIDECAR_LR": "1.5e-4",
            "GRAPH_LM_LOSS_WEIGHT": "0.07",
            "GRAPHCG_LOSS_WEIGHT": "2e-6",
            "ANALOGY_LOSS_WEIGHT": "1.6e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "3.5e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "1.8e-5",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "2.0e-6",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "1.0e-6",
            "TORIC_BGG_LOSS_WEIGHT": "1.5e-6",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "7.5e-7",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "7.5e-7",
        },
        "Aggressive 1.5K BPB profile for under-fast early curves: larger microbatch, higher matrix/scalar/tied-embedding learning rates, light GraphCG pressure, and moderate advanced losses to test whether the BPB bottleneck is under-optimization rather than structure.",
    ),
    Profile(
        "gate1500_fast_main_lr_aux_conflict_recovery",
        {
            "TRAIN_BATCH_TOKENS": "983040",
            "MATRIX_LR": "0.048",
            "SCALAR_LR": "0.048",
            "TIED_EMBED_LR": "0.058",
            "WARMDOWN_ITERS": "450",
            "WARMUP_STEPS": "20",
            "TOKENGT_FIRST_CLASS_LR": "1.6e-4",
            "TOKENGT_GRAPH_RADIUS": "4",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.045",
            "TOKENGT_EDGE_WEIGHT": "0.030",
            "TOKENGT_TORUS_WEIGHT": "0.010",
            "TOKENGT_IDENTIFIER_DIM": "24",
            "TOKENGT_IDENTIFIER_WEIGHT": "0.014",
            "TOKENGT_ENDPOINT_WEIGHT": "0.018",
            "TOKENGT_EDGE_TOKEN_WEIGHT": "0.016",
            "GRAPH_OUTPUT_FLATTENING_LR": "1.6e-4",
            "GRAPH_OUTPUT_EDGE_RADIUS": "4",
            "GRAPH_OUTPUT_NODE_WEIGHT": "0.050",
            "GRAPH_OUTPUT_EDGE_WEIGHT": "0.050",
            "GRAPH_OUTPUT_EDGE_TOKEN_WEIGHT": "0.035",
            "GRAPH_OUTPUT_SCORE_CORRECTION_WEIGHT": "0.018",
            "GRAPH_LM_LOSS_WEIGHT_START": "0.005",
            "GRAPH_LM_LOSS_WEIGHT": "0.060",
            "GRAPH_LM_WARMUP_STEPS": "800",
            "GRAPH_LM_HOLD_STEPS": "120",
            "OAI_GFLOWNET_LOSS_WEIGHT": "1.2e-5",
            "OAI_GFLOWNET_LR": "1.5e-4",
            "OAI_GFLOWNET_ENTROPY_WEIGHT": "5e-6",
            "OAI_GFLOWNET_ENTROPY_TARGET": "1.0",
            "OAI_MTP_LOSS_WEIGHT": "0.0012",
            "OAI_MTP_EVERY": "2",
            "OAI_MTP_MAX_SEQUENCES": "1",
            "GRAPHCG_LOSS_WEIGHT": "1.0e-6",
            "GRAPHCG_BPB_ORTHOGONAL_WEIGHT": "0.040",
            "GRAPHCG_COVARIANCE_CONFLICT_DAMPING": "0.75",
            "ANALOGY_LOSS_WEIGHT": "2.6e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "5.0e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "1.2e-5",
            "RETRIEVAL_GATE_MIN": "0.05",
            "RETRIEVAL_GATE_CENTER": "0.35",
            "RETRIEVAL_GATE_SOFTNESS": "0.10",
            "SIDECAR_UNCERTAINTY_ALPHA": "0.20",
            "SIDECAR_UNCERTAINTY_MAX": "1.4",
            "TORICGT_SIDECAR_LOSS_WEIGHT_START": "0.02",
            "TORICGT_SIDECAR_WARMUP_STEPS": "700",
            "TORICGT_SIDECAR_HOLD_STEPS": "150",
            "AUX_GRAD_ALIGNED_BOOST": "0.8",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "3.2e-6",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "1.5e-6",
            "TORIC_BGG_LOSS_WEIGHT": "2.0e-6",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "3.0e-7",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "8.0e-7",
        },
        "Artifact-reviewed conflict-recovery profile: keep first-class causal graphification active and increase the local radius to 4, preserve toric/vector-bundle/BGG pressure at low nonzero values, but slow graph-LM ramp, reduce MTP cadence, lower GraphCG/memory pressure, and increase GFlowNet entropy discipline after the 1.5K artifacts showed auxiliary-gradient conflict above the BPB target.",
    ),
    Profile(
        "gate1500_structural_toric_heavy",
        {
            "TRAIN_BATCH_TOKENS": "917504",
            "MATRIX_LR": "0.038",
            "SCALAR_LR": "0.038",
            "TIED_EMBED_LR": "0.048",
            "WARMDOWN_ITERS": "700",
            "WARMUP_STEPS": "10",
            "TOKENGT_FIRST_CLASS_LR": "2.4e-4",
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.070",
            "TOKENGT_EDGE_WEIGHT": "0.050",
            "TOKENGT_TORUS_WEIGHT": "0.018",
            "GRAPH_OUTPUT_FLATTENING_LR": "2.2e-4",
            "GRAPH_OUTPUT_EDGE_RADIUS": "3",
            "GRAPH_OUTPUT_NODE_WEIGHT": "0.070",
            "GRAPH_OUTPUT_EDGE_WEIGHT": "0.075",
            "TORICGT_SIDECAR_LR": "2.0e-4",
            "GRAPH_LM_LOSS_WEIGHT": "0.10",
            "GRAPHCG_LOSS_WEIGHT": "3e-6",
            "ANALOGY_LOSS_WEIGHT": "2.0e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "6.0e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "2.0e-5",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "5.0e-6",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "2.0e-6",
            "TORIC_BGG_LOSS_WEIGHT": "2.0e-6",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "1.0e-6",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "1.2e-6",
        },
        "Structural/tropical experiment: test whether stronger first-class TokenGT graphification plus toric fan/binomial/moment pressure can improve early BPB by giving the baseline better local graph inductive bias.",
    ),
    Profile(
        "gate1500_bgg_memory_probe",
        {
            "TRAIN_BATCH_TOKENS": "983040",
            "MATRIX_LR": "0.040",
            "SCALAR_LR": "0.040",
            "TIED_EMBED_LR": "0.050",
            "WARMDOWN_ITERS": "650",
            "WARMUP_STEPS": "10",
            "TOKENGT_FIRST_CLASS_LR": "1.6e-4",
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.038",
            "TOKENGT_EDGE_WEIGHT": "0.026",
            "TOKENGT_TORUS_WEIGHT": "0.010",
            "GRAPH_OUTPUT_FLATTENING_LR": "1.7e-4",
            "GRAPH_OUTPUT_EDGE_RADIUS": "3",
            "GRAPH_OUTPUT_NODE_WEIGHT": "0.050",
            "GRAPH_OUTPUT_EDGE_WEIGHT": "0.055",
            "TORICGT_SIDECAR_LR": "2.2e-4",
            "GRAPH_LM_LOSS_WEIGHT": "0.12",
            "GRAPHCG_LOSS_WEIGHT": "3e-6",
            "ANALOGY_LOSS_WEIGHT": "3.0e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "3.5e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "3.5e-5",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "2.0e-6",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "1.0e-6",
            "TORIC_BGG_LOSS_WEIGHT": "4.0e-6",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "1.0e-6",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "1.0e-6",
        },
        "BGG/memory experiment: emphasize resolution consistency, standard-filtration structure, analogy, and trajectory-memory retrieval while keeping toric geometry active, to test whether better reusable reasoning state improves byte prediction early.",
    ),
    Profile(
        "gate2500_experimental_family_selective",
        {
            "TRAIN_BATCH_TOKENS": "524288",
            "MATRIX_LR": "0.035",
            "SCALAR_LR": "0.035",
            "TIED_EMBED_LR": "0.043",
            "WARMDOWN_ITERS": "1850",
            "WARMUP_STEPS": "20",
            "TOKENGT_FIRST_CLASS_LR": "1.7e-4",
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.040",
            "TOKENGT_EDGE_WEIGHT": "0.030",
            "TOKENGT_TORUS_WEIGHT": "0.010",
            "TORICGT_SIDECAR_LR": "1.6e-4",
            "GRAPHCG_LOSS_WEIGHT": "4e-6",
            "ANALOGY_LOSS_WEIGHT": "2.0e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "4.0e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "2.0e-5",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "2.5e-6",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "1.5e-6",
            "TORIC_BGG_LOSS_WEIGHT": "2.0e-6",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "7.5e-7",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "8e-7",
        },
        "2.5K gate profile kept as a slower fallback after nuanced metric review.",
    ),
    Profile(
        "gate2500_fast_bpb_low_aux",
        {
            "TRAIN_BATCH_TOKENS": "524288",
            "MATRIX_LR": "0.034",
            "SCALAR_LR": "0.034",
            "TIED_EMBED_LR": "0.042",
            "WARMDOWN_ITERS": "1800",
            "WARMUP_STEPS": "20",
            "TOKENGT_FIRST_CLASS_LR": "1.2e-4",
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.030",
            "TOKENGT_EDGE_WEIGHT": "0.020",
            "TOKENGT_TORUS_WEIGHT": "0.008",
            "TORICGT_SIDECAR_LR": "1.2e-4",
            "GRAPHCG_LOSS_WEIGHT": "8e-6",
            "ANALOGY_LOSS_WEIGHT": "1.0e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "2.0e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "1.0e-5",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "5e-7",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "5e-7",
            "TORIC_BGG_LOSS_WEIGHT": "5e-7",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "5e-7",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "5e-7",
        },
        "2.5K fallback profile: keep all metrics visible, reduce auxiliary gradient pressure, and spend more optimization budget on the OAI FineWeb BPB objective.",
    ),
    Profile(
        "large_batch_lower_lr_sidecar_light",
        {
            "TRAIN_BATCH_TOKENS": "524288",
            "MATRIX_LR": "0.025",
            "SCALAR_LR": "0.025",
            "TIED_EMBED_LR": "0.035",
            "WARMDOWN_ITERS": "3500",
            "WARMUP_STEPS": "20",
            "TOKENGT_FIRST_CLASS_LR": "1.0e-4",
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.025",
            "TOKENGT_EDGE_WEIGHT": "0.018",
            "TOKENGT_TORUS_WEIGHT": "0.006",
            "TORICGT_SIDECAR_LR": "1.5e-4",
            "GRAPHCG_LOSS_WEIGHT": "1.5e-5",
            "ANALOGY_LOSS_WEIGHT": "1.5e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "3e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "1.5e-5",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "5e-7",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "5e-7",
            "TORIC_BGG_LOSS_WEIGHT": "5e-7",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "5e-7",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "5e-7",
        },
        "Lower-risk BPB control with first-class FineWeb graphification enabled, all advanced ToricGT metrics active, and minimal nonzero advanced training pressure.",
    ),
    Profile(
        "large_batch_modest_lr_all_metrics_light_train",
        {
            "TRAIN_BATCH_TOKENS": "524288",
            "MATRIX_LR": "0.028",
            "SCALAR_LR": "0.028",
            "TIED_EMBED_LR": "0.037",
            "WARMDOWN_ITERS": "3600",
            "WARMUP_STEPS": "20",
            "TOKENGT_FIRST_CLASS_LR": "1.4e-4",
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.035",
            "TOKENGT_EDGE_WEIGHT": "0.025",
            "TOKENGT_TORUS_WEIGHT": "0.010",
            "TORICGT_SIDECAR_LR": "1.5e-4",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "GRAPHCG_LOSS_WEIGHT": "1.5e-5",
            "ANALOGY_LOSS_WEIGHT": "1.5e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "3e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "1.5e-5",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "2e-6",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "1e-6",
            "TORIC_BGG_LOSS_WEIGHT": "1e-6",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "1e-6",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "1e-6",
        },
        "Codex review of the first attempt recommended a modest LR bump over the light sidecar profile while keeping all advanced metrics active; this is the default graphified restart profile.",
    ),
    Profile(
        "large_batch_modest_lr_geometry_bgg_subset",
        {
            "TRAIN_BATCH_TOKENS": "524288",
            "MATRIX_LR": "0.028",
            "SCALAR_LR": "0.028",
            "TIED_EMBED_LR": "0.037",
            "WARMDOWN_ITERS": "3600",
            "WARMUP_STEPS": "20",
            "TOKENGT_FIRST_CLASS_LR": "1.5e-4",
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.040",
            "TOKENGT_EDGE_WEIGHT": "0.025",
            "TOKENGT_TORUS_WEIGHT": "0.012",
            "TORICGT_SIDECAR_LR": "1.6e-4",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "GRAPHCG_LOSS_WEIGHT": "1.5e-5",
            "ANALOGY_LOSS_WEIGHT": "1.25e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "2.5e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "1.25e-5",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "4e-6",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "0",
            "TORIC_BGG_LOSS_WEIGHT": "3e-6",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "0",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "1e-6",
        },
        "Nontrivial toric-geometry plus finite category-O/BGG subset when BPB plateaus but sidecar metrics look stable.",
    ),
    Profile(
        "large_batch_topology_memory_subset",
        {
            "TRAIN_BATCH_TOKENS": "524288",
            "MATRIX_LR": "0.024",
            "SCALAR_LR": "0.024",
            "TIED_EMBED_LR": "0.034",
            "WARMDOWN_ITERS": "4000",
            "WARMUP_STEPS": "20",
            "TOKENGT_FIRST_CLASS_LR": "1.2e-4",
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.030",
            "TOKENGT_EDGE_WEIGHT": "0.030",
            "TOKENGT_TORUS_WEIGHT": "0.010",
            "TORICGT_SIDECAR_LR": "1.4e-4",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "GRAPHCG_LOSS_WEIGHT": "1.5e-5",
            "ANALOGY_LOSS_WEIGHT": "1.5e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "2.5e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "2.2e-5",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "0",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "1e-6",
            "TORIC_BGG_LOSS_WEIGHT": "0",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "3e-6",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "3e-6",
        },
        "Topology, persistent-homology, memory-retrieval, vector-bundle/sheaf subset for analogy/memory pressure under a lower BPB LR.",
    ),
    Profile(
        "large_batch_all_advanced_low_weight",
        {
            "TRAIN_BATCH_TOKENS": "524288",
            "MATRIX_LR": "0.026",
            "SCALAR_LR": "0.026",
            "TIED_EMBED_LR": "0.036",
            "WARMDOWN_ITERS": "3800",
            "WARMUP_STEPS": "20",
            "TOKENGT_FIRST_CLASS_LR": "1.4e-4",
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.040",
            "TOKENGT_EDGE_WEIGHT": "0.030",
            "TOKENGT_TORUS_WEIGHT": "0.012",
            "TORICGT_SIDECAR_LR": "1.5e-4",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "GRAPHCG_LOSS_WEIGHT": "1.5e-5",
            "ANALOGY_LOSS_WEIGHT": "1.5e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "3e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "1.5e-5",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "2e-6",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "2e-6",
            "TORIC_BGG_LOSS_WEIGHT": "2e-6",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "2e-6",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "2e-6",
        },
        "All advanced ToricGT families train at low weight while BPB remains the dominant objective.",
    ),
    Profile(
        "artifact_margin_conservative_aux",
        {
            "TRAIN_BATCH_TOKENS": "524288",
            "MATRIX_LR": "0.022",
            "SCALAR_LR": "0.022",
            "TIED_EMBED_LR": "0.032",
            "WARMDOWN_ITERS": "4200",
            "WARMUP_STEPS": "20",
            "TOKENGT_FIRST_CLASS_LR": "8.0e-5",
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.020",
            "TOKENGT_EDGE_WEIGHT": "0.015",
            "TOKENGT_TORUS_WEIGHT": "0.005",
            "TORICGT_SIDECAR_LR": "1.2e-4",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "GRAPHCG_LOSS_WEIGHT": "1.0e-5",
            "ANALOGY_LOSS_WEIGHT": "1.0e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "2.0e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "1.0e-5",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "5e-7",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "5e-7",
            "TORIC_BGG_LOSS_WEIGHT": "5e-7",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "5e-7",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "5e-7",
        },
        "Conservative all-metric run when artifact margin or BPB stability becomes the dominant constraint.",
    ),
    Profile(
        "large_batch_low_lr_longer_warmdown",
        {
            "TRAIN_BATCH_TOKENS": "524288",
            "MATRIX_LR": "0.020",
            "SCALAR_LR": "0.020",
            "TIED_EMBED_LR": "0.030",
            "WARMDOWN_ITERS": "4200",
            "WARMUP_STEPS": "20",
            "TOKENGT_FIRST_CLASS_LR": "8.0e-5",
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.020",
            "TOKENGT_EDGE_WEIGHT": "0.015",
            "TOKENGT_TORUS_WEIGHT": "0.005",
            "TORICGT_SIDECAR_LR": "1.2e-4",
            "GRAPHCG_LOSS_WEIGHT": "1.0e-5",
            "ANALOGY_LOSS_WEIGHT": "1.0e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "2.0e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "1.0e-5",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "5e-7",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "5e-7",
            "TORIC_BGG_LOSS_WEIGHT": "5e-7",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "5e-7",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "5e-7",
        },
        "If BPB remains high, reduce optimization noise and keep auxiliary pressure smaller.",
    ),
    Profile(
        "large_batch_mid_lr_sidecar_medium",
        {
            "TRAIN_BATCH_TOKENS": "524288",
            "MATRIX_LR": "0.032",
            "SCALAR_LR": "0.032",
            "TIED_EMBED_LR": "0.040",
            "WARMDOWN_ITERS": "3600",
            "WARMUP_STEPS": "20",
            "TOKENGT_FIRST_CLASS_LR": "1.8e-4",
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.045",
            "TOKENGT_EDGE_WEIGHT": "0.030",
            "TOKENGT_TORUS_WEIGHT": "0.012",
            "TORICGT_SIDECAR_LR": "2.0e-4",
            "GRAPHCG_LOSS_WEIGHT": "2.0e-5",
            "ANALOGY_LOSS_WEIGHT": "2.0e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "4.0e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "2.0e-5",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "2e-6",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "1e-6",
            "TORIC_BGG_LOSS_WEIGHT": "2e-6",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "1e-6",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "1e-6",
        },
        "If lower LR underfits by the gate step, use a middle LR while preserving the larger batch.",
    ),
    Profile(
        "quarter_batch_conservative_lr",
        {
            "TRAIN_BATCH_TOKENS": "262144",
            "MATRIX_LR": "0.026",
            "SCALAR_LR": "0.026",
            "TIED_EMBED_LR": "0.036",
            "WARMDOWN_ITERS": "3800",
            "WARMUP_STEPS": "20",
            "TOKENGT_FIRST_CLASS_LR": "1.2e-4",
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.030",
            "TOKENGT_EDGE_WEIGHT": "0.022",
            "TOKENGT_TORUS_WEIGHT": "0.008",
            "TORICGT_SIDECAR_LR": "1.5e-4",
            "GRAPHCG_LOSS_WEIGHT": "1.0e-5",
            "ANALOGY_LOSS_WEIGHT": "1.0e-5",
            "TOKENGT_GRAPH_LOSS_WEIGHT": "2.5e-5",
            "TRAJECTORY_MEMORY_LOSS_WEIGHT": "1.0e-5",
            "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
            "TORIC_GEOMETRY_LOSS_WEIGHT": "5e-7",
            "TORIC_VECTOR_BUNDLE_LOSS_WEIGHT": "5e-7",
            "TORIC_BGG_LOSS_WEIGHT": "5e-7",
            "KOSZUL_PERSISTENCE_LOSS_WEIGHT": "5e-7",
            "COMBINATORIAL_TORIC_LOSS_WEIGHT": "5e-7",
        },
        "Fallback for 24GB stability if 524288 tokens is too slow or memory-heavy.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--campaign-id", default="")
    parser.add_argument("--max-runs", type=int, default=25)
    parser.add_argument("--target-bpb", type=float, default=1.19)
    parser.add_argument("--steps-per-run", type=int, default=1500)
    parser.add_argument("--profile-offset", type=int, default=0)
    parser.add_argument(
        "--prior-run-id",
        default="toricgt-oai-sidecar-bpb112-campaign-20260616T222606Z-run001-large_batch_lower_lr_sidecar_light-20260616T222607Z",
    )
    parser.add_argument(
        "--prior-log",
        default=(
            "runs/oai_sidecar/"
            "toricgt-oai-sidecar-bpb112-campaign-20260616T222606Z-run001-large_batch_lower_lr_sidecar_light-20260616T222607Z/"
            "train.log"
        ),
    )
    parser.add_argument(
        "--prior-analysis-dir",
        default="",
        help="Optional completed full-analysis directory. Its metrics.json and next_profile_decision.json seed run 1.",
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "amelie-iska-math"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "toricgt-parameter-golf"))
    parser.add_argument("--conda-bin", default=os.environ.get("CONDA_BIN", "/home/iska/miniconda3/bin/conda"))
    parser.add_argument("--conda-env", default=os.environ.get("CONDA_ENV", "tokengt"))
    parser.add_argument("--graph-data-path", default="/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/curated_hf_shards")
    parser.add_argument("--fineweb-data", default="amelie-iska/parameter-golf/data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="amelie-iska/parameter-golf/data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--matched-docs-jsonl", default="amelie-iska/parameter-golf/data/docs_selected.jsonl")
    parser.add_argument(
        "--matched-docs-parquet",
        action="append",
        default=[],
        help="Parquet glob(s) for ConvexTok regret analysis samples when matched docs JSONL is absent. May be repeated.",
    )
    parser.add_argument("--matched-docs-parquet-text-column", default="text")
    parser.add_argument("--sentencepiece-baseline-tokenizer", default="amelie-iska/parameter-golf/data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--codex-review", action="store_true", default=env_truthy("TORICGT_CODEX_REVIEW", "0"))
    parser.add_argument("--full-analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--analysis-records", type=int, default=3)
    parser.add_argument(
        "--analysis-cas-max-points",
        type=int,
        default=6,
        help="Exact Sage/Macaulay2 toric embedding audit size. Six points keeps resolutions exact and bounded.",
    )
    parser.add_argument("--analysis-cas-macaulay2-timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--analysis-retries",
        type=int,
        default=1,
        help="Retry the exact full-analysis pass this many times after a failure before strict campaign stop.",
    )
    parser.add_argument(
        "--analysis-retry-timeout-multiplier",
        type=float,
        default=2.0,
        help="Multiply the Macaulay2 timeout on each full-analysis retry. The CAS audit remains exact on the selected subset.",
    )
    parser.add_argument(
        "--followup-runs-after-meta",
        type=int,
        default=0,
        help="Deprecated follow-up budget. Defaults to 0 so campaigns run exactly --max-runs attempts.",
    )
    parser.add_argument(
        "--stop-on-target",
        action="store_true",
        default=env_truthy("TORICGT_STOP_ON_TARGET", "0"),
        help="Stop early when a run reaches target BPB. Disabled by default so the campaign can select the best out of the exact run budget.",
    )
    parser.add_argument(
        "--final-full-train-best",
        action="store_true",
        default=env_truthy("TORICGT_FINAL_FULL_TRAIN_BEST", "0"),
        help="After the exact short-run sweep completes, launch one fresh longer run using the best profile/settings from the sweep.",
    )
    parser.add_argument(
        "--final-full-train-steps",
        type=int,
        default=int(os.environ.get("TORICGT_FINAL_FULL_TRAIN_STEPS", "25000")),
        help="Iteration count for --final-full-train-best. Uses the same full training shard set, but does not stop at the short gate length.",
    )
    parser.add_argument(
        "--upload-best-short-to-hf",
        action="store_true",
        default=env_truthy("TORICGT_UPLOAD_BEST_SHORT_TO_HF", "0"),
        help="After the exact short-run sweep completes, upload the best short-run checkpoint and int8 artifact to Hugging Face.",
    )
    parser.add_argument(
        "--hf-checkpoint-repo",
        default=os.environ.get("TORICGT_HF_CHECKPOINT_REPO", "AmelieSchreiber/toricgt-checkpoints"),
        help="Hugging Face model repo for --upload-best-short-to-hf. Defaults to the existing ToricGT checkpoint repo.",
    )
    parser.add_argument(
        "--hf-token-file",
        default=os.environ.get("TORICGT_HF_TOKEN_FILE", "keys.txt"),
        help="Local file containing an hf_* token. The token is read into the environment and never printed.",
    )
    parser.add_argument(
        "--submission-step-candidate",
        type=int,
        default=int(os.environ.get("TORICGT_SUBMISSION_STEP_CANDIDATE", "0")),
        help="After the short-run sweep, rerun the best profile for this many steps as an exact submission candidate. Use 900 to test whether the pre-1K checkpoint is better; 0 disables.",
    )
    parser.add_argument(
        "--create-parameter-golf-pr",
        action="store_true",
        default=env_truthy("TORICGT_CREATE_PARAMETER_GOLF_PR", "0"),
        help="After the short-run sweep and optional submission-step candidate, create a Parameter Golf record folder, commit it in the local fork, push it, and open a PR.",
    )
    parser.add_argument(
        "--parameter-golf-repo",
        default=os.environ.get("TORICGT_PARAMETER_GOLF_REPO", "amelie-iska/parameter-golf"),
        help="Local clone of the Parameter Golf fork used for record-folder submission packaging.",
    )
    parser.add_argument(
        "--parameter-golf-track",
        default=os.environ.get("TORICGT_PARAMETER_GOLF_TRACK", "track_non_record_16mb"),
        help="Records subfolder to use for the generated submission. Defaults to non-record because ConvexTok/custom graphification needs extra validation.",
    )
    parser.add_argument(
        "--parameter-golf-author",
        default=os.environ.get("TORICGT_PARAMETER_GOLF_AUTHOR", "Amelie Schreiber"),
    )
    parser.add_argument(
        "--parameter-golf-github-id",
        default=os.environ.get("TORICGT_PARAMETER_GOLF_GITHUB_ID", "amelie-iska"),
    )
    parser.add_argument(
        "--parameter-golf-submission-name",
        default=os.environ.get("TORICGT_PARAMETER_GOLF_SUBMISSION_NAME", "ToricGT ConvexTok-2048 Graphified FoT"),
    )
    parser.add_argument(
        "--parameter-golf-pr-base-repo",
        default=os.environ.get("TORICGT_PARAMETER_GOLF_PR_BASE_REPO", "openai/parameter-golf"),
    )
    parser.add_argument(
        "--parameter-golf-pr-base",
        default=os.environ.get("TORICGT_PARAMETER_GOLF_PR_BASE", "main"),
    )
    parser.add_argument(
        "--parameter-golf-pr-draft",
        action="store_true",
        default=env_truthy("TORICGT_PARAMETER_GOLF_PR_DRAFT", "0"),
        help="Open the Parameter Golf PR as a draft. Disabled by default when --create-parameter-golf-pr is used because that flag means submit.",
    )
    parser.add_argument(
        "--parameter-golf-include-model-artifact",
        action="store_true",
        default=env_truthy("TORICGT_PARAMETER_GOLF_INCLUDE_MODEL_ARTIFACT", "0"),
        help="Copy final_model.int8.ptz into the record folder. Disabled by default; the model artifact is uploaded to Hugging Face and the PR carries reproducible code/logs/manifests.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_log(log_path: Path) -> dict[str, float | int | str | None]:
    vals: list[dict[str, float]] = []
    trains: list[dict[str, float]] = []
    final_bpb: float | None = None
    final_loss: float | None = None
    artifact_size: int | None = None
    checkpoint_path: str | None = None
    checkpoint_step: int | None = None
    if not log_path.exists():
        return {"exists": 0, "path": str(log_path)}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := VAL_RE.search(line):
            vals.append({key: float(value) for key, value in match.groupdict().items()})
        if " train_loss:" in line:
            row = parse_train_metric_line(line)
            if row:
                trains.append(row)
        if match := FINAL_RE.search(line):
            final_loss = float(match.group("loss"))
            final_bpb = float(match.group("bpb"))
        if match := SIZE_RE.search(line):
            artifact_size = int(match.group("size"))
        if match := CHECKPOINT_RE.search(line):
            checkpoint_path = match.group("path")
            checkpoint_step = int(match.group("step"))
    latest_val = vals[-1] if vals else {}
    latest_train: dict[str, float] = {}
    for row in trains:
        latest_train.update(row)
    return {
        "exists": 1,
        "path": str(log_path),
        "val_step": int(latest_val.get("step", -1)),
        "val_loss": latest_val.get("loss"),
        "val_bpb": latest_val.get("bpb"),
        "final_int8_loss": final_loss,
        "final_int8_bpb": final_bpb,
        "artifact_bytes": artifact_size,
        "checkpoint_path": checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "train_step": int(latest_train.get("step", -1)),
        "train_loss": latest_train.get("loss"),
        "train_bpb": latest_train.get("train_bpb"),
        "train_bpt": latest_train.get("train_bpt"),
        "graph_lm_loss": latest_train.get("graph_lm_loss"),
        "graph_lm_bpb": latest_train.get("graph_lm_bpb"),
        "graph_lm_weight": latest_train.get("graph_lm_weight"),
        "oai_gflownet_loss": latest_train.get("oai_gflownet_loss"),
        "oai_gflownet_entropy": latest_train.get("oai_gflownet_entropy"),
        "oai_gflownet_reward": latest_train.get("oai_gflownet_reward"),
        "oai_fot_loss": latest_train.get("oai_fot_loss"),
        "oai_fot_entropy": latest_train.get("oai_fot_entropy"),
        "oai_fot_reward": latest_train.get("oai_fot_reward"),
        "oai_fot_diversity": latest_train.get("oai_fot_diversity"),
        "oai_mtp_loss": latest_train.get("oai_mtp_loss"),
        "oai_mtp_weight": latest_train.get("oai_mtp_weight"),
        "sidecar_loss": latest_train.get("sidecar"),
        "graphcg_loss": latest_train.get("graphcg"),
        "analogy_loss": latest_train.get("analogy"),
        "tokengt_graph_loss": latest_train.get("tokengt"),
        "memory_loss": latest_train.get("memory"),
        "toric_geometry_loss": latest_train.get("toric"),
        "toric_vector_bundle_1d_cone_ce_loss": latest_train.get("vector_bundle_1d_cone"),
        "toric_bgg_loss": latest_train.get("bgg"),
        "koszul_persistence_loss": latest_train.get("koszul"),
        "toric_cca_topology_loss": latest_train.get("combinatorial_toric"),
        "derived_signature_loss": latest_train.get("derived_signature"),
    }


def parse_train_metric_line(line: str) -> dict[str, float]:
    """Parse a train log line without regex backtracking.

    Some training lines contain only BPB/loss fields, while analysis lines add
    graph, FoT, GFlowNet, toric, BGG, and persistence metrics.  A single broad
    regex can spend unbounded time backtracking on lines that omit optional
    sidecar fields.  Token parsing is deterministic and keeps the controller
    from stalling between runs.
    """

    aliases = {
        "train_loss": "loss",
        "train_bpb": "train_bpb",
        "train_bpt": "train_bpt",
        "graph_lm_loss": "graph_lm_loss",
        "graph_lm_bpb": "graph_lm_bpb",
        "graph_lm_w": "graph_lm_weight",
        "oai_gfn": "oai_gflownet_loss",
        "gfn_H": "oai_gflownet_entropy",
        "gfn_R": "oai_gflownet_reward",
        "oai_fot": "oai_fot_loss",
        "fot_H": "oai_fot_entropy",
        "fot_R": "oai_fot_reward",
        "fot_div": "oai_fot_diversity",
        "mtp": "oai_mtp_loss",
        "mtp_w": "oai_mtp_weight",
        "sidecar_loss": "sidecar",
        "graphcg": "graphcg",
        "analogy": "analogy",
        "tokengt_graph": "tokengt",
        "memory": "memory",
        "toric": "toric",
        "vb1d": "vector_bundle_1d_cone",
        "vb": "vector_bundle_1d_cone",
        "bgg": "bgg",
        "koszul": "koszul",
        "cca": "combinatorial_toric",
        "derived": "derived_signature",
    }
    row: dict[str, float] = {}
    for token in line.split():
        if ":" not in token:
            continue
        key, raw_value = token.split(":", 1)
        raw_value = raw_value.rstrip(",;")
        if key == "step":
            try:
                step, total = raw_value.split("/", 1)
                row["step"] = float(step)
                row["total"] = float(total)
            except ValueError:
                continue
            continue
        mapped = aliases.get(key)
        if mapped is None:
            continue
        try:
            row[mapped] = float(raw_value)
        except ValueError:
            continue
    return row


def metric_bpb(metrics: dict[str, float | int | str | None]) -> float:
    value = metrics.get("final_int8_bpb")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    value = metrics.get("val_bpb")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float("inf")


def profile_by_name(name: str) -> Profile | None:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    return None


def choose_profile(run_index: int, history: list[dict[str, object]], offset: int) -> Profile:
    if history:
        for row in reversed(history):
            decision = row.get("analysis_decision", {})
            if isinstance(decision, dict):
                hinted = profile_by_name(str(decision.get("next_profile_hint", "")))
                if hinted is not None:
                    return hinted
    if not history:
        return PROFILES[offset % len(PROFILES)]
    known = {profile.name: profile for profile in PROFILES}
    rows: list[tuple[str, dict[str, float | int | str | None]]] = []
    for item in history:
        profile_name = str(item.get("profile", ""))
        metrics = item.get("metrics", {})
        if profile_name in known and isinstance(metrics, dict):
            rows.append((profile_name, metrics))
    tried = {name for name, _ in rows}
    if len(tried) < len(PROFILES):
        start = max(0, int(offset))
        for idx in range(len(PROFILES)):
            candidate = PROFILES[(start + idx + run_index - 1) % len(PROFILES)]
            if candidate.name not in tried:
                return candidate
    grouped: dict[str, list[dict[str, float | int | str | None]]] = {}
    for name, metrics in rows:
        grouped.setdefault(name, []).append(metrics)
    scored: list[tuple[float, Profile]] = []
    total = max(1, sum(len(values) for values in grouped.values()))
    for profile in PROFILES:
        values = grouped.get(profile.name, [])
        if not values:
            scored.append((float("-inf"), profile))
            continue
        bpbs = [metric_bpb(metrics) for metrics in values]
        finite_bpbs = [value for value in bpbs if math.isfinite(value)]
        if not finite_bpbs:
            center = 9.0
        else:
            center = min(finite_bpbs)
        train_bpbs = [
            float(metrics["train_bpb"])
            for metrics in values
            if isinstance(metrics.get("train_bpb"), (int, float)) and math.isfinite(float(metrics["train_bpb"]))
        ]
        graph_bpbs = [
            float(metrics["graph_lm_bpb"])
            for metrics in values
            if isinstance(metrics.get("graph_lm_bpb"), (int, float)) and math.isfinite(float(metrics["graph_lm_bpb"]))
        ]
        train_bonus = 0.002 * max(0.0, 2.0 - min(train_bpbs)) if train_bpbs else 0.0
        graph_penalty = 0.001 * max(0.0, min(graph_bpbs) - 2.5) if graph_bpbs else 0.0
        artifact_penalty = 0.0
        for metrics in values:
            artifact = metrics.get("artifact_bytes")
            if isinstance(artifact, (int, float)) and float(artifact) > 16_000_000:
                artifact_penalty += 0.010
        exploration_bonus = 0.012 * math.sqrt(math.log(total + 1.0) / max(1, len(values)))
        score = center + graph_penalty + artifact_penalty - train_bonus - exploration_bonus
        scored.append((score, profile))
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def decision_env_overrides(decision: dict[str, object] | None) -> dict[str, str]:
    if not isinstance(decision, dict):
        return {}
    raw = decision.get("env_overrides")
    if not isinstance(raw, dict):
        return {}
    clean: dict[str, str] = {}
    for key, value in raw.items():
        key_s = str(key).strip()
        if not key_s:
            continue
        clean[key_s] = str(value)
    return clean


def latest_env_overrides(history: list[dict[str, object]]) -> dict[str, str]:
    if not history:
        return {}
    for row in reversed(history):
        decision = row.get("analysis_decision")
        overrides = decision_env_overrides(decision if isinstance(decision, dict) else None)
        if overrides:
            return overrides
    return {}


def shell_env(env: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))


def write_report(
    path: Path,
    title: str,
    run_id: str,
    profile: Profile | None,
    metrics: dict[str, float | int | str | None],
    next_profile: Profile | None,
    returncode: int | None = None,
    analysis_decision: dict[str, object] | None = None,
    target_bpb: float = 1.19,
    env_overrides_used: dict[str, str] | None = None,
    next_env_overrides: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpb = metric_bpb(metrics)
    gap = bpb - float(target_bpb) if math.isfinite(bpb) else float("inf")
    lines = [
        f"# {title}",
        "",
        f"- timestamp UTC: `{utc_iso()}`",
        f"- run id: `{run_id}`",
        f"- return code: `{returncode}`" if returncode is not None else "- return code: `not applicable`",
        f"- parsed log: `{metrics.get('path')}`",
        f"- target BPB: `{float(target_bpb):.6f}`",
        f"- selected BPB: `{bpb:.6f}`" if math.isfinite(bpb) else "- selected BPB: `unavailable`",
        f"- target gap: `{gap:.6f}`" if math.isfinite(gap) else "- target gap: `unavailable`",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
    ]
    if profile is not None:
        lines.extend(
            [
                "## Hyperparameters Used",
                "",
                f"- profile: `{profile.name}`",
                f"- rationale: {profile.rationale}",
                f"- adaptive overrides applied: `{bool(env_overrides_used)}`",
                "",
                "```json",
                json.dumps(profile.env, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    if env_overrides_used:
        lines.extend(
            [
                "## Adaptive Overrides Used",
                "",
                "```json",
                json.dumps(env_overrides_used, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    if next_profile is not None:
        lines.extend(
            [
                "## Next Run Decision",
                "",
                f"- next profile: `{next_profile.name}`",
                f"- rationale: {next_profile.rationale}",
                "",
                "```json",
                json.dumps(next_profile.env, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    if next_env_overrides:
        lines.extend(
            [
                "## Next Adaptive Overrides",
                "",
                "These values come from the checkpoint-backed full-analysis pass and override the base profile on the next fresh step-0 restart.",
                "",
                "```json",
                json.dumps(next_env_overrides, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    if analysis_decision:
        lines.extend(
            [
                "## Full Analysis Decision",
                "",
                "```json",
                json.dumps(analysis_decision, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Each campaign attempt restarts from step 0.  The active OAI FineWeb tokenizer stream remains the primary BPB objective.  "
            "For ConvexTok runs, tokenisation is a byte-boundary DAG with exact min-plus shortest-path encoding; LP lower-bound regret, active-path tropical margins, and toric vocabulary-face metrics are reviewed as tokenizer-specific evidence rather than merged into generic auxiliary losses.  "
            "FineWeb graphification is first-class and enabled by default: the main GPT input stream receives causal TokenGT-style node/edge structural embeddings before the transformer blocks, and optional OAI-FineWeb-only output flattening maps graph-output states back to SentencePiece sequence order for BPB scoring.  "
            "Curated graph data is also bonafide primary LM training data: graph records are serialized into causal topological node/edge token sequences when directed and acyclic, and deterministic random-order graph token sequences when noncausal or cyclic.  Graph-data hidden states are not flattened by default; they remain graph structured for graph-in/graph-out training and sidecar analysis.  "
            "ToricGT sidecar losses remain active from step 0 for full-rank GraphCG, analogy lattice structure, graph supervision, and trajectory-memory retrieval, but their weights are kept small so they regularize rather than dominate BPB.  "
            "The OAI path also trains embedding-space GFlowNet graph-of-thought and Forest-of-Thought heads as BPB-facing training-only trajectory objectives; these shape hidden reasoning forests and retrieval behavior without entering the compressed artifact unless explicitly exported later.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_codex_review(report: Path) -> None:
    prompt = (
        "Review this ToricGT OAI-baseline threshold-gated BPB campaign report. "
        "Review every available metric and generated visualization summary, including BPB/train BPB, first-class FineWeb graphification, "
        "bonafide graph-LM primary loss on graphified node/edge token records, GraphCG, analogical memory, TokenGT graph losses, "
        "toric/tropical geometry, persistent homology, vector-bundle/sheaf, "
        "BGG category O, Koszul/resolution, combinatorial commutative algebra, scheduled graph-LM weights, teacher distillation, "
        "adaptive graph radius, graph-output flattening CE lift/regression, calibration loss, score-correction gates, "
        "ConvexTok tokenizer-regret metrics, LP lower-bound gap ratios, min-plus tokenisation-DAG active paths, tropical path margins, toric vocabulary-face entropy, "
        "OAI embedding-space GFlowNet graph-of-thought trajectory-balance metrics, OAI embedding-space Forest-of-Thought sparse activation/UCB/self-correction/consensus/trajectory-balance metrics, OAI multi-token prediction metrics, "
        "BPB-first auxiliary staging multipliers, family-level gradient conflict route scales, FoT BPB-delta reward metrics, and reward-coupled FoT entropy/diversity controls, "
        "non-destructive score-first TTA metrics, "
        "auxiliary-gradient routing cosines/projection coefficients, retrieval gates, uncertainty-weighted advanced multipliers, "
        "GraphCG BPB-orthogonalization metrics, artifact size, and round-trip quantization. "
        "If the report's full-analysis decision includes a sidecar_metric_review path, read that markdown/JSON and account for every observed "
        "`toricgt_sidecar/*` metric description, trend, BPB correlation, and family weight guidance before recommending the next restart profile. "
        "If the 1.5K gate missed BPB < 1.19, recommend exact hyperparameter/config changes and explain why the next run should restart from step 0. "
        "Do not collapse all advanced losses into one bin: reason separately about tropical/toric fan losses, BGG/category-O losses, "
        "vector-bundle/sheaf losses, persistent homology/Koszul losses, combinatorial toric algebra, ConvexTok tokenizer regret, tropical shortest-path tokenisation, GraphCG, analogy, memory, and graph-LM. "
        "For loss-like metrics, interpret positive BPB correlation as possible evidence that reducing that loss could help BPB; "
        "do not automatically treat every positive correlation as harmful. "
        "Stay exploratory: this project has only a small number of short FoT-enabled 1K-step runs, so recommend inventive, evidence-based changes instead of prematurely collapsing the sweep. "
        "Specifically evaluate whether BPB-first staging protected early CE descent, whether conflict routing projected or damped the right families, whether FoT reward_bpb_delta and corrected_byte_nll are moving in the right direction, and whether FoT entropy/diversity control should adjust temperature, UCB, sparse pressure, or loss weight next run. "
        f"Report path: {report}"
    )
    stdout_path = report.with_name(f"{report.stem}.codex-review.stdout.log")
    stderr_path = report.with_name(f"{report.stem}.codex-review.stderr.log")
    try:
        stdout = stdout_path.open("w", encoding="utf-8")
        stderr = stderr_path.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                ["codex", "exec", prompt],
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        finally:
            stdout.close()
            stderr.close()
        (report.with_name(f"{report.stem}.codex-review.pid")).write_text(f"{proc.pid}\n", encoding="utf-8")
    except Exception as exc:
        report.with_name(f"{report.stem}.codex-review.failed.txt").write_text(
            f"failed to launch codex review: {exc!r}\n",
            encoding="utf-8",
        )


def launch_training(
    args: argparse.Namespace,
    run_id: str,
    profile: Profile,
    run_index: int,
    env_overrides: dict[str, str] | None = None,
) -> int:
    repo = Path(args.repo_root).resolve()
    work_dir = repo / "runs" / "oai_sidecar" / run_id
    ckpt_dir = repo / "checkpoints" / run_id
    notes_dir = repo / "training_notes" / args.campaign_id
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)
    train_gpt = repo / "amelie-iska" / "parameter-golf" / "train_gpt.py"
    fineweb_data = (repo / args.fineweb_data).resolve() if not Path(args.fineweb_data).is_absolute() else Path(args.fineweb_data)
    tokenizer_path = (repo / args.tokenizer_path).resolve() if not Path(args.tokenizer_path).is_absolute() else Path(args.tokenizer_path)
    base_env = {
        "PYTHONPATH": str(repo / "src"),
        "PYTHONUNBUFFERED": "1",
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_RUN_ID": run_id,
        "WANDB_RUN_NAME": run_id,
        "WANDB_RESUME": "never",
        "RUN_ID": run_id,
        "DATA_PATH": str(fineweb_data),
        "TOKENIZER_PATH": str(tokenizer_path),
        "RESUME_CHECKPOINT": "",
        "START_STEP": "0",
        "TORICGT_SIDECAR": "1",
        "REQUIRE_TORICGT_SIDECAR": "1",
        "GRAPH_DATA_PATH": args.graph_data_path,
        "GRAPH_LM_PRIMARY": "1",
        "REQUIRE_GRAPH_LM_PRIMARY": "1",
        "GRAPH_LM_SEQ_LEN": "1024",
        "GRAPH_LM_BATCH_SIZE": os.environ.get("GRAPH_LM_BATCH_SIZE", "2"),
        "GRAPH_LM_EVERY": os.environ.get("GRAPH_LM_EVERY", "4"),
        "GRAPH_LM_LOSS_WEIGHT": "0.08",
        "GRAPH_LM_LOSS_WEIGHT_START": "0.025",
        "GRAPH_LM_WARMUP_STEPS": "350",
        "GRAPH_LM_HOLD_STEPS": "25",
        "AUX_GRAD_ROUTING": "1",
        "AUX_GRAD_ROUTE_GRAPH_LM": "1",
        "AUX_GRAD_ROUTE_SIDECAR": "1",
        "AUX_GRAD_ROUTE_TEACHER": "1",
        "AUX_GRAD_CONFLICT_PROJECTION": "1",
        "AUX_GRAD_ALIGNED_BOOST": "1.0",
        "AUX_CONFLICT_CONTROLLER": "1",
        "AUX_CONFLICT_DAMP_MIN": "0.10",
        "AUX_CONFLICT_BOOST_MAX": "1.35",
        "BPB_FIRST_AUX_STAGING": "1",
        "BPB_FIRST_CORE_STEPS": "250",
        "BPB_FIRST_RAMP_STEPS": "500",
        "BPB_FIRST_MIN_AUX_MULT": "0.02",
        "BPB_FIRST_REQUIRE_NEGATIVE_SLOPE": "1",
        "BPB_FIRST_CURVATURE_GUARD": "1",
        "BPB_FIRST_BAD_SLOPE_MULT": "0.25",
        "BPB_FIRST_BAD_CURVATURE_MULT": "0.35",
        "TEACHER_CHECKPOINT": os.environ.get("TEACHER_CHECKPOINT", ""),
        "TEACHER_DISTILL_WEIGHT": os.environ.get("TEACHER_DISTILL_WEIGHT", "0"),
        "TEACHER_DISTILL_WEIGHT_START": os.environ.get("TEACHER_DISTILL_WEIGHT_START", "0"),
        "TEACHER_DISTILL_UNTIL_STEP": os.environ.get("TEACHER_DISTILL_UNTIL_STEP", "900"),
        "TEACHER_DISTILL_DECAY_STEPS": os.environ.get("TEACHER_DISTILL_DECAY_STEPS", "300"),
        "TEACHER_DISTILL_TEMPERATURE": os.environ.get("TEACHER_DISTILL_TEMPERATURE", "2.0"),
        "TEACHER_DISTILL_MAX_SEQUENCES": os.environ.get("TEACHER_DISTILL_MAX_SEQUENCES", "2"),
        "FINEWEB_CASEOPS": os.environ.get("FINEWEB_CASEOPS", "0"),
        "VOCAB_SIZE": str(int(args.vocab_size)),
        "MODEL_DIM": "512",
        "NUM_LAYERS": "9",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2",
        "TIE_EMBEDDINGS": "1",
        "TRAIN_SEQ_LEN": "1024",
        "FINEWEB_GRAPHIFY": "1",
        "TOKENGT_FIRST_CLASS": "1",
        "TOKENGT_FIRST_CLASS_LR": "1.4e-4",
        "TOKENGT_GRAPH_RADIUS": "4",
        "TOKENGT_DISTANCE_FEATURES": "functional",
        "TOKENGT_TOKEN_CLASS_BUCKETS": "64",
        "TOKENGT_POSITION_BUCKETS": "256",
        "TOKENGT_STRUCTURAL_WEIGHT": "0.035",
        "TOKENGT_EDGE_WEIGHT": "0.025",
        "TOKENGT_TORUS_WEIGHT": "0.010",
        "TOKENGT_IDENTIFIER_DIM": "24",
        "TOKENGT_IDENTIFIER_WEIGHT": "0.014",
        "TOKENGT_ENDPOINT_WEIGHT": "0.018",
        "TOKENGT_EDGE_TOKEN_WEIGHT": "0.016",
        "CONVEXTOK_DAG_FEATURES": os.environ.get("CONVEXTOK_DAG_FEATURES", "1"),
        "CONVEXTOK_DAG_FEATURE_WEIGHT": os.environ.get("CONVEXTOK_DAG_FEATURE_WEIGHT", "0.018"),
        "CONVEXTOK_TORIC_REG_WEIGHT": os.environ.get("CONVEXTOK_TORIC_REG_WEIGHT", "0"),
        "CONVEXTOK_TORIC_REG_TOPK": os.environ.get("CONVEXTOK_TORIC_REG_TOPK", "96"),
        "OAI_FINEWEB_OUTPUT_FLATTENING": "1",
        "GRAPH_OUTPUT_FLATTENING": "1",
        "GRAPH_OUTPUT_FLATTENING_LR": "1.4e-4",
        "GRAPH_OUTPUT_EDGE_RADIUS": "4",
        "GRAPH_OUTPUT_DISTANCE_FEATURES": "functional",
        "GRAPH_OUTPUT_NODE_WEIGHT": "0.05",
        "GRAPH_OUTPUT_EDGE_WEIGHT": "0.05",
        "GRAPH_OUTPUT_VIRTUAL_EDGE_TOKENS": "1",
        "GRAPH_OUTPUT_EDGE_TOKEN_WEIGHT": "0.035",
        "GRAPH_OUTPUT_SCORE_CORRECTION": "1",
        "GRAPH_OUTPUT_SCORE_CORRECTION_WEIGHT": "0.024",
        "GRAPH_OUTPUT_CALIBRATION_LOSS_WEIGHT": "0.012",
        "GRAPH_OUTPUT_CALIBRATION_MARGIN": "0.0",
        "GRAPH_OUTPUT_CALIBRATION_EVERY": os.environ.get("GRAPH_OUTPUT_CALIBRATION_EVERY", "50"),
        "GRAPH_OUTPUT_CALIBRATION_MAX_SEQUENCES": os.environ.get("GRAPH_OUTPUT_CALIBRATION_MAX_SEQUENCES", "1"),
        "VAL_BATCH_SIZE": "524288",
        "VAL_MAX_TOKENS": os.environ.get("VAL_MAX_TOKENS", "8388608"),
        "EVAL_INITIAL": os.environ.get("EVAL_INITIAL", "0"),
        "ITERATIONS": str(args.steps_per_run),
        "MAX_WALLCLOCK_SECONDS": "0",
        "VAL_LOSS_EVERY": str(args.steps_per_run),
        "TRAIN_LOG_EVERY": os.environ.get("TRAIN_LOG_EVERY", "1"),
        "CHECKPOINT_EVERY": str(args.steps_per_run),
        "CHECKPOINT_DIR": str(ckpt_dir),
        "TORICGT_SIDECAR_SEQ_LEN": "256",
        "TORICGT_SIDECAR_BATCH_SIZE": os.environ.get("TORICGT_SIDECAR_BATCH_SIZE", "4"),
        "TORICGT_SIDECAR_EVERY": os.environ.get("TORICGT_SIDECAR_EVERY", "4"),
        "TORICGT_SIDECAR_LOSS_WEIGHT": "1.0",
        "TORICGT_SIDECAR_LOSS_WEIGHT_START": "0.05",
        "TORICGT_SIDECAR_WARMUP_STEPS": "500",
        "TORICGT_SIDECAR_HOLD_STEPS": "100",
        "TORICGT_SIDECAR_COMPUTE_ALL_METRICS": "1",
        "RETRIEVAL_CONDITIONED_AUX": "1",
        "RETRIEVAL_GATE_MIN": "0.20",
        "RETRIEVAL_GATE_CENTER": "0.20",
        "RETRIEVAL_GATE_SOFTNESS": "0.08",
        "SIDECAR_UNCERTAINTY_WEIGHTING": "1",
        "SIDECAR_UNCERTAINTY_ALPHA": "0.35",
        "SIDECAR_UNCERTAINTY_CENTER": "2.6",
        "SIDECAR_UNCERTAINTY_SCALE": "1.0",
        "SIDECAR_UNCERTAINTY_MAX": "2.0",
        "GRAPHCG_BPB_ORTHOGONAL_WEIGHT": "0.02",
        "GRAPHCG_COVARIANCE_CONFLICT_DAMPING": "0.50",
        "DERIVED_SIGNATURE_LOSS_WEIGHT": os.environ.get("DERIVED_SIGNATURE_LOSS_WEIGHT", "1e-6"),
        "DERIVED_SIGNATURE_MAX_VERTICES": os.environ.get("DERIVED_SIGNATURE_MAX_VERTICES", "8"),
        "DERIVED_SIGNATURE_MAX_EDGES": os.environ.get("DERIVED_SIGNATURE_MAX_EDGES", "64"),
        "ADVANCED_LAGRANGIAN_CONTROLLER": os.environ.get("ADVANCED_LAGRANGIAN_CONTROLLER", "1"),
        "LAGRANGIAN_DUAL_LR": os.environ.get("LAGRANGIAN_DUAL_LR", "0.025"),
        "LAGRANGIAN_DECAY": os.environ.get("LAGRANGIAN_DECAY", "0.985"),
        "LAGRANGIAN_MIN_MULTIPLIER": os.environ.get("LAGRANGIAN_MIN_MULTIPLIER", "0.25"),
        "LAGRANGIAN_MAX_MULTIPLIER": os.environ.get("LAGRANGIAN_MAX_MULTIPLIER", "2.25"),
        "LAGRANGIAN_BPB_CEILING": os.environ.get("LAGRANGIAN_BPB_CEILING", "3.05"),
        "LAGRANGIAN_BPB_SOFTNESS": os.environ.get("LAGRANGIAN_BPB_SOFTNESS", "0.35"),
        "TORIC_FAN_CURRICULUM": os.environ.get("TORIC_FAN_CURRICULUM", "1"),
        "TORIC_FAN_COARSE_STEPS": os.environ.get("TORIC_FAN_COARSE_STEPS", "450"),
        "TORIC_FAN_INTERMEDIATE_STEPS": os.environ.get("TORIC_FAN_INTERMEDIATE_STEPS", "950"),
        "MEMORY_SHEAF_GATE_MIN": os.environ.get("MEMORY_SHEAF_GATE_MIN", "0.15"),
        "MEMORY_SHEAF_GATE_THRESHOLD": os.environ.get("MEMORY_SHEAF_GATE_THRESHOLD", "0.12"),
        "MEMORY_SHEAF_GATE_SOFTNESS": os.environ.get("MEMORY_SHEAF_GATE_SOFTNESS", "0.18"),
        "MEMORY_SHEAF_CE_WEIGHT": os.environ.get("MEMORY_SHEAF_CE_WEIGHT", "1.0"),
        "SCORE_FIRST_TTA": os.environ.get("SCORE_FIRST_TTA", "1"),
        "SCORE_FIRST_TTA_STEPS": os.environ.get("SCORE_FIRST_TTA_STEPS", "64"),
        "SCORE_FIRST_TTA_LR": os.environ.get("SCORE_FIRST_TTA_LR", "2e-5"),
        "SCORE_FIRST_TTA_COMMIT": os.environ.get("SCORE_FIRST_TTA_COMMIT", "0"),
        "OAI_GFLOWNET": os.environ.get("OAI_GFLOWNET", "1"),
        "OAI_GFLOWNET_LR": os.environ.get("OAI_GFLOWNET_LR", "2e-4"),
        "OAI_GFLOWNET_EVERY": os.environ.get("OAI_GFLOWNET_EVERY", "4"),
        "OAI_GFLOWNET_LOSS_WEIGHT": os.environ.get("OAI_GFLOWNET_LOSS_WEIGHT", "2e-5"),
        "OAI_GFLOWNET_ENTROPY_WEIGHT": os.environ.get("OAI_GFLOWNET_ENTROPY_WEIGHT", "2e-6"),
        "OAI_GFLOWNET_ENTROPY_TARGET": os.environ.get("OAI_GFLOWNET_ENTROPY_TARGET", "1.8"),
        "OAI_GFLOWNET_NUM_ACTIONS": os.environ.get("OAI_GFLOWNET_NUM_ACTIONS", "16"),
        "OAI_GFLOWNET_HIDDEN_DIM": os.environ.get("OAI_GFLOWNET_HIDDEN_DIM", "192"),
        "OAI_GFLOWNET_MAX_SEQUENCES": os.environ.get("OAI_GFLOWNET_MAX_SEQUENCES", "1"),
        "OAI_GFLOWNET_MAX_POSITIONS": os.environ.get("OAI_GFLOWNET_MAX_POSITIONS", "128"),
        "OAI_EMBEDDING_FOT": os.environ.get("OAI_EMBEDDING_FOT", "1"),
        "OAI_FOT_LR": os.environ.get("OAI_FOT_LR", "1.5e-4"),
        "OAI_FOT_EVERY": os.environ.get("OAI_FOT_EVERY", "4"),
        "OAI_FOT_LOSS_WEIGHT": os.environ.get("OAI_FOT_LOSS_WEIGHT", "1.5e-5"),
        "OAI_FOT_NUM_TREES": os.environ.get("OAI_FOT_NUM_TREES", "3"),
        "OAI_FOT_MAX_DEPTH": os.environ.get("OAI_FOT_MAX_DEPTH", "4"),
        "OAI_FOT_BRANCHING": os.environ.get("OAI_FOT_BRANCHING", "3"),
        "OAI_FOT_TOPK_TREES": os.environ.get("OAI_FOT_TOPK_TREES", "2"),
        "OAI_FOT_HIDDEN_DIM": os.environ.get("OAI_FOT_HIDDEN_DIM", "192"),
        "OAI_FOT_MAX_SEQUENCES": os.environ.get("OAI_FOT_MAX_SEQUENCES", "1"),
        "OAI_FOT_MAX_POSITIONS": os.environ.get("OAI_FOT_MAX_POSITIONS", "128"),
        "OAI_FOT_CONSENSUS_BUCKETS": os.environ.get("OAI_FOT_CONSENSUS_BUCKETS", "64"),
        "OAI_FOT_CORRECTION_SCALE": os.environ.get("OAI_FOT_CORRECTION_SCALE", "0.08"),
        "OAI_FOT_UCB_EXPLORATION": os.environ.get("OAI_FOT_UCB_EXPLORATION", "1.25"),
        "OAI_FOT_TEMPERATURE": os.environ.get("OAI_FOT_TEMPERATURE", "0.70"),
        "OAI_FOT_SPARSE_WEIGHT": os.environ.get("OAI_FOT_SPARSE_WEIGHT", "1.0"),
        "OAI_FOT_UCB_WEIGHT": os.environ.get("OAI_FOT_UCB_WEIGHT", "0.45"),
        "OAI_FOT_CORRECTION_WEIGHT": os.environ.get("OAI_FOT_CORRECTION_WEIGHT", "0.55"),
        "OAI_FOT_CONSENSUS_WEIGHT": os.environ.get("OAI_FOT_CONSENSUS_WEIGHT", "0.80"),
        "OAI_FOT_TB_WEIGHT": os.environ.get("OAI_FOT_TB_WEIGHT", "1.0"),
        "OAI_FOT_SUBTB_WEIGHT": os.environ.get("OAI_FOT_SUBTB_WEIGHT", "0.20"),
        "OAI_FOT_COMPLEXITY_WEIGHT": os.environ.get("OAI_FOT_COMPLEXITY_WEIGHT", "0.04"),
        "OAI_FOT_REWARD_ADVANCED_BONUS": os.environ.get("OAI_FOT_REWARD_ADVANCED_BONUS", "0.05"),
        "OAI_FOT_REWARD_MODE": os.environ.get("OAI_FOT_REWARD_MODE", "bpb_delta"),
        "OAI_FOT_BPB_DELTA_WEIGHT": os.environ.get("OAI_FOT_BPB_DELTA_WEIGHT", "1.0"),
        "OAI_FOT_REWARD_GRAPH_WEIGHT": os.environ.get("OAI_FOT_REWARD_GRAPH_WEIGHT", "0.10"),
        "OAI_FOT_REWARD_CONSENSUS_WEIGHT": os.environ.get("OAI_FOT_REWARD_CONSENSUS_WEIGHT", "0.20"),
        "OAI_FOT_REWARD_COMPLEXITY_WEIGHT": os.environ.get("OAI_FOT_REWARD_COMPLEXITY_WEIGHT", "0.02"),
        "OAI_FOT_REWARD_FLOOR": os.environ.get("OAI_FOT_REWARD_FLOOR", "1e-4"),
        "OAI_FOT_ADAPTIVE_CONTROL": os.environ.get("OAI_FOT_ADAPTIVE_CONTROL", "1"),
        "OAI_FOT_REWARD_TARGET": os.environ.get("OAI_FOT_REWARD_TARGET", "0.08"),
        "OAI_FOT_DIVERSITY_TARGET": os.environ.get("OAI_FOT_DIVERSITY_TARGET", "0.12"),
        "OAI_FOT_ENTROPY_HIGH": os.environ.get("OAI_FOT_ENTROPY_HIGH", "0.985"),
        "OAI_FOT_ENTROPY_LOW": os.environ.get("OAI_FOT_ENTROPY_LOW", "0.75"),
        "OAI_FOT_TEMP_MIN": os.environ.get("OAI_FOT_TEMP_MIN", "0.45"),
        "OAI_FOT_TEMP_MAX": os.environ.get("OAI_FOT_TEMP_MAX", "1.10"),
        "OAI_FOT_UCB_MIN": os.environ.get("OAI_FOT_UCB_MIN", "0.40"),
        "OAI_FOT_UCB_MAX": os.environ.get("OAI_FOT_UCB_MAX", "1.80"),
        "OAI_MTP": os.environ.get("OAI_MTP", "1"),
        "OAI_MTP_EVERY": os.environ.get("OAI_MTP_EVERY", "2"),
        "OAI_MTP_LOSS_WEIGHT": os.environ.get("OAI_MTP_LOSS_WEIGHT", "0.003"),
        "OAI_MTP_OFFSETS": os.environ.get("OAI_MTP_OFFSETS", "2"),
        "OAI_MTP_MAX_SEQUENCES": os.environ.get("OAI_MTP_MAX_SEQUENCES", "1"),
        "SEED": str(1337 + run_index),
    }
    env = {**base_env, **profile.env}
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    launch_record = notes_dir / f"RUN-{run_index:03d}-{run_id}-LAUNCH.md"
    launch_record.write_text(
        "\n".join(
            [
                f"# Launch {run_id}",
                "",
                f"- profile: `{profile.name}`",
                f"- rationale: {profile.rationale}",
                f"- adaptive overrides applied: `{bool(env_overrides)}`",
                f"- work dir: `{work_dir}`",
                f"- checkpoint dir: `{ckpt_dir}`",
                f"- W&B: `https://wandb.ai/{args.wandb_entity}/{args.wandb_project}/runs/{run_id}`",
                "",
                "## Adaptive Overrides",
                "",
                "```json",
                json.dumps(env_overrides or {}, indent=2, sort_keys=True),
                "```",
                "",
                "## Effective Environment",
                "",
                "```json",
                json.dumps(env, indent=2, sort_keys=True),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    command = (
        f"cd {shlex.quote(str(work_dir))} && "
        f"set -o pipefail && "
        f"{shell_env(env)} "
        f"{shlex.quote(args.conda_bin)} run --no-capture-output -n {shlex.quote(args.conda_env)} "
        f"python {shlex.quote(str(train_gpt))} 2>&1 | tee train.log"
    )
    command_path = notes_dir / f"RUN-{run_index:03d}-{run_id}-command.sh"
    command_path.write_text(command + "\n", encoding="utf-8")
    if args.dry_run:
        print(command)
        return 0
    print(f"[{utc_iso()}] launching {run_id} with {profile.name}", flush=True)
    proc = subprocess.run(["bash", "-lc", command], cwd=repo)
    return int(proc.returncode)


def run_convextok_regret_analysis(args: argparse.Namespace, output_dir: Path) -> None:
    if not str(args.tokenizer_path).endswith(".convextok.json"):
        return
    repo = Path(args.repo_root).resolve()
    tokenizer_path = (repo / args.tokenizer_path).resolve() if not Path(args.tokenizer_path).is_absolute() else Path(args.tokenizer_path)
    docs_jsonl = (repo / args.matched_docs_jsonl).resolve() if not Path(args.matched_docs_jsonl).is_absolute() else Path(args.matched_docs_jsonl)
    baseline = (
        (repo / args.sentencepiece_baseline_tokenizer).resolve()
        if not Path(args.sentencepiece_baseline_tokenizer).is_absolute()
        else Path(args.sentencepiece_baseline_tokenizer)
    )
    parquet_patterns = [str(pattern) for pattern in getattr(args, "matched_docs_parquet", []) if str(pattern).strip()]
    if not docs_jsonl.exists() and not parquet_patterns:
        note = output_dir / "convextok_regret_missing_docs.md"
        note.write_text(
            "\n".join(
                [
                    "# ConvexTok Regret Analysis Not Run",
                    "",
                    f"- expected matched docs JSONL: `{docs_jsonl}`",
                    "- no `--matched-docs-parquet` globs were provided.",
                    "- reason: no document sample source was present for this run.",
                    "- action: run the matched FineWeb ConvexTok export or pass local Parquet globs from the shared ToricGT corpus.",
                ]
            ),
            encoding="utf-8",
        )
        return
    command = [
        args.conda_bin,
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "scripts/analyze_convextok_regret.py",
        "--convextok-tokenizer",
        str(tokenizer_path),
        "--sentencepiece-tokenizer",
        str(baseline),
        "--max-docs",
        "24",
        "--max-doc-bytes",
        "2048",
        "--max-candidates",
        "1800",
        "--output-json",
        str(output_dir / "convextok_regret.json"),
        "--dag-output-json",
        str(output_dir / "convextok_tokenisation_dag.json"),
    ]
    if docs_jsonl.exists():
        command.extend(["--docs-jsonl", str(docs_jsonl)])
    else:
        for pattern in parquet_patterns:
            command.extend(["--docs-parquet", pattern])
        command.extend(["--parquet-text-column", str(args.matched_docs_parquet_text_column)])
    (output_dir / "convextok_regret_command.json").write_text(
        json.dumps(command, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "convextok_regret.stdout.log").open("w", encoding="utf-8") as stdout, (
        output_dir / "convextok_regret.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=repo, stdout=stdout, stderr=stderr, text=True, check=False)


def run_full_iteration_analysis(
    args: argparse.Namespace,
    *,
    run_id: str,
    run_index: int,
    metrics: dict[str, float | int | str | None],
) -> tuple[int, dict[str, object]]:
    if not args.full_analysis:
        return 0, {}
    repo = Path(args.repo_root).resolve()
    notes_dir = repo / "training_notes" / args.campaign_id
    log_path = repo / "runs" / "oai_sidecar" / run_id / "train.log"
    checkpoint = metrics.get("checkpoint_path")
    if not checkpoint:
        raise FileNotFoundError(f"No checkpoint path was parsed from {log_path}; cannot run full analysis")
    checkpoint_path = Path(str(checkpoint))
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo / checkpoint_path
    run_path = f"{args.wandb_entity}/{args.wandb_project}/{run_id}"
    graph_train_glob = str(Path(args.graph_data_path) / "train" / "*.parquet")
    max_attempts = 1 + max(0, int(args.analysis_retries))
    last_returncode = 0
    last_decision: dict[str, object] = {}
    failure_records: list[dict[str, object]] = []
    for attempt in range(max_attempts):
        suffix = "full-analysis" if attempt == 0 else f"full-analysis-retry{attempt:02d}"
        output_dir = notes_dir / f"RUN-{run_index:03d}-{run_id}-{suffix}"
        output_dir.mkdir(parents=True, exist_ok=True)
        timeout = int(
            round(
                float(args.analysis_cas_macaulay2_timeout_seconds)
                * (float(args.analysis_retry_timeout_multiplier) ** attempt)
            )
        )
        command = [
            args.conda_bin,
            "run",
            "--no-capture-output",
            "-n",
            args.conda_env,
            "env",
            f"PYTHONPATH={repo / 'src'}",
            "python",
            "scripts/run_oai_sidecar_full_iteration_analysis.py",
            "--run-id",
            run_id,
            "--run-path",
            run_path,
            "--checkpoint",
            str(checkpoint_path),
            "--train-log",
            str(log_path),
            "--output-dir",
            str(output_dir),
            "--tokenizer-path",
            args.tokenizer_path,
            "--graph-train-glob",
            graph_train_glob,
            "--records",
            str(args.analysis_records),
            "--embedding-records",
            str(args.analysis_records),
            "--cas-max-points",
            str(args.analysis_cas_max_points),
            "--cas-macaulay2-timeout-seconds",
            str(timeout),
        ]
        if args.strict_analysis:
            command.append("--strict")
        else:
            command.append("--no-strict")
        stdout_path = notes_dir / f"RUN-{run_index:03d}-{run_id}-{suffix}.stdout.log"
        stderr_path = notes_dir / f"RUN-{run_index:03d}-{run_id}-{suffix}.stderr.log"
        command_path = notes_dir / f"RUN-{run_index:03d}-{run_id}-{suffix}-command.json"
        command_path.write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{utc_iso()}] running full analysis attempt {attempt + 1}/{max_attempts} "
            f"for {run_id} with Macaulay2 timeout {timeout}s",
            flush=True,
        )
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(command, cwd=repo, stdout=stdout, stderr=stderr, text=True)
        last_returncode = int(proc.returncode)
        decision_path = output_dir / "next_profile_decision.json"
        last_decision = {}
        if decision_path.exists():
            try:
                last_decision = json.loads(decision_path.read_text(encoding="utf-8"))
            except Exception as exc:
                last_decision = {"next_profile_hint": "", "reason": f"decision JSON parse failed: {exc}"}
        if last_returncode == 0:
            run_convextok_regret_analysis(args, output_dir)
            if failure_records:
                retry_success = notes_dir / f"RUN-{run_index:03d}-{run_id}-FULL-ANALYSIS-RETRY-SUCCEEDED.md"
                retry_success.write_text(
                    "\n".join(
                        [
                            "# Full Analysis Retry Succeeded",
                            "",
                            f"- run id: `{run_id}`",
                            f"- successful attempt: `{attempt + 1}`",
                            f"- output dir: `{output_dir}`",
                            "",
                            "## Previous Failed Attempts",
                            "",
                            "```json",
                            json.dumps(failure_records, indent=2, sort_keys=True),
                            "```",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
            return 0, last_decision
        failure_records.append(
            {
                "attempt": attempt + 1,
                "returncode": last_returncode,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "output_dir": str(output_dir),
                "command": str(command_path),
                "macaulay2_timeout_seconds": timeout,
            }
        )
        print(
            f"[{utc_iso()}] full analysis attempt {attempt + 1}/{max_attempts} failed for {run_id}; "
            f"returncode={last_returncode}",
            flush=True,
        )
    failure = notes_dir / f"RUN-{run_index:03d}-{run_id}-FULL-ANALYSIS-FAILED.md"
    failure.write_text(
        "\n".join(
            [
                "# Full Analysis Failed After Retries",
                "",
                f"- run id: `{run_id}`",
                f"- final return code: `{last_returncode}`",
                "",
                "The campaign is intentionally blocked here so that the next run is not launched without complete metric and visualization review.",
                "",
                "## Attempts",
                "",
                "```json",
                json.dumps(failure_records, indent=2, sort_keys=True),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return int(last_returncode), last_decision


def write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}", "_path": str(path)}
    return payload if isinstance(payload, dict) else {"_payload": payload, "_path": str(path)}


def run_index_from_id(campaign_id: str, run_id: str) -> int | None:
    match = re.search(rf"{re.escape(campaign_id)}-r(?P<idx>\d{{3}})-", run_id)
    if not match:
        return None
    return int(match.group("idx"))


def profile_from_run_id(campaign_id: str, run_id: str) -> str:
    match = re.match(rf"{re.escape(campaign_id)}-r\d{{3}}-(?P<profile>.*)-\d{{8}}T\d{{6}}Z$", run_id)
    if match:
        return match.group("profile")
    return "unknown"


def completed_run_indices(history: list[dict[str, object]], campaign_id: str) -> set[int]:
    indices: set[int] = set()
    for row in history:
        run_id = str(row.get("run_id", ""))
        idx = run_index_from_id(campaign_id, run_id)
        if idx is not None:
            indices.add(idx)
    return indices


def recover_logged_runs(repo: Path, args: argparse.Namespace, history: list[dict[str, object]]) -> list[dict[str, object]]:
    """Recover completed run directories that were logged but not persisted.

    This protects long campaigns from controller-side parsing failures.  It does
    not invent metrics: only existing `train.log` files under `runs/oai_sidecar`
    are parsed, and recovered rows are marked explicitly in the state.
    """

    campaign_id = str(args.campaign_id)
    known = completed_run_indices(history, campaign_id)
    run_root = repo / "runs" / "oai_sidecar"
    recovered: list[dict[str, object]] = []
    if not run_root.exists():
        return history
    candidates: list[tuple[int, Path]] = []
    for path in run_root.glob(f"{campaign_id}-r[0-9][0-9][0-9]-*"):
        if not path.is_dir():
            continue
        idx = run_index_from_id(campaign_id, path.name)
        if idx is None or idx in known:
            continue
        log_path = path / "train.log"
        if not log_path.exists():
            continue
        metrics = parse_log(log_path)
        if not metrics.get("checkpoint_path") and not metrics.get("final_int8_bpb"):
            continue
        candidates.append((idx, path))
    for idx, path in sorted(candidates, key=lambda item: item[0]):
        metrics = parse_log(path / "train.log")
        profile_name = profile_from_run_id(campaign_id, path.name)
        row = {
            "run_id": path.name,
            "profile": profile_name,
            "env_overrides_used": {},
            "metrics": metrics,
            "returncode": 0 if metrics.get("checkpoint_path") or metrics.get("final_int8_bpb") else 1,
            "analysis_returncode": 0,
            "analysis_decision": {
                "next_profile_hint": "",
                "reason": "Recovered from existing train.log after controller restart; no full-analysis decision was available.",
            },
            "recovered_from_log": True,
        }
        history.append(row)
        recovered.append(row)
    if recovered:
        print(
            f"[{utc_iso()}] recovered {len(recovered)} completed run(s) from logs: "
            + ", ".join(str(row.get("run_id")) for row in recovered),
            flush=True,
        )
    return history


def finite_metric(row: dict[str, object], key: str) -> float | None:
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def completed_campaign_rows(history: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in history
        if row.get("profile") != "prior_external" and isinstance(row.get("metrics"), dict)
    ]


def mean(values: list[float]) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return None
    return sum(values) / len(values)


def profile_summary(history: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in completed_campaign_rows(history):
        grouped.setdefault(str(row.get("profile", "unknown")), []).append(row)
    summaries: list[dict[str, object]] = []
    for profile, rows in sorted(grouped.items()):
        bpbs = [metric_bpb(row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}) for row in rows]
        bpbs = [value for value in bpbs if math.isfinite(value)]
        train_bpbs = [finite_metric(row, "train_bpb") for row in rows]
        graph_lm_bpbs = [finite_metric(row, "graph_lm_bpb") for row in rows]
        graph_lm_weights = [finite_metric(row, "graph_lm_weight") for row in rows]
        artifacts = [finite_metric(row, "artifact_bytes") for row in rows]
        sidecars = [finite_metric(row, "sidecar_loss") for row in rows]
        fot_losses = [finite_metric(row, "oai_fot_loss") for row in rows]
        fot_rewards = [finite_metric(row, "oai_fot_reward") for row in rows]
        fot_diversity = [finite_metric(row, "oai_fot_diversity") for row in rows]
        toric = [finite_metric(row, "toric_geometry_loss") or finite_metric(row, "toric") for row in rows]
        vector_bundle_1d = [
            finite_metric(row, "toric_vector_bundle_1d_cone_ce_loss") or finite_metric(row, "vector_bundle_1d_cone")
            for row in rows
        ]
        bgg = [finite_metric(row, "bgg_loss") or finite_metric(row, "bgg") for row in rows]
        derived = [finite_metric(row, "derived_signature_loss") or finite_metric(row, "derived_signature") for row in rows]
        summary = {
            "profile": profile,
            "runs": len(rows),
            "best_bpb": min(bpbs) if bpbs else None,
            "mean_bpb": mean(bpbs),
            "worst_bpb": max(bpbs) if bpbs else None,
            "mean_train_bpb": mean([v for v in train_bpbs if v is not None]),
            "mean_graph_lm_bpb": mean([v for v in graph_lm_bpbs if v is not None]),
            "mean_graph_lm_weight": mean([v for v in graph_lm_weights if v is not None]),
            "mean_artifact_bytes": mean([v for v in artifacts if v is not None]),
            "mean_sidecar_loss": mean([v for v in sidecars if v is not None]),
            "mean_oai_fot_loss": mean([v for v in fot_losses if v is not None]),
            "mean_oai_fot_reward": mean([v for v in fot_rewards if v is not None]),
            "mean_oai_fot_diversity": mean([v for v in fot_diversity if v is not None]),
            "mean_toric_loss": mean([v for v in toric if v is not None]),
            "mean_toric_vector_bundle_1d_cone_ce_loss": mean([v for v in vector_bundle_1d if v is not None]),
            "mean_bgg_loss": mean([v for v in bgg if v is not None]),
            "mean_derived_signature_loss": mean([v for v in derived if v is not None]),
        }
        summaries.append(summary)
    summaries.sort(key=lambda row: float(row["best_bpb"]) if isinstance(row.get("best_bpb"), (int, float)) else float("inf"))
    return summaries


def profile_env_markdown(profile_name: str) -> list[str]:
    profile = profile_by_name(profile_name)
    if profile is None:
        return [f"- profile `{profile_name}` is not in the current profile table."]
    active_advanced = {
        key: value
        for key, value in profile.env.items()
        if key.endswith("_LOSS_WEIGHT") and key not in {"GRAPHCG_LOSS_WEIGHT", "ANALOGY_LOSS_WEIGHT", "TOKENGT_GRAPH_LOSS_WEIGHT", "TRAJECTORY_MEMORY_LOSS_WEIGHT"}
    }
    return [
        f"- rationale: {profile.rationale}",
        f"- LR: matrix `{profile.env.get('MATRIX_LR')}`, scalar `{profile.env.get('SCALAR_LR')}`, tied embedding `{profile.env.get('TIED_EMBED_LR')}`",
        f"- first-class FineWeb TokenGT: lr `{profile.env.get('TOKENGT_FIRST_CLASS_LR')}`, radius `{profile.env.get('TOKENGT_GRAPH_RADIUS')}`, distance features `{profile.env.get('TOKENGT_DISTANCE_FEATURES', 'functional/base default')}`, structural `{profile.env.get('TOKENGT_STRUCTURAL_WEIGHT')}`, one-dimensional-edge `{profile.env.get('TOKENGT_EDGE_WEIGHT')}`, torus `{profile.env.get('TOKENGT_TORUS_WEIGHT')}`, identifier `{profile.env.get('TOKENGT_IDENTIFIER_WEIGHT', 'base default')}`, endpoint `{profile.env.get('TOKENGT_ENDPOINT_WEIGHT', 'base default')}`, edge-token `{profile.env.get('TOKENGT_EDGE_TOKEN_WEIGHT', 'base default')}`",
        f"- OAI-FineWeb-only graph-output flattening: lr `{profile.env.get('GRAPH_OUTPUT_FLATTENING_LR', 'base default')}`, radius `{profile.env.get('GRAPH_OUTPUT_EDGE_RADIUS', 'base default')}`, distance features `{profile.env.get('GRAPH_OUTPUT_DISTANCE_FEATURES', 'functional/base default')}`, node `{profile.env.get('GRAPH_OUTPUT_NODE_WEIGHT', 'base default')}`, edge `{profile.env.get('GRAPH_OUTPUT_EDGE_WEIGHT', 'base default')}`, virtual-edge `{profile.env.get('GRAPH_OUTPUT_EDGE_TOKEN_WEIGHT', 'base default')}`, score-correction `{profile.env.get('GRAPH_OUTPUT_SCORE_CORRECTION_WEIGHT', 'base default')}`",
        "- graph radius policy: BPB-gated recovery profiles now start at local radius 4; the adaptive analysis may later widen to 5-6 only when BPB slope, graph correlations, and flattening calibration support wider neighborhoods.",
        "- flattening calibration: enabled by base environment with a small regression-only loss that penalizes graph-output flattening only when it worsens FineWeb CE versus the raw graph hidden state.",
        f"- bonafide graph LM primary weight: `{profile.env.get('GRAPH_LM_LOSS_WEIGHT', 'base default')}`",
        f"- embedding Forest-of-Thought: enabled `{profile.env.get('OAI_EMBEDDING_FOT', 'base default')}`, weight `{profile.env.get('OAI_FOT_LOSS_WEIGHT', 'base default')}`, trees `{profile.env.get('OAI_FOT_NUM_TREES', 'base default')}`, depth `{profile.env.get('OAI_FOT_MAX_DEPTH', 'base default')}`, branching `{profile.env.get('OAI_FOT_BRANCHING', 'base default')}`",
        f"- sidecar LR: `{profile.env.get('TORICGT_SIDECAR_LR')}`",
        f"- base sidecar weights: graphcg `{profile.env.get('GRAPHCG_LOSS_WEIGHT')}`, analogy `{profile.env.get('ANALOGY_LOSS_WEIGHT')}`, tokengt_graph `{profile.env.get('TOKENGT_GRAPH_LOSS_WEIGHT')}`, memory `{profile.env.get('TRAJECTORY_MEMORY_LOSS_WEIGHT')}`",
        f"- advanced weights: `{json.dumps(active_advanced, sort_keys=True)}`",
        f"- controllers: Lagrangian `{profile.env.get('ADVANCED_LAGRANGIAN_CONTROLLER', 'base default')}`, toric fan curriculum `{profile.env.get('TORIC_FAN_CURRICULUM', 'base default')}`, derived signature `{profile.env.get('DERIVED_SIGNATURE_LOSS_WEIGHT', 'base default')}`, memory sheaf gate threshold `{profile.env.get('MEMORY_SHEAF_GATE_THRESHOLD', 'base default')}`",
    ]


def write_meta_analysis(notes_dir: Path, history: list[dict[str, object]], args: argparse.Namespace, *, phase: str) -> Path:
    rows = completed_campaign_rows(history)
    summaries = profile_summary(history)
    best = min(rows, key=lambda row: metric_bpb(row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}), default=None)
    worst = max(rows, key=lambda row: metric_bpb(row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}), default=None)
    best_profile = str(summaries[0]["profile"]) if summaries else ""
    table = [
        "| profile | runs | best BPB | mean BPB | mean train BPB | mean graph-LM BPB | mean graph-LM weight | mean FoT loss | mean FoT reward | mean FoT diversity | worst BPB | mean artifact bytes | mean sidecar |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        table.append(
            "| {profile} | {runs} | {best} | {avg} | {train_avg} | {graph_avg} | {graph_weight} | {fot_loss} | {fot_reward} | {fot_div} | {worst} | {artifact} | {sidecar} |".format(
                profile=row["profile"],
                runs=row["runs"],
                best=f"{row['best_bpb']:.6f}" if isinstance(row.get("best_bpb"), (int, float)) else "n/a",
                avg=f"{row['mean_bpb']:.6f}" if isinstance(row.get("mean_bpb"), (int, float)) else "n/a",
                train_avg=f"{row['mean_train_bpb']:.6f}" if isinstance(row.get("mean_train_bpb"), (int, float)) else "n/a",
                graph_avg=f"{row['mean_graph_lm_bpb']:.6f}" if isinstance(row.get("mean_graph_lm_bpb"), (int, float)) else "n/a",
                graph_weight=f"{row['mean_graph_lm_weight']:.4f}" if isinstance(row.get("mean_graph_lm_weight"), (int, float)) else "n/a",
                fot_loss=f"{row['mean_oai_fot_loss']:.6g}" if isinstance(row.get("mean_oai_fot_loss"), (int, float)) else "n/a",
                fot_reward=f"{row['mean_oai_fot_reward']:.6g}" if isinstance(row.get("mean_oai_fot_reward"), (int, float)) else "n/a",
                fot_div=f"{row['mean_oai_fot_diversity']:.6g}" if isinstance(row.get("mean_oai_fot_diversity"), (int, float)) else "n/a",
                worst=f"{row['worst_bpb']:.6f}" if isinstance(row.get("worst_bpb"), (int, float)) else "n/a",
                artifact=f"{row['mean_artifact_bytes']:.0f}" if isinstance(row.get("mean_artifact_bytes"), (int, float)) else "n/a",
                sidecar=f"{row['mean_sidecar_loss']:.6g}" if isinstance(row.get("mean_sidecar_loss"), (int, float)) else "n/a",
            )
        )
    analysis_gate_sentence = (
        "Use strict exact analysis as a gate. A failed GUDHI/Sage/Macaulay2 or screenshot bundle blocks the next run unless a retry succeeds."
        if bool(args.strict_analysis)
        else "Keep analyses non-blocking for this sweep. A failed GUDHI/Sage/Macaulay2 or screenshot bundle should be recorded, but it must not prevent the configured run budget from completing."
    )
    recommendations = [
        "Keep `TRAIN_SEQ_LEN=1024` and `grad_accum_steps=8`; those are the current speed and memory controls.",
        "Keep every advanced metric family enabled in logging. Sweep training pressure through profile weights, not by disabling observability.",
        "Reject profiles that improve auxiliary diagnostics while worsening exported int8 BPB or pushing the artifact near the 16,000,000-byte cap.",
        "Favor the best BPB profile and its nearest LR/auxiliary-weight neighbors for the follow-up phase; do not jump to high auxiliary weights unless BPB and artifact margin justify it.",
        analysis_gate_sentence,
    ]
    if best_profile:
        recommendations.append(f"Current best evidence favors `{best_profile}` as the anchor for the next hyperparameter neighborhood.")
    meta_path = notes_dir / f"{phase}-META-ANALYSIS-{utc_stamp()}.md"
    repo_root = notes_dir.parents[1] if len(notes_dir.parents) > 1 else Path.cwd()
    planning_dir = repo_root / "planning"
    planning_dir.mkdir(parents=True, exist_ok=True)
    planning_path = planning_dir / f"{phase.upper()}-BPB119-FOLLOWUP-PLAN-{utc_stamp()}.md"
    best_profile_details = profile_env_markdown(best_profile) if best_profile else ["No completed profile rows are available."]
    mechanism_review = [
        "For the next phase, do not treat advanced training as one undifferentiated auxiliary block.",
        "Review `bpb_aux_control/stage_multiplier`, slope and curvature metrics to decide whether the BPB-first stage is too protective or too permissive.",
        "Review each `aux_grad_routing/*_cosine`, `*_route_scale`, and `*_projection_coeff` to identify families that are repeatedly aligned, neutral, or destructive.",
        "Review `oai_fot/reward_bpb_delta`, `oai_fot/reward_corrected_byte_nll`, `oai_fot/runtime_after_*`, entropy, and diversity together; high entropy with poor BPB-delta reward means less exploration, not more FoT loss.",
        "Keep exploration broad across run families.  The current evidence base is short-run and sparse, so the follow-up should test several theories rather than collapse onto one profile.",
    ]
    lines = [
        f"# {phase.replace('-', ' ').title()} Meta-Analysis",
        "",
        f"- generated UTC: `{utc_iso()}`",
        f"- target BPB: `{args.target_bpb}`",
        f"- completed campaign rows: `{len(rows)}`",
        f"- best run: `{best.get('run_id') if best else 'n/a'}`",
        f"- best BPB: `{metric_bpb(best.get('metrics', {}) if best and isinstance(best.get('metrics'), dict) else {}) if best else 'n/a'}`",
        f"- worst run: `{worst.get('run_id') if worst else 'n/a'}`",
        "",
        "## Profile Comparison",
        "",
        *table,
        "",
        "## Best Profile Details",
        "",
        *best_profile_details,
        "",
        "## Training Adjustment Plan",
        "",
        *[f"- {item}" for item in recommendations],
        "",
        "## Mechanism Review Requirements",
        "",
        *[f"- {item}" for item in mechanism_review],
        "",
        "## Raw History",
        "",
        "```json",
        json.dumps(rows, indent=2, sort_keys=True),
        "```",
        "",
    ]
    meta_path.write_text("\n".join(lines), encoding="utf-8")
    planning_lines = [
        f"# {phase.replace('-', ' ').title()} Follow-Up BPB Plan",
        "",
        f"- generated UTC: `{utc_iso()}`",
        f"- source meta-analysis: `{meta_path}`",
        f"- target BPB: `{args.target_bpb}`",
        f"- completed rows: `{len(rows)}`",
        "",
        "## Required Mindset",
        "",
        "The follow-up campaign remains exploratory.  We have not yet observed a dramatic BPB drop from the new FoT path, so the agent should form and test multiple mechanistic hypotheses instead of repeating one conservative profile.",
        "",
        "## Evidence Summary",
        "",
        *table,
        "",
        "## Hypotheses To Test",
        "",
        "1. BPB-first staging may improve early CE descent by preventing non-BPB auxiliary objectives from rotating the first optimizer steps.",
        "2. If route scales repeatedly damp the same family, lower that family's configured weight or delay it; if a family is repeatedly aligned, test a bounded increase.",
        "3. FoT should be judged by BPB-delta reward and corrected byte NLL.  Structural reward without byte-likelihood lift should not justify higher FoT weight.",
        "4. If graph-LM BPB is already easy while FineWeb BPB is poor, slow graph-LM ramp or lower graph mixture pressure while keeping graphification features active.",
        "5. If graph-output flattening CE lift is positive, keep the flattening path and test stronger residual score correction; if it regresses, keep calibration and lower graph-output weights.",
        "",
        "## Follow-Up Actions",
        "",
        *[f"- {item}" for item in recommendations],
        "",
        "## Mechanism Review Requirements",
        "",
        *[f"- {item}" for item in mechanism_review],
        "",
        "## Raw History",
        "",
        "```json",
        json.dumps(rows, indent=2, sort_keys=True),
        "```",
        "",
    ]
    planning_path.write_text("\n".join(planning_lines), encoding="utf-8")
    return meta_path


def best_completed_row(history: list[dict[str, object]]) -> dict[str, object] | None:
    rows = completed_campaign_rows(history)
    finite_rows = [
        row
        for row in rows
        if math.isfinite(metric_bpb(row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}))
    ]
    if not finite_rows:
        return None
    return min(finite_rows, key=lambda row: metric_bpb(row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}))


def read_hf_token(token_file: Path) -> str:
    if not token_file.exists():
        raise FileNotFoundError(f"HF token file does not exist: {token_file}")
    text = token_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"hf_[A-Za-z0-9_\\-]+", text)
    if not match:
        raise RuntimeError(f"HF token file did not contain an hf_* token: {token_file}")
    return match.group(0)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def upload_best_short_checkpoint_to_hf(
    args: argparse.Namespace,
    notes_dir: Path,
    best: dict[str, object],
    *,
    phase: str,
) -> int:
    repo = Path(args.repo_root).resolve()
    metrics = best.get("metrics", {})
    if not isinstance(metrics, dict):
        return 1
    run_id = str(best.get("run_id", "unknown-run"))
    profile = str(best.get("profile", "unknown-profile"))
    checkpoint = Path(str(metrics.get("checkpoint_path") or ""))
    train_log = Path(str(metrics.get("path") or ""))
    run_dir = train_log.parent if train_log.exists() else repo / "runs" / "oai_sidecar" / run_id
    staging = repo / "outputs" / "hf_toricgt_checkpoints" / f"{phase}-{run_id}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    if copy_if_exists(checkpoint, staging / "checkpoints" / checkpoint.name):
        copied.append(f"checkpoints/{checkpoint.name}")
    for filename in ("final_model.int8.ptz", "final_model.pt", "train.log"):
        src = run_dir / filename
        if copy_if_exists(src, staging / filename):
            copied.append(filename)
    for src in sorted((run_dir / "logs").glob("*")) if (run_dir / "logs").exists() else []:
        if src.is_file() and copy_if_exists(src, staging / "logs" / src.name):
            copied.append(f"logs/{src.name}")

    (staging / "RUN_ID.txt").write_text(run_id + "\n", encoding="utf-8")
    card_lines = [
        "# ToricGT Checkpoint",
        "",
        f"- uploaded UTC: `{utc_iso()}`",
        f"- source campaign: `{args.campaign_id}`",
        f"- source phase: `{phase}`",
        f"- selected run: `{run_id}`",
        f"- profile: `{profile}`",
        f"- profile rationale: `{profile_by_name(profile).rationale if profile_by_name(profile) else 'profile not found in current table'}`",
        f"- checkpoint step: `{metrics.get('checkpoint_step')}`",
        f"- train BPB: `{metrics.get('train_bpb')}`",
        f"- validation BPB: `{metrics.get('val_bpb')}`",
        f"- int8+zlib round-trip BPB: `{metrics.get('final_int8_bpb')}`",
        f"- compressed artifact bytes: `{metrics.get('artifact_bytes')}`",
        f"- checkpoint path in source workspace: `{metrics.get('checkpoint_path')}`",
        "",
        "## Selection Rule",
        "",
        "This artifact was selected as the best completed short run in the configured fixed-budget sweep, using exported/int8 BPB when available and validation BPB otherwise. No hard target threshold was used for early stopping.",
        "",
        "## Active Techniques",
        "",
        "- deterministic ConvexTok-2048 tokenization",
        "- first-class TokenGT graphification of FineWeb with OAI-only sequential flattening for BPB scoring",
        "- tokenization-DAG, min-plus/tropical path, and toric vocabulary-face features",
        "- graph-output score correction and calibration",
        "- embedding-space GFlowNet and Forest-of-Thought heads",
        "- multi-token prediction",
        "- full-rank GraphCG, trajectory-memory retrieval, toric geometry, vector-bundle 1D-cone/sheaf, Toric BGG category-O, Koszul persistence, combinatorial toric commutative algebra, and derived-signature metrics/losses",
        "",
        "## Copied Files",
        "",
        *[f"- `{item}`" for item in copied],
        "",
    ]
    (staging / "README.md").write_text("\n".join(card_lines), encoding="utf-8")

    hf_bin = shutil.which("hf") or "/home/iska/.local/bin/hf"
    token_file = Path(args.hf_token_file)
    if not token_file.is_absolute():
        token_file = repo / token_file
    upload_report = notes_dir / f"{phase}-HF-UPLOAD-{utc_stamp()}.md"
    try:
        token = read_hf_token(token_file)
        env = dict(os.environ)
        env["HF_TOKEN"] = token
        create = subprocess.run(
            [hf_bin, "repos", "create", str(args.hf_checkpoint_repo), "--type", "model", "--exist-ok"],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        upload = subprocess.run(
            [
                hf_bin,
                "upload",
                str(args.hf_checkpoint_repo),
                str(staging),
                ".",
                "--type",
                "model",
                "--commit-message",
                f"Upload best ToricGT short sweep checkpoint {run_id}",
            ],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        status = 0 if create.returncode == 0 and upload.returncode == 0 else 1
        upload_report.write_text(
            "\n".join(
                [
                    "# Hugging Face Best Short-Run Upload",
                    "",
                    f"- generated UTC: `{utc_iso()}`",
                    f"- repo: `{args.hf_checkpoint_repo}`",
                    f"- run: `{run_id}`",
                    f"- staging: `{staging}`",
                    f"- create return code: `{create.returncode}`",
                    f"- upload return code: `{upload.returncode}`",
                    "",
                    "## Upload Output",
                    "",
                    "```text",
                    upload.stdout.strip(),
                    "```",
                    "",
                    "## Upload Errors",
                    "",
                    "```text",
                    upload.stderr.strip(),
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        if status == 0:
            print(f"[{utc_iso()}] uploaded best short checkpoint {run_id} to {args.hf_checkpoint_repo}", flush=True)
        else:
            print(f"[{utc_iso()}] HF upload failed for {run_id}; see {upload_report}", flush=True)
        return status
    except Exception as exc:
        upload_report.write_text(
            f"# Hugging Face Upload Failed\n\n- generated UTC: `{utc_iso()}`\n- run: `{run_id}`\n- error: `{type(exc).__name__}: {exc}`\n",
            encoding="utf-8",
        )
        print(f"[{utc_iso()}] HF upload failed for {run_id}: {type(exc).__name__}: {exc}", flush=True)
        return 1


SUBMISSION_TORICGT_MODULES = [
    "embedding_forest_of_thought.py",
    "oai_sidecar.py",
    "combinatorial_toric_metrics.py",
    "derived_category_metrics.py",
    "koszul_persistence.py",
    "trajectory_memory.py",
    "toric_bgg.py",
    "toric_geometry_tasks.py",
    "toric_vector_bundles.py",
    "symbolic_multigraded_resolution.py",
    "topological_reasoning.py",
    "gudhi_persistence.py",
    "got_trajectory.py",
    "cas_backed_losses.py",
    "cas_certificates.py",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_run_dir(repo: Path, row: dict[str, object]) -> Path:
    metrics = row.get("metrics", {})
    if isinstance(metrics, dict):
        log_path = metrics.get("path")
        if isinstance(log_path, str) and log_path:
            path = Path(log_path)
            if not path.is_absolute():
                path = repo / path
            if path.exists():
                return path.parent
    return repo / "runs" / "oai_sidecar" / str(row.get("run_id", ""))


def submission_source_files(args: argparse.Namespace) -> list[Path]:
    repo = Path(args.repo_root).resolve()
    pg_repo = repo / args.parameter_golf_repo if not Path(args.parameter_golf_repo).is_absolute() else Path(args.parameter_golf_repo)
    tokenizer = Path(args.tokenizer_path)
    if not tokenizer.is_absolute():
        tokenizer = repo / tokenizer
    files = [
        pg_repo / "train_gpt.py",
        pg_repo / "convextok.py",
        pg_repo / "requirements.txt",
        tokenizer,
    ]
    source_pkg = repo / "src" / "toricgt"
    files.extend(source_pkg / name for name in SUBMISSION_TORICGT_MODULES)
    return files


def submission_code_bytes(args: argparse.Namespace) -> int:
    total = 0
    for path in submission_source_files(args):
        if path.exists():
            total += path.stat().st_size
    total += len("# minimal package marker for Parameter Golf record\n".encode("utf-8"))
    total += 512  # run_submission.sh reserve; exact record bytes are recomputed after writing.
    return total


def row_model_artifact_bytes(repo: Path, row: dict[str, object]) -> int | None:
    artifact = row_run_dir(repo, row) / "final_model.int8.ptz"
    if artifact.exists():
        return artifact.stat().st_size
    metrics = row.get("metrics", {})
    if isinstance(metrics, dict):
        value = metrics.get("artifact_bytes")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return int(value)
    return None


def row_self_contained_submission_bytes(args: argparse.Namespace, row: dict[str, object]) -> int | None:
    repo = Path(args.repo_root).resolve()
    model_bytes = row_model_artifact_bytes(repo, row)
    if model_bytes is None:
        return None
    return int(model_bytes) + submission_code_bytes(args)


def best_submission_row(args: argparse.Namespace, history: list[dict[str, object]]) -> dict[str, object] | None:
    candidates: list[dict[str, object]] = []
    for row in completed_campaign_rows(history):
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        if not math.isfinite(metric_bpb(metrics)):
            continue
        total_bytes = row_self_contained_submission_bytes(args, row)
        if total_bytes is None or total_bytes > 16_000_000:
            continue
        final_artifact = row_run_dir(Path(args.repo_root).resolve(), row) / "final_model.int8.ptz"
        train_log = row_run_dir(Path(args.repo_root).resolve(), row) / "train.log"
        if final_artifact.exists() and train_log.exists():
            candidates.append(row)
    if not candidates:
        return None
    return min(candidates, key=lambda row: metric_bpb(row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}))


def seed_for_row(args: argparse.Namespace, row: dict[str, object], fallback: int) -> int:
    idx = run_index_from_id(str(args.campaign_id), str(row.get("run_id", "")))
    if idx is None:
        return fallback
    return 1337 + idx


def launch_submission_step_candidate(
    args: argparse.Namespace,
    notes_dir: Path,
    best: dict[str, object],
    *,
    steps: int,
    run_index: int,
) -> dict[str, object] | None:
    if steps <= 0:
        return None
    profile = profile_by_name(str(best.get("profile", "")))
    if profile is None:
        raise RuntimeError(f"best run profile is not available in current profile table: {best.get('profile')}")
    env_overrides = best.get("env_overrides_used", {})
    if not isinstance(env_overrides, dict):
        env_overrides = {}
    env_overrides = {str(key): str(value) for key, value in env_overrides.items()}
    env_overrides["SEED"] = str(seed_for_row(args, best, fallback=1337 + run_index))
    run_id = short_run_id(f"{args.campaign_id}-submit{steps}", run_index, profile.name)
    original_steps = int(args.steps_per_run)
    args.steps_per_run = int(steps)
    note = notes_dir / f"SUBMISSION-STEP-{steps}-CANDIDATE-{utc_stamp()}.md"
    note.write_text(
        "\n".join(
            [
                f"# Submission Step-{steps} Candidate Rerun",
                "",
                f"- generated UTC: `{utc_iso()}`",
                f"- source best run: `{best.get('run_id')}`",
                f"- profile: `{profile.name}`",
                f"- rerun id: `{run_id}`",
                f"- steps: `{steps}`",
                f"- seed override: `{env_overrides.get('SEED')}`",
                "",
                "This is an exact rerun to test whether a pre-1K checkpoint/export is better than the terminal 1K export. It is not inferred from intermediate train BPB logs.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        print(f"[{utc_iso()}] launching exact {steps}-step submission candidate {run_id}", flush=True)
        returncode = launch_training(args, run_id, profile, run_index, env_overrides=env_overrides)
        metrics = parse_log(Path(args.repo_root).resolve() / "runs" / "oai_sidecar" / run_id / "train.log")
        row = {
            "run_id": run_id,
            "profile": profile.name,
            "env_overrides_used": env_overrides,
            "metrics": metrics,
            "returncode": returncode,
            "analysis_returncode": 0,
            "analysis_decision": {
                "next_profile_hint": "",
                "reason": f"Exact {steps}-step submission candidate rerun for Parameter Golf packaging.",
            },
            "submission_step_candidate": True,
        }
        report = notes_dir / f"SUBMISSION-STEP-{steps}-CANDIDATE-{run_id}-REPORT.md"
        write_report(
            report,
            f"Submission Step-{steps} Candidate Report",
            run_id,
            profile,
            metrics,
            None,
            returncode,
            row["analysis_decision"],
            target_bpb=float(args.target_bpb),
            env_overrides_used=env_overrides,
        )
        return row
    finally:
        args.steps_per_run = original_steps


def copy_submission_dependency_package(args: argparse.Namespace, record_dir: Path) -> list[str]:
    repo = Path(args.repo_root).resolve()
    source_pkg = repo / "src" / "toricgt"
    target_pkg = record_dir / "toricgt"
    target_pkg.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    init_text = "# Minimal ToricGT dependency package for this Parameter Golf record.\n"
    (target_pkg / "__init__.py").write_text(init_text, encoding="utf-8")
    copied.append("toricgt/__init__.py")
    for name in SUBMISSION_TORICGT_MODULES:
        src = source_pkg / name
        if not src.exists():
            raise FileNotFoundError(f"missing ToricGT dependency for submission: {src}")
        dst = target_pkg / name
        shutil.copy2(src, dst)
        copied.append(f"toricgt/{name}")
    return copied


def write_parameter_golf_submission(
    args: argparse.Namespace,
    notes_dir: Path,
    best: dict[str, object],
    history: list[dict[str, object]],
    *,
    phase: str,
) -> tuple[Path, Path]:
    repo = Path(args.repo_root).resolve()
    pg_repo = repo / args.parameter_golf_repo if not Path(args.parameter_golf_repo).is_absolute() else Path(args.parameter_golf_repo)
    if not pg_repo.exists():
        raise FileNotFoundError(f"Parameter Golf repo does not exist: {pg_repo}")
    record_slug = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}_ToricGT_ConvexTok2048_Graphified_FoT_BestOf10"
    record_dir = pg_repo / "records" / str(args.parameter_golf_track) / record_slug
    if record_dir.exists():
        record_dir = pg_repo / "records" / str(args.parameter_golf_track) / f"{record_slug}_{utc_stamp()}"
    record_dir.mkdir(parents=True, exist_ok=False)

    metrics = best.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    run_id = str(best.get("run_id", "unknown-run"))
    profile_name = str(best.get("profile", "unknown-profile"))
    run_dir = row_run_dir(repo, best)
    final_artifact = run_dir / "final_model.int8.ptz"
    train_log = run_dir / "train.log"
    tokenizer = Path(args.tokenizer_path)
    if not tokenizer.is_absolute():
        tokenizer = repo / tokenizer

    copied: list[str] = []
    for filename in ("train_gpt.py", "convextok.py", "requirements.txt"):
        src = pg_repo / filename
        if not src.exists():
            raise FileNotFoundError(f"missing Parameter Golf dependency: {src}")
        shutil.copy2(src, record_dir / filename)
        copied.append(filename)
    shutil.copy2(tokenizer, record_dir / tokenizer.name)
    copied.append(tokenizer.name)
    copied.extend(copy_submission_dependency_package(args, record_dir))
    run_script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            ": \"${DATA_PATH:?set DATA_PATH to the ConvexTok-2048 FineWeb shard directory}\"",
            "PYTHONPATH=. \\",
            f"TOKENIZER_PATH=./{tokenizer.name} \\",
            "VOCAB_SIZE=2048 \\",
            "FINEWEB_CASEOPS=0 \\",
            "FINEWEB_GRAPHIFY=1 TOKENGT_FIRST_CLASS=1 GRAPH_OUTPUT_FLATTENING=1 OAI_FINEWEB_OUTPUT_FLATTENING=1 \\",
            "OAI_GFLOWNET=1 OAI_EMBEDDING_FOT=1 OAI_MTP=1 \\",
            "TORICGT_SIDECAR=1 REQUIRE_TORICGT_SIDECAR=1 GRAPH_LM_PRIMARY=1 REQUIRE_GRAPH_LM_PRIMARY=1 \\",
            "python train_gpt.py \"$@\"",
            "",
        ]
    )
    (record_dir / "run_submission.sh").write_text(run_script, encoding="utf-8")
    (record_dir / "run_submission.sh").chmod(0o755)
    copied.append("run_submission.sh")
    if train_log.exists():
        shutil.copy2(train_log, record_dir / "train.log")
        copied.append("train.log")
    if args.parameter_golf_include_model_artifact and final_artifact.exists():
        shutil.copy2(final_artifact, record_dir / "final_model.int8.ptz")
        copied.append("final_model.int8.ptz")

    code_bytes = 0
    for path in record_dir.rglob("*"):
        if path.is_file() and path.name != "final_model.int8.ptz":
            code_bytes += path.stat().st_size
    model_bytes = final_artifact.stat().st_size if final_artifact.exists() else row_model_artifact_bytes(repo, best) or 0
    total_bytes = code_bytes + int(model_bytes)
    artifact_sha = sha256_file(final_artifact) if final_artifact.exists() else ""
    checkpoint_path = Path(str(metrics.get("checkpoint_path") or ""))
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path.exists() else ""
    history_rows = completed_campaign_rows(history)
    run_table = [
        "| run | profile | train BPB | val BPB | int8 BPB | self-contained bytes | status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in history_rows:
        row_metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        row_bytes = row_self_contained_submission_bytes(args, row)
        run_table.append(
            "| {run} | {profile} | {train} | {val} | {final} | {bytes} | {status} |".format(
                run=row.get("run_id"),
                profile=row.get("profile"),
                train=f"{row_metrics.get('train_bpb'):.6f}" if isinstance(row_metrics.get("train_bpb"), (int, float)) else "n/a",
                val=f"{row_metrics.get('val_bpb'):.6f}" if isinstance(row_metrics.get("val_bpb"), (int, float)) else "n/a",
                final=f"{row_metrics.get('final_int8_bpb'):.6f}" if isinstance(row_metrics.get("final_int8_bpb"), (int, float)) else "n/a",
                bytes=str(row_bytes) if row_bytes is not None else "n/a",
                status="selected" if row.get("run_id") == run_id else "candidate",
            )
        )
    caveat = (
        "This is packaged as a non-record submission by default. The run uses ConvexTok-2048 and a graphified OAI-baseline adaptation, "
        "so the BPB path needs extra independent verification before any SOTA claim. The record folder includes the tokenizer and the exact "
        "minimal ToricGT modules required by `train_gpt.py`, and the README reports a self-contained code/tokenizer/model size estimate."
    )
    readme = [
        "# ToricGT ConvexTok-2048 Graphified FoT",
        "",
        caveat,
        "",
        "## Result",
        "",
        f"- selected run: `{run_id}`",
        f"- selected profile: `{profile_name}`",
        f"- checkpoint step: `{metrics.get('checkpoint_step')}`",
        f"- train BPB: `{metrics.get('train_bpb')}`",
        f"- validation BPB: `{metrics.get('val_bpb')}`",
        f"- int8+zlib round-trip BPB: `{metrics.get('final_int8_bpb')}`",
        f"- compressed model bytes: `{model_bytes}`",
        f"- self-contained code/tokenizer/dependency bytes: `{code_bytes}`",
        f"- total estimated artifact bytes: `{total_bytes}` / `16000000`",
        f"- final model SHA256: `{artifact_sha}`",
        f"- checkpoint SHA256: `{checkpoint_sha}`",
        "",
        "## Technique Summary",
        "",
        "This submission adapts the OpenAI Parameter Golf baseline rather than replacing it with the full ToricGT research model. The byte LM remains the BPB objective, while the input/output stream is augmented with first-class graph structure.",
        "",
        "- **ConvexTok-2048 deterministic tokenizer**: byte-boundary tokenization DAG with LP/token-rank/price/byte-length features.",
        "- **TokenGT-style FineWeb graphification**: token nodes, causal one-dimensional edge tokens, distance/endpoint/identifier channels, and toric phase features are injected into the baseline hidden stream.",
        "- **OAI-FineWeb-only flattening**: graph-output states are flattened back to the tokenizer sequence for BPB scoring; non-FineWeb graph data remains graph structured.",
        "- **Tropical/toric tokenization features**: min-plus path structure and toric vocabulary-face regularization expose tokenizer active paths as trainable geometry.",
        "- **Embedding-space GFlowNet and Forest-of-Thought heads**: training-only trajectory objectives encourage useful reasoning forests and memory retrieval without increasing the compressed inference artifact.",
        "- **Low-weight advanced sidecar losses**: full-rank GraphCG, analogical memory, TokenGT graph supervision, toric geometry, vector-bundle 1D-cone/sheaf, Toric BGG category-O, Koszul persistence, combinatorial toric commutative algebra, and derived signatures are active with BPB-first staging.",
        "",
        "## Best-Of-10 Sweep",
        "",
        *run_table,
        "",
        "## Reproduction",
        "",
        "The record folder is self-contained with the minimal ToricGT modules required by the adapted `train_gpt.py`. It still expects the Parameter Golf FineWeb binary shards to be available through `DATA_PATH`; no network access is used during evaluation.",
        "",
        "```bash",
        "DATA_PATH=/path/to/fineweb10B_convextok2048_det \\",
        "./run_submission.sh",
        "```",
        "",
        "## Rule Notes",
        "",
        "- The official README states the cap is `16,000,000` decimal bytes for code plus compressed model. This folder reports the stricter self-contained size including the tokenizer and dependency modules.",
        "- Tokenizer changes require extra proof that BPB is correct. The campaign logs include native validation BPB and int8 round-trip BPB; ConvexTok regret/tokenization-DAG analyses are retained in the ToricGT training notes.",
        "- This local campaign was not an 8xH100 10-minute record run. It is submitted as a unique non-record graph/tropical/toric OAI-baseline adaptation unless later rerun under official record conditions.",
        "- Score-first test-time adaptation is configured as non-committing in these short runs; no validation tokens are used for training before scoring.",
        "",
    ]
    (record_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    submission = {
        "author": args.parameter_golf_author,
        "github_id": args.parameter_golf_github_id,
        "name": args.parameter_golf_submission_name,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "blurb": "OpenAI baseline adaptation with ConvexTok-2048, first-class TokenGT graphification, OAI-only graph-output flattening, and training-only tropical/toric/FoT side objectives.",
        "val_bpb": metrics.get("final_int8_bpb") if isinstance(metrics.get("final_int8_bpb"), (int, float)) else metrics.get("val_bpb"),
        "native_val_bpb": metrics.get("val_bpb"),
        "final_int8_zlib_roundtrip_bpb": metrics.get("final_int8_bpb"),
        "train_bpb": metrics.get("train_bpb"),
        "bytes_total": total_bytes,
        "bytes_model_int8_zlib": model_bytes,
        "bytes_code_tokenizer_dependencies": code_bytes,
        "track": args.parameter_golf_track,
        "selected_run_id": run_id,
        "selected_profile": profile_name,
        "checkpoint_step": metrics.get("checkpoint_step"),
        "tokenizer": "ConvexTok-2048 deterministic",
        "vocab_size": int(args.vocab_size),
        "fineweb_graphification": True,
        "tokengt_first_class": True,
        "graph_output_flattening_scope": "oai_fineweb_bpb_only",
        "oai_gflownet": True,
        "embedding_forest_of_thought": True,
        "multi_token_prediction": True,
        "advanced_sidecar_metrics": True,
        "official_record_claim": False,
        "model_artifact_sha256": artifact_sha,
        "checkpoint_sha256": checkpoint_sha,
        "huggingface_repo": args.hf_checkpoint_repo,
        "copied_files": copied,
    }
    (record_dir / "submission.json").write_text(json.dumps(submission, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "generated_utc": utc_iso(),
        "phase": phase,
        "record_dir": str(record_dir),
        "selected_run": best,
        "self_contained_artifact_bytes": total_bytes,
        "code_tokenizer_dependency_bytes": code_bytes,
        "model_artifact_bytes": model_bytes,
        "model_artifact_path": str(final_artifact),
        "model_artifact_sha256": artifact_sha,
        "history": history_rows,
    }
    (record_dir / "best_model_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if final_artifact.exists():
        (record_dir / "final_model.int8.ptz.sha256").write_text(f"{artifact_sha}  final_model.int8.ptz\n", encoding="utf-8")
    note = notes_dir / f"{phase}-PARAMETER-GOLF-RECORD-{utc_stamp()}.md"
    note.write_text(
        "\n".join(
            [
                "# Parameter Golf Record Folder Generated",
                "",
                f"- generated UTC: `{utc_iso()}`",
                f"- record dir: `{record_dir}`",
                f"- selected run: `{run_id}`",
                f"- selected BPB: `{metric_bpb(metrics)}`",
                f"- total estimated bytes: `{total_bytes}`",
                f"- include model artifact in PR folder: `{args.parameter_golf_include_model_artifact}`",
                "",
                "The compressed model artifact itself is uploaded to Hugging Face by the campaign when `--upload-best-short-to-hf` is enabled. The PR folder carries reproducible code, dependencies, logs, tokenizer, and a manifest.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return record_dir, note


def create_parameter_golf_pr(args: argparse.Namespace, record_dir: Path, notes_dir: Path, *, phase: str) -> int:
    repo = Path(args.repo_root).resolve()
    pg_repo = repo / args.parameter_golf_repo if not Path(args.parameter_golf_repo).is_absolute() else Path(args.parameter_golf_repo)
    branch = f"codex/toricgt-convextok2048-{utc_stamp().lower()}"
    rel_record = record_dir.relative_to(pg_repo)
    body = notes_dir / f"{phase}-PARAMETER-GOLF-PR-BODY.md"
    body.write_text(
        "\n".join(
            [
                "## Summary",
                "",
                "Adds a non-record Parameter Golf submission folder for ToricGT ConvexTok-2048 graphified OAI-baseline experiments.",
                "",
                "## Scope",
                "",
                f"- Only adds `{rel_record}`.",
                "- Includes `README.md`, `submission.json`, `train.log`, adapted `train_gpt.py`, `convextok.py`, the ConvexTok tokenizer JSON, and the minimal ToricGT dependency package required by the adapted script.",
                "- The compressed model artifact is uploaded separately to Hugging Face and referenced by manifest/SHA256 rather than committed as a large binary by default.",
                "",
                "## Rule Notes",
                "",
                "- Packaged as `track_non_record_16mb` by default because it uses a custom tokenizer and graphification path that should receive extra BPB verification before any record claim.",
                "- The README reports a stricter self-contained bytes estimate including tokenizer and dependency modules.",
                "",
                "## Validation",
                "",
                "- The ToricGT campaign selected the best completed short run by exported int8 BPB under the self-contained 16MB estimate.",
                "- `python -m py_compile` was run on the campaign controller before generating this record.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    commands = [
        ["git", "checkout", "-B", branch],
        ["git", "add", str(rel_record)],
        ["git", "commit", "-m", "Add ToricGT ConvexTok graphified non-record submission"],
        ["git", "push", "-u", "origin", branch],
    ]
    report = notes_dir / f"{phase}-PARAMETER-GOLF-PR-{utc_stamp()}.md"
    outputs: list[dict[str, object]] = []
    for command in commands:
        proc = subprocess.run(command, cwd=pg_repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outputs.append({"command": command, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
        if proc.returncode != 0:
            report.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
            print(f"[{utc_iso()}] Parameter Golf PR prep failed at {' '.join(command)}; see {report}", flush=True)
            return int(proc.returncode)
    pr_command = [
        "gh",
        "pr",
        "create",
        "--repo",
        str(args.parameter_golf_pr_base_repo),
        "--head",
        f"{args.parameter_golf_github_id}:{branch}",
        "--base",
        str(args.parameter_golf_pr_base),
        "--title",
        "ToricGT ConvexTok-2048 graphified FoT non-record submission",
        "--body-file",
        str(body),
    ]
    if args.parameter_golf_pr_draft:
        pr_command.append("--draft")
    proc = subprocess.run(pr_command, cwd=pg_repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    outputs.append({"command": pr_command, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
    report.write_text(
        "\n".join(
            [
                "# Parameter Golf PR Creation",
                "",
                f"- generated UTC: `{utc_iso()}`",
                f"- record folder: `{rel_record}`",
                f"- branch: `{branch}`",
                f"- return code: `{proc.returncode}`",
                f"- PR output: `{proc.stdout.strip()}`",
                "",
                "## Command Outputs",
                "",
                "```json",
                json.dumps(outputs, indent=2),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if proc.returncode == 0:
        print(f"[{utc_iso()}] opened Parameter Golf PR: {proc.stdout.strip()}", flush=True)
    else:
        print(f"[{utc_iso()}] Parameter Golf PR creation failed; see {report}", flush=True)
    return int(proc.returncode)


def launch_best_full_training(
    args: argparse.Namespace,
    notes_dir: Path,
    best: dict[str, object],
    *,
    run_index: int,
) -> int:
    profile = profile_by_name(str(best.get("profile", "")))
    if profile is None:
        raise RuntimeError(f"best run profile is not available in current profile table: {best.get('profile')}")
    env_overrides = best.get("env_overrides_used", {})
    if not isinstance(env_overrides, dict):
        env_overrides = {}
    full_run_id = short_run_id(f"{args.campaign_id}-bestfull", run_index, profile.name)
    original_steps = int(args.steps_per_run)
    args.steps_per_run = int(args.final_full_train_steps)
    selection_note = notes_dir / f"BEST-SETTINGS-FULL-TRAIN-{utc_stamp()}.md"
    selection_note.write_text(
        "\n".join(
            [
                "# Best Settings Full Training Launch",
                "",
                f"- generated UTC: `{utc_iso()}`",
                f"- selected short run: `{best.get('run_id')}`",
                f"- selected profile: `{profile.name}`",
                f"- selected BPB: `{metric_bpb(best.get('metrics', {}) if isinstance(best.get('metrics'), dict) else {})}`",
                f"- full run id: `{full_run_id}`",
                f"- full steps: `{args.steps_per_run}`",
                f"- full data path: `{args.fineweb_data}`",
                f"- tokenizer path: `{args.tokenizer_path}`",
                "",
                "## Selected Short-Run Metrics",
                "",
                "```json",
                json.dumps(best.get("metrics", {}), indent=2, sort_keys=True),
                "```",
                "",
                "## Selected Adaptive Overrides",
                "",
                "```json",
                json.dumps(env_overrides, indent=2, sort_keys=True),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        print(
            f"[{utc_iso()}] launching full-data best-settings run {full_run_id} for {args.steps_per_run} steps from {best.get('run_id')}",
            flush=True,
        )
        return launch_training(args, full_run_id, profile, run_index, env_overrides={str(k): str(v) for k, v in env_overrides.items()})
    finally:
        args.steps_per_run = original_steps


def main() -> None:
    args = parse_args()
    if not args.campaign_id:
        args.campaign_id = f"toricgt-oai-sidecar-bpb112-campaign-{utc_stamp()}"
    repo = Path(args.repo_root).resolve()
    notes_dir = repo / "training_notes" / args.campaign_id
    notes_dir.mkdir(parents=True, exist_ok=True)
    state_path = notes_dir / "campaign_state.json"
    history: list[dict[str, object]] = []
    existing_state = read_json_if_exists(state_path)
    if isinstance(existing_state.get("history"), list):
        history = [row for row in existing_state.get("history", []) if isinstance(row, dict)]
        print(f"[{utc_iso()}] resuming campaign with {len(history)} state row(s) from {state_path}", flush=True)
    prior_analysis_dir = Path(str(args.prior_analysis_dir)) if str(args.prior_analysis_dir).strip() else None
    if not history and prior_analysis_dir is not None:
        if not prior_analysis_dir.is_absolute():
            prior_analysis_dir = repo / prior_analysis_dir
        prior_analysis_metrics = read_json_if_exists(prior_analysis_dir / "metrics.json")
        prior_analysis_decision = read_json_if_exists(prior_analysis_dir / "next_profile_decision.json")
        prior_analysis_validation = read_json_if_exists(prior_analysis_dir / "validation.json")
        prior_sidecar_summary = read_json_if_exists(prior_analysis_dir / "sidecar_metric_review" / "sidecar_metric_summary.json")
        prior_analysis_report = prior_analysis_dir / "FULL-ITERATION-REPORT.md"
        if not prior_analysis_metrics or "_read_error" in prior_analysis_metrics:
            raise FileNotFoundError(f"prior analysis metrics are unavailable: {prior_analysis_dir / 'metrics.json'}")
        if not prior_analysis_decision or "_read_error" in prior_analysis_decision:
            raise FileNotFoundError(
                f"prior analysis decision is unavailable: {prior_analysis_dir / 'next_profile_decision.json'}"
            )
        if not prior_analysis_report.exists():
            raise FileNotFoundError(f"prior analysis report is unavailable: {prior_analysis_report}")
        original_prior_analysis_decision = prior_analysis_decision
        prior_analysis_decision = build_adaptive_decision(
            prior_analysis_metrics,
            prior_analysis_validation if isinstance(prior_analysis_validation, dict) else {},
            prior_sidecar_summary if isinstance(prior_sidecar_summary, dict) else {},
            run_id=f"prior_import:{prior_analysis_dir.name}",
            target_bpb=float(args.target_bpb),
        )
        if isinstance(original_prior_analysis_decision, dict):
            original_hint = str(original_prior_analysis_decision.get("next_profile_hint", "") or "").strip()
            if original_hint:
                prior_analysis_decision["next_profile_hint"] = original_hint
                prior_analysis_decision["imported_full_analysis_hint_authoritative"] = True
            original_env = original_prior_analysis_decision.get("env_overrides")
            if isinstance(original_env, dict) and original_env:
                merged_env = dict(prior_analysis_decision.get("env_overrides") or {})
                merged_env.update({str(key): str(value) for key, value in original_env.items()})
                prior_analysis_decision["env_overrides"] = merged_env
                prior_analysis_decision["imported_full_analysis_env_authoritative"] = True
        prior_analysis_decision["recomputed_from_prior_analysis_with_current_policy"] = True
        prior_analysis_decision["original_prior_next_profile_hint"] = (
            original_prior_analysis_decision.get("next_profile_hint")
            if isinstance(original_prior_analysis_decision, dict)
            else None
        )
        imported = notes_dir / f"PRIOR-FULL-ANALYSIS-{prior_analysis_dir.name}.md"
        imported.write_text(
            "\n".join(
                [
                    "# Imported Prior Full Analysis",
                    "",
                    f"- imported UTC: `{utc_iso()}`",
                    f"- source directory: `{prior_analysis_dir}`",
                    f"- report: `{prior_analysis_report}`",
                    "",
                    "## Decision",
                    "",
                    "```json",
                    json.dumps(prior_analysis_decision, indent=2, sort_keys=True),
                    "```",
                    "",
                    "The first fresh run in this campaign is chosen from this analysis decision.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        history.append(
            {
                "run_id": f"prior_full_analysis:{prior_analysis_dir.name}",
                "metrics": prior_analysis_metrics,
                "analysis_decision": prior_analysis_decision,
                "analysis_returncode": 0,
                "profile": "prior_full_analysis",
            }
        )
    prior_log = repo / args.prior_log if not Path(args.prior_log).is_absolute() else Path(args.prior_log)
    if not history and prior_log.exists():
        prior_metrics = parse_log(prior_log)
        first_profile = choose_profile(1, [{"metrics": prior_metrics}], args.profile_offset)
        prior_report = notes_dir / f"PRIOR-{args.prior_run_id}-ANALYSIS.md"
        write_report(
            prior_report,
            f"Prior {int(args.steps_per_run)}-Step Gate Analysis",
            args.prior_run_id,
            None,
            prior_metrics,
            first_profile,
            target_bpb=float(args.target_bpb),
        )
        history.append({"run_id": args.prior_run_id, "metrics": prior_metrics, "profile": "prior_external"})
    history = recover_logged_runs(repo, args, history)
    write_state(
        state_path,
        {
            "campaign_id": args.campaign_id,
            "updated_utc": utc_iso(),
            "target_bpb": args.target_bpb,
            "primary_run_budget": int(args.max_runs),
            "followup_run_budget": int(args.followup_runs_after_meta),
            "history": history,
        },
    )
    total_runs = int(args.max_runs)
    meta_written = False
    existing_indices = completed_run_indices(history, str(args.campaign_id))
    start_run_index = max(existing_indices) + 1 if existing_indices else 1
    if start_run_index > 1:
        print(f"[{utc_iso()}] continuing campaign at run index {start_run_index}/{total_runs}", flush=True)
    for run_index in range(start_run_index, total_runs + 1):
        profile = choose_profile(run_index, history, args.profile_offset)
        run_id = short_run_id(str(args.campaign_id), run_index, profile.name)
        env_overrides_used = latest_env_overrides(history)
        returncode = launch_training(args, run_id, profile, run_index, env_overrides=env_overrides_used)
        if args.dry_run:
            return
        log_path = repo / "runs" / "oai_sidecar" / run_id / "train.log"
        metrics = parse_log(log_path)
        analysis_returncode = 0
        analysis_decision: dict[str, object] = {}
        if returncode == 0:
            analysis_returncode, analysis_decision = run_full_iteration_analysis(
                args,
                run_id=run_id,
                run_index=run_index,
                metrics=metrics,
            )
        elif args.strict_analysis:
            analysis_decision = {
                "next_profile_hint": "",
                "reason": "training failed before full analysis; strict campaign stopped",
            }
        candidate_history = history + [
            {
                "metrics": metrics,
                "analysis_decision": analysis_decision,
                "analysis_returncode": analysis_returncode,
            }
        ]
        next_profile = choose_profile(run_index + 1, candidate_history, args.profile_offset)
        next_env_overrides = decision_env_overrides(analysis_decision)
        report = notes_dir / f"RUN-{run_index:03d}-{run_id}-REPORT.md"
        write_report(
            report,
            f"{int(args.steps_per_run)}-Step Fresh-Start Campaign Run Report",
            run_id,
            profile,
            metrics,
            next_profile,
            returncode,
            analysis_decision,
            target_bpb=float(args.target_bpb),
            env_overrides_used=env_overrides_used,
            next_env_overrides=next_env_overrides,
        )
        history.append(
            {
                "run_id": run_id,
                "profile": profile.name,
                "env_overrides_used": env_overrides_used,
                "metrics": metrics,
                "returncode": returncode,
                "analysis_returncode": analysis_returncode,
                "analysis_decision": analysis_decision,
            }
        )
        write_state(
            state_path,
            {
                "campaign_id": args.campaign_id,
                "updated_utc": utc_iso(),
                "target_bpb": args.target_bpb,
                "primary_run_budget": int(args.max_runs),
                "followup_run_budget": int(args.followup_runs_after_meta),
                "history": history,
            },
        )
        if args.codex_review:
            run_codex_review(report)
        bpb = metric_bpb(metrics)
        if args.strict_analysis and (returncode != 0 or analysis_returncode != 0):
            print(
                f"[{utc_iso()}] strict campaign stopped after {run_id}; "
                f"training_returncode={returncode} analysis_returncode={analysis_returncode}",
                flush=True,
            )
            return
        if returncode != 0:
            print(f"[{utc_iso()}] run {run_id} failed with code {returncode}; continuing with next profile", flush=True)
        if args.stop_on_target and math.isfinite(bpb) and bpb <= args.target_bpb:
            synopsis = notes_dir / f"TARGET-REACHED-{utc_stamp()}.md"
            synopsis.write_text(
                f"# Target Reached\n\n- run: `{run_id}`\n- BPB: `{bpb:.8f}`\n- target: `{args.target_bpb}`\n",
                encoding="utf-8",
            )
            print(f"[{utc_iso()}] target reached by {run_id}: {bpb:.6f}", flush=True)
            return
    best = best_completed_row(history)
    if best is None:
        best = min(history, key=lambda row: metric_bpb(row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}))
    if int(args.submission_step_candidate) > 0 and best is not None:
        candidate = launch_submission_step_candidate(
            args,
            notes_dir,
            best,
            steps=int(args.submission_step_candidate),
            run_index=total_runs + 90,
        )
        if candidate is not None:
            history.append(candidate)
            write_state(
                state_path,
                {
                    "campaign_id": args.campaign_id,
                    "updated_utc": utc_iso(),
                    "target_bpb": args.target_bpb,
                    "primary_run_budget": int(args.max_runs),
                    "followup_run_budget": int(args.followup_runs_after_meta),
                    "submission_step_candidate": int(args.submission_step_candidate),
                    "history": history,
                },
            )
    submission_best = best_submission_row(args, history)
    if submission_best is not None:
        best = submission_best
    final_meta = write_meta_analysis(notes_dir, history, args, phase=f"final-{total_runs}-run")
    hf_upload_returncode: int | None = None
    if args.upload_best_short_to_hf:
        hf_upload_returncode = upload_best_short_checkpoint_to_hf(
            args,
            notes_dir,
            best,
            phase=f"best-of-{total_runs}-short-runs",
        )
    parameter_golf_record_dir: str | None = None
    parameter_golf_pr_returncode: int | None = None
    if args.create_parameter_golf_pr:
        record_dir, record_note = write_parameter_golf_submission(
            args,
            notes_dir,
            best,
            history,
            phase=f"best-of-{total_runs}-short-runs",
        )
        parameter_golf_record_dir = str(record_dir)
        parameter_golf_pr_returncode = create_parameter_golf_pr(
            args,
            record_dir,
            notes_dir,
            phase=f"best-of-{total_runs}-short-runs",
        )
    synopsis = notes_dir / f"CAMPAIGN-SYNOPSIS-{utc_stamp()}.md"
    synopsis.write_text(
        "\n".join(
            [
                "# Campaign Synopsis",
                "",
                f"- campaign: `{args.campaign_id}`",
                f"- target BPB: `{args.target_bpb}`",
                f"- runs completed: `{len(completed_campaign_rows(history))}`",
                f"- best run: `{best.get('run_id')}`",
                f"- HF upload return code: `{hf_upload_returncode}`",
                f"- Parameter Golf record dir: `{parameter_golf_record_dir}`",
                f"- Parameter Golf PR return code: `{parameter_golf_pr_returncode}`",
                f"- final meta-analysis: `{final_meta}`",
                f"- best metrics:",
                "",
                "```json",
                json.dumps(best.get("metrics", {}), indent=2, sort_keys=True),
                "```",
                "",
                "The short-run campaign exhausted its configured exact run budget. The best-run checkpoint is the selected short-run artifact.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if args.final_full_train_best:
        final_returncode = launch_best_full_training(args, notes_dir, best, run_index=total_runs + 1)
        final_state = {
            "campaign_id": args.campaign_id,
            "updated_utc": utc_iso(),
            "short_run_budget": total_runs,
            "short_runs_completed": len(completed_campaign_rows(history)),
            "best_short_run": best.get("run_id"),
            "best_short_metrics": best.get("metrics", {}),
            "hf_upload_returncode": hf_upload_returncode,
            "parameter_golf_record_dir": parameter_golf_record_dir,
            "parameter_golf_pr_returncode": parameter_golf_pr_returncode,
            "final_full_train_steps": int(args.final_full_train_steps),
            "final_full_train_returncode": final_returncode,
        }
        write_state(notes_dir / "final_full_train_state.json", final_state)
        print(
            f"[{utc_iso()}] best-settings full training completed with return code {final_returncode}",
            flush=True,
        )
        return
    print(f"[{utc_iso()}] campaign exhausted; synopsis {synopsis}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
