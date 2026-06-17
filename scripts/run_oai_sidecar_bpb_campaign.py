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
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


VAL_RE = re.compile(r"step:(?P<step>\d+)/(?P<total>\d+) val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+)")
TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+) train_loss:(?P<loss>[0-9.]+)"
    r"(?: train_bpb:(?P<train_bpb>[0-9.eE+-]+) train_bpt:(?P<train_bpt>[0-9.eE+-]+))?.*?"
    r"(?: graph_lm_loss:(?P<graph_lm_loss>[0-9.eE+-]+) graph_lm_bpb:(?P<graph_lm_bpb>[0-9.eE+-]+) "
    r"graph_lm_w:(?P<graph_lm_weight>[0-9.eE+-]+))?.*?"
    r"sidecar_loss:(?P<sidecar>[0-9.eE+-]+) graphcg:(?P<graphcg>[0-9.eE+-]+) "
    r"analogy:(?P<analogy>[0-9.eE+-]+) tokengt_graph:(?P<tokengt>[0-9.eE+-]+) "
    r"memory:(?P<memory>[0-9.eE+-]+)"
    r"(?: toric:(?P<toric>[0-9.eE+-]+) (?:vb1d|vb):(?P<vector_bundle_1d_cone>[0-9.eE+-]+) "
    r"bgg:(?P<bgg>[0-9.eE+-]+) koszul:(?P<koszul>[0-9.eE+-]+) "
    r"cca:(?P<combinatorial_toric>[0-9.eE+-]+))?"
)
FINAL_RE = re.compile(r"final_int8_zlib_roundtrip_exact val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+)")
SIZE_RE = re.compile(r"Total submission size int8\+zlib: (?P<size>\d+) bytes")
CHECKPOINT_RE = re.compile(r"checkpoint_saved:(?P<path>.*?) step:(?P<step>\d+) val_bpb:(?P<bpb>[-+0-9.eE]+|None)")


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
        "gate1500_high_batch_toric_bgg_memory",
        {
            "TRAIN_BATCH_TOKENS": "983040",
            "MATRIX_LR": "0.042",
            "SCALAR_LR": "0.042",
            "TIED_EMBED_LR": "0.052",
            "WARMDOWN_ITERS": "650",
            "WARMUP_STEPS": "10",
            "TOKENGT_FIRST_CLASS_LR": "2.0e-4",
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.050",
            "TOKENGT_EDGE_WEIGHT": "0.036",
            "TOKENGT_TORUS_WEIGHT": "0.012",
            "GRAPH_OUTPUT_FLATTENING_LR": "1.8e-4",
            "GRAPH_OUTPUT_EDGE_RADIUS": "3",
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
            "TOKENGT_GRAPH_RADIUS": "3",
            "TOKENGT_STRUCTURAL_WEIGHT": "0.040",
            "TOKENGT_EDGE_WEIGHT": "0.028",
            "TOKENGT_TORUS_WEIGHT": "0.008",
            "GRAPH_OUTPUT_FLATTENING_LR": "1.5e-4",
            "GRAPH_OUTPUT_EDGE_RADIUS": "3",
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
        default=10,
        help="After max-runs primary attempts miss target, write a cross-run meta-analysis and run this many more fresh attempts.",
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
        if match := TRAIN_RE.search(line):
            trains.append({key: float(value) for key, value in match.groupdict().items() if value is not None})
        if match := FINAL_RE.search(line):
            final_loss = float(match.group("loss"))
            final_bpb = float(match.group("bpb"))
        if match := SIZE_RE.search(line):
            artifact_size = int(match.group("size"))
        if match := CHECKPOINT_RE.search(line):
            checkpoint_path = match.group("path")
            checkpoint_step = int(match.group("step"))
    latest_val = vals[-1] if vals else {}
    latest_train = trains[-1] if trains else {}
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
    }


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
        decision = history[-1].get("analysis_decision", {})
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
            if isinstance(artifact, (int, float)) and float(artifact) > 15_800_000:
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
            "Each campaign attempt restarts from step 0.  The OAI SP1024 FineWeb stream remains the primary BPB objective.  "
            "FineWeb graphification is first-class and enabled by default: the main GPT input stream receives causal TokenGT-style node/edge structural embeddings before the transformer blocks, and optional OAI-FineWeb-only output flattening maps graph-output states back to SentencePiece sequence order for BPB scoring.  "
            "Curated graph data is also bonafide primary LM training data: graph records are serialized into causal topological node/edge token sequences when directed and acyclic, and deterministic random-order graph token sequences when noncausal or cyclic.  Graph-data hidden states are not flattened by default; they remain graph structured for graph-in/graph-out training and sidecar analysis.  "
            "ToricGT sidecar losses remain active from step 0 for full-rank GraphCG, analogy lattice structure, graph supervision, and trajectory-memory retrieval, but their weights are kept small so they regularize rather than dominate BPB.",
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
        "auxiliary-gradient routing cosines/projection coefficients, retrieval gates, uncertainty-weighted advanced multipliers, "
        "GraphCG BPB-orthogonalization metrics, artifact size, and round-trip quantization. "
        "If the report's full-analysis decision includes a sidecar_metric_review path, read that markdown/JSON and account for every observed "
        "`toricgt_sidecar/*` metric description, trend, BPB correlation, and family weight guidance before recommending the next restart profile. "
        "If the 1.5K gate missed BPB < 1.19, recommend exact hyperparameter/config changes and explain why the next run should restart from step 0. "
        "Do not collapse all advanced losses into one bin: reason separately about tropical/toric fan losses, BGG/category-O losses, "
        "vector-bundle/sheaf losses, persistent homology/Koszul losses, combinatorial toric algebra, GraphCG, analogy, memory, and graph-LM. "
        "For loss-like metrics, interpret positive BPB correlation as possible evidence that reducing that loss could help BPB; "
        "do not automatically treat every positive correlation as harmful. "
        f"Report path: {report}"
    )
    subprocess.run(["codex", "exec", prompt], check=False)


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
        "GRAPH_LM_BATCH_SIZE": "4",
        "GRAPH_LM_EVERY": "1",
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
        "TEACHER_CHECKPOINT": os.environ.get("TEACHER_CHECKPOINT", ""),
        "TEACHER_DISTILL_WEIGHT": os.environ.get("TEACHER_DISTILL_WEIGHT", "0"),
        "TEACHER_DISTILL_WEIGHT_START": os.environ.get("TEACHER_DISTILL_WEIGHT_START", "0"),
        "TEACHER_DISTILL_UNTIL_STEP": os.environ.get("TEACHER_DISTILL_UNTIL_STEP", "900"),
        "TEACHER_DISTILL_DECAY_STEPS": os.environ.get("TEACHER_DISTILL_DECAY_STEPS", "300"),
        "TEACHER_DISTILL_TEMPERATURE": os.environ.get("TEACHER_DISTILL_TEMPERATURE", "2.0"),
        "TEACHER_DISTILL_MAX_SEQUENCES": os.environ.get("TEACHER_DISTILL_MAX_SEQUENCES", "2"),
        "FINEWEB_CASEOPS": os.environ.get("FINEWEB_CASEOPS", "0"),
        "VOCAB_SIZE": "1024",
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
        "TOKENGT_GRAPH_RADIUS": "3",
        "TOKENGT_TOKEN_CLASS_BUCKETS": "64",
        "TOKENGT_POSITION_BUCKETS": "256",
        "TOKENGT_STRUCTURAL_WEIGHT": "0.035",
        "TOKENGT_EDGE_WEIGHT": "0.025",
        "TOKENGT_TORUS_WEIGHT": "0.010",
        "TOKENGT_IDENTIFIER_DIM": "24",
        "TOKENGT_IDENTIFIER_WEIGHT": "0.014",
        "TOKENGT_ENDPOINT_WEIGHT": "0.018",
        "TOKENGT_EDGE_TOKEN_WEIGHT": "0.016",
        "OAI_FINEWEB_OUTPUT_FLATTENING": "1",
        "GRAPH_OUTPUT_FLATTENING": "1",
        "GRAPH_OUTPUT_FLATTENING_LR": "1.4e-4",
        "GRAPH_OUTPUT_EDGE_RADIUS": "3",
        "GRAPH_OUTPUT_NODE_WEIGHT": "0.05",
        "GRAPH_OUTPUT_EDGE_WEIGHT": "0.05",
        "GRAPH_OUTPUT_VIRTUAL_EDGE_TOKENS": "1",
        "GRAPH_OUTPUT_EDGE_TOKEN_WEIGHT": "0.035",
        "GRAPH_OUTPUT_SCORE_CORRECTION": "1",
        "GRAPH_OUTPUT_SCORE_CORRECTION_WEIGHT": "0.024",
        "GRAPH_OUTPUT_CALIBRATION_LOSS_WEIGHT": "0.012",
        "GRAPH_OUTPUT_CALIBRATION_MARGIN": "0.0",
        "GRAPH_OUTPUT_CALIBRATION_EVERY": "25",
        "GRAPH_OUTPUT_CALIBRATION_MAX_SEQUENCES": "2",
        "VAL_BATCH_SIZE": "524288",
        "ITERATIONS": str(args.steps_per_run),
        "MAX_WALLCLOCK_SECONDS": "0",
        "VAL_LOSS_EVERY": str(args.steps_per_run),
        "TRAIN_LOG_EVERY": "100",
        "CHECKPOINT_EVERY": str(args.steps_per_run),
        "CHECKPOINT_DIR": str(ckpt_dir),
        "TORICGT_SIDECAR_SEQ_LEN": "256",
        "TORICGT_SIDECAR_BATCH_SIZE": "8",
        "TORICGT_SIDECAR_EVERY": "1",
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
        "SCORE_FIRST_TTA": os.environ.get("SCORE_FIRST_TTA", "0"),
        "SCORE_FIRST_TTA_STEPS": os.environ.get("SCORE_FIRST_TTA_STEPS", "0"),
        "SCORE_FIRST_TTA_LR": os.environ.get("SCORE_FIRST_TTA_LR", "0"),
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
        toric = [finite_metric(row, "toric_geometry_loss") or finite_metric(row, "toric") for row in rows]
        vector_bundle_1d = [
            finite_metric(row, "toric_vector_bundle_1d_cone_ce_loss") or finite_metric(row, "vector_bundle_1d_cone")
            for row in rows
        ]
        bgg = [finite_metric(row, "bgg_loss") or finite_metric(row, "bgg") for row in rows]
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
            "mean_toric_loss": mean([v for v in toric if v is not None]),
            "mean_toric_vector_bundle_1d_cone_ce_loss": mean([v for v in vector_bundle_1d if v is not None]),
            "mean_bgg_loss": mean([v for v in bgg if v is not None]),
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
        f"- first-class FineWeb TokenGT: lr `{profile.env.get('TOKENGT_FIRST_CLASS_LR')}`, radius `{profile.env.get('TOKENGT_GRAPH_RADIUS')}`, structural `{profile.env.get('TOKENGT_STRUCTURAL_WEIGHT')}`, one-dimensional-edge `{profile.env.get('TOKENGT_EDGE_WEIGHT')}`, torus `{profile.env.get('TOKENGT_TORUS_WEIGHT')}`, identifier `{profile.env.get('TOKENGT_IDENTIFIER_WEIGHT', 'base default')}`, endpoint `{profile.env.get('TOKENGT_ENDPOINT_WEIGHT', 'base default')}`, edge-token `{profile.env.get('TOKENGT_EDGE_TOKEN_WEIGHT', 'base default')}`",
        f"- OAI-FineWeb-only graph-output flattening: lr `{profile.env.get('GRAPH_OUTPUT_FLATTENING_LR', 'base default')}`, radius `{profile.env.get('GRAPH_OUTPUT_EDGE_RADIUS', 'base default')}`, node `{profile.env.get('GRAPH_OUTPUT_NODE_WEIGHT', 'base default')}`, edge `{profile.env.get('GRAPH_OUTPUT_EDGE_WEIGHT', 'base default')}`, virtual-edge `{profile.env.get('GRAPH_OUTPUT_EDGE_TOKEN_WEIGHT', 'base default')}`, score-correction `{profile.env.get('GRAPH_OUTPUT_SCORE_CORRECTION_WEIGHT', 'base default')}`",
        "- graph radius policy: profile starts in the local radius 2-3 regime; the adaptive analysis may widen to 4-6 only when BPB slope, graph correlations, and flattening calibration support wider neighborhoods.",
        "- flattening calibration: enabled by base environment with a small regression-only loss that penalizes graph-output flattening only when it worsens FineWeb CE versus the raw graph hidden state.",
        f"- bonafide graph LM primary weight: `{profile.env.get('GRAPH_LM_LOSS_WEIGHT', 'base default')}`",
        f"- sidecar LR: `{profile.env.get('TORICGT_SIDECAR_LR')}`",
        f"- base sidecar weights: graphcg `{profile.env.get('GRAPHCG_LOSS_WEIGHT')}`, analogy `{profile.env.get('ANALOGY_LOSS_WEIGHT')}`, tokengt_graph `{profile.env.get('TOKENGT_GRAPH_LOSS_WEIGHT')}`, memory `{profile.env.get('TRAJECTORY_MEMORY_LOSS_WEIGHT')}`",
        f"- advanced weights: `{json.dumps(active_advanced, sort_keys=True)}`",
    ]


def write_meta_analysis(notes_dir: Path, history: list[dict[str, object]], args: argparse.Namespace, *, phase: str) -> Path:
    rows = completed_campaign_rows(history)
    summaries = profile_summary(history)
    best = min(rows, key=lambda row: metric_bpb(row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}), default=None)
    worst = max(rows, key=lambda row: metric_bpb(row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}), default=None)
    best_profile = str(summaries[0]["profile"]) if summaries else ""
    table = [
        "| profile | runs | best BPB | mean BPB | mean train BPB | mean graph-LM BPB | mean graph-LM weight | worst BPB | mean artifact bytes | mean sidecar |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        table.append(
            "| {profile} | {runs} | {best} | {avg} | {train_avg} | {graph_avg} | {graph_weight} | {worst} | {artifact} | {sidecar} |".format(
                profile=row["profile"],
                runs=row["runs"],
                best=f"{row['best_bpb']:.6f}" if isinstance(row.get("best_bpb"), (int, float)) else "n/a",
                avg=f"{row['mean_bpb']:.6f}" if isinstance(row.get("mean_bpb"), (int, float)) else "n/a",
                train_avg=f"{row['mean_train_bpb']:.6f}" if isinstance(row.get("mean_train_bpb"), (int, float)) else "n/a",
                graph_avg=f"{row['mean_graph_lm_bpb']:.6f}" if isinstance(row.get("mean_graph_lm_bpb"), (int, float)) else "n/a",
                graph_weight=f"{row['mean_graph_lm_weight']:.4f}" if isinstance(row.get("mean_graph_lm_weight"), (int, float)) else "n/a",
                worst=f"{row['worst_bpb']:.6f}" if isinstance(row.get("worst_bpb"), (int, float)) else "n/a",
                artifact=f"{row['mean_artifact_bytes']:.0f}" if isinstance(row.get("mean_artifact_bytes"), (int, float)) else "n/a",
                sidecar=f"{row['mean_sidecar_loss']:.6g}" if isinstance(row.get("mean_sidecar_loss"), (int, float)) else "n/a",
            )
        )
    recommendations = [
        "Keep `TRAIN_SEQ_LEN=1024` and `grad_accum_steps=8`; those are the current speed and memory controls.",
        "Keep every advanced metric family enabled in logging. Sweep training pressure through profile weights, not by disabling observability.",
        "Reject profiles that improve auxiliary diagnostics while worsening exported int8 BPB or pushing the artifact near the 16,000,000-byte cap.",
        "Favor the best BPB profile and its nearest LR/auxiliary-weight neighbors for the follow-up phase; do not jump to high auxiliary weights unless BPB and artifact margin justify it.",
        "Use strict exact analysis as a gate. A failed GUDHI/Sage/Macaulay2 or screenshot bundle blocks the next run unless a retry succeeds.",
    ]
    if best_profile:
        recommendations.append(f"Current best evidence favors `{best_profile}` as the anchor for the next hyperparameter neighborhood.")
    meta_path = notes_dir / f"{phase}-META-ANALYSIS-{utc_stamp()}.md"
    best_profile_details = profile_env_markdown(best_profile) if best_profile else ["No completed profile rows are available."]
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
        "## Raw History",
        "",
        "```json",
        json.dumps(rows, indent=2, sort_keys=True),
        "```",
        "",
    ]
    meta_path.write_text("\n".join(lines), encoding="utf-8")
    return meta_path


def main() -> None:
    args = parse_args()
    if not args.campaign_id:
        args.campaign_id = f"toricgt-oai-sidecar-bpb112-campaign-{utc_stamp()}"
    repo = Path(args.repo_root).resolve()
    notes_dir = repo / "training_notes" / args.campaign_id
    notes_dir.mkdir(parents=True, exist_ok=True)
    state_path = notes_dir / "campaign_state.json"
    history: list[dict[str, object]] = []
    prior_analysis_dir = Path(str(args.prior_analysis_dir)) if str(args.prior_analysis_dir).strip() else None
    if prior_analysis_dir is not None:
        if not prior_analysis_dir.is_absolute():
            prior_analysis_dir = repo / prior_analysis_dir
        prior_analysis_metrics = read_json_if_exists(prior_analysis_dir / "metrics.json")
        prior_analysis_decision = read_json_if_exists(prior_analysis_dir / "next_profile_decision.json")
        prior_analysis_report = prior_analysis_dir / "FULL-ITERATION-REPORT.md"
        if not prior_analysis_metrics or "_read_error" in prior_analysis_metrics:
            raise FileNotFoundError(f"prior analysis metrics are unavailable: {prior_analysis_dir / 'metrics.json'}")
        if not prior_analysis_decision or "_read_error" in prior_analysis_decision:
            raise FileNotFoundError(
                f"prior analysis decision is unavailable: {prior_analysis_dir / 'next_profile_decision.json'}"
            )
        if not prior_analysis_report.exists():
            raise FileNotFoundError(f"prior analysis report is unavailable: {prior_analysis_report}")
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
    if prior_log.exists():
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
    total_runs = int(args.max_runs) + max(0, int(args.followup_runs_after_meta))
    meta_written = False
    for run_index in range(1, total_runs + 1):
        if run_index == int(args.max_runs) + 1 and not meta_written:
            meta_path = write_meta_analysis(notes_dir, history, args, phase="primary-25-run")
            write_state(
                state_path,
                {
                    "campaign_id": args.campaign_id,
                    "updated_utc": utc_iso(),
                    "target_bpb": args.target_bpb,
                    "history": history,
                    "primary_meta_analysis": str(meta_path),
                    "followup_runs_planned": int(args.followup_runs_after_meta),
                },
            )
            if args.codex_review:
                run_codex_review(meta_path)
            meta_written = True
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
        if math.isfinite(bpb) and bpb <= args.target_bpb:
            synopsis = notes_dir / f"TARGET-REACHED-{utc_stamp()}.md"
            synopsis.write_text(
                f"# Target Reached\n\n- run: `{run_id}`\n- BPB: `{bpb:.8f}`\n- target: `{args.target_bpb}`\n",
                encoding="utf-8",
            )
            print(f"[{utc_iso()}] target reached by {run_id}: {bpb:.6f}", flush=True)
            return
    best = min(history, key=lambda row: metric_bpb(row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}))
    final_meta = write_meta_analysis(notes_dir, history, args, phase="final-35-run")
    synopsis = notes_dir / f"CAMPAIGN-SYNOPSIS-{utc_stamp()}.md"
    synopsis.write_text(
        "\n".join(
            [
                "# Campaign Synopsis",
                "",
                f"- campaign: `{args.campaign_id}`",
                f"- target BPB: `{args.target_bpb}`",
                f"- runs completed: `{len(history)}`",
                f"- best run: `{best.get('run_id')}`",
                f"- final meta-analysis: `{final_meta}`",
                f"- best metrics:",
                "",
                "```json",
                json.dumps(best.get("metrics", {}), indent=2, sort_keys=True),
                "```",
                "",
                "The campaign exhausted its configured primary and follow-up run budgets without reaching target.  Inspect the final meta-analysis before starting another campaign.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[{utc_iso()}] campaign exhausted; synopsis {synopsis}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
