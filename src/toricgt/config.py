"""Configuration dataclasses for ToricGT experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


AttentionKind = Literal["softmax", "tropical", "tropical_ring", "hybrid"]


@dataclass(frozen=True)
class ModelConfig:
    """Small TokenGT-style encoder configuration.

    The default is deliberately lightweight for CPU validation. Increase
    ``d_model``, ``num_layers``, and ``ffn_multiplier`` for the 8M-35M regime.
    """

    d_model: int = 192
    num_heads: int = 6
    num_layers: int = 6
    ffn_multiplier: int = 4
    dropout: float = 0.1
    attention: AttentionKind = "tropical_ring"
    ring_block_size: int = 256
    max_nodes: int = 256
    max_edges: int = 1024
    node_feature_dim: int = 16
    edge_feature_dim: int = 16
    output_dim: int = 16
    torus_rank: int = 2
    theta: float = 0.6180339887498948
    use_soft_moe: bool = True
    soft_moe_num_experts: int = 4
    soft_moe_slots_per_expert: int = 2
    soft_moe_residual_scale: float = 0.1
    soft_moe_start_layer: int | None = None
    use_gflownet_head: bool = True
    gflownet_num_actions: int = 12
    gflownet_hidden_dim: int = 512
    use_trajectory_memory_head: bool = False
    trajectory_memory_projection_dim: int = 128
    trajectory_memory_teacher_temperature: float = 0.20
    trajectory_memory_retrieval_temperature: float = 0.20
    trajectory_memory_distill_weight: float = 0.25
    trajectory_memory_quality_weight: float = 0.10
    trajectory_memory_topology_weight: float = 0.20
    trajectory_memory_graphcg_weight: float = 0.30
    trajectory_memory_toric_weight: float = 0.20
    trajectory_memory_dag_weight: float = 0.20
    trajectory_memory_derived_weight: float = 0.20
    output_derived_category_certificates: bool = False
    derived_category_max_vertices: int = 8


@dataclass(frozen=True)
class DataConfig:
    """Dataset curation and leakage-controlled split configuration."""

    dataset_names: tuple[str, ...] = (
        "AI-MO/NuminaMath-CoT",
        "open-r1/OpenR1-Math-220k",
        "open-r1/codeforces-cots",
        "openai/gsm8k",
        "openai/frontierscience",
        "openai/healthbench",
        "openai/healthbench-professional",
        "openai/graphwalks",
        "EleutherAI/hendrycks_math",
        "HuggingFaceH4/MATH-500",
        "terrycraddock/Tree_Of_Thoughts_BASE_24k",
        "gss1147/Got_Math_500K",
        "unimorph/universal_morphologies",
        "Sefaria/Rabbinic-Hebrew-English-Pairs",
        "Sefaria/hebrew_library",
        "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
        "Jackrong/GPT-OSS-120B-Distilled-Reasoning-math",
        "reasoning-degeneration-dev/algorithmic-sft-training-data-v1",
        "lamm-mit/graph-reasoning-messages-11K",
        "sequelbox/DAG-Reasoning-DeepSeek-R1-0528",
        "Gryphe/Opus-4.6-Reasoning-24k",
        "nvidia/Nemotron-RL-ReasoningGym-v1",
        "nvidia/Nemotron-Content-Safety-Reasoning-Dataset",
        "nvidia/PhysicalAI-Traffic-Anomaly-Reasoning",
    )
    output_dir: str = "data/curated"
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    minhash_shingles: int = 5
    similarity_threshold: float = 0.82
    seed: int = 17


@dataclass(frozen=True)
class TrainConfig:
    """Training loop defaults for one RTX 4090 without disturbing other jobs."""

    batch_size: int = 4
    grad_accum_steps: int = 8
    max_steps: int = 100_000
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 2_000
    clip_grad_norm: float = 1.0
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    device: str = "cuda"
    log_interval: int = 20
    eval_interval: int = 1_000
    ckpt_interval: int = 2_000
    num_workers: int = 2
    use_wandb: bool = True
    wandb_project: str = "toricgt"
    tags: list[str] = field(default_factory=lambda: ["tokengt", "tropical-ring-attention"])
