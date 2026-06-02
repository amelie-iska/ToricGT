"""Dense random-order autoregressive language model for Parameter Golf.

This module is intentionally a narrow contest adapter.  It keeps the ToricGT
ideas that fit the byte-budget setting: graph-order projection, toric phase
features on target positions, hybrid softmax/tropical-ring attention, recurrent
depth sharing, and optional PolarQuant KV perturbation during evaluation.
Soft-MoE remains available in the research graph model but is disabled here by
default because dense shared weights are usually the better bytes-per-quality
tradeoff under a 16 MB artifact cap.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .config import AttentionKind
from .koszul_persistence import KoszulPersistenceConfig, koszul_persistence_loss
from .tropical_attention import TransformerBlock
from .topological_reasoning import ReasoningTopologyConfig, reasoning_step_topology_loss
from .toric_geometry_tasks import LowRankToricGeometryProbe, ToricGeometryConfig
from .trajectory_memory import TrajectoryMemoryConfig, TrajectoryRetrievalHead


@dataclass(frozen=True)
class RandomOrderLMConfig:
    """Configuration for the Parameter-Golf random-order ToricGT adapter."""

    vocab_size: int = 260
    max_seq_len: int = 1024
    d_model: int = 384
    num_heads: int = 6
    num_layers: int = 7
    recurrent_passes: int = 2
    ffn_multiplier: int = 4
    dropout: float = 0.1
    attention: AttentionKind = "hybrid"
    ring_block_size: int = 256
    seed: int = 17
    theta: float = 0.6180339887498948
    beta: float = 1.4142135623730951
    use_soft_moe: bool = False
    polarquant_kv_bits: int = 0
    polarquant_train: bool = False
    use_gflownet_policy: bool = True
    gflownet_num_actions: int = 8
    gflownet_hidden_dim: int = 128
    gflownet_action_scale: float = 0.05
    use_bigram_hash: bool = True
    bigram_hash_buckets: int = 4096
    bigram_hash_weight: float = 0.35
    use_caseops_features: bool = True
    caseops_weight: float = 0.35
    use_revealed_neighbor_context: bool = False
    revealed_neighbor_radius: int = 2
    revealed_neighbor_context_weight: float = 0.75
    use_revealed_context_prior: bool = False
    revealed_context_prior_alpha: float = 0.25
    revealed_context_prior_weight: float = 0.35
    use_smear_gate: bool = True
    smear_temperature_min: float = 0.55
    smear_temperature_max: float = 1.75
    use_toric_memory: bool = True
    toric_memory_slots: int = 32
    toric_memory_weight: float = 0.08
    use_toric_geometry_tasks: bool = True
    toric_geometry_num_exponents: int = 16
    toric_geometry_exponent_dim: int = 4
    toric_geometry_probe_rank: int = 12
    toric_geometry_quant_bits: int = 6
    toric_geometry_teacher_temperature: float = 0.55
    toric_geometry_margin: float = 0.18
    toric_geometry_fan_weight: float = 1.0
    toric_geometry_bend_weight: float = 0.3
    toric_geometry_binom_weight: float = 0.3
    toric_geometry_moment_weight: float = 0.5
    toric_geometry_coxeter_weight: float = 0.2
    toric_geometry_braid_weight: float = 0.1
    toric_geometry_leaf_weight: float = 0.25
    toric_geometry_max_positions: int = 256
    use_graphcg: bool = False
    graphcg_num_directions: int = 12
    graphcg_alpha: float = 0.12
    graphcg_temperature: float = 0.2
    graphcg_max_codes: int = 256
    graphcg_orthogonal_weight: float = 0.2
    graphcg_covariance_weight: float = 0.05
    graphcg_sparsity_weight: float = 0.0001
    use_analogy_lattice: bool = False
    analogy_lattice_max_pairs: int = 256
    analogy_lattice_stride: int = 1
    analogy_lattice_temperature: float = 0.2
    analogy_lattice_basis_weight: float = 0.5
    analogy_lattice_parallelogram_weight: float = 0.5
    analogy_lattice_topology_weight: float = 0.25
    analogy_topology_max_points_per_group: int = 16
    analogy_topology_max_groups: int = 24
    analogy_topology_k: int = 4
    analogy_topology_filtration_levels: int = 4
    analogy_topology_radius_min: float = 0.55
    analogy_topology_radius_max: float = 1.65
    analogy_topology_chain_weight: float = 0.25
    analogy_topology_inclusion_weight: float = 0.1
    analogy_topology_directed: bool = True
    analogy_topology_directed_weight: float = 0.35
    analogy_topology_skew_scale: float = 0.35
    analogy_topology_cycle_weight: float = 0.1
    analogy_hdbscan_enabled: bool = True
    analogy_hdbscan_weight: float = 0.2
    analogy_hdbscan_min_cluster_size: int = 4
    analogy_hdbscan_min_samples: int = 4
    analogy_hdbscan_stability_threshold: float = 0.18
    analogy_step_topology_weight: float = 0.25
    analogy_step_topology_max_points: int = 24
    analogy_step_topology_max_windows: int = 6
    analogy_step_topology_window_size: int = 32
    analogy_step_topology_step_stride: int = 8
    analogy_step_topology_time_bias: float = 0.18
    use_koszul_persistence: bool = True
    koszul_max_points: int = 24
    koszul_max_windows: int = 4
    koszul_window_size: int = 32
    koszul_step_stride: int = 8
    koszul_num_parameters: int = 3
    koszul_temperature: float = 0.12
    koszul_chart_exponents: int = 12
    koszul_rank_temperature: float = 0.05
    aux_mtp_offsets: int = 2
    contrastive_temperature: float = 0.2
    trajectory_flow_viscosity: float = 0.05
    use_trajectory_memory_head: bool = False
    trajectory_memory_projection_dim: int = 128
    trajectory_memory_teacher_temperature: float = 0.20
    trajectory_memory_retrieval_temperature: float = 0.20
    trajectory_memory_distill_weight: float = 0.25
    trajectory_memory_quality_weight: float = 0.10
    trajectory_memory_topology_weight: float = 0.20
    trajectory_memory_graphcg_weight: float = 0.30
    trajectory_memory_toric_weight: float = 0.20
    target_artifact_bytes: int = 15_600_000
    byte_offset: int = 4
    pad_token_id: int = 0
    weight_tying: bool = True

    @property
    def bos_token_id(self) -> int:
        return self.vocab_size


@dataclass(frozen=True)
class RandomOrderBatch:
    previous_tokens: torch.Tensor
    target_tokens: torch.Tensor
    target_positions: torch.Tensor
    permutation: torch.Tensor


def stable_order_seed(base_seed: int, sample_id: int, pass_id: int = 0, length: int = 0) -> int:
    """Derive a reproducible order seed independent of token content."""

    payload = f"{base_seed}:{sample_id}:{pass_id}:{length}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


def random_order_permutation(
    length: int,
    seed: int,
    sample_id: int = 0,
    pass_id: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a content-independent random order for one graph/sequence."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_order_seed(seed, sample_id, pass_id, length))
    order = torch.randperm(length, generator=generator)
    if device is not None:
        order = order.to(device=device)
    return order


def random_order_batch(
    tokens: torch.Tensor,
    seed: int,
    sample_ids: torch.Tensor | None = None,
    pass_id: int = 0,
    bos_token_id: int | None = None,
    permutation: torch.Tensor | None = None,
) -> RandomOrderBatch:
    """Create score-before-update inputs for random-order AR training.

    ``previous_tokens[:, k]`` contains BOS for ``k=0`` and otherwise the token
    revealed at the previous random-order step.  The current target token is
    never present in the same row that scores it.
    """

    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, length]")
    batch, length = tokens.shape
    device = tokens.device
    if bos_token_id is None:
        bos_token_id = int(tokens.max().item()) + 1
    if permutation is None:
        if sample_ids is None:
            sample_ids = torch.arange(batch, device=device)
        orders = [
            random_order_permutation(
                length,
                seed=seed,
                sample_id=int(sample_ids[i].detach().cpu().item()),
                pass_id=pass_id,
                device=device,
            )
            for i in range(batch)
        ]
        permutation = torch.stack(orders, dim=0)
    if permutation.shape != tokens.shape:
        raise ValueError("permutation must have the same [batch, length] shape as tokens")
    target_tokens = torch.gather(tokens, dim=1, index=permutation)
    previous_tokens = torch.empty_like(target_tokens)
    previous_tokens[:, 0] = bos_token_id
    if length > 1:
        previous_tokens[:, 1:] = target_tokens[:, :-1]
    return RandomOrderBatch(
        previous_tokens=previous_tokens,
        target_tokens=target_tokens,
        target_positions=permutation,
        permutation=permutation,
    )


def byte_encode(text: str, byte_offset: int = 4) -> list[int]:
    return [int(byte) + byte_offset for byte in text.encode("utf-8", errors="replace")]


def byte_decode(tokens: list[int] | torch.Tensor, byte_offset: int = 4) -> str:
    if isinstance(tokens, torch.Tensor):
        raw = tokens.detach().cpu().tolist()
    else:
        raw = tokens
    data = bytes(max(0, min(255, int(tok) - byte_offset)) for tok in raw if int(tok) >= byte_offset)
    return data.decode("utf-8", errors="replace")


class DenseRandomOrderToricLM(nn.Module):
    """Dense Parameter-Golf model with ToricGT random-order graph projection."""

    def __init__(self, config: RandomOrderLMConfig) -> None:
        super().__init__()
        if config.use_soft_moe:
            raise ValueError("Parameter-Golf random-order LM is dense by default; set use_soft_moe=False")
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size + 1, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.toric_phase = nn.Linear(4, config.d_model, bias=False)
        self.bigram_hash = (
            nn.Embedding(config.bigram_hash_buckets, config.d_model)
            if config.use_bigram_hash and config.bigram_hash_buckets > 0
            else None
        )
        self.caseops = nn.Linear(10, config.d_model, bias=False) if config.use_caseops_features else None
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ffn_multiplier=config.ffn_multiplier,
                    attention=self._attention_for_layer(layer_idx),
                    dropout=config.dropout,
                    ring_block_size=config.ring_block_size,
                    use_soft_moe=False,
                    polarquant_kv_bits=config.polarquant_kv_bits,
                    polarquant_train=config.polarquant_train,
                )
                for layer_idx in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.smear_gate = (
            nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, 1),
            )
            if config.use_smear_gate
            else None
        )
        if config.use_toric_memory and config.toric_memory_slots > 0:
            self.toric_memory_key = nn.Linear(6, config.d_model, bias=False)
            self.toric_memory_query = nn.Linear(config.d_model, config.d_model, bias=False)
            self.toric_memory_value = nn.Parameter(torch.empty(config.toric_memory_slots, config.d_model))
        else:
            self.toric_memory_key = None
            self.toric_memory_query = None
            self.toric_memory_value = None
        if config.use_gflownet_policy:
            self.gflownet_policy = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.gflownet_hidden_dim),
                nn.GELU(),
                nn.Linear(config.gflownet_hidden_dim, config.gflownet_num_actions),
            )
            self.gflownet_action_embedding = nn.Embedding(config.gflownet_num_actions, config.d_model)
            self.gflownet_flow = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.gflownet_hidden_dim),
                nn.GELU(),
                nn.Linear(config.gflownet_hidden_dim, 1),
            )
            self.gflownet_log_z = nn.Parameter(torch.zeros(()))
        else:
            self.gflownet_policy = None
            self.gflownet_action_embedding = None
            self.gflownet_flow = None
            self.gflownet_log_z = None
        self.graphcg_direction_basis = (
            nn.Parameter(torch.empty(config.graphcg_num_directions, config.d_model))
            if config.use_graphcg and config.graphcg_num_directions > 1
            else None
        )
        if config.weight_tying:
            self.output = None
            self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))
        else:
            self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.output_bias = None
        self.aux_mtp_heads = nn.ModuleList(
            [nn.Linear(config.d_model, config.vocab_size) for _ in range(max(0, config.aux_mtp_offsets))]
        )
        self.trajectory_memory_head = (
            TrajectoryRetrievalHead(
                config.d_model,
                TrajectoryMemoryConfig(
                    projection_dim=config.trajectory_memory_projection_dim,
                    teacher_temperature=config.trajectory_memory_teacher_temperature,
                    retrieval_temperature=config.trajectory_memory_retrieval_temperature,
                    distill_weight=config.trajectory_memory_distill_weight,
                    quality_weight=config.trajectory_memory_quality_weight,
                    topology_weight=config.trajectory_memory_topology_weight,
                    graphcg_weight=config.trajectory_memory_graphcg_weight,
                    toric_weight=config.trajectory_memory_toric_weight,
                ),
            )
            if config.use_trajectory_memory_head
            else None
        )
        self.toric_geometry_probe = (
            LowRankToricGeometryProbe(
                config.d_model,
                ToricGeometryConfig(
                    enabled=True,
                    num_exponents=config.toric_geometry_num_exponents,
                    exponent_dim=config.toric_geometry_exponent_dim,
                    probe_rank=config.toric_geometry_probe_rank,
                    quant_bits=config.toric_geometry_quant_bits,
                    theta=config.theta,
                    beta=config.beta,
                    teacher_temperature=config.toric_geometry_teacher_temperature,
                    margin=config.toric_geometry_margin,
                    fan_weight=config.toric_geometry_fan_weight,
                    bend_weight=config.toric_geometry_bend_weight,
                    binom_weight=config.toric_geometry_binom_weight,
                    moment_weight=config.toric_geometry_moment_weight,
                    coxeter_weight=config.toric_geometry_coxeter_weight,
                    braid_weight=config.toric_geometry_braid_weight,
                    leaf_weight=config.toric_geometry_leaf_weight,
                    max_positions=config.toric_geometry_max_positions,
                ),
            )
            if config.use_toric_geometry_tasks
            else None
        )
        radius = max(0, int(config.revealed_neighbor_radius))
        if config.use_revealed_neighbor_context and radius > 0:
            self.revealed_left_logits = nn.ModuleList(
                [nn.Embedding(config.vocab_size + 1, config.vocab_size, padding_idx=config.vocab_size) for _ in range(radius)]
            )
            self.revealed_right_logits = nn.ModuleList(
                [nn.Embedding(config.vocab_size + 1, config.vocab_size, padding_idx=config.vocab_size) for _ in range(radius)]
            )
        else:
            self.revealed_left_logits = None
            self.revealed_right_logits = None
        self.apply(self._init_module)
        if self.toric_memory_value is not None:
            nn.init.normal_(self.toric_memory_value, mean=0.0, std=0.02)
        if self.graphcg_direction_basis is not None:
            nn.init.orthogonal_(self.graphcg_direction_basis)
        if self.revealed_left_logits is not None and self.revealed_right_logits is not None:
            for table in list(self.revealed_left_logits) + list(self.revealed_right_logits):
                nn.init.zeros_(table.weight)
        nn.init.zeros_(self.output_bias) if self.output_bias is not None else None

    def _init_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _attention_for_layer(self, layer_idx: int) -> str:
        if self.config.attention != "hybrid":
            return self.config.attention
        split = max(1, self.config.num_layers // 2)
        return "softmax" if layer_idx < split else "tropical_ring"

    @property
    def bos_token_id(self) -> int:
        return self.config.bos_token_id

    def _phase_features(self, positions: torch.Tensor) -> torch.Tensor:
        pos = positions.to(dtype=torch.float32)
        theta = 2 * math.pi * self.config.theta * pos
        beta = 2 * math.pi * self.config.beta * pos
        return torch.stack(
            [torch.sin(theta), torch.cos(theta), torch.sin(beta), torch.cos(beta)],
            dim=-1,
        )

    def _bigram_hash_ids(self, previous_tokens: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        """Hash strict-prefix byte context and target location into small buckets.

        At reveal step ``k``, ``previous_tokens[:, k]`` is the token revealed at
        step ``k-1`` and the shifted context below is the token revealed at
        step ``k-2``.  This makes the feature prefix-causal while giving the
        compact Parameter-Golf model a cheap n-gram-like memory.
        """

        if self.bigram_hash is None:
            raise RuntimeError("bigram hash is disabled")
        previous_previous = torch.empty_like(previous_tokens)
        previous_previous[:, 0] = self.bos_token_id
        if previous_tokens.shape[1] > 1:
            previous_previous[:, 1:] = previous_tokens[:, :-1]
        step_ids = torch.arange(previous_tokens.shape[1], device=previous_tokens.device).view(1, -1)
        hashed = (
            previous_previous.to(torch.int64) * 1_000_003
            + previous_tokens.to(torch.int64) * 917_609
            + target_positions.to(torch.int64) * 65_537
            + step_ids.to(torch.int64) * 32_771
        )
        return torch.remainder(hashed, self.config.bigram_hash_buckets).to(torch.long)

    def _caseops_features(self, previous_tokens: torch.Tensor) -> torch.Tensor:
        """Return byte-class operator features from already revealed tokens."""

        byte = previous_tokens.to(torch.float32) - float(self.config.byte_offset)
        valid = ((byte >= 0) & (byte <= 255)).to(torch.float32)
        lower = ((byte >= ord("a")) & (byte <= ord("z"))).to(torch.float32)
        upper = ((byte >= ord("A")) & (byte <= ord("Z"))).to(torch.float32)
        digit = ((byte >= ord("0")) & (byte <= ord("9"))).to(torch.float32)
        space = ((byte == ord(" ")) | (byte == ord("\t"))).to(torch.float32)
        newline = ((byte == ord("\n")) | (byte == ord("\r"))).to(torch.float32)
        punctuation = (
            ((byte >= ord("!")) & (byte <= ord("/")))
            | ((byte >= ord(":")) & (byte <= ord("@")))
            | ((byte >= ord("[")) & (byte <= ord("`")))
            | ((byte >= ord("{")) & (byte <= ord("~")))
        ).to(torch.float32)
        high_bit = (byte >= 128).to(torch.float32)
        bos = (previous_tokens == self.bos_token_id).to(torch.float32)
        norm_byte = torch.where(valid.bool(), byte / 255.0, torch.zeros_like(byte))
        return torch.stack(
            [norm_byte, valid, lower, upper, digit, space, newline, punctuation, high_bit, bos],
            dim=-1,
        )

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones(length, length, dtype=torch.bool, device=device).tril()
        return mask.view(1, 1, length, length)

    def _backbone_hidden(
        self,
        previous_tokens: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        if previous_tokens.shape != target_positions.shape:
            raise ValueError("previous_tokens and target_positions must have the same shape")
        if previous_tokens.ndim != 2:
            raise ValueError("inputs must have shape [batch, length]")
        _, length = previous_tokens.shape
        if length > self.config.max_seq_len:
            raise ValueError(f"sequence length {length} exceeds max_seq_len {self.config.max_seq_len}")
        x = self.token_embedding(previous_tokens)
        x = x + self.position_embedding(target_positions.clamp(0, self.config.max_seq_len - 1))
        x = x + self.toric_phase(self._phase_features(target_positions).to(device=x.device, dtype=x.dtype))
        if self.bigram_hash is not None:
            hash_ids = self._bigram_hash_ids(previous_tokens, target_positions)
            x = x + self.config.bigram_hash_weight * self.bigram_hash(hash_ids)
        if self.caseops is not None:
            features = self._caseops_features(previous_tokens).to(device=x.device, dtype=x.dtype)
            x = x + self.config.caseops_weight * self.caseops(features)
        x = self.drop(x)
        mask = self._causal_mask(length, device=previous_tokens.device)
        for _ in range(max(1, self.config.recurrent_passes)):
            for block in self.blocks:
                x = block(x, mask=mask, token_mask=None)
        return self.norm(x)

    def _project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.output is None:
            return F.linear(hidden, self.token_embedding.weight[: self.config.vocab_size], self.output_bias)
        return self.output(hidden)

    def _apply_smear_gate(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.smear_gate is None:
            return logits, None
        raw = self.smear_gate(hidden).squeeze(-1).float()
        lo = float(self.config.smear_temperature_min)
        hi = float(self.config.smear_temperature_max)
        if hi <= lo:
            raise ValueError("smear_temperature_max must be greater than smear_temperature_min")
        temperature = lo + (hi - lo) * torch.sigmoid(raw)
        scaled = logits.float() / temperature.unsqueeze(-1).clamp_min(1e-4)
        return scaled.to(dtype=logits.dtype), temperature

    def _revealed_neighbor_context_logits(
        self,
        target_tokens: torch.Tensor,
        target_positions: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return legal score-before-update logits from revealed graph neighbors.

        Random-order decoding turns a byte string into a graph completion task:
        positions are vertices and the reveal prefix is the known induced
        subgraph.  For step ``k`` this head may use tokens already revealed at
        original positions ``p-r`` or ``p+r`` for ``r <= radius``.  It never
        reads the current target or any unrevealed future token before scoring
        the current position.
        """

        table_context_enabled = self.revealed_left_logits is not None and self.revealed_right_logits is not None
        prior_enabled = bool(self.config.use_revealed_context_prior) and abs(
            float(self.config.revealed_context_prior_weight)
        ) > 0.0
        if not table_context_enabled and not prior_enabled:
            return None
        if target_tokens.shape != target_positions.shape:
            raise ValueError("target_tokens and target_positions must have the same shape")
        if target_tokens.ndim != 2:
            raise ValueError("target tensors must have shape [batch, length]")
        batch, length = target_tokens.shape
        device = target_tokens.device
        vocab = int(self.config.vocab_size)
        unknown = vocab
        context_logits = torch.zeros(batch, length, vocab, device=device, dtype=torch.float32)
        known_counts = torch.zeros(batch, length, device=device, dtype=torch.float32)
        clipped_targets = target_tokens.clamp(0, vocab - 1).to(torch.long)
        positions = target_positions.clamp(0, length - 1).to(torch.long)
        alpha = max(float(self.config.revealed_context_prior_alpha), 1.0e-6)
        prior_weight = float(self.config.revealed_context_prior_weight)
        uniform_log_prob = -math.log(max(1, vocab))
        if prior_enabled:
            token_one_hot = F.one_hot(clipped_targets, num_classes=vocab).to(torch.float32)
            strict_prefix_counts = alpha + torch.cumsum(token_one_hot, dim=1) - token_one_hot
            strict_prefix_total = alpha * vocab + torch.arange(length, device=device, dtype=torch.float32).view(
                1,
                length,
                1,
            )
            log_prior = strict_prefix_counts.clamp_min(1.0e-8).log() - strict_prefix_total.log()
            context_logits = context_logits + prior_weight * (log_prior - uniform_log_prob)
        if table_context_enabled:
            neighbor_logits = torch.zeros_like(context_logits)
            tokens_by_position = torch.full((batch, length), unknown, device=device, dtype=torch.long)
            tokens_by_position.scatter_(1, positions, clipped_targets)
            reveal_step = torch.arange(length, device=device, dtype=torch.long).view(1, length).expand(batch, -1)
            step_by_position = torch.empty(batch, length, device=device, dtype=torch.long)
            step_by_position.scatter_(1, positions, reveal_step)
            for offset, (left_table, right_table) in enumerate(
                zip(self.revealed_left_logits, self.revealed_right_logits),
                start=1,
            ):
                left_pos = positions - offset
                left_valid = left_pos >= 0
                left_idx = left_pos.clamp(0, length - 1)
                left_step = step_by_position.gather(1, left_idx)
                left_known = left_valid & (left_step < reveal_step)
                left_tok = tokens_by_position.gather(1, left_idx)
                left_tok = torch.where(left_known, left_tok, torch.full_like(left_tok, unknown))
                neighbor_logits = neighbor_logits + left_table(left_tok).float()
                known_counts = known_counts + left_known.to(torch.float32)

                right_pos = positions + offset
                right_valid = right_pos < length
                right_idx = right_pos.clamp(0, length - 1)
                right_step = step_by_position.gather(1, right_idx)
                right_known = right_valid & (right_step < reveal_step)
                right_tok = tokens_by_position.gather(1, right_idx)
                right_tok = torch.where(right_known, right_tok, torch.full_like(right_tok, unknown))
                neighbor_logits = neighbor_logits + right_table(right_tok).float()
                known_counts = known_counts + right_known.to(torch.float32)
            context_logits = context_logits + float(self.config.revealed_neighbor_context_weight) * neighbor_logits
        neighbor_slots = 2.0 * len(self.revealed_left_logits) if table_context_enabled else 1.0
        return context_logits.to(dtype=dtype), known_counts / max(1.0, neighbor_slots)

    def _add_revealed_neighbor_context(
        self,
        aux: dict[str, torch.Tensor],
        target_tokens: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context = self._revealed_neighbor_context_logits(
            target_tokens,
            target_positions,
            dtype=aux["logits"].dtype,
        )
        if context is None:
            return aux
        context_logits, known_fraction = context
        aux = dict(aux)
        aux["neural_logits"] = aux["logits"]
        aux["logits"] = aux["logits"] + context_logits
        aux["revealed_neighbor_known_fraction"] = known_fraction.mean()
        aux["revealed_neighbor_context_norm"] = context_logits.float().norm(dim=-1).mean()
        aux["revealed_context_prior_weight"] = torch.as_tensor(
            float(self.config.revealed_context_prior_weight),
            device=context_logits.device,
        )
        return aux

    def _toric_memory_basis(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        slots = torch.arange(self.config.toric_memory_slots, device=device, dtype=torch.float32)
        theta = 2 * math.pi * self.config.theta * slots
        beta = 2 * math.pi * self.config.beta * slots
        cocycle = theta + beta + 2 * math.pi * self.config.theta * slots.square()
        basis = torch.stack(
            [
                torch.sin(theta),
                torch.cos(theta),
                torch.sin(beta),
                torch.cos(beta),
                torch.sin(cocycle),
                torch.cos(theta - beta),
            ],
            dim=-1,
        )
        return basis.to(dtype=dtype)

    def _toric_memory_context(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.toric_memory_key is None or self.toric_memory_query is None or self.toric_memory_value is None:
            return None
        basis = self._toric_memory_basis(hidden.device, hidden.dtype)
        keys = F.normalize(self.toric_memory_key(basis), dim=-1)
        queries = F.normalize(self.toric_memory_query(hidden), dim=-1)
        scores = torch.matmul(queries, keys.transpose(0, 1)) * math.sqrt(hidden.shape[-1])
        weights = F.softmax(scores.float(), dim=-1).to(dtype=hidden.dtype)
        context = torch.matmul(weights, self.toric_memory_value.to(dtype=hidden.dtype))
        entropy = -(weights.float().clamp_min(1e-8) * weights.float().clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy.mean() / math.log(max(2, self.config.toric_memory_slots))
        return context, entropy

    def _gflownet_context(
        self,
        hidden: torch.Tensor,
        sample: bool,
    ) -> dict[str, torch.Tensor]:
        if self.gflownet_policy is None or self.gflownet_action_embedding is None:
            return {}
        policy_hidden = torch.nan_to_num(hidden.float(), nan=0.0, posinf=30.0, neginf=-30.0).to(dtype=hidden.dtype)
        policy_logits = self.gflownet_policy(policy_hidden)
        policy_logits_f = torch.nan_to_num(policy_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(
            min=-30.0,
            max=30.0,
        )
        policy_logits = policy_logits_f.to(dtype=policy_logits.dtype)
        policy_log_probs = F.log_softmax(policy_logits_f, dim=-1)
        policy_probs = policy_log_probs.exp()
        entropy = -(policy_probs * policy_log_probs).sum(dim=-1)
        if sample:
            flat_logits = policy_logits_f.reshape(-1, policy_logits_f.shape[-1])
            action_ids = torch.distributions.Categorical(logits=flat_logits).sample().view(policy_logits.shape[:-1])
            action_logprob = policy_log_probs.gather(-1, action_ids.unsqueeze(-1)).squeeze(-1)
            context = self.gflownet_action_embedding(action_ids)
            action_mass = F.one_hot(action_ids, num_classes=self.config.gflownet_num_actions).float().mean(dim=(0, 1))
        else:
            action_ids = policy_probs.argmax(dim=-1)
            action_logprob = policy_log_probs.gather(-1, action_ids.unsqueeze(-1)).squeeze(-1)
            context = torch.matmul(policy_probs.to(dtype=hidden.dtype), self.gflownet_action_embedding.weight.to(hidden.dtype))
            action_mass = policy_probs.mean(dim=(0, 1))
        action_mass = action_mass.clamp_min(1e-8)
        diversity = -(action_mass * action_mass.log()).sum() / math.log(max(2, self.config.gflownet_num_actions))
        return {
            "context": context.to(dtype=hidden.dtype),
            "policy_logits": policy_logits,
            "policy_entropy": entropy.mean(),
            "action_ids": action_ids,
            "action_logprob": action_logprob,
            "action_diversity": diversity,
        }

    def _graphcg_losses(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """GraphCG-style disentanglement over hidden-state edit directions.

        The original GraphCG objective learns steerable latent directions by
        making same-direction edits identifiable and by suppressing
        cross-direction correlation.  For the compact Parameter-Golf adapter we
        keep the same geometry but avoid a separate generator: hidden states are
        edited directly along a small learned basis and trained with a small
        auxiliary contrastive/covariance penalty.
        """

        if self.graphcg_direction_basis is None:
            return {}
        max_codes = max(2, int(self.config.graphcg_max_codes))
        codes = hidden.float().reshape(-1, hidden.shape[-1])
        if codes.shape[0] > max_codes:
            index = torch.linspace(0, codes.shape[0] - 1, steps=max_codes, device=codes.device).long()
            codes = codes.index_select(0, index)
        codes = F.normalize(codes, dim=-1)
        directions = F.normalize(self.graphcg_direction_basis.float(), dim=-1)
        alpha = float(self.config.graphcg_alpha)
        temperature = max(float(self.config.graphcg_temperature), 1e-4)

        edited_near = F.normalize(codes[:, None, :] + alpha * directions[None, :, :], dim=-1)
        edited_far = F.normalize(codes[:, None, :] + (2.0 * alpha) * directions[None, :, :], dim=-1)
        sim = torch.einsum("ndh,nkh->ndk", edited_near, edited_far) / temperature
        n_codes, n_dirs, _ = edited_near.shape
        labels = torch.arange(n_dirs, device=hidden.device).expand(n_codes, n_dirs).reshape(-1)
        code_loss = F.cross_entropy(sim.reshape(n_codes * n_dirs, n_dirs), labels)

        eye = torch.eye(n_dirs, device=hidden.device, dtype=directions.dtype)
        gram = directions @ directions.transpose(0, 1)
        offdiag_count = max(1, n_dirs * (n_dirs - 1))
        orthogonal_loss = (gram - eye).pow(2).sum() / offdiag_count

        coords = codes @ directions.transpose(0, 1)
        coords = coords - coords.mean(dim=0, keepdim=True)
        denom = max(1, coords.shape[0] - 1)
        cov = coords.transpose(0, 1) @ coords / denom
        diag = cov.diagonal().clamp_min(1e-6)
        corr = cov / torch.sqrt(diag[:, None] * diag[None, :])
        covariance_loss = (corr - eye).pow(2).sum() / offdiag_count
        sparsity_loss = directions.abs().mean()
        total = (
            code_loss
            + float(self.config.graphcg_orthogonal_weight) * orthogonal_loss
            + float(self.config.graphcg_covariance_weight) * covariance_loss
            + float(self.config.graphcg_sparsity_weight) * sparsity_loss
        )
        return {
            "graphcg_loss": total,
            "graphcg_code_loss": code_loss.detach(),
            "graphcg_orthogonal_loss": orthogonal_loss.detach(),
            "graphcg_covariance_loss": covariance_loss.detach(),
            "graphcg_sparsity_loss": sparsity_loss.detach(),
            "graphcg_basis_coherence": (gram - eye).abs().amax().detach(),
            "graphcg_axis_variance": coords.var(dim=0, unbiased=False).mean().detach(),
        }

    def _byte_class_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map byte-token ids to coarse relation classes for analogy mining."""

        byte = (tokens - int(self.config.byte_offset)).clamp(0, 255)
        cls = torch.full_like(byte, 8)
        cls = torch.where((byte >= 97) & (byte <= 122), torch.ones_like(cls), cls)
        cls = torch.where((byte >= 65) & (byte <= 90), torch.full_like(cls, 2), cls)
        cls = torch.where((byte >= 48) & (byte <= 57), torch.full_like(cls, 3), cls)
        cls = torch.where(
            (byte == 9) | (byte == 11) | (byte == 12) | (byte == 13) | (byte == 32),
            torch.full_like(cls, 4),
            cls,
        )
        cls = torch.where(byte == 10, torch.full_like(cls, 5), cls)
        punctuation = (
            ((byte >= 33) & (byte <= 47))
            | ((byte >= 58) & (byte <= 64))
            | ((byte >= 91) & (byte <= 96))
            | ((byte >= 123) & (byte <= 126))
        )
        cls = torch.where(punctuation, torch.full_like(cls, 6), cls)
        cls = torch.where(byte >= 128, torch.full_like(cls, 7), cls)
        return cls

    def _analogy_lattice_losses(
        self,
        hidden: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Explicit analogical/functor loss over hidden relation vectors.

        Repeated coarse byte transitions should induce reusable displacement
        vectors in hidden space.  This is the direct category-theoretic analogy
        objective: if two observed arrows have the same coarse source/target
        type, then their hidden differences should agree, and those differences
        should be expressible in the GraphCG lattice basis when that basis is
        enabled.
        """

        if not self.config.use_analogy_lattice:
            return {}
        zero = hidden.float().sum() * 0.0
        stride = max(1, int(self.config.analogy_lattice_stride))
        if hidden.ndim != 3 or target_tokens.ndim != 2 or hidden.shape[1] <= stride:
            return {
                "analogy_lattice_loss": zero,
                "analogy_functor_loss": zero.detach(),
                "analogy_parallelogram_loss": zero.detach(),
                "analogy_topology_loss": zero.detach(),
                "analogy_barcode_loss": zero.detach(),
                "analogy_simplex_closure_loss": zero.detach(),
                "analogy_filtration_inclusion_loss": zero.detach(),
                "analogy_chain_map_loss": zero.detach(),
                "analogy_directed_topology_loss": zero.detach(),
                "analogy_directed_transitive_loss": zero.detach(),
                "analogy_directed_cycle_loss": zero.detach(),
                "analogy_directed_chain_map_loss": zero.detach(),
                "analogy_directed_asymmetry": zero.detach(),
                "analogy_directed_skew_norm": zero.detach(),
                "analogy_hdbscan_loss": zero.detach(),
                "analogy_hdbscan_stability": zero.detach(),
                "analogy_hdbscan_persistent_edge_density": zero.detach(),
                "analogy_hdbscan_outlier_score": zero.detach(),
                "analogy_hdbscan_core_radius": zero.detach(),
                "analogy_filtration_edge_density": zero.detach(),
                "analogy_filtration_triangle_density": zero.detach(),
                "analogy_step_topology_loss": zero.detach(),
                "analogy_step_barcode_loss": zero.detach(),
                "analogy_step_simplex_closure_loss": zero.detach(),
                "analogy_step_filtration_inclusion_loss": zero.detach(),
                "analogy_step_boundary_residual": zero.detach(),
                "analogy_step_dirichlet_energy": zero.detach(),
                "analogy_step_dec_conservation_loss": zero.detach(),
                "analogy_step_dec_mass_residual": zero.detach(),
                "analogy_step_dec_vorticity_drift": zero.detach(),
                "analogy_step_dec_kinetic_energy": zero.detach(),
                "analogy_step_dec_kinetic_energy_drift": zero.detach(),
                "analogy_step_dec_hodge_balance": zero.detach(),
                "analogy_step_dec_wedge_interior_residual": zero.detach(),
                "analogy_step_directed_topology_loss": zero.detach(),
                "analogy_step_directed_transitive_loss": zero.detach(),
                "analogy_step_directed_cycle_flux": zero.detach(),
                "analogy_step_directed_chain_commutator": zero.detach(),
                "analogy_step_directed_asymmetry": zero.detach(),
                "analogy_step_analogical_map_loss": zero.detach(),
                "analogy_step_directed_map_loss": zero.detach(),
                "analogy_step_transport_entropy": zero.detach(),
                "analogy_step_hdbscan_loss": zero.detach(),
                "analogy_step_hdbscan_stability": zero.detach(),
                "analogy_step_hdbscan_noise_fraction": zero.detach(),
                "analogy_step_hdbscan_persistent_edge_density": zero.detach(),
                "analogy_step_hdbscan_core_radius": zero.detach(),
                "analogy_step_edge_density": zero.detach(),
                "analogy_step_triangle_density": zero.detach(),
                "analogy_step_cycle_rank": zero.detach(),
                "analogy_step_betti0": zero.detach(),
                "analogy_step_windows": zero.detach(),
                "analogy_basis_loss": zero.detach(),
                "analogy_axis_entropy": zero.detach(),
                "analogy_lattice_margin": zero.detach(),
                "analogy_relation_groups": zero.detach(),
                "analogy_topology_groups": zero.detach(),
                "analogy_graphcg_chart_dim": zero.detach(),
                "analogy_graphcg_chart_energy": zero.detach(),
            }

        relation = hidden[:, stride:, :].float() - hidden[:, :-stride, :].float()
        relation = F.normalize(relation.reshape(-1, relation.shape[-1]), dim=-1)
        topology_relation = relation
        chart_directions = None
        graphcg_chart_dim = zero
        graphcg_chart_energy = zero
        if self.graphcg_direction_basis is not None:
            chart_directions = F.normalize(self.graphcg_direction_basis.float(), dim=-1)
            topology_relation = F.normalize(relation @ chart_directions.transpose(0, 1), dim=-1)
            hidden_chart = hidden.float() @ chart_directions.transpose(0, 1)
            graphcg_chart_dim = relation.new_tensor(float(chart_directions.shape[0]))
            graphcg_chart_energy = hidden_chart.pow(2).mean()
        else:
            hidden_chart = hidden.float()
        classes = self._byte_class_ids(target_tokens)
        keys = (classes[:, :-stride] * 16 + classes[:, stride:]).reshape(-1)
        max_pairs = max(2, int(self.config.analogy_lattice_max_pairs))
        if relation.shape[0] > max_pairs:
            index = torch.linspace(0, relation.shape[0] - 1, steps=max_pairs, device=relation.device).long()
            relation = relation.index_select(0, index)
            topology_relation = topology_relation.index_select(0, index)
            keys = keys.index_select(0, index)

        unique, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
        repeated = counts[inverse] > 1
        if bool(repeated.any()):
            sums = relation.new_zeros(unique.shape[0], relation.shape[-1])
            sums.index_add_(0, inverse[repeated], relation[repeated])
            group_counts = counts.clamp_min(1).to(dtype=relation.dtype).unsqueeze(-1)
            means = F.normalize(sums / group_counts, dim=-1)
            residual = relation[repeated] - means[inverse[repeated]]
            functor_loss = residual.pow(2).mean()
            relation_groups = (counts > 1).to(dtype=relation.dtype).sum()
        else:
            functor_loss = zero
            relation_groups = zero

        topology_terms = []
        barcode_terms = []
        closure_terms = []
        inclusion_terms = []
        chain_map_terms = []
        directed_terms = []
        directed_transitive_terms = []
        directed_cycle_terms = []
        directed_chain_terms = []
        directed_asymmetry_terms = []
        directed_skew_terms = []
        hdbscan_terms = []
        hdbscan_stability_terms = []
        hdbscan_persistent_edge_terms = []
        hdbscan_outlier_terms = []
        hdbscan_core_radius_terms = []
        edge_density_terms = []
        triangle_density_terms = []
        topology_groups = zero
        max_topology_groups = max(1, int(self.config.analogy_topology_max_groups))
        max_group_points = max(4, int(self.config.analogy_topology_max_points_per_group))
        topology_k = max(1, int(self.config.analogy_topology_k))
        topology_temperature = max(float(self.config.analogy_lattice_temperature), 1e-4)
        filtration_levels = max(2, int(self.config.analogy_topology_filtration_levels))
        radius_min = float(self.config.analogy_topology_radius_min)
        radius_max = max(radius_min + 1e-4, float(self.config.analogy_topology_radius_max))
        filtration_radii = torch.linspace(radius_min, radius_max, steps=filtration_levels, device=relation.device)
        use_directed_topology = bool(self.config.analogy_topology_directed)
        skew_scale = float(self.config.analogy_topology_skew_scale)
        use_hdbscan = bool(self.config.analogy_hdbscan_enabled)
        hdbscan_weight = float(self.config.analogy_hdbscan_weight)
        hdbscan_min_cluster_size = max(2, int(self.config.analogy_hdbscan_min_cluster_size))
        hdbscan_min_samples = max(1, int(self.config.analogy_hdbscan_min_samples))
        hdbscan_threshold = float(self.config.analogy_hdbscan_stability_threshold)
        repeated_group_ids = unique[counts >= 4][:max_topology_groups]
        eye_cache: dict[int, torch.Tensor] = {}
        for group_id in repeated_group_ids:
            group = topology_relation[keys == group_id]
            if group.shape[0] > max_group_points:
                index = torch.linspace(0, group.shape[0] - 1, steps=max_group_points, device=group.device).long()
                group = group.index_select(0, index)
            n_group = int(group.shape[0])
            if n_group < 4:
                continue
            distances = torch.cdist(group, group, p=2)
            nonzero = distances[distances > 1e-6]
            if nonzero.numel() == 0:
                continue
            scale = nonzero.median().detach().clamp_min(1e-4)
            normalized_distances = distances / scale
            eye = eye_cache.get(n_group)
            if eye is None or eye.device != group.device:
                eye = torch.eye(n_group, device=group.device, dtype=group.dtype)
                eye_cache[n_group] = eye
            if use_directed_topology and group.shape[-1] >= 2:
                half = group.shape[-1] // 2
                left = group[:, :half]
                right = group[:, half : half + half]
                skew = (left @ right.transpose(0, 1) - right @ left.transpose(0, 1)) / math.sqrt(max(1, half))
                skew = skew / skew.detach().abs().median().clamp_min(1e-4)
                skew = skew.clamp(-4.0, 4.0) * skew_scale
            else:
                skew = normalized_distances.new_zeros(normalized_distances.shape)
            masked_distances = normalized_distances + eye * 1.0e6
            k = min(topology_k, n_group - 1)
            knn = masked_distances.topk(k, dim=-1, largest=False).values
            barcode_loss = knn[:, 0].mean()
            hdbscan_loss = zero
            hdbscan_stability = zero
            hdbscan_persistent_edge_density = zero
            hdbscan_outlier_score = zero
            hdbscan_core_radius = zero
            if use_hdbscan:
                core_k = min(max(1, hdbscan_min_samples), n_group - 1)
                core_radius = masked_distances.topk(core_k, dim=-1, largest=False).values[:, -1].detach()
                mutual_reachability = torch.maximum(
                    normalized_distances,
                    torch.maximum(core_radius[:, None], core_radius[None, :]),
                )
                mutual_reachability = mutual_reachability + eye * 1.0e6
                persistent_edges = []
                for radius in filtration_radii:
                    persistent_edges.append(
                        torch.sigmoid((radius - mutual_reachability) / topology_temperature) * (1.0 - eye)
                    )
                persistent_affinity = torch.stack(persistent_edges, dim=0).mean(dim=0)
                stable_degree = persistent_affinity.sum(dim=-1) / max(1, n_group - 1)
                stable_member = (stable_degree.detach() * max(1, n_group - 1)) >= float(hdbscan_min_cluster_size - 1)
                pair_weight = torch.relu(persistent_affinity.detach() - hdbscan_threshold).pow(2)
                pair_weight = pair_weight * stable_member.to(pair_weight.dtype)[:, None]
                pair_weight = pair_weight * stable_member.to(pair_weight.dtype)[None, :]
                pair_weight = pair_weight * (1.0 - eye)
                pair_mass = pair_weight.sum()
                if bool((pair_mass > 1e-8).detach().cpu().item()):
                    hdbscan_loss = (pair_weight * normalized_distances.pow(2)).sum() / pair_mass.clamp_min(1e-8)
                hdbscan_stability = stable_degree.mean()
                hdbscan_persistent_edge_density = (pair_weight > 0).to(dtype=relation.dtype).sum() / max(
                    1,
                    n_group * (n_group - 1),
                )
                hdbscan_outlier_score = torch.relu(hdbscan_threshold - stable_degree).mean()
                hdbscan_core_radius = core_radius.mean()
            level_closure_terms = []
            level_edge_density_terms = []
            level_triangle_density_terms = []
            level_inclusion_terms = []
            level_chain_terms = []
            level_directed_transitive_terms = []
            level_directed_cycle_terms = []
            level_directed_asymmetry_terms = []
            level_directed_skew_terms = []
            prev_adjacency = None
            prev_prolongation = None
            prev_directed = None
            prev_directed_prolongation = None
            for radius in filtration_radii:
                adjacency = torch.sigmoid((radius - normalized_distances) / topology_temperature) * (1.0 - eye)
                directed_adjacency = torch.sigmoid((radius - normalized_distances + skew) / topology_temperature) * (
                    1.0 - eye
                )
                two_step = adjacency @ adjacency / max(1, n_group - 2)
                closure_loss = (two_step * (1.0 - adjacency)).mean()
                directed_two_step = directed_adjacency @ directed_adjacency / max(1, n_group - 2)
                directed_transitive_loss = (directed_two_step * (1.0 - directed_adjacency)).mean()
                directed_cycle_flux = torch.einsum(
                    "ij,jk,ki->",
                    directed_adjacency,
                    directed_adjacency,
                    directed_adjacency,
                ) / max(1, n_group * (n_group - 1) * (n_group - 2))
                reverse_cycle_flux = torch.einsum(
                    "ji,kj,ik->",
                    directed_adjacency,
                    directed_adjacency,
                    directed_adjacency,
                ) / max(1, n_group * (n_group - 1) * (n_group - 2))
                directed_cycle_loss = (directed_cycle_flux - reverse_cycle_flux).abs()
                directed_asymmetry = (directed_adjacency - directed_adjacency.transpose(0, 1)).abs().mean()
                directed_skew_norm = (skew * (1.0 - eye)).abs().mean()
                edge_density = adjacency.sum() / max(1, n_group * (n_group - 1))
                triangle_density = torch.einsum("ij,jk,ik->", adjacency, adjacency, adjacency) / max(
                    1,
                    n_group * (n_group - 1) * (n_group - 2),
                )
                prolongation = adjacency + eye
                prolongation = prolongation / prolongation.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                directed_prolongation = directed_adjacency + eye
                directed_prolongation = directed_prolongation / directed_prolongation.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                level_closure_terms.append(closure_loss)
                level_edge_density_terms.append(edge_density)
                level_triangle_density_terms.append(triangle_density)
                level_directed_transitive_terms.append(directed_transitive_loss)
                level_directed_cycle_terms.append(directed_cycle_loss)
                level_directed_asymmetry_terms.append(directed_asymmetry)
                level_directed_skew_terms.append(directed_skew_norm)
                if prev_adjacency is not None and prev_prolongation is not None:
                    # Inclusion K_r -> K_R should only add simplices.  The
                    # row-stochastic prolongations model the induced chain map
                    # on 0-chains; nested maps should commute along the
                    # filtration up to the local softening temperature.
                    level_inclusion_terms.append(torch.relu(prev_adjacency - adjacency).pow(2).mean())
                    level_chain_terms.append((prolongation @ prev_prolongation - prev_prolongation @ prolongation).pow(2).mean())
                    if prev_directed is not None and prev_directed_prolongation is not None:
                        level_inclusion_terms.append(torch.relu(prev_directed - directed_adjacency).pow(2).mean())
                        level_chain_terms.append(
                            (
                                directed_prolongation @ prev_directed_prolongation
                                - prev_directed_prolongation @ directed_prolongation
                            )
                            .pow(2)
                            .mean()
                        )
                prev_adjacency = adjacency
                prev_directed = directed_adjacency
                prev_prolongation = prolongation
                prev_directed_prolongation = directed_prolongation
            closure_loss = torch.stack(level_closure_terms).mean()
            edge_density = torch.stack(level_edge_density_terms).mean()
            triangle_density = torch.stack(level_triangle_density_terms).mean()
            inclusion_loss = torch.stack(level_inclusion_terms).mean() if level_inclusion_terms else zero
            chain_map_loss = torch.stack(level_chain_terms).mean() if level_chain_terms else zero
            directed_transitive_loss = torch.stack(level_directed_transitive_terms).mean()
            directed_cycle_loss = torch.stack(level_directed_cycle_terms).mean()
            directed_chain_map_loss = chain_map_loss
            directed_asymmetry = torch.stack(level_directed_asymmetry_terms).mean()
            directed_skew_norm = torch.stack(level_directed_skew_terms).mean()
            directed_topology_loss = directed_transitive_loss + float(self.config.analogy_topology_cycle_weight) * directed_cycle_loss
            topology_terms.append(
                barcode_loss
                + closure_loss
                + float(self.config.analogy_topology_inclusion_weight) * inclusion_loss
                + float(self.config.analogy_topology_chain_weight) * chain_map_loss
                + float(self.config.analogy_topology_directed_weight) * directed_topology_loss
                + hdbscan_weight * hdbscan_loss
            )
            barcode_terms.append(barcode_loss)
            closure_terms.append(closure_loss)
            inclusion_terms.append(inclusion_loss)
            chain_map_terms.append(chain_map_loss)
            directed_terms.append(directed_topology_loss)
            directed_transitive_terms.append(directed_transitive_loss)
            directed_cycle_terms.append(directed_cycle_loss)
            directed_chain_terms.append(directed_chain_map_loss)
            directed_asymmetry_terms.append(directed_asymmetry)
            directed_skew_terms.append(directed_skew_norm)
            hdbscan_terms.append(hdbscan_loss)
            hdbscan_stability_terms.append(hdbscan_stability)
            hdbscan_persistent_edge_terms.append(hdbscan_persistent_edge_density)
            hdbscan_outlier_terms.append(hdbscan_outlier_score)
            hdbscan_core_radius_terms.append(hdbscan_core_radius)
            edge_density_terms.append(edge_density)
            triangle_density_terms.append(triangle_density)
        if topology_terms:
            topology_loss = torch.stack(topology_terms).mean()
            barcode_loss = torch.stack(barcode_terms).mean()
            simplex_closure_loss = torch.stack(closure_terms).mean()
            filtration_inclusion_loss = torch.stack(inclusion_terms).mean()
            chain_map_loss = torch.stack(chain_map_terms).mean()
            directed_topology_loss = torch.stack(directed_terms).mean()
            directed_transitive_loss = torch.stack(directed_transitive_terms).mean()
            directed_cycle_loss = torch.stack(directed_cycle_terms).mean()
            directed_chain_map_loss = torch.stack(directed_chain_terms).mean()
            directed_asymmetry = torch.stack(directed_asymmetry_terms).mean()
            directed_skew_norm = torch.stack(directed_skew_terms).mean()
            hdbscan_loss = torch.stack(hdbscan_terms).mean()
            hdbscan_stability = torch.stack(hdbscan_stability_terms).mean()
            hdbscan_persistent_edge_density = torch.stack(hdbscan_persistent_edge_terms).mean()
            hdbscan_outlier_score = torch.stack(hdbscan_outlier_terms).mean()
            hdbscan_core_radius = torch.stack(hdbscan_core_radius_terms).mean()
            filtration_edge_density = torch.stack(edge_density_terms).mean()
            filtration_triangle_density = torch.stack(triangle_density_terms).mean()
            topology_groups = relation.new_tensor(float(len(topology_terms)))
        else:
            topology_loss = zero
            barcode_loss = zero
            simplex_closure_loss = zero
            filtration_inclusion_loss = zero
            chain_map_loss = zero
            directed_topology_loss = zero
            directed_transitive_loss = zero
            directed_cycle_loss = zero
            directed_chain_map_loss = zero
            directed_asymmetry = zero
            directed_skew_norm = zero
            hdbscan_loss = zero
            hdbscan_stability = zero
            hdbscan_persistent_edge_density = zero
            hdbscan_outlier_score = zero
            hdbscan_core_radius = zero
            filtration_edge_density = zero
            filtration_triangle_density = zero

        step_topology = reasoning_step_topology_loss(
            hidden_chart,
            config=ReasoningTopologyConfig(
                max_points=int(self.config.analogy_step_topology_max_points),
                max_windows=int(self.config.analogy_step_topology_max_windows),
                window_size=int(self.config.analogy_step_topology_window_size),
                step_stride=int(self.config.analogy_step_topology_step_stride),
                levels=int(self.config.analogy_topology_filtration_levels),
                radius_min=float(self.config.analogy_topology_radius_min),
                radius_max=float(self.config.analogy_topology_radius_max),
                temperature=float(self.config.analogy_lattice_temperature),
                skew_scale=float(self.config.analogy_topology_skew_scale),
                time_bias=float(self.config.analogy_step_topology_time_bias),
                hdbscan_min_cluster_size=int(self.config.analogy_hdbscan_min_cluster_size),
                hdbscan_min_samples=int(self.config.analogy_hdbscan_min_samples),
                hdbscan_stability_threshold=float(self.config.analogy_hdbscan_stability_threshold),
            ),
        )
        topology_loss = topology_loss + float(self.config.analogy_step_topology_weight) * step_topology[
            "reasoning_step_topology_loss"
        ]

        if hidden.shape[1] > 2 * stride:
            rel_left = hidden[:, stride:-stride, :].float() - hidden[:, :-2 * stride, :].float()
            rel_right = hidden[:, 2 * stride :, :].float() - hidden[:, stride:-stride, :].float()
            key_left = classes[:, :-2 * stride] * 16 + classes[:, stride:-stride]
            key_right = classes[:, stride:-stride] * 16 + classes[:, 2 * stride :]
            same_arrow = key_left == key_right
            if bool(same_arrow.any()):
                closure = F.normalize(rel_left[same_arrow], dim=-1) - F.normalize(rel_right[same_arrow], dim=-1)
                parallelogram_loss = closure.pow(2).mean()
            else:
                parallelogram_loss = zero
        else:
            parallelogram_loss = zero

        if self.graphcg_direction_basis is not None:
            directions = F.normalize(self.graphcg_direction_basis.float(), dim=-1)
            coords = relation @ directions.transpose(0, 1)
            abs_coords = coords.abs()
            temperature = max(float(self.config.analogy_lattice_temperature), 1e-4)
            axis_probs = torch.softmax(abs_coords / temperature, dim=-1)
            signed_axis_probs = axis_probs * coords.sign()
            reconstructed = F.normalize(signed_axis_probs @ directions, dim=-1)
            basis_loss = (1.0 - (relation * reconstructed).sum(dim=-1)).mean()
            axis_entropy = -(axis_probs * (axis_probs + 1e-8).log()).sum(dim=-1).mean() / math.log(
                max(2, directions.shape[0])
            )
            if directions.shape[0] > 1:
                top2 = abs_coords.topk(2, dim=-1).values
                lattice_margin = (top2[:, 0] - top2[:, 1]).mean()
            else:
                lattice_margin = abs_coords.mean()
        else:
            basis_loss = zero
            axis_entropy = zero
            lattice_margin = zero

        total = (
            functor_loss
            + float(self.config.analogy_lattice_basis_weight) * basis_loss
            + float(self.config.analogy_lattice_parallelogram_weight) * parallelogram_loss
            + float(self.config.analogy_lattice_topology_weight) * topology_loss
        )
        return {
            "analogy_lattice_loss": total,
            "analogy_functor_loss": functor_loss.detach(),
            "analogy_parallelogram_loss": parallelogram_loss.detach(),
            "analogy_topology_loss": topology_loss.detach(),
            "analogy_barcode_loss": barcode_loss.detach(),
            "analogy_simplex_closure_loss": simplex_closure_loss.detach(),
            "analogy_filtration_inclusion_loss": filtration_inclusion_loss.detach(),
            "analogy_chain_map_loss": chain_map_loss.detach(),
            "analogy_directed_topology_loss": directed_topology_loss.detach(),
            "analogy_directed_transitive_loss": directed_transitive_loss.detach(),
            "analogy_directed_cycle_loss": directed_cycle_loss.detach(),
            "analogy_directed_chain_map_loss": directed_chain_map_loss.detach(),
            "analogy_directed_asymmetry": directed_asymmetry.detach(),
            "analogy_directed_skew_norm": directed_skew_norm.detach(),
            "analogy_hdbscan_loss": hdbscan_loss.detach(),
            "analogy_hdbscan_stability": hdbscan_stability.detach(),
            "analogy_hdbscan_persistent_edge_density": hdbscan_persistent_edge_density.detach(),
            "analogy_hdbscan_outlier_score": hdbscan_outlier_score.detach(),
            "analogy_hdbscan_core_radius": hdbscan_core_radius.detach(),
            "analogy_filtration_edge_density": filtration_edge_density.detach(),
            "analogy_filtration_triangle_density": filtration_triangle_density.detach(),
            "analogy_step_topology_loss": step_topology["reasoning_step_topology_loss"].detach(),
            "analogy_step_barcode_loss": step_topology["reasoning_step_barcode_loss"].detach(),
            "analogy_step_simplex_closure_loss": step_topology["reasoning_step_simplex_closure_loss"].detach(),
            "analogy_step_filtration_inclusion_loss": step_topology[
                "reasoning_step_filtration_inclusion_loss"
            ].detach(),
            "analogy_step_boundary_residual": step_topology["reasoning_step_boundary_residual"].detach(),
            "analogy_step_dirichlet_energy": step_topology["reasoning_step_dirichlet_energy"].detach(),
            "analogy_step_dec_conservation_loss": step_topology["reasoning_step_dec_conservation_loss"].detach(),
            "analogy_step_dec_mass_residual": step_topology["reasoning_step_dec_mass_residual"].detach(),
            "analogy_step_dec_vorticity_drift": step_topology["reasoning_step_dec_vorticity_drift"].detach(),
            "analogy_step_dec_kinetic_energy": step_topology["reasoning_step_dec_kinetic_energy"].detach(),
            "analogy_step_dec_kinetic_energy_drift": step_topology[
                "reasoning_step_dec_kinetic_energy_drift"
            ].detach(),
            "analogy_step_dec_hodge_balance": step_topology["reasoning_step_dec_hodge_balance"].detach(),
            "analogy_step_dec_wedge_interior_residual": step_topology[
                "reasoning_step_dec_wedge_interior_residual"
            ].detach(),
            "analogy_step_directed_topology_loss": step_topology["reasoning_step_directed_topology_loss"].detach(),
            "analogy_step_directed_transitive_loss": step_topology[
                "reasoning_step_directed_transitive_loss"
            ].detach(),
            "analogy_step_directed_cycle_flux": step_topology["reasoning_step_directed_cycle_flux"].detach(),
            "analogy_step_directed_chain_commutator": step_topology[
                "reasoning_step_directed_chain_commutator"
            ].detach(),
            "analogy_step_directed_asymmetry": step_topology["reasoning_step_directed_asymmetry"].detach(),
            "analogy_step_analogical_map_loss": step_topology["reasoning_step_analogical_map_loss"].detach(),
            "analogy_step_directed_map_loss": step_topology["reasoning_step_directed_map_loss"].detach(),
            "analogy_step_transport_entropy": step_topology["reasoning_step_transport_entropy"].detach(),
            "analogy_step_hdbscan_loss": step_topology["reasoning_step_hdbscan_loss"].detach(),
            "analogy_step_hdbscan_stability": step_topology["reasoning_step_hdbscan_stability"].detach(),
            "analogy_step_hdbscan_noise_fraction": step_topology[
                "reasoning_step_hdbscan_noise_fraction"
            ].detach(),
            "analogy_step_hdbscan_persistent_edge_density": step_topology[
                "reasoning_step_hdbscan_persistent_edge_density"
            ].detach(),
            "analogy_step_hdbscan_core_radius": step_topology["reasoning_step_hdbscan_core_radius"].detach(),
            "analogy_step_edge_density": step_topology["reasoning_step_edge_density"].detach(),
            "analogy_step_triangle_density": step_topology["reasoning_step_triangle_density"].detach(),
            "analogy_step_cycle_rank": step_topology["reasoning_step_cycle_rank"].detach(),
            "analogy_step_betti0": step_topology["reasoning_step_betti0"].detach(),
            "analogy_step_windows": step_topology["reasoning_step_windows"].detach(),
            "analogy_basis_loss": basis_loss.detach(),
            "analogy_axis_entropy": axis_entropy.detach(),
            "analogy_lattice_margin": lattice_margin.detach(),
            "analogy_relation_groups": relation_groups.detach(),
            "analogy_topology_groups": topology_groups.detach(),
            "analogy_graphcg_chart_dim": graphcg_chart_dim.detach(),
            "analogy_graphcg_chart_energy": graphcg_chart_energy.detach(),
        }

    def forward_from_previous(
        self,
        previous_tokens: torch.Tensor,
        target_positions: torch.Tensor,
        sample_gflownet: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        hidden = self._backbone_hidden(previous_tokens, target_positions)
        aux = self._gflownet_context(hidden, sample=sample_gflownet)
        if aux:
            hidden_for_logits = hidden + self.config.gflownet_action_scale * aux["context"]
        else:
            hidden_for_logits = hidden
        toric_memory = self._toric_memory_context(hidden_for_logits)
        toric_memory_entropy = None
        if toric_memory is not None:
            toric_context, toric_memory_entropy = toric_memory
            hidden_for_logits = hidden_for_logits + self.config.toric_memory_weight * toric_context
        logits = self._project_logits(hidden_for_logits)
        logits, smear_temperature = self._apply_smear_gate(hidden_for_logits, logits)
        if not return_aux:
            return logits
        out: dict[str, torch.Tensor] = {"logits": logits, "hidden": hidden}
        out.update({key: value for key, value in aux.items() if key != "context"})
        if smear_temperature is not None:
            out["smear_temperature"] = smear_temperature.mean()
        if toric_memory_entropy is not None:
            out["toric_memory_entropy"] = toric_memory_entropy.float()
        if self.gflownet_flow is not None:
            flow_hidden = torch.nan_to_num(hidden.float(), nan=0.0, posinf=30.0, neginf=-30.0).to(dtype=hidden.dtype)
            flow_log = self.gflownet_flow(flow_hidden).squeeze(-1)
            out["flow_log"] = torch.nan_to_num(flow_log.float(), nan=0.0, posinf=30.0, neginf=-30.0).to(
                dtype=flow_log.dtype
            )
        return out

    def _supervised_and_gflownet_losses(
        self,
        aux: dict[str, torch.Tensor],
        target_tokens: torch.Tensor,
        target_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = aux["logits"]
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = target_tokens.reshape(-1)
        per_token_nll = F.cross_entropy(flat_logits, flat_targets, reduction="none").view_as(target_tokens)
        loss = per_token_nll.mean()
        out: dict[str, torch.Tensor] = {"loss": loss}
        if "neural_logits" in aux:
            neural_nll = F.cross_entropy(
                aux["neural_logits"].reshape(-1, aux["neural_logits"].shape[-1]),
                flat_targets,
                reduction="none",
            ).view_as(target_tokens)
            out["neural_loss"] = neural_nll.mean()
        if "revealed_neighbor_known_fraction" in aux:
            out["revealed_neighbor_known_fraction"] = aux["revealed_neighbor_known_fraction"].float()
        if "revealed_neighbor_context_norm" in aux:
            out["revealed_neighbor_context_norm"] = aux["revealed_neighbor_context_norm"].float()
        if "revealed_context_prior_weight" in aux:
            out["revealed_context_prior_weight"] = aux["revealed_context_prior_weight"].float()
        hidden = aux.get("hidden")
        if hidden is not None and self.aux_mtp_heads:
            mtp_losses = []
            for offset, head in enumerate(self.aux_mtp_heads, start=1):
                if target_tokens.shape[1] <= offset:
                    continue
                offset_logits = head(hidden[:, :-offset, :])
                offset_targets = target_tokens[:, offset:]
                mtp_losses.append(
                    F.cross_entropy(
                        offset_logits.reshape(-1, offset_logits.shape[-1]),
                        offset_targets.reshape(-1),
                    )
                )
            if mtp_losses:
                out["mtp_loss"] = torch.stack(mtp_losses).mean()
        if "action_logprob" in aux and self.gflownet_log_z is not None:
            log_reward = -per_token_nll.detach()
            tb_error = self.gflownet_log_z + aux["action_logprob"].float() - log_reward.float()
            out["gflownet_loss"] = tb_error.pow(2).mean()
        elif "flow_log" in aux:
            out["gflownet_loss"] = (aux["flow_log"].float() + per_token_nll.detach().float()).pow(2).mean()
        if "policy_entropy" in aux:
            out["gflownet_entropy"] = aux["policy_entropy"].float()
        if "action_diversity" in aux:
            out["gflownet_action_diversity"] = aux["action_diversity"].float()
        if "smear_temperature" in aux:
            out["smear_temperature"] = aux["smear_temperature"].float()
        if "toric_memory_entropy" in aux:
            out["toric_memory_entropy"] = aux["toric_memory_entropy"].float()
        hidden = aux.get("hidden")
        if hidden is not None:
            out.update(self._graphcg_losses(hidden))
            out.update(self._analogy_lattice_losses(hidden, target_tokens))
            if self.toric_geometry_probe is not None and target_positions is not None:
                out.update(self.toric_geometry_probe(hidden, target_positions, target_tokens))
            if self.trajectory_memory_head is not None:
                out.update(
                    self.trajectory_memory_head(
                        hidden,
                        target_positions,
                        per_token_nll,
                        graphcg_basis=self.graphcg_direction_basis,
                    )
                )
            if bool(self.config.use_koszul_persistence):
                out.update(
                    koszul_persistence_loss(
                        hidden,
                        target_positions,
                        config=KoszulPersistenceConfig(
                            max_points=int(self.config.koszul_max_points),
                            max_windows=int(self.config.koszul_max_windows),
                            window_size=int(self.config.koszul_window_size),
                            step_stride=int(self.config.koszul_step_stride),
                            num_parameters=int(self.config.koszul_num_parameters),
                            temperature=float(self.config.koszul_temperature),
                            chart_exponents=int(self.config.koszul_chart_exponents),
                            theta=float(self.config.theta),
                            beta=float(self.config.beta),
                            rank_temperature=float(self.config.koszul_rank_temperature),
                        ),
                    )
                )
        if hidden is not None and hidden.shape[0] > 1:
            pooled = F.normalize(hidden.mean(dim=1).float(), dim=-1)
            sim = pooled @ pooled.transpose(0, 1) / max(float(self.config.contrastive_temperature), 1e-4)
            labels = torch.arange(pooled.shape[0], device=pooled.device)
            out["contrastive_loss"] = F.cross_entropy(sim, labels)
        if hidden is not None and hidden.shape[1] > 2:
            velocity = hidden[:, 1:, :].float() - hidden[:, :-1, :].float()
            acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
            kinetic = velocity.pow(2).mean()
            viscous = acceleration.pow(2).mean()
            norm_variation = hidden.float().norm(dim=-1).var(dim=1, unbiased=False).mean()
            out["trajectory_flow_loss"] = (
                viscous
                + float(self.config.trajectory_flow_viscosity) * kinetic
                + 0.001 * norm_variation
            )
            out["trajectory_kinetic_energy"] = kinetic.detach()
            out["trajectory_viscous_dissipation"] = viscous.detach()
        return out

    def forward(
        self,
        tokens: torch.Tensor,
        sample_ids: torch.Tensor | None = None,
        pass_id: int = 0,
        permutation: torch.Tensor | None = None,
        return_order: bool = False,
        sample_gflownet: bool = False,
        gflownet_samples: int = 1,
    ) -> dict[str, torch.Tensor]:
        batch = random_order_batch(
            tokens,
            seed=self.config.seed,
            sample_ids=sample_ids,
            pass_id=pass_id,
            bos_token_id=self.bos_token_id,
            permutation=permutation,
        )
        order_aux: dict[str, torch.Tensor] | None = None
        if self.config.use_gflownet_policy and gflownet_samples > 1:
            sample_losses = []
            sample_logps = []
            entropies = []
            diversities = []
            last_aux: dict[str, torch.Tensor] | None = None
            for _ in range(gflownet_samples):
                aux = self.forward_from_previous(
                    batch.previous_tokens,
                    batch.target_positions,
                    sample_gflownet=True,
                    return_aux=True,
                )
                if not isinstance(aux, dict):
                    raise RuntimeError("expected auxiliary output")
                aux = self._add_revealed_neighbor_context(aux, batch.target_tokens, batch.target_positions)
                losses = self._supervised_and_gflownet_losses(aux, batch.target_tokens, batch.target_positions)
                logp = F.log_softmax(aux["logits"], dim=-1).gather(-1, batch.target_tokens.unsqueeze(-1)).squeeze(-1)
                sample_logps.append(logp)
                sample_losses.append(losses["loss"])
                if "gflownet_entropy" in losses:
                    entropies.append(losses["gflownet_entropy"])
                if "gflownet_action_diversity" in losses:
                    diversities.append(losses["gflownet_action_diversity"])
                last_aux = aux
            order_aux = last_aux
            mixture_logp = torch.logsumexp(torch.stack(sample_logps, dim=0), dim=0) - math.log(gflownet_samples)
            loss = -mixture_logp.mean()
            logits = last_aux["logits"] if last_aux is not None else torch.empty(0, device=tokens.device)
            aux_losses: dict[str, torch.Tensor] = {
                "loss": loss,
                "single_sample_loss": torch.stack(sample_losses).mean(),
            }
            if entropies:
                aux_losses["gflownet_entropy"] = torch.stack(entropies).mean()
            if diversities:
                aux_losses["gflownet_action_diversity"] = torch.stack(diversities).mean()
        else:
            aux = self.forward_from_previous(
                batch.previous_tokens,
                batch.target_positions,
                sample_gflownet=sample_gflownet and self.config.use_gflownet_policy,
                return_aux=True,
            )
            if not isinstance(aux, dict):
                raise RuntimeError("expected auxiliary output")
            aux = self._add_revealed_neighbor_context(aux, batch.target_tokens, batch.target_positions)
            logits = aux["logits"]
            order_aux = aux
            aux_losses = self._supervised_and_gflownet_losses(aux, batch.target_tokens, batch.target_positions)
            loss = aux_losses["loss"]
        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "loss": loss,
            "bpb": loss.detach() / math.log(2),
        }
        out.update({key: value for key, value in aux_losses.items() if key != "loss"})
        if return_order:
            out.update(
                {
                    "previous_tokens": batch.previous_tokens,
                    "target_tokens": batch.target_tokens,
                    "target_positions": batch.target_positions,
                    "permutation": batch.permutation,
                }
            )
            if order_aux is not None and "action_ids" in order_aux:
                out["gflownet_action_ids"] = order_aux["action_ids"]
        return out

    @torch.no_grad()
    def score_with_bias_adaptation(
        self,
        tokens: torch.Tensor,
        sample_ids: torch.Tensor | None = None,
        pass_id: int = 0,
        permutation: torch.Tensor | None = None,
        lr: float = 0.025,
        decay: float = 0.98,
        clip: float = 3.0,
        gflownet_samples: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Evaluate a legal score-before-update output-bias adapter.

        The base model computes prefix-causal log probabilities.  A per-sequence
        bias vector is then updated only after each revealed target has been
        scored.  This is a small, deterministic test-time adaptation analogue:
        it can improve local byte frequencies without touching deployment
        weights or reading future bytes.
        """

        batch = random_order_batch(
            tokens,
            seed=self.config.seed,
            sample_ids=sample_ids,
            pass_id=pass_id,
            bos_token_id=self.bos_token_id,
            permutation=permutation,
        )
        sample_log_probs = []
        for sample in range(max(1, gflownet_samples)):
            aux = self.forward_from_previous(
                batch.previous_tokens,
                batch.target_positions,
                sample_gflownet=self.config.use_gflownet_policy and gflownet_samples > 1,
                return_aux=True,
            )
            if not isinstance(aux, dict):
                raise RuntimeError("expected auxiliary output")
            aux = self._add_revealed_neighbor_context(aux, batch.target_tokens, batch.target_positions)
            sample_log_probs.append(F.log_softmax(aux["logits"].float(), dim=-1))
        base_log_probs = torch.logsumexp(torch.stack(sample_log_probs, dim=0), dim=0) - math.log(
            max(1, gflownet_samples)
        )
        bias = torch.zeros(tokens.shape[0], self.config.vocab_size, device=tokens.device, dtype=base_log_probs.dtype)
        losses = []
        for step in range(tokens.shape[1]):
            adapted_log_probs = F.log_softmax(base_log_probs[:, step, :] + bias, dim=-1)
            target = batch.target_tokens[:, step]
            losses.append(-adapted_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1))
            probs = adapted_log_probs.exp()
            probs.scatter_add_(1, target.unsqueeze(-1), -torch.ones_like(target, dtype=probs.dtype).unsqueeze(-1))
            bias = (decay * bias - lr * probs).clamp(-clip, clip)
        loss = torch.stack(losses, dim=1).mean()
        return {
            "loss": loss,
            "bpb": loss / math.log(2),
            "bias_norm": bias.norm(dim=-1).mean(),
        }

    @torch.no_grad()
    def causal_future_permutation_error(
        self,
        tokens: torch.Tensor,
        step: int | None = None,
        permutation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Measure whether future token changes alter the current-step logits."""

        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, length]")
        length = tokens.shape[1]
        if step is None:
            step = max(0, length // 2)
        if not 0 <= step < length:
            raise ValueError("step must be in range")
        if permutation is None:
            permutation = torch.arange(length, device=tokens.device).view(1, -1).expand(tokens.shape[0], -1)
        mutated = tokens.clone()
        if step + 1 < length:
            future = mutated[:, step + 1 :].flip(dims=[1])
            mutated[:, step + 1 :] = torch.remainder(future + 17, max(1, self.config.vocab_size - self.config.byte_offset))
            mutated[:, step + 1 :] = mutated[:, step + 1 :].clamp_min(self.config.byte_offset)
        out_a = self(tokens, permutation=permutation, return_order=True)
        out_b = self(mutated, permutation=permutation, return_order=True)
        return (out_a["logits"][:, step, :] - out_b["logits"][:, step, :]).float().abs().max()

    @torch.no_grad()
    def sample_random_order(
        self,
        length: int,
        seed: int,
        sample_id: int = 0,
        temperature: float = 1.0,
        top_k: int | None = None,
        device: torch.device | str | None = None,
        sample_gflownet: bool | None = None,
    ) -> torch.Tensor:
        """Generate one sequence by filling a random target-position order."""

        if device is None:
            device = next(self.parameters()).device
        if sample_gflownet is None:
            sample_gflownet = self.config.use_gflownet_policy
        order = random_order_permutation(length, seed=seed, sample_id=sample_id, device=device).view(1, length)
        previous = torch.full((1, length), self.config.pad_token_id, dtype=torch.long, device=device)
        previous[:, 0] = self.bos_token_id
        generated = torch.full((1, length), self.config.pad_token_id, dtype=torch.long, device=device)
        for step in range(length):
            logits = self.forward_from_previous(previous, order, sample_gflownet=sample_gflownet)
            if isinstance(logits, dict):
                logits = logits["logits"]
            logits = logits[:, step, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, _ = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
                logits = logits.masked_fill(logits < values[:, -1:], torch.finfo(logits.dtype).min)
            token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(1)
            position = order[:, step]
            generated.scatter_(1, position.view(1, 1), token.view(1, 1))
            if step + 1 < length:
                previous[:, step + 1] = token
        return generated.squeeze(0)

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


def estimate_uncompressed_quantized_bytes(
    model: nn.Module,
    bits: Literal[4, 6, 8] = 8,
    exclude_prefixes: tuple[str, ...] = ("aux_", "graphcg_direction_basis", "toric_geometry_probe.", "trajectory_memory_head."),
) -> int:
    """Conservative tensor-only byte estimate before zip compression."""

    total_bits = 0
    for name, tensor in model.state_dict().items():
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if torch.is_floating_point(tensor):
            total_bits += tensor.numel() * bits
        else:
            total_bits += tensor.numel() * tensor.element_size() * 8
    return (total_bits + 7) // 8
