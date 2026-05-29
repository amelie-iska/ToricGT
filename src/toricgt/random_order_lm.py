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
from .tropical_attention import TransformerBlock


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
    use_smear_gate: bool = True
    smear_temperature_min: float = 0.55
    smear_temperature_max: float = 1.75
    use_toric_memory: bool = True
    toric_memory_slots: int = 32
    toric_memory_weight: float = 0.08
    use_graphcg: bool = False
    graphcg_num_directions: int = 12
    graphcg_alpha: float = 0.12
    graphcg_temperature: float = 0.2
    graphcg_max_codes: int = 256
    graphcg_orthogonal_weight: float = 0.2
    graphcg_covariance_weight: float = 0.05
    graphcg_sparsity_weight: float = 0.0001
    aux_mtp_offsets: int = 2
    contrastive_temperature: float = 0.2
    trajectory_flow_viscosity: float = 0.05
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
        self.apply(self._init_module)
        if self.toric_memory_value is not None:
            nn.init.normal_(self.toric_memory_value, mean=0.0, std=0.02)
        if self.graphcg_direction_basis is not None:
            nn.init.orthogonal_(self.graphcg_direction_basis)
        nn.init.zeros_(self.output_bias) if self.output_bias is not None else None

    def _init_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
        policy_logits = self.gflownet_policy(hidden)
        policy_logits_f = policy_logits.float()
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
            out["flow_log"] = self.gflownet_flow(hidden).squeeze(-1)
        return out

    def _supervised_and_gflownet_losses(
        self,
        aux: dict[str, torch.Tensor],
        target_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = aux["logits"]
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = target_tokens.reshape(-1)
        per_token_nll = F.cross_entropy(flat_logits, flat_targets, reduction="none").view_as(target_tokens)
        loss = per_token_nll.mean()
        out: dict[str, torch.Tensor] = {"loss": loss}
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
                losses = self._supervised_and_gflownet_losses(aux, batch.target_tokens)
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
            logits = aux["logits"]
            order_aux = aux
            aux_losses = self._supervised_and_gflownet_losses(aux, batch.target_tokens)
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
    exclude_prefixes: tuple[str, ...] = ("aux_",),
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
