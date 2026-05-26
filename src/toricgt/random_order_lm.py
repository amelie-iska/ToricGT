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
        if config.weight_tying:
            self.output = None
            self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))
        else:
            self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.output_bias = None
        self.apply(self._init_module)
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
        logits = self._project_logits(hidden_for_logits)
        if not return_aux:
            return logits
        out: dict[str, torch.Tensor] = {"logits": logits, "hidden": hidden}
        out.update({key: value for key, value in aux.items() if key != "context"})
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
        return out

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


def estimate_uncompressed_quantized_bytes(model: nn.Module, bits: Literal[4, 6, 8] = 8) -> int:
    """Conservative tensor-only byte estimate before zip compression."""

    total_bits = 0
    for tensor in model.state_dict().values():
        if torch.is_floating_point(tensor):
            total_bits += tensor.numel() * bits
        else:
            total_bits += tensor.numel() * tensor.element_size() * 8
    return (total_bits + 7) // 8
