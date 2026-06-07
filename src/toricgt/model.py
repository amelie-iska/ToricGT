"""TokenGT-style graph-to-graph model with tropical ring attention."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .derived_category_metrics import derived_category_objects_from_batch
from .gflownet import EmbeddingPolicy
from .graph_tokenizer import GraphBatch, GraphTokenizer, attention_mask_from_token_mask
from .tropical_attention import TransformerBlock
from .trajectory_memory import TrajectoryMemoryConfig, TrajectoryRetrievalHead


class ToricTokenGT(nn.Module):
    """Graph-to-graph encoder-decoder over node and edge tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = GraphTokenizer(
            node_feature_dim=config.node_feature_dim,
            edge_feature_dim=config.edge_feature_dim,
            d_model=config.d_model,
            max_nodes=config.max_nodes,
            max_edges=config.max_edges,
            torus_rank=config.torus_rank,
            theta=config.theta,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ffn_multiplier=config.ffn_multiplier,
                    attention=self._layer_attention_mode(layer_idx),
                    dropout=config.dropout,
                    ring_block_size=config.ring_block_size,
                    use_soft_moe=self._use_soft_moe(layer_idx),
                    soft_moe_num_experts=config.soft_moe_num_experts,
                    soft_moe_slots_per_expert=config.soft_moe_slots_per_expert,
                    soft_moe_residual_scale=config.soft_moe_residual_scale,
                )
                for layer_idx in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.node_head = nn.Linear(config.d_model, config.output_dim)
        self.edge_head = nn.Linear(config.d_model, config.output_dim)
        self.lm_token_emb = (
            nn.Embedding(config.lm_vocab_size, config.d_model)
            if config.use_lm_head and config.use_lm_token_embeddings
            else None
        )
        self.tie_lm_head = bool(
            config.use_lm_head and config.tie_lm_head_to_token_embeddings and self.lm_token_emb is not None
        )
        self.lm_head = (
            None
            if self.tie_lm_head or not config.use_lm_head
            else nn.Linear(config.d_model, config.lm_vocab_size)
        )
        self.lm_output_bias = nn.Parameter(torch.zeros(config.lm_vocab_size)) if self.tie_lm_head else None
        self.lm_position_emb = (
            nn.Embedding(max(1, config.lm_max_positions), config.d_model)
            if config.use_lm_head and config.use_lm_position_embeddings
            else None
        )
        self.lm_toric_phase = (
            nn.Linear(4, config.d_model, bias=False)
            if config.use_lm_head and config.use_lm_toric_position_features
            else None
        )
        self.lm_context_hash = (
            nn.Embedding(config.lm_context_hash_buckets, config.d_model)
            if config.use_lm_head and config.use_lm_context_hash_embeddings and config.lm_context_hash_buckets > 0
            else None
        )
        self.lm_revealed_neighbor_hash = (
            nn.Embedding(config.lm_revealed_neighbor_hash_buckets, config.d_model)
            if config.use_lm_head
            and config.use_lm_revealed_neighbor_hash
            and config.lm_revealed_neighbor_hash_buckets > 0
            else None
        )
        self.lm_caseops = (
            nn.Linear(config.lm_caseops_feature_dim, config.d_model, bias=False)
            if config.use_lm_head and config.use_lm_caseops_features
            else None
        )
        self.lm_smear_gate = (
            nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
            if config.use_lm_head and config.use_lm_smear_gate
            else None
        )
        self.lm_bigram_bias = (
            nn.Embedding(config.lm_vocab_size, config.lm_vocab_size)
            if config.use_lm_head and config.use_lm_bigram_bias
            else None
        )
        self.graph_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.output_dim),
        )
        self.gflownet_policy = (
            EmbeddingPolicy(
                embedding_dim=config.d_model,
                num_actions=config.gflownet_num_actions,
                hidden_dim=config.gflownet_hidden_dim,
            )
            if config.use_gflownet_head
            else None
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
                    dag_weight=config.trajectory_memory_dag_weight,
                    derived_weight=config.trajectory_memory_derived_weight,
                ),
            )
            if config.use_trajectory_memory_head
            else None
        )

    def _layer_attention_mode(self, layer_idx: int) -> str:
        if self.config.attention != "hybrid":
            return self.config.attention
        return "softmax" if layer_idx < self.config.num_layers // 2 else "tropical_ring"

    def _use_soft_moe(self, layer_idx: int) -> bool:
        if not self.config.use_soft_moe:
            return False
        start = self.config.soft_moe_start_layer
        if start is None:
            start = self.config.num_layers // 2
        return layer_idx >= start

    def _lm_positions(self, batch: GraphBatch, node_count: int, device: torch.device) -> torch.Tensor:
        if batch.lm_target_positions is not None:
            positions = batch.lm_target_positions[:, :node_count].to(device=device, dtype=torch.long)
        else:
            positions = torch.arange(node_count, device=device, dtype=torch.long).view(1, -1)
            positions = positions.expand(batch.node_features.shape[0], -1)
        return positions

    def _lm_phase_features(self, positions: torch.Tensor) -> torch.Tensor:
        pos = positions.to(dtype=torch.float32)
        theta = 2.0 * math.pi * self.config.theta * pos
        beta = 2.0 * math.pi * 1.4142135623730951 * pos
        return torch.stack([torch.sin(theta), torch.cos(theta), torch.sin(beta), torch.cos(beta)], dim=-1)

    def _previous_reveal_tokens(self, input_ids: torch.Tensor, ranks: torch.Tensor | None) -> torch.Tensor:
        bos = int(max(0, min(self.config.lm_bos_token_id, self.config.lm_vocab_size - 1)))
        if ranks is None:
            prev = torch.empty_like(input_ids)
            prev[:, 0] = bos
            if input_ids.shape[1] > 1:
                prev[:, 1:] = input_ids[:, :-1]
            return prev
        valid_rank = ranks.to(device=input_ids.device, dtype=torch.long)
        order = torch.argsort(valid_rank, dim=1, stable=True)
        sorted_inputs = torch.gather(input_ids, 1, order)
        prev_sorted = torch.empty_like(sorted_inputs)
        prev_sorted[:, 0] = bos
        if sorted_inputs.shape[1] > 1:
            prev_sorted[:, 1:] = sorted_inputs[:, :-1]
        prev = torch.empty_like(input_ids)
        prev.scatter_(1, order, prev_sorted)
        return prev

    def _lm_context_hash_ids(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ranks: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.lm_context_hash is None:
            raise RuntimeError("LM context hash is disabled")
        prev_reveal = self._previous_reveal_tokens(input_ids, ranks)
        if ranks is None:
            ranks_i64 = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand_as(input_ids)
        else:
            ranks_i64 = ranks.to(device=input_ids.device, dtype=torch.long)
        hashed = (
            prev_reveal.to(torch.int64) * 1_000_003
            + input_ids.to(torch.int64) * 917_609
            + positions.to(torch.int64) * 65_537
            + ranks_i64.to(torch.int64) * 32_771
        )
        return torch.remainder(hashed, int(self.config.lm_context_hash_buckets)).to(torch.long)

    def _lm_revealed_neighbor_context(
        self,
        batch: GraphBatch,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ranks: torch.Tensor,
        node_count: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Aggregate rank-safe graph-neighbor hashes from already revealed nodes."""

        if self.lm_revealed_neighbor_hash is None:
            return None, None
        if batch.edge_index.numel() == 0 or batch.edge_mask.numel() == 0:
            return None, None
        device = input_ids.device
        edge_index = batch.edge_index.to(device=device, dtype=torch.long)
        edge_mask = batch.edge_mask.to(device=device, dtype=torch.bool)
        node_mask = batch.node_mask[:, :node_count].to(device=device, dtype=torch.bool)
        context = input_ids.new_zeros((input_ids.shape[0], node_count, self.config.d_model), dtype=torch.float32)
        counts = input_ids.new_zeros((input_ids.shape[0], node_count), dtype=torch.float32)
        buckets = int(self.config.lm_revealed_neighbor_hash_buckets)
        for batch_idx in range(input_ids.shape[0]):
            active = edge_mask[batch_idx]
            if not bool(active.any().detach().cpu()):
                continue
            endpoints = edge_index[batch_idx, active, :].clamp(0, max(node_count - 1, 0))
            src = endpoints[:, 0]
            dst = endpoints[:, 1]
            valid = node_mask[batch_idx, src] & node_mask[batch_idx, dst]
            if not bool(valid.any().detach().cpu()):
                continue
            src = src[valid]
            dst = dst[valid]
            src_rank = ranks[batch_idx, src]
            dst_rank = ranks[batch_idx, dst]
            for query, neighbor, direction_code in (
                (dst, src, 1),
                (src, dst, 2),
            ):
                query_rank = ranks[batch_idx, query]
                neighbor_rank = ranks[batch_idx, neighbor]
                legal = neighbor_rank < query_rank
                if not bool(legal.any().detach().cpu()):
                    continue
                query = query[legal]
                neighbor = neighbor[legal]
                hashed = (
                    input_ids[batch_idx, neighbor].to(torch.int64) * 1_000_003
                    + input_ids[batch_idx, query].to(torch.int64) * 917_609
                    + positions[batch_idx, query].to(torch.int64) * 65_537
                    + ranks[batch_idx, query].to(torch.int64) * 32_771
                    + int(direction_code) * 8_191
                )
                hash_ids = torch.remainder(hashed, buckets).to(torch.long)
                context[batch_idx].index_add_(0, query, self.lm_revealed_neighbor_hash(hash_ids).float())
                counts[batch_idx].index_add_(0, query, torch.ones_like(query, dtype=torch.float32))
        if not bool((counts > 0).any().detach().cpu()):
            return None, None
        context = context / counts.clamp_min(1.0).unsqueeze(-1)
        known = ((counts > 0) & node_mask).to(torch.float32)
        known_fraction = known.sum() / node_mask.to(torch.float32).sum().clamp_min(1.0)
        return context.to(device=device, dtype=dtype), known_fraction

    def _apply_lm_smear_gate(
        self,
        node_x: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.lm_smear_gate is None:
            return logits, None
        lo = float(self.config.lm_smear_temperature_min)
        hi = float(self.config.lm_smear_temperature_max)
        if hi <= lo:
            raise ValueError("lm_smear_temperature_max must be greater than lm_smear_temperature_min")
        raw = self.lm_smear_gate(node_x).squeeze(-1).float()
        temperature = lo + (hi - lo) * torch.sigmoid(raw)
        scaled = logits.float() / temperature.unsqueeze(-1).clamp_min(1e-4)
        return scaled.to(dtype=logits.dtype), temperature.mean()

    def forward(
        self,
        batch: GraphBatch,
        *,
        include_derived_category: bool | None = None,
        include_memory_trace: bool = False,
        memory_trace_top_k: int = 3,
    ) -> dict[str, torch.Tensor | object]:
        tok = self.tokenizer(batch)
        x = tok.tokens
        input_ids = None
        lm_input = None
        node_rank = None
        lm_revealed_neighbor_known_fraction = None
        if batch.node_causal_rank is not None:
            node_rank = batch.node_causal_rank[:, : tok.node_positions.numel()].to(device=x.device, dtype=torch.long)
        if batch.lm_input_ids is not None:
            input_ids = batch.lm_input_ids[:, : tok.node_positions.numel()].to(
                device=x.device,
                dtype=torch.long,
            ).clamp(0, self.config.lm_vocab_size - 1)
        if self.lm_token_emb is not None and input_ids is not None:
            lm_input = self.lm_token_emb(input_ids)
        if self.lm_position_emb is not None:
            positions = self._lm_positions(batch, tok.node_positions.numel(), x.device)
            pos_emb = self.lm_position_emb(positions.clamp(0, self.config.lm_max_positions - 1))
            lm_input = pos_emb * float(self.config.lm_position_weight) if lm_input is None else lm_input + float(
                self.config.lm_position_weight
            ) * pos_emb
        if self.lm_toric_phase is not None:
            positions = self._lm_positions(batch, tok.node_positions.numel(), x.device)
            phase = self.lm_toric_phase(self._lm_phase_features(positions).to(device=x.device, dtype=x.dtype))
            lm_input = phase * float(self.config.lm_toric_position_weight) if lm_input is None else lm_input + float(
                self.config.lm_toric_position_weight
            ) * phase
        if self.lm_context_hash is not None and input_ids is not None:
            positions = self._lm_positions(batch, tok.node_positions.numel(), x.device)
            hash_ids = self._lm_context_hash_ids(input_ids, positions, node_rank)
            hashed = self.lm_context_hash(hash_ids)
            lm_input = hashed * float(self.config.lm_context_hash_weight) if lm_input is None else lm_input + float(
                self.config.lm_context_hash_weight
            ) * hashed
        if self.lm_revealed_neighbor_hash is not None and input_ids is not None and node_rank is not None:
            positions = self._lm_positions(batch, tok.node_positions.numel(), x.device)
            neighbor_context, lm_revealed_neighbor_known_fraction = self._lm_revealed_neighbor_context(
                batch,
                input_ids,
                positions,
                node_rank,
                tok.node_positions.numel(),
                x.dtype,
            )
            if neighbor_context is not None:
                lm_input = (
                    neighbor_context * float(self.config.lm_revealed_neighbor_hash_weight)
                    if lm_input is None
                    else lm_input + float(self.config.lm_revealed_neighbor_hash_weight) * neighbor_context
                )
        if self.lm_caseops is not None and batch.lm_input_features is not None:
            features = batch.lm_input_features[:, : tok.node_positions.numel(), :].to(device=x.device, dtype=x.dtype)
            caseops = self.lm_caseops(features)
            lm_input = caseops * float(self.config.lm_caseops_weight) if lm_input is None else lm_input + float(
                self.config.lm_caseops_weight
            ) * caseops
        if lm_input is not None:
            if batch.lm_mask is not None:
                lm_input = lm_input * batch.lm_mask[:, : tok.node_positions.numel()].to(
                    device=x.device,
                    dtype=lm_input.dtype,
                ).unsqueeze(-1)
            x = x.clone()
            x[:, tok.node_positions, :] = x[:, tok.node_positions, :] + lm_input
        causal_rank = tok.causal_rank if self.config.use_causal_graph_attention else None
        mask = attention_mask_from_token_mask(tok.token_mask, causal_rank=causal_rank)
        for _ in range(max(1, int(self.config.recurrent_passes))):
            for block in self.blocks:
                if self.training and self.config.activation_checkpointing:
                    x = checkpoint(
                        lambda block_x, block=block: block(block_x, mask=mask, token_mask=tok.token_mask),
                        x,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, mask=mask, token_mask=tok.token_mask)
        x = self.norm(x)

        node_x = x[:, tok.node_positions, :]
        edge_x = x[:, tok.edge_positions, :]
        denom = tok.token_mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
        pooled = (x * tok.token_mask.unsqueeze(-1).to(x.dtype)).sum(dim=1) / denom
        outputs = {
            "node": self.node_head(node_x),
            "edge": self.edge_head(edge_x),
            "graph": self.graph_head(pooled),
            "token_embeddings": x,
            "node_embeddings": node_x,
            "edge_embeddings": edge_x,
            "token_mask": tok.token_mask,
            "node_mask": batch.node_mask,
            "edge_index": batch.edge_index,
            "edge_mask": batch.edge_mask,
        }
        if lm_revealed_neighbor_known_fraction is not None:
            outputs["lm_revealed_neighbor_known_fraction"] = lm_revealed_neighbor_known_fraction
        if self.config.use_lm_head:
            if self.tie_lm_head and self.lm_token_emb is not None:
                lm_logits = F.linear(node_x, self.lm_token_emb.weight[: self.config.lm_vocab_size], self.lm_output_bias)
            elif self.lm_head is not None:
                lm_logits = self.lm_head(node_x)
            else:
                raise RuntimeError("LM head is enabled but no output projection is configured")
            if self.lm_bigram_bias is not None and batch.lm_input_ids is not None:
                input_ids = batch.lm_input_ids.to(device=lm_logits.device, dtype=torch.long).clamp(
                    0,
                    self.config.lm_vocab_size - 1,
                )
                lm_logits = lm_logits + self.config.lm_bigram_bias_scale * self.lm_bigram_bias(input_ids)
            lm_logits, smear_temperature = self._apply_lm_smear_gate(node_x, lm_logits)
            outputs["lm_logits"] = lm_logits
            if smear_temperature is not None:
                outputs["lm_smear_temperature"] = smear_temperature
        if self.gflownet_policy is not None:
            forward_logits, backward_logits = self.gflownet_policy(pooled)
            outputs["gflownet_forward_logits"] = forward_logits
            outputs["gflownet_backward_logits"] = backward_logits
            outputs["gflownet_log_z"] = self.gflownet_policy.log_z
        emit_derived = self.config.output_derived_category_certificates if include_derived_category is None else bool(include_derived_category)
        if emit_derived:
            outputs["derived_category"] = derived_category_objects_from_batch(
                batch.edge_index,
                node_mask=batch.node_mask,
                edge_mask=batch.edge_mask,
                max_vertices=self.config.derived_category_max_vertices,
            )
        if include_memory_trace:
            if self.trajectory_memory_head is None:
                outputs["memory_trace"] = {
                    "kind": "trajectory_memory_analogical_retrieval_trace",
                    "enabled": False,
                    "reason": "trajectory_memory_head_disabled",
                }
            else:
                positions = torch.arange(node_x.shape[1], device=node_x.device).unsqueeze(0).expand(node_x.shape[0], -1)
                neutral_nll = node_x.new_zeros(node_x.shape[:2])
                outputs["memory_trace"] = self.trajectory_memory_head.trace(
                    node_x,
                    positions,
                    neutral_nll,
                    trajectory_node_mask=batch.node_mask,
                    trajectory_edge_index=batch.edge_index,
                    trajectory_edge_mask=batch.edge_mask,
                    top_k=memory_trace_top_k,
                )
        return outputs

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def soft_moe_diagnostics(self) -> list[dict[str, float]]:
        diagnostics: list[dict[str, float]] = []
        for layer_idx, block in enumerate(self.blocks):
            if not getattr(block, "uses_soft_moe", False):
                continue
            diag = block.ffn.diagnostics()
            if diag is None:
                continue
            diagnostics.append(
                {
                    "layer": float(layer_idx),
                    "expert_mass_min": float(diag.expert_mass.min()),
                    "expert_mass_max": float(diag.expert_mass.max()),
                    "dispatch_entropy_mean": float(diag.dispatch_entropy.mean()),
                    "combine_entropy": float(diag.combine_entropy),
                }
            )
        return diagnostics

    def set_active_soft_moe_experts(self, expert_ids: list[int] | tuple[int, ...] | None) -> None:
        """Restrict all Soft-MoE blocks to selected experts for curricula."""

        for block in self.blocks:
            if getattr(block, "uses_soft_moe", False):
                block.set_active_experts(expert_ids)
