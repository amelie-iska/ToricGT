"""TokenGT-style graph-to-graph model with tropical ring attention."""

from __future__ import annotations

import torch
from torch import nn

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
        self.lm_head = nn.Linear(config.d_model, config.lm_vocab_size) if config.use_lm_head else None
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
        mask = attention_mask_from_token_mask(tok.token_mask)
        for block in self.blocks:
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
        if self.lm_head is not None:
            outputs["lm_logits"] = self.lm_head(node_x)
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
