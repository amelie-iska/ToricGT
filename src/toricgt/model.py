"""TokenGT-style graph-to-graph model with tropical ring attention."""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .gflownet import EmbeddingPolicy
from .graph_tokenizer import GraphBatch, GraphTokenizer, attention_mask_from_token_mask
from .tropical_attention import TransformerBlock


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

    def forward(self, batch: GraphBatch) -> dict[str, torch.Tensor]:
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
            "token_mask": tok.token_mask,
        }
        if self.gflownet_policy is not None:
            forward_logits, backward_logits = self.gflownet_policy(pooled)
            outputs["gflownet_forward_logits"] = forward_logits
            outputs["gflownet_backward_logits"] = backward_logits
            outputs["gflownet_log_z"] = self.gflownet_policy.log_z
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
