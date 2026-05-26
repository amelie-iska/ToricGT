"""Graph-to-token and token-to-graph utilities for TokenGT-style models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GraphBatch:
    """Dense padded graph batch.

    ``edge_index`` stores endpoint ids for edge tokens and has shape
    ``[batch, max_edges, 2]``. Padding masks are true for valid items.
    """

    node_features: torch.Tensor
    edge_features: torch.Tensor
    edge_index: torch.Tensor
    node_mask: torch.Tensor
    edge_mask: torch.Tensor


@dataclass
class TokenBatch:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    token_type: torch.Tensor
    node_positions: torch.Tensor
    edge_positions: torch.Tensor


class GraphTokenizer(nn.Module):
    """Tokenize nodes and edges with endpoint-aware structural embeddings."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        d_model: int,
        max_nodes: int,
        max_edges: int,
        torus_rank: int = 2,
        theta: float = 0.6180339887498948,
    ) -> None:
        super().__init__()
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.d_model = d_model
        self.node_proj = nn.Linear(node_feature_dim, d_model)
        self.edge_proj = nn.Linear(edge_feature_dim, d_model)
        self.type_emb = nn.Embedding(2, d_model)
        self.node_id = nn.Embedding(max_nodes, d_model)
        self.edge_id = nn.Embedding(max_edges, d_model)
        self.endpoint_proj = nn.Linear(2 * d_model, d_model)
        self.toric_proj = nn.Linear(2 * torus_rank, d_model)
        self.register_buffer("torus_freq", self._make_torus_freq(max_nodes, torus_rank, theta), persistent=False)

    @staticmethod
    def _make_torus_freq(max_nodes: int, rank: int, theta: float) -> torch.Tensor:
        idx = torch.arange(max_nodes, dtype=torch.float32).unsqueeze(-1)
        freqs = torch.arange(1, rank + 1, dtype=torch.float32).unsqueeze(0)
        angles = 2.0 * torch.pi * theta * idx * freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, batch: GraphBatch) -> TokenBatch:
        bsz, n_nodes, _ = batch.node_features.shape
        _, n_edges, _ = batch.edge_features.shape
        if n_nodes > self.max_nodes or n_edges > self.max_edges:
            raise ValueError("batch exceeds configured max_nodes/max_edges")

        device = batch.node_features.device
        node_ids = torch.arange(n_nodes, device=device).expand(bsz, n_nodes)
        edge_ids = torch.arange(n_edges, device=device).expand(bsz, n_edges)

        node_tok = self.node_proj(batch.node_features)
        node_tok = node_tok + self.type_emb(torch.zeros_like(node_ids)) + self.node_id(node_ids)
        toric = self.toric_proj(self.torus_freq[:n_nodes].to(device)).unsqueeze(0)
        node_tok = node_tok + toric

        endpoints = batch.edge_index.clamp(min=0, max=max(n_nodes - 1, 0))
        src = self.node_id(endpoints[..., 0])
        dst = self.node_id(endpoints[..., 1])
        edge_tok = self.edge_proj(batch.edge_features)
        edge_tok = edge_tok + self.type_emb(torch.ones_like(edge_ids)) + self.edge_id(edge_ids)
        edge_tok = edge_tok + self.endpoint_proj(torch.cat([src, dst], dim=-1))

        tokens = torch.cat([node_tok, edge_tok], dim=1)
        token_mask = torch.cat([batch.node_mask, batch.edge_mask], dim=1)
        token_type = torch.cat([torch.zeros_like(node_ids), torch.ones_like(edge_ids)], dim=1)
        node_positions = torch.arange(n_nodes, device=device)
        edge_positions = torch.arange(n_nodes, n_nodes + n_edges, device=device)
        return TokenBatch(tokens, token_mask, token_type, node_positions, edge_positions)


def attention_mask_from_token_mask(token_mask: torch.Tensor) -> torch.Tensor:
    """Build [batch, 1, query, key] boolean attention mask."""

    return token_mask[:, None, :, None] & token_mask[:, None, None, :]

