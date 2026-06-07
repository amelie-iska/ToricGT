"""Graph-to-token and token-to-graph utilities for TokenGT-style models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    lm_input_ids: Optional[torch.Tensor] = None
    lm_target_ids: Optional[torch.Tensor] = None
    lm_mask: Optional[torch.Tensor] = None
    lm_target_byte_lengths: Optional[torch.Tensor] = None
    lm_target_positions: Optional[torch.Tensor] = None
    lm_input_features: Optional[torch.Tensor] = None
    node_causal_rank: Optional[torch.Tensor] = None


@dataclass
class TokenBatch:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    token_type: torch.Tensor
    node_positions: torch.Tensor
    edge_positions: torch.Tensor
    causal_rank: Optional[torch.Tensor] = None


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
        if batch.edge_mask.numel() > 0:
            active_edges = int(batch.edge_mask.long().sum(dim=1).max().detach().cpu().item())
            active_edges = max(1, min(active_edges, n_edges))
        else:
            active_edges = 0
        edge_features = batch.edge_features[:, :active_edges, :]
        edge_index = batch.edge_index[:, :active_edges, :]
        edge_mask = batch.edge_mask[:, :active_edges]
        n_edges = active_edges
        node_ids = torch.arange(n_nodes, device=device).expand(bsz, n_nodes)
        edge_ids = torch.arange(n_edges, device=device).expand(bsz, n_edges)

        node_tok = self.node_proj(batch.node_features)
        node_tok = node_tok + self.type_emb(torch.zeros_like(node_ids)) + self.node_id(node_ids)
        toric = self.toric_proj(self.torus_freq[:n_nodes].to(device)).unsqueeze(0)
        node_tok = node_tok + toric

        endpoints = edge_index.clamp(min=0, max=max(n_nodes - 1, 0))
        src = self.node_id(endpoints[..., 0])
        dst = self.node_id(endpoints[..., 1])
        edge_tok = self.edge_proj(edge_features)
        edge_tok = edge_tok + self.type_emb(torch.ones_like(edge_ids)) + self.edge_id(edge_ids)
        edge_tok = edge_tok + self.endpoint_proj(torch.cat([src, dst], dim=-1))

        tokens = torch.cat([node_tok, edge_tok], dim=1)
        token_mask = torch.cat([batch.node_mask, edge_mask], dim=1)
        token_type = torch.cat([torch.zeros_like(node_ids), torch.ones_like(edge_ids)], dim=1)
        node_positions = torch.arange(n_nodes, device=device)
        edge_positions = torch.arange(n_nodes, n_nodes + n_edges, device=device)
        causal_rank = None
        if batch.node_causal_rank is not None:
            node_rank = batch.node_causal_rank[:, :n_nodes].to(device=device, dtype=torch.long)
            node_rank = node_rank.masked_fill(~batch.node_mask[:, :n_nodes].to(device=device, dtype=torch.bool), 2**30)
            if n_edges > 0:
                src_rank = torch.gather(node_rank, 1, endpoints[..., 0])
                dst_rank = torch.gather(node_rank, 1, endpoints[..., 1])
                edge_rank = torch.maximum(src_rank, dst_rank).masked_fill(~edge_mask.to(device=device, dtype=torch.bool), 2**30)
            else:
                edge_rank = torch.empty((bsz, 0), device=device, dtype=torch.long)
            causal_rank = torch.cat([node_rank, edge_rank], dim=1)
        return TokenBatch(tokens, token_mask, token_type, node_positions, edge_positions, causal_rank)


def attention_mask_from_token_mask(
    token_mask: torch.Tensor,
    causal_rank: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build [batch, 1, query, key] boolean attention mask.

    When ``causal_rank`` is supplied, each query can only attend to keys whose
    reveal/topological rank is no larger than the query rank.  This gives a
    TokenGT analogue of autoregressive decoding for FineWeb chains, DAGs, and
    deterministic random-order graph reveals.
    """

    mask = token_mask[:, None, :, None] & token_mask[:, None, None, :]
    if causal_rank is not None:
        query_rank = causal_rank[:, None, :, None]
        key_rank = causal_rank[:, None, None, :]
        mask = mask & (key_rank <= query_rank)
    return mask
