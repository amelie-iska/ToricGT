from dataclasses import replace

import pytest
import torch

from scripts.train import move_batch
from toricgt.config import ModelConfig
from toricgt.graph_tokenizer import GraphBatch
from toricgt.model import ToricTokenGT


def graph_fixture(cfg):
    generator = torch.Generator().manual_seed(17)
    return GraphBatch(
        torch.randn(1, 4, cfg.node_feature_dim, generator=generator),
        torch.randn(1, 5, cfg.edge_feature_dim, generator=generator),
        torch.tensor([[[1, 2], [2, 0], [0, 1], [1, 0], [0, 0]]]),
        torch.tensor([[True, True, True, False]]),
        torch.tensor([[True, True, True, True, False]]),
        lm_input_ids=torch.tensor([[4, 1, 7, 0]]),
        node_causal_rank=torch.tensor([[2, 0, 0, -1]]),
        node_identity_ids=torch.tensor([[7, 3, 1, -1]]),
        edge_identity_ids=torch.tensor([[8, 2, 4, 6, -1]]),
    )


@pytest.mark.parametrize("attention", ["softmax", "hybrid"])
@torch.no_grad()
def test_transported_identities_preserve_aligned_scores(attention):
    torch.manual_seed(29)
    cfg = ModelConfig(
        d_model=16, num_heads=2, num_layers=2, max_nodes=8, max_edges=10,
        attention=attention, use_causal_graph_attention=True, use_lm_head=True,
        lm_vocab_size=16, gflownet_hidden_dim=16, use_lm_token_embeddings=True,
        use_lm_position_embeddings=True, use_lm_toric_position_features=True,
        use_lm_context_hash_embeddings=True, use_lm_revealed_neighbor_hash=True,
    )
    model = ToricTokenGT(cfg).eval()
    original = graph_fixture(cfg)
    nodes = torch.tensor([3, 2, 0, 1])
    edges = torch.tensor([0, 4, 2, 3, 1])
    permuted = GraphBatch(
        original.node_features[:, nodes],
        original.edge_features[:, edges],
        nodes.argsort()[original.edge_index[:, edges]],
        original.node_mask[:, nodes],
        original.edge_mask[:, edges],
        lm_input_ids=original.lm_input_ids[:, nodes],
        node_causal_rank=original.node_causal_rank[:, nodes],
        node_identity_ids=original.node_identity_ids[:, nodes],
        edge_identity_ids=original.edge_identity_ids[:, edges],
    )
    permuted, _ = move_batch(permuted, torch.zeros(1), "cpu")
    expected = model(original)
    actual = model(permuted)
    for name in ("node", "node_embeddings", "lm_logits"):
        torch.testing.assert_close(actual[name][:, permuted.node_mask[0]], expected[name][:, nodes[permuted.node_mask[0]]])
    for name in ("edge", "edge_embeddings"):
        torch.testing.assert_close(actual[name][:, permuted.edge_mask[0]], expected[name][:, edges[permuted.edge_mask[0]]])
    torch.testing.assert_close(actual["graph"], expected["graph"])

    row_indexed = model(replace(permuted, node_identity_ids=None, edge_identity_ids=None))
    assert not torch.allclose(row_indexed["node"][:, permuted.node_mask[0]], actual["node"][:, permuted.node_mask[0]])

    changed = permuted.node_features.clone()
    changed[:, permuted.node_causal_rank[0] == 2] += 10
    altered = model(replace(permuted, node_features=changed))
    early = (permuted.node_causal_rank[0] < 2) & permuted.node_mask[0]
    torch.testing.assert_close(actual["lm_logits"][:, early], altered["lm_logits"][:, early], rtol=0, atol=0)


@torch.no_grad()
def test_explicit_identity_validates_unmasked_indices():
    cfg = ModelConfig(d_model=16, num_heads=2, num_layers=1, max_nodes=8, max_edges=10)
    model = ToricTokenGT(cfg).eval()
    original = graph_fixture(cfg)
    with pytest.raises(ValueError, match="node identity ID outside"):
        model(replace(original, node_identity_ids=torch.tensor([[8, 3, 1, -99]])))
    with pytest.raises(ValueError, match="node identity IDs must be unique"):
        model(replace(original, node_identity_ids=torch.tensor([[7, 3, 7, -99]])))
    with pytest.raises(ValueError, match="edge identity IDs must be integer"):
        model(replace(original, edge_identity_ids=original.edge_identity_ids.float()))
