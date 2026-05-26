import json

import torch

from toricgt.config import ModelConfig
from toricgt.graph_dataset import collate_graph_items, graph_json_to_item
from toricgt.metrics import masked_mse
from toricgt.model import ToricTokenGT


def test_graph_json_item_with_padding_trains_finite():
    payload = {
        "task_family": "unit",
        "nodes": [
            {"id": "problem", "type": "problem", "text": "show a short proof"},
            {"id": "step_000", "type": "reasoning_step", "text": "use the definition"},
            {"id": "answer", "type": "answer", "text": "done"},
        ],
        "edges": [
            {"source": "problem", "target": "step_000", "type": "depends_on"},
            {"source": "step_000", "target": "answer", "type": "supports_answer"},
        ],
    }
    cfg = ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_nodes=16,
        max_edges=32,
        attention="tropical_ring",
        ring_block_size=8,
        soft_moe_start_layer=0,
    )
    item = graph_json_to_item(json.dumps(payload), cfg)
    batch, target = collate_graph_items([item, item])
    model = ToricTokenGT(cfg)
    out = model(batch)
    loss = masked_mse(out["node"], target, batch.node_mask)
    loss.backward()
    assert torch.isfinite(loss)
