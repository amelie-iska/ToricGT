import json

import torch

from toricgt.config import ModelConfig
from toricgt.got_trajectory import graph_json_has_branch_merge
from toricgt.graph_dataset import collate_graph_items, graph_json_to_item, record_to_graph_json, text_to_graph_json
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


def test_text_fallback_builds_branching_merging_graph():
    text = "Start with the hypothesis. Try a direct proof. Also try contradiction. Merge the cases. Finish the answer."
    graph_json = text_to_graph_json(text, dataset="unit", task_family="got", record_id="r0", max_spans=8)
    payload = json.loads(graph_json)

    assert payload["trajectory_kind"] == "branch_merge_dag"
    assert graph_json_has_branch_merge(payload)


def test_record_to_graph_json_upgrades_linear_graph():
    payload = {
        "nodes": [
            {"id": "n0", "text": "a"},
            {"id": "n1", "text": "b"},
            {"id": "n2", "text": "c"},
            {"id": "n3", "text": "d"},
        ],
        "edges": [
            {"source": "n0", "target": "n1", "type": "next"},
            {"source": "n1", "target": "n2", "type": "next"},
            {"source": "n2", "target": "n3", "type": "next"},
        ],
    }
    cfg = ModelConfig(max_nodes=16, max_edges=32)
    upgraded = json.loads(record_to_graph_json({"graph_json": json.dumps(payload)}, cfg))

    assert graph_json_has_branch_merge(upgraded)
