import json
from pathlib import Path

import numpy as np
import torch

from toricgt.config import ModelConfig
from toricgt.got_trajectory import graph_json_has_branch_merge, got_dag_summary_np
from toricgt.graph_dataset import (
    CuratedGraphIterableDataset,
    collate_graph_items,
    graph_json_to_item,
    interleave_dataset_paths,
    load_competition_token_memmap,
    record_to_graph_json,
    text_to_graph_json,
)
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
    assert "context_order" not in {edge["type"] for edge in payload["edges"]}
    node_ids = {node["id"]: idx for idx, node in enumerate(payload["nodes"])}
    edges = np.asarray([[node_ids[edge["source"]], node_ids[edge["target"]]] for edge in payload["edges"]], dtype=np.int64)
    summary = got_dag_summary_np(len(payload["nodes"]), edges)
    assert summary["branch_merge_edge_fraction"] > summary["linear_chain_fraction"]


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
    assert "next" not in {edge["type"] for edge in upgraded["edges"]}
    assert upgraded["suppressed_linear_chain_edges"] == 3


def write_fake_challenge_bin(path, tokens):
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    with path.open("wb") as handle:
        header.tofile(handle)
        np.asarray(tokens, dtype="<u2").tofile(handle)


def test_fineweb_bin_tokens_stream_as_branch_merge_graph(tmp_path):
    token_path = tmp_path / "fineweb_train_000000.bin"
    write_fake_challenge_bin(token_path, list(range(1, 33)))
    tokens = load_competition_token_memmap(token_path)
    assert int(tokens.shape[0]) == 32

    cfg = ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_nodes=16,
        max_edges=64,
        attention="tropical_ring",
        ring_block_size=8,
    )
    dataset = CuratedGraphIterableDataset(
        [token_path],
        cfg,
        fineweb_tokenizer_path=tmp_path / "missing.model",
        fineweb_tokens_per_graph=16,
        fineweb_stride_tokens=8,
    )
    item = next(iter(dataset))
    batch, target = collate_graph_items([item, item])
    model = ToricTokenGT(cfg)
    out = model(batch)
    loss = masked_mse(out["node"], target, batch.node_mask)

    assert batch.node_mask.any()
    assert batch.edge_mask.any()
    assert torch.isfinite(loss)


def test_interleaved_paths_emit_jsonl_and_fineweb_bin(tmp_path):
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(json.dumps({"text": "A. B. C. D.", "record_id": "jsonl"}) + "\n", encoding="utf-8")
    token_path = tmp_path / "fineweb_train_000001.bin"
    write_fake_challenge_bin(token_path, list(range(1, 65)))
    cfg = ModelConfig(max_nodes=16, max_edges=64)
    dataset = CuratedGraphIterableDataset(
        [jsonl_path, token_path],
        cfg,
        interleave_paths=True,
        fineweb_tokenizer_path=tmp_path / "missing.model",
        fineweb_tokens_per_graph=16,
        fineweb_stride_tokens=16,
    )
    iterator = iter(dataset)
    first = next(iterator)
    second = next(iterator)

    assert first.graph.node_mask.any()
    assert second.graph.node_mask.any()
    assert first.graph.edge_mask.any()
    assert second.graph.edge_mask.any()
    metadata = [first.metadata, second.metadata]
    assert any(item["dataset"] == "fineweb10B_sp1024" for item in metadata)
    assert any(item["task_family"] == "fineweb_language_modeling_graph" for item in metadata)


def test_path_type_interleaving_places_fineweb_bins_early():
    paths = [
        *[Path(f"train-{idx:05d}.parquet") for idx in range(4)],
        *[Path(f"fineweb_train_{idx:06d}.bin") for idx in range(2)],
    ]
    ordered = interleave_dataset_paths(paths)

    assert ordered[0].suffix == ".parquet"
    assert ordered[1].suffix == ".bin"
    assert ordered[2].suffix == ".parquet"
    assert ordered[3].suffix == ".bin"
