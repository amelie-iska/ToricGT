import json

from scripts.train_parameter_golf_random_order import ParquetByteChunkDataset
from toricgt.random_order_lm import advanced_byte_offset, advanced_special_token_map


def test_advanced_dataset_projection_marks_reasoning_memory_edges_and_analogies():
    special_ids = advanced_special_token_map()
    dataset = ParquetByteChunkDataset(
        parquet_glob="missing/*.parquet",
        seq_len=64,
        byte_offset=advanced_byte_offset(),
        special_token_mode="reasoning_memory",
    )
    graph_json = json.dumps(
        {
            "nodes": [
                {"id": "problem", "type": "problem", "text": "show the relation"},
                {"id": "step_000", "type": "reasoning_step", "text": "map A to B"},
                {"id": "memory_000", "type": "memory_event", "text": "retrieve the prior proof"},
                {"id": "analogy_000", "type": "analogy_step", "text": "A:B::C:D"},
            ],
            "edges": [
                {"source": "problem", "target": "step_000", "type": "depends_on"},
                {"source": "memory_000", "target": "step_000", "type": "memory_read"},
                {"source": "step_000", "target": "analogy_000", "type": "analogy_link"},
            ],
            "targets": {"answer": "done"},
        }
    )

    projection = dataset._graph_projection(graph_json)
    assert "<|got_begin|>" in projection
    assert "<|reason_step_begin|>" in projection
    assert "<|memory_begin|>" in projection
    assert "<|memory_read|>" in projection
    assert "<|memory_link|>" in projection
    assert "<|analogy_begin|>" in projection
    assert projection.strip().endswith("<|got_end|>")

    row_text = dataset._row_text(
        {
            "question": "How should the graph be traversed?",
            "reasoning": "First choose a directed edge.\nThen close the simplex.",
            "answer": "Use the memory-backed analogy.",
            "graph_json": graph_json,
        }
    )
    encoded = dataset._encode_text(row_text)
    for marker in (
        "<|got_begin|>",
        "<|reason_step_begin|>",
        "<|reason_step_end|>",
        "<|memory_read|>",
        "<|memory_link|>",
        "<|analogy_begin|>",
        "<|analogy_end|>",
    ):
        assert special_ids[marker] in encoded
