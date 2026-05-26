import json

from toricgt.synthetic import generate_synthetic_records, reduce_rotation_word


def test_rotation_word_reduction_crossing_count():
    assert reduce_rotation_word(["V", "U"]) == (1, 1, 1)
    assert reduce_rotation_word(["U", "V"]) == (1, 1, 0)
    assert reduce_rotation_word(["v", "U"]) == (1, -1, -1)


def test_synthetic_records_have_graph_json():
    records = list(generate_synthetic_records(4, seed=0))
    assert len(records) == 4
    for record in records:
        graph = json.loads(record.graph_json)
        assert graph["nodes"]
        assert graph["edges"]
        assert record.answer
