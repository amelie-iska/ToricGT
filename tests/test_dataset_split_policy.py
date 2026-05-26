import json

from toricgt.datasets import DATASET_SPECS, TEST_TIME_SCALING_DATASETS, normalize_record


def test_new_external_datasets_are_forced_to_test_split():
    specs = {spec.name: spec for spec in DATASET_SPECS}
    for dataset_name in TEST_TIME_SCALING_DATASETS:
        assert specs[dataset_name].forced_split == "test"


def test_graphwalks_normalization_is_graph_native_and_test_only():
    spec = next(spec for spec in DATASET_SPECS if spec.name == "openai/graphwalks")
    row = {
        "prompt": """
The graph has the following edges:
a -> b
b -> c
Operation:
Perform a BFS from node a with depth 2.
Final Answer: [c]
""",
        "answer_nodes": ["c"],
        "problem_type": "bfs",
        "date_added": "02-27-2026",
    }
    normalized = normalize_record(spec, row, "train", 0)
    graph = json.loads(normalized["graph_json"])
    assert normalized["split"] == "test"
    assert graph["task_family"] == "openai_graphwalks"
    assert any(edge["type"] == "directed_edge" for edge in graph["edges"])
    assert any(edge["type"] == "returns_answer_node" for edge in graph["edges"])
