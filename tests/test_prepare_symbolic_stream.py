import json
import subprocess
import sys

import pytest

from scripts.prepare_symbolic_stream import prepare_symbolic_stream
from toricgt.config import ModelConfig
from toricgt.prepared_stream import PreparedBatchStream, open_prepared_stream
from toricgt.toricmora_symbolic_splits import GSM8K, MATH_500, OPENR1_MATH


def row(record_id, source_id=None, *, source="authored-arithmetic", graph=True, **metadata):
    record = {"record_id": record_id, "source": source, "source_id": source_id} | metadata
    result = {"record": record}
    if graph:
        result["graph"] = {
            "nodes": [
                {"id": "q", "type": "question", "text": "1 + 2"},
                {"id": "a", "type": "answer", "text": "3"},
            ],
            "edges": [{"source": "q", "target": "a", "type": "derived"}],
        }
    return result


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8")


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_cli_curation_is_deterministic_and_prepared_reader_consumes_only_eligible_rows(tmp_path):
    fixture = [
        row("z-train", "arithmetic-2", retrieval_allowed=True),
        row("a-train", "arithmetic-1", retrieval_allowed=True),
        row("validation", declared_holdout="validation"),
        row("heldout", source=MATH_500, graph=False, canonical_question="1 + 5", canonical_answer="6"),
        row("trace", "authored-uuid", source=OPENR1_MATH, graph=False,
            canonical_question="1  + 5", canonical_answer="6"),
        row("near", graph=False) | {"graph": "authored holdout payload should never be written"},
        row("gsm-authored-test", "authored-test-1", source=GSM8K, graph=False, declared_holdout="test"),
        row("conflict-test", "conflicting-problem", graph=False, declared_holdout="test"),
        row("conflict-val", "conflicting-problem", graph=False, declared_holdout="validation"),
    ]
    records = tmp_path / "records.jsonl"
    write_jsonl(records, fixture)
    link = {"left_record_id": "trace", "right_record_id": "near", "review_ref": "authored-review"}
    links = write_json(tmp_path / "links.json", {"similarity_links": [link]})
    retrieval = write_json(tmp_path / "index_export.json", {"record_ids": ["a-train"]})

    out = tmp_path / "prepared"
    result = subprocess.run(
        [sys.executable, "scripts/prepare_symbolic_stream.py", "--records", str(records),
         "--similarity-links", str(links), "--retrieval-inventory", str(retrieval), "--output", str(out)],
        check=True, text=True, capture_output=True,
    )
    report = json.loads(result.stdout)
    assert report["manifest"] == str(out / "manifest.json")
    assert report["records_by_split"] == {"quarantine": 2, "test": 4, "train": 2, "validation": 1}
    assert report["quarantine_groups_by_reason"] == {"conflicting_declared_holdouts": 1}
    assert report["retrieval_records"] == 1
    assert set(p.name for p in out.iterdir()) == {
        "manifest.json", "review.json", "retrieval.json", "train.parquet", "validation.parquet",
    }
    with open_prepared_stream(report["manifest"], report["manifest_sha256"]) as (shards, admission):
        assert admission["train_records"] == 2
        assert [record["record_id"] for record in shards["train"].records(0, 2)] == ["a-train", "z-train"]
        assert shards["validation"].inventory() == {"validation"}
        batch, target = PreparedBatchStream(shards["train"], ModelConfig(max_nodes=3, max_edges=3), 2).next_batch()
        assert batch.node_mask.sum().item() == 4
        assert target.shape[:2] == (2, 3)

    write_jsonl(records, list(reversed(fixture))[:4])
    rest = tmp_path / "rest.jsonl"
    write_jsonl(rest, list(reversed(fixture))[4:])
    write_json(links, {"similarity_links": [link | {"left_record_id": "near", "right_record_id": "trace"}]})
    again = prepare_symbolic_stream(
        [rest, records], tmp_path / "again", similarity_links=links, retrieval_inventory=retrieval,
    )
    assert again["manifest_sha256"] == report["manifest_sha256"]
    assert all(b"authored holdout payload" not in path.read_bytes() for path in out.iterdir())


@pytest.mark.parametrize(
    ("bad_rows", "message"),
    [
        ([row("unreviewed", source="other")], "unsupported symbolic benchmark source"),
        ([row("missing-id", source=GSM8K)], "GSM8K metadata requires a source_id"),
        ([row("dupe"), row("dupe")], "duplicate record_id"),
        ([row("missing-graph", "arithmetic-1", graph=False)], "requires a graph"),
        ([row("bad-edge", "arithmetic-1") | {"graph": {
            "nodes": [{"id": "x", "type": "node", "text": "x"}],
            "edges": [{"source": "x", "target": "absent", "type": "directed"}],
        }}], "edge endpoints"),
    ],
)
def test_invalid_input_leaves_no_output(tmp_path, bad_rows, message):
    records = tmp_path / "invalid.jsonl"
    output = tmp_path / "prepared"
    write_jsonl(records, bad_rows)
    with pytest.raises(ValueError, match=message):
        prepare_symbolic_stream([records], output)
    assert list(tmp_path.iterdir()) == [records]


def test_retrieval_requires_group_permission_and_existing_output_is_preserved(tmp_path):
    records = tmp_path / "records.jsonl"
    write_jsonl(records, [
        row("train-permitted", "arithmetic-1", retrieval_allowed=True),
        row("train-withheld", "arithmetic-1"),
        row("authored-test", "test-continued", source=GSM8K, graph=False, declared_holdout="test"),
    ])
    retrieval = write_json(tmp_path / "retrieval.json", {"record_ids": ["train-permitted", "authored-test"]})
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match=r'"retrieval_disallowed":1,"test":1'):
        prepare_symbolic_stream([records], output, retrieval_inventory=retrieval)
    assert not output.exists()

    result = prepare_symbolic_stream([records], output)
    assert result["train_records"] == 2
    assert result["validation_records"] == result["retrieval_records"] == 0
    assert json.loads((output / "manifest.json").read_text())["validation"] is None
    with pytest.raises(FileExistsError, match="fresh prepared output"):
        prepare_symbolic_stream([records], output)
    assert json.loads((output / "retrieval.json").read_text()) == {"record_ids": []}


def test_native_recipe_capacity_rejects_a_graph_before_publication(tmp_path):
    records = tmp_path / "records.jsonl"
    graph = {
        "nodes": [
            {"id": str(index), "type": "node", "text": "x"}
            for index in range(ModelConfig().max_nodes + 1)
        ],
        "edges": [],
    }
    write_jsonl(records, [row("too-many-nodes", "arithmetic-1") | {"graph": graph}])
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="node capacity"):
        prepare_symbolic_stream([records], output)
    assert not output.exists()
