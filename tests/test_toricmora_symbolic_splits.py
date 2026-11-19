import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from toricgt.toricmora_symbolic_splits import (
    CONFLICT,
    MATH_500,
    OPENR1_MATH,
    ReviewedSimilarityLink,
    SymbolicRecord,
    audit_symbolic_splits,
)


def test_transitive_identity_preserves_benchmark_holdout_and_is_order_independent():
    records = (
        SymbolicRecord("benchmark", MATH_500, canonical_question="Compute  3 * 7.", canonical_answer="21"),
        SymbolicRecord("trace-a", OPENR1_MATH, "authored-problem-7", "Compute 3 * 7.", "21"),
        SymbolicRecord("trace-b", OPENR1_MATH, "authored-problem-7", "Multiply 7 by 3.", "21"),
        SymbolicRecord("rewrite", "authored-arithmetic", canonical_question="Add three sevens.", canonical_answer="21"),
        SymbolicRecord("tree", "authored-graph-rewrite", declared_holdout="test"),
    )
    links = (
        ReviewedSimilarityLink("trace-b", "rewrite", "authored-review-1"),
        ReviewedSimilarityLink("rewrite", "tree", "authored-review-2"),
    )
    report = audit_symbolic_splits(records, links)
    assert report == audit_symbolic_splits(
        reversed(records),
        (replace(link, left_record_id=link.right_record_id, right_record_id=link.left_record_id) for link in reversed(links)),
    )
    assert len({row.group_id for row in report.assignments}) == 1
    assert report.groups_by_split == {"test": 1}
    assert report.records_by_split == {"test": 5}
    assert {row.split for row in report.assignments} == {"test"}
    assert not any(row.retrieval_eligible for row in report.assignments)
    assert report.training_exposure is None
    assert report.retrieval_exposure is None

    trace_only = audit_symbolic_splits((records[1],))
    both_traces = audit_symbolic_splits(records[1:3])
    assert {row.group_id for row in both_traces.assignments} == {trace_only.assignments[0].group_id}


def test_unmatched_identity_stays_separate_and_declared_holdout_conflict_quarantines():
    records = (
        SymbolicRecord("arithmetic-0", "authored-arithmetic", "7", "Evaluate X.", "2", "validation"),
        SymbolicRecord("benchmark", MATH_500, canonical_question="Evaluate X.", canonical_answer="2"),
        SymbolicRecord("different-answer", "authored-arithmetic", canonical_question="Evaluate X.", canonical_answer="3"),
        SymbolicRecord("no-answer", "authored-arithmetic", canonical_question="Evaluate X."),
        SymbolicRecord("namespace", "authored-graph-rewrite", "7"),
        SymbolicRecord("case-sensitive", "authored-arithmetic", canonical_question="Evaluate x.", canonical_answer="2"),
    )
    report = audit_symbolic_splits(records)
    assignments = {row.record_id: row for row in report.assignments}
    assert assignments["arithmetic-0"].group_id == assignments["benchmark"].group_id
    assert assignments["arithmetic-0"].split == assignments["benchmark"].split == "quarantine"
    assert report.quarantine_groups_by_reason == {CONFLICT: 1}
    assert report.records_by_split["quarantine"] == 2
    assert len({row.group_id for row in report.assignments}) == 5


def test_inventories_are_independent_and_retrieval_permission_is_group_wide():
    records = (
        SymbolicRecord("a", "authored-arithmetic", "arithmetic-1", retrieval_allowed=True),
        SymbolicRecord("b", "authored-arithmetic", "arithmetic-1"),
        SymbolicRecord("c", "authored-arithmetic", "arithmetic-2", retrieval_allowed=True),
        SymbolicRecord("sealed", MATH_500),
    )
    report = audit_symbolic_splits(
        records,
        seen_training_record_ids=("a", "b", "c", "sealed", "sealed"),
        indexed_retrieval_record_ids=("a", "c", "unknown"),
    )
    assignments = {row.record_id: row for row in report.assignments}
    assert assignments["a"].split == assignments["b"].split == assignments["c"].split == "train"
    assert not assignments["a"].retrieval_eligible
    assert not assignments["b"].retrieval_eligible
    assert assignments["c"].retrieval_eligible
    assert report.training_exposure.records_checked == 4
    assert report.training_exposure.violations == {"test": 1}
    assert report.retrieval_exposure.records_checked == 3
    assert report.retrieval_exposure.violations == {"retrieval_disallowed": 1, "unknown_record": 1}


def test_incomplete_or_ambiguous_identity_metadata_fails_closed():
    record = SymbolicRecord("a", "authored-arithmetic")
    with pytest.raises(ValueError, match="duplicate record_id"):
        audit_symbolic_splits((record, record))
    with pytest.raises(ValueError, match="unknown record"):
        audit_symbolic_splits((record,), (ReviewedSimilarityLink("a", "missing", "authored-review-1"),))
    with pytest.raises(ValueError, match="review reference"):
        ReviewedSimilarityLink("a", "b", "")
    with pytest.raises(ValueError, match="requires a source_id"):
        SymbolicRecord("trace", OPENR1_MATH)


def test_cli_reports_leakage_and_returns_a_failure_status():
    entry_point = Path(__file__).resolve().parents[1] / "src" / "toricgt" / "toricmora_symbolic_splits.py"
    manifest = {
        "records": [{"record_id": "authored-holdout", "source": MATH_500}],
        "seen_training_record_ids": [],
        "indexed_retrieval_record_ids": ["authored-holdout"],
    }
    result = subprocess.run(
        [sys.executable, str(entry_point), "-"],
        check=False,
        input=json.dumps(manifest),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["training_exposure"] == {"records_checked": 0, "violations": {}}
    assert report["retrieval_exposure"] == {"records_checked": 1, "violations": {"test": 1}}
