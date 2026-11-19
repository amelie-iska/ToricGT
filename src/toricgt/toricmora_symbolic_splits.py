"""Audit split and retrieval isolation for symbolic benchmark metadata.

Pass a JSON manifest containing records and optional reviewed similarity_links,
seen_training_record_ids, and indexed_retrieval_record_ids. Record IDs identify
individual rows. Source IDs identify problems shared by trace variants. Exact
keys are case-preserving canonical question/answer pairs, with both required.
The manifest author supplies a review reference for each accepted similarity
edge. A training group's retrieval eligibility requires permission from every
member. An omitted inventory yields no exposure audit. The command reports
quarantined groups in JSON and exits with status 1 for any exposure violation.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

MATH_500 = "HuggingFaceH4/MATH-500"
OPENR1_MATH = "open-r1/OpenR1-Math-220k"
GSM8K = "openai/gsm8k"
SYMBOLIC_SOURCES = frozenset({
    MATH_500, OPENR1_MATH, GSM8K, "authored-arithmetic", "authored-graph-rewrite",
})
Holdout = Literal["validation", "test"]
Split = Literal["train", "validation", "test", "quarantine"]
IdentityKey = tuple[int, str, str]
CONFLICT = "conflicting_declared_holdouts"


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _canonical(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


@dataclass(frozen=True)
class SymbolicRecord:
    record_id: str
    source: str
    source_id: str | None = None
    canonical_question: str = ""
    canonical_answer: str = ""
    declared_holdout: Holdout | None = None
    retrieval_allowed: bool = False

    def __post_init__(self) -> None:
        if not _nonempty_string(self.record_id):
            raise ValueError("symbolic records require a nonempty record_id")
        if not isinstance(self.source, str) or self.source not in SYMBOLIC_SOURCES:
            raise ValueError("unsupported symbolic benchmark source")
        if self.source_id is not None and not _nonempty_string(self.source_id):
            raise ValueError("source_id must be a nonempty string when supplied")
        if self.source == OPENR1_MATH and self.source_id is None:
            raise ValueError("OpenR1 metadata requires a source_id (problem UUID)")
        if self.source == GSM8K and self.source_id is None:
            raise ValueError("GSM8K metadata requires a source_id")
        if not all(isinstance(key, str) for key in (self.canonical_question, self.canonical_answer)):
            raise ValueError("canonical keys must be strings")
        if self.declared_holdout not in (None, "validation", "test"):
            raise ValueError("declared_holdout must be validation or test")
        if not isinstance(self.retrieval_allowed, bool):
            raise ValueError("retrieval_allowed must be a boolean")

    def _identity_keys(self) -> tuple[IdentityKey, ...]:
        keys = []
        if self.source_id is not None:
            keys.append((0, self.source, self.source_id))
        question = _canonical(self.canonical_question)
        answer = _canonical(self.canonical_answer)
        if question and answer:
            keys.append((1, question, answer))
        return (*keys, (2, self.record_id, ""))


@dataclass(frozen=True)
class ReviewedSimilarityLink:
    left_record_id: str
    right_record_id: str
    review_ref: str

    def __post_init__(self) -> None:
        if not all(
            _nonempty_string(value)
            for value in (self.left_record_id, self.right_record_id, self.review_ref)
        ):
            raise ValueError("similarity links require two record IDs and a review reference")
        if self.left_record_id == self.right_record_id:
            raise ValueError("similarity links require distinct record IDs")


@dataclass(frozen=True)
class Assignment:
    record_id: str
    group_id: str
    split: Split
    retrieval_eligible: bool


@dataclass(frozen=True)
class ExposureAudit:
    records_checked: int
    violations: dict[str, int]


@dataclass(frozen=True)
class SplitAudit:
    assignments: tuple[Assignment, ...]
    records_by_split: dict[str, int]
    groups_by_split: dict[str, int]
    quarantine_groups_by_reason: dict[str, int]
    training_exposure: ExposureAudit | None
    retrieval_exposure: ExposureAudit | None


def _audit_exposure(
    record_ids: Iterable[str] | None,
    assignments: dict[str, Assignment],
    *,
    retrieval: bool,
) -> ExposureAudit | None:
    if record_ids is None:
        return None
    if isinstance(record_ids, (str, bytes)):
        raise ValueError("exposure inventories must contain record IDs")
    unique_ids = set()
    for record_id in record_ids:
        if not _nonempty_string(record_id):
            raise ValueError("exposure inventories require nonempty record IDs")
        unique_ids.add(record_id)

    violations: Counter[str] = Counter()
    for record_id in unique_ids:
        assignment = assignments.get(record_id)
        if assignment is None:
            violations["unknown_record"] += 1
        elif assignment.split != "train":
            violations[assignment.split] += 1
        elif retrieval and not assignment.retrieval_eligible:
            violations["retrieval_disallowed"] += 1
    return ExposureAudit(len(unique_ids), dict(sorted(violations.items())))


def audit_symbolic_splits(
    records: Iterable[SymbolicRecord],
    similarity_links: Iterable[ReviewedSimilarityLink] = (),
    *,
    seen_training_record_ids: Iterable[str] | None = None,
    indexed_retrieval_record_ids: Iterable[str] | None = None,
) -> SplitAudit:
    """Resolve components before assigning roles or checking either inventory."""
    members = tuple(records)
    by_id = {record.record_id: i for i, record in enumerate(members)}
    if len(by_id) != len(members):
        raise ValueError("duplicate record_id in symbolic manifest")

    parent = list(range(len(members)))
    size = [1] * len(members)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = root(left), root(right)
        if left != right:
            if size[left] < size[right]:
                left, right = right, left
            parent[right] = left
            size[left] += size[right]

    key_owner: dict[IdentityKey, int] = {}
    anchors = []
    for index, record in enumerate(members):
        keys = record._identity_keys()
        anchors.append(min(keys))
        for key in keys[:-1]:
            union(index, key_owner.setdefault(key, index))
    for link in similarity_links:
        if link.left_record_id not in by_id or link.right_record_id not in by_id:
            raise ValueError("similarity link refers to an unknown record")
        union(by_id[link.left_record_id], by_id[link.right_record_id])

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(members)):
        groups[root(index)].append(index)

    assignments: dict[str, Assignment] = {}
    records_by_split: Counter[str] = Counter()
    groups_by_split: Counter[str] = Counter()
    quarantine_reasons: Counter[str] = Counter()
    for group in groups.values():
        anchor = min(anchors[index] for index in group)
        anchor_json = json.dumps(anchor, ensure_ascii=True, separators=(",", ":"))
        group_id = sha256(anchor_json.encode("ascii")).hexdigest()
        holdouts: set[Holdout] = set()
        for index in group:
            record = members[index]
            if record.source == MATH_500:
                holdouts.add("test")
            if record.declared_holdout is not None:
                holdouts.add(record.declared_holdout)

        split: Split
        if len(holdouts) > 1:
            split = "quarantine"
            quarantine_reasons[CONFLICT] += 1
        elif holdouts:
            split = holdouts.pop()
        else:
            # Compare integers to keep the 80/10/10 boundaries exact.
            bucket = int(group_id[:12], 16) * 10
            limit = 16**12
            if bucket < 8 * limit:
                split = "train"
            elif bucket < 9 * limit:
                split = "validation"
            else:
                split = "test"

        retrieval_eligible = split == "train" and all(members[i].retrieval_allowed for i in group)
        for index in group:
            record_id = members[index].record_id
            assignments[record_id] = Assignment(record_id, group_id, split, retrieval_eligible)
        records_by_split[split] += len(group)
        groups_by_split[split] += 1

    return SplitAudit(
        assignments=tuple(assignments[record_id] for record_id in sorted(assignments)),
        records_by_split=dict(sorted(records_by_split.items())),
        groups_by_split=dict(sorted(groups_by_split.items())),
        quarantine_groups_by_reason=dict(quarantine_reasons),
        training_exposure=_audit_exposure(seen_training_record_ids, assignments, retrieval=False),
        retrieval_exposure=_audit_exposure(indexed_retrieval_record_ids, assignments, retrieval=True),
    )


def _read_manifest(path: str) -> dict:
    manifest = json.loads(sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8"))
    fields = (
        "records",
        "similarity_links",
        "seen_training_record_ids",
        "indexed_retrieval_record_ids",
    )
    if not isinstance(manifest, dict) or "records" not in manifest or manifest.keys() - set(fields):
        raise ValueError("expected a symbolic manifest with records and optional links and inventories")
    if any(not isinstance(manifest[name], list) for name in manifest):
        raise ValueError("manifest fields must be JSON arrays")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="symbolic metadata JSON, or - for standard input")
    args = parser.parse_args()
    try:
        manifest = _read_manifest(args.manifest)
        audit = audit_symbolic_splits(
            (SymbolicRecord(**row) for row in manifest["records"]),
            (ReviewedSimilarityLink(**row) for row in manifest.get("similarity_links", ())),
            seen_training_record_ids=manifest.get("seen_training_record_ids"),
            indexed_retrieval_record_ids=manifest.get("indexed_retrieval_record_ids"),
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(audit), indent=2))
    exposures = (audit.training_exposure, audit.retrieval_exposure)
    if any(exposure and exposure.violations for exposure in exposures):
        sys.exit(1)


if __name__ == "__main__":
    main()
