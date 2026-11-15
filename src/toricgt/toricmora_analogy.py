"""Evaluate payload-exchange analogies in abstract typed trees and forests.

These fixtures use unordered children with directed, typed edges. The verifier
recognizes the benchmark grammar and infers a two-payload exchange from A and B.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from itertools import combinations
from typing import Literal

Scope = Literal["subtree", "tree", "forest"]
Edge = tuple[str, str, str]  # parent ID, child ID, edge type
Signature = tuple[str, tuple[tuple[str, "Signature"], ...]]
SCOPES: tuple[Scope, ...] = ("subtree", "tree", "forest")
MAX_FOREST_NODES = 256


@dataclass(frozen=True)
class TypedForest:
    nodes: tuple[tuple[str, str], ...]  # node ID, node type
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        if len(self.nodes) > MAX_FOREST_NODES:
            raise ValueError(f"a forest can contain at most {MAX_FOREST_NODES} nodes")
        if not self.nodes or len(self._kinds) != len(self.nodes):
            raise ValueError("a forest needs nonempty, uniquely identified nodes")
        children_seen: set[str] = set()
        for parent, child, _ in self.edges:
            if parent not in self._kinds or child not in self._kinds:
                raise ValueError("edge endpoints must be declared nodes")
            if child in children_seen:
                raise ValueError("a forest node can have at most one parent")
            children_seen.add(child)
        if len(self._signatures) != len(self.nodes):
            raise ValueError("a forest cannot contain cycles")

    @cached_property
    def _kinds(self) -> dict[str, str]:
        return dict(self.nodes)

    @cached_property
    def _children(self) -> dict[str, list[tuple[str, str]]]:
        children: dict[str, list[tuple[str, str]]] = {node: [] for node, _ in self.nodes}
        for parent, child, role in self.edges:
            children[parent].append((role, child))
        return children

    @cached_property
    def roots(self) -> tuple[str, ...]:
        children = {child for _, child, _ in self.edges}
        return tuple(node for node, _ in self.nodes if node not in children)

    @cached_property
    def _signatures(self) -> dict[str, Signature]:
        # Build signatures bottom-up so traversal does not depend on IDs or input order.
        visit = list(self.roots)
        for node in visit:
            visit.extend(child for _, child in self._children[node])
        signatures: dict[str, Signature] = {}
        for node in reversed(visit):
            children = tuple(sorted((role, signatures[child]) for role, child in self._children[node]))
            signatures[node] = self._kinds[node], children
        return signatures


def forest_correspondence(source: TypedForest, target: TypedForest) -> dict[str, str] | None:
    """Match complete forests, allowing interchangeable equal-signature siblings."""

    def keyed_roots(forest: TypedForest) -> list[tuple[Signature, str]]:
        return sorted((forest._signatures[root], root) for root in forest.roots)

    source_roots, target_roots = keyed_roots(source), keyed_roots(target)
    if [signature for signature, _ in source_roots] != [signature for signature, _ in target_roots]:
        return None

    pending = [(left, right) for (_, left), (_, right) in zip(source_roots, target_roots)]
    mapping = {}
    for left, right in pending:
        mapping[left] = right
        left_children = sorted(
            (role, source._signatures[child], child) for role, child in source._children[left]
        )
        right_children = sorted(
            (role, target._signatures[child], child) for role, child in target._children[right]
        )
        pending.extend(
            (child_left, child_right)
            for (_, _, child_left), (_, _, child_right) in zip(left_children, right_children)
        )
    return mapping


@dataclass(frozen=True)
class AnalogyCase:
    scope: Scope
    source: TypedForest
    source_result: TypedForest
    query: TypedForest
    candidates: tuple[TypedForest, ...]
    target_index: int

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise ValueError("unsupported analogy scope")
        if not 0 <= self.target_index < len(self.candidates):
            raise ValueError("target_index must designate a candidate")


@dataclass(frozen=True)
class AnalogyWitness:
    source_to_query_branches: dict[str, str]
    query_to_candidate_nodes: dict[str, str]
    removed_query_edges: tuple[Edge, Edge]
    added_query_edges: tuple[Edge, Edge]


def _anchors(forest: TypedForest) -> dict[Signature, tuple[str, Edge]]:
    anchors = {}
    for parent, branch, role in forest.edges:
        if role != "part" or forest._kinds[parent] not in {"compute", "routine"}:
            continue
        if forest._kinds[branch] != "branch":
            continue
        markers = [child for kind, child in forest._children[branch] if kind == "anchor"]
        payloads = [child for kind, child in forest._children[branch] if kind == "payload"]
        if len(markers) != 1 or len(payloads) != 1:
            continue
        unchanged_children = tuple(
            sorted(
                (role, forest._signatures[child])
                for role, child in forest._children[branch]
                if role != "payload"
            )
        )
        signature = "branch", unchanged_children
        if signature in anchors:
            return {}  # An exchange between repeated anchor roles is ambiguous.
        anchors[signature] = branch, (branch, payloads[0], "payload")
    return anchors


def _exchange(forest: TypedForest, edges: tuple[Edge, Edge]) -> tuple[TypedForest, tuple[Edge, Edge]]:
    (left, left_child, role), (right, right_child, _) = edges
    added = (left, right_child, role), (right, left_child, role)
    remaining = tuple(edge for edge in forest.edges if edge not in edges)
    return TypedForest(forest.nodes, remaining + added), added


def verify_analogy(case: AnalogyCase) -> dict[int, AnalogyWitness]:
    """Infer a unique anchor exchange, then certify each eligible target."""
    source_anchors = _anchors(case.source)
    query_anchors = _anchors(case.query)
    matching_exchanges = []
    for left, right in combinations(source_anchors, 2):
        removed = source_anchors[left][1], source_anchors[right][1]
        try:
            exchanged, _ = _exchange(case.source, removed)
        except ValueError:
            continue  # Nested anchors can produce an invalid swap.
        if forest_correspondence(exchanged, case.source_result) is not None:
            matching_exchanges.append((left, right))
    if len(matching_exchanges) != 1:
        return {}

    left, right = matching_exchanges[0]
    if left not in query_anchors or right not in query_anchors:
        return {}
    matching_anchors = sorted(source_anchors.keys() & query_anchors.keys())
    correspondence = {source_anchors[key][0]: query_anchors[key][0] for key in matching_anchors}
    removed = query_anchors[left][1], query_anchors[right][1]
    try:
        expected, added = _exchange(case.query, removed)
    except ValueError:
        return {}
    accepted = {}
    for index, candidate in enumerate(case.candidates):
        mapping = forest_correspondence(expected, candidate)
        if mapping is not None:
            accepted[index] = AnalogyWitness(correspondence, mapping, removed, added)
    return accepted


def _example(scope: Scope, rng: random.Random, *, query: bool) -> tuple[TypedForest, list[Edge]]:
    nodes: list[tuple[str, str]] = []
    edges: list[Edge] = []

    def node(kind: str, parent: str | None = None, role: str = "") -> str:
        identifier = str(len(nodes))
        nodes.append((identifier, kind))
        if parent is not None:
            edges.append((parent, identifier, role))
        return identifier

    def branch(parent: str, ticks: int, length: int) -> Edge:
        anchor_branch = node("branch", parent, "part")
        anchor = node("anchor", anchor_branch, "anchor")
        for _ in range(ticks):
            node("tick", anchor, "mark")
        payload = node("payload", anchor_branch, "payload")
        tail = payload
        for _ in range(length):
            tail = node("step", tail, "next")
        return anchor_branch, payload, "payload"

    if scope == "subtree":
        context = node("context")
        scratch = node("scratch", context, "side")
        branch(scratch, 1, rng.randint(1, 4))
        root = node("compute", context, "body")
    elif scope == "tree":
        root = node("compute")
    else:
        background = node("background")
        node("tick", background, "mark")

    lengths = list(range(1, 5)) if not query else list(range(3, 7))
    rng.shuffle(lengths)
    payload_edges = []
    for ticks, length in enumerate(lengths, start=1):
        if scope == "forest":
            root = node("routine")
        payload_edges.append(branch(root, ticks, length))
    return TypedForest(tuple(nodes), tuple(edges)), payload_edges


def _relabel(forest: TypedForest, rng: random.Random) -> TypedForest:
    ids = rng.sample(range(1 << 32), len(forest.nodes))
    mapping = {old: f"n{new:08x}" for (old, _), new in zip(forest.nodes, ids)}
    nodes = [(mapping[old], kind) for old, kind in forest.nodes]
    edges = [(mapping[parent], mapping[child], role) for parent, child, role in forest.edges]
    rng.shuffle(nodes)
    rng.shuffle(edges)
    return TypedForest(tuple(nodes), tuple(edges))


def generate_analogy_cases(*, seed: int = 17, examples_per_scope: int = 16) -> list[AnalogyCase]:
    """Generate reproducible cases with five edge-swapped negatives per target."""
    if not 1 <= examples_per_scope <= 256:
        raise ValueError("examples_per_scope must be between 1 and 256")
    rng = random.Random(seed)
    exchanges = list(combinations(range(4), 2))
    cases = []
    for scope in SCOPES:
        for _ in range(examples_per_scope):
            source, source_edges = _example(scope, rng, query=False)
            query, query_edges = _example(scope, rng, query=True)
            left, right = gold = rng.choice(exchanges)
            source_result, _ = _exchange(source, (source_edges[left], source_edges[right]))
            order = rng.sample(exchanges, len(exchanges))
            candidates = []
            for left, right in order:
                candidate, _ = _exchange(query, (query_edges[left], query_edges[right]))
                candidates.append(_relabel(candidate, rng))
            cases.append(
                AnalogyCase(
                    scope,
                    _relabel(source, rng),
                    _relabel(source_result, rng),
                    _relabel(query, rng),
                    tuple(candidates),
                    order.index(gold),
                )
            )
    return cases


ForestEmbedder = Callable[[TypedForest], Sequence[float]]


def _typed_counts(forest: TypedForest) -> Counter[tuple[str, ...]]:
    counts = Counter((kind,) for _, kind in forest.nodes)
    counts.update(
        (forest._kinds[parent], role, forest._kinds[child]) for parent, child, role in forest.edges
    )
    return counts


def anchor_payload_sizes(forest: TypedForest) -> tuple[float, ...]:
    """Count payload nodes in the canonical order of the benchmark's anchor roles."""
    anchors = _anchors(forest)
    sizes = []
    for anchor in sorted(anchors):
        _, (_, payload, _) = anchors[anchor]
        pending = [payload]
        for node in pending:
            pending.extend(child for _, child in forest._children[node])
        sizes.append(float(len(pending)))
    return tuple(sizes)


def _offset_ranks(case: AnalogyCase, embed: ForestEmbedder | None) -> tuple[int, ...]:
    forests = (case.source, case.source_result, case.query, *case.candidates)
    if embed is None:
        counts = [_typed_counts(forest) for forest in forests]
        vocabulary = sorted(set().union(*counts))
        vectors = [tuple(float(count[key]) for key in vocabulary) for count in counts]
    else:
        vectors = [tuple(map(float, embed(forest))) for forest in forests]
        dimension = len(vectors[0])
        if not dimension or any(len(vector) != dimension for vector in vectors):
            raise ValueError("embed must return fixed-size, nonempty vectors")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("embed must return finite values")
    source, source_result, query, *candidates = vectors
    expected = [b - a + c for a, b, c in zip(source, source_result, query)]
    distances = [math.dist(expected, vector) for vector in candidates]
    best = min(distances)
    return tuple(
        index for index, value in enumerate(distances) if math.isclose(value, best, abs_tol=1e-12)
    )


@dataclass
class _Summary:
    cases: int = 0
    offset_credit: float = 0.0
    offset_ties: int = 0
    verified_targets: int = 0
    verified_negatives: int = 0
    negatives: int = 0

    def metrics(self) -> dict[str, int | float]:
        return {
            "cases": self.cases,
            "offset_top1_expected": self.offset_credit / self.cases,
            "offset_tie_rate": self.offset_ties / self.cases,
            "structural_target_recall": self.verified_targets / self.cases,
            "structural_false_accept_rate": self.verified_negatives / max(1, self.negatives),
        }


def evaluate_analogies(
    cases: Iterable[AnalogyCase], embed: ForestEmbedder | None = None
) -> dict[str, dict[str, int | float]]:
    """Compare B - A + C nearest-neighbor retrieval with structural verification."""
    summaries = {"all": _Summary()}
    for case in cases:
        offset = _offset_ranks(case, embed)
        verified = verify_analogy(case)
        target_hit = case.target_index in verified
        for summary in (summaries["all"], summaries.setdefault(case.scope, _Summary())):
            summary.cases += 1
            summary.offset_credit += (case.target_index in offset) / len(offset)
            summary.offset_ties += len(offset) > 1
            summary.verified_targets += target_hit
            summary.verified_negatives += len(verified) - target_hit
            summary.negatives += len(case.candidates) - 1
    if not summaries["all"].cases:
        raise ValueError("at least one analogy case is required")
    return {scope: summary.metrics() for scope, summary in summaries.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--examples-per-scope", type=int, default=16)
    parser.add_argument(
        "--embedding",
        choices=("typed-counts", "anchor-payload-sizes"),
        default="typed-counts",
        help="feature encoding for B - A + C",
    )
    parser.add_argument("--include-example", action="store_true", help="include one full case per scope")
    args = parser.parse_args()
    try:
        cases = generate_analogy_cases(seed=args.seed, examples_per_scope=args.examples_per_scope)
    except ValueError as error:
        parser.error(str(error))
    embed = anchor_payload_sizes if args.embedding == "anchor-payload-sizes" else None
    report = {
        "seed": args.seed,
        "examples_per_scope": args.examples_per_scope,
        "offset_embedding": args.embedding,
        "metrics": evaluate_analogies(cases, embed),
    }
    if args.include_example:
        report["examples"] = [asdict(case) for case in cases[::args.examples_per_scope]]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
