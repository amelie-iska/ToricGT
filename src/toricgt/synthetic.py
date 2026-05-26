"""Synthetic graph-record generators for ToricGT curricula."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SyntheticRecord:
    """A normalized JSONL-compatible graph reasoning record."""

    record_id: str
    dataset: str
    task_family: str
    question: str
    answer: str
    reasoning: str
    graph_json: str
    metadata_json: str


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _graph(nodes: list[dict], edges: list[dict], task_family: str) -> str:
    return _json({"task_family": task_family, "nodes": nodes, "edges": edges})


def reduce_rotation_word(word: list[str]) -> tuple[int, int, int]:
    """Reduce a word in U,V to ``omega^c U^a V^b`` for ``VU=omega UV``."""

    degrees = {"U": (1, 0), "u": (-1, 0), "V": (0, 1), "v": (0, -1)}
    a = 0
    b = 0
    c = 0
    prefix_v = 0
    for token in word:
        du, dv = degrees[token]
        c += prefix_v * du
        a += du
        b += dv
        prefix_v += dv
    return a, b, c


def rotation_word_record(index: int, rng: random.Random, length: int = 12, denominator: int = 17) -> SyntheticRecord:
    alphabet = ["U", "u", "V", "v"]
    word = [rng.choice(alphabet) for _ in range(length)]
    a, b, c = reduce_rotation_word(word)
    word_text = " ".join(word)
    nodes = [{"id": "problem", "type": "problem", "text": f"Reduce {word_text} with VU=omega UV."}]
    edges = []
    previous = "problem"
    for pos, token in enumerate(word):
        node_id = f"letter_{pos:03d}"
        nodes.append({"id": node_id, "type": "generator", "text": token})
        edges.append({"source": previous, "target": node_id, "type": "next_symbol"})
        previous = node_id
    answer = f"omega^{c} U^{a} V^{b}"
    nodes.append({"id": "answer", "type": "answer", "text": answer})
    edges.append({"source": previous, "target": "answer", "type": "reduces_to"})
    metadata = {"length": length, "denominator": denominator, "word": word, "a": a, "b": b, "c": c}
    return SyntheticRecord(
        record_id=f"synthetic_rotation_{index:08d}",
        dataset="synthetic/rotation_words",
        task_family="rotation_algebra",
        question=f"Reduce the rotation algebra word {word_text}.",
        answer=answer,
        reasoning=f"Move all U powers before V powers and count signed V-before-U crossings: {c}.",
        graph_json=_graph(nodes, edges, "rotation_algebra"),
        metadata_json=_json(metadata),
    )


def tropical_shortest_path_record(index: int, rng: random.Random, nodes_count: int = 6, edge_prob: float = 0.35) -> SyntheticRecord:
    """Generate a small DAG shortest-path trace as a min-plus graph record."""

    weighted_edges: list[tuple[int, int, int]] = []
    for src in range(nodes_count):
        for dst in range(src + 1, nodes_count):
            if rng.random() < edge_prob or dst == src + 1:
                weighted_edges.append((src, dst, rng.randint(1, 9)))
    dist = [10**9] * nodes_count
    parent: list[int | None] = [None] * nodes_count
    dist[0] = 0
    trace: list[str] = []
    for src, dst, weight in weighted_edges:
        if dist[src] + weight < dist[dst]:
            dist[dst] = dist[src] + weight
            parent[dst] = src
            trace.append(f"relax {src}->{dst} with weight {weight}: d[{dst}]={dist[dst]}")
    path = []
    cur: int | None = nodes_count - 1
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path = list(reversed(path))

    graph_nodes = [{"id": f"v{i}", "type": "state", "text": f"d[{i}]={dist[i] if dist[i] < 10**9 else 'inf'}"} for i in range(nodes_count)]
    graph_nodes.append({"id": "problem", "type": "problem", "text": "Compute a shortest path from v0 to the terminal node."})
    graph_edges = [{"source": "problem", "target": "v0", "type": "source"}]
    for src, dst, weight in weighted_edges:
        graph_edges.append({"source": f"v{src}", "target": f"v{dst}", "type": "weighted_edge", "weight": weight})
    answer = "->".join(f"v{i}" for i in path)
    graph_nodes.append({"id": "answer", "type": "answer", "text": answer})
    graph_edges.append({"source": f"v{nodes_count - 1}", "target": "answer", "type": "supports_answer"})
    metadata = {"nodes": nodes_count, "edges": weighted_edges, "distances": dist, "path": path}
    return SyntheticRecord(
        record_id=f"synthetic_tropical_sp_{index:08d}",
        dataset="synthetic/tropical_shortest_paths",
        task_family="tropical_shortest_path",
        question="Find the shortest path in the weighted acyclic graph.",
        answer=answer,
        reasoning="\n".join(trace),
        graph_json=_graph(graph_nodes, graph_edges, "tropical_shortest_path"),
        metadata_json=_json(metadata),
    )


def generate_synthetic_records(count: int, seed: int = 17) -> Iterable[SyntheticRecord]:
    rng = random.Random(seed)
    for index in range(count):
        if index % 2 == 0:
            yield rotation_word_record(index, rng, length=rng.randint(6, 18), denominator=rng.choice([13, 17, 29, 31]))
        else:
            yield tropical_shortest_path_record(index, rng, nodes_count=rng.randint(5, 9), edge_prob=0.35)


def write_jsonl(path: str | Path, count: int, seed: int = 17) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in generate_synthetic_records(count=count, seed=seed):
            handle.write(_json(asdict(record)) + "\n")
