#!/usr/bin/env python3
"""Create deterministic local data for ToricGT analysis and inference smoke runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def branch_merge_graph(record_id: str, title: str, steps: list[str], *, task_family: str) -> str:
    nodes: list[dict[str, Any]] = [
        {"id": "problem", "type": "problem", "text": title},
        {"id": "branch_a", "type": "reasoning_branch", "text": steps[0]},
        {"id": "branch_b", "type": "reasoning_branch", "text": steps[1]},
        {"id": "merge_0", "type": "merge_step", "text": steps[2]},
        {"id": "certificate", "type": "certificate", "text": steps[3]},
        {"id": "answer", "type": "answer", "text": steps[4]},
    ]
    edges = [
        {"source": "problem", "target": "branch_a", "type": "branch_left"},
        {"source": "problem", "target": "branch_b", "type": "branch_right"},
        {"source": "branch_a", "target": "merge_0", "type": "merge_candidate"},
        {"source": "branch_b", "target": "merge_0", "type": "merge_candidate"},
        {"source": "merge_0", "target": "certificate", "type": "verifies"},
        {"source": "certificate", "target": "answer", "type": "supports_answer"},
        {"source": "branch_a", "target": "certificate", "type": "auxiliary_check"},
        {"source": "branch_b", "target": "certificate", "type": "auxiliary_check"},
    ]
    return json.dumps(
        {
            "record_id": record_id,
            "dataset": "toricgt_analysis_fixture",
            "task_family": task_family,
            "trajectory_kind": "branch_merge_dag",
            "nodes": nodes,
            "edges": edges,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def expand_solution(seed: str, clauses: list[str], repeats: int = 10) -> str:
    paragraphs = []
    for idx in range(repeats):
        body = " ".join(clauses)
        paragraphs.append(
            f"Step {idx + 1}: {seed}. {body} "
            f"The audit should expose the active certificate, the map residual, "
            f"the chain relation, and the metric that would be logged during training."
        )
    return "\n".join(paragraphs)


def make_rows() -> list[dict[str, Any]]:
    specs = [
        {
            "record_id": "fixture_toric_tropical_attention",
            "task_family": "toric_tropical_graph_reasoning",
            "question": "Explain how tropical active faces are embedded into a toric normal fan for a graph-token attention head.",
            "answer": "Final answer: use the Newton polytope of affine attention candidates; the tropical argmax selects a normal-fan cone inside the toric chart.",
            "clauses": [
                "The affine candidates define exponent vectors and a lifted Newton polytope.",
                "The max-plus attention value selects an exposed face.",
                "The selected face determines a cone of the normal fan.",
                "The toric audit records chamber id, margin, bend, and binomial consistency.",
            ],
        },
        {
            "record_id": "fixture_gudhi_two_parameter_persistence",
            "task_family": "gudhi_persistence_simplicial_maps",
            "question": "Build the two-parameter persistence module for reasoning level and Rips radius.",
            "answer": "Final answer: vertices, edges, and triangles generate a bigraded chain complex over F2[x_level,y_radius], with commuting inclusion squares.",
            "clauses": [
                "Reasoning level reveals more vertices in the trajectory prefix.",
                "Radius increases the Vietoris-Rips simplex tree.",
                "The x and y structure maps are inclusions.",
                "Macaulay2 verifies homogeneous boundary maps and computes HH_i, res, and betti.",
            ],
        },
        {
            "record_id": "fixture_bgg_category_o_resolution",
            "task_family": "toric_bgg_category_o",
            "question": "Review a finite Toric BGG certificate with standard-filtration and differential checks.",
            "answer": "Final answer: the certificate is acceptable when standard leakage is low, D_{k-1}D_k vanishes, and the Betti profile matches the expected Koszul pattern.",
            "clauses": [
                "The finite poset labels standard objects.",
                "The sparse boundary matrices encode the resolution.",
                "The chain-map residual checks functor-like transfer.",
                "Gale-dual labels are compared only after the local differential audit passes.",
            ],
        },
        {
            "record_id": "fixture_hebrew_root_template_graph",
            "task_family": "hebrew_morphology_graph",
            "question": "Disambiguate a Hebrew root-template graph represented with radical, binyan, feature, and surface-form nodes.",
            "answer": "Final answer: preserve multiple analyzer hypotheses and score the root, binyan, feature bundle, and radical-slot alignment jointly.",
            "clauses": [
                "The root node connects to ordered radical nodes.",
                "Template slots attach to surface spans.",
                "Binyan and feature nodes constrain compatible forms.",
                "Paradigm edges test whether the graph learned a root family rather than a string lookup.",
            ],
        },
        {
            "record_id": "fixture_noncommutative_torus_phase",
            "task_family": "noncommutative_torus_phase_memory",
            "question": "Reduce a finite clock-shift word and audit the projected irrational torus phase leaf.",
            "answer": "Final answer: the normal form stores U and V exponents plus the cocycle phase, and the projected leaf residual checks coherent phase transport.",
            "clauses": [
                "The relation VU equals omega times UV accumulates a crossing exponent.",
                "Finite Weyl pairs verify the rational approximants.",
                "The phase path is projected to a commutative torus for visualization.",
                "Large leaf residual means the sinusoidal channel is being used incoherently.",
            ],
        },
        {
            "record_id": "fixture_bpb_control_graph",
            "task_family": "parameter_golf_bpb_control",
            "question": "State the score-before-update constraint for BPB and explain how structural probes should be gated.",
            "answer": "Final answer: every prediction must depend only on scored prefix bytes; structural probes remain training-only unless exported BPB improves.",
            "clauses": [
                "The online state is updated only after the scored byte loss is recorded.",
                "Validation streams cannot be searched by future bytes or seed sweeps.",
                "Auxiliary toric, BGG, topology, and memory losses are monitored with ablations.",
                "The artifact gate is serialized bytes and deterministic BPB, not diagnostic attractiveness.",
            ],
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        solution = expand_solution(spec["question"], spec["clauses"], repeats=9)
        graph_json = branch_merge_graph(
            spec["record_id"],
            spec["question"],
            [*spec["clauses"], spec["answer"]],
            task_family=spec["task_family"],
        )
        text = "\n\n".join([spec["question"], solution, spec["answer"]])
        content_hash = hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()
        rows.append(
            {
                "record_id": spec["record_id"],
                "dataset": "toricgt_analysis_fixture",
                "task_family": spec["task_family"],
                "language": "en",
                "question": spec["question"],
                "answer": spec["answer"],
                "solution": solution,
                "reasoning": solution,
                "metadata_json": json.dumps({"fixture": True, "analysis_use": "geometry_inference"}),
                "text": text,
                "graph_json": graph_json,
                "estimated_tokens": max(256, len(text.split())),
                "quality_flags_json": json.dumps({"synthetic_fixture": True, "leakage_safe": True}),
                "content_hash": content_hash,
                "group_hash": content_hash,
                "split": "analysis_fixture",
            }
        )
    return rows


def make_point_clouds() -> dict[str, Any]:
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    spiral_t = np.linspace(0.0, 4.0 * np.pi, 36)
    spiral = np.stack([(0.08 * spiral_t) * np.cos(spiral_t), (0.08 * spiral_t) * np.sin(spiral_t)], axis=1)
    branch = np.asarray(
        [[0.0, 0.0], [0.4, 0.3], [0.8, 0.6], [1.2, 0.0], [0.8, -0.5], [0.4, -0.2], [1.6, 0.1], [2.0, 0.0]],
        dtype=float,
    )
    return {
        "point_clouds": [
            {"record_id": "fixture_circle_loop", "points": circle.round(6).tolist()},
            {"record_id": "fixture_spiral_phase_leaf", "points": spiral.round(6).tolist()},
            {"record_id": "fixture_branch_merge_points", "points": branch.round(6).tolist()},
        ]
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/analysis_fixtures")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = make_rows()
    table = pa.Table.from_pylist(rows)
    parquet_path = out_dir / "toricgt_analysis_fixture.parquet"
    jsonl_path = out_dir / "toricgt_analysis_fixture.jsonl"
    points_path = out_dir / "toricgt_analysis_points.json"
    pq.write_table(table, parquet_path)
    write_jsonl(jsonl_path, rows)
    points_path.write_text(json.dumps(make_point_clouds(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "toricgt.analysis_fixture_manifest.v1",
        "records": len(rows),
        "parquet": str(parquet_path),
        "jsonl": str(jsonl_path),
        "points_json": str(points_path),
        "task_families": sorted({str(row["task_family"]) for row in rows}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
