from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from toricgt.config import ModelConfig
from toricgt.graph_dataset import CuratedGraphIterableDataset
from toricgt.got_trajectory import graph_json_has_branch_merge


def load_fixture_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_analysis_fixture_data.py"
    spec = importlib.util.spec_from_file_location("create_analysis_fixture_data", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_analysis_fixture_rows_are_branch_merge_graphs() -> None:
    module = load_fixture_module()
    rows = module.make_rows()
    task_families = {row["task_family"] for row in rows}

    assert len(rows) == 6
    assert {
        "toric_tropical_graph_reasoning",
        "gudhi_persistence_simplicial_maps",
        "toric_bgg_category_o",
        "hebrew_morphology_graph",
        "noncommutative_torus_phase_memory",
        "parameter_golf_bpb_control",
    } <= task_families
    for row in rows:
        payload = json.loads(row["graph_json"])
        assert graph_json_has_branch_merge(payload)
        assert payload["trajectory_kind"] == "branch_merge_dag"


def test_analysis_fixture_writer_streams_through_curated_loader(tmp_path: Path) -> None:
    module = load_fixture_module()
    out_dir = tmp_path / "fixtures"
    rows = module.make_rows()
    table = module.pa.Table.from_pylist(rows)
    parquet_path = out_dir / "toricgt_analysis_fixture.parquet"
    out_dir.mkdir(parents=True)
    module.pq.write_table(table, parquet_path)

    cfg = ModelConfig(max_nodes=64, max_edges=128)
    dataset = CuratedGraphIterableDataset([parquet_path], cfg, parquet_batch_size=2)
    item = next(iter(dataset))

    assert int(item.graph.node_mask.sum().item()) == 6
    assert int(item.graph.edge_mask.sum().item()) == 8
    assert item.metadata["dataset"] == "toricgt_analysis_fixture"
    assert item.metadata["causal_rank_kind"] == "topological_dag"


def test_analysis_fixture_point_clouds_include_loop_spiral_and_branch() -> None:
    module = load_fixture_module()
    payload = module.make_point_clouds()
    clouds = payload["point_clouds"]
    ids = {cloud["record_id"] for cloud in clouds}

    assert ids == {"fixture_circle_loop", "fixture_spiral_phase_leaf", "fixture_branch_merge_points"}
    assert len(clouds[0]["points"]) >= 16
    assert len(clouds[1]["points"]) >= 16
    assert len(clouds[2]["points"]) == 8
