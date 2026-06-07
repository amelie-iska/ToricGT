from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from toricgt.derived_category_metrics import derived_category_objects_from_batch


def load_analyzer_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_derived_category_examples.py"
    spec = importlib.util.spec_from_file_location("analyze_derived_category_examples", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_analyze_derived_category_examples_writes_report_and_figures(tmp_path):
    module = load_analyzer_module()
    edge_index = torch.tensor([[[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]]])
    node_mask = torch.ones(1, 6, dtype=torch.bool)
    edge_mask = torch.ones(1, edge_index.shape[1], dtype=torch.bool)
    objects = derived_category_objects_from_batch(edge_index, node_mask=node_mask, edge_mask=edge_mask, max_vertices=5)
    example_dir = tmp_path / "examples"
    example_dir.mkdir()
    (example_dir / "step_00001000.json").write_text(
        json.dumps({"step": 1000, "objects": objects}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "analysis"

    result = module.analyze(example_dir, out_dir, max_files=4, max_objects=4)

    assert len(result["rows"]) == 1
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "derived_category_summary.csv").exists()
    assert (out_dir / "REPORT.md").exists()
    assert any(path.endswith("chain_complex.png") for path in result["figures"])
    assert any(path.endswith("betti_resolution.png") for path in result["figures"])
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["object_count"] == 1
    assert summary["steps"] == [1000]
