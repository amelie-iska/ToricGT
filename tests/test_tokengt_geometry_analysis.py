import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from toricgt.config import ModelConfig, TrainConfig
from toricgt.model import ToricTokenGT


def _load_infer_module():
    path = Path("scripts/infer_tokengt_with_geometry.py").resolve()
    spec = importlib.util.spec_from_file_location("infer_tokengt_with_geometry_for_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tokengt_geometry_suite_smoke(tmp_path):
    cfg = ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=1,
        max_nodes=16,
        max_edges=64,
        attention="tropical_ring",
        ring_block_size=8,
        use_trajectory_memory_head=True,
    )
    checkpoint = tmp_path / "toricgt_step_00000001.pt"
    model = ToricTokenGT(cfg)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "train_config": TrainConfig(device="cpu", precision="fp32").__dict__,
            "step": 1,
        },
        checkpoint,
    )
    data = tmp_path / "records.jsonl"
    data.write_text(
        json.dumps(
            {
                "text": "Start with a premise. Split into two approaches. Compare both alternatives. Merge them into a conclusion.",
                "dataset": "unit",
                "task_family": "got",
                "record_id": "r0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with data.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "text": "Begin with a hypothesis. Explore two branches. Reconcile them with a shared lemma. Finish with the answer.",
                    "dataset": "unit",
                    "task_family": "got",
                    "record_id": "r1",
                }
            )
            + "\n"
        )
    output_dir = tmp_path / "analysis"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_tokengt_reasoning_geometry_suite.py",
            "--checkpoint",
            str(checkpoint),
            "--data-glob",
            str(data),
            "--output-dir",
            str(output_dir),
            "--records",
            "2",
            "--batch-size",
            "1",
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--no-rich-legacy-geometry",
            "--emit-embedding-payloads",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    summary = json.loads((output_dir / "reasoning_geometry_summary.json").read_text(encoding="utf-8"))
    assert summary["checkpoint_family"] == "tokengt_graph"
    assert summary["records"] == 2
    assert summary["oai_competition_bpb_available"] == 0.0
    assert summary["embedding_payloads_enabled"] is True
    assert summary["embedding_payload_manifest"] == "embeddings/manifest.json"
    assert (output_dir / "tokengt_geometry_records.csv").exists()
    assert (output_dir / "derived_category_objects.json").exists()
    assert (output_dir / "graph_energy_topology_triangle.png").exists()
    embedding_manifest = json.loads((output_dir / "embeddings" / "manifest.json").read_text(encoding="utf-8"))
    assert embedding_manifest["records"] == 2
    payload_npz = output_dir / embedding_manifest["payloads"][0]["relative_npz"]
    with np.load(payload_npz) as payload:
        assert "hidden" in payload.files
        assert "projected" in payload.files
        assert "complex_projected" in payload.files
        assert payload["hidden"].ndim == 2
    assert list((output_dir / "trajectories").glob("*trajectory_3d.png"))
    assert list((output_dir / "topology").glob("*topology_heatmaps.png"))
    simplex_tree_html = next((output_dir / "trajectories").glob("*_reasoning_step_simplex_tree_3d.html"))
    simplex_text = simplex_tree_html.read_text(encoding="utf-8")
    assert "reasoning_step_simplex_tree_3d" in simplex_text
    assert "simplex_tree_payload" in simplex_text
    assert "reasoning_level_slider" in simplex_text
    assert "radius_slider" in simplex_text
    assert "radius" in simplex_text
    assert "NLL unavailable" in simplex_text or "node NLL" in simplex_text
    analogical_html = next((output_dir / "analogical").glob("*_analogical_simplex_maps_3d.html"))
    analogical_text = analogical_html.read_text(encoding="utf-8")
    assert "analogical_simplex_maps_3d" in analogical_text
    assert "vectorized_persistence_comparisons" in analogical_text
    assert "full_reasoning_trajectory_simplex_tree_map" in analogical_text or "No analogy emitted" in analogical_text
    assert "reasoning_level_slider" in analogical_text
    assert "radius_slider" in analogical_text
    slepian_html = next((output_dir / "trajectories").glob("*_slepian_torus_surface.html"))
    slepian_text = slepian_html.read_text(encoding="utf-8")
    assert "slepian_torus_surface" in slepian_text
    assert "slepian_torus_payload" in slepian_text
    plot_manifest = json.loads((output_dir / "plot_family_manifest.json").read_text(encoding="utf-8"))
    family_status = {item["family"]: item["status"] for item in plot_manifest["families"]}
    assert family_status["reasoning_step_simplex_tree_3d"] == "ok"
    assert family_status["analogical_simplex_maps_3d"] == "ok"
    assert family_status["slepian_torus_surface"] == "ok"
    assert "tokengt_graph" in result.stdout


def test_inference_branching_screenshots_assert_report_by_default() -> None:
    infer = _load_infer_module()
    args = infer.argparse.Namespace(assert_branching_reasoning_report=True)
    cmd = infer.branching_screenshot_command(args, Path("branching"), Path("screens"))
    assert "--assert-branching-report" in cmd
    assert "--interaction-audit" in cmd


def test_inference_branching_screenshots_allows_explicit_debug_opt_out() -> None:
    infer = _load_infer_module()
    args = infer.argparse.Namespace(assert_branching_reasoning_report=False)
    cmd = infer.branching_screenshot_command(args, Path("branching"), Path("screens"))
    assert "--assert-branching-report" not in cmd
    assert "--interaction-audit" in cmd
