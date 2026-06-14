import json
import subprocess
import sys

import numpy as np
import torch

from toricgt.config import ModelConfig, TrainConfig
from toricgt.model import ToricTokenGT


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
            "1",
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
    assert summary["records"] == 1
    assert summary["oai_competition_bpb_available"] == 0.0
    assert summary["embedding_payloads_enabled"] is True
    assert summary["embedding_payload_manifest"] == "embeddings/manifest.json"
    assert (output_dir / "tokengt_geometry_records.csv").exists()
    assert (output_dir / "derived_category_objects.json").exists()
    assert (output_dir / "graph_energy_topology_triangle.png").exists()
    embedding_manifest = json.loads((output_dir / "embeddings" / "manifest.json").read_text(encoding="utf-8"))
    assert embedding_manifest["records"] == 1
    payload_npz = output_dir / embedding_manifest["payloads"][0]["relative_npz"]
    with np.load(payload_npz) as payload:
        assert "hidden" in payload.files
        assert "projected" in payload.files
        assert "complex_projected" in payload.files
        assert payload["hidden"].ndim == 2
    assert list((output_dir / "trajectories").glob("*trajectory_3d.png"))
    assert list((output_dir / "topology").glob("*topology_heatmaps.png"))
    assert "tokengt_graph" in result.stdout
