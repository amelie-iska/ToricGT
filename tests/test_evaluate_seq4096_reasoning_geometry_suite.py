from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_seq4096_reasoning_geometry_suite.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_seq4096_reasoning_geometry_suite", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fake_seq4096_state(vocab_size: int = 32, model_dim: int = 16, layers: int = 4) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "tok_emb.weight": torch.randn(vocab_size, model_dim) * 0.02,
        "prev_token_bias.weight": torch.randn(vocab_size, vocab_size) * 0.01,
        "hash_ngram_bias.weight": torch.randn(48, vocab_size) * 0.01,
        "skip_weights": torch.randn(1, model_dim) * 0.01,
    }
    for idx in range(layers):
        state[f"blocks.{idx}.attn.q_gain"] = torch.randn(4) * 0.01
        state[f"blocks.{idx}.attn.c_q.weight"] = torch.randn(model_dim, model_dim) * 0.02
        state[f"blocks.{idx}.attn.c_k.weight"] = torch.randn(model_dim // 2, model_dim) * 0.02
        state[f"blocks.{idx}.attn.c_v.weight"] = torch.randn(model_dim // 2, model_dim) * 0.02
        state[f"blocks.{idx}.attn.proj.weight"] = torch.randn(model_dim, model_dim) * 0.02
        state[f"blocks.{idx}.mlp.fc.weight"] = torch.randn(3 * model_dim, model_dim) * 0.02
        state[f"blocks.{idx}.mlp.proj.weight"] = torch.randn(model_dim, 3 * model_dim) * 0.02
        state[f"blocks.{idx}.attn_scale"] = torch.randn(model_dim) * 0.01
        state[f"blocks.{idx}.mlp_scale"] = torch.randn(model_dim) * 0.01
        state[f"blocks.{idx}.resid_mix"] = torch.randn(2, model_dim) * 0.01
    return state


def write_sample_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "step:3500/4500 train_loss:2.0800 train_time:1ms step_avg:1ms train_bpb:1.2200",
                "step:3500/4500 val_loss:2.0700 val_bpb:1.2200 train_time:1ms step_avg:1ms",
                "step:3750/4500 train_loss:2.0300 train_time:1ms step_avg:1ms train_bpb:1.1850",
                "step:3750/4500 val_loss:2.0483 val_bpb:1.2131 train_time:1ms step_avg:1ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_seq4096_reasoning_geometry_suite_writes_old_suite_shape(tmp_path: Path) -> None:
    module = load_module()
    checkpoint = tmp_path / "run_step_003750.pt"
    torch.save({"step": 3750, "model": fake_seq4096_state()}, checkpoint)
    log_path = tmp_path / "train.log"
    write_sample_log(log_path)
    output_dir = tmp_path / "analysis"

    summary = module.run_analysis(
        Namespace(
            checkpoint=str(checkpoint),
            log=str(log_path),
            output_dir=str(output_dir),
            run_path="entity/project/run-id",
            target_bpb=1.2,
            max_points=16,
            records=4,
            seed=10017,
            topology_max_points=12,
            topology_window_size=12,
            topology_levels=3,
        )
    )

    assert summary["analysis_source"] == "seq4096_checkpoint_embedding_proxy"
    assert summary["current_run_only"] == 1.0
    assert summary["families_available"]["topology"] == 1.0
    assert summary["families_available"]["graphcg"] == 1.0
    assert (output_dir / "geometry" / "reasoning_geometry_summary.json").exists()
    assert (output_dir / "simplex" / "reasoning_simplex_summary.json").exists()
    assert (output_dir / "geometry" / "triangles" / "reasoning_k_bpb.png").exists()
    assert (output_dir / "geometry" / "tetrahedra" / "toric_gfn_bpb.png").exists()
    assert (output_dir / "simplex" / "reasoning_k_bpb_triangle.png").exists()
    assert (output_dir / "artifact_inventory.json").exists()
    assert (output_dir / "analysis_review_prompt.md").exists()

    inventory = json.loads((output_dir / "artifact_inventory.json").read_text(encoding="utf-8"))
    assert inventory["review_required"] == 1.0
    assert any("directed_filtration" in path for path in inventory["image_files"])
    assert any("toric_slepian_audit" in path for path in inventory["image_files"])
    assert any("graphcg_basis_disentanglement" in path for path in inventory["image_files"])
    assert any("analogical_transport_map" in path for path in inventory["image_files"])


def test_seq4096_energy_landscape_uses_static_3d_axis(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    projections: list[str | None] = []
    original_add_subplot = Figure.add_subplot

    def capture_add_subplot(self, *args, **kwargs):
        projections.append(kwargs.get("projection"))
        return original_add_subplot(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "add_subplot", capture_add_subplot)
    proj3 = torch.randn(18, 3).numpy()
    record = {
        "record_id": "synthetic",
        "_arrays": {
            "proj3": proj3,
            "energy": torch.linspace(0.5, 1.5, 18).numpy(),
        },
    }

    output = tmp_path / "energy_3d.png"
    module.plot_energy_landscape(record, output)

    assert output.exists()
    assert "3d" in projections


def test_seq4096_triangle_uses_dark_filled_reasoning_heatmap(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    contour_calls = 0
    saved_facecolors: list[tuple[float, float, float, float]] = []
    original_tricontourf = Axes.tricontourf
    original_savefig = Figure.savefig

    def capture_tricontourf(self, *args, **kwargs):
        nonlocal contour_calls
        contour_calls += 1
        return original_tricontourf(self, *args, **kwargs)

    def capture_savefig(self, *args, **kwargs):
        saved_facecolors.append(tuple(self.get_facecolor()))
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "tricontourf", capture_tricontourf)
    monkeypatch.setattr(Figure, "savefig", capture_savefig)
    records = [
        {
            "record_id": f"R{idx}",
            "score/trajectory_depth": 0.2 + idx * 0.1,
            "score/relative_k": 0.8 - idx * 0.08,
            "score/bpb_quality": 0.3 + idx * 0.09,
        }
        for idx in range(6)
    ]

    output = tmp_path / "triangle.png"
    module.plot_triangle(records, module.TRIANGLE_SPECS["reasoning_k_bpb"], output)

    assert output.exists()
    assert contour_calls >= 1
    assert saved_facecolors
    assert saved_facecolors[-1][0] < 0.05
    assert saved_facecolors[-1][1] < 0.08
    assert saved_facecolors[-1][2] < 0.12
