import importlib.util
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from toricgt.parameter_golf_export import quantized_state_dict
from toricgt.random_order_lm import DenseRandomOrderToricLM, RandomOrderLMConfig
from toricgt.toric_bgg import (
    ToricBGGConfig,
    ToricBGGProbe,
    boundary_square_residual,
    chain_standard_mask,
    gale_dual_consistency,
    koszul_linearity_residual,
    standard_filtration_leakage,
    toric_bgg_metric_provenance,
    toy_bgg_certificate,
)


def _load_train_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_parameter_golf_random_order.py"
    spec = importlib.util.spec_from_file_location("train_parameter_golf_random_order", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_config_get_float_preserves_explicit_zero_guard_scalars():
    train_script = _load_train_script()
    config = {
        "training": {
            "shock_guard_update_scale": 0.0,
            "shock_guard_grad_norm": "0.00",
            "robust_micro_loss_guard_min_scale": 0.0,
        }
    }
    assert train_script.config_get_float(config, "training", "shock_guard_update_scale", 0.35) == 0.0
    assert train_script.config_get_float(config, "training", "shock_guard_grad_norm", 0.75) == 0.0
    assert train_script.config_get_float(config, "training", "robust_micro_loss_guard_min_scale", 0.10) == 0.0
    assert train_script.config_get_float(config, "training", "missing", 0.35) == 0.35


def test_fineweb_mix_ratio_is_phase_control():
    train_script = _load_train_script()
    config = {
        "phase_curriculum": {
            "enabled": True,
            "phases": [
                {
                    "name": "calibrated",
                    "start_step": 1000,
                    "end_step": 1500,
                    "fineweb_mix_ratio": 0.35,
                }
            ],
        }
    }
    controls, name, index = train_script.active_phase_controls(config, 1250)
    assert name == "calibrated"
    assert index == 0
    assert controls["fineweb_mix_ratio"] == 0.35


def test_two_variable_toy_bgg_certificate_has_zero_boundary_square():
    cert = toy_bgg_certificate()
    assert boundary_square_residual(cert.boundaries).item() == 0.0
    assert cert.standard_mask.shape == (3, 3)
    assert cert.standard_mask[2, 0]
    assert not cert.standard_mask[0, 2]


def test_standard_filtration_leakage_and_koszul_linearity():
    mask = chain_standard_mask(4)
    probs = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )
    labels = torch.tensor([0, 1])
    leakage = standard_filtration_leakage(probs, labels, mask)
    assert torch.isclose(leakage, torch.tensor(0.35))
    mass = torch.tensor([0.0, 0.5, 0.5])
    residual = koszul_linearity_residual(mass, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 3]))
    assert torch.isclose(residual, torch.tensor(0.5))


def test_gale_dual_consistency_zero_for_identity_pair():
    signatures = torch.eye(4)
    loss = gale_dual_consistency(signatures, signatures, torch.eye(4))
    assert loss.item() == 0.0


def test_toric_bgg_probe_reports_finite_metrics():
    hidden = torch.randn(2, 12, 16)
    positions = torch.arange(12).expand(2, 12)
    tokens = torch.randint(4, 32, (2, 12))
    probe = ToricBGGProbe(16, ToricBGGConfig(num_standard_tokens=5, probe_rank=4, signature_dim=6))
    out = probe(hidden, positions, tokens)
    assert out["toric_bgg_loss"].isfinite()
    assert out["toric_bgg_d2_residual"].isfinite()
    assert out["toric_bgg_standard_leakage"].isfinite()
    assert out["toric_bgg_koszul_linearity_residual"].isfinite()
    assert out["toric_bgg_gale_dual_consistency"].isfinite()
    assert out["toric_bgg_exact_certificate_available"].item() == 1.0
    assert out["toric_bgg_provenance_exact_finite_chain"].item() == 1.0
    provenance = toric_bgg_metric_provenance()
    assert provenance["toric_bgg_d2_residual"] == "exact_finite_chain_complex_boundary_square"
    assert provenance["toric_bgg_standard_leakage"] == "exact_finite_chain_poset_standard_mask"


def test_toric_bgg_probe_is_training_only_for_artifact_export():
    cfg = RandomOrderLMConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        num_heads=4,
        num_layers=1,
        recurrent_passes=1,
        ffn_multiplier=2,
        dropout=0.0,
        use_toric_bgg=True,
        toric_bgg_num_standard_tokens=4,
        toric_bgg_probe_rank=4,
        toric_bgg_signature_dim=4,
        use_toric_geometry_tasks=False,
        use_trajectory_memory_head=False,
        use_graphcg=False,
    )
    model = DenseRandomOrderToricLM(cfg)
    payload = quantized_state_dict(model, bits=8)
    assert any(name.startswith("toric_bgg_probe.") for name in payload["excluded"])
    assert not any(name.startswith("toric_bgg_probe.") for name in payload["tensors"])


def test_all_phase_configs_keep_bgg_loss_late_gated() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "config/train.parameter_golf_all_phases.yaml",
        "config/train.parameter_golf_all_phases_medium_conservative.yaml",
    ):
        config = yaml.safe_load((root / relative).read_text(encoding="utf-8"))
        assert float(config["training"]["toric_bgg_loss_weight"]) == 0.0
        phases = config["phase_curriculum"]["phases"]
        seen_category_o = False
        for phase in phases:
            if phase["name"] == "toric_bgg_category_o":
                seen_category_o = True
            if not seen_category_o:
                assert float(phase.get("toric_bgg_loss_weight", 0.0)) == 0.0, (relative, phase["name"])
        assert seen_category_o


def test_render_bgg_category_o_report_writes_provenance_panels(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "render_bgg_category_o_report.py"
    out = tmp_path / "bgg_report"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    html = (out / "index.html").read_text(encoding="utf-8")
    payload = yaml.safe_load((out / "bgg_category_o_report.json").read_text(encoding="utf-8"))
    assert "Toric BGG Category O Report" in html
    assert "exact_finite_chain_complex_boundary_square" in html
    assert payload["metric_provenance"]["toric_bgg_standard_leakage"] == "exact_finite_chain_poset_standard_mask"
    assert payload["gates"]["late_gate_required"] == 1.0
