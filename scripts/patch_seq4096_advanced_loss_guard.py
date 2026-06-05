#!/usr/bin/env python3
"""Install reproducible Seq4096 advanced-loss patches.

The Parameter-Golf trainer lives under the ignored ``amelie-iska/`` checkout on
this machine, so this tracked helper keeps step-0 advanced-loss launches
reproducible without committing generated model/checkpoint artifacts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"patch anchor not found: {old[:96]!r}")
    return text.replace(old, new, 1)


def install_finite_guard(text: str) -> tuple[str, bool]:
    if "def finite_aux_scalar" in text and "advanced/aux_loss_nonfinite_suppressed" in text:
        return text, False

    text = replace_once(
        text,
        """    metrics: dict[str, Tensor] = {}\n    weighted = zero\n\n    if weights[\"graphcg\"] > 0.0:\n""",
        """    metrics: dict[str, Tensor] = {}\n    weighted = zero\n\n    def finite_aux_scalar(name: str, value: Tensor, max_value: float = 1.0e4) -> Tensor:\n        finite = torch.isfinite(value.detach())\n        metrics[f\"advanced/{name}_nonfinite\"] = (~finite).to(dtype=torch.float32)\n        safe = torch.where(torch.isfinite(value), value, zero)\n        safe = torch.nan_to_num(safe, nan=0.0, posinf=max_value, neginf=0.0)\n        return safe.clamp_min(0.0)\n\n    if weights[\"graphcg\"] > 0.0:\n""",
    )
    replacements = {
        """        graphcg = off.square().mean() / (diag.square().mean().detach() + 1e-6)\n        probs = (diag.abs() + 1e-6) / (diag.abs().sum() + 1e-6)\n        entropy = -(probs * probs.log()).sum() / math.log(max(probs.numel(), 2))\n        weighted = weighted + weights[\"graphcg\"] * graphcg\n""": """        graphcg = off.square().mean() / (diag.square().mean().detach() + 1e-6)\n        graphcg = finite_aux_scalar(\"graphcg_loss\", graphcg)\n        probs = (diag.abs() + 1e-6) / (diag.abs().sum() + 1e-6)\n        entropy = torch.nan_to_num(-(probs * probs.log()).sum() / math.log(max(probs.numel(), 2)), nan=0.0, posinf=1.0, neginf=0.0)\n        weighted = weighted + weights[\"graphcg\"] * graphcg\n""",
        """        toric = (1.0 - entropy) + F.relu(0.15 - margin).mean()\n        weighted = weighted + weights[\"toric\"] * toric\n""": """        toric = (1.0 - entropy) + F.relu(0.15 - margin).mean()\n        toric = finite_aux_scalar(\"toric_tropical_loss\", toric)\n        weighted = weighted + weights[\"toric\"] * toric\n""",
        """        slepian = 1.0 - concentration\n        weighted = weighted + weights[\"slepian\"] * slepian\n""": """        slepian = 1.0 - concentration\n        slepian = finite_aux_scalar(\"slepian_pollak_loss\", slepian)\n        weighted = weighted + weights[\"slepian\"] * slepian\n""",
        """        koszul = d2_resid + 0.25 * leakage\n        weighted = weighted + weights[\"koszul\"] * koszul\n""": """        koszul = d2_resid + 0.25 * leakage\n        koszul = finite_aux_scalar(\"koszul_bgg_loss\", koszul)\n        weighted = weighted + weights[\"koszul\"] * koszul\n""",
        """        analogy = analogue / denom\n        weighted = weighted + weights[\"analogy\"] * analogy\n""": """        analogy = analogue / denom\n        analogy = finite_aux_scalar(\"analogy_loss\", analogy)\n        weighted = weighted + weights[\"analogy\"] * analogy\n""",
        """    total = scale * weighted\n    metrics[\"advanced/weighted_unscaled_loss\"] = weighted.detach()\n""": """    weighted = finite_aux_scalar(\"weighted_unscaled_loss\", weighted)\n    total = finite_aux_scalar(\"aux_loss_total\", scale * weighted)\n    metrics[\"advanced/weighted_unscaled_loss\"] = weighted.detach()\n""",
        """            aux_loss, aux_metrics = advanced_embedding_losses(base_model, args, x, runtime_scale=runtime_scale)\n            if runtime_scale > 0.0 and args.advanced_loss_max_ce_ratio > 0.0:\n""": """            aux_loss, aux_metrics = advanced_embedding_losses(base_model, args, x, runtime_scale=runtime_scale)\n            if not torch.isfinite(aux_loss.detach()):\n                aux_metrics[\"advanced/aux_loss_nonfinite_suppressed\"] = torch.ones((), device=device, dtype=torch.float32)\n                aux_loss = torch.zeros_like(ce_loss)\n            else:\n                aux_metrics[\"advanced/aux_loss_nonfinite_suppressed\"] = torch.zeros((), device=device, dtype=torch.float32)\n            if runtime_scale > 0.0 and args.advanced_loss_max_ce_ratio > 0.0:\n""",
    }
    for old, new in replacements.items():
        text = replace_once(text, old, new)
    return text, True


def install_combinatorial_bridge(text: str) -> tuple[str, bool]:
    if "toric_cca_topology_loss" in text and "combinatorial_toric_cca_topology_loss" in text:
        return text, False

    anchor = """        metrics.update({\n            \"advanced/koszul_bgg_loss\": koszul.detach(),\n            \"advanced/koszul_d2_residual\": d2_resid.detach(),\n            \"advanced/bgg_standard_leakage\": leakage.detach(),\n        })\n\n    if weights[\"analogy\"] > 0.0 and seq.size(0) >= 6:\n"""
    bridge = """        metrics.update({\n            \"advanced/koszul_bgg_loss\": koszul.detach(),\n            \"advanced/koszul_d2_residual\": d2_resid.detach(),\n            \"advanced/bgg_standard_leakage\": leakage.detach(),\n        })\n\n    if (weights[\"toric\"] > 0.0 or weights[\"koszul\"] > 0.0) and seq.size(0) >= 8:\n        try:\n            from toricgt.combinatorial_toric_metrics import (\n                CombinatorialToricConfig,\n                combinatorial_toric_cca_topology_loss,\n            )\n\n            positions = torch.arange(seq.size(0), device=seq.device, dtype=torch.long).view(1, -1)\n            cca_metrics = combinatorial_toric_cca_topology_loss(\n                seq.unsqueeze(0),\n                positions,\n                config=CombinatorialToricConfig(\n                    max_points=min(12, int(getattr(args, \"advanced_loss_sample_tokens\", 64))),\n                    max_windows=1,\n                    window_size=min(32, seq.size(0)),\n                    step_stride=16,\n                    num_chambers=max(4, int(getattr(args, \"toric_tropical_fan_bins\", 8))),\n                    temperature=0.16,\n                    max_loss_value=16.0,\n                ),\n            )\n            cca_loss = finite_aux_scalar(\n                \"toric_cca_topology_loss\",\n                cca_metrics[\"toric_cca_topology_loss\"],\n                max_value=16.0,\n            )\n            cca_weight = 0.15 * weights[\"toric\"] + 0.15 * weights[\"koszul\"]\n            weighted = weighted + cca_weight * cca_loss\n            for cca_key, cca_value in cca_metrics.items():\n                if isinstance(cca_value, torch.Tensor):\n                    metrics[f\"advanced/{cca_key}\"] = cca_value.detach()\n            metrics[\"advanced/toric_cca_weight\"] = torch.as_tensor(\n                cca_weight,\n                device=seq.device,\n                dtype=torch.float32,\n            )\n            metrics[\"advanced/toric_cca_bridge_failed\"] = torch.zeros((), device=seq.device, dtype=torch.float32)\n        except Exception:\n            metrics[\"advanced/toric_cca_bridge_failed\"] = torch.ones((), device=seq.device, dtype=torch.float32)\n\n    if weights[\"analogy\"] > 0.0 and seq.size(0) >= 6:\n"""
    return replace_once(text, anchor, bridge), True


def install_polarquant_warmup(text: str) -> tuple[str, bool]:
    changed = False
    if "polarquant_train_start_step" not in text:
        text = replace_once(
            text,
            """    polarquant_train_sample_tokens = int(os.environ.get(\"POLARQUANT_TRAIN_SAMPLE_TOKENS\", 0))\n    polarquant_eval_sample_tokens = int(os.environ.get(\"POLARQUANT_EVAL_SAMPLE_TOKENS\", 0))\n    polarquant_seed = int(os.environ.get(\"POLARQUANT_SEED\", 271828))\n""",
            """    polarquant_train_sample_tokens = int(os.environ.get(\"POLARQUANT_TRAIN_SAMPLE_TOKENS\", 0))\n    polarquant_eval_sample_tokens = int(os.environ.get(\"POLARQUANT_EVAL_SAMPLE_TOKENS\", 0))\n    polarquant_seed = int(os.environ.get(\"POLARQUANT_SEED\", 271828))\n    polarquant_train_start_step = int(os.environ.get(\"POLARQUANT_TRAIN_START_STEP\", 0))\n    polarquant_train_warmup_steps = int(os.environ.get(\"POLARQUANT_TRAIN_WARMUP_STEPS\", 0))\n""",
        )
        changed = True
    if "polarquant_train_scale" not in text:
        text = replace_once(
            text,
            """        self.polarquant_train_sample_tokens = int(polarquant_train_sample_tokens)\n        self.polarquant_eval_sample_tokens = int(polarquant_eval_sample_tokens)\n        if self.head_dim % 2 != 0:\n""",
            """        self.polarquant_train_sample_tokens = int(polarquant_train_sample_tokens)\n        self.polarquant_eval_sample_tokens = int(polarquant_eval_sample_tokens)\n        self.register_buffer(\"polarquant_train_scale\", torch.ones((), dtype=torch.float32), persistent=False)\n        if self.head_dim % 2 != 0:\n""",
        )
        text = replace_once(
            text,
            """            k = sampled_polarquant_kv_perturb(\n                k,\n                self.polarquant_signs.to(device=k.device),\n                self.polarquant_kv_bits,\n                sample_tokens,\n            )\n            v = sampled_polarquant_kv_perturb(\n                v,\n                self.polarquant_signs.to(device=v.device),\n                self.polarquant_kv_bits,\n                sample_tokens,\n            )\n""",
            """            k_perturbed = sampled_polarquant_kv_perturb(\n                k,\n                self.polarquant_signs.to(device=k.device),\n                self.polarquant_kv_bits,\n                sample_tokens,\n            )\n            v_perturbed = sampled_polarquant_kv_perturb(\n                v,\n                self.polarquant_signs.to(device=v.device),\n                self.polarquant_kv_bits,\n                sample_tokens,\n            )\n            if self.training:\n                train_scale = self.polarquant_train_scale.to(device=k.device, dtype=k.dtype).clamp(0.0, 1.0)\n                k = k + train_scale * (k_perturbed - k)\n                v = v + train_scale * (v_perturbed - v)\n            else:\n                k = k_perturbed\n                v = v_perturbed\n""",
        )
        text = replace_once(
            text,
            """def advanced_loss_runtime_scale(args: object, step: int, best_val_bpb: float) -> float:\n""",
            """def polarquant_train_runtime_scale(args: object, step: int) -> float:\n    if not bool(getattr(args, \"polarquant_train\", False)):\n        return 0.0\n    start_step = int(getattr(args, \"polarquant_train_start_step\", 0))\n    if step < start_step:\n        return 0.0\n    warmup = int(getattr(args, \"polarquant_train_warmup_steps\", 0))\n    if warmup <= 0:\n        return 1.0\n    return min(max(step - start_step + 1, 0) / float(warmup), 1.0)\n\n\ndef set_polarquant_train_scale_(model: nn.Module, scale: float) -> None:\n    for module in model.modules():\n        if hasattr(module, \"polarquant_train_scale\"):\n            module.polarquant_train_scale.fill_(float(scale))\n\n\ndef advanced_loss_runtime_scale(args: object, step: int, best_val_bpb: float) -> float:\n""",
        )
        text = replace_once(
            text,
            """        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n        scale = lr_mul(step, elapsed_ms)\n        zero_grad_all()\n""",
            """        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n        scale = lr_mul(step, elapsed_ms)\n        polarquant_train_scale = polarquant_train_runtime_scale(args, step)\n        set_polarquant_train_scale_(base_model, polarquant_train_scale)\n        zero_grad_all()\n""",
        )
        text = replace_once(
            text,
            """                \"advanced/backprop_enabled\": float(advanced_runtime_scale_avg.item() > 0.0),\n                \"train/perplexity\": math.exp(min(float(train_loss.item()), 20.0)),\n""",
            """                \"advanced/backprop_enabled\": float(advanced_runtime_scale_avg.item() > 0.0),\n                \"polarquant/train_scale\": polarquant_train_scale,\n                \"polarquant/train_start_step\": args.polarquant_train_start_step,\n                \"polarquant/train_warmup_steps\": args.polarquant_train_warmup_steps,\n                \"train/perplexity\": math.exp(min(float(train_loss.item()), 20.0)),\n""",
        )
        changed = True
    return text, changed


def patch_trainer(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    applied: list[str] = []
    for name, installer in (
        ("finite_guard", install_finite_guard),
        ("combinatorial_bridge", install_combinatorial_bridge),
        ("polarquant_warmup", install_polarquant_warmup),
    ):
        text, changed = installer(text)
        if changed:
            applied.append(name)
    if applied:
        path.write_text(text, encoding="utf-8")
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if not args.trainer.exists():
        raise SystemExit(f"missing trainer: {args.trainer}")
    applied = patch_trainer(args.trainer)
    subprocess.run([args.python, "-m", "py_compile", str(args.trainer)], check=True)
    status = "patched:" + ",".join(applied) if applied else "already_present"
    print(f"advanced_loss_patches:{status} trainer:{args.trainer}")


if __name__ == "__main__":
    main()
