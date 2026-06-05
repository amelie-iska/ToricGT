#!/usr/bin/env python3
"""Install the finite advanced-loss guard in the live Seq4096 trainer copy.

The Parameter-Golf trainer lives under the ignored ``amelie-iska/`` checkout on
this machine, so this tracked helper keeps the step-0 advanced-loss launch
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


def patch_trainer(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "def finite_aux_scalar" in text and "advanced/aux_loss_nonfinite_suppressed" in text:
        return False

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
    path.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if not args.trainer.exists():
        raise SystemExit(f"missing trainer: {args.trainer}")
    changed = patch_trainer(args.trainer)
    subprocess.run([args.python, "-m", "py_compile", str(args.trainer)], check=True)
    print(f"advanced_loss_guard:{'patched' if changed else 'already_present'} trainer:{args.trainer}")


if __name__ == "__main__":
    main()
