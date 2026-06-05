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
        if "advanced/log_only" in text and "effective_total = weighted.new_zeros(()) if log_only else scale * weighted" in text:
            return text, False
        text = replace_once(
            text,
            """    total = finite_aux_scalar("aux_loss_total", scale * weighted)
    metrics["advanced/weighted_unscaled_loss"] = weighted.detach()
    metrics["advanced/runtime_scale"] = torch.as_tensor(scale, device=weighted.device, dtype=weighted.dtype)
""",
            """    effective_total = weighted.new_zeros(()) if log_only else scale * weighted
    total = finite_aux_scalar("aux_loss_total", effective_total)
    metrics["advanced/weighted_unscaled_loss"] = weighted.detach()
    metrics["advanced/log_only"] = torch.as_tensor(float(log_only), device=weighted.device, dtype=weighted.dtype)
    metrics["advanced/runtime_scale"] = torch.as_tensor(scale, device=weighted.device, dtype=weighted.dtype)
""",
        )
        return text, True

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
        """    total = finite_aux_scalar(\"aux_loss_total\", scale * weighted)\n    metrics[\"advanced/weighted_unscaled_loss\"] = weighted.detach()\n    metrics[\"advanced/runtime_scale\"] = torch.as_tensor(scale, device=weighted.device, dtype=weighted.dtype)\n""": """    effective_total = weighted.new_zeros(()) if log_only else scale * weighted\n    total = finite_aux_scalar(\"aux_loss_total\", effective_total)\n    metrics[\"advanced/weighted_unscaled_loss\"] = weighted.detach()\n    metrics[\"advanced/log_only\"] = torch.as_tensor(float(log_only), device=weighted.device, dtype=weighted.dtype)\n    metrics[\"advanced/runtime_scale\"] = torch.as_tensor(scale, device=weighted.device, dtype=weighted.dtype)\n""",
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
    if "k_perturbed = sampled_polarquant_kv_perturb" in text:
        text = replace_once(
            text,
            """            k_perturbed = sampled_polarquant_kv_perturb(\n                k,\n                self.polarquant_signs.to(device=k.device),\n                self.polarquant_kv_bits,\n                sample_tokens,\n            )\n            v_perturbed = sampled_polarquant_kv_perturb(\n                v,\n                self.polarquant_signs.to(device=v.device),\n                self.polarquant_kv_bits,\n                sample_tokens,\n            )\n            if self.training:\n                train_scale = self.polarquant_train_scale.to(device=k.device, dtype=k.dtype).clamp(0.0, 1.0)\n                k = k + train_scale * (k_perturbed - k)\n                v = v + train_scale * (v_perturbed - v)\n            else:\n                k = k_perturbed\n                v = v_perturbed\n""",
            """            k = sampled_polarquant_kv_perturb(\n                k,\n                self.polarquant_signs.to(device=k.device),\n                self.polarquant_kv_bits,\n                sample_tokens,\n            )\n            v = sampled_polarquant_kv_perturb(\n                v,\n                self.polarquant_signs.to(device=v.device),\n                self.polarquant_kv_bits,\n                sample_tokens,\n            )\n""",
        )
        changed = True
    return text, changed


def install_nonfinite_update_skip(text: str) -> tuple[str, bool]:
    if "train/nonfinite_update_skip" in text and "optim/grad_norm" in text:
        return text, False

    old = """        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
"""
    new = """        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        grad_norm_value = 0.0
        nonfinite_update_skip = 0.0
        if (
            not torch.isfinite(train_loss.detach())
            or not torch.isfinite(train_aux_loss.detach())
            or not math.isfinite(train_bpb)
        ):
            nonfinite_update_skip = 1.0
        else:
            for group in optimizer_muon.param_groups:
                group["momentum"] = muon_momentum

            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["base_lr"] * scale

            if args.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
                grad_norm_value = float(
                    torch.nan_to_num(
                        grad_norm.detach().float(),
                        nan=float("inf"),
                        posinf=float("inf"),
                        neginf=float("inf"),
                    ).item()
                )
                if not math.isfinite(grad_norm_value):
                    nonfinite_update_skip = 1.0
            if nonfinite_update_skip < 0.5:
                for opt in optimizers:
                    opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
"""
    text = replace_once(text, old, new)
    text = replace_once(
        text,
        """                "optim/scalar_lr": args.scalar_lr * scale,
            }
""",
        """                "optim/scalar_lr": args.scalar_lr * scale,
                "optim/grad_norm": grad_norm_value,
                "train/nonfinite_update_skip": nonfinite_update_skip,
            }
""",
    )
    return text, True


def install_nonfinite_source_telemetry(text: str) -> tuple[str, bool]:
    if "train/nonfinite_microbatch_skip_count" in text and "nonfinite_param_count" in text:
        return text, False

    text = replace_once(
        text,
        """    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
""",
        """    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    nonfinite_audit_every = int(os.environ.get("NONFINITE_AUDIT_EVERY", 25))
""",
    )
    text = replace_once(
        text,
        """        train_token_count = torch.zeros((), device=device, dtype=torch.float64)
        train_byte_count = torch.zeros((), device=device, dtype=torch.float64)
        for micro_step in range(grad_accum_steps):
""",
        """        train_token_count = torch.zeros((), device=device, dtype=torch.float64)
        train_byte_count = torch.zeros((), device=device, dtype=torch.float64)
        nonfinite_microbatch_skip_count = torch.zeros((), device=device, dtype=torch.float32)
        nonfinite_ce_loss_count = torch.zeros((), device=device, dtype=torch.float32)
        nonfinite_aux_loss_count = torch.zeros((), device=device, dtype=torch.float32)
        for micro_step in range(grad_accum_steps):
""",
    )
    text = replace_once(
        text,
        """            if not torch.isfinite(aux_loss.detach()):
                aux_metrics["advanced/aux_loss_nonfinite_suppressed"] = torch.ones((), device=device, dtype=torch.float32)
                aux_loss = torch.zeros_like(ce_loss)
            else:
                aux_metrics["advanced/aux_loss_nonfinite_suppressed"] = torch.zeros((), device=device, dtype=torch.float32)
            if runtime_scale > 0.0 and args.advanced_loss_max_ce_ratio > 0.0:
""",
        """            ce_loss_is_finite = torch.isfinite(ce_loss.detach())
            if not bool(ce_loss_is_finite.item()):
                nonfinite_ce_loss_count += 1.0
                aux_metrics["advanced/ce_loss_nonfinite"] = torch.ones((), device=device, dtype=torch.float32)
            else:
                aux_metrics["advanced/ce_loss_nonfinite"] = torch.zeros((), device=device, dtype=torch.float32)
            if not torch.isfinite(aux_loss.detach()):
                nonfinite_aux_loss_count += 1.0
                aux_metrics["advanced/aux_loss_nonfinite_suppressed"] = torch.ones((), device=device, dtype=torch.float32)
                aux_loss = torch.zeros_like(ce_loss)
            else:
                aux_metrics["advanced/aux_loss_nonfinite_suppressed"] = torch.zeros((), device=device, dtype=torch.float32)
            if runtime_scale > 0.0 and args.advanced_loss_max_ce_ratio > 0.0:
""",
    )
    text = replace_once(
        text,
        """            loss = ce_loss + aux_loss
            train_loss += ce_loss.detach()
            train_aux_loss += aux_loss.detach()
            advanced_runtime_scale_sum += torch.as_tensor(runtime_scale, device=device, dtype=torch.float32)
            for key, value in aux_metrics.items():
                advanced_metric_sums[key] = advanced_metric_sums.get(key, torch.zeros((), device=device)) + value.detach()
            (loss * grad_scale).backward()
""",
        """            loss = ce_loss + aux_loss
            loss_is_finite = torch.isfinite(loss.detach())
            train_loss += torch.nan_to_num(ce_loss.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            train_aux_loss += torch.nan_to_num(aux_loss.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            advanced_runtime_scale_sum += torch.as_tensor(runtime_scale, device=device, dtype=torch.float32)
            for key, value in aux_metrics.items():
                advanced_metric_sums[key] = advanced_metric_sums.get(key, torch.zeros((), device=device)) + value.detach()
            if not bool(loss_is_finite.item()):
                nonfinite_microbatch_skip_count += 1.0
                continue
            (loss * grad_scale).backward()
""",
    )
    text = replace_once(
        text,
        """            or not math.isfinite(train_bpb)
        ):
            nonfinite_update_skip = 1.0
        else:
""",
        """            or not math.isfinite(train_bpb)
            or float(nonfinite_microbatch_skip_count.item()) > 0.0
        ):
            nonfinite_update_skip = 1.0
        else:
""",
    )
    text = replace_once(
        text,
        """        zero_grad_all()

        step += 1
""",
        """        audit_params = args.nonfinite_audit_every > 0 and (
            step <= 10 or step % args.nonfinite_audit_every == 0 or nonfinite_update_skip > 0.5
        )
        nonfinite_param_count = 0.0
        nonfinite_grad_count = 0.0
        max_abs_param = 0.0
        max_abs_grad = 0.0
        if audit_params:
            with torch.no_grad():
                for param in base_model.parameters():
                    data = param.detach()
                    finite_data = torch.isfinite(data)
                    nonfinite_param_count += float((~finite_data).sum().item())
                    if finite_data.any():
                        max_abs_param = max(max_abs_param, float(data[finite_data].abs().max().item()))
                    if param.grad is not None:
                        grad = param.grad.detach()
                        finite_grad = torch.isfinite(grad)
                        nonfinite_grad_count += float((~finite_grad).sum().item())
                        if finite_grad.any():
                            max_abs_grad = max(max_abs_grad, float(grad[finite_grad].abs().max().item()))
        zero_grad_all()

        step += 1
""",
    )
    text = replace_once(
        text,
        """                "train/nonfinite_update_skip": nonfinite_update_skip,
            }
""",
        """                "train/nonfinite_update_skip": nonfinite_update_skip,
                "train/nonfinite_microbatch_skip_count": float(nonfinite_microbatch_skip_count.item()),
                "train/nonfinite_ce_loss_count": float(nonfinite_ce_loss_count.item()),
                "train/nonfinite_aux_loss_count": float(nonfinite_aux_loss_count.item()),
                "train/nonfinite_param_count": nonfinite_param_count,
                "train/nonfinite_grad_count": nonfinite_grad_count,
                "optim/max_abs_param": max_abs_param,
                "optim/max_abs_grad": max_abs_grad,
            }
""",
    )
    return text, True


def patch_trainer(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    applied: list[str] = []
    for name, installer in (
        ("finite_guard", install_finite_guard),
        ("combinatorial_bridge", install_combinatorial_bridge),
        ("polarquant_warmup", install_polarquant_warmup),
        ("nonfinite_update_skip", install_nonfinite_update_skip),
        ("nonfinite_source_telemetry", install_nonfinite_source_telemetry),
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
