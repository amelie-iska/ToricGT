#!/usr/bin/env python3
# Example:
#   conda run -n tokengt env PYTHONPATH=src python scripts/propose_training_adjustments.py \
#     --analysis-dir outputs/manual_analysis/oai-bpb-postbounce-02000_step2250_20260602T002659Z \
#     --output-json outputs/manual_analysis/oai-bpb-postbounce-02000_step2250_20260602T002659Z/training_adjustment_proposal.json
"""Propose small training-control changes from a completed analysis gate.

This is intentionally conservative.  It is not an optimizer that edits configs
blindly; it converts the same metric files reviewed by Codex into an auditable
proposal that can be accepted, rejected, or modified in the next review turn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def metric(stats: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in stats:
        if row.get("metric") == name:
            return row
    return {}


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    args = parse_args()
    root = Path(args.analysis_dir)
    stats_raw = load_json(root / "metrics" / "metric_stats.json")
    stats = stats_raw if isinstance(stats_raw, list) else []
    geometry = load_json(root / "geometry" / "reasoning_geometry_summary.json")

    train_bpb = metric(stats, "train/bpb")
    train_loss = metric(stats, "train/loss")
    graphcg_loss = metric(stats, "train/graphcg_loss")
    graphcg_cov = metric(stats, "train/graphcg_covariance_loss")
    analogy_map = metric(stats, "train/analogy_step_analogical_map_loss")
    toric_entropy = metric(stats, "train/toric_active_face_entropy")

    bpb_slope = f(train_bpb, "recent_slope_per_1k")
    loss_slope = f(train_loss, "recent_slope_per_1k")
    best_bpb = float(geometry.get("best_bpb", 0.0) or 0.0)
    mean_bpb = float(geometry.get("mean_bpb", 0.0) or 0.0)
    branch_gap = max(0.0, mean_bpb - best_bpb) if mean_bpb and best_bpb else 0.0
    toric_margin = float(geometry.get("mean_toric_shadow_mean_margin", 0.0) or 0.0)
    active_face_margin = float(geometry.get("mean_toric_geometry_active_face_margin", 0.0) or 0.0)

    proposal: dict[str, Any] = {
        "analysis_dir": str(root),
        "signals": {
            "train_bpb_recent_slope_per_1k": bpb_slope,
            "train_loss_recent_slope_per_1k": loss_slope,
            "branch_gap_mean_minus_best_bpb": branch_gap,
            "toric_shadow_mean_margin": toric_margin,
            "toric_active_face_margin": active_face_margin,
            "graphcg_loss_category": graphcg_loss.get("category"),
            "graphcg_covariance_category": graphcg_cov.get("category"),
            "analogy_map_category": analogy_map.get("category"),
            "toric_entropy_category": toric_entropy.get("category"),
        },
        "recommended_phase_delta": {},
        "rationale": [],
        "search_policy": "multi_fidelity_codex_gate",
    }

    delta = proposal["recommended_phase_delta"]
    rationale = proposal["rationale"]

    if bpb_slope < -1.0 and loss_slope < -0.6:
        delta["lr_multiplier"] = "increase_small_or_hold"
        rationale.append("BPB and loss have strong negative recent slopes; avoid large resets.")
    elif bpb_slope < -0.25:
        delta["lr_multiplier"] = "hold_or_increase_10pct"
        rationale.append("BPB is improving but not dramatically; a small LR increase is viable.")
    else:
        delta["lr_multiplier"] = "decrease_or_rollback"
        rationale.append("BPB slope is weak or positive; prefer rollback or lower LR.")

    if graphcg_cov.get("category") == "not_as_desired" or graphcg_loss.get("category") != "as_desired":
        delta["graphcg_loss_weight"] = "enable_5e-5_to_1e-4"
        delta["analogy_lattice_loss_weight"] = "tiny_1e-5_to_2e-5"
        rationale.append("GraphCG basis/covariance diagnostics are drifting; activate a small chart-learning loss.")

    if branch_gap > 0.75:
        delta["gflownet_loss_weight"] = "enable_tiny_2.5e-4_to_5e-4"
        delta["gflownet_entropy_weight"] = "tiny_5e-5_to_1e-4_at_target_2.0"
        rationale.append("Branch search finds much better terminals than the mean branch; promote a tiny GFlowNet objective so the policy begins learning that branch gap while BPB remains dominant.")

    if toric_margin < 0.04 or active_face_margin < 0.0:
        delta["toric_geometry_loss_weight"] = "keep_zero_until_margins_improve"
        delta["koszul_persistence_loss_weight"] = "keep_zero_until_margins_improve"
        rationale.append("Toric active-face margins are weak; heavy toric/Koszul losses would likely fight BPB.")

    if analogy_map.get("category") == "not_as_desired":
        delta["contrastive_loss_weight"] = "increase_small_while_graphcg_on"
        rationale.append("Analogical map residual is drifting; a small contrastive increase should help before topology losses are promoted.")

    if not rationale:
        rationale.append("No decisive metric failure detected; continue current phase and recheck at the next gate.")

    output_json = Path(args.output_json) if args.output_json else root / "training_adjustment_proposal.json"
    output_json.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md = Path(args.output_md) if args.output_md else root / "training_adjustment_proposal.md"
    lines = [
        "# Training Adjustment Proposal",
        "",
        f"- Analysis directory: `{root}`",
        f"- Search policy: `{proposal['search_policy']}`",
        "",
        "## Signals",
    ]
    for key, value in proposal["signals"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Recommended Phase Delta"])
    for key, value in delta.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Rationale"])
    for item in rationale:
        lines.append(f"- {item}")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(proposal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
