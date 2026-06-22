#!/usr/bin/env python3
"""Publish a completed ToricGT 160M FoT checkpoint bundle to Hugging Face.

The script intentionally refuses to publish startup checkpoints unless
``--allow-low-step`` is set.  It stages a dated checkpoint name plus supporting
artifacts, then uses the logged-in ``hf`` CLI token without printing secrets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "AmelieSchreiber/ToricGT_160M_FoT"
DEFAULT_RUN_ID = "toricblm-mup-codex55-full-20260622T141538Z"


def checkpoint_step(path: Path) -> int:
    match = re.search(r"_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--min-step", type=int, default=25_000)
    parser.add_argument("--allow-low-step", action="store_true")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--staging-root", default=str(ROOT / "hf_uploads" / "ToricGT_160M_FoT"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else ROOT / "checkpoints" / run_id
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "runs" / "oai_sidecar" / run_id
    checkpoints = sorted(checkpoint_dir.glob("*_step_*.pt"), key=checkpoint_step)
    if not checkpoints:
        raise SystemExit(f"No checkpoint files found in {checkpoint_dir}")
    ckpt = checkpoints[-1]
    step = checkpoint_step(ckpt)
    if step < args.min_step and not args.allow_low_step:
        raise SystemExit(
            f"Latest checkpoint is step {step}, below required min step {args.min_step}; refusing upload."
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = Path(args.staging_root) / f"{stamp}_{run_id}_step_{step:06d}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    dated_ckpt_name = f"{stamp}_{run_id}_step_{step:06d}.pt"
    staged_ckpt = staging / "checkpoints" / dated_ckpt_name
    staged_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, staged_ckpt)

    copied: dict[str, str | None] = {
        "checkpoint": str(staged_ckpt),
        "train_log": copy_if_exists(run_dir / "train.log", staging / "artifacts" / stamp / "train.log"),
        "config_env": copy_if_exists(
            ROOT / "configs" / "toricblm_mup_170m_codex55.env",
            staging / "artifacts" / stamp / "toricblm_mup_170m_codex55.env",
        ),
        "mup_base_shapes": copy_if_exists(
            ROOT / "configs" / "mup" / "toricblm_base512_delta768_layers9.bsh",
            staging / "artifacts" / stamp / "toricblm_base512_delta768_layers9.bsh",
        ),
        "late_stage_manifest": copy_if_exists(
            Path("/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/codex55_tot_late_stage/manifest.json"),
            staging / "artifacts" / stamp / "codex55_tot_late_stage_manifest.json",
        ),
        "analysis_plan": copy_if_exists(
            ROOT / "planning" / "TORICBLM-25K-RUN-ANALYSIS-20260622.md",
            staging / "artifacts" / stamp / "TORICBLM-25K-RUN-ANALYSIS-20260622.md",
        ),
        "mup_plan": copy_if_exists(
            ROOT / "planning" / "TORICBLM-MUP-MUTRANSFER-PLAN-20260622.md",
            staging / "artifacts" / stamp / "TORICBLM-MUP-MUTRANSFER-PLAN-20260622.md",
        ),
        "model_card": copy_if_exists(
            ROOT / "hf_model_cards" / "ToricGT_160M_FoT" / "README.md",
            staging / "README.md",
        ),
    }

    manifest = {
        "repo_id": args.repo_id,
        "run_id": run_id,
        "published_at_utc": stamp,
        "checkpoint_step": step,
        "checkpoint_source": str(ckpt),
        "checkpoint_repo_path": f"checkpoints/{dated_ckpt_name}",
        "checkpoint_bytes": ckpt.stat().st_size,
        "checkpoint_sha256": sha256_file(ckpt),
        "supporting_artifacts": copied,
        "notes": (
            "Published by ToricGT post-training publisher. "
            "FineWeb remains the BPB validation stream; toricgt-curated-splits train "
            "and codex5.5_ToT train-only 3-pass records are graph-training streams."
        ),
    }
    (staging / "artifacts" / stamp).mkdir(parents=True, exist_ok=True)
    (staging / "artifacts" / stamp / "upload_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps({"staging": str(staging), **manifest}, indent=2, sort_keys=True))
    if args.dry_run:
        return

    subprocess.run(
        [
            "hf",
            "upload",
            args.repo_id,
            str(staging),
            ".",
            "--type",
            "model",
            "--commit-message",
            f"Upload {run_id} step {step:06d} checkpoint {stamp}",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
