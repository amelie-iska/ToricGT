#!/usr/bin/env python3
"""Watch split-safe non-protein FoT curation and launch ToricBLM training."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from finalize_nonprotein_fot_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def training_running() -> bool:
    result = subprocess.run(
        ["bash", "-lc", "ps -eo args | rg 'train_gpt.py' | rg -v rg || true"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return bool(result.stdout.strip())


def tmux_session_exists(session: str) -> bool:
    result = subprocess.run(["tmux", "has-session", "-t", session], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0


def launch_training(config_path: Path, launch_script: Path, *, session_prefix: str) -> str:
    stamp = utc_stamp()
    session = f"{session_prefix}_{stamp}"
    if tmux_session_exists(session):
        raise RuntimeError(f"tmux session already exists: {session}")
    log_path = ROOT / "logs" / f"{session}.log"
    command = (
        f"cd {ROOT} && "
        f"CONFIG_PATH={config_path} RUN_STAMP={stamp} "
        f"bash {launch_script} 2>&1 | tee {log_path}"
    )
    subprocess.run(["tmux", "new", "-d", "-s", session, command], cwd=ROOT, check=True)
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest-name", default="toricblm_nonprotein_similarity_fot_split_manifest.json")
    parser.add_argument("--min-train-rows", type=int, default=450_000)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--config-path", type=Path, default=ROOT / "configs" / "toricblm_mup_170m_convextok8192_biomed_fot_structure.env")
    parser.add_argument("--launch-script", type=Path, default=ROOT / "scripts" / "launch_toricblm_mup_full_codex55.sh")
    parser.add_argument("--session-prefix", default="toricblm_full_fot_split_safe")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "training_notes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifests" / args.manifest_name
    while True:
        manifest = build_manifest(args.out_dir, min_train_rows=int(args.min_train_rows), validate_all=False)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
        status_path = args.report_dir / f"nonprotein_fot_watch_status_{utc_stamp()}.json"
        status_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
        counts = {
            modality: manifest.get("writer_counts", {}).get(modality, {}).get("train", 0)
            for modality in ("dna", "rna", "small_molecule")
        }
        print(
            f"watch_status ready={manifest['ready_for_training']} counts={counts} "
            f"manifest={manifest_path}",
            flush=True,
        )
        if manifest["ready_for_training"]:
            if training_running():
                print("training already running; watcher exits without launching duplicate.", flush=True)
                return
            if args.dry_run:
                print("dry_run ready; watcher exits without launch.", flush=True)
                return
            session = launch_training(args.config_path, args.launch_script, session_prefix=args.session_prefix)
            print(f"launched_training_session={session}", flush=True)
            return
        time.sleep(max(10, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
