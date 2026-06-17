#!/usr/bin/env python3
"""Hand off an already-running campaign to the patched adaptive controller.

The current tmux campaign process has already imported its Python source, so it
cannot see edits made after launch.  This watcher waits until the current run
finishes the configured gate, kills the old campaign sessions only after the
run has produced its final checkpoint/roundtrip evidence, runs the patched full
analysis, then launches a new adaptive campaign from step 0 using the analysis
as prior evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_RE = re.compile(r"checkpoint_saved:(?P<path>.*?) step:(?P<step>\d+) val_bpb:(?P<bpb>[-+0-9.eE]+|None)")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--current-campaign-id", required=True)
    parser.add_argument("--current-run-id", required=True)
    parser.add_argument("--old-campaign-session", required=True)
    parser.add_argument("--old-watch-session", default="")
    parser.add_argument("--new-campaign-id", default="")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--target-bpb", type=float, default=1.19)
    parser.add_argument("--max-runs", type=int, default=25)
    parser.add_argument("--followup-runs-after-meta", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--conda-bin", default="/home/iska/miniconda3/bin/conda")
    parser.add_argument("--conda-env", default="tokengt")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "amelie-iska-math"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "toricgt-parameter-golf"))
    parser.add_argument("--graph-data-path", default="/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/curated_hf_shards")
    parser.add_argument("--tokenizer-path", default="amelie-iska/parameter-golf/data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--analysis-records", type=int, default=3)
    parser.add_argument("--analysis-cas-max-points", type=int, default=6)
    parser.add_argument("--analysis-cas-macaulay2-timeout-seconds", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"[{utc_iso()}] $ {' '.join(shlex.quote(part) for part in command)}", flush=True)
    proc = subprocess.run(command, cwd=cwd, text=True)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)
    return proc


def run_shell(command: str, *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"[{utc_iso()}] $ {command}", flush=True)
    proc = subprocess.run(["bash", "-lc", command], cwd=cwd, text=True)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)
    return proc


def train_log(repo: Path, run_id: str) -> Path:
    return repo / "runs" / "oai_sidecar" / run_id / "train.log"


def latest_checkpoint(log_path: Path) -> Path | None:
    if not log_path.exists():
        return None
    latest: Path | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := CHECKPOINT_RE.search(line):
            latest = Path(match.group("path"))
    return latest


def gate_complete(log_path: Path, steps: int) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return (
        f"step:{steps}/{steps} val_loss:" in text
        and "final_int8_zlib_roundtrip_exact" in text
        and latest_checkpoint(log_path) is not None
    )


def kill_session(name: str, repo: Path) -> None:
    if not name:
        return
    run(["tmux", "has-session", "-t", name], cwd=repo, check=False)
    run(["tmux", "kill-session", "-t", name], cwd=repo, check=False)


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    new_campaign_id = args.new_campaign_id or f"{args.current_campaign_id}-adaptive-{utc_stamp()}"
    notes_dir = repo / "training_notes" / new_campaign_id
    notes_dir.mkdir(parents=True, exist_ok=True)
    log_file = notes_dir / "adaptive_takeover.log"
    # Mirror stdout to a persistent note through shell redirection when launched
    # by tmux; write a small JSON state here as well for direct inspection.
    state_path = notes_dir / "adaptive_takeover_state.json"
    current_log = train_log(repo, args.current_run_id)
    print(f"[{utc_iso()}] waiting for {current_log} to reach gate {args.steps}", flush=True)
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        return
    while not gate_complete(current_log, args.steps):
        state_path.write_text(
            json.dumps(
                {
                    "updated_utc": utc_iso(),
                    "state": "waiting_for_current_gate",
                    "current_run_id": args.current_run_id,
                    "train_log": str(current_log),
                    "gate_steps": args.steps,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        time.sleep(max(5, int(args.poll_seconds)))
    ckpt = latest_checkpoint(current_log)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {current_log}")
    print(f"[{utc_iso()}] current gate complete; checkpoint={ckpt}", flush=True)
    kill_session(args.old_watch_session, repo)
    kill_session(args.old_campaign_session, repo)
    analysis_dir = notes_dir / f"PRIOR-RUN-FULL-ANALYSIS-{args.current_run_id}"
    graph_glob = str(Path(args.graph_data_path) / "train" / "*.parquet")
    analysis_cmd = [
        args.conda_bin,
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "env",
        f"PYTHONPATH={repo / 'src'}",
        "python",
        "scripts/run_oai_sidecar_full_iteration_analysis.py",
        "--run-id",
        args.current_run_id,
        "--run-path",
        f"{args.wandb_entity}/{args.wandb_project}/{args.current_run_id}",
        "--checkpoint",
        str(ckpt),
        "--train-log",
        str(current_log),
        "--output-dir",
        str(analysis_dir),
        "--tokenizer-path",
        args.tokenizer_path,
        "--graph-train-glob",
        graph_glob,
        "--records",
        str(args.analysis_records),
        "--embedding-records",
        str(args.analysis_records),
        "--cas-max-points",
        str(args.analysis_cas_max_points),
        "--cas-macaulay2-timeout-seconds",
        str(args.analysis_cas_macaulay2_timeout_seconds),
        "--strict",
    ]
    run(analysis_cmd, cwd=repo, check=True)
    campaign_session = f"toricgt_oai_gate1500_{new_campaign_id}"
    watch_session = f"toricgt_full_analysis_codex_watch_{new_campaign_id}"
    campaign_log = notes_dir / "campaign_supervisor.log"
    watch_log = notes_dir / "full_analysis_codex_watch.log"
    campaign_cmd = (
        f"cd {shlex.quote(str(repo))} && "
        f"{shlex.quote(args.conda_bin)} run --no-capture-output -n {shlex.quote(args.conda_env)} "
        f"env PYTHONPATH=src python scripts/run_oai_sidecar_bpb_campaign.py "
        f"--campaign-id {shlex.quote(new_campaign_id)} "
        f"--max-runs {int(args.max_runs)} "
        f"--followup-runs-after-meta {int(args.followup_runs_after_meta)} "
        f"--target-bpb {float(args.target_bpb)} "
        f"--steps-per-run {int(args.steps)} "
        f"--profile-offset 0 "
        f"--full-analysis --strict-analysis "
        f"--analysis-retries 1 "
        f"--analysis-retry-timeout-multiplier 2.0 "
        f"--analysis-records {int(args.analysis_records)} "
        f"--analysis-cas-max-points {int(args.analysis_cas_max_points)} "
        f"--analysis-cas-macaulay2-timeout-seconds {int(args.analysis_cas_macaulay2_timeout_seconds)} "
        f"--graph-data-path {shlex.quote(args.graph_data_path)} "
        f"--prior-log {shlex.quote(str(current_log))} "
        f"--prior-analysis-dir {shlex.quote(str(analysis_dir))} "
        f"--codex-review > {shlex.quote(str(campaign_log))} 2>&1"
    )
    watch_cmd = (
        f"cd {shlex.quote(str(repo))} && "
        f"{shlex.quote(args.conda_bin)} run --no-capture-output -n {shlex.quote(args.conda_env)} "
        f"python scripts/watch_full_analysis_codex_reviews.py "
        f"--campaign-notes-dir {shlex.quote(str(notes_dir))} "
        f"--poll-seconds 60 --max-reviews 35 > {shlex.quote(str(watch_log))} 2>&1"
    )
    run(["tmux", "new-session", "-d", "-s", campaign_session, campaign_cmd], cwd=repo, check=True)
    run(["tmux", "new-session", "-d", "-s", watch_session, watch_cmd], cwd=repo, check=True)
    state_path.write_text(
        json.dumps(
            {
                "updated_utc": utc_iso(),
                "state": "adaptive_campaign_launched",
                "previous_run_id": args.current_run_id,
                "analysis_dir": str(analysis_dir),
                "new_campaign_id": new_campaign_id,
                "campaign_session": campaign_session,
                "watch_session": watch_session,
                "campaign_log": str(campaign_log),
                "watch_log": str(watch_log),
                "takeover_log": str(log_file),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[{utc_iso()}] adaptive campaign launched: {new_campaign_id}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

