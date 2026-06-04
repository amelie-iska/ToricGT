#!/usr/bin/env python3
"""Keep Seq4096 live analysis attached to the newest training run.

The 4K recovery gate can relaunch fresh ``toricgt_seq4096_*`` runs while the
training loop is active.  A per-run ``watch_seq4096_analysis.py`` process is
useful, but it is easy to leave it attached to an old run after a recovery
restart.  This supervisor watches tmux for the newest active training session
and starts one compatible checkpoint-analysis watcher per run.

This intentionally uses the Seq4096-compatible analysis suite.  The historical
``watch_training_analysis.py`` suite expects RandomOrderLM checkpoints with a
``RandomOrderLMConfig`` payload; compact Seq4096 competition checkpoints are
GPT-style payloads and do not safely load in those entrypoints yet.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


STAMP_RE = re.compile(r"(\d{8}T\d{6}Z)")
STEP_RE = re.compile(r"_step_(\d+)\.pt$")
LOG_STEP_RE = re.compile(r"step:(\d+)/\d+")
RESUME_STEP_RE = re.compile(r"checkpoint_resumed:.* step:(\d+)")
RESUME_CHECKPOINT_PATH_RE = re.compile(r"resume_checkpoint:.*_step_(\d+)\.pt")
EXCLUDED_SESSION_FRAGMENTS = (
    "_analysis_",
    "_full_diag_",
    "_mirror_",
    "_gate_",
    "live_full_analysis",
    "live_periodic",
)


@dataclass(frozen=True)
class WatcherLaunch:
    run_id: str
    session: str
    command_file: Path
    watcher_log: Path
    output_root: Path
    start_step: int
    shell_text: str


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/iska/Documents/amelie/bio/ToricGT")
    parser.add_argument("--python", default="/home/iska/miniconda3/envs/tokengt/bin/python")
    parser.add_argument("--live-root", default="", help="Defaults to <root>/outputs/live_periodic_reviews")
    parser.add_argument("--target-bpb", type=float, default=1.2)
    parser.add_argument("--gate-step", type=int, default=4000)
    parser.add_argument("--interval-steps", type=int, default=250)
    parser.add_argument("--watcher-poll-seconds", type=float, default=30.0)
    parser.add_argument("--supervisor-poll-seconds", type=float, default=60.0)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "toricgt-parameter-golf"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "amelie-iska-math"))
    parser.add_argument("--tmux-prefix", default="toricgt_seq4096_live_full_analysis_")
    parser.add_argument("--codex-review-hook", default="scripts/codex_training_review_resume.sh")
    parser.add_argument("--codex-review-tmux-prefix", default="toricgt_codex_review_seq4096_live")
    parser.add_argument("--once", action="store_true", help="Attach once and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Write no files and launch no tmux sessions.")
    return parser.parse_args()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def tmux_sessions() -> list[str]:
    result = run_command(["tmux", "list-sessions", "-F", "#S"])
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def is_training_session(session: str) -> bool:
    return session.startswith("toricgt_seq4096") and not any(
        fragment in session for fragment in EXCLUDED_SESSION_FRAGMENTS
    )


def active_training_sessions(sessions: Iterable[str]) -> list[str]:
    return [session for session in sessions if is_training_session(session)]


def session_stamp(session: str) -> str:
    matches = STAMP_RE.findall(session)
    return matches[-1] if matches else "00000000T000000Z"


def latest_active_run(sessions: Iterable[str]) -> str:
    candidates = active_training_sessions(sessions)
    if not candidates:
        return ""
    return sorted(candidates, key=lambda item: (session_stamp(item), item))[-1]


def tmux_has_session(session: str) -> bool:
    return run_command(["tmux", "has-session", "-t", session]).returncode == 0


def latest_checkpoint_step(checkpoint_dir: Path) -> int:
    if not checkpoint_dir.exists():
        return 0
    best = 0
    for path in checkpoint_dir.glob("*_step_*.pt"):
        match = STEP_RE.search(path.name)
        if not match:
            continue
        best = max(best, int(match.group(1)))
    return best


def latest_log_step(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    best = 0
    for match in LOG_STEP_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
        best = max(best, int(match.group(1)))
    return best


def resume_step(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    best = 0
    for match in RESUME_STEP_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
        best = max(best, int(match.group(1)))
    return best


def resume_checkpoint_path_step(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    best = 0
    for match in RESUME_CHECKPOINT_PATH_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
        best = max(best, int(match.group(1)))
    return best


def start_step_for_run(checkpoint_dir: Path, log_path: Path, interval_steps: int) -> int:
    interval = max(1, int(interval_steps))
    checkpoint_step = latest_checkpoint_step(checkpoint_dir)
    if checkpoint_step > 0:
        return max(0, checkpoint_step - interval)
    base = max(latest_log_step(log_path), resume_step(log_path), resume_checkpoint_path_step(log_path))
    if base <= 0:
        return 0
    return base // interval * interval


def quote(value: str | Path | int | float) -> str:
    return shlex.quote(str(value))


def sanitize_session_name(value: str, limit: int = 180) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)
    return (clean[:limit].rstrip("_") or "toricgt_seq4096_live_full_analysis")


def build_watcher_launch(
    *,
    root: Path,
    python_bin: Path,
    live_root: Path,
    run_id: str,
    start_step: int,
    interval_steps: int,
    watcher_poll_seconds: float,
    target_bpb: float,
    gate_step: int,
    project: str,
    entity: str,
    tmux_prefix: str,
    codex_review_hook: str = "",
    codex_review_tmux_prefix: str = "",
) -> WatcherLaunch:
    checkpoint_dir = root / "amelie-iska" / "parameter-golf" / "checkpoints" / run_id
    log_path = root / "amelie-iska" / "parameter-golf" / "logs" / f"{run_id}.txt"
    output_root = live_root / run_id
    supervisor_dir = root / "logs" / run_id / "supervisor"
    command_file = supervisor_dir / "live_periodic_full_analysis_command.sh"
    watcher_log = root / "logs" / f"{run_id}.live_periodic_full_analysis.txt"
    session = sanitize_session_name(f"{tmux_prefix}{run_id}")
    lines = [
        "#!/usr/bin/env bash",
        f"cd {quote(root)} || exit 1",
        f"export PYTHONPATH=src WANDB_PROJECT={quote(project)} WANDB_ENTITY={quote(entity)}",
        " ".join(
            [
                "exec",
                quote(python_bin),
                "scripts/watch_seq4096_analysis.py",
                "--checkpoint-dir",
                quote(checkpoint_dir),
                "--log",
                quote(log_path),
                "--run-path",
                quote(f"{entity}/{project}/{run_id}"),
                "--run-id",
                quote(run_id),
                "--output-root",
                quote(output_root),
                "--start-step",
                quote(start_step),
                "--analyze-start-step",
                "--interval-steps",
                quote(interval_steps),
                "--poll-seconds",
                quote(watcher_poll_seconds),
                "--target-bpb",
                quote(target_bpb),
                "--gate-step",
                quote(gate_step),
                "--training-tmux",
                quote(run_id),
                "--python",
                quote(python_bin),
            ]
            + (
                [
                    "--codex-review-hook",
                    quote((root / codex_review_hook).resolve() if codex_review_hook and not Path(codex_review_hook).is_absolute() else codex_review_hook),
                ]
                if codex_review_hook
                else []
            )
            + (
                ["--codex-review-tmux-prefix", quote(codex_review_tmux_prefix)]
                if codex_review_tmux_prefix
                else []
            )
            + [
                "2>&1",
                "|",
                "tee",
                "-a",
                quote(watcher_log),
            ]
        ),
    ]
    return WatcherLaunch(
        run_id=run_id,
        session=session,
        command_file=command_file,
        watcher_log=watcher_log,
        output_root=output_root,
        start_step=int(start_step),
        shell_text="\n".join(lines) + "\n",
    )


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_stamp()} {message}\n")


def ensure_run_watcher(args: argparse.Namespace, run_id: str, log_path: Path) -> WatcherLaunch | None:
    root = Path(args.root)
    live_root = Path(args.live_root) if args.live_root else root / "outputs" / "live_periodic_reviews"
    checkpoint_dir = root / "amelie-iska" / "parameter-golf" / "checkpoints" / run_id
    train_log = root / "amelie-iska" / "parameter-golf" / "logs" / f"{run_id}.txt"
    if not checkpoint_dir.exists() or not train_log.exists():
        append_log(log_path, f"skip run={run_id} missing_checkpoint_dir_or_log")
        return None
    start_step = start_step_for_run(checkpoint_dir, train_log, args.interval_steps)
    launch = build_watcher_launch(
        root=root,
        python_bin=Path(args.python),
        live_root=live_root,
        run_id=run_id,
        start_step=start_step,
        interval_steps=args.interval_steps,
        watcher_poll_seconds=args.watcher_poll_seconds,
        target_bpb=args.target_bpb,
        gate_step=args.gate_step,
        project=args.project,
        entity=args.entity,
        tmux_prefix=args.tmux_prefix,
        codex_review_hook=args.codex_review_hook,
        codex_review_tmux_prefix=args.codex_review_tmux_prefix,
    )
    if tmux_has_session(launch.session):
        return launch
    if args.dry_run:
        append_log(log_path, f"dry_run would_launch run={run_id} session={launch.session}")
        return launch
    launch.output_root.mkdir(parents=True, exist_ok=True)
    launch.command_file.parent.mkdir(parents=True, exist_ok=True)
    launch.command_file.write_text(launch.shell_text, encoding="utf-8")
    launch.command_file.chmod(0o755)
    result = run_command(["tmux", "new-session", "-d", "-s", launch.session, "bash", str(launch.command_file)])
    if result.returncode != 0:
        append_log(log_path, f"launch_failed run={run_id} session={launch.session} stderr={result.stderr.strip()}")
        return launch
    append_log(
        log_path,
        (
            f"launched run={run_id} session={launch.session} start_step={launch.start_step} "
            f"output_root={launch.output_root} log={launch.watcher_log} command={launch.command_file}"
        ),
    )
    return launch


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    live_root = Path(args.live_root) if args.live_root else root / "outputs" / "live_periodic_reviews"
    supervisor_log = live_root / "live_sidecar_supervisor.log"
    append_log(supervisor_log, f"live Seq4096 analysis supervisor starting root={root} live_root={live_root}")
    while True:
        run_id = latest_active_run(tmux_sessions())
        if run_id:
            ensure_run_watcher(args, run_id, supervisor_log)
        else:
            append_log(supervisor_log, "no active toricgt_seq4096 training session detected")
        if args.once:
            break
        time.sleep(max(1.0, float(args.supervisor_poll_seconds)))


if __name__ == "__main__":
    main()
