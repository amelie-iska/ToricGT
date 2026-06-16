from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def test_prune_old_outputs_dry_run_keeps_files(tmp_path: Path) -> None:
    for idx in range(4):
        path = tmp_path / f"branching_reasoning_trajectory_{idx}"
        path.mkdir()
        (path / "artifact.txt").write_text(str(idx), encoding="utf-8")
        stamp = time.time() - (10 + idx) * 3600
        path.touch()
        (path / "artifact.txt").touch()
        import os

        os.utime(path, (stamp, stamp))

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/prune_old_outputs.py",
            "--output-root",
            str(tmp_path),
            "--pattern",
            "branching_reasoning_trajectory_*",
            "--keep-latest",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["dry_run"] is True
    assert payload["candidate_count"] == 2
    assert all((tmp_path / f"branching_reasoning_trajectory_{idx}").exists() for idx in range(4))
