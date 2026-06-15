from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_toricgt_status_is_read_only_json() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/toricgt_status.py", "--repo", ".", "--json"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "toricgt.status.v1"
    assert payload["read_only"] is True
    assert isinstance(payload["processes"], list)
    assert isinstance(payload["tmux_sessions"], list)
