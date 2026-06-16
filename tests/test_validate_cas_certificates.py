from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_validate_cas_certificates_preflight_schema() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/validate_cas_certificates.py",
            "--preflight",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "toricgt.cas_toolchain_preflight_report.v1"
    assert "backends" in payload
    assert "tools" in payload
    assert "python_packages" in payload
    assert "gudhi" in payload["python_packages"]
    assert "gfan" in payload["tools"]
    assert "normaliz" in payload["tools"]
    assert "macaulay2" in payload["backends"]
    assert "sage" in payload["backends"]
