from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_render_html_screenshots_cli(tmp_path: Path) -> None:
    try:
        import playwright  # noqa: F401
        import PIL  # noqa: F401
    except Exception:
        pytest.skip("Playwright and Pillow are required for screenshot rendering")

    source = tmp_path / "site"
    source.mkdir()
    (source / "index.html").write_text(
        "<!doctype html><html><body style='background:#030712;color:#e8fbff'><h1>ToricGT screenshot smoke</h1></body></html>",
        encoding="utf-8",
    )
    out = tmp_path / "screens"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/render_html_screenshots.py",
            "--source-dir",
            str(source),
            "--output-dir",
            str(out),
            "--wait-ms",
            "100",
            "--timeout-ms",
            "10000",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 1
    assert manifest["ok_count"] == 1
    assert manifest["source_index_linked"] is True
    assert (out / "contact_sheet.png").exists()
    assert (out / "index.html").exists()
    assert list((out / "screenshots").glob("*.png"))
    source_index = (source / "index.html").read_text(encoding="utf-8")
    assert "Rendered HTML Screenshot Audit" in source_index
    assert "contact_sheet.png" in source_index
