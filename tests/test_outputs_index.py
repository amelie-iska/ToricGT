from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_outputs_index_links_branching_checkpoint_and_gudhi_artifacts(tmp_path: Path) -> None:
    latest = tmp_path / "latest_branching_reasoning_trajectory_report"
    latest.mkdir(parents=True)
    (latest / "branching_reasoning_trajectory.html").write_text("<html>latest</html>", encoding="utf-8")
    (latest / "manifest.json").write_text(
        json.dumps({"schema": "toricgt.branching_reasoning_visual_report.v1", "nodes": 128, "analogy_emitted": True}),
        encoding="utf-8",
    )
    (latest / "html_screenshots").mkdir()
    (latest / "html_screenshots" / "contact_sheet.png").write_bytes(b"png")

    checkpoint = tmp_path / "branching_reasoning_embedding_payload_20260616T000000Z"
    checkpoint.mkdir()
    (checkpoint / "branching_reasoning_trajectory.html").write_text("<html>checkpoint</html>", encoding="utf-8")
    (checkpoint / "manifest.json").write_text(
        json.dumps({"source_mode": "checkpoint_embedding_payload", "nodes": 64}),
        encoding="utf-8",
    )

    gudhi = tmp_path / "analysis" / "gudhi_persistence"
    gudhi.mkdir(parents=True)
    (gudhi / "index.html").write_text("<html>gudhi</html>", encoding="utf-8")
    (gudhi / "summary.json").write_text(json.dumps({"records": 2}), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/render_outputs_index.py",
            "--output-root",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "toricgt.outputs_index.v1"
    index = tmp_path / "index.html"
    html = index.read_text(encoding="utf-8")
    assert "Latest Branching Reasoning Simplex Report" in html
    assert "Checkpoint Branching" in html
    assert "GUDHI Persistence" in html
    assert "contact_sheet.png" in html
