from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_codex_review_loop_ignores_stale_target_bpb_metric(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "codex_training_review_resume.sh"
    analysis_dir = tmp_path / "analysis" / "step-00004000"
    metrics_dir = analysis_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metric_stats.json").write_text(
        json.dumps(
            [
                {"metric": "fineweb/target_bpb", "last_value": 1.2},
                {"metric": "fineweb/val_bpb", "last_value": 1.23},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (metrics_dir / "wandb_history.csv").write_text(
        "fineweb/target_bpb,fineweb/val_bpb\n1.2,1.23\n",
        encoding="utf-8",
    )
    state_path = tmp_path / "loop_state.json"
    state_path.write_text(
        json.dumps(
            {
                "best_bpb": 1.2,
                "best_bpb_metric": "fineweb/target_bpb",
                "best_bpb_step": 3750,
                "iteration_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "BPB_LOOP_STATE": str(state_path),
            "BPB_LOOP_STOP_FILE": str(tmp_path / "stop"),
            "BPB_TARGET": "1.2",
            "BPB_MAX_REVIEW_ITERATIONS": "10",
        }
    )

    result = subprocess.run(
        [
            str(script),
            "--analysis-dir",
            str(analysis_dir),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--step",
            "4000",
            "--run-path",
            "entity/project/run-id",
            "--dry-run",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    status = json.loads((analysis_dir / "logs" / "bpb_codex_loop_status.json").read_text(encoding="utf-8"))
    assert status["target_reached"] is False
    assert status["best_bpb"] == 1.23
    assert status["best_bpb_metric"] == "fineweb/val_bpb"
