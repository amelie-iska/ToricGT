from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_embedding_cas_sidecar_runs_sage_and_macaulay2(tmp_path: Path) -> None:
    if shutil.which("sage") is None or shutil.which("M2") is None:
        pytest.skip("SageMath and Macaulay2 are required for the embedding CAS sidecar")

    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    npz = embeddings / "record_000_embedding_payload.npz"
    hidden = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.4],
            [1.0, 1.0, 0.8],
            [0.4, 0.7, 1.0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        npz,
        hidden=hidden,
        projected=hidden[:, :3],
        energy=np.zeros((hidden.shape[0],), dtype=np.float32),
        nll=np.zeros((hidden.shape[0],), dtype=np.float32),
        edges=np.asarray([[0, 1], [1, 3], [2, 3]], dtype=np.int64),
        complex_projected=hidden[:, :3],
        complex_nll=np.zeros((hidden.shape[0],), dtype=np.float32),
        complex_mass=np.ones((hidden.shape[0],), dtype=np.float32),
        complex_edges=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
    )
    manifest = {
        "schema": "toricgt.embedding_payload_manifest.v1",
        "records": 1,
        "payloads": [
            {
                "record_index": 0,
                "npz": str(npz),
                "relative_npz": "embeddings/record_000_embedding_payload.npz",
            }
        ],
    }
    manifest_path = embeddings / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    out = tmp_path / "cas"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_embedding_cas_sidecar.py",
            "--embedding-manifest",
            str(manifest_path),
            "--output-dir",
            str(out),
            "--records",
            "1",
            "--max-points",
            "5",
            "--exponent-dim",
            "2",
            "--quantization-scale",
            "6",
            "--sage-timeout-seconds",
            "120",
            "--macaulay2-timeout-seconds",
            "120",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    summary = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert summary["records"] == 1
    assert summary["sage_backend"]["available"] is True
    assert summary["macaulay2_backend"]["available"] is True
    record_path = out / summary["record_pages"][0]["json"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["sage_normal_fan"]["provenance"] == "exact_cas/sage"
    assert record["macaulay2_toric_ideal"]["provenance"] == "exact_cas/macaulay2"
    assert record["exponent_metadata"]["method"] == "actual_hidden_pca_rank_quantized_nonnegative_exponents"
    assert (out / "index.html").exists()
