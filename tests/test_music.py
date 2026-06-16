from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

from toricgt.music import TorusMusicConfig, compose_torus_music, generate_wav
from toricgt.slepian_torus import dpss_basis, toric_slepian_audit


def test_slepian_basis_is_sorted_and_concentrated() -> None:
    eig, basis = dpss_basis(length=32, half_bandwidth=0.09, modes=5)

    assert eig.shape == (5,)
    assert basis.shape == (32, 5)
    assert all(float(eig[i]) >= float(eig[i + 1]) for i in range(len(eig) - 1))
    assert 0.0 <= float(eig[-1]) <= float(eig[0]) <= 1.0

    audit = toric_slepian_audit(
        phase_u=[(0.61803398875 * i) % 1.0 for i in range(32)],
        phase_v=[(0.41421356237 * i) % 1.0 for i in range(32)],
    )
    assert 0.0 <= float(audit["slepian_concentration"]) <= 1.0
    assert 0.0 <= float(audit["slepian_leakage"]) <= 1.0


def test_compose_torus_music_is_deterministic() -> None:
    config = TorusMusicConfig(seconds=2.0, bpm=120.0, sample_rate=8_000, seed=19)

    first = compose_torus_music(config)
    second = compose_torus_music(config)

    assert [event.midi for event in first] == [event.midi for event in second]
    assert len(first) > 4
    assert {event.expert for event in first}.issubset({0, 1, 2, 3})
    assert {event.active_face for event in first}.issubset({0, 1, 2, 3})
    assert all(0.0 <= event.slepian_weight <= 1.0 for event in first)


def test_generate_wav_and_metadata(tmp_path) -> None:
    output = tmp_path / "toric.wav"
    config = TorusMusicConfig(seconds=0.8, bpm=120.0, sample_rate=8_000, seed=31)

    wav_path, metadata_path, events = generate_wav(output, config)

    assert wav_path.exists()
    assert metadata_path.exists()
    assert events
    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getnchannels() == 2
        assert wav.getframerate() == 8_000
        assert wav.getnframes() > 0

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["config"]["theta"] == config.theta
    assert "slepian_concentration" in metadata["slepian_audit"]
    assert "slepian_weight" in metadata["events"][0]
    assert metadata["events"][0]["role"] in {"lead", "torus_pad"}


def test_inference_cli_exposes_optional_slepian_music_export() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/infer_tokengt_with_geometry.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    assert "--emit-slepian-music" in result.stdout
    assert "--slepian-music-output-dir" in result.stdout
    assert "--slepian-music-modes" in result.stdout
    assert "--emit-branching-reasoning-report" in result.stdout
    assert "--branching-reasoning-max-nodes" in result.stdout
    assert "--branching-reasoning-screenshots" in result.stdout
