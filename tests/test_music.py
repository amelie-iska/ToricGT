from __future__ import annotations

import json
import wave

from toricgt.music import TorusMusicConfig, compose_torus_music, generate_wav


def test_compose_torus_music_is_deterministic() -> None:
    config = TorusMusicConfig(seconds=2.0, bpm=120.0, sample_rate=8_000, seed=19)

    first = compose_torus_music(config)
    second = compose_torus_music(config)

    assert [event.midi for event in first] == [event.midi for event in second]
    assert len(first) > 4
    assert {event.expert for event in first}.issubset({0, 1, 2, 3})
    assert {event.active_face for event in first}.issubset({0, 1, 2, 3})


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
    assert metadata["events"][0]["role"] in {"lead", "torus_pad"}
