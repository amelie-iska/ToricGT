"""Algorithmic music from ToricGT torus constraints.

The generator is intentionally lightweight: it uses only Python's standard
library so it can run from the existing Conda environment without an audio
stack beyond a system WAV player.  The composition rule mirrors the model's
research objects: irrational torus orbits propose graph states, a tropical
argmax chooses active faces, a four-slot router selects timbres, and the
result is rendered as a small stereo WAV.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import struct
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


GOLDEN_ROTATION = (math.sqrt(5.0) - 1.0) / 2.0
SILVER_ROTATION = math.sqrt(2.0) - 1.0


@dataclass(frozen=True)
class TorusMusicConfig:
    """Configuration for a deterministic toric music render."""

    seconds: float = 48.0
    bpm: float = 84.0
    sample_rate: int = 44_100
    theta: float = GOLDEN_ROTATION
    beta: float = SILVER_ROTATION
    seed: int = 1729
    root_midi: int = 36
    amplitude: float = 0.78
    swing: float = 0.0
    pad_every: int = 8
    mood: str = "dark_analog"


@dataclass(frozen=True)
class TorusNoteEvent:
    """A single rendered note plus its algebraic provenance."""

    start: float
    duration: float
    midi: int
    velocity: float
    pan: float
    expert: int
    active_face: int
    torus_x: int
    torus_y: int
    phase_u: float
    phase_v: float
    role: str


def _wrap01(x: float) -> float:
    return x - math.floor(x)


def _torus_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    du = abs(a[0] - b[0])
    dv = abs(a[1] - b[1])
    du = min(du, 1.0 - du)
    dv = min(dv, 1.0 - dv)
    return math.sqrt(du * du + dv * dv)


def _pitch_class(x: int, y: int, mood: str = "dark_analog") -> int:
    # Circle-of-fifths and major-third coordinates on a Tonnetz-like torus.
    raw = (7 * x + 4 * y) % 12
    if mood == "dark_analog":
        dark_minor = [0, 2, 3, 5, 7, 8, 10]
        return min(dark_minor, key=lambda pc: min((raw - pc) % 12, (pc - raw) % 12))
    return raw


def _nearest_midi(pitch_class: int, previous: int, root_midi: int) -> int:
    candidates = [root_midi + pitch_class + 12 * octave for octave in range(-2, 5)]
    return min(candidates, key=lambda note: (abs(note - previous), abs(note - 60)))


def _router_expert(u: float, v: float) -> tuple[int, list[float]]:
    centers = [(0.18, 0.22), (0.76, 0.22), (0.25, 0.78), (0.78, 0.72)]
    logits = [-11.0 * _torus_distance((u, v), center) for center in centers]
    peak = max(logits)
    weights = [math.exp(logit - peak) for logit in logits]
    total = sum(weights)
    weights = [weight / total for weight in weights]
    return max(range(4), key=weights.__getitem__), weights


def compose_torus_music(config: TorusMusicConfig) -> list[TorusNoteEvent]:
    """Create note events from an irrational torus walk and tropical choices."""

    if config.seconds <= 0:
        raise ValueError("seconds must be positive")
    if config.bpm <= 0:
        raise ValueError("bpm must be positive")
    if config.sample_rate < 8_000:
        raise ValueError("sample_rate must be at least 8000")

    rng = random.Random(config.seed)
    beat = 60.0 / config.bpm
    step = beat / 2.0
    n_steps = max(1, int(config.seconds / step))
    previous_midi = 60
    events: list[TorusNoteEvent] = []

    # The path is a finite shadow of an irrational rotation on T^2.
    u = _wrap01(rng.random() * 0.17 + config.theta)
    v = _wrap01(rng.random() * 0.23 + config.beta)
    previous_xy = (0, 0)

    for t in range(n_steps):
        u = _wrap01(u + config.theta + 0.015 * math.sin(2.0 * math.pi * v))
        v = _wrap01(v + config.beta + 0.012 * math.sin(2.0 * math.pi * u))
        expert, weights = _router_expert(u, v)

        x = int(math.floor(12 * u)) % 12
        y = int(math.floor(7 * v)) % 7
        target = (_wrap01(u + config.theta), _wrap01(v + config.beta))
        candidates = [
            (x, y),
            ((x + 1) % 12, y),
            (x, (y + 1) % 7),
            ((x + 1) % 12, (y + 1) % 7),
        ]

        best_face = 0
        best_score = -float("inf")
        for face, (cx, cy) in enumerate(candidates):
            point = ((cx + 0.5) / 12.0, (cy + 0.5) / 7.0)
            pitch = _pitch_class(cx, cy, config.mood)
            consonance = 0.20 if pitch in {0, 3, 5, 7, 10} else -0.06
            cocycle = math.sin(2.0 * math.pi * config.theta * (previous_xy[0] * cy - previous_xy[1] * cx))
            router_bias = 0.10 * weights[face % 4]
            score = -_torus_distance(point, target) + consonance + 0.07 * cocycle + router_bias
            if score > best_score:
                best_score = score
                best_face = face

        active_x, active_y = candidates[best_face]
        pitch_class = _pitch_class(active_x, active_y, config.mood)
        midi = _nearest_midi(pitch_class, previous_midi, config.root_midi)
        # Keep the lead line in a tense mid-register and let bass own the lows.
        while midi < 55:
            midi += 12
        while midi > 76:
            midi -= 12
        previous_midi = midi
        previous_xy = (active_x, active_y)

        swing_offset = config.swing * step if t % 2 else 0.0
        duration = step * (0.92 if config.mood == "dark_analog" else 1.35)
        velocity = 0.26 + 0.26 * weights[expert] + 0.04 * math.sin(2.0 * math.pi * u)
        pan = max(-0.85, min(0.85, 2.0 * u - 1.0))
        events.append(
            TorusNoteEvent(
                start=t * step + swing_offset,
                duration=duration,
                midi=midi,
                velocity=velocity,
                pan=pan,
                expert=expert,
                active_face=best_face,
                torus_x=active_x,
                torus_y=active_y,
                phase_u=u,
                phase_v=v,
                role="lead",
            )
        )

        if config.mood == "dark_analog":
            # A strict low arpeggiator supplies the dark analog-synth pulse.
            pulse_pattern = [0, 12, 15, 19, 24, 19, 15, 12]
            pulse_midi = config.root_midi + pulse_pattern[t % len(pulse_pattern)]
            if t % 16 in {7, 15}:
                pulse_midi += 2 if best_face in {1, 3} else -2
            events.append(
                TorusNoteEvent(
                    start=t * step + 0.004 * math.sin(2.0 * math.pi * u),
                    duration=step * 0.74,
                    midi=pulse_midi,
                    velocity=0.38 + 0.10 * weights[expert],
                    pan=-0.12 + 0.24 * ((t % 4) / 3.0),
                    expert=expert,
                    active_face=best_face,
                    torus_x=active_x,
                    torus_y=active_y,
                    phase_u=u,
                    phase_v=v,
                    role="pulse_bass",
                )
            )

        if config.pad_every > 0 and t % config.pad_every == 0:
            pad_pitch_classes = [
                _pitch_class(active_x, active_y, config.mood),
                _pitch_class((active_x + 1) % 12, active_y, config.mood),
                _pitch_class(active_x, (active_y + 1) % 7, config.mood),
            ]
            for j, pad_pc in enumerate(pad_pitch_classes):
                pad_midi = _nearest_midi(pad_pc, 48 + 7 * j, config.root_midi - 12)
                while pad_midi > 60:
                    pad_midi -= 12
                events.append(
                    TorusNoteEvent(
                        start=t * step + 0.015 * j,
                        duration=step * config.pad_every * 2.7,
                        midi=pad_midi - (12 if j == 0 else 0),
                        velocity=0.08 + 0.035 * j,
                        pan=-0.45 + 0.45 * j,
                        expert=expert,
                        active_face=best_face,
                        torus_x=active_x,
                        torus_y=active_y,
                        phase_u=u,
                        phase_v=v,
                        role="torus_pad",
                    )
                )

    return events


def _midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _envelope(t: float, duration: float, role: str) -> float:
    if role == "pulse_bass":
        attack = min(0.010, duration * 0.08)
        release = min(0.055, duration * 0.22)
    elif role == "torus_pad":
        attack = min(0.75, duration * 0.28)
        release = min(1.40, duration * 0.45)
    else:
        attack = min(0.018, duration * 0.12)
        release = min(0.11, duration * 0.28)
    if t < attack:
        return t / max(attack, 1e-6)
    if t > duration - release:
        return max(0.0, (duration - t) / max(release, 1e-6))
    if role == "pulse_bass":
        return 0.70 + 0.30 * math.exp(-2.4 * t / max(duration, 1e-6))
    if role == "torus_pad":
        return 0.74 + 0.12 * math.sin(math.pi * t / max(duration, 1e-6))
    return 0.76 + 0.18 * math.sin(math.pi * t / max(duration, 1e-6))


def render_torus_music(config: TorusMusicConfig, events: Iterable[TorusNoteEvent]) -> list[tuple[float, float]]:
    """Render note events to a stereo buffer."""

    tail = 1.0
    n_frames = int((config.seconds + tail) * config.sample_rate)
    buffer = [[0.0, 0.0] for _ in range(n_frames)]
    two_pi = 2.0 * math.pi

    for event in events:
        start = max(0, int(event.start * config.sample_rate))
        stop = min(n_frames, int((event.start + event.duration) * config.sample_rate))
        if stop <= start:
            continue
        frequency = _midi_to_hz(event.midi)
        left_gain = math.sqrt(0.5 * (1.0 - event.pan))
        right_gain = math.sqrt(0.5 * (1.0 + event.pan))
        timbre = 1.0 + 0.09 * event.expert
        phase_offset = two_pi * event.phase_u
        for idx in range(start, stop):
            local_t = (idx - start) / config.sample_rate
            env = _envelope(local_t, event.duration, event.role)
            phase = two_pi * frequency * local_t + phase_offset
            fm = 0.018 * math.sin(two_pi * (frequency * config.theta / 4.0) * local_t + two_pi * event.phase_v)
            if event.role == "pulse_bass":
                detune = 1.006 + 0.001 * event.expert
                saw = 2.0 * ((frequency * local_t + event.phase_u) % 1.0) - 1.0
                saw_detuned = 2.0 * ((frequency * detune * local_t + event.phase_v) % 1.0) - 1.0
                sub = math.sin(0.5 * phase)
                gate = 0.74 + 0.26 * math.sin(two_pi * 2.0 * local_t)
                harmonic = 0.50 * saw + 0.35 * saw_detuned + 0.45 * sub
                sample = event.velocity * env * gate * harmonic / 1.30
            elif event.role == "torus_pad":
                slow = math.sin(two_pi * 0.085 * local_t + two_pi * event.phase_v)
                harmonic = (
                    0.70 * math.sin(phase + fm)
                    + 0.42 * math.sin(phase * 1.003 + 0.8 + slow * 0.12)
                    + 0.18 * math.sin(2.0 * phase + timbre)
                )
                sample = event.velocity * env * harmonic / 1.55
            else:
                pulse = 0.78 + 0.22 * math.sin(two_pi * 4.0 * local_t + two_pi * event.phase_v)
                harmonic = (
                    0.72 * math.sin(phase + fm)
                    + 0.34 * math.sin(2.0 * phase + timbre)
                    + 0.10 * math.sin(3.0 * phase + two_pi * config.beta)
                )
                sample = event.velocity * env * pulse * harmonic / 1.35
            if event.role == "torus_pad":
                sample *= 0.58
            buffer[idx][0] += sample * left_gain
            buffer[idx][1] += sample * right_gain

    # Small cross-channel delay: enough spatial structure to hear the orbit.
    delay = int(0.23 * config.sample_rate)
    for idx in range(delay, n_frames):
        buffer[idx][0] += 0.12 * buffer[idx - delay][1]
        buffer[idx][1] += 0.12 * buffer[idx - delay][0]

    # Gentle high-frequency damping gives the PCM synth a darker tone without
    # requiring scipy or an external filter implementation.
    prev_l = 0.0
    prev_r = 0.0
    alpha = 0.18
    for idx, (left, right) in enumerate(buffer):
        prev_l = prev_l + alpha * (left - prev_l)
        prev_r = prev_r + alpha * (right - prev_r)
        buffer[idx][0] = prev_l
        buffer[idx][1] = prev_r

    peak = max((max(abs(l), abs(r)) for l, r in buffer), default=1.0)
    scale = config.amplitude / max(peak, 1e-9)
    return [(l * scale, r * scale) for l, r in buffer]


def write_wav(path: Path, config: TorusMusicConfig, samples: Iterable[tuple[float, float]]) -> None:
    """Write stereo 16-bit PCM WAV audio."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(config.sample_rate)
        frames = bytearray()
        for left, right in samples:
            left_i = int(max(-1.0, min(1.0, left)) * 32767)
            right_i = int(max(-1.0, min(1.0, right)) * 32767)
            frames.extend(struct.pack("<hh", left_i, right_i))
        wav.writeframes(frames)


def write_metadata(path: Path, config: TorusMusicConfig, events: list[TorusNoteEvent]) -> None:
    """Write JSON provenance for the rendered music."""

    payload = {
        "model": "ToricGT torus-constrained algorithmic music",
        "description": (
            "Irrational T^2 orbit with tropical active-face selection, "
            "noncommutative-torus cocycle bias, and four-expert Soft-MoE-style routing."
        ),
        "config": asdict(config),
        "events": [asdict(event) for event in events],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_wav(output: Path, config: TorusMusicConfig) -> tuple[Path, Path, list[TorusNoteEvent]]:
    """Generate a WAV plus sidecar metadata."""

    events = compose_torus_music(config)
    samples = render_torus_music(config, events)
    write_wav(output, config, samples)
    metadata_path = output.with_suffix(".json")
    write_metadata(metadata_path, config, events)
    return output, metadata_path, events


def play_wav(path: Path) -> str | None:
    """Play a WAV file with the first available local player."""

    players = [
        ("aplay", ["aplay", "-q", str(path)]),
        ("paplay", ["paplay", str(path)]),
        ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)]),
        ("afplay", ["afplay", str(path)]),
        ("play", ["play", "-q", str(path)]),
    ]
    failure_markers = ("failed", "error", "no such device", "host is down", "no more combinations")
    for name, command in players:
        if shutil.which(name):
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            output = f"{result.stdout}\n{result.stderr}".lower()
            if result.returncode == 0 and not any(marker in output for marker in failure_markers):
                return name
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate torus-constrained ToricGT algorithmic music.")
    parser.add_argument("--output", type=Path, default=Path("outputs/music/toricgt_torus_music.wav"))
    parser.add_argument("--seconds", type=float, default=48.0)
    parser.add_argument("--bpm", type=float, default=84.0)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--theta", type=float, default=GOLDEN_ROTATION)
    parser.add_argument("--beta", type=float, default=SILVER_ROTATION)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--root-midi", type=int, default=36)
    parser.add_argument("--amplitude", type=float, default=0.78)
    parser.add_argument("--mood", choices=["dark_analog", "toric"], default="dark_analog")
    parser.add_argument("--play", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TorusMusicConfig(
        seconds=args.seconds,
        bpm=args.bpm,
        sample_rate=args.sample_rate,
        theta=args.theta,
        beta=args.beta,
        seed=args.seed,
        root_midi=args.root_midi,
        amplitude=args.amplitude,
        mood=args.mood,
    )
    output, metadata, events = generate_wav(args.output, config)
    print(f"Wrote {output}")
    print(f"Wrote {metadata}")
    print(f"Rendered {len(events)} toric note events")
    if args.play:
        if os.environ.get("CI"):
            print("CI is set; skipping playback")
            return
        player = play_wav(output)
        if player:
            print(f"Played with {player}")
        else:
            print("No local WAV player found; generated file is ready to play manually")


if __name__ == "__main__":
    main()
