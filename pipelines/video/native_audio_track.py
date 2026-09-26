"""
Native audio track — Wan/Seedance/Hailuo co-processed A/V scaffold.

Full neural audio codecs need weights; this module delivers the production
contract competitors advertise:

1. Build a timed audio plan (dialogue / SFX / ambience beds) from the prompt.
2. Prefer voice refs from the multimodal bus when present.
3. Synthesize a lightweight stereo bed (numpy) when no ref audio exists.
4. Mux onto the final video via audio_mux.
5. Emit lip-sync energy curve for the lip_sync_driver.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "AudioPlan",
    "AudioEvent",
    "plan_native_audio",
    "synthesize_stereo_bed",
    "energy_envelope",
    "attach_native_audio",
]


@dataclass(slots=True)
class AudioEvent:
    kind: str  # dialogue|sfx|ambience|music
    start_sec: float
    duration_sec: float
    hint: str = ""
    ref_path: str = ""


@dataclass(slots=True)
class AudioPlan:
    duration_sec: float
    sample_rate: int = 32000
    events: list[AudioEvent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def plan_native_audio(
    prompt: str,
    *,
    duration_sec: float = 6.0,
    voice_refs: list[str] | None = None,
) -> AudioPlan:
    text = (prompt or "").lower()
    dur = max(1.0, float(duration_sec))
    events: list[AudioEvent] = []
    notes: list[str] = []

    # Ambience always
    ambience = "room tone"
    for key, label in (
        ("rain", "rain ambience"),
        ("ocean", "ocean waves"),
        ("city", "city traffic bed"),
        ("forest", "forest wind"),
        ("crowd", "crowd murmur"),
        ("office", "office hush"),
    ):
        if key in text:
            ambience = label
            break
    events.append(AudioEvent(kind="ambience", start_sec=0.0, duration_sec=dur, hint=ambience))

    # Dialogue if speech verbs / quotes
    if any(k in text for k in ("says", "said", "speak", "talk", "dialogue", "sing", '"', "'")):
        events.append(
            AudioEvent(
                kind="dialogue",
                start_sec=min(0.4, dur * 0.1),
                duration_sec=min(dur * 0.6, dur - 0.4),
                hint="spoken dialogue",
                ref_path=(voice_refs or [""])[0] if voice_refs else "",
            )
        )
        notes.append("dialogue_planned")

    # SFX hits from action verbs
    for verb, when in (
        ("door", 0.2),
        ("gunshot", 0.5),
        ("explosion", 0.55),
        ("footstep", 0.15),
        ("whoosh", 0.3),
        ("splash", 0.4),
    ):
        if verb in text:
            events.append(
                AudioEvent(
                    kind="sfx",
                    start_sec=min(dur * when, dur - 0.3),
                    duration_sec=0.35,
                    hint=verb,
                )
            )

    if voice_refs:
        notes.append(f"voice_refs={len(voice_refs)}")
    return AudioPlan(duration_sec=dur, events=events, notes=notes)


def _write_wav(path: Path, samples: np.ndarray, *, sr: int = 32000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # stereo float -> int16
    if samples.ndim == 1:
        samples = np.stack([samples, samples], axis=1)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def synthesize_stereo_bed(plan: AudioPlan, out_path: str | Path) -> Path:
    """Procedural stereo bed — not a neural codec, but a real attachable track."""
    sr = int(plan.sample_rate)
    n = int(sr * plan.duration_sec)
    t = np.arange(n, dtype=np.float64) / sr
    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)

    for ev in plan.events:
        i0 = int(ev.start_sec * sr)
        i1 = min(n, int((ev.start_sec + ev.duration_sec) * sr))
        if i1 <= i0:
            continue
        seg_t = t[i0:i1] - t[i0]
        if ev.kind == "ambience":
            # Soft noise bed + slow LFO
            rng = np.random.default_rng(abs(hash(ev.hint)) % (2**32))
            noise = rng.normal(0, 0.04, size=i1 - i0)
            lfo = 0.5 + 0.5 * np.sin(2 * math.pi * 0.15 * seg_t)
            left[i0:i1] += noise * lfo
            right[i0:i1] += noise * (1.0 - 0.3 * lfo)
        elif ev.kind == "dialogue":
            # Formant-ish buzz (placeholder voice)
            fund = 140.0 if "female" not in ev.hint else 220.0
            env = np.sin(np.pi * np.clip(seg_t / max(seg_t[-1], 1e-6), 0, 1)) ** 0.6
            sig = (
                0.12
                * env
                * (
                    np.sin(2 * math.pi * fund * seg_t)
                    + 0.4 * np.sin(2 * math.pi * fund * 2 * seg_t)
                    + 0.2 * np.sin(2 * math.pi * fund * 3 * seg_t)
                )
            )
            left[i0:i1] += sig
            right[i0:i1] += sig * 0.95
        elif ev.kind == "sfx":
            env = np.exp(-seg_t * 8.0)
            burst = 0.25 * env * np.sin(2 * math.pi * 400 * seg_t * (1 + seg_t))
            left[i0:i1] += burst
            right[i0:i1] += burst * 0.85
        elif ev.kind == "music":
            chord = sum(np.sin(2 * math.pi * f * seg_t) for f in (220.0, 277.0, 330.0)) / 3.0
            left[i0:i1] += 0.06 * chord
            right[i0:i1] += 0.06 * chord * 0.9

    peak = max(1e-6, float(np.max(np.abs(np.stack([left, right])))))
    left /= peak * 1.2
    right /= peak * 1.2
    return _write_wav(Path(out_path), np.stack([left, right], axis=1), sr=sr)


def energy_envelope(wav_path: str | Path, *, fps: float = 24.0, duration_sec: float = 0.0) -> np.ndarray:
    """Per-frame audio energy for lip-sync correlation."""
    path = Path(wav_path)
    if not path.is_file():
        return np.zeros(1, dtype=np.float32)
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        nframes = w.getnframes()
        raw = w.readframes(nframes)
        sw = w.getsampwidth()
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    dur = duration_sec if duration_sec > 0 else len(data) / max(sr, 1)
    n_frames = max(1, int(round(dur * fps)))
    win = max(1, len(data) // n_frames)
    env = []
    for i in range(n_frames):
        chunk = data[i * win : (i + 1) * win]
        env.append(float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0)
    arr = np.asarray(env, dtype=np.float32)
    if arr.max() > 1e-6:
        arr = arr / arr.max()
    return arr


def attach_native_audio(
    video_path: str | Path,
    plan: AudioPlan,
    work_dir: str | Path,
    *,
    prefer_ref: str | Path | None = None,
) -> Path:
    """Synthesize or reuse ref audio and mux onto video."""
    from .audio_mux import mux_audio_onto_video

    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    if prefer_ref and Path(prefer_ref).is_file():
        audio = Path(prefer_ref)
    else:
        audio = synthesize_stereo_bed(plan, wd / "native_audio.wav")
    return mux_audio_onto_video(video_path, audio, out_path=Path(video_path))
