"""Extract and mux audio from reference clips onto final output."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

from .video_io import ffmpeg_available

__all__ = ["extract_audio", "mux_audio_onto_video", "collect_segment_audio"]


def extract_audio(
    video_path: str | Path, out_path: str | Path, *, start_sec: float = 0.0, duration_sec: float = 0.0
) -> Optional[Path]:
    if not ffmpeg_available():
        return None
    src = Path(video_path)
    if not src.is_file():
        return None
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    if start_sec > 0:
        cmd.extend(["-ss", str(start_sec)])
    cmd.extend(["-i", str(src), "-vn", "-acodec", "aac"])
    if duration_sec > 0:
        cmd.extend(["-t", str(duration_sec)])
    cmd.append(str(out))
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return out if out.is_file() else None
    except subprocess.CalledProcessError:
        return None


def mux_audio_onto_video(video_path: str | Path, audio_path: str | Path, out_path: Optional[str | Path] = None) -> Path:
    vid = Path(video_path)
    aud = Path(audio_path)
    out = Path(out_path) if out_path else vid
    if not ffmpeg_available() or not aud.is_file():
        return vid
    tmp = vid.with_suffix(".mux.tmp.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(vid),
        "-i",
        str(aud),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(tmp),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    if out == vid:
        shutil.move(str(tmp), str(vid))
    else:
        shutil.move(str(tmp), str(out))
    return out


def collect_segment_audio(
    clip_paths: Sequence[str],
    work_dir: str | Path,
    *,
    durations: Optional[Sequence[float]] = None,
) -> Optional[Path]:
    """Concatenate per-segment audio beds into one track."""
    if not ffmpeg_available():
        return None
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    parts: List[Path] = []
    for i, cp in enumerate(clip_paths):
        if not cp or not Path(cp).is_file():
            continue
        dur = float(durations[i]) if durations and i < len(durations) else 0.0
        ap = wd / f"seg_{i:02d}.aac"
        if extract_audio(cp, ap, duration_sec=dur):
            parts.append(ap)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    list_file = wd / "audio_concat.txt"
    list_file.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8")
    merged = wd / "master_audio.aac"
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(merged)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return merged if merged.is_file() else None
    except subprocess.CalledProcessError:
        return None
