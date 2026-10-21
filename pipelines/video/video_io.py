"""Video I/O helpers: probe, extract frames, encode, retime.

Also holds what used to be: timeline, transition_fx, stitch, audio_mux, parallel_segments.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .types import TransitionType

# ==================================================================================================
# video_io
# ==================================================================================================


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe_video(path: str | Path) -> dict[str, Any]:
    """Return fps, width, height, duration_sec, frame_count (best effort)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    info: dict[str, Any] = {"path": str(p), "fps": 24.0, "width": 0, "height": 0, "duration_sec": 0.0, "frame_count": 0}
    if ffmpeg_available():
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(p),
        ]
        raw = subprocess.check_output(cmd, text=True)
        data = json.loads(raw)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                info["width"] = int(stream.get("width") or 0)
                info["height"] = int(stream.get("height") or 0)
                fr = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "24/1"
                if "/" in str(fr):
                    num, den = str(fr).split("/", 1)
                    info["fps"] = float(num) / max(1.0, float(den))
                else:
                    info["fps"] = float(fr)
                info["frame_count"] = int(stream.get("nb_frames") or 0)
                break
        fmt = data.get("format") or {}
        if fmt.get("duration"):
            info["duration_sec"] = float(fmt["duration"])
    if info["frame_count"] <= 0 and info["duration_sec"] > 0 and info["fps"] > 0:
        info["frame_count"] = int(info["duration_sec"] * info["fps"])
    return info


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    max_frames: int | None = None,
    fps: float | None = None,
    prefix: str = "frame",
) -> list[Path]:
    """Extract frames to ``out_dir/frame_000001.png`` via ffmpeg or OpenCV fallback."""
    p = Path(video_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if ffmpeg_available():
        pattern = str(out / f"{prefix}_%06d.png")
        cmd = ["ffmpeg", "-y", "-i", str(p)]
        if fps is not None and fps > 0:
            cmd.extend(["-vf", f"fps={fps}"])
        if max_frames is not None and max_frames > 0:
            cmd.extend(["-frames:v", str(int(max_frames))])
        cmd.append(pattern)
        subprocess.run(cmd, check=True, capture_output=True)
        paths = sorted(out.glob(f"{prefix}_*.png"))
        return paths
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("ffmpeg not found and cv2 unavailable") from exc
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {p}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = 1
    if fps is not None and fps > 0 and src_fps > 0:
        step = max(1, int(round(src_fps / fps)))
    paths: list[Path] = []
    i = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            fp = out / f"{prefix}_{saved + 1:06d}.png"
            cv2.imwrite(str(fp), frame)
            paths.append(fp)
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        i += 1
    cap.release()
    return paths


def read_frame_rgb(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_frame_rgb(path: str | Path, rgb: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def write_video_from_frames(
    frame_paths: list[str | Path],
    out_path: str | Path,
    *,
    fps: float = 24.0,
    audio_path: str | None = None,
) -> Path:
    """Encode PNG sequence to mp4."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frame_paths:
        raise ValueError("write_video_from_frames: no frames")
    if ffmpeg_available():
        list_file = out.with_suffix(".frames.txt")
        lines = [f"file '{Path(fp).resolve().as_posix()}'\n" for fp in frame_paths]
        list_file.write_text("".join(lines), encoding="utf-8")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-r",
            str(float(fps)),
            "-i",
            str(list_file),
            "-pix_fmt",
            "yuv420p",
        ]
        if audio_path and Path(audio_path).is_file():
            cmd.extend(["-i", str(audio_path), "-c:a", "aac", "-shortest"])
        cmd.append(str(out))
        subprocess.run(cmd, check=True, capture_output=True)
        list_file.unlink(missing_ok=True)
        return out
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("ffmpeg required or install opencv-python") from exc
    first = read_frame_rgb(frame_paths[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (w, h))
    for fp in frame_paths:
        rgb = read_frame_rgb(fp)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if bgr.shape[1] != w or bgr.shape[0] != h:
            bgr = cv2.resize(bgr, (w, h))
        writer.write(bgr)
    writer.release()
    return out


# ==================================================================================================
# timeline: Master timeline: normalize fps, retime segments, transitions.
# ==================================================================================================


def frame_count_for_duration(duration_sec: float, fps: float) -> int:
    return max(1, int(round(float(duration_sec) * float(fps))))


def normalize_segment_window(
    clip_duration_sec: float,
    target_duration_sec: float,
    *,
    in_sec: float = 0.0,
) -> tuple[float, float, float]:
    """
    Pick in/out window inside clip and speed factor to hit target duration.

    Returns (in_sec, out_sec, speed_factor).
    """
    avail = max(0.01, clip_duration_sec - in_sec)
    need = max(0.01, target_duration_sec)
    if avail >= need:
        return in_sec, in_sec + need, 1.0
    speed = avail / need
    return in_sec, in_sec + avail, speed


# ==================================================================================================
# transition_fx: Extended transition effects between segments.
# ==================================================================================================


def transition_overlap_frames_fx(transition: TransitionType, fps: float) -> int:
    t = transition.value if hasattr(transition, "value") else str(transition)
    if t == TransitionType.DISSOLVE.value:
        return max(2, int(round(fps * 0.35)))
    if t == TransitionType.MATCH_ACTION.value:
        return max(1, int(round(fps * 0.12)))
    if t == TransitionType.WHIP.value:
        return max(2, int(round(fps * 0.18)))
    if t == TransitionType.FLASH.value:
        return max(1, int(round(fps * 0.08)))
    if t == TransitionType.DIP.value:
        return max(2, int(round(fps * 0.25)))
    return 0


def _crossfade(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return np.clip((1.0 - t) * a.astype(np.float32) + t * b.astype(np.float32), 0, 255).astype(np.uint8)


def _whip_blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    try:
        import cv2

        shift = int((t - 0.5) * a.shape[1] * 0.35)
        m = np.float32([[1, 0, shift], [0, 1, 0]])
        wa = cv2.warpAffine(a, m, (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.warpAffine(b, m, (b.shape[1], b.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        return _crossfade(wa, wb, t)
    except ImportError:
        return _crossfade(a, b, t)


def _flash_blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    peak = max(0.0, 1.0 - abs(t - 0.5) * 4.0)
    base = _crossfade(a, b, t)
    flash = np.full_like(base, 255)
    return np.clip(base.astype(np.float32) * (1.0 - peak * 0.55) + flash * (peak * 0.55), 0, 255).astype(np.uint8)


def _dip_blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    if t < 0.5:
        fade = t * 2.0
        return np.clip(a.astype(np.float32) * (1.0 - fade), 0, 255).astype(np.uint8)
    fade = (t - 0.5) * 2.0
    return np.clip(b.astype(np.float32) * fade, 0, 255).astype(np.uint8)


def apply_transition(
    frames_a: Sequence[np.ndarray],
    frames_b: Sequence[np.ndarray],
    transition: TransitionType,
    overlap: int,
) -> list[np.ndarray]:
    if overlap <= 0 or not frames_a or not frames_b:
        return list(frames_a) + list(frames_b)
    overlap = min(overlap, len(frames_a), len(frames_b))
    head = list(frames_a[:-overlap])
    tail = list(frames_b[overlap:])
    blend: list[np.ndarray] = []
    tname = transition.value if hasattr(transition, "value") else str(transition)
    for i in range(overlap):
        t = (i + 1) / (overlap + 1)
        a = frames_a[-overlap + i]
        b = frames_b[i]
        if tname == TransitionType.WHIP.value:
            blend.append(_whip_blend(a, b, t))
        elif tname == TransitionType.FLASH.value:
            blend.append(_flash_blend(a, b, t))
        elif tname == TransitionType.DIP.value:
            blend.append(_dip_blend(a, b, t))
        else:
            blend.append(_crossfade(a, b, t))
    return head + blend + tail


# ==================================================================================================
# stitch: Stitch segments with transitions onto master timeline.
# ==================================================================================================


def stitch_frame_lists(
    segment_frames: list[list[Path]],
    out_path: str | Path,
    *,
    fps: float = 24.0,
    transitions: list[TransitionType] | None = None,
) -> Path:
    transitions = transitions or [TransitionType.CUT] * len(segment_frames)
    merged: list[np.ndarray] = []
    for i, paths in enumerate(segment_frames):
        frames = [read_frame_rgb(p) for p in paths]
        if i == 0:
            merged = frames
            continue
        tr = transitions[i] if i < len(transitions) else TransitionType.CUT
        overlap = transition_overlap_frames_fx(tr, fps) if tr != TransitionType.CUT else 0
        merged = apply_transition(merged, frames, tr, overlap)
    out = Path(out_path)
    tmp = out.parent / f"{out.stem}_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    fpsaths: list[Path] = []
    for j, rgb in enumerate(merged):
        fp = tmp / f"final_{j + 1:06d}.png"
        save_frame_rgb(fp, rgb)
        fpsaths.append(fp)
    return write_video_from_frames(fpsaths, out, fps=fps)


# ==================================================================================================
# audio_mux: Extract and mux audio from reference clips onto final output.
# ==================================================================================================


def extract_audio(
    video_path: str | Path, out_path: str | Path, *, start_sec: float = 0.0, duration_sec: float = 0.0
) -> Path | None:
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


def mux_audio_onto_video(video_path: str | Path, audio_path: str | Path, out_path: str | Path | None = None) -> Path:
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
    durations: Sequence[float] | None = None,
) -> Path | None:
    """Concatenate per-segment audio beds into one track."""
    if not ffmpeg_available():
        return None
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
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


# ==================================================================================================
# parallel_segments: Parallel segment processing for multi-core / multi-GPU pipelines.
# ==================================================================================================


@dataclass(slots=True)
class SegmentWorkItem:
    index: int
    seg_dir: Path


@dataclass(slots=True)
class SegmentWorkResult:
    index: int
    frames: list[Path]
    quality: Any


def run_segments_parallel(
    items: Sequence[SegmentWorkItem],
    worker: Callable[[SegmentWorkItem], tuple[list[Path], Any]],
    *,
    max_workers: int = 2,
) -> list[SegmentWorkResult]:
    """Run independent segment jobs; results sorted by segment index."""
    if max_workers <= 1 or len(items) <= 1:
        out = [SegmentWorkResult(i.index, *worker(i)) for i in items]
        return sorted(out, key=lambda r: r.index)

    results: list[SegmentWorkResult] = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futs = {pool.submit(worker, item): item for item in items}
        for fut in as_completed(futs):
            item = futs[fut]
            frames, quality = fut.result()
            results.append(SegmentWorkResult(item.index, frames, quality))
    return sorted(results, key=lambda r: r.index)


__all__ = [
    "extract_frames",
    "ffmpeg_available",
    "probe_video",
    "read_frame_rgb",
    "save_frame_rgb",
    "write_video_from_frames",
    "frame_count_for_duration",
    "normalize_segment_window",
    "apply_transition",
    "transition_overlap_frames_fx",
    "stitch_frame_lists",
    "extract_audio",
    "mux_audio_onto_video",
    "collect_segment_audio",
    "SegmentWorkItem",
    "SegmentWorkResult",
    "run_segments_parallel",
]
