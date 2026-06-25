"""Video I/O helpers: probe, extract frames, encode, retime."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

__all__ = [
    "extract_frames",
    "ffmpeg_available",
    "load_frame_paths",
    "probe_video",
    "read_frame_rgb",
    "save_frame_rgb",
    "write_video_from_frames",
]


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe_video(path: str | Path) -> Dict[str, Any]:
    """Return fps, width, height, duration_sec, frame_count (best effort)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    info: Dict[str, Any] = {"path": str(p), "fps": 24.0, "width": 0, "height": 0, "duration_sec": 0.0, "frame_count": 0}
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
    max_frames: Optional[int] = None,
    fps: Optional[float] = None,
    prefix: str = "frame",
) -> List[Path]:
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
    paths: List[Path] = []
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


def load_frame_paths(dir_path: str | Path, *, pattern: str = "*.png") -> List[Path]:
    return sorted(Path(dir_path).glob(pattern))


def write_video_from_frames(
    frame_paths: List[str | Path],
    out_path: str | Path,
    *,
    fps: float = 24.0,
    audio_path: Optional[str] = None,
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
