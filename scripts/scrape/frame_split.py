"""Extract training frames from animated GIFs and video posts.

Booru sites host mp4/webm/gif uploads alongside still images. SDX training
expects static RGB frames, so the scraper splits these into JPEGs under
``images/<parent_md5>_f000001.jpg`` and emits one manifest row per frame.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageSequence

_log = logging.getLogger(__name__)

VIDEO_EXTS = frozenset({"webm", "mp4", "mov", "mkv", "avi", "m4v"})
GIF_EXT = "gif"
ALL_SPLITTABLE_EXTS = VIDEO_EXTS | {GIF_EXT}


@dataclass
class ExtractedFrame:
    """One frame ready for the training manifest."""

    rel_path: str  # e.g. images/abc_f000001.jpg
    abs_path: Path
    md5: str
    width: int
    height: int
    frame_index: int


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def _looks_like_video_file(path: Path, *, min_bytes: int = 512) -> bool:
    """Reject empty downloads, HTML error pages, and other obvious non-video blobs."""
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < min_bytes:
        return False
    try:
        head = path.read_bytes()[:16]
    except OSError:
        return False
    if head.startswith((b"<!DOCTYPE", b"<html", b"<?xml", b"{")):
        return False
    return True


def _ffprobe_readable(path: Path) -> bool:
    if not ffprobe_available():
        return True
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and "video" in (proc.stdout or "").lower()


def normalize_ext(ext: str) -> str:
    return (ext or "").lower().lstrip(".")


def is_splittable_ext(ext: str) -> bool:
    return normalize_ext(ext) in ALL_SPLITTABLE_EXTS


def is_animated_gif(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            return int(getattr(im, "n_frames", 1) or 1) > 1
    except Exception:
        return False


def needs_frame_split(path: Path, ext: str) -> bool:
    """True when the downloaded file should be exploded into training frames."""
    ext = normalize_ext(ext)
    if ext in VIDEO_EXTS:
        return True
    if ext == GIF_EXT:
        return is_animated_gif(path)
    return False


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_rgb_jpeg(img: Image.Image, dest: Path, *, quality: int = 92) -> tuple[int, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rgb = img.convert("RGB")
    rgb.save(dest, "JPEG", quality=quality, optimize=True)
    return rgb.size


def _subsample_indices(total: int, max_frames: int) -> List[int]:
    if total <= 0:
        return []
    if max_frames <= 0 or total <= max_frames:
        return list(range(total))
    step = total / max_frames
    return sorted({min(total - 1, int(i * step)) for i in range(max_frames)})


def _extract_gif_frames(
    src: Path,
    out_dir: Path,
    parent_md5: str,
    *,
    max_frames: int,
    jpeg_quality: int,
) -> List[ExtractedFrame]:
    frames: List[ExtractedFrame] = []
    with Image.open(src) as im:
        total = int(getattr(im, "n_frames", 1) or 1)
        pick = _subsample_indices(total, max_frames)
        for out_i, src_i in enumerate(pick, start=1):
            im.seek(src_i)
            frame = im.copy()
            fname = f"{parent_md5}_f{out_i:06d}.jpg"
            dest = out_dir / fname
            if dest.is_file():
                w, h = Image.open(dest).size
            else:
                w, h = _save_rgb_jpeg(frame, dest, quality=jpeg_quality)
            frames.append(
                ExtractedFrame(
                    rel_path=str(Path("images") / fname),
                    abs_path=dest,
                    md5=_file_md5(dest),
                    width=w,
                    height=h,
                    frame_index=out_i,
                )
            )
    return frames


def _extract_video_frames_ffmpeg(
    src: Path,
    out_dir: Path,
    parent_md5: str,
    *,
    fps: float,
    max_frames: int,
) -> List[ExtractedFrame]:
    tmp_dir = out_dir / f".split_{parent_md5}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(tmp_dir / "frame_%06d.jpg")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    vf_parts = []
    if fps > 0:
        vf_parts.append(f"fps={fps}")
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])
    if max_frames > 0:
        cmd.extend(["-frames:v", str(int(max_frames))])
    cmd.extend(["-q:v", "2", pattern])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed on {src.name}: {err[:500]}")
    paths = sorted(tmp_dir.glob("frame_*.jpg"))
    if not paths:
        raise RuntimeError(f"ffmpeg produced no frames for {src.name}")
    frames: List[ExtractedFrame] = []
    for i, tmp_path in enumerate(paths, start=1):
        fname = f"{parent_md5}_f{i:06d}.jpg"
        dest = out_dir / fname
        if not dest.is_file():
            shutil.move(str(tmp_path), str(dest))
        else:
            tmp_path.unlink(missing_ok=True)
        w, h = Image.open(dest).size
        frames.append(
            ExtractedFrame(
                rel_path=str(Path("images") / fname),
                abs_path=dest,
                md5=_file_md5(dest),
                width=w,
                height=h,
                frame_index=i,
            )
        )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return frames


def _extract_video_frames_cv2(
    src: Path,
    out_dir: Path,
    parent_md5: str,
    *,
    fps: float,
    max_frames: int,
) -> List[ExtractedFrame]:
    import cv2

    # Force the FFMPEG backend; default backend order can mis-detect files as
    # image sequences and spam VIDEOIO(CV_IMAGES) pattern errors.
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video with cv2+ffmpeg backend: {src}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    step = 1
    if fps > 0 and src_fps > 0:
        step = max(1, int(round(src_fps / fps)))
    frames: List[ExtractedFrame] = []
    i = saved = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            saved += 1
            fname = f"{parent_md5}_f{saved:06d}.jpg"
            dest = out_dir / fname
            if not dest.is_file():
                cv2.imwrite(str(dest), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            frames.append(
                ExtractedFrame(
                    rel_path=str(Path("images") / fname),
                    abs_path=dest,
                    md5=_file_md5(dest),
                    width=w,
                    height=h,
                    frame_index=saved,
                )
            )
            if max_frames > 0 and saved >= max_frames:
                break
        i += 1
    cap.release()
    return frames


def existing_frames(out_dir: Path, parent_md5: str) -> List[ExtractedFrame]:
    """Return already-extracted frames for a parent post (resume)."""
    paths = sorted(out_dir.glob(f"{parent_md5}_f*.jpg"))
    frames: List[ExtractedFrame] = []
    for p in paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            w, h = 0, 0
        # frame_000001 -> index from suffix
        idx = 0
        try:
            idx = int(p.stem.rsplit("_f", 1)[-1])
        except (IndexError, ValueError):
            pass
        frames.append(
            ExtractedFrame(
                rel_path=str(Path("images") / p.name),
                abs_path=p,
                md5=_file_md5(p),
                width=w,
                height=h,
                frame_index=idx,
            )
        )
    return frames


def extract_training_frames(
    src: Path,
    images_dir: Path,
    parent_md5: str,
    ext: str,
    *,
    fps: float = 1.0,
    max_frames: int = 120,
    jpeg_quality: int = 92,
) -> List[ExtractedFrame]:
    """Split ``src`` into JPEG frames, or return a single-frame list for stills."""
    ext = normalize_ext(ext)
    existing = existing_frames(images_dir, parent_md5)
    if existing:
        return existing

    if not needs_frame_split(src, ext):
        return []

    if ext == GIF_EXT:
        return _extract_gif_frames(src, images_dir, parent_md5, max_frames=max_frames, jpeg_quality=jpeg_quality)

    if not _looks_like_video_file(src):
        _log.warning("skip frame split (not a video file): %s", src.name)
        return []

    if not _ffprobe_readable(src):
        _log.warning("skip frame split (ffprobe found no video stream): %s", src.name)
        return []

    if ffmpeg_available():
        try:
            return _extract_video_frames_ffmpeg(
                src, images_dir, parent_md5, fps=fps, max_frames=max_frames
            )
        except (RuntimeError, OSError) as exc:
            _log.warning("ffmpeg frame split failed for %s: %s", src.name, exc)

    try:
        return _extract_video_frames_cv2(src, images_dir, parent_md5, fps=fps, max_frames=max_frames)
    except Exception as exc:
        _log.warning("opencv frame split failed for %s: %s", src.name, exc)
        return []
