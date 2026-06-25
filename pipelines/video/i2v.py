"""Image-to-video: anchor frame 0, borrow motion from reference clip."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .shot_planner import plan_video_from_prompt
from .types import VideoMode, VideoPlan
from .video_io import save_frame_rgb

__all__ = ["build_i2v_plan", "prepare_anchor_frame"]


def prepare_anchor_frame(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    width: int,
    height: int,
) -> Path:
    """Resize/center-crop user image to timeline resolution."""
    from PIL import Image

    src = Path(image_path)
    out = Path(out_dir) / "anchor_frame.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    w, h = img.size
    target_ar = width / max(1, height)
    src_ar = w / max(1, h)
    if src_ar > target_ar:
        new_w = int(h * target_ar)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ar)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    save_frame_rgb(out, __import__("numpy").array(img))
    return out


def build_i2v_plan(
    prompt: str,
    anchor_image: str | Path,
    *,
    duration_sec: float = 4.0,
    fps: float = 24.0,
    width: int = 1280,
    height: int = 720,
    reference_clips: Optional[Sequence[str]] = None,
) -> VideoPlan:
    plan = plan_video_from_prompt(
        prompt,
        mode=VideoMode.I2V,
        duration_sec=duration_sec,
        fps=fps,
        width=width,
        height=height,
        anchor_image_note="identity locked from anchor image, consistent subject",
    )
    plan.metadata["anchor_image"] = str(Path(anchor_image).resolve())
    if reference_clips:
        plan.metadata["reference_clips"] = [str(Path(p).resolve()) for p in reference_clips]
    return plan
