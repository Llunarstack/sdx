"""
Part compositing utilities.

Takes the output of `utils.generation.image_dissection.dissect_images_to_parts`
and builds:

- an init image (RGB) to seed img2img
- an inpaint mask (L): white=inpaint, black=keep

This lets sampling preserve user-requested regions (e.g. "use the hat from ref1")
while the diffusion model regenerates the rest to match the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from utils.generation.image_dissection import DissectedPart


@dataclass(frozen=True, slots=True)
class CompositeSpec:
    init_image_path: str
    mask_path: str
    preserved_parts: int
    background_locked: bool


def _load_rgb(path: Union[str, Path], *, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def _load_mask(path: Union[str, Path], *, size: Tuple[int, int]) -> Image.Image:
    m = Image.open(path).convert("L")
    if m.size != size:
        m = m.resize(size, Image.Resampling.NEAREST)
    return m


def build_init_and_inpaint_mask(
    *,
    reference_images: Sequence[Union[str, Path]],
    parts: Sequence[DissectedPart],
    output_dir: Union[str, Path],
    target_size: Tuple[int, int],
    lock_background_if_requested: bool = True,
) -> CompositeSpec:
    """
    Compose an init image and an inpaint mask from dissection outputs.

    Mask semantics follow `sample.py`: white=inpaint, black=keep.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    W, H = int(target_size[0]), int(target_size[1])
    size = (W, H)

    # Choose background base.
    bg_part = next((p for p in parts if p.request.role == "background" or p.request.part.lower() == "background"), None)
    if bg_part is not None and bg_part.crop_path:
        base = _load_rgb(bg_part.crop_path, size=size)
        background_locked = bool(lock_background_if_requested)
    else:
        # Default: use first ref as a loose base if available; otherwise blank.
        if reference_images:
            base = _load_rgb(reference_images[0], size=size)
        else:
            base = Image.new("RGB", size, (0, 0, 0))
        background_locked = False

    base_arr = np.array(base).astype(np.uint8)

    # Preserve-mask: white means "preserve/keep"; we will invert at end.
    preserve = np.zeros((H, W), dtype=np.uint8)

    if background_locked:
        preserve[:, :] = 255

    preserved_parts = 0

    # Overlay preserved parts.
    for p in parts:
        if p.request.role == "background" or p.request.part.lower() == "background":
            continue
        if not p.crop_path or not p.mask_path:
            continue
        try:
            crop = _load_rgb(p.crop_path, size=size)
            m = _load_mask(p.mask_path, size=size)
        except Exception:
            continue

        crop_arr = np.array(crop).astype(np.uint8)
        m_arr = (np.array(m).astype(np.uint8) >= 128).astype(np.uint8) * 255

        # Composite crop onto base using mask.
        m3 = (m_arr[..., None] / 255.0).astype(np.float32)
        base_arr = (
            (base_arr.astype(np.float32) * (1.0 - m3) + crop_arr.astype(np.float32) * m3).round().astype(np.uint8)
        )

        # Update preserve map (keep these pixels).
        preserve = np.maximum(preserve, m_arr)
        preserved_parts += 1

    init = Image.fromarray(base_arr, mode="RGB")

    # Inpaint mask: white=inpaint, black=keep.
    # If we are not background-locked, we inpaint everything except preserved parts.
    inpaint = (255 - preserve).astype(np.uint8)
    mask = Image.fromarray(inpaint, mode="L")

    init_path = out_dir / "composite_init.png"
    mask_path = out_dir / "composite_mask.png"
    init.save(init_path)
    mask.save(mask_path)

    return CompositeSpec(
        init_image_path=str(init_path),
        mask_path=str(mask_path),
        preserved_parts=int(preserved_parts),
        background_locked=bool(background_locked),
    )


__all__ = ["CompositeSpec", "build_init_and_inpaint_mask"]
