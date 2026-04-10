"""
Pixel-perfect canvas alignment for **DiT + VAE + block-AR** sampling.

Keeps RGB height/width on an integer grid that matches:

- **VAE**: latent cells are ``image_px // 8`` (SD-style KL/RAE bridge in this repo).
- **DiT PatchEmbed**: each token covers ``patch_size`` latent cells per axis, so one token spans
  ``8 * patch_size`` RGB pixels when ``patch_size`` is isotropic.

Block-AR masks are defined on ``sqrt(num_patches)``; using the model's own ``x_embedder.img_size``
for latent shape already guarantees token count matches. This module snaps **requested** pixel
sizes so decode/resize pipelines avoid half-pixel drift and crisp edges for line art / UI tiles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Union

LATENT_TO_PIXEL: int = 8  # RGB pixels per latent cell (this repo's VAE convention)


SnapMode = Literal["floor", "ceil", "nearest"]


def snap_to_multiple(n: int, m: int, *, mode: SnapMode = "nearest") -> int:
    """Round integer ``n`` to a multiple of positive ``m``."""
    if m <= 0:
        raise ValueError("m must be positive")
    n, m = int(n), int(m)
    if mode == "floor":
        return max(m, (n // m) * m)
    if mode == "ceil":
        return max(m, ((n + m - 1) // m) * m)
    # nearest
    down = max(m, (n // m) * m)
    up = max(m, ((n + m - 1) // m) * m)
    return up if abs(up - n) < abs(n - down) else down


def _patch_size_latent(model: Any) -> int:
    emb = getattr(model, "x_embedder", None)
    if emb is None:
        return 1
    ps = getattr(emb, "patch_size", 1)
    if isinstance(ps, int):
        return max(1, int(ps))
    if isinstance(ps, (tuple, list)) and len(ps) >= 1:
        return max(1, int(ps[0]))
    return 1


def dit_rgb_stride_px(model: Any) -> int:
    """
    RGB pixels spanned by one DiT patch token along one axis (``8 * patch_size`` latent).
    Falls back to ``LATENT_TO_PIXEL`` when ``x_embedder`` is missing.
    """
    pl = _patch_size_latent(model)
    return int(LATENT_TO_PIXEL) * pl


def pixel_stride_for_pipeline(
    *,
    model: Any = None,
    override_stride_px: Optional[int] = None,
) -> int:
    """
    Conservative stride for snapping **width/height**: at least VAE alignment (8), and DiT patch
    alignment when ``model`` is provided.
    """
    if override_stride_px is not None:
        s = int(override_stride_px)
        if s < LATENT_TO_PIXEL:
            raise ValueError(f"override_stride_px must be >= {LATENT_TO_PIXEL}")
        return s
    s = LATENT_TO_PIXEL
    if model is not None:
        s = max(s, dit_rgb_stride_px(model))
    return s


def latent_hw_from_pixels(height_px: int, width_px: int) -> Tuple[int, int]:
    """``(latent_h, latent_w)`` for SD-style ``// 8`` latent."""
    h = max(1, int(height_px) // LATENT_TO_PIXEL)
    w = max(1, int(width_px) // LATENT_TO_PIXEL)
    return h, w


def pixels_from_latent_hw(latent_h: int, latent_w: int) -> Tuple[int, int]:
    """Exact RGB size for a full latent grid."""
    return int(latent_h) * LATENT_TO_PIXEL, int(latent_w) * LATENT_TO_PIXEL


@dataclass(frozen=True)
class PixelPerfectCanvas:
    """Canonical sizes after snapping (pixels and latent)."""

    height_px: int
    width_px: int
    latent_h: int
    latent_w: int
    stride_px: int
    snap_mode: str
    aligned_to_dit_patch: bool

    def as_dict(self) -> Dict[str, Union[int, str, bool]]:
        return {
            "height_px": self.height_px,
            "width_px": self.width_px,
            "latent_h": self.latent_h,
            "latent_w": self.latent_w,
            "pixel_perfect_stride_px": self.stride_px,
            "pixel_perfect_snap_mode": self.snap_mode,
            "pixel_perfect_dit_patch_aligned": self.aligned_to_dit_patch,
        }


def resolve_pixel_perfect_hw(
    height_px: int,
    width_px: int,
    *,
    model: Any = None,
    stride_px: Optional[int] = None,
    mode: SnapMode = "nearest",
    square: bool = False,
    min_side: int = LATENT_TO_PIXEL,
    max_side: int = 16384,
) -> Tuple[int, int, PixelPerfectCanvas]:
    """
    Snap ``height_px`` × ``width_px`` to a pipeline-aligned grid.

    Returns ``(h, w, spec)`` where ``h,w`` are multiples of the chosen stride and within bounds.
    """
    stride = pixel_stride_for_pipeline(model=model, override_stride_px=stride_px)

    lo = max(int(min_side), stride)
    hi = max(lo, int(max_side))

    h0, w0 = int(height_px), int(width_px)
    h0 = max(lo, min(hi, h0))
    w0 = max(lo, min(hi, w0))
    if square:
        s = max(h0, w0)
        s = max(lo, min(hi, s))
        h0 = w0 = snap_to_multiple(s, stride, mode=mode)

    h = snap_to_multiple(h0, stride, mode=mode)
    w = snap_to_multiple(w0, stride, mode=mode)
    h = max(lo, min(hi, h))
    w = max(lo, min(hi, w))

    lh, lw = latent_hw_from_pixels(h, w)
    if model is None:
        patch_aligned = False
    else:
        dpx = dit_rgb_stride_px(model)
        patch_aligned = dpx > 0 and h % dpx == 0 and w % dpx == 0
    spec = PixelPerfectCanvas(
        height_px=h,
        width_px=w,
        latent_h=lh,
        latent_w=lw,
        stride_px=stride,
        snap_mode=mode,
        aligned_to_dit_patch=patch_aligned,
    )
    return h, w, spec


def validate_pixels_against_dit(
    height_px: int,
    width_px: int,
    model: Any,
) -> None:
    """
    Raise ``ValueError`` if pixel canvas does not match ``model.x_embedder.img_size`` latent shape.
    """
    emb = getattr(model, "x_embedder", None)
    if emb is None:
        return
    img_size = getattr(emb, "img_size", None)
    if img_size is None:
        return
    if isinstance(img_size, int):
        eh = ew = int(img_size)
    else:
        eh, ew = int(img_size[0]), int(img_size[1])
    lh, lw = latent_hw_from_pixels(height_px, width_px)
    if lh != eh or lw != ew:
        raise ValueError(
            f"pixel canvas {height_px}x{width_px} -> latent {lh}x{lw} != DiT x_embedder.img_size ({eh}x{ew})"
        )


def ar_block_grid_side(num_patches: int) -> Optional[int]:
    """Return ``P`` with ``P*P == num_patches`` if square; else ``None``."""
    np_ = int(num_patches)
    if np_ <= 0:
        return None
    p = int(round(math.sqrt(float(np_))))
    return p if p * p == np_ else None


def validate_latent_matches_ar_grid(latent_h: int, latent_w: int, num_patches: int) -> None:
    """Ensure latent grid matches a square token layout (required for 2D block-AR)."""
    p = ar_block_grid_side(num_patches)
    if p is None:
        raise ValueError(f"num_patches={num_patches} is not a perfect square; cannot verify AR grid")
    if int(latent_h) != p or int(latent_w) != p:
        raise ValueError(f"latent {latent_h}x{latent_w} != AR token grid {p}x{p} (num_patches={num_patches})")


def tag_manifest_pixel_perfect(
    row: Mapping[str, Any],
    spec: Union[PixelPerfectCanvas, Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Merge pixel-perfect fields into a manifest / JSONL row for ViT QA and book tooling.

    Skips keys that already exist when ``overwrite`` is False.
    """
    out = dict(row)
    d = spec.as_dict() if isinstance(spec, PixelPerfectCanvas) else dict(spec)
    for k, v in d.items():
        if not overwrite and k in out:
            continue
        out[k] = v
    return out


__all__ = [
    "LATENT_TO_PIXEL",
    "PixelPerfectCanvas",
    "ar_block_grid_side",
    "dit_rgb_stride_px",
    "latent_hw_from_pixels",
    "pixels_from_latent_hw",
    "pixel_stride_for_pipeline",
    "resolve_pixel_perfect_hw",
    "snap_to_multiple",
    "tag_manifest_pixel_perfect",
    "validate_latent_matches_ar_grid",
    "validate_pixels_against_dit",
]
