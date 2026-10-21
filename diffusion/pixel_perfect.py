"""
Pixel-perfect canvas alignment for **DiT + VAE + block-AR** sampling.

Keeps RGB height/width on an integer grid that matches:

- **VAE**: latent cells are ``image_px // 8`` (SD-style KL/RAE bridge in this repo).
- **DiT PatchEmbed**: each token covers ``patch_size`` latent cells per axis, so one token spans
  ``8 * patch_size`` RGB pixels when ``patch_size`` is isotropic.

The DiTs accept any latent grid that tiles into whole patches (positions and block-AR masks are built
per grid), so this module snaps **requested** pixel sizes onto that stride and picks the sampling
canvas (``sampling_canvas_hw``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

LATENT_TO_PIXEL: int = 8  # RGB pixels per latent cell (this repo's VAE convention)

SnapMode = Literal["floor", "ceil", "nearest"]


def snap_to_multiple(n: int, m: int, *, mode: SnapMode = "nearest") -> int:
    """Round integer ``n`` to a multiple of positive ``m``."""
    if m <= 0:
        raise ValueError("m must be positive")
    n, m = int(n), int(m)
    down = max(m, (n // m) * m)
    up = max(m, -(-n // m) * m)
    if mode == "floor":
        return down
    if mode == "ceil":
        return up
    return up if abs(up - n) < abs(n - down) else down


def dit_patch_size(model: Any) -> int:
    """Latent cells per patch side; the largest size for dynamic patching (``SuperiorViT``), else 1."""
    model = getattr(model, "_orig_mod", model)
    sizes = getattr(getattr(model, "patch_embed", None), "patch_sizes", None)
    if sizes:
        return max(int(p) for p in sizes)
    ps = getattr(getattr(model, "x_embedder", None), "patch_size", None)
    if ps is None:
        ps = getattr(model, "patch_size", 1)
    if isinstance(ps, (tuple, list)):
        ps = ps[0] if ps else 1
    return max(1, int(ps))


def dit_rgb_stride_px(model: Any) -> int:
    """RGB pixels spanned by one DiT patch token along one axis (``8 * patch_size``)."""
    return int(LATENT_TO_PIXEL) * dit_patch_size(model)


def pixel_stride_for_pipeline(
    *,
    model: Any = None,
    override_stride_px: int | None = None,
) -> int:
    """Pixel stride for snapping width/height: VAE alignment (8), or the DiT patch stride when ``model`` is given."""
    if override_stride_px is not None:
        s = int(override_stride_px)
        if s < LATENT_TO_PIXEL:
            raise ValueError(f"override_stride_px must be >= {LATENT_TO_PIXEL}")
        return s
    return LATENT_TO_PIXEL if model is None else dit_rgb_stride_px(model)


def sampling_canvas_hw(
    height_px: int,
    width_px: int,
    *,
    native_px: int,
    stride_px: int,
    max_area_ratio: float = 2.0,
) -> tuple[int, int]:
    """Pixel ``(h, w)`` of the first sampling pass for a requested size.

    The requested size snapped to ``stride_px`` when its area is within ``max_area_ratio`` of the native
    ``native_px`` square (the sampler's SD3 resolution shift covers that range). Otherwise the requested
    aspect ratio at the native pixel area (aspect-ratio bucketing, SDXL / NovelAI), leaving the final size
    to hires-fix or a resize; ``max_area_ratio=1`` always does this.
    """
    h0, w0 = max(1, int(height_px)), max(1, int(width_px))
    area_ratio = (h0 * w0) / float(native_px * native_px)
    if 1.0 / max_area_ratio <= area_ratio <= max_area_ratio:
        h, w = h0, w0
    else:
        aspect = w0 / h0
        h, w = native_px / aspect**0.5, native_px * aspect**0.5
    return snap_to_multiple(round(h), stride_px), snap_to_multiple(round(w), stride_px)


def latent_hw_from_pixels(height_px: int, width_px: int) -> tuple[int, int]:
    """``(latent_h, latent_w)`` for the SD-style ``// 8`` latent."""
    return max(1, int(height_px) // LATENT_TO_PIXEL), max(1, int(width_px) // LATENT_TO_PIXEL)


def pixels_from_latent_hw(latent_h: int, latent_w: int) -> tuple[int, int]:
    """Exact RGB size for a full latent grid."""
    return int(latent_h) * LATENT_TO_PIXEL, int(latent_w) * LATENT_TO_PIXEL


@dataclass(frozen=True, slots=True)
class PixelPerfectCanvas:
    """Canonical sizes after snapping (pixels and latent)."""

    height_px: int
    width_px: int
    latent_h: int
    latent_w: int
    stride_px: int
    snap_mode: str
    aligned_to_dit_patch: bool

    def as_dict(self) -> dict[str, int | str | bool]:
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
    stride_px: int | None = None,
    mode: SnapMode = "nearest",
    square: bool = False,
    min_side: int = LATENT_TO_PIXEL,
    max_side: int = 16384,
) -> tuple[int, int, PixelPerfectCanvas]:
    """Snap ``height_px`` x ``width_px`` onto the pipeline stride within ``[min_side, max_side]``.

    Returns ``(h, w, spec)``.
    """
    stride = pixel_stride_for_pipeline(model=model, override_stride_px=stride_px)
    lo = max(int(min_side), stride)
    hi = max(lo, int(max_side))

    def clamp(v: int) -> int:
        return max(lo, min(hi, v))

    h0, w0 = clamp(int(height_px)), clamp(int(width_px))
    if square:
        h0 = w0 = max(h0, w0)
    h = clamp(snap_to_multiple(h0, stride, mode=mode))
    w = clamp(snap_to_multiple(w0, stride, mode=mode))

    lh, lw = latent_hw_from_pixels(h, w)
    if model is None:
        patch_aligned = False
    else:
        dpx = dit_rgb_stride_px(model)
        patch_aligned = h % dpx == 0 and w % dpx == 0
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


def validate_pixels_against_dit(height_px: int, width_px: int, model: Any) -> None:
    """Raise ``ValueError`` unless the pixel canvas tiles into whole DiT patches."""
    stride = dit_rgb_stride_px(model)
    if int(height_px) % stride or int(width_px) % stride:
        lh, lw = latent_hw_from_pixels(height_px, width_px)
        raise ValueError(
            f"pixel canvas {height_px}x{width_px} (latent {lh}x{lw}) is not a multiple of the {stride}px DiT patch"
        )


def tag_manifest_pixel_perfect(
    row: Mapping[str, Any],
    spec: PixelPerfectCanvas | Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy of a manifest row with the pixel-perfect fields merged in; existing keys win unless ``overwrite``."""
    out = dict(row)
    d = spec.as_dict() if isinstance(spec, PixelPerfectCanvas) else dict(spec)
    for k, v in d.items():
        if overwrite or k not in out:
            out[k] = v
    return out


__all__ = [
    "LATENT_TO_PIXEL",
    "PixelPerfectCanvas",
    "dit_patch_size",
    "dit_rgb_stride_px",
    "latent_hw_from_pixels",
    "pixels_from_latent_hw",
    "pixel_stride_for_pipeline",
    "resolve_pixel_perfect_hw",
    "sampling_canvas_hw",
    "snap_to_multiple",
    "tag_manifest_pixel_perfect",
    "validate_pixels_against_dit",
]
