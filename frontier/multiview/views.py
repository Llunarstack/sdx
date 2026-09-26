"""
"Perfect Angles" — render the one shared field from canonical viewpoints.

Straight from the original SDX-3D idea: a ``<viewpoint>`` control that returns
front / side / top / three-quarter images that are *guaranteed* to be the same
object, because they are all cameras looking at the same tri-plane field. No more
"front view is a dragon, side view is a different dragon."

Provides named orthogonal-ish viewpoints, a turntable orbit, and a contact-sheet
tiler so you can eyeball all angles at once (and save a PNG).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from .neus_render import NeusRenderer, look_at_rays
from .triplane import SpatialLatentField

# Camera *directions* (the camera sits at dir * radius, looking at the origin).
CANONICAL_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, 0.0, -1.0),
    "back": (0.0, 0.0, 1.0),
    "right": (1.0, 0.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "top": (0.0, 1.0, 0.0),
    "bottom": (0.0, -1.0, 0.0),
    "three_quarter": (0.7, 0.5, -0.7),
}


def viewpoint_rays(
    name: str,
    *,
    radius: float = 2.5,
    fov_deg: float = 45.0,
    height: int = 64,
    width: int = 64,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Camera rays for a named canonical viewpoint."""
    if name not in CANONICAL_DIRECTIONS:
        raise KeyError(f"unknown viewpoint {name!r}; options: {sorted(CANONICAL_DIRECTIONS)}")
    d = torch.tensor(CANONICAL_DIRECTIONS[name], dtype=torch.float32, device=device)
    cam_pos = torch.nn.functional.normalize(d, dim=0) * radius
    # For top/bottom the world-up is parallel to the view dir, so pick a different up.
    up = (0.0, 0.0, 1.0) if name in ("top", "bottom") else (0.0, 1.0, 0.0)
    return look_at_rays(cam_pos, torch.zeros(3, device=device), fov_deg=fov_deg, height=height, width=width, up=up)


def _render_image(
    field: SpatialLatentField,
    planes: tuple[Tensor, Tensor, Tensor],
    renderer: NeusRenderer,
    rays_o: Tensor,
    rays_d: Tensor,
    height: int,
    width: int,
    *,
    near: float,
    far: float,
    background: float,
) -> np.ndarray:
    """Render rays and composite the RGB over a flat background using opacity."""
    with torch.no_grad():
        out = renderer.render(field, planes, rays_o, rays_d, near=near, far=far)
    rgb = out["rgb"].reshape(height, width, 3)
    alpha = out["opacity"].reshape(height, width, 1).clamp(0, 1)
    composited = rgb * alpha + background * (1.0 - alpha)
    return composited.clamp(0, 1).cpu().numpy()


def perfect_angles(
    field: SpatialLatentField,
    planes: tuple[Tensor, Tensor, Tensor],
    renderer: NeusRenderer,
    *,
    names: list[str] | None = None,
    radius: float = 2.5,
    fov_deg: float = 45.0,
    resolution: int = 64,
    near: float = 0.5,
    far: float = 5.0,
    background: float = 1.0,
) -> dict[str, np.ndarray]:
    """Render a dict of ``name -> (H, W, 3)`` images, all of the same object."""
    names = names or ["front", "right", "back", "left", "top", "three_quarter"]
    device = planes[0].device
    out: dict[str, np.ndarray] = {}
    for name in names:
        o, d = viewpoint_rays(name, radius=radius, fov_deg=fov_deg, height=resolution, width=resolution, device=device)
        out[name] = _render_image(
            field, planes, renderer, o, d, resolution, resolution, near=near, far=far, background=background
        )
    return out


def turntable(
    field: SpatialLatentField,
    planes: tuple[Tensor, Tensor, Tensor],
    renderer: NeusRenderer,
    *,
    n_frames: int = 8,
    elevation_deg: float = 20.0,
    radius: float = 2.5,
    fov_deg: float = 45.0,
    resolution: int = 64,
    near: float = 0.5,
    far: float = 5.0,
    background: float = 1.0,
) -> list[np.ndarray]:
    """Render an orbit of frames around the object (azimuth sweep)."""
    device = planes[0].device
    el = math.radians(elevation_deg)
    frames: list[np.ndarray] = []
    for i in range(n_frames):
        az = 2 * math.pi * i / n_frames
        cam = torch.tensor(
            [radius * math.cos(el) * math.sin(az), radius * math.sin(el), radius * math.cos(el) * math.cos(az)],
            dtype=torch.float32,
            device=device,
        )
        o, d = look_at_rays(cam, torch.zeros(3, device=device), fov_deg=fov_deg, height=resolution, width=resolution)
        frames.append(
            _render_image(
                field, planes, renderer, o, d, resolution, resolution, near=near, far=far, background=background
            )
        )
    return frames


def contact_sheet(images: list[np.ndarray], *, cols: int = 3, pad: int = 4, background: float = 1.0) -> np.ndarray:
    """Tile images into a single ``(H, W, 3)`` grid with padding."""
    if not images:
        raise ValueError("no images to tile")
    h, w, _ = images[0].shape
    rows = math.ceil(len(images) / cols)
    sheet = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), background, dtype=np.float32)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        sheet[y : y + h, x : x + w] = img
    return sheet


def save_image(path: str, image: np.ndarray) -> None:
    """Write a float ``[0, 1]`` image (H, W, 3) to a PNG."""
    from PIL import Image

    arr = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)
