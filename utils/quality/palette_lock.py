"""Hex palette lock — Recraft / Ideogram non-language colour channel.

Steer image chrominance toward up to 16 ``#RRGGBB`` colours without a hard LUT.
Nearby colours pull; distant colours (skin, sky, unless they match a swatch)
stay put.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ["hex_to_rgb", "apply_palette_lock", "score_palette_lock", "palette_from_prompt"]

_HEX_RE = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")


def hex_to_rgb(token: str) -> tuple[float, float, float] | None:
    t = (token or "").strip()
    if t.startswith("#"):
        t = t[1:]
    if len(t) == 3 and re.fullmatch(r"[0-9a-fA-F]{3}", t):
        t = "".join(ch * 2 for ch in t)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", t):
        return None
    return int(t[0:2], 16) / 255.0, int(t[2:4], 16) / 255.0, int(t[4:6], 16) / 255.0


def palette_from_prompt(
    prompt: str, extra: list[str] | tuple[str, ...] | None = None
) -> list[tuple[float, float, float]]:
    tokens = list(_HEX_RE.findall(prompt or ""))
    if extra:
        tokens.extend(str(t) for t in extra)
    rgb: list[tuple[float, float, float]] = []
    seen: set[tuple[int, int, int]] = set()
    for token in tokens:
        trip = hex_to_rgb(token)
        if trip is None:
            continue
        key = (int(trip[0] * 255), int(trip[1] * 255), int(trip[2] * 255))
        if key in seen:
            continue
        seen.add(key)
        rgb.append(trip)
        if len(rgb) >= 16:
            break
    return rgb


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB ``[..., 3]`` in 0–1 → Lab. D65, sRGB."""
    lin = _srgb_to_linear(np.clip(rgb, 0.0, 1.0))
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = lin @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    l = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([l, a, b], axis=-1)


def _lab_ab(rgb: np.ndarray) -> np.ndarray:
    return _rgb_to_lab(rgb)[..., 1:3]


def apply_palette_lock(
    img_np: np.ndarray,
    hexes: list[str] | tuple[str, ...] | None,
    *,
    strength: float = 0.22,
    prompt: str = "",
) -> np.ndarray:
    swatches = palette_from_prompt(prompt, extra=list(hexes or []))
    s = float(max(0.0, min(1.0, strength)))
    if s <= 1e-6 or not swatches or img_np.ndim != 3 or img_np.shape[2] < 3:
        return img_np
    arr = np.asarray(img_np, dtype=np.float32)
    rgb = np.clip(arr[..., :3] / 255.0, 0.0, 1.0)
    pix_ab = _lab_ab(rgb.reshape(-1, 3).astype(np.float64))
    pal_ab = _lab_ab(np.asarray(swatches, dtype=np.float64).reshape(-1, 3))
    # (N, P) chroma distance to each swatch
    delta = pix_ab[:, None, :] - pal_ab[None, :, :]
    dist = np.sqrt((delta * delta).sum(axis=-1) + 1e-8)
    nearest = dist.argmin(axis=1)
    dmin = dist[np.arange(dist.shape[0]), nearest]
    target = pal_ab[nearest]
    # Pull only colours already in the palette neighbourhood (sigma in Lab a/b).
    weight = s * np.exp(-(dmin**2) / (2.0 * (28.0**2)))
    mixed = pix_ab + weight[:, None] * (target - pix_ab)
    out_ab = mixed.reshape(rgb.shape[0], rgb.shape[1], 2)
    # Rebuild RGB by converting Lab with original L + new ab ≈ chroma-only grade.
    lab = _rgb_to_lab(rgb.astype(np.float64))
    lab[..., 1:3] = out_ab
    graded = _lab_to_srgb(lab)
    out = arr.copy()
    out[..., :3] = np.clip(np.round(graded * 255.0), 0, 255)
    return out.astype(img_np.dtype)


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (l + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps = 216 / 24389
    kappa = 24389 / 27
    xr = np.where(fx**3 > eps, fx**3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(l > kappa * eps, fy**3, l / kappa)
    zr = np.where(fz**3 > eps, fz**3, (116.0 * fz - 16.0) / kappa)
    xyz = np.stack([xr * 0.95047, yr, zr * 1.08883], axis=-1)
    m_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float64,
    )
    lin = xyz @ m_inv.T
    lin = np.clip(lin, 0.0, None)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0)


def score_palette_lock(
    img_np: np.ndarray,
    hexes: list[str] | tuple[str, ...] | None = None,
    *,
    prompt: str = "",
) -> float:
    """1 = dominant chromas sit on the named swatches; 0.5 if no palette."""
    swatches = palette_from_prompt(prompt, extra=list(hexes or []))
    if not swatches or img_np.ndim != 3 or img_np.shape[2] < 3:
        return 0.5
    small = img_np[:: max(1, img_np.shape[0] // 64), :: max(1, img_np.shape[1] // 64), :3]
    rgb = np.clip(small.astype(np.float64) / 255.0, 0.0, 1.0).reshape(-1, 3)
    pix_ab = _lab_ab(rgb)
    pal_ab = _lab_ab(np.asarray(swatches, dtype=np.float64).reshape(-1, 3))
    dist = np.sqrt(((pix_ab[:, None, :] - pal_ab[None, :, :]) ** 2).sum(axis=-1) + 1e-8)
    dmin = dist.min(axis=1)
    # Saturated pixels (chroma > 8) should land near a swatch.
    chroma = np.sqrt((pix_ab**2).sum(axis=1))
    mask = chroma > 8.0
    if not np.any(mask):
        return 0.55
    mean_d = float(dmin[mask].mean())
    return float(np.clip(1.0 - mean_d / 60.0, 0.0, 1.0))
