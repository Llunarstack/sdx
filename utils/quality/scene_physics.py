"""Still-image occlusion and reflection heuristics (uint8 RGB HWC, CPU numpy)."""

from __future__ import annotations

import re

import numpy as np

__all__ = [
    "prompt_wants_occlusion",
    "prompt_wants_reflection",
    "score_occlusion",
    "score_reflection",
    "score_spatial_color_bind",
    "occlusion_score",
    "reflection_score",
    "spatial_bind_score",
]

_OCCLUSION_KEYS = (
    "behind",
    "through",
    "fence",
    "gaps",
    "hidden behind",
    "occlud",
)

_REFLECTION_KEYS = (
    "reflection",
    "reflected",
)

_CHROME_SURFACES = ("granite", "water", "puddle", "table")

_COLOR_RGB: dict[str, np.ndarray] = {
    "red": np.array([0.78, 0.12, 0.12], dtype=np.float32),
    "blue": np.array([0.12, 0.18, 0.78], dtype=np.float32),
    "green": np.array([0.12, 0.62, 0.18], dtype=np.float32),
    "yellow": np.array([0.82, 0.78, 0.12], dtype=np.float32),
    "orange": np.array([0.88, 0.45, 0.10], dtype=np.float32),
    "purple": np.array([0.48, 0.18, 0.62], dtype=np.float32),
    "pink": np.array([0.88, 0.38, 0.58], dtype=np.float32),
    "white": np.array([0.88, 0.88, 0.88], dtype=np.float32),
    "black": np.array([0.08, 0.08, 0.08], dtype=np.float32),
    "brown": np.array([0.42, 0.26, 0.14], dtype=np.float32),
    "gray": np.array([0.48, 0.48, 0.48], dtype=np.float32),
    "grey": np.array([0.48, 0.48, 0.48], dtype=np.float32),
    "gold": np.array([0.78, 0.62, 0.18], dtype=np.float32),
    "golden": np.array([0.78, 0.62, 0.18], dtype=np.float32),
    "silver": np.array([0.68, 0.70, 0.74], dtype=np.float32),
}
_COLOR_ALT = "|".join(re.escape(k) for k in _COLOR_RGB)
_LEFT_OF_COLORS = re.compile(
    rf"\b({_COLOR_ALT})\b.{{0,40}}?\b(?:to\s+the\s+)?left\s+of\b.{{0,24}}?\b({_COLOR_ALT})\b",
    re.I | re.S,
)
_RIGHT_OF_COLORS = re.compile(
    rf"\b({_COLOR_ALT})\b.{{0,40}}?\b(?:to\s+the\s+)?right\s+of\b.{{0,24}}?\b({_COLOR_ALT})\b",
    re.I | re.S,
)
_LEFT_RIGHT_BIND = re.compile(
    rf"\b({_COLOR_ALT})\b.{{0,48}}?\bon\s+the\s+left\b.{{0,80}}?\b({_COLOR_ALT})\b.{{0,48}}?\bon\s+the\s+right\b",
    re.I | re.S,
)


def prompt_wants_occlusion(prompt: str) -> bool:
    """True when the prompt implies a foreground occluder (fence, glass, gaps)."""
    text = (prompt or "").lower()
    if not text:
        return False
    return any(k in text for k in _OCCLUSION_KEYS)


def prompt_wants_reflection(prompt: str) -> bool:
    """True for table/water/chrome reflections, not a hall-of-mirrors prompt alone."""
    text = (prompt or "").lower()
    if not text:
        return False
    if any(k in text for k in _REFLECTION_KEYS):
        return True
    if "mirror of" in text:
        return True
    if "chrome" in text and any(s in text for s in _CHROME_SURFACES):
        return True
    return False


def _as_rgb(rgb: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[0] < 4 or arr.shape[1] < 4 or arr.shape[2] < 3:
        return None
    return arr[..., :3]


def _gray(rgb: np.ndarray) -> np.ndarray:
    ch = rgb.astype(np.float32, copy=False)
    return 0.299 * ch[..., 0] + 0.587 * ch[..., 1] + 0.114 * ch[..., 2]


def _sobel_abs(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    return np.abs(gx) + np.abs(gy)


def score_occlusion(rgb: np.ndarray) -> float:
    """Foreground occluder has high-frequency overlay; subject not a smear.

    Combines center-vs-frame edge energy, vertical-bar anisotropy (fences/mesh),
    and overall edge magnitude so a fused blob does not score like a mesh.
    """
    arr = _as_rgb(rgb)
    if arr is None:
        return 0.0
    gray = _gray(arr)
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    edge = _sobel_abs(gray)
    h, w = edge.shape
    y0, y1 = int(h * 0.20), max(int(h * 0.20) + 1, int(h * 0.80))
    x0, x1 = int(w * 0.20), max(int(w * 0.20) + 1, int(w * 0.80))
    center = edge[y0:y1, x0:x1]
    mean_center = float(center.mean()) if center.size else 0.0
    mean_all = float(edge.mean())
    fill = float(np.clip(mean_center / (mean_all + 1e-6), 0.0, 1.0))
    gx_m = float(gx.mean())
    gy_m = float(gy.mean())
    aniso = gx_m / (gx_m + gy_m + 1e-6)
    energy = float(np.tanh(mean_all / 18.0))
    return float(np.clip(0.42 * fill + 0.38 * aniso + 0.20 * energy, 0.0, 1.0))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size < 2 or y.size != x.size:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom < 1e-8:
        return 0.0
    return float(np.clip(np.dot(x, y) / denom, -1.0, 1.0))


def score_reflection(rgb: np.ndarray) -> float:
    """Lower half resembles a vertically flipped, darker version of upper half
    (table/water reflection) rather than a second identical object.

    Compare upper[::-1] vs lower with a scale/shift; penalize if lower is as
    bright as upper (second object).
    """
    arr = _as_rgb(rgb)
    if arr is None:
        return 0.0
    gray = _gray(arr)
    h = gray.shape[0]
    half = h // 2
    if half < 2:
        return 0.0
    upper = gray[:half]
    lower = gray[half : half + half]
    flipped = upper[::-1]
    # Pearson is cosine of mean-centered vectors (scale + shift invariant).
    corr = max(0.0, _pearson(flipped, lower))
    upper_m = float(upper.mean())
    lower_m = float(lower.mean())
    if lower_m >= 0.95 * upper_m:
        darkness_factor = 0.30
    else:
        darkness_factor = 1.0
    return float(np.clip(corr * darkness_factor, 0.0, 1.0))


def _mean_rgb(rgb: np.ndarray) -> np.ndarray:
    ch = rgb.astype(np.float32) / 255.0
    return ch.reshape(-1, 3).mean(axis=0)


def _color_match(mean_rgb: np.ndarray, name: str) -> float:
    target = _COLOR_RGB.get(name.lower())
    if target is None:
        return 0.0
    a = mean_rgb.astype(np.float32)
    b = target
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def _parse_left_right_colors(prompt: str) -> tuple[str, str] | None:
    text = prompt or ""
    m = _LEFT_OF_COLORS.search(text)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    m = _LEFT_RIGHT_BIND.search(text)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    m = _RIGHT_OF_COLORS.search(text)
    if m:
        return m.group(2).lower(), m.group(1).lower()
    return None


def score_spatial_color_bind(rgb: np.ndarray, prompt: str) -> float | None:
    """Left/right halves match the colors named in a spatial bind prompt.

    Returns None when the prompt has no parseable color pair so pick-best can skip.
    """
    pair = _parse_left_right_colors(prompt)
    if pair is None:
        return None
    arr = _as_rgb(rgb)
    if arr is None:
        return 0.0
    left_name, right_name = pair
    w = arr.shape[1]
    mid = max(1, w // 2)
    left = _mean_rgb(arr[:, :mid])
    right = _mean_rgb(arr[:, mid:])
    correct = 0.5 * (_color_match(left, left_name) + _color_match(right, right_name))
    swapped = 0.5 * (_color_match(left, right_name) + _color_match(right, left_name))
    return float(np.clip(0.5 + 0.5 * (correct - swapped), 0.0, 1.0))


def _load_rgb_path(image_path: str) -> np.ndarray | None:
    try:
        from PIL import Image

        return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def occlusion_score(image_path: str) -> tuple[float | None, dict]:
    """Eval-harness wrapper around ``score_occlusion``."""
    arr = _load_rgb_path(image_path)
    if arr is None:
        return None, {"error": "image_unreadable"}
    return float(score_occlusion(arr)), {"metric": "occlusion"}


def reflection_score(image_path: str) -> tuple[float | None, dict]:
    """Eval-harness wrapper around ``score_reflection``."""
    arr = _load_rgb_path(image_path)
    if arr is None:
        return None, {"error": "image_unreadable"}
    return float(score_reflection(arr)), {"metric": "reflection"}


def spatial_bind_score(image_path: str, prompt: str) -> tuple[float | None, dict]:
    """Eval-harness wrapper around ``score_spatial_color_bind``."""
    arr = _load_rgb_path(image_path)
    if arr is None:
        return None, {"error": "image_unreadable"}
    val = score_spatial_color_bind(arr, prompt)
    if val is None:
        return None, {"skipped": "no_color_pair"}
    return float(val), {"metric": "spatial_color_bind"}
