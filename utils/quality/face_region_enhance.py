"""
Face-focused post-process: detect frontal faces (OpenCV Haar) and apply local
sharpen + micro-contrast. Bridges the gap to ADetailer-style workflows without
requiring GFPGAN/CodeFormer in-tree.

Requires ``opencv-python`` for detection; if import or cascade fails, callers should skip.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _detect_face_rects_bgr(bgr: np.ndarray, *, min_side: int = 48) -> List[Tuple[int, int, int, int]]:
    try:
        import cv2
    except ImportError:
        return []

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return []
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_side, min_side))
    if faces is None or len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def enhance_faces_in_rgb(
    rgb: np.ndarray,
    *,
    padding: float = 0.25,
    sharpen_amount: float = 0.35,
    micro_contrast: float = 1.04,
    max_faces: int = 4,
    min_face_side: int = 48,
) -> np.ndarray:
    """
    Return a copy of *rgb* (H,W,3 uint8) with up to *max_faces* frontal regions sharpened.

    *padding* expands each bbox by this fraction of max(w,h) before processing.
    *micro_contrast* is passed to ``quality.contrast`` on the patch (1.0 = off).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)

    try:
        import cv2
    except ImportError:
        return rgb

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rects = _detect_face_rects_bgr(bgr, min_side=min_face_side)
    if not rects:
        return rgb

    from utils.quality.quality import contrast, sharpen

    out = rgb.copy()
    H, W = out.shape[:2]

    for fi, (x, y, w, h) in enumerate(rects[: max(1, int(max_faces))]):
        pad = int(max(w, h) * float(padding))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)
        if x2 <= x1 or y2 <= y1:
            continue

        patch = out[y1:y2, x1:x2].copy()
        ph, pw = patch.shape[:2]
        # Ellipse mask centered on original face (not full padded rect) for soft blend
        fcx = x + w // 2 - x1
        fcy = y + h // 2 - y1
        rx = max(4, int(w * 0.55))
        ry = max(4, int(h * 0.65))

        enhanced = sharpen(patch, amount=float(sharpen_amount))
        if abs(float(micro_contrast) - 1.0) >= 1e-6:
            enhanced = contrast(enhanced.astype(np.float32), factor=float(micro_contrast))
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        mask = np.zeros((ph, pw), dtype=np.float32)
        cv2.ellipse(mask, (fcx, fcy), (rx, ry), 0, 0, 360, 1.0, thickness=-1)
        sigma = max(3.0, min(rx, ry) * 0.25)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        mask = np.clip(mask, 0.0, 1.0)
        m3 = mask[:, :, np.newaxis]
        blended = (m3 * enhanced.astype(np.float32) + (1.0 - m3) * patch.astype(np.float32)).round()
        out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return out


def blend_reference_rgb(
    base: np.ndarray,
    reference: np.ndarray,
    *,
    alpha: float = 0.12,
) -> np.ndarray:
    """
    Crude whole-image linear blend toward a resized *reference* (color/style pull).
    *alpha* in [0, 0.5] recommended; no learned identity lock.
    """
    if base.ndim != 3 or base.shape[2] != 3:
        return base
    a = float(np.clip(alpha, 0.0, 0.5))
    if a <= 0:
        return base

    try:
        from PIL import Image
    except ImportError:
        return base

    b = np.clip(np.asarray(base), 0, 255).astype(np.uint8)
    H, W = b.shape[:2]
    ref = np.clip(np.asarray(reference), 0, 255).astype(np.uint8)
    if ref.ndim != 3 or ref.shape[2] != 3:
        return base
    ref_im = Image.fromarray(ref, mode="RGB").resize((W, H), Image.Resampling.LANCZOS)
    r = np.asarray(ref_im, dtype=np.float32)
    out = (1.0 - a) * b.astype(np.float32) + a * r
    return np.clip(out.round(), 0, 255).astype(np.uint8)
