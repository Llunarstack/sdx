"""Image fit helpers for inference-time output sizing."""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from PIL import Image

ResizeMode = Literal["stretch", "center_crop", "saliency_crop"]


def _resize_np(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    pil = Image.fromarray(img.astype(np.uint8), mode="RGB")
    pil = pil.resize((int(out_w), int(out_h)), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def _center_crop_box(h: int, w: int, out_h: int, out_w: int) -> Tuple[int, int, int, int]:
    target_ar = float(out_w) / max(1.0, float(out_h))
    src_ar = float(w) / max(1.0, float(h))
    if src_ar > target_ar:
        ch = h
        cw = max(1, int(round(h * target_ar)))
        y0 = 0
        x0 = max(0, (w - cw) // 2)
    else:
        cw = w
        ch = max(1, int(round(w / max(1e-6, target_ar))))
        x0 = 0
        y0 = max(0, (h - ch) // 2)
    return y0, x0, ch, cw


def _face_boost_map(img: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    try:
        import cv2
    except Exception:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    out = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for x, y, fw, fh in faces:
        x2 = min(img.shape[1], x + fw)
        y2 = min(img.shape[0], y + fh)
        out[y:y2, x:x2] += float(strength)
    return out


def _saliency_map(img: np.ndarray, face_bias: float = 0.0) -> np.ndarray:
    # Lightweight saliency: edge magnitude + local contrast + optional face boost.
    arr = img.astype(np.float32) / 255.0
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gx = np.abs(np.gradient(gray, axis=1))
    gy = np.abs(np.gradient(gray, axis=0))
    edge = gx + gy
    contrast = np.abs(gray - gray.mean())
    sal = 0.7 * edge + 0.3 * contrast
    if face_bias > 0:
        sal = sal + _face_boost_map(img, face_bias)
    sal = sal - sal.min()
    m = float(sal.max())
    if m > 1e-8:
        sal = sal / m
    return sal.astype(np.float32)


def _best_saliency_crop_box(img: np.ndarray, out_h: int, out_w: int, face_bias: float = 0.0) -> Tuple[int, int, int, int]:
    h, w = img.shape[:2]
    y0, x0, ch, cw = _center_crop_box(h, w, out_h, out_w)
    sal = _saliency_map(img, face_bias=face_bias)
    if cw == w and ch == h:
        return y0, x0, ch, cw
    if ch == h:
        # Horizontal sliding window
        col = sal.sum(axis=0)
        csum = np.concatenate([[0.0], np.cumsum(col)])
        best_x, best_val = x0, -1e30
        for x in range(0, max(1, w - cw + 1)):
            v = float(csum[x + cw] - csum[x])
            if v > best_val:
                best_val, best_x = v, x
        return 0, int(best_x), ch, cw
    # Vertical sliding window
    row = sal.sum(axis=1)
    csum = np.concatenate([[0.0], np.cumsum(row)])
    best_y, best_val = y0, -1e30
    for y in range(0, max(1, h - ch + 1)):
        v = float(csum[y + ch] - csum[y])
        if v > best_val:
            best_val, best_y = v, y
    return int(best_y), 0, ch, cw


def fit_image_to_size(
    img: np.ndarray,
    out_h: int,
    out_w: int,
    *,
    mode: ResizeMode = "stretch",
    saliency_face_bias: float = 0.0,
) -> np.ndarray:
    """Resize RGB uint8 image using stretch, center-crop, or saliency-crop."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("fit_image_to_size expects HxWx3 image")
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output size must be positive")
    h, w = img.shape[:2]
    if h == out_h and w == out_w:
        return img.astype(np.uint8)
    m = str(mode or "stretch").lower()
    if m == "stretch":
        return _resize_np(img, out_h, out_w)
    if m == "center_crop":
        y0, x0, ch, cw = _center_crop_box(h, w, out_h, out_w)
    elif m == "saliency_crop":
        y0, x0, ch, cw = _best_saliency_crop_box(img, out_h, out_w, face_bias=float(saliency_face_bias))
    else:
        raise ValueError(f"Unknown resize mode: {mode}")
    crop = img[y0 : y0 + ch, x0 : x0 + cw]
    return _resize_np(crop, out_h, out_w)

