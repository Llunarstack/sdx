"""
Scorers for the competitor-weakness eval harness.

Every scorer returns ``(score | None, detail)`` where ``score`` is in [0, 1]
(higher = better) or ``None`` when the backend needed to grade that item is not
available in this environment. The harness reports coverage explicitly, so a
missing OCR engine or unbuilt native counter degrades the report rather than
crashing it.

Backends and their availability:
  * CLIP adherence  -> always (transformers CLIP; weights usually cached).
  * Object count    -> native object counter (native/rust) when built, else None.
  * In-image text   -> pytesseract/easyocr when installed, else None.
"""

from __future__ import annotations

import re

import numpy as np

# --- CLIP adherence ---------------------------------------------------------

_CLIP: dict = {}


def _clip(model_id: str, device: str):
    key = (model_id, device)
    if key in _CLIP:
        return _CLIP[key]
    import torch
    from transformers import CLIPModel, CLIPProcessor

    proc = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _CLIP[key] = (model, proc, torch)
    return _CLIP[key]


def clip_adherence(
    image_path: str,
    prompt: str,
    *,
    model_id: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
) -> tuple[float | None, dict]:
    """
    CLIP image-text cosine, remapped from ~[-1,1] to [0,1].

    Note: CLIP is bag-of-words-ish and largely blind to negation and exact
    counts; it measures *topical* adherence, which is why the harness pairs it
    with dedicated count/text scorers for those categories.
    """
    try:
        from PIL import Image

        model, proc, torch = _clip(model_id, device)
        img = Image.open(image_path).convert("RGB")
        inputs = proc(text=[prompt], images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        from utils.generation.clip_alignment import _clip_feature_tensor

        with torch.inference_mode():
            imf = _clip_feature_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
            txf = _clip_feature_tensor(
                model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
            )
            imf = imf / (imf.norm(dim=-1, keepdim=True) + 1e-8)
            txf = txf / (txf.norm(dim=-1, keepdim=True) + 1e-8)
            cos = float((imf * txf).sum(-1).item())
        return max(0.0, min(1.0, (cos + 1.0) / 2.0)), {"cosine": cos}
    except Exception as e:  # pragma: no cover - env dependent
        return None, {"error": f"{type(e).__name__}: {e}"}


# --- Object count -----------------------------------------------------------


def count_objects(image_path: str, *, min_area_frac: float = 0.005) -> int | None:
    """
    Count salient foreground blobs via connected components.

    Prefers the native counter (native/rust, when built); falls back to a NumPy
    connected-components implementation so the harness works today. Returns None
    only if the image can't be read.
    """
    try:
        from PIL import Image

        arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    except Exception:
        return None

    # Try native accelerator first (Rust sdx_image_ops); fall back to NumPy.
    n = _native_count_blobs(arr, min_area_frac)
    if n is not None:
        return int(n)
    return _count_blobs_numpy(arr, min_area_frac=min_area_frac)


def _ensure_native_on_path() -> None:
    """Best-effort: put ``native/_experimental/python`` on sys.path for sdx_native."""
    import sys
    from pathlib import Path

    # scorers.py is at scripts/tools/eval/ -> repo root is parents[3].
    root = Path(__file__).resolve().parents[3]
    native_py = root / "native" / "_experimental" / "python"
    if native_py.is_dir() and str(native_py) not in sys.path:
        sys.path.insert(0, str(native_py))


def _native_count_blobs(arr: np.ndarray, min_area_frac: float) -> int | None:
    try:
        _ensure_native_on_path()
        from sdx_native.image_ops_native import count_blobs as _native_count  # type: ignore

        return _native_count(arr, min_area_frac=min_area_frac)
    except Exception:
        return None


def _otsu_threshold(gray: np.ndarray) -> float:
    """Otsu threshold in [0,1] from a 256-bin histogram."""
    hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    total = gray.size
    sum_all = float(np.dot(np.arange(256), hist))
    w_b = 0.0
    sum_b = 0.0
    best_t, best_var = 0, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    # Place the threshold between bins so a class sitting exactly on best_t
    # (e.g. flat-colored objects) lands on the intended side of the comparison.
    return (best_t + 0.5) / 255.0


def _count_blobs_numpy(gray: np.ndarray, *, min_area_frac: float) -> int:
    """
    Reference connected-components blob count.

    Objects are the foreground class; the *background* is inferred from which
    class dominates the image border (robust to bright- or dark-background
    images). Components touching the border, or larger than 60% of the frame,
    are treated as background/leaks and not counted.
    """
    h, w = gray.shape
    total = gray.size
    thr = _otsu_threshold(gray)
    dark = gray < thr  # one of {dark, light} is the object class

    # Background = the class that occupies most of the border ring.
    border = np.concatenate([dark[0, :], dark[-1, :], dark[:, 0], dark[:, -1]])
    border_dark_frac = float(border.mean()) if border.size else 0.5
    # If the border is mostly dark, background is dark -> objects are light.
    mask = ~dark if border_dark_frac >= 0.5 else dark

    min_area = max(1, int(min_area_frac * total))
    max_area = int(0.60 * total)
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    count = 0
    stack: list[tuple[int, int]] = []
    for i in range(h):
        for j in range(w):
            if mask[i, j] and labels[i, j] == 0:
                cur += 1
                area = 0
                touches_border = False
                stack.append((i, j))
                labels[i, j] = cur
                while stack:
                    y, x = stack.pop()
                    area += 1
                    if y == 0 or x == 0 or y == h - 1 or x == w - 1:
                        touches_border = True
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = cur
                            stack.append((ny, nx))
                if min_area <= area <= max_area and not touches_border:
                    count += 1
    return count


def count_score(image_path: str, expected_count: int) -> tuple[float | None, dict]:
    """1.0 if detected == expected, decaying with absolute error."""
    n = count_objects(image_path)
    if n is None:
        return None, {"detected": None}
    err = abs(n - int(expected_count))
    score = 1.0 / (1.0 + err)  # 1.0 exact, 0.5 off-by-one, ...
    return score, {"detected": n, "expected": int(expected_count), "abs_error": err}


# --- In-image text (OCR) ----------------------------------------------------


def _ocr_available() -> str | None:
    try:
        import pytesseract  # noqa: F401

        return "pytesseract"
    except Exception:
        pass
    try:
        import easyocr  # noqa: F401

        return "easyocr"
    except Exception:
        pass
    return None


def _normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def text_score(image_path: str, expected_text: str) -> tuple[float | None, dict]:
    """
    Fraction of the expected string recovered by OCR (character-level containment).
    Returns None if no OCR engine is installed.
    """
    engine = _ocr_available()
    if engine is None:
        return None, {"engine": None}
    try:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        if engine == "pytesseract":
            import pytesseract

            got = pytesseract.image_to_string(img)
        else:
            import easyocr
            import numpy as _np

            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            got = " ".join(reader.readtext(_np.asarray(img), detail=0))
    except Exception as e:  # pragma: no cover - env dependent
        return None, {"engine": engine, "error": str(e)}

    want = _normalize_text(expected_text)
    have = _normalize_text(got)
    if not want:
        return None, {"engine": engine}
    score = 1.0 if want in have else _lcs_ratio(want, have)
    return score, {"engine": engine, "expected": expected_text, "ocr": got.strip()[:80]}


# --- Scene physics (optional; utils.quality.scene_physics may not exist yet) -


def occlusion_score(image_path: str) -> tuple[float | None, dict]:
    """Cheap occlusion plausibility. None when scene_physics is not importable."""
    try:
        from utils.quality.scene_physics import occlusion_score as _score
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}
    try:
        return _score(image_path)
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}


def reflection_score(image_path: str) -> tuple[float | None, dict]:
    """Cheap reflection plausibility. None when scene_physics is not importable."""
    try:
        from utils.quality.scene_physics import reflection_score as _score
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}
    try:
        return _score(image_path)
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}


def spatial_bind_score(image_path: str, prompt: str) -> tuple[float | None, dict]:
    """Left/right color placement. None when no parseable color pair or import fails."""
    try:
        from utils.quality.scene_physics import spatial_bind_score as _score
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}
    try:
        return _score(image_path, prompt)
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}


def _lcs_ratio(a: str, b: str) -> float:
    if not a:
        return 0.0
    # Longest common substring ratio (cheap, good enough for short target strings).
    best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            best = max(best, k)
    return best / len(a)


def availability(model_id: str = "openai/clip-vit-base-patch32", device: str = "cpu") -> dict:
    """Report which scorers can actually grade in this environment."""
    clip_ok = False
    try:
        _clip(model_id, device)
        clip_ok = True
    except Exception:
        clip_ok = False
    native = False
    try:
        _ensure_native_on_path()
        from sdx_native.image_ops_native import available as _native_available

        native = bool(_native_available())
    except Exception:
        native = False
    return {
        "clip_adherence": clip_ok,
        "count": True,  # always available (numpy fallback)
        "count_native": native,
        "text_ocr": _ocr_available(),
        "occlusion": True,
        "reflection": True,
    }


__all__ = [
    "clip_adherence",
    "count_objects",
    "count_score",
    "text_score",
    "occlusion_score",
    "reflection_score",
    "spatial_bind_score",
    "availability",
]
