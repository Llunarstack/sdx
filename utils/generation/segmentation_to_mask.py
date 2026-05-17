"""
**Text → inpaint mask** for regional edits.

Tries pretrained **Grounding DINO** (box) + **SAM2** (mask refinement) under ``pretrained/`` when available;
otherwise falls back to :mod:`edit_masks` heuristics from simple phrase keywords.

Mask convention matches ``sample.py`` / MDM inpaint: **white** = regenerate, **black** = preserve.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from PIL import Image, ImageFilter

from utils.generation.edit_masks import heuristic_inpaint_mask, normalize_heuristic_region

_PRETRAINED = Path(__file__).resolve().parents[2] / "pretrained"
GDINO_REPO_PATH = _PRETRAINED / "GroundingDINO-Base"
SAM2_REPO_PATH = _PRETRAINED / "SAM2-Hiera-Large"

BBox = Tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class SegmentationMaskResult:
    mask: Image.Image  # mode "L"
    bbox: Optional[BBox]
    mode: str
    notes: str = ""


_PHRASE_TO_REGION_TERMS = (
    (("background", "backdrop", "sky behind", "environment"), "background"),
    (("face", "facial", "eyes", "eyebrow", "mouth", "lips"), "face"),
    (("hands", "fingers", "palm"), "hands"),
    (("outfit", "dress", "shirt", "jacket", "clothing", "pants"), "clothing"),
)


def phrase_to_fallback_region(text: str) -> Optional[str]:
    """If *text* looks like an edit noun phrase, map to a coarse heuristic region."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if re.search(r"\b(full|everything|whole image|entire image)\b", t):
        return "full"
    for terms, reg in _PHRASE_TO_REGION_TERMS:
        if any(x in t for x in terms):
            return reg
    return None


def _pick_device_preference() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


_gdino_pl: Any = None
_sam_pl: Any = None
_gdino_unavailable = object()
_sam_unavailable = object()


def _load_gdino():
    global _gdino_pl
    if _gdino_pl is _gdino_unavailable:
        return None
    if _gdino_pl is not None:
        return _gdino_pl
    try:
        import transformers
    except ImportError:
        _gdino_pl = _gdino_unavailable
        return None
    if not GDINO_REPO_PATH.exists():
        _gdino_pl = _gdino_unavailable
        return None
    dev = _pick_device_preference()
    dev_id = 0 if dev.startswith("cuda") else -1
    try:
        pl = transformers.pipeline(
            task="zero-shot-object-detection",
            model=str(GDINO_REPO_PATH),
            device=dev_id,
        )
    except Exception:
        _gdino_pl = _gdino_unavailable
        return None
    _gdino_pl = pl
    return _gdino_pl


def _load_sam2():
    global _sam_pl
    if _sam_pl is _sam_unavailable:
        return None
    if _sam_pl is not None:
        return _sam_pl
    try:
        import transformers
    except ImportError:
        _sam_pl = _sam_unavailable
        return None
    if not SAM2_REPO_PATH.exists():
        _sam_pl = _sam_unavailable
        return None
    dev = _pick_device_preference()
    dev_id = 0 if dev.startswith("cuda") else -1
    try:
        pl = transformers.pipeline(
            task="mask-generation",
            model=str(SAM2_REPO_PATH),
            device=dev_id,
        )
    except Exception:
        _sam_pl = _sam_unavailable
        return None
    _sam_pl = pl
    return _sam_pl


def _clip_box(box: BBox, w: int, h: int) -> BBox:
    x0, y0, x1, y1 = box
    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(x0 + 1, min(w, int(x1)))
    y1 = max(y0 + 1, min(h, int(y1)))
    return (x0, y0, x1, y1)


def _bbox_to_mask(size: Tuple[int, int], box: BBox) -> Image.Image:
    assert np is not None
    w, h = size
    x0, y0, x1, y1 = _clip_box(box, w, h)
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 255
    return Image.fromarray(m, mode="L")


def _mask_from_sam_output(raw: object, fallback: Image.Image, w: int, h: int) -> Image.Image:
    """Normalize SAM pipeline output → ``L`` mask."""
    assert np is not None
    if isinstance(raw, list) and raw:
        first = raw[0]
        blob = first.get("mask") if isinstance(first, dict) else None
        if blob is None:
            return fallback
        if hasattr(blob, "convert"):
            m = blob.convert("L").resize((w, h), Image.Resampling.NEAREST)
            return m
        ma = np.array(blob).astype(np.uint8)
        if ma.ndim == 3:
            ma = ma[..., 0]
        m_arr = ((ma > 0).astype(np.uint8)) * 255
        return Image.fromarray(m_arr, mode="L").resize((w, h), Image.Resampling.NEAREST)
    return fallback


def detect_box_for_phrase(
    pil_rgb: Image.Image,
    phrase: str,
    *,
    threshold: float = 0.2,
) -> Tuple[Optional[BBox], float]:
    """Run Grounding DINO if available; return (bbox, score)."""
    gdino = _load_gdino()
    w, h = pil_rgb.size
    if gdino is None:
        return None, 0.0
    try:
        det = gdino(pil_rgb, candidate_labels=[phrase.strip()], threshold=float(threshold))
        if isinstance(det, list) and det:
            best = max(det, key=lambda d: float(d.get("score", 0.0)))
            b = best.get("box") or {}
            box = (
                int(b.get("xmin", 0)),
                int(b.get("ymin", 0)),
                int(b.get("xmax", w)),
                int(b.get("ymax", h)),
            )
            return _clip_box(box, w, h), float(best.get("score", 0.0))
    except Exception:
        return None, 0.0
    return None, 0.0


def build_segmentation_mask_for_edit(
    image: Union[Image.Image, str, Path],
    target_phrase: str,
    *,
    feather_radius: float = 4.0,
    gdino_threshold: float = 0.2,
    use_vision_models: bool = True,
) -> SegmentationMaskResult:
    """
    Build a grayscale inpaint mask for *target_phrase* (“the face”, “the sword”, …).

    Args:
        image: RGB source (or path).
        target_phrase: Object / region wording for detection (short is better).
        feather_radius: Gaussian blur radius on ``L`` mask (0 to skip).
        gdino_threshold: Detection confidence floor for Grounding DINO.
        use_vision_models: If False, only keyword → heuristic_masks path.
    """
    phrase = (target_phrase or "").strip()
    if not phrase:
        raise ValueError("target_phrase must be non-empty")

    if isinstance(image, (str, Path)):
        pil = Image.open(Path(image)).convert("RGB")
    else:
        pil = image.convert("RGB")

    w, h = pil.size

    heuristic_label = phrase_to_fallback_region(phrase)
    fallback_region = normalize_heuristic_region(heuristic_label) if heuristic_label else None

    if not use_vision_models:
        if fallback_region:
            mask = heuristic_inpaint_mask(w, h, fallback_region, feather_radius=max(0.0, float(feather_radius)))
            return SegmentationMaskResult(mask=mask, bbox=None, mode="heuristic_only", notes="models disabled")
        mask = heuristic_inpaint_mask(w, h, "subject", feather_radius=max(0.0, float(feather_radius)))
        return SegmentationMaskResult(mask=mask, bbox=None, mode="heuristic_subject_default", notes="models disabled")

    box, score = detect_box_for_phrase(pil, phrase, threshold=gdino_threshold)
    bbox_mask: Optional[Image.Image] = None
    if box is None and fallback_region:
        mask = heuristic_inpaint_mask(w, h, fallback_region, feather_radius=max(0.0, float(feather_radius)))
        return SegmentationMaskResult(
            mask=mask,
            bbox=None,
            mode="heuristic",
            notes="gdino_miss",
        )

    if box is None:
        mask = heuristic_inpaint_mask(w, h, "subject", feather_radius=max(0.0, float(feather_radius)))
        return SegmentationMaskResult(mask=mask, bbox=None, mode="heuristic_subject_default", notes="gdino_miss")

    assert np is not None
    bbox_mask = _bbox_to_mask((w, h), box)
    sam = _load_sam2()
    if sam is not None:
        try:
            sam_out = sam(pil, input_boxes=[list(box)])
            mask = _mask_from_sam_output(sam_out, bbox_mask, w, h)
            mode = "gdino_plus_sam2"
            notes = f"score={score:.3f}"
        except Exception:
            mask = bbox_mask
            mode = "gdino_bbox_only"
            notes = f"sam_fallback score={score:.3f}"
    else:
        mask = bbox_mask
        mode = "gdino_bbox_only"
        notes = f"score={score:.3f}"

    if feather_radius and feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))

    return SegmentationMaskResult(mask=mask, bbox=box, mode=mode, notes=notes)


__all__ = [
    "GDINO_REPO_PATH",
    "SAM2_REPO_PATH",
    "BBox",
    "SegmentationMaskResult",
    "build_segmentation_mask_for_edit",
    "detect_box_for_phrase",
    "phrase_to_fallback_region",
]
