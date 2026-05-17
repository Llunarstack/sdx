"""
Prompt-driven image dissection helpers.

Goal
----
Given a user prompt and one or more reference images, extract the *parts* the user
explicitly asks for (e.g. "use the hat from image 1 and the background from image 2")
and produce:

1) A structured plan of requested parts.
2) Optional region crops/masks when heavy vision models are available.
3) Textual "visual facts" that can be fed into existing RAG text pipelines
   (`utils.prompt.rag_prompt.merge_facts_into_prompt`).

This module is designed to be safe to import in environments without `torch`.
The heavy region extraction (SAM/GroundingDINO/OwlViT) is **optional** and will
gracefully fall back to plan-only + facts-only behavior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _TORCH_AVAILABLE = False


ImageLike = Union[str, Path]
BBox = Tuple[int, int, int, int]  # (x0, y0, x1, y1)

_PRETRAINED = Path(__file__).resolve().parents[2] / "pretrained"
_GDINO_PATH = _PRETRAINED / "GroundingDINO-Base"
_SAM2_PATH = _PRETRAINED / "SAM2-Hiera-Large"


@dataclass(frozen=True, slots=True)
class PartRequest:
    """A single user-requested part."""

    part: str
    source_index: int = 0  # 0-based reference image index
    role: str = "foreground"  # "foreground" | "background" | "style" | "text" | ...
    hint: str = ""  # freeform extra info


@dataclass(slots=True)
class DissectedPart:
    """Result of attempting to extract a requested part."""

    request: PartRequest
    bbox: Optional[BBox] = None
    mask_path: Optional[str] = None
    crop_path: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


_IMG_REF_RE = re.compile(
    r"\b(?:image|img|ref|reference)\s*(?P<idx>\d+)\b",
    re.IGNORECASE,
)

# Matches patterns like:
# - "use the hat from image 1"
# - "take background from ref2"
# - "use eyes from img 3"
_PART_FROM_IMG_RE = re.compile(
    r"\b(?:use|take|extract|borrow|copy|keep)\s+"
    r"(?:the\s+)?(?P<part>[a-z0-9][a-z0-9 _\\-]{0,48}?)\s+"
    r"(?:from|in)\s+(?:the\s+)?(?:image|img|ref|reference)\s*(?P<idx>\d+)\b",
    re.IGNORECASE,
)

# Background-only shorthand:
_BG_FROM_IMG_RE = re.compile(
    r"\b(?:use|take|keep)\s+(?:the\s+)?background\s+from\s+(?:image|img|ref|reference)\s*(?P<idx>\d+)\b",
    re.IGNORECASE,
)


def parse_part_requests(prompt: str, *, default_source_index: int = 0) -> List[PartRequest]:
    """
    Extract PartRequest entries from a user prompt.

    This is intentionally conservative: if we can't parse a request, we don't invent one.
    """
    p = (prompt or "").strip()
    if not p:
        return []

    reqs: List[PartRequest] = []

    # 1) Explicit "X from image N" requests.
    for m in _PART_FROM_IMG_RE.finditer(p):
        part = (m.group("part") or "").strip()
        idx = int(m.group("idx")) - 1
        if part:
            role = "background" if "background" in part.lower() else "foreground"
            reqs.append(PartRequest(part=part, source_index=max(0, idx), role=role))

    # 2) Background shorthand (avoids "part" being parsed as "background from image").
    for m in _BG_FROM_IMG_RE.finditer(p):
        idx = int(m.group("idx")) - 1
        reqs.append(PartRequest(part="background", source_index=max(0, idx), role="background"))

    # 3) If user references images but doesn't specify parts, emit a soft request.
    # Example: "use image 2 as reference for outfit" (no explicit "from image" phrase).
    if not reqs:
        refs = [int(m.group("idx")) - 1 for m in _IMG_REF_RE.finditer(p)]
        refs = [r for r in refs if r >= 0]
        if refs:
            # Minimal: treat as "style/identity reference" rather than a hard crop request.
            reqs.append(PartRequest(part="reference", source_index=refs[0], role="style"))

    # Deduplicate (same part+idx+role).
    seen = set()
    out: List[PartRequest] = []
    for r in reqs:
        key = (r.part.lower().strip(), int(r.source_index), r.role.lower().strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    # If no explicit refs, allow default implicit reference if prompt uses "use this image" language.
    if not out and re.search(r"\b(use|keep)\s+(this|the)\s+image\b", p, re.IGNORECASE):
        out.append(PartRequest(part="reference", source_index=int(default_source_index), role="style"))

    return out


def visual_facts_from_requests(
    prompt: str,
    requests: Sequence[PartRequest],
    *,
    num_reference_images: int,
) -> List[str]:
    """
    Build short textual "facts" describing what to extract, suitable for RAG merging.
    """
    facts: List[str] = []
    if not requests:
        return facts

    facts.append("Reference-image dissection requested. Treat these as constraints when generating the final image.")
    facts.append(f"Number of reference images provided: {int(num_reference_images)}")

    for r in requests:
        img_n = int(r.source_index) + 1
        facts.append(f"Use `{r.part}` from reference image {img_n} (role={r.role}).")

    # Add the original prompt as a reminder for dissection intent, without duplicating too much.
    short = (prompt or "").strip()
    if short:
        facts.append(f"User intent: {short[:240] + ('...' if len(short) > 240 else '')}")
    return facts


def dissect_images_to_parts(
    prompt: str,
    reference_images: Sequence[ImageLike],
    *,
    output_dir: Union[str, Path],
    default_source_index: int = 0,
    enable_heavy_models: bool = True,
) -> Tuple[List[PartRequest], List[DissectedPart], List[str]]:
    """
    High-level helper used by RAG/generation.

    Returns (requests, parts, facts). If heavy models aren't available, `parts` will
    be empty (or bbox/mask/crop None) but `facts` will still exist.
    """
    reqs = parse_part_requests(prompt, default_source_index=default_source_index)
    facts = visual_facts_from_requests(prompt, reqs, num_reference_images=len(reference_images))

    parts: List[DissectedPart] = [DissectedPart(request=r) for r in reqs]

    # If heavy models are disabled, return plan+facts only.
    # If torch is unavailable, we still support non-ML extraction (background full-mask,
    # bbox-only masks) and only skip *model inference*.
    if not enable_heavy_models:
        return reqs, parts, facts

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to load models via transformers pipelines. If unavailable, fall back to bbox-only masks.
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return reqs, parts, facts

    gdino = None
    sam = None
    device = "cpu"
    if _TORCH_AVAILABLE:
        try:
            import torch as _torch

            device = "cuda:0" if _torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        try:
            import transformers

            if _GDINO_PATH.exists():
                try:
                    gdino = transformers.pipeline(
                        task="zero-shot-object-detection",
                        model=str(_GDINO_PATH),
                        device=0 if device.startswith("cuda") else -1,
                    )
                except Exception:
                    gdino = None

            # SAM2: try the generic mask-generation pipeline. Some installs may not support SAM2 yet.
            if _SAM2_PATH.exists():
                try:
                    sam = transformers.pipeline(
                        task="mask-generation",
                        model=str(_SAM2_PATH),
                        device=0 if device.startswith("cuda") else -1,
                    )
                except Exception:
                    sam = None
        except Exception:
            gdino = None
            sam = None

    def _load_pil(img: ImageLike) -> Image.Image:
        p = Path(img)
        return Image.open(p).convert("RGB")

    def _clip_box(box: BBox, w: int, h: int) -> BBox:
        x0, y0, x1, y1 = box
        x0 = max(0, min(w - 1, int(x0)))
        y0 = max(0, min(h - 1, int(y0)))
        x1 = max(x0 + 1, min(w, int(x1)))
        y1 = max(y0 + 1, min(h, int(y1)))
        return (x0, y0, x1, y1)

    def _bbox_mask(size: Tuple[int, int], box: BBox) -> Image.Image:
        w, h = size
        x0, y0, x1, y1 = _clip_box(box, w, h)
        m = np.zeros((h, w), dtype=np.uint8)
        m[y0:y1, x0:x1] = 255
        return Image.fromarray(m, mode="L")

    def _save_mask_and_crop(
        *,
        pil: Image.Image,
        mask: Image.Image,
        stem: str,
    ) -> Tuple[str, str]:
        mask_p = out_dir / f"{stem}_mask.png"
        crop_p = out_dir / f"{stem}_crop.png"
        mask.save(mask_p)
        # Save crop as RGB (mask kept separately).
        arr = np.array(pil)
        m = np.array(mask).astype(np.float32) / 255.0
        if m.ndim == 2:
            m3 = np.repeat(m[..., None], 3, axis=2)
        else:
            m3 = m
        cut = (arr.astype(np.float32) * m3).round().astype(np.uint8)
        Image.fromarray(cut).save(crop_p)
        return str(mask_p), str(crop_p)

    # Process each request independently.
    for i, dp in enumerate(parts):
        r = dp.request
        if r.source_index < 0 or r.source_index >= len(reference_images):
            dp.metadata["error"] = "bad_source_index"
            continue
        pil = _load_pil(reference_images[r.source_index])
        w, h = pil.size
        stem = f"ref{r.source_index + 1}_{re.sub(r'[^a-z0-9]+', '_', r.part.lower()).strip('_')}_{i:02d}"

        # Background: treat as full-image mask/crop.
        if r.role == "background" or r.part.lower().strip() == "background":
            full = Image.fromarray(np.full((h, w), 255, dtype=np.uint8), mode="L")
            mp, cp = _save_mask_and_crop(pil=pil, mask=full, stem=stem)
            dp.mask_path = mp
            dp.crop_path = cp
            dp.bbox = (0, 0, w, h)
            dp.confidence = 1.0
            dp.metadata["mode"] = "background_full"
            continue

        # Foreground part: try grounding bbox.
        box: Optional[BBox] = None
        score = 0.0
        if gdino is not None:
            try:
                # Candidate label is the requested part.
                det = gdino(pil, candidate_labels=[r.part], threshold=0.15)
                if isinstance(det, list) and det:
                    best = max(det, key=lambda d: float(d.get("score", 0.0)))
                    b = best.get("box") or {}
                    # pipeline returns {"xmin","ymin","xmax","ymax"} in pixels
                    box = (int(b.get("xmin", 0)), int(b.get("ymin", 0)), int(b.get("xmax", w)), int(b.get("ymax", h)))
                    score = float(best.get("score", 0.0))
            except Exception:
                box = None

        if box is None:
            # Fallback: center crop-ish box (not great, but non-crashing).
            cx0, cy0 = int(w * 0.2), int(h * 0.2)
            cx1, cy1 = int(w * 0.8), int(h * 0.8)
            box = (cx0, cy0, cx1, cy1)
            score = 0.05
            dp.metadata["bbox_fallback"] = True

        dp.bbox = _clip_box(box, w, h)

        # Mask refinement: try SAM mask-generation pipeline, else use bbox mask.
        mask_img: Image.Image
        if sam is not None:
            try:
                # For mask-generation pipeline, pass a box prompt when supported.
                # Many implementations accept `points_per_batch` etc; keep minimal.
                out_masks = sam(pil, input_boxes=[list(dp.bbox)])
                # Expected: list of dicts with 'mask' as np array or PIL.
                # Be permissive across versions.
                if isinstance(out_masks, list) and len(out_masks) > 0:
                    first_any: Any = None
                    for _it in out_masks:
                        first_any = _it
                        break
                    first = first_any if isinstance(first_any, dict) else {}
                    m0 = first.get("mask")
                    if m0 is not None:
                        if hasattr(m0, "convert"):
                            mask_img = m0.convert("L")
                        else:
                            ma = np.array(m0).astype(np.uint8)
                            if ma.ndim == 3:
                                ma = ma[..., 0]
                            mask_img = Image.fromarray((ma > 0).astype(np.uint8) * 255, mode="L")
                    else:
                        mask_img = _bbox_mask((w, h), dp.bbox)
                else:
                    mask_img = _bbox_mask((w, h), dp.bbox)
            except Exception:
                mask_img = _bbox_mask((w, h), dp.bbox)
        else:
            mask_img = _bbox_mask((w, h), dp.bbox)

        mp, cp = _save_mask_and_crop(pil=pil, mask=mask_img, stem=stem)
        dp.mask_path = mp
        dp.crop_path = cp
        dp.confidence = float(max(0.0, min(1.0, score)))
        dp.metadata["mode"] = (
            "gdino+sam"
            if sam is not None and gdino is not None
            else ("gdino+bbox" if gdino is not None else "bbox_only")
        )

    return reqs, parts, facts


__all__ = [
    "PartRequest",
    "DissectedPart",
    "parse_part_requests",
    "visual_facts_from_requests",
    "dissect_images_to_parts",
]
