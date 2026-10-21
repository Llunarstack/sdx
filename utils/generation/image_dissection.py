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

Safe to import without ``torch``. Region extraction reuses the cached Grounding DINO / SAM2 pipelines of
``segmentation_to_mask`` and falls back to a centre box when they are not installed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ImageLike = str | Path
BBox = tuple[int, int, int, int]  # (x0, y0, x1, y1)


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
    bbox: BBox | None = None
    mask_path: str | None = None
    crop_path: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


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


def parse_part_requests(prompt: str, *, default_source_index: int = 0) -> list[PartRequest]:
    """
    Extract PartRequest entries from a user prompt.

    This is intentionally conservative: if we can't parse a request, we don't invent one.
    """
    p = (prompt or "").strip()
    if not p:
        return []

    reqs: list[PartRequest] = []

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
    out: list[PartRequest] = []
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
) -> list[str]:
    """
    Build short textual "facts" describing what to extract, suitable for RAG merging.
    """
    facts: list[str] = []
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
    output_dir: str | Path,
    default_source_index: int = 0,
    enable_heavy_models: bool = True,
) -> tuple[list[PartRequest], list[DissectedPart], list[str]]:
    """
    High-level helper used by RAG/generation.

    Returns (requests, parts, facts). With ``enable_heavy_models=False`` the parts carry no masks; without
    Grounding DINO a part's mask is the centre box.
    """
    reqs = parse_part_requests(prompt, default_source_index=default_source_index)
    facts = visual_facts_from_requests(prompt, reqs, num_reference_images=len(reference_images))

    parts: list[DissectedPart] = [DissectedPart(request=r) for r in reqs]

    if not enable_heavy_models:
        return reqs, parts, facts

    import numpy as np
    from PIL import Image

    from utils.generation.segmentation_to_mask import detect_box_for_phrase, mask_for_box

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_mask_and_crop(pil: Image.Image, mask: Image.Image, stem: str) -> tuple[str, str]:
        mask_p, crop_p = out_dir / f"{stem}_mask.png", out_dir / f"{stem}_crop.png"
        mask.save(mask_p)
        alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
        Image.fromarray((np.asarray(pil, dtype=np.float32) * alpha).round().astype(np.uint8)).save(crop_p)
        return str(mask_p), str(crop_p)

    for i, dp in enumerate(parts):
        r = dp.request
        if not 0 <= r.source_index < len(reference_images):
            dp.metadata["error"] = "bad_source_index"
            continue
        pil = Image.open(Path(reference_images[r.source_index])).convert("RGB")
        w, h = pil.size
        stem = f"ref{r.source_index + 1}_{re.sub(r'[^a-z0-9]+', '_', r.part.lower()).strip('_')}_{i:02d}"

        if r.role == "background" or r.part.lower().strip() == "background":
            dp.mask_path, dp.crop_path = _save_mask_and_crop(pil, Image.new("L", (w, h), 255), stem)
            dp.bbox = (0, 0, w, h)
            dp.confidence = 1.0
            dp.metadata["mode"] = "background_full"
            continue

        box, score = detect_box_for_phrase(pil, r.part, threshold=0.15)
        if box is None:
            # No detector or no hit: the central 60% of the image stands in for the part.
            box, score = (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)), 0.05
            dp.metadata["bbox_fallback"] = True
        dp.bbox = box
        mask, sam_used = mask_for_box(pil, box)
        dp.mask_path, dp.crop_path = _save_mask_and_crop(pil, mask, stem)
        dp.confidence = float(max(0.0, min(1.0, score)))
        dp.metadata["mode"] = ("center" if dp.metadata.get("bbox_fallback") else "gdino") + (
            "+sam" if sam_used else "+bbox"
        )

    return reqs, parts, facts


__all__ = [
    "PartRequest",
    "DissectedPart",
    "parse_part_requests",
    "visual_facts_from_requests",
    "dissect_images_to_parts",
]
