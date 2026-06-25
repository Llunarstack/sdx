"""Kling/Runway-style Elements library — tagged multi-ref subjects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "ElementDef",
    "ElementsLibrary",
    "compile_element_refs",
    "parse_elements",
    "resolve_element_images",
]


@dataclass(slots=True)
class ElementDef:
    """Reusable visual DNA: character, location, style, or prop."""

    id: str
    images: List[str] = field(default_factory=list)
    video_ref: str = ""
    bind_subject: bool = False
    reference_sheet: bool = False
    role: str = "character"
    tags: List[str] = field(default_factory=list)
    reference_strength: float = 0.85
    negative: str = ""
    text_hint: str = ""


@dataclass(slots=True)
class ElementsLibrary:
    elements: Dict[str, ElementDef] = field(default_factory=dict)

    def get(self, element_id: str) -> Optional[ElementDef]:
        return self.elements.get(element_id)

    def ids(self) -> List[str]:
        return list(self.elements.keys())


def parse_elements(raw: Any) -> ElementsLibrary:
    lib = ElementsLibrary()
    if not isinstance(raw, Mapping):
        return lib
    for k, v in raw.items():
        eid = str(k)
        if isinstance(v, str):
            lib.elements[eid] = ElementDef(id=eid, images=[v], role="character")
            continue
        if not isinstance(v, Mapping):
            continue
        imgs = v.get("images") or v.get("refs") or []
        if isinstance(imgs, str):
            imgs = [imgs]
        single = str(v.get("image") or v.get("ref") or "")
        if single and single not in imgs:
            imgs = [single] + list(imgs)
        lib.elements[eid] = ElementDef(
            id=eid,
            images=[str(x) for x in imgs if str(x).strip()],
            video_ref=str(v.get("video_ref") or v.get("video") or v.get("clip") or ""),
            bind_subject=bool(v.get("bind_subject") or v.get("bind") or v.get("lock_subject")),
            reference_sheet=bool(v.get("reference_sheet") or v.get("sheet")),
            role=str(v.get("role") or v.get("type") or "character"),
            tags=[str(t) for t in (v.get("tags") or [])],
            reference_strength=float(v.get("reference_strength") or v.get("strength") or 0.85),
            negative=str(v.get("negative") or ""),
            text_hint=str(v.get("text") or v.get("hint") or ""),
        )
    return lib


def resolve_element_images(
    element: ElementDef,
    work_dir: str | Path,
    *,
    max_refs: int = 3,
) -> List[str]:
    """Return up to ``max_refs`` image paths; optionally build reference sheet."""
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    paths = [p for p in element.images if p and Path(p).is_file()]
    if element.reference_sheet and len(paths) <= 1 and paths:
        from .reference_sheet import build_reference_sheet

        sheet = build_reference_sheet(paths[0], wd / f"sheet_{element.id}")
        paths = [str(p) for p in sheet.views if Path(p).is_file()]
    return paths[:max_refs]


def compile_element_refs(
    library: ElementsLibrary,
    element_ids: Sequence[str],
    work_dir: str | Path,
) -> tuple[List[str], List[str], List[str]]:
    """
    Compile element IDs → (image_paths, weights, video_refs).

    ``weights`` are ``path:strength`` strings for ``--style-ref``.
    """
    images: List[str] = []
    weights: List[str] = []
    videos: List[str] = []
    for eid in element_ids:
        el = library.get(eid)
        if not el:
            continue
        refs = resolve_element_images(el, work_dir)
        for rp in refs:
            if rp not in images:
                images.append(rp)
                weights.append(f"{rp}:{el.reference_strength:.2f}")
        if el.video_ref and Path(el.video_ref).is_file():
            videos.append(el.video_ref)
    return images, weights, videos
