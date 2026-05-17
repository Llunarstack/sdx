"""
Style Genome — structured invented aesthetics for agentic image generation.

A genome is an orthogonal bundle (palette, line, surface, camera, signature) compiled
into prompt / negative / style-conditioning strings for SDX.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from utils.prompt.fast_paths import merge_fragments


@dataclass
class StyleGenome:
    """Invented style identity (not a single artist name clone)."""

    id: str
    name: str
    palette: str = ""
    line: str = ""
    surface: str = ""
    camera: str = ""
    lighting: str = ""
    signature: str = ""
    anti_clone: Tuple[str, ...] = ()
    positive_fragments: Tuple[str, ...] = ()
    negative_fragments: Tuple[str, ...] = ()
    reasoning: str = ""

    def axis_tokens(self) -> List[str]:
        out: List[str] = []
        for part in (self.palette, self.line, self.surface, self.camera, self.lighting, self.signature):
            p = (part or "").strip()
            if p:
                out.append(p)
        out.extend(x for x in self.positive_fragments if (x or "").strip())
        return out

    def style_head_string(self) -> str:
        """T5 style channel text (medium + look, not full scene)."""
        parts = self.axis_tokens()[:8]
        if not parts:
            return self.name.strip()
        return ", ".join(parts)

    def compile_positive(self, base_prompt: str) -> str:
        base = (base_prompt or "").strip()
        extras = ", ".join(self.axis_tokens())
        if extras:
            return merge_fragments(base, extras)
        return base

    def compile_negative(self, base_negative: str) -> str:
        neg = (base_negative or "").strip()
        chunks: List[str] = []
        if self.negative_fragments:
            chunks.append(", ".join(self.negative_fragments))
        if self.anti_clone:
            chunks.append(", ".join(self.anti_clone))
        if chunks:
            return merge_fragments(neg, ", ".join(chunks))
        return neg

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["anti_clone"] = list(self.anti_clone)
        d["positive_fragments"] = list(self.positive_fragments)
        d["negative_fragments"] = list(self.negative_fragments)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StyleGenome:
        def _tuple(key: str) -> Tuple[str, ...]:
            raw = data.get(key) or ()
            if isinstance(raw, str):
                return tuple(x.strip() for x in raw.split(",") if x.strip())
            return tuple(str(x).strip() for x in raw if str(x).strip())

        gid = str(data.get("id") or "").strip() or f"genome_{uuid.uuid4().hex[:10]}"
        return cls(
            id=gid,
            name=str(data.get("name") or "Invented style").strip(),
            palette=str(data.get("palette") or "").strip(),
            line=str(data.get("line") or "").strip(),
            surface=str(data.get("surface") or "").strip(),
            camera=str(data.get("camera") or "").strip(),
            lighting=str(data.get("lighting") or "").strip(),
            signature=str(data.get("signature") or "").strip(),
            anti_clone=_tuple("anti_clone"),
            positive_fragments=_tuple("positive_fragments"),
            negative_fragments=_tuple("negative_fragments"),
            reasoning=str(data.get("reasoning") or "").strip(),
        )


def genome_from_json(text: str) -> StyleGenome:
    data = json.loads(text)
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("Style genome JSON must be an object or non-empty array")
    return StyleGenome.from_dict(data)


def nearest_catalog_style_overlap(genome: StyleGenome) -> Tuple[str, float]:
    """
    Return (best_matching_style_id, overlap_score 0–1) against built-in STYLE_SPECS keywords.
    Higher = closer to an existing catalog style (less novel).
    """
    try:
        from config.defaults.style_guidance import STYLE_SPECS
    except ImportError:
        return "", 0.0

    text = " ".join(genome.axis_tokens()).lower()
    if not text:
        return "", 0.0

    best_id = ""
    best_score = 0.0
    tokens = set(re.findall(r"[a-z0-9]+", text))
    if not tokens:
        return "", 0.0

    for spec in STYLE_SPECS:
        kw_text = ", ".join(spec.keywords)
        try:
            from utils.prompt.style_native import text_overlap

            overlap = text_overlap(" ".join(genome.axis_tokens()), kw_text)
        except Exception:
            kw_tokens: set[str] = set()
            for kw in spec.keywords:
                kw_tokens.update(re.findall(r"[a-z0-9]+", kw.lower()))
            if not kw_tokens:
                continue
            overlap = len(tokens & kw_tokens) / max(len(kw_tokens), 1)
        if overlap > best_score:
            best_score = overlap
            best_id = spec.id

    return best_id, min(1.0, best_score)


def is_genome_novel_enough(genome: StyleGenome, *, max_catalog_overlap: float = 0.55) -> bool:
    _, score = nearest_catalog_style_overlap(genome)
    return score < max_catalog_overlap


__all__ = [
    "StyleGenome",
    "genome_from_json",
    "is_genome_novel_enough",
    "nearest_catalog_style_overlap",
]
