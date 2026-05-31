"""
**Scene brief** — synthesized plan of what the final image should contain.

The brain merges user prompt constraints, reference understanding (OCR/VLM),
dissection facts, and RAG into a single artifact the generator follows.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from utils.prompt.rag_prompt import merge_facts_into_prompt

from .understand import ImageUnderstanding


@dataclass(slots=True)
class SceneElement:
    """One thing that must appear (or be preserved) in the output."""

    name: str
    source: str = "user_prompt"
    description: str = ""
    reference_index: int = -1
    must_include: bool = True
    ocr_text: str = ""


@dataclass(slots=True)
class SceneBrief:
    """What the Visual Brain decided belongs in the final image."""

    user_prompt: str
    negative_prompt: str = ""
    elements: List[SceneElement] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    reference_paths: List[str] = field(default_factory=list)
    control_image: str = ""
    init_image: str = ""
    inpaint_mask: str = ""
    expected_text: str = ""
    enriched_prompt: str = ""
    reasoning: str = ""

    def to_facts(self) -> List[str]:
        """Flatten brief into RAG-style fact strings."""
        facts = list(self.facts)
        facts.append(f"Primary user request (must satisfy): {self.user_prompt.strip()}")
        for el in self.elements:
            if el.must_include:
                facts.append(f"Include {el.name}: {el.description or el.source}".strip())
            if el.ocr_text:
                facts.append(f"Legible text from reference: {el.ocr_text[:120]}")
        if self.expected_text:
            facts.append(f"Rendered text must read: {self.expected_text}")
        return facts

    def build_generation_prompt(self) -> str:
        """Merge user prompt + facts for T5 encoding."""
        if self.enriched_prompt.strip():
            base = self.enriched_prompt.strip()
        else:
            base = self.user_prompt.strip()
        merged = merge_facts_into_prompt(base, self.to_facts())
        return merged.strip()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SceneBrief":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        els = [SceneElement(**e) for e in data.pop("elements", [])]
        return cls(elements=els, **data)


def synthesize_scene_brief(
    user_prompt: str,
    *,
    understandings: Sequence[ImageUnderstanding],
    dissection_facts: Optional[Sequence[str]] = None,
    rag_facts: Optional[Sequence[str]] = None,
    expected_text: str = "",
    negative_prompt: str = "",
    init_image: str = "",
    inpaint_mask: str = "",
    control_image: str = "",
    creative_enriched: str = "",
    creative_reasoning: str = "",
) -> SceneBrief:
    """
    Merge user prompt + reference understanding into a scene brief.
    """
    refs = [u.path for u in understandings if u.path]
    facts: List[str] = list(rag_facts or []) + list(dissection_facts or [])
    elements: List[SceneElement] = [
        SceneElement(name="user_subject", source="user_prompt", description=user_prompt.strip(), must_include=True)
    ]

    for i, u in enumerate(understandings):
        desc = u.caption or f"reference {i + 1}"
        elements.append(
            SceneElement(
                name=f"ref_{i + 1}",
                source=u.source,
                description=desc[:400],
                reference_index=i,
                must_include=False,
                ocr_text=u.ocr_text[:200] if u.ocr_text else "",
            )
        )
        if u.caption:
            facts.append(f"Reference image {i + 1} shows: {u.caption[:300]}")
        if u.ocr_text:
            facts.append(f"Reference image {i + 1} text: {u.ocr_text[:200]}")

    ctrl = control_image
    if not ctrl:
        for u in understandings:
            if u.control_maps.get("canny"):
                ctrl = u.control_maps["canny"]
                break

    return SceneBrief(
        user_prompt=user_prompt.strip(),
        negative_prompt=negative_prompt.strip(),
        elements=elements,
        facts=facts,
        reference_paths=refs,
        control_image=ctrl,
        init_image=init_image,
        inpaint_mask=inpaint_mask,
        expected_text=str(expected_text or "").strip(),
        enriched_prompt=creative_enriched.strip(),
        reasoning=creative_reasoning.strip(),
    )


def prompt_coverage_score(brief: SceneBrief, metrics: Dict[str, float]) -> float:
    """
    Heuristic: how well verify metrics + OCR align with user intent.
    """
    comp = float(metrics.get("composite", 0.0) or 0.0)
    clip = float(metrics.get("clip", 0.0) or 0.0)
    ocr = float(metrics.get("ocr_match", 1.0) or 1.0)
    base = 0.45 * comp + 0.35 * min(1.0, max(0.0, (clip + 0.2) / 0.6)) + 0.20 * ocr
    if brief.expected_text and ocr < 0.65:
        base *= 0.85
    return float(max(0.0, min(1.0, base)))


__all__ = [
    "SceneBrief",
    "SceneElement",
    "prompt_coverage_score",
    "synthesize_scene_brief",
]
