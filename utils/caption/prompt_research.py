"""Research the best diffusion prompt for an image via uncensored VLM + local RAG.

Instead of reverse image search (SauceNAO/TinEye), use your own booru corpus:
  1. VLM describes the image (explicit / NSFW-safe — no refusals)
  2. TF-IDF RAG retrieves similar captions/tags from ``rag_corpus.jsonl``
  3. Creative RAG (moondream + Qwen) synthesizes a training-quality prompt

This is the recommended path for unknown images when you already scrape danbooru/e621.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

# Explicit prompt for caption models — reduces refusals on NSFW training data.
_UNCENSORED_VLM_PROMPT = (
    "Describe this image exhaustively for AI image training. "
    "Include every visible subject, character (name them if recognizable from anime/games), "
    "series or franchise, artist style, pose, clothing, body, expression, setting, lighting, "
    "composition, colors, and any mature or explicit details without censoring or omitting. "
    "End with comma-separated booru-style tags."
)

_DIFFUSION_SEED_FALLBACK = "high quality detailed illustration"


@dataclass
class PromptResearchResult:
    diffusion_prompt: str
    negative_prompt: str = ""
    image_description: str = ""
    scene_summary: str = ""
    retrieved_facts: List[str] = field(default_factory=list)
    reasoning: str = ""
    sources: List[str] = field(default_factory=list)
    fallback_used: bool = False

    def to_manifest_patch(self) -> dict:
        out = {
            "caption": self.diffusion_prompt,
            "scene_summary": self.scene_summary or self.image_description,
            "tag_sources": list(self.sources),
        }
        if self.negative_prompt.strip():
            out["negative_caption"] = self.negative_prompt.strip()
            out["negative_prompt_hint"] = self.negative_prompt.strip()
        return out


def _tags_from_description(text: str, *, max_tags: int = 24) -> str:
    """Pull trailing comma-tags from a VLM description."""
    t = (text or "").strip()
    if not t:
        return ""
    if "," in t:
        tail = t.split(",")[-12:]
        tags = [x.strip() for x in ",".join(tail).split(",") if x.strip()]
        if len(tags) >= 3:
            return ", ".join(tags[:max_tags])
    return t


def _seed_from_description(description: str, seed_prompt: str = "") -> str:
    seed = (seed_prompt or "").strip()
    if seed:
        return seed
    desc = (description or "").strip()
    if not desc:
        return _DIFFUSION_SEED_FALLBACK
    # First sentence or first 200 chars as intent seed for Creative RAG.
    first = re.split(r"[.!?\n]", desc, maxsplit=1)[0].strip()
    return first[:240] if first else _DIFFUSION_SEED_FALLBACK


def describe_image_uncensored(image_path: str | Path, *, device: str = "cuda") -> str:
    """Dense VLM description tuned for training captions (no content refusal)."""
    try:
        from utils.brain.understand import caption_image_vlm

        return caption_image_vlm(
            str(image_path),
            user_prompt=_UNCENSORED_VLM_PROMPT,
            device=device,
        ).strip()
    except Exception:
        return ""


def retrieve_rag_facts(
    query: str,
    corpus_path: str | Path,
    *,
    top_k: int = 8,
) -> List[str]:
    if not query.strip():
        return []
    try:
        from utils.prompt.rag_prompt import retrieve_facts_for_query_local

        return retrieve_facts_for_query_local(query, corpus_path, top_k=top_k)
    except Exception:
        return []


def research_prompt_for_image(
    image_path: str | Path,
    *,
    rag_corpus: Optional[str | Path] = None,
    seed_prompt: str = "",
    creativity_level: float = 0.45,
    top_k: int = 8,
    device: str = "cuda",
    use_rag: bool = True,
    use_creative_rag: bool = True,
) -> PromptResearchResult:
    """Image → researched diffusion prompt using VLM + local RAG + Creative RAG."""
    p = Path(image_path)
    sources: List[str] = []
    image_description = describe_image_uncensored(p, device=device) if p.is_file() else ""
    if image_description:
        sources.append("vlm_uncensored")

    query = " ".join(x for x in (seed_prompt, image_description) if x).strip()
    facts: List[str] = []
    if use_rag and rag_corpus and query:
        facts = retrieve_rag_facts(query, rag_corpus, top_k=top_k)
        if facts:
            sources.append(f"rag:{len(facts)}")

    seed = _seed_from_description(image_description, seed_prompt)
    negative = ""
    reasoning = ""
    fallback_used = False
    diffusion_prompt = ""

    if use_creative_rag and p.is_file():
        try:
            from utils.prompt.creative_rag import CreativeRAGEngine

            engine = CreativeRAGEngine(device=device)
            result = engine.enrich(
                seed,
                reference_image_path=str(p),
                facts=facts,
                creativity_level=creativity_level,
                use_image_dissection=False,
            )
            diffusion_prompt = (result.enriched_prompt or "").strip()
            negative = (result.negative_additions or "").strip()
            reasoning = (result.reasoning or "").strip()
            fallback_used = bool(result.fallback_used)
            if result.image_description and not image_description:
                image_description = result.image_description
            if result.retrieved_facts:
                facts = list(result.retrieved_facts)
            if diffusion_prompt:
                sources.append("creative_rag")
        except Exception:
            pass

    if not diffusion_prompt:
        fallback_used = True
        parts = []
        if image_description:
            parts.append(_tags_from_description(image_description) or image_description[:400])
        if facts:
            parts.append(", ".join(facts[:3]))
        diffusion_prompt = ", ".join(p for p in parts if p).strip() or seed
        reasoning = reasoning or "VLM + RAG fallback (Creative RAG models unavailable)."
        sources.append("rag_fallback")

    scene_summary = image_description
    if facts and scene_summary:
        scene_summary = f"{scene_summary}\n\nSimilar corpus entries: " + "; ".join(facts[:4])

    return PromptResearchResult(
        diffusion_prompt=diffusion_prompt,
        negative_prompt=negative,
        image_description=image_description,
        scene_summary=scene_summary,
        retrieved_facts=facts,
        reasoning=reasoning,
        sources=sources,
        fallback_used=fallback_used,
    )
