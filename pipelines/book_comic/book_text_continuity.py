"""
**Text / lettering / script continuity** helpers for books and comics.

Prompt-only: pair with ``--expected-text`` / OCR repair for strongest dialogue lock.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pipelines.book_comic.prompt_lexicon import merge_prompt_fragments


def _text(spec: Mapping[str, Any], *keys: str) -> str:
    """First truthy value among *keys*, stringified and stripped."""
    for key in keys:
        value = spec.get(key)
        if value:
            return str(value).strip()
    return ""


def lettering_visual_memory_fragment(lettering: Mapping[str, Any]) -> str:
    """Build a fragment from visual-memory ``lettering`` object."""
    bits: list[str] = []
    for template, keys in (
        ("consistent speech balloon style: {}", ("balloon_style",)),
        ("hand-lettered typography mood: {}", ("typography_mood", "font_vibe")),
        ("integrated sfx lettering: {}", ("sfx_style",)),
        ("dialogue orthography lock: {}", ("orthography", "language_lock")),
        ("{}", ("extra",)),
    ):
        value = _text(lettering, *keys)
        if value:
            bits.append(template.format(value))
    if lettering.get("match_quoted_script"):
        bits.append("printed dialogue must match quoted script text exactly in balloons")
    return merge_prompt_fragments(*bits)


def text_continuity_clause(spec: Mapping[str, Any]) -> str:
    """
    Strong continuity for recurring on-page text, object labels, and chapter leitmotifs.

    Expected keys (all optional):

    - ``locked_phrases`` / ``must_include``: list[str] or semicolon str
    - ``object_labels``: list of ``{"id": "sword", "label": "EXCALIBUR"}`` for prop text
    - ``chapter_motto``: recurring title fragment
    - ``narration_caption_style``: e.g. "italic rectangular captions top-left"
    - ``strict_script``: if true, insist lettering matches provided wording
    """
    bits: list[str] = []
    if spec.get("strict_script"):
        bits.append("strict script fidelity: every visible word in balloons and captions matches the writer script")

    raw_p = spec.get("locked_phrases") or spec.get("must_include") or spec.get("must_include_phrases")
    phrases: list[str] = []
    if isinstance(raw_p, list):
        phrases = [str(x).strip() for x in raw_p if str(x).strip()]
    elif isinstance(raw_p, str) and raw_p.strip():
        phrases = [p.strip() for p in raw_p.split(";") if p.strip()]
    if phrases:
        blob = "; ".join(phrases[:12])
        tail = " (and same recurring wording elsewhere)" if len(phrases) > 12 else ""
        bits.append(f"recurring exact text on page when shown: {blob}{tail}")

    motto = _text(spec, "chapter_motto")
    if motto:
        bits.append(f"chapter leitmotif text: {motto}")

    cap = _text(spec, "narration_caption_style")
    if cap:
        bits.append(f"caption box discipline: {cap}")

    objs = spec.get("object_labels")
    if isinstance(objs, list):
        for o in objs:
            if not isinstance(o, dict):
                continue
            oid = _text(o, "id")
            lab = _text(o, "label")
            if oid and lab:
                bits.append(f"object {oid} always reads as printed text {lab!r} when labeled")
            elif lab:
                bits.append(f"recurring printed label: {lab}")

    ex = _text(spec, "extra")
    if ex:
        bits.append(ex)
    return merge_prompt_fragments(*bits)
