"""
**Text / lettering / script continuity** helpers for books and comics.

Prompt-only: pair with ``--expected-text`` / OCR repair for strongest dialogue lock.
"""

from __future__ import annotations

from typing import Any, List, Mapping

from pipelines.book_comic.prompt_lexicon import merge_prompt_fragments


def lettering_visual_memory_fragment(lettering: Mapping[str, Any]) -> str:
    """Build a fragment from visual-memory ``lettering`` object."""
    bits: List[str] = []
    bal = str(lettering.get("balloon_style", "") or "").strip()
    if bal:
        bits.append(f"consistent speech balloon style: {bal}")
    typ = str(lettering.get("typography_mood", "") or lettering.get("font_vibe", "") or "").strip()
    if typ:
        bits.append(f"hand-lettered typography mood: {typ}")
    sfx = str(lettering.get("sfx_style", "") or "").strip()
    if sfx:
        bits.append(f"integrated sfx lettering: {sfx}")
    ortho = str(lettering.get("orthography", "") or lettering.get("language_lock", "") or "").strip()
    if ortho:
        bits.append(f"dialogue orthography lock: {ortho}")
    tail = str(lettering.get("extra", "") or "").strip()
    if tail:
        bits.append(tail)
    if bool(lettering.get("match_quoted_script", False)):
        bits.append("printed dialogue must match quoted script text exactly in balloons")
    return merge_prompt_fragments(*bits)


def text_continuity_clause(spec: Mapping[str, Any]) -> str:
    """
    Strong continuity for recurring on-page text, object labels, and chapter leitmotifs.

    Expected keys (all optional):

    - ``locked_phrases`` / ``must_include``: list[str] or semicolon str
    - ``object_labels``: list of ``{{"id": "sword", "label": "EXCALIBUR"}}`` for prop text
    - ``chapter_motto``: recurring title fragment
    - ``narration_caption_style``: e.g. \"italic rectangular captions top-left\"
    - ``strict_script``: if true, insist lettering matches provided wording
    """
    bits: List[str] = []
    if bool(spec.get("strict_script", False)):
        bits.append(
            "strict script fidelity: every visible word in balloons and captions matches the writer script"
        )

    raw_p = spec.get("locked_phrases") or spec.get("must_include") or spec.get("must_include_phrases")
    phrases: List[str] = []
    if isinstance(raw_p, list):
        phrases = [str(x).strip() for x in raw_p if str(x).strip()]
    elif isinstance(raw_p, str) and raw_p.strip():
        phrases = [p.strip() for p in raw_p.split(";") if p.strip()]
    if phrases:
        blob = "; ".join(phrases[:12])
        tail = " (and same recurring wording elsewhere)" if len(phrases) > 12 else ""
        bits.append(f"recurring exact text on page when shown: {blob}{tail}")

    motto = str(spec.get("chapter_motto", "") or "").strip()
    if motto:
        bits.append(f"chapter leitmotif text: {motto}")

    cap = str(spec.get("narration_caption_style", "") or "").strip()
    if cap:
        bits.append(f"caption box discipline: {cap}")

    objs = spec.get("object_labels")
    if isinstance(objs, list):
        for o in objs:
            if not isinstance(o, dict):
                continue
            oid = str(o.get("id", "") or "").strip()
            lab = str(o.get("label", "") or "").strip()
            if oid and lab:
                bits.append(f"object {oid} always reads as printed text {lab!r} when labeled")
            elif lab:
                bits.append(f"recurring printed label: {lab}")

    ex = str(spec.get("extra", "") or "").strip()
    if ex:
        bits.append(ex)
    return merge_prompt_fragments(*bits)
