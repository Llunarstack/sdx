"""
Layered **prompt layout** compiler for clearer T5 conditioning and repeatable control.

Encoders attend more reliably when information is **grouped and ordered**: intent and
subjects before scene stack; quality either compact early (``subject_first``) or
leading (``quality_first``). This module turns structured JSON into one positive
and one negative string.

Use with ``sample.py --prompt-layout path.json`` or call :func:`compile_prompt_layout`.
Sampling uses :func:`substitute_compiled_layout_in_t5_prompt` / ``--t5-layout-encode`` for T5, and
:func:`triple_clip_caption` so CLIP-L and CLIP-bigG (triple mode) get the same labeled, 77-token-friendly line.

Schema (all keys optional except you should set at least one content section):

- ``preset_order``: ``quality_first`` | ``subject_first`` | ``scene_first``
- ``quality``, ``intent``, ``interaction``, ``props``, ``environment``, ``camera``,
  ``lighting``, ``composition``, ``style``, ``color_script``: string or list of strings
- ``subjects``: list of strings **or** list of objects ``{"label": "left", "tokens": [...], "negative": [...]}``
- ``negative``: global negatives (string or list)
- ``wrap_subject_labels``: bool (default ``true``) — ``(label: tok, tok)`` per subject dict
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "CompiledPromptLayout",
    "DEFAULT_PRESET_ORDER",
    "PRESET_SECTION_ORDER",
    "T5_SECTION_LABELS",
    "compile_prompt_layout",
    "load_prompt_layout_file",
    "merge_prompt_with_layout",
    "layout_tail_suffix",
    "substitute_compiled_layout_in_t5_prompt",
    "t5_segment_texts_for_full_prompt",
    "t5_segment_texts_from_layout",
    "triple_clip_caption",
]

# Short labels so T5 can attend to section boundaries without special tokens.
T5_SECTION_LABELS: Dict[str, str] = {
    "quality": "QUALITY",
    "intent": "INTENT",
    "subjects": "SUBJECTS",
    "interaction": "INTERACTION",
    "props": "PROPS",
    "environment": "ENVIRONMENT",
    "camera": "CAMERA",
    "lighting": "LIGHTING",
    "composition": "COMPOSITION",
    "style": "STYLE",
    "color_script": "COLOR",
}

SectionValue = Union[str, Sequence[str], None]

# Order of section names in the final prompt (only non-empty sections are emitted).
PRESET_SECTION_ORDER: Dict[str, Tuple[str, ...]] = {
    "quality_first": (
        "quality",
        "intent",
        "subjects",
        "interaction",
        "props",
        "environment",
        "camera",
        "lighting",
        "composition",
        "style",
        "color_script",
    ),
    "subject_first": (
        "intent",
        "subjects",
        "interaction",
        "quality",
        "environment",
        "props",
        "camera",
        "lighting",
        "composition",
        "style",
        "color_script",
    ),
    "scene_first": (
        "intent",
        "environment",
        "camera",
        "lighting",
        "composition",
        "subjects",
        "interaction",
        "props",
        "quality",
        "style",
        "color_script",
    ),
}

DEFAULT_PRESET_ORDER = "subject_first"


@dataclass
class CompiledPromptLayout:
    """Result of :func:`compile_prompt_layout`."""

    positive: str
    negative: str
    preset: str = DEFAULT_PRESET_ORDER
    sections_used: Tuple[str, ...] = field(default_factory=tuple)
    # (section_name, body) in preset order — used for T5-friendly encoding.
    section_blocks: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def to_t5_encoder_string(self) -> str:
        """
        Newline-separated, labeled blocks for T5 cross-attention (same semantics as :attr:`positive`,
        clearer boundaries than a single comma-separated line).
        """
        if not self.section_blocks:
            return (self.positive or "").strip()
        lines = [
            "Image generation: the labeled sections below describe one coherent image; use all of them together.",
        ]
        for name, body in self.section_blocks:
            b = (body or "").strip()
            if not b:
                continue
            label = T5_SECTION_LABELS.get(name, name.replace("_", " ").upper())
            lines.append(f"{label}: {b}.")
        return "\n".join(lines)


def _as_str_list(value: SectionValue) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    out: List[str] = []
    for x in value:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out


def _dedupe_csv(tokens: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in tokens:
        k = t.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(t.strip())
    return out


def _join_csv(tokens: Sequence[str]) -> str:
    return ", ".join(_dedupe_csv(list(tokens)))


def _compile_subjects(
    subjects: Any,
    *,
    wrap_labels: bool,
    neg_out: List[str],
) -> str:
    if subjects is None:
        return ""
    if isinstance(subjects, str):
        return subjects.strip()
    if not isinstance(subjects, list):
        return ""

    chunks: List[str] = []
    for i, item in enumerate(subjects):
        if isinstance(item, str) and item.strip():
            chunks.append(item.strip())
        elif isinstance(item, Mapping):
            m = dict(item)
            label = str(m.get("label", m.get("id", "")) or "").strip()
            toks = _as_str_list(m.get("tokens", m.get("positive", m.get("appearance", []))))
            if not toks:
                continue
            body = _join_csv(toks)
            if label and wrap_labels:
                chunks.append(f"({label}: {body})")
            elif label:
                chunks.append(f"{label}: {body}")
            else:
                chunks.append(body)
            neg_out.extend(_as_str_list(m.get("negative", m.get("avoid", []))))
        else:
            continue
    return _join_csv(chunks)


def _section_strings(data: Mapping[str, Any], subject_neg_out: List[str]) -> Dict[str, str]:
    """Map section name -> single comma-joined string (empty if missing)."""
    out: Dict[str, str] = {}
    for key in (
        "quality",
        "intent",
        "interaction",
        "props",
        "environment",
        "camera",
        "lighting",
        "composition",
        "style",
        "color_script",
    ):
        out[key] = _join_csv(_as_str_list(data.get(key)))
    wrap = bool(data.get("wrap_subject_labels", True))
    subj = _compile_subjects(data.get("subjects"), wrap_labels=wrap, neg_out=subject_neg_out)
    out["subjects"] = subj
    return out


def compile_prompt_layout(data: Mapping[str, Any]) -> CompiledPromptLayout:
    """
    Build ``(positive, negative)`` from a layout dict (e.g. parsed JSON).

    Unknown top-level keys are ignored. Per-subject ``negative`` lists are merged into
    the compiled negative, then the global ``negative`` field.
    """
    if not isinstance(data, Mapping):
        raise TypeError("layout data must be a mapping")
    preset = str(data.get("preset_order", DEFAULT_PRESET_ORDER) or DEFAULT_PRESET_ORDER).strip()
    order = PRESET_SECTION_ORDER.get(preset)
    if order is None:
        preset = DEFAULT_PRESET_ORDER
        order = PRESET_SECTION_ORDER[DEFAULT_PRESET_ORDER]

    subj_neg_tokens: List[str] = []
    sec = _section_strings(data, subj_neg_tokens)
    pos_parts: List[str] = []
    used: List[str] = []
    blocks: List[Tuple[str, str]] = []
    for name in order:
        if name == "subjects":
            s = sec.get("subjects", "")
        else:
            s = sec.get(name, "")
        if s:
            pos_parts.append(s)
            used.append(name)
            blocks.append((name, s))

    positive = ", ".join(pos_parts).strip()

    negative = _join_csv(subj_neg_tokens + _as_str_list(data.get("negative")))

    return CompiledPromptLayout(
        positive=positive,
        negative=negative,
        preset=preset,
        sections_used=tuple(used),
        section_blocks=tuple(blocks),
    )


def load_prompt_layout_file(path: Union[str, Path]) -> CompiledPromptLayout:
    """Load JSON object from path and compile."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"prompt layout not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(raw, dict):
        raise ValueError("prompt layout JSON must be an object at the root")
    return compile_prompt_layout(raw)


def substitute_compiled_layout_in_t5_prompt(full_text: str, compiled: CompiledPromptLayout) -> str:
    """
    Replace the compiled flat :attr:`CompiledPromptLayout.positive` substring (first occurrence)
    with :meth:`CompiledPromptLayout.to_t5_encoder_string`, preserving any prefix/suffix
    (quality boosts, content-control tags, user merge tail).
    """
    if full_text is None:
        return ""
    core = (compiled.positive or "").strip()
    if not full_text or not core or core not in full_text:
        return full_text
    structured = compiled.to_t5_encoder_string()
    if structured == core:
        return full_text
    return full_text.replace(core, structured, 1)


def t5_segment_texts_from_layout(compiled: CompiledPromptLayout) -> List[str]:
    """One string per non-empty layout section for segmented T5 tokenization."""
    out: List[str] = []
    for name, body in compiled.section_blocks:
        b = (body or "").strip()
        if not b:
            continue
        label = T5_SECTION_LABELS.get(name, name.replace("_", " ").upper())
        out.append(f"{label}: {b}.")
    if not out:
        p = (compiled.positive or "").strip()
        if p:
            out.append(p)
    return out


def layout_tail_suffix(full_text: str, compiled: CompiledPromptLayout) -> str:
    """
    Text after the first occurrence of the compiled flat positive (merged user prompt, control tags, …).
    Empty if the core cannot be found (e.g. emphasis rewrote the span).
    """
    if not full_text:
        return ""
    core = (compiled.positive or "").strip()
    if not core or core not in full_text:
        return ""
    return full_text[full_text.index(core) + len(core) :].strip(" ,")


def triple_clip_caption(compiled: CompiledPromptLayout, flat_full: str) -> str:
    """
    One string for **both** CLIP-L and CLIP-bigG in triple mode: short labeled sections (high signal
    in the first tokens) plus any suffix after the layout core. CLIP tokenizers truncate at 77 tokens.
    """
    flat = (flat_full or "").strip()
    if not compiled.section_blocks:
        return flat
    parts: List[str] = []
    for name, body in compiled.section_blocks:
        b = (body or "").strip()
        if not b:
            continue
        lab = T5_SECTION_LABELS.get(name, name.replace("_", " ").upper())
        parts.append(f"{lab}: {b}")
    head = " . ".join(parts)
    if not head:
        return flat
    tail = layout_tail_suffix(flat_full, compiled)
    if tail:
        return f"{head} . {tail}"
    return head


def t5_segment_texts_for_full_prompt(compiled: CompiledPromptLayout, flat_full: str) -> List[str]:
    """
    Segmented T5: layout sections plus an ``OTHER:`` segment for merged user / control text after the core,
    so triple conditioning stays aligned with the full :func:`merge_prompt_with_layout` string.
    """
    segs = t5_segment_texts_from_layout(compiled)
    tail = layout_tail_suffix(flat_full, compiled)
    if tail:
        segs.append(f"OTHER: {tail}.")
    if not segs and flat_full.strip():
        return [flat_full.strip()]
    return segs


def merge_prompt_with_layout(
    layout_positive: str,
    user_prompt: str,
    *,
    layout_first: bool = True,
) -> str:
    """Combine compiled layout positive with an extra user prompt fragment."""
    a = (layout_positive or "").strip()
    b = (user_prompt or "").strip()
    if not a:
        return b
    if not b:
        return a
    if layout_first:
        return f"{a}, {b}"
    return f"{b}, {a}"
