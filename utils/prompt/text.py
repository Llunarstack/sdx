"""Prompt text: comma tags, ``(emphasis)`` weights, the layered prompt layout, breakdown, i18n hints, JSONL lint, ``@artist`` mentions and the block composer."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from config.prompts import dedupe_tags, merge_csv_unique


def split_tags(text: str) -> list[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def str_list(value: object) -> list[str]:
    """Stripped non-empty strings from a JSON field that holds a string or a list of strings."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [x.strip() for x in value if isinstance(x, str) and x.strip()]
    return []


def join_tags(tags: Sequence[str]) -> str:
    return ", ".join(t.strip() for t in tags if t and str(t).strip())


def append_unique(base: str, additions: Sequence[str]) -> str:
    return merge_csv_unique(base or "", join_tags([str(t) for t in additions]))


def append_csv(base: str, fragment: str) -> str:
    """Merge ``fragment`` into ``base`` without repeats; an empty fragment leaves ``base`` as written."""
    frag = (fragment or "").strip().strip(",")
    return merge_csv_unique(base, frag) if frag else (base or "").strip()


def token_set(text: str) -> set[str]:
    """Lowercased words of comma- and space-separated prompt text."""
    return {w.lower() for part in (text or "").split(",") for w in part.split()}


_WORD = re.compile(r"[a-z][a-z0-9'-]*")
_STOPWORDS = frozenset(
    "a an the of on in at to and or with without for from by no not very too over overly is are as into".split()
)
# A multi-word negative tag that names a defect stays even when the positive mentions the same thing:
# "bad hands" or "extra fingers" still belong in the negative of a prompt about hands.
_DEFECT_WORDS = frozenset(
    "bad worst low poor poorly ugly deformed malformed disfigured mutated mutation extra missing fused broken "
    "twisted distorted warped melted melting mangled duplicate duplicated cropped wrong incorrect floating "
    "unnatural gross messy artifacts artifact".split()
)


def filter_negative_by_positive(positive: str, negative: str) -> str:
    """Drop negative tags that name what the positive asks for, so CFG does not push away from it.

    A whole tag goes when one of its content words appears in ``positive``, unless it is a multi-word
    defect description.
    Single words are never cut out of a tag: trimming "plastic skin" to "plastic" or "deformed face"
    to "face" would negate something new.
    """
    wanted = {w for w in _WORD.findall((positive or "").lower()) if w not in _STOPWORDS}
    if not wanted:
        return negative
    kept = []
    for tag in split_tags(negative):
        words = {w for w in _WORD.findall(tag.lower()) if w not in _STOPWORDS}
        if words & wanted and not (len(words) > 1 and words & _DEFECT_WORDS):
            continue
        kept.append(tag)
    return join_tags(kept) or " "


try:
    import torch
except ModuleNotFoundError as e:
    if e.name != "torch":
        raise
    torch = None  # type: ignore[assignment]


def parse_prompt_emphasis(prompt: str) -> tuple[str, list]:
    """
    Parse ``(word)`` → weight 1.2, ``[word]`` → 0.8.

    Returns ``(cleaned_prompt, segments)`` where *segments* are
    ``[(char_start, char_end, weight), ...]`` in *cleaned* string coordinates.
    """
    cleaned = ""
    segments: list = []
    parts = re.split(r"(\([^)]*\)|\[[^\]]*\])", prompt)
    for p in parts:
        if p.startswith("(") and p.endswith(")"):
            content = p[1:-1]
            start = len(cleaned)
            cleaned += content
            segments.append((start, len(cleaned), 1.2))
        elif p.startswith("[") and p.endswith("]"):
            content = p[1:-1]
            start = len(cleaned)
            cleaned += content
            segments.append((start, len(cleaned), 0.8))
        else:
            start = len(cleaned)
            cleaned += p
            if p:
                segments.append((start, len(cleaned), 1.0))
    return cleaned.strip(), segments


def token_weights_from_cleaned_segments(
    cleaned: str,
    segments: list,
    tokenizer,
    max_length: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """
    Return ``(max_length,)`` per-token weights using tokenizer ``offset_mapping``, or ``None``
    if the tokenizer does not support it.
    """
    if torch is None:  # pragma: no cover (torch required for tensor ops)
        raise ModuleNotFoundError("torch is required for token-weight tensor operations.")
    if dtype is None:
        dtype = torch.float32
    try:
        enc = tokenizer(
            [cleaned],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
    except TypeError:
        return None
    offset_mapping = enc.get("offset_mapping")
    if offset_mapping is None:
        return None
    offset_mapping = offset_mapping[0]
    weights: list[float] = []
    for s, e in offset_mapping:
        if s == 0 and e == 0:  # padding
            weights.append(1.0)
            continue
        w = 1.0
        for seg_start, seg_end, seg_w in segments:
            if s < seg_end and e > seg_start:
                w = seg_w
                break
        weights.append(w)
    t = torch.tensor(weights, dtype=dtype)
    if device is not None:
        t = t.to(device=device)
    return t


def batch_encoder_token_weights(
    captions: list[str],
    tokenizer,
    max_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    text_bundle=None,
) -> tuple[list[str], torch.Tensor | None]:
    """
    For each caption, strip emphasis brackets for T5 text and build aligned weights.

    Returns:
        ``(cleaned_captions, token_weights)`` with ``token_weights`` shape ``(B, L)`` where
        ``L == max_length`` for T5-only, or ``max_length + N`` for triple/penta text
        (extra N tokens = 1.0; N is 2 for triple, 4 for penta).

    If offset mapping is unavailable for any row, returns ``(cleaned_captions, None)`` — caller
    should still encode *cleaned_captions* but omit ``token_weights`` in ``model_kwargs``.
    """
    if torch is None:  # pragma: no cover (torch required for tensor ops)
        raise ModuleNotFoundError("torch is required for token-weight tensor operations.")
    cleaned_caps: list[str] = []
    segment_lists: list[tuple[str, list]] = []
    for c in captions:
        cl, segs = parse_prompt_emphasis(c or "")
        cleaned_caps.append(cl)
        segment_lists.append((cl, segs))

    rows: list[torch.Tensor] = []
    for cl, segs in segment_lists:
        w = token_weights_from_cleaned_segments(cl, segs, tokenizer, max_length, device=None, dtype=torch.float32)
        if w is None:
            return cleaned_caps, None
        rows.append(w)

    stacked = torch.stack(rows, dim=0).to(device=device, dtype=dtype)
    extra = 0
    if text_bundle is not None and getattr(text_bundle, "fusion", None) is not None:
        extra = int(text_bundle.extra_fusion_tokens())
    if extra > 0:
        b = stacked.shape[0]
        stacked = torch.cat([stacked, torch.ones(b, extra, device=device, dtype=dtype)], dim=1)
    return cleaned_caps, stacked


# Short labels so T5 can attend to section boundaries without special tokens.
T5_SECTION_LABELS: dict[str, str] = {
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


# Order of section names in the final prompt (only non-empty sections are emitted).
PRESET_SECTION_ORDER: dict[str, tuple[str, ...]] = {
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


@dataclass(slots=True)
class CompiledPromptLayout:
    """Result of :func:`compile_prompt_layout`."""

    positive: str
    negative: str
    preset: str = DEFAULT_PRESET_ORDER
    sections_used: tuple[str, ...] = field(default_factory=tuple)
    # (section_name, body) in preset order — used for T5-friendly encoding.
    section_blocks: tuple[tuple[str, str], ...] = field(default_factory=tuple)

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


def _join_csv(tokens: Sequence[str]) -> str:
    return ", ".join(dedupe_tags(list(tokens)))


def _compile_subjects(
    subjects: Any,
    *,
    wrap_labels: bool,
    neg_out: list[str],
) -> str:
    if subjects is None:
        return ""
    if isinstance(subjects, str):
        return subjects.strip()
    if not isinstance(subjects, list):
        return ""

    chunks: list[str] = []
    for item in subjects:
        if isinstance(item, str) and item.strip():
            chunks.append(item.strip())
        elif isinstance(item, Mapping):
            m = dict(item)
            label = str(m.get("label", m.get("id", "")) or "").strip()
            toks = str_list(m.get("tokens", m.get("positive", m.get("appearance", []))))
            if not toks:
                continue
            body = _join_csv(toks)
            if label and wrap_labels:
                chunks.append(f"({label}: {body})")
            elif label:
                chunks.append(f"{label}: {body}")
            else:
                chunks.append(body)
            neg_out.extend(str_list(m.get("negative", m.get("avoid", []))))
        else:
            continue
    return _join_csv(chunks)


def _section_strings(data: Mapping[str, Any], subject_neg_out: list[str]) -> dict[str, str]:
    """Map section name -> single comma-joined string (empty if missing)."""
    out: dict[str, str] = {}
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
        out[key] = _join_csv(str_list(data.get(key)))
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

    subj_neg_tokens: list[str] = []
    sec = _section_strings(data, subj_neg_tokens)
    pos_parts: list[str] = []
    used: list[str] = []
    blocks: list[tuple[str, str]] = []
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

    negative = _join_csv(subj_neg_tokens + str_list(data.get("negative")))

    return CompiledPromptLayout(
        positive=positive,
        negative=negative,
        preset=preset,
        sections_used=tuple(used),
        section_blocks=tuple(blocks),
    )


def load_prompt_layout_file(path: str | Path) -> CompiledPromptLayout:
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


def t5_segment_texts_from_layout(compiled: CompiledPromptLayout) -> list[str]:
    """One string per non-empty layout section for segmented T5 tokenization."""
    out: list[str] = []
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
    One string for short-context CLIP encoders in triple/penta mode: labeled sections (high signal
    in the first tokens) plus any suffix after the layout core. CLIP tokenizers truncate at 77 tokens.
    """
    flat = (flat_full or "").strip()
    if not compiled.section_blocks:
        return flat
    parts: list[str] = []
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


def multi_clip_caption(compiled: CompiledPromptLayout, flat_full: str) -> str:
    """Alias for :func:`triple_clip_caption` (CLIP-L / bigG / H in penta mode)."""
    return triple_clip_caption(compiled, flat_full)


def t5_segment_texts_for_full_prompt(compiled: CompiledPromptLayout, flat_full: str) -> list[str]:
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


BreakdownFormat = Literal["ordered", "labeled"]
BreakdownOrder = Literal["subject_first", "quality_first", "scene_first"]

# Lowercase substrings → layout section (first match wins by highest score per segment).
_KEYWORD_BUCKETS: dict[str, tuple[str, ...]] = {
    "quality": (
        "masterpiece",
        "best quality",
        "high quality",
        "highly detailed",
        "absurdres",
        "8k",
        "4k",
        "uhd",
        "hdr",
        "sharp focus",
        "detailed",
    ),
    "style": (
        "anime",
        "manga",
        "oil painting",
        "watercolor",
        "photorealistic",
        "realistic",
        "cinematic",
        "digital art",
        "concept art",
        "illustration",
        "3d render",
        "pixel art",
        "impressionist",
        "baroque",
        "ukiyo",
        "sketch",
        "pastel drawing",
    ),
    "camera": (
        "wide angle",
        "telephoto",
        "macro",
        "depth of field",
        "bokeh",
        "lens flare",
        "35mm",
        "50mm",
        "85mm",
        "f/1.4",
        "f/2.8",
        "close-up",
        "closeup",
        "full body",
        "bust portrait",
        "headshot",
        "establishing shot",
        "dutch angle",
        "low angle",
        "high angle",
        "bird's eye",
        "worm's eye",
    ),
    "lighting": (
        "golden hour",
        "blue hour",
        "rim light",
        "backlight",
        "softbox",
        "studio lighting",
        "neon light",
        "candlelight",
        "volumetric",
        "god rays",
        "chiaroscuro",
        "ambient",
        "hard light",
        "soft light",
        "three-point lighting",
        "moonlight",
        "sunset lighting",
    ),
    "composition": (
        "rule of thirds",
        "symmetrical",
        "leading lines",
        "negative space",
        "centered composition",
        "dynamic composition",
        "frame within",
        "foreground",
        "midground",
        "background",
    ),
    "environment": (
        "indoor",
        "outdoor",
        "forest",
        "jungle",
        "city",
        "street",
        "alley",
        "beach",
        "ocean",
        "desert",
        "mountain",
        "sky",
        "clouds",
        "underwater",
        "office",
        "bedroom",
        "kitchen",
        "castle",
        "temple",
        "ruins",
        "space station",
        "cyberpunk city",
        "meadow",
        "snow",
        "rain",
        "fog",
        "night sky",
    ),
    "interaction": (
        "fighting",
        "dancing",
        "hugging",
        "kissing",
        "talking",
        "running",
        "walking together",
        "looking at",
        "holding hands",
        "high five",
        "shaking hands",
        "chasing",
        "embracing",
        "arguing",
        "dueling",
        "battling",
    ),
    "props": (
        "holding ",
        "wearing ",
        "carrying ",
        "sitting on ",
        "standing on ",
        "leaning on ",
        "sword",
        "shield",
        "staff",
        "gun",
        "bow",
        "book",
        "cup",
        "phone",
        "backpack",
        "helmet",
        "crown",
        "wings",
        "tail",
        "hat",
        "glasses",
    ),
    "color_script": (
        "warm colors",
        "cool colors",
        "cold colors",
        "monochrome",
        "sepia",
        "teal and orange",
        "pastel palette",
        "muted colors",
        "vibrant colors",
        "desaturated",
        "high contrast",
        "low contrast",
        "color grading",
    ),
}


def warrant_prompt_breakdown(prompt: str) -> bool:
    """Enable auto breakdown for long or clause-heavy prompts."""
    p = (prompt or "").strip()
    if not p:
        return False
    if len(p) >= 120:
        return True
    chunks = [c.strip() for c in re.split(r"[,;]", p) if c.strip()]
    return len(chunks) >= 5


def _split_segments(prompt: str) -> list[str]:
    parts: list[str] = []
    for chunk in re.split(r"[,;]", prompt):
        c = chunk.strip()
        if c:
            parts.append(c)
    return parts


def _classify_segment(seg: str) -> str:
    low = seg.lower()
    best = "subjects"
    best_n = 0
    for cat, kws in _KEYWORD_BUCKETS.items():
        n = sum(1 for k in kws if k in low)
        if n > best_n:
            best_n = n
            best = cat
    if best_n == 0:
        return "subjects"
    return best


@dataclass(slots=True)
class PromptBreakdownResult:
    """Structured buckets + rendered strings."""

    sections: dict[str, list[str]] = field(default_factory=dict)
    ordered_flat: str = ""
    labeled_t5: str = ""
    section_order: tuple[str, ...] = ()


def breakdown_prompt_to_sections(prompt: str) -> dict[str, list[str]]:
    """Map section name → list of clause strings."""
    buckets: dict[str, list[str]] = {k: [] for k in _KEYWORD_BUCKETS}
    buckets["subjects"] = []
    for seg in _split_segments(prompt):
        cat = _classify_segment(seg)
        buckets.setdefault(cat, []).append(seg)
    return buckets


def build_breakdown(
    prompt: str,
    *,
    order: BreakdownOrder = "subject_first",
) -> PromptBreakdownResult:
    """
    Classify clauses, reorder by ``PRESET_SECTION_ORDER[order]``, emit flat + labeled strings.

    ``subjects`` holds everything not matching a specialized bucket strongly enough.
    Automatic breakdown does not emit ``intent`` (reserved for hand-written layouts).
    """
    prompt = (prompt or "").strip()
    raw = breakdown_prompt_to_sections(prompt)
    merged: dict[str, list[str]] = {
        "subjects": dedupe_tags(raw.get("subjects", [])),
        **{k: dedupe_tags(raw.get(k, [])) for k in _KEYWORD_BUCKETS},
    }
    order_names = PRESET_SECTION_ORDER.get(order, PRESET_SECTION_ORDER["subject_first"])
    ordered_parts: list[str] = []
    section_blocks: list[tuple[str, str]] = []

    for name in order_names:
        if name == "intent":
            continue
        body_list = merged.get(name, [])
        if not body_list:
            continue
        joined = ", ".join(body_list)
        ordered_parts.append(joined)
        section_blocks.append((name, joined))

    flat = ", ".join(ordered_parts).strip().strip(",")
    if not flat:
        flat = prompt

    lines = [
        "Image generation: the labeled sections below describe one coherent image; use all of them together.",
    ]
    for name, body in section_blocks:
        b = (body or "").strip()
        if not b:
            continue
        label = T5_SECTION_LABELS.get(name, name.replace("_", " ").upper())
        lines.append(f"{label}: {b}.")
    labeled = "\n".join(lines)

    return PromptBreakdownResult(
        sections={k: list(v) for k, v in merged.items() if v},
        ordered_flat=flat,
        labeled_t5=labeled,
        section_order=order_names,
    )


def apply_prompt_breakdown(
    prompt: str,
    *,
    order: BreakdownOrder = "subject_first",
    output_format: BreakdownFormat = "ordered",
) -> tuple[str, str]:
    """
    Returns ``(string_for_cache_and_clip, string_for_t5)``.

    When ``output_format`` is ``ordered``, both strings are identical (reordered flat prompt).
    When ``labeled``, first is flat reordered, second is multi-line labeled block for T5.
    """
    res = build_breakdown(prompt, order=order)
    if output_format == "labeled":
        return res.ordered_flat, res.labeled_t5
    return res.ordered_flat, res.ordered_flat


def normalize_lang_code(lang: str) -> str:
    if not lang:
        return "en"
    s = str(lang).strip().lower()
    if not s:
        return "en"

    if s in ("en", "eng", "english"):
        return "en"

    if s in ("zh", "zh-cn", "zh_hans", "zh-hans", "中文", "zh_cn"):
        return "zh"
    if s in ("zh-tw", "zh_hant", "zh-hant", "zh_tw"):
        return "zh"

    if s.startswith("es"):
        return "es"
    if s.startswith("fr"):
        return "fr"
    if s.startswith("ru"):
        return "ru"
    if s.startswith("ja"):
        return "ja"
    if s.startswith("it"):
        return "it"
    if s.startswith("ar"):
        return "ar"
    if s in ("la", "lat", "latin"):
        return "la"

    if s.startswith("hi"):
        return "hi"
    if s.startswith("bn"):
        return "bn"
    if s.startswith("ta"):
        return "ta"
    if s.startswith("te"):
        return "te"
    if s.startswith("mr"):
        return "mr"
    if s.startswith("kn"):
        return "kn"
    if s.startswith("ml"):
        return "ml"
    if s.startswith("gu"):
        return "gu"
    if s.startswith("pa"):
        return "pa"
    if s.startswith("ur"):
        return "ur"

    return "en"


def _examples_to_string(examples: list[str], limit: int = 6) -> str:
    cleaned = [e for e in examples if e]
    if not cleaned:
        return ""
    return ", ".join(cleaned[:limit])


def generic_suggestion(category: str, examples: list[str], lang: str) -> str:
    lang_code = normalize_lang_code(lang)
    ex = _examples_to_string(examples)
    ex_part = f" (e.g., {ex})" if ex else ""

    templates = {
        "en": "Add related keyword terms for '{category}' early in the caption and keep it consistent across the dataset."
        + ex_part,
        "es": "Para reforzar '{category}', agrega terminos relacionados al inicio del caption y mantente consistente en todo el dataset."
        + ex_part,
        "fr": "Pour renforcer '{category}', ajoutez des termes lies des le debut des captions et gardez la coherence dans tout le dataset."
        + ex_part,
        "ru": "chtoby usilit '{category}', dobavlyayte svyazannye slova v nachale caption i derzhite yedinstvo po vsemu datasetu."
        + ex_part,
        "ja": "tame ni '{category}' o tsuyokusuru: kyappushon no hayai tokoro ni kanren kyuudo o irete, zenkoku de ichiji ni shite kudasai."
        + ex_part,
        "zh": "xiangqianghua '{category}', qing zai caption de zaoqi bufen fangru guanlian keyici, bing zai quanbu shuju zhong baochi yizhi."
        + ex_part,
        "hi": "'{category}' ko behtar banane ke liye, caption ki shuruaat me sambandhit keywords add karein aur poore dataset me consistent rakhein."
        + ex_part,
        "bn": "'{category}' aro bhalo korte, caption-er shuru te sambandhito keywords add korun ebong poora dataset-e consistent rakhen."
        + ex_part,
        "ta": "'{category}'-ai balappadutha, captions-in aarambathile sambandhamaana keywords serthu, muzhum dataset-il ore maadhiri vaitthukollungal."
        + ex_part,
        "te": "'{category}' ni balam cheyyadaniki, caption start-lo sambandhit keywords add cheyyandi, mariyu motta dataset-lo consistency unchandi."
        + ex_part,
        "mr": "'{category}' majboot karanyasathi, caption cha suruvatichya pankhtit sambandhit keywords joda ani poora dataset madhye consistency theva."
        + ex_part,
        "kn": "'{category}' anu balapaadalu, caption shuruvall-e sambandhita keywords add maadi mattu poorti dataset-alli consistency irisi."
        + ex_part,
        "ml": "'{category}' balam aakkikan, caption-inde aarambhathil sambandhita keywords cherkkuka, poora dataset-ile consistency paalikuka."
        + ex_part,
        "gu": "'{category}' ne majboot banava mate, caption ni shuruaat ma sambandhit keywords jodo ane poora dataset ma consistency rakhjo."
        + ex_part,
        "pa": "'{category}' nu behtar banan layi, caption di shuruaat vich sambandhit keywords jodo te poore dataset vich consistency rakho."
        + ex_part,
        "ur": "'{category}' ko behtar banane ke liye, caption ke shuru me mutalliq keywords shamil karein aur poore dataset me consistency rakhein."
        + ex_part,
        "it": "Per rafforzare '{category}', aggiungi termini correlati all'inizio dei caption e mantieni coerenza in tutto il dataset."
        + ex_part,
        "ar": "litaqwiyat '{category}', adif kalimat muhtamimah fi awwal al-caption wa ahtafidh bi-al-itqan fi jami' al-dataset."
        + ex_part,
        "la": "Ad '{category}' roborandum, adde vocabula coniuncta primo in captione et eadem manere per totam tuam materiam."
        + ex_part,
    }

    return templates.get(lang_code, templates["en"]).format(category=category)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+", flags=re.ASCII)


def tokenize_normalized(text: str) -> list[str]:
    """
    Lightweight tokenizer for prompt adherence linting.

    - case-insensitive
    - alnum-only tokens
    - used for overlap detection and token-set heuristics (not for exact model tokenization)
    """
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _get_str(d: dict, keys: Iterable[str]) -> str | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str):
            s = v.strip()
            if s:
                return s
    return None


def get_pos_neg_text(row: dict) -> tuple[str, str]:
    """
    Extract (caption/positive, negative) from a manifest row.
    Keys match SDX conventions:
      - positive: caption | text
      - negative: negative_caption | negative_prompt | negative_text
    """
    pos = _get_str(row, ("caption", "text")) or ""
    neg = _get_str(row, ("negative_caption", "negative_prompt", "negative_text")) or ""
    return pos, neg


@dataclass(frozen=True, slots=True)
class PromptLintOptions:
    min_caption_len_chars: int = 0
    max_caption_tokens: int = 0  # distinct token set size heuristic
    top_overlap_tokens: int = 10
    fail_on_overlap: bool = False


def lint_jsonl_path(path: Path, opts: PromptLintOptions) -> dict[str, object]:
    """
    Stream a JSONL file and compute prompt lint stats.
    Expects one JSON object per non-empty line.
    """
    total_lines = 0
    json_parse_errors = 0

    empty_caption_rows = 0
    json_missing_pos_tokens = 0
    rows_over_max_tokens = 0

    ok_rows = 0
    caption_token_count_sum = 0
    caption_token_count_min = 2**31 - 1
    caption_token_count_max = 0

    overlap_rows = 0
    max_overlap_distinct_tokens = 0
    overlap_token_counts: dict[str, int] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            t = line.strip()
            if not t:
                continue
            try:
                row = json.loads(t)
            except Exception:
                json_parse_errors += 1
                continue

            if not isinstance(row, dict):
                json_parse_errors += 1
                continue

            pos, neg = get_pos_neg_text(row)
            if not pos or (opts.min_caption_len_chars > 0 and len(pos) < opts.min_caption_len_chars):
                empty_caption_rows += 1
                continue
            if not _TOKEN_RE.search(pos):
                json_missing_pos_tokens += 1
                continue

            pos_set = set(tokenize_normalized(pos))
            neg_set = set(tokenize_normalized(neg))

            tok_len = len(pos_set)
            ok_rows += 1
            caption_token_count_sum += tok_len
            caption_token_count_min = min(caption_token_count_min, tok_len)
            caption_token_count_max = max(caption_token_count_max, tok_len)

            if opts.max_caption_tokens > 0 and tok_len > opts.max_caption_tokens:
                rows_over_max_tokens += 1

            if pos_set and neg_set:
                overlap = pos_set.intersection(neg_set)
                if overlap:
                    overlap_rows += 1
                    overlap_n = len(overlap)
                    max_overlap_distinct_tokens = max(max_overlap_distinct_tokens, overlap_n)
                    for tok in overlap:
                        overlap_token_counts[tok] = overlap_token_counts.get(tok, 0) + 1

    top = sorted(overlap_token_counts.items(), key=lambda kv: kv[1], reverse=True)
    if opts.top_overlap_tokens > 0:
        top = top[: opts.top_overlap_tokens]

    return {
        "promptlint_file": str(path),
        "lines_total": total_lines,
        "json_parse_errors": json_parse_errors,
        "empty_caption_rows": empty_caption_rows,
        "json_missing_pos_tokens": json_missing_pos_tokens,
        "rows_over_max_tokens": rows_over_max_tokens,
        "rows_ok": ok_rows,
        "caption_token_count_set_size_avg": (caption_token_count_sum / ok_rows) if ok_rows else 0.0,
        "caption_token_count_set_size_min": caption_token_count_min if ok_rows else 0,
        "caption_token_count_set_size_max": caption_token_count_max if ok_rows else 0,
        "pos_neg_overlap_rows": overlap_rows,
        "pos_neg_overlap_max_distinct_tokens": max_overlap_distinct_tokens,
        "top_overlap_tokens": top,
        "fail_on_overlap": opts.fail_on_overlap,
    }


def should_fail(stats: dict[str, object], opts: PromptLintOptions) -> bool:
    if not opts.fail_on_overlap:
        return False
    return int(stats.get("pos_neg_overlap_rows", 0)) > 0


@dataclass
class ArtistEntry:
    canonical: str
    aliases: set[str] = field(default_factory=set)
    count: int = 0

    def to_dict(self) -> dict:
        return {
            "canonical": self.canonical,
            "aliases": sorted(self.aliases),
            "count": self.count,
        }


def _alias_keys(name: str) -> set[str]:
    """Generate lookup keys for an artist name (case/underscore/space variants)."""
    raw = name.strip()
    if not raw:
        return set()
    norm = normalize_artist_tag(raw)
    keys = {
        raw.lower(),
        norm.lower(),
        norm.lower().replace(" ", "_"),
        norm.lower().replace(" ", ""),
        raw.lower().replace("_", " "),
        raw.lower().replace(" ", "_"),
    }
    return {k for k in keys if k}


class ArtistRegistry:
    """In-memory artist alias map with fuzzy resolution."""

    def __init__(self) -> None:
        self._by_key: dict[str, str] = {}  # alias key -> canonical caption tag
        self._entries: dict[str, ArtistEntry] = {}  # canonical -> entry

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, name: str, *, count: int = 1) -> None:
        canonical = normalize_artist_tag(name)
        if not canonical:
            return
        ent = self._entries.setdefault(canonical, ArtistEntry(canonical=canonical))
        ent.count += int(count)
        for key in _alias_keys(name):
            ent.aliases.add(key)
            # Prefer higher-count artist when aliases collide across artists.
            prev = self._by_key.get(key)
            if prev is None or ent.count >= self._entries[prev].count:
                self._by_key[key] = canonical

    def resolve(self, query: str) -> str:
        """Map a user-typed artist handle to the trained caption tag."""
        q = query.strip()
        if not q:
            return q
        for key in _alias_keys(q):
            if key in self._by_key:
                return self._by_key[key]
        # Unknown artist: still normalize to booru caption form (works after full-site training).
        return normalize_artist_tag(q)

    def known(self, query: str) -> bool:
        return any(k in self._by_key for k in _alias_keys(query))

    def save(self, path: str | os.PathLike[str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in sorted(self._entries.items())}
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ArtistRegistry:
        reg = cls()
        p = Path(path)
        if not p.is_file():
            return reg
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for _key, val in raw.items():
                if isinstance(val, dict):
                    name = val.get("canonical") or _key
                    reg.add(name, count=int(val.get("count") or 0))
                    for alias in val.get("aliases") or []:
                        reg.add(str(alias), count=0)
                elif isinstance(val, (int, float)):
                    reg.add(str(_key), count=int(val))
                else:
                    reg.add(str(_key), count=1)
        return reg


_GLOBAL: ArtistRegistry | None = None


def default_index_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("SDX_ARTIST_INDEX", "").strip()
    if env:
        paths.append(Path(env))
    data = os.environ.get("SDX_DATA", "").strip()
    if data:
        paths.append(Path(data) / "artist_index.json")
        paths.append(Path(data) / "combined" / "artist_index.json")
    # repo-relative fallbacks
    root = Path(__file__).resolve().parents[2]
    paths.append(root / "data" / "artist_index.json")
    return paths


def get_registry(path: str | os.PathLike[str] | None = None) -> ArtistRegistry:
    """Return a cached registry (loads from disk on first call)."""
    global _GLOBAL
    if path is not None:
        return ArtistRegistry.load(path)
    if _GLOBAL is None:
        _GLOBAL = ArtistRegistry()
        for p in default_index_paths():
            if p.is_file():
                _GLOBAL = ArtistRegistry.load(p)
                break
    return _GLOBAL


def build_from_manifests(manifest_paths: Iterable[str | os.PathLike[str]]) -> ArtistRegistry:
    """Scan JSONL manifests for ``artist_tags`` fields and tag captions."""
    reg = ArtistRegistry()
    for mp in manifest_paths:
        p = Path(mp)
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            artists = row.get("artist_tags") or []
            if isinstance(artists, str):
                artists = [a.strip() for a in artists.split(",") if a.strip()]
            for a in artists:
                reg.add(str(a), count=1)
    return reg


# Unicode letters/digits for international artist names (pixiv, etc.).
_ARTIST_BODY = r"[\w\-.()]+"
_ARTIST_EXPLICIT_RE = re.compile(
    rf"@artist:(?:'([^']+)'|\"([^\"]+)\"|({_ARTIST_BODY}))",
    re.UNICODE,
)
_ARTIST_RE = re.compile(
    rf"@(?:'([^']+)'|\"([^\"]+)\"|({_ARTIST_BODY}))",
    re.UNICODE,
)


def normalize_artist_tag(name: str) -> str:
    """Lowercase, underscores->spaces, collapse whitespace (danbooru caption form)."""
    tag = name.strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", tag).strip()


def _resolve_name(raw: str, registry) -> str:
    if registry is not None:
        return registry.resolve(raw)
    return normalize_artist_tag(raw)


def expand_artist_mentions(
    prompt: str,
    *,
    strength: float = 1.0,
    registry=None,
) -> tuple[str, list[str]]:
    """Replace ``@artist`` mentions with trained artist tags.

    Returns ``(expanded_prompt, artists)``. Pass an :class:`ArtistRegistry` (or
    use ``get_registry()``) to resolve aliases from scraped data.
    """
    if not prompt or "@" not in prompt:
        return prompt, []

    if registry is None:
        try:
            registry = get_registry()
        except Exception:
            registry = None

    artists: list[str] = []

    def _wrap(tag: str) -> str:
        artists.append(tag)
        if abs(strength - 1.0) > 1e-3:
            return f"({tag}:{strength:.2f})"
        return tag

    def _sub_explicit(m: re.Match) -> str:
        raw = m.group(1) or m.group(2) or m.group(3) or ""
        tag = _resolve_name(raw, registry)
        return _wrap(tag) if tag else m.group(0)

    def _sub(m: re.Match) -> str:
        raw = m.group(1) or m.group(2) or m.group(3) or ""
        if raw.lower() == "artist":
            return m.group(0)
        tag = _resolve_name(raw, registry)
        if not tag:
            return m.group(0)
        return _wrap(tag)

    expanded = _ARTIST_EXPLICIT_RE.sub(_sub_explicit, prompt)
    expanded = _ARTIST_RE.sub(_sub, expanded)
    return expanded, artists


# ``+category: traits`` or ``+category traits`` — category names are case-insensitive.
_BLOCK_RE = re.compile(
    r"\+\s*([a-z][a-z0-9_-]*)\s*:\s*([^+|]+)|"
    r"\+\s*([a-z][a-z0-9_-]*)\s+([^+|]+)",
    re.IGNORECASE,
)

# Category → booru anchor tags prepended before the user's traits.
_CATEGORY_ANCHORS: dict[str, list[str]] = {
    "character": [],
    "char": [],
    "person": [],
    "building": ["architecture", "building"],
    "architecture": ["architecture"],
    "house": ["house", "building"],
    "skyscraper": ["skyscraper", "cityscape"],
    "city": ["cityscape", "city"],
    "landscape": ["landscape", "scenery"],
    "scenery": ["scenery", "landscape"],
    "vehicle": ["vehicle"],
    "car": ["car", "motor vehicle"],
    "truck": ["truck", "motor vehicle"],
    "motorcycle": ["motorcycle", "motor vehicle"],
    "aircraft": ["aircraft"],
    "ship": ["ship", "watercraft"],
    "train": ["train", "railroad"],
    "object": ["still life", "object focus"],
    "prop": ["object focus"],
    "weapon": ["weapon"],
    "food": ["food"],
    "animal": ["animal"],
    "creature": ["creature"],
    "plant": ["plant", "nature"],
    "flower": ["flower"],
    "tree": ["tree"],
    "furniture": ["furniture"],
    "clothing": ["clothes"],
    "outfit": ["clothes"],
    "armor": ["armor"],
    "scene": [],
    "background": ["background"],
    "interior": ["indoor", "interior"],
    "exterior": ["outdoors"],
    "lighting": [],
    "light": [],
    "weather": [],
    "style": [],
    "mood": [],
    "quality": ["masterpiece", "best quality"],
    "camera": [],
    "pose": [],
    "expression": ["expression"],
    "hair": [],
    "eyes": [],
    "skin": [],
    "noun": [],
    "adj": [],
    "adjective": [],
}


@dataclass
class ComposedPrompt:
    positive: str
    artists: list[str] = field(default_factory=list)
    blocks: dict[str, list[str]] = field(default_factory=dict)
    base_text: str = ""


def _split_traits(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[,;]", text) if p.strip()]
    return parts


def parse_blocks(prompt: str) -> tuple[str, dict[str, list[str]]]:
    """Return ``(remainder, {category: [traits...]})``."""
    blocks: dict[str, list[str]] = {}

    def _collect(m: re.Match) -> str:
        cat = (m.group(1) or m.group(3) or "").strip().lower()
        body = (m.group(2) or m.group(4) or "").strip()
        if cat and body:
            blocks.setdefault(cat, []).extend(_split_traits(body))
        return " "

    remainder = _BLOCK_RE.sub(_collect, prompt)
    remainder = re.sub(r"\s+", " ", remainder).strip(" ,|")
    return remainder, blocks


def compose_prompt(
    prompt: str,
    *,
    artist_strength: float = 1.0,
    artist_registry: ArtistRegistry | None = None,
    artist_index: str | None = None,
    prepend_quality: bool = False,
) -> ComposedPrompt:
    """Turn ``@artist`` / ``+category`` syntax into a training-style caption."""
    raw = (prompt or "").strip()
    if not raw:
        return ComposedPrompt(positive="")

    reg = artist_registry or (get_registry(artist_index) if artist_index else get_registry())

    base, blocks = parse_blocks(raw)
    expanded, artists = expand_artist_mentions(base, strength=artist_strength, registry=reg)

    segments: list[str] = []
    if prepend_quality:
        segments.append("masterpiece, best quality")

    if expanded:
        segments.append(expanded)

    for cat, traits in blocks.items():
        anchors = list(_CATEGORY_ANCHORS.get(cat, []))
        trait_str = ", ".join(traits)
        if anchors:
            segments.append(merge_csv_unique(", ".join(anchors), trait_str))
        else:
            segments.append(trait_str)

    positive = merge_csv_unique(*segments)
    return ComposedPrompt(positive=positive, artists=artists, blocks=blocks, base_text=base)
