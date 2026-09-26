"""
Compile free-text prompts into regional box layouts (Ideogram-style spatial CFG).

Conservative heuristics: exact counts, occlusion, reflections, binary spatial
relations, between / next-to, holding, foreground/background, quoted glyph text,
and left/right people binding.
Returns ``None`` when no layout signal is found.
"""

from __future__ import annotations

import re

from .regional_box_prompting import BoxLayoutSpec, BoxRegion

__all__ = [
    "prompt_needs_spatial_layout",
    "compile_spatial_layout",
    "quoted_glyph_texts",
]

_WORD_NUMBERS: dict[str, int] = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}
_NUMBER_WORDS = "|".join(re.escape(w) for w in _WORD_NUMBERS)
_PERSON_TERMS = r"(?:woman|women|girl|girls|man|men|boy|boys|person|people)"

_QUOTED_RE = re.compile(
    r'["“]([^"”]{1,64})["”]|\'([^\']{1,64})\'|\[text:\s*([^\]]{1,64})\]',
    re.I,
)

_EXACT_COUNT_DIGIT = re.compile(
    r"\bexactly\s+(?P<n>[2-6])\s+(?:(?P<adj>[a-z][\w-]*)\s+)?(?P<noun>[a-z][\w-]*)\b",
    re.I,
)
_EXACT_COUNT_WORD = re.compile(
    rf"\bexactly\s+(?P<nword>{_NUMBER_WORDS})\s+(?:(?P<adj>[a-z][\w-]*)\s+)?(?P<noun>[a-z][\w-]*)\b",
    re.I,
)
_WORD_COUNT = re.compile(
    rf"\b(?P<nword>{_NUMBER_WORDS})\s+(?:(?P<adj>[a-z][\w-]*)\s+)?(?P<noun>[a-z][\w-]*)\b",
    re.I,
)

_LEFT_OF = re.compile(
    r"(?P<a>.+?)\s+(?:to\s+the\s+)?left\s+of\s+(?P<b>.+?)(?:[,.;]|$)",
    re.I,
)
_RIGHT_OF = re.compile(
    r"(?P<a>.+?)\s+(?:to\s+the\s+)?right\s+of\s+(?P<b>.+?)(?:[,.;]|$)",
    re.I,
)
_BEHIND = re.compile(
    r"(?P<a>.+?)\s+(?:behind|in\s+back\s+of)\s+(?P<b>.+?)(?:[,.;]|$)",
    re.I,
)
_IN_FRONT = re.compile(
    r"(?P<a>.+?)\s+in\s+front\s+of\s+(?P<b>.+?)(?:[,.;]|$)",
    re.I,
)
_ON_TOP = re.compile(
    r"(?P<a>.+?)\s+(?:"
    r"on\s+top\s+of(?:\s+a)?|"
    r"sitting\s+on(?:\s+a)?|"
    r"stacked\s+on(?:\s+a)?"
    r")\s+(?P<b>.+?)(?:[,.;]|$)",
    re.I,
)
_UNDER = re.compile(
    r"(?P<a>.+?)\s+(?:under|below|beneath)\s+(?P<b>.+?)(?:[,.;]|$)",
    re.I,
)
_BETWEEN = re.compile(
    r"(?P<a>.+?)\s+between\s+(?P<b>.+?)\s+and\s+(?P<c>.+?)(?:[,.;]|$)",
    re.I,
)
_NEXT_TO = re.compile(
    r"(?P<a>.+?)\s+(?:next\s+to|beside)\s+(?P<b>.+?)(?:[,.;]|$)",
    re.I,
)
_HOLDING = re.compile(
    r"\bholding\s+(?:a|an|the\s+)?(?P<obj>[\w][\w\s-]{0,32}?)(?:[,.;]| in | with |$)",
    re.I,
)
_FOREGROUND = re.compile(
    r"(?P<fg>.{3,72}?)\s+in\s+the\s+foreground",
    re.I,
)
_BACKGROUND = re.compile(
    r"(?P<bg>.{3,72}?)\s+in\s+the\s+background",
    re.I,
)

_OCCLUSION_HINT = re.compile(
    r"behind\s+(?:a|an|the\s+)?chain-link\s+fence|"
    r"behind\s+(?:a|an|the\s+)?(?:[\w-]+\s+)*fence\b|"
    r"through\s+the\s+gaps|"
    r"behind\s+(?:a|an|the\s+)?wine\s+glass|"
    r"partly\s+hidden\s+behind|"
    r"visible\s+through|"
    r"behind\s+(?:a|an|the\s+)?barrier\b",
    re.I,
)
_OCC_PARTLY_HIDDEN = re.compile(
    r"(?P<subj>.+?)\s+partly\s+hidden\s+behind\s+(?:a|an|the\s+)?(?P<occ>.+?)(?:[,.;]|$)",
    re.I,
)
_OCC_BEHIND = re.compile(
    r"(?P<subj>.+?)\s+behind\s+(?:a|an|the\s+)?(?P<occ>"
    r"wine\s+glass|"
    r"(?:[\w-]+\s+)*fence|"
    r"(?:[\w-]+\s+)?barrier|"
    r"(?:[\w-]+\s+)?glass"
    r")\b",
    re.I,
)
_OCC_NOUN = re.compile(
    r"\b((?:chain-link\s+)?fence|wine\s+glass|glass|barrier)\b",
    re.I,
)

_REFLECTION_WORD = re.compile(r"\b(?:reflection|reflected\s+in)\b", re.I)
_CHROME_SURFACE = re.compile(
    r"\bchrome\b.*\b(?:granite|water|puddle|tabletop)\b|"
    r"\b(?:granite|water|puddle|tabletop)\b.*\bchrome\b",
    re.S | re.I,
)

_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.I)

_OCC_NEG = "fused with subject, melted into person, missing occluder"
_OCC_SUBJ_NEG = "fused into fence, melted into glass, no occlusion"
_OCC_GLOBAL_NEG = "fused into occluder, melted through barrier, missing occlusion"
_REFL_NEG = "two identical objects stacked, duplicate kettle, second mug"


def quoted_glyph_texts(prompt: str) -> list[str]:
    """Quoted / ``[text:...]`` strings to render."""
    text = prompt or ""
    out: list[str] = []
    seen: set[str] = set()
    for m in _QUOTED_RE.finditer(text):
        chunk = next(g for g in m.groups() if g is not None).strip()
        if chunk and chunk not in seen:
            seen.add(chunk)
            out.append(chunk)
    return out


def prompt_needs_spatial_layout(prompt: str) -> bool:
    """Return True when ``compile_spatial_layout`` would produce a spec."""
    return compile_spatial_layout(prompt) is not None


def _first_clause(subject: str) -> str:
    return subject.split(",", 1)[0].strip()


def _strip_article(name: str) -> str:
    return _ARTICLE_RE.sub("", name.strip())


def _slug(name: str) -> str:
    slug = re.sub(r"[^\w]+", "_", _strip_article(name).lower()).strip("_")
    return slug or "region"


def _parse_count(prompt: str) -> tuple[int, str, str] | None:
    text = prompt or ""
    for pat in (_EXACT_COUNT_DIGIT, _EXACT_COUNT_WORD, _WORD_COUNT):
        m = pat.search(text)
        if not m:
            continue
        n_raw = m.group("n") if "n" in m.groupdict() and m.group("n") else None
        if n_raw is not None:
            n = int(n_raw)
        else:
            n = _WORD_NUMBERS.get((m.group("nword") or "").lower(), 0)
        if n < 2 or n > 6:
            continue
        adj = (m.group("adj") or "").strip()
        noun = (m.group("noun") or "").strip()
        if not noun:
            continue
        return n, adj, noun
    return None


def _count_regions(n: int, adj: str, noun: str) -> list[BoxRegion]:
    margin_x = 0.04
    gap = 0.03
    y1, y2 = 0.28, 0.88
    usable = 1.0 - 2 * margin_x
    box_w = (usable - (n - 1) * gap) / n
    adj_bit = f"{adj} " if adj else ""
    sing = noun.rstrip("s") if noun.endswith("s") and len(noun) > 3 else noun
    pos = f"one {adj_bit}{noun}, isolated {sing}, correct count"
    neg = f"extra {noun}, crowd, merged objects, duplicate"
    regions: list[BoxRegion] = []
    for i in range(n):
        x1 = margin_x + i * (box_w + gap)
        x2 = x1 + box_w
        regions.append(
            BoxRegion(
                name=f"{noun}_{i + 1}",
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                prompt=pos,
                negative=neg,
                priority=8,
            )
        )
    return regions


def _binary_pair(
    a: str,
    b: str,
    *,
    a_box: tuple[float, float, float, float],
    b_box: tuple[float, float, float, float],
    a_priority: int,
    b_priority: int,
    a_suffix: str = "",
    b_suffix: str = "",
) -> list[BoxRegion]:
    a_clause = _first_clause(a).strip()
    b_clause = _first_clause(b).strip()
    a_name = _slug(a_clause)
    b_name = _slug(b_clause)
    return [
        BoxRegion(
            name=a_name,
            x1=a_box[0],
            y1=a_box[1],
            x2=a_box[2],
            y2=a_box[3],
            prompt=f"{a_clause}{a_suffix}".strip(),
            negative="merged objects, duplicate, wrong placement",
            priority=a_priority,
        ),
        BoxRegion(
            name=b_name,
            x1=b_box[0],
            y1=b_box[1],
            x2=b_box[2],
            y2=b_box[3],
            prompt=f"{b_clause}{b_suffix}".strip(),
            negative="merged objects, duplicate, wrong placement",
            priority=b_priority,
        ),
    ]


def _try_binary_spatial(prompt: str) -> list[BoxRegion] | None:
    text = prompt or ""
    m = _LEFT_OF.search(text)
    if m:
        return _binary_pair(
            m.group("a"),
            m.group("b"),
            a_box=(0.04, 0.12, 0.48, 0.95),
            b_box=(0.52, 0.12, 0.96, 0.95),
            a_priority=10,
            b_priority=8,
        )
    m = _RIGHT_OF.search(text)
    if m:
        return _binary_pair(
            m.group("a"),
            m.group("b"),
            a_box=(0.52, 0.12, 0.96, 0.95),
            b_box=(0.04, 0.12, 0.48, 0.95),
            a_priority=10,
            b_priority=8,
        )
    m = _BEHIND.search(text)
    if m:
        return _binary_pair(
            m.group("a"),
            m.group("b"),
            a_box=(0.08, 0.04, 0.92, 0.50),
            b_box=(0.22, 0.38, 0.78, 0.96),
            a_priority=3,
            b_priority=10,
        )
    m = _IN_FRONT.search(text)
    if m:
        return _binary_pair(
            m.group("a"),
            m.group("b"),
            a_box=(0.22, 0.38, 0.78, 0.96),
            b_box=(0.08, 0.04, 0.92, 0.50),
            a_priority=10,
            b_priority=3,
        )
    m = _ON_TOP.search(text)
    if m:
        return _binary_pair(
            m.group("a"),
            m.group("b"),
            a_box=(0.22, 0.04, 0.78, 0.48),
            b_box=(0.16, 0.42, 0.84, 0.96),
            a_priority=10,
            b_priority=8,
        )
    m = _UNDER.search(text)
    if m:
        return _binary_pair(
            m.group("a"),
            m.group("b"),
            a_box=(0.16, 0.42, 0.84, 0.96),
            b_box=(0.22, 0.04, 0.78, 0.48),
            a_priority=8,
            b_priority=10,
        )
    return None


def _occluder_prompt(noun: str) -> str:
    if re.search(r"\bfence\b", noun, re.I):
        return f"{noun} in the foreground, sharp wire mesh, in front of the subject"
    return f"{noun} in the foreground, in front of the subject"


def _try_occlusion(prompt: str) -> list[BoxRegion] | None:
    text = prompt or ""
    if not _OCCLUSION_HINT.search(text):
        return None

    subj = ""
    occ = ""
    for pat in (_OCC_PARTLY_HIDDEN, _OCC_BEHIND):
        m = pat.search(text)
        if m:
            subj = (m.group("subj") or "").strip()
            occ = (m.group("occ") or "").strip()
            if subj or occ:
                break

    if not occ:
        noun_m = _OCC_NOUN.search(text)
        occ = noun_m.group(1) if noun_m else "occluder"
    if not subj:
        head = re.split(
            r"\b(?:partly\s+hidden\s+behind|behind|visible\s+through|through\s+the\s+gaps)\b",
            text,
            maxsplit=1,
            flags=re.I,
        )[0]
        subj = head.strip(" ,") or "the subject"

    occ_noun = _first_clause(_strip_article(occ)) or "occluder"
    subj_clause = _first_clause(subj) or "the subject"
    return [
        BoxRegion(
            name="occluder",
            x1=0.05,
            y1=0.08,
            x2=0.95,
            y2=0.92,
            prompt=_occluder_prompt(occ_noun),
            negative=_OCC_NEG,
            priority=12,
        ),
        BoxRegion(
            name="subject",
            x1=0.18,
            y1=0.12,
            x2=0.82,
            y2=0.88,
            prompt=f"{subj_clause} visible through gaps, not melted into the occluder",
            negative=_OCC_SUBJ_NEG,
            priority=6,
        ),
    ]


def _try_reflection(prompt: str) -> list[BoxRegion] | None:
    text = prompt or ""
    if not _REFLECTION_WORD.search(text) and not _CHROME_SURFACE.search(text):
        return None

    head = re.split(
        r"\b(?:a\s+sharp\s+)?reflections?\b|\breflected\s+in\b",
        text,
        maxsplit=1,
        flags=re.I,
    )[0].strip(" ,")
    subj = _first_clause(head) if head else _first_clause(text)
    if not subj:
        subj = text.strip()
    return [
        BoxRegion(
            name="subject",
            x1=0.15,
            y1=0.05,
            x2=0.85,
            y2=0.58,
            prompt=subj,
            negative=_REFL_NEG,
            priority=10,
        ),
        BoxRegion(
            name="reflection",
            x1=0.15,
            y1=0.52,
            x2=0.85,
            y2=0.98,
            prompt=(
                "reflection of the subject, vertically inverted, glossy surface, "
                "not a second copy of the object sitting below"
            ),
            negative=_REFL_NEG,
            priority=8,
        ),
    ]


def _try_between(prompt: str) -> list[BoxRegion] | None:
    m = _BETWEEN.search(prompt or "")
    if not m:
        return None
    a = _first_clause(m.group("a")).strip()
    b = _first_clause(m.group("b")).strip()
    c = _first_clause(m.group("c")).strip()
    if not a or not b or not c:
        return None
    neg = "merged objects, duplicate, wrong placement"
    columns = (
        (b, (0.04, 0.12, 0.34, 0.95), 8),
        (a, (0.33, 0.12, 0.67, 0.95), 10),
        (c, (0.66, 0.12, 0.96, 0.95), 8),
    )
    return [
        BoxRegion(
            name=_slug(label),
            x1=box[0],
            y1=box[1],
            x2=box[2],
            y2=box[3],
            prompt=label,
            negative=neg,
            priority=prio,
        )
        for label, box, prio in columns
    ]


def _try_next_to(prompt: str) -> list[BoxRegion] | None:
    m = _NEXT_TO.search(prompt or "")
    if not m:
        return None
    return _binary_pair(
        m.group("a"),
        m.group("b"),
        a_box=(0.04, 0.12, 0.48, 0.95),
        b_box=(0.52, 0.12, 0.96, 0.95),
        a_priority=10,
        b_priority=8,
    )


def _try_holding(prompt: str) -> list[BoxRegion] | None:
    text = prompt or ""
    if re.search(r"\bholding\s+(?:still|on|onto|back|off)\b", text, re.I):
        return None
    m = _HOLDING.search(text)
    if not m:
        return None
    obj = _first_clause(_strip_article(m.group("obj") or "")).strip()
    if len(obj) < 2:
        return None
    holder = _first_clause(text[: m.start()]).strip() or "the subject"
    if len(holder) > 80:
        holder = "the subject"
    return [
        BoxRegion(
            name="holder",
            x1=0.16,
            y1=0.08,
            x2=0.90,
            y2=0.82,
            prompt=f"{holder}, occupying the frame",
            negative="empty hands, missing held object, floating prop",
            priority=8,
        ),
        BoxRegion(
            name="held_object",
            x1=0.06,
            y1=0.48,
            x2=0.46,
            y2=0.94,
            prompt=f"{obj} held in the hand, gripped, not floating",
            negative="missing object, extra copy, fused into the body",
            priority=11,
        ),
    ]


def _try_foreground_background(prompt: str) -> list[BoxRegion] | None:
    text = prompt or ""
    fg_m = _FOREGROUND.search(text)
    bg_m = _BACKGROUND.search(text)
    if not fg_m or not bg_m:
        return None
    fg = _first_clause(fg_m.group("fg")).strip()
    bg = _first_clause(bg_m.group("bg")).strip()
    if not fg or not bg or fg.lower() == bg.lower():
        return None
    return [
        BoxRegion(
            name="background",
            x1=0.04,
            y1=0.04,
            x2=0.96,
            y2=0.62,
            prompt=f"{bg} in the background, farther, slightly smaller",
            negative="foreground object occupying the back, merged planes",
            priority=4,
        ),
        BoxRegion(
            name="foreground",
            x1=0.18,
            y1=0.38,
            x2=0.92,
            y2=0.96,
            prompt=f"{fg} in the foreground, nearer, in front",
            negative="pushed to the background, missing occluder",
            priority=11,
        ),
    ]


def _try_attribute_bind(prompt: str) -> list[BoxRegion] | None:
    text = prompt or ""
    if not re.search(r"\bleft\b", text, re.I) or not re.search(r"\bright\b", text, re.I):
        return None
    left_m = re.search(
        rf"(?P<subj>(?:{_PERSON_TERMS})[^,.;]*?)\s+(?:on\s+the\s+)?left\b|"
        rf"\bleft[^,.;]*?(?P<subj2>(?:{_PERSON_TERMS})[^,.;]*)",
        text,
        re.I,
    )
    right_m = re.search(
        rf"(?P<subj>(?:{_PERSON_TERMS})[^,.;]*?)\s+(?:on\s+the\s+)?right\b|"
        rf"\bright[^,.;]*?(?P<subj2>(?:{_PERSON_TERMS})[^,.;]*)",
        text,
        re.I,
    )
    if not left_m or not right_m:
        return None
    left_subj = (left_m.group("subj") or left_m.group("subj2") or "").strip()
    right_subj = (right_m.group("subj") or right_m.group("subj2") or "").strip()
    if not left_subj or not right_subj:
        return None
    return _binary_pair(
        left_subj,
        right_subj,
        a_box=(0.04, 0.12, 0.48, 0.95),
        b_box=(0.52, 0.12, 0.96, 0.95),
        a_priority=10,
        b_priority=10,
    )


def _glyph_box(prompt: str) -> tuple[float, float, float, float]:
    low = (prompt or "").lower()
    if re.search(r"\b(?:sign|shop|storefront)\b", low):
        return (0.18, 0.08, 0.82, 0.42)
    if re.search(r"\b(?:poster|banner)\b", low):
        return (0.12, 0.22, 0.88, 0.62)
    return (0.2, 0.18, 0.8, 0.48)


def _glyph_region(prompt: str, texts: list[str]) -> BoxRegion:
    text = texts[0]
    x1, y1, x2, y2 = _glyph_box(prompt)
    return BoxRegion(
        name="glyph_text",
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        prompt=f'perfectly spelled lettering that reads "{text}"',
        negative="misspelled text, garbled letters, extra words",
        priority=12,
    )


def compile_spatial_layout(prompt: str) -> BoxLayoutSpec | None:
    """Return a regional box spec or None if the prompt has no layout signal."""
    text = (prompt or "").strip()
    if not text:
        return None

    regions: list[BoxRegion] = []
    global_negative = ""

    count = _parse_count(text)
    if count is not None:
        n, adj, noun = count
        regions.extend(_count_regions(n, adj, noun))

    if not regions:
        occluded = _try_occlusion(text)
        if occluded:
            regions.extend(occluded)
            global_negative = _OCC_GLOBAL_NEG

    if not regions:
        reflected = _try_reflection(text)
        if reflected:
            regions.extend(reflected)
            global_negative = _REFL_NEG

    if not regions:
        spatial = _try_binary_spatial(text)
        if spatial:
            regions.extend(spatial)

    if not regions:
        between = _try_between(text)
        if between:
            regions.extend(between)

    if not regions:
        next_to = _try_next_to(text)
        if next_to:
            regions.extend(next_to)

    if not regions:
        holding = _try_holding(text)
        if holding:
            regions.extend(holding)

    if not regions:
        depth = _try_foreground_background(text)
        if depth:
            regions.extend(depth)

    if not regions:
        bound = _try_attribute_bind(text)
        if bound:
            regions.extend(bound)

    glyphs = quoted_glyph_texts(text)
    if glyphs:
        regions.append(_glyph_region(text, glyphs))

    if not regions:
        return None

    return BoxLayoutSpec(
        global_prompt=text,
        global_negative=global_negative,
        regions=regions,
        feather_px=10,
        overlap_mode="priority",
    )
