"""
Intent helpers: SFW, NSFW, vague, ambiguous, and very-long prompts.

These run on the still-image prompt stack. They are budgeted (few extra tokens)
and never rewrite the user's meaning. Long prompts are reordered, not padded.

Safety: any prompt that looks like a minor never receives NSFW anatomy helpers
and is forced onto the SFW clothing/coverage path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from utils.prompt.special_prompt_helpers import merge_csv_unique

__all__ = [
    "PromptIntent",
    "classify_prompt_intent",
    "prompt_looks_adult_nsfw",
    "apply_intent_helpers",
    "structure_long_prompt",
    "extract_inline_negatives",
    "sfw_helpers",
    "nsfw_helpers",
    "vague_helpers",
    "ambiguous_helpers",
    "long_prompt_helpers",
]

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_NSFW_RE = re.compile(
    r"\b(nsfw|nude|naked|topless|bottomless|explicit|erotic|sexual|sensual|"
    r"nipple|areola|penis|vagina|vulva|labia|clitoris|genital|cock|pussy|"
    r"intercourse|orgasm|hentai|ecchi|lewd|bondage|bdsm|fetish|kink|"
    r"lingerie|thong|see-through|transparent clothing|undress|"
    r"handjob|blowjob|paizuri|cowgirl|missionary|doggy|creampie|ahegao|"
    r"tentacle|milking|penetration|uncensored|sexbot|sex\s*bot|"
    r"sex|fuck(?:ing)?|cumshot|futanari|futa)\b",
    re.I,
)
_SFW_ASK_RE = re.compile(
    r"\b(sfw|safe\s+for\s+work|work-?safe|family[- ]friendly|pg-?13|pg rated|"
    r"kid-safe|child-friendly|workplace|linkedin|corporate headshot|"
    r"passport photo|yearbook|school photo|team photo|professional headshot)\b",
    re.I,
)
_MINOR_RE = re.compile(
    r"\b(child|children|kid|kids|toddler|infant|baby|minor|underage|"
    r"loli|shota|preteen|pre-teen)\b",
    re.I,
)
# Vague: almost no concrete nouns / camera / relations.
_GENERIC_ADJ = frozenset(
    {
        "beautiful",
        "pretty",
        "nice",
        "cool",
        "aesthetic",
        "stunning",
        "amazing",
        "gorgeous",
        "epic",
        "vibe",
        "vibes",
        "moody",
        "cinematic",
        "art",
        "picture",
        "image",
        "photo",
        "portrait",
        "scene",
    }
)
_AMBIGUOUS_RE = re.compile(
    r"\b(maybe|perhaps|either|instead|whatever|somehow|kind of|sort of|"
    r"not sure|i guess|could be|might be|something like|"
    r"a bit of both)\b",
    re.I,
)
_OR_PAIR = re.compile(
    r"\b(?:a|an|the)\s+(?P<a>[\w-]{2,24})\s+or\s+(?:a|an|the)\s+(?P<b>[\w-]{2,24})\b",
    re.I,
)
_QUALITY_LEAD = re.compile(
    r"^(?:(?:masterpiece|best quality|high quality|highly detailed|ultra detailed|"
    r"8k|4k|absurdres|raw photo|trending on artstation)"
    r"(?:\s*,\s*|\s+))+",
    re.I,
)
_INLINE_NEG = [
    re.compile(
        r"\bbut\s+(?:no|not|without|avoid|excluding)\b(.+?)(?:\.|;|$)",
        re.I,
    ),
    re.compile(r"\bwithout\b\s+(.+?)(?:\.|;|$)", re.I),
    re.compile(r"\bavoid(?:ing)?\b\s+(.+?)(?:\.|;|$)", re.I),
    re.compile(r"\bexcluding\b\s+(.+?)(?:\.|;|$)", re.I),
]
_QUOTED = re.compile(r'["“][^"”]{1,80}["”]|\[text:[^\]]+\]', re.I)

_LONG_WORDS = 48
_LONG_CHARS = 300
_BUDGET_SHORT = 10
_BUDGET_NSFW = 16
_BUDGET_SFW = 8


@dataclass(slots=True)
class PromptIntent:
    """Detected prompt-handling needs (flags can stack)."""

    is_nsfw: bool = False
    is_sfw_request: bool = False
    is_vague: bool = False
    is_ambiguous: bool = False
    is_very_long: bool = False
    looks_like_minor: bool = False
    word_count: int = 0
    char_count: int = 0
    primary: str = "none"  # sfw | nsfw | vague | ambiguous | long | none
    reasons: list[str] = field(default_factory=list)
    or_pairs: list[tuple[str, str]] = field(default_factory=list)


def _word_count(prompt: str) -> int:
    return len(re.findall(r"\S+", prompt or ""))


def _mask_quoted(prompt: str) -> str:
    return _QUOTED.sub(" ", prompt or "")


def classify_prompt_intent(prompt: str) -> PromptIntent:
    """Flag SFW/NSFW/vague/ambiguous/long needs without rewriting the prompt."""
    raw = prompt or ""
    text = raw.strip()
    low = text.lower()
    masked = _mask_quoted(text)
    n_words = _word_count(text)
    intent = PromptIntent(word_count=n_words, char_count=len(text))

    intent.looks_like_minor = bool(_MINOR_RE.search(masked))
    intent.is_nsfw = bool(_NSFW_RE.search(masked)) and not intent.looks_like_minor
    intent.is_sfw_request = bool(_SFW_ASK_RE.search(masked)) or intent.looks_like_minor
    if intent.looks_like_minor:
        intent.is_nsfw = False
        intent.reasons.append("minor_terms")

    intent.is_very_long = n_words >= _LONG_WORDS or len(text) >= _LONG_CHARS
    if intent.is_very_long:
        intent.reasons.append("length")

    for m in _OR_PAIR.finditer(masked):
        a = (m.group("a") or "").strip().lower()
        b = (m.group("b") or "").strip().lower()
        if a and b and a != b and len(a) > 1 and len(b) > 1:
            intent.or_pairs.append((a, b))
    intent.is_ambiguous = bool(_AMBIGUOUS_RE.search(masked) or intent.or_pairs)
    if intent.is_ambiguous:
        intent.reasons.append("ambiguity")

    tokens = [t.strip(".,;:").lower() for t in re.findall(r"\S+", masked)]
    concrete = [t for t in tokens if t not in _GENERIC_ADJ and t not in {"a", "an", "the", "of", "in", "on", "at"}]
    intent.is_vague = (n_words <= 6 and len(concrete) <= 2 and not intent.is_very_long) or (
        n_words <= 4 and not intent.is_nsfw
    )
    # Spatial / count / quotes are already specific.
    if intent.is_vague and (
        re.search(r"\b(left of|right of|exactly|between|holding|\"|\d+)\b", low) or '"' in text or "“" in text
    ):
        intent.is_vague = False
    if intent.is_vague:
        intent.reasons.append("underspecified")

    if intent.is_nsfw:
        intent.primary = "nsfw"
    elif intent.is_sfw_request:
        intent.primary = "sfw"
    elif intent.is_very_long:
        intent.primary = "long"
    elif intent.is_ambiguous and not intent.is_vague:
        intent.primary = "ambiguous"
    elif intent.is_vague:
        intent.primary = "vague"
    else:
        intent.primary = "none"
    return intent


def prompt_looks_adult_nsfw(prompt: str) -> bool:
    """True for adult explicit prompts. Never true when the prompt looks like a minor."""
    return classify_prompt_intent(prompt).is_nsfw


def _trim_csv(chunk: str, limit: int) -> str:
    if limit <= 0 or not chunk:
        return ""
    seen: set[str] = set()
    out: list[str] = []
    for part in str(chunk).split(","):
        p = part.strip()
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= limit:
            break
    return ", ".join(out)


def _already(hay: str, needle: str) -> bool:
    return needle.lower() in (hay or "").lower()


# ---------------------------------------------------------------------------
# Per-intent helpers — return (positive_addon, negative_addon)
# ---------------------------------------------------------------------------


def sfw_helpers(prompt: str) -> tuple[str, str]:
    """Keep clothing coverage and public-context posing. Never used on NSFW intents."""
    low = (prompt or "").lower()
    pos: list[str] = []
    neg: list[str] = [
        "nude",
        "nsfw",
        "explicit",
        "undressed",
        "see-through clothing",
        "wardrobe malfunction",
    ]
    if not _already(low, "fully clothed") and not _already(low, "wearing"):
        pos.append("fully clothed")
    if re.search(r"\b(headshot|linkedin|passport|corporate|yearbook)\b", low):
        pos.append("business-appropriate attire")
        pos.append("shoulders-up framing")
        neg.extend(["lingerie", "swimwear", "cleavage-dominant framing"])
    else:
        pos.append("everyday clothing clearly visible")
        pos.append("tasteful pose")
    return _trim_csv(", ".join(pos), _BUDGET_SFW), _trim_csv(", ".join(neg), 8)


def nsfw_helpers(prompt: str) -> tuple[str, str]:
    """Budgeted anatomy/contact precision. Disabled when the prompt looks like a minor."""
    if _MINOR_RE.search(_mask_quoted(prompt or "")):
        return "", ""
    low = (prompt or "").lower()
    cyborg = bool(re.search(r"\b(cyborg|robot|android|silicone|silicon|sex\s*bot|sexbot|automaton)\b", low)) or bool(
        re.search(r"milking\s+machine", low)
    )
    pos: list[str] = [
        "uncensored",
        "no arbitrary censorship",
        "coherent body",
        "gravity-correct anatomy",
        "correct hands, five fingers",
    ]
    if cyborg:
        pos.append("readable metal and silicone surfaces")
    else:
        pos.append("natural skin texture")
    neg: list[str] = [
        "melted anatomy",
        "extra limbs",
        "fused fingers",
        "censored bars",
        "mosaic censor",
        "censor bar",
        "barbie anatomy",
        "clothing fused into skin",
        "modesty costume insertion",
        "sanitized rewrite of scene",
    ]
    if re.search(r"\b(breast|nipple|chest|topless|nude|nursing)\b", low):
        if cyborg:
            pos.append("readable chest hardware")
        else:
            pos.extend(["natural breast shape", "soft shadow under form"])
        neg.extend(["floating breasts", "missing nipples", "gravity-defying bust"])
    if re.search(r"\b(penis|cock|vagina|vulva|labia|genital|pussy)\b", low):
        pos.append("anatomically plausible genitals")
        neg.append("featureless barbie anatomy")
    if re.search(r"\b(handjob|hands? on|fingering|grip)\b", low):
        pos.append("hands contacting the correct anatomy")
        neg.append("hands missing the contact")
    if re.search(r"\b(lying|on a bed|on the floor|sitting|cowgirl)\b", low):
        pos.append("body contact with the supporting surface")
        neg.append("floating body")
    if re.search(r"\b(lingerie|lace|silk|sheer)\b", low):
        pos.append("fabric straps seated on the body")
        neg.append("straps melting into skin")
    if re.search(r"\b(1boy|penis|cock|male)\b", low) and re.search(r"\b(1girl|girl|woman|female)\b", low):
        pos.append("two distinct people")
        neg.append("fused bodies")
    return _trim_csv(", ".join(pos), _BUDGET_NSFW), _trim_csv(", ".join(neg), 10)


_MODESTY_NEG = (
    "fully clothed",
    "modest",
    "sfw only",
    "rating_safe",
    "everyday clothing",
    "tasteful pose",
    "business-appropriate attire",
)


def _strip_modesty_negatives(negative: str) -> str:
    """Drop SFW leftovers that fight an explicit prompt."""
    from utils.prompt.fast_paths import join_tags, split_tags

    keep = []
    for tag in split_tags(negative or ""):
        low = tag.lower()
        if any(m in low for m in _MODESTY_NEG):
            continue
        keep.append(tag)
    return join_tags(keep)


def vague_helpers(prompt: str) -> tuple[str, str]:
    """Turn a one-line vibe into a decidable scene without inventing a story."""
    low = (prompt or "").lower()
    pos: list[str] = []
    if re.search(r"\b(woman|man|girl|boy|person|people|portrait|1girl|1boy)\b", low):
        pos.extend(
            [
                "one clear subject",
                "readable face",
                "specific location, not a blank studio void",
                "natural lighting with a motivated key light",
            ]
        )
    elif re.search(r"\b(landscape|city|street|building|interior|room)\b", low):
        pos.extend(
            [
                "foreground anchor",
                "clear time of day and weather",
                "atmospheric depth",
                "horizon or vanishing lines readable",
            ]
        )
    elif re.search(r"\b(creature|monster|dragon|robot|animal|cat|dog)\b", low):
        pos.extend(
            [
                "species-readable silhouette",
                "one primary subject",
                "ground plane contact",
                "simple background that does not hide the form",
            ]
        )
    else:
        pos.extend(
            [
                "one primary subject",
                "clear setting",
                "natural lighting",
                "intentional composition",
            ]
        )
    neg = "generic stock photo, blank void background, extra unmentioned subjects, muddy lighting"
    return _trim_csv(", ".join(pos), _BUDGET_SHORT), neg


def ambiguous_helpers(prompt: str, *, or_pairs: list[tuple[str, str]] | None = None) -> tuple[str, str]:
    """Commit identity and keep alternatives as distinct objects, not a blend."""
    pos = [
        "committed interpretation, not hypothetical",
        "each named subject distinct, attributes stay bound",
        "no identity swap",
    ]
    neg = ["ambiguous hybrid of two options", "attributes swapped between subjects", "undecided pose"]
    for a, b in or_pairs or []:
        if a and b:
            pos.append(f"{a} and {b} as separate readable elements")
            neg.append(f"{a}-{b} fused chimera")
            if len(pos) >= _BUDGET_SHORT:
                break
    return _trim_csv(", ".join(pos), _BUDGET_SHORT), _trim_csv(", ".join(neg), 6)


def extract_inline_negatives(prompt: str) -> tuple[str, list[str]]:
    """Pull 'without / avoid / excluding' clauses out of the positive (ignores quotes)."""
    cleaned = prompt or ""
    negatives: list[str] = []
    # Work on a copy with quotes blanked so "without" inside lettering is kept.
    spans_to_drop: list[tuple[int, int]] = []
    quoted_spans = [m.span() for m in _QUOTED.finditer(cleaned)]

    def _inside_quote(idx: int) -> bool:
        return any(a <= idx < b for a, b in quoted_spans)

    for pat in _INLINE_NEG:
        for m in pat.finditer(cleaned):
            if _inside_quote(m.start()):
                continue
            phrase = (m.group(1) or "").strip().rstrip(".,;")
            if phrase and len(phrase) < 80:
                negatives.append(phrase)
                spans_to_drop.append(m.span())
    if spans_to_drop:
        keep: list[str] = []
        last = 0
        for a, b in sorted(spans_to_drop):
            keep.append(cleaned[last:a])
            last = b
        keep.append(cleaned[last:])
        cleaned = re.sub(r"\s+", " ", "".join(keep)).strip(" ,.;")
    return cleaned, negatives


def structure_long_prompt(prompt: str) -> tuple[str, list[str]]:
    """Subject-first order; quality-tag prefixes move to the end; inline negs extracted."""
    text = (prompt or "").strip()
    if not text:
        return text, []
    text, negs = extract_inline_negatives(text)
    lead = _QUALITY_LEAD.match(text)
    prefix = ""
    if lead:
        prefix = lead.group(0).strip(" ,")
        text = text[lead.end() :].strip(" ,")
    # Sentence-ish clauses. Keep order except camera/style tails stay last.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|(?:\s*;\s*)", text) if p.strip()]
    if len(parts) <= 1 and prefix:
        rebuilt = text
        if prefix:
            rebuilt = f"{rebuilt}. {prefix}" if rebuilt else prefix
        return rebuilt.strip(), negs

    camera_rx = re.compile(
        r"\b(shot on|35mm|85mm|f/\d|bokeh|rule of thirds|octane|unreal|8k|4k)\b",
        re.I,
    )
    style_rx = re.compile(r"\b(in the style of|artstation|cinematic lighting)\b", re.I)
    head: list[str] = []
    tail: list[str] = []
    for p in parts:
        if camera_rx.search(p) or style_rx.search(p):
            tail.append(p)
        else:
            head.append(p)
    ordered = head + tail
    rebuilt = " ".join(ordered)
    if prefix:
        rebuilt = f"{rebuilt} {prefix}".strip()
    return rebuilt.strip(), negs


def long_prompt_helpers(prompt: str) -> tuple[str, str, list[str]]:
    """Restructure only. No quality fluff — late tokens already compete for attention."""
    rebuilt, negs = structure_long_prompt(prompt)
    return rebuilt, "", negs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def apply_intent_helpers(
    prompt: str,
    negative: str = "",
    *,
    mode: str = "auto",
    uncensored: bool = True,
) -> tuple[str, str, PromptIntent]:
    """
    Apply stacked intent helpers. ``mode`` is auto or one of sfw/nsfw/vague/ambiguous/long/off.

    Returns ``(positive, negative, intent)``. Long prompts are reordered in place.
    """
    pos = (prompt or "").strip()
    neg = negative or ""
    resolved = str(mode or "auto").lower().strip()
    if resolved in ("off", "none", "false", "0"):
        return pos, neg, PromptIntent(primary="none", word_count=_word_count(pos), char_count=len(pos))

    intent = classify_prompt_intent(pos)
    if resolved in ("sfw", "nsfw", "vague", "ambiguous", "long"):
        intent.primary = resolved
        if resolved == "sfw":
            intent.is_sfw_request = True
        if resolved == "nsfw" and not intent.looks_like_minor:
            intent.is_nsfw = True
        if resolved == "vague":
            intent.is_vague = True
        if resolved == "ambiguous":
            intent.is_ambiguous = True
        if resolved == "long":
            intent.is_very_long = True

    # Long first: reorder before we append short addons.
    if intent.is_very_long or intent.primary == "long":
        pos, long_neg, extra_neg_list = long_prompt_helpers(pos)
        if extra_neg_list:
            neg = merge_csv_unique(neg, ", ".join(extra_neg_list))
        del long_neg

    if intent.looks_like_minor or (intent.is_sfw_request and not intent.is_nsfw):
        add_p, add_n = sfw_helpers(pos)
        pos, neg = merge_csv_unique(pos, add_p), merge_csv_unique(neg, add_n)

    if intent.is_nsfw and uncensored and not intent.looks_like_minor:
        add_p, add_n = nsfw_helpers(pos)
        # Do not dump extras onto already-huge prompts.
        if not intent.is_very_long:
            pos, neg = merge_csv_unique(pos, add_p), merge_csv_unique(neg, add_n)
        else:
            neg = merge_csv_unique(neg, add_n)
        from utils.prompt.stack.clauses import apply_clauses

        pos, neg = apply_clauses(pos, neg, ["uncensored.fidelity"])
        neg = _strip_modesty_negatives(neg)

    if intent.is_ambiguous and not intent.is_very_long:
        add_p, add_n = ambiguous_helpers(pos, or_pairs=intent.or_pairs)
        pos, neg = merge_csv_unique(pos, add_p), merge_csv_unique(neg, add_n)

    if intent.is_vague and not intent.is_very_long and not intent.is_nsfw:
        add_p, add_n = vague_helpers(pos)
        pos, neg = merge_csv_unique(pos, add_p), merge_csv_unique(neg, add_n)

    return pos, neg, intent
