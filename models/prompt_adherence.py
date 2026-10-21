"""Rule-based prompt structure for adherence tooling.

``PromptParser`` pulls (subject, attributes, relations, negation, count) out of raw prompt text: the
training-free text-embedding modulation in ``utils/generation/quality_stack``, the video prompt graph and the
attention heatmap below all build on it. ``attribute_binding_heatmap`` turns captured cross-attention into a
2D map of where the parsed subjects landed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class SemanticTriple:
    """A (subject, attribute, relation) triple parsed from a prompt."""

    subject: str
    attributes: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    negated: bool = False
    count: int = 1
    token_indices: list[int] = field(default_factory=list)  # word positions of the attributes + subject


@dataclass(slots=True)
class ParsedPrompt:
    triples: list[SemanticTriple]
    raw: str
    negation_indices: list[int] = field(default_factory=list)  # word positions of negation cues
    count_constraints: dict[str, int] = field(default_factory=dict)  # subject -> count


class PromptParser:
    """Rule-based parser for clothing/object subjects, their preceding attributes, negation, counts and
    spatial relations; works on raw words before tokenisation."""

    # Colour, size, material, pattern and condition attributes.
    _ATTR_RE = re.compile(
        "|".join(
            (
                r"\b(red|orange|yellow|green|blue|purple|pink|black|white|gray|grey|brown|golden|silver|dark|light|bright|pale|vivid|neon)\b",
                r"\b(large|small|big|tiny|huge|tall|short|wide|narrow|thick|thin|long)\b",
                r"\b(wooden|metal|leather|silk|cotton|wool|plastic|glass|stone|fabric|denim|lace)\b",
                r"\b(striped|checkered|floral|plain|patterned|solid|transparent|opaque|shiny|matte|glossy)\b",
                r"\b(old|new|worn|torn|clean|dirty|wet|dry|crumpled|pressed|fitted|loose|tight)\b",
            )
        ),
        re.I,
    )
    _SUBJECT_RE = re.compile(
        r"\b(shirt|blouse|dress|skirt|pants|jeans|jacket|coat|hoodie|sweater|vest|suit|tie|scarf|hat|cap|shoes|boots|sneakers|heels|gloves|socks|underwear|bra|bikini|swimsuit|uniform|robe|cloak|armor|helmet"
        r"|bag|backpack|purse|wallet|watch|glasses|sunglasses|necklace|bracelet|ring|earrings|sword|gun|staff|wand|book|phone|laptop|cup|bottle|umbrella|flower|crown|mask|cube|sphere|cone|cylinder|pyramid|box|ball|block|torus)\b",
        re.I,
    )

    # Spatial phrases, longest first: (words, relation on the preceding subject, inverse on the following one)
    _SPATIAL = (
        (("to", "the", "left", "of"), "left", "right"),
        (("to", "the", "right", "of"), "right", "left"),
        (("left", "of"), "left", "right"),
        (("right", "of"), "right", "left"),
        (("on", "top", "of"), "above", "below"),
        (("in", "front", "of"), "in front", "behind"),
        (("next", "to"), "beside", "beside"),
        (("beside",), "beside", "beside"),
        (("above",), "above", "below"),
        (("below",), "below", "above"),
        (("beneath",), "below", "above"),
        (("under",), "below", "above"),
        (("behind",), "behind", "in front"),
    )

    _NEGATION_RE = re.compile(r"\b(no|not|without|lacking|missing|absent)\b")
    _NEGATION_PHRASES = (("devoid", "of"), ("free", "of"))
    _NEGATION_SPAN = 3  # words after a negation cue that count as negated

    _COUNTS = {
        "one": 1,
        "a": 1,
        "an": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    def parse(self, prompt: str) -> ParsedPrompt:
        """Parse a prompt string into semantic triples."""
        words = [w.strip(".,;:!?()[]\"'") for w in prompt.lower().split()]

        negation_indices: list[int] = []
        neg_positions: set[int] = set()
        for i, w in enumerate(words):
            cue_len = 2 if tuple(words[i : i + 2]) in self._NEGATION_PHRASES else int(bool(self._NEGATION_RE.match(w)))
            if cue_len:
                negation_indices.append(i)
                neg_positions.update(range(i + cue_len, min(i + cue_len + self._NEGATION_SPAN, len(words))))

        triples: list[SemanticTriple] = []
        count_constraints: dict[str, int] = {}
        for i, word in enumerate(words):
            count = self._COUNTS.get(word)
            if count is not None and i + 1 < len(words):
                count_constraints[words[i + 1]] = count
            if not self._SUBJECT_RE.match(word):
                continue
            # Attributes bind from the 4 words before the subject, unless negated.
            attr_idx = [j for j in range(max(0, i - 4), i) if self._ATTR_RE.match(words[j]) and j not in neg_positions]
            triples.append(
                SemanticTriple(
                    subject=word,
                    attributes=[words[j] for j in attr_idx],
                    negated=i in neg_positions,
                    count=count_constraints.get(word, 1),
                    token_indices=[*attr_idx, i],
                )
            )

        for k, rel, inverse in self._spatial_relations(words):
            before = [t for t in triples if t.token_indices[-1] < k]
            after = [t for t in triples if t.token_indices[-1] > k]
            if before and after:
                before[-1].relations.append(rel)
                after[0].relations.append(inverse)

        return ParsedPrompt(
            triples=triples,
            raw=prompt,
            negation_indices=negation_indices,
            count_constraints=count_constraints,
        )

    def _spatial_relations(self, words: list[str]):
        """Yield ``(start index, relation, inverse)`` per spatial phrase, longest match first."""
        i = 0
        while i < len(words):
            for phrase, rel, inverse in self._SPATIAL:
                if tuple(words[i : i + len(phrase)]) == phrase:
                    yield i, rel, inverse
                    i += len(phrase)
                    break
            else:
                i += 1


def attribute_binding_heatmap(
    attn_weights: torch.Tensor,
    parsed: ParsedPrompt,
    *,
    spatial_h: int,
    spatial_w: int,
) -> torch.Tensor:
    """Head-averaged cross-attention onto the parsed subject/attribute tokens as an ``(H, W)`` map.

    ``attn_weights`` is ``(heads, N, L)`` or ``(B, heads, N, L)`` (first sample used). When ``N`` does not
    match ``spatial_h * spatial_w`` a square grid is assumed; with no parsed tokens the first 16 are used.
    """
    w = attn_weights
    if w.dim() == 4:
        w = w[0]
    if w.dim() != 3:
        raise ValueError(f"expected attn (H,N,L) or (B,H,N,L), got {tuple(attn_weights.shape)}")
    hw = int(spatial_h) * int(spatial_w)
    if w.shape[1] != hw:
        side = int(round(w.shape[1] ** 0.5))
        spatial_h = spatial_w = side
    heat = w.float().mean(dim=0)  # N, L
    token_idx = [int(i) for t in parsed.triples for i in t.token_indices if 0 <= int(i) < heat.shape[1]]
    if not token_idx:
        token_idx = list(range(min(heat.shape[1], 16)))
    sel = heat[:, token_idx].mean(dim=1)
    return sel.reshape(spatial_h, spatial_w)


__all__ = ["ParsedPrompt", "PromptParser", "SemanticTriple", "attribute_binding_heatmap"]
