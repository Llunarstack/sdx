"""
Multi-character / multi-outfit / interaction-hard prompts.

T5/DiT follow **disambiguated** captions better than a flat comma soup. Use:

- **Count tags** early (``2girls``, ``1boy 1girl``, ``3girls``).
- **Per-subject blocks**: identity + outfit + pose + screen position (left/right).
- **Relations** after subjects: ``holding hands``, ``facing each other``.

Training data should mirror the same structure (see ``TRAINING_CAPTION_GUIDE``).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# Extra negatives when multiple distinct identities/outfits must not bleed together.
MULTI_SUBJECT_NEGATIVE_EXTRAS: Tuple[str, ...] = (
    "same outfit on both",
    "outfit swap",
    "clothes mixed between characters",
    "wrong outfit on wrong person",
    "identical twins when not intended",
    "cloned face on both",
    "merged bodies",
    "conjoined",
    "shared limbs between people",
    "pose copied onto wrong figure",
    "mirror duplicate pose error",
)

# Positives that encourage separability (used with composition_mode ``multi_character``).
MULTI_SUBJECT_POSITIVE_EXTRAS: Tuple[str, ...] = (
    "each person has distinct clothing",
    "each figure has own pose",
    "readable silhouettes",
    "spatial separation between subjects",
)


def default_subject_labels(n: int) -> List[str]:
    if n <= 0:
        return []
    if n == 1:
        return ["subject"]
    return [f"character {i + 1}" for i in range(n)]


def merge_character_sheet_positives(
    blocks: Sequence[str],
    *,
    labels: Optional[Sequence[str]] = None,
) -> str:
    """
    Join multiple character-sheet positive strings with **labeled** segments so the
    text encoder sees separate subjects (not one blended description).
    """
    parts: List[str] = []
    raw = [b.strip() for b in blocks if b and str(b).strip()]
    if not raw:
        return ""
    if len(raw) == 1:
        return raw[0]
    lab = list(labels) if labels is not None else default_subject_labels(len(raw))
    while len(lab) < len(raw):
        lab.append(f"character {len(lab) + 1}")
    for i, chunk in enumerate(raw):
        label = lab[i].strip() if i < len(lab) else f"character {i + 1}"
        parts.append(f"({label}: {chunk})")
    return ", ".join(parts)


def merge_character_sheet_negatives(blocks: Sequence[str]) -> str:
    """Deduped merge of negative strings from several sheets."""
    seen = set()
    out: List[str] = []
    for b in blocks:
        if not b or not str(b).strip():
            continue
        for tok in str(b).split(","):
            t = tok.strip()
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
    return ", ".join(out)


def multi_sheet_extra_negatives_csv() -> str:
    """Comma-separated negatives focused on outfit/face/pose confusion (multi-sheet scenes)."""
    return ", ".join(MULTI_SUBJECT_NEGATIVE_EXTRAS)


TRAINING_CAPTION_GUIDE = """
Multi-character training captions (JSONL ``caption`` / text field):

1. Start with counts: ``2girls``, ``1boy 1girl``, ``3girls``, ``multiple boys``, etc.
2. Describe **A** then **B** with explicit anchors: ``left: 1girl, long red hair, blue dress,
   arms crossed`` — ``right: 1girl, short black hair, white suit, waving``.
3. Add interaction after both: ``looking at each other``, ``standing back to back``.
4. Avoid repeating the same outfit phrase for both without color/shape words; models blend tokens.
5. For hard poses, name limbs: ``left character's right hand holds right character's left hand``.

Same structure at inference (--composition-mode multi_character, --scene-blueprint, labeled
character sheets, or ``--prompt-layout`` JSON via ``utils.prompt.prompt_layout``).
""".strip()


__all__ = [
    "MULTI_SUBJECT_NEGATIVE_EXTRAS",
    "MULTI_SUBJECT_POSITIVE_EXTRAS",
    "TRAINING_CAPTION_GUIDE",
    "default_subject_labels",
    "merge_character_sheet_positives",
    "merge_character_sheet_negatives",
    "multi_sheet_extra_negatives_csv",
]
