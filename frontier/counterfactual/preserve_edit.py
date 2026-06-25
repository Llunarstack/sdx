"""
Counterfactual editing: change one attribute, lock everything else.

Maps to img2img strength + regional masks + negative "do not change X".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class PreserveSpec:
    locked_phrases: Tuple[str, ...]
    change_phrase: str
    strength_hint: float  # 0..1 img2img / latent edit strength


@dataclass(frozen=True)
class CounterfactualEdit:
    original: str
    edited: str
    preserve: PreserveSpec
    inpaint_region: str = ""  # optional box name


_CHANGE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bchange\s+(?:the\s+)?(.+?)\s+to\s+(.+?)(?:\.|$|,)", "change"),
    (r"\bmake\s+(?:the\s+)?(.+?)\s+(.+?)(?:\.|$|,)", "make"),
    (r"\breplace\s+(?:the\s+)?(.+?)\s+with\s+(.+?)(?:\.|$|,)", "replace"),
    (r"\bturn\s+(?:the\s+)?(.+?)\s+into\s+(.+?)(?:\.|$|,)", "turn"),
)


class PreserveEditPlanner:
    """Parse natural-language edit requests into preserve/change specs."""

    def parse(self, edit_instruction: str, *, base_prompt: str = "") -> CounterfactualEdit | None:
        text = (edit_instruction or "").strip()
        if not text:
            return None
        for pattern, _kind in _CHANGE_PATTERNS:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if not m:
                continue
            target, new_val = m.group(1).strip(), m.group(2).strip()
            locked = _lock_phrases(base_prompt or text, exclude={target.lower()})
            edited = _apply_substitution(base_prompt or text, target, new_val)
            preserve = PreserveSpec(
                locked_phrases=tuple(locked),
                change_phrase=f"{target} as {new_val}",
                strength_hint=0.45 if len(locked) > 3 else 0.35,
            )
            return CounterfactualEdit(original=base_prompt or text, edited=edited, preserve=preserve)
        return None

    def negative_lock(self, spec: PreserveSpec) -> str:
        if not spec.locked_phrases:
            return "altered composition, different subject"
        parts = [f"do not change {p}" for p in spec.locked_phrases[:5]]
        return ", ".join(parts)

    def regional_prompts(self, edit: CounterfactualEdit) -> Tuple[str, str]:
        """(global_locked, local_change) for two-pass regional."""
        global_p = ", ".join(edit.preserve.locked_phrases) if edit.preserve.locked_phrases else edit.original
        local = edit.preserve.change_phrase
        return global_p, local


def _lock_phrases(prompt: str, exclude: set[str]) -> List[str]:
    chunks = [c.strip() for c in re.split(r"[,;]", prompt) if c.strip()]
    locked: List[str] = []
    for c in chunks:
        if any(ex in c.lower() for ex in exclude):
            continue
        if len(c.split()) >= 2:
            locked.append(c)
    return locked[:8]


def _apply_substitution(prompt: str, target: str, new_val: str) -> str:
    return re.sub(re.escape(target), new_val, prompt, count=1, flags=re.IGNORECASE)


__all__ = ["CounterfactualEdit", "PreserveEditPlanner", "PreserveSpec"]
