"""
Prompt mutator — generate creative variants without re-prompting the user.

Used for best-of-N, auto-refine, and "explore" mode. This is structurally different
from appending static guidance tags.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence, Tuple


class MutationAxis(str, Enum):
    TIME_SHIFT = "time_shift"
    WEATHER_SWAP = "weather_swap"
    SCALE_FLIP = "scale_flip"
    PALETTE_INVERT = "palette_invert"
    MEDIUM_SWAP = "medium_swap"
    MOOD_FLIP = "mood_flip"
    DETAIL_ZOOM = "detail_zoom"


@dataclass(frozen=True)
class PromptMutation:
    axis: MutationAxis
    original: str
    mutated: str
    seed_salt: int


_TIME = (("dawn", "midnight"), ("noon", "blue hour"), ("summer", "winter"))
_WEATHER = (("clear sky", "sudden rain"), ("calm", "windstorm"), ("fog", "harsh sun"))
_MOOD = (("serene", "ominous"), ("joyful", "melancholic"), ("chaotic", "still"))
_MEDIUM = (("oil painting", "charcoal sketch"), ("photograph", "watercolor"), ("3d render", "linocut"))
_SCALE = (("wide establishing shot", "extreme close-up"), ("tiny figure", "colossal figure"))


class PromptMutator:
    """Deterministic creative mutations from prompt + seed."""

    def __init__(self, axes: Sequence[MutationAxis] | None = None) -> None:
        self.axes = tuple(axes or MutationAxis)

    def mutate_one(self, prompt: str, *, seed: int = 0, axis: MutationAxis | None = None) -> PromptMutation:
        text = (prompt or "").strip()
        ax = axis or self._pick_axis(text, seed)
        mutated, salt = self._apply(text, ax, seed)
        return PromptMutation(ax, text, mutated, salt)

    def mutate_batch(self, prompt: str, *, seed: int = 0, count: int = 4) -> List[PromptMutation]:
        text = (prompt or "").strip()
        out: List[PromptMutation] = []
        for i in range(max(1, count)):
            ax = self._pick_axis(text, seed + i * 997)
            mutated, salt = self._apply(text, ax, seed + i)
            if mutated != text or i == 0:
                out.append(PromptMutation(ax, text, mutated, salt))
        return out[:count]

    def _pick_axis(self, prompt: str, seed: int) -> MutationAxis:
        h = int(hashlib.md5(f"{prompt}:{seed}".encode()).hexdigest()[:8], 16)
        axes = list(self.axes)
        return axes[h % len(axes)]

    def _apply(self, text: str, axis: MutationAxis, seed: int) -> Tuple[str, int]:
        salt = seed & 0xFFFF
        if axis == MutationAxis.TIME_SHIFT:
            return self._swap_pair(text, _TIME, seed), salt
        if axis == MutationAxis.WEATHER_SWAP:
            return self._swap_pair(text, _WEATHER, seed), salt
        if axis == MutationAxis.MOOD_FLIP:
            return self._swap_pair(text, _MOOD, seed), salt
        if axis == MutationAxis.MEDIUM_SWAP:
            return self._swap_pair(text, _MEDIUM, seed), salt
        if axis == MutationAxis.SCALE_FLIP:
            return self._swap_pair(text, _SCALE, seed), salt
        if axis == MutationAxis.PALETTE_INVERT:
            extra = "inverted color palette, complementary shadow hues" if "inverted" not in text.lower() else text
            return f"{text}, {extra}" if text else extra, salt
        if axis == MutationAxis.DETAIL_ZOOM:
            if re.search(r"\bclose[- ]?up\b", text, re.I):
                return re.sub(r"\bclose[- ]?up\b", "wide environmental context", text, count=1, flags=re.I), salt
            return f"{text}, extreme macro detail on focal texture", salt
        return text, salt

    def _swap_pair(self, text: str, pairs: Sequence[Tuple[str, str]], seed: int) -> str:
        for a, b in pairs:
            if re.search(rf"\b{re.escape(a)}\b", text, re.I):
                return re.sub(rf"\b{re.escape(a)}\b", b, text, count=1, flags=re.I)
            if re.search(rf"\b{re.escape(b)}\b", text, re.I):
                return re.sub(rf"\b{re.escape(b)}\b", a, text, count=1, flags=re.I)
        pick = pairs[seed % len(pairs)]
        return f"{text}, {pick[1]} atmosphere" if text else pick[1]


__all__ = ["MutationAxis", "PromptMutation", "PromptMutator"]
