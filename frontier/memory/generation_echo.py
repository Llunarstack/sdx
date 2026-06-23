"""
Cross-run memory of generation failures — not full RL, just "don't step on the same rake."

Stores artifact tags from rejected outputs and merges them into negative prompts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence


@dataclass
class ArtifactEcho:
    prompt_fingerprint: str
    tags: List[str]
    severity: float


# Common AI failure modes to recognize from short tag lists
_DEFAULT_ARTIFACT_TAGS = (
    "extra fingers",
    "melted hands",
    "duplicate face",
    "warped text",
    "asymmetric eyes",
    "plastic skin",
    "jpeg artifacts",
    "floating objects",
)


class GenerationEchoMemory:
    """
    Ring buffer of recent failure echoes keyed by coarse prompt fingerprint.

    Fingerprint = first N lowercase words (cheap; swap for hash embedding later).
    """

    def __init__(self, capacity: int = 64, fingerprint_words: int = 6) -> None:
        self.capacity = max(4, int(capacity))
        self.fingerprint_words = max(2, int(fingerprint_words))
        self._echoes: Deque[ArtifactEcho] = deque(maxlen=self.capacity)
        self._by_fp: Dict[str, List[str]] = {}

    @staticmethod
    def fingerprint(prompt: str, *, words: int = 6) -> str:
        parts = (prompt or "").lower().split()
        return " ".join(parts[:words]) if parts else ""

    def record_failure(
        self,
        prompt: str,
        tags: Sequence[str],
        *,
        severity: float = 0.8,
    ) -> ArtifactEcho:
        fp = self.fingerprint(prompt, words=self.fingerprint_words)
        clean_tags = [t.strip().lower() for t in tags if t.strip()]
        echo = ArtifactEcho(prompt_fingerprint=fp, tags=clean_tags, severity=float(severity))
        self._echoes.append(echo)
        bucket = self._by_fp.setdefault(fp, [])
        for t in clean_tags:
            if t not in bucket:
                bucket.append(t)
        return echo

    def tags_for_prompt(self, prompt: str) -> List[str]:
        fp = self.fingerprint(prompt, words=self.fingerprint_words)
        return list(self._by_fp.get(fp, []))

    def negative_suffix(self, prompt: str, *, min_severity: float = 0.5) -> str:
        tags: List[str] = []
        fp = self.fingerprint(prompt, words=self.fingerprint_words)
        for echo in self._echoes:
            if echo.severity < min_severity:
                continue
            if echo.prompt_fingerprint == fp or self._overlap(fp, echo.prompt_fingerprint):
                tags.extend(echo.tags)
        # always include baseline artifact guard at low weight
        tags.extend(_DEFAULT_ARTIFACT_TAGS[:3])
        seen: set[str] = set()
        out: List[str] = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return ", ".join(out)

    @staticmethod
    def _overlap(a: str, b: str) -> bool:
        wa, wb = set(a.split()), set(b.split())
        if not wa or not wb:
            return False
        return len(wa & wb) / min(len(wa), len(wb)) >= 0.5

    def clear(self) -> None:
        self._echoes.clear()
        self._by_fp.clear()
