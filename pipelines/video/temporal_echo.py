"""Temporal Echo — shots visually rhyme with earlier beats (motifs, palette, camera)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

__all__ = [
    "EchoLink",
    "TemporalEchoPlan",
    "parse_echo_config",
    "plan_temporal_echoes",
]

import re


@dataclass(slots=True)
class EchoLink:
    shot_id: str
    echoes_shot_id: str
    echo_strength: float
    prompt_suffix: str


@dataclass(slots=True)
class TemporalEchoPlan:
    links: List[EchoLink] = field(default_factory=list)


def parse_echo_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, Mapping):
        return {
            "enabled": bool(raw.get("enabled", True)),
            "auto_rhyme": bool(raw.get("auto_rhyme", True)),
            "strength": float(raw.get("strength") or 0.45),
            "manual": list(raw.get("links") or raw.get("echoes") or []),
        }
    return {"enabled": bool(raw)}


def _visual_tokens(prompt: str) -> set[str]:
    stop = {"the", "a", "an", "and", "with", "in", "on", "at", "to", "of", "shot", "camera"}
    words = re.findall(r"[a-z]{4,}", (prompt or "").lower())
    return {w for w in words if w not in stop}


def plan_temporal_echoes(
    shots: Sequence[Any],
    config: Mapping[str, Any],
) -> TemporalEchoPlan:
    if not config.get("enabled"):
        return TemporalEchoPlan()
    strength = float(config.get("strength") or 0.45)
    links: List[EchoLink] = []

    manual = config.get("manual") or []
    for row in manual:
        if not isinstance(row, Mapping):
            continue
        links.append(
            EchoLink(
                shot_id=str(row.get("shot") or row.get("to") or ""),
                echoes_shot_id=str(row.get("echoes") or row.get("from") or ""),
                echo_strength=float(row.get("strength") or strength),
                prompt_suffix=str(row.get("prompt") or "visual rhyme echoing earlier beat"),
            )
        )

    if config.get("auto_rhyme"):
        tokens_per_shot = [(str(getattr(s, "id", "")), _visual_tokens(str(getattr(s, "prompt", "")))) for s in shots]
        for i, (sid, tok) in enumerate(tokens_per_shot):
            if i < 2:
                continue
            best_j = -1
            best_overlap = 0
            for j in range(i - 1):
                overlap = len(tok & tokens_per_shot[j][1])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_j = j
            if best_j >= 0 and best_overlap >= 3:
                ref_id = tokens_per_shot[best_j][0]
                shared = tok & tokens_per_shot[best_j][1]
                frag = f"visual echo rhyming with {ref_id}: {', '.join(sorted(shared)[:4])}"
                links.append(
                    EchoLink(
                        shot_id=sid,
                        echoes_shot_id=ref_id,
                        echo_strength=strength,
                        prompt_suffix=frag,
                    )
                )
    return TemporalEchoPlan(links=links)
