"""Scar Timeline — injuries and damage accumulate on character bibles across shots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

__all__ = [
    "ScarRecord",
    "ScarTimelineReport",
    "parse_scar_config",
    "track_scar_timeline",
]


@dataclass(slots=True)
class ScarRecord:
    character_id: str
    injuries: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ScarTimelineReport:
    by_shot: Dict[str, Dict[str, List[str]]]
    prompt_injections: Dict[str, str]


_INJURY_WORDS = ("bruised", "bleeding", "cut", "scarred", "bandaged", "limping", "burnt", "wounded")


def parse_scar_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, Mapping):
        return {"enabled": bool(raw.get("enabled", True))}
    return {"enabled": bool(raw)}


def _injuries_from_shot(shot: Any) -> Dict[str, List[str]]:
    raw = getattr(shot, "injuries", None) or getattr(shot, "scars", None) or {}
    out: Dict[str, List[str]] = {}
    if isinstance(raw, Mapping):
        for cid, inj in raw.items():
            if isinstance(inj, list):
                out[str(cid)] = [str(x) for x in inj]
            else:
                out[str(cid)] = [str(inj)]
    prompt = str(getattr(shot, "prompt", "")).lower()
    for w in _INJURY_WORDS:
        if w in prompt:
            for cid in getattr(shot, "characters", []) or []:
                out.setdefault(str(cid), []).append(w)
    return out


def track_scar_timeline(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> ScarTimelineReport:
    if not config.get("enabled"):
        return ScarTimelineReport(by_shot={}, prompt_injections={})
    ledger: Dict[str, List[str]] = {}
    by_shot: Dict[str, Dict[str, List[str]]] = {}
    injections: Dict[str, str] = {}

    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        new_inj = _injuries_from_shot(sh)
        for cid, inj_list in new_inj.items():
            for inj in inj_list:
                if inj not in ledger.get(cid, []):
                    ledger.setdefault(cid, []).append(inj)
        snap = {cid: list(injs) for cid, injs in ledger.items()}
        by_shot[sid] = snap
        if ledger:
            parts = [f"{cid}: {', '.join(injs)}" for cid, injs in ledger.items() if injs]
            injections[sid] = f"visible accumulated injuries — {'; '.join(parts)}"
    return ScarTimelineReport(by_shot=by_shot, prompt_injections=injections)
