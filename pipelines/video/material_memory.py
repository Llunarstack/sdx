"""Material Memory — wet, mud, blood, soot persist on entities until explicitly cleaned."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "MaterialState",
    "MaterialIssue",
    "MaterialMemoryReport",
    "parse_material_config",
    "track_material_memory",
]

_STATES = ("dry", "wet", "muddy", "bloody", "sooty", "torn", "burned", "frozen")
_IRREVERSIBLE = {"bloody", "torn", "burned"}


@dataclass(slots=True)
class MaterialState:
    entity_id: str
    state: str


@dataclass(slots=True)
class MaterialIssue:
    level: str
    code: str
    message: str
    shot_id: str


@dataclass(slots=True)
class MaterialMemoryReport:
    timeline: List[Dict[str, str]]
    issues: List[MaterialIssue]
    prompt_injections: Dict[str, str]


def parse_material_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, Mapping):
        return {"enabled": bool(raw.get("enabled", True)), "strict": bool(raw.get("strict", False))}
    return {"enabled": bool(raw)}


def _entity_materials(shot: Any) -> Dict[str, str]:
    raw = getattr(shot, "material_state", None) or getattr(shot, "materials", None) or {}
    if isinstance(raw, Mapping):
        return {str(k): str(v).lower() for k, v in raw.items()}
    return {}


_STATE_PROMPTS: Dict[str, str] = {
    "wet": "clothes and hair still wet, water dripping",
    "muddy": "mud stains on boots and hem, dirty splatter",
    "bloody": "blood stains persist on fabric and skin",
    "sooty": "soot smudges on face and clothing",
    "torn": "torn fabric, visible rips unchanged",
    "burned": "char marks and scorch patterns remain",
    "frozen": "frost on lashes and coat edges",
    "dry": "clean dry clothing",
}


def track_material_memory(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
    initial: Optional[Mapping[str, str]] = None,
) -> MaterialMemoryReport:
    if not config.get("enabled"):
        return MaterialMemoryReport(timeline=[], issues=[], prompt_injections={})
    strict = bool(config.get("strict"))
    level = "error" if strict else "warn"
    ledger: Dict[str, str] = {str(k): str(v).lower() for k, v in (initial or {}).items()}
    timeline: List[Dict[str, str]] = []
    issues: List[MaterialIssue] = []
    injections: Dict[str, str] = {}

    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        mats = _entity_materials(sh)
        snap = dict(ledger)
        for eid, st in mats.items():
            prev = ledger.get(eid, "dry")
            if prev in _IRREVERSIBLE and st == "dry":
                issues.append(
                    MaterialIssue(
                        level=level,
                        code="material_reset",
                        message=f"{eid} cannot go {prev}→dry without cleaning scene",
                        shot_id=sid,
                    )
                )
            ledger[eid] = st
        for eid, st in ledger.items():
            if st not in ("dry", "") and eid not in mats:
                frag = _STATE_PROMPTS.get(st, f"{st} material state persists")
                injections[sid] = f"{injections.get(sid, '')}, {eid} {frag}".strip(", ")
        timeline.append({"shot_id": sid, **snap, **{f"mat_{k}": v for k, v in ledger.items()}})
    return MaterialMemoryReport(timeline=timeline, issues=issues, prompt_injections=injections)
