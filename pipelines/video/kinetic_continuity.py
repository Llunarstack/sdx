"""Kinetic Ledger — cross-shot velocity and energy state must conserve narrative physics."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "KineticState",
    "KineticIssue",
    "KineticLedger",
    "parse_kinetic_config",
    "track_kinetic_ledger",
]

_ENERGY_VERBS: Dict[str, float] = {
    "sprint": 0.95,
    "run": 0.85,
    "runs": 0.85,
    "chase": 0.9,
    "jump": 0.8,
    "leap": 0.85,
    "explode": 1.0,
    "collapse": 0.1,
    "walk": 0.35,
    "walks": 0.35,
    "stand": 0.05,
    "stands": 0.05,
    "sit": 0.02,
    "idle": 0.0,
    "sleep": 0.0,
    "land": 0.4,
    "lands": 0.4,
    "fall": 0.7,
    "falls": 0.7,
    "stop": 0.05,
    "stops": 0.05,
    "brake": 0.15,
}


@dataclass(slots=True)
class KineticState:
    energy: float  # 0..1
    vertical: float  # -1 falling, 0 grounded, 1 airborne
    verb: str = ""


@dataclass(slots=True)
class KineticIssue:
    level: str
    code: str
    message: str
    shot_id: str
    related_shot_id: str = ""


@dataclass(slots=True)
class KineticLedger:
    states: List[KineticState] = field(default_factory=list)
    issues: List[KineticIssue] = field(default_factory=list)
    shot_prompt_patches: Dict[str, str] = field(default_factory=dict)


def parse_kinetic_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, Mapping):
        return {
            "enabled": bool(raw.get("enabled", True)),
            "strict": bool(raw.get("strict", False)),
            "max_energy_jump": float(raw.get("max_energy_jump") or 0.55),
        }
    return {"enabled": bool(raw)}


def _infer_kinetic(prompt: str, explicit: Optional[Mapping[str, Any]] = None) -> KineticState:
    if isinstance(explicit, Mapping):
        return KineticState(
            energy=float(explicit.get("energy") or 0.5),
            vertical=float(explicit.get("vertical") or 0.0),
            verb=str(explicit.get("verb") or ""),
        )
    p = (prompt or "").lower()
    energy = 0.3
    vertical = 0.0
    verb = ""
    for v, e in _ENERGY_VERBS.items():
        if re.search(rf"\b{re.escape(v)}\b", p):
            energy = e
            verb = v
            if v in ("jump", "leap", "falls", "fall"):
                vertical = 1.0 if v in ("jump", "leap") else -1.0
            elif v in ("land", "lands"):
                vertical = 0.0
            break
    if "airborne" in p or "mid-air" in p:
        vertical = 1.0
    if "exhausted" in p or "gasping" in p:
        energy = min(energy, 0.25)
    return KineticState(energy=energy, vertical=vertical, verb=verb)


def track_kinetic_ledger(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> KineticLedger:
    if not config.get("enabled"):
        return KineticLedger()
    max_jump = float(config.get("max_energy_jump") or 0.55)
    strict = bool(config.get("strict"))
    level = "error" if strict else "warn"

    states: List[KineticState] = []
    issues: List[KineticIssue] = []
    patches: Dict[str, str] = {}

    prev: Optional[KineticState] = None
    prev_id = ""
    for i, sh in enumerate(shots):
        sid = str(getattr(sh, "id", f"shot_{i}"))
        kin_raw = getattr(sh, "kinetic", None)
        st = _infer_kinetic(str(getattr(sh, "prompt", "")), kin_raw if isinstance(kin_raw, Mapping) else None)
        states.append(st)

        if prev is not None:
            de = abs(st.energy - prev.energy)
            if de > max_jump and not _narrative_reset(str(getattr(sh, "prompt", ""))):
                issues.append(
                    KineticIssue(
                        level=level,
                        code="energy_discontinuity",
                        message=f"Energy jump {prev.energy:.2f}→{st.energy:.2f} across cut (max {max_jump})",
                        shot_id=prev_id,
                        related_shot_id=sid,
                    )
                )
                patches[sid] = _patch_for_energy(prev.energy, st.energy)
            if prev.vertical > 0.5 and st.vertical <= 0 and st.verb not in ("land", "lands", "fall", "falls"):
                issues.append(
                    KineticIssue(
                        level="info",
                        code="missing_landing_beat",
                        message=f"Airborne prior shot — consider landing beat before {sid}",
                        shot_id=sid,
                    )
                )
                patches[sid] = (patches.get(sid, "") + ", grounded landing recovery").strip(", ")
        prev = st
        prev_id = sid

    return KineticLedger(states=states, issues=issues, shot_prompt_patches=patches)


def _narrative_reset(prompt: str) -> bool:
    p = prompt.lower()
    return any(x in p for x in ("later", "meanwhile", "hours later", "new scene", "cut to", "title card"))


def _patch_for_energy(prev_e: float, curr_e: float) -> str:
    if curr_e > prev_e:
        return "acceleration from prior momentum, building speed naturally"
    return "deceleration from prior momentum, slowing naturally"
