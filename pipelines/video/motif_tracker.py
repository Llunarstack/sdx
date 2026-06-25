"""Motif Haunting — recurring visual symbols must appear until narratively resolved."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

__all__ = [
    "VisualMotif",
    "MotifHauntReport",
    "parse_motifs",
    "audit_motif_haunting",
    "motif_prompt_injections",
]

import re


@dataclass(slots=True)
class VisualMotif:
    id: str
    description: str
    must_appear_by_shot: int = -1  # index; -1 = optional
    resolved_after_shot: int = -1
    haunt_until_resolved: bool = True
    weight: float = 1.0


@dataclass(slots=True)
class MotifHauntReport:
    ok: bool
    unresolved: List[str] = field(default_factory=list)
    appearances: Dict[str, List[str]] = field(default_factory=dict)
    injections: Dict[str, str] = field(default_factory=dict)  # shot_id → prompt frag


def parse_motifs(raw: Any) -> Dict[str, VisualMotif]:
    out: Dict[str, VisualMotif] = {}
    if isinstance(raw, Mapping):
        for mid, spec in raw.items():
            if isinstance(spec, str):
                out[str(mid)] = VisualMotif(id=str(mid), description=spec)
            elif isinstance(spec, Mapping):
                out[str(mid)] = VisualMotif(
                    id=str(mid),
                    description=str(spec.get("description") or spec.get("prompt") or mid),
                    must_appear_by_shot=int(
                        spec["must_appear_by_shot"]
                        if spec.get("must_appear_by_shot") is not None
                        else (spec.get("by_shot") if spec.get("by_shot") is not None else -1)
                    ),
                    resolved_after_shot=int(
                        spec["resolved_after_shot"]
                        if spec.get("resolved_after_shot") is not None
                        else (spec.get("resolve_after") if spec.get("resolve_after") is not None else -1)
                    ),
                    haunt_until_resolved=bool(spec.get("haunt", spec.get("haunt_until_resolved", True))),
                    weight=float(spec.get("weight") or 1.0),
                )
    return out


def _motif_in_text(motif: VisualMotif, text: str) -> bool:
    t = (text or "").lower()
    desc = motif.description.lower()
    if desc and desc in t:
        return True
    tokens = [x for x in re.split(r"[\s,_-]+", motif.id.lower()) if len(x) > 2]
    return any(tok in t for tok in tokens)


def audit_motif_haunting(motifs: Mapping[str, VisualMotif], shots: Sequence[Any]) -> MotifHauntReport:
    appearances: Dict[str, List[str]] = {m: [] for m in motifs}
    injections: Dict[str, str] = {}
    unresolved: List[str] = []

    for i, sh in enumerate(shots):
        sid = str(getattr(sh, "id", f"shot_{i}"))
        prompt = str(getattr(sh, "prompt", "") or "")
        for mid, motif in motifs.items():
            if _motif_in_text(motif, prompt):
                appearances[mid].append(sid)
            elif motif.haunt_until_resolved:
                resolved_at = motif.resolved_after_shot
                if resolved_at < 0 or i <= resolved_at:
                    # Motif should haunt background until seen or resolved
                    if not appearances[mid] and (motif.must_appear_by_shot < 0 or i >= motif.must_appear_by_shot):
                        injections[sid] = _merge_frag(
                            injections.get(sid, ""),
                            f"subtle background presence of {motif.description}",
                        )
                    elif appearances[mid] and (resolved_at < 0 or i < resolved_at):
                        injections[sid] = _merge_frag(
                            injections.get(sid, ""),
                            f"recurring motif {motif.description} in frame",
                        )

    for mid, motif in motifs.items():
        if motif.must_appear_by_shot >= 0 and not appearances[mid]:
            # Check if any shot up to index saw it
            seen_in_time = False
            for i, sh in enumerate(shots):
                if i > motif.must_appear_by_shot:
                    break
                if _motif_in_text(motif, str(getattr(sh, "prompt", ""))):
                    seen_in_time = True
                    break
            if not seen_in_time:
                unresolved.append(mid)

    ok = len(unresolved) == 0
    return MotifHauntReport(ok=ok, unresolved=unresolved, appearances=appearances, injections=injections)


def _merge_frag(a: str, b: str) -> str:
    if not a:
        return b
    if b.lower() in a.lower():
        return a
    return f"{a}, {b}"


def motif_prompt_injections(report: MotifHauntReport) -> Dict[str, str]:
    return dict(report.injections)
