"""Narrative Debt — plot threads opened must pay off by mandated shot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Set

__all__ = [
    "PlotThread",
    "NarrativeDebtReport",
    "parse_narrative_threads",
    "audit_narrative_debt",
]

import re


@dataclass(slots=True)
class PlotThread:
    id: str
    description: str
    opened_shot: str = ""
    payoff_by_shot: int = -1
    resolved: bool = False


@dataclass(slots=True)
class NarrativeDebtReport:
    ok: bool
    open_threads: List[str] = field(default_factory=list)
    resolved_threads: List[str] = field(default_factory=list)
    unpaid: List[str] = field(default_factory=list)
    injections: Dict[str, str] = field(default_factory=dict)


def parse_narrative_threads(raw: Any) -> Dict[str, PlotThread]:
    out: Dict[str, PlotThread] = {}
    if isinstance(raw, Mapping):
        for tid, spec in raw.items():
            if isinstance(spec, str):
                out[str(tid)] = PlotThread(id=str(tid), description=spec)
            elif isinstance(spec, Mapping):
                payoff = spec.get("payoff_by_shot")
                out[str(tid)] = PlotThread(
                    id=str(tid),
                    description=str(spec.get("description") or spec.get("prompt") or tid),
                    opened_shot=str(spec.get("opened_shot") or spec.get("open") or ""),
                    payoff_by_shot=int(payoff if payoff is not None else -1),
                )
    return out


def _thread_in_text(thread: PlotThread, text: str) -> bool:
    t = (text or "").lower()
    if thread.description.lower() in t:
        return True
    tokens = [x for x in re.split(r"[\s,_-]+", thread.id.lower()) if len(x) > 2]
    return any(tok in t for tok in tokens)


def audit_narrative_debt(
    threads: Mapping[str, PlotThread],
    shots: Sequence[Any],
) -> NarrativeDebtReport:
    if not threads:
        return NarrativeDebtReport(ok=True)
    open_set: Set[str] = set()
    resolved: List[str] = []
    unpaid: List[str] = []
    injections: Dict[str, str] = {}
    opened_at: Dict[str, int] = {}

    for i, sh in enumerate(shots):
        sid = str(getattr(sh, "id", f"shot_{i}"))
        prompt = str(getattr(sh, "prompt", ""))
        shot_threads = set(getattr(sh, "threads", []) or getattr(sh, "narrative_threads", []) or [])
        for tid, thread in threads.items():
            if tid in shot_threads or _thread_in_text(thread, prompt):
                if tid not in open_set:
                    open_set.add(tid)
                    opened_at[tid] = i
                if "resolve" in prompt.lower() or "reveal" in prompt.lower() or "payoff" in prompt.lower():
                    if tid in open_set:
                        open_set.discard(tid)
                        resolved.append(tid)
        for tid in list(open_set):
            thread = threads[tid]
            deadline = thread.payoff_by_shot
            if deadline >= 0 and i >= deadline and tid not in resolved:
                unpaid.append(tid)
            elif tid in open_set:
                injections[sid] = f"{injections.get(sid, '')}, subtle reminder of {thread.description}".strip(", ")

    ok = len(unpaid) == 0
    return NarrativeDebtReport(
        ok=ok,
        open_threads=list(open_set),
        resolved_threads=resolved,
        unpaid=unpaid,
        injections=injections,
    )
