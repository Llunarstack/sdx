"""Absence Pulse — amplify negative space and withheld detail for horror/minimal beats."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AbsencePulsePlan", "absence_pulse_for_mode"]


@dataclass(frozen=True)
class AbsencePulsePlan:
    mode: str
    positive: str
    negative: str
    cfg_dip: float


def absence_pulse_for_mode(mode: str = "horror") -> AbsencePulsePlan:
    m = (mode or "horror").lower()
    if m == "comedy":
        return AbsencePulsePlan(
            mode=m,
            positive="awkward empty space, held pause, minimal movement",
            negative="busy cluttered action",
            cfg_dip=0.92,
        )
    if m == "minimal":
        return AbsencePulsePlan(
            mode=m,
            positive="vast negative space, subject isolated, withheld detail",
            negative="information overload, busy frame",
            cfg_dip=0.88,
        )
    return AbsencePulsePlan(
        mode="horror",
        positive="oppressive negative space, unseen threat implied, withheld reveal",
        negative="fully visible monster, bright flat lighting",
        cfg_dip=0.85,
    )
