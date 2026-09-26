from __future__ import annotations

from typing import Final

from ..intents.goal_spec import GoalSpec

# (needles checked as substrings in lowered narrative, first match wins)
_DOMAIN_RULES: Final[tuple[tuple[tuple[str, ...], tuple[str, str]], ...]] = (
    (("logo", "wordmark", "brand"), ("visual-design-domain", "brand")),
    (("diagram", "graph", "equation", "formula"), ("visual-design-domain", "stem")),
    (("ui", "dashboard", "app screen", "wireframe"), ("visual-design-domain", "ui_ux")),
    (("textbook", "didactic"), ("visual-design-domain", "textbook")),
)


def goal_spec_to_sample_hints(spec: GoalSpec) -> dict[str, list[str]]:
    """
    Map high-level GoalSpec fields to CLI flag fragments (conceptual hints).

    Returns lists of argv tokens *without* the leading ``--``.
    Caller still chooses whether to subprocess ``sample.py`` or build argparse directly.
    """
    nar = spec.narrative.lower()
    for needles, pair in _DOMAIN_RULES:
        if any(n in nar for n in needles):
            return {"extras": [pair[0], pair[1]]}
    return {"extras": []}


__all__ = ["goal_spec_to_sample_hints"]
