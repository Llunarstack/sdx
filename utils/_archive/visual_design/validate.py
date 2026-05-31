"""Registry integrity checks (used by tests and optional CI)."""

from __future__ import annotations

from typing import List

from .registry import DOMAIN_NEGATIVES, DOMAIN_POSITIVES

_TIERS = ("lite", "standard", "strong")


def validate_visual_design_registry() -> List[str]:
    """
    Return a list of issues (empty = OK).
    Ensures every domain has all tiers in positives and negatives.
    """
    issues: List[str] = []

    pos_domains = set(DOMAIN_POSITIVES.keys())
    neg_domains = set(DOMAIN_NEGATIVES.keys())
    if pos_domains != neg_domains:
        only_pos = sorted(pos_domains - neg_domains)
        only_neg = sorted(neg_domains - pos_domains)
        if only_pos:
            issues.append(f"domains only in positives: {only_pos}")
        if only_neg:
            issues.append(f"domains only in negatives: {only_neg}")

    for d in sorted(pos_domains | neg_domains):
        for tier_map, label in (
            (DOMAIN_POSITIVES.get(d, {}), "positives"),
            (DOMAIN_NEGATIVES.get(d, {}), "negatives"),
        ):
            if not tier_map:
                issues.append(f"domain {d!r} missing {label} map")
                continue
            for t in _TIERS:
                v = tier_map.get(t, "").strip()
                if not v:
                    issues.append(f"domain {d!r} missing {label} tier {t!r}")

    return issues


def assert_visual_design_registry_valid() -> None:
    errs = validate_visual_design_registry()
    if errs:
        raise AssertionError("visual_design registry invalid:\n" + "\n".join(errs))


__all__ = ["validate_visual_design_registry", "assert_visual_design_registry_valid"]
