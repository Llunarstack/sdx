"""Print / aspect intent presets + per-facet ref source policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

__all__ = [
    "PrintIntent",
    "resolve_print_intent",
    "PRINT_INTENTS",
    "RefSourceChoice",
    "RefSourcePolicy",
    "default_ref_policy",
    "apply_ref_policy_to_search",
]

PRINT_INTENTS: dict[str, dict[str, int | str]] = {
    "phone": {"width": 576, "height": 1024, "label": "phone wallpaper"},
    "portrait": {"width": 768, "height": 1024, "label": "portrait"},
    "square": {"width": 1024, "height": 1024, "label": "square social"},
    "desktop": {"width": 1280, "height": 720, "label": "desktop wallpaper"},
    "poster": {"width": 768, "height": 1152, "label": "poster print"},
    "album": {"width": 1024, "height": 1024, "label": "album cover"},
}


@dataclass(slots=True)
class PrintIntent:
    key: str
    width: int
    height: int
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sample_argv(self) -> list[str]:
        return ["--width", str(self.width), "--height", str(self.height)]


def resolve_print_intent(name: str) -> PrintIntent | None:
    key = str(name or "").strip().lower()
    if not key or key not in PRINT_INTENTS:
        return None
    row = PRINT_INTENTS[key]
    return PrintIntent(
        key=key,
        width=int(row["width"]),
        height=int(row["height"]),
        label=str(row.get("label") or key),
    )


SourceMode = Literal["upload", "search", "both", "none"]


@dataclass
class RefSourceChoice:
    facet: str
    mode: SourceMode = "both"
    upload_paths: list[str] = field(default_factory=list)


@dataclass
class RefSourcePolicy:
    """Per-facet: use user uploads, web search, both, or skip."""

    choices: dict[str, RefSourceChoice] = field(default_factory=dict)
    nsfw_web_allowed: bool = False
    web_consent: bool = True

    def mode_for(self, facet: str) -> SourceMode:
        c = self.choices.get(facet)
        return c.mode if c else "both"

    def to_dict(self) -> dict[str, Any]:
        return {
            "nsfw_web_allowed": self.nsfw_web_allowed,
            "web_consent": self.web_consent,
            "choices": {k: asdict(v) for k, v in self.choices.items()},
        }


def default_ref_policy(
    *,
    nsfw: bool = False,
    web_consent: bool = True,
    uploads: dict[str, list[str]] | None = None,
) -> RefSourcePolicy:
    uploads = uploads or {}
    facets = ("subject", "attire", "style", "identity", "pose", "place")
    choices: dict[str, RefSourceChoice] = {}
    for f in facets:
        paths = list(uploads.get(f) or [])
        if paths:
            choices[f] = RefSourceChoice(facet=f, mode="upload", upload_paths=paths)
        else:
            choices[f] = RefSourceChoice(facet=f, mode="search" if web_consent else "none")
    return RefSourcePolicy(
        choices=choices,
        nsfw_web_allowed=bool(nsfw and web_consent),
        web_consent=web_consent,
    )


def apply_ref_policy_to_search(
    policy: RefSourcePolicy,
    *,
    nsfw: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    """
    Returns (allow_web, facets_to_search).
    Blocks web entirely without consent; NSFW needs nsfw_web_allowed.
    """
    if not policy.web_consent:
        return False, ()
    if nsfw and not policy.nsfw_web_allowed:
        return False, ()
    search_facets = []
    for facet, choice in policy.choices.items():
        if choice.mode in ("search", "both") and not choice.upload_paths:
            search_facets.append(facet)
        elif choice.mode == "both":
            search_facets.append(facet)
    # Core scrape facets only
    allowed = tuple(f for f in ("subject", "attire", "style") if f in search_facets)
    return True, allowed
