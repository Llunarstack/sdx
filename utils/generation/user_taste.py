"""User taste quiz + personal negative bank for stills generation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "UserTaste",
    "load_user_taste",
    "save_user_taste",
    "build_taste_quiz",
    "apply_taste_quiz_answers",
    "apply_taste_to_prompts",
    "default_taste_path",
]


def default_taste_path() -> Path:
    return Path("outputs") / "user_taste.json"


@dataclass
class UserTaste:
    """Persistent preferences across perfect-gen / sample sessions."""

    favorite_artists: list[str] = field(default_factory=list)
    style_axis: str = ""  # photoreal | anime | painterly | 3d | mixed
    lighting_bias: str = ""  # dark | bright | neon | soft | dramatic
    default_pose: str = ""
    default_place: str = ""
    aspect_intent: str = ""  # phone | poster | square | desktop | portrait
    negative_bank: list[str] = field(default_factory=list)
    hates: list[str] = field(default_factory=list)  # free-text hates → negatives
    likes_notes: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserTaste:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})


_QUIZ: list[dict[str, Any]] = [
    {
        "id": "style_axis",
        "prompt": "Realism vs stylized — what do you usually want?",
        "choices": ["photoreal", "anime", "painterly", "3d", "mixed"],
    },
    {
        "id": "lighting_bias",
        "prompt": "Default lighting mood?",
        "choices": ["soft", "bright", "dark", "neon", "dramatic"],
    },
    {
        "id": "favorite_artists",
        "prompt": "Favorite artists/styles to lean on? (comma-separated, or 'none')",
        "choices": [],
    },
    {
        "id": "hates",
        "prompt": "What do you hate in AI art? (comma-separated → personal negative bank)",
        "choices": [],
    },
    {
        "id": "aspect_intent",
        "prompt": "Typical output shape?",
        "choices": ["phone", "poster", "square", "desktop", "portrait"],
    },
    {
        "id": "default_pose",
        "prompt": "Default pose when you don't specify one?",
        "choices": [
            "standing, looking at viewer, full body",
            "sitting, relaxed",
            "portrait upper body",
            "dynamic action",
        ],
    },
]


def build_taste_quiz() -> list[dict[str, Any]]:
    return [dict(q) for q in _QUIZ]


def apply_taste_quiz_answers(taste: UserTaste, answers: dict[str, str]) -> UserTaste:
    if "style_axis" in answers:
        taste.style_axis = str(answers["style_axis"]).strip()
    if "lighting_bias" in answers:
        taste.lighting_bias = str(answers["lighting_bias"]).strip()
    if "aspect_intent" in answers:
        taste.aspect_intent = str(answers["aspect_intent"]).strip()
    if "default_pose" in answers:
        taste.default_pose = str(answers["default_pose"]).strip()
    if "default_place" in answers:
        taste.default_place = str(answers["default_place"]).strip()
    if "favorite_artists" in answers:
        raw = str(answers["favorite_artists"]).strip()
        if raw.lower() not in ("none", "n/a", ""):
            taste.favorite_artists = [a.strip() for a in raw.split(",") if a.strip()]
    if "hates" in answers:
        raw = str(answers["hates"]).strip()
        hates = [h.strip() for h in raw.split(",") if h.strip()]
        taste.hates = hates
        # Expand into negative bank tokens
        bank = list(taste.negative_bank)
        for h in hates:
            if h.lower() not in {b.lower() for b in bank}:
                bank.append(h)
        # Common AI-hate expansions
        expansions = {
            "plastic": "plastic skin, waxy skin, oversmoothed",
            "hands": "bad hands, extra fingers, fused fingers",
            "text": "watermark, signature, text artifacts",
            "blurry": "blurry, low detail, soft focus mush",
        }
        blob = " ".join(hates).lower()
        for key, extra in expansions.items():
            if key in blob:
                for tok in extra.split(","):
                    t = tok.strip()
                    if t and t.lower() not in {b.lower() for b in bank}:
                        bank.append(t)
        taste.negative_bank = bank
    taste.updated_at = time.time()
    return taste


def apply_taste_to_prompts(
    positive: str,
    negative: str = "",
    taste: UserTaste | None = None,
) -> tuple[str, str]:
    """Merge taste defaults into pos/neg prompts."""
    if taste is None:
        return positive, negative
    pos = str(positive or "").rstrip(",")
    neg = str(negative or "").rstrip(",")
    bits: list[str] = []
    if taste.style_axis and taste.style_axis not in pos.lower():
        axis_map = {
            "photoreal": "photorealistic, natural skin texture",
            "anime": "anime illustration, clean lineart",
            "painterly": "painterly brushwork, traditional media feel",
            "3d": "3d render, subsurface scattering",
            "mixed": "",
        }
        add = axis_map.get(taste.style_axis.lower(), taste.style_axis)
        if add:
            bits.append(add)
    if taste.lighting_bias:
        light_map = {
            "soft": "soft diffused lighting",
            "bright": "bright airy lighting",
            "dark": "moody low-key lighting",
            "neon": "neon rim lighting, cyberpunk glow",
            "dramatic": "dramatic chiaroscuro lighting",
        }
        bits.append(light_map.get(taste.lighting_bias.lower(), taste.lighting_bias))
    for artist in taste.favorite_artists[:3]:
        token = f"{artist} style"
        if token.lower() not in pos.lower():
            bits.append(token)
    for b in bits:
        if b.lower() not in pos.lower():
            pos = f"{pos}, {b}" if pos else b
    # Negatives
    bank = list(taste.negative_bank) + list(taste.hates)
    for tok in bank:
        if tok and tok.lower() not in neg.lower():
            neg = f"{neg}, {tok}" if neg else tok
    return pos.strip(", "), neg.strip(", ")


def load_user_taste(path: str | Path | None = None) -> UserTaste:
    p = Path(path) if path else default_taste_path()
    if not p.is_file():
        return UserTaste()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return UserTaste.from_dict(data)
    except Exception:
        pass
    return UserTaste()


def save_user_taste(taste: UserTaste, path: str | Path | None = None) -> Path:
    p = Path(path) if path else default_taste_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    taste.updated_at = time.time()
    p.write_text(json.dumps(taste.to_dict(), indent=2), encoding="utf-8")
    return p
