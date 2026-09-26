"""
Best-model **quality policy** — compete mode (max aggression).

Every profile soft-fills the strongest practical stack so SDX outruns closed
APIs on *control surface*: anti-AI look, guidance, anatomy, text repair,
multi-candidate pick, and edit loops. Escape: ``--no-quality-defaults``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ProfileRecipe:
    """Max-strength soft defaults for one prompt profile."""

    name: str
    # Sampler / look
    sampler_preset: str | None = "superior"
    op_mode: str | None = None
    holy_grail: bool = True
    holy_grail_preset: str = "aggressive"
    cfg_scale: float | None = None
    cfg_rescale: float | None = 0.7
    apg_parallel_eta: float | None = 0.0
    apg_momentum_beta: float | None = 0.2
    fdg_cfg_strength: float | None = 0.5
    zeresfdg_strength: float | None = 0.85
    qsilk_micrograin: float | None = 0.12
    steps: int | None = 32
    # Anti-AI / human look
    human_made: str = "strong"
    less_ai: bool = True
    naturalize: bool = True
    naturalize_deep: bool = True
    anti_ai_pack: str = "strong"
    human_media: str | None = "dslr"
    shortcomings_mitigation: str = "all"
    anatomy_guidance: str = "strong"
    diversity: bool = True
    anti_bleed: bool = True
    anti_artifacts: bool = True
    strong_watermark: bool = True
    # Mid-network residuals
    naturalness_strength: float = 0.45
    anatomy_attention_strength: float = 0.4
    glyph_residual_strength: float = 0.14
    # Prompt / pick / repair
    expand_prompt: bool = True
    anti_slop: bool = True
    photo_realism_prefer: str | None = "documentary"
    hand_mode: str | None = "detailed"
    pose_naturalness: str | None = "natural"
    typography_mode: str | None = None
    text_in_image: bool = False
    ocr_fix: bool = False
    # Test-time selection
    num: int | None = 3
    pick_best: str | None = "superior_composite"
    # Frontier / self-correct
    frontier_subject: bool = False
    superior_self_correct: bool = True
    adherence_negation_scale: float = 0.55
    adherence_binding_boost: float = 0.15
    # Guidance interval (skip early CFG — research default from GuidanceInterval)
    cfg_skip_early_frac: float | None = 0.08
    cfg_skip_late_frac: float | None = 0.0
    # Soft token emphasis for hands/text/counts
    token_emphasis: bool = True


# ``--pick-best`` choices that currently exist (sample CLI / test_time_pick).
_KNOWN_PICK_BEST = frozenset(
    {
        "auto",
        "none",
        "clip",
        "edge",
        "ocr",
        "vit",
        "aesthetic",
        "combo",
        "combo_vit",
        "combo_vit_hq",
        "combo_vit_realism",
        "combo_count_vit",
        "combo_exposure",
        "combo_structural",
        "combo_architecture",
        "combo_hq",
        "combo_count",
        "combo_realism",
        "combo_contact",
        "combo_hand",
        "combo_occlusion",
        "combo_reflection",
        "combo_spatial_bind",
        "combo_human",
        "aesthetic_realism",
        "superior_composite",
    }
)


def _pick_best_or_realism(preferred: str) -> str:
    """Use ``preferred`` when the pick-best metric exists, else combo_vit_realism."""
    return preferred if preferred in _KNOWN_PICK_BEST else "combo_vit_realism"


# Compete-mode profiles — first match wins.
PROFILES: dict[str, ProfileRecipe] = {
    "text_render": ProfileRecipe(
        name="text_render",
        sampler_preset="sdxl",
        holy_grail_preset="balanced",
        cfg_scale=6.5,
        steps=36,
        human_made="standard",
        naturalize=False,
        naturalize_deep=False,
        human_media=None,
        anatomy_guidance="none",
        anatomy_attention_strength=0.0,
        naturalness_strength=0.15,
        glyph_residual_strength=0.22,
        typography_mode="poster",
        text_in_image=True,
        ocr_fix=True,
        photo_realism_prefer=None,
        hand_mode=None,
        pose_naturalness=None,
        pick_best="ocr",
        num=4,
        frontier_subject=False,
        anti_slop=False,
        fdg_cfg_strength=0.35,
    ),
    "photoreal": ProfileRecipe(
        name="photoreal",
        sampler_preset="flux",
        op_mode="portrait",
        holy_grail_preset="photoreal",
        cfg_scale=3.8,
        steps=32,
        fdg_cfg_strength=0.55,
        human_media="dslr",
        photo_realism_prefer="documentary",
        naturalness_strength=0.48,
        anatomy_attention_strength=0.38,
        hand_mode="detailed",
        pose_naturalness="natural",
        frontier_subject=True,
        pick_best="combo_vit_realism",
        num=3,
    ),
    "portrait": ProfileRecipe(
        name="portrait",
        sampler_preset="flux",
        op_mode="portrait",
        holy_grail_preset="photoreal",
        cfg_scale=3.8,
        steps=34,
        fdg_cfg_strength=0.55,
        human_media="photographic",
        photo_realism_prefer="studio_portrait",
        naturalness_strength=0.5,
        anatomy_attention_strength=0.42,
        hand_mode="detailed",
        pose_naturalness="intimate_natural",
        frontier_subject=True,
        pick_best="combo_vit_realism",
        num=3,
    ),
    "anime": ProfileRecipe(
        name="anime",
        sampler_preset="pixai",
        op_mode="anime_char",
        holy_grail_preset="pixai",
        cfg_scale=5.5,
        cfg_rescale=0.7,
        apg_parallel_eta=None,
        apg_momentum_beta=None,
        fdg_cfg_strength=0.25,
        zeresfdg_strength=0.5,
        steps=28,
        human_made="standard",
        naturalize=False,
        naturalize_deep=False,
        human_media=None,
        anatomy_guidance="strong",
        naturalness_strength=0.28,
        anatomy_attention_strength=0.35,
        glyph_residual_strength=0.08,
        photo_realism_prefer=None,
        hand_mode="stable",
        pose_naturalness="dynamic_natural",
        anti_slop=False,
        pick_best="combo_vit",
        num=3,
        frontier_subject=True,
    ),
    "pixai": ProfileRecipe(
        name="pixai",
        sampler_preset="pixai_pro",
        op_mode="pro",
        holy_grail_preset="pixai",
        cfg_scale=6.0,
        cfg_rescale=0.7,
        steps=36,
        human_made="standard",
        naturalize=False,
        anatomy_guidance="strong",
        naturalness_strength=0.3,
        anatomy_attention_strength=0.38,
        hand_mode="stable",
        pose_naturalness="dynamic_natural",
        pick_best="combo_vit",
        num=3,
        frontier_subject=True,
    ),
    "illustration": ProfileRecipe(
        name="illustration",
        sampler_preset="zit",
        holy_grail_preset="illustration",
        cfg_scale=6.5,
        steps=30,
        human_made="standard",
        naturalize=False,
        naturalize_deep=False,
        human_media=None,
        photo_realism_prefer=None,
        naturalness_strength=0.32,
        anatomy_attention_strength=0.3,
        hand_mode="stable",
        anti_slop=False,
        pick_best="combo_vit",
        num=3,
    ),
    "people": ProfileRecipe(
        name="people",
        sampler_preset="superior",
        op_mode="fullbody",
        holy_grail_preset="aggressive",
        cfg_scale=6.5,
        steps=34,
        fdg_cfg_strength=0.5,
        photo_realism_prefer="cinematic",
        naturalness_strength=0.45,
        anatomy_attention_strength=0.45,
        hand_mode="grip",
        pose_naturalness="dynamic_natural",
        frontier_subject=True,
        pick_best="combo_vit_hq",
        num=4,
    ),
    "occlusion": ProfileRecipe(
        name="occlusion",
        holy_grail_preset="aggressive",
        anatomy_guidance="strong",
        anatomy_attention_strength=0.45,
        pick_best=_pick_best_or_realism("combo_occlusion"),
        num=3,
        frontier_subject=True,
    ),
    "reflection": ProfileRecipe(
        name="reflection",
        holy_grail_preset="aggressive",
        anatomy_guidance="strong",
        pick_best=_pick_best_or_realism("combo_reflection"),
        num=3,
        frontier_subject=True,
    ),
    "sfw": ProfileRecipe(
        name="sfw",
        holy_grail_preset="photoreal",
        cfg_scale=5.5,
        anatomy_guidance="strong",
        anatomy_attention_strength=0.35,
        hand_mode="stable",
        pose_naturalness="natural",
        pick_best="combo_vit_realism",
        num=3,
        frontier_subject=True,
    ),
    "nsfw": ProfileRecipe(
        name="nsfw",
        holy_grail_preset="aggressive",
        # Do not force DSLR / human-made / anti-slop photoreal onto hentai or silicon.
        human_made="none",
        less_ai=False,
        naturalize=False,
        naturalize_deep=False,
        anti_ai_pack="none",
        human_media=None,
        photo_realism_prefer=None,
        anti_slop=False,
        anatomy_guidance="strong",
        anatomy_attention_strength=0.5,
        hand_mode="detailed",
        pose_naturalness="intimate_natural",
        pick_best=_pick_best_or_realism("combo_contact"),
        num=3,
        frontier_subject=True,
    ),
    "bind": ProfileRecipe(
        name="bind",
        holy_grail_preset="aggressive",
        anatomy_guidance="strong",
        pick_best=_pick_best_or_realism("combo_spatial_bind"),
        num=3,
        frontier_subject=True,
    ),
    "default": ProfileRecipe(
        name="default",
        sampler_preset="superior",
        holy_grail_preset="aggressive",
        cfg_scale=6.5,
        steps=32,
        fdg_cfg_strength=0.45,
        photo_realism_prefer=None,
        naturalness_strength=0.4,
        anatomy_attention_strength=0.35,
        hand_mode="stable",
        pose_naturalness="natural",
        pick_best="superior_composite",
        num=3,
        frontier_subject=True,
        anti_slop=True,
    ),
}


@dataclass(slots=True)
class QualityPolicy:
    """Global compete-mode switches."""

    enabled: bool = True
    aggression: str = "max"  # max | balanced (balanced softens num/pick)
    enable_holy_grail: bool = True
    enable_adherence_modulate: bool = True
    enable_glyph: bool = True
    # Multi-candidate pick is on by default in max mode (costs more VRAM/time)
    enable_pick_best: bool = True
    enable_ocr_auto: bool = True
    enable_frontier_subject: bool = True
    enable_token_emphasis: bool = True
    enable_cfg_skip_early: bool = True
    # Still-image competitor gaps (spatial / glyphs / binding / anti-float)
    enable_auto_layout: bool = True
    enable_prompt_ground: bool = True
    enable_glyph_canvas: bool = True
    enable_contact_shadow: bool = True
    enable_design_brief: bool = True
    enable_community_recipes: bool = True
    enable_box_attn: bool = True
    enable_prompt_reinject: bool = True
    edit_gate_default: str = "auto"
    profiles: dict[str, ProfileRecipe] = field(default_factory=lambda: dict(PROFILES))
    train_shortcomings_mitigation: str = "all"
    train_anatomy_guidance: str = "strong"
    # Stronger text-cond modulation than v1
    adherence_negation_scale: float = 0.55
    adherence_binding_boost: float = 0.15


POLICY = QualityPolicy()

_NSFW_PROFILE_RE = re.compile(
    r"\b(nsfw|nude|naked|explicit|erotic|hentai|topless|lingerie|"
    r"penis|vagina|pussy|handjob|blowjob|paizuri|cowgirl|missionary|"
    r"creampie|ahegao|tentacle|uncensored|sexbot|sex\s*bot|"
    r"intercourse|orgasm|lewd|ecchi|bondage|bdsm|sex|fuck(?:ing)?)\b",
    re.I,
)
_MINOR_PROFILE_RE = re.compile(
    r"\b(child|children|kid|kids|toddler|infant|baby|minor|loli|shota)\b",
    re.I,
)


def match_profile_name(prompt: str, *, style: str = "") -> str:
    """First matching profile id for this prompt (see PROFILES)."""
    text = f"{prompt} {style}".lower()
    if any(
        k in text
        for k in (
            "text that says",
            "text says",
            "sign that says",
            "storefront",
            "typograph",
            "lettering",
            "[text:",
        )
    ) or ('"' in (prompt or "") and len(prompt or "") < 400):
        if '"' in (prompt or "") or "“" in (prompt or "") or "[text:" in text:
            return "text_render"
    if not _MINOR_PROFILE_RE.search(text) and _NSFW_PROFILE_RE.search(text):
        return "nsfw"
    if any(
        k in text
        for k in (
            "sfw",
            "safe for work",
            "work-safe",
            "family-friendly",
            "linkedin",
            "passport photo",
            "corporate headshot",
        )
    ):
        return "sfw"
    if any(k in text for k in ("pixai", "tsubaki", "serin", "moonlit garden", "full-body illustration")):
        return "pixai"
    if any(k in text for k in ("anime", "manga", "cel shading", "waifu", "danbooru", "visual novel")):
        return "anime"
    if any(
        k in text for k in ("illustration", "concept art", "painterly", "comic", "storybook", "watercolor", "oil paint")
    ):
        return "illustration"
    # Occlusion / reflection / bind before portrait so fence scenes pick physics.
    if any(
        k in text
        for k in (
            "fence",
            "occlud",
            "chain-link",
            "through the gaps",
            "partly hidden behind",
            "visible through",
            "wine glass",
        )
    ):
        return "occlusion"
    if (
        "reflection" in text
        or "reflected" in text
        or ("chrome" in text and "granite" in text)
        or "puddle" in text
        or "mirror surface" in text
    ):
        return "reflection"
    if "on the left" in text and "on the right" in text:
        return "bind"
    if "left of" in text:
        color_hits = sum(
            1
            for c in (
                "red",
                "blue",
                "green",
                "yellow",
                "orange",
                "purple",
                "pink",
                "white",
                "black",
                "brown",
            )
            if c in text
        )
        if color_hits >= 2:
            return "bind"
    if any(k in text for k in ("portrait", "headshot", "beauty shot", "passport photo")):
        return "portrait"
    if any(
        k in text
        for k in (
            "photo",
            "photoreal",
            "dslr",
            "raw photo",
            "8k photo",
            "cinematic photo",
            "documentary",
            "street photo",
            "hyperreal",
            "realistic",
        )
    ):
        return "photoreal"
    if any(
        k in text
        for k in (
            "person",
            "people",
            "man",
            "woman",
            "girl",
            "boy",
            "hands",
            "full body",
            "character",
        )
    ):
        return "people"
    return "default"


def recipe_for_prompt(prompt: str, *, style: str = "") -> ProfileRecipe:
    name = match_profile_name(prompt, style=style)
    return POLICY.profiles.get(name, POLICY.profiles["default"])


def soft_set(args: Any, name: str, value: Any, *, unset: tuple[Any, ...] | None = None) -> bool:
    """Set getattr(args,name) only when it looks unset. Returns True if applied."""
    if value is None or not hasattr(args, name):
        return False
    cur = getattr(args, name)
    if unset is None:
        if isinstance(value, bool):
            unset = (False,)
        elif isinstance(value, (int, float)):
            unset = (0, 0.0, -1, -1.0, None)
        else:
            unset = (None, "", "none", "off", "0")
    if cur in unset:
        setattr(args, name, value)
        return True
    return False


__all__ = [
    "POLICY",
    "PROFILES",
    "ProfileRecipe",
    "QualityPolicy",
    "match_profile_name",
    "recipe_for_prompt",
    "soft_set",
]
