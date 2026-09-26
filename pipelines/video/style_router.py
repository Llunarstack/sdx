"""Universal style router — realistic / anime / cartoon / VFX / stop-motion / …

One prompt → engine preset + motion grammar + animation principles + VIDEOMAX
repair profile. Prevents live-action rules from crushing cel holds, and anime
timing from making VFX plates look like cartoons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "StyleRoute",
    "STYLE_PROFILES",
    "list_styles",
    "route_style",
    "apply_style_route_to_options",
]


@dataclass
class StyleRoute:
    style: str
    engine: str
    grammar: str
    animation_preset: str
    post_grade: str = ""
    videomax_profile: dict[str, Any] = field(default_factory=dict)
    sample_extras: list[str] = field(default_factory=list)
    positive_addon: str = ""
    negative_addon: str = ""
    principles_prompt: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Per-medium VIDEOMAX repair dials (override after base VIDEOMAX).
STYLE_PROFILES: dict[str, dict[str, Any]] = {
    "realistic": {
        "engine": "realistic",
        "grammar": "realistic",
        "animation_preset": "realistic",
        "post_grade": "cinematic",
        "cues": ("photoreal", "live action", "documentary", "handheld", "real footage"),
        "videomax": {
            "motion_shutter": True,
            "contact_ground": True,
            "physics_gate": True,
            "physics_repair": True,
            "permanence_repair": True,
            "occlusion_resolve": True,
            "min_permanence": 0.60,
            "min_physics": 0.50,
            "min_contact": 0.45,
            "min_identity": 0.60,
            "deflicker_strength": 0.88,
        },
        "sample_extras": ["--photo-postprocess", "--human-made", "standard", "--less-ai", "--naturalize"],
        "invention": "maxwave",
    },
    "film": {
        "engine": "realistic",
        "grammar": "film",
        "animation_preset": "realistic",
        "post_grade": "cinematic",
        "cues": ("feature film", "anamorphic", "imax", "35mm", "cinematic film", "narrative film"),
        "videomax": {
            "motion_shutter": True,
            "contact_ground": True,
            "physics_gate": True,
            "physics_repair": True,
            "camera_stabilize": True,
            "min_permanence": 0.62,
            "min_identity": 0.62,
            "deflicker_strength": 0.90,
        },
        "sample_extras": ["--photo-postprocess", "--human-made", "standard"],
        "invention": "maxwave",
    },
    "anime_2d": {
        "engine": "anime_2d",
        "grammar": "anime_2d",
        "animation_preset": "anime_tv",
        "post_grade": "vibrant",
        "cues": ("anime", "manga", "cel shaded", "ghibli", "sakuga", "waifu", "key animation"),
        "videomax": {
            "motion_shutter": False,
            "contact_ground": False,  # floats / holds OK
            "physics_gate": False,
            "physics_repair": False,
            "permanence_repair": False,  # intentional holds
            "count_bind_repair": True,
            "identity_bind": True,
            "extremity_lock": True,
            "hf_deshimmer": False,  # keep line chatter
            "deflicker_strength": 0.50,
            "min_permanence": 0.35,
            "min_physics": 0.0,
            "min_contact": 0.0,
            "min_identity": 0.52,
            "temporal_smooth": 1,
            "motion_beat_keyframes": True,
        },
        "sample_extras": ["--invention-stack", "artwave", "--boost-quality"],
        "invention": "artwave",
    },
    "cartoon": {
        "engine": "spider_verse",  # strong graphic default; grammar is cartoon
        "grammar": "cartoon",
        "animation_preset": "looney",
        "post_grade": "vibrant",
        "cues": ("cartoon", "toon", "looney", "rubber hose", "western animation", "comic book"),
        "videomax": {
            "motion_shutter": False,
            "contact_ground": False,
            "physics_gate": False,
            "physics_repair": False,
            "permanence_repair": False,
            "secondary_track": True,
            "identity_bind": True,
            "deflicker_strength": 0.55,
            "min_permanence": 0.30,
            "min_physics": 0.0,
            "min_contact": 0.0,
            "motion_beat_keyframes": True,
        },
        "sample_extras": ["--invention-stack", "artwave"],
        "invention": "artwave",
    },
    "spider_verse": {
        "engine": "spider_verse",
        "grammar": "spider_verse",
        "animation_preset": "spider_verse",
        "post_grade": "vibrant",
        "cues": ("spider-verse", "spider verse", "halftone", "comic frame rate"),
        "videomax": {
            "motion_shutter": False,
            "contact_ground": False,
            "physics_gate": False,
            "permanence_repair": False,
            "deflicker_strength": 0.45,  # keep frame-rate chatter
            "min_permanence": 0.28,
            "motion_beat_keyframes": True,
        },
        "sample_extras": ["--invention-stack", "artwave"],
        "invention": "artwave",
    },
    "pixar_3d": {
        "engine": "pixar_3d",
        "grammar": "pixar_3d",
        "animation_preset": "pixar",
        "post_grade": "cinematic",
        "cues": ("pixar", "dreamworks", "stylized 3d", "3d animated film", "cgi character"),
        "videomax": {
            "motion_shutter": False,
            "contact_ground": True,
            "physics_gate": True,
            "physics_repair": True,
            "permanence_repair": True,
            "secondary_track": True,
            "min_permanence": 0.55,
            "min_physics": 0.42,
            "min_contact": 0.40,
            "min_identity": 0.58,
            "deflicker_strength": 0.80,
        },
        "sample_extras": ["--boost-quality", "--invention-stack", "maxwave"],
        "invention": "maxwave",
    },
    "cgi": {
        "engine": "cgi",
        "grammar": "cgi",
        "animation_preset": "realistic",
        "post_grade": "cinematic",
        "cues": ("cgi", "lookdev", "octane", "redshift", "unreal engine", "pbr render", "production render"),
        "videomax": {
            "motion_shutter": True,
            "contact_ground": True,
            "physics_gate": True,
            "physics_repair": True,
            "permanence_repair": True,
            "occlusion_resolve": True,
            "depth_interpolate": True,
            "min_permanence": 0.65,
            "min_physics": 0.50,
            "min_identity": 0.58,
            "deflicker_strength": 0.86,
        },
        "sample_extras": ["--boost-quality", "--invention-stack", "maxwave"],
        "invention": "maxwave",
    },
    "vfx": {
        "engine": "realistic",
        "grammar": "vfx",
        "animation_preset": "realistic",
        "post_grade": "cinematic",
        "cues": ("vfx", "explosion", "houdini", "particle", "simulation", "compositing", "fx plate", "volumetric"),
        "videomax": {
            "motion_shutter": True,
            "contact_ground": True,
            "physics_gate": True,
            "physics_repair": True,
            "permanence_repair": True,
            "secondary_track": True,
            "occlusion_resolve": True,
            "depth_interpolate": True,
            "min_permanence": 0.68,
            "min_physics": 0.52,
            "min_secondary": 0.40,
            "min_occlusion": 0.35,
            "deflicker_strength": 0.85,
        },
        "sample_extras": ["--boost-quality", "--invention-stack", "maxwave"],
        "invention": "maxwave",
    },
    "product": {
        "engine": "realistic",
        "grammar": "product",
        "animation_preset": "realistic",
        "post_grade": "product_clean",
        "cues": ("product shot", "turntable", "sku", "packshot", "ecommerce", "catalog", "logo reveal"),
        "videomax": {
            "glyph_lock": True,
            "motion_shutter": False,
            "contact_ground": True,
            "physics_gate": True,
            "permanence_repair": True,
            "camera_stabilize": True,
            "min_permanence": 0.78,
            "min_glyph": 0.60,
            "min_identity": 0.55,
            "deflicker_strength": 0.92,
            "extremity_lock": False,
        },
        "sample_extras": ["--invention-stack", "alphacut,maxwave"],
        "invention": "maxwave",
    },
    "stop_motion": {
        "engine": "claymation",
        "grammar": "stop_motion",
        "animation_preset": "stop_motion",
        "post_grade": "muted",
        "cues": ("stop motion", "stop-motion", "claymation", "clay", "wallace", "puppet animation"),
        "videomax": {
            "motion_shutter": False,
            "contact_ground": True,
            "physics_gate": False,
            "physics_repair": False,
            "permanence_repair": True,
            "hf_deshimmer": False,
            "deflicker_strength": 0.35,  # keep thumb-jitter
            "min_permanence": 0.45,
            "min_physics": 0.0,
            "temporal_smooth": 1,
        },
        "sample_extras": ["--invention-stack", "artwave"],
        "invention": "artwave",
    },
    "lego": {
        "engine": "lego",
        "grammar": "stop_motion",
        "animation_preset": "stop_motion",
        "post_grade": "vibrant",
        "cues": ("lego", "minifig", "brickfilm", "brick film"),
        "videomax": {
            "motion_shutter": False,
            "physics_gate": False,
            "hf_deshimmer": False,
            "deflicker_strength": 0.40,
            "permanence_repair": True,
            "min_permanence": 0.50,
            "glyph_lock": True,
        },
        "sample_extras": ["--invention-stack", "gameassets,artwave"],
        "invention": "gameassets",
    },
    "pixel_art": {
        "engine": "pixel_art",
        "grammar": "pixel_art",
        "animation_preset": "anime_tv",
        "post_grade": "",
        "cues": ("pixel art", "8-bit", "16-bit", "retro game", "pixel animation"),
        "videomax": {
            "motion_shutter": False,
            "contact_ground": False,
            "physics_gate": False,
            "permanence_repair": True,
            "hf_deshimmer": False,
            "deflicker_strength": 0.25,
            "depth_interpolate": False,
            "min_permanence": 0.40,
            "temporal_smooth": 0,
        },
        "sample_extras": ["--invention-stack", "gameassets"],
        "invention": "gameassets",
    },
    "low_poly": {
        "engine": "low_poly",
        "grammar": "low_poly",
        "animation_preset": "realistic",
        "post_grade": "muted",
        "cues": ("low poly", "ps1", "n64", "playstation", "affine texture"),
        "videomax": {
            "motion_shutter": False,
            "physics_gate": False,
            "deflicker_strength": 0.55,
            "min_permanence": 0.45,
        },
        "sample_extras": ["--invention-stack", "gameassets,artwave"],
        "invention": "gameassets",
    },
    "voxel": {
        "engine": "voxel",
        "grammar": "pixel_art",
        "animation_preset": "stop_motion",
        "post_grade": "vibrant",
        "cues": ("minecraft", "voxel", "blocky", "cube world"),
        "videomax": {
            "motion_shutter": False,
            "physics_gate": True,
            "physics_repair": True,
            "deflicker_strength": 0.50,
            "min_permanence": 0.50,
        },
        "sample_extras": ["--invention-stack", "gameassets"],
        "invention": "gameassets",
    },
    "vector": {
        "engine": "vector",
        "grammar": "vector",
        "animation_preset": "pixar",
        "post_grade": "",
        "cues": ("vector", "motion graphics", "flat design", "after effects style"),
        "videomax": {
            "motion_shutter": False,
            "contact_ground": False,
            "physics_gate": False,
            "glyph_lock": True,
            "deflicker_strength": 0.70,
            "min_permanence": 0.55,
            "min_glyph": 0.50,
        },
        "sample_extras": ["--invention-stack", "artwave"],
        "invention": "artwave",
    },
    "dream_logic": {
        "engine": "dream_logic",
        "grammar": "dream_logic",
        "animation_preset": "ghibli",
        "post_grade": "dreamy_orton",
        "cues": ("dream", "surreal", "impossible architecture", "salvador", "morphing reality"),
        "videomax": {
            "physics_gate": False,
            "physics_repair": False,
            "permanence_repair": False,
            "contact_ground": False,
            "deflicker_strength": 0.60,
            "min_permanence": 0.25,
            "min_physics": 0.0,
            "identity_bind": True,
        },
        "sample_extras": ["--invention-stack", "artwave,maxwave"],
        "invention": "artwave",
    },
    "hybrid": {
        "engine": "hybrid",
        "grammar": "hybrid",
        "animation_preset": "pixar",
        "post_grade": "cinematic",
        "cues": ("hybrid", "mixed media", "live action anime", "rotoscope"),
        "videomax": {
            "physics_gate": True,
            "permanence_repair": True,
            "identity_bind": True,
            "secondary_track": True,
            "min_permanence": 0.50,
            "min_identity": 0.55,
        },
        "sample_extras": ["--invention-stack", "maxwave,artwave"],
        "invention": "maxwave",
    },
}


def list_styles() -> list[str]:
    return sorted(STYLE_PROFILES.keys())


def route_style(prompt: str = "", *, style_hint: str = "", force: str = "") -> StyleRoute:
    """Detect medium and build a full style route."""
    forced = str(force or "").strip().lower().replace("-", "_")
    if forced in ("auto", "none", ""):
        forced = ""
    text = f"{prompt} {style_hint}".lower()

    style_id = forced
    if not style_id:
        # Priority order matters (specific before generic)
        order = (
            "product",
            "lego",
            "voxel",
            "pixel_art",
            "stop_motion",
            "spider_verse",
            "low_poly",
            "vector",
            "dream_logic",
            "vfx",
            "cgi",
            "pixar_3d",
            "anime_2d",
            "cartoon",
            "film",
            "hybrid",
            "realistic",
        )
        for sid in order:
            cues = STYLE_PROFILES[sid].get("cues") or ()
            if any(c in text for c in cues):
                style_id = sid
                break
        if not style_id:
            # Fall back to style_engines matcher
            try:
                from pipelines.video.style_engines import match_engine_from_prompt

                eng = match_engine_from_prompt(prompt, style_hint=style_hint)
                style_id = eng.value if eng.value in STYLE_PROFILES else "realistic"
                if style_id == "claymation":
                    style_id = "stop_motion"
            except Exception:
                style_id = "realistic"

    if style_id not in STYLE_PROFILES:
        style_id = "realistic"
    prof = STYLE_PROFILES[style_id]

    principles_txt = ""
    try:
        from pipelines.video.animation_principles import preset_principles, principles_prompt

        principles_txt = principles_prompt(preset_principles(str(prof.get("animation_preset") or "")))
    except Exception:
        pass

    engine_pos = engine_neg = ""
    try:
        from pipelines.video.style_engines import engine_by_id

        ep = engine_by_id(str(prof["engine"]))
        if ep:
            engine_pos, engine_neg = ep.positive, ep.negative
    except Exception:
        pass

    pos_bits = [x for x in (engine_pos, principles_txt) if x]
    extras = list(prof.get("sample_extras") or [])
    inv = str(prof.get("invention") or "maxwave")
    if "--invention-stack" not in extras:
        extras = ["--invention-stack", inv, *extras]

    notes = [f"style={style_id}", f"engine={prof['engine']}", f"grammar={prof['grammar']}"]
    if principles_txt:
        notes.append(f"principles={principles_txt[:80]}")

    return StyleRoute(
        style=style_id,
        engine=str(prof["engine"]),
        grammar=str(prof["grammar"]),
        animation_preset=str(prof.get("animation_preset") or ""),
        post_grade=str(prof.get("post_grade") or ""),
        videomax_profile=dict(prof.get("videomax") or {}),
        sample_extras=extras,
        positive_addon=", ".join(pos_bits),
        negative_addon=engine_neg,
        principles_prompt=principles_txt,
        notes=notes,
    )


def apply_style_route_to_options(opts: Any, route: StyleRoute) -> Any:
    """Stamp engine/grammar/post_grade + style VIDEOMAX dials onto ProcessOptions."""
    from dataclasses import replace

    from pipelines.video.motion_grammar import apply_motion_grammar, grammar_for_engine

    grammar = grammar_for_engine(route.grammar)
    opts = apply_motion_grammar(opts, grammar)
    fields = getattr(type(opts), "__dataclass_fields__", {}) or {}
    kw: dict[str, Any] = {}
    if "motion_grammar" in fields:
        kw["motion_grammar"] = route.grammar
    if "post_grade" in fields and route.post_grade:
        kw["post_grade"] = route.post_grade
    # Merge style-specific videomax dials (never disable VIDEOMAX repairs / floors)
    videomax_on = bool(getattr(opts, "videomax", False))
    _repair_keys = {
        "physics_gate",
        "physics_repair",
        "occlusion_resolve",
        "count_bind",
        "count_bind_repair",
        "permanence_repair",
        "identity_bind",
        "identity_lock",
        "extremity_lock",
        "shot_chain",
        "flow_consistency",
        "deflicker",
        "lip_sync",
    }
    for k, v in (route.videomax_profile or {}).items():
        if k not in fields:
            continue
        if videomax_on and k.startswith("min_"):
            try:
                kw[k] = max(float(getattr(opts, k, 0.0) or 0.0), float(v or 0.0))
            except (TypeError, ValueError):
                kw[k] = v
        elif videomax_on and (k.endswith("_repair") or k in _repair_keys):
            kw[k] = bool(getattr(opts, k, False)) or bool(v)
        else:
            kw[k] = v
    # Append principle / engine garnish onto motion_grammar prompt fields
    if "motion_grammar_positive" in fields and route.positive_addon:
        cur = str(getattr(opts, "motion_grammar_positive", "") or "")
        add = route.positive_addon
        if add and add.lower() not in cur.lower():
            kw["motion_grammar_positive"] = f"{cur}, {add}".strip(", ")
    if "motion_grammar_negative" in fields and route.negative_addon:
        cur = str(getattr(opts, "motion_grammar_negative", "") or "")
        add = route.negative_addon
        if add and add.lower() not in cur.lower():
            kw["motion_grammar_negative"] = f"{cur}, {add}".strip(", ")
    if "style_route" in fields:
        kw["style_route"] = route.to_dict()
    if kw:
        try:
            opts = replace(opts, **kw)
        except TypeError:
            for k, v in kw.items():
                try:
                    setattr(opts, k, v)
                except Exception:
                    pass
    return opts
