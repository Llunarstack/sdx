"""Unified prompt stack: context types, intent clauses, prompt intelligence, content controls, the stages, the runner and the ``sample.py`` bridge."""

from __future__ import annotations

import math
import os
import re
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any

from utils.prompt.content_controls import apply_content_controls, infer_content_controls_from_prompt
from utils.prompt.text import append_csv, append_unique, join_tags, split_tags, token_set


class StackMode(str, Enum):
    """Which stages run (inference full, preview subset, training guidance-only)."""

    INFERENCE = "inference"
    PREVIEW = "preview"
    TRAINING = "training"


@dataclass
class PromptArtifacts:
    """Staged fragments produced before the stack runs (layout, multi-instance, …)."""

    layout_negative: str = ""
    multi_instance_negative: str = ""
    detailed_scene_negative: str = ""
    visual_design_negative: str = ""
    character_negative: str = ""
    scene_negative: str = ""
    photo_negative: str = ""  # filled by guidance stage


@dataclass
class PromptContext:
    """
    Mutable input/output carrier for :func:`run_prompt_stack`.

    *positive* / *negative* are updated in place through stages.
    *args* is the argparse Namespace from ``sample.py`` when present.
    """

    positive: str
    negative: str = ""
    mode: StackMode = StackMode.INFERENCE
    args: Any = None
    artifacts: PromptArtifacts = field(default_factory=PromptArtifacts)
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)

    # Post-stack hints (sample.py encoding)
    t5_positive_hint: str = ""
    apply_scale_distortion_negative: bool = False
    # Text encoder the prompt is fitted to (token budget stage); 0 = sample_helpers.TEXT_MAX_TOKENS.
    tokenizer: Any = None
    max_tokens: int = 0


@dataclass
class PromptResult:
    positive: str
    negative: str
    t5_positive_hint: str = ""
    trace: list[str] = field(default_factory=list)
    analysis: Mapping[str, Any] | None = None
    resolved_controls: Mapping[str, str] | None = None


@dataclass(frozen=True)
class IntentClause:
    """Named bundle of positive/negative tags with optional activation predicate."""

    name: str
    positive: Sequence[str] = ()
    negative: Sequence[str] = ()
    description: str = ""

    def apply_positive(self, text: str) -> str:
        return append_unique(text, self.positive)

    def apply_negative(self, text: str) -> str:
        return append_unique(text, self.negative)


# Built-in clauses for stack extensions and book/visual-memory hooks.
CLAUSE_REGISTRY: dict[str, IntentClause] = {
    "uncensored.fidelity": IntentClause(
        name="uncensored.fidelity",
        positive=(
            "faithful to prompt description",
            "no arbitrary censorship",
            "anatomically coherent when nude",
        ),
        negative=(
            "random censor bar",
            "lens censorship blur",
            "sanitized rewrite of scene",
            "modesty costume insertion",
        ),
        description="Narrative fidelity for uncensored model runs.",
    ),
    "hands.stable": IntentClause(
        name="hands.stable",
        positive=("detailed hands", "correct finger count", "natural hand pose"),
        negative=("extra fingers", "fused fingers", "malformed hands", "missing fingers"),
        description="Hand stability booster.",
    ),
    "composition.single": IntentClause(
        name="composition.single",
        positive=("solo", "single subject", "one person"),
        negative=("duplicate person", "cloned face", "twin duplicate"),
        description="Single-subject guardrails.",
    ),
    "quality.micro": IntentClause(
        name="quality.micro",
        positive=(
            "fine texture detail",
            "material-accurate surface",
            "clean edges",
            "subtle micro-contrast",
        ),
        negative=("mushy detail", "plastic texture", "smudged fine detail"),
        description="Micro-detail quality clause.",
    ),
    "style.chaos": IntentClause(
        name="style.chaos",
        positive=(
            "visually unforgettable",
            "deliberately uncanny art direction",
            "impossible material honesty",
            "aggressive compositional tension",
        ),
        negative=(
            "forgettable average",
            "template AI look",
            "instagram filter pack",
            "boring safe composition",
            "generic stock photo",
        ),
        description="Maximum originality / anti-slop clause for style explore.",
    ),
    "style.surreal": IntentClause(
        name="style.surreal",
        positive=(
            "dream logic scale",
            "liminal atmosphere",
            "synesthetic color mood",
            "designed accident framing",
        ),
        negative=("literal boring snapshot", "flat realism only", "explainable mundane scene"),
        description="Surreal / liminal style explore booster.",
    ),
    "style.glitch": IntentClause(
        name="style.glitch",
        positive=(
            "datamoshed accents",
            "chromatic aberration design",
            "CRT phosphor aesthetic",
            "intentional compression artifact beauty",
        ),
        negative=("clean 4k digital", "sterile corporate render", "perfect smooth gradients"),
        description="Digital corruption / glitch art clause.",
    ),
    "style.apocalypse": IntentClause(
        name="style.apocalypse",
        positive=(
            "epic scale ruin",
            "dust-choked atmosphere",
            "weathered mythic weight",
            "particulate air density",
        ),
        negative=("cheerful picnic", "pristine modern city", "empty greybox void"),
        description="Post-collapse epic atmosphere clause.",
    ),
}


def apply_clauses(
    positive: str,
    negative: str,
    clause_names: Sequence[str],
    *,
    registry: Mapping[str, IntentClause] | None = None,
) -> tuple[str, str]:
    reg = registry or CLAUSE_REGISTRY
    p, n = positive, negative
    for name in clause_names:
        clause = reg.get(str(name).strip())
        if clause is None:
            continue
        p = clause.apply_positive(p)
        n = clause.apply_negative(n)
    return p, n


def list_clauses() -> list[str]:
    return sorted(CLAUSE_REGISTRY.keys())


@dataclass
class PromptAnalysis:
    """Structured read of a user prompt before generation."""

    token_count: int = 0
    char_count: int = 0
    comma_rich: bool = False
    domains: list[str] = field(default_factory=list)
    complexity: str = "simple"  # simple | moderate | complex | extreme
    missing_quality: bool = False
    photographic: bool = False
    has_text_intent: bool = False
    solo_subject: bool = False
    multi_subject: bool = False
    suggested_controls: dict[str, str] = field(default_factory=dict)
    suggested_positive: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


_QUALITY_MARKERS = frozenset(
    {
        "masterpiece",
        "best quality",
        "high quality",
        "highres",
        "absurdres",
        "8k",
        "ultra detailed",
        "detailed",
    }
)

_PHOTO_MARKERS = (
    "photoreal",
    "raw photo",
    "dslr",
    "35mm",
    "film photograph",
    "iphone photo",
    "shot on",
    "f/1.",
    "bokeh",
)

_TEXT_MARKERS = (
    "sign that says",
    "text that says",
    "lettering",
    "speech bubble",
    "caption reads",
    'reads "',
    'says "',
)


def _detect_domains(p_lower: str, tags: set[str]) -> list[str]:
    domains: list[str] = []
    if any(x in p_lower for x in ("anime", "manga", "cel shading", "1girl", "1boy")) or tags & {
        "1girl",
        "1boy",
        "anime",
    }:
        domains.append("anime")
    if any(x in p_lower for x in _PHOTO_MARKERS):
        domains.append("photographic")
    if any(x in p_lower for x in ("3d render", "octane", "blender", "unreal engine", "cgi")):
        domains.append("3d")
    if any(x in p_lower for x in ("interior", "architecture", "facade", "building")):
        domains.append("architecture")
    if any(x in p_lower for x in ("vehicle", "car", "motorcycle", "aircraft")):
        domains.append("vehicles")
    if any(x in p_lower for x in ("horror", "eldritch", "cosmic", "nightmare")):
        domains.append("horror")
    if any(x in p_lower for x in ("dragon", "furry", "anthro", "robot", "mecha")):
        domains.append("creature")
    return domains


def analyze_prompt(prompt: str) -> PromptAnalysis:
    raw = (prompt or "").strip()
    p_lower = raw.lower()
    tags = token_set(raw)
    tag_list = split_tags(raw)
    n_tags = len(tag_list)
    n_char = len(raw)

    analysis = PromptAnalysis(
        token_count=n_tags,
        char_count=n_char,
        comma_rich="," in raw,
        domains=_detect_domains(p_lower, tags),
    )

    if n_tags >= 22 or n_char >= 520:
        analysis.complexity = "extreme"
    elif n_tags >= 12 or n_char >= 280:
        analysis.complexity = "complex"
    elif n_tags >= 6 or n_char >= 120:
        analysis.complexity = "moderate"

    analysis.missing_quality = not any(m in p_lower for m in _QUALITY_MARKERS)
    analysis.photographic = "photographic" in analysis.domains
    analysis.has_text_intent = any(m in p_lower for m in _TEXT_MARKERS)
    analysis.solo_subject = bool(tags & {"solo", "1girl", "1boy", "1other"}) or "single " in p_lower
    analysis.multi_subject = bool(tags & {"2girls", "2boys", "3girls"}) or any(
        x in p_lower for x in ("crowd", "group", "2girls", "couple", "duo")
    )

    suggested: dict[str, str] = {}
    if analysis.complexity in ("complex", "extreme"):
        suggested["adherence_pack"] = "standard"
    if analysis.complexity == "extreme":
        suggested["adherence_pack"] = "strict"
    if analysis.photographic:
        suggested["human_media_mode"] = "photographic"
    if analysis.solo_subject and not analysis.multi_subject:
        suggested["composition_mode"] = "single_subject"
        suggested["people_layout"] = "solo"
    if analysis.missing_quality and analysis.complexity in ("moderate", "complex", "extreme"):
        analysis.suggested_positive.append("detailed")
        analysis.notes.append("Consider --quality-pack top or masterpiece tags for short prompts.")

    analysis.suggested_controls = suggested
    return analysis


def apply_intelligence(
    positive: str,
    analysis: PromptAnalysis,
    *,
    auto_quality: bool = True,
    auto_controls: bool = True,
) -> tuple[str, dict[str, str]]:
    """Optionally prepend light quality tags and return control hints."""
    out = positive
    controls: dict[str, str] = dict(analysis.suggested_controls) if auto_controls else {}
    if auto_quality and analysis.missing_quality and analysis.complexity in ("simple", "moderate"):
        if "detailed" in analysis.suggested_positive:
            out = append_unique(out, analysis.suggested_positive)
    return out, controls


def _get(args: Any, name: str, default: str = "none") -> str:
    if args is None:
        return default
    return str(getattr(args, name, default) or default)


def _get_bool(args: Any, name: str, default: bool = False) -> bool:
    if args is None:
        return default
    return bool(getattr(args, name, default))


@dataclass
class ContentControlState:
    """Explicit + inferred content-control modes (single source of truth)."""

    pose_mode: str = "none"
    view_angle: str = "none"
    subject_sex: str = "none"
    scene_domain: str = "none"
    clothing_mode: str = "none"
    background_mode: str = "none"
    people_layout: str = "none"
    relationship_mode: str = "none"
    object_layout: str = "none"
    hand_mode: str = "none"
    pose_naturalness: str = "none"
    typography_mode: str = "none"
    quality_pack: str = "none"
    lighting_mode: str = "none"
    skin_detail_mode: str = "none"
    body_proportion: str = "none"
    interaction_intensity: str = "none"
    advanced_pose: str = "none"
    object_interaction: str = "none"
    environment_type: str = "none"
    style_mode: str = "none"
    composition_mode: str = "none"
    artist_composition: str = "none"
    anti_ai_pack: str = "none"
    human_media_mode: str = "none"
    lora_scaffold: str = "none"
    adherence_pack: str = "none"
    # flags
    style_lock: bool = False
    anti_style_bleed: bool = False
    anti_duplicate_subjects: bool = False
    anti_perspective_drift: bool = False
    cleanup_conflicting_tags: bool = False
    allow_text_in_image: bool = False
    one_shot_boost: bool = True

    def to_apply_kwargs(self) -> dict[str, Any]:
        return asdict(self)


def merge_content_control_overrides(state: ContentControlState, overrides: Mapping[str, Any]) -> ContentControlState:
    """Apply intelligence (or other) overrides without passing unknown keys to the dataclass."""
    if not overrides:
        return state
    data = state.to_apply_kwargs()
    valid = {f.name for f in fields(ContentControlState)}
    for key, val in overrides.items():
        if key not in valid or val is None:
            continue
        if isinstance(data.get(key), bool):
            if val:
                data[key] = bool(val)
        elif data.get(key) == "none" and val:
            data[key] = str(val)
    return ContentControlState(**data)


def resolve_content_controls(args: Any, prompt: str, *, auto_infer: bool = True) -> ContentControlState:
    """CLI flags (same names as the fields; ``--text-in-image`` -> ``allow_text_in_image``), then modes inferred
    from the prompt fill whatever the CLI left at ``"none"``."""
    data: dict[str, Any] = {}
    for f in fields(ContentControlState):
        src = "text_in_image" if f.name == "allow_text_in_image" else f.name
        data[f.name] = _get_bool(args, src, f.default) if isinstance(f.default, bool) else _get(args, src)
    if data["lora_scaffold"] == "none" and _get_bool(args, "lora_scaffold_auto") and getattr(args, "lora", None):
        data["lora_scaffold"] = "blend"
    state = ContentControlState(**data)
    if not auto_infer or args is None or not _get_bool(args, "auto_content_fix", True):
        return state
    return merge_content_control_overrides(state, infer_content_controls_from_prompt(prompt))


def apply_resolved_controls(positive: str, negative: str, state: ContentControlState) -> tuple[str, str]:
    return apply_content_controls(positive, negative, **state.to_apply_kwargs())


DEFAULT_EXTRA_TOKENS = 96
DEFAULT_NEGATIVE_EXTRA_TOKENS = 160
_PIECE = re.compile(r"\w+|[^\w\s]")


def _encoder_max_tokens(ctx: PromptContext) -> int:
    if ctx.max_tokens:
        return int(ctx.max_tokens)
    try:
        from utils.generation.sample_core import TEXT_MAX_TOKENS

        return int(TEXT_MAX_TOKENS)
    except ImportError:
        return 300


def tag_token_counts(tags: list[str], tokenizer: Any = None) -> list[int]:
    """Encoder tokens per tag (no special tokens); 1.35 per word piece without a tokenizer."""
    if not tags:
        return []
    if tokenizer is not None:
        try:
            return [len(ids) for ids in tokenizer(tags, add_special_tokens=False)["input_ids"]]
        except Exception:
            pass
    return [math.ceil(1.35 * len(_PIECE.findall(t))) for t in tags]


def _leading_text(text: str, n_tags: int) -> str:
    """``text`` up to the end of its ``n_tags``-th non-empty comma tag, unchanged."""
    if n_tags <= 0:
        return ""
    seen = 0
    for m in re.finditer(r"[^,]+", text or ""):
        if m.group().strip():
            seen += 1
            if seen == n_tags:
                return text[: m.end()].strip()
    return (text or "").strip()


def _user_keys(text: str) -> set[str]:
    from utils.prompt.text import parse_prompt_emphasis

    cleaned, _ = parse_prompt_emphasis(text or "")
    return {t.lower() for t in split_tags(cleaned)}


def fit_tags(
    protected: list[str],
    ordered: list[str],
    *,
    limit: int,
    extra: int,
    tokenizer: Any = None,
) -> tuple[list[str], int]:
    """Protected tags plus as many ``ordered`` tags as fit; returns (kept, dropped count)."""
    seen: set[str] = set()
    head: list[str] = []
    for t in protected:
        if t.lower() not in seen:
            seen.add(t.lower())
            head.append(t)
    tail = []
    for t in ordered:
        if t.lower() not in seen:
            seen.add(t.lower())
            tail.append(t)
    head_counts = tag_token_counts(head, tokenizer)
    used = sum(head_counts) + max(0, len(head) - 1) + 1  # commas + EOS
    cap = limit if extra < 0 else min(limit, used + extra)
    kept = list(head)
    dropped = 0
    for tag, n in zip(tail, tag_token_counts(tail, tokenizer), strict=True):
        cost = n + (1 if kept else 0)
        if used + cost <= cap:
            kept.append(tag)
            used += cost
        else:
            dropped += 1
    return kept, dropped


def stage_token_budget(ctx: PromptContext) -> None:
    args = ctx.args
    if args is None:
        return
    from utils.prompt.art_direction import gate_tag

    ad = getattr(args, "_art_direction", None)
    user = str(getattr(args, "_user_prompt", "") or ctx.metadata.get("input_positive", "") or "")
    limit = _encoder_max_tokens(ctx)
    extra = int(getattr(args, "prompt_budget", DEFAULT_EXTRA_TOKENS))
    neg_extra = int(getattr(args, "negative_budget", DEFAULT_NEGATIVE_EXTRA_TOKENS))

    keys = _user_keys(user)
    tags = split_tags(ctx.positive)
    first_user = next((i for i, t in enumerate(tags) if t.lower() in keys), len(tags) if keys else 0)
    head_end = first_user
    while head_end < len(tags) and tags[head_end].lower() in keys:
        head_end += 1
    protected = tags[:head_end] + [t for t in tags[head_end:] if t.lower() in keys]
    additions = [t for t in tags[head_end:] if t.lower() not in keys]
    caption = split_tags(ad.caption()) if ad is not None else []
    caption += split_tags(str(getattr(args, "_shot_caption", "") or ""))  # utils.prompt.realism shot spec
    gated = [t for t in additions if ad is None or gate_tag(t, ad, user_text=user)]
    kept, dropped = fit_tags(protected, caption + gated, limit=limit, extra=extra, tokenizer=ctx.tokenizer)
    # The leading user text is kept byte-for-byte so emphasis offsets computed on it stay valid.
    head = _leading_text(ctx.positive, head_end)
    head_keys = {t.lower() for t in tags[:head_end]}
    tail = [t for t in kept if t.lower() not in head_keys]
    ctx.positive = ", ".join(p for p in (head, join_tags(tail)) if p)

    user_neg = str(getattr(args, "_user_negative", "") or "")
    neg_keys = _user_keys(user_neg)
    neg_tags = split_tags(ctx.negative)
    neg_protected = [t for t in neg_tags if t.lower() in neg_keys]
    neg_rest = [
        t
        for t in neg_tags
        if t.lower() not in neg_keys and (ad is None or gate_tag(t, ad, user_text=user, negative=True))
    ]
    neg_kept, neg_dropped = fit_tags(neg_protected, neg_rest, limit=limit, extra=neg_extra, tokenizer=ctx.tokenizer)
    ctx.negative = join_tags(neg_kept) or " "

    ctx.metadata["budget"] = {
        "limit": limit,
        "extra": extra,
        "gated": len(additions) - len(gated),
        "dropped": dropped,
        "negative_gated": len(neg_tags) - len(neg_protected) - len(neg_rest),
        "negative_dropped": neg_dropped,
    }
    ctx.trace.append(f"budget:{len(kept)}tags")
    if hasattr(args, "prompt"):
        args.prompt = ctx.positive


def stage_intelligence(ctx: PromptContext) -> None:
    args = ctx.args
    if args is None or not getattr(args, "prompt_stack_intelligence", True):
        return

    analysis = analyze_prompt(ctx.positive)
    ctx.metadata["analysis"] = analysis
    ctx.positive, extra_controls = apply_intelligence(
        ctx.positive,
        analysis,
        auto_quality=bool(getattr(args, "prompt_stack_auto_quality", True)),
        auto_controls=bool(getattr(args, "auto_content_fix", True)),
    )
    if extra_controls:
        ctx.metadata["intelligence_controls"] = extra_controls
    ctx.trace.append(f"intelligence:{analysis.complexity}")


def _get_active_genome(args: Any):
    if args is None:
        return None
    genome = getattr(args, "_active_style_genome", None)
    if genome is not None:
        return genome
    path = str(getattr(args, "style_genome_file", "") or "").strip()
    if not path:
        return None
    try:
        from utils.prompt.style import resolve_style_genome_for_args

        return resolve_style_genome_for_args(args, str(getattr(args, "prompt", "") or ""))
    except Exception:
        return None


def stage_style_genome(ctx: PromptContext) -> None:
    """Merge genome axes into positive/negative; set style head on args."""
    args = ctx.args
    if args is not None and not getattr(args, "style_genome_enabled", True):
        return
    if args is not None and not (
        getattr(args, "_active_style_genome", None)
        or str(getattr(args, "style_genome_file", "") or "").strip()
        or int(getattr(args, "invent_styles", 0) or 0) > 0
    ):
        return

    genome = _get_active_genome(args)
    if genome is None:
        return

    chaos_level = float(getattr(args, "style_chaos_level", 0.0) or 0.0) if args is not None else 0.0
    if chaos_level > 0.01:
        try:
            from utils.prompt.style import apply_chaos_level

            genome = apply_chaos_level(genome, chaos_level)
        except Exception:
            pass

    ctx.positive = genome.compile_positive(ctx.positive)
    ctx.negative = genome.compile_negative(ctx.negative)
    ctx.metadata["style_genome"] = genome.to_dict()
    ctx.trace.append(f"style_genome:{genome.id}")
    if chaos_level > 0.35:
        ctx.trace.append(f"style_genome:chaos_{chaos_level:.0%}")

    if args is not None:
        if not (getattr(args, "style", None) or "").strip():
            args.style = genome.style_head_string()
        if hasattr(args, "prompt"):
            args.prompt = ctx.positive


def stage_special_helpers(ctx: PromptContext) -> None:
    args = ctx.args
    if args is None:
        return
    mode = str(getattr(args, "prompt_special_helpers", "auto") or "auto").strip().lower()
    if mode in ("off", "none", "false", "0", "skip"):
        return
    try:
        from utils.prompt.special_prompt_helpers import apply_special_helpers

        pos, neg = apply_special_helpers(ctx.positive, ctx.negative, category=mode)
        if pos != ctx.positive or neg != ctx.negative:
            ctx.positive, ctx.negative = pos, neg
            ctx.trace.append(f"special:{mode}")
    except Exception as exc:
        ctx.metadata["special_helpers_error"] = str(exc)
        ctx.trace.append("special:failed")


def stage_game(ctx: PromptContext) -> None:
    """Game asset conventions (sprite sheet grid, seamless tiles, icon silhouette, true pixel art, ...)."""
    if ctx.args is None:
        return
    from utils.prompt.game import apply_game_prompt, resolve_game_profile

    prof = resolve_game_profile(ctx.args)
    if prof is None:
        return
    ctx.positive, ctx.negative = apply_game_prompt(ctx.positive, ctx.negative, prof)
    ctx.metadata["game"] = prof
    ctx.trace.append(f"game:{prof.asset}{'+pixel' if prof.pixel_art else ''}")


def stage_content_controls(ctx: PromptContext) -> None:
    args = ctx.args
    auto = bool(getattr(args, "auto_content_fix", True)) if args is not None else True
    state = resolve_content_controls(args, ctx.positive, auto_infer=auto)

    intel = ctx.metadata.get("intelligence_controls") or {}
    if intel:
        state = merge_content_control_overrides(state, intel)

    try:
        pos, neg = apply_resolved_controls(ctx.positive, ctx.negative, state)
        ctx.positive, ctx.negative = pos, neg
        ctx.metadata["resolved_controls"] = {
            k: v for k, v in state.to_apply_kwargs().items() if k.endswith("_mode") or k.endswith("_pack")
        }
        ctx.trace.append("content_controls")
    except Exception as exc:
        ctx.metadata["content_controls_error"] = str(exc)
        ctx.trace.append("content_controls:failed")


def stage_clauses(ctx: PromptContext) -> None:
    args = ctx.args
    if args is None:
        return
    raw = str(getattr(args, "prompt_clauses", "") or "").strip()
    if not raw:
        return
    names = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
    ctx.positive, ctx.negative = apply_clauses(ctx.positive, ctx.negative, names)
    ctx.trace.append(f"clauses:{len(names)}")


def stage_post_enrich(ctx: PromptContext) -> None:
    """Character/scene negatives and scale-distortion guard."""
    art = ctx.artifacts
    negative = ctx.negative

    if ctx.apply_scale_distortion_negative:
        scale_neg = (
            "distorted proportions, warped anatomy, stretched limbs, "
            "elongated limbs, tiny head, giant hands, bad perspective scale"
        )
        negative = append_csv(negative, scale_neg)
        ctx.trace.append("post:scale_distortion")

    if art.character_negative:
        negative = append_csv(negative, art.character_negative)
        ctx.trace.append("post:character_neg")
    if art.scene_negative:
        negative = append_csv(negative, art.scene_negative)
        ctx.trace.append("post:scene_neg")

    ctx.negative = negative


def stage_prompt_breakdown(ctx: PromptContext) -> None:
    if ctx.mode == StackMode.PREVIEW:
        return
    args = ctx.args
    if args is None:
        return
    pbrk = str(getattr(args, "prompt_breakdown", "off") or "off").lower()
    if pbrk == "off" or not ctx.positive.strip():
        return
    if getattr(args, "_layout_compiled", None) is not None:
        return
    try:
        from utils.prompt.text import PRESET_SECTION_ORDER, apply_prompt_breakdown, warrant_prompt_breakdown

        if pbrk == "on" or (pbrk == "auto" and warrant_prompt_breakdown(ctx.positive)):
            fmt = str(getattr(args, "prompt_breakdown_format", "ordered") or "ordered").lower()
            if fmt not in ("ordered", "labeled"):
                fmt = "ordered"
            order = str(getattr(args, "prompt_breakdown_order", "subject_first") or "subject_first").lower()
            if order not in PRESET_SECTION_ORDER:
                order = "subject_first"
            flat, t5s = apply_prompt_breakdown(
                ctx.positive,
                order=order,  # type: ignore[arg-type]
                output_format=fmt,  # type: ignore[arg-type]
            )
            if flat:
                ctx.positive = flat
                args.prompt = flat
                if t5s != flat:
                    ctx.t5_positive_hint = t5s
                ctx.trace.append(f"breakdown:{pbrk}")
                print(
                    f"Prompt breakdown ({pbrk}, order={order}, format={fmt}).",
                    file=sys.stderr,
                )
    except Exception as exc:
        print(f"Prompt breakdown skipped: {exc}", file=sys.stderr)


def stage_neg_filter(ctx: PromptContext) -> None:
    args = ctx.args
    if args is not None and getattr(args, "no_neg_filter", False):
        ctx.trace.append("neg_filter:skipped")
        return
    from utils.prompt.text import filter_negative_by_positive

    # Conflicts are judged against what was asked for (the user's words and the art-direction plan),
    # not pack text: an anti-AI positive mentioning skin must not cancel the "plastic skin" negative.
    asked = str(getattr(args, "_user_prompt", "") or "") if args is not None else ""
    ad = getattr(args, "_art_direction", None) if args is not None else None
    if asked and ad is not None:
        asked = f"{asked}, {ad.caption()}"
    raw = ctx.negative
    filtered = filter_negative_by_positive(asked or ctx.positive, raw)
    if not filtered.strip():
        filtered = " "
    if filtered != raw:
        ctx.trace.append("neg_filter:applied")
        if ctx.mode != StackMode.PREVIEW:
            print(
                f'Negative prompt filtered (conflict resolution): "{raw[:60]}{"..." if len(raw) > 60 else ""}" '
                f'-> "{filtered[:60]}{"..." if len(filtered) > 60 else ""}"',
                file=sys.stderr,
            )
    else:
        ctx.trace.append("neg_filter:none")
    ctx.negative = filtered


def stage_guidance(ctx: PromptContext) -> None:
    """Apply shortcomings, art medium, style guidance, and photo realism to positive."""
    args = ctx.args
    if args is None:
        return

    prompt = ctx.positive
    if not prompt.strip():
        return

    sm_mode = str(getattr(args, "shortcomings_mitigation", "none") or "none").lower()
    ag_mode = str(getattr(args, "art_guidance_mode", "none") or "none").lower()
    ag_anat = str(getattr(args, "anatomy_guidance", "none") or "none").lower()
    sg_mode = str(getattr(args, "style_guidance_mode", "none") or "none").lower()

    ctx.metadata["guidance"] = {
        "shortcomings": sm_mode,
        "art_guidance": ag_mode,
        "anatomy": ag_anat,
        "style_guidance": sg_mode,
    }

    if sm_mode in ("auto", "all"):
        try:
            from config.prompts import mitigation_fragments

            pos_sm, neg_sm = mitigation_fragments(
                prompt,
                sm_mode,  # type: ignore[arg-type]
                include_2d_pack=bool(getattr(args, "shortcomings_2d", False)),
            )
            if pos_sm:
                prompt = append_csv(prompt, pos_sm)
                ctx.trace.append("guidance:shortcomings_pos")
            if neg_sm:
                ctx.metadata.setdefault("guidance_neg_parts", []).append(neg_sm)
                ctx.trace.append("guidance:shortcomings_neg")
        except Exception:
            pass

    if ag_mode in ("auto", "all") or ag_anat in ("lite", "strong"):
        try:
            from config.styles import guidance_fragments

            ag_pos, ag_neg = guidance_fragments(
                prompt,
                ag_mode,  # type: ignore[arg-type]
                include_photography=not bool(getattr(args, "no_art_guidance_photography", False)),
                anatomy_mode=ag_anat,  # type: ignore[arg-type]
            )
            ctx.metadata.setdefault("guidance_neg_parts", []).append(ag_neg)
            if ag_pos:
                prompt = append_csv(prompt, ag_pos)
                ctx.trace.append("guidance:art_medium_pos")
        except Exception:
            pass

    if prompt.strip():
        try:
            from config.styles import style_guidance_fragments

            sg_eff = sg_mode if sg_mode in ("auto", "all") else "none"
            sg_pos, sg_neg = style_guidance_fragments(
                prompt,
                sg_eff,  # type: ignore[arg-type]
                include_artist_refs=bool(getattr(args, "style_guidance_artists", True)),
            )
            ctx.metadata.setdefault("guidance_neg_parts", []).append(sg_neg)
            if sg_pos:
                prompt = append_csv(prompt, sg_pos)
                ctx.trace.append("guidance:style_pos")
        except Exception:
            pass

    if ctx.mode != StackMode.TRAINING:
        prompt, photo_neg, photo_meta = _apply_photo_realism(prompt, args)
        if photo_neg:
            ctx.artifacts.photo_negative = photo_neg
            ctx.trace.append("guidance:photo_realism")
        if photo_meta:
            ctx.metadata["photo"] = photo_meta
            for key, val in photo_meta.items():
                setattr(args, key, val)

    ctx.positive = prompt
    if args is not None and hasattr(args, "prompt"):
        args.prompt = prompt


def _apply_photo_realism(prompt: str, args: Any) -> tuple[str, str, dict]:
    try:
        from utils.prompt.art_direction import (
            infer_photo_realism_controls,
            is_photographic_prompt,
            photo_realism_fragments,
            recommend_photo_post_profile,
        )
    except ImportError:
        return prompt, "", {}

    pr_pack = str(getattr(args, "photo_realism_pack", "none") or "none")
    pr_grade = str(getattr(args, "photo_color_grade", "none") or "none")
    pr_light = str(getattr(args, "photo_lighting_technique", "none") or "none")
    pr_filter = str(getattr(args, "photo_filter", "none") or "none")
    pr_grain = str(getattr(args, "photo_grain_style", "none") or "none")
    pr_strength = float(getattr(args, "photo_realism_strength", 1.0) or 1.0)

    if getattr(args, "auto_photo_realism", True):
        inf = infer_photo_realism_controls(prompt)
        if pr_pack == "none" and inf.get("photo_realism_pack"):
            pr_pack = str(inf["photo_realism_pack"])
        if pr_grade == "none" and inf.get("photo_color_grade"):
            pr_grade = str(inf["photo_color_grade"])
        if pr_light == "none" and inf.get("photo_lighting_technique"):
            pr_light = str(inf["photo_lighting_technique"])
        if pr_filter == "none" and inf.get("photo_filter"):
            pr_filter = str(inf["photo_filter"])

    pr_pos, pr_neg = photo_realism_fragments(
        photo_realism_pack=pr_pack,
        photo_color_grade=pr_grade,
        photo_lighting_technique=pr_light,
        photo_filter=pr_filter,
        photo_grain_style=pr_grain,
        strength=pr_strength,
    )
    meta: dict = {
        "_pr_grade_ef": str(pr_grade or "none"),
        "_pr_filter_ef": str(pr_filter or "none"),
        "_pr_grain_ef": str(pr_grain or "none"),
        "_pr_pack_ef": str(pr_pack or "none"),
        "_is_photo_prompt": False,
        "_auto_pick_metric": "",
    }
    if pr_pos:
        prompt = append_csv(prompt, pr_pos)
        if str(getattr(args, "human_media_mode", "none") or "none") == "none":
            args.human_media_mode = "photographic"
    try:
        meta["_is_photo_prompt"] = bool(is_photographic_prompt(prompt))
    except Exception:
        pass
    if bool(getattr(args, "realism_autopilot", True)) and (meta["_is_photo_prompt"] or meta["_pr_pack_ef"] != "none"):
        try:
            rp = recommend_photo_post_profile(
                photo_realism_pack=meta["_pr_pack_ef"],
                photo_color_grade=meta["_pr_grade_ef"],
                photo_filter=meta["_pr_filter_ef"],
                photo_grain_style=meta["_pr_grain_ef"],
            )
            if (
                str(meta["_pr_grain_ef"]).lower() == "none"
                and str(rp.get("photo_grain_style", "none")).lower() != "none"
            ):
                meta["_pr_grain_ef"] = str(rp["photo_grain_style"])
                args.photo_grain_style = meta["_pr_grain_ef"]
            if abs(float(getattr(args, "photo_post_strength", 0.6) or 0.6) - 0.6) < 1e-6:
                try:
                    args.photo_post_strength = float(rp.get("photo_post_strength", "0.62"))
                except Exception:
                    pass
            meta["_auto_pick_metric"] = str(rp.get("pick_best_metric", "") or "")
        except Exception:
            pass
    return prompt, pr_neg or "", meta


def _training_guidance_namespace(
    *,
    shortcomings_mode: str = "none",
    shortcomings_2d: bool = False,
    art_guidance_mode: str = "none",
    anatomy_guidance: str = "none",
    style_guidance_mode: str = "none",
    style_guidance_artists: bool = True,
    no_art_guidance_photography: bool = False,
) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        shortcomings_mitigation=shortcomings_mode,
        shortcomings_2d=shortcomings_2d,
        art_guidance_mode=art_guidance_mode,
        anatomy_guidance=anatomy_guidance,
        style_guidance_mode=style_guidance_mode,
        style_guidance_artists=style_guidance_artists,
        no_art_guidance_photography=no_art_guidance_photography,
    )


def apply_training_guidance_pair(
    caption: str,
    negative_caption: str = "",
    *,
    shortcomings_mode: str = "none",
    shortcomings_2d: bool = False,
    art_guidance_mode: str = "none",
    anatomy_guidance: str = "none",
    style_guidance_mode: str = "none",
    style_guidance_artists: bool = True,
    include_art_guidance_photography: bool = True,
) -> tuple[str, str]:
    """
    Training-side helper: same shortcomings / art / style guidance fragments as ``sample.py`` (pos + neg).
    """
    ctx = PromptContext(
        positive=caption,
        negative=negative_caption or "",
        mode=StackMode.TRAINING,
        args=_training_guidance_namespace(
            shortcomings_mode=shortcomings_mode,
            shortcomings_2d=shortcomings_2d,
            art_guidance_mode=art_guidance_mode,
            anatomy_guidance=anatomy_guidance,
            style_guidance_mode=style_guidance_mode,
            style_guidance_artists=style_guidance_artists,
            no_art_guidance_photography=not include_art_guidance_photography,
        ),
    )
    stage_guidance(ctx)
    negative = negative_caption or ""
    for part in ctx.metadata.get("guidance_neg_parts") or []:
        if part:
            negative = append_csv(negative, part)
    return ctx.positive, negative


def merge_guidance_for_training_caption(
    caption: str,
    *,
    shortcomings_mode: str = "none",
    shortcomings_2d: bool = False,
    art_guidance_mode: str = "none",
    anatomy_guidance: str = "none",
    style_guidance_mode: str = "none",
    style_guidance_artists: bool = True,
    no_art_guidance_photography: bool = False,
) -> str:
    """Training-side helper: positive guidance only (no photo / neg stack)."""
    pos, _ = apply_training_guidance_pair(
        caption,
        "",
        shortcomings_mode=shortcomings_mode,
        shortcomings_2d=shortcomings_2d,
        art_guidance_mode=art_guidance_mode,
        anatomy_guidance=anatomy_guidance,
        style_guidance_mode=style_guidance_mode,
        style_guidance_artists=style_guidance_artists,
        include_art_guidance_photography=not no_art_guidance_photography,
    )
    return pos


def stage_negative_bootstrap(ctx: PromptContext) -> None:
    """Build base negative from config, staged attrs, and guidance negs."""
    args = ctx.args
    positive = ctx.positive
    art = ctx.artifacts

    try:
        from config.prompts import (
            ANTI_AI_LOOK_NEGATIVE,
            ANTI_AI_LOOK_NEGATIVE_STRONG,
            DEFAULT_NEGATIVE_PROMPT,
            TEXT_IN_IMAGE_NEGATIVE,
            TEXT_IN_IMAGE_PHRASES,
        )
    except ImportError:
        DEFAULT_NEGATIVE_PROMPT = " "
        TEXT_IN_IMAGE_NEGATIVE = "garbled text, watermark, low quality, blurry"
        TEXT_IN_IMAGE_PHRASES = ("sign that says", "text that says", "lettering")
        ANTI_AI_LOOK_NEGATIVE = "oversaturated, plastic skin, synthetic, CGI"
        ANTI_AI_LOOK_NEGATIVE_STRONG = ANTI_AI_LOOK_NEGATIVE

    user_neg = ""
    staged_neg = ""
    if args is not None:
        staged_neg = (getattr(args, "negative_prompt", None) or "").strip()
        # sample.py records the user's own negative; policy packs appended to args.negative_prompt since
        # then are additions on top of the default negative, not a replacement for it.
        user_neg = str(getattr(args, "_user_negative", staged_neg) or "").strip()

    if user_neg:
        negative = user_neg
        ctx.trace.append("negative:user")
    elif args is not None and getattr(args, "text_in_image", False):
        negative = TEXT_IN_IMAGE_NEGATIVE
        ctx.trace.append("negative:text_in_image_flag")
    else:
        p_lower = positive.lower()
        if any(phrase in p_lower for phrase in TEXT_IN_IMAGE_PHRASES):
            negative = TEXT_IN_IMAGE_NEGATIVE
            ctx.trace.append("negative:text_in_image_heuristic")
        else:
            negative = DEFAULT_NEGATIVE_PROMPT
            ctx.trace.append("negative:default")
    if staged_neg and staged_neg != user_neg:
        negative = append_csv(negative, staged_neg)
        ctx.trace.append("negative:staged_args")

    for frag, label in (
        (art.layout_negative, "layout"),
        (art.multi_instance_negative, "multi_instance"),
        (art.detailed_scene_negative, "detailed_scene"),
        (art.visual_design_negative, "visual_design"),
    ):
        if frag:
            negative = append_csv(negative, frag)
            ctx.trace.append(f"negative:staged_{label}")

    if args is not None:
        negative = _append_flag_negatives(negative, args, ANTI_AI_LOOK_NEGATIVE, ANTI_AI_LOOK_NEGATIVE_STRONG)

    # Guidance negatives (computed in guidance stage)
    for part in ctx.metadata.get("guidance_neg_parts") or []:
        if part:
            negative = append_csv(negative, part)

    if ctx.mode != StackMode.TRAINING:
        if art.photo_negative:
            negative = append_csv(negative, art.photo_negative)

        if args is not None and getattr(args, "less_ai", False):
            if str(getattr(args, "anti_ai_pack", "none") or "none") == "none":
                args.anti_ai_pack = "lite"
            if str(getattr(args, "human_media_mode", "none") or "none") == "none":
                args.human_media_mode = "photographic"

        if args is not None and len(getattr(args, "lora", []) or []) > 1:
            try:
                from config.prompts import LORA_STACK_NEGATIVE

                negative = append_csv(negative, LORA_STACK_NEGATIVE)
                ctx.trace.append("negative:lora_stack")
            except ImportError:
                pass

    ctx.negative = negative


def _append_flag_negatives(negative: str, args: Any, anti_ai: str, anti_ai_strong: str) -> str:
    if getattr(args, "naturalize", False):
        neg = anti_ai_strong if getattr(args, "naturalize_deep", False) else anti_ai
        negative = append_csv(negative, neg)
    if getattr(args, "anti_bleed", False):
        try:
            from config.prompts import CONCEPT_BLEEDING_NEGATIVE

            negative = append_csv(negative, CONCEPT_BLEEDING_NEGATIVE)
        except ImportError:
            pass
    if getattr(args, "diversity", False):
        try:
            from config.prompts import FLUX_FACE_DIVERSITY_NEGATIVE

            negative = append_csv(negative, FLUX_FACE_DIVERSITY_NEGATIVE)
        except ImportError:
            pass
    if getattr(args, "anti_artifacts", False):
        try:
            from config.prompts import ARTIFACT_NEGATIVES

            negative = append_csv(negative, ARTIFACT_NEGATIVES)
        except ImportError:
            pass
    if getattr(args, "strong_watermark", False):
        try:
            from config.prompts import WATERMARK_NEGATIVE_STRONG

            negative = append_csv(negative, WATERMARK_NEGATIVE_STRONG)
        except ImportError:
            pass
    return negative


StageFn = Callable[[PromptContext], None]

_DEFAULT_INFERENCE_STAGES: list[StageFn] = []
_PREVIEW_STAGES: list[StageFn] = []
_TRAINING_STAGES: list[StageFn] = []


def _load_stages() -> None:
    global _DEFAULT_INFERENCE_STAGES, _PREVIEW_STAGES, _TRAINING_STAGES
    if _DEFAULT_INFERENCE_STAGES:
        return

    _TRAINING_STAGES = [stage_guidance]
    _PREVIEW_STAGES = [
        stage_intelligence,
        stage_style_genome,
        stage_special_helpers,
        stage_game,
        stage_guidance,
        stage_negative_bootstrap,
        stage_content_controls,
        stage_clauses,
        stage_token_budget,
        stage_neg_filter,
    ]
    _DEFAULT_INFERENCE_STAGES = [
        stage_intelligence,
        stage_style_genome,
        stage_special_helpers,
        stage_game,
        stage_guidance,
        stage_negative_bootstrap,
        stage_content_controls,
        stage_clauses,
        stage_post_enrich,
        stage_token_budget,
        stage_prompt_breakdown,
        stage_neg_filter,
    ]


def run_prompt_stack(ctx: PromptContext, stages: Sequence[StageFn] | None = None) -> PromptResult:
    """
    Run the prompt stack on *ctx* (mutates positive/negative in place).

    Returns :class:`PromptResult` with trace and analysis metadata.
    """
    _load_stages()
    if stages is None:
        if ctx.mode == StackMode.TRAINING:
            stages = _TRAINING_STAGES
        elif ctx.mode == StackMode.PREVIEW:
            stages = _PREVIEW_STAGES
        else:
            stages = _DEFAULT_INFERENCE_STAGES

    ctx.metadata.setdefault("input_positive", ctx.positive)
    for stage in stages:
        stage(ctx)

    analysis = ctx.metadata.get("analysis")
    analysis_dict = None
    if analysis is not None and hasattr(analysis, "__dataclass_fields__"):
        from dataclasses import asdict

        analysis_dict = asdict(analysis)

    return PromptResult(
        positive=ctx.positive,
        negative=ctx.negative,
        t5_positive_hint=ctx.t5_positive_hint,
        trace=list(ctx.trace),
        analysis=analysis_dict,
        resolved_controls=ctx.metadata.get("resolved_controls"),
    )


def _fallback_negative(args: Any, positive: str) -> str:
    """Match ``stage_negative_bootstrap`` when the full stack fails."""
    user = (getattr(args, "negative_prompt", None) or "").strip() if args is not None else ""
    if user:
        return user
    try:
        from config.prompts import DEFAULT_NEGATIVE_PROMPT, TEXT_IN_IMAGE_NEGATIVE, TEXT_IN_IMAGE_PHRASES
    except ImportError:
        return " "
    if args is not None and getattr(args, "text_in_image", False):
        return TEXT_IN_IMAGE_NEGATIVE
    p_lower = (positive or "").lower()
    if any(phrase in p_lower for phrase in TEXT_IN_IMAGE_PHRASES):
        return TEXT_IN_IMAGE_NEGATIVE
    return DEFAULT_NEGATIVE_PROMPT


def apply_sample_prompt_stack(
    args: Any,
    prompt_to_encode: str,
    *,
    character_negative_additions: str = "",
    scene_negative_additions: str = "",
    apply_scale_distortion: bool = False,
    tokenizer: Any = None,
    max_tokens: int = 0,
) -> tuple[str, str]:
    """
    Run the unified prompt stack for ``sample.py`` (guidance -> negative -> controls -> budget -> finalize).

    Mutates *args* (``args.prompt``, ``human_media_mode``, photo post fields, etc.). *tokenizer* and
    *max_tokens* describe the text encoder so the budget stage counts real tokens.
    """
    artifacts = PromptArtifacts(
        layout_negative=str(getattr(args, "_prompt_layout_negative", "") or "").strip(),
        multi_instance_negative=str(getattr(args, "_multi_instance_negative", "") or "").strip(),
        detailed_scene_negative=str(getattr(args, "_detailed_scene_negative", "") or "").strip(),
        visual_design_negative=str(getattr(args, "_visual_design_negative", "") or "").strip(),
        character_negative=(character_negative_additions or "").strip(),
        scene_negative=(scene_negative_additions or "").strip(),
    )

    if args is not None and (
        int(getattr(args, "invent_styles", 0) or 0) > 0 or str(getattr(args, "style_genome_file", "") or "").strip()
    ):
        try:
            from utils.prompt.style import resolve_style_genome_for_args

            resolve_style_genome_for_args(args, prompt_to_encode)
        except Exception as exc:
            print(f"Warning: style genome resolve failed: {exc}", file=sys.stderr)

    ctx = PromptContext(
        positive=prompt_to_encode,
        negative="",
        mode=StackMode.INFERENCE,
        args=args,
        artifacts=artifacts,
        apply_scale_distortion_negative=apply_scale_distortion,
        tokenizer=tokenizer,
        max_tokens=int(max_tokens or 0),
    )

    try:
        result = run_prompt_stack(ctx)
    except Exception as exc:
        print(f"Warning: prompt stack failed: {exc}", file=sys.stderr)
        if os.environ.get("SDX_DEBUG", "").strip():
            traceback.print_exc()
        return prompt_to_encode, _fallback_negative(args, prompt_to_encode)

    args._encode_t5_positive_hint = result.t5_positive_hint or ""
    if result.analysis:
        args._prompt_stack_analysis = result.analysis
    if result.resolved_controls:
        args._prompt_stack_controls = result.resolved_controls
    if os.environ.get("SDX_PROMPT_STACK_TRACE", "").strip():
        print("Prompt stack trace:", " -> ".join(result.trace), file=sys.stderr)

    return result.positive, result.negative
