"""Guidance + photo-realism stages (shared with training via merge_guidance_only)."""

from __future__ import annotations

from typing import Any, Tuple

from ..context import PromptContext, StackMode
from ..tokens import append_csv


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
            from config.defaults.ai_image_shortcomings import mitigation_fragments

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
            from config.defaults.art_mediums import guidance_fragments

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
            from config.defaults.style_guidance import style_guidance_fragments

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


def _apply_photo_realism(prompt: str, args: Any) -> Tuple[str, str, dict]:
    try:
        from utils.prompt.photo_realism import (
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
) -> Tuple[str, str]:
    """
    Training-side helper: same guidance fragments as ``sample.py`` / ``t2i_dataset`` (pos + neg).

    Mirrors ``apply_shortcomings_to_caption_pair`` + art + style without duplicating config logic.
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
