"""Tests for utils.prompt.stack (PromptStack v2)."""

from __future__ import annotations

from types import SimpleNamespace

from utils.prompt.stack import (
    PromptContext,
    StackMode,
    analyze_prompt,
    append_unique,
    apply_clauses,
    list_clauses,
    merge_guidance_for_training_caption,
    resolve_content_controls,
    run_prompt_stack,
    split_tags,
)
from utils.prompt.stack.clauses import CLAUSE_REGISTRY


def test_split_and_append_unique():
    assert split_tags("a, b, a") == ["a", "b", "a"]
    assert "c" in split_tags(append_unique("a, b", ["b", "c"]))


def test_analyze_prompt_complexity():
    short = analyze_prompt("1girl, red dress")
    assert short.complexity == "simple"
    long = analyze_prompt(", ".join([f"tag{i}" for i in range(25)]))
    assert long.complexity in ("complex", "extreme")


def test_clause_registry_uncensored():
    pos, neg = apply_clauses("", "", ["uncensored.fidelity"])
    assert "censorship" in neg.lower() or "censor" in neg.lower()
    assert "faithful" in pos.lower()
    assert "uncensored.fidelity" in CLAUSE_REGISTRY
    assert len(list_clauses()) >= 4


def test_style_chaos_clauses():
    pos, neg = apply_clauses("portrait", "low quality", ["style.chaos", "style.glitch"])
    assert "unforgettable" in pos.lower() or "uncanny" in pos.lower()
    assert "AI slop" in neg or "ai slop" in neg.lower() or "template" in neg.lower()


def test_resolve_content_controls_infer_view():
    args = SimpleNamespace(
        pose_mode="none",
        view_angle="none",
        subject_sex="none",
        scene_domain="none",
        clothing_mode="none",
        background_mode="none",
        people_layout="none",
        relationship_mode="none",
        object_layout="none",
        hand_mode="none",
        pose_naturalness="none",
        typography_mode="none",
        quality_pack="none",
        lighting_mode="none",
        skin_detail_mode="none",
        body_proportion="none",
        interaction_intensity="none",
        style_mode="none",
        composition_mode="none",
        artist_composition="none",
        anti_ai_pack="none",
        human_media_mode="none",
        lora_scaffold="none",
        lora_scaffold_auto=False,
        adherence_pack="none",
        style_lock=False,
        anti_style_bleed=False,
        anti_duplicate_subjects=False,
        anti_perspective_drift=False,
        cleanup_conflicting_tags=False,
        text_in_image=False,
        one_shot_boost=False,
        auto_content_fix=True,
        lora=None,
    )
    state = resolve_content_controls(args, "city street, two point perspective, vanishing point")
    assert state.artist_composition == "perspective"


def test_preview_stack_runs_stages():
    args = SimpleNamespace(
        prompt="1girl, portrait",
        negative_prompt="",
        shortcomings_mitigation="none",
        shortcomings_2d=False,
        art_guidance_mode="none",
        anatomy_guidance="none",
        style_guidance_mode="none",
        style_guidance_artists=True,
        no_art_guidance_photography=False,
        auto_photo_realism=False,
        photo_realism_pack="none",
        photo_color_grade="none",
        photo_lighting_technique="none",
        photo_filter="none",
        photo_grain_style="none",
        photo_realism_strength=1.0,
        realism_autopilot=False,
        quality_pack="top",
        adherence_pack="none",
        auto_content_fix=True,
        prompt_clauses="",
        prompt_stack_intelligence=True,
        prompt_stack_auto_quality=True,
        one_shot_boost=True,
        less_ai=False,
        lora=[],
        text_in_image=False,
        no_neg_filter=False,
        scene_domain="none",
        view_angle="none",
        pose_mode="none",
        style_mode="none",
        clothing_mode="none",
        composition_mode="none",
        artist_composition="none",
        people_layout="none",
        hand_mode="none",
        lighting_mode="none",
        anti_ai_pack="none",
        human_media_mode="none",
        lora_scaffold="none",
        lora_scaffold_auto=False,
    )
    ctx = PromptContext(positive="1girl, portrait", mode=StackMode.PREVIEW, args=args)
    result = run_prompt_stack(ctx)
    assert result.positive
    assert result.negative
    assert "content_controls" in result.trace
    assert any("neg_filter" in step for step in result.trace)


def test_merge_guidance_for_training_caption():
    out = merge_guidance_for_training_caption(
        "concept art environment, matte painting",
        shortcomings_mode="auto",
    )
    assert out.strip()


def test_training_guidance_unified_entry():
    from data.caption_utils import (
        apply_art_guidance_to_caption_pair,
        apply_shortcomings_to_caption_pair,
        apply_style_guidance_to_caption_pair,
        apply_training_guidance_to_caption_pair,
    )

    caption = "1girl, portrait, detailed face, anime style"
    neg = "low quality"
    unified_pos, unified_neg = apply_training_guidance_to_caption_pair(
        caption,
        neg,
        shortcomings_mode="auto",
        art_guidance_mode="auto",
        anatomy_guidance="lite",
        style_guidance_mode="auto",
    )
    step_pos, step_neg = caption, neg
    step_pos, step_neg = apply_shortcomings_to_caption_pair(step_pos, step_neg, mode="auto", include_2d=False)
    step_pos, step_neg = apply_art_guidance_to_caption_pair(
        step_pos,
        step_neg,
        mode="auto",
        include_photography=True,
        anatomy_mode="lite",
    )
    step_pos, step_neg = apply_style_guidance_to_caption_pair(step_pos, step_neg, mode="auto", include_artist_refs=True)
    assert unified_pos.strip() == step_pos.strip()
    assert unified_neg.strip() == step_neg.strip()
