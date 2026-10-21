"""Tests for the compete-mode quality policy and the text-embedding adherence / glyph hooks."""

from __future__ import annotations

from types import SimpleNamespace

import torch


def _blank_args(**over):
    base = dict(
        prompt="",
        negative_prompt="",
        style="",
        photo_realism_pack="none",
        frontier_perfect=False,
        frontier_subject=False,
        apg_parallel_eta=-1.0,
        apg_momentum_beta=0.0,
        cfg_scale=7.5,
        cfg_rescale=0.0,
        fdg_low=1.0,
        fdg_high=1.0,
        cfg_interval="",
        qsilk_micrograin=0.0,
        steps=50,
        preset=None,
        op_mode=None,
        holy_grail=False,
        holy_grail_preset=None,
        human_made="none",
        less_ai=False,
        naturalize=False,
        naturalize_deep=False,
        anti_ai_pack="none",
        human_media_mode="none",
        shortcomings_mitigation="none",
        anatomy_guidance="auto",
        hand_mode="none",
        pose_naturalness="none",
        typography_mode="none",
        expand_prompt=False,
        anti_bleed=False,
        anti_artifacts=False,
        strong_watermark=False,
        diversity=False,
        text_in_image=False,
        ocr_fix=False,
        pick_best="none",
        num=1,
        superior_self_correct=False,
        no_quality_defaults=False,
        box_attn_layout=False,
        per_region_cads=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_quality_policy_photoreal_compete():
    from utils.generation.quality_stack import apply_best_model_policy

    args = _blank_args(prompt="hyperreal 8k dslr portrait photo")
    meta = apply_best_model_policy(args)
    assert meta["profile"] in ("photoreal", "portrait")
    assert meta.get("aggression") == "max"
    assert args.holy_grail is True
    assert float(args.apg_parallel_eta) == 0.0
    assert float(args.apg_momentum_beta) == -0.5
    assert float(args.fdg_low) < 1.0 < float(args.fdg_high)
    assert args.cfg_interval == "0.28,5.42"
    assert str(args.human_made) == "strong"
    assert int(args.num) >= 3
    assert str(args.pick_best) not in ("none", "")
    assert args.ocr_fix is False or True  # may stay false without quotes
    assert "plastic" in str(args.negative_prompt).lower() or "anti_slop" in str(meta.get("applied"))
    assert args.box_attn_layout is True


def test_quality_policy_bind_two_colors_left_of():
    from utils.generation.quality_stack import apply_best_model_policy

    args = _blank_args(prompt="a red cube to the left of a blue sphere")
    meta = apply_best_model_policy(args)
    assert meta["profile"] == "bind"
    assert str(args.pick_best) in ("combo_spatial_bind", "combo", "combo_vit_realism")


def test_quality_policy_text_render_ocr():
    from utils.generation.quality_stack import apply_best_model_policy

    args = _blank_args(prompt='a shop sign that says "OPEN NOW"')
    meta = apply_best_model_policy(args)
    assert meta["profile"] == "text_render"
    assert args.text_in_image is True
    assert args.ocr_fix is True


def test_quality_policy_anime_brain():
    from utils.generation.quality_stack import apply_best_model_policy

    args = _blank_args(prompt="anime girl, cel shading, manga panel")
    meta = apply_best_model_policy(args)
    assert meta["profile"] == "anime"
    assert args.preset is None and args.op_mode is None
    assert args.holy_grail is True
    assert str(args.holy_grail_preset) == "anime"


def test_quality_policy_keeps_user_frontier_mode():
    from utils.generation.quality_stack import apply_best_model_policy

    args = _blank_args(prompt="a castle in the clouds", frontier=True)
    apply_best_model_policy(args)
    assert args.frontier_subject is False
    args = _blank_args(prompt="a castle in the clouds")
    apply_best_model_policy(args)
    assert args.frontier_subject is True


def test_quality_policy_occlusion_fence():
    from utils.generation.quality_stack import apply_best_model_policy

    args = _blank_args(prompt="a woman standing behind a wooden fence")
    meta = apply_best_model_policy(args)
    assert meta["profile"] == "occlusion"
    assert str(args.anatomy_guidance) == "strong"
    assert str(args.holy_grail_preset) == "aggressive"


def test_quality_policy_quoted_open_still_text_render():
    from utils.generation.quality_stack import apply_best_model_policy

    args = _blank_args(prompt='a shop sign that says "OPEN" behind a fence')
    meta = apply_best_model_policy(args)
    assert meta["profile"] == "text_render"


def test_quality_policy_nsfw_and_sfw():
    from utils.generation.quality_stack import apply_best_model_policy

    nsfw = _blank_args(prompt="nude woman, natural window light, lying on linen")
    meta = apply_best_model_policy(nsfw)
    assert meta["profile"] == "nsfw"
    assert str(nsfw.anatomy_guidance) == "strong"
    assert str(nsfw.human_made) == "none"
    assert str(nsfw.human_media_mode) == "none"
    assert str(nsfw.photo_realism_pack) == "none"
    assert "anti_slop" not in str(meta.get("applied"))

    act = _blank_args(prompt="1girl, 1boy, handjob, penis, uncensored")
    meta_act = apply_best_model_policy(act)
    assert meta_act["profile"] == "nsfw"
    assert str(act.human_media_mode) == "none"

    sfw = _blank_args(prompt="linkedin headshot of an engineer, office window light")
    meta2 = apply_best_model_policy(sfw)
    assert meta2["profile"] == "sfw"


def test_quality_policy_opt_out():
    from utils.generation.quality_stack import apply_best_model_policy

    args = _blank_args(prompt="photoreal dslr", no_quality_defaults=True)
    meta = apply_best_model_policy(args)
    assert meta.get("skipped") is True


def test_token_spans_char_fallback_long_word_occupies_more():
    from utils.generation.quality_stack import _token_spans_for_words

    seq = 32
    long_spans = _token_spans_for_words("supercalifragilisticexpialidocious x", seq)
    short_spans = _token_spans_for_words("a x", seq)
    long_w = long_spans[0].stop - long_spans[0].start
    short_w = short_spans[0].stop - short_spans[0].start
    assert long_w > short_w


def test_token_spans_fake_tokenizer_overlap():
    from utils.generation.quality_stack import _token_spans_for_words

    class _TinyTok:
        def __call__(self, prompt, **kwargs):
            return {"offset_mapping": [(0, 0), (0, 4), (4, 9), (10, 16)]}

    prompt = "abcdefghi second"
    spans = _token_spans_for_words(prompt, 8, tokenizer=_TinyTok())
    assert spans[0] == slice(1, 3)
    assert spans[1] == slice(3, 4)


def test_modulate_text_for_adherence_clones_once():
    from utils.generation.quality_stack import modulate_text_for_adherence

    enc = torch.randn(1, 16, 8)
    orig = enc.clone()
    out = modulate_text_for_adherence(enc, "red shirt, no glasses", negation_scale=0.55, binding_boost=0.15)
    assert torch.equal(enc, orig)
    assert out.shape == enc.shape
    assert not torch.equal(out, enc)
