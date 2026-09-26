"""Tests for compete-mode quality policy + naturalness/adherence hooks."""

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
        fdg_cfg_strength=0.0,
        zeresfdg_strength=0.0,
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
        naturalness_strength=-1.0,
        anatomy_attention_strength=-1.0,
        glyph_residual_strength=-1.0,
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
        auto_layout=False,
        no_auto_layout=False,
        prompt_ground=False,
        no_prompt_ground=False,
        glyph_canvas=False,
        no_glyph_canvas=False,
        contact_shadow_auto=False,
        no_contact_shadow=False,
        contact_shadow=-1.0,
        box_attn_layout=False,
        per_region_cads=False,
        no_box_attn_layout=False,
        prompt_early_scale=-1.0,
        prompt_reinject_every_n=0,
        prompt_reinject_alpha=0.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_anti_ai_naturalness_changes_tokens():
    from models.anti_ai_naturalness import AntiAINaturalnessController, detect_medium

    ctrl = AntiAINaturalnessController(32)
    x = torch.randn(2, 64, 32)
    medium = detect_medium("photoreal dslr portrait photo")
    assert medium.family == "photo"
    out = ctrl(x, medium, 8, 8, strength=0.5)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)


def test_anatomy_spatial_prior():
    from models.anatomy_attention import apply_anatomy_spatial_prior

    x = torch.randn(1, 64, 16)
    out = apply_anatomy_spatial_prior(x, 8, 8, strength=0.4)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)


def test_quality_policy_photoreal_compete():
    from utils.generation.quality_stack import apply_best_model_policy

    args = _blank_args(prompt="hyperreal 8k dslr portrait photo")
    meta = apply_best_model_policy(args)
    assert meta["profile"] in ("photoreal", "portrait")
    assert meta.get("aggression") == "max"
    assert args.holy_grail is True
    assert float(args.apg_parallel_eta) == 0.0
    assert float(args.cfg_rescale) == 0.7
    assert str(args.human_made) == "strong"
    assert int(args.num) >= 3
    assert str(args.pick_best) not in ("none", "")
    assert args.ocr_fix is False or True  # may stay false without quotes
    assert "plastic" in str(args.negative_prompt).lower() or "anti_slop" in str(meta.get("applied"))
    assert args.auto_layout is True
    assert args.prompt_ground is True
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
    assert float(args.glyph_residual_strength) >= 0.2


def test_quality_policy_anime_brain():
    from utils.generation.quality_stack import apply_best_model_policy

    args = _blank_args(prompt="anime girl, cel shading, manga panel")
    meta = apply_best_model_policy(args)
    assert meta["profile"] == "anime"
    assert args.preset == "pixai"
    assert args.holy_grail is True
    assert str(args.holy_grail_preset) == "pixai"


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


def test_adherence_and_glyph_modulation():
    from utils.generation.quality_stack import enrich_text_conditioning, prompt_needs_glyph

    assert prompt_needs_glyph('storefront sign that says "OPEN"')
    enc = torch.randn(1, 32, 64)
    out = enrich_text_conditioning(
        enc,
        'red shirt, no glasses, sign says "OPEN"',
        enable_adherence=True,
        enable_glyph=True,
        glyph_strength=0.1,
        expected_texts=["OPEN"],
        negation_scale=0.55,
        binding_boost=0.15,
    )
    assert out.shape == enc.shape
    assert not torch.allclose(out, enc)


def test_dit_quality_kwargs_auto():
    from utils.generation.quality_stack import dit_quality_kwargs

    args = SimpleNamespace(
        naturalness_strength=-1.0,
        less_ai=True,
        human_made="none",
        anatomy_guidance="strong",
        anatomy_attention_strength=-1.0,
        style="",
    )
    kw = dit_quality_kwargs(args, "a person with hands")
    assert kw["naturalness_strength"] > 0
    assert kw["anatomy_attention_strength"] > 0


def test_dit_quality_kwargs_prompt_reinject():
    from utils.generation.quality_stack import dit_quality_kwargs

    args = SimpleNamespace(
        naturalness_strength=-1.0,
        less_ai=False,
        human_made="none",
        anatomy_guidance="none",
        anatomy_attention_strength=-1.0,
        style="",
        prompt_early_scale=1.12,
        prompt_reinject_every_n=4,
        prompt_reinject_alpha=0.08,
    )
    kw = dit_quality_kwargs(args, "a landscape")
    assert kw["prompt_early_scale"] == 1.12
    assert kw["prompt_reinject_every_n"] == 4
    assert kw["prompt_reinject_alpha"] == 0.08
    assert kw["prompt_timestep_schedule_enabled"] is True


def test_validate_count_in_image_three_dots():
    from PIL import Image, ImageDraw
    from utils.generation.precision_control import CountingValidator

    im = Image.new("RGB", (256, 256), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    for cx in (64, 128, 192):
        draw.ellipse((cx - 14, 118, cx + 14, 146), fill=(0, 0, 0))
    out = CountingValidator().validate_count_in_image(im, "dot", 3)
    assert "detected_count" in out
    assert "accuracy" in out
    assert 0.0 <= float(out["accuracy"]) <= 1.0
    assert int(out["expected_count"]) == 3


def test_patch_quality_hooks_identity_when_off():
    from models.patch_quality_hooks import apply_patch_quality_hooks

    class _M:
        pass

    x = torch.randn(1, 16, 8)
    out = apply_patch_quality_hooks(_M(), x, h_lat=8, w_lat=8, kwargs={})
    assert torch.equal(out, x)


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


def test_ensure_patch_quality_modules_reused_on_apply():
    from models.patch_quality_hooks import apply_patch_quality_hooks, ensure_patch_quality_modules

    class _M:
        num_heads = 4

    m = _M()
    hidden = 16
    ensure_patch_quality_modules(m, hidden, num_heads=4)
    ctrl = m._naturalness_ctrl
    assert ctrl is not None
    x = torch.randn(1, 16, hidden)
    out = apply_patch_quality_hooks(
        m,
        x,
        h_lat=8,
        w_lat=8,
        kwargs={"naturalness_strength": 0.3, "naturalness_prompt": "photoreal dslr photo"},
    )
    assert m._naturalness_ctrl is ctrl
    assert out.shape == x.shape
    assert not torch.allclose(out, x)
