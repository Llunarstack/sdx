"""Tests for book challenging-content prompt fragments.

Also holds what used to be: test_book_challenging_content, test_book_helpers, test_book_manifest_utils, test_book_model_readiness, test_book_project, test_book_prompt_intel, test_book_style_authenticity, test_book_style_fusion, test_book_text_continuity, test_book_train_preset, test_book_train_scripts, test_book_training_helpers, test_book_visual_memory, test_consistency_text_continuity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pipelines.book_comic import render, story
from pipelines.book_comic import training as bth
from pipelines.book_comic.render import BookProject, manifest_summary_lines, validate_book_manifest
from pipelines.book_comic.scripts import prepare_and_train_book as prep
from pipelines.book_comic.scripts import train_book_model as tbm
from pipelines.book_comic.story import (
    approximate_token_estimate,
    composed_prompt_length_report,
    find_cast_mentions,
    freeform_style_fusion,
    fusion_fragment_from_preset,
    fusion_from_cli,
    lettering_visual_memory_fragment,
    load_visual_memory,
    peek_visual_memory_book_style,
    positive_block_from_mapping,
    primary_style_from_book_type,
    resolve_authenticity_bundle,
    resolve_effective_medium,
    text_continuity_clause,
)
from training.cli import apply_book_train_preset_to_args

# ==================================================================================================
# test_book_challenging_content: Tests for book challenging-content prompt fragments.
# ==================================================================================================


def test_challenge_pack_positive_mature_coherence() -> None:
    frag = story.challenge_pack_positive("mature_coherence")
    assert "mature-rated" in frag


def test_challenge_pack_positive_max_includes_mature() -> None:
    frag = story.challenge_pack_positive("max")
    assert "mature-rated" in frag
    assert "surreal" in frag


def test_challenging_content_from_mapping() -> None:
    block = {"pack": "surreal_weird", "tags": ["hands_heavy"], "extra": "dream logic"}
    out = story.challenging_content_from_mapping(block)
    assert "surreal" in out
    assert "hands-heavy" in out.lower() or "hands" in out.lower()
    assert "dream logic" in out


def test_positive_block_from_mapping_challenging_content() -> None:
    spec = {
        "challenging_content": {"pack": "technical_hard", "tags": ["reflections_glass"]},
    }
    out = positive_block_from_mapping(spec)
    assert "perspective" in out or "reflection" in out


def test_visual_memory_challenge_clause_adds_mature_fidelity() -> None:
    frag = story.visual_memory_challenge_clause({"book_challenge_pack": "none"})
    assert "mature-rated" in frag


# ==================================================================================================
# test_book_helpers: Tests for pipelines.book_comic.render shortcomings wiring.
# ==================================================================================================


def _args(**overrides):
    base = dict(
        book_accuracy="none",
        sample_candidates=0,
        pick_best="auto",
        no_boost_quality=False,
        boost_quality=False,
        subject_first=False,
        no_subject_first=False,
        save_prompt=False,
        post_sharpen=-1.0,
        post_naturalize=False,
        no_post_naturalize=False,
        post_grain=-1.0,
        post_micro_contrast=-1.0,
        prepend_quality_if_short=False,
        no_prepend_quality_if_short=False,
        shortcomings_mitigation="",
        shortcomings_2d=False,
        no_shortcomings_2d=False,
        art_guidance_mode="",
        art_guidance_photography=False,
        no_art_guidance_photography=False,
        anatomy_guidance="",
        style_guidance_mode="",
        style_guidance_artists=False,
        no_style_guidance_artists=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_production_vit_preset():
    pv = render.preset_for_book_accuracy("production_vit")
    assert pv.pick_best == "combo_vit_hq"
    assert pv.sample_candidates == 6


def test_production_fidelity_preset():
    pf = render.preset_for_book_accuracy("production_fidelity")
    assert pf.pick_best == "combo_vit_hq"
    assert pf.sample_candidates == 8
    assert pf.shortcomings_mitigation == "all"


def test_derive_book_page_seed():
    assert render.derive_book_page_seed(10, 0) == 10
    assert render.derive_book_page_seed(10, 1) == 10 + render.BOOK_PAGE_SEED_STRIDE


def test_audit_book_run_flags_vit_warning():
    _, warns = render.audit_book_run_flags(
        pick_best="combo_vit",
        sample_candidates=2,
        pick_vit_ckpt="",
        beam_width=0,
    )
    assert any("pick-vit-ckpt" in w for w in warns)


def test_audit_book_run_flags_beam_sample_mismatch():
    _, warns = render.audit_book_run_flags(
        pick_best="clip",
        sample_candidates=4,
        beam_width=4,
    )
    assert any("beam" in w.lower() for w in warns)


def test_normalize_book_prompt_fragment():
    assert render.normalize_book_prompt_fragment("  a  \n b , ") == "a b"


def test_extend_sample_py_adapter_control_cmd_lora_and_control():
    from types import SimpleNamespace

    ns = SimpleNamespace(
        style="",
        auto_style_from_prompt=False,
        tags="",
        tags_file="",
        control_image="",
        control_type="auto",
        control_scale=0.85,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        control_guidance_decay=1.0,
        control=["edge.png:canny:0.7"],
        holy_grail=False,
        lora=["a.safetensors:0.6:style"],
        no_lora_normalize_scales=False,
        lora_max_total_scale=1.5,
        lora_default_role="style",
        lora_role_budgets="",
        lora_stage_policy="auto",
        lora_layers="last",
        lora_role_stage_weights="",
        lora_trigger="mytrigger",
        lora_scaffold="none",
        lora_scaffold_auto=False,
        reference_image="",
        reference_strength=1.0,
        reference_tokens=4,
        reference_clip_model="openai/clip-vit-large-patch14",
        reference_adapter_pt="",
    )
    cmd = ["python", "sample.py"]
    render.extend_sample_py_adapter_control_cmd(cmd, ns)
    s = " ".join(cmd)
    assert "--control edge.png:canny:0.7" in s
    assert "--lora a.safetensors:0.6:style" in s
    assert "--lora-layers last" in s
    assert "--lora-trigger mytrigger" in s


def test_adapter_control_argv_for_sample_matches_extend():
    from types import SimpleNamespace

    ns = SimpleNamespace(
        style="ink",
        style_strength=0.5,
        auto_style_from_prompt=False,
        tags="",
        tags_file="",
        control_image="",
        control_type="auto",
        control_scale=0.85,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        control_guidance_decay=1.0,
        control=[],
        holy_grail=False,
        lora=[],
        no_lora_normalize_scales=False,
        lora_max_total_scale=1.5,
        lora_default_role="style",
        lora_role_budgets="",
        lora_stage_policy="auto",
        lora_layers="all",
        lora_role_stage_weights="",
        lora_trigger="",
        lora_scaffold="none",
        lora_scaffold_auto=False,
        reference_image="",
        reference_strength=1.0,
        reference_tokens=4,
        reference_clip_model="openai/clip-vit-large-patch14",
        reference_adapter_pt="",
    )
    frag = render.adapter_control_argv_for_sample(ns)
    assert "--style" in frag
    assert "ink" in frag


def test_extend_sample_py_sdx_enhance_cmd_hires_finishing_and_spectral():
    ns = SimpleNamespace(
        flow_matching_sample=False,
        flow_solver="euler",
        domain_prior_latent=0.0,
        spectral_coherence_latent=0.08,
        spectral_coherence_cutoff=0.12,
        hires_fix=True,
        hires_scale=1.6,
        hires_steps=12,
        hires_strength=0.4,
        hires_cfg_scale=-1.0,
        finishing_preset="illustration",
        sharpen=0.0,
        contrast=1.0,
        saturation=1.0,
        clarity=0.1,
        tone_punch=0.0,
        chroma_smooth=0.0,
        polish=0.0,
        face_enhance=False,
        face_enhance_sharpen=0.35,
        face_enhance_contrast=1.04,
        face_enhance_padding=0.25,
        face_enhance_max=4,
        post_reference_image="",
        post_reference_alpha=0.0,
        face_restore_shell="",
        no_refine=False,
        refine_t=50,
        refine_gate="auto",
        refine_gate_threshold=0.7,
    )
    cmd = ["python", "sample.py"]
    render.extend_sample_py_sdx_enhance_cmd(cmd, ns)
    s = " ".join(cmd)
    assert "--spectral-coherence-latent 0.08" in s
    assert "--spectral-coherence-cutoff 0.12" in s
    assert "--hires-fix" in s
    assert "--hires-scale 1.6" in s
    assert "--hires-steps 12" in s
    assert "--finishing-preset illustration" in s
    assert "--clarity 0.1" in s
    assert "--refine-gate auto" in s
    assert "--refine-gate-threshold 0.7" in s


def test_audit_book_run_flags_clip_and_adherence_hints():
    _, warns = render.audit_book_run_flags(
        pick_best="combo_vit",
        sample_candidates=2,
        pick_vit_ckpt="",
        clip_guard_threshold=0.25,
        clip_monitor_every=3,
        adherence_pack="strict",
        pick_vit_use_adherence=False,
    )
    assert any("clip-guard" in w.lower() for w in warns)
    assert any("clip-monitor" in w.lower() for w in warns)
    assert any("pick-vit-use-adherence" in w for w in warns)


def test_extend_sample_py_adherence_quality_cmd_packs_and_clip():
    ns = SimpleNamespace(
        quality_pack="illustrative",
        adherence_pack="strict",
        clip_guard_threshold=0.24,
        clip_guard_model="openai/clip-vit-base-patch32",
        clip_guard_t_frac=0.2,
        clip_guard_steps=10,
        clip_monitor_every=0,
        clip_monitor_threshold=0.22,
        clip_monitor_cfg_boost=0.12,
        clip_monitor_rewind=0.0,
        volatile_cfg_boost=0.1,
        volatile_cfg_quantile=0.7,
        volatile_cfg_window=8,
        sag_scale=0.15,
        no_auto_expected_text=False,
        no_auto_constraint_boost=False,
        hard_style="realistic",
        dual_stage_layout=True,
        dual_stage_div=2,
        dual_layout_steps=20,
        dual_detail_steps=18,
        dual_detail_strength=0.4,
        deterministic=False,
        no_cache=False,
    )
    cmd = ["python", "sample.py"]
    render.extend_sample_py_adherence_quality_cmd(cmd, ns)
    s = " ".join(cmd)
    assert "--quality-pack illustrative" in s
    assert "--adherence-pack strict" in s
    assert "--clip-guard-threshold 0.24" in s
    assert "--volatile-cfg-boost 0.1" in s
    assert "--sag-scale 0.15" in s
    assert "--hard-style realistic" in s
    assert "--dual-stage-layout" in s
    assert "--dual-layout-steps 20" in s


def test_extend_sample_argv_visual_design_preset_via_book_helpers():
    """extend_sample_py_adherence_quality_cmd delegates visual argv to utils.visual_design."""
    ns = SimpleNamespace(
        quality_pack="none",
        adherence_pack="none",
        clip_guard_threshold=0.0,
        clip_guard_model="openai/clip-vit-base-patch32",
        clip_guard_t_frac=0.22,
        clip_guard_steps=12,
        clip_monitor_every=0,
        clip_monitor_threshold=0.22,
        clip_monitor_cfg_boost=0.12,
        clip_monitor_rewind=0.0,
        volatile_cfg_boost=0.0,
        volatile_cfg_quantile=0.72,
        volatile_cfg_window=6,
        sag_scale=0.0,
        no_auto_expected_text=False,
        no_auto_constraint_boost=False,
        hard_style="",
        dual_stage_layout=False,
        dual_stage_div=2,
        dual_layout_steps=24,
        dual_detail_steps=20,
        dual_detail_strength=0.38,
        deterministic=False,
        no_cache=False,
        visual_design_domain="none",
        visual_design_intensity="standard",
        visual_design_negative_pack=False,
        visual_design_preset="keynote_slide",
    )
    cmd: list[str] = []
    render.extend_sample_py_adherence_quality_cmd(cmd, ns)
    assert cmd == ["--visual-design-preset", "keynote_slide"]


def test_extend_sample_py_adherence_includes_visual_design():
    ns = SimpleNamespace(
        quality_pack="none",
        adherence_pack="none",
        clip_guard_threshold=0.0,
        clip_guard_model="openai/clip-vit-base-patch32",
        clip_guard_t_frac=0.22,
        clip_guard_steps=12,
        clip_monitor_every=0,
        clip_monitor_threshold=0.22,
        clip_monitor_cfg_boost=0.12,
        clip_monitor_rewind=0.0,
        volatile_cfg_boost=0.0,
        volatile_cfg_quantile=0.72,
        volatile_cfg_window=6,
        sag_scale=0.0,
        no_auto_expected_text=False,
        no_auto_constraint_boost=False,
        hard_style="",
        dual_stage_layout=False,
        dual_stage_div=2,
        dual_layout_steps=24,
        dual_detail_steps=20,
        dual_detail_strength=0.38,
        deterministic=False,
        no_cache=False,
        visual_design_domain="stem",
        visual_design_intensity="strong",
        visual_design_negative_pack=True,
    )
    cmd: list[str] = []
    render.extend_sample_py_adherence_quality_cmd(cmd, ns)
    s = " ".join(cmd)
    assert "--visual-design-domain stem" in s
    assert "--visual-design-intensity strong" in s
    assert "--visual-design-negative-pack" in s


def test_adherence_quality_argv_for_sample_matches_extend():
    ns = SimpleNamespace(
        quality_pack="none",
        adherence_pack="none",
        clip_guard_threshold=0.0,
        clip_guard_model="openai/clip-vit-base-patch32",
        clip_guard_t_frac=0.22,
        clip_guard_steps=12,
        clip_monitor_every=2,
        clip_monitor_threshold=0.2,
        clip_monitor_cfg_boost=0.1,
        clip_monitor_rewind=0.05,
        volatile_cfg_boost=0.0,
        volatile_cfg_quantile=0.72,
        volatile_cfg_window=6,
        sag_scale=0.0,
        no_auto_expected_text=True,
        no_auto_constraint_boost=True,
        hard_style="",
        dual_stage_layout=False,
        dual_stage_div=2,
        dual_layout_steps=24,
        dual_detail_steps=20,
        dual_detail_strength=0.38,
        deterministic=True,
        no_cache=True,
    )
    cmd = ["x"]
    render.extend_sample_py_adherence_quality_cmd(cmd, ns)
    frag = render.adherence_quality_argv_for_sample(ns)
    assert cmd[1:] == frag


def test_sdx_enhance_argv_for_sample_matches_extend():
    ns = SimpleNamespace(
        flow_matching_sample=True,
        flow_solver="heun",
        domain_prior_latent=0.0,
        spectral_coherence_latent=0.0,
        spectral_coherence_cutoff=0.15,
        hires_fix=False,
        hires_scale=1.5,
        hires_steps=15,
        hires_strength=0.35,
        hires_cfg_scale=-1.0,
        finishing_preset="none",
        sharpen=0.0,
        contrast=1.0,
        saturation=1.0,
        clarity=0.0,
        tone_punch=0.0,
        chroma_smooth=0.0,
        polish=0.0,
        face_enhance=False,
        face_enhance_sharpen=0.35,
        face_enhance_contrast=1.04,
        face_enhance_padding=0.25,
        face_enhance_max=4,
        post_reference_image="",
        post_reference_alpha=0.0,
        face_restore_shell="",
        no_refine=True,
        refine_t=50,
        refine_gate="off",
        refine_gate_threshold=0.62,
    )
    cmd = ["x"]
    render.extend_sample_py_sdx_enhance_cmd(cmd, ns)
    frag = render.sdx_enhance_argv_for_sample(ns)
    assert cmd[1:] == frag


def test_preset_defaults_for_book_accuracy():
    fast = render.preset_for_book_accuracy("fast")
    assert fast.shortcomings_mitigation == "none"
    assert fast.shortcomings_2d is False
    balanced = render.preset_for_book_accuracy("balanced")
    assert balanced.shortcomings_mitigation == "auto"
    assert balanced.shortcomings_2d is True
    assert balanced.art_guidance_mode == "auto"
    assert balanced.anatomy_guidance == "lite"
    assert balanced.style_guidance_mode == "auto"


def test_resolve_overrides_shortcomings():
    settings = render.resolve_book_sample_settings(
        _args(
            book_accuracy="balanced",
            shortcomings_mitigation="all",
            no_shortcomings_2d=True,
        )
    )
    assert settings.shortcomings_mitigation == "all"
    assert settings.shortcomings_2d is False


def test_append_sample_py_quality_flags_includes_shortcomings():
    settings = render.resolve_book_sample_settings(_args(book_accuracy="balanced"))
    cmd = ["python", "sample.py"]
    render.append_sample_py_quality_flags(cmd, settings, pick_expected_text="")
    assert "--shortcomings-mitigation" in cmd
    assert "--shortcomings-2d" in cmd
    assert "--art-guidance-mode" in cmd
    assert "--anatomy-guidance" in cmd
    assert "--style-guidance-mode" in cmd


def test_append_sample_py_quality_flags_forwards_vit_pick_and_ar():
    settings = render.resolve_book_sample_settings(
        _args(book_accuracy="none", sample_candidates=2, pick_best="combo_vit_hq")
    )
    cmd = ["python", "sample.py"]
    render.append_sample_py_quality_flags(
        cmd,
        settings,
        pick_expected_text="",
        pick_vit_ckpt=r"C:\models\vit\best.pt",
        pick_vit_use_adherence=True,
        pick_vit_ar_blocks=2,
        pick_report_json=r"C:\tmp\run_pick.json",
        pick_auto_no_clip=True,
    )
    joined = " ".join(cmd)
    assert "--pick-best combo_vit_hq" in joined
    assert "--pick-vit-ckpt" in joined
    assert "best.pt" in joined
    assert "--pick-vit-use-adherence" in joined
    assert "--pick-vit-ar-blocks 2" in joined
    assert "--pick-report-json" in joined
    assert "run_pick.json" in joined
    assert "--pick-auto-no-clip" in joined


def test_append_sample_py_beam_flags():
    cmd = ["python", "sample.py"]
    render.append_sample_py_beam_flags(
        cmd,
        beam_width=4,
        beam_steps=8,
        beam_metric="combo_vit",
        beam2_width=3,
        beam2_steps=5,
        beam2_metric="edge",
        beam2_at_frac=0.7,
        beam2_noise=0.02,
    )
    s = " ".join(cmd)
    assert "--beam-width 4" in s
    assert "--beam-steps 8" in s
    assert "--beam-metric combo_vit" in s
    assert "--beam2-width 3" in s
    assert "--beam2-steps 5" in s
    assert "--beam2-metric edge" in s
    assert "--beam2-at-frac 0.7" in s
    assert "--beam2-noise 0.02" in s


def test_append_sample_py_quality_flags_combo_vit_realism_forwards_expected_text():
    settings = render.resolve_book_sample_settings(
        _args(book_accuracy="none", sample_candidates=2, pick_best="combo_vit_realism")
    )
    cmd = ["python", "sample.py"]
    render.append_sample_py_quality_flags(cmd, settings, pick_expected_text="BONJOUR")
    assert "--expected-text BONJOUR" in " ".join(cmd)


def test_append_sample_py_quality_flags_combo_count_forwards_count_args():
    settings = render.resolve_book_sample_settings(
        _args(book_accuracy="none", sample_candidates=2, pick_best="combo_count")
    )
    cmd = ["python", "sample.py"]
    render.append_sample_py_quality_flags(
        cmd,
        settings,
        pick_expected_text="",
        pick_expected_count=5,
        pick_expected_count_target="objects",
        pick_expected_count_object="coin",
    )
    joined = " ".join(cmd)
    assert "--pick-best combo_count" in joined
    assert "--expected-count 5" in joined
    assert "--expected-count-target objects" in joined
    assert "--expected-count-object coin" in joined


def test_compose_book_page_prompt_order():
    s = render.compose_book_page_prompt(
        user_prompt="page scene",
        narration_prefix="nar",
        consistency_block="cons",
        style_fusion_block="fuse",
        user_style_fragment="userpaint",
        panel_hint="panel",
        rolling_context="roll",
    )
    assert s.index("nar") < s.index("cons") < s.index("fuse") < s.index("userpaint")
    assert s.index("userpaint") < s.index("panel") < s.index("roll") < s.index("page scene")


def test_expand_page_prompt_template():
    s = render.expand_page_prompt_template(
        "p{page1} of {total_pages} idx{page0} last{total_pages0} {chapter}",
        page_index=2,
        total_pages=10,
        chapter="3",
    )
    assert "p3 of 10" in s
    assert "idx2" in s
    assert "last9" in s
    assert s.endswith(" 3")


def test_build_extra_ocr_flags_includes_shortcomings():
    settings = render.resolve_book_sample_settings(_args(book_accuracy="production"))
    flags = render.build_extra_ocr_sample_flags(settings)
    assert "--shortcomings-mitigation" in flags
    assert "all" in flags
    assert "--shortcomings-2d" in flags
    assert "--art-guidance-mode" in flags
    assert "--anatomy-guidance" in flags
    assert "--style-guidance-mode" in flags


# ==================================================================================================
# test_book_manifest_utils: Tests for ``pipelines.book_comic.render``.
# ==================================================================================================


def test_validate_book_manifest_ok_minimal() -> None:
    m = {
        "ckpt": "models/x.pt",
        "entries": [{"kind": "page", "index": 0, "path": "pages/p000.png", "prompt": "a", "seed": 1}],
    }
    errs, warns = validate_book_manifest(m)
    assert not errs


def test_validate_book_manifest_missing_ckpt() -> None:
    errs, _ = validate_book_manifest({"entries": []})
    assert any("ckpt" in e.lower() for e in errs)


def test_validate_vit_pick_without_ckpt_warns() -> None:
    m = {
        "ckpt": "z.pt",
        "pick_best": "combo_vit_hq",
        "entries": [],
    }
    _, warns = validate_book_manifest(m)
    assert any("pick_vit" in w.lower() or "vit" in w.lower() for w in warns)


def test_validate_book_manifest_check_files(tmp_path) -> None:
    img = tmp_path / "pages" / "page_000.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    m = {
        "ckpt": "z.pt",
        "entries": [{"kind": "page", "index": 0, "path": "pages/page_000.png", "prompt": "a"}],
    }
    from pipelines.book_comic.render import validate_book_manifest

    errs, _ = validate_book_manifest(m, project_root=tmp_path, check_files=True)
    assert not errs


def test_validate_book_manifest_missing_file(tmp_path) -> None:
    m = {
        "ckpt": "z.pt",
        "entries": [{"kind": "page", "index": 0, "path": "pages/missing.png"}],
    }
    from pipelines.book_comic.render import validate_book_manifest

    errs, _ = validate_book_manifest(m, project_root=tmp_path, check_files=True)
    assert errs


def test_manifest_summary_includes_pick() -> None:
    lines = manifest_summary_lines({"pick_best": "combo", "sample_candidates": 2, "entries": [], "ckpt": "a.pt"})
    assert any("pick_best=combo" in x for x in lines)
    assert any("sample_candidates=2" in x for x in lines)


# ==================================================================================================
# test_book_model_readiness: Tests for ``pipelines.book_comic.render`` and ViT peek helper.
# ==================================================================================================


def test_peek_vit_quality_config(tmp_path: Path) -> None:
    from vit_quality.model import peek_vit_quality_config

    p = tmp_path / "vq.pt"
    torch.save({"config": {"use_ar_conditioning": True, "image_size": 384}}, p)
    cfg = peek_vit_quality_config(p)
    assert cfg.get("use_ar_conditioning") is True
    assert int(cfg.get("image_size", 0)) == 384


def test_run_book_preflight_off() -> None:

    ns = SimpleNamespace(ckpt="__no_such_ckpt__", pick_vit_ckpt="", lora=[], control=[])
    e, w = render.run_book_preflight(ns, mode="off")
    assert e == [] and w == []


def test_run_book_preflight_strict_missing_ckpt(tmp_path: Path) -> None:

    ns = SimpleNamespace(
        ckpt=str(tmp_path / "nope.pt"),
        pick_vit_ckpt="",
        lora=[],
        control=[],
        control_image="",
        reference_image="",
        reference_adapter_pt="",
        tags_file="",
        character_sheet="",
        visual_memory="",
        consistency_json="",
    )
    errs, _ = render.run_book_preflight(ns, mode="strict", resolved_pick_best="clip")
    assert errs


def test_collect_dual_model_ar_mismatch() -> None:

    ns = SimpleNamespace(pick_vit_ckpt="x.pt", pick_vit_ar_blocks=4, pick_vit_ar_from_ckpt=False)
    w = render.collect_dual_model_alignment_warnings(
        ns,
        dit_ar_blocks=2,
        vit_cfg={"use_ar_conditioning": True, "image_size": 224},
    )
    assert any("num_ar_blocks" in x or "2" in x for x in w)


def test_book_model_stack_snapshot() -> None:

    ns = SimpleNamespace(
        ckpt="c.pt",
        pick_vit_ckpt="",
        lora=["a.safetensors:1"],
        control=[],
        pick_vit_ar_blocks=-1,
        pick_vit_ar_from_ckpt=False,
    )
    snap = render.book_model_stack_snapshot(ns, dit_ar_blocks=0, vit_cfg={})
    assert snap["dit_num_ar_blocks"] == 0
    assert snap["lora_count"] == 1


def test_missing_path_errors_parse_drive_letter_specs(tmp_path: Path) -> None:
    r"""``C:\...:0.8`` specs must keep the drive letter (regression: split on the first colon)."""

    weights = tmp_path / "style.safetensors"
    weights.write_bytes(b"x")
    ckpt = tmp_path / "dit.pt"
    ckpt.write_bytes(b"x")
    ns = SimpleNamespace(ckpt=str(ckpt), lora=[f"{weights}:0.8:style"], control=[f"{weights}:canny:0.7"])
    assert render.collect_missing_path_errors(ns) == []


def test_pick_vit_ar_blocks_zero_is_a_real_regime() -> None:
    """``--pick-vit-ar-blocks 0`` must not be treated as unset (regression: ``int(x or -1)``)."""

    ns = SimpleNamespace(pick_vit_ckpt="x.pt", pick_vit_ar_blocks=0, pick_vit_ar_from_ckpt=False)
    w = render.collect_dual_model_alignment_warnings(ns, dit_ar_blocks=2, vit_cfg={"use_ar_conditioning": True})
    assert not any("is unset" in x for x in w)
    assert any("resolves to 0" in x for x in w)
    snap = render.book_model_stack_snapshot(ns, vit_cfg={})
    assert snap["pick_vit_ar_blocks_cli"] == 0


# ==================================================================================================
# test_book_project: Tests for ``pipelines.book_comic.render`` layout.
# ==================================================================================================


def test_book_project_layout_and_flush(tmp_path: Path) -> None:
    proj = BookProject.open(tmp_path / "my_book", create=True)
    proj.title = "Test Vol 1"
    proj.ckpt = "results/run/best.pt"
    front = proj.cover_path("front")
    back = proj.cover_path("back")
    page0 = proj.page_path(0)
    front.write_bytes(b"png")
    back.write_bytes(b"png")
    page0.write_bytes(b"png")
    proj.add_entry({"kind": "front_cover", "path": proj.rel(front), "prompt": "front"})
    proj.add_entry({"kind": "back_cover", "path": proj.rel(back), "prompt": "back"})
    proj.add_entry({"kind": "page", "index": 0, "path": proj.rel(page0), "prompt": "p0"})
    proj.flush()
    proj.sync_legacy_front_cover(front)
    assert (tmp_path / "my_book" / "book.json").is_file()
    assert (tmp_path / "my_book" / "covers" / "front.png").is_file()
    assert (tmp_path / "my_book" / "covers" / "back.png").is_file()
    assert (tmp_path / "my_book" / "cover" / "cover.png").is_file()
    loaded = BookProject.open(tmp_path / "my_book", create=False)
    assert loaded.title == "Test Vol 1"
    assert len(loaded.entries) == 3


def test_book_project_open_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "book"
    BookProject.open(root, create=True).flush()
    again = BookProject.open(root, create=False)
    assert again.root == root.resolve()


# ==================================================================================================
# test_book_prompt_intel: Tests for pipelines.book_comic.story.
# ==================================================================================================


def test_approximate_token_estimate_empty():
    assert approximate_token_estimate("") == 0


def test_find_cast_mentions():
    r = find_cast_mentions("Ren talks to Kai near the bike", ["Ren", "Kai", "Yuki"])
    assert "Ren" in r.found and "Kai" in r.found
    assert "Yuki" in r.missing
    assert "Yuki" in r.soft_reminder_fragment()


def test_length_report_ignores_empty_blocks_in_token_estimate():
    prompt = "a quiet rooftop at dusk, two figures"
    rep = composed_prompt_length_report(user_prompt=prompt)
    assert rep["total_chars"] == len(prompt)
    assert rep["approx_tokens"] == approximate_token_estimate(prompt)


# ==================================================================================================
# test_book_style_authenticity: Tests for pipelines.book_comic.story.
# ==================================================================================================


def test_peek_visual_memory_book_style(tmp_path: Path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"version": 1, "book_style": "webtoon", "entities": {}}), encoding="utf-8")
    assert peek_visual_memory_book_style(p) == "webtoon"


def test_resolve_effective_medium_from_book_type():
    assert resolve_effective_medium(medium="auto", book_type="comic") == "comic_us"
    assert resolve_effective_medium(medium="auto", book_type="novel_cover") == "illustration"


def test_resolve_authenticity_bundle_none():
    b = resolve_authenticity_bundle(level="none", medium="manga", book_type="manga")
    assert b["positive"] == "" and b["negative"] == ""


def test_resolve_authenticity_bundle_manga_strong():
    b = resolve_authenticity_bundle(level="strong", medium="manga", book_type="manga")
    assert "manga" in b["positive"].lower() or "ink" in b["positive"].lower()
    assert b["effective_medium"] == "manga"
    assert "synthetic" in b["negative"].lower() or "ai" in b["negative"].lower()


def test_peek_visual_memory_book_style_non_utf8_returns_empty(tmp_path: Path):
    p = tmp_path / "vm_latin1.json"
    p.write_bytes('{"book_style": "café"}'.encode("latin-1"))
    assert peek_visual_memory_book_style(p) == ""


# ==================================================================================================
# test_book_style_fusion: Tests for pipelines.book_comic.story.
# ==================================================================================================


def test_primary_style_from_book_type():
    assert primary_style_from_book_type("comic") == "comic_us"
    assert primary_style_from_book_type("novel_cover") == "illustration"


def test_fusion_preset_manga_comic():
    s = fusion_fragment_from_preset("manga_comic")
    assert "manga" in s.lower() or "comic" in s.lower() or "hybrid" in s.lower()


def test_fusion_from_cli_secondary_only():
    s = fusion_from_cli(preset="none", secondary="comic_us", primary_book_style="manga")
    assert s
    assert "primary" in s.lower() or "manga" in s.lower()


def test_freeform_empty_secondary():
    assert freeform_style_fusion("manga", "") == ""


# ==================================================================================================
# test_book_text_continuity: Tests for pipelines.book_comic.story.
# ==================================================================================================


def test_lettering_visual_memory_fragment():
    s = lettering_visual_memory_fragment(
        {
            "balloon_style": "rounded tails",
            "sfx_style": "brush impact",
            "match_quoted_script": True,
        }
    )
    assert "balloon" in s.lower()
    assert "script" in s.lower()


def test_text_continuity_clause():
    s = text_continuity_clause(
        {
            "strict_script": True,
            "locked_phrases": ["I am the storm", "Chapter 7"],
            "object_labels": [{"id": "blade", "label": "VOID"}],
        }
    )
    assert "storm" in s.lower()
    assert "void" in s.lower()


# ==================================================================================================
# test_book_train_preset: Book training preset wiring for ``train.py``.
# ==================================================================================================


def test_apply_book_train_preset_sets_model() -> None:
    args = argparse.Namespace(
        book_train_preset="fast",
        book_ar_profile="auto",
        model="",
        image_size=0,
        global_batch_size=0,
        lr=0.0,
        passes=-1,
        max_steps=-1,
        ar_profile="",
        num_ar_blocks=-1,
        ar_block_order="",
    )
    apply_book_train_preset_to_args(args)
    assert args.model == "DiT-B/2-Text"
    assert args.use_hierarchical_captions is True
    assert args.boost_adherence_caption is True


def test_apply_book_train_preset_off_is_noop() -> None:
    args = argparse.Namespace(book_train_preset="", model="DiT-XL/2-Text")
    apply_book_train_preset_to_args(args)
    assert args.model == "DiT-XL/2-Text"


# ==================================================================================================
# test_book_train_scripts: Book training launchers: prepare_and_train_book forwards options the trainer wrapper parses back identically.
# ==================================================================================================

_NOT_FORWARDED = {"data_path", "manifest_jsonl", "max_steps"}


def test_split_passthrough():
    assert tbm.split_passthrough(["--a", "--", "--b", "--"]) == (["--a"], ["--b", "--"])
    assert tbm.split_passthrough(["--a"]) == (["--a"], [])


@pytest.mark.parametrize(
    "argv",
    [
        [],
        [
            "--model", "M", "--image-size", "640", "--global-batch-size", "8", "--lr", "1e-4", "--passes", "0",
            "--max-steps-train", "0", "--ar-profile", "comic_snake", "--num-ar-blocks", "0", "--ar-block-order", "zorder",
            "--ar-curriculum-mode", "linear", "--ar-curriculum-warmup-steps", "5", "--ar-curriculum-ramp-start", "1",
            "--ar-curriculum-ramp-end", "9", "--ar-curriculum-start-blocks", "0", "--ar-curriculum-target-blocks", "4",
            "--ar-order-mix", "raster,snake", "--num-workers", "0", "--dry-run", "--no-compile", "--no-xformers",
            "--native-preflight", "--strict-native", "--manifest-min-caption-len", "5",
            "--manifest-max-caption-len", "500", "--init-from", "base.pt",
        ],
    ],
)  # fmt: skip
def test_trainer_command_round_trips(argv):
    args = prep._build_parser().parse_args(argv)
    cmd = prep._trainer_command(args, Path("m.jsonl"), "data", ["--extra", "1"])
    own, passthrough = tbm.split_passthrough(cmd[2:])
    parsed = tbm._build_parser().parse_args(own)

    assert passthrough == ["--extra", "1"]
    assert parsed.max_steps == args.max_steps_train
    for dest, value in vars(parsed).items():
        if dest not in _NOT_FORWARDED and not dest.startswith("manifest_gate"):
            assert value == getattr(args, dest), dest


def test_help_is_cp1252_safe():
    for parser in (prep._build_parser(), tbm._build_parser()):
        parser.format_help().encode("cp1252")


# ==================================================================================================
# test_book_training_helpers: Tests for pipelines.book_comic.training.
# ==================================================================================================


def _test_book_training_helpers_args(**overrides):
    base = dict(
        book_train_preset="balanced",
        data_path="data/train",
        manifest_jsonl="",
        results_dir="results/book_train",
        model="",
        image_size=0,
        global_batch_size=0,
        lr=0.0,
        passes=-1,
        max_steps=-1,
        ar_profile="auto",
        num_ar_blocks=-1,
        ar_block_order="",
        ar_curriculum_mode="none",
        ar_curriculum_warmup_steps=0,
        ar_curriculum_ramp_start=0,
        ar_curriculum_ramp_end=0,
        ar_curriculum_start_blocks=-1,
        ar_curriculum_target_blocks=-1,
        ar_order_mix="",
        dry_run=False,
        no_compile=False,
        no_xformers=False,
        num_workers=-1,
        resume="",
        init_from="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_preset_for_book_train_production():
    p = bth.preset_for_book_train("production")
    assert p.train_shortcomings_mitigation == "all"
    assert p.train_art_guidance_mode == "all"
    assert p.train_anatomy_guidance == "strong"
    assert p.use_hierarchical_captions is True


def test_build_train_command_init_from_forwarded():
    args = _test_book_training_helpers_args(manifest_jsonl="data/m.jsonl", init_from="results/sdx/best.pt")
    settings = bth.resolve_book_train_settings(args)
    cmd = bth.build_train_command(
        root=Path("c:/repo"),
        python_exe="python",
        args=args,
        settings=settings,
        passthrough_train_args=[],
    )
    i = cmd.index("--init-from")
    assert cmd[i + 1] == "results/sdx/best.pt"


def test_build_train_command_rejects_resume_and_init_from():
    args = _test_book_training_helpers_args(manifest_jsonl="data/m.jsonl", resume="a.pt", init_from="b.pt")
    settings = bth.resolve_book_train_settings(args)
    try:
        bth.build_train_command(
            root=Path("c:/repo"),
            python_exe="python",
            args=args,
            settings=settings,
            passthrough_train_args=[],
        )
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "not both" in str(e)


def test_build_train_command_includes_book_guidance():
    args = _test_book_training_helpers_args(
        book_train_preset="balanced",
        manifest_jsonl="data/manifest.jsonl",
        dry_run=True,
        ar_curriculum_mode="linear",
        ar_curriculum_ramp_start=1000,
        ar_curriculum_ramp_end=5000,
        ar_order_mix="raster,zorder",
    )
    settings = bth.resolve_book_train_settings(args)
    cmd = bth.build_train_command(
        root=Path("c:/repo"),
        python_exe="python",
        args=args,
        settings=settings,
        passthrough_train_args=["--seed", "123"],
    )
    joined = " ".join(cmd)
    assert "--train-shortcomings-mitigation auto" in joined
    assert "--train-art-guidance-mode auto" in joined
    assert "--train-style-guidance-mode auto" in joined
    assert "--use-hierarchical-captions" in cmd
    assert "--num-ar-blocks" in cmd
    assert "--ar-block-order" in cmd
    assert "--ar-curriculum-mode" in cmd
    assert "--ar-order-mix" in cmd
    assert "--dry-run" in cmd
    assert "--seed" in cmd


def test_native_manifest_preflight_without_rust_tools(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"image_path":"a.png","caption":"hello"}\n', encoding="utf-8")

    monkeypatch.setattr(
        bth, "run_rust_jsonl_validate", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no rust"))
    )
    monkeypatch.setattr(
        bth, "run_rust_jsonl_stats", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no rust"))
    )

    info = bth.run_native_manifest_preflight(manifest)
    assert info["exists"] is True
    assert "fnv1a64=" in info["fingerprint"] or "linecrc" in info["fingerprint"]
    assert info["rust_validate_ok"] is None
    assert info["rust_stats_ok"] is None


def test_build_hf_export_command():
    cmd = bth.build_hf_export_command(
        root=Path("c:/repo"),
        python_exe="python",
        dataset="org/bookset",
        out_dir=Path("data/out"),
        manifest_name="m.jsonl",
        max_samples=123,
        streaming=True,
        shuffle_seed=7,
    )
    joined = " ".join(cmd)
    assert "manifest.py hf-export" in joined
    assert "--dataset org/bookset" in joined
    assert "--manifest-name m.jsonl" in joined
    assert "--max-samples 123" in joined
    assert "--streaming" in cmd
    assert "--shuffle-seed" in cmd


def test_build_hf_export_command_targets_existing_script():
    repo = Path(__file__).resolve().parents[1]
    cmd = bth.build_hf_export_command(root=repo, python_exe="python", dataset="org/x", out_dir=Path("out"))
    assert Path(cmd[1]).is_file(), cmd[1]


def test_build_caption_normalize_command():
    cmd = bth.build_caption_normalize_command(
        root=Path("c:/repo"),
        python_exe="python",
        inp_manifest=Path("in.jsonl"),
        out_manifest=Path("out.jsonl"),
        shortcomings_mitigation="all",
        shortcomings_2d=True,
        art_guidance_mode="auto",
        art_guidance_photography=False,
        anatomy_guidance="strong",
        style_guidance_mode="all",
        style_guidance_artists=False,
    )
    joined = " ".join(cmd)
    assert "captions.py normalize" in joined
    assert "--shortcomings-mitigation all" in joined
    assert "--shortcomings-2d" in cmd
    assert "--no-art-guidance-photography" in cmd
    assert "--style-guidance-mode all" in joined
    assert "--no-style-guidance-artists" in cmd


def test_resolve_train_humanization_pack_strong():
    out = bth.resolve_train_humanization_pack("strong")
    assert out["shortcomings_mitigation"] == "all"
    assert out["anatomy_guidance"] == "strong"
    assert out["style_guidance_mode"] == "all"


def test_resolve_book_ar_profile_and_explicit_overrides():
    auto = bth.resolve_book_ar_profile("auto")
    assert auto == {}
    z = bth.resolve_book_ar_profile("zorder")
    assert z["num_ar_blocks"] == 2
    assert z["ar_block_order"] == "zorder"
    vit = bth.resolve_book_ar_profile("vit_strong")
    assert vit["num_ar_blocks"] == 4
    assert vit["ar_block_order"] == "zorder"
    snake = bth.resolve_book_ar_profile("comic_snake")
    assert snake["ar_block_order"] == "snake"

    settings = bth.resolve_book_train_settings(
        _test_book_training_helpers_args(ar_profile="layout", num_ar_blocks=4, ar_block_order="spiral")
    )
    assert settings.num_ar_blocks == 4
    assert settings.ar_block_order == "spiral"


def test_build_train_command_forwards_zero_curriculum_blocks():
    """0 is a real curriculum block count; only -1 means unset (regression: ``int(x or -1)``)."""
    args = _test_book_training_helpers_args(
        manifest_jsonl="m.jsonl", ar_curriculum_start_blocks=0, ar_curriculum_target_blocks=0
    )
    cmd = bth.build_train_command(
        root=Path("c:/repo"),
        python_exe="python",
        args=args,
        settings=bth.resolve_book_train_settings(args),
        passthrough_train_args=[],
    )
    assert cmd[cmd.index("--ar-curriculum-start-blocks") + 1] == "0"
    assert cmd[cmd.index("--ar-curriculum-target-blocks") + 1] == "0"
    unset = bth.build_train_command(
        root=Path("c:/repo"),
        python_exe="python",
        args=_test_book_training_helpers_args(manifest_jsonl="m.jsonl"),
        settings=bth.resolve_book_train_settings(_test_book_training_helpers_args()),
        passthrough_train_args=[],
    )
    assert "--ar-curriculum-target-blocks" not in unset


# ==================================================================================================
# test_book_visual_memory: Tests for ``pipelines.book_comic.story``.
# ==================================================================================================


def test_load_and_entity_override_prompt(tmp_path: Path) -> None:
    data = {
        "version": 1,
        "book_style": "webtoon",
        "entities": {
            "a": {
                "kind": "character",
                "display_name": "A",
                "canonical_look": "blue hair",
                "structure": {"default_viewing_angle": "eye level"},
                "page_overrides": [
                    {
                        "from_page": 1,
                        "to_page": 1,
                        "patch": {"structure": {"default_viewing_angle": "dramatic low angle"}},
                    }
                ],
            }
        },
        "page_patches": [{"from_page": 0, "to_page": 0, "extra_prompt": "establishing wide shot"}],
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    mem = load_visual_memory(p)

    p0 = mem.prompt_fragment_for_page(0)
    assert "establishing wide shot" in p0
    assert "eye level" in p0

    p1 = mem.prompt_fragment_for_page(1)
    assert "dramatic low angle" in p1
    assert "blue hair" in p1


def test_cover_skips_page_overrides(tmp_path: Path) -> None:
    data = {
        "version": 1,
        "entities": {
            "a": {
                "display_name": "A",
                "canonical_look": "round glasses",
                "structure": {"default_viewing_angle": "bird's eye"},
                "page_overrides": [
                    {"from_page": 0, "to_page": 99, "patch": {"canonical_look": "no glasses"}},
                ],
            }
        },
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    mem = load_visual_memory(p)
    cov = mem.prompt_fragment_for_cover()
    assert "round glasses" in cov
    assert "no glasses" not in cov


def test_entities_list_form(tmp_path: Path) -> None:
    data = {
        "version": 1,
        "entities": [{"id": "x", "kind": "prop", "display_name": "Sword", "canonical_look": "curved saber"}],
    }
    p = tmp_path / "l.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    mem = load_visual_memory(p)
    assert "x" in mem.entity_ids()
    assert "saber" in mem.prompt_fragment_for_page(0).lower()


def test_apply_entity_page_patch_and_save(tmp_path: Path) -> None:
    p = tmp_path / "mut.json"
    p.write_text(json.dumps({"version": 1, "entities": {}}), encoding="utf-8")
    mem = load_visual_memory(p)
    mem.apply_entity_page_patch("new_char", from_page=3, to_page=10, patch={"canonical_look": "red cloak"})
    out = tmp_path / "out.json"
    mem.save(out)
    mem2 = load_visual_memory(out)
    assert "new_char" in mem2.entity_ids()
    assert "red cloak" in mem2.prompt_fragment_for_page(5)


def test_effective_entity_keyerror(tmp_path: Path) -> None:
    p = tmp_path / "empty_ent.json"
    p.write_text(json.dumps({"version": 1, "entities": {}}), encoding="utf-8")
    m = load_visual_memory(p)
    with pytest.raises(KeyError):
        m.effective_entity("missing", 0)


def test_visual_memory_lettering_and_style_mix(tmp_path: Path) -> None:
    p = tmp_path / "mix.json"
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "book_style": "manga",
                "lettering": {"balloon_style": "oval", "typography_mood": "hand ink"},
                "style_mix": {"preset": "manga_comic"},
                "user_style_anchor": "pastel wash under ink",
                "entities": {},
            }
        ),
        encoding="utf-8",
    )
    mem = load_visual_memory(p)
    frag = mem.prompt_fragment_for_page(0)
    assert "oval" in frag.lower() or "balloon" in frag.lower()
    assert "hybrid" in frag.lower() or "manga" in frag.lower()
    assert "pastel" in frag.lower()


def test_invalid_version(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"version": 99, "entities": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_visual_memory(p)


def test_null_fields_do_not_render_none(tmp_path: Path) -> None:
    p = tmp_path / "nulls.json"
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "user_style_anchor": None,
                "entities": [
                    {"id": None, "canonical_look": "skipped"},
                    {"id": "a", "canonical_look": None, "notes": None, "structure": {"extra": None}},
                ],
            }
        ),
        encoding="utf-8",
    )
    mem = load_visual_memory(p)
    assert mem.entity_ids() == ["a"]
    assert "None" not in mem.prompt_fragment_for_page(0)
    assert "None" not in mem.prompt_fragment_for_cover()


def test_book_scene_split_pages():
    from scripts.tools.book import split_into_page_prompts

    raw = "intro\n## Page 1\nhero runs\n\n---PAGE---\n\n### page 2\nvillain laughs\n"
    assert split_into_page_prompts(raw) == ["intro", "hero runs", "villain laughs"]


# ==================================================================================================
# test_consistency_text_continuity: Consistency JSON text_continuity merged into positive block.
# ==================================================================================================


def test_positive_block_includes_text_continuity():
    spec = {
        "character": "a hero",
        "text_continuity": {
            "chapter_motto": "Never look back",
            "locked_phrases": ["Remember the bridge"],
        },
    }
    out = positive_block_from_mapping(spec)
    assert "Never look back" in out
    assert "bridge" in out.lower()


def test_positive_block_ignores_json_nulls():
    spec = {
        "character": {"label": "Ren", "hair": None},
        "props": [{"description": None, "label": "x"}, {"description": "lantern", "label": None}],
        "vehicle": {"description": None},
        "lettering": {"extra": None},
    }
    out = positive_block_from_mapping(spec)
    assert "None" not in out
    assert "recurring character Ren" in out
    assert "same recurring prop: lantern" in out
