"""Tests for pipelines.book_comic.book_helpers shortcomings wiring."""

from types import SimpleNamespace

from pipelines.book_comic import book_helpers


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
    pv = book_helpers.preset_for_book_accuracy("production_vit")
    assert pv.pick_best == "combo_vit_hq"
    assert pv.sample_candidates == 6


def test_production_fidelity_preset():
    pf = book_helpers.preset_for_book_accuracy("production_fidelity")
    assert pf.pick_best == "combo_vit_hq"
    assert pf.sample_candidates == 8
    assert pf.shortcomings_mitigation == "all"


def test_derive_book_page_seed():
    assert book_helpers.derive_book_page_seed(10, 0) == 10
    assert book_helpers.derive_book_page_seed(10, 1) == 10 + book_helpers.BOOK_PAGE_SEED_STRIDE


def test_audit_book_run_flags_vit_warning():
    _, warns = book_helpers.audit_book_run_flags(
        pick_best="combo_vit",
        sample_candidates=2,
        pick_vit_ckpt="",
        beam_width=0,
    )
    assert any("pick-vit-ckpt" in w for w in warns)


def test_audit_book_run_flags_beam_sample_mismatch():
    _, warns = book_helpers.audit_book_run_flags(
        pick_best="clip",
        sample_candidates=4,
        beam_width=4,
    )
    assert any("beam" in w.lower() for w in warns)


def test_normalize_book_prompt_fragment():
    assert book_helpers.normalize_book_prompt_fragment("  a  \n b , ") == "a b"


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
    book_helpers.extend_sample_py_adapter_control_cmd(cmd, ns)
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
    frag = book_helpers.adapter_control_argv_for_sample(ns)
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
    book_helpers.extend_sample_py_sdx_enhance_cmd(cmd, ns)
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
    _, warns = book_helpers.audit_book_run_flags(
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
    book_helpers.extend_sample_py_adherence_quality_cmd(cmd, ns)
    s = " ".join(cmd)
    assert "--quality-pack illustrative" in s
    assert "--adherence-pack strict" in s
    assert "--clip-guard-threshold 0.24" in s
    assert "--volatile-cfg-boost 0.1" in s
    assert "--sag-scale 0.15" in s
    assert "--hard-style realistic" in s
    assert "--dual-stage-layout" in s
    assert "--dual-layout-steps 20" in s


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
    book_helpers.extend_sample_py_adherence_quality_cmd(cmd, ns)
    frag = book_helpers.adherence_quality_argv_for_sample(ns)
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
    book_helpers.extend_sample_py_sdx_enhance_cmd(cmd, ns)
    frag = book_helpers.sdx_enhance_argv_for_sample(ns)
    assert cmd[1:] == frag


def test_preset_defaults_for_book_accuracy():
    fast = book_helpers.preset_for_book_accuracy("fast")
    assert fast.shortcomings_mitigation == "none"
    assert fast.shortcomings_2d is False
    balanced = book_helpers.preset_for_book_accuracy("balanced")
    assert balanced.shortcomings_mitigation == "auto"
    assert balanced.shortcomings_2d is True
    assert balanced.art_guidance_mode == "auto"
    assert balanced.anatomy_guidance == "lite"
    assert balanced.style_guidance_mode == "auto"


def test_resolve_overrides_shortcomings():
    settings = book_helpers.resolve_book_sample_settings(
        _args(
            book_accuracy="balanced",
            shortcomings_mitigation="all",
            no_shortcomings_2d=True,
        )
    )
    assert settings.shortcomings_mitigation == "all"
    assert settings.shortcomings_2d is False


def test_append_sample_py_quality_flags_includes_shortcomings():
    settings = book_helpers.resolve_book_sample_settings(_args(book_accuracy="balanced"))
    cmd = ["python", "sample.py"]
    book_helpers.append_sample_py_quality_flags(cmd, settings, pick_expected_text="")
    assert "--shortcomings-mitigation" in cmd
    assert "--shortcomings-2d" in cmd
    assert "--art-guidance-mode" in cmd
    assert "--anatomy-guidance" in cmd
    assert "--style-guidance-mode" in cmd


def test_append_sample_py_quality_flags_forwards_vit_pick_and_ar():
    settings = book_helpers.resolve_book_sample_settings(
        _args(book_accuracy="none", sample_candidates=2, pick_best="combo_vit_hq")
    )
    cmd = ["python", "sample.py"]
    book_helpers.append_sample_py_quality_flags(
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
    book_helpers.append_sample_py_beam_flags(
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
    settings = book_helpers.resolve_book_sample_settings(
        _args(book_accuracy="none", sample_candidates=2, pick_best="combo_vit_realism")
    )
    cmd = ["python", "sample.py"]
    book_helpers.append_sample_py_quality_flags(cmd, settings, pick_expected_text="BONJOUR")
    assert "--expected-text BONJOUR" in " ".join(cmd)


def test_append_sample_py_quality_flags_combo_count_forwards_count_args():
    settings = book_helpers.resolve_book_sample_settings(_args(book_accuracy="none", sample_candidates=2, pick_best="combo_count"))
    cmd = ["python", "sample.py"]
    book_helpers.append_sample_py_quality_flags(
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
    s = book_helpers.compose_book_page_prompt(
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
    s = book_helpers.expand_page_prompt_template(
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
    settings = book_helpers.resolve_book_sample_settings(_args(book_accuracy="production"))
    flags = book_helpers.build_extra_ocr_sample_flags(settings)
    assert "--shortcomings-mitigation" in flags
    assert "all" in flags
    assert "--shortcomings-2d" in flags
    assert "--art-guidance-mode" in flags
    assert "--anatomy-guidance" in flags
    assert "--style-guidance-mode" in flags

