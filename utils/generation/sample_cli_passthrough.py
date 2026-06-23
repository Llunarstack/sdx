"""
Forward ``sample.py`` CLI flags from an argparse namespace into a subprocess argv list.

Used by OCR repair re-invocation so new flags do not require duplicating 200+ lines in ``sample.py``.
"""

from __future__ import annotations

from typing import Any, List


def append_sample_repair_passthrough(cmd: List[str], args: Any) -> None:
    """Append generation knobs from *args* that should match the parent ``sample.py`` run."""
    if int(getattr(args, "width", 0) or 0) > 0:
        cmd.extend(["--width", str(int(getattr(args, "width", 0) or 0))])
    if int(getattr(args, "height", 0) or 0) > 0:
        cmd.extend(["--height", str(int(getattr(args, "height", 0) or 0))])
    _rm = str(getattr(args, "resize_mode", "stretch") or "stretch").lower()
    if _rm in ("stretch", "center_crop", "saliency_crop"):
        cmd.extend(["--resize-mode", _rm])
    if float(getattr(args, "resize_saliency_face_bias", 0.0) or 0.0) > 0:
        cmd.extend(["--resize-saliency-face-bias", str(float(getattr(args, "resize_saliency_face_bias", 0.0) or 0.0))])

    if (
        getattr(args, "dynamic_threshold_type", "percentile") != "percentile"
        or getattr(args, "dynamic_threshold_value", 0.0) > 0.0
        or getattr(args, "dynamic_threshold_percentile", 0.0) > 0.0
    ):
        cmd.extend(
            [
                "--dynamic-threshold-percentile",
                str(getattr(args, "dynamic_threshold_percentile", 0.0)),
                "--dynamic-threshold-type",
                getattr(args, "dynamic_threshold_type", "percentile"),
                "--dynamic-threshold-value",
                str(getattr(args, "dynamic_threshold_value", 0.0)),
            ]
        )
    if getattr(args, "vae_tiling", False):
        cmd.append("--vae-tiling")
    if getattr(args, "deterministic", False):
        cmd.append("--deterministic")
    if getattr(args, "no_cache", False):
        cmd.append("--no-cache")
    if getattr(args, "no_refine", False):
        cmd.append("--no-refine")
    if getattr(args, "refine_t", None) is not None:
        cmd.extend(["--refine-t", str(getattr(args, "refine_t", 50))])
    if str(getattr(args, "refine_gate", "off") or "off").lower() in ("off", "auto"):
        cmd.extend(["--refine-gate", str(getattr(args, "refine_gate", "off"))])
    if getattr(args, "refine_gate_threshold", None) is not None:
        cmd.extend(["--refine-gate-threshold", str(getattr(args, "refine_gate_threshold", 0.62))])
    if getattr(args, "no_neg_filter", False):
        cmd.append("--no-neg-filter")
    cmd.append("--text-in-image")

    if getattr(args, "gender_swap", False):
        cmd.append("--gender-swap")
    if getattr(args, "anatomy_scale", ""):
        cmd.extend(["--anatomy-scale", str(getattr(args, "anatomy_scale"))])
    if getattr(args, "object_scale", ""):
        cmd.extend(["--object-scale", str(getattr(args, "object_scale"))])
    if getattr(args, "scene_scale", ""):
        cmd.extend(["--scene-scale", str(getattr(args, "scene_scale"))])
    for flag, name, default in (
        ("--photo-realism-pack", "photo_realism_pack", "none"),
        ("--photo-color-grade", "photo_color_grade", "none"),
        ("--photo-lighting-technique", "photo_lighting_technique", "none"),
        ("--photo-filter", "photo_filter", "none"),
        ("--photo-grain-style", "photo_grain_style", "none"),
    ):
        val = str(getattr(args, name, default) or default)
        if val != default:
            cmd.extend([flag, val])
    if float(getattr(args, "photo_realism_strength", 1.0) or 1.0) != 1.0:
        cmd.extend(["--photo-realism-strength", str(getattr(args, "photo_realism_strength"))])
    if not bool(getattr(args, "auto_photo_realism", True)):
        cmd.append("--no-auto-photo-realism")
    if not bool(getattr(args, "photo_postprocess", True)):
        cmd.append("--no-photo-postprocess")
    if float(getattr(args, "photo_post_strength", 0.6) or 0.6) != 0.6:
        cmd.extend(["--photo-post-strength", str(getattr(args, "photo_post_strength"))])
    if not bool(getattr(args, "realism_autopilot", True)):
        cmd.append("--no-realism-autopilot")
    if getattr(args, "character_sheet", ""):
        cmd.extend(["--character-sheet", str(getattr(args, "character_sheet"))])
    if getattr(args, "label_multi_character_sheets", False):
        cmd.append("--label-multi-character-sheets")
    if getattr(args, "character_prompt_extra", ""):
        cmd.extend(["--character-prompt-extra", str(getattr(args, "character_prompt_extra"))])
    if getattr(args, "character_negative_extra", ""):
        cmd.extend(["--character-negative-extra", str(getattr(args, "character_negative_extra"))])
    if getattr(args, "prompt_layout", ""):
        cmd.extend(["--prompt-layout", str(getattr(args, "prompt_layout"))])
    if getattr(args, "box_layout", ""):
        cmd.extend(["--box-layout", str(getattr(args, "box_layout"))])
    if str(getattr(args, "box_layout_mode", "regional_cfg") or "regional_cfg").lower() != "regional_cfg":
        cmd.extend(["--box-layout-mode", str(getattr(args, "box_layout_mode"))])
    if str(getattr(args, "t5_layout_encode", "auto") or "auto").lower() != "auto":
        cmd.extend(["--t5-layout-encode", str(getattr(args, "t5_layout_encode"))])
    if getattr(args, "scene_blueprint", ""):
        cmd.extend(["--scene-blueprint", str(getattr(args, "scene_blueprint"))])
    if float(getattr(args, "scene_blueprint_strength", 1.0)) != 1.0:
        cmd.extend(["--scene-blueprint-strength", str(getattr(args, "scene_blueprint_strength"))])
    if float(getattr(args, "character_strength", 1.0)) != 1.0:
        cmd.extend(["--character-strength", str(getattr(args, "character_strength"))])
    if bool(getattr(args, "uncensored_mode", False)):
        cmd.append("--uncensored-mode")
    for flag, attr in (
        ("--clothing-mode", "clothing_mode"),
        ("--background-mode", "background_mode"),
        ("--people-layout", "people_layout"),
        ("--relationship-mode", "relationship_mode"),
        ("--object-layout", "object_layout"),
        ("--artist-composition", "artist_composition"),
        ("--hand-mode", "hand_mode"),
        ("--pose-naturalness", "pose_naturalness"),
        ("--typography-mode", "typography_mode"),
        ("--quality-pack", "quality_pack"),
        ("--adherence-pack", "adherence_pack"),
        ("--lighting-mode", "lighting_mode"),
        ("--skin-detail-mode", "skin_detail_mode"),
        ("--style-mode", "style_mode"),
    ):
        val = str(getattr(args, attr, "none") or "none")
        if val != "none":
            cmd.extend([flag, val])
    if bool(getattr(args, "style_lock", False)):
        cmd.append("--style-lock")
    if bool(getattr(args, "anti_style_bleed", False)):
        cmd.append("--anti-style-bleed")
    if getattr(args, "preset", None):
        cmd.extend(["--preset", str(getattr(args, "preset"))])
    if getattr(args, "op_mode", None):
        cmd.extend(["--op-mode", str(getattr(args, "op_mode"))])
    if getattr(args, "holy_grail_preset", None):
        cmd.extend(["--holy-grail-preset", str(getattr(args, "holy_grail_preset"))])
    if getattr(args, "hard_style", None):
        cmd.extend(["--hard-style", str(getattr(args, "hard_style"))])
    if getattr(args, "boost_quality", False):
        cmd.append("--boost-quality")

    try:
        from pipelines.book_comic import book_helpers

        book_helpers.extend_sample_py_adapter_control_cmd(cmd, args)
        book_helpers.extend_sample_py_sdx_enhance_cmd(cmd, args)
        book_helpers.extend_sample_py_adherence_quality_cmd(cmd, args)
    except ImportError:
        _append_adapter_control_fallback(cmd, args)

    if getattr(args, "naturalize", False):
        cmd.append("--naturalize")
        cmd.extend(["--naturalize-grain", str(getattr(args, "naturalize_grain", 0.015))])
    if getattr(args, "naturalize_deep", False):
        cmd.append("--naturalize-deep")
    if getattr(args, "face_enhance", False):
        cmd.append("--face-enhance")
        cmd.extend(["--face-enhance-sharpen", str(getattr(args, "face_enhance_sharpen", 0.35))])
        cmd.extend(["--face-enhance-contrast", str(getattr(args, "face_enhance_contrast", 1.04))])
        cmd.extend(["--face-enhance-padding", str(getattr(args, "face_enhance_padding", 0.25))])
        cmd.extend(["--face-enhance-max", str(getattr(args, "face_enhance_max", 4))])
    pri = str(getattr(args, "post_reference_image", "") or "").strip()
    if pri:
        cmd.extend(["--post-reference-image", pri])
        cmd.extend(["--post-reference-alpha", str(float(getattr(args, "post_reference_alpha", 0.0) or 0.0))])
    frsh = str(getattr(args, "face_restore_shell", "") or "").strip()
    if frsh:
        cmd.extend(["--face-restore-shell", frsh])
    if float(getattr(args, "sag_blur_sigma", 0.0) or 0.0) > 0 and float(getattr(args, "sag_scale", 0.0) or 0.0) > 0:
        cmd.extend(["--sag-blur-sigma", str(getattr(args, "sag_blur_sigma", 0.0))])
        cmd.extend(["--sag-scale", str(getattr(args, "sag_scale", 0.0))])
    if getattr(args, "less_ai", False):
        cmd.append("--less-ai")
    _hm = str(getattr(args, "human_made", "none") or "none").lower()
    if _hm not in ("none", "off", "0", ""):
        cmd.extend(["--human-made", _hm])
        _hms = float(getattr(args, "human_made_strength", -1.0) or -1.0)
        if _hms >= 0.0:
            cmd.extend(["--human-made-strength", str(_hms)])
    if str(getattr(args, "anti_ai_pack", "none") or "none") != "none":
        cmd.extend(["--anti-ai-pack", str(getattr(args, "anti_ai_pack"))])
    if str(getattr(args, "human_media_mode", "none") or "none") != "none":
        cmd.extend(["--human-media", str(getattr(args, "human_media_mode"))])
    if getattr(args, "anti_bleed", False):
        cmd.append("--anti-bleed")
    if getattr(args, "diversity", False):
        cmd.append("--diversity")
    if getattr(args, "anti_artifacts", False):
        cmd.append("--anti-artifacts")
    if getattr(args, "strong_watermark", False):
        cmd.append("--strong-watermark")
    _rsm = str(getattr(args, "shortcomings_mitigation", "none") or "none").lower()
    if _rsm in ("auto", "all"):
        cmd.extend(["--shortcomings-mitigation", _rsm])
    if getattr(args, "shortcomings_2d", False):
        cmd.append("--shortcomings-2d")
    _rag = str(getattr(args, "art_guidance_mode", "none") or "none").lower()
    if _rag in ("auto", "all"):
        cmd.extend(["--art-guidance-mode", _rag])
    if getattr(args, "no_art_guidance_photography", False):
        cmd.append("--no-art-guidance-photography")
    _anat = str(getattr(args, "anatomy_guidance", "none") or "none").lower()
    if _anat in ("lite", "strong"):
        cmd.extend(["--anatomy-guidance", _anat])
    _sg = str(getattr(args, "style_guidance_mode", "none") or "none").lower()
    if _sg in ("auto", "all"):
        cmd.extend(["--style-guidance-mode", _sg])
    if not bool(getattr(args, "style_guidance_artists", True)):
        cmd.append("--no-style-guidance-artists")
    if not getattr(args, "auto_content_fix", True):
        cmd.append("--no-auto-content-fix")
    if not getattr(args, "one_shot_boost", True):
        cmd.append("--no-one-shot-boost")


def _append_adapter_control_fallback(cmd: List[str], args: Any) -> None:
    """Minimal adapter/control forward when book_helpers is unavailable."""
    if getattr(args, "style", ""):
        cmd.extend(["--style", str(getattr(args, "style"))])
        cmd.extend(["--style-strength", str(getattr(args, "style_strength", 0.7))])
    if getattr(args, "control_image", ""):
        cmd.extend(["--control-image", str(getattr(args, "control_image"))])
        cmd.extend(["--control-type", str(getattr(args, "control_type", "auto"))])
        cmd.extend(["--control-scale", str(getattr(args, "control_scale", 0.85))])
    if getattr(args, "control", None):
        cmd.extend(["--control"] + [str(x) for x in getattr(args, "control", [])])
    if getattr(args, "lora", None):
        cmd.extend(["--lora"] + [str(x) for x in getattr(args, "lora", [])])
    if bool(getattr(args, "holy_grail", False)):
        cmd.append("--holy-grail")
