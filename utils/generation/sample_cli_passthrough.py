"""``sample.py`` argv from argparse values (keyed by dest), and the flag set an OCR repair re-run inherits.

Building argv from dests through the real parser means a renamed or removed flag fails here, at build time,
instead of as an "unrecognized arguments" error inside a subprocess.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

__all__ = ["REPAIR_DESTS", "append_sample_repair_passthrough", "sample_argv"]

# Generation settings the OCR repair re-run (``--ocr-fix``) must share with the parent run. The caller sets
# prompt/seed/steps/solver/CFG and the inpaint knobs itself; prompt-rewriting features (invention stack,
# cast, corpus, explore-styles, ...) are absent because the repair receives the already-rewritten prompt.
REPAIR_DESTS: tuple[str, ...] = (
    # canvas
    "width",
    "height",
    "resize_mode",
    "resize_saliency_face_bias",
    "vae_tiling",
    "deterministic",
    "no_cache",
    "inference_amp",
    # sampler / guidance
    "dynamic_threshold_percentile",
    "dynamic_threshold_type",
    "dynamic_threshold_value",
    "eta",
    "sde_band",
    "eps_scale",
    "eps_scale_k",
    "flow_matching_sample",
    "force_vp_sample",
    "flow_solver",
    "flow_schedule",
    "flow_karras_rho",
    "flow_shift",
    "guidance_schedule",
    "guidance_schedule_linear_start",
    "guidance_schedule_linear_end",
    "guidance_schedule_cosine_min",
    "guidance_schedule_cosine_max",
    "fdg_low",
    "fdg_high",
    "apg_parallel_eta",
    "apg_norm_threshold",
    "apg_momentum_beta",
    "cfg_zero_star",
    "cfg_zero_init_frac",
    "cfg_interval",
    "tcfg_damping",
    "slg_scale",
    "slg_skip_blocks",
    "pag_scale",
    "pag_blocks",
    "sag_blur_sigma",
    "sag_scale",
    "volatile_cfg_boost",
    "volatile_cfg_quantile",
    "volatile_cfg_window",
    "dual_stage_layout",
    "dual_stage_div",
    "dual_layout_steps",
    "dual_detail_steps",
    "dual_detail_strength",
    "clip_guard_threshold",
    "clip_guard_model",
    "clip_guard_t_frac",
    "clip_guard_steps",
    "clip_monitor_every",
    "clip_monitor_threshold",
    "clip_monitor_cfg_boost",
    "clip_monitor_rewind",
    "domain_prior_latent",
    "spectral_coherence_latent",
    "spectral_coherence_cutoff",
    "refine",
    "no_refine",
    "refine_t",
    "restart_cycles",
    "restart_steps",
    "refine_gate",
    "refine_gate_threshold",
    "hires_fix",
    "hires_scale",
    "hires_steps",
    "hires_strength",
    "hires_cfg_scale",
    # adapters / control / holy grail
    "style",
    "style_strength",
    "auto_style_from_prompt",
    "tags",
    "tags_file",
    "control_image",
    "control",
    "control_type",
    "control_scale",
    "control_guidance_start",
    "control_guidance_end",
    "control_guidance_decay",
    "lora",
    "no_lora_normalize_scales",
    "lora_max_total_scale",
    "lora_default_role",
    "lora_role_budgets",
    "lora_stage_policy",
    "lora_layers",
    "lora_role_stage_weights",
    "lora_trigger",
    "lora_scaffold",
    "lora_scaffold_auto",
    "reference_image",
    "reference_strength",
    "reference_tokens",
    "reference_clip_model",
    "reference_adapter_pt",
    "holy_grail",
    "holy_grail_cfg_early_ratio",
    "holy_grail_cfg_late_ratio",
    "holy_grail_control_mult",
    "holy_grail_adapter_mult",
    "holy_grail_no_frontload_control",
    "holy_grail_late_adapter_boost",
    "holy_grail_cads_strength",
    "holy_grail_cads_min_strength",
    "holy_grail_cads_power",
    "holy_grail_unsharp_sigma",
    "holy_grail_unsharp_amount",
    "holy_grail_clamp_quantile",
    "holy_grail_clamp_floor",
    # prompt packs and modes
    "no_neg_filter",
    "gender_swap",
    "anatomy_scale",
    "object_scale",
    "scene_scale",
    "character_sheet",
    "label_multi_character_sheets",
    "character_prompt_extra",
    "character_negative_extra",
    "character_strength",
    "prompt_layout",
    "box_layout",
    "box_layout_mode",
    "t5_layout_encode",
    "scene_blueprint",
    "scene_blueprint_strength",
    "uncensored_mode",
    "clothing_mode",
    "background_mode",
    "people_layout",
    "relationship_mode",
    "object_layout",
    "artist_composition",
    "hand_mode",
    "pose_naturalness",
    "typography_mode",
    "quality_pack",
    "adherence_pack",
    "lighting_mode",
    "skin_detail_mode",
    "style_mode",
    "style_lock",
    "anti_style_bleed",
    "preset",
    "op_mode",
    "holy_grail_preset",
    "hard_style",
    "boost_quality",
    "no_quality_defaults",
    "auto_expected_text",
    "auto_constraint_boost",
    "visual_design_preset",
    "visual_design_domain",
    "visual_design_intensity",
    "visual_design_negative_pack",
    "anti_bleed",
    "diversity",
    "anti_artifacts",
    "strong_watermark",
    "shortcomings_mitigation",
    "shortcomings_2d",
    "art_guidance_mode",
    "no_art_guidance_photography",
    "anatomy_guidance",
    "style_guidance_mode",
    "style_guidance_artists",
    "auto_content_fix",
    "one_shot_boost",
    # finish
    "sharpen",
    "contrast",
    "saturation",
    "clarity",
    "tone_punch",
    "chroma_smooth",
    "polish",
    "finishing_preset",
    "naturalize",
    "naturalize_grain",
    "naturalize_deep",
    "less_ai",
    "human_made",
    "human_made_strength",
    "anti_ai_pack",
    "human_media_mode",
    "photo_realism_pack",
    "photo_color_grade",
    "photo_lighting_technique",
    "photo_filter",
    "photo_grain_style",
    "photo_realism_strength",
    "auto_photo_realism",
    "photo_postprocess",
    "photo_post_strength",
    "realism_autopilot",
    "camera_isp",
    "camera_iso",
    "camera_jpeg_quality",
    "save_16bit",
    "face_enhance",
    "face_enhance_sharpen",
    "face_enhance_contrast",
    "face_enhance_padding",
    "face_enhance_max",
    "post_reference_image",
    "post_reference_alpha",
    "face_restore_shell",
)


@lru_cache(maxsize=1)
def _parser() -> argparse.ArgumentParser:
    from utils.generation.sample_cli_parser import build_sample_parser

    return build_sample_parser()


@lru_cache(maxsize=1)
def _actions_by_dest() -> dict[str, list[argparse.Action]]:
    out: dict[str, list[argparse.Action]] = {}
    for a in _parser()._actions:
        if a.option_strings and not isinstance(a, argparse._HelpAction):
            out.setdefault(a.dest, []).append(a)
    return out


def _encode(dest: str, value: Any) -> list[str]:
    for action in _actions_by_dest()[dest]:
        flags = action.option_strings
        if isinstance(action, argparse.BooleanOptionalAction):
            return [flags[0] if value else next(f for f in flags if f.startswith("--no-"))]
        if isinstance(action, argparse._StoreConstAction):
            if action.const == value:
                return [flags[0]]
            continue
        if isinstance(value, (list, tuple)):
            return [flags[0], *map(str, value)]
        return [flags[0], str(value)]
    raise ValueError(f"no sample.py flag sets {dest}={value!r}")


def sample_argv(values: Mapping[str, Any]) -> list[str]:
    """``sample.py`` argv for each value that differs from its parser default (keys are argparse dests).

    Raises ``KeyError`` for a dest ``sample.py`` does not define.
    """
    argv: list[str] = []
    for dest, value in values.items():
        if dest not in _actions_by_dest():
            raise KeyError(f"sample.py has no option for dest {dest!r}")
        default = _parser().get_default(dest)
        if value is None or value == default:
            continue
        if isinstance(value, (list, tuple)) and list(value) == list(default or []):
            continue
        argv.extend(_encode(dest, value))
    return argv


def append_sample_repair_passthrough(cmd: list[str], args: Any) -> None:
    """Append the parent run's :data:`REPAIR_DESTS` settings (non-default only) to a repair ``sample.py`` argv."""
    cmd.extend(sample_argv({d: getattr(args, d) for d in REPAIR_DESTS if hasattr(args, d)}))
    cmd.append("--text-in-image")
