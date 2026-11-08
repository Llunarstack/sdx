"""``sample.py`` as a subprocess: argv from argparse values, the OCR-repair passthrough, img2img / inpaint runs, ComfyUI export."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

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
    "realism",
    "shot_spec",
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


try:
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise ImportError("sample_edit_runner requires Pillow") from e


def resolve_repo_root() -> Path:
    """SDX repository root (parent of ``sample.py``)."""
    return Path(__file__).resolve().parents[2]


def resolve_sample_py(repo_root: Path | None = None) -> Path:
    """Absolute path to the repository ``sample.py`` entrypoint."""
    root = repo_root if repo_root is not None else resolve_repo_root()
    p = root / "sample.py"
    if not p.is_file():
        raise FileNotFoundError(f"sample.py not found at {p} (unexpected layout?)")
    return p


def build_sample_command(
    *,
    ckpt: str,
    prompt: str,
    out_path: str | Path,
    repo_root: Path | None = None,
    sample_py: Path | None = None,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 28,
    cfg_scale: float = 7.0,
    seed: int | None = None,
    num: int = 1,
    device: str = "cuda",
    init_image_path: str | Path | None = None,
    mask_image_path: str | Path | None = None,
    strength: float = 0.65,
    inpaint_mode: str = "mdm",
    scheduler: str = "ays_dit",
    solver: str = "dpmpp_2m",
    timestep_schedule: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """
    Construct a ``sample.py`` argv list (executable + args). Caller runs it with no shell.

    Raises:
        ValueError: invalid dimensions, steps, ``strength``, or mask without init image.
    """
    if int(width) < 1 or int(height) < 1:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    if int(steps) < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    sf = float(strength)
    if not 0.0 <= sf <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {sf}")

    root = repo_root if repo_root is not None else resolve_repo_root()
    script = sample_py if sample_py is not None else resolve_sample_py(root)

    if mask_image_path and not init_image_path:
        raise ValueError("Inpainting requires both init image and mask paths.")

    cmd: list[str] = [
        sys.executable,
        str(script),
        "--ckpt",
        ckpt,
        "--prompt",
        prompt,
        "--out",
        str(out_path),
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
        "--steps",
        str(int(steps)),
        "--cfg-scale",
        str(float(cfg_scale)),
        "--num",
        str(int(max(1, num))),
        "--device",
        device,
        "--scheduler",
        scheduler or "ays_dit",
        "--solver",
        solver or "dpmpp_2m",
    ]
    ts = (timestep_schedule or "").strip()
    if ts:
        cmd.extend(["--timestep-schedule", ts])
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])
    if negative_prompt.strip():
        cmd.extend(["--negative-prompt", negative_prompt])

    init_p = Path(init_image_path) if init_image_path else None
    if init_p is not None:
        cmd.extend(["--init-image", str(init_p), "--strength", str(float(strength))])

    mask_p = Path(mask_image_path) if mask_image_path else None
    if mask_p is not None:
        cmd.extend(["--mask", str(mask_p), "--inpaint-mode", str(inpaint_mode or "mdm")])

    if extra_args:
        for a in extra_args:
            if not isinstance(a, str) or not a.strip():
                raise ValueError(f"extra_args entries must be non-empty strings, got {a!r}")
        cmd.extend(list(extra_args))

    return cmd


def run_sample_inference(
    *,
    ckpt: str,
    prompt: str,
    out_path: str | Path,
    cwd: str | Path | None = None,
    check: bool = True,
    capture_output: bool = True,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 28,
    cfg_scale: float = 7.0,
    seed: int | None = None,
    num: int = 1,
    device: str = "cuda",
    init_image_path: str | Path | None = None,
    mask_image_path: str | Path | None = None,
    strength: float = 0.65,
    inpaint_mode: str = "mdm",
    scheduler: str = "ays_dit",
    solver: str = "dpmpp_2m",
    timestep_schedule: str | None = None,
    repo_root: Path | None = None,
    sample_py: Path | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Run ``sample.py`` subprocess.

    ``cwd`` defaults to repository root so relative checkpoint paths behave like the CLI.
    """
    root = resolve_repo_root()
    cw = cwd if cwd is not None else root

    cmd = build_sample_command(
        ckpt=ckpt,
        prompt=prompt,
        out_path=out_path,
        repo_root=repo_root,
        sample_py=sample_py,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        num=num,
        device=device,
        init_image_path=init_image_path,
        mask_image_path=mask_image_path,
        strength=strength,
        inpaint_mode=inpaint_mode,
        scheduler=scheduler,
        solver=solver,
        timestep_schedule=timestep_schedule,
        extra_args=extra_args,
    )

    result = subprocess.run(
        cmd,
        cwd=str(cw),
        capture_output=capture_output,
        text=True,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        tail = (stderr + "\n" + stdout)[-4000:]
        raise RuntimeError(f"sample.py failed ({result.returncode}): {tail}")
    return result


def run_edit_with_pillow(
    *,
    ckpt: str,
    prompt: str,
    negative_prompt: str,
    base_image: Image.Image,
    mask_image: Image.Image | None,
    width: int,
    height: int,
    steps: int = 28,
    cfg_scale: float = 7.0,
    seed: int | None = None,
    img2img_strength: float = 0.65,
    inpaint_mode: str = "mdm",
    device: str = "cuda",
    scheduler: str = "ays_dit",
    solver: str = "dpmpp_2m",
    timestep_schedule: str | None = None,
    extra_args: list[str] | None = None,
    keep_temp_dir: bool = False,
) -> Image.Image:
    """
    Save ``base_image`` (+ optional grayscale ``mask_image``) to temp files and run inpaint/img2img.

    Mask convention matches ``sample.py``: **white** = inpaint / regenerate region.
    """
    base_rgb = base_image.convert("RGB").resize((int(width), int(height)), Image.Resampling.LANCZOS)
    mask_l: Image.Image | None = None
    if mask_image is not None:
        mask_l = mask_image.convert("L").resize((int(width), int(height)), Image.Resampling.LANCZOS)

    root = resolve_repo_root()
    tmp_home = tempfile.mkdtemp(prefix="sdx_edit_")
    tmp = Path(tmp_home)
    try:
        init_path = tmp / "init.png"
        out_path = tmp / "out.png"
        base_rgb.save(init_path, format="PNG")
        mask_path: Path | None = None
        if mask_l is not None:
            mask_path = tmp / "mask.png"
            mask_l.save(mask_path, format="PNG")

        run_sample_inference(
            ckpt=ckpt,
            prompt=prompt,
            out_path=out_path,
            cwd=root,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            init_image_path=init_path,
            mask_image_path=mask_path,
            strength=img2img_strength,
            inpaint_mode=inpaint_mode,
            device=device,
            scheduler=scheduler,
            solver=solver,
            timestep_schedule=timestep_schedule,
            extra_args=extra_args,
        )
        if not out_path.is_file():
            raise RuntimeError(f"sample.py reported success but output missing: {out_path}")
        return Image.open(out_path).convert("RGB")
    finally:
        if not keep_temp_dir:
            shutil.rmtree(tmp_home, ignore_errors=True)


def sample_args_to_comfy_workflow(
    args: Mapping[str, Any],
    *,
    title: str = "SDX Export",
) -> dict[str, Any]:
    """Build a portable Comfy-style node graph (API-loadable skeleton)."""
    ckpt = str(args.get("ckpt", "results/best.pt"))
    prompt = str(args.get("prompt", ""))
    negative = str(args.get("negative_prompt", ""))
    steps = int(args.get("steps", 50))
    cfg = float(args.get("cfg_scale", 7.5))
    seed = int(args.get("seed", 42))
    w = int(args.get("width", 0) or 1024)
    h = int(args.get("height", 0) or 1024)

    return {
        "version": "sdx-comfy-export-1",
        "title": title,
        "notes": "Generated by SDX — map CheckpointLoader to your Comfy DiT/SDX bridge node.",
        "nodes": {
            "1": {
                "class_type": "SDXLoadCheckpoint",
                "inputs": {"ckpt_path": ckpt},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative},
            },
            "4": {
                "class_type": "SDXSampler",
                "inputs": {
                    "steps": steps,
                    "cfg": cfg,
                    "seed": seed,
                    "width": w,
                    "height": h,
                    "box_layout": str(args.get("box_layout", "") or ""),
                    "holy_grail": bool(args.get("holy_grail", False)),
                    "frontier": bool(args.get("frontier", False)),
                },
            },
            "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": "sdx_export"}},
        },
        "links": [
            ["1", 0, "4", 0],
            ["2", 0, "4", 1],
            ["3", 0, "4", 2],
            ["4", 0, "5", 0],
        ],
    }


def write_comfy_workflow(path: str | Path, workflow: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    return p
