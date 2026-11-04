"""``sample.py`` run: flag resolution, the denoising passes and latent refinement, decode / post / pick-best / save, and programmatic text-to-image."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusion.sampling import dit_patch_size, dit_rgb_stride_px, sanitize_holy_grail_kwargs, snap_to_multiple
from PIL import Image

from utils.generation.sample_cli_parser import build_sample_parser
from utils.generation.sample_core import (
    SAMPLE_PY,
    _maybe_append_text_says,
    _refine_gate_score,
    encode_conditioning,
    load_models,
    prepare_init_latents,
    renoise_for_grid,
)
from utils.generation.sample_prompt import build_prompt_plan, compose_prompt_additions, compose_user_prompt
from utils.quality.finishing import save_rgb
from utils.safety.minor_guard import is_sexual, require_allowed, require_image_allowed


def to_rgb_u8(images: torch.Tensor) -> list[np.ndarray]:
    """``(N, 3, H, W)`` in ``[0, 1]`` -> list of ``(H, W, 3)`` uint8 (cheap previews for critics)."""
    arr = images.detach().float().permute(0, 2, 3, 1).cpu().numpy()
    return [(a * 255).round().astype(np.uint8) for a in arr]


def pick_indices(args: Any, plan: Any, images: list, metric: str, device: Any, *, art_plan: dict | None = None):
    """``pick_best_indices`` with the run's critic settings; returns ``(best_index, scores)``."""
    from utils.quality.test_time_pick import pick_best_indices

    return pick_best_indices(
        images,
        plan.prompt,
        metric,
        str(device),
        plan.expected_ocr,
        getattr(args, "pick_clip_model", "openai/clip-vit-base-patch32"),
        int(getattr(args, "expected_count", 0) or 0),
        str(getattr(args, "expected_count_target", "auto") or "auto"),
        str(getattr(args, "expected_count_object", "") or ""),
        str(getattr(args, "pick_vit_ckpt", "") or ""),
        bool(getattr(args, "pick_vit_use_adherence", False)),
        int(getattr(args, "pick_vit_ar_blocks", -1) or -1),
        plan=art_plan,
    )


def decode_images(args: Any, m: Any, x0: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int], bool]:
    """Decode to ``[0, 1]`` RGB at the requested size; returns (images, (out_w, out_h), needs a fit-resize)."""
    image = m.decode_unit(x0)
    dec_h, dec_w = int(image.shape[-2]), int(image.shape[-1])
    out_w, out_h = dec_w, dec_h
    stretch = str(getattr(args, "resize_mode", "stretch") or "stretch") == "stretch"
    if args.width > 0 or args.height > 0:
        out_w, out_h = m.req_w, m.req_h
        if (out_w, out_h) != (dec_w, dec_h) and stretch:
            image = torch.nn.functional.interpolate(
                image, size=(out_h, out_w), mode="bilinear", align_corners=False, antialias=True
            )
        if out_w * out_h > 1.5 * dec_w * dec_h:
            print(
                f"Note: decoded {dec_w}x{dec_h} is upscaled to {out_w}x{out_h}; --hires-fix renders that detail instead.",
                file=sys.stderr,
            )
    return image, (out_w, out_h), (out_w, out_h) != (dec_w, dec_h) and not stretch


def _fit(args: Any, img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """``--resize-mode center_crop / saliency_crop``: fit the decode to the requested aspect ratio."""
    from utils.quality.finishing import to_uint8

    try:
        from utils.quality.finishing import fit_image_to_size

        return fit_image_to_size(
            to_uint8(img),
            out_h,
            out_w,
            mode=str(getattr(args, "resize_mode", "stretch") or "stretch"),
            saliency_face_bias=float(getattr(args, "resize_saliency_face_bias", 0.0) or 0.0),
        ).astype(np.float32)
    except Exception as e:
        print(f"Resize mode fallback to stretch: {e}", file=sys.stderr)
        resized = Image.fromarray(to_uint8(img), mode="RGB").resize((out_w, out_h), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32)


def _finishing(args: Any, img: np.ndarray) -> np.ndarray:
    """Contrast / saturation / tone / chroma / clarity / polish / sharpen and ``--finishing-preset`` baselines."""
    sat = float(getattr(args, "saturation", 1.0))
    preset = str(getattr(args, "finishing_preset", "none") or "none").lower()
    knobs = ("clarity", "tone_punch", "chroma_smooth", "polish")
    if not (
        args.contrast != 1.0
        or args.sharpen > 0
        or abs(sat - 1.0) > 1e-6
        or any(float(getattr(args, k, 0.0)) > 0 for k in knobs)
        or preset != "none"
    ):
        return img
    try:
        from utils.quality.finishing import (
            FINISHING_PRESET_BASELINES,
            chroma_smooth_light,
            contrast,
            gentle_s_curve_luminance,
            luminance_clarity,
            polish_pass,
            saturation_rgb,
            sharpen,
        )

        b_clarity, b_tone, b_chroma = FINISHING_PRESET_BASELINES.get(preset, (0.0, 0.0, 0.0))
        clarity = min(0.55, float(getattr(args, "clarity", 0.0)) + b_clarity)
        tone = min(0.42, float(getattr(args, "tone_punch", 0.0)) + b_tone)
        chroma = min(0.65, float(getattr(args, "chroma_smooth", 0.0)) + b_chroma)
        polish = float(getattr(args, "polish", 0.0))
        if args.contrast != 1.0:
            img = contrast(img, factor=args.contrast)
        if abs(sat - 1.0) > 1e-6:
            img = saturation_rgb(img, factor=sat)
        if tone > 0:
            img = gentle_s_curve_luminance(img, strength=tone)
        if chroma > 0:
            img = chroma_smooth_light(img, amount=chroma)
        if clarity > 0:
            img = luminance_clarity(img, amount=clarity)
        if polish > 0:
            img = polish_pass(img, amount=polish)
        if args.sharpen > 0:
            img = sharpen(img, amount=args.sharpen)
    except Exception:
        pass
    return img


def _artistic(args: Any, img: np.ndarray, seed: int) -> np.ndarray:
    """Composition guide, value structure, lost-and-found edges and micro detail (``artistic_post_process``)."""
    wanted = (
        str(getattr(args, "composition_guide", "none") or "none") != "none"
        or bool(getattr(args, "value_structure", False))
        or any(
            float(getattr(args, k, 0.0) or 0.0) > 0
            for k in (
                "value_shadow_lift",
                "value_highlight_roll",
                "value_midtone_contrast",
                "lost_found_edges",
                "micro_detail",
            )
        )
    )
    if not wanted:
        return img
    try:
        from utils.quality.finishing import ArtisticPostConfig, apply_artistic_pipeline

        cfg = ArtisticPostConfig(
            composition_mode=str(getattr(args, "composition_guide", "none") or "none"),
            composition_strength=float(getattr(args, "composition_guide_strength", 0.15) or 0.15),
            value_structure=bool(getattr(args, "value_structure", False)),
            value_shadow_lift=float(getattr(args, "value_shadow_lift", 0.0) or 0.0),
            value_highlight_roll=float(getattr(args, "value_highlight_roll", 0.0) or 0.0),
            value_midtone_contrast=float(getattr(args, "value_midtone_contrast", 0.0) or 0.0),
            lost_found_strength=float(getattr(args, "lost_found_edges", 0.0) or 0.0),
            lost_found_seed=seed,
            micro_detail=float(getattr(args, "micro_detail", 0.0) or 0.0),
        )
        return apply_artistic_pipeline(img, cfg)
    except Exception as e:
        print(f"Artistic post-process failed (non-fatal): {e}", file=sys.stderr)
        return img


def postprocess(args: Any, plan: Any, images: torch.Tensor, out_size: tuple[int, int], fit: bool) -> list:
    """Deterministic per-candidate post in float32 ``[0, 255]`` (no grain: the finish adds capture noise later)."""
    out_w, out_h = out_size
    photo_post = bool(getattr(args, "photo_postprocess", True)) and (
        plan.photo_grade.lower() != "none" or plan.photo_filter.lower() != "none"
    )
    processed = []
    for i in range(plan.num_gen):
        img = images[i].detach().float().permute(1, 2, 0).cpu().numpy() * 255.0
        if fit:
            img = _fit(args, img, out_w, out_h)
        img = _finishing(args, img)
        if photo_post:
            try:
                from utils.quality.finishing import apply_photo_color_grade, apply_photo_filter

                strength = float(getattr(args, "photo_post_strength", 0.6) or 0.6)
                img = apply_photo_color_grade(img, preset=plan.photo_grade, strength=strength)
                img = apply_photo_filter(img, filter_name=plan.photo_filter, strength=strength)
            except Exception:
                pass
        img = _artistic(args, img, int(args.seed) + i + 1000)
        try:
            from utils.quality.finishing import apply_human_made_pipeline, config_from_args

            hm_cfg = config_from_args(args)
            if hm_cfg is not None:
                img = apply_human_made_pipeline(img, hm_cfg)
        except Exception as e:
            print(f"Human-made post-process failed (non-fatal): {e}", file=sys.stderr)
        if getattr(args, "face_enhance", False):
            try:
                from utils.quality.finishing import enhance_faces_in_rgb

                img = enhance_faces_in_rgb(
                    img,
                    padding=float(getattr(args, "face_enhance_padding", 0.25)),
                    sharpen_amount=float(getattr(args, "face_enhance_sharpen", 0.35)),
                    micro_contrast=float(getattr(args, "face_enhance_contrast", 1.04)),
                    max_faces=int(getattr(args, "face_enhance_max", 4)),
                ).astype(np.float32)
            except Exception:
                pass
        ref = str(getattr(args, "post_reference_image", "") or "").strip()
        alpha = float(getattr(args, "post_reference_alpha", 0.0) or 0.0)
        if ref and alpha > 0:
            try:
                from utils.quality.finishing import blend_reference_rgb

                ref_np = np.array(Image.open(ref).convert("RGB"))
                img = blend_reference_rgb(img, ref_np, alpha=alpha).astype(np.float32)
            except Exception:
                pass
        processed.append(img)
    return processed


def pick_metric(args: Any, plan: Any) -> str:
    """``--pick-best`` (``auto`` routes on expected text, counts, photo prompts and the ViT critic)."""
    metric = (getattr(args, "pick_best", None) or "none").lower()
    if metric != "auto":
        return metric
    # --hard-prompts: the critic for the prompt's hardest constraint (count, left/right colours, hands, people).
    hard = str(getattr(args, "_hard_pick_metric", "") or "")
    if hard and not plan.expected_texts:
        print(f"pick-best auto -> {hard} (hard prompt)", file=sys.stderr)
        return hard
    photo = plan.is_photo_prompt or plan.photo_pack.lower() != "none"
    if bool(getattr(args, "realism_autopilot", True)) and plan.auto_pick_metric.strip():
        metric = plan.auto_pick_metric.strip().lower()
    elif str(getattr(args, "pick_vit_ckpt", "") or "").strip():
        if plan.expected_texts:
            metric = "combo_vit_realism"
        elif plan.has_count:
            metric = "combo_count_vit"
        else:
            metric = "combo_vit_realism" if photo else "combo_vit_hq"
    elif bool(getattr(args, "pick_auto_no_clip", False)):
        if plan.expected_texts:
            metric = "ocr"
        elif plan.has_count:
            metric = "combo_count"
        else:
            metric = "aesthetic_realism" if photo else "aesthetic"
    elif plan.expected_texts:
        metric = "combo"
    elif plan.has_count:
        metric = "combo_count"
    else:
        metric = "combo_realism" if photo else "combo_hq"
    print(f"pick-best auto -> {metric}", file=sys.stderr)
    return metric


def choose_best(args: Any, m: Any, plan: Any, processed: list, report: dict) -> tuple[str, int]:
    """Rank the pre-finish candidates; returns (metric, best index)."""
    from utils.quality.critics import resolve_plan
    from utils.quality.finishing import to_uint8

    metric = pick_metric(args, plan)
    if plan.num_gen <= 1 or metric == "none":
        return metric, 0
    # Deterministic pre-finish candidates, judged against what the user asked for, not the enriched prompt.
    art_plan = resolve_plan(
        str(getattr(args, "_raw_prompt_before_compose", "") or args.prompt or ""), photo_color_grade=plan.photo_grade
    )
    best, scores = pick_indices(args, plan, [to_uint8(p) for p in processed], metric, m.device, art_plan=art_plan)
    print(f"pick-best ({metric}): scores={scores} -> best index {best}")
    report["pick_best"] = {
        "metric": metric,
        "plan": dict(art_plan),
        "scores": [float(x) for x in scores],
        "best_index": int(best),
        "expected_text": plan.expected_ocr,
        "expected_count": int(getattr(args, "expected_count", 0) or 0),
        "expected_count_target": str(getattr(args, "expected_count_target", "auto") or "auto"),
        "expected_count_object": str(getattr(args, "expected_count_object", "") or ""),
        "pick_clip_model": str(getattr(args, "pick_clip_model", "openai/clip-vit-base-patch32") or ""),
        "pick_vit_ckpt": str(getattr(args, "pick_vit_ckpt", "") or ""),
        "pick_vit_use_adherence": bool(getattr(args, "pick_vit_use_adherence", False)),
        "pick_vit_ar_blocks": int(getattr(args, "pick_vit_ar_blocks", -1) or -1),
        "pick_auto_no_clip": bool(getattr(args, "pick_auto_no_clip", False)),
    }
    from utils.prompt.game import resolve_game_profile

    prof = resolve_game_profile(args)
    if prof is not None and prof.critic:
        # Game assets also have to work as assets: a clean pixel grid, a seamless wrap, an icon that reads at 32 px.
        from utils.quality.pixel_art import game_score

        g = np.asarray([game_score(to_uint8(p), prof.critic) for p in processed])
        s = np.asarray(scores, dtype=np.float64)
        combined = (s - s.min()) / (float(s.max() - s.min()) or 1.0) + g
        best = int(combined.argmax())
        report["pick_best"].update(game_critic=prof.critic, game_scores=g.round(4).tolist(), best_index=best)
        print(f"game critic ({prof.critic}): {g.round(3).tolist()} -> best index {best}")
    return metric, int(best)


def _capture_finish(args: Any, plan: Any, report: dict):
    """(finish plan or None, lens-only plan or None) for the medium-routed camera / film finish."""
    from utils.quality.finishing import NO_LENS, plan_finish, user_lens

    cam = str(getattr(args, "camera_isp", "auto") or "auto").lower()
    photo_post = bool(getattr(args, "photo_postprocess", True))
    spec = getattr(args, "_shot_spec", None)  # --realism: set only for photographic prompts
    wanted = (
        cam != "auto"
        or str(getattr(args, "human_made", "none") or "none").lower() not in ("none", "off", "0", "")
        or (bool(getattr(args, "naturalize", False)) and float(getattr(args, "naturalize_grain", 0.015) or 0.0) > 0)
        or (photo_post and plan.photo_grain.lower() != "none")
        or spec is not None
    )
    vignette = float(getattr(args, "vignette", 0.0) or 0.0)
    ca = float(getattr(args, "chromatic_aberration", 0.0) or 0.0)
    finish = None
    if wanted:
        finish = plan_finish(
            str(getattr(args, "_raw_prompt_before_compose", "") or args.prompt or ""),
            camera=cam,
            iso=int(getattr(args, "camera_iso", 0) or 0) or int(getattr(spec, "iso", 0) or 0),
            grain_style=plan.photo_grain if photo_post else "none",
            human_media=str(getattr(args, "human_media_mode", "none") or "none"),
            vignette=vignette,
            chromatic_aberration=ca,
        )
        if spec is not None:
            from utils.quality.realism import tune_finish

            finish = tune_finish(finish, spec, user_vignette=vignette, user_ca=ca)
    if finish is None:
        # No capture finish (not wanted, or a non-photographic medium): explicit lens knobs still get one
        # linear-light lens pass.
        return None, user_lens(NO_LENS, vignette, ca) if (vignette > 0 or ca > 0) else None
    print(f"Capture finish: {finish.name}, ISO {finish.iso}", file=sys.stderr)
    report["capture_finish"] = {"profile": finish.name, "iso": int(finish.iso)}
    return finish, None


def capture_finish(args: Any, plan: Any, processed: list, report: dict) -> list[np.ndarray]:
    """The camera / film finish (or lens-only pass), once per candidate after ranking; float ``[0, 255]``."""
    from utils.quality.finishing import apply_finish, apply_lens_srgb

    finish, lens_only = _capture_finish(args, plan, report)
    correct = None
    if finish is not None and getattr(args, "_shot_spec", None) is not None:
        from utils.quality.critics import plan_from_prompt
        from utils.quality.realism import realism_correct, realism_strength

        raw = str(getattr(args, "_raw_prompt_before_compose", "") or args.prompt or "")
        vivid = plan_from_prompt(raw).get("grade") in ("vivid", "neon")
        strength = realism_strength(args)
        report["realism"] = {"strength": strength, "shot_spec": getattr(args, "shot_spec_plan", None)}

        def correct(im: np.ndarray) -> np.ndarray:
            return realism_correct(im, strength, vivid_ok=vivid)

    finals = []
    for i, img in enumerate(processed):
        if finish is not None:
            if correct is not None:
                img = correct(img)
            img = apply_finish(
                img,
                finish,
                seed=int(args.seed) + i,
                jpeg_quality=int(getattr(args, "camera_jpeg_quality", 0) or 0),
                jpeg=not bool(getattr(args, "save_16bit", False)),
            )
        elif lens_only is not None:
            img = apply_lens_srgb(img, lens_only)
        finals.append(img)
    return finals


def game_finish(args: Any, finals: list, report: dict) -> tuple[list, list]:
    """Seamless tiles, an exact pixel grid + palette, transparent sprite backgrounds.

    Returns (finals, native-resolution sprites or None per candidate). Pixel art comes back uint8 so
    saving can't dither its palette.
    """
    from utils.prompt.game import resolve_game_profile

    prof = resolve_game_profile(args)
    if prof is None or not (prof.pixel_art or prof.tileable or prof.transparent_bg):
        return finals, [None] * len(finals)
    from utils.quality.finishing import to_uint8
    from utils.quality.pixel_art import background_mask, make_tileable, pixel_snap

    out, natives = [], []
    for i, img in enumerate(finals):
        native = None
        if prof.tileable:
            img = make_tileable(img)
        if prof.pixel_art:
            img, native = pixel_snap(
                img,
                pixel_size=int(getattr(args, "pixel_size", 0) or 0),
                grid=prof.grid,
                palette_colors=prof.palette_colors,
                transparent_bg=prof.transparent_bg,
                seed=int(args.seed) + i,
            )
        elif prof.transparent_bg:
            u8 = to_uint8(img)
            img = np.dstack([u8, np.where(background_mask(u8, tol=28.0), 0, 255).astype(np.uint8)])
        out.append(img)
        natives.append(native)
    report["game"] = {
        "asset": prof.asset,
        "pixel_art": prof.pixel_art,
        "palette_colors": prof.palette_colors,
        "transparent_bg": prof.transparent_bg,
        "tileable": prof.tileable,
        "native_size": list(natives[0].shape[:2]) if natives and natives[0] is not None else None,
    }
    print(f"Game finish: {prof.asset}" + (f", sprite {report['game']['native_size']}" if prof.pixel_art else ""))
    return out, natives


def save_game_image(img: np.ndarray, native: np.ndarray | None, path: Path, *, sixteen_bit: bool, seed: int) -> None:
    """save_rgb, except alpha images go out as PNG with alpha, plus the native-size sprite next to them."""
    if img.shape[-1] == 4:
        if path.suffix.lower() == ".png":
            Image.fromarray(img).save(path)
        else:
            print(f"Transparent background needs .png; {path.name} is saved without alpha", file=sys.stderr)
            save_rgb(img[..., :3], path, seed=seed)
    else:
        save_rgb(img, path, sixteen_bit=sixteen_bit, seed=seed)
    if native is not None:
        sprite = path.with_name(f"{path.stem}_sprite.png")
        Image.fromarray(native).save(sprite)
        print(f"Saved native sprite: {sprite} ({native.shape[1]}x{native.shape[0]})")


def save_targets(args: Any, n: int, metric: str, best: int) -> list[tuple[int, Path, str]]:
    """(candidate index, path, log label): ``--pick-save-all`` candidates, then the pick or every sample."""
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem, ext = out_path.stem, out_path.suffix or ".png"
    targets = []
    if getattr(args, "pick_save_all", False) and n > 1:
        targets += [(i, out_path.parent / f"{stem}_cand{i}{ext}", "Saved candidate") for i in range(n)]
    if n == 1:
        targets.append((0, out_path, "Saved"))
    elif metric != "none":
        targets.append((best, out_path, f"Saved best ({metric})"))
    else:
        targets += [(i, out_path.parent / f"{stem}_{i}{ext}", "Saved") for i in range(n)]
    return targets


def after_save(args: Any, plan: Any, metric: str, best: int, finals: list, report: dict) -> None:
    """Character session, feedback log, params sidecar, pick report, OCR repair, face-restore hook and grid."""
    out_path = Path(args.out)
    stem, ext = out_path.stem, out_path.suffix or ".png"
    n = plan.num_gen
    session = str(getattr(args, "save_character_session", "") or "").strip()
    if session:
        from utils.generation.sample_prompt import save_character_session

        save_character_session(
            session,
            {
                "name": Path(session).stem,
                "prompt_additions": str(getattr(args, "prompt", "") or ""),
                "negative_prompt": str(getattr(args, "negative_prompt", "") or ""),
                "reference_images": [str(args.init_image)] if getattr(args, "init_image", "") else [],
                "last_output": str(out_path),
                "seed": int(getattr(args, "seed", 0) or 0),
            },
        )
        print(f"Saved character session: {session}", file=sys.stderr)

    if bool(getattr(args, "log_feedback", False)) or str(getattr(args, "feedback_log", "") or "").strip():
        try:
            from utils.training.preference import default_feedback_log, record_generation, record_pick

            log = str(getattr(args, "feedback_log", "") or "").strip() or str(default_feedback_log())
            tag = "cand" if getattr(args, "pick_save_all", False) else ""
            cands = [str(out_path.parent / f"{stem}_{tag}{i}{ext}") for i in range(n)] if n > 1 else []
            prompt = str(getattr(args, "prompt", "") or "")
            record_generation(
                str(out_path),
                prompt=prompt,
                ckpt=str(getattr(args, "ckpt", "") or ""),
                seed=int(getattr(args, "seed", 0) or 0),
                candidates=cands if len(cands) > 1 else None,
                log_path=log,
            )
            losers = [c for i, c in enumerate(cands) if i != best]
            if metric != "none" and losers:
                record_pick(str(out_path), losers, prompt=prompt, log_path=log)
        except Exception as e:
            print(f"feedback log skipped: {e}", file=sys.stderr)

    if getattr(args, "save_prompt", False):
        lines = [
            f"prompt: {args.prompt}",
            f"negative: {(args.negative_prompt or '').strip() or '(default)'}",
            f"seed: {args.seed}",
            f"steps: {args.steps}",
            f"cfg_scale: {getattr(args, 'cfg_scale', 7.5)}",
            f"scheduler: {getattr(args, 'timestep_schedule', None) or getattr(args, 'scheduler', 'ays_dit')}, "
            f"solver: {getattr(args, 'solver', 'dpmpp_2m')}",
            f"hires_fix: {getattr(args, 'hires_fix', False)}, hires_scale: {getattr(args, 'hires_scale', 1.5)}, "
            f"hires_steps: {getattr(args, 'hires_steps', 15)}, saturation: {getattr(args, 'saturation', 1.0)}",
            f"finishing_preset: {getattr(args, 'finishing_preset', 'none')}, clarity: {getattr(args, 'clarity', 0.0)}, "
            f"tone_punch: {getattr(args, 'tone_punch', 0.0)}, chroma_smooth: {getattr(args, 'chroma_smooth', 0.0)}, "
            f"polish: {getattr(args, 'polish', 0.0)}",
        ]
        sidecar = out_path.parent / f"{stem}.txt"
        sidecar.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved params: {sidecar}")

    report_path = str(getattr(args, "pick_report_json", "") or "").strip()
    if report_path:
        try:
            import json

            rp = Path(report_path).expanduser()
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved pick report: {rp}")
        except Exception as e:
            print(f"pick-report-json: failed to write ({e})", file=sys.stderr)

    if getattr(args, "ocr_fix", False) and plan.expected_texts and n == 1:
        _ocr_repair(args, plan, out_path)

    shell = str(getattr(args, "face_restore_shell", "") or "").strip()
    if shell:
        try:
            import shlex

            # {src}/{dst} become literal path tokens before splitting; no shell ever parses the result.
            cmd = shlex.split(shell.replace("{src}", str(out_path)).replace("{dst}", str(out_path)))
            subprocess.run(cmd, shell=False, check=False)
        except Exception as e:
            print(f"face-restore-shell failed: {e}", file=sys.stderr)

    if n > 1 and getattr(args, "grid", False) and finals:
        try:
            from utils.quality.finishing import to_uint8

            saved = [to_uint8(f, np.random.default_rng(int(args.seed) + i)) for i, f in enumerate(finals)]
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            h, w = saved[0].shape[:2]
            grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
            for i in range(n):
                r, c = divmod(i, cols)
                grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = saved[i]
            grid_path = out_path.parent / f"{stem}_grid{ext}"
            Image.fromarray(grid).save(grid_path)
            print(f"Saved grid: {grid_path}")
        except Exception as e:
            print(f"Grid save failed: {e}", file=sys.stderr)


def _ocr_repair(args: Any, plan: Any, out_path: Path) -> None:
    """``--ocr-fix``: OCR-validate the saved image and re-run ``sample.py`` as an MDM inpaint of the text box."""
    if int(getattr(args, "ocr_repair_iter", 0)) >= int(getattr(args, "ocr_iters", 0)):
        return
    try:
        from utils.generation.editing import create_text_rendering_pipeline

        expected = plan.expected_texts
        pipe = create_text_rendering_pipeline()
        pil = Image.open(out_path).convert("RGB")
        acc = float(pipe["engine"].validate_text_rendering(pil, expected).get("accuracy_score", 0.0))
        threshold = float(getattr(args, "ocr_threshold", 0.65))
        if acc >= threshold:
            print(f"OCR fix: acc={acc:.3f} >= {threshold}; done.", file=sys.stderr)
            return
        mask = pipe["inpainting"].create_text_edit_mask(pil, target_text=expected[0] if len(expected) == 1 else None)
        if mask is None:
            print("OCR fix: could not build mask; stopping.", file=sys.stderr)
            return
        dilate = int(getattr(args, "ocr_mask_dilate", 0))
        if dilate > 0:
            import cv2

            kernel = np.ones((dilate * 2 + 1, dilate * 2 + 1), dtype=np.uint8)
            mask = Image.fromarray(cv2.dilate(np.array(mask.convert("L")), kernel, iterations=1), mode="L")
        mask_dir = out_path.parent / "ocr_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"{out_path.stem}_ocrmask_iter{args.ocr_repair_iter}.png"
        mask.save(mask_path)
        cmd = [
            sys.executable,
            str(SAMPLE_PY),
            "--ckpt",
            args.ckpt,
            "--prompt",
            _maybe_append_text_says(args.prompt, expected),
            "--negative-prompt",
            args.negative_prompt or "",
            "--out",
            str(out_path),
            "--num",
            "1",
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--scheduler",
            getattr(args, "timestep_schedule", None) or getattr(args, "scheduler", "ays_dit"),
            "--solver",
            getattr(args, "solver", "dpmpp_2m"),
            "--karras-rho",
            str(getattr(args, "karras_rho", 7.0)),
            "--cfg-scale",
            str(getattr(args, "cfg_scale", 7.5)),
            "--cfg-rescale",
            str(getattr(args, "cfg_rescale", 0.0)),
            "--strength",
            str(getattr(args, "ocr_inpaint_strength", 0.55)),
            "--init-image",
            str(out_path),
            "--mask",
            str(mask_path),
            "--inpaint-mode",
            "mdm",
            "--expected-text",
            ",".join(expected),
            "--ocr-fix",
            "--ocr-threshold",
            str(threshold),
            "--ocr-iters",
            str(getattr(args, "ocr_iters", 2)),
            "--ocr-mask-dilate",
            str(dilate),
            "--ocr-inpaint-strength",
            str(getattr(args, "ocr_inpaint_strength", 0.55)),
            "--ocr-repair-iter",
            str(int(getattr(args, "ocr_repair_iter", 0)) + 1),
        ]
        from utils.generation.sample_argv import append_sample_repair_passthrough

        append_sample_repair_passthrough(cmd, args)
        print(f"OCR fix: acc={acc:.3f} < {threshold}; repairing with mdm inpainting...", file=sys.stderr)
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"OCR fix failed (non-fatal): {e}", file=sys.stderr)
        return
    # The repair run overwrote --out; nothing after it should touch the old image.
    sys.exit(0)


# Model kwargs that carry one row per sample.
_PER_SAMPLE_KWARGS = (
    "encoder_hidden_states",
    "style_embedding",
    "token_weights",
    "creativity",
    "reference_tokens",
    "control_image",
    "control_type",
    "control_scale",
    "edit_latents",
)


def resolve_inference_amp(args, device) -> tuple[bool, torch.dtype | None]:
    """Map ``--inference-amp`` to ``(enabled, autocast dtype)`` for ``sample_loop``."""
    mode = str(getattr(args, "inference_amp", "auto") or "auto").strip().lower()
    if mode == "fp16":
        return True, torch.float16
    if mode == "bf16":
        return True, torch.bfloat16
    if mode == "auto" and getattr(device, "type", None) == "cuda":
        return True, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return False, None


def _with_batch(kw: dict, n: int) -> dict:
    """Repeat batch-1 per-sample kwargs to ``n`` rows (beam branches share one prompt)."""
    return {
        k: v.expand(n, *v.shape[1:]) if k in _PER_SAMPLE_KWARGS and torch.is_tensor(v) and v.shape[0] == 1 else v
        for k, v in kw.items()
    }


def _f(args: Any, name: str, default: float) -> float:
    """Float knob; an explicit 0 is honoured (only a missing / None value takes the default)."""
    value = getattr(args, name, None)
    return float(default if value is None else value)


class Sampler:
    """One run's sampler settings; every pass goes through ``run`` so they share the same guidance stack."""

    def __init__(self, args: Any, m: Any, plan: Any, cond: Any, init: Any, feature_kw: dict):
        self.args, self.m, self.plan, self.c, self.init = args, m, plan, cond, init
        self.feature_kw = feature_kw
        cfg_scale = getattr(args, "cfg_scale", 7.5)
        cfg_rescale = getattr(args, "cfg_rescale", 0.0)
        dyn_p = getattr(args, "dynamic_threshold_percentile", 0.0)
        if getattr(args, "originality", 0.0):
            # Lower CFG a little so the model can explore beyond the most literal token match.
            strength = max(0.0, min(1.0, float(args.originality)))
            cfg_scale = max(1.0, cfg_scale - strength * 2.0)
            if cfg_rescale == 0.0 and cfg_scale > 6.0:
                cfg_rescale = 0.6
        # High CFG without rescale oversaturates; Imagen thresholding is not auto-enabled because on VAE latents
        # (|x| well above 1) it rescales the whole latent toward grey.
        if cfg_scale > 10 and cfg_rescale == 0.0 and dyn_p == 0.0:
            cfg_rescale = 0.7
            print(f"High CFG ({cfg_scale}): auto-enabled cfg_rescale=0.7 to reduce oversaturation.", file=sys.stderr)
        self.cfg_scale = cfg_scale
        holy = sanitize_holy_grail_kwargs(_holy_grail_kwargs(args))
        if holy["holy_grail_enable"]:
            print("Holy-grail diffusion policy enabled.", file=sys.stderr)
        # --per-region-cads: CADS condition noise keeps a box layout's regions from collapsing into one another early.
        cads_layout = getattr(args, "per_region_cads", False) and getattr(args, "_box_layout_spec", None) is not None
        if cads_layout and holy["holy_grail_cads_strength"] <= 0.0:
            holy["holy_grail_cads_strength"] = 0.15
        schedule = getattr(args, "guidance_schedule", None)
        schedule = schedule.strip() if isinstance(schedule, str) and schedule.strip() else None
        if holy["holy_grail_enable"] and schedule:
            print("--guidance-schedule ignored when holy-grail is enabled.", file=sys.stderr)
            schedule = None
        if init.flow:
            print(
                f"Using flow-matching sampler (solver={getattr(args, 'flow_solver', 'dpmpp_2m') or 'dpmpp_2m'}); "
                "pair with checkpoints trained using --flow-matching-training.",
                file=sys.stderr,
            )
        flow_shift = getattr(args, "flow_shift", None)
        if flow_shift is None:
            flow_shift = getattr(m.cfg, "flow_shift", None) or 1.0
        amp_on, amp_dtype = resolve_inference_amp(args, m.device)
        self.common = dict(
            model_kwargs_cond=cond.cond,
            model_kwargs_uncond=cond.uncond,
            cfg_scale=cfg_scale,
            cfg_rescale=cfg_rescale,
            eta=getattr(args, "eta", None),
            device=m.device,
            dtype=torch.float32,
            dynamic_threshold_percentile=dyn_p,
            dynamic_threshold_type=getattr(args, "dynamic_threshold_type", "percentile"),
            dynamic_threshold_value=getattr(args, "dynamic_threshold_value", 0.0),
            solver=getattr(args, "solver", "dpmpp_2m"),
            karras_rho=float(getattr(args, "karras_rho", 7.0)),
            volatile_cfg_boost=_f(args, "volatile_cfg_boost", 0.0),
            volatile_cfg_quantile=_f(args, "volatile_cfg_quantile", 0.72),
            volatile_cfg_window=int(getattr(args, "volatile_cfg_window", 6) or 6),
            flow_matching_sample=bool(init.flow),
            flow_solver=str(getattr(args, "flow_solver", "dpmpp_2m") or "dpmpp_2m"),
            flow_schedule=str(getattr(args, "flow_schedule", "ays") or "ays"),
            flow_karras_rho=_f(args, "flow_karras_rho", 7.0),
            # SD3 resolution shift: the trained shift at the native grid, rescaled for each sampled grid.
            flow_shift=float(flow_shift),
            flow_shift_ref_hw=(m.latent_size, m.latent_size),
            sde_band=getattr(args, "sde_band", None),
            eps_scale=float(getattr(args, "eps_scale", 1.0)),
            eps_scale_k=float(getattr(args, "eps_scale_k", 0.0)),
            control_guidance_start=_f(args, "control_guidance_start", 0.0),
            control_guidance_end=_f(args, "control_guidance_end", 1.0),
            control_guidance_decay=_f(args, "control_guidance_decay", 1.0),
            inference_amp=amp_on,
            inference_amp_dtype=amp_dtype,
            cfg_guidance_schedule=schedule,
            cfg_guidance_linear_start_multiplier=float(getattr(args, "guidance_schedule_linear_start", 0.7)),
            cfg_guidance_linear_end_multiplier=float(getattr(args, "guidance_schedule_linear_end", 1.0)),
            cfg_guidance_cosine_min_multiplier=float(getattr(args, "guidance_schedule_cosine_min", 0.65)),
            cfg_guidance_cosine_max_multiplier=float(getattr(args, "guidance_schedule_cosine_max", 1.0)),
            regional_cfg_plan=getattr(args, "_regional_cfg_plan", None),
            **holy,
            **_guidance_stack_kwargs(args),
        )
        # Trajectories that start from noise (or an img2img start) use the user's schedule and SAG / CLIP monitor.
        self.trajectory = dict(
            scheduler=getattr(args, "scheduler", "ays_dit"),
            timestep_schedule=getattr(args, "timestep_schedule", None),
            sag_blur_sigma=_f(args, "sag_blur_sigma", 0.0),
            sag_scale=_f(args, "sag_scale", 0.0),
            **self._clip_monitor_kwargs(),
        )
        early_exit = bool(getattr(args, "ada_early_exit", False))
        self.early_exit = dict(
            ada_early_exit_delta_threshold=getattr(args, "ada_exit_delta_threshold", 0.0) if early_exit else 0.0,
            ada_early_exit_patience=int(getattr(args, "ada_exit_patience", 0)) if early_exit else 0,
            ada_early_exit_min_steps=int(getattr(args, "ada_exit_min_steps", 0)),
        )

    def _clip_monitor_kwargs(self) -> dict:
        """``--clip-monitor-every``: mid-loop CLIP cosine check that boosts CFG (and optionally rewinds) when low."""
        args, m = self.args, self.m
        every = int(getattr(args, "clip_monitor_every", 0) or 0)
        if every <= 0:
            return {}
        from utils.generation.guidance import latent_x0_clip_cosine

        model_id = str(getattr(args, "clip_guard_model", "openai/clip-vit-base-patch32"))

        def clip_fn(_step: int, x0p: torch.Tensor) -> float:
            return latent_x0_clip_cosine(
                x0p,
                prompt=self.plan.prompt,
                model_id=model_id,
                device=m.device,
                vae=m.vae,
                latent_scale=m.latent_scale,
                ae_type=m.ae_type,
                rae_bridge=m.rae_bridge,
            )

        return dict(
            periodic_alignment_interval=every,
            periodic_alignment_threshold=_f(args, "clip_monitor_threshold", 0.22),
            periodic_alignment_cfg_boost=_f(args, "clip_monitor_cfg_boost", 0.12),
            periodic_alignment_fn=clip_fn,
            periodic_alignment_rewind_strength=_f(args, "clip_monitor_rewind", 0.0),
        )

    def run(self, shape: tuple, **kw: Any) -> Any:
        with torch.inference_mode():
            return self.m.diffusion.sample_loop(self.m.model, shape, **{**self.common, **kw})

    def resolve_from(self, x_t: torch.Tensor, t_s: int, n_steps: int, cfg: float | None = None) -> torch.Tensor:
        """Backward solve from the noised latent ``x_t`` at timestep ``t_s`` with the run's sampler settings."""
        return self.run(
            x_t.shape,
            cfg_scale=self.cfg_scale if cfg is None else cfg,
            num_inference_steps=max(4, int(n_steps)),
            x_init=x_t,
            start_timestep=t_s,
            scheduler="euler",
        )

    def renoise_resolve(self, x0: torch.Tensor, t_s: int, n_steps: int, cycles: int = 1) -> torch.Tensor:
        """Forward-noise a clean latent to ``t_s`` and solve back (one Restart cycle, VP or flow)."""
        return self.m.diffusion.restart(
            x0, lambda x_t, t: self.resolve_from(x_t, t, n_steps), t_max=t_s, cycles=cycles, flow=self.init.flow
        )

    def renoise_refine(self, x_lo: torch.Tensor, lh: int, lw: int, strength: float, n_steps: int, cfg: float):
        """Upscale + re-noise a clean latent for the ``lh x lw`` grid, then denoise it (hires-fix, dual-stage)."""
        x_t, t0 = renoise_for_grid(
            self.m.diffusion,
            x_lo,
            lh,
            lw,
            strength,
            native_latent_hw=(self.m.latent_size, self.m.latent_size),
            flow=self.init.flow,
        )
        print(f"Refine pass at latent {lh}x{lw}: t_start={t0}, {n_steps} steps, cfg={cfg}.", file=sys.stderr)
        return self.resolve_from(x_t, t0, n_steps, cfg)

    def pick_preview(self, x0_pred: torch.Tensor, metric: str) -> tuple[int, list[float]]:

        with torch.no_grad():
            return pick_indices(self.args, self.plan, to_rgb_u8(self.m.decode_unit(x0_pred)), metric, self.m.device)


def _holy_grail_kwargs(args: Any) -> dict:
    return dict(
        holy_grail_enable=bool(getattr(args, "holy_grail", False)),
        holy_grail_cfg_early_ratio=_f(args, "holy_grail_cfg_early_ratio", 0.72),
        holy_grail_cfg_late_ratio=_f(args, "holy_grail_cfg_late_ratio", 1.0),
        holy_grail_control_mult=_f(args, "holy_grail_control_mult", 1.0),
        holy_grail_adapter_mult=_f(args, "holy_grail_adapter_mult", 1.0),
        holy_grail_frontload_control=not bool(getattr(args, "holy_grail_no_frontload_control", False)),
        holy_grail_late_adapter_boost=_f(args, "holy_grail_late_adapter_boost", 1.15),
        holy_grail_cads_strength=_f(args, "holy_grail_cads_strength", 0.0),
        holy_grail_cads_min_strength=_f(args, "holy_grail_cads_min_strength", 0.0),
        holy_grail_cads_power=_f(args, "holy_grail_cads_power", 1.0),
        holy_grail_unsharp_sigma=_f(args, "holy_grail_unsharp_sigma", 0.0),
        holy_grail_unsharp_amount=_f(args, "holy_grail_unsharp_amount", 0.0),
        holy_grail_clamp_quantile=_f(args, "holy_grail_clamp_quantile", 0.0),
        holy_grail_clamp_floor=_f(args, "holy_grail_clamp_floor", 1.0),
    )


def _guidance_stack_kwargs(args: Any) -> dict:
    """APG / FDG / CFG-Zero* / interval / TCFG / PAG / SLG, feature + block caches, QSilk, dynamic width."""
    return dict(
        fdg_low=float(getattr(args, "fdg_low", 1.0)),
        fdg_high=float(getattr(args, "fdg_high", 1.0)),
        feature_cache_delta_threshold=_f(args, "feature_cache_delta", 0.0),
        feature_cache_max_reuse=int(getattr(args, "feature_cache_max_reuse", 2) or 2),
        block_cache_threshold=_f(args, "block_cache_thresh", 0.0),
        block_cache_recompute_every=int(getattr(args, "block_cache_recompute_every", 4) or 4),
        taylor_cache=bool(getattr(args, "taylor_cache", False)),
        taylor_cache_order=int(getattr(args, "taylor_cache_order", 1) or 1),
        taylor_cache_interval=int(getattr(args, "block_cache_recompute_every", 4) or 4),
        apg_parallel_eta=float(getattr(args, "apg_parallel_eta", -1.0)),
        apg_norm_threshold=float(getattr(args, "apg_norm_threshold", 0.0)),
        cfg_zero_star=bool(getattr(args, "cfg_zero_star", False)),
        cfg_zero_init_frac=float(getattr(args, "cfg_zero_init_frac", 0.04)),
        qsilk_micrograin=_f(args, "qsilk_micrograin", 0.0),
        dynamic_dit_width=bool(getattr(args, "dynamic_dit_width", False)),
        dynamic_dit_early=_f(args, "dynamic_dit_early", 0.88),
        dynamic_sdt=bool(getattr(args, "dynamic_sdt", False)),
        apg_momentum_beta=float(getattr(args, "apg_momentum_beta", 0.0)),
        cfg_interval=getattr(args, "cfg_interval", None) or None,
        tcfg_damping=_f(args, "tcfg_damping", 0.0),
        pag_scale=_f(args, "pag_scale", 0.0),
        pag_blocks=str(getattr(args, "pag_blocks", "auto") or "auto"),
        slg_scale=_f(args, "slg_scale", 0.0),
        slg_skip_blocks=str(getattr(args, "slg_skip_blocks", "auto") or "auto"),
        cfg_rejection_rerank=bool(getattr(args, "cfg_rejection_rerank", False))
        and int(getattr(args, "num", 1) or 1) > 1,
        dbc_separate_cfg=bool(getattr(args, "dbc_separate_cfg", False)) or _f(args, "block_cache_thresh", 0.0) > 0.0,
    )


def _beam_metric(args: Any, *names: str) -> str:
    """First of ``names`` that is set (not auto / none), else the ViT combo critic."""
    for name in names:
        value = str(getattr(args, name, "") or "").strip().lower()
        if value and value not in ("auto", "none"):
            return value
    return "combo_vit_hq" if str(getattr(args, "pick_vit_ckpt", "") or "").strip() else "combo_vit"


def _dual_stage(s: Sampler) -> torch.Tensor | None:
    """``--dual-stage-layout``: compose on a coarse grid, then a detail pass at the canvas grid."""
    args, m, init = s.args, s.m, s.init
    if not bool(getattr(args, "dual_stage_layout", False)):
        return None
    if m.ae_type != "kl" or args.mask or args.init_image or args.init_latent or init.x_init is not None:
        print("Dual-stage layout disabled (need KL VAE, no inpaint/img2img/from-z/init).", file=sys.stderr)
        return None
    div = max(2, int(getattr(args, "dual_stage_div", 2) or 2))
    patch = dit_patch_size(m.model)
    ch, cw = snap_to_multiple(round(m.lat_h / div), patch), snap_to_multiple(round(m.lat_w / div), patch)
    layout_steps = int(getattr(args, "dual_layout_steps", 24) or 24)
    print(f"Dual-stage: layout latent {ch}x{cw} ({layout_steps} steps) -> {m.lat_h}x{m.lat_w}...", file=sys.stderr)
    x0 = s.run((s.plan.num_gen, 4, ch, cw), num_inference_steps=layout_steps, **s.trajectory, **s.feature_kw)
    with torch.no_grad():
        return s.renoise_refine(
            x0,
            m.lat_h,
            m.lat_w,
            float(getattr(args, "dual_detail_strength", 0.38)),
            int(getattr(args, "dual_detail_steps", 20) or 20),
            s.cfg_scale,
        )


def _single_trajectory_ok(s: Sampler) -> bool:
    return s.plan.num_gen == 1 and not s.init.flow and s.init.inpaint_mask is None


def _beam_search(s: Sampler, report: dict) -> int:
    """``--beam-width/--beam-steps``: denoise several seeds a few steps, keep the best preview; returns steps used."""
    args, init = s.args, s.init
    width, steps = int(getattr(args, "beam_width", 0) or 0), int(getattr(args, "beam_steps", 0) or 0)
    if not (width > 1 and steps > 0 and _single_trajectory_ok(s) and init.x_init is None):
        return 0
    try:
        metric = _beam_metric(args, "beam_metric", "pick_best")
        print(f"Beam search: width={width} steps={steps} metric={metric}", file=sys.stderr)
        x_t, x0_pred, t_resume = s.run(
            (width, s.m.latent_channels, s.m.lat_h, s.m.lat_w),
            num_inference_steps=steps,
            model_kwargs_cond=_with_batch(s.c.cond, width),
            model_kwargs_uncond=_with_batch(s.c.uncond, width),
            return_intermediate_state=True,
            **s.trajectory,
            **s.early_exit,
        )
        best, scores = s.pick_preview(x0_pred, metric)
    except Exception as e:
        print(f"Beam search disabled (fallback): {e}", file=sys.stderr)
        return 0
    print(f"beam scores={scores} -> keep {best} resume_t={t_resume}", file=sys.stderr)
    report["beam1"] = {
        "metric": metric,
        "scores": [float(x) for x in scores],
        "best_index": int(best),
        "resume_t": int(t_resume),
        "beam_width": width,
        "beam_steps": steps,
    }
    init.x_init, init.start_timestep = x_t[best : best + 1].contiguous(), int(t_resume)
    return steps


def _micro_beam(s: Sampler, steps_main: int, report: dict) -> int:
    """``--beam2-*``: branch from a mid-trajectory latent with small noise and re-pick; returns steps left."""
    args, init = s.args, s.init
    width, steps = int(getattr(args, "beam2_width", 0) or 0), int(getattr(args, "beam2_steps", 0) or 0)
    at, noise = _f(args, "beam2_at_frac", 0.65), _f(args, "beam2_noise", 0.03)
    if not (width > 1 and steps > 0 and _single_trajectory_ok(s) and steps_main >= steps + 2):
        return steps_main
    try:
        metric = _beam_metric(args, "beam2_metric", "beam_metric", "pick_best")
        split = int(round(float(steps_main) * min(0.95, max(0.05, at))))
        split = max(1, min(split, steps_main - (steps + 1)))
        shape = (1, s.m.latent_channels, s.m.lat_h, s.m.lat_w)
        x_mid, _, t_mid = s.run(
            shape,
            num_inference_steps=split,
            x_init=init.x_init,
            start_timestep=init.start_timestep,
            return_intermediate_state=True,
            **s.trajectory,
            **s.early_exit,
        )
        x_seed = x_mid.expand(width, -1, -1, -1).contiguous()
        if noise > 0.0:
            x_seed = x_seed + noise * torch.randn_like(x_seed)
        x2, x0_2, t2 = s.run(
            (width, *shape[1:]),
            num_inference_steps=steps,
            x_init=x_seed,
            start_timestep=int(t_mid),
            model_kwargs_cond=_with_batch(s.c.cond, width),
            model_kwargs_uncond=_with_batch(s.c.uncond, width),
            return_intermediate_state=True,
            **s.trajectory,
            **s.early_exit,
        )
        best, scores = s.pick_preview(x0_2, metric)
    except Exception as e:
        print(f"Micro-beam disabled (fallback): {e}", file=sys.stderr)
        return steps_main
    print(
        f"Micro-beam: split={split}/{steps_main} width={width} steps={steps} "
        f"metric={metric} scores={scores} -> keep {best} resume_t={t2}",
        file=sys.stderr,
    )
    report["beam2"] = {
        "metric": metric,
        "scores": [float(x) for x in scores],
        "best_index": int(best),
        "resume_t": int(t2),
        "beam2_width": width,
        "beam2_steps": steps,
        "beam2_at_frac": at,
        "beam2_noise": noise,
        "split_steps": split,
    }
    init.x_init, init.start_timestep = x2[best : best + 1].contiguous(), int(t2)
    return max(1, steps_main - split - steps)


def _denoise(s: Sampler, report: dict) -> torch.Tensor:
    """Main trajectory, after the optional beam searches."""
    args, init = s.args, s.init
    print(f"Sampling (steps={args.steps}, num={s.plan.num_gen}, cfg_scale={s.cfg_scale})...")
    steps_main = int(args.steps) - _beam_search(s, report)
    steps_main = _micro_beam(s, max(1, steps_main), report)
    return s.run(
        (s.plan.num_gen, s.m.latent_channels, s.m.lat_h, s.m.lat_w),
        num_inference_steps=steps_main,
        x_init=init.x_init,
        start_timestep=init.start_timestep,
        inpaint_mask=init.inpaint_mask,
        inpaint_x0=init.inpaint_x0,
        inpaint_noise=init.inpaint_noise,
        inpaint_freeze_known=bool(args.mask and args.inpaint_mode == "mdm"),
        **s.trajectory,
        **s.early_exit,
        **s.feature_kw,
    )


def _refine_latents(s: Sampler, x0: torch.Tensor) -> torch.Tensor:
    """Post-trajectory passes: CFG-rejection rerank, CLIP guard, self-correct, latent priors, hires-fix, Restart."""
    args, m = s.args, s.m
    if bool(getattr(args, "cfg_rejection_rerank", False)) and s.plan.num_gen > 1:
        probe = getattr(m.diffusion, "_last_guidance_probe", None)
        order = probe.rerank_indices() if probe is not None else None
        if order and len(order) == int(x0.shape[0]):
            x0 = x0[order]
            print(f"CFG-rejection rerank order={order}", file=sys.stderr)

    if _f(args, "clip_guard_threshold", 0.0) > 0.0:
        try:
            from utils.generation.guidance import maybe_clip_refine_latent

            x0 = maybe_clip_refine_latent(
                x0,
                prompt=s.plan.prompt,
                sim_threshold=float(args.clip_guard_threshold),
                model_id=str(getattr(args, "clip_guard_model", "openai/clip-vit-base-patch32")),
                device=m.device,
                vae=m.vae,
                latent_scale=m.latent_scale,
                ae_type=m.ae_type,
                rae_bridge=m.rae_bridge,
                num_timesteps=s.init.num_timesteps,
                t_frac=float(getattr(args, "clip_guard_t_frac", 0.22)),
                refine_steps=int(getattr(args, "clip_guard_steps", 12)),
                refiner=s.renoise_resolve,
            )
        except Exception as e:
            print(f"CLIP guard refine skipped: {e}", file=sys.stderr)
    elif getattr(args, "superior_self_correct", False):
        try:
            from utils.agentic.quality_loop import SelfCorrectConfig, SelfCorrectPolicy

            policy = SelfCorrectPolicy(
                SelfCorrectConfig(clip_model_id=str(getattr(args, "pick_clip_model", "openai/clip-vit-base-patch32")))
            )
            score = policy.score(
                x0,
                s.plan.prompt,
                vae=m.vae,
                latent_scale=m.latent_scale,
                ae_type=m.ae_type,
                rae_bridge=m.rae_bridge,
                device=m.device,
            )
            if policy.needs_correction(score):
                t_s, n_st = policy.refine_plan(s.init.num_timesteps)
                x0 = s.renoise_resolve(x0, t_s, n_st)
                policy.note_refinement_done()
                print(f"Superior self-correct: refined (score={score:.3f}, t={t_s})", file=sys.stderr)
        except Exception as e:
            print(f"Superior self-correct skipped: {e}", file=sys.stderr)

    if _f(args, "domain_prior_latent", 0.0) > 0.0:
        from utils.generation.guidance import highfreq_layout_prior

        x0 = highfreq_layout_prior(x0, strength=float(args.domain_prior_latent))
    if _f(args, "spectral_coherence_latent", 0.0) > 0.0:
        from utils.generation.guidance import spectral_latent_lowfreq_blend

        x0 = spectral_latent_lowfreq_blend(
            x0,
            strength=float(args.spectral_coherence_latent),
            cutoff_frac=_f(args, "spectral_coherence_cutoff", 0.15),
        )
    x0 = _hires_fix(s, x0)
    if str(getattr(args, "refine", "off") or "off") == "restart" and not getattr(args, "no_refine", False):
        if _refine_gate_open(s, x0):
            x0 = s.renoise_resolve(
                x0,
                int(getattr(args, "refine_t", 200)),
                int(getattr(args, "restart_steps", 6) or 6),
                cycles=int(getattr(args, "restart_cycles", 2) or 2),
            )
    return x0


def _hires_fix(s: Sampler, x0: torch.Tensor) -> torch.Tensor:
    """``--hires-fix``: upscale the clean latent to the requested size, re-noise and denoise it there."""
    args, m = s.args, s.m
    if not getattr(args, "hires_fix", False):
        return x0
    if m.ae_type != "kl":
        print("Hires-fix skipped: only supported for KL VAE checkpoints.", file=sys.stderr)
        return x0
    if args.mask or args.init_image or args.init_latent:
        print("Hires-fix skipped: not combined with inpaint / img2img / from-z.", file=sys.stderr)
        return x0
    if int(getattr(args, "width", 0) or 0) > 0 or int(getattr(args, "height", 0) or 0) > 0:
        th_px, tw_px = m.req_h, m.req_w
    else:
        scale = max(_f(args, "hires_scale", 1.5), 1.0)
        th_px, tw_px = round(m.canvas_h * scale), round(m.canvas_w * scale)
    stride = dit_rgb_stride_px(m.model)
    tlh, tlw = snap_to_multiple(th_px, stride) // 8, snap_to_multiple(tw_px, stride) // 8
    if tlh * tlw <= x0.shape[-2] * x0.shape[-1]:
        print(f"Hires-fix skipped: latent already {x0.shape[-2]}x{x0.shape[-1]} (target {tlh}x{tlw}).", file=sys.stderr)
        return x0
    cfg_h = float(getattr(args, "hires_cfg_scale", -1.0))
    with torch.no_grad():
        return s.renoise_refine(
            x0,
            tlh,
            tlw,
            float(getattr(args, "hires_strength", 0.35)),
            max(1, int(getattr(args, "hires_steps", 15))),
            float(s.cfg_scale) if cfg_h < 0 else cfg_h,
        )


def _refine_gate_open(s: Sampler, x0: torch.Tensor) -> bool:
    """``--refine-gate auto``: skip the Restart refine when a cheap preview already scores well."""
    args = s.args
    if str(getattr(args, "refine_gate", "off") or "off").lower() != "auto":
        return True
    try:
        with torch.no_grad():
            rgb = to_rgb_u8(s.m.decode_unit(x0[:1]))[0]
        score, details = _refine_gate_score(image_rgb_u8=rgb, expected_texts=s.plan.expected_texts)
    except Exception as e:
        print(f"refine-gate fallback: {e}", file=sys.stderr)
        return True
    thr = _f(args, "refine_gate_threshold", 0.62)
    print(f"refine-gate: score={score:.3f} thr={thr:.3f} run_refine={score < thr} details={details}", file=sys.stderr)
    return score < thr


def export_attention(args: Any, m: Any, cond: Any, x0: torch.Tensor) -> None:
    """``--save-attn`` / ``--explain-adherence``: cross-attention maps at t=0 for the first sample."""
    adherence_path = str(getattr(args, "explain_adherence", "") or "").strip()
    if not (args.save_attn or adherence_path):
        return
    with torch.no_grad():
        out = m.model(x0[:1], torch.zeros(1, device=m.device, dtype=torch.long), return_attn=True, **cond.cond)
    if not isinstance(out, tuple):
        print("Model does not support attention export; skipping.", file=sys.stderr)
        return
    _, attn = out
    if args.save_attn:
        torch.save({"attn": attn.cpu(), "prompt": args.prompt}, args.save_attn)
        print(f"Saved attention: {args.save_attn}")
    if adherence_path:
        from utils.generation.sample_prompt import export_adherence_heatmap

        export_adherence_heatmap(
            attn,
            args.prompt or "",
            adherence_path,
            latent_h=int(x0.shape[-2]),
            latent_w=int(x0.shape[-1]),
            box_spec=getattr(args, "_box_layout_spec", None),
        )
        print(f"Saved adherence heatmap: {adherence_path}")


def _injects_noise(args: Any, flow: bool) -> bool:
    """Whether the solver adds noise each step; frontier ``step_noise_scales`` curves only scale that noise."""
    from diffusion.gaussian_diffusion import canonicalize_flow_solver, canonicalize_vp_solver

    if flow:
        solver = canonicalize_flow_solver(str(getattr(args, "flow_solver", "dpmpp_2m") or "dpmpp_2m"))
        stochastic = ("dpmpp_2m_sde",)
    else:
        solver = canonicalize_vp_solver(str(getattr(args, "solver", "dpmpp_2m") or "dpmpp_2m"))
        stochastic = ("ddim", "heun", "dpmpp_2m_sde")
    eta = getattr(args, "eta", None)
    eta = (1.0 if solver == "dpmpp_2m_sde" else 0.0) if eta is None else float(eta)
    return solver in stochastic and eta > 0.0


def sample_latents(args: Any, m: Any, plan: Any, cond: Any, init: Any, report: dict) -> torch.Tensor:
    """Denoise (dual-stage, beam or plain), then refine in latent space; returns clean DiT latents."""
    from utils.generation.guidance import sanitize_diffusion_feature_kwargs
    from utils.generation.sample_prompt import apply_sample_feature_diffusion_phase

    feature_kw = sanitize_diffusion_feature_kwargs(
        apply_sample_feature_diffusion_phase(args, steps=int(getattr(args, "steps", 50) or 50))
    )
    if "step_noise_scales" in feature_kw and not _injects_noise(args, init.flow):
        del feature_kw["step_noise_scales"]
        print(
            "Frontier noise curve skipped: the solver injects no noise (use --eta > 0 with ddim, heun or "
            "dpmpp_2m_sde).",
            file=sys.stderr,
        )
    s = Sampler(args, m, plan, cond, init, feature_kw)
    x0 = _dual_stage(s)
    if x0 is None:
        x0 = _denoise(s, report)
    return _refine_latents(s, x0)


_log = logging.getLogger(__name__)


def tensor_bchw01_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a single batch image tensor ``(1, 3, H, W)`` in ``[0, 1]`` to RGB PIL."""
    if image.dim() != 4 or int(image.shape[0]) != 1 or int(image.shape[1]) != 3:
        raise ValueError("expected tensor shape (1, 3, H, W)")
    img_np = image[0].permute(1, 2, 0).detach().float().cpu().numpy()
    img_np = (img_np * 255.0).clip(0.0, 255.0).round().astype("uint8")
    return Image.fromarray(img_np, mode="RGB")


@torch.no_grad()
def sample_one_image_pil(
    *,
    model: torch.nn.Module,
    diffusion: Any,
    tokenizer: Any,
    text_encoder: torch.nn.Module,
    vae: torch.nn.Module,
    device: str | torch.device,
    prompt: str,
    negative_prompt: str = "",
    image_size: int = 512,
    cfg_scale: float = 7.5,
    num_inference_steps: int = 50,
    latent_scale: float = 0.18215,
    ae_type: str = "kl",
    max_length: int = 300,
    dtype: torch.dtype = torch.float32,
    text_bundle: Any = None,
    seed: int | None = None,
    rae_bridge: Any = None,
    size_embed_dim: int = 0,
    eval_mode: bool = True,
    **sample_loop_kwargs: Any,
) -> Image.Image:
    """
    Encode ``prompt`` / ``negative_prompt``, run ``diffusion.sample_loop``, decode with ``vae``.

    Extra keyword arguments are forwarded to ``diffusion.sample_loop`` (e.g. ``scheduler``,
    ``solver``, ``eta``). Models that use ``size_embed`` must pass a non-zero
    ``size_embed_dim`` matching the checkpoint (same convention as ``sample.py``).
    """
    # Local import keeps `utils.generation` importable without loading `sample` at package import time.
    from sample import encode_text

    from utils.modeling.encoders import encode_kwargs_for_captions

    dev = torch.device(device) if isinstance(device, str) else device
    if seed is not None:
        torch.manual_seed(int(seed))
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

    if eval_mode:
        model.eval()
        vae.eval()
        text_encoder.eval()

    negative_prompt = require_allowed(prompt, negative_prompt)
    if cfg_scale < 1.0:
        require_allowed(f"{prompt}, {negative_prompt}")
    both = encode_text(
        [prompt, negative_prompt],
        tokenizer,
        text_encoder,
        dev,
        max_length=max_length,
        dtype=dtype,
        text_bundle=text_bundle,
        **encode_kwargs_for_captions([prompt, negative_prompt], text_bundle),
    )
    cond, uncond = both[0:1], both[1:2]
    model_kwargs_cond: dict[str, Any] = {"encoder_hidden_states": cond}
    model_kwargs_uncond: dict[str, Any] = {"encoder_hidden_states": uncond}

    if int(size_embed_dim or 0) > 0:
        lh = max(1, int(image_size) // 8)
        lw = max(1, int(image_size) // 8)
        bsz = int(cond.shape[0])
        sz = torch.tensor([[float(lh), float(lw)]], device=dev, dtype=torch.float32).expand(bsz, -1)
        model_kwargs_cond["size_embed"] = sz
        model_kwargs_uncond["size_embed"] = sz

    latent_size = max(1, int(image_size) // 8)
    shape = (1, 4, latent_size, latent_size)

    loop_kw = dict(sample_loop_kwargs)
    eta = float(loop_kw.pop("eta", 0.0))

    x0 = diffusion.sample_loop(
        model,
        shape,
        model_kwargs_cond=model_kwargs_cond,
        model_kwargs_uncond=model_kwargs_uncond,
        cfg_scale=float(cfg_scale),
        num_inference_steps=int(num_inference_steps),
        eta=eta,
        device=dev,
        dtype=dtype,
        **loop_kw,
    )

    if ae_type == "kl":
        x0 = x0 / float(latent_scale)
    elif ae_type == "rae" and rae_bridge is not None:
        x0 = rae_bridge.dit_to_rae(x0)
    else:
        if ae_type == "rae" and rae_bridge is None:
            _log.warning("ae_type='rae' but rae_bridge is None; decoding DiT latent as-is")

    decoded = vae.decode(x0)
    image = decoded.sample if hasattr(decoded, "sample") else decoded
    image = (image * 0.5 + 0.5).clamp(0, 1)
    pil = tensor_bchw01_to_pil(image)
    require_image_allowed(pil, sexual=is_sexual(prompt) or (cfg_scale < 1.0 and is_sexual(negative_prompt)))
    return pil



def explicit_flags(parser: Any, argv: list[str]) -> frozenset[str]:
    """Destinations of the options the user typed (``--flag``, ``--flag=v``, unique ``--fl`` prefixes)."""
    options = {opt: action.dest for action in parser._actions for opt in action.option_strings}
    dests = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0]
        dest = options.get(flag)
        if dest is None:
            hits = {d for opt, d in options.items() if opt.startswith(flag)}
            dest = hits.pop() if len(hits) == 1 else None
        if dest is not None:
            dests.add(dest)
    return frozenset(dests)


def keep_explicit(args: Any, apply: Any, *apply_args: Any) -> None:
    """Run a soft-default applier, then restore every flag the user typed (text it only extended is kept)."""
    typed = {k: getattr(args, k) for k in getattr(args, "_explicit_flags", ()) if hasattr(args, k)}
    apply(args, *apply_args)
    for k, v in typed.items():
        new = getattr(args, k)
        if isinstance(v, str) and v and isinstance(new, str) and new.startswith(v):
            continue
        setattr(args, k, v)


def _expand_presets(args: Any) -> None:
    from config.presets import apply_op_mode_to_args, apply_preset_to_args
    from diffusion.sampling import apply_holy_grail_preset_to_args, recommend_holy_grail_preset

    done = args.__dict__.setdefault("_expanded_presets", set())
    if getattr(args, "preset", None) and ("preset", args.preset) not in done:
        apply_preset_to_args(args, args.preset)
        done.add(("preset", args.preset))
    if getattr(args, "op_mode", None) and ("op_mode", args.op_mode) not in done:
        apply_op_mode_to_args(args, args.op_mode)
        done.add(("op_mode", args.op_mode))
    hg = str(getattr(args, "holy_grail_preset", None) or "").strip().lower()
    if hg and ("holy_grail", hg) not in done:
        name = hg
        if hg == "auto":
            name = recommend_holy_grail_preset(
                prompt=str(getattr(args, "prompt", "") or ""),
                style=str(getattr(args, "style", "") or ""),
                has_control=bool(getattr(args, "control_image", "") or getattr(args, "control", [])),
                has_lora=bool(getattr(args, "lora", [])),
            )
        apply_holy_grail_preset_to_args(args, name)
        done.add(("holy_grail", hg))


def expand_presets(args: Any) -> None:
    """Soft-apply ``--preset`` / ``--op-mode`` / ``--holy-grail-preset`` (``auto`` reads the prompt), once each."""
    keep_explicit(args, _expand_presets)


def parse_args(parser: Any, argv: list[str]) -> Any:
    args = parser.parse_args(argv)
    args._explicit_flags = explicit_flags(parser, argv)
    return args


def run_early_exit(args: Any, argv: list[str]) -> bool:
    """``--export-comfy-workflow`` and ``--auto-refine N`` (N child runs of ``sample.py``); True when handled."""
    if str(getattr(args, "export_comfy_workflow", "") or "").strip():
        from utils.generation.sample_argv import sample_args_to_comfy_workflow, write_comfy_workflow

        out = write_comfy_workflow(str(args.export_comfy_workflow).strip(), sample_args_to_comfy_workflow(vars(args)))
        print(f"Wrote Comfy workflow: {out}")
        return True
    n = int(getattr(args, "auto_refine", 0) or 0)
    if n <= 1:
        return False
    from utils.generation.sample_prompt import run_auto_refine_candidates, run_creative_refine

    stripped, skip = [], False
    for tok in argv:
        if skip or tok == "--auto-refine" or tok.startswith("--auto-refine="):
            skip = tok == "--auto-refine"
            continue
        stripped.append(tok)
    cmd = [sys.executable, str(SAMPLE_PY), *stripped]
    stem = str(Path(args.out).with_suffix("")) + "_ar"
    seed = int(getattr(args, "seed", 42) or 42)
    mutate = int(getattr(args, "creative_mutate", 0) or 0)
    if bool(getattr(args, "frontier_creative", False)) or mutate > 0:
        res = run_creative_refine(
            sample_cmd=cmd,
            base_prompt=str(getattr(args, "prompt", "") or ""),
            num_candidates=n,
            seed_base=seed,
            out_stem=stem,
            mutate_count=mutate if mutate > 0 else n,
            base_serendipity=float(getattr(args, "frontier_serendipity", 0.25) or 0.25),
        )
    else:
        res = run_auto_refine_candidates(sample_cmd=cmd, num_candidates=n, seed_base=seed, out_stem=stem)
    shutil.copy(res.outputs[res.best_index], args.out)
    best_prompt = res.prompts[res.best_index] if res.prompts else ""
    note = f" prompt={best_prompt[:80]!r}" if best_prompt else ""
    print(
        f"Auto-refine: kept candidate {res.best_index} (score={res.scores[res.best_index]:.3f}){note} -> {args.out}",
        file=sys.stderr,
    )
    return True


def _expand_explore_styles(args: Any) -> None:
    if getattr(args, "explore_styles_insane", False):
        args.explore_styles = True
        args.style_inventor_mode = "apocalypse"
        args.style_chaos_level = max(float(getattr(args, "style_chaos_level", 0) or 0), 0.95)
        args.invent_styles = max(int(getattr(args, "invent_styles", 0) or 0), 4)
        clauses = str(getattr(args, "prompt_clauses", "") or "").strip()
        insane = "style.chaos,style.surreal,style.apocalypse"
        args.prompt_clauses = f"{clauses},{insane}".strip(",") if clauses else insane
        print("Explore-styles-insane: apocalypse mode engaged.", file=sys.stderr)
    if getattr(args, "explore_styles", False):
        if int(getattr(args, "invent_styles", 0) or 0) < 1:
            args.invent_styles = 3
        if str(getattr(args, "pick_best", "none") or "none").strip().lower() in ("none", ""):
            args.pick_best = "combo"
        print(
            f"Explore-styles: inventing {args.invent_styles} genomes "
            f"(mode={getattr(args, 'style_inventor_mode', 'normal')}, "
            f"chaos={float(getattr(args, 'style_chaos_level', 0) or 0):.2f}); "
            f"use --style-genome-index 0..{args.invent_styles - 1} per run or `python -m scripts.tools prompts explore-styles`",
            file=sys.stderr,
        )


def resolve_quality_defaults(args: Any, report: dict) -> None:
    """User-requested expansions, art direction, then the quality policy and the presets it picks."""
    keep_explicit(args, _expand_explore_styles)

    from utils.generation.sample_core import apply_best_model_policy
    from utils.prompt.art_direction import apply_art_direction_to_args
    from utils.prompt.realism import apply_realism_to_args, apply_shot_spec_override

    apply_shot_spec_override(args)
    apply_art_direction_to_args(args)
    report["art_direction"] = args.art_direction_plan
    apply_realism_to_args(args)  # shot spec (camera, shutter, ISO) for photographic prompts
    report["shot_spec"] = args.shot_spec_plan
    keep_explicit(args, apply_best_model_policy)
    expand_presets(args)
    if str(getattr(args, "human_made", "none") or "none").lower() not in ("none", "off", "0", ""):
        from utils.quality.finishing import apply_human_made_prompt_flags

        keep_explicit(args, apply_human_made_prompt_flags)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_sample_parser()
    args = parse_args(parser, argv)
    jobs = str(getattr(args, "jobs_jsonl", "") or "").strip()
    if not jobs:
        _generate(parser, args, argv)
        return
    # Batch mode: later flags win in argparse, so each job's prompt/seed/out override the base argv.
    # ponytail: models load with the first job's args, so prompt-picked LoRA-bank adapters stick for the
    # whole batch; pass --no-lora-bank (rsi_loop does) or run separate calls if adapters must vary.
    m = None
    for line in Path(jobs).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        job = json.loads(line)
        job_argv = [*argv, "--prompt", str(job["prompt"]), "--seed", str(int(job.get("seed", 42))), "--out", job["out"]]
        m = _generate(parser, parse_args(parser, job_argv), job_argv, m)


def _generate(parser: Any, args: Any, argv: list[str], m: Any = None) -> Any:
    """One generation; pass ``m`` from a previous call to skip reloading checkpoint and encoders."""
    report: dict = {
        "prompt": str(getattr(args, "prompt", "") or ""),
        "negative_prompt": str(getattr(args, "negative_prompt", "") or ""),
        "seed": int(getattr(args, "seed", 0) or 0),
        "steps": int(getattr(args, "steps", 0) or 0),
        "num": int(getattr(args, "num", 0) or 0),
    }
    expand_presets(args)
    if run_early_exit(args, argv):
        return m
    if float(getattr(args, "linear_attn_fraction", 0.0) or 0.0) > 0.0:
        from models.attention import set_linear_attention_fraction

        set_linear_attention_fraction(float(args.linear_attn_fraction))

    compose_user_prompt(args, parser)
    resolve_quality_defaults(args, report)
    extras = compose_prompt_additions(args)
    if m is None:
        m = load_models(args)
    else:
        torch.manual_seed(int(args.seed))  # load_models seeds the first job; noise comes from the global RNG
    with torch.inference_mode():
        plan = build_prompt_plan(args, m, extras)
        cond = encode_conditioning(args, m, plan)
        init = prepare_init_latents(args, m, plan.num_gen)
        x0 = sample_latents(args, m, plan, cond, init, report)
        export_attention(args, m, cond, x0)

        images, out_size, fit = decode_images(args, m, x0)
        processed = postprocess(args, plan, images, out_size, fit)
        metric, best = choose_best(args, m, plan, processed, report)
        finals = capture_finish(args, plan, processed, report)
        finals, natives = game_finish(args, finals, report)
    require_image_allowed(finals, sexual=cond.sexual)  # reads RGB of RGBA sprites, so nothing hides under alpha
    rgb_finals = [f[..., :3] for f in finals]
    for i, path, label in save_targets(args, plan.num_gen, metric, best):
        save_game_image(
            finals[i],
            natives[i],
            path,
            sixteen_bit=bool(getattr(args, "save_16bit", False)),
            seed=int(args.seed) + i,
        )
        print(f"{label}: {path}")
    after_save(args, plan, metric, best, rgb_finals, report)
    return m
