"""Output phase of ``sample.py``: decode, per-candidate post, pick-best, capture finish, save and follow-ups."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from utils.generation.sample_helpers import SAMPLE_PY, _maybe_append_text_says


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
    from utils.quality.camera_isp import to_uint8

    try:
        from utils.image_resize import fit_image_to_size

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
        from utils.quality import (
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
        from utils.quality.artistic_post_process import ArtisticPostConfig, apply_artistic_pipeline

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
                from utils.quality import apply_photo_color_grade, apply_photo_filter

                strength = float(getattr(args, "photo_post_strength", 0.6) or 0.6)
                img = apply_photo_color_grade(img, preset=plan.photo_grade, strength=strength)
                img = apply_photo_filter(img, filter_name=plan.photo_filter, strength=strength)
            except Exception:
                pass
        img = _artistic(args, img, int(args.seed) + i + 1000)
        try:
            from utils.quality.human_made import apply_human_made_pipeline, config_from_args

            hm_cfg = config_from_args(args)
            if hm_cfg is not None:
                img = apply_human_made_pipeline(img, hm_cfg)
        except Exception as e:
            print(f"Human-made post-process failed (non-fatal): {e}", file=sys.stderr)
        if getattr(args, "face_enhance", False):
            try:
                from utils.quality.face_region_enhance import enhance_faces_in_rgb

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
                from utils.quality.face_region_enhance import blend_reference_rgb

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
    from utils.quality.art_critic import resolve_plan
    from utils.quality.camera_isp import to_uint8

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
    return metric, int(best)


def _capture_finish(args: Any, plan: Any, report: dict):
    """(finish plan or None, lens-only plan or None) for the medium-routed camera / film finish."""
    from utils.quality.camera_isp import NO_LENS, plan_finish, user_lens

    cam = str(getattr(args, "camera_isp", "auto") or "auto").lower()
    photo_post = bool(getattr(args, "photo_postprocess", True))
    wanted = (
        cam != "auto"
        or str(getattr(args, "human_made", "none") or "none").lower() not in ("none", "off", "0", "")
        or (bool(getattr(args, "naturalize", False)) and float(getattr(args, "naturalize_grain", 0.015) or 0.0) > 0)
        or (photo_post and plan.photo_grain.lower() != "none")
    )
    vignette = float(getattr(args, "vignette", 0.0) or 0.0)
    ca = float(getattr(args, "chromatic_aberration", 0.0) or 0.0)
    finish = None
    if wanted:
        finish = plan_finish(
            str(getattr(args, "_raw_prompt_before_compose", "") or args.prompt or ""),
            camera=cam,
            iso=int(getattr(args, "camera_iso", 0) or 0),
            grain_style=plan.photo_grain if photo_post else "none",
            human_media=str(getattr(args, "human_media_mode", "none") or "none"),
            vignette=vignette,
            chromatic_aberration=ca,
        )
    if finish is None:
        # No capture finish (not wanted, or a non-photographic medium): explicit lens knobs still get one
        # linear-light lens pass.
        return None, user_lens(NO_LENS, vignette, ca) if (vignette > 0 or ca > 0) else None
    print(f"Capture finish: {finish.name}, ISO {finish.iso}", file=sys.stderr)
    report["capture_finish"] = {"profile": finish.name, "iso": int(finish.iso)}
    return finish, None


def capture_finish(args: Any, plan: Any, processed: list, report: dict) -> list[np.ndarray]:
    """The camera / film finish (or lens-only pass), once per candidate after ranking; float ``[0, 255]``."""
    from utils.quality.camera_isp import apply_finish, apply_lens_srgb

    finish, lens_only = _capture_finish(args, plan, report)
    finals = []
    for i, img in enumerate(processed):
        if finish is not None:
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
        from utils.generation.sample_features import save_character_session

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
            from utils.training.feedback_bus import default_feedback_log, record_generation, record_pick

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
            from utils.quality.camera_isp import to_uint8

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
        from utils.generation.text_rendering import create_text_rendering_pipeline

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
        from utils.generation.sample_cli_passthrough import append_sample_repair_passthrough

        append_sample_repair_passthrough(cmd, args)
        print(f"OCR fix: acc={acc:.3f} < {threshold}; repairing with mdm inpainting...", file=sys.stderr)
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"OCR fix failed (non-fatal): {e}", file=sys.stderr)
        return
    # The repair run overwrote --out; nothing after it should touch the old image.
    sys.exit(0)


__all__ = [
    "after_save",
    "capture_finish",
    "choose_best",
    "decode_images",
    "pick_indices",
    "pick_metric",
    "postprocess",
    "save_targets",
    "to_rgb_u8",
]
