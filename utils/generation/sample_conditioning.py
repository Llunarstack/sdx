"""Conditioning phase of ``sample.py``: text embeddings, model kwargs, control / reference images, init latents."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from utils.generation.sample_helpers import (
    _T5_CACHE_MAX,
    TEXT_MAX_TOKENS,
    _maybe_rae_to_dit,
    _parse_control_spec,
    _parse_lora_spec,
    _parse_weighted_style_mix,
    _t5_cache,
    encode_text,
)
from utils.safety.minor_guard import is_sexual, require_allowed, require_image_allowed


@dataclass
class Conditioning:
    cond: dict
    uncond: dict
    sexual: bool


@dataclass
class InitLatents:
    """Starting point of the trajectory: pure noise (all None), img2img / from-z, or structured inpaint."""

    flow: bool
    num_timesteps: int
    x_init: torch.Tensor | None = None
    start_timestep: int | None = None
    inpaint_mask: torch.Tensor | None = None
    inpaint_x0: torch.Tensor | None = None
    inpaint_noise: torch.Tensor | None = None


def conditioning_image_paths(args) -> list[str]:
    """Every user-supplied image that conditions generation, for the minor guard's image layer."""
    from utils.generation.krea_controls import _collect_style_refs

    keys = ("init_image", "control_image", "reference_image", "post_reference_image", "creative_rag_image")
    paths = [getattr(args, k, "") for k in keys]
    for k in ("creative_rag_images", "dissect_refs"):
        paths += str(getattr(args, k, "") or "").split(",")
    paths += [_parse_control_spec(s)[0] for s in getattr(args, "control", None) or []]
    paths += [r.path for r in _collect_style_refs(args)]
    return list(dict.fromkeys(p for p in (str(x or "").strip() for x in paths) if p))


def _load_rgb_tensor(path: str, width: int, height: int) -> torch.Tensor:
    """``(3, H, W)`` in ``[-1, 1]``, resized to the canvas."""
    pil = Image.open(path).convert("RGB")
    if pil.size != (width, height):
        pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(pil, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)


def _style_text(args: Any, cfg: Any, prompt: str) -> str:
    """``--style`` (``a::0.6 | b::0.4`` mixes) or, with ``--auto-style-from-prompt``, a style read off the prompt."""
    style = (args.style or "").strip()
    if not (getattr(cfg, "style_embed_dim", 0) and not style and getattr(args, "auto_style_from_prompt", False)):
        return style
    try:
        from config.defaults.style_artists import extract_style_from_text

        style = extract_style_from_text(prompt) or ""
        if style:
            print(f'Auto-style from prompt: "{style[:50]}{"..." if len(style) > 50 else ""}"', file=sys.stderr)
    except Exception:
        pass
    return style


def encode_conditioning(args: Any, m: Any, plan: Any) -> Conditioning:
    """Screen every positive channel, encode (cached), and build the cond / uncond model kwargs."""
    prompt, num_gen, device = plan.prompt, plan.num_gen, m.device
    style = _style_text(args, m.cfg, prompt)
    style_mix = _parse_weighted_style_mix(style)
    if style_mix:
        style_key = "|".join(f"{t}::{w:.4f}" for t, w in style_mix)
    else:
        style_key = style if getattr(m.cfg, "style_embed_dim", 0) else ""

    layout = getattr(args, "_layout_compiled", None)
    layout_enc = str(getattr(args, "t5_layout_encode", "auto") or "auto").lower()
    if layout_enc == "auto":
        layout_enc = "blocks" if layout is not None else "flat"
    t5_prompt = str(getattr(args, "_encode_t5_positive_hint", "") or "").strip() or prompt
    segment_texts = clip_caps = long_clip_caps = None
    if layout is not None and layout_enc == "blocks":
        from utils.prompt.prompt_layout import substitute_compiled_layout_in_t5_prompt

        t5_prompt = substitute_compiled_layout_in_t5_prompt(prompt, layout)
    elif layout is not None and layout_enc == "segmented":
        from utils.prompt.prompt_layout import t5_segment_texts_for_full_prompt

        segment_texts = t5_segment_texts_for_full_prompt(layout, prompt)
    if m.text_bundle is not None and layout is not None:
        from utils.prompt.prompt_layout import multi_clip_caption

        clip_caps = [multi_clip_caption(layout, prompt)]
    if m.text_bundle is not None and getattr(m.text_bundle, "mode", "") == "penta":
        long_clip_caps = [prompt]

    negative = plan.negative
    box_regions = getattr(getattr(args, "_box_layout_spec", None), "regions", None) or []
    adapters = [_parse_lora_spec(s)[:2] for s in getattr(args, "lora", None) or []]
    positive_union = ", ".join(
        [
            prompt,
            t5_prompt,
            style,
            *(segment_texts or []),
            *(clip_caps or []),
            *(r.prompt for r in box_regions),
            *(Path(p).stem for p, scale in adapters if scale >= 0),
        ]
    )
    negative = require_allowed(positive_union, negative)
    # A negative-scale adapter subtracts its concept the way a negative prompt does.
    if negative_adapters := ", ".join(Path(p).stem for p, scale in adapters if scale < 0):
        require_allowed(positive_union, negative_adapters)
    sexual = is_sexual(positive_union)
    if float(getattr(args, "cfg_scale", 7.5) or 0.0) < 1.0:
        # Below CFG 1 the uncond branch gets positive weight, so the negative is part of the prompt.
        require_allowed(f"{prompt}, {negative}")
        sexual = sexual or is_sexual(negative)
    if sexual:
        require_image_allowed(conditioning_image_paths(args), sexual=True, conditioning=True)

    def enc(captions, **kw):
        return encode_text(captions, m.tokenizer, m.text_encoder, device, text_bundle=m.text_bundle, **kw)

    layout_tag = (layout_enc, str(getattr(args, "prompt_layout", "") or "")) if layout is not None else ("flat", "")
    t5_hash = hashlib.sha256(t5_prompt.encode("utf-8")).hexdigest()[:20]
    cache_key = None if getattr(args, "no_cache", False) else (prompt, negative, style_key, layout_tag, t5_hash)
    if cache_key is not None and cache_key in _t5_cache:
        cond_emb, uncond_emb, style_emb = (t.to(device) if t is not None else None for t in _t5_cache[cache_key])
        print("T5 cache hit.")
    else:
        if segment_texts is not None:
            cond_emb = enc(
                [t5_prompt], clip_captions=clip_caps, long_clip_captions=long_clip_caps, segment_texts=segment_texts
            )
            uncond_emb = enc([negative])
        else:
            both_long = long_clip_caps
            if m.text_bundle is not None and getattr(m.text_bundle, "mode", "") == "penta":
                both_long = [prompt, negative]
            both = enc([t5_prompt, negative], clip_captions=clip_caps, long_clip_captions=both_long)
            cond_emb, uncond_emb = both[0:1], both[1:2]
        style_emb = None
        if style and getattr(m.cfg, "style_embed_dim", 0):
            if len(style_mix) > 1:
                weights = torch.tensor([w for _, w in style_mix], device=device, dtype=torch.float32)
                style_enc = enc([t for t, _ in style_mix]).mean(dim=1)
                style_emb = (style_enc * weights[:, None].to(style_enc.dtype)).sum(dim=0, keepdim=True)
            else:
                style_emb = enc([style]).mean(dim=1)
        if cache_key is not None:
            while len(_t5_cache) >= _T5_CACHE_MAX:
                _t5_cache.pop(next(iter(_t5_cache)))
            _t5_cache[cache_key] = tuple(t.cpu() if t is not None else None for t in (cond_emb, uncond_emb, style_emb))

    cond = {"encoder_hidden_states": cond_emb.expand(num_gen, -1, -1)}
    uncond = {"encoder_hidden_states": uncond_emb.expand(num_gen, -1, -1)}
    if style_emb is not None:
        cond["style_embedding"] = style_emb.expand(num_gen, -1)
        cond["style_strength"] = args.style_strength
    if plan.emphasis_segments:
        from utils.prompt.prompt_emphasis import token_weights_from_cleaned_segments

        tw = token_weights_from_cleaned_segments(
            prompt, plan.emphasis_segments, m.tokenizer, TEXT_MAX_TOKENS, device=device
        )
        if tw is not None and tw.shape[0] == cond_emb.shape[1]:
            # (word) -> 1.2, [word] -> 0.8 on the cond tokens; batch-shaped so batched CFG and regional
            # forwards can stack it, and explicit ones for the negative branch.
            tw = tw.view(1, -1, 1).expand(num_gen, -1, -1)
            cond["token_weights"] = tw
            uncond["token_weights"] = torch.ones_like(tw)
    if getattr(m.cfg, "creativity_embed_dim", 0) and args.creativity is not None:
        c0 = max(0.0, min(1.0, float(args.creativity)))
        jitter = max(0.0, float(getattr(args, "creativity_jitter", 0.0) or 0.0))
        if jitter > 0:
            g = torch.Generator(device=device).manual_seed(int(args.seed) + 90211)
            deltas = torch.randn(num_gen, generator=g, device=device, dtype=torch.float32) * jitter
            cond["creativity"] = (c0 + deltas).clamp(0.0, 1.0).to(dtype=cond_emb.dtype)
        else:
            cond["creativity"] = torch.full((num_gen,), c0, device=device, dtype=cond_emb.dtype)
    for flag in ("internal_guidance_block", "genesis_ops"):
        if bool(getattr(args, flag, False)):
            cond[flag] = uncond[flag] = True
    _control_kwargs(args, m, cond, num_gen)
    _reference_kwargs(args, m, cond, num_gen)
    args._regional_cfg_plan = _regional_plan(args, m, enc)
    return Conditioning(cond=cond, uncond=uncond, sexual=sexual)


def _control_kwargs(args: Any, m: Any, cond: dict, num_gen: int) -> None:
    """ControlNet inputs: ``--control-image`` / stacked ``--control`` specs, else box-layout sketches.

    Tensors are batch-shaped like the text embeddings, so batched CFG can stack the branches.
    """
    from models.controlnet import control_type_to_id, infer_control_type_from_path

    default_type = str(getattr(args, "control_type", "auto") or "auto")
    default_scale = float(getattr(args, "control_scale", 0.85))
    specs: list[tuple[str, str, float]] = []
    if args.control_image:
        specs.append((str(args.control_image), default_type, default_scale))
    for raw in list(getattr(args, "control", []) or []):
        path, ctype, scale = _parse_control_spec(raw, default_type=default_type, default_scale=default_scale)
        if path:
            specs.append((path, ctype, scale))
    box_spec = getattr(args, "_box_layout_spec", None)
    if box_spec is not None and not specs:
        try:
            from utils.generation.regional_box_sketch import build_composite_sketch_tensor, spec_has_sketches

            if spec_has_sketches(box_spec):
                sketch = build_composite_sketch_tensor(
                    box_spec, m.image_size, source_dir=getattr(box_spec, "source_dir", None), device=m.device
                )
                cond["control_image"] = sketch.unsqueeze(0).expand(num_gen, -1, -1, -1)
                cond["control_scale"] = float(getattr(args, "box_sketch_control_scale", 0.75) or 0.75)
                cond["control_type"] = torch.full(
                    (num_gen,), int(control_type_to_id("scribble")), device=m.device, dtype=torch.long
                )
                print("Box-layout sketch control image enabled (scribble).", file=sys.stderr)
        except Exception as e:
            print(f"Warning: box sketch control image skipped: {e}", file=sys.stderr)
    if not specs:
        return
    images, type_ids, scales = [], [], []
    for path, ctype, scale in specs:
        images.append(_load_rgb_tensor(path, m.canvas_w, m.canvas_h))
        name = str(ctype or "auto").strip().lower()
        if name in ("", "auto"):
            name = infer_control_type_from_path(str(path))
        type_ids.append(int(control_type_to_id(name)))
        scales.append(float(max(0.0, scale)))
    types = torch.tensor(type_ids, device=m.device, dtype=torch.long)
    if len(images) == 1:
        cond["control_image"] = images[0].to(m.device).expand(num_gen, -1, -1, -1)
        cond["control_scale"] = scales[0]
        cond["control_type"] = types.expand(num_gen)
    else:
        stack = torch.stack(images, dim=0).to(m.device)  # (K, C, H, W)
        cond["control_image"] = stack.expand(num_gen, *stack.shape)
        cond["control_scale"] = torch.tensor(scales, device=m.device, dtype=torch.float32).expand(num_gen, -1)
        cond["control_type"] = types.expand(num_gen, -1)


def _reference_kwargs(args: Any, m: Any, cond: dict, num_gen: int) -> None:
    """``--reference-image`` / ``--style-ref`` / moodboards -> reference tokens."""
    ref_path = str(getattr(args, "reference_image", "") or "").strip()
    multi = any(
        str(getattr(args, k, "") or "").strip()
        for k in ("style_ref", "style_references_json", "moodboard_json", "moodboard_images")
    )
    if not (multi or (ref_path and float(getattr(args, "reference_strength", 0.0) or 0.0) > 0)):
        return
    try:
        from utils.generation.krea_controls import inject_reference_conditioning

        inject_reference_conditioning(cond, args=args, cfg=m.cfg, device=m.device, num_gen=num_gen)
    except Exception as e:
        print(f"Reference image conditioning skipped: {e}", file=sys.stderr)


def _regional_plan(args: Any, m: Any, enc: Any) -> Any:
    """Per-box embeddings and masks for regional CFG (``--box-layout`` in ``regional_cfg`` mode)."""
    spec = getattr(args, "_box_layout_spec", None)
    if (
        spec is None
        or str(getattr(args, "box_layout_mode", "regional_cfg") or "regional_cfg").lower() != "regional_cfg"
    ):
        return None
    try:
        from utils.generation.regional_box_prompting import encode_regional_plan

        plan = encode_regional_plan(
            spec,
            encode_fn=enc,
            device=m.device,
            latent_h=m.lat_h,
            latent_w=m.lat_w,
            pixel_size=max(m.canvas_h, m.canvas_w),
            base_negative=str(getattr(args, "negative_prompt", "") or ""),
        )
    except Exception as e:
        print(f"Warning: regional box CFG disabled: {e}", file=sys.stderr)
        return None
    sketches = sum(1 for r in spec.regions if r.sketch_path or r.strokes)
    msg = f"Regional CFG enabled for {len(spec.regions)} box(es)."
    print(msg + (f" {sketches} with draw+describe sketches." if sketches else ""), file=sys.stderr)
    return plan


def _auto_init_from_dissection(args: Any, m: Any) -> None:
    """``--auto-init-from-dissection``: composite reference parts into an init image + inpaint mask."""
    refs = [p.strip() for p in str(getattr(args, "dissect_refs", "") or "").split(",") if p.strip()]
    if not (bool(getattr(args, "auto_init_from_dissection", False)) and refs and not args.init_image and not args.mask):
        return
    try:
        from utils.generation.image_dissection import dissect_images_to_parts
        from utils.generation.part_compositing import build_init_and_inpaint_mask

        out_dir = Path("runs") / "dissection_composites"
        _reqs, parts, _facts = dissect_images_to_parts(
            str(getattr(args, "prompt", "") or ""), refs, output_dir=out_dir, enable_heavy_models=True
        )
        spec = build_init_and_inpaint_mask(
            reference_images=refs,
            parts=parts,
            output_dir=out_dir,
            target_size=(m.canvas_w, m.canvas_h),
            lock_background_if_requested=bool(getattr(args, "dissection_lock_background", False)),
        )
    except Exception as e:
        print(f"Auto dissection init skipped: {e}", file=sys.stderr)
        return
    if spec.preserved_parts > 0:
        args.init_image = spec.init_image_path
        args.mask = spec.mask_path
        # MDM inpaint behaves better for preserving known regions.
        if str(getattr(args, "inpaint_mode", "legacy")) == "legacy":
            args.inpaint_mode = "mdm"
        print(
            f"Auto dissection init enabled: parts={spec.preserved_parts}, "
            f"background_locked={spec.background_locked} -> init={args.init_image}, mask={args.mask}",
            file=sys.stderr,
        )


def _noise_to(diffusion: Any, z0: torch.Tensor, t_start: int, num_timesteps: int, flow: bool) -> torch.Tensor:
    """Forward-noise a clean latent to ``t_start`` (VP ``q_sample`` or the flow interpolation)."""
    if flow:
        s0 = float(t_start) / float(max(num_timesteps - 1, 1))
        return (1.0 - s0) * z0 + s0 * torch.randn_like(z0)
    return diffusion.q_sample(z0, torch.tensor([t_start], device=z0.device).expand(z0.shape[0]))


def prepare_init_latents(args: Any, m: Any, num_gen: int) -> InitLatents:
    """``--init-latent`` (from-z), ``--init-image`` (img2img) and ``--mask`` (inpaint) starting latents."""
    _auto_init_from_dissection(args, m)
    num_timesteps = int(getattr(m.cfg, "num_timesteps", 1000))
    flow = bool(getattr(args, "flow_matching_sample", False)) or (
        bool(getattr(m.cfg, "flow_matching_training", False)) and not bool(getattr(args, "force_vp_sample", False))
    )
    if flow and args.mask:
        print(
            "Flow sampling disabled: structured inpaint uses VP q_sample; use --force-vp-sample or drop --mask.",
            file=sys.stderr,
        )
        flow = False
    init = InitLatents(flow=flow, num_timesteps=num_timesteps)
    if not (args.init_latent or args.init_image):
        return init
    t_start = min(max(1, int(args.strength * num_timesteps)), num_timesteps - 1)
    init.start_timestep = t_start
    if args.init_latent:
        z = torch.load(args.init_latent, map_location=m.device, weights_only=True)
        z = _maybe_rae_to_dit(
            (z.unsqueeze(0) if z.dim() == 3 else z).to(device=m.device, dtype=torch.float32), m.ae_type, m.rae_bridge
        )
        init.x_init = _noise_to(m.diffusion, z, t_start, num_timesteps, flow)
        print(f"From-z: strength={args.strength} -> t_start={t_start}")
    else:
        img = _load_rgb_tensor(args.init_image, m.canvas_w, m.canvas_h).unsqueeze(0).to(m.device, torch.float32)
        with torch.no_grad():
            enc = m.vae.encode(img)
            z0 = enc.latent_dist.sample() * m.latent_scale if hasattr(enc, "latent_dist") else enc.latent
            z0 = _maybe_rae_to_dit(z0, m.ae_type, m.rae_bridge)
        if args.mask:
            pil_mask = Image.open(args.mask).convert("L")
            if pil_mask.size != (m.canvas_w, m.canvas_h):
                pil_mask = pil_mask.resize((m.canvas_w, m.canvas_h), Image.Resampling.LANCZOS)
            mask = torch.from_numpy((np.asarray(pil_mask, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32))
            mask = torch.nn.functional.interpolate(
                mask.to(m.device).view(1, 1, m.canvas_h, m.canvas_w), size=(m.lat_h, m.lat_w), mode="nearest"
            )
            t_b = torch.tensor([t_start], device=m.device).expand(z0.shape[0])
            noise_known = torch.randn_like(z0)
            if args.inpaint_mode == "mdm":
                # MDM-style fill-in: known and masked regions get different forward noise at t_start, and the
                # sampler re-imposes q_sample(x0_known, t) on the known region after every step.
                x_masked = m.diffusion.q_sample(z0, t_b, noise=torch.randn_like(z0))
                init.x_init = mask * x_masked + (1 - mask) * m.diffusion.q_sample(z0, t_b, noise=noise_known)
                init.inpaint_mask, init.inpaint_x0, init.inpaint_noise = mask, z0, noise_known
            else:
                init.x_init = mask * noise_known + (1 - mask) * m.diffusion.q_sample(z0, t_b, noise=noise_known)
            print(f"Inpainting: strength={args.strength} -> t_start={t_start}")
        else:
            init.x_init = _noise_to(m.diffusion, z0, t_start, num_timesteps, flow)
            print(f"Img2img: strength={args.strength} -> t_start={t_start}")
    if init.x_init.shape[0] != num_gen:
        for name in ("x_init", "inpaint_x0", "inpaint_noise", "inpaint_mask"):
            t = getattr(init, name)
            if t is not None:
                setattr(init, name, t.expand(num_gen, -1, -1, -1))
    return init


__all__ = ["Conditioning", "InitLatents", "conditioning_image_paths", "encode_conditioning", "prepare_init_latents"]
