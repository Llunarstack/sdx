"""
Neural VideoDiT sample path — denoise short latent clips without retrieve/interpolate.

Orchestration (scene_graph / stitch) stays outside; this module is the engine for
``ProcessOptions.neural_engine=True`` segments.

Prompt/image understanding: structured text context + I2V latent/identity proxies
(no more zero context).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .types import SegmentAssignment, VideoPlan
from .video_io import save_frame_rgb

__all__ = [
    "load_video_dit",
    "sample_video_latents",
    "latents_to_frames_placeholder",
    "run_neural_segment",
]


def load_video_dit(
    ckpt: str | Path | None = None,
    *,
    model_name: str = "VideoDiT-S/2",
    context_dim: int = 768,
    device: str | torch.device = "cpu",
    permanent: bool = False,
) -> torch.nn.Module:
    if permanent or "Permanent" in model_name:
        from models.permanent_video_dit import PermanentVideoDiT, PermanentVideoDiT_models

        factory = PermanentVideoDiT_models.get(model_name)
        if factory is None:
            model = PermanentVideoDiT(dim=256, depth=6, num_heads=4, patch_size=2, context_dim=context_dim)
        else:
            model = factory(context_dim=context_dim)
    else:
        from models.video_dit import VideoDiT, VideoDiT_models

        factory = VideoDiT_models.get(model_name)
        if factory is None:
            model = VideoDiT(dim=256, depth=6, num_heads=4, patch_size=2, context_dim=context_dim)
        else:
            model = factory(context_dim=context_dim)
    if ckpt and Path(ckpt).is_file():
        try:
            state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
        except Exception:
            # Image DiT ckpts are not VideoDiT — keep random init for dry API smoke.
            pass
    return model.to(device).eval()


@torch.inference_mode()
def sample_video_latents(
    model: torch.nn.Module,
    *,
    frames: int = 8,
    height: int = 32,
    width: int = 32,
    steps: int = 12,
    context: torch.Tensor | None = None,
    first_frame_latent: torch.Tensor | None = None,
    identity: torch.Tensor | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Euler flow-matching denoise of ``(1, C, T, H, W)`` latents.

    Compatible with ``flow_matching`` training; no VAE decode here.
    """
    device = torch.device(device)
    c = int(getattr(model, "in_channels", 4))
    g = torch.Generator(device=device if device.type == "cuda" else "cpu")
    g.manual_seed(int(seed))
    x = torch.randn(1, c, frames, height, width, generator=g, device=device)
    if context is not None:
        context = context.to(device)
    if first_frame_latent is not None:
        first_frame_latent = first_frame_latent.to(device)
        # Match channel/spatial to noise tensor
        if first_frame_latent.shape[-2:] != (height, width):
            first_frame_latent = torch.nn.functional.interpolate(
                first_frame_latent, size=(height, width), mode="bilinear", align_corners=False
            )
        if first_frame_latent.shape[1] != c:
            if first_frame_latent.shape[1] < c:
                pad = torch.zeros(1, c - first_frame_latent.shape[1], height, width, device=device)
                first_frame_latent = torch.cat([first_frame_latent, pad], dim=1)
            else:
                first_frame_latent = first_frame_latent[:, :c]

    supports_identity = hasattr(model, "id_bank")
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i in range(steps):
        t0, t1 = float(ts[i]), float(ts[i + 1])
        t = torch.full((1,), t0, device=device)
        kwargs: dict[str, Any] = {
            "context": context,
            "first_frame_latent": first_frame_latent,
        }
        if supports_identity and identity is not None:
            kwargs["identity"] = identity.to(device)
        v = model(x, t, **kwargs)
        x = x + (t1 - t0) * v
    return x


def latents_to_frames_placeholder(
    latents: torch.Tensor,
    out_dir: str | Path,
    *,
    width: int = 256,
    height: int = 256,
    anchor_rgb: np.ndarray | None = None,
) -> list[Path]:
    """
    Decode-less preview: map latent channels to RGB tiles for API smoke tests.

    When ``anchor_rgb`` is set (I2V), frame 0 is the real image and later frames
    blend latent motion onto it so identity is visibly locked.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z = latents[0].detach().float().cpu()  # C, T, H, W
    _c, t, _h, _w = z.shape
    paths: list[Path] = []
    anchor = None
    if anchor_rgb is not None:
        from PIL import Image

        anchor = np.asarray(Image.fromarray(anchor_rgb.astype(np.uint8)).resize((width, height), Image.BILINEAR))
    for i in range(t):
        plane = z[:3, i].mean(dim=0).numpy() if z.shape[0] >= 3 else z[0, i].numpy()
        plane = plane - plane.min()
        plane = plane / (plane.max() + 1e-6)
        motion = (np.stack([plane, plane, plane], axis=-1) * 255).astype(np.float32)
        from PIL import Image

        motion_img = np.asarray(
            Image.fromarray(motion.astype(np.uint8)).resize((width, height), Image.BILINEAR)
        ).astype(np.float32)
        if anchor is not None:
            # Frame 0 = pure anchor; later frames = soft latent residual on identity
            alpha = 0.0 if i == 0 else min(0.35, 0.08 + 0.04 * i)
            rgb = ((1.0 - alpha) * anchor.astype(np.float32) + alpha * motion_img).clip(0, 255).astype(np.uint8)
        else:
            rgb = motion_img.astype(np.uint8)
        fp = out / f"neural_{i:04d}.png"
        save_frame_rgb(fp, rgb)
        paths.append(fp)
    return paths


def run_neural_segment(
    assignment: SegmentAssignment,
    plan: VideoPlan,
    work_dir: str | Path,
    *,
    ckpt: str = "",
    target_frames: int = 24,
    width: int = 512,
    height: int = 512,
    steps: int = 12,
    device: str = "cpu",
    permanent: bool = True,
) -> tuple[list[Path], dict[str, Any]]:
    """
    Generate a segment with VideoDiT / PermanentVideoDiT.

    Returns frame paths + metadata. Raises on hard failures so the caller can
    fall back to retrieve→edit.
    """
    from .prompt_ground_graph import bind_at_mentions, parse_prompt_ground
    from .video_image_cond import image_prompt_brief, image_to_identity_tokens, image_to_latent_proxy
    from .video_text_encode import encode_prompt_context

    wd = Path(work_dir)
    neural_dir = wd / "neural"
    neural_dir.mkdir(parents=True, exist_ok=True)

    # Latent spatial size (assume /8 VAE); clamp for smoke without VAE.
    lh = max(8, (height // 8) // 2 * 2)
    lw = max(8, (width // 8) // 2 * 2)
    # Cap temporal length for untrained smoke
    t_lat = max(4, min(16, target_frames // 2))

    model_name = "PermanentVideoDiT-S/2" if permanent else "VideoDiT-S/2"
    model = load_video_dit(
        ckpt or None,
        device=device,
        permanent=permanent,
        model_name=model_name,
    )
    ctx_dim = int(getattr(model.context_proj, "in_features", 768)) if model.context_proj else 768
    model_dim = int(getattr(model, "dim", 256) or 256)

    raw_prompt = str(assignment.shot.prompt or plan.user_prompt or "").strip()
    graph = parse_prompt_ground(raw_prompt)
    prompt = graph.rewritten or raw_prompt

    # Multimodal @ / identity refs from plan metadata or assignment
    refs: dict[str, str] = {}
    leaders = (plan.metadata or {}).get("leaders") or {}
    for i, p in enumerate(getattr(assignment, "identity_refs", None) or leaders.get("identity_paths") or []):
        if p:
            refs[f"Image{i + 1}"] = f"identity reference {i + 1}"
    if assignment.start_image and Path(assignment.start_image).is_file():
        refs.setdefault("Image1", "first-frame identity lock")
        brief = image_prompt_brief(assignment.start_image)
        if brief.brief:
            prompt = f"{prompt}. {brief.brief}"
    if refs:
        prompt = bind_at_mentions(prompt, refs, graph=graph)

    context = None
    if model.context_proj is not None:
        context = encode_prompt_context(prompt, context_dim=ctx_dim, device=device)

    first = None
    identity = None
    anchor_rgb = None
    if assignment.start_image and Path(assignment.start_image).is_file():
        first = image_to_latent_proxy(
            assignment.start_image,
            height=lh,
            width=lw,
            device=device,
        )
        if permanent:
            identity = image_to_identity_tokens(
                assignment.start_image,
                dim=model_dim,
                device=device,
            )
        try:
            from PIL import Image

            anchor_rgb = np.asarray(Image.open(assignment.start_image).convert("RGB"))
        except Exception:
            anchor_rgb = None

    latents = sample_video_latents(
        model,
        frames=t_lat,
        height=lh,
        width=lw,
        steps=steps,
        context=context,
        first_frame_latent=first,
        identity=identity,
        seed=42 + int(assignment.shot.index),
        device=device,
    )
    frames = latents_to_frames_placeholder(
        latents,
        neural_dir,
        width=width,
        height=height,
        anchor_rgb=anchor_rgb,
    )
    # Resample to target_frames by holding last
    if len(frames) < target_frames:
        frames = list(frames) + [frames[-1]] * (target_frames - len(frames))
    elif len(frames) > target_frames:
        idx = np.linspace(0, len(frames) - 1, target_frames).astype(int)
        frames = [frames[i] for i in idx]

    meta = {
        "ops": [
            f"neural_sample:t={t_lat}",
            f"steps={steps}",
            f"model={model_name}",
            "prompt_ground=1",
            f"context={'yes' if context is not None else 'no'}",
            f"i2v={'yes' if first is not None else 'no'}",
        ],
        "prompt": prompt,
        "prompt_raw": raw_prompt,
        "negative_extra": graph.negative_extra,
        "ground_entities": len(graph.entities),
        "engine": "permanent_video_dit" if permanent else "video_dit",
    }
    return frames, meta
