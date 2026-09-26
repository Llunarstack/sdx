"""Library surface for single-image generation from a loaded DiT stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import torch
from PIL import Image


@dataclass(slots=True)
class ImageGenerateConfig:
    prompt: str
    negative_prompt: str = ""
    image_size: int = 512
    cfg_scale: float = 7.5
    steps: int = 50
    seed: int | None = None
    latent_scale: float = 0.18215
    max_length: int = 300
    inference_amp: str = "auto"
    auto_layout: bool = True
    invention_stack: str = "off"
    invention_spectra: bool = False
    invention_adaptive_steps: bool = False
    invention_auto_spectra: bool = False


_CONFIG_FIELD_NAMES = frozenset(f.name for f in fields(ImageGenerateConfig))


class ImageGenerationPipeline:
    """Loaded DiT + diffusion + VAE + text encoder. Not the 200-flag CLI."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        diffusion: Any,
        tokenizer: Any,
        text_encoder: torch.nn.Module,
        vae: torch.nn.Module,
        device: str | torch.device,
        config: ImageGenerateConfig | None = None,
        **stack_kwargs: Any,
    ) -> None:
        self.model = model
        self.diffusion = diffusion
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.device = device
        self.config = config if config is not None else ImageGenerateConfig(prompt="")
        self._stack_kwargs = dict(stack_kwargs)

    @torch.inference_mode()
    def generate(self, prompt: str | None = None, /, **overrides: Any) -> Image.Image:
        """Generate one image. ``prompt`` overrides ``config.prompt`` when given."""
        from utils.generation.simple_latent_generate import sample_one_image_pil

        known = {k: overrides.pop(k) for k in list(overrides) if k in _CONFIG_FIELD_NAMES}
        if prompt is not None:
            known["prompt"] = prompt
        cfg = replace(self.config, **known)
        payload = asdict(cfg)
        steps = int(overrides.pop("num_inference_steps", payload.pop("steps")))
        auto_layout = bool(payload.pop("auto_layout", True))
        inf_amp_mode = str(payload.pop("inference_amp", "auto") or "auto").strip().lower()
        gen_prompt = str(payload.get("prompt") or "")
        if auto_layout and gen_prompt.strip():
            try:
                from utils.generation.prompt_scene import compile_spatial_layout
                from utils.generation.regional_box_prompting import layout_text_from_regions

                spec = compile_spatial_layout(gen_prompt)
                if spec is not None:
                    layout_line = layout_text_from_regions(spec)
                    if layout_line:
                        payload["prompt"] = f"{gen_prompt}. {layout_line}" if gen_prompt else layout_line
                    gn = str(getattr(spec, "global_negative", "") or "").strip()
                    if gn:
                        neg = str(payload.get("negative_prompt") or "")
                        if gn.lower() not in neg.lower():
                            payload["negative_prompt"] = f"{neg}, {gn}".strip(", ").strip()
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning("auto_layout skipped: %s", e)
        grounded = str(payload.get("prompt") or "")
        if grounded.strip() and auto_layout:
            try:
                from pipelines.video.prompt_ground_graph import parse_prompt_ground

                graph = parse_prompt_ground(grounded)
                if len(graph.entities) >= 2 or graph.negations:
                    payload["prompt"] = graph.rewritten or grounded
                    extra = str(graph.negative_extra or "").strip()
                    if extra:
                        neg = str(payload.get("negative_prompt") or "")
                        if extra.lower() not in neg.lower():
                            payload["negative_prompt"] = f"{neg}, {extra}".strip(", ").strip()
            except Exception:
                pass
        inv_mode = str(payload.pop("invention_stack", "off") or "off").strip().lower()
        inv_spectra = bool(payload.pop("invention_spectra", False))
        inv_adapt = bool(payload.pop("invention_adaptive_steps", False))
        inv_auto_sp = bool(payload.pop("invention_auto_spectra", False))
        if inv_mode not in ("off", "none", "0", "false", ""):
            try:
                from utils.generation.inventions.wire import prepare_invention_runtime

                rt = prepare_invention_runtime(
                    str(payload.get("prompt") or ""),
                    str(payload.get("negative_prompt") or ""),
                    enable=inv_mode,
                    base_steps=int(steps),
                    adaptive_steps=inv_adapt,
                    auto_spectra=inv_auto_sp,
                    invention_spectra=inv_spectra,
                )
                payload["prompt"] = rt.positive
                if rt.negative:
                    payload["negative_prompt"] = rt.negative
                if rt.steps is not None:
                    steps = int(rt.steps)
                inv_spectra = bool(rt.invention_spectra)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning("invention_stack skipped: %s", e)
        call = {
            "model": self.model,
            "diffusion": self.diffusion,
            "tokenizer": self.tokenizer,
            "text_encoder": self.text_encoder,
            "vae": self.vae,
            "device": self.device,
            **self._stack_kwargs,
            **payload,
            "num_inference_steps": steps,
            **overrides,
        }
        if inf_amp_mode == "auto":
            dev = torch.device(self.device) if isinstance(self.device, str) else self.device
            if getattr(dev, "type", None) == "cuda":
                call.setdefault("inference_amp", True)
        if inv_spectra:
            call["invention_spectra"] = True
        return sample_one_image_pil(**call)
