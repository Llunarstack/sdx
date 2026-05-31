"""
Helpers for triple / penta ``encode_text`` — CLIP vs LongCLIP caption routing.

Training JSONL may include ``prompt_layout`` (dict) or ``prompt_layout_path``; when present,
short CLIP encoders get :func:`utils.prompt.prompt_layout.multi_clip_caption` and LongCLIP
gets the full flat caption (penta).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.modeling.text_encoder_bundle import attach_fusion_weights, load_text_encoder_bundle

LayoutSpec = Union[None, str, Path, dict]


def _compile_layout_spec(spec: LayoutSpec) -> Any:
    """Return :class:`~utils.prompt.prompt_layout.CompiledPromptLayout`."""
    from utils.prompt.prompt_layout import compile_prompt_layout, load_prompt_layout_file

    if spec is None:
        raise ValueError("layout spec is None")
    if isinstance(spec, dict):
        return compile_prompt_layout(spec)
    p = Path(spec)
    if p.is_file():
        return load_prompt_layout_file(p)
    if isinstance(spec, str) and spec.strip().startswith("{"):
        raw = json.loads(spec)
        if isinstance(raw, dict):
            return compile_prompt_layout(raw)
    raise ValueError(f"invalid prompt layout spec: {spec!r}")


def clip_and_long_captions_for_layout(
    flat_caption: str,
    layout_spec: LayoutSpec,
) -> tuple[str, str]:
    """
    CLIP string (labeled sections, 77-token friendly) and LongCLIP string (full flat text).
    """
    from utils.prompt.prompt_layout import multi_clip_caption

    flat = (flat_caption or "").strip()
    if layout_spec is None:
        return flat, flat
    try:
        compiled = _compile_layout_spec(layout_spec)
        clip = multi_clip_caption(compiled, flat or compiled.positive)
        long_cap = flat or compiled.positive
        return clip, long_cap
    except Exception:
        return flat, flat


def prepare_multi_encoder_kwargs(
    t5_captions: List[str],
    text_bundle: Any = None,
    *,
    layout_specs: Optional[List[Optional[LayoutSpec]]] = None,
    flat_full_prompts: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Build ``clip_captions`` / ``long_clip_captions`` kwargs for multi-encoder bundles.

    When ``layout_specs[i]`` is set, CLIP encoders use the compiled layout caption; LongCLIP
    (penta) uses ``flat_full_prompts[i]`` or ``t5_captions[i]``.
    """
    if text_bundle is None or getattr(text_bundle, "fusion", None) is None:
        return {}
    mode = str(getattr(text_bundle, "mode", "t5") or "t5").lower()
    if mode not in ("triple", "penta"):
        return {}

    n = len(t5_captions)
    flats = list(flat_full_prompts) if flat_full_prompts is not None else list(t5_captions)
    if len(flats) < n:
        flats.extend([t5_captions[i] for i in range(len(flats), n)])
    specs: List[Optional[LayoutSpec]] = list(layout_specs) if layout_specs is not None else [None] * n
    if len(specs) < n:
        specs.extend([None] * (n - len(specs)))

    clip_caps: List[str] = []
    long_caps: List[str] = []
    for i in range(n):
        flat = flats[i] if i < len(flats) else t5_captions[i]
        spec = specs[i] if i < len(specs) else None
        clip_s, long_s = clip_and_long_captions_for_layout(flat, spec)
        clip_caps.append(clip_s)
        long_caps.append(long_s)

    out: Dict[str, List[str]] = {"clip_captions": clip_caps}
    if mode == "penta":
        out["long_clip_captions"] = long_caps
    return out


def encode_kwargs_for_captions(
    captions: List[str],
    text_bundle: Any = None,
    *,
    layout_specs: Optional[List[Optional[LayoutSpec]]] = None,
    flat_full_prompts: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Extra kwargs for :func:`train.encode_text` / :class:`TextEncoderBundle.encode`.

    Pass ``layout_specs`` from a training batch (JSONL ``prompt_layout`` fields) for
    layout-aware CLIP / LongCLIP routing.
    """
    if layout_specs is not None or flat_full_prompts is not None:
        return prepare_multi_encoder_kwargs(
            captions,
            text_bundle,
            layout_specs=layout_specs,
            flat_full_prompts=flat_full_prompts,
        )
    if text_bundle is None or getattr(text_bundle, "fusion", None) is None:
        return {}
    mode = str(getattr(text_bundle, "mode", "t5") or "t5").lower()
    if mode not in ("triple", "penta"):
        return {}
    caps = list(captions)
    out: Dict[str, List[str]] = {"clip_captions": caps}
    if mode == "penta":
        out["long_clip_captions"] = caps
    return out


def load_text_bundle_for_training(
    cfg: object,
    device: Any,
    fusion_sd: Optional[dict] = None,
    *,
    out_dim: int = 4096,
):
    """Load triple/penta bundle and attach checkpoint fusion weights when provided."""
    bundle = load_text_encoder_bundle(cfg, device, out_dim=out_dim)
    if bundle is not None and fusion_sd is not None:
        attach_fusion_weights(bundle, fusion_sd)
    return bundle


__all__ = [
    "clip_and_long_captions_for_layout",
    "encode_kwargs_for_captions",
    "load_text_bundle_for_training",
    "prepare_multi_encoder_kwargs",
]
