"""
Universal asset **sniffer** — fingerprint any image-gen ecosystem file.

Given a state dict (or path to ``.safetensors`` / ``.pt`` / ``.ckpt``), report
what it is: which ecosystem it was trained for (SD1.5, SD2, SDXL, SD3/MMDiT,
Flux, PixArt, generic DiT, sdx) and what kind of asset it is (full checkpoint,
LoRA-family adapter, textual-inversion embedding, VAE, text encoder, ESRGAN-
style upscaler). Downstream bridges (``adapter_bridge``, ``embedding_bridge``,
``latent_probe``) consume this report to pick a conversion path.

Detection is key-signature based — no network, no model construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

__all__ = [
    "AssetReport",
    "load_state_dict",
    "sniff_asset",
]

# LoRA-family tensor suffixes (kohya, PEFT/diffusers, LyCORIS).
_ADAPTER_SUFFIXES = (
    ".lora_down.weight",
    ".lora_up.weight",
    ".lora_A.weight",
    ".lora_B.weight",
    ".hada_w1_a",
    ".hada_w1_b",
    ".hada_w2_a",
    ".hada_w2_b",
    ".lokr_w1",
    ".lokr_w2",
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".diff",
    ".diff_b",
    ".dora_scale",
    ".dora_magnitude_vector",
)


@dataclass(slots=True)
class AssetReport:
    """What a foreign file is and which ecosystem it belongs to."""

    kind: str = "unknown"
    """checkpoint | adapter | embedding | vae | text_encoder | upscaler | unknown"""
    family: str = "unknown"
    """sd15 | sd2 | sdxl (incl. Pony/Illustrious/NoobAI/Playground/Kolors) |
    sd3 | flux (incl. Chroma) | auraflow | pixart | hunyuan_dit | lumina |
    hidream | qwen_image | cascade | dit | clip | t5 | esrgan | swinir | unknown"""
    adapter_algo: str = ""
    """For adapters: lora | loha | lokr | full_diff (empty otherwise; mixed files
    report the dominant algorithm)."""
    num_tensors: int = 0
    details: dict[str, str] = field(default_factory=dict)


def load_state_dict(path_or_state: str | Path | dict) -> dict[str, torch.Tensor]:
    """Load a state dict from .safetensors/.pt/.ckpt (or pass a dict through)."""
    if isinstance(path_or_state, dict):
        return path_or_state
    path = Path(path_or_state)
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def _has_prefix(keys: list[str], prefix: str) -> bool:
    return any(k.startswith(prefix) for k in keys)


def _has_sub(keys: list[str], sub: str) -> bool:
    return any(sub in k for k in keys)


def _adapter_algo(keys: list[str]) -> str:
    counts = {
        "loha": sum(1 for k in keys if ".hada_w1_a" in k),
        "lokr": sum(1 for k in keys if ".lokr_w1" in k or ".lokr_w1_a" in k),
        "lora": sum(1 for k in keys if k.endswith((".lora_down.weight", ".lora_A.weight"))),
        "full_diff": sum(1 for k in keys if k.endswith(".diff")),
    }
    best = max(counts, key=lambda a: counts[a])
    return best if counts[best] > 0 else "lora"


def _adapter_family(keys: list[str]) -> str:
    if _has_sub(keys, "double_stream_blocks") or _has_sub(keys, "single_stream_blocks"):
        return "hidream"
    if _has_sub(keys, "double_blocks") or _has_sub(keys, "single_blocks"):
        return "flux"
    if _has_sub(keys, "joint_transformer_blocks"):
        return "auraflow"
    if _has_sub(keys, "joint_blocks"):
        return "sd3"
    if _has_sub(keys, "Wqkv"):
        return "hunyuan_dit"
    if _has_sub(keys, "img_mlp") or _has_sub(keys, "img_mod"):
        return "qwen_image"
    if _has_prefix(keys, "lora_prior_"):
        return "cascade"
    if _has_prefix(keys, "lora_te2_") or _has_sub(keys, "text_encoder_2"):
        return "sdxl"
    if _has_sub(keys, "input_blocks") or _has_sub(keys, "down_blocks"):
        # UNet-shaped target; SDXL adapters usually carry te2, handled above.
        return "sd15"
    if _has_sub(keys, "transformer_blocks") or _has_sub(keys, "lora_transformer_"):
        return "dit"
    return "unknown"


def _checkpoint_family(keys: list[str], state: dict[str, torch.Tensor]) -> str:
    # Ordered most-specific-first; several ecosystems share substrings
    # ("single_transformer_blocks" contains "transformer_blocks", Flux and SD3
    # both have "context_embedder", ...).
    if _has_sub(keys, "double_stream_blocks."):
        return "hidream"
    if _has_sub(keys, "double_blocks.") or _has_sub(keys, "single_blocks."):
        return "flux"
    if _has_sub(keys, "joint_blocks."):
        return "sd3"
    if _has_sub(keys, "joint_transformer_blocks.") or _has_sub(keys, "register_tokens"):
        return "auraflow"
    if _has_sub(keys, "Wqkv") and _has_sub(keys, "blocks."):
        return "hunyuan_dit"
    if _has_sub(keys, "clip_txt_pooled_mapper"):
        return "cascade"
    if _has_sub(keys, "cap_embedder"):
        return "lumina"
    if _has_sub(keys, "transformer_blocks."):
        if _has_sub(keys, "img_mlp") or _has_sub(keys, "img_mod"):
            return "qwen_image"
        if _has_sub(keys, "single_transformer_blocks."):
            return "flux"
        if _has_sub(keys, "adaln_single"):
            return "pixart"
        if _has_sub(keys, "context_embedder"):
            return "sd3"
    if _has_prefix(keys, "conditioner.embedders.1.") or _has_sub(keys, "add_embedding."):
        # Also covers SDXL-architecture derivatives: Pony, Illustrious, NoobAI,
        # Animagine, Playground, Kolors — same UNet, so same bridge path.
        return "sdxl"
    if _has_prefix(keys, "model.diffusion_model."):
        # SD1.x vs SD2.x: cross-attention context width (768 vs 1024).
        for k in keys:
            if "attn2.to_k.weight" in k:
                w = state.get(k)
                if w is not None and w.ndim == 2:
                    return "sd2" if int(w.shape[1]) >= 1024 else "sd15"
        return "sd15"
    if _has_sub(keys, "cross_attn.") and _has_prefix(keys, "blocks."):
        return "pixart"
    if _has_prefix(keys, "blocks.") and (_has_sub(keys, ".attn.qkv.") or _has_sub(keys, "x_embedder")):
        return "dit"
    return "unknown"


def sniff_asset(path_or_state: str | Path | dict) -> AssetReport:
    """Fingerprint a state dict into an :class:`AssetReport`."""
    state = load_state_dict(path_or_state)
    keys = list(state.keys())
    report = AssetReport(num_tensors=len(keys))
    if not keys:
        return report

    # Textual-inversion embeddings: tiny dicts in one of the known layouts.
    if "string_to_param" in state or "emb_params" in keys:
        report.kind, report.family = "embedding", "clip"
        return report
    if set(keys) <= {"clip_l", "clip_g"}:
        report.kind, report.family = "embedding", "sdxl"
        return report

    # LoRA-family adapters (checked before checkpoints: adapter keys embed
    # target-model names like down_blocks that would confuse family checks).
    if any(k.endswith(_ADAPTER_SUFFIXES) for k in keys):
        report.kind = "adapter"
        report.adapter_algo = _adapter_algo(keys)
        report.family = _adapter_family(keys)
        report.details["text_encoder_tensors"] = str(sum(1 for k in keys if k.startswith(("lora_te", "text_encoder"))))
        return report

    # Standalone VAE (diffusers or CompVis layout).
    is_vae = (_has_prefix(keys, "decoder.") and _has_prefix(keys, "encoder.")) or _has_prefix(
        keys, "first_stage_model.decoder."
    )
    if is_vae and not _has_sub(keys, "diffusion_model"):
        report.kind, report.family = "vae", "unknown"
        return report

    # Upscalers: RRDBNet (ESRGAN/RealESRGAN) or SwinIR residual groups.
    if _has_sub(keys, ".rdb1.conv1.") or _has_sub(keys, "model.1.sub."):
        report.kind, report.family = "upscaler", "esrgan"
        return report
    if _has_sub(keys, "residual_group."):
        report.kind, report.family = "upscaler", "swinir"
        return report

    # Standalone text encoders.
    if _has_prefix(keys, "text_model.encoder.layers."):
        report.kind, report.family = "text_encoder", "clip"
        return report
    if _has_sub(keys, "encoder.block.") and _has_sub(keys, "SelfAttention"):
        report.kind, report.family = "text_encoder", "t5"
        return report

    family = _checkpoint_family(keys, state)
    if family != "unknown":
        report.kind, report.family = "checkpoint", family
        # Sub-family markers that don't change the bridge path but matter for
        # defaults (text encoder, guidance): Chroma is a pruned Flux; Kolors is
        # an SDXL UNet driven by ChatGLM (4096-d encoder_hid_proj).
        if family == "flux" and _has_sub(keys, "distilled_guidance_layer"):
            report.details["variant"] = "chroma"
        if family == "sdxl" and _has_sub(keys, "encoder_hid_proj"):
            report.details["variant"] = "kolors"
        return report

    # Single raw tensor → likely an embedding vector dump.
    if len(keys) == 1 and torch.is_tensor(next(iter(state.values()))):
        t = next(iter(state.values()))
        if t.ndim <= 2 and t.numel() < 1_000_000:
            report.kind, report.family = "embedding", "unknown"
            return report
    return report
