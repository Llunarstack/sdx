"""Resolve local `pretrained/` paths vs Hugging Face hub IDs."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def model_dir() -> Path:
    return repo_root() / "pretrained"


def resolve_model_path(folder_name: str, hf_fallback: str) -> str:
    """
    If `model/<folder_name>` exists and is non-empty, use it.
    Otherwise return `hf_fallback` (hub id).
    """
    local = model_dir() / folder_name
    if local.is_dir():
        try:
            if any(local.iterdir()):
                return str(local)
        except OSError:
            pass
    return hf_fallback


# Defaults aligned with scripts/download/download_revolutionary_stack.py


def default_t5_path() -> str:
    return resolve_model_path("T5-XXL", "google/t5-v1_1-xxl")


def default_clip_l_path() -> str:
    return resolve_model_path("CLIP-ViT-L-14", "openai/clip-vit-large-patch14")


def default_clip_bigg_path() -> str:
    return resolve_model_path("CLIP-ViT-bigG-14", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")


def default_dinov2_large_path() -> str:
    return resolve_model_path("DINOv2-Large", "facebook/dinov2-large")


def default_siglip_path() -> str:
    return resolve_model_path("SigLIP-SO400M", "google/siglip-so400m-patch14-384")


def default_qwen_path() -> str:
    return resolve_model_path("Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct")


def default_cascade_prior_path() -> str:
    return resolve_model_path("StableCascade-Prior", "stabilityai/stable-cascade-prior")


def default_cascade_decoder_path() -> str:
    return resolve_model_path("StableCascade-Decoder", "stabilityai/stable-cascade")
