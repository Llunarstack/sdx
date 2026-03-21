#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR_DEFAULT = os.path.join(ROOT, "model")


def _has_model_weights(local_dir: str) -> bool:
    exts = (".safetensors", ".bin", ".pt", ".ckpt")
    for root, _dirs, files in os.walk(local_dir):
        for f in files:
            if f.endswith(exts):
                return True
    return False


def _download(
    repo_id: str,
    local_dir: str,
    allow_patterns: list[str],
    max_workers: int,
    *,
    hf_transfer: bool,
) -> None:
    from huggingface_hub import snapshot_download

    os.makedirs(local_dir, exist_ok=True)
    if hf_transfer:
        # Faster transport path when hf_transfer is installed.
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        max_workers=max_workers,
    )


def _try_download(
    repo_id: str,
    local_dir: str,
    allow_patterns: list[str],
    max_workers: int,
    *,
    hf_transfer: bool,
    skip_existing: bool,
    continue_on_error: bool,
) -> bool:
    try:
        if skip_existing and os.path.isdir(local_dir) and _has_model_weights(local_dir):
            print(f"Skipping (already has weights): {repo_id} -> {local_dir}")
            return True
        _download(
            repo_id,
            local_dir,
            allow_patterns,
            max_workers,
            hf_transfer=hf_transfer,
        )
        return True
    except Exception as e:
        msg = f"Failed: {repo_id} -> {local_dir}: {e}"
        if continue_on_error:
            print(msg, file=sys.stderr)
            return False
        raise


def main() -> int:
    p = argparse.ArgumentParser(description="Download a curated high-end SDX model stack.")
    p.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument(
        "--mode",
        choices=("fast", "balanced", "low-mem"),
        default="balanced",
        help="fast: max throughput, balanced: safe default, low-mem: lower RAM/network pressure",
    )
    p.add_argument("--hf-transfer", action="store_true", help="Enable HF transfer acceleration if available")
    p.add_argument("--skip-existing", action="store_true", help="Skip repos that already have model weights")
    p.add_argument("--continue-on-error", action="store_true", help="Skip repos that fail and keep going")
    args = p.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Install dependency first: pip install huggingface_hub", file=sys.stderr)
        return 1

    model_dir = args.model_dir
    mode = str(args.mode)
    user_workers = int(args.max_workers)
    if mode == "fast":
        mw = max(12, user_workers)
    elif mode == "low-mem":
        mw = min(2, user_workers)
    else:
        mw = min(6, user_workers)

    cont = bool(args.continue_on_error)
    use_hf_transfer = bool(args.hf_transfer or mode == "fast")
    skip_existing = bool(args.skip_existing or mode in ("balanced", "low-mem"))
    print(
        f"Downloader config: mode={mode} max_workers={mw} hf_transfer={use_hf_transfer} skip_existing={skip_existing}"
    )
    jobs = [
        # Triple text encoder stack
        (
            "google/t5-v1_1-xxl",
            "T5-XXL",
            [
                "config.json",
                "generation_config.json",
                "*.safetensors",
                "pytorch_model.bin",
                "special_tokens_map.json",
                "spiece.model",
                "tokenizer_config.json",
            ],
        ),
        (
            "openai/clip-vit-large-patch14",
            "CLIP-ViT-L-14",
            [
                "config.json",
                "preprocessor_config.json",
                "*.safetensors",
                "pytorch_model.bin",
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
        ),
        (
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            "CLIP-ViT-bigG-14",
            [
                "config.json",
                "preprocessor_config.json",
                "*.safetensors",
                "pytorch_model.bin",
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
        ),
        # Best LLM
        (
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen2.5-14B-Instruct",
            [
                "config.json",
                "generation_config.json",
                "*.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ],
        ),
        # ViT/DiT support encoders
        (
            "facebook/dinov2-large",
            "DINOv2-Large",
            [
                "config.json",
                "preprocessor_config.json",
                "*.safetensors",
                "pytorch_model.bin",
            ],
        ),
        (
            "google/siglip-so400m-patch14-384",
            "SigLIP-SO400M",
            [
                "config.json",
                "preprocessor_config.json",
                "*.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ],
        ),
        # Cascaded diffusion stage models
        (
            "stabilityai/stable-cascade-prior",
            "StableCascade-Prior",
            [
                "model_index.json",
                "scheduler/*",
                "prior/*",
                "text_encoder/*",
                "tokenizer/*",
                "*.json",
                "*.safetensors",
                "*.bin",
            ],
        ),
        (
            "stabilityai/stable-cascade",
            "StableCascade-Decoder",
            [
                "model_index.json",
                "scheduler/*",
                "decoder/*",
                "vqgan/*",
                "*.json",
                "*.safetensors",
                "*.bin",
            ],
        ),
        # RAE-oriented representation backbone (for latent bridge experiments)
        (
            "facebook/dinov2-giant",
            "DINOv2-Giant",
            [
                "config.json",
                "preprocessor_config.json",
                "*.safetensors",
                "pytorch_model.bin",
            ],
        ),
    ]

    ok = 0
    failed = 0
    for repo_id, folder, allow in jobs:
        success = _try_download(
            repo_id,
            os.path.join(model_dir, folder),
            allow,
            mw,
            hf_transfer=use_hf_transfer,
            skip_existing=skip_existing,
            continue_on_error=cont,
        )
        ok += int(success)
        failed += int(not success)

    print(f"Done. downloaded={ok} failed={failed} target_dir={model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
