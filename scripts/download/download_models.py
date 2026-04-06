#!/usr/bin/env python3
"""
Download Hugging Face models for better generation and prompt understanding.
Saves to pretrained/ (or --model-dir). Defaults are chosen to save disk space.

- T5 (prompt understanding): --t5 downloads only T5-XXL (best, 4096-dim). Use --t5-xl or --t5-large
  for smaller GPUs (saves ~several GB by not downloading the others).
- VAE (decode quality): --vae downloads only sd-vae-ft-mse (train default) + sdxl-vae-fp16-fix (best decode).
  Use --vae-all to get sd-vae-ft-ema and sdxl-vae as well (usually not needed).
- CLIP: only with --clip (optional, for future T5+CLIP). Not included in --all.
- LLM: --llm (SmolLM 360M) or --llm-best (Qwen2.5-7B). --all includes only the default LLM (360M); add --llm-best if needed.

Use --all to download the minimal recommended set: T5-XXL, 2 VAEs, SmolLM 360M (saves space).
To free space after downloading everything, run: python scripts/download/remove_unused_models.py
"""

import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_DIR_DEFAULT = os.path.join(ROOT, "pretrained")

# Text encoders: better prompt understanding (research: SDXL, FLUX, PixArt use T5-XXL)
T5_REPOS = [
    ("google/t5-v1_1-xxl", "T5-XXL"),  # 4096-dim, best understanding (default)
    ("google/t5-v1_1-xl", "T5-XL"),  # 1024-dim, lighter / faster
    ("google/t5-v1_1-large", "T5-Large"),  # 768-dim, lightest option
]
# VAEs: decode quality (same latent space; pick one for training, try others at inference)
VAE_REPOS = [
    ("stabilityai/sd-vae-ft-mse", "sd-vae-ft-mse"),  # SD 1.5 MSE fine-tuned (current default)
    ("stabilityai/sd-vae-ft-ema", "sd-vae-ft-ema"),  # SD 1.5 EMA (alternative)
    ("stabilityai/sdxl-vae", "sdxl-vae"),  # SDXL: better decode quality
    ("madebyollin/sdxl-vae-fp16-fix", "sdxl-vae-fp16-fix"),  # SDXL VAE, fp16-safe
]
# CLIP: optional for future dual-encoder (T5 + CLIP) pipelines
CLIP_REPOS = [
    ("openai/clip-vit-large-patch14", "CLIP-ViT-L-14"),
]
LLM_DEFAULT = "HuggingFaceTB/SmolLM2-360M-Instruct"
LLM_BEST = "Qwen/Qwen2.5-7B-Instruct"

ADVANCED_REPOS = [
    ("creative-graphic-design/LongCLIP-L", "LongCLIP-L"),
    ("vikhyatk/moondream2", "moondream2"),
    ("prs-eth/marigold-depth-v1-1", "Marigold-Depth-v1-1"),
    ("prs-eth/marigold-normals-v1-1", "Marigold-Normals-v1-1"),
    ("madebyollin/taesd", "TAESD"),
    ("madebyollin/taesdxl", "TAESDXL"),
    ("openai/consistency-decoder", "Consistency-Decoder"),
    ("facebook/convnextv2-large-22k-384", "ConvNeXtV2-Large"),
    ("camenduru/improved-aesthetic-predictor", "LAION-Aesthetic-v2"),
    # Non-standard repo layout but useful for identity-preserving experiments.
    ("camenduru/AnyDoor", "AnyDoor-Ref"),
]

# Only download files needed to load the model (no README, .gitattributes, LICENSE, etc.)
ALLOW_T5 = [
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer_config.json",
]
ALLOW_VAE = [
    "config.json",
    "*.safetensors",
    "diffusion_pytorch_model.bin",
]
ALLOW_CLIP = [
    "config.json",
    "preprocessor_config.json",
    "*.safetensors",
    "pytorch_model.bin",
]
ALLOW_LLM = [
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]

ALLOW_ADVANCED_GENERIC = [
    "config.json",
    "model_index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.ckpt",
    "*.pth",
]

CODEFORMER_FILES = {
    "weights/CodeFormer/codeformer.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "weights/facelib/detection_Resnet50_Final.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "weights/facelib/parsing_parsenet.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
}


def download(repo_id: str, local_dir: str, max_workers: int = 4, allow_patterns=None):
    from huggingface_hub import snapshot_download

    kwargs = {"repo_id": repo_id, "local_dir": local_dir, "max_workers": max_workers}
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    snapshot_download(**kwargs)
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Download best HF models for image output: T5, VAE, LLMs into model/.")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR_DEFAULT, help="Base dir (default: project pretrained/)")
    parser.add_argument(
        "--t5", action="store_true", help="Download T5 text encoders: XXL (best), XL, Large (prompt understanding)"
    )
    parser.add_argument(
        "--vae", action="store_true", help="Download VAEs: sd-vae-ft-mse, sd-vae-ft-ema, sdxl-vae, sdxl-vae-fp16-fix"
    )
    parser.add_argument(
        "--clip", action="store_true", help="Download CLIP ViT-L/14 (optional, for future T5+CLIP dual-encoder)"
    )
    parser.add_argument("--llm", action="store_true", help="Download default LLM (SmolLM2-360M) for prompt expansion")
    parser.add_argument(
        "--llm-best", action="store_true", help="Download best LLM (Qwen2.5-7B-Instruct) for prompt expansion"
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Download advanced optional models (LongCLIP, moondream2, Marigold, TAESD, CodeFormer, ConvNeXtV2, consistency decoder, aesthetic predictor, AnyDoor ref).",
    )
    parser.add_argument("--all", action="store_true", help="Download all T5 sizes, all VAEs, CLIP, and both LLMs")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel download workers")
    args = parser.parse_args()

    do_t5 = args.t5 or args.all
    do_vae = args.vae or args.all
    do_clip = args.clip or args.all
    do_llm = args.llm or args.all
    do_llm_best = args.llm_best or args.all
    do_advanced = args.advanced

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Install: pip install huggingface_hub", file=sys.stderr)
        return 1

    os.makedirs(args.model_dir, exist_ok=True)
    n = 0

    if do_t5:
        for repo_id, folder in T5_REPOS:
            local_dir = os.path.join(args.model_dir, folder)
            print(f"Downloading T5 (essential files only): {repo_id} -> {local_dir}")
            download(repo_id, local_dir, args.max_workers, allow_patterns=ALLOW_T5)
            print(f"  -> Use: --text_encoder {local_dir}")
            n += 1

    if do_vae:
        for repo_id, folder in VAE_REPOS:
            local_dir = os.path.join(args.model_dir, folder)
            print(f"Downloading VAE (essential files only): {repo_id} -> {local_dir}")
            download(repo_id, local_dir, args.max_workers, allow_patterns=ALLOW_VAE)
            print(f"  -> Use: --vae_model {local_dir}")
            n += 1

    if do_clip:
        for repo_id, folder in CLIP_REPOS:
            local_dir = os.path.join(args.model_dir, folder)
            print(f"Downloading CLIP (essential files only): {repo_id} -> {local_dir}")
            download(repo_id, local_dir, args.max_workers, allow_patterns=ALLOW_CLIP)
            print(f"  -> For future dual-encoder: load from {local_dir}")
            n += 1

    if do_llm:
        local_dir = os.path.join(args.model_dir, "SmolLM2-360M-Instruct")
        print(f"Downloading LLM (essential files only): {LLM_DEFAULT} -> {local_dir}")
        download(LLM_DEFAULT, local_dir, args.max_workers, allow_patterns=ALLOW_LLM)
        n += 1

    if do_llm_best:
        local_dir = os.path.join(args.model_dir, "Qwen2.5-7B-Instruct")
        print(f"Downloading LLM (essential files only): {LLM_BEST} -> {local_dir}")
        download(LLM_BEST, local_dir, args.max_workers, allow_patterns=ALLOW_LLM)
        n += 1

    if do_advanced:
        for repo_id, folder in ADVANCED_REPOS:
            local_dir = os.path.join(args.model_dir, folder)
            print(f"Downloading advanced model (essential files only): {repo_id} -> {local_dir}")
            download(repo_id, local_dir, args.max_workers, allow_patterns=ALLOW_ADVANCED_GENERIC)
            n += 1
        codeformer_root = os.path.join(args.model_dir, "CodeFormer")
        for rel, url in CODEFORMER_FILES.items():
            target = os.path.join(codeformer_root, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            print(f"Downloading CodeFormer weight: {url} -> {target}")
            urllib.request.urlretrieve(url, target)
        n += 1

    if n == 0:
        print("Choose at least one: --t5, --vae, --clip, --llm, --llm-best, --advanced, or --all", file=sys.stderr)
        return 1

    print(f"Done. {n} model(s) in {args.model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
