"""
Hugging Face scaffold registry and config-only downloads.

Downloads repo metadata (configs, tokenizers, small files) without checkpoint
weights so local folders exist for wiring while runtime still falls back to
the hub when weights are needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from utils.modeling.model_paths import model_dir, repo_root

# Small files only — no weight checkpoints.
CONFIG_ONLY_ALLOW: tuple[str, ...] = (
    "**/*.json",
    "**/*.jinja",
    "**/*.yaml",
    "**/*.yml",
    "**/*.txt",
    "**/*.md",
    "**/.gitattributes",
    "**/tokenizer.model",
    "**/spiece.model",
    "**/vocab.json",
    "**/merges.txt",
    "**/special_tokens_map.json",
    "**/tokenizer_config.json",
    "**/preprocessor_config.json",
    "**/processor_config.json",
    "**/generation_config.json",
    "**/model_index.json",
)

CONFIG_ONLY_IGNORE: tuple[str, ...] = (
    "*.safetensors",
    "**/*.safetensors",
    "*.bin",
    "**/*.bin",
    "*.pt",
    "**/*.pt",
    "*.pth",
    "**/*.pth",
    "*.ckpt",
    "**/*.ckpt",
    "*.onnx",
    "**/*.onnx",
    "*.gguf",
    "**/*.gguf",
    "*.h5",
    "**/*.h5",
    "*.zip",
    "**/*.zip",
    "*.tar",
    "**/*.tar",
    "*.tar.gz",
    "**/*.tar.gz",
)

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")


@dataclass(frozen=True, slots=True)
class HFModelEntry:
    """One row in the SDX HF scaffold registry."""

    name: str
    repo_id: str
    role: str
    local_folder: str


def _entry(name: str, repo_id: str, role: str) -> HFModelEntry:
    return HFModelEntry(name=name, repo_id=repo_id, role=role, local_folder=name)


# Curated stack: core + boost models wired through model_paths / hf_loaders.
HF_SCAFFOLD_REGISTRY: tuple[HFModelEntry, ...] = (
    # Text / CLIP
    _entry("T5-XXL", "google/t5-v1_1-xxl", "text_encoder"),
    _entry("CLIP-ViT-L-14", "openai/clip-vit-large-patch14", "text_encoder"),
    _entry("CLIP-ViT-bigG-14", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "text_encoder"),
    _entry("CLIP-ViT-H-14", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "text_encoder"),
    _entry("LongCLIP-L", "creative-graphic-design/LongCLIP-L", "text_encoder"),
    _entry("SigLIP-SO400M", "google/siglip-so400m-patch14-384", "vision_encoder"),
    _entry("SigLIP-base-patch16-224", "google/siglip-base-patch16-224", "vision_encoder"),
    _entry("SigLIP2-SO400M", "google/siglip2-so400m-patch16-384", "vision_encoder"),
    # Vision backbones
    _entry("DINOv2-Large", "facebook/dinov2-large", "vision_encoder"),
    _entry("DINOv2-Giant", "facebook/dinov2-giant", "vision_encoder"),
    _entry("DINOv2-Base", "facebook/dinov2-base", "vision_encoder"),
    _entry("DINOv2-Small", "facebook/dinov2-small", "vision_encoder"),
    _entry("ConvNeXtV2-Large", "facebook/convnextv2-large-22k-384", "vision_encoder"),
    _entry("MobileCLIP-S2", "apple/MobileCLIP-S2", "vision_encoder"),
    _entry("DINOv3-ViT-S16", "facebook/dinov3-vits16-pretrain-lvd1689m", "vision_encoder"),
    # LLM / VLM
    _entry("Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "llm"),
    _entry("GenSearcher-8B", "GenSearcher/Gen-Searcher-8B", "vlm"),
    _entry("moondream2", "vikhyatoolkit/moondream2", "vlm"),
    _entry("Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-2B-Instruct", "vlm"),
    _entry("MiniCPM-V-2", "openbmb/MiniCPM-V-2", "vlm"),
    _entry("Kosmos-2-patch14-224", "microsoft/kosmos-2-patch14-224", "vlm"),
    _entry("Florence-2-base", "microsoft/Florence-2-base", "vlm"),
    _entry("Florence-2-large", "microsoft/Florence-2-large", "vlm"),
    _entry("BLIP-image-captioning-base", "Salesforce/blip-image-captioning-base", "caption"),
    _entry("BLIP-image-captioning-large", "Salesforce/blip-image-captioning-large", "caption"),
    _entry("BLIP2-opt-2.7b", "Salesforce/blip2-opt-2.7b", "caption"),
    _entry("InstructBLIP-Vicuna-7B", "Salesforce/instructblip-vicuna-7b", "caption"),
    _entry("vit-gpt2-image-captioning", "nlpconnect/vit-gpt2-image-captioning", "caption"),
    _entry("SmolVLM-256M-Instruct", "HuggingFaceTB/SmolVLM-256M-Instruct", "vlm"),
    _entry("PaliGemma-3B", "google/paligemma-3b-pt-224", "vlm"),
    _entry("MiniCPM-Llama3-V-2_5", "openbmb/MiniCPM-Llama3-V-2_5", "vlm"),
    _entry("SmolVLM2-2B-Instruct", "HuggingFaceTB/SmolVLM2-2B-Instruct", "vlm"),
    _entry("Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "vlm"),
    _entry("Phi-3.5-vision-instruct", "microsoft/Phi-3.5-vision-instruct", "vlm"),
    _entry("Florence-2-base-ft", "microsoft/Florence-2-base-ft", "vlm"),
    _entry("GIT-base-coco", "microsoft/git-base-coco", "caption"),
    _entry("GIT-large-coco", "microsoft/git-large-coco", "caption"),
    _entry("BLIP2-Flan-T5-XL", "Salesforce/blip2-flan-t5-xl", "caption"),
    _entry("LLaVA-1.5-7B", "llava-hf/llava-1.5-7b-hf", "vlm"),
    _entry("InternVL2-2B", "OpenGVLab/InternVL2-2B", "vlm"),
    _entry("Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct", "vlm"),
    _entry("Phi-3-vision-128k-instruct", "microsoft/Phi-3-vision-128k-instruct", "vlm"),
    _entry("LLaVA-v1.6-Mistral-7B", "llava-hf/llava-v1.6-mistral-7b-hf", "vlm"),
    _entry("Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "vlm"),
    _entry("InternVL2-1B", "OpenGVLab/InternVL2-1B", "vlm"),
    _entry("PaliGemma2-3B", "google/paligemma2-3b-pt-224", "vlm"),
    _entry("Florence-2-large-ft", "microsoft/Florence-2-large-ft", "vlm"),
    _entry("MiniCPM-V-2_6", "openbmb/MiniCPM-V-2_6", "vlm"),
    _entry("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "llm"),
    _entry("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "llm"),
    _entry("SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", "llm"),
    _entry("T5-XL", "google/t5-v1_1-xl", "text_encoder"),
    _entry("T5-Large", "google/t5-v1_1-large", "text_encoder"),
    _entry("CLIP-ViT-B-32", "openai/clip-vit-base-patch32", "text_encoder"),
    _entry("LongCLIP-B", "creative-graphic-design/LongCLIP-B", "text_encoder"),
    _entry("UMT5-XXL", "google/umt5-xxl", "text_encoder"),
    _entry("CLIP-ViT-B-16", "openai/clip-vit-base-patch16", "text_encoder"),
    # Quality / preference
    _entry("ImageReward", "zai-org/ImageReward", "reward"),
    _entry("PickScore_v1", "yuvalkirstain/PickScore_v1", "reward"),
    _entry("HPSv2-hf", "adams-story/HPSv2-hf", "reward"),
    _entry("PerceptCLIP_IQA", "PerceptCLIP/PerceptCLIP_IQA", "reward"),
    _entry("LAION-Aesthetic-v2", "camenduru/improved-aesthetic-predictor", "reward"),
    _entry("OneAlign", "Q-Future/OneAlign", "reward"),
    _entry("CAFE-Aesthetic", "cafeai/cafe_aesthetic_classifier", "reward"),
    _entry("MUSIQ", "chengaz/MUSIQ", "reward"),
    _entry("CLIP-IQA", "djghosh/aesthetics-scorer", "reward"),
    # Vision encoders (alignment / REPA experiments)
    _entry("MetaCLIP-B16", "facebook/metaclip-b16-fullcc2.5b", "vision_encoder"),
    _entry("AIMv2-Large", "apple/aimv2-large-patch14-384", "vision_encoder"),
    _entry("AltCLIP-ViT-H-14", "BAAI/AltCLIP-ViT-H-14", "vision_encoder"),
    _entry("EVA02-CLIP-L-14", "BAAI/EVA02-CLIP-L-14", "vision_encoder"),
    _entry("EVA02-CLIP-bigE-14", "BAAI/EVA02-CLIP-bigE-14", "vision_encoder"),
    _entry("CLIP-ViT-L-336", "openai/clip-vit-large-patch14-336", "text_encoder"),
    _entry("CLIP-ViT-L-laion2B", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "text_encoder"),
    # Detection / OCR / layout
    _entry("GroundingDINO-Base", "IDEA-Research/grounding-dino-base", "detector"),
    _entry("GroundingDINO-Tiny", "IDEA-Research/grounding-dino-tiny", "detector"),
    _entry("GroundingDINO-SwinT", "IDEA-Research/grounding-dino-swint-ogc", "detector"),
    _entry("GroundingDINO-1.5-SwinT", "IDEA-Research/grounding-dino-1.5-swint-ogc", "detector"),
    _entry("CountGD", "nikigoli/CountGD", "detector"),
    _entry("OwlViT-base-patch32", "google/owlvit-base-patch32", "detector"),
    _entry("OWLv2-base-patch16-ensemble", "google/owlv2-base-patch16-ensemble", "detector"),
    _entry("OwlViT-large-patch14", "google/owlvit-large-patch14", "detector"),
    _entry("OWLv2-Large", "google/owlv2-large-patch14-ensemble", "detector"),
    _entry("Table-Transformer-Detection", "microsoft/table-transformer-detection", "detector"),
    _entry("TrOCR-Large-Printed", "microsoft/trocr-large-printed", "ocr"),
    _entry("TrOCR-base-handwritten", "microsoft/trocr-base-handwritten", "ocr"),
    _entry("TrOCR-small-printed", "microsoft/trocr-small-printed", "ocr"),
    _entry("GOT-OCR2", "stepfun-ai/GOT-OCR2_0", "ocr"),
    _entry("CRAFT-text-detector", "boomb0om/CRAFT-text-detector", "ocr"),
    _entry("Donut-base", "naver-clova-ix/donut-base", "ocr"),
    _entry("Donut-docvqa", "naver-clova-ix/donut-base-finetuned-docvqa", "ocr"),
    _entry("LayoutLMv3-base", "microsoft/layoutlmv3-base", "ocr"),
    _entry("TrOCR-large-handwritten", "microsoft/trocr-large-handwritten", "ocr"),
    _entry("TrOCR-base-printed", "microsoft/trocr-base-printed", "ocr"),
    _entry("Pix2Struct-base", "google/pix2struct-base", "ocr"),
    _entry("DePlot", "google/deplot", "ocr"),
    _entry("vit-gpt2-coco", "nlpconnect/vit-gpt2-image-captioning-finetuned-coco", "caption"),
    _entry("NSFW-Detector", "Falconsai/nsfw_image_detection", "safety"),
    _entry("NSFW-Alt-Detector", "AdamCodd/vit-base-nsfw-detector", "safety"),
    _entry("Watermark-Detector", "prithivMLmods/Watermark-Detection-Model-10K-v1", "qa"),
    # Depth / control / segmentation
    _entry("Depth-Anything-V2-Large", "depth-anything/Depth-Anything-V2-Large", "depth"),
    _entry("Depth-Anything-V2-Small", "depth-anything/Depth-Anything-V2-Small-hf", "depth"),
    _entry("Marigold-Depth-v1-1", "prs-eth/marigold-depth-v1-1", "depth"),
    _entry("Marigold-Normals-v1-1", "prs-eth/marigold-normals-v1-1", "normals"),
    _entry("ControlNet-Canny", "lllyasviel/sd-controlnet-canny", "control"),
    _entry("ControlNet-Depth", "lllyasviel/sd-controlnet-depth", "control"),
    _entry("ControlNet-OpenPose", "lllyasviel/sd-controlnet-openpose", "control"),
    _entry("ControlNet-Lineart", "lllyasviel/sd-controlnet-lineart", "control"),
    _entry("ControlNet-Scribble", "lllyasviel/sd-controlnet-scribble", "control"),
    _entry("ControlNet-MLSD", "lllyasviel/sd-controlnet-mlsd", "control"),
    _entry("ControlNet-SoftEdge", "lllyasviel/sd-controlnet-softedge", "control"),
    _entry("ControlNet-Seg", "lllyasviel/sd-controlnet-seg", "control"),
    _entry("ControlNet-Normal", "lllyasviel/sd-controlnet-normal", "control"),
    _entry("ControlNet-HED", "lllyasviel/sd-controlnet-hed", "control"),
    _entry("ControlNet-Canny-SDXL", "diffusers/controlnet-canny-sdxl-1.0", "control"),
    _entry("ControlNet-Depth-SDXL", "diffusers/controlnet-depth-sdxl-1.0", "control"),
    _entry("ControlNet-OpenPose-SDXL", "diffusers/controlnet-openpose-sdxl-1.0", "control"),
    _entry("ControlNet-Union-SDXL", "xinsir/controlnet-union-sdxl-1.0-promax", "control"),
    _entry("DPT-Hybrid-Midas", "Intel/dpt-hybrid-midas", "depth"),
    _entry("DPT-Large", "Intel/dpt-large", "depth"),
    _entry("Depth-Anything-V2-Base", "depth-anything/Depth-Anything-V2-Base-hf", "depth"),
    _entry("ZoeDepth", "Intel/zoedepth-nyu-kitti", "depth"),
    _entry("Metric3D-ViT-Small", "JUGGHM/Metric3D", "depth"),
    _entry("SAM2-Hiera-Large", "facebook/sam2-hiera-large-hf", "segmentation"),
    _entry("SAM2-Hiera-Base", "facebook/sam2-hiera-base-plus-hf", "segmentation"),
    _entry("SAM2-Hiera-Small", "facebook/sam2-hiera-small-hf", "segmentation"),
    _entry("SAM2-Hiera-Tiny", "facebook/sam2-hiera-tiny-hf", "segmentation"),
    _entry("SAM-ViT-Base", "facebook/sam-vit-base", "segmentation"),
    _entry("SAM-ViT-Huge", "facebook/sam-vit-huge", "segmentation"),
    _entry("DETR-ResNet50", "facebook/detr-resnet-50", "detector"),
    _entry("Mask2Former-Swin-Base", "facebook/mask2former-swin-base-coco-panoptic", "segmentation"),
    # VAE / decode / restore
    _entry("StableCascade-Prior", "stabilityai/stable-cascade-prior", "cascade"),
    _entry("StableCascade-Decoder", "stabilityai/stable-cascade", "cascade"),
    _entry("TAESD", "madebyollin/taesd", "vae"),
    _entry("TAESDXL", "madebyollin/taesdxl", "vae"),
    _entry("sd-vae-ft-mse", "stabilityai/sd-vae-ft-mse", "vae"),
    _entry("sd-vae-ft-ema", "stabilityai/sd-vae-ft-ema", "vae"),
    _entry("sdxl-vae", "stabilityai/sdxl-vae", "vae"),
    _entry("sdxl-vae-fp16-fix", "madebyollin/sdxl-vae-fp16-fix", "vae"),
    _entry("Consistency-Decoder", "openai/consistency-decoder", "vae"),
    _entry("CodeFormer", "sczhou/CodeFormer", "face_restore"),
    _entry("GFPGAN", "TencentARC/GFPGAN", "face_restore"),
    _entry("Real-ESRGAN", "ai-forever/Real-ESRGAN", "upscale"),
    _entry("SwinIR-classical-x2", "caidas/swin2SR-classical-sr-x2-64", "upscale"),
    _entry("AnyDoor-Ref", "camenduru/AnyDoor", "reference"),
)


def scaffold_registry() -> List[HFModelEntry]:
    return list(HF_SCAFFOLD_REGISTRY)


def scaffold_by_role(role: str) -> List[HFModelEntry]:
    return [e for e in HF_SCAFFOLD_REGISTRY if e.role == role]


def has_local_weights(local_dir: str | Path) -> bool:
    p = Path(local_dir)
    if not p.is_dir():
        return False
    for root, _dirs, files in os.walk(p):
        for fname in files:
            if fname.endswith(_WEIGHT_SUFFIXES):
                return True
    return False


def download_scaffold_entry(
    entry: HFModelEntry,
    *,
    model_root: str | Path | None = None,
    max_workers: int = 4,
    skip_existing: bool = True,
) -> str:
    """Download config-only scaffold for one registry entry."""
    from huggingface_hub import snapshot_download

    root = Path(model_root) if model_root is not None else model_dir()
    local_dir = root / entry.local_folder
    os.makedirs(local_dir, exist_ok=True)
    if skip_existing and local_dir.is_dir() and any(local_dir.iterdir()):
        print(f"Skipping (folder exists): {entry.repo_id} -> {local_dir}")
        return str(local_dir)
    print(f"Scaffold (config-only): {entry.repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=entry.repo_id,
        local_dir=str(local_dir),
        allow_patterns=list(CONFIG_ONLY_ALLOW),
        ignore_patterns=list(CONFIG_ONLY_IGNORE),
        max_workers=max_workers,
    )
    return str(local_dir)


def download_scaffold_batch(
    entries: Sequence[HFModelEntry],
    *,
    model_root: str | Path | None = None,
    max_workers: int = 4,
    skip_existing: bool = True,
    continue_on_error: bool = True,
) -> dict[str, object]:
    ok: List[str] = []
    failed: List[str] = []
    for entry in entries:
        try:
            download_scaffold_entry(
                entry,
                model_root=model_root,
                max_workers=max_workers,
                skip_existing=skip_existing,
            )
            ok.append(entry.name)
        except Exception as exc:
            msg = f"{entry.name} ({entry.repo_id}): {exc}"
            failed.append(msg)
            if not continue_on_error:
                raise
    return {
        "ok": ok,
        "failed": failed,
        "model_root": str(model_root or model_dir()),
        "repo_root": str(repo_root()),
    }


def resolve_entries(
    *,
    names: Optional[Sequence[str]] = None,
    roles: Optional[Sequence[str]] = None,
) -> List[HFModelEntry]:
    entries = list(HF_SCAFFOLD_REGISTRY)
    if roles:
        role_set = {str(r).strip().lower() for r in roles}
        entries = [e for e in entries if e.role.lower() in role_set]
    if names:
        name_set = {str(n).strip() for n in names}
        entries = [e for e in entries if e.name in name_set]
    return entries


__all__ = [
    "CONFIG_ONLY_ALLOW",
    "CONFIG_ONLY_IGNORE",
    "HFModelEntry",
    "HF_SCAFFOLD_REGISTRY",
    "download_scaffold_batch",
    "download_scaffold_entry",
    "has_local_weights",
    "resolve_entries",
    "scaffold_by_role",
    "scaffold_registry",
]
