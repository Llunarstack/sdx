"""Resolve local `pretrained/` paths vs Hugging Face hub IDs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence


def repo_root() -> Path:
    # utils/modeling/model_paths.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    return repo_root() / "pretrained"


def resolve_model_path(folder_name: str, hf_fallback: str) -> str:
    """
    If `model/<folder_name>` exists and is non-empty, use it.
    Otherwise return `hf_fallback` (hub id).
    """
    local_model_path = model_dir() / folder_name
    if local_model_path.is_dir():
        try:
            if any(local_model_path.iterdir()):
                return str(local_model_path)
        except OSError:
            pass
    return hf_fallback


def resolve_model_path_require_weights(
    folder_name: str,
    hf_fallback: str,
    *,
    weight_globs: Optional[Sequence[str]] = None,
) -> str:
    """
    Like `resolve_model_path`, but only uses the local folder if it appears to
    contain model weights (so "config-only" minimal downloads won't break
    runtime by masking the HF fallback).

    `weight_globs` defaults to common HF weight file extensions.
    """
    local_model_path = model_dir() / folder_name
    if not local_model_path.is_dir():
        return hf_fallback

    patterns = list(weight_globs or ("*.safetensors", "*.bin", "*.pt", "*.pth"))
    try:
        for pat in patterns:
            if any(local_model_path.glob(pat)):
                return str(local_model_path)
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


def default_dinov2_giant_path() -> str:
    return resolve_model_path("DINOv2-Giant", "facebook/dinov2-giant")


def default_siglip_path() -> str:
    return resolve_model_path("SigLIP-SO400M", "google/siglip-so400m-patch14-384")


def default_qwen_path() -> str:
    return resolve_model_path("Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct")


def default_cascade_prior_path() -> str:
    return resolve_model_path("StableCascade-Prior", "stabilityai/stable-cascade-prior")


def default_cascade_decoder_path() -> str:
    return resolve_model_path("StableCascade-Decoder", "stabilityai/stable-cascade")


def default_gen_searcher_8b_path() -> str:
    """
    Local folder or Hugging Face repo id for Gen-Searcher-8B.

    Local folder convention:
        pretrained/GenSearcher-8B
    """
    return resolve_model_path("GenSearcher-8B", "GenSearcher/Gen-Searcher-8B")


def default_image_reward_path() -> str:
    return resolve_model_path("ImageReward", "zai-org/ImageReward")


def default_pickscore_path() -> str:
    return resolve_model_path("PickScore_v1", "yuvalkirstain/PickScore_v1")


def default_grounding_dino_base_path() -> str:
    return resolve_model_path("GroundingDINO-Base", "IDEA-Research/grounding-dino-base")


def default_countgd_path() -> str:
    return resolve_model_path("CountGD", "nikigoli/CountGD")


def default_trocr_large_printed_path() -> str:
    return resolve_model_path("TrOCR-Large-Printed", "microsoft/trocr-large-printed")


def default_perceptclip_iqa_path() -> str:
    return resolve_model_path("PerceptCLIP_IQA", "PerceptCLIP/PerceptCLIP_IQA")


def default_depth_anything_v2_large_path() -> str:
    return resolve_model_path_require_weights(
        "Depth-Anything-V2-Large",
        "depth-anything/Depth-Anything-V2-Large",
    )


def default_sam2_hiera_large_path() -> str:
    return resolve_model_path("SAM2-Hiera-Large", "facebook/sam2-hiera-large-hf")


def default_realesrgan_path() -> str:
    return resolve_model_path("Real-ESRGAN", "ai-forever/Real-ESRGAN")


def default_longclip_l_path() -> str:
    return resolve_model_path("LongCLIP-L", "creative-graphic-design/LongCLIP-L")


def default_moondream2_path() -> str:
    return resolve_model_path("moondream2", "vikhyatoolkit/moondream2")


def default_marigold_depth_path() -> str:
    return resolve_model_path("Marigold-Depth-v1-1", "prs-eth/marigold-depth-v1-1")


def default_marigold_normals_path() -> str:
    return resolve_model_path("Marigold-Normals-v1-1", "prs-eth/marigold-normals-v1-1")


def default_taesd_path() -> str:
    return resolve_model_path("TAESD", "madebyollin/taesd")


def default_taesdxl_path() -> str:
    return resolve_model_path("TAESDXL", "madebyollin/taesdxl")


def default_codeformer_path() -> str:
    return resolve_model_path("CodeFormer", "sczhou/CodeFormer")


def default_consistency_decoder_path() -> str:
    return resolve_model_path("Consistency-Decoder", "openai/consistency-decoder")


def default_convnextv2_large_path() -> str:
    return resolve_model_path("ConvNeXtV2-Large", "facebook/convnextv2-large-22k-384")


def default_laion_aesthetic_v2_path() -> str:
    return resolve_model_path("LAION-Aesthetic-v2", "camenduru/improved-aesthetic-predictor")


def default_anydoor_ref_path() -> str:
    return resolve_model_path("AnyDoor-Ref", "camenduru/AnyDoor")


# ---------------------------------------------------------------------------
# Additional optional "boost" models (critics / VLM / detectors).
# These are often downloaded "config-only" to save disk. For those, use
# weight-aware resolution so HF fallback is used unless weights exist locally.
# ---------------------------------------------------------------------------


def default_hpsv2_path() -> str:
    return resolve_model_path_require_weights("HPSv2-hf", "adams-story/HPSv2-hf")


def default_blip_caption_base_path() -> str:
    return resolve_model_path_require_weights(
        "BLIP-image-captioning-base",
        "Salesforce/blip-image-captioning-base",
    )


def default_kosmos2_path() -> str:
    return resolve_model_path_require_weights(
        "Kosmos-2-patch14-224",
        "microsoft/kosmos-2-patch14-224",
    )


def default_craft_text_detector_path() -> str:
    # This one is small enough that we typically have weights locally.
    return resolve_model_path_require_weights(
        "CRAFT-text-detector",
        "boomb0om/CRAFT-text-detector",
    )


def default_owlvit_base_path() -> str:
    return resolve_model_path_require_weights(
        "OwlViT-base-patch32",
        "google/owlvit-base-patch32",
    )


def default_donut_base_path() -> str:
    return resolve_model_path_require_weights(
        "Donut-base",
        "naver-clova-ix/donut-base",
    )


def default_florence2_base_path() -> str:
    return resolve_model_path_require_weights(
        "Florence-2-base",
        "microsoft/Florence-2-base",
    )


def default_blip_caption_large_path() -> str:
    return resolve_model_path_require_weights(
        "BLIP-image-captioning-large",
        "Salesforce/blip-image-captioning-large",
    )


def default_blip2_opt_path() -> str:
    return resolve_model_path_require_weights(
        "BLIP2-opt-2.7b",
        "Salesforce/blip2-opt-2.7b",
    )


def default_qwen2_vl_2b_path() -> str:
    return resolve_model_path_require_weights(
        "Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2-VL-2B-Instruct",
    )


def default_minicpm_v2_path() -> str:
    return resolve_model_path_require_weights(
        "MiniCPM-V-2",
        "openbmb/MiniCPM-V-2",
    )


def default_clip_h14_path() -> str:
    return resolve_model_path_require_weights(
        "CLIP-ViT-H-14",
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    )


def default_siglip_base_path() -> str:
    return resolve_model_path_require_weights(
        "SigLIP-base-patch16-224",
        "google/siglip-base-patch16-224",
    )


def default_depth_anything_v2_small_path() -> str:
    return resolve_model_path_require_weights(
        "Depth-Anything-V2-Small",
        "depth-anything/Depth-Anything-V2-Small-hf",
    )


def default_owlv2_base_path() -> str:
    return resolve_model_path_require_weights(
        "OWLv2-base-patch16-ensemble",
        "google/owlv2-base-patch16-ensemble",
    )


def default_controlnet_canny_path() -> str:
    return resolve_model_path_require_weights(
        "ControlNet-Canny",
        "lllyasviel/sd-controlnet-canny",
    )


def default_controlnet_depth_path() -> str:
    return resolve_model_path_require_weights(
        "ControlNet-Depth",
        "lllyasviel/sd-controlnet-depth",
    )


def default_controlnet_openpose_path() -> str:
    return resolve_model_path_require_weights(
        "ControlNet-OpenPose",
        "lllyasviel/sd-controlnet-openpose",
    )


def default_controlnet_lineart_path() -> str:
    return resolve_model_path_require_weights(
        "ControlNet-Lineart",
        "lllyasviel/sd-controlnet-lineart",
    )


def default_controlnet_scribble_path() -> str:
    return resolve_model_path_require_weights(
        "ControlNet-Scribble",
        "lllyasviel/sd-controlnet-scribble",
    )


def default_dpt_hybrid_midas_path() -> str:
    return resolve_model_path_require_weights(
        "DPT-Hybrid-Midas",
        "Intel/dpt-hybrid-midas",
    )


def default_depth_anything_v2_base_path() -> str:
    return resolve_model_path_require_weights(
        "Depth-Anything-V2-Base",
        "depth-anything/Depth-Anything-V2-Base-hf",
    )


def default_metaclip_b16_path() -> str:
    return resolve_model_path_require_weights(
        "MetaCLIP-B16",
        "facebook/metaclip-b16-fullcc2.5b",
    )


def default_aimv2_large_path() -> str:
    return resolve_model_path_require_weights(
        "AIMv2-Large",
        "apple/aimv2-large-patch14-384",
    )


def default_altclip_h14_path() -> str:
    return resolve_model_path_require_weights(
        "AltCLIP-ViT-H-14",
        "BAAI/AltCLIP-ViT-H-14",
    )


def default_smolvlm_256m_path() -> str:
    return resolve_model_path_require_weights(
        "SmolVLM-256M-Instruct",
        "HuggingFaceTB/SmolVLM-256M-Instruct",
    )


def default_paligemma_3b_path() -> str:
    return resolve_model_path_require_weights(
        "PaliGemma-3B",
        "google/paligemma-3b-pt-224",
    )


def default_minicpm_llama3_v_path() -> str:
    return resolve_model_path_require_weights(
        "MiniCPM-Llama3-V-2_5",
        "openbmb/MiniCPM-Llama3-V-2_5",
    )


def default_vit_gpt2_caption_path() -> str:
    return resolve_model_path_require_weights(
        "vit-gpt2-image-captioning",
        "nlpconnect/vit-gpt2-image-captioning",
    )


def default_instructblip_vicuna_7b_path() -> str:
    return resolve_model_path_require_weights(
        "InstructBLIP-Vicuna-7B",
        "Salesforce/instructblip-vicuna-7b",
    )


def default_florence2_large_path() -> str:
    return resolve_model_path_require_weights(
        "Florence-2-large",
        "microsoft/Florence-2-large",
    )


def default_onealign_path() -> str:
    return resolve_model_path_require_weights(
        "OneAlign",
        "Q-Future/OneAlign",
    )


def default_grounding_dino_tiny_path() -> str:
    return resolve_model_path_require_weights(
        "GroundingDINO-Tiny",
        "IDEA-Research/grounding-dino-tiny",
    )


def default_owlvit_large_path() -> str:
    return resolve_model_path_require_weights(
        "OwlViT-large-patch14",
        "google/owlvit-large-patch14",
    )


def default_trocr_base_handwritten_path() -> str:
    return resolve_model_path_require_weights(
        "TrOCR-base-handwritten",
        "microsoft/trocr-base-handwritten",
    )


def default_sam_vit_base_path() -> str:
    return resolve_model_path_require_weights(
        "SAM-ViT-Base",
        "facebook/sam-vit-base",
    )


def default_cafe_aesthetic_path() -> str:
    return resolve_model_path_require_weights(
        "CAFE-Aesthetic",
        "cafeai/cafe_aesthetic_classifier",
    )


def default_controlnet_mlsd_path() -> str:
    return resolve_model_path_require_weights(
        "ControlNet-MLSD",
        "lllyasviel/sd-controlnet-mlsd",
    )


def default_controlnet_softedge_path() -> str:
    return resolve_model_path_require_weights(
        "ControlNet-SoftEdge",
        "lllyasviel/sd-controlnet-softedge",
    )


def default_controlnet_seg_path() -> str:
    return resolve_model_path_require_weights(
        "ControlNet-Seg",
        "lllyasviel/sd-controlnet-seg",
    )


def default_controlnet_normal_path() -> str:
    return resolve_model_path_require_weights(
        "ControlNet-Normal",
        "lllyasviel/sd-controlnet-normal",
    )


def default_dpt_large_path() -> str:
    return resolve_model_path_require_weights(
        "DPT-Large",
        "Intel/dpt-large",
    )


def default_grounding_dino_swint_path() -> str:
    return resolve_model_path_require_weights(
        "GroundingDINO-SwinT",
        "IDEA-Research/grounding-dino-swint-ogc",
    )


def default_owlv2_large_path() -> str:
    return resolve_model_path_require_weights(
        "OWLv2-Large",
        "google/owlv2-large-patch14-ensemble",
    )


def default_table_transformer_detection_path() -> str:
    return resolve_model_path_require_weights(
        "Table-Transformer-Detection",
        "microsoft/table-transformer-detection",
    )


def default_smolvlm2_2b_path() -> str:
    return resolve_model_path_require_weights(
        "SmolVLM2-2B-Instruct",
        "HuggingFaceTB/SmolVLM2-2B-Instruct",
    )


def default_qwen2_vl_7b_path() -> str:
    return resolve_model_path_require_weights(
        "Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
    )


def default_phi35_vision_path() -> str:
    return resolve_model_path_require_weights(
        "Phi-3.5-vision-instruct",
        "microsoft/Phi-3.5-vision-instruct",
    )


def default_florence2_base_ft_path() -> str:
    return resolve_model_path_require_weights(
        "Florence-2-base-ft",
        "microsoft/Florence-2-base-ft",
    )


def default_git_base_coco_path() -> str:
    return resolve_model_path_require_weights(
        "GIT-base-coco",
        "microsoft/git-base-coco",
    )


def default_git_large_coco_path() -> str:
    return resolve_model_path_require_weights(
        "GIT-large-coco",
        "microsoft/git-large-coco",
    )


def default_qwen25_7b_path() -> str:
    return resolve_model_path_require_weights(
        "Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    )


def default_smollm2_360m_path() -> str:
    return resolve_model_path_require_weights(
        "SmolLM2-360M-Instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
    )


def default_clip_l336_path() -> str:
    return resolve_model_path_require_weights(
        "CLIP-ViT-L-336",
        "openai/clip-vit-large-patch14-336",
    )


def default_clip_l_laion2b_path() -> str:
    return resolve_model_path_require_weights(
        "CLIP-ViT-L-laion2B",
        "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    )


def default_sam_vit_huge_path() -> str:
    return resolve_model_path_require_weights(
        "SAM-ViT-Huge",
        "facebook/sam-vit-huge",
    )


def default_trocr_small_printed_path() -> str:
    return resolve_model_path_require_weights(
        "TrOCR-small-printed",
        "microsoft/trocr-small-printed",
    )


def default_got_ocr2_path() -> str:
    return resolve_model_path_require_weights(
        "GOT-OCR2",
        "stepfun-ai/GOT-OCR2_0",
    )


def default_eva02_clip_l14_path() -> str:
    return resolve_model_path_require_weights(
        "EVA02-CLIP-L-14",
        "BAAI/EVA02-CLIP-L-14",
    )


def default_musiq_path() -> str:
    return resolve_model_path_require_weights(
        "MUSIQ",
        "chengaz/MUSIQ",
    )


def default_t5_xl_path() -> str:
    return resolve_model_path_require_weights("T5-XL", "google/t5-v1_1-xl")


def default_t5_large_path() -> str:
    return resolve_model_path_require_weights("T5-Large", "google/t5-v1_1-large")


def default_clip_b32_path() -> str:
    return resolve_model_path_require_weights(
        "CLIP-ViT-B-32",
        "openai/clip-vit-base-patch32",
    )


def default_siglip2_so400m_path() -> str:
    return resolve_model_path_require_weights(
        "SigLIP2-SO400M",
        "google/siglip2-so400m-patch16-384",
    )


def default_dinov2_base_path() -> str:
    return resolve_model_path_require_weights("DINOv2-Base", "facebook/dinov2-base")


def default_dinov2_small_path() -> str:
    return resolve_model_path_require_weights("DINOv2-Small", "facebook/dinov2-small")


def default_eva02_clip_bige14_path() -> str:
    return resolve_model_path_require_weights(
        "EVA02-CLIP-bigE-14",
        "BAAI/EVA02-CLIP-bigE-14",
    )


def default_longclip_b_path() -> str:
    return resolve_model_path_require_weights(
        "LongCLIP-B",
        "creative-graphic-design/LongCLIP-B",
    )


def default_llava_15_7b_path() -> str:
    return resolve_model_path_require_weights(
        "LLaVA-1.5-7B",
        "llava-hf/llava-1.5-7b-hf",
    )


def default_internvl2_2b_path() -> str:
    return resolve_model_path_require_weights(
        "InternVL2-2B",
        "OpenGVLab/InternVL2-2B",
    )


def default_qwen25_vl_3b_path() -> str:
    return resolve_model_path_require_weights(
        "Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    )


def default_blip2_flan_t5_xl_path() -> str:
    return resolve_model_path_require_weights(
        "BLIP2-Flan-T5-XL",
        "Salesforce/blip2-flan-t5-xl",
    )


def default_phi3_vision_path() -> str:
    return resolve_model_path_require_weights(
        "Phi-3-vision-128k-instruct",
        "microsoft/Phi-3-vision-128k-instruct",
    )


def default_controlnet_hed_path() -> str:
    return resolve_model_path_require_weights(
        "ControlNet-HED",
        "lllyasviel/sd-controlnet-hed",
    )


def default_controlnet_canny_sdxl_path() -> str:
    return resolve_model_path_require_weights(
        "ControlNet-Canny-SDXL",
        "diffusers/controlnet-canny-sdxl-1.0",
    )


def default_controlnet_depth_sdxl_path() -> str:
    return resolve_model_path_require_weights(
        "ControlNet-Depth-SDXL",
        "diffusers/controlnet-depth-sdxl-1.0",
    )


def default_zoedepth_path() -> str:
    return resolve_model_path_require_weights(
        "ZoeDepth",
        "Intel/zoedepth-nyu-kitti",
    )


def default_metric3d_vit_small_path() -> str:
    return resolve_model_path_require_weights(
        "Metric3D-ViT-Small",
        "JUGGHM/Metric3D",
    )


def default_detr_resnet50_path() -> str:
    return resolve_model_path_require_weights(
        "DETR-ResNet50",
        "facebook/detr-resnet-50",
    )


def default_mask2former_swin_base_path() -> str:
    return resolve_model_path_require_weights(
        "Mask2Former-Swin-Base",
        "facebook/mask2former-swin-base-coco-panoptic",
    )


def default_sam2_hiera_tiny_path() -> str:
    return resolve_model_path_require_weights(
        "SAM2-Hiera-Tiny",
        "facebook/sam2-hiera-tiny-hf",
    )


def default_layoutlmv3_base_path() -> str:
    return resolve_model_path_require_weights(
        "LayoutLMv3-base",
        "microsoft/layoutlmv3-base",
    )


def default_trocr_large_handwritten_path() -> str:
    return resolve_model_path_require_weights(
        "TrOCR-large-handwritten",
        "microsoft/trocr-large-handwritten",
    )


def default_donut_docvqa_path() -> str:
    return resolve_model_path_require_weights(
        "Donut-docvqa",
        "naver-clova-ix/donut-base-finetuned-docvqa",
    )


def default_nsfw_detector_path() -> str:
    return resolve_model_path_require_weights(
        "NSFW-Detector",
        "Falconsai/nsfw_image_detection",
    )


def default_sd_vae_ft_mse_path() -> str:
    return resolve_model_path_require_weights(
        "sd-vae-ft-mse",
        "stabilityai/sd-vae-ft-mse",
    )


def default_sdxl_vae_fp16_fix_path() -> str:
    return resolve_model_path_require_weights(
        "sdxl-vae-fp16-fix",
        "madebyollin/sdxl-vae-fp16-fix",
    )


def default_llava_v16_mistral_7b_path() -> str:
    return resolve_model_path_require_weights(
        "LLaVA-v1.6-Mistral-7B",
        "llava-hf/llava-v1.6-mistral-7b-hf",
    )


def default_qwen25_vl_7b_path() -> str:
    return resolve_model_path_require_weights(
        "Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    )


def default_internvl2_1b_path() -> str:
    return resolve_model_path_require_weights(
        "InternVL2-1B",
        "OpenGVLab/InternVL2-1B",
    )


def default_paligemma2_3b_path() -> str:
    return resolve_model_path_require_weights(
        "PaliGemma2-3B",
        "google/paligemma2-3b-pt-224",
    )


def default_florence2_large_ft_path() -> str:
    return resolve_model_path_require_weights(
        "Florence-2-large-ft",
        "microsoft/Florence-2-large-ft",
    )


def default_mobileclip_s2_path() -> str:
    return resolve_model_path_require_weights(
        "MobileCLIP-S2",
        "apple/MobileCLIP-S2",
    )


def default_dinov3_vits16_path() -> str:
    return resolve_model_path_require_weights(
        "DINOv3-ViT-S16",
        "facebook/dinov3-vits16-pretrain-lvd1689m",
    )


def default_clip_b16_path() -> str:
    return resolve_model_path_require_weights(
        "CLIP-ViT-B-16",
        "openai/clip-vit-base-patch16",
    )


def default_sd_vae_ft_ema_path() -> str:
    return resolve_model_path_require_weights(
        "sd-vae-ft-ema",
        "stabilityai/sd-vae-ft-ema",
    )


def default_sdxl_vae_path() -> str:
    return resolve_model_path_require_weights(
        "sdxl-vae",
        "stabilityai/sdxl-vae",
    )


def default_controlnet_openpose_sdxl_path() -> str:
    return resolve_model_path_require_weights(
        "ControlNet-OpenPose-SDXL",
        "diffusers/controlnet-openpose-sdxl-1.0",
    )


def default_controlnet_union_sdxl_path() -> str:
    return resolve_model_path_require_weights(
        "ControlNet-Union-SDXL",
        "xinsir/controlnet-union-sdxl-1.0-promax",
    )


def default_sam2_hiera_base_path() -> str:
    return resolve_model_path_require_weights(
        "SAM2-Hiera-Base",
        "facebook/sam2-hiera-base-plus-hf",
    )


def default_sam2_hiera_small_path() -> str:
    return resolve_model_path_require_weights(
        "SAM2-Hiera-Small",
        "facebook/sam2-hiera-small-hf",
    )


def default_grounding_dino_15_swint_path() -> str:
    return resolve_model_path_require_weights(
        "GroundingDINO-1.5-SwinT",
        "IDEA-Research/grounding-dino-1.5-swint-ogc",
    )


def default_pix2struct_base_path() -> str:
    return resolve_model_path_require_weights(
        "Pix2Struct-base",
        "google/pix2struct-base",
    )


def default_deplot_path() -> str:
    return resolve_model_path_require_weights(
        "DePlot",
        "google/deplot",
    )


def default_trocr_base_printed_path() -> str:
    return resolve_model_path_require_weights(
        "TrOCR-base-printed",
        "microsoft/trocr-base-printed",
    )


def default_watermark_detector_path() -> str:
    return resolve_model_path_require_weights(
        "Watermark-Detector",
        "prithivMLmods/Watermark-Detection-Model-10K-v1",
    )


def default_vit_gpt2_coco_path() -> str:
    return resolve_model_path_require_weights(
        "vit-gpt2-coco",
        "nlpconnect/vit-gpt2-image-captioning-finetuned-coco",
    )


def default_swinir_classical_path() -> str:
    return resolve_model_path_require_weights(
        "SwinIR-classical-x2",
        "caidas/swin2SR-classical-sr-x2-64",
    )


def default_gfpgan_path() -> str:
    return resolve_model_path_require_weights(
        "GFPGAN",
        "TencentARC/GFPGAN",
    )


def default_umt5_xxl_path() -> str:
    return resolve_model_path_require_weights(
        "UMT5-XXL",
        "google/umt5-xxl",
    )


def default_qwen25_3b_path() -> str:
    return resolve_model_path_require_weights(
        "Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    )


def default_minicpm_v26_path() -> str:
    return resolve_model_path_require_weights(
        "MiniCPM-V-2_6",
        "openbmb/MiniCPM-V-2_6",
    )


def default_nsfw_alt_detector_path() -> str:
    return resolve_model_path_require_weights(
        "NSFW-Alt-Detector",
        "AdamCodd/vit-base-nsfw-detector",
    )


def default_clip_iqa_path() -> str:
    return resolve_model_path_require_weights(
        "CLIP-IQA",
        "djghosh/aesthetics-scorer",
    )


def pretrained_catalog() -> List[Dict[str, str]]:
    """
    Canonical pretrained model map used by SDX.

    Returns list rows with:
      - name
      - local_folder
      - hf_fallback
      - resolved
    """
    catalog_rows = [
        ("T5-XXL", "google/t5-v1_1-xxl", default_t5_path()),
        ("CLIP-ViT-L-14", "openai/clip-vit-large-patch14", default_clip_l_path()),
        ("CLIP-ViT-bigG-14", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", default_clip_bigg_path()),
        ("DINOv2-Large", "facebook/dinov2-large", default_dinov2_large_path()),
        ("DINOv2-Giant", "facebook/dinov2-giant", default_dinov2_giant_path()),
        ("SigLIP-SO400M", "google/siglip-so400m-patch14-384", default_siglip_path()),
        ("Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct", default_qwen_path()),
        ("StableCascade-Prior", "stabilityai/stable-cascade-prior", default_cascade_prior_path()),
        ("StableCascade-Decoder", "stabilityai/stable-cascade", default_cascade_decoder_path()),
        ("GenSearcher-8B", "GenSearcher/Gen-Searcher-8B", default_gen_searcher_8b_path()),
        ("ImageReward", "zai-org/ImageReward", default_image_reward_path()),
        ("PickScore_v1", "yuvalkirstain/PickScore_v1", default_pickscore_path()),
        ("GroundingDINO-Base", "IDEA-Research/grounding-dino-base", default_grounding_dino_base_path()),
        ("CountGD", "nikigoli/CountGD", default_countgd_path()),
        ("TrOCR-Large-Printed", "microsoft/trocr-large-printed", default_trocr_large_printed_path()),
        ("PerceptCLIP_IQA", "PerceptCLIP/PerceptCLIP_IQA", default_perceptclip_iqa_path()),
        ("Depth-Anything-V2-Large", "depth-anything/Depth-Anything-V2-Large", default_depth_anything_v2_large_path()),
        ("SAM2-Hiera-Large", "facebook/sam2-hiera-large-hf", default_sam2_hiera_large_path()),
        ("Real-ESRGAN", "ai-forever/Real-ESRGAN", default_realesrgan_path()),
        ("LongCLIP-L", "creative-graphic-design/LongCLIP-L", default_longclip_l_path()),
        ("moondream2", "vikhyatoolkit/moondream2", default_moondream2_path()),
        ("Marigold-Depth-v1-1", "prs-eth/marigold-depth-v1-1", default_marigold_depth_path()),
        ("Marigold-Normals-v1-1", "prs-eth/marigold-normals-v1-1", default_marigold_normals_path()),
        ("TAESD", "madebyollin/taesd", default_taesd_path()),
        ("TAESDXL", "madebyollin/taesdxl", default_taesdxl_path()),
        ("CodeFormer", "sczhou/CodeFormer", default_codeformer_path()),
        ("Consistency-Decoder", "openai/consistency-decoder", default_consistency_decoder_path()),
        ("ConvNeXtV2-Large", "facebook/convnextv2-large-22k-384", default_convnextv2_large_path()),
        ("LAION-Aesthetic-v2", "christophschuhmann/improved-aesthetic-predictor", default_laion_aesthetic_v2_path()),
        ("AnyDoor-Ref", "camenduru/AnyDoor", default_anydoor_ref_path()),
        # Boost models (critics / VLM / detectors)
        ("HPSv2-hf", "adams-story/HPSv2-hf", default_hpsv2_path()),
        ("BLIP-image-captioning-base", "Salesforce/blip-image-captioning-base", default_blip_caption_base_path()),
        ("Kosmos-2-patch14-224", "microsoft/kosmos-2-patch14-224", default_kosmos2_path()),
        ("CRAFT-text-detector", "boomb0om/CRAFT-text-detector", default_craft_text_detector_path()),
        ("OwlViT-base-patch32", "google/owlvit-base-patch32", default_owlvit_base_path()),
        ("Donut-base", "naver-clova-ix/donut-base", default_donut_base_path()),
        ("Florence-2-base", "microsoft/Florence-2-base", default_florence2_base_path()),
        ("BLIP-image-captioning-large", "Salesforce/blip-image-captioning-large", default_blip_caption_large_path()),
        ("BLIP2-opt-2.7b", "Salesforce/blip2-opt-2.7b", default_blip2_opt_path()),
        ("Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-2B-Instruct", default_qwen2_vl_2b_path()),
        ("MiniCPM-V-2", "openbmb/MiniCPM-V-2", default_minicpm_v2_path()),
        ("CLIP-ViT-H-14", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", default_clip_h14_path()),
        ("SigLIP-base-patch16-224", "google/siglip-base-patch16-224", default_siglip_base_path()),
        (
            "Depth-Anything-V2-Small",
            "depth-anything/Depth-Anything-V2-Small-hf",
            default_depth_anything_v2_small_path(),
        ),
        ("OWLv2-base-patch16-ensemble", "google/owlv2-base-patch16-ensemble", default_owlv2_base_path()),
        ("ControlNet-Canny", "lllyasviel/sd-controlnet-canny", default_controlnet_canny_path()),
        ("ControlNet-Depth", "lllyasviel/sd-controlnet-depth", default_controlnet_depth_path()),
        ("ControlNet-OpenPose", "lllyasviel/sd-controlnet-openpose", default_controlnet_openpose_path()),
        ("ControlNet-Lineart", "lllyasviel/sd-controlnet-lineart", default_controlnet_lineart_path()),
        ("ControlNet-Scribble", "lllyasviel/sd-controlnet-scribble", default_controlnet_scribble_path()),
        ("DPT-Hybrid-Midas", "Intel/dpt-hybrid-midas", default_dpt_hybrid_midas_path()),
        ("Depth-Anything-V2-Base", "depth-anything/Depth-Anything-V2-Base-hf", default_depth_anything_v2_base_path()),
        ("MetaCLIP-B16", "facebook/metaclip-b16-fullcc2.5b", default_metaclip_b16_path()),
        ("AIMv2-Large", "apple/aimv2-large-patch14-384", default_aimv2_large_path()),
        ("AltCLIP-ViT-H-14", "BAAI/AltCLIP-ViT-H-14", default_altclip_h14_path()),
        ("SmolVLM-256M-Instruct", "HuggingFaceTB/SmolVLM-256M-Instruct", default_smolvlm_256m_path()),
        ("PaliGemma-3B", "google/paligemma-3b-pt-224", default_paligemma_3b_path()),
        ("MiniCPM-Llama3-V-2_5", "openbmb/MiniCPM-Llama3-V-2_5", default_minicpm_llama3_v_path()),
        ("vit-gpt2-image-captioning", "nlpconnect/vit-gpt2-image-captioning", default_vit_gpt2_caption_path()),
        ("InstructBLIP-Vicuna-7B", "Salesforce/instructblip-vicuna-7b", default_instructblip_vicuna_7b_path()),
        ("Florence-2-large", "microsoft/Florence-2-large", default_florence2_large_path()),
        ("OneAlign", "Q-Future/OneAlign", default_onealign_path()),
        ("GroundingDINO-Tiny", "IDEA-Research/grounding-dino-tiny", default_grounding_dino_tiny_path()),
        ("OwlViT-large-patch14", "google/owlvit-large-patch14", default_owlvit_large_path()),
        ("TrOCR-base-handwritten", "microsoft/trocr-base-handwritten", default_trocr_base_handwritten_path()),
        ("SAM-ViT-Base", "facebook/sam-vit-base", default_sam_vit_base_path()),
        ("CAFE-Aesthetic", "cafeai/cafe_aesthetic_classifier", default_cafe_aesthetic_path()),
        ("ControlNet-MLSD", "lllyasviel/sd-controlnet-mlsd", default_controlnet_mlsd_path()),
        ("ControlNet-SoftEdge", "lllyasviel/sd-controlnet-softedge", default_controlnet_softedge_path()),
        ("ControlNet-Seg", "lllyasviel/sd-controlnet-seg", default_controlnet_seg_path()),
        ("ControlNet-Normal", "lllyasviel/sd-controlnet-normal", default_controlnet_normal_path()),
        ("DPT-Large", "Intel/dpt-large", default_dpt_large_path()),
        ("GroundingDINO-SwinT", "IDEA-Research/grounding-dino-swint-ogc", default_grounding_dino_swint_path()),
        ("OWLv2-Large", "google/owlv2-large-patch14-ensemble", default_owlv2_large_path()),
        (
            "Table-Transformer-Detection",
            "microsoft/table-transformer-detection",
            default_table_transformer_detection_path(),
        ),
        ("SmolVLM2-2B-Instruct", "HuggingFaceTB/SmolVLM2-2B-Instruct", default_smolvlm2_2b_path()),
        ("Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", default_qwen2_vl_7b_path()),
        ("Phi-3.5-vision-instruct", "microsoft/Phi-3.5-vision-instruct", default_phi35_vision_path()),
        ("Florence-2-base-ft", "microsoft/Florence-2-base-ft", default_florence2_base_ft_path()),
        ("GIT-base-coco", "microsoft/git-base-coco", default_git_base_coco_path()),
        ("GIT-large-coco", "microsoft/git-large-coco", default_git_large_coco_path()),
        ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", default_qwen25_7b_path()),
        ("SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", default_smollm2_360m_path()),
        ("CLIP-ViT-L-336", "openai/clip-vit-large-patch14-336", default_clip_l336_path()),
        ("CLIP-ViT-L-laion2B", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", default_clip_l_laion2b_path()),
        ("SAM-ViT-Huge", "facebook/sam-vit-huge", default_sam_vit_huge_path()),
        ("TrOCR-small-printed", "microsoft/trocr-small-printed", default_trocr_small_printed_path()),
        ("GOT-OCR2", "stepfun-ai/GOT-OCR2_0", default_got_ocr2_path()),
        ("EVA02-CLIP-L-14", "BAAI/EVA02-CLIP-L-14", default_eva02_clip_l14_path()),
        ("MUSIQ", "chengaz/MUSIQ", default_musiq_path()),
        ("T5-XL", "google/t5-v1_1-xl", default_t5_xl_path()),
        ("T5-Large", "google/t5-v1_1-large", default_t5_large_path()),
        ("CLIP-ViT-B-32", "openai/clip-vit-base-patch32", default_clip_b32_path()),
        ("SigLIP2-SO400M", "google/siglip2-so400m-patch16-384", default_siglip2_so400m_path()),
        ("DINOv2-Base", "facebook/dinov2-base", default_dinov2_base_path()),
        ("DINOv2-Small", "facebook/dinov2-small", default_dinov2_small_path()),
        ("EVA02-CLIP-bigE-14", "BAAI/EVA02-CLIP-bigE-14", default_eva02_clip_bige14_path()),
        ("LongCLIP-B", "creative-graphic-design/LongCLIP-B", default_longclip_b_path()),
        ("LLaVA-1.5-7B", "llava-hf/llava-1.5-7b-hf", default_llava_15_7b_path()),
        ("InternVL2-2B", "OpenGVLab/InternVL2-2B", default_internvl2_2b_path()),
        ("Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct", default_qwen25_vl_3b_path()),
        ("BLIP2-Flan-T5-XL", "Salesforce/blip2-flan-t5-xl", default_blip2_flan_t5_xl_path()),
        ("Phi-3-vision-128k-instruct", "microsoft/Phi-3-vision-128k-instruct", default_phi3_vision_path()),
        ("ControlNet-HED", "lllyasviel/sd-controlnet-hed", default_controlnet_hed_path()),
        ("ControlNet-Canny-SDXL", "diffusers/controlnet-canny-sdxl-1.0", default_controlnet_canny_sdxl_path()),
        ("ControlNet-Depth-SDXL", "diffusers/controlnet-depth-sdxl-1.0", default_controlnet_depth_sdxl_path()),
        ("ZoeDepth", "Intel/zoedepth-nyu-kitti", default_zoedepth_path()),
        ("Metric3D-ViT-Small", "JUGGHM/Metric3D", default_metric3d_vit_small_path()),
        ("DETR-ResNet50", "facebook/detr-resnet-50", default_detr_resnet50_path()),
        ("Mask2Former-Swin-Base", "facebook/mask2former-swin-base-coco-panoptic", default_mask2former_swin_base_path()),
        ("SAM2-Hiera-Tiny", "facebook/sam2-hiera-tiny-hf", default_sam2_hiera_tiny_path()),
        ("LayoutLMv3-base", "microsoft/layoutlmv3-base", default_layoutlmv3_base_path()),
        ("TrOCR-large-handwritten", "microsoft/trocr-large-handwritten", default_trocr_large_handwritten_path()),
        ("Donut-docvqa", "naver-clova-ix/donut-base-finetuned-docvqa", default_donut_docvqa_path()),
        ("NSFW-Detector", "Falconsai/nsfw_image_detection", default_nsfw_detector_path()),
        ("sd-vae-ft-mse", "stabilityai/sd-vae-ft-mse", default_sd_vae_ft_mse_path()),
        ("sdxl-vae-fp16-fix", "madebyollin/sdxl-vae-fp16-fix", default_sdxl_vae_fp16_fix_path()),
        ("LLaVA-v1.6-Mistral-7B", "llava-hf/llava-v1.6-mistral-7b-hf", default_llava_v16_mistral_7b_path()),
        ("Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", default_qwen25_vl_7b_path()),
        ("InternVL2-1B", "OpenGVLab/InternVL2-1B", default_internvl2_1b_path()),
        ("PaliGemma2-3B", "google/paligemma2-3b-pt-224", default_paligemma2_3b_path()),
        ("Florence-2-large-ft", "microsoft/Florence-2-large-ft", default_florence2_large_ft_path()),
        ("MobileCLIP-S2", "apple/MobileCLIP-S2", default_mobileclip_s2_path()),
        ("DINOv3-ViT-S16", "facebook/dinov3-vits16-pretrain-lvd1689m", default_dinov3_vits16_path()),
        ("CLIP-ViT-B-16", "openai/clip-vit-base-patch16", default_clip_b16_path()),
        ("sd-vae-ft-ema", "stabilityai/sd-vae-ft-ema", default_sd_vae_ft_ema_path()),
        ("sdxl-vae", "stabilityai/sdxl-vae", default_sdxl_vae_path()),
        ("ControlNet-OpenPose-SDXL", "diffusers/controlnet-openpose-sdxl-1.0", default_controlnet_openpose_sdxl_path()),
        ("ControlNet-Union-SDXL", "xinsir/controlnet-union-sdxl-1.0-promax", default_controlnet_union_sdxl_path()),
        ("SAM2-Hiera-Base", "facebook/sam2-hiera-base-plus-hf", default_sam2_hiera_base_path()),
        ("SAM2-Hiera-Small", "facebook/sam2-hiera-small-hf", default_sam2_hiera_small_path()),
        (
            "GroundingDINO-1.5-SwinT",
            "IDEA-Research/grounding-dino-1.5-swint-ogc",
            default_grounding_dino_15_swint_path(),
        ),
        ("Pix2Struct-base", "google/pix2struct-base", default_pix2struct_base_path()),
        ("DePlot", "google/deplot", default_deplot_path()),
        ("TrOCR-base-printed", "microsoft/trocr-base-printed", default_trocr_base_printed_path()),
        ("Watermark-Detector", "prithivMLmods/Watermark-Detection-Model-10K-v1", default_watermark_detector_path()),
        ("vit-gpt2-coco", "nlpconnect/vit-gpt2-image-captioning-finetuned-coco", default_vit_gpt2_coco_path()),
        ("SwinIR-classical-x2", "caidas/swin2SR-classical-sr-x2-64", default_swinir_classical_path()),
        ("GFPGAN", "TencentARC/GFPGAN", default_gfpgan_path()),
        ("UMT5-XXL", "google/umt5-xxl", default_umt5_xxl_path()),
        ("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", default_qwen25_3b_path()),
        ("MiniCPM-V-2_6", "openbmb/MiniCPM-V-2_6", default_minicpm_v26_path()),
        ("NSFW-Alt-Detector", "AdamCodd/vit-base-nsfw-detector", default_nsfw_alt_detector_path()),
        ("CLIP-IQA", "djghosh/aesthetics-scorer", default_clip_iqa_path()),
    ]
    catalog: List[Dict[str, str]] = []
    for name, hf_fallback, resolved_path in catalog_rows:
        catalog.append(
            {
                "name": name,
                "local_folder": str(model_dir() / name),
                "hf_fallback": hf_fallback,
                "resolved": str(resolved_path),
            }
        )
    return catalog


def verify_gen_searcher_8b_local(path: str) -> Dict[str, object]:
    """
    Verify required local files for Gen-Searcher-8B sharded checkpoint.

    Returns a small report with:
      - is_local_dir
      - all_required_present
      - missing (list[str])
      - found_shards (list[str])
    """
    base_path = Path(path)
    required: List[str] = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ]
    if not base_path.is_dir():
        return {
            "is_local_dir": False,
            "all_required_present": False,
            "missing": list(required),
            "found_shards": [],
        }
    missing = [filename for filename in required if not (base_path / filename).is_file()]
    found_shards = sorted(
        [file_path.name for file_path in base_path.glob("model-*-of-*.safetensors") if file_path.is_file()]
    )
    return {
        "is_local_dir": True,
        "all_required_present": len(missing) == 0,
        "missing": missing,
        "found_shards": found_shards,
    }
