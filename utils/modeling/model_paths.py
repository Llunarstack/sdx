"""Resolve local `pretrained/` paths vs Hugging Face hub IDs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


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
    return resolve_model_path("Depth-Anything-V2-Large", "depth-anything/Depth-Anything-V2-Large")


def default_sam2_hiera_large_path() -> str:
    return resolve_model_path("SAM2-Hiera-Large", "facebook/sam2-hiera-large-hf")


def default_realesrgan_path() -> str:
    return resolve_model_path("Real-ESRGAN", "ai-forever/Real-ESRGAN")


def pretrained_catalog() -> List[Dict[str, str]]:
    """
    Canonical pretrained model map used by SDX.

    Returns list rows with:
      - name
      - local_folder
      - hf_fallback
      - resolved
    """
    rows = [
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
    ]
    out: List[Dict[str, str]] = []
    for name, hf, resolved in rows:
        out.append(
            {
                "name": name,
                "local_folder": str(model_dir() / name),
                "hf_fallback": hf,
                "resolved": str(resolved),
            }
        )
    return out


def verify_gen_searcher_8b_local(path: str) -> Dict[str, object]:
    """
    Verify required local files for Gen-Searcher-8B sharded checkpoint.

    Returns a small report with:
      - is_local_dir
      - all_required_present
      - missing (list[str])
      - found_shards (list[str])
    """
    p = Path(path)
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
    if not p.is_dir():
        return {
            "is_local_dir": False,
            "all_required_present": False,
            "missing": list(required),
            "found_shards": [],
        }
    missing = [n for n in required if not (p / n).is_file()]
    found_shards = sorted([x.name for x in p.glob("model-*-of-*.safetensors") if x.is_file()])
    return {
        "is_local_dir": True,
        "all_required_present": len(missing) == 0,
        "missing": missing,
        "found_shards": found_shards,
    }
