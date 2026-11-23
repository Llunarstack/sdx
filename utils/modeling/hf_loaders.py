"""Lazy Hugging Face model loaders (captioners, OCR, depth, detectors, reward scorers), the reward panel and ControlNet control-map helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from PIL import Image, ImageFilter

from utils.modeling.model_paths import (
    default_blip2_flan_t5_xl_path,
    default_blip2_opt_path,
    default_blip_caption_base_path,
    default_blip_caption_large_path,
    default_cafe_aesthetic_path,
    default_clip_h14_path,
    default_clip_iqa_path,
    default_controlnet_canny_path,
    default_controlnet_canny_sdxl_path,
    default_controlnet_depth_path,
    default_controlnet_depth_sdxl_path,
    default_controlnet_hed_path,
    default_controlnet_lineart_path,
    default_controlnet_mlsd_path,
    default_controlnet_normal_path,
    default_controlnet_openpose_path,
    default_controlnet_openpose_sdxl_path,
    default_controlnet_scribble_path,
    default_controlnet_seg_path,
    default_controlnet_softedge_path,
    default_controlnet_union_sdxl_path,
    default_depth_anything_v2_base_path,
    default_depth_anything_v2_large_path,
    default_depth_anything_v2_small_path,
    default_dpt_large_path,
    default_florence2_base_path,
    default_florence2_large_path,
    default_git_base_coco_path,
    default_git_large_coco_path,
    default_got_ocr2_path,
    default_grounding_dino_swint_path,
    default_hpsv2_path,
    default_image_reward_path,
    default_internvl2_2b_path,
    default_kosmos2_path,
    default_llava_15_7b_path,
    default_llava_v16_mistral_7b_path,
    default_marigold_depth_path,
    default_marigold_normals_path,
    default_moondream2_path,
    default_nsfw_detector_path,
    default_onealign_path,
    default_owlv2_base_path,
    default_owlvit_base_path,
    default_paligemma2_3b_path,
    default_perceptclip_iqa_path,
    default_pickscore_path,
    default_qwen2_vl_2b_path,
    default_qwen2_vl_7b_path,
    default_qwen25_vl_3b_path,
    default_qwen25_vl_7b_path,
    default_siglip2_so400m_path,
    default_smolvlm2_2b_path,
    default_smolvlm_256m_path,
    default_trocr_base_handwritten_path,
    default_trocr_large_printed_path,
    default_vit_gpt2_caption_path,
    default_vit_gpt2_coco_path,
    default_watermark_detector_path,
    default_zoedepth_path,
    has_local_weights,
)


def _image_text_to_text_cls() -> Any:
    """VLM auto-class: ``AutoModelForImageTextToText`` (transformers >= 4.46); v5 removed ``AutoModelForVision2Seq``."""
    import transformers

    return getattr(transformers, "AutoModelForImageTextToText", None) or transformers.AutoModelForVision2Seq


def _has_weights_at(path: str) -> bool:
    return has_local_weights(path)


@dataclass
class _Cache:
    store: dict[str, Any]

    def get_or_load(self, key: str, loader: Callable[[], Any]) -> Any:
        if key not in self.store:
            self.store[key] = loader()
        return self.store[key]


_CACHE = _Cache(store={})
_LOG = logging.getLogger(__name__)
_T = TypeVar("_T")


def _safe_infer(label: str, fn: Callable[[], _T], default: _T) -> _T:
    """Run optional HF inference; log failures instead of silent empty returns."""
    try:
        return fn()
    except Exception as exc:
        _LOG.warning("%s failed: %s: %s", label, type(exc).__name__, exc)
        return default


def caption_blip(path: str, *, device: str = "cuda", variant: str = "base") -> str:
    """BLIP image caption (base or large)."""
    if variant == "large":
        model_id = default_blip_caption_large_path()
    else:
        model_id = default_blip_caption_base_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        proc = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        return proc, model

    def _run() -> str:
        proc, model = _CACHE.get_or_load(f"blip_{variant}", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = proc(img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80)
        return proc.decode(out[0], skip_special_tokens=True).strip()

    return _safe_infer(f"caption_blip/{variant}", _run, "")


def caption_blip2(path: str, *, device: str = "cuda") -> str:
    model_id = default_blip2_opt_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        proc = Blip2Processor.from_pretrained(model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        return proc, model

    def _run() -> str:
        proc, model = _CACHE.get_or_load("blip2", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = proc(img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80)
        return proc.decode(out[0], skip_special_tokens=True).strip()

    return _safe_infer("caption_blip2", _run, "")


def caption_florence2(
    path: str,
    *,
    device: str = "cuda",
    task: str = "<CAPTION>",
    model_key: str = "base",
) -> str:
    model_id = default_florence2_large_path() if model_key == "large" else default_florence2_base_path()
    cache_key = f"florence2_{model_key}"
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        return proc, model

    def _run() -> str:
        proc, model = _CACHE.get_or_load(cache_key, _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = proc(text=task, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100)
        text = proc.batch_decode(out, skip_special_tokens=False)[0]
        parsed = proc.post_process_generation(text, task=task, image_size=img.size)
        if isinstance(parsed, dict):
            return str(parsed.get(task, parsed)).strip()
        return str(parsed).strip()

    return _safe_infer(f"caption_florence2/{model_key}", _run, "")


def caption_kosmos2(path: str, *, device: str = "cuda") -> str:
    model_id = default_kosmos2_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id)
        model = _image_text_to_text_cls().from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("kosmos2", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = proc(images=img, text="<grounding>", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64)
        return proc.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return ""


def ocr_trocr(path: str, *, device: str = "cuda") -> str:
    model_id = default_trocr_large_printed_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        proc = TrOCRProcessor.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("trocr", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = proc(img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128)
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""


def depth_map(path: str, output_path: str, *, device: str = "cuda", prefer: str = "small") -> str:
    """Write a grayscale depth PNG; returns output path or empty string."""
    if prefer == "dpt":
        model_id = default_dpt_large_path()
    elif prefer == "zoe":
        model_id = default_zoedepth_path()
    elif prefer == "large":
        model_id = default_depth_anything_v2_large_path()
    elif prefer == "base":
        model_id = default_depth_anything_v2_base_path()
    else:
        model_id = default_depth_anything_v2_small_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> Any:
        from transformers import pipeline

        return pipeline("depth-estimation", model=model_id, device=0 if device.startswith("cuda") else -1)

    try:
        pipe = _CACHE.get_or_load(f"depth_{prefer}", _load)
        from PIL import Image

        result = pipe(Image.open(path).convert("RGB"))
        depth = result["depth"]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        depth.save(output_path)
        return output_path
    except Exception:
        return ""


def score_hpsv2(rgb_uint8: np.ndarray, prompt: str, *, device: str = "cuda") -> float | None:
    """Human preference score in [0, 1] or None if model unavailable."""
    model_id = default_hpsv2_path()
    if not _has_weights_at(model_id):
        return None

    def _load() -> tuple[Any, Any]:
        from transformers import AutoModel, AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("hpsv2", _load)
        import torch
        from PIL import Image

        pil = Image.fromarray(rgb_uint8)
        inputs = proc(text=[prompt], images=[pil], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits_per_image
        score = float(logits.item())
        return float(torch.sigmoid(torch.tensor((score - 25.0) / 5.0)).item())
    except Exception:
        return None


def score_pickscore(rgb_uint8: np.ndarray, prompt: str, *, device: str = "cuda") -> float | None:
    model_id = default_pickscore_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return None

    def _load() -> tuple[Any, Any]:
        from transformers import AutoModel, AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("pickscore", _load)
        import torch
        from PIL import Image

        pil = Image.fromarray(rgb_uint8)
        inputs = proc(text=[prompt], images=[pil], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits_per_image
        score = float(logits.item())
        return float(torch.sigmoid(torch.tensor((score - 25.0) / 5.0)).item())
    except Exception:
        return None


def score_clip_h14(rgb_uint8: np.ndarray, prompt: str, *, device: str = "cuda") -> float | None:
    model_id = default_clip_h14_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return None

    def _load() -> tuple[Any, Any]:
        from transformers import CLIPModel, CLIPProcessor

        proc = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("clip_h14", _load)
        import torch
        from PIL import Image

        pil = Image.fromarray(rgb_uint8)
        inputs = proc(text=[prompt], images=[pil], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits_per_image
        score = float(logits.item())
        return float(torch.sigmoid(torch.tensor(score / 10.0)).item())
    except Exception:
        return None


def score_onealign(rgb_uint8: np.ndarray, *, prompt: str = "", device: str = "cuda") -> float | None:
    model_id = default_onealign_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return None

    def _load() -> Any:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

    try:
        model = _CACHE.get_or_load("onealign", _load)
        import torch
        from PIL import Image

        pil = Image.fromarray(rgb_uint8)
        with torch.no_grad():
            if hasattr(model, "score"):
                raw = model.score(pil, prompt or "quality")
                return float(torch.sigmoid(torch.tensor(float(raw))).item())
            if hasattr(model, "predict"):
                raw = model.predict(pil)
                return float(np.clip(float(raw), 0.0, 1.0))
        return None
    except Exception:
        return None


def score_cafe_aesthetic(rgb_uint8: np.ndarray, *, device: str = "cuda") -> float | None:
    model_id = default_cafe_aesthetic_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return None

    def _load() -> tuple[Any, Any]:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("cafe_aesthetic", _load)
        import torch
        from PIL import Image

        pil = Image.fromarray(rgb_uint8)
        inputs = proc(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # Higher class index often maps to higher aesthetic in binary/multi-class heads.
        score = float(probs[0, -1].item()) if probs.shape[-1] > 1 else float(probs[0, 0].item())
        return score
    except Exception:
        return None


def score_perceptclip(rgb_uint8: np.ndarray, *, prompt: str = "", device: str = "cuda") -> float | None:
    model_id = default_perceptclip_iqa_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return None

    def _load() -> tuple[Any, Any]:
        from transformers import AutoModel, AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("perceptclip", _load)
        import torch
        from PIL import Image

        pil = Image.fromarray(rgb_uint8)
        if prompt.strip():
            inputs = proc(text=[prompt], images=[pil], return_tensors="pt", padding=True)
        else:
            inputs = proc(images=[pil], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        logits = getattr(out, "logits_per_image", None) or getattr(out, "logits", None)
        if logits is None:
            return None
        score = float(logits.squeeze().item())
        return float(torch.sigmoid(torch.tensor(score / 10.0)).item())
    except Exception:
        return None


def score_image_reward(rgb_uint8: np.ndarray, prompt: str, *, device: str = "cuda") -> float | None:
    ir_path = default_image_reward_path()
    local = Path(ir_path)
    if local.is_dir() and not _has_weights_at(local):
        return None

    def _load() -> Any:
        import ImageReward as RM

        path_s = str(local) if local.is_dir() else ir_path
        return RM.load(path_s, device=device)

    try:
        model = _CACHE.get_or_load("image_reward", _load)
        import torch
        from PIL import Image

        pil = Image.fromarray(rgb_uint8)
        raw = float(model.score(prompt, pil))
        return float(torch.sigmoid(torch.tensor(raw / 2.0)).item())
    except Exception:
        return None


def caption_git(path: str, *, device: str = "cuda", variant: str = "base") -> str:
    model_id = default_git_large_coco_path() if variant == "large" else default_git_base_coco_path()
    cache_key = f"git_{variant}"
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load(cache_key, _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = proc(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(pixel_values=inputs.pixel_values, max_new_tokens=64)
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""


def caption_smolvlm2(path: str, *, user_prompt: str = "", device: str = "cuda") -> str:
    model_id = default_smolvlm2_2b_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id)
        model = _image_text_to_text_cls().from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("smolvlm2", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        prompt = user_prompt.strip() or "Describe this image."
        inputs = proc(text=prompt, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120)
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""


def detect_grounding_dino(path: str, queries: list[str], *, device: str = "cuda", threshold: float = 0.25) -> list[str]:
    if not queries:
        return []
    model_id = default_grounding_dino_swint_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return []

    def _load() -> tuple[Any, Any]:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("grounding_dino", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        text = ". ".join(queries) + "."
        inputs = proc(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=threshold, target_sizes=[img.size[::-1]]
        )
        if not results:
            return []
        labels = results[0].get("labels", [])
        return [str(x) for x in labels if str(x)]
    except Exception:
        return []


def detect_objects(path: str, queries: list[str], *, device: str = "cuda") -> list[str]:
    hits = detect_grounding_dino(path, queries, device=device)
    if hits:
        return hits
    return detect_owlv2(path, queries, device=device)


def ocr_got(path: str, *, device: str = "cuda") -> str:
    model_id = default_got_ocr2_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        return tok, model

    try:
        tok, model = _CACHE.get_or_load("got_ocr2", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = tok(img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256)
        return tok.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return ""


def marigold_depth_map(path: str, output_path: str, *, device: str = "cuda") -> str:
    model_id = default_marigold_depth_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> Any:
        from diffusers import MarigoldDepthPipeline

        return MarigoldDepthPipeline.from_pretrained(model_id).to(device)

    try:
        pipe = _CACHE.get_or_load("marigold_depth", _load)
        from PIL import Image

        result = pipe(Image.open(path).convert("RGB"), num_inference_steps=4)
        depth = result.prediction[0] if hasattr(result, "prediction") else result.images[0]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        depth.save(output_path)
        return output_path
    except Exception:
        return ""


def score_siglip2(rgb_uint8: np.ndarray, prompt: str, *, device: str = "cuda") -> float | None:
    model_id = default_siglip2_so400m_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return None

    def _load() -> tuple[Any, Any]:
        from transformers import AutoModel, AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("siglip2", _load)
        import torch
        from PIL import Image

        pil = Image.fromarray(rgb_uint8)
        inputs = proc(text=[prompt], images=[pil], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits_per_image
        score = float(logits.item())
        return float(torch.sigmoid(torch.tensor(score / 10.0)).item())
    except Exception:
        return None


def score_nsfw_probability(rgb_uint8: np.ndarray, *, device: str = "cuda") -> float | None:
    """Return NSFW probability in [0, 1] or None."""
    model_id = default_nsfw_detector_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return None

    def _load() -> tuple[Any, Any]:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("nsfw", _load)
        import torch
        from PIL import Image

        pil = Image.fromarray(rgb_uint8)
        inputs = proc(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # Label 1 is typically nsfw for Falconsai model.
        idx = 1 if probs.shape[-1] > 1 else 0
        return float(probs[0, idx].item())
    except Exception:
        return None


def caption_llava(path: str, *, user_prompt: str = "", device: str = "cuda") -> str:
    model_id = default_llava_15_7b_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        proc = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("llava", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        prompt = f"USER: <image>\n{user_prompt.strip() or 'Describe this image.'}\nASSISTANT:"
        inputs = proc(text=prompt, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120)
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""


def caption_internvl2(path: str, *, user_prompt: str = "", device: str = "cuda") -> str:
    model_id = default_internvl2_2b_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        return tok, model

    try:
        tok, model = _CACHE.get_or_load("internvl2", _load)
        from PIL import Image

        img = Image.open(path).convert("RGB")
        prompt = user_prompt.strip() or "Describe this image in detail."
        if hasattr(model, "chat"):
            return str(model.chat(tok, img, prompt, generation_config={"max_new_tokens": 120})).strip()
        return ""
    except Exception:
        return ""


def caption_blip2_flan(path: str, *, device: str = "cuda") -> str:
    model_id = default_blip2_flan_t5_xl_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        proc = Blip2Processor.from_pretrained(model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("blip2_flan", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = proc(img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80)
        return proc.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return ""


def caption_qwen25_vl(path: str, *, user_prompt: str = "", device: str = "cuda") -> str:
    model_id = default_qwen25_vl_3b_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = _image_text_to_text_cls().from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("qwen25_vl", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        prompt = user_prompt.strip() or "Describe this image."
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120)
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""


def score_clip_iqa(rgb_uint8: np.ndarray, *, device: str = "cuda") -> float | None:
    model_id = default_clip_iqa_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return None

    def _load() -> tuple[Any, Any]:
        from transformers import AutoImageProcessor, AutoModel

        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("clip_iqa", _load)
        import torch
        from PIL import Image

        pil = Image.fromarray(rgb_uint8)
        inputs = proc(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        logits = getattr(out, "logits", None) or getattr(out, "pooler_output", None)
        if logits is None:
            return None
        raw = (
            float(logits.squeeze().mean().item())
            if hasattr(logits.squeeze(), "mean")
            else float(logits.squeeze().item())
        )
        return float(np.clip(raw / 10.0, 0.0, 1.0))
    except Exception:
        return None


def score_watermark_probability(rgb_uint8: np.ndarray, *, device: str = "cuda") -> float | None:
    model_id = default_watermark_detector_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return None

    def _load() -> tuple[Any, Any]:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("watermark", _load)
        import torch
        from PIL import Image

        pil = Image.fromarray(rgb_uint8)
        inputs = proc(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        idx = 1 if probs.shape[-1] > 1 else 0
        return float(probs[0, idx].item())
    except Exception:
        return None


def caption_llava_v16(path: str, *, user_prompt: str = "", device: str = "cuda") -> str:
    model_id = default_llava_v16_mistral_7b_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        proc = AutoProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("llava_v16", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        prompt = user_prompt.strip() or "Describe this image."
        inputs = proc(text=prompt, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120)
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    except Exception:
        return caption_llava(path, user_prompt=user_prompt, device=device)


def caption_paligemma2(path: str, *, user_prompt: str = "", device: str = "cuda") -> str:
    model_id = default_paligemma2_3b_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        proc = AutoProcessor.from_pretrained(model_id)
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("paligemma2", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        prompt = user_prompt.strip() or "caption en"
        inputs = proc(text=prompt, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100)
        return proc.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return ""


def caption_vit_gpt2_coco(path: str, *, device: str = "cuda") -> str:
    model_id = default_vit_gpt2_coco_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer, ViTImageProcessor

        proc = ViTImageProcessor.from_pretrained(model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
        return proc, tok, model

    try:
        proc, tok, model = _CACHE.get_or_load("vit_gpt2_coco", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = proc(img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64)
        return tok.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return ""


def caption_qwen25_vl_7b(path: str, *, user_prompt: str = "", device: str = "cuda") -> str:
    model_id = default_qwen25_vl_7b_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = _image_text_to_text_cls().from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("qwen25_vl_7b", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        prompt = user_prompt.strip() or "Describe this image."
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120)
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""


def normals_map(path: str, output_path: str, *, device: str = "cuda") -> str:
    """Marigold surface normals PNG when diffusers + weights available."""
    model_id = default_marigold_normals_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> Any:
        from diffusers import MarigoldNormalsPipeline

        return MarigoldNormalsPipeline.from_pretrained(model_id).to(device)

    try:
        pipe = _CACHE.get_or_load("marigold_normals", _load)
        from PIL import Image

        result = pipe(Image.open(path).convert("RGB"), num_inference_steps=4)
        normal = result.prediction[0] if hasattr(result, "prediction") else result.images[0]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        normal.save(output_path)
        return output_path
    except Exception:
        return ""


def caption_vit_gpt2(path: str, *, device: str = "cuda") -> str:
    model_id = default_vit_gpt2_caption_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer, ViTImageProcessor

        proc = ViTImageProcessor.from_pretrained(model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
        return proc, tok, model

    try:
        proc, tok, model = _CACHE.get_or_load("vit_gpt2", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = proc(img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64)
        return tok.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return ""


def caption_smolvlm(path: str, *, user_prompt: str = "", device: str = "cuda") -> str:
    model_id = default_smolvlm_256m_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id)
        model = _image_text_to_text_cls().from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("smolvlm", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        prompt = user_prompt.strip() or "Describe this image."
        inputs = proc(text=prompt, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100)
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""


def ocr_trocr_handwritten(path: str, *, device: str = "cuda") -> str:
    model_id = default_trocr_base_handwritten_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        proc = TrOCRProcessor.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("trocr_hw", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = proc(img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128)
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""


def detect_owlv2(path: str, queries: list[str], *, device: str = "cuda", threshold: float = 0.12) -> list[str]:
    """Return matched query labels for an image."""
    if not queries:
        return []
    model_id = default_owlv2_base_path()
    if not _has_weights_at(model_id) and "/" not in model_id:
        model_id = default_owlvit_base_path()
        if not _has_weights_at(model_id) and "/" not in model_id:
            return []

    def _load() -> tuple[Any, Any]:
        from transformers import AutoModel, AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load("owlv2", _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        texts = [[q] for q in queries]
        inputs = proc(text=texts, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]], device=device)
        results = proc.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)
        if not results:
            return []
        scores = results[0]["scores"]
        labels = results[0]["labels"]
        hits: list[str] = []
        for idx, score in zip(labels.tolist(), scores.tolist(), strict=True):
            if score >= threshold and 0 <= idx < len(queries):
                label = queries[idx]
                if label not in hits:
                    hits.append(label)
        return hits
    except Exception:
        return []


def caption_image_chain(
    path: str,
    *,
    user_prompt: str = "",
    device: str = "cuda",
    backends: list[str] | None = None,
) -> tuple[str, str]:
    """
    Try caption backends in order. Returns (caption, backend_name).
    """
    order = (
        backends
        if backends is not None
        else [
            "moondream",
            "qwen25_vl_7b",
            "qwen25_vl",
            "paligemma2",
            "llava_v16",
            "internvl2",
            "llava",
            "smolvlm2",
            "smolvlm",
            "git_large",
            "git_base",
            "blip2_flan",
            "blip2",
            "florence2_large",
            "blip_large",
            "florence2",
            "vit_gpt2_coco",
            "vit_gpt2",
            "blip",
            "kosmos2",
            "qwen2_vl_7b",
            "qwen2_vl",
        ]
    )
    for name in order:
        cap = ""
        if name == "moondream":
            md_path = default_moondream2_path()
            if _has_weights_at(md_path) or "/" in md_path:
                try:
                    from utils.prompt.rag import CreativeRAGEngine

                    engine = CreativeRAGEngine()
                    result = engine.enrich(
                        prompt=user_prompt or "Describe this image in detail for image generation.",
                        reference_image_path=path,
                        creativity_level=0.3,
                    )
                    cap = (result.image_description or result.enriched_prompt or "").strip()
                except Exception:
                    cap = ""
        elif name == "blip2":
            cap = caption_blip2(path, device=device)
        elif name == "blip2_flan":
            cap = caption_blip2_flan(path, device=device)
        elif name == "llava":
            cap = caption_llava(path, user_prompt=user_prompt, device=device)
        elif name == "internvl2":
            cap = caption_internvl2(path, user_prompt=user_prompt, device=device)
        elif name == "qwen25_vl":
            cap = caption_qwen25_vl(path, user_prompt=user_prompt, device=device)
        elif name == "qwen25_vl_7b":
            cap = caption_qwen25_vl_7b(path, user_prompt=user_prompt, device=device)
        elif name == "paligemma2":
            cap = caption_paligemma2(path, user_prompt=user_prompt, device=device)
        elif name == "llava_v16":
            cap = caption_llava_v16(path, user_prompt=user_prompt, device=device)
        elif name == "smolvlm":
            cap = caption_smolvlm(path, user_prompt=user_prompt, device=device)
        elif name == "smolvlm2":
            cap = caption_smolvlm2(path, user_prompt=user_prompt, device=device)
        elif name == "git_base":
            cap = caption_git(path, device=device, variant="base")
        elif name == "git_large":
            cap = caption_git(path, device=device, variant="large")
        elif name == "vit_gpt2":
            cap = caption_vit_gpt2(path, device=device)
        elif name == "vit_gpt2_coco":
            cap = caption_vit_gpt2_coco(path, device=device)
        elif name == "florence2_large":
            cap = caption_florence2(path, device=device, model_key="large")
        elif name == "blip_large":
            cap = caption_blip(path, device=device, variant="large")
        elif name == "blip":
            cap = caption_blip(path, device=device, variant="base")
        elif name == "florence2":
            cap = caption_florence2(path, device=device)
        elif name == "kosmos2":
            cap = caption_kosmos2(path, device=device)
        elif name == "qwen2_vl":
            cap = _caption_qwen2_vl(path, user_prompt=user_prompt, device=device, size="2b")
        elif name == "qwen2_vl_7b":
            cap = _caption_qwen2_vl(path, user_prompt=user_prompt, device=device, size="7b")
        if cap:
            return cap, name
    return "", ""


def _caption_qwen2_vl(path: str, *, user_prompt: str, device: str, size: str = "2b") -> str:
    model_id = default_qwen2_vl_7b_path() if size == "7b" else default_qwen2_vl_2b_path()
    cache_key = f"qwen2_vl_{size}"
    if not _has_weights_at(model_id) and "/" not in model_id:
        return ""

    def _load() -> tuple[Any, Any]:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = _image_text_to_text_cls().from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        return proc, model

    try:
        proc, model = _CACHE.get_or_load(cache_key, _load)
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        prompt = user_prompt.strip() or "Describe this image."
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120)
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""


ControlPathFn = Callable[[], str]

CONTROLNET_REGISTRY: dict[str, ControlPathFn] = {
    "canny": default_controlnet_canny_path,
    "depth": default_controlnet_depth_path,
    "openpose": default_controlnet_openpose_path,
    "lineart": default_controlnet_lineart_path,
    "scribble": default_controlnet_scribble_path,
    "mlsd": default_controlnet_mlsd_path,
    "softedge": default_controlnet_softedge_path,
    "seg": default_controlnet_seg_path,
    "normal": default_controlnet_normal_path,
    "hed": default_controlnet_hed_path,
    "canny_sdxl": default_controlnet_canny_sdxl_path,
    "depth_sdxl": default_controlnet_depth_sdxl_path,
    "openpose_sdxl": default_controlnet_openpose_sdxl_path,
    "union_sdxl": default_controlnet_union_sdxl_path,
}

# Maps that can be synthesized without neural weights.
PIL_PROXY_TYPES: frozenset[str] = frozenset({"canny", "softedge", "hed"})


def list_controlnet_types() -> list[str]:
    return sorted(CONTROLNET_REGISTRY.keys())


def resolve_controlnet_path(control_type: str) -> str:
    key = str(control_type).strip().lower()
    fn = CONTROLNET_REGISTRY.get(key)
    if fn is None:
        raise KeyError(f"Unknown control type: {control_type}")
    return fn()


def controlnet_has_weights(control_type: str) -> bool:
    try:
        path = resolve_controlnet_path(control_type)
        return has_local_weights(path)
    except KeyError:
        return False


def extract_pil_proxy(path: str, output_path: str, control_type: str) -> str:
    """Write a cheap proxy control map (canny or softedge)."""
    kind = str(control_type).strip().lower()
    if kind not in PIL_PROXY_TYPES:
        return ""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return ""
    if kind == "canny":
        out = img.convert("L").filter(ImageFilter.FIND_EDGES)
    elif kind == "hed":
        out = img.convert("L").filter(ImageFilter.EMBOSS).filter(ImageFilter.FIND_EDGES)
    else:
        out = img.convert("L").filter(ImageFilter.GaussianBlur(radius=1)).filter(ImageFilter.FIND_EDGES)
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest)
    return str(dest)


def extract_control_maps_batch(
    path: str,
    output_dir: Path,
    *,
    types: Sequence[str] = ("canny", "softedge", "depth"),
    device: str = "cuda",
) -> dict[str, str]:
    """Merge PIL proxies + HF depth/normal loaders."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    maps: dict[str, str] = {}

    for kind in types:
        key = str(kind).strip().lower()
        if key in PIL_PROXY_TYPES:
            dest = out_dir / f"{stem}_{key}.png"
            out = extract_pil_proxy(path, str(dest), key)
            if out:
                maps[key] = out
            continue
        if key == "depth":
            try:
                dest = out_dir / f"{stem}_depth.png"
                for prefer in ("small", "base", "large", "dpt", "zoe"):
                    out = depth_map(str(path), str(dest), device=device, prefer=prefer)
                    if out:
                        maps["depth"] = out
                        break
                if "depth" not in maps:
                    out = marigold_depth_map(str(path), str(dest), device=device)
                    if out:
                        maps["depth"] = out
            except Exception:
                pass
        elif key == "normals":
            try:
                dest = out_dir / f"{stem}_normals.png"
                out = normals_map(str(path), str(dest), device=device)
                if out:
                    maps["normals"] = out
            except Exception:
                pass

    return maps


@dataclass(slots=True)
class HFRewardWeights:
    hpsv2: float = 0.16
    pickscore: float = 0.16
    clip_h14: float = 0.12
    onealign: float = 0.12
    cafe_aesthetic: float = 0.12
    perceptclip: float = 0.12
    image_reward: float = 0.10
    siglip2: float = 0.07
    clip_iqa: float = 0.07


@dataclass(slots=True)
class HFRewardBreakdown:
    composite: float
    parts: dict[str, float] = field(default_factory=dict)


class HFRewardPanel:
    """Score RGB uint8 images with multiple HF reward backends."""

    def __init__(
        self,
        weights: HFRewardWeights | None = None,
        *,
        device: str = "cuda",
    ) -> None:
        self.weights = weights or HFRewardWeights()
        self.device = device

    def score_breakdown(self, rgb_uint8: np.ndarray, *, prompt: str = "") -> HFRewardBreakdown:

        parts: dict[str, float] = {}
        w = self.weights

        if w.hpsv2 > 0 and prompt.strip():
            v = score_hpsv2(rgb_uint8, prompt, device=self.device)
            if v is not None:
                parts["hpsv2"] = float(v)

        if w.pickscore > 0 and prompt.strip():
            v = score_pickscore(rgb_uint8, prompt, device=self.device)
            if v is not None:
                parts["pickscore"] = float(v)

        if w.clip_h14 > 0 and prompt.strip():
            v = score_clip_h14(rgb_uint8, prompt, device=self.device)
            if v is not None:
                parts["clip_h14"] = float(v)

        if w.onealign > 0:
            v = score_onealign(rgb_uint8, prompt=prompt, device=self.device)
            if v is not None:
                parts["onealign"] = float(v)

        if w.cafe_aesthetic > 0:
            v = score_cafe_aesthetic(rgb_uint8, device=self.device)
            if v is not None:
                parts["cafe_aesthetic"] = float(v)

        if w.perceptclip > 0:
            v = score_perceptclip(rgb_uint8, prompt=prompt, device=self.device)
            if v is not None:
                parts["perceptclip"] = float(v)

        if w.image_reward > 0 and prompt.strip():
            v = score_image_reward(rgb_uint8, prompt, device=self.device)
            if v is not None:
                parts["image_reward"] = float(v)

        if w.siglip2 > 0 and prompt.strip():
            v = score_siglip2(rgb_uint8, prompt, device=self.device)
            if v is not None:
                parts["siglip2"] = float(v)

        if w.clip_iqa > 0:
            v = score_clip_iqa(rgb_uint8, device=self.device)
            if v is not None:
                parts["clip_iqa"] = float(v)

        if not parts:
            return HFRewardBreakdown(composite=0.5, parts={})

        weight_map = {
            "hpsv2": w.hpsv2,
            "pickscore": w.pickscore,
            "clip_h14": w.clip_h14,
            "onealign": w.onealign,
            "cafe_aesthetic": w.cafe_aesthetic,
            "perceptclip": w.perceptclip,
            "image_reward": w.image_reward,
            "siglip2": w.siglip2,
            "clip_iqa": w.clip_iqa,
        }
        tw = sum(weight_map[k] for k in parts) or 1.0
        composite = float(sum(weight_map[k] * parts[k] for k in parts) / tw)
        return HFRewardBreakdown(composite=composite, parts=parts)

    def score(self, rgb_uint8: np.ndarray, *, prompt: str = "") -> float:
        return self.score_breakdown(rgb_uint8, prompt=prompt).composite

    def rank_indices(
        self,
        images: list[np.ndarray],
        *,
        prompt: str = "",
    ) -> list[int]:
        scores = [self.score(im, prompt=prompt) for im in images]
        return sorted(range(len(scores)), key=lambda i: -scores[i])
