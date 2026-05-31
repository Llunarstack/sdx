# Local model stack (`pretrained/`)

When folders exist under `pretrained/`, paths resolve automatically (see `utils/modeling/model_paths.py`). For the full runtime picture (repo folders, `ViT/` vs DiT, diagrams), see the **[README § Architecture and pipeline](../README.md#architecture-and-pipeline)** and **[FILES.md](FILES.md)**.

| Role | Folder / default |
|------|------------------|
| T5-XXL | `pretrained/T5-XXL` or `google/t5-v1_1-xxl` |
| CLIP ViT-L/14 | `pretrained/CLIP-ViT-L-14` or `openai/clip-vit-large-patch14` |
| CLIP ViT-bigG/14 | `pretrained/CLIP-ViT-bigG-14` or LAION hub id |
| DINOv2 (REPA / ViT) | `pretrained/DINOv2-Large` or `facebook/dinov2-large` |
| SigLIP | `pretrained/SigLIP-SO400M` |
| Qwen LLM | `pretrained/Qwen2.5-14B-Instruct` |
| Stable Cascade | `pretrained/StableCascade-Prior`, `pretrained/StableCascade-Decoder` |
| LongCLIP-L (long prompts) | `pretrained/LongCLIP-L` or `creative-graphic-design/LongCLIP-L` |
| moondream2 (caption/VQA helper) | `pretrained/moondream2` or `vikhyatoolkit/moondream2` |
| Marigold depth/normals | `pretrained/Marigold-Depth-v1-1`, `pretrained/Marigold-Normals-v1-1` |
| TAESD / TAESDXL (fast preview VAE) | `pretrained/TAESD`, `pretrained/TAESDXL` |
| Consistency Decoder | `pretrained/Consistency-Decoder` or `openai/consistency-decoder` |
| ConvNeXtV2-Large | `pretrained/ConvNeXtV2-Large` |
| LAION Aesthetic v2 | `pretrained/LAION-Aesthetic-v2` |
| CodeFormer (face restore helper) | `pretrained/CodeFormer` |
| AnyDoor reference weights | `pretrained/AnyDoor-Ref` |
| HPSv2 (preference scorer) | `pretrained/HPSv2-hf` or `adams-story/HPSv2-hf` |
| BLIP captioning (base) | `pretrained/BLIP-image-captioning-base` or `Salesforce/blip-image-captioning-base` |
| Kosmos-2 VLM | `pretrained/Kosmos-2-patch14-224` or `microsoft/kosmos-2-patch14-224` |
| CRAFT text detector | `pretrained/CRAFT-text-detector` or `boomb0om/CRAFT-text-detector` |
| OwlViT (open-vocab detector) | `pretrained/OwlViT-base-patch32` or `google/owlvit-base-patch32` |
| Donut (doc/layout understanding) | `pretrained/Donut-base` or `naver-clova-ix/donut-base` |
| Florence-2 (caption / vision tasks) | `pretrained/Florence-2-base` or `microsoft/Florence-2-base` |
| BLIP-2 (strong captions) | `pretrained/BLIP2-opt-2.7b` or `Salesforce/blip2-opt-2.7b` |
| Qwen2-VL-2B (compact VLM) | `pretrained/Qwen2-VL-2B-Instruct` or `Qwen/Qwen2-VL-2B-Instruct` |
| Depth-Anything-V2-Small | `pretrained/Depth-Anything-V2-Small` or hub id |
| CLIP ViT-H/14 | `pretrained/CLIP-ViT-H-14` or LAION hub id |
| OWLv2 detector | `pretrained/OWLv2-base-patch16-ensemble` |
| ControlNet Canny | `pretrained/ControlNet-Canny` or `lllyasviel/sd-controlnet-canny` |
| ControlNet Depth / OpenPose / Lineart / Scribble | `pretrained/ControlNet-*` or lllyasviel hub ids |
| SmolVLM-256M (tiny VLM) | `pretrained/SmolVLM-256M-Instruct` |
| Florence-2-large | `pretrained/Florence-2-large` |
| OneAlign (IQA/reward) | `pretrained/OneAlign` or `Q-Future/OneAlign` |
| MetaCLIP / AIMv2 / AltCLIP | vision encoder alternatives for alignment experiments |
| CAFE Aesthetic | `pretrained/CAFE-Aesthetic` |
| GroundingDINO-Tiny | lightweight open-vocab detector scaffold |
| SmolVLM2-2B / Qwen2-VL-7B / Phi-3.5-vision | additional VLM scaffolds |
| GIT-base/large-coco | caption models |
| ControlNet MLSD / SoftEdge / Seg / Normal | extended control scaffolds |
| GroundingDINO-SwinT / OWLv2-Large | stronger detectors |
| MUSIQ / PerceptCLIP / ImageReward | extended HF reward panel |
| EVA02-CLIP-L-14 / CLIP-ViT-L-336 | CLIP alignment variants |
| GOT-OCR2 / TrOCR-small | document OCR scaffolds |
| SAM-ViT-Huge | segmentation scaffold |
| T5-XL / T5-Large / CLIP-ViT-B-32 | lighter text encoder scaffolds |
| SigLIP2 / DINOv2-Base/Small | lighter vision encoder scaffolds |
| LLaVA-1.5-7B / InternVL2-2B / Qwen2.5-VL-3B | additional VLMs |
| ControlNet HED / SDXL canny+depth | extended control scaffolds |
| ZoeDepth / Metric3D / Marigold depth | depth estimation alternatives |
| DETR / Mask2Former / SAM2-Tiny | detection & segmentation |
| LayoutLMv3 / Donut-docvqa | document layout OCR |
| NSFW-Detector | safety gate scaffold |
| sd-vae-ft-mse / sdxl-vae-fp16-fix | VAE decode scaffolds |
| LLaVA-v1.6 / Qwen2.5-VL-7B / PaliGemma2 / InternVL2-1B | latest VLM wave |
| DINOv3 / MobileCLIP-S2 / UMT5-XXL | next-gen encoders |
| ControlNet Union/OpenPose SDXL | SDXL control scaffolds |
| SAM2 Base/Small + GroundingDINO-1.5 | mid-tier seg/detect |
| Pix2Struct / DePlot / vit-gpt2-coco | doc/chart/caption helpers |
| Watermark-Detector / CLIP-IQA | QA + reward scaffolds |
| GFPGAN / SwinIR-classical | face restore + upscale scaffolds |

Registry index: ``utils/modeling/hf_index.py`` (`summary()`, `role_counts()`). Upscale helpers: ``utils/modeling/hf_upscale.py``.

## Config-only scaffold (no checkpoint download)

Fetch folder structure + configs/tokenizers without weight files:

```bash
python scripts/download/download_hf_scaffold.py --all
python scripts/download/download_hf_scaffold.py --role vlm --role reward
python scripts/download/download_hf_scaffold.py --list
```

Runtime uses `resolve_model_path_require_weights()` so config-only local folders do **not** block Hugging Face fallback. Lazy loaders: `utils/modeling/hf_loaders.py`, unified reward panel: `utils/modeling/hf_reward.py`.

## How this maps to the SDX pipeline

- **DiT generation** uses **text** encoders (T5, optionally + CLIP fusion) and **image latents** from VAE/RAE — see the main **[README § Architecture and pipeline](../README.md#architecture-and-pipeline)** for full diagrams (DiT vs `ViT/` QA tools vs REPA).
- **`ViT/`** in this repo is a **separate** scoring/ranking stack (timm ViT on manifests), not the same module as the DiT generator.
- **REPA** uses a **frozen vision** model (often DINOv2) to align DiT **internal** features during training.

## Training

- **T5 only (default):** `python train.py ...` — same as before.
- **Triple text encoders:** T5 + CLIP-L + CLIP-bigG with a small trainable fusion:
  ```bash
  python train.py --data-path ... --text-encoder-mode triple
  ```
- **Penta text encoders:** T5 + CLIP-L + CLIP-bigG + CLIP-H + LongCLIP-L (4 fused CLIP tokens):
  ```bash
  python train.py --data-path ... --text-encoder-mode penta
  ```
  Download weights (or config scaffolds):
  ```bash
  python scripts/download/download_models.py --penta-text-encoders
  python scripts/download/download_hf_scaffold.py --penta
  python -m scripts.tools.ops.pretrained_status --text-encoder-mode penta
  ```
JSONL fields (optional, for triple/penta training):

- ``prompt_layout``: inline layout object (same schema as ``examples/prompt_layout.example.json``)
- ``prompt_layout_path``: path to layout JSON (relative to manifest / image root)

When set, CLIP encoders receive labeled section captions; LongCLIP (penta) receives the full training caption.

## Sampling

`sample.py` reads `text_encoder_mode` from the checkpoint config and loads the same stack.

## Other

- **Qwen prompt expansion:** `utils/analysis/llm_client.py` (`load_qwen_causal_lm`, `expand_prompt_qwen`).
- **Stable Cascade (Diffusers):** `python scripts/cascade_generate.py --prompt "..."` — separate from DiT; does not share the DiT forward pass.
- **Advanced optional downloads:** `python scripts/download/download_models.py --advanced` to fetch LongCLIP/moondream2/Marigold/TAESD/CodeFormer/Consistency-Decoder/ConvNeXtV2/LAION Aesthetic v2/AnyDoor reference.

---

## Model enhancements (shared blocks & fusion)

Small, composable pieces under [`models/model_enhancements.py`](../models/model_enhancements.py) and optional flags on multimodal / cascade / RAE modules.

### `model_enhancements.py`

| Class / helper | Role |
|----------------|------|
| **`RMSNorm`** | Root mean square norm (no mean centering); lighter than LayerNorm. |
| **`DropPath`** | Stochastic depth on a residual branch (training). |
| **`TokenFiLM`** | Global cond vector → per-channel scale/shift on `(B, N, D)` tokens. |
| **`SE1x1`** | Squeeze–excitation gating on `(B, D)` or `(B, N, D)`. |
| **`apply_gate_residual`** | `x + gate * branch` with broadcast gate. |

Import: `from models import RMSNorm, DropPath, TokenFiLM, SE1x1` (also `models.model_enhancements`).

### `NativeMultimodalTransformer`

| Option | Default | Effect |
|--------|---------|--------|
| **`cross_attn_heads`** | `0` | If &gt; 0, vision tokens **cross-attend** to text (+ extra) **before** joint `TransformerEncoder`. |
| **`output_norm`** | `"layernorm"` | `"rmsnorm"` uses **`RMSNorm`** after the encoder stack. |
| **`film_cond_dim`** | `0` | If &gt; 0, **`TokenFiLM`** on **vision** tokens; forward requires **`film_cond`** `(B, dim)`. |

### `CascadedMultimodalDiffusion`

| Option | Default | Effect |
|--------|---------|--------|
| **`blend_base_refine`** | `1.0` | `final = (1-w)*base_out + w*refine_out` (ablation / stability). `1.0` matches previous “refine only” behavior. |
| **`detach_base_for_refine`** (forward kw) | `False` | Refine stage sees **detached** `base_out` (train refine with frozen base). |

### `RAELatentBridge`

| Option | Default | Effect |
|--------|---------|--------|
| **`learnable_output_scale`** | `False` | Scalar **`scale_dit`** / **`scale_rae`** on conv outputs (training stability). |

See also [`models/native_multimodal_transformer.py`](../models/native_multimodal_transformer.py), [`models/cascaded_multimodal_diffusion.py`](../models/cascaded_multimodal_diffusion.py), [CODEBASE.md](CODEBASE.md).
