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
  Checkpoints store `text_encoder_fusion` next to the DiT weights.

## Sampling

`sample.py` reads `text_encoder_mode` from the checkpoint config and loads the same stack.

## Other

- **Qwen prompt expansion:** `utils/analysis/llm_client.py` (`load_qwen_causal_lm`, `expand_prompt_qwen`).
- **Stable Cascade (Diffusers):** `python scripts/cascade_generate.py --prompt "..."` — separate from DiT; does not share the DiT forward pass.

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
