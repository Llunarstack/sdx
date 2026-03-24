# Local model stack (`model/`)

When folders exist under `model/`, paths resolve automatically (see `utils/modeling/model_paths.py`). For the full runtime picture (repo folders, `ViT/` vs DiT, diagrams), see the **[README § Architecture and pipeline](../README.md#architecture-and-pipeline)** and **[FILES.md](FILES.md)**.

| Role | Folder / default |
|------|------------------|
| T5-XXL | `model/T5-XXL` or `google/t5-v1_1-xxl` |
| CLIP ViT-L/14 | `model/CLIP-ViT-L-14` or `openai/clip-vit-large-patch14` |
| CLIP ViT-bigG/14 | `model/CLIP-ViT-bigG-14` or LAION hub id |
| DINOv2 (REPA / ViT) | `model/DINOv2-Large` or `facebook/dinov2-large` |
| SigLIP | `model/SigLIP-SO400M` |
| Qwen LLM | `model/Qwen2.5-14B-Instruct` |
| Stable Cascade | `model/StableCascade-Prior`, `model/StableCascade-Decoder` |

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
