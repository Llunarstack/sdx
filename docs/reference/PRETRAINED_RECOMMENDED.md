# Recommended pretrained stack (2026)

Metadata-only guide. **No weights are downloaded by this doc.** Point `pretrained/<Folder>` at hub ids below (or run scaffold / download scripts when you choose to fetch).

Resolution order is implemented in `utils/modeling/model_paths.py`: **local folder first** (including legacy names), then hub fallback.

## Primary upgrades (use these)

| Role | Local folder | Hub id | Replaces |
|------|--------------|--------|----------|
| **T2I text encoder (best Qwen)** | `Qwen3-8B` | `Qwen/Qwen3-8B` | T5-only / older LLM TE for modern DiT/FLUX.2 Klein-style stacks |
| **Prompt LLM** | `Qwen3-14B` | `Qwen/Qwen3-14B` | `Qwen2.5-14B-Instruct` |
| **VLM / caption / Qwen-Image TE path** | `Qwen3-VL-8B-Instruct` | `Qwen/Qwen3-VL-8B-Instruct` | `Qwen2.5-VL-7B-Instruct` |
| **REPA / dense vision** | `DINOv3-ViT-L16` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | `DINOv2-Large` |
| **Vision–text align** | `SigLIP2-SO400M` | `google/siglip2-so400m-patch16-384` | `SigLIP-SO400M` |
| **Preference reward** | `HPSv3` | `MizzenAI/HPSv3` | `HPSv2-hf` |
| **Light VLM / creative RAG** | `moondream3-preview` | `moondream/moondream3-preview` | `moondream2` |

## Still recommended (keep)

| Role | Folder | Hub id | Notes |
|------|--------|--------|-------|
| Classic DiT TE | `T5-XXL` | `google/t5-v1_1-xxl` | Default SDX DiT train/sample path until a ckpt is Qwen-conditioned |
| Multilingual TE | `UMT5-XXL` | `google/umt5-xxl` | Optional |
| CLIP fusion (triple/penta) | `CLIP-ViT-L-14`, `CLIP-ViT-bigG-14`, `CLIP-ViT-H-14`, `LongCLIP-L` | OpenAI / LAION / LongCLIP | Unchanged |
| Depth | `Depth-Anything-V2-Large` | `depth-anything/Depth-Anything-V2-Large` | Prefer DA3 when available |
| Depth (2025+) | `Depth-Anything-V3-Large` | `depth-anything/DA3-LARGE-1.1` | New default depth scaffold |
| Seg | `SAM2-Hiera-Large` | `facebook/sam2-hiera-large-hf` | Keep |
| Foundation T2I (external) | `Qwen-Image` / `Qwen-Image-2512` | `Qwen/Qwen-Image*` | Sibling Diffusers stack, not the SDX DiT |
| Foundation T2I (external) | `FLUX.2-klein-9B` / `FLUX.2-dev` | `black-forest-labs/FLUX.2*` | Uses Qwen3-8B as TE in Klein |
| Modern DiT VAE | `DC-AE-f32c32` | `mit-han-lab/dc-ae-f32c32-mix-1.0` | Latent experiments |
| Extra VLMs | `InternVL3-8B`, `Gemma-3-4B-IT` | OpenGVLab / Google | Caption / critique alternatives |

## Why these picks

- **Qwen3-8B** is what current FLUX.2 Klein / Comfy text-encoder workflows standardize on for LLM-as-TE.
- **Qwen3-14B** beats Qwen2.5-14B for prompt expansion / agentic text.
- **Qwen3-VL-8B-Instruct** supersedes Qwen2.5-VL for captioning and multimodal critique; Qwen-Image historically used Qwen2.5-VL-7B as TE — Qwen3-VL is the upgrade path for new work.
- **DINOv3** outperforms DINOv2 / prior SSL on dense features (REPA, critics).
- **SigLIP2** is the current SigLIP line.
- **HPSv3** (ICCV 2025) is the wide-spectrum preference scorer above HPSv2.
- **moondream3-preview** is the current Moondream VLM (MoE, longer context).

## Code entry points

```python
from utils.modeling.model_paths import (
    default_qwen_path,                 # Qwen3-14B
    default_qwen3_text_encoder_path,   # Qwen3-8B
    default_qwen_vl_path,              # Qwen3-VL-8B
    default_repa_vision_path,          # DINOv3-L
    default_siglip_path,               # SigLIP2
    default_hps_path,                  # prefers HPSv3 (alias: default_hpsv2_path)
    default_moondream_path,            # prefers moondream3 (alias: default_moondream2_path)
    default_depth_anything_path,       # prefers DA3
)
```

Scaffold registry: `utils/modeling/hf_scaffold.py`.  
Status: `python -m scripts.tools.ops.pretrained_status`.

## Fetch later (when you want weights)

```bash
# Config/tokenizer scaffolds only (no weight blobs)
python scripts/download/download_hf_scaffold.py --name Qwen3-8B --name Qwen3-14B --name Qwen3-VL-8B-Instruct
python scripts/download/download_hf_scaffold.py --name DINOv3-ViT-L16 --name SigLIP2-SO400M --name HPSv3 --name moondream3-preview
python scripts/download/download_hf_scaffold.py --name FLUX.2-klein-9B --name Depth-Anything-V3-Large --name DC-AE-f32c32

# Full revolutionary stack (includes weights — heavy)
python scripts/download/download_revolutionary_stack.py
```
