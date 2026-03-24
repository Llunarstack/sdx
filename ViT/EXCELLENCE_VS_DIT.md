# ViT stack vs DiT — how to get “100×” leverage (research + SDX)

**Important distinction:** **`ViT/` in this repo is not a replacement for the DiT generator.**  
- **DiT** (`models/dit_text.py`, `train.py`): *generates* latents → images (diffusion).  
- **`ViT/`** (`ViT/model.py`): *scores* images + captions (quality, adherence) for **dataset QA**, **filtering**, **retrieval**, and **best-of-N** — same *role* as a reward model or IQA network, not the same loss as diffusion.

So “100× better than DiT” in practice means: **stack a strong discriminative stack on top of (or beside) DiT** so that *training data*, *sampling selection*, and *optional finetuning* compound — not swapping ViT for DiT inside the denoiser (that would be a different project).

---

## 1. What recent papers suggest (2024–2026)

### A. Faster / stronger *diffusion transformers* (for future DiT work in SDX)

These target **generation** (replace or extend `DiT_Text`), not the `ViT/` QA module:

| Direction | Idea | Pointer |
|-----------|------|---------|
| **Window / local attention in DiT** | Reduce redundant global attention; shifted-window style behavior | **Swin-DiT** — pseudo shifted windows, efficiency + quality (e.g. arXiv [2505.13219](https://arxiv.org/abs/2505.13219)) |
| **Large unified vision diffusion** | Joint spatial-temporal VAE + scalable DiT, multi-task | **LaVin-DiT** (CVPR 2025; arXiv [2411.11505](https://arxiv.org/abs/2411.11505)) |
| **Flexible resolution** | Tokens that scale with image layout, not fixed grid | **FiT** — flexible vision transformer for diffusion (arXiv [2402.12376](https://arxiv.org/abs/2402.12376)) |

*Integration path in SDX:* new backbones or blocks under `models/`, new `train.py` flags — see `config/train_config.py` and `IMPROVEMENTS.md`.

### B. Reward / preference alignment (pairs with **your** ViT scores)

| Direction | Idea | Pointer |
|-----------|------|---------|
| **Reward finetuning** | Optimize diffusion with learned rewards | **DRaFT** — direct reward finetuning through sampling (ICLR 2024) |
| **Stable large-scale reward training** | Predict reward *differences* on trajectories | **PRDP** — proximal reward difference prediction (arXiv [2402.08714](https://arxiv.org/abs/2402.08714)) |
| **Richer than scalar scores** | Critiques + edits as supervision | **RPO**-style VLM critique pipelines (see recent “preference optimization” literature) |

*Integration path:* export ViT scores → weighted JSONL → filter / reweight pre-training; later, plug scores into external RLHF-style trainers that consume image+prompt pairs.

### C. No-reference **image quality** (IQA) with transformers

| Direction | Idea | Pointer |
|-----------|------|---------|
| **Multiscale + attention** | Coarse/fine cues for blind IQA | **MS-SCANet**-style multiscale transformers (e.g. arXiv [2602.04032](https://arxiv.org/abs/2602.04032)) |
| **Ranking + consistency** | Relative ranking losses, augmentation robustness | ADTRS-style relative ranking + self-consistency (arXiv [2409.07115](https://arxiv.org/abs/2409.07115)) |

*Integration path:* same as current `ViT/losses.py` ranking loss — increase **pairwise / listwise** data and consider **multi-scale inputs** (future code: second branch or higher `image_size`).

---

## 2. Practical “100×” checklist inside **this** repo

### Short term (no new architectures)

1. **Stronger timm backbone** — use `ViT/backbone_presets.py` candidates (`vit_large_*`, `swin_*`, `convnext_*`) with `--model-name` in `ViT/train.py`.
2. **More ranking supervision** — raise `--ranking-loss-weight`, curate *pairs* (same prompt, better/worse image).
3. **EMA + TTA** — already in `ViT/ema.py`, `ViT/tta.py`; keep for inference stability.
4. **Bigger `text_feat_dim` + richer `text_feature_vector`** — if captions carry structure, upgrade the text side (see `ViT/dataset.py`) so adherence isn’t from 8 random dims.
5. **Wire into sampling** — merge ViT scores with `utils/quality/test_time_pick.py` / `sample.py --pick-best` (ensemble: CLIP + edge + OCR + ViT).

### Medium term

6. **Frozen semantic encoder** — concatenate **DINOv2** or **CLIP** image embeddings before the fusion MLP (aligns with REPA philosophy: strong frozen vision features).
7. **Multi-crop / multi-scale scoring** — average predictions over scales (extends TTA).
8. **Latent-space scoring** — small head on VAE latents for speed (score before decode).

### Long term

9. **Swin-DiT / FiT-style** changes land in **`models/`** DiT, not in `ViT/` QA.
10. **Reward finetuning** — export ViT as one channel in PRDP/DRaFT-style pipelines (external or future `scripts/training/`).

---

## 3. Mental model

```text
DiT  ──generates──►  images
                      │
ViT / CLIP / OCR  ──scores──►  filter data · pick-best · (optional) RLHF
```

Used well, the **discriminative** stack doesn’t replace DiT — it **amplifies** every DiT sample by choosing better data and better outputs.

---

## See also

- [ViT/README.md](README.md) — CLI and JSONL format  
- [ViT/backbone_presets.py](backbone_presets.py) — suggested `timm` names  
- [docs/MODEL_STACK.md](../docs/MODEL_STACK.md) — DINOv2, CLIP, triple encoder  
- [README.md § Who is who](../README.md#who-is-who-easy-to-confuse) — DiT vs `ViT/` package
