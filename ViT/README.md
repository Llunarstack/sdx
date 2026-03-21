# ViT module (quality + prompt adherence)

## Why this is not “replacing DiT”

**DiT** = diffusion **generator** in `models/`. **`ViT/`** = **scoring / ranking** on finished images (plus caption features). Stacking them is how you get large gains: clean training data, better best-of-N picks, optional reward finetuning. Read **[EXCELLENCE_VS_DIT.md](EXCELLENCE_VS_DIT.md)** for 2024–2026 paper pointers (Swin-DiT, FiT, LaVin-DiT, PRDP, multiscale IQA) and a practical checklist. Suggested **timm** backbones: **[backbone_presets.py](backbone_presets.py)**.

---

This folder is a concrete starter implementation for your **ViT idea**:

- train a Vision Transformer to score:
  - **quality** (`quality_label` -> binary)
  - **prompt adherence** (`adherence_score` -> regression in `[0,1]`)
- run inference on JSONL manifests and append `vit_quality_prob` + `vit_adherence_score`.

## Files

- `config.py` - dataclass for ViT training config
- `dataset.py` - JSONL dataset reader + text feature vector
- `model.py` - `ViTQualityAdherenceModel` with dual heads
- `losses.py` - pairwise ranking loss for stronger ordering
- `ema.py` - exponential moving average model weights
- `tta.py` - test-time augmentation inference helper
- `prompt_system.py` - prompt decomposition (add vs avoid) + compose negative-in-positive
- `prompt_tool.py` - CLI for single prompt or JSONL prompt-plan augmentation
- `train.py` - training loop, saves `best.pt`/`last.pt` (+ optional `best_ema.pt`)
- `infer.py` - annotate manifest rows with ViT scores (+ optional TTA / EMA)
- `rank.py` - rank/filter scored manifests by weighted final score
- `export_embeddings.py` - export fused ViT embeddings to `.npz` for retrieval

## Expected JSONL fields

Required:
- `image_path` (or `path` / `image`)
- `caption` (or `text`)

Optional labels (for supervised training):
- `quality_label` (or `quality`) in `{0,1}`
- `adherence_score` (or `prompt_adherence`) in `[0,1]`

## Train

```bash
python ViT/train.py \
  --manifest-jsonl data/manifest.jsonl \
  --out-dir vit_runs \
  --epochs 5 \
  --ranking-loss-weight 0.2 \
  --save-ema
```

Stronger backbone (VRAM permitting):

```bash
python ViT/train.py --manifest-jsonl data/manifest.jsonl --out-dir vit_runs \
  --model-name vit_large_patch16_224 --batch-size 8
```

Run `python ViT/train.py --help` for a list of suggested `--model-name` values (from `backbone_presets.py`).

## Infer

```bash
python ViT/infer.py \
  --ckpt vit_runs/best.pt \
  --manifest-jsonl data/manifest.jsonl \
  --out data/manifest_vit_scored.jsonl \
  --use-ema \
  --tta
```

## Rank best rows

```bash
python ViT/rank.py \
  --input data/manifest_vit_scored.jsonl \
  --output data/manifest_vit_ranked.jsonl \
  --quality-weight 0.6 \
  --adherence-weight 0.4 \
  --top-k 50000
```

## Export embeddings (retrieval/rerank)

```bash
python ViT/export_embeddings.py \
  --ckpt vit_runs/best.pt \
  --manifest-jsonl data/manifest.jsonl \
  --out-npz data/vit_embeddings.npz
```

## Prompt system (negative inside positive)

```bash
python ViT/prompt_tool.py \
  --prompt "1girl, cinematic light, no blurry, without watermark, sharp eyes"
```

```bash
python ViT/prompt_tool.py \
  --json-in data/manifest.jsonl \
  --json-out data/manifest.prompt_plan.jsonl
```

This adds:
- `vit_prompt_add` (what to add)
- `vit_prompt_avoid` (what to avoid)
- `vit_prompt_composed` (single positive prompt with embedded avoid constraints)

## Why this helps SDX

- You can use ViT scores as dataset QA gates before diffusion training.
- You can weight rows by `vit_quality_prob` or filter low-adherence examples.
- You can use scores for best-of-N ranking alongside existing CLIP/edge/OCR pickers.
- You can use ranking loss + EMA + TTA for stronger ordering robustness.
- You can reuse exported embeddings for nearest-neighbor search and style buckets.

