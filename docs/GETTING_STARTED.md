# Getting Started with SDX

New to SDX? This guide explains what it does and gets you running in minutes.

---

## What Does SDX Do?

SDX is a tool for training custom image generation models on your own data.

**Basic workflow:**
- Input: Your images + text descriptions
- Process: Train a diffusion transformer model
- Output: A model that generates images from text

**Advanced capabilities:**
- Precise per-step generation control (Holy Grail scheduling)
- Create novel artistic styles beyond artist imitation (Style Genome)
- Automatic quality filtering for difficult prompts (TCIS)
- Modern training techniques (flow matching, preference learning, distillation)

---

## Quick Start (30 seconds)

```bash
# Install dependencies
pip install -r requirements.txt

# Generate your first image (uses pretrained weights)
python demo.py

# Output is saved to out.png
```

---

## Core Concepts

### Training (train.py)

Train a model on your image dataset.

Input:
- Image folder with captions (or JSONL manifest)
- Configuration (batch size, learning rate, epochs)

Output:
- Trained checkpoint file (best.pt)

```bash
python train.py --data-path my_images/ --results-dir outputs/
```

Features:
- Flow matching (faster, newer approach)
- DPO training (learn from preference feedback)
- LoRA adapters (efficient training)
- Multi-GPU support (distributed training)

### Sampling (sample.py)

Generate images from a trained model.

Input:
- Text prompt
- Checkpoint file

Output:
- Generated image

```bash
python sample.py --ckpt outputs/best.pt --prompt "a cat wearing sunglasses" --out result.png
```

Features:
- Holy Grail scheduling (per-step adaptation)
- TCIS quality filtering (hard prompt reliability)
- Style Genome (novel aesthetic generation)
- LoRA blending (combine multiple models)

### Style Genome

Create original visual aesthetics instead of artist imitation.

```bash
# Generate 6 novel styles for a prompt
python -m scripts.tools explore_styles \
  --prompt "samurai warrior" --genomes 6 --mode apocalypse
```

Five style axes:
- Palette: colors and dominant hues
- Line: stroke mechanics and contours
- Surface: texture and material properties
- Camera: composition and perspective
- Signature: recurring visual motifs

---

## First Training Run

### Step 1: Prepare Data

Create a folder with images and captions:

```
my_dataset/
  ├── image1.png
  ├── image1.txt          # caption: "a red car"
  ├── image2.png
  ├── image2.txt          # caption: "a blue bicycle"
  └── ...
```

Or create a manifest.jsonl:

```json
{"image_path": "image1.png", "caption": "a red car"}
{"image_path": "image2.png", "caption": "a blue bicycle"}
```

### Step 2: Verify GPU Setup

```bash
python -m toolkit.training.env_health
```

Checks:
- CUDA availability
- VRAM (minimum 16GB recommended)
- Library installation

### Step 3: Start Training

```bash
python train.py \
  --data-path my_dataset/ \
  --results-dir results/ \
  --flow-matching-training \
  --num-epochs 20
```

Training process:
1. Images loaded and encoded to latent space
2. Model initialized
3. Training loop executes (loss decreases over time)
4. Checkpoints saved every 100 steps
5. Best checkpoint saved to results/best.pt

Timeline:
- 100 images on RTX 3090: ~20 hours
- 100 images on RTX 4090: ~5 hours
- Scales with number of images and GPU capability

### Step 4: Generate Images

```bash
python sample.py \
  --ckpt results/best.pt \
  --prompt "your description here" \
  --out result.png
```

Sample configurations:

```bash
# High quality (slower)
python sample.py --ckpt results/best.pt \
  --prompt "a red car, professional photo" \
  --steps 50 --cfg-scale 8.0 --out car.png

# Fast generation
python sample.py --ckpt results/best.pt \
  --prompt "a red car" \
  --steps 20 --cfg-scale 5.0 --out car_fast.png

# With style variation
python sample.py --ckpt results/best.pt \
  --prompt "a red car" \
  --invent-styles 1 --style-chaos-level 0.8 --out car_styled.png
```

---

## Configuration Parameters

Key training flags (full list: `python train.py --help`):

| Parameter | Purpose | Example |
|-----------|---------|---------|
| --data-path | Image dataset location | my_dataset/ |
| --results-dir | Checkpoint output directory | results/ |
| --num-epochs | Training passes through data | 20 |
| --batch-size | Images per training step | 4 |
| --learning-rate | Training speed | 5e-5 |
| --flow-matching-training | Use flow matching objective | (recommended) |
| --use-bf16 | 16-bit precision (faster, less memory) | (recommended) |
| --grad-checkpointing | Memory optimization | (default enabled) |

---

## Sampling Parameters

| Parameter | Purpose | Range | Notes |
|-----------|---------|-------|-------|
| --steps | Number of generation steps | 20-60 | Higher = better quality, slower |
| --cfg-scale | Prompt adherence strength | 5.0-15.0 | Higher = follow prompt strictly |
| --holy-grail-preset | Adaptive scheduling mode | auto, high_quality | auto recommended |
| --num | Batch generate N images | 1-10 | Multi-image generation |
| --pick-best | Auto quality filtering | auto | Uses ViT model to score |

---

## Common Use Cases

### Fine-tune Pretrained Model

Train on your data using a pretrained checkpoint as initialization:

```bash
python train.py \
  --data-path my_dataset/ \
  --init-ckpt pretrained.pt \
  --results-dir results/
```

Training converges faster with pretrained weights.

### Blend Multiple Trained Models

Train separate models, then mix them:

```bash
# Train style-specific models
python train.py --data-path anime_images/ --results-dir anime_model/
python train.py --data-path photo_images/ --results-dir photo_model/

# Generate with both models blended
python sample.py \
  --ckpt anime_model/best.pt \
  --lora photo_model/best.pt:0.5:style \
  --prompt "a character"
```

### Hard Prompt Generation (Text, Layouts)

For difficult prompts (text in images, exact layouts), use quality filtering:

```bash
python -m scripts.tools hybrid_dit_vit_generate \
  --ckpt results/best.pt \
  --vit-ckpt vit_quality_model.pt \
  --prompt "poster with text HELLO WORLD" \
  --num 10 --pick-best auto
```

---

## Troubleshooting

**CUDA out of memory**
- Reduce --batch-size to 1 or 2
- Use --image-size 256 (smaller images)
- Grad checkpointing is on by default (reduces memory)

**"No module named X"**
- Reinstall: pip install -r requirements.txt

**Training too slow**
- Enable --flow-matching-training (faster)
- Enable --use-bf16 (16-bit precision)
- Reduce --num-epochs
- Use a larger batch size if VRAM allows

**Generated images are low quality**
- Train longer: increase --num-epochs (50+ recommended)
- Use more data (minimum 100 images)
- Write detailed captions

---

## Advanced Topics

For deeper exploration:

- [Training Techniques](TRAINING_TEXT_TO_PIXELS.md) - Flow matching, DPO, distillation
- [Sampling Guide](HOLY_GRAIL_OVERVIEW.md) - Holy Grail scheduling, TCIS
- [Style Genome](../README.md#style-genome-invent-original-looks) - Custom aesthetic creation
- [Codebase Architecture](CODEBASE_GUIDE.md) - How SDX is structured

Advanced sampling options:
- --style-inventor-mode insane for style variation
- --holy-grail-preset high_quality for maximum quality
- --flow-matching-sample for flow-trained checkpoints

---

## Frequently Asked Questions

**How much training data do I need?**

Minimum: 50-100 images. Recommended: 500-5000. Model quality scales with data quantity and quality.

**How long is training?**

Depends on GPU and data:
- RTX 3090 with 100 images: approximately 20 hours
- RTX 4090 with 100 images: approximately 5 hours

**Do I need a GPU?**

Training requires GPU (NVIDIA with CUDA). Sampling is 10-100x faster on GPU but possible on CPU.

**Can I use data from other artists?**

Only use images you have rights to. Respect copyright and intellectual property.

**Flow matching vs regular training?**

Flow matching is the newer approach. Use it for new projects unless you have specific requirements otherwise.

**Commercial use?**

SDX is Apache 2.0 licensed. Yes, commercial use is permitted. See LICENSE file.

---

## Resources

- [Documentation Index](README.md) - Complete reference
- [Reproducibility](guides/REPRODUCIBILITY.md) - Same seed / deterministic runs
- [Main README](../README.md) - Feature overview and architecture
- [GitHub Repository](https://github.com/Llunarstack/sdx) - Source code and issues

---

## Summary

You now understand:
- What SDX does (training and sampling text-to-image models)
- How to prepare data
- Basic training and sampling workflows
- Advanced features available

Start with the quick demo, then train on your own data. The codebase is designed to be readable—review train.py and sample.py to understand the full pipeline.
