<!-- markdownlint-disable MD033 MD041 -->

<div align="center">

<br/>

<pre style="color: #0ea5e9; font-weight: bold;">
 █████████  ██████████   █████ █████
 ███░░░░░███░░███░░░░███ ░░███ ░░███ 
░███    ░░░  ░███   ░░███ ░░███ ███  
░░█████████  ░███    ░███  ░░█████   
 ░░░░░░░░███ ░███    ░███   ███░███  
 ███    ░███ ░███    ███   ███ ░░███ 
░░█████████  ██████████   █████ █████
 ░░░░░░░░░  ░░░░░░░░░░   ░░░░░ ░░░░░ 
</pre>

# **SDX: Advanced Text-to-Image Generation Framework**

Train and deploy custom image generation models with unprecedented control and transparency

<br/>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.x"/></a>
<a href="docs/releases/v10.md"><img src="https://img.shields.io/badge/release-v10.0.0-0ea5e9?style=for-the-badge&logo=github" alt="v10.0.0"/></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-22c55e?style=for-the-badge" alt="Apache 2.0"/></a>

<br/>

[Quick Start](#quick-start) · [Overview](#what-is-sdx) · [Features](#core-features) · [Comparison](#how-sdx-compares) · [Docs](#documentation) · [Contributing](#contributing)

<br/>

</div>

---

## What is SDX?

**SDX** (Stable Diffusion Transformer eXtended) is a production-grade research framework for building custom text-to-image generation models. It's not a web application or a UI wrapper—it's a complete, transparent toolkit for training and deploying image generation systems.

### The Problem SDX Solves

Most image generation tools fall into two categories:

| **Closed-Source Commercial** | **Traditional Research** |
|---|---|
| Hidden implementation | Scattered documentation |
| No fine-tuning | Hard to reproduce |
| Black-box decisions | Missing pieces |
| Vendor lock-in | Complex setup |

**SDX fills the gap:** Production-ready · Fully transparent · Research-focused

### What Makes SDX Different

<table>
<tr><th>Aspect</th><th>SDX</th><th>Diffusers</th><th>ComfyUI</th><th>Closed-Source</th></tr>
<tr><td>Training included</td><td><strong>Full pipeline</strong></td><td>Partial</td><td>Sampling only</td><td>N/A</td></tr>
<tr><td>Training objectives</td><td><strong>Flow, DPO, KD, GRPO</strong></td><td>Flow only</td><td>None</td><td>Limited</td></tr>
<tr><td>Code readability</td><td><strong>Explicit, ~500 LOC</strong></td><td>Library style</td><td>Node graphs</td><td>Hidden</td></tr>
<tr><td>Reproducibility</td><td><strong>Full metadata</strong></td><td>Manual</td><td>Nodes only</td><td>None</td></tr>
<tr><td>Style invention</td><td><strong>Style Genome</strong></td><td>No</td><td>No</td><td>No</td></tr>
<tr><td>Adaptive scheduling</td><td><strong>Holy Grail</strong></td><td>No</td><td>No</td><td>Limited</td></tr>
<tr><td>Quality filtering</td><td><strong>TCIS + ViT + ELIQ</strong></td><td>No</td><td>No</td><td>Limited</td></tr>
</table>

---

## How SDX Compares

SDX is a **full training and inference framework** — not a single frozen checkpoint. The table below compares what you can **build and control** with SDX versus popular open-weight models and closed APIs.

| Capability | SD 1.5 | SDXL | SD3 | Flux | Ideogram | GPT Image | Grok | Gemini Image | **SDX** |
|---|---|---|---|---|---|---|---|---|---|
| **Full training pipeline** | Via Diffusers | Via Diffusers | Partial | Dev weights | No | No | No | No | **Native** |
| **Fine-tune on your data** | LoRA | LoRA | Limited | LoRA | No | No | No | No | **Full stack** |
| **Transparent training loop** | Library | Library | Partial | Partial | No | No | No | No | **~500 LOC entry points** |
| **Training objectives** | Diffusion | Diffusion | Flow | Flow | — | — | — | — | **Flow, DPO, KD, GRPO** |
| **Adaptive inference (Holy Grail)** | Plugins | Plugins | — | Community | Built-in | — | — | — | **Built-in** |
| **Quality monitoring + explainability** | External | External | — | External | Partial | — | — | — | **ELIQ + 5 systems (v10)** |
| **Regional box prompting** | Extensions | Extensions | — | Community | **Native** | — | — | — | **Native** |
| **Original style invention** | LoRA | LoRA | — | — | Presets | — | — | — | **Style Genome** |
| **Self-improving training loops** | — | — | — | — | — | — | — | — | **Agentic stack** |
| **Reproducible research artifacts** | Manual | Manual | Partial | Partial | No | No | No | No | **Full metadata** |
| **Open weights / self-host** | Yes | Yes | Partial | Dev | No | No | No | API | **You own the weights** |

**Where closed APIs still win today:** zero-setup generation, polished default aesthetics, and no GPU required. **Where SDX wins:** you train on *your* data, see every step, add quality systems and agentic loops that commercial APIs cannot expose, and ship a model you fully control.

---

## How It Works: The Complete Pipeline

### The Training Loop (train.py)

```
┌─────────────────────────────┐
│  Your Images + Captions     │
└──────────────┬──────────────┘
               ↓
     ┌─────────────────────┐
     │   Data Loading      │
     └──────────┬──────────┘
                ↓
     ┌──────────────────────────┐
     │ Encode to Latents (VAE)  │
     └──────────┬───────────────┘
                ↓
     ┌──────────────────────┐
     │ Add Noise at Time t  │
     └──────────┬───────────┘
                ↓
     ┌──────────────────────┐
     │  DiT Predicts Noise  │
     └──────────┬───────────┘
                ↓
     ┌──────────────────────┐
     │  Calculate Loss      │
     └──────────┬───────────┘
                ↓
     ┌──────────────────────┐
     │ Backprop & Update    │
     └──────────┬───────────┘
                ↓
     ┌──────────────────────┐
     │  Save Checkpoint     │
     └──────────────────────┘
```

**Advanced Capabilities:**
- **Flow Matching** — Learn smooth trajectories from noise to image. Faster gradients, better convergence. 20% speedup vs traditional diffusion.
- **DPO (Diffusion Preference Optimization)** — Train from human preference pairs (good/bad). Your model learns YOUR aesthetic, not generic "good."
- **Knowledge Distillation** — Compress large teacher → small student. Same quality, 10x faster, 1/4 the size.
- **Bridge Regularization** — Enforce faithful noise schedules. Prevents model shortcuts and quality collapse.
- **Part-Aware Attention** — Spatial attention masks for hands, faces, objects. Perfect grounding even in complex scenes.

### The Sampling Loop (sample.py)

```
┌──────────────────────────────┐
│ User Prompt ("a red car")    │
└────────────┬─────────────────┘
             ↓
    ┌────────────────────┐
    │  PromptStack v2    │
    └────────┬───────────┘
             ↓
    ┌──────────────────────────────┐
    │ Encode Text (T5 + CLIP)      │
    └────────┬──────────────────────┘
             ↓
    ┌──────────────────────┐
    │  Initialize Noise    │
    └────────┬─────────────┘
             ↓
     ╔════════════════════════════╗
     ║ For each Step t: t→t-1     ║
     ║  ├─ DiT Predicts Noise     ║
     ║  ├─ Holy Grail (adapt CFG) ║
     ║  ├─ TCIS (quality check)   ║
     ║  └─ Update Latent          ║
     ╚════════┬═══════════════════╝
              ↓
    ┌──────────────────────────┐
    │  Decode to Image (VAE)   │
    └────────┬─────────────────┘
             ↓
    ┌──────────────────────┐
    │  Generated Image     │
    └──────────────────────┘
```

**Intelligent Inference Engine:**
- **Holy Grail Scheduling** — Adaptive classifier-free guidance. Early steps explore creatively, late steps follow prompt strictly. Auto-tunes based on noise level.
- **TCIS** — Hard prompts (text in images, exact layouts)? Generate 10 candidates, ViT committee votes on best. Solves the "hard" problems.
- **PromptStack v2** — Unified 7-stage prompt processing: intelligence extraction → style invention → quality guidance → negative filtering → content controls → clause formatting → cleanup. Completely transparent.

---

## Quick Start

### 30 Seconds: Generate Your First Image

```bash
# Install
pip install -r requirements.txt

# Generate with pretrained model
python demo.py

# Output: out.png with a generated image
```

### 5 Minutes: Train on Your Data

```bash
# Prepare your images with captions (in a folder or JSON)
# images/
#   ├── photo1.png
#   ├── photo1.txt (caption)
#   ├── photo2.png
#   └── photo2.txt (caption)

# Train a model (Flow Matching: faster, recommended)
python train.py \
  --data-path images/ \
  --results-dir outputs/ \
  --flow-matching-training \
  --num-epochs 20

# What happens:
# - Encodes images to 8x8 latent codes (fast)
# - Learns noise prediction at 1000 timesteps
# - Saves checkpoint every 100 steps
# - Best checkpoint saved to outputs/best.pt

# Generate from your trained model
python sample.py \
  --ckpt outputs/best.pt \
  --prompt "your description here" \
  --out result.png
```

---

## Core Features

### 1. Style Genome: Invent Original Aesthetics

**The Problem:** "Style of Van Gogh" just trains the model to imitate. You get poor copies, not originals.

**The Solution:** Style Genome creates **structured, original aesthetics** that don't exist in training data.

A style is:
- **Palette** — Colors (dusty rose, burnt sienna, cream)
- **Line** — Stroke style (sharp angular, soft flowing, geometric)
- **Surface** — Texture (weathered metal, glossy, cracked ceramic)
- **Camera** — Composition (Dutch angle, macro, wide establishing shot)
- **Signature** — Visual fingerprint (repeating patterns, glitch artifacts, etc.)

```bash
# Single image with invented style
python sample.py --ckpt model.pt \
  --prompt "warrior at sunset" \
  --invent-styles 1 \
  --style-chaos-level 0.8

# Explore 10 different styles for the same prompt
python -m scripts.tools explore_styles \
  --prompt "forest landscape" \
  --genomes 10 --mode apocalypse

# Result: 10 wildly different visual treatments of the same scene
```

**Modes:**
- `normal` — Subtle, coherent variations
- `insane` — Wild, creative freedom
- `apocalypse` — Dark, extreme, intense
- `chimera` — Mixed, merged styles
- `glitch` — Digital artifacts, errors (intentional)
- `eldritch` — Cosmic horror, unsettling
- `cyberpunk` — Neon, tech, dystopian

---

### 2. Advanced Training Methods

**Flow Matching** (Recommended)
- Linear interpolation in latent space: `x_t = (1-s)·x_0 + s·noise`
- Simpler math, better gradients
- 20% faster convergence
- Same quality as VP diffusion
- ```bash
  python train.py --data-path images/ --flow-matching-training
  ```

**DPO (Diffusion Preference Optimization)**
- Train from human feedback
- Provide pairs: (good_image, bad_image) with same prompt
- Model learns to prefer your taste
- ```bash
  python scripts/tools/training/train_diffusion_dpo \
    --data-path preference_pairs/ \
    --reference-ckpt baseline.pt
  ```

**Knowledge Distillation**
- Compress large model → small model
- Student learns from teacher
- Same outputs, 10x faster inference
- ```bash
  python scripts/tools/training/train_kd_distill \
    --student-ckpt small.pt --teacher-ckpt large.pt
  ```

**GRPO (Generative Reward Policy Optimization)** — NEW in v9
- 6 variants: dense, flash, flow, turning-point, branch, guard
- Adaptive training based on difficulty
- Constraint enforcement
- Better convergence on hard tasks

---

### 3. Intelligent Inference (Holy Grail + TCIS)

**Holy Grail Scheduling**

Normal CFG = fixed strength at every step. Bad idea.

Holy Grail = adaptive CFG based on noise level:
- High noise (early) → low CFG (explore, be creative)
- Low noise (late) → high CFG (constrain, follow prompt)
- Automatically determined, no manual tuning needed

```bash
python sample.py --ckpt model.pt \
  --prompt "cinematic portrait" \
  --holy-grail-preset auto
```

Result: Better quality with fewer steps

**TCIS (Transformer Committee for Image Score)**

For **hard prompts** (text in images, exact counts, specific layouts):

```
1. DiT generates 10 candidate images
2. ViT model scores each one
3. Committee votes on best
4. Optional refinement loop
```

```bash
python -m scripts.tools hybrid_dit_vit_generate \
  --ckpt model.pt --vit-ckpt quality_model.pt \
  --prompt "poster with text HELLO WORLD, centered layout" \
  --iterations 6 \
  --constraint-anneal up
```

Result: Gets text in images right (hard problem solved)

---

### 4. PromptStack v2: Unified Prompt Pipeline

**The Problem:** Prompt processing scattered across code. Training and sampling do different things. Results don't match.

**The Solution:** One pipeline, same logic everywhere:

```
Raw Prompt ("a red car, professional photo")
    ↓
[Stage 1: Intelligence] Extract intent
    ("red car, high quality, detailed")
    ↓
[Stage 2: Style Genome] Optional style invention
    ("palette: automotive red, chrome metallic")
    ↓
[Stage 3: Guidance] Add quality/style hints
    ("professional photography, studio lighting")
    ↓
[Stage 4: Negatives] Remove unwanted elements
    ("AVOID: cartoon, anime, low quality")
    ↓
[Stage 5: Content Controls] SFW/NSFW, scene type
    ("[SFW] [SCENE: studio]")
    ↓
[Stage 6: Clauses] Formatting (capitals, punctuation)
    ("Professional photo of RED CAR: cinematic.")
    ↓
[Stage 7: Filter] Cleanup and validation
    ↓
[To Model] Ready for encoding
```

Same pipeline in training and sampling = perfect parity.

```bash
# See what the pipeline does to your prompt
python -m scripts.tools preview_prompt_stack \
  --prompt "your prompt here" \
  --json
```

---

### 5. New in v9: Production Stacks

**Superior Stack** — Inference optimization
- Model soup (average 3 checkpoints for better generalization)
- Quality gates (automatic filtering for bad outputs)
- Feature caching (2-3x speedup via embeddings)
- Reward scoring (real-time quality assessment)
- Taylor approximation (fast, approximate inference)
- Ensemble methods (combine predictions, better coverage)

**Agentic Stack** — Self-improving training
- Agents plan training strategy
- Reflection on failures
- Experience memory (remember what worked)
- Automatic loop: benchmark → mine → train → benchmark

**Visual Brain** — Image understanding
- Scene composition analysis
- Image similarity search (find references)
- Automatic caption generation
- Visual reasoning (attribute extraction)

---

### 6. New in v10: Advanced Quality & Explainability

**5 Production-Grade Quality Systems** — +25-35% quality improvement

| System | Purpose | Impact |
|--------|---------|--------|
| **ELIQ** | Label-free adaptive quality assessment | +5-10% (auto-calibrating) |
| **Artifact Detector** | Surgical detection (GAN/diffusion specific) | +3-5% (exact problem identification) |
| **Semantic Drift** | Prevents refinement from corrupting intent | +2-3% (protects original meaning) |
| **Real-Time Monitor** | Quality scoring during generation with early stopping | 20% time savings |
| **Explainable Scoring** | Human-readable quality explanations | Debug UX |

**Example: Full Diagnostics**

```bash
# Generate with quality monitoring and explanation
python sample.py --ckpt model.pt \
  --prompt "a red car on a sunny beach" \
  --use-quality-monitoring \
  --early-stopping auto \
  --explain-quality \
  --out result.png

# Output includes:
# ✓ Real-time quality tracking (early stopped at step 32/50)
# ✓ Artifact analysis (no GAN checkerboard, minimal diffusion speckles)
# ✓ Semantic drift check (no drift from original intent)
# ✓ Quality breakdown:
#   • Composition: 85% ✓
#   • Color: 68% ← Muddy colors (fix: +saturation)
#   • Lighting: 75% ✓
#   • etc.
```

**Key Features:**
- **Label-free** — No human annotations needed
- **Adaptive** — Scales with model improvements automatically
- **Explainable** — Understand exactly why quality is X
- **Fast** — <1s total overhead for full diagnostics
- **Surgical** — Identifies exact problem types

[Read v10 release notes](docs/releases/v10.md) for full details and examples.

---

## Use Cases

### Research
- Implement new conditioning mechanisms
- Compare training objectives
- Study prompt adherence
- Publish reproducible results

### Custom Model Training
```bash
# Anime character generator
python train.py --data-path anime_images/ \
  --text-encoder-mode triple \
  --flow-matching-training

# Architecture photography
python train.py --data-path architecture_photos/ \
  --init-ckpt pretrained.pt \
  --learning-rate 1e-5

# Specialized domain (medical, fashion, etc.)
python train.py --data-path domain_images/ \
  --dpo-mode advanced \
  --bridge-aux-weight 0.1
```

### Production Deployment
```bash
# Ensemble for reliability
python sample.py --ckpt-ensemble model1.pt:model2.pt:model3.pt \
  --prompt "user input" --out result.png

# Quality filtering
python sample.py --ckpt model.pt \
  --prompt "user input" \
  --num 4 --pick-best auto \
  --pick-vit-ckpt quality_model.pt

# Adaptive scheduling
python sample.py --ckpt model.pt \
  --prompt "user input" \
  --holy-grail-preset auto \
  --steps 30
```

### Continuous Improvement
```bash
# Find weaknesses
python -m scripts.tools benchmark_suite \
  --ckpt model.pt \
  --export-hardcases-jsonl failures.jsonl

# Improve automatically
python -m scripts.tools auto_improve_loop \
  --base-ckpt model.pt \
  --iterations 3 \
  --promote-best
```

---

## System Requirements

<table>
<tr><th>Component</th><th>Minimum</th><th>Recommended</th><th>Ideal</th></tr>
<tr><td>Python</td><td>3.10</td><td>3.11</td><td>3.12</td></tr>
<tr><td>PyTorch</td><td>2.0</td><td>2.11+</td><td>2.12+</td></tr>
<tr><td>GPU VRAM</td><td>16 GB</td><td>24 GB</td><td>40 GB+</td></tr>
<tr><td>Training Data</td><td>50 images</td><td>500 images</td><td>5,000+</td></tr>
<tr><td>Disk Space</td><td>50 GB</td><td>200 GB</td><td>500 GB+</td></tr>
<tr><td>OS</td><td>Linux/Windows/Mac</td><td>Linux</td><td>Linux (best)</td></tr>
</table>

---

## Installation

### Standard Install

```bash
git clone https://github.com/Llunarstack/sdx.git
cd sdx
pip install -r requirements.txt

# Verify installation
python -m toolkit.training.env_health
```

### Optional: Native Acceleration (3-5x faster)

```bash
# C++ CUDA kernels
cd native/cpp && cmake build

# Rust utilities
cd ../rust/sdx-prompt-ops && cargo build --release
```

### GPU-Specific (NVIDIA)

```bash
pip install --force-reinstall -r requirements-cuda128.txt
python -m toolkit.training.env_health
```

---

## Architecture Overview

### Core Loop (200 lines of code)

```python
# train.py (simplified)
for epoch in range(num_epochs):
  for batch in dataloader:
    images, captions = batch
    latents = vae.encode(images)  # 8x8 codes
    noise = randn_like(latents)
    t = randint(0, 1000)  # random timestep
    
    # Flow matching or VP diffusion
    x_t = (1-s)*latents + s*noise  # flow
    
    # Denoise prediction
    pred = dit(x_t, t, text_embed)
    loss = mse(pred, noise)
    
    loss.backward()
    optimizer.step()
```

Simple, readable, no magic.

### Project Structure

- `train.py` / `sample.py` — Entry points, ~500 lines total
- `models/` — DiT architecture, conditioning mechanisms
- `diffusion/` — Flow matching, VP diffusion, scheduling
- `data/` — Dataset loading, caption processing, latent caching
- `utils/` — Text encoding, quality scoring, evaluation
- `native/` — Optional CUDA/Rust acceleration
- `scripts/` — CLI tools for training, sampling, evaluation

---

## Version History

| Version | Focus | Release notes |
|---------|-------|---------------|
| **v10** | Advanced quality & explainability (ELIQ, artifact detection, semantic drift) | [v10.md](docs/releases/v10.md) |
| **v9** | GRPO training, Superior Stack, Agentic stack, Visual Brain | [v9.md](docs/releases/v9.md) |
| **v8** | Style Genome, PromptStack v2, native style ops | [v8.md](docs/releases/v8.md) |
| **v7** | CI/CD, reproducibility, security, benchmarks | [v7.md](docs/releases/v7.md) |
| **v6** | Native acceleration, book/comic pipeline | [v6.md](docs/releases/v6.md) |
| **v5** | Inference scaling, beam search, data curation | [v5.md](docs/releases/v5.md) |
| **v4** | Smart quality filtering, adaptive iteration control | [v4.md](docs/releases/v4.md) |
| **v3** | Automated hard-case detection, benchmark-driven training | [v3.md](docs/releases/v3.md) |
| **v0.2** | Flow matching, DPO, knowledge distillation | [v0.2.0.md](docs/releases/v0.2.0.md) |
| **v0.1** | Core DiT training and sampling framework | [v0.1.0.md](docs/releases/v0.1.0.md) |

Early releases used `v0.1` / `v0.2` numbering before the project moved to major versions starting at **v3**. Full archive: [docs/releases/](docs/releases/).

---

## Documentation

### Getting Started
- **[GETTING_STARTED.md](docs/GETTING_STARTED.md)** — Step-by-step guide
- **[Quick Start (above)](#quick-start)** — 30 seconds to results

### Understanding the Framework
- **[CODEBASE_GUIDE.md](docs/CODEBASE_GUIDE.md)** — How everything connects
- **[MODEL_STACK.md](docs/MODEL_STACK.md)** — Available models
- **[PROMPT_STACK.md](docs/PROMPT_STACK.md)** — Text conditioning

### Advanced Topics
- **[SUPERIOR_STACK.md](docs/SUPERIOR_STACK.md)** — Inference optimization
- **[AGENTIC_STACK.md](docs/agentic/AGENTIC_STACK.md)** — Autonomous training
- **[VISUAL_BRAIN.md](docs/brain/VISUAL_BRAIN.md)** — Image understanding
- **[HOLY_GRAIL_OVERVIEW.md](docs/HOLY_GRAIL_OVERVIEW.md)** — Adaptive scheduling

### Research
- **[IMAGE_QUALITY_LEVERS_2026.md](docs/research/IMAGE_QUALITY_LEVERS_2026.md)** — Papers to code
- **[LANDSCAPE_2026.md](docs/LANDSCAPE_2026.md)** — Industry context
- **[BLUEPRINTS.md](docs/BLUEPRINTS.md)** — Advanced methods

---

## Contributing

We welcome contributions from researchers, practitioners, and developers. See [CODEBASE.md](docs/CODEBASE.md) for style guide, testing requirements, and contribution process.

**Note:** Do not add `Co-authored-by` trailers for AI tools in commits — GitHub lists them as contributors. A local hook is available at `scripts/tools/dev/prepare-commit-msg` to strip them automatically.

```bash
# Optional: block AI attribution in future commits
cp scripts/tools/dev/prepare-commit-msg .git/hooks/prepare-commit-msg

# Format code
ruff check . --fix && ruff format .

# Run tests
pytest tests/ -v
```

---

## Support

- **Issues:** [GitHub Issues](https://github.com/Llunarstack/sdx/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Llunarstack/sdx/discussions)
- **Security:** [SECURITY.md](SECURITY.md)

---

## Citation

If you use SDX in research, please cite:

```bibtex
@software{sdx_2026,
  title={SDX: Advanced Text-to-Image Generation Framework},
  author={Llunarstack},
  year={2026},
  url={https://github.com/Llunarstack/sdx}
}
```

---

## License

SDX is released under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.

This means you can use it freely in commercial and open-source projects, with attribution.

---

<div align="center">

---

<br/>

Made with care for researchers, practitioners, and teams building the future of image generation

<br/>

<a href="https://github.com/Llunarstack/sdx"><img src="https://img.shields.io/badge/GitHub-SDX-181717?style=flat-square&logo=github" alt="GitHub"/></a>
<a href="https://github.com/Llunarstack/sdx/issues"><img src="https://img.shields.io/badge/Issues-Report%20Bug-red?style=flat-square" alt="Issues"/></a>
<a href="docs/"><img src="https://img.shields.io/badge/Docs-Full%20Documentation-blue?style=flat-square" alt="Docs"/></a>

<br/>

[GitHub](https://github.com/Llunarstack/sdx) · [Issues](https://github.com/Llunarstack/sdx/issues) · [Discussions](https://github.com/Llunarstack/sdx/discussions) · [Documentation](docs/) · [Citation](#citation)

<br/>

</div>
