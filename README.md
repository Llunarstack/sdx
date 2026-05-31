<!-- markdownlint-disable MD033 MD041 -->

<div align="center">

<br/>

```
   ███████╗██████╗ ██╗  ██╗
   ██╔════╝██╔══██╗╚██╗██╔╝
   ███████╗██║  ██║ ╚███╔╝ 
   ╚════██║██║  ██║ ██╔██╗ 
   ███████║██████╔╝██╔╝ ██╗
   ╚══════╝╚═════╝ ╚═╝  ╚═╝
```

# **SDX: Advanced Text-to-Image Generation Framework**

**Train and deploy custom image generation models with unprecedented control and transparency**

<br/>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.x"/></a>
<a href="docs/releases/v9.md"><img src="https://img.shields.io/badge/release-v9.0.0-0ea5e9?style=for-the-badge" alt="v9.0.0"/></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-22c55e?style=for-the-badge" alt="Apache 2.0"/></a>

<br/>

[**Quick Start**](#quick-start) · [**Features**](#core-features) · [**Use Cases**](#use-cases) · [**Documentation**](#documentation) · [**Contributing**](#contributing)

<br/>

</div>

---

## What is SDX?

**SDX** is a production-grade research framework for building custom text-to-image generation models. Unlike general diffusion libraries, SDX provides:

- **Complete training pipelines** with flow matching, DPO, and knowledge distillation
- **Advanced inference optimization** with model ensembles, quality filtering, and adaptive scheduling
- **Novel style invention** that creates original aesthetics instead of copying artists
- **Transparent, readable code** where you understand exactly what's happening
- **Research integration** connecting academic papers to working implementations

Built for researchers, practitioners, and teams who want full control over their image generation systems.

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

# Train a model
python train.py \
  --data-path images/ \
  --results-dir outputs/ \
  --flow-matching-training

# Generate from your trained model
python sample.py \
  --ckpt outputs/best.pt \
  --prompt "your description" \
  --out result.png
```

---

## Core Features

### 1. Style Genome: Invent Original Aesthetics

Stop copying artists. Create structured, original visual styles:

```bash
# Generate a unique style for your prompt
python sample.py --ckpt model.pt \
  --prompt "warrior at sunset" \
  --invent-styles 1 \
  --style-chaos-level 0.8

# Explore multiple style variations
python -m scripts.tools explore_styles \
  --prompt "forest landscape" \
  --genomes 10 --mode apocalypse
```

**What gets invented:** palette, line, surface, camera angle, visual signature

### 2. Advanced Training Methods

Choose from multiple training objectives:

```bash
# Flow matching (faster, recommended for new projects)
python train.py --data-path images/ --flow-matching-training

# Learn from feedback (DPO)
python scripts/tools/training/train_diffusion_dpo \
  --data-path preferences/ \
  --reference-ckpt baseline.pt

# Shrink your model (knowledge distillation)
python scripts/tools/training/train_kd_distill \
  --student-ckpt small.pt --teacher-ckpt large.pt

# New in v9: 6 GRPO variants for adaptive training
python scripts/tools/training/train_flow_grpo \
  --data-path images/ --ckpt-init base.pt
```

### 3. Intelligent Inference

Generate better images with adaptive scheduling:

```bash
# Single image generation
python sample.py --ckpt model.pt --prompt "..." --out result.png

# Generate and pick the best (quality filtered)
python sample.py --ckpt model.pt \
  --prompt "..." --num 4 --pick-best auto

# For hard prompts (text, exact layouts)
python -m scripts.tools hybrid_dit_vit_generate \
  --ckpt model.pt --vit-ckpt quality_model.pt \
  --prompt "poster with text HELLO WORLD" \
  --iterations 6 --constraint-anneal up

# Model ensemble (blend multiple models)
python sample.py --ckpt-ensemble model1.pt:model2.pt:model3.pt \
  --prompt "..."
```

### 4. Reproducibility & Evaluation

All training is reproducible:

```bash
# Training saves configuration
# Later, reproduce exactly:
python train.py --config results/config.train.json

# Evaluate against baselines
python examples/run_baseline_eval.py \
  --ckpt results/best.pt \
  --execute --output scores.json

# Compare multiple models
python -m scripts.tools benchmark_suite \
  --ckpt model1.pt model2.pt \
  --seed-list 42,123,999 \
  --robustness-penalty 0.15
```

### 5. New in v9: Production Stacks

**Superior Stack** — Inference optimization and ensembles
- Model soup (average multiple checkpoints)
- Quality gates and filtering
- Feature caching (2-3x speedup)
- Reward-based scoring

**Agentic Stack** — Self-improving training
- Agent-driven optimization loops
- Planning and reflection
- Experience memory and replay

**Visual Brain** — Image understanding
- Scene analysis and composition
- Image similarity search
- Automatic caption generation

---

## Use Cases

### For Researchers
- Compare training objectives (flow matching vs VP diffusion)
- Implement new conditioning mechanisms
- Study prompt adherence and quality trade-offs
- Reproduce papers in a modern, readable codebase

### For Practitioners
- Train models on custom datasets
- Create original visual styles (not artist imitation)
- Deploy with adaptive scheduling
- Measure and improve quality systematically

### For Teams
- Version control training runs with metadata
- Evaluate models fairly across seeds
- Build production pipelines with quality gates
- Share transparent, auditable training processes

### Example Projects
```bash
# Create an anime character generator
python train.py --data-path anime_dataset/ \
  --text-encoder-mode triple \
  --flow-matching-training

# Fine-tune on specialized domain
python train.py --data-path architecture_photos/ \
  --init-ckpt pretrained.pt \
  --learning-rate 1e-5

# Optimize for consistency
python -m scripts.tools auto_improve_loop \
  --base-ckpt model.pt \
  --iterations 3 \
  --promote-best
```

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11+ |
| PyTorch | 2.0 | 2.11+ |
| GPU VRAM | 16GB | 24GB+ |
| Disk Space | 50GB | 200GB+ |
| OS | Linux/Windows/Mac | Linux preferred |

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

### Optional: Native Acceleration

For 3-5x faster inference:

```bash
cd native/cpp && cmake build
cd ../rust/sdx-prompt-ops && cargo build --release
```

### Optional: GPU-Specific (NVIDIA)

```bash
pip install --force-reinstall -r requirements-cuda128.txt
```

---

## Architecture

```
Your Data
    ↓
train.py ──→ Model Checkpoint
    ↓
sample.py ──→ Generated Images
```

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `train.py` | Main training loop (200 lines) |
| `sample.py` | Generation with Holy Grail scheduling (300 lines) |
| `models/` | DiT architecture and conditioning |
| `diffusion/` | Flow matching, VP diffusion, scheduling |
| `utils/` | Organized into specialized packages |
| `diffusion/sampling_extras/` | Holy Grail, TCIS, quality gates |
| `native/` | Optional C++/CUDA/Rust acceleration |

Everything is readable. No hidden magic.

---

## Documentation

### Getting Started
- **[GETTING_STARTED.md](docs/GETTING_STARTED.md)** — Complete beginner guide
- **[Quick Start (above)](#quick-start)** — 30 seconds to first image

### Learning the Framework
- **[CODEBASE_GUIDE.md](docs/CODEBASE_GUIDE.md)** — How everything is organized
- **[MODEL_STACK.md](docs/MODEL_STACK.md)** — Available models and encoders
- **[PROMPT_STACK.md](docs/PROMPT_STACK.md)** — Text conditioning pipeline

### Advanced Topics
- **[SUPERIOR_STACK.md](docs/SUPERIOR_STACK.md)** — Inference optimization
- **[AGENTIC_STACK.md](docs/agentic/AGENTIC_STACK.md)** — Autonomous training
- **[VISUAL_BRAIN.md](docs/brain/VISUAL_BRAIN.md)** — Image understanding
- **[TRAINING_TEXT_TO_PIXELS.md](docs/TRAINING_TEXT_TO_PIXELS.md)** — Text encoding details
- **[HOLY_GRAIL_OVERVIEW.md](docs/HOLY_GRAIL_OVERVIEW.md)** — Adaptive scheduling

### Research & References
- **[Release Notes](docs/releases/)** — What's new in each version (v9, v8, v7, ...)
- **[IMAGE_QUALITY_LEVERS_2026.md](docs/research/IMAGE_QUALITY_LEVERS_2026.md)** — Papers → implementation
- **[LANDSCAPE_2026.md](docs/LANDSCAPE_2026.md)** — Industry context and architecture decisions
- **[BLUEPRINTS.md](docs/BLUEPRINTS.md)** — Flow matching, distillation, advanced methods

---

## Version History

| Version | Release | Key Features |
|---------|---------|--------------|
| **v9** | May 2026 | GRPO training, Superior/Agentic/Brain stacks, full documentation |
| **v8** | May 2026 | Style Genome, PromptStack v2, native ops |
| **v7** | April 2026 | CI/CD, reproducibility, evaluation framework |
| **v6** | April 2026 | Native acceleration, book generation, ViT quality |
| **v5** | April 2026 | Inference scaling, beam search, data curation |
| **v0.2** | March 2026 | Flow matching, DPO, KD, modular organization |
| **v0.1** | March 2026 | Core framework: train.py, sample.py |

Full release notes: [docs/releases/](docs/releases/)

---

## Contributing

We welcome contributions! Here's how:

1. **Check the codebase:** [CODEBASE.md](docs/CODEBASE.md) covers style, structure, and testing
2. **Run tests:** `pytest tests/ -v`
3. **Format code:** `ruff check . --fix && ruff format .`
4. **Submit PR** with a clear description of what changed and why

---

## Support & Community

- **Documentation:** See [Documentation](#documentation) above
- **Issues:** [GitHub Issues](https://github.com/Llunarstack/sdx/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Llunarstack/sdx/discussions)
- **Security:** See [SECURITY.md](SECURITY.md) to report vulnerabilities

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

You're free to use SDX for commercial and personal projects.

---

<div align="center">

Made with care for researchers and practitioners  
[GitHub](https://github.com/Llunarstack/sdx) · [Issues](https://github.com/Llunarstack/sdx/issues) · [Documentation](docs/)

</div>
