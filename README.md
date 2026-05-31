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

# **SDX: Advanced Text-to-Image Generation with Diffusion Transformers**

**A research framework for training & sampling custom image generation models with unprecedented control**

<br/>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.x"/></a>
<a href="docs/releases/v8.md"><img src="https://img.shields.io/badge/release-v8-0ea5e9?style=for-the-badge" alt="v8"/></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-22c55e?style=for-the-badge" alt="Apache 2.0"/></a>

<br/>

[**Quick start**](#quick-start) · [**What is SDX?**](#what-is-sdx) · [**Key features**](#-key-features) · [**Style Genome**](#style-genome-invent-original-looks) · [**Training**](#training) · [**Sampling**](#sampling) · [**Docs**](#documentation)

<br/>

</div>

---

## What is SDX?

**SDX** (Stable Diffusion Transformer eXtended) is a research framework for building and deploying custom text-to-image generation models. Unlike general-purpose diffusion libraries, SDX is purpose-built for:

- **Training custom models** on your own data with advanced objectives (flow matching, DPO, distillation)
- **Precise control** with per-step adapters, LoRA routing, and dynamic CFG scheduling
- **Style invention** — generate novel aesthetics beyond artist imitation using Style Genome
- **Research and experimentation** with transparent, modular code (not node graphs)

<br/>

> **New in [v8](docs/releases/v8.md):** **Style Genome** (invent orthogonal aesthetics, not artist-name clones) · **PromptStack v2** (one pipeline for `sample.py` and training captions) · chaos / fusion / apocalypse modes · native style ops (Rust, CUDA, Go, Mojo) with Python fallbacks.

---

## Key Features

### Style Genome: Invent Novel Aesthetics

Instead of "in the style of Artist X," create structured aesthetic systems with:

- Palette - color theory and dominant hues
- Line - stroke mechanics and contour style
- Surface - texture and material properties
- Camera - perspective and composition angles
- Signature - recurring motifs and visual fingerprints

### PromptStack v2: Transparent Prompt Pipeline

Single, traceable pipeline from user input to model conditioning:

```
Raw prompt → Intelligence → Style genome → Guidance → Negatives → Encoders → DiT
```

Same logic in training and inference—reproducible, debuggable, observable.

### Holy Grail Scheduling: Per-Step Adaptation

Adaptive generation instead of fixed CFG scales:

- Dynamic CFG strength based on noise level
- Conditional LoRA blending per step
- Solver and step count optimization
- Works with flow matching and VP diffusion

### TCIS: Quality Filtering with ViT Committee

For difficult prompts (text in images, exact layouts):

- DiT generates candidate images
- ViT model scores quality and adherence
- Consensus selection and optional refinement

### Advanced Training

Modern training objectives and techniques:

- Flow matching: faster convergence, better latent paths
- Bridge regularization: balanced schedule adherence
- Part-aware attention: hands, faces, objects ground correctly
- DPO and knowledge distillation: learn from feedback and teacher models

---

## What's New in v8

| Feature | v7 | v8 |
|---|:---:|:---:|
| Prompt pipeline | Fragmented | PromptStack v2 (unified, traceable) |
| Style system | Artist tags | Style Genome (5-axis invention) |
| Exploration | Manual prompts | explore_styles CLI + manifests |
| Native operations | Basic | Rust/CUDA/Go/Mojo optimized ops |

Full details: [docs/releases/v8.md](docs/releases/v8.md)  
Previous versions: [v7](docs/releases/v7.md) · [v6](docs/releases/v6.md) · [v5](docs/releases/v5.md)

---

## Game-Changing Capabilities

<table>
<tr>
<td width="33%" valign="top">

### Style Genome

Invent **novel** look systems as structured genomes — compile to pos/neg/style-channel text. Modes from `normal` to `apocalypse`; fusion, hypermutation, and chaos presets.

```bash
python -m scripts.tools explore_styles \
  --prompt "samurai at dusk" --mode chimera --chaos 0.9
```

</td>
<td width="33%" valign="top">

### PromptStack v2

One ordered pipeline: intelligence → genome → guidance → negatives → controls → clauses → filter. **Same guidance stage** in training via `caption_utils`.

```bash
python -m scripts.tools preview_prompt_stack \
  --prompt "portrait, rim light" --json
```

</td>
<td width="33%" valign="top">

### Holy Grail + TCIS

Per-step **CFG / control / adapter** scheduling — not fixed constants. **TCIS** loops DiT proposals through a ViT committee for hard prompts.

```bash
python sample.py ... --holy-grail-preset auto
python -m scripts.tools hybrid_dit_vit_generate ...
```

</td>
</tr>
</table>

---

## System diagram

```mermaid
flowchart LR
  subgraph in[" "]
    PR[Prompt]
  end
  subgraph enc["Encoders"]
    TE[T5 / triple CLIP]
  end
  subgraph core[" "]
    PS[PromptStack v2]
    SG[Style Genome]
    DIT[DiT]
    DE[Diffusion / flow]
    HG[Holy Grail]
    VAE[VAE]
  end
  PR --> PS
  PS --> SG
  SG --> TE
  TE --> DIT --> DE --> HG --> VAE --> IM[Image]
```

<details>
<summary>ASCII fallback (any editor)</summary>

```text
  Prompt → PromptStack v2 → Style Genome? → T5/triple → DiT → diffusion/flow
         → Holy Grail + extras → VAE → image
```

</details>

---

## Why SDX?

A transparent research framework, not a checkpoint fork.

Most diffusion tools fall into categories:
- ComfyUI: node graph workflows (good for inference, hard to reason about)
- diffusers: library for sampling (missing training and style systems)
- Closed-source: proprietary, can't see how they work

**SDX advantages:**

- **Full stack**: data loading → training → inference scheduling
- **Transparent**: train.py is 200 lines, sample.py is 300 lines—read the full pipeline
- **Research-grade**: Flow matching, DPO, TCIS, Holy Grail all integrated
- **Training-first**: not just sampling. Full checkpoint metadata and config snapshots

| Feature | SDX | diffusers | ComfyUI |
|---|:---:|:---:|:---:|
| Training loop | Yes | No | No |
| Flow matching + VP + bridge | Yes | Partial | No |
| Multi-LoRA role routing | Yes | Basic | Plugins |
| Holy Grail adaptive CFG | Yes | No | No |
| Style Genome | Yes | No | No |
| Run reproducibility | Yes | No | No |

---

---

## Use Cases

**Research and Custom Models**

- Train diffusion transformers on your datasets (anime, architecture, products, etc.)
- Compare training objectives (flow matching vs VP diffusion vs bridge loss)
- Export checkpoints with reproducibility metadata

**Style Invention and Curation**

- Generate novel aesthetic systems from prompts
- Rank variants with ViT quality scoring (TCIS)
- Batch-generate images with consistent styles

**Production Sampling**

- Per-step CFG, LoRA, and control scheduling (Holy Grail)
- Quality filtering for difficult prompts (TCIS)
- Adaptive step counts and solver selection

**Custom Data and Training**

- Train on custom datasets (folder or JSONL manifest format)
- Hierarchical captions (global, local, entity-level)
- Part-aware attention for hands, faces, and objects

---

## Quick Start

```bash
# Install & health check
pip install -r requirements.txt
python -m toolkit.training.env_health    # VRAM + CUDA check

# Try sampling with pretrained weights
python demo.py                           # one-command generation
python -m scripts.tools quick_test       # CPU-only smoke test

# Train on your own data
python train.py --data-path datasets/train --results-dir results --flow-matching-training

# Generate from your checkpoint
python sample.py --ckpt results/*/best.pt \
  --prompt "cinematic portrait, dramatic lighting" \
  --holy-grail-preset auto --out out.png
```

---

## Style Genome — invent original looks

A **genome** is an invented aesthetic bundle (not “in the style of Artist X”):

| Axis | Example |
|------|---------|
| palette | oxidized copper, tea-stained paper |
| line | broken contour, dry brush |
| surface | chalk dust, cracked glaze |
| camera | worm's-eye, tilted horizon |
| signature | recurring motif, border bleed |

**Single image with invention:**

```bash
python sample.py --ckpt results/.../best.pt \
  --prompt "lone figure in rain" \
  --invent-styles 1 --style-inventor-mode insane \
  --style-chaos-level 0.8 --out out.png
```

**Explore many genomes → manifest → batch:**

```bash
python -m scripts.tools explore_styles \
  --prompt "void priest in cathedral" --genomes 6 --mode apocalypse

python sample.py --ckpt ... --explore-styles-insane --num 4 --out dir/
```

**Insane shortcut on sample.py:** `--explore-styles-insane` (invents + chaos clauses + multi-candidate pick).

Modules: `utils/prompt/style_genome.py`, `style_inventor.py`, `style_explore.py`, `style_genome_chaos.py` · stack stage: `utils/prompt/stack/stages/style_genome.py`

---

## Training

```bash
python train.py --data-path datasets/train --results-dir results
```

**Flow matching (recommended for new runs):**

```bash
python train.py --data-path datasets/train --flow-matching-training --results-dir results
```

**Triple encoders (T5 + CLIP-L + CLIP-bigG):**

```bash
python train.py --data-path datasets/train --text-encoder-mode triple --results-dir results
```

**Multi-GPU:**

```bash
torchrun --nproc_per_node=2 train.py --data-path datasets/train --results-dir results
```

| Flag | Purpose |
|------|---------|
| `--flow-matching-training` | Rectified-flow objective |
| `--bridge-aux-weight` | Bridge regularization |
| `--use-hierarchical-captions` | Global / local / entity captions |
| `--attn-grounding-loss-weight` | Part-aware attention grounding |
| `--grad-checkpointing` | Lower VRAM (default on) |

`python train.py --help` for the full list · [TRAINING_TEXT_TO_PIXELS.md](docs/TRAINING_TEXT_TO_PIXELS.md)

---

## Sampling

```bash
python sample.py --ckpt results/.../best.pt --prompt "..." \
  --holy-grail-preset auto --cfg-scale 6 --steps 40 --out out.png
```

**Adapters:** `--lora path:scale:role` · **styles:** `--style "anime::0.7 | cinematic::0.3"`

**Flow-trained ckpt:** add `--flow-matching-sample --flow-solver heun`

**Pick-best / beam:** `--num 4 --pick-best auto --pick-vit-ckpt vq/runs/best.pt`

**Hard prompts (TCIS):**

```bash
python -m scripts.tools hybrid_dit_vit_generate \
  --ckpt results/.../best.pt --vit-ckpt vq/runs/best.pt \
  --prompt "poster title NEON STORM, exactly 2 characters" \
  --num 6 --iterations 4 --pick-best combo_hq --out out.png
```

See [Holy Grail README](diffusion/holy_grail/README.md) · [TCIS overview](docs/TCIS_OVERVIEW.md)

---

## Architecture

| Layer | Location |
|-------|----------|
| DiT + text | `models/dit_text.py` |
| Encoders | `utils/modeling/text_encoder_bundle.py` |
| PromptStack | `utils/prompt/stack/` |
| Style Genome | `utils/prompt/style_*.py` |
| Diffusion / flow | `diffusion/` |
| Holy Grail | `diffusion/holy_grail/` |
| Native (optional) | `native/` → [native/README.md](native/README.md) |

```text
datasets/ → train.py → checkpoints/ → sample.py → images
```

---

## Data formats

**Folder mode** — `image.png` + `image.txt` (caption line 1, optional negative line 2).

**JSONL** — one object per line:

```json
{"image_path": "/path/img.png", "caption": "...", "negative_caption": "blurry"}
```

```bash
python train.py --manifest-jsonl data/train.jsonl --results-dir results
```

---

## Repo layout

```text
sdx/
├── train.py · sample.py · demo.py · inference.py
├── config/ · data/ · diffusion/ · models/ · utils/
│   └── utils/prompt/stack/     # PromptStack v2
│   └── utils/prompt/style_*    # Style Genome
├── scripts/tools/              # explore_styles, preview_prompt_stack, …
├── native/                     # Rust · Zig · C · C++ · cuda · Go · Mojo
├── vit_quality/                # ViT quality / TCIS scoring
├── pipelines/book_comic/       # sequential art
└── docs/                       # full index → docs/README.md
```

---

## Documentation

| Doc | Topic |
|-----|--------|
| [docs/README.md](docs/README.md) | Full index |
| [docs/releases/v8.md](docs/releases/v8.md) | **v8 release notes** |
| [docs/PROMPT_STACK.md](docs/PROMPT_STACK.md) | PromptStack v2 stages |
| [docs/HOLY_GRAIL_OVERVIEW.md](docs/HOLY_GRAIL_OVERVIEW.md) | Adaptive sampling |
| [docs/TCIS_OVERVIEW.md](docs/TCIS_OVERVIEW.md) | Hybrid DiT + ViT loop |
| [docs/HOW_GENERATION_WORKS.md](docs/HOW_GENERATION_WORKS.md) | Train → sample walkthrough |
| [native/README.md](native/README.md) | Native build + layout |
| [scripts/tools/README.md](scripts/tools/README.md) | CLI tooling index |

---

## Pretrained weights

```bash
python scripts/download/download_models.py --t5 --vae
python -m scripts.tools pretrained_status
```

Local `pretrained/<name>` overrides Hugging Face hub IDs — see [MODEL_STACK.md](docs/MODEL_STACK.md).

---

## Contributing

```bash
ruff check . && ruff format .
pytest tests/ -m "not cuda and not slow" -q
python -m scripts.tools quick_test
```

[CONTRIBUTING.md](CONTRIBUTING.md) · mirror CI: [docs/recipes/local_ci_mirror.md](docs/recipes/local_ci_mirror.md)

---

## FAQ

**Is this production-ready?**  
Structure and tooling are operator-grade; image quality depends on your data and training budget.

**Need native CUDA/Rust?**  
No — Python fallbacks cover style ops, manifest stats, and pick-best paths.

**v7 vs v8?**  
v7 = CI + eval + security baseline. v8 = **Style Genome + PromptStack v2 + native style layer** on top of that baseline.

---

## Acknowledgements

Built on ideas from [DiT](https://github.com/facebookresearch/DiT), [ControlNet](https://github.com/lllyasviel/ControlNet), [FLUX](https://github.com/black-forest-labs/flux), and the broader diffusion research community — [INSPIRATION.md](docs/INSPIRATION.md).

## License

Apache 2.0 — [LICENSE](LICENSE)
