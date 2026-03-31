<!-- markdownlint-disable MD033 MD041 -->

<div align="center">

# SDX

### Text-to-image diffusion transformers — research depth, operator clarity

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.x"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-3DDC84?style=flat-square" alt="Apache 2.0"/></a>
  <a href="docs/README.md"><img src="https://img.shields.io/badge/Docs-docs%2FREADME-24292f?style=flat-square&logo=github" alt="Docs"/></a>
</p>

<p align="center">
  <a href="#quick-start"><strong>Quick start</strong></a> ·
  <a href="#training">Training</a> ·
  <a href="#sampling">Sampling</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#data-formats">Data formats</a> ·
  <a href="#key-docs">Docs</a>
</p>

</div>

---

SDX is a modular text-to-image training and inference framework built on Diffusion Transformers (DiT). It is designed to be transparent — training lives in `train.py`, sampling in `sample.py`, and every module boundary is explicit.

**Core stack:** DiT · T5 / triple text encoders · LoRA/DoRA/LyCORIS routing · VP diffusion + flow matching + bridge/OT objectives · Holy Grail adaptive sampling · optional native CUDA acceleration.

---

## What SDX gives you

| Capability | Details |
| :--- | :--- |
| **Core model** | DiT with text conditioning via `models/dit_text.py` |
| **Text encoders** | T5-XXL (default) or triple mode: T5 + CLIP-ViT-L/14 + CLIP-ViT-bigG/14 |
| **Training objectives** | VP diffusion · flow matching · bridge auxiliary · OT noise-latent coupling |
| **Prompt adherence** | Part-aware attention grounding + token coverage auxiliary losses |
| **Adapter system** | Multi-LoRA/DoRA/LyCORIS stacking with per-role budgets and depth routing |
| **Adaptive sampling** | Holy Grail: per-step CFG/control/adapter scheduling + CADS condition annealing |
| **Inference controls** | CFG schedulers · speculative CFG · SAG · reference-token injection · img2img |
| **Reproducibility** | Per-run `run_manifest.json` + `config.train.json` snapshots |
| **Performance** | bf16 · `torch.compile` · gradient checkpointing · DDP-ready |

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Verify your environment
python -m toolkit.training.env_health

# Train on your dataset
python train.py --data-path datasets/train --results-dir results

# Sample from a trained checkpoint
python sample.py \
  --ckpt results/000-DiT-XL-2-Text/checkpoints/best.pt \
  --prompt "cinematic portrait, dramatic lighting" \
  --out out.png
```

Optional: refresh to CUDA 12.8 wheels:

```bash
pip install --force-reinstall -r requirements-cuda128.txt
```

---

## Training

### Minimal run

```bash
python train.py \
  --data-path datasets/train \
  --model DiT-XL/2-Text \
  --results-dir results
```

### Common training recipes

**Flow matching** (recommended for new checkpoints):

```bash
python train.py \
  --data-path datasets/train \
  --flow-matching-training \
  --results-dir results
```

**Triple text encoders** (T5 + CLIP-L + CLIP-bigG):

```bash
python train.py \
  --data-path datasets/train \
  --text-encoder-mode triple \
  --results-dir results
```

**Part-aware training** with attention grounding:

```bash
python train.py \
  --data-path datasets/train \
  --use-hierarchical-captions \
  --attn-grounding-loss-weight 0.1 \
  --attn-token-coverage-loss-weight 0.05 \
  --results-dir results
```

**Resume from checkpoint:**

```bash
python train.py \
  --data-path datasets/train \
  --resume results/000-DiT-XL-2-Text/checkpoints/best.pt \
  --results-dir results
```

### Key training flags

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--model` | `DiT-XL/2-Text` | Model variant |
| `--text-encoder-mode` | `t5` | `t5` or `triple` |
| `--flow-matching-training` | off | Use flow objective instead of VP |
| `--bridge-aux-weight` | `0.0` | Bridge regularization weight |
| `--ot-noise-pair-reg` | `0.0` | OT noise-latent coupling |
| `--use-hierarchical-captions` | off | Compose global/local/entity captions |
| `--attn-grounding-loss-weight` | `0.0` | Foreground attention grounding loss |
| `--attn-token-coverage-loss-weight` | `0.0` | Token coverage auxiliary loss |
| `--val-split` | `0.0` | Fraction of data held out for validation |
| `--early-stopping-patience` | `0` | Stop after N val checks with no improvement |
| `--grad-checkpointing` | on | Reduce VRAM at cost of speed |
| `--use-bf16` | on | Mixed precision training |
| `--strict-warnings` | off | Escalate project warnings to errors |

Run `python train.py --help` for the full list.

---

## Sampling

### Basic usage

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "your prompt here" \
  --out out.png
```

### With adapters and style blending

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "hero character, dynamic pose, city at night" \
  --lora char.safetensors:0.9:character style.safetensors:0.6:style \
  --lora-stage-policy auto \
  --cfg-scale 6.0 \
  --steps 40 \
  --out out.png
```

Adapter format: `path:scale:role` where role is one of `character`, `style`, `detail`, `composition`, `other`.

Style blending: `--style "anime::0.7 | cinematic::0.3"` — weights are normalised automatically.

### Holy Grail adaptive sampling

Holy Grail applies per-step CFG, control, and adapter scheduling automatically. Enable with a preset:

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "photoreal portrait, soft window light" \
  --holy-grail-preset auto \
  --out out.png
```

Available presets: `auto` · `balanced` · `photoreal` · `anime` · `illustration` · `aggressive`

`auto` picks a preset heuristically from your prompt, style, and LoRA roles.

Fine-grained control:

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "..." \
  --holy-grail \
  --holy-grail-cfg-early-ratio 0.72 \
  --holy-grail-cfg-late-ratio 1.0 \
  --holy-grail-cads-strength 0.03 \
  --holy-grail-unsharp-sigma 0.6 \
  --holy-grail-unsharp-amount 0.18 \
  --out out.png
```

See [`diffusion/holy_grail/README.md`](diffusion/holy_grail/README.md) for the full flag reference.

### Flow-trained checkpoint

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "cinematic sci-fi alley, volumetric fog" \
  --flow-matching-sample \
  --flow-solver heun \
  --cfg-scale 6.0 \
  --steps 32 \
  --out out.png
```

### Key sampling flags

| Flag | Purpose |
| :--- | :--- |
| `--cfg-scale` | Classifier-free guidance scale (default: 7.5) |
| `--steps` | Number of denoising steps |
| `--scheduler` | Timestep schedule: `ddim`, `euler`, `karras_rho`, etc. |
| `--solver` | Solver: `ddim` or `heun` |
| `--negative-prompt` | Negative conditioning text |
| `--lora` | One or more adapters as `path:scale:role` |
| `--lora-stage-policy` | Depth routing: `auto`, `character_focus`, `style_focus`, `balanced` |
| `--flow-matching-sample` | Use flow-matching sampler (for flow-trained checkpoints) |
| `--holy-grail-preset` | Adaptive sampling preset |
| `--width` / `--height` | Output resolution (default: model native) |
| `--seed` | Random seed for reproducibility |

Run `python sample.py --help` for the full list.

---

## Architecture

### Pipeline overview

```
datasets/train/  ──►  train.py  ──►  checkpoint/  ──►  sample.py  ──►  images
                       │                                    │
                  DiT + diffusion                    CFG + scheduler
                  objectives                         + adapters
```

### Model stack

| Component | Location | Notes |
| :--- | :--- | :--- |
| DiT core | `models/dit_text.py` | Patch embed → DiT blocks → head |
| Text encoders | `utils/modeling/text_encoder_bundle.py` | T5-XXL or T5 + CLIP-L + CLIP-bigG |
| Adapter routing | `models/lora.py` | LoRA/DoRA/LyCORIS, per-role budgets |
| ControlNet | `models/controlnet.py` | Optional image conditioning |
| Diffusion engine | `diffusion/gaussian_diffusion.py` | VP + flow + Holy Grail wiring |
| Flow matching | `diffusion/flow_matching.py` | Rectified-flow objective |
| Holy Grail | `diffusion/holy_grail/` | Adaptive per-step scheduling |
| VAE / RAE | via `diffusers` | Latent encode/decode |

### Training objectives

SDX supports composable training objectives:

- **VP diffusion** — standard epsilon/v/x0 prediction with min-SNR weighting
- **Flow matching** — rectified-flow velocity prediction (`--flow-matching-training`)
- **Bridge auxiliary** — VP bridge regularisation on shuffled latent pairs (`--bridge-aux-weight`)
- **OT coupling** — optimal-transport noise-latent pairing (`--ot-noise-pair-reg`)
- **Part-aware losses** — attention grounding + token coverage (`--attn-grounding-loss-weight`)
- **REPA** — representation alignment with a frozen DINOv2/CLIP encoder (`--repa-weight`)

---

## Data formats

### Folder mode

```text
datasets/train/
  subject_a/
    img_001.png
    img_001.txt      ← caption on line 1, optional negative on line 2
  subject_b/
    img_002.jpg
    img_002.txt
```

```bash
python train.py --data-path datasets/train --results-dir results
```

### JSONL manifest mode

One JSON object per line:

```json
{"image_path": "/abs/path/img.png", "caption": "your caption here"}
```

Extended fields (all optional):

```json
{
  "image_path": "/abs/path/img.png",
  "caption": "main caption",
  "negative_caption": "blurry, low quality",
  "caption_global": "scene description",
  "caption_local": "subject detail",
  "style": "anime",
  "grounding_mask": "relative/path/mask.png",
  "weight": 1.5
}
```

```bash
python train.py --manifest-jsonl /path/to/manifest.jsonl --results-dir results
```

---

## Pretrained weights

Download T5, VAE, and optional LLMs into `pretrained/`:

```bash
# Minimal recommended set (T5-XXL + VAE)
python scripts/download/download_models.py --t5 --vae

# Everything including CLIP and LLM
python scripts/download/download_models.py --all

# Download a specific LLM for prompt expansion
python scripts/download/download_llm.py --best
```

Weights are resolved automatically at runtime via `utils/modeling/model_paths.py`. If a local `pretrained/<name>` folder exists and is non-empty, it is used; otherwise the Hugging Face hub ID is used as fallback.

| Model | Local path | HF fallback |
| :--- | :--- | :--- |
| T5-XXL | `pretrained/T5-XXL` | `google/t5-v1_1-xxl` |
| CLIP ViT-L/14 | `pretrained/CLIP-ViT-L-14` | `openai/clip-vit-large-patch14` |
| CLIP ViT-bigG/14 | `pretrained/CLIP-ViT-bigG-14` | `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` |
| DINOv2-Large | `pretrained/DINOv2-Large` | `facebook/dinov2-large` |
| VAE (default) | `pretrained/sd-vae-ft-mse` | `stabilityai/sd-vae-ft-mse` |

---

## Repo layout

```text
sdx/
├── train.py                  # Main training entry point
├── sample.py                 # Main inference entry point
├── inference.py              # Lightweight programmatic inference API
│
├── config/                   # TrainConfig dataclass + model build kwargs + defaults
├── data/                     # Dataset loaders, caption processing, bucket sampler
├── diffusion/                # Diffusion engine, schedules, losses, Holy Grail
│   └── holy_grail/           # Adaptive per-step CFG/control/adapter scheduling
├── models/                   # DiT core, attention, adapters, ControlNet, MoE
├── training/                 # CLI parser + config mapping (split from train loop)
├── utils/                    # Training, generation, prompt, checkpoint, modeling utils
│
├── scripts/                  # Download, setup, and tooling scripts
├── native/                   # Optional C++/CUDA/Rust acceleration
├── ViT/                      # Vision Transformer standalone training module
├── pipelines/                # High-level generation pipelines (book/comic, etc.)
├── toolkit/                  # QoL helpers: env health, seeds, timing, manifest digest
│
├── pretrained/               # Downloaded model weights (gitignored)
├── datasets/                 # Your training data (gitignored)
├── results/                  # Training run outputs (gitignored)
├── docs/                     # All documentation
└── external/                 # Reference repos (DiT, ControlNet, Flux, etc.)
```

---

## Key docs

| Document | What it covers |
| :--- | :--- |
| [`docs/README.md`](docs/README.md) | Full documentation index |
| [`docs/CODEBASE.md`](docs/CODEBASE.md) | Where things live and why |
| [`docs/HOW_GENERATION_WORKS.md`](docs/HOW_GENERATION_WORKS.md) | End-to-end train → checkpoint → sample walkthrough |
| [`docs/PROMPT_STACK.md`](docs/PROMPT_STACK.md) | Prompt assembly, controls, and filtering |
| [`docs/MODEL_STACK.md`](docs/MODEL_STACK.md) | Model weights, roles, and download paths |
| [`docs/DIFFUSION_LEVERAGE_ROADMAP.md`](docs/DIFFUSION_LEVERAGE_ROADMAP.md) | High-impact quality priorities |
| [`docs/MODEL_WEAKNESSES.md`](docs/MODEL_WEAKNESSES.md) | Known failure modes and mitigations |
| [`docs/QUALITY_AND_ISSUES.md`](docs/QUALITY_AND_ISSUES.md) | Practical quality playbook |
| [`diffusion/holy_grail/README.md`](diffusion/holy_grail/README.md) | Holy Grail adaptive sampling reference |

---

## Contributing

Small, focused PRs are preferred. Docs, tooling, and quality improvements are all welcome.

```bash
# Lint before submitting
ruff check .
ruff format .
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide.

Good first contributions:
- Tighten docs for one module (`data/`, `diffusion/`, `models/`, `utils/`)
- Add a dry-run test for a new flag in `sample.py` or `train.py`
- Improve a CLI help string
- Profile and document one inference or training path

---

## FAQ

**Is SDX production-ready?**
The codebase is production-oriented in structure and tooling. Output quality depends on your data, training budget, and checkpoint.

**Do I need the native CUDA modules?**
No. All core train/sample paths work without them. Native modules improve specific performance paths (RMSNorm, RoPE, SiLU-gate).

**Is this only for anime/illustration?**
No. The stack is domain-agnostic. Anime/manga is one well-tested path, but the system supports photoreal, illustration, and other domains with appropriate data.

**What is Holy Grail?**
An adaptive sampling system that schedules CFG scale, ControlNet strength, and adapter multipliers per denoising step — rather than keeping them fixed. See [`diffusion/holy_grail/README.md`](diffusion/holy_grail/README.md).

---

## Acknowledgements

SDX builds on ideas from the broader diffusion research community.

- [facebookresearch/DiT](https://github.com/facebookresearch/DiT)
- [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)
- [black-forest-labs/flux](https://github.com/black-forest-labs/flux)
- [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)

See [`docs/INSPIRATION.md`](docs/INSPIRATION.md) for extended context.

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
