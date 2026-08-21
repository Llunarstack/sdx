<p align="center">
  <strong>SDX</strong> · Stable Diffusion Transformer eXtended<br/>
  <sub>Train, layout, direct, and deploy your own image &amp; video models — fully open.</sub>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="docs/releases/v12.md"><img src="https://img.shields.io/badge/release-v12.0.0-0ea5e9?style=flat-square" alt="v12"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-22c55e?style=flat-square" alt="License"/></a>
  <img src="https://img.shields.io/badge/tests-1300%2B-22c55e?style=flat-square" alt="Tests"/>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#why-sdx">Why SDX</a> ·
  <a href="#how-sdx-compares">Compare</a> ·
  <a href="#what-you-get">Features</a> ·
  <a href="#pipelines">Pipelines</a> ·
  <a href="#new-in-v12">v12</a> ·
  <a href="#documentation">Docs</a> ·
  <a href="#glossary">Glossary</a>
</p>

---

## What is SDX?

**SDX** is an open research framework for building **your own** text-to-image and text/image-to-video systems. It is not a hosted model, not an API wrapper, and not a scoreboard of someone else's weights.

You own the training loop, the sampling stack, the layout controls, the video director, and the quality critics — end to end, on your hardware.

```
your data  →  train.py  →  checkpoint  →  sample.py / video studio  →  images & clips
```

---

## Why SDX

Most products give you **pixels**. SDX gives you the **pipeline**.

| You need… | Typical path | With SDX |
|-----------|--------------|----------|
| Model on *your* data | Fine-tune scripts + glue | One `train.py` path (flow, DPO, GRPO) |
| Layout you can inspect | Vendor UI or plugin zoo | Regional box JSON + scene graphs |
| Quality that retries | Single-shot generate | TCIS critic loop + pick-best |
| Video you can direct | Closed APIs / black boxes | Open scene JSON → plan → stitch |
| Reproducible science | Varies by repo | Metadata, provenance, **1300+** tests |

If you want a pretty demo button, use a hosted API. If you want to **train, measure, and ship** your own system, use SDX.

---

## Quick start

```bash
git clone https://github.com/Llunarstack/sdx.git && cd sdx
pip install -r requirements.txt

# Image (demo checkpoint)
python demo.py

# Train on your folder
python train.py --data-path images/ --flow-matching-training --epochs 20

# Sample
python sample.py --ckpt outputs/best.pt --prompt "your prompt" --out result.png

# Video — plan from one scene file (v12)
python -m scripts.tools video_generate --scene examples/scene_frontier.example.json --plan-only
```

**Health check:** `python -m toolkit.training.env_health` · **Tests:** `pytest tests/ -q`

---

## How SDX compares

SDX is a **framework you train** — not a hosted model scoreboard. Leaders below are compared on *capability classes* (landscape snapshot: mid‑2026). Names and versions move weekly; the structural gaps usually do not.

Legend: **✓** strong / built-in · **◐** partial / LoRA-only / plugin · **ext** community extensions · **✗** not available · **n/a** wrong product shape

Full research tables + sources → [docs/COMPETITIVE_ANALYSIS.md](docs/COMPETITIVE_ANALYSIS.md)

### Landscape: what the leaders are actually good at

#### Closed image APIs

| System | Best known for | Weak / missing |
|--------|----------------|----------------|
| **GPT Image 2** (OpenAI) | Prompt following, editing loops, text-in-image, API production | Closed; no train-your-weights; house look; no local air-gap |
| **Midjourney V8** | Aesthetic craft, lighting, cinematic / concept art | Weak automation API; limited literal control; no self-host; no your-data train |
| **Gemini / Nano Banana** (Google) | Fast multimodal edit, references, 4K-class Pro path | Closed; ops locked to Google stack |
| **Ideogram 4** | Typography / posters / layout; **open weights** on latest line | Still not a full research train stack; video not the product |
| **Recraft V4** | Brand / design graphics, vectors, commercial layout | Closed design tool, not a trainable DiT lab |
| **Firefly Image 5** | Adobe Creative Cloud fit, commercial licensing story | Ecosystem lock-in; not a research pipeline |
| **Seedream 5** (ByteDance) | Cinematic stills, semantic prompts | API-first; limited ownership |

#### Open-weight image models

| System | Best known for | Weak / missing |
|--------|----------------|----------------|
| **FLUX.2** (BFL) | Photoreal quality bar; LoRA on klein; partner APIs | Training = mostly LoRA/adapters; licenses vary (dev vs klein); no in-tree critic/video director |
| **SD 3.5** | Deepest LoRA / ControlNet / Comfy ecosystem | Needs tooling glue; preference RL not first-class |
| **Qwen-Image** | Multilingual / CJK text-in-image; Apache-friendly | Newer ecosystem vs SD; not a full studio |
| **Z-Image Turbo** | Speed / volume on mid GPUs | Ecosystem thinner; quality vs frontier tradeoff |
| **SDXL** (legacy) | Ubiquitous fine-tunes, tiny GPUs | Older U-Net quality ceiling vs DiT/flow leaders |
| **ComfyUI / A1111** | Node / UI orchestration over *any* weights | You assemble the science; no opinionated train→critique→video stack |

#### Video (closed + open)

| System | Best known for | Weak / missing |
|--------|----------------|----------------|
| **Veo 3.1** (Google) | Overall fidelity + native audio | Closed; watermark / cloud; no open director graph |
| **Kling 3** | Motion, multi-shot value, 4K-class paths | Closed API; data residency / ToS constraints |
| **Runway Gen-4.5** | Motion brush, editor, character control | Hosted product; not a forkable training framework |
| **Luma Ray / Seedance** | I2V atmosphere / multi-shot narrative | Closed; limited reproducibility |
| **Wan / LTX** (open-ish) | Self-hostable video weights emerging | Still model-centric; little open *scene-graph film studio* |

### Capability matrix (image)

| Capability | GPT Image | Midjourney | Ideogram | FLUX.2 | SD3.5 | Qwen-Image | Comfy | **SDX** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Best-in-class hosted aesthetics | ✓ | **✓** | ◐ | ✓ | ◐ | ◐ | n/a | n/a¹ |
| Text-in-image | **✓** | ◐ | **✓** | ✓ | ◐ | **✓** | ext | ◐² |
| Conversational / API edit | **✓** | ◐ | ◐ | ✓ | ext | ✓ | ext | ◐ |
| Download / run weights | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| LoRA / adapter fine-tune | ✗ | ✗ | ✓ | ✓ | **✓** | ◐ | **✓** | **✓** |
| Full train loop on *your* folder | ✗ | ✗ | ✗ | ◐ | ◐ | ◐ | ext | **✓** |
| Flow matching in-tree | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ext | **✓** |
| DPO / GRPO preference train | ✗ | ✗ | ✗ | ✗ | ext | ✗ | ext | **✓** |
| Regional box layout (native) | ◐ | ◐ | ✓ | ext | ext | ext | ext | **✓** |
| Self-critique + retry (TCIS / pick-best) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ | **✓** |
| Adaptive CFG stack (Holy Grail) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ext | **✓** |
| Style invention (Style Genome) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Open scene-graph video director | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ext | **✓** |
| Continuity / filmmaker modules | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Air-gap + reproducible metadata | ✗ | ✗ | ◐ | ◐ | ◐ | ◐ | ◐ | **✓** |
| Research horizon registry | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |

¹ SDX is not a published aesthetics leaderboard checkpoint — you train (or bring) weights.  
² Glyph canvas + OCR pick-best help; closed typography specialists still lead raw OCR demos.

### Capability matrix (video)

| Capability | Veo | Kling | Runway | Luma | Wan / LTX | **SDX** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Hosted clip quality / audio | **✓** | **✓** | ✓ | ✓ | ◐ | n/a¹ |
| Motion brush / editor UX | ◐ | ◐ | **✓** | ◐ | ✗ | ◐ |
| Multi-shot storyboard | ✓ | **✓** | ✓ | ◐ | ◐ | **✓** |
| Open scene JSON → plan → stitch | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Identity / continuity validators | ◐ | ◐ | ✓ | ◐ | ✗ | **✓** |
| Segment critic + retry | ✗ | ✗ | ◐ | ✗ | ✗ | **✓** |
| Trainable / forkable pipeline | ✗ | ✗ | ✗ | ✗ | ◐ | **✓** |
| Self-host entire director stack | ✗ | ✗ | ✗ | ✗ | ◐ | **✓** |

¹ Default SDX video path is an open **compositional studio** (retrieve → keyframe → motion → polish). Optional VideoDiT exists; frontier clip quality still belongs to hosted Veo/Kling/Runway unless you train.

### What SDX can do that *they* generally cannot

These are structural — hard for a closed API or a weights-only release to match without becoming a different product:

| SDX edge | Why others miss it |
|----------|--------------------|
| **End-to-end `train.py` on your data** (flow + VP + LoRA/DoRA) | Closed APIs forbid it; Flux/SD mostly ship **LoRA recipes**, not one readable research loop |
| **In-tree DPO + GRPO family** | Preference / RL fine-tune lives as scattered papers + third-party scripts elsewhere |
| **TCIS closed loop** — generate → multi-scorer critique → disagree → retry | Hosted models are almost always **single-shot**; Comfy can approximate with hand-built graphs |
| **Holy Grail adaptive sampling** as first-class presets | CFG schedules exist as lore/nodes, not an owned stack with tests |
| **Style Genome** — invent style axes without artist-name cloning | Competitors lock “house style” or rely on artist LoRAs |
| **Regional boxes + prompt→layout + glyph canvas** in one repo | Ideogram-class layout is closed or plugin-glued |
| **Open scene-graph film studio** (JSON director, continuity, frontier filmmakers) | Veo/Kling/Runway are products; Wan/LTX are weights — neither is an open director OS |
| **`frontier/` idea registry** — cinema / narrative / realism experiments | No major vendor ships a browsable research horizon next to production entry points |
| **Foreign adapter bridge** (sniff LoRA/LyCORIS → DiT roles) | Cross-arch weight reuse is usually “start over in Comfy” |
| **Air-gap + 1300+ tests + provenance metadata** | SaaS cannot be air-gapped; many FOSS UIs skip scientific reproducibility |

### Where they still win (be honest)

| They win | Prefer |
|----------|--------|
| Zero-ops pretty defaults | Midjourney, GPT Image, Veo |
| Absolute text-in-image demos | Ideogram, GPT Image, Qwen-Image |
| Deepest node ecosystem today | ComfyUI + SD/FLUX weights |
| Hosted video fidelity + native audio | Veo, Kling, Runway |
| Adobe / enterprise creative suite | Firefly |

Use those when you need **pixels tomorrow**. Use SDX when you need to **own the loop**: data → train → sample → critique → direct.

### What this is *not*

| Claim we avoid | Reality |
|----------------|---------|
| “Beats Midjourney on aesthetics” | No bake-off claimed; taste is subjective and APIs churn |
| “Drop-in Flux / Veo replacement” | Different shape: framework + training + director, not one mega-checkpoint |
| “One demo ckpt does everything” | You train or bring weights; SDX supplies the stack |

Deeper strategy + weakness→fix map: [docs/COMPETITIVE_ANALYSIS.md](docs/COMPETITIVE_ANALYSIS.md) · field notes: [docs/research/LANDSCAPE_2026.md](docs/research/LANDSCAPE_2026.md)

---

## What you get

<details open>
<summary><strong>Image generation</strong> — train.py + sample.py</summary>

- DiT + VAE latent diffusion, flow matching, DPO, GRPO family
- Holy Grail adaptive CFG, TCIS committee scoring, Style Genome
- Regional box prompting (layout JSON with per-region prompts)
- Agentic quality: pick-best / critics, artifacts, drift repair, eval harness

</details>

<details open>
<summary><strong>Video studio (v12)</strong> — pipelines/video/</summary>

One **scene JSON** → retrieve → keyframe edit → motion → polish → stitch.

- **Studio:** engine router (realistic / anime / voxel / …), director mode, character & world bibles
- **Controls:** elements, motion brush, FLF2V, storyboard cuts
- **Continuity:** eyeline, props, light motivation, thumbnail rehearsal
- **Frontier:** 25 filmmaker modules (tension curve, causal ripples, witness lens, …)

```bash
python -m scripts.tools video_generate --list-frontier
python -m scripts.tools video_generate --scene examples/scene_studio.example.json --preflight
```

</details>

<details>
<summary><strong>Frontier research</strong> — frontier/</summary>

80+ experimental modules: layout, guidance, narrative, realism, cinema, fusion. Browse the registry:

```python
from frontier.registry import list_ideas
implemented = list_ideas(status="implemented")
```

</details>

---

## Pipelines

Diagrams use **tables** (not Mermaid) so they render cleanly on GitHub mobile and dark mode.

### Training (`train.py`)

| Step | What happens |
|------|----------------|
| 1 | Load images + captions |
| 2 | VAE encode → latent space |
| 3 | Add noise @ timestep (flow or VP) |
| 4 | DiT predicts noise / velocity |
| 5 | Loss + backward (optional DPO / GRPO aux) |
| 6 | Checkpoint + metadata |

### Sampling (`sample.py`)

| Step | What happens |
|------|----------------|
| 1 | PromptStack cleans & expands prompt |
| 2 | Optional box layout → regional masks |
| 3 | T5 / CLIP encode conditioning |
| 4 | Denoise loop (Holy Grail CFG, regional blend) |
| 5 | VAE decode → image |

### Video (`pipelines/video/`)

| Step | What happens |
|------|----------------|
| 1 | `compile_scene_graph()` — studio + frontier layers |
| 2 | Shot plan + per-segment overrides |
| 3 | Retrieve reference clips |
| 4 | Keyframe img2img edits |
| 5 | Motion transfer, FLF2V, polish stack |
| 6 | Stitch + provenance |

---

## New in v12

| Area | Highlights |
|------|------------|
| **Video** | Scene-graph TI2V, 60+ modules, CLI tools |
| **Frontier** | 25 filmmaker modules + horizon expansion |
| **Quality** | Continuity validators, thumbnail-first rehearsal |
| **DX** | 1300+ tests, ruff-clean CI, docs restructure |

[Full v12 release notes →](docs/releases/v12.md)

### v1 → v12

| | v1 (foundation) | **v12 (now)** |
|---|-----------------|---------------|
| Scope | Train + sample images | Image + **video studio** + frontier |
| Video | ✗ | Scene JSON director pipeline |
| Layout | ✗ | Regional boxes + storyboard |
| Tests | few | **1300+** |
| Research | packages | `frontier/` + `research/` |

[Full comparison →](docs/releases/VERSION_COMPARISON.md)

---

## Project structure

```
sdx/
├── train.py · sample.py · demo.py     # Image entry points
├── models/ · diffusion/               # DiT, schedulers, sampling
├── frontier/                          # Experimental research
├── pipelines/video/                   # TI2V scene-graph studio (v12)
├── utils/generation/                  # Layout, CFG, sample features
└── tests/                             # 1300+ tests
```

---

## System requirements

| | Minimum | Recommended |
|---|---------|-------------|
| Python | 3.10 | 3.11+ |
| PyTorch | 2.0 | 2.2+ |
| GPU VRAM | 16 GB | 24 GB+ |

---

## Documentation

| Topic | Link |
|-------|------|
| Getting started | [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| Codebase map | [docs/CODEBASE.md](docs/CODEBASE.md) |
| Video pipeline | [pipelines/video/README.md](pipelines/video/README.md) |
| Frontier | [frontier/README.md](frontier/README.md) |
| Competitive notes | [docs/COMPETITIVE_ANALYSIS.md](docs/COMPETITIVE_ANALYSIS.md) |
| v12 release | [docs/releases/v12.md](docs/releases/v12.md) |
| v1 → v12 | [docs/releases/VERSION_COMPARISON.md](docs/releases/VERSION_COMPARISON.md) |
| **Jargon & acronyms** | [docs/GLOSSARY.md](docs/GLOSSARY.md) |

---

## Contributing

```bash
ruff check . --fix && ruff format .
pytest tests/ -q
```

### Are Cursor / Claude permanent contributors?

**No.** GitHub's contributor graph only counts **git commit author names**. AI assistants are not collaborators unless their name appears on commits.

To keep the graph human-only:

1. **Hook (recommended):** copy `scripts/tools/dev/prepare-commit-msg` to `.git/hooks/prepare-commit-msg` — strips `Co-authored-by: Cursor` / Claude trailers before each commit.
2. **Rewrite history (once):** `scripts/tools/dev/cursorfix.sh` reattributes old Cursor-authored commits to you.
3. **Policy:** do not add AI `Co-authored-by` lines (see README in dev scripts).

---

## Version history

| Version | Focus | Notes |
|---------|--------|--------|
| **[v12](docs/releases/v12.md)** | AI film studio video, frontier horizon | **Current** · tag `v12.0.0` |
| [v11](docs/releases/v11.md) | Regional box layout, frontier research, package restructure | `v11.0.0` |
| [v10](docs/releases/v10.md) | ELIQ, artifacts, explainable quality | `v10.0.0` |
| [v9](docs/releases/v9.md) | GRPO family, Superior Stack, agentic training | `v9.0.0` |
| [v8](docs/releases/v8.md) | Style Genome, unified PromptStack | `v8.0.0` |
| [v7](docs/releases/v7.md) | CI, reproducibility, security, benchmarks | `v7.0.0` |
| [v6](docs/releases/v6.md) | Native acceleration, book/comic pipeline | `v6.0.0` |
| [v5](docs/releases/v5.md) | Inference scaling, beam search, data curation | `v5.0.0` |
| [v4](docs/releases/v4.md) | Smart quality filtering, adaptive iteration | `v4` |
| [v3](docs/releases/v3.md) | Hard-case detection, benchmark training loops | `v3` |
| [v0.2](docs/releases/v0.2.0.md) | Flow matching, DPO, knowledge distillation | `v0.2.0` |
| [v0.1](docs/releases/v0.1.0.md) | Foundation train + sample framework | `v0.1.0` |

[Full timeline & v1 → v12 comparison →](docs/releases/VERSION_COMPARISON.md) · **[What does this jargon mean? →](docs/GLOSSARY.md)**

<details>
<summary><strong>Version history in plain English</strong> (click to expand)</summary>

| Version | What it actually means |
|---------|------------------------|
| **v0.1** | First SDX: train your own image model from your data, generate from text. |
| **v0.2** | Faster training (flow matching), learn from preferences (DPO), compress models (distillation). |
| **v3** | Auto-find weak prompts/images, benchmark, retrain in a loop to improve. |
| **v4** | Detect bad generations mid-run, retry, filter low-quality outputs. |
| **v5** | Generate many candidates, pick the best; tools to clean training data. |
| **v6** | Faster native code; pipeline for illustrated books and comics. |
| **v7** | Automated tests on every push, reproducible runs, security + eval benchmarks. |
| **v8** | Invent new art styles; one prompt system for train + sample; smarter CFG schedule. |
| **v9** | RL-style fine-tuning (GRPO); AI helpers that score and refine each other. |
| **v10** | Label-free quality scoring, glitch detection, human-readable quality reports. |
| **v11** | Draw boxes on the image for per-region prompts; reorganized code folders. |
| **v12** | Full video pipeline from one JSON scene; 25+ director rules; 1300+ tests. |

</details>

---

## Glossary

Short definitions for terms used above. [Full glossary with every acronym →](docs/GLOSSARY.md)

| Term | Plain English |
|------|----------------|
| **DiT** | The transformer network that generates images by removing noise step by step. |
| **VAE / latent** | Compresses images to a smaller grid for fast training; expands back to pixels at the end. |
| **CFG** | How tightly the image follows your prompt (higher = more literal). |
| **Flow matching** | A modern training method, often faster than classic diffusion. |
| **DPO** | Train from “A is better than B” preference pairs. |
| **GRPO** | Generate several outputs, rank them, train toward the winners (RL-style). |
| **Holy Grail** | SDX’s adaptive CFG — strong guidance late, looser early. |
| **TCIS** | Generate multiple images; scorers vote for the best match to your prompt. |
| **Style Genome** | Recipe system for inventing new visual styles (not copying artists). |
| **PromptStack** | Cleans and expands prompts the same way in training and sampling. |
| **ELIQ** | Scores image quality without needing human ratings. |
| **Regional box layout** | Put prompts in rectangles on the canvas (character here, sky there). |
| **Frontier** | Experimental folder — new ideas live here before going production. |
| **Agentic** | Several small AI tools (score, refine, validate) instead of one black box. |
| **TI2V / scene graph** | Text or image in → video out, directed by one JSON scene file. |
| **FLF2V** | You set the first and last frame; SDX fills in the motion between. |
| **CI / ruff / pytest** | Auto tests and style checks that run when you push code. |

---

## Citation

```bibtex
@software{sdx_2026,
  title={SDX: Advanced Text-to-Image and Video Generation Framework},
  author={Llunarstack},
  year={2026},
  version={12.0.0},
  url={https://github.com/Llunarstack/sdx}
}
```

---

<p align="center">
  <sub>Apache 2.0 · <a href="https://github.com/Llunarstack/sdx/issues">Issues</a> · <a href="https://github.com/Llunarstack/sdx/releases">Releases</a></sub>
</p>
