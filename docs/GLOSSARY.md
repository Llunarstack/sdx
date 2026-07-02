# SDX Glossary — Plain English

Quick reference for acronyms and jargon used in the README, version history, and release notes.

---

## Version history explained

What each release actually added, without the buzzwords.

| Version | In plain English |
|---------|------------------|
| **v0.1** | The first usable SDX: load your images and captions, train a custom image model, generate pictures from text. You can read and change the whole pipeline. |
| **v0.2** | Better training recipes: faster **flow matching**, learn from human **preferences (DPO)**, and shrink big models into smaller ones (**distillation**). |
| **v3** | The model finds its own weak spots (hard prompts/images), runs **benchmarks** automatically, and trains in loops to fix them. |
| **v4** | Smarter generation: detect when quality is bad mid-run, try again, and filter weak outputs instead of keeping everything. |
| **v5** | Generate several candidates and pick the best (**beam search** / inference scaling). Tools to clean and rank **training data**. |
| **v6** | Speed-ups with compiled **native code**. A full pipeline to generate **illustrated books and comics** (multi-page, consistent). |
| **v7** | “Production ready” basics: automated **tests on every push (CI)**, same-results-twice **reproducibility**, security policy, standard **eval benchmarks**. |
| **v8** | Invent **new art styles** from rules (not copy artists). One shared **prompt pipeline** for training and generation. Better default sampling (**Holy Grail**). |
| **v9** | Advanced **RL-style training (GRPO)** — six variants. **Agentic** helpers (models that score, refine, and validate each other). **Superior Stack** = bundled best inference tricks. |
| **v10** | Deep **quality tools**: score images without human labels (**ELIQ**), catch AI glitches (**artifacts**), explain *why* an image scored low, stop “fixing” that breaks the prompt (**drift**). |
| **v11** | Put objects in **boxes on the canvas** (like Ideogram). Experimental code moved to **`frontier/`**; stable code to **`innovations/`**. |
| **v12** | **Video from text/images**: one JSON scene file → shots, camera, continuity checks, 25+ “filmmaker” rules. **`frontier/`** grew to 80+ experiments. **803+ automated tests**. |

---

## Core acronyms & terms

### Project & architecture

| Term | Stands for / means | What it does in SDX |
|------|--------------------|---------------------|
| **SDX** | Stable Diffusion Transformer eXtended | This project — open framework to train and run your own image/video models. |
| **DiT** | Diffusion Transformer | The main neural network that denoises images. Transformer-based (like GPT) but for pixels/latents. |
| **VAE** | Variational Autoencoder | Compresses images to a smaller **latent** tensor for fast training; decodes back to pixels when done. |
| **Latent** | — | Compact representation of an image (e.g. 64× smaller). Training happens here, not on full 1024×1024 pixels. |
| **T5 / CLIP** | Text encoders | Turn your prompt into numbers the DiT understands. T5 = long text; CLIP = short tags + image similarity. |
| **CFG** | Classifier-Free Guidance | How strongly the model follows the prompt vs. being creative. Higher = more literal, lower = more random. |
| **VP** | Variance Preserving | Classic diffusion noise schedule (as opposed to flow matching). |

### Training methods

| Term | Stands for / means | What it does in SDX |
|------|--------------------|---------------------|
| **Flow matching** | — | A newer training objective; often converges faster than classic diffusion. Flag: `--flow-matching-training`. |
| **DPO** | Direct Preference Optimization | Train from pairs: “image A is better than B for this prompt” — no separate reward model needed. |
| **GRPO** | Group Relative Policy Optimization | RL-style fine-tuning: generate a batch, rank by reward, update toward winners. SDX has 6 GRPO variants. |
| **Distillation** | Knowledge distillation | Teach a small/fast model to mimic a larger one so inference is cheaper. |
| **RLHF** | Reinforcement Learning from Human Feedback | Broad family of “learn from preferences/rewards”; DPO and GRPO are specific methods. |

### Inference & quality

| Term | Stands for / means | What it does in SDX |
|------|--------------------|---------------------|
| **Holy Grail** | SDX name for adaptive CFG | Changes guidance strength across denoise steps — explore early, lock in late. |
| **TCIS** | Text-Conditioned Image Selection | Generate multiple images; a **committee** of scorers picks the best for text/layout-heavy prompts. |
| **PromptStack** | — | Multi-step prompt cleanup/expansion so training and sampling use the same text logic. |
| **Style Genome** | — | Structured recipe for inventing original looks (palette, line, camera, etc.) without copying artists. |
| **ELIQ** | Enhanced Label-free Image Quality | Scores image quality **without** human ratings — adapts during generation. |
| **Artifact detector** | — | Finds typical AI flaws (extra fingers, melted textures, etc.). |
| **Semantic drift** | — | When refinement steps slowly change the image away from what the prompt asked for; SDX can detect and repair. |
| **Explainable scoring** | — | Quality breakdown humans can read (“text match: good, anatomy: poor”), not just one number. |

### Layout & control (v11+)

| Term | Stands for / means | What it does in SDX |
|------|--------------------|---------------------|
| **Regional box prompting** | — | Draw rectangles on the image; each box gets its own prompt (character left, sky right, etc.). |
| **Omost canvas** | Inspired by Omost | Describe a scene as labeled regions; converts to box layout JSON. |
| **LAMIC** | Layout-Aware Multi-region Integration (schedule) | Blends global + per-region predictions across denoise steps so layout stays stable. |
| **ConsistCompose / loc tokens** | — | Special text tokens that tell the model *where* in the frame something should appear. |

### Video pipeline (v12)

| Term | Stands for / means | What it does in SDX |
|------|--------------------|---------------------|
| **TI2V** | Text/Image-to-Video | Start from a prompt and/or reference image(s) and produce video. |
| **Scene graph** | — | One JSON file: characters, shots, duration, camera, effects — the single source of truth. |
| **I2V** | Image-to-Video | Animate from a starting frame (and optional motion reference clip). |
| **T2V** | Text-to-Video | Video driven mainly by text (may still retrieve reference footage). |
| **FLF2V** | First-Last-Frame-to-Video | You supply start + end images; SDX interpolates motion between them. |
| **Keyframe edit** | — | Edit specific frames with the image model, then interpolate between them. |
| **Motion transfer** | — | Copy movement from a reference video onto your generated content. |
| **Storyboard cuts** | — | Multiple shots with duration and camera verbs (push in, orbit, etc.). |
| **Continuity validators** | — | Rules that catch mistakes across shots: wrong eyeline, props teleporting, weather flipping. |
| **Thumbnail rehearsal** | — | Cheap tiny previews of each shot before spending GPU on full resolution. |
| **Frontier filmmaker modules** | — | Experimental rules (tension curve, causal ripples, witness lens, …) that shape prompts and edits like a director would. |

### Packages & research

| Term | Stands for / means | What it does in SDX |
|------|--------------------|---------------------|
| **`innovations/`** | — | Production-quality features: agentic stack, photorealism, semantics, control. |
| **`frontier/`** | — | Experimental ideas — try here first; promote to production when stable. |
| **Agentic** | — | Multiple specialized “agents” (scorers, refiners, validators) working together, not one monolithic model. |
| **Superior Stack** | Marketing name in v9 | Bundle of best inference optimizations (caching, CFG tricks, ensembles, etc.). |

### Dev & ops

| Term | Stands for / means | What it does in SDX |
|------|--------------------|---------------------|
| **CI** | Continuous Integration | GitHub Actions runs lint + tests on every push so broken code doesn’t land on `main`. |
| **Ruff** | Python linter/formatter | Fast style and bug checker (`ruff check .`). |
| **Pytest** | — | Automated test runner (`pytest tests/ -q`). |
| **VRAM** | Video RAM (GPU memory) | How much GPU memory you need; 16 GB minimum, 24 GB+ recommended for training. |
| **DX** | Developer experience | Tests, docs, CLI ergonomics — how pleasant it is to work on the codebase. |
| **Provenance** | — | Metadata saved with outputs: which checkpoint, prompt, seeds, and pipeline steps were used. |

### Other models (comparison table)

| Term | What it is |
|------|------------|
| **SDXL** | Stability AI’s large open image model (1024px). SDX trains similar-capability models you own. |
| **Flux** | Black Forest Labs’ popular open-weights image model. |
| **Ideogram** | Commercial service known for strong **text-in-image** and **layout** control. SDX’s box layout is analogous but self-hosted. |

---

## Still confused?

- [GETTING_STARTED.md](GETTING_STARTED.md) — install and first run  
- [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) — where code lives  
- [releases/VERSION_COMPARISON.md](releases/VERSION_COMPARISON.md) — feature matrix across versions  
- [pipelines/video/README.md](../pipelines/video/README.md) — video pipeline details  
