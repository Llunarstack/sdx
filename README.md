<h1 align="center">SDX</h1>

<p align="center">
  <b>Stable Diffusion Transformer eXtended</b><br/>
  Train, direct, measure and improve your own image &amp; video models. Fully open, end to end.
</p>

<p align="center">
  <a href="https://github.com/Llunarstack/sdx/actions/workflows/ci.yml"><img src="https://github.com/Llunarstack/sdx/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://github.com/Llunarstack/sdx/releases"><img src="https://img.shields.io/badge/release-v14.0.0-0ea5e9?style=flat-square" alt="Release v14.0.0"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.x"/></a>
  <a href="native/README.md"><img src="https://img.shields.io/badge/Rust-kernels-B7410E?style=flat-square&logo=rust&logoColor=white" alt="Rust kernels"/></a>
  <img src="https://img.shields.io/badge/tests-1750%2B-22c55e?style=flat-square" alt="1750+ tests"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-22c55e?style=flat-square" alt="Apache 2.0"/></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#whats-new-in-v14">What's new</a> ·
  <a href="#natural-looking-images">Natural look</a> ·
  <a href="#100-opt-in-techniques">100 techniques</a> ·
  <a href="#train-with-confidence">Training</a> ·
  <a href="#how-sdx-compares">Compare</a> ·
  <a href="#documentation">Docs</a>
</p>

> [!TIP]
> **v14 is out.** A measure-fix-verify **natural-look pass**, **100 opt-in sampling and finishing techniques**,
> **Rust image kernels** (9-36x faster than NumPy), a **training-readiness gate**, resumable training and a codebase
> cut from ~970 to ~330 files. [What's new](#whats-new-in-v14) · [Release notes](docs/CHANGELOG.md#sdx-v14--natural-look-100-techniques-rust-kernels--training-readiness)

---

## What is SDX?

**SDX is an open framework for building your own text-to-image and text/image-to-video systems.** It is not a hosted
model, an API wrapper, or a leaderboard of someone else's weights. You own every stage: the training loop, the sampler,
layout control, the quality critics, the video director and the loop that improves the model from feedback.

```
your images + captions ──► train.py ──► checkpoint ──► sample.py / video studio ──► images & clips
        ▲                     │ readiness gate             │ critics · --natural · --ideas      │
        └──── feedback · RSI · DPO / GRPO ◄────────── bench ablate (measure every change) ◄─────┘
```

| | |
|---|---|
| **Train** | Diffusion Transformer (DiT) on VAE latents: flow matching or VP diffusion, LoRA / DoRA, DPO and GRPO preference training, resumable bit-exact checkpoints. |
| **Generate** | PromptStack prompt cleanup, adaptive CFG (Holy Grail), APG / CFG-Zero* / PAG / guidance intervals, tiled hires-fix, outpainting, detail control. |
| **Control** | Regional box layouts, scene graphs, ControlNet-style conditioning, style references, Style Genome. |
| **Look natural** | AI-look critics plus `--natural`, which fixes each measured tell and verifies every fix. |
| **Measure** | 150-prompt benchmark, `bench ablate` (score, win-rate, slowdown, VRAM per variant), CPU end-to-end tests. |
| **Video** | One scene JSON in, a directed multi-shot clip out: retrieval, keyframes, motion, continuity, VIDEOMAX. |

---

## Quick start

```bash
git clone https://github.com/Llunarstack/sdx.git && cd sdx
pip install -r requirements.txt
(cd native/rust && cargo build --release)       # optional: Rust image kernels (NumPy fallback otherwise)
python -m toolkit.env_health                    # CUDA, VRAM, packages
```

Prove the pipeline on a tiny dataset before spending GPU hours:

```bash
python -m scripts.tools make_smoke_dataset --out smoke --count 16 --size 256
python -m scripts.tools readiness check --data-path smoke/train --results-dir results --image-size 256
```

Then train and sample:

```bash
python train.py --data-path my_dataset/ --results-dir results/ --flow-matching-training --epochs 20

python sample.py --ckpt results/<run>/checkpoints/best.pt --prompt "a rainy street at dusk, 35mm photo" \
  --out street.png --natural

python -m scripts.tools ui                      # local web UI at http://127.0.0.1:7860
```

> [!NOTE]
> SDX ships the framework, not trained weights: train on your data (or bring a compatible checkpoint). The tests,
> the readiness check and the smoke dataset all run on CPU.

---

## What's new in v14

| | What you get | Try it |
|---|---|---|
| 🌿 **Natural-look pass** | Measures each AI-look tell, applies the fix meant for it, keeps the fix only if the detectors confirm it. | `--natural` |
| 🧪 **100 opt-in techniques** | Noise shaping, CFG schedules, guidance edits, trajectory guards and 46 image finishes, all off by default. | `--ideas`, `--list-ideas` |
| 🦀 **Rust image kernels** | Error diffusion, blue noise, bilateral, median, Kuwahara: 9-36x NumPy, ~460x pure Python. | `cargo build --release` |
| 🚦 **Training-readiness gate** | Dataset audit, overfit, checkpoint round-trip and resume checks; long runs need a READY report. | `scripts.tools readiness` |
| 💾 **Bulletproof training** | Atomic checkpoints with optimizer, scaler, RNG and data position; bit-exact resume; NaN/Inf guard; run manifest. | `--resume`, `--save-every-steps` |
| 🖼️ **Big canvases** | Tiled (MultiDiffusion) hires-fix and outpainting at any size. | `--hires-tiled`, `--outpaint` |
| ✨ **Detail control** | Tell the model there is less noise mid-sampling: more fine detail, or smoother with a negative amount. | `--detail-amount 0.2` |
| 📊 **Benchmark & ablations** | 150 prompts in 10 categories; per-variant score, category deltas, win-rate, slowdown and VRAM. | `bench ablate --ideas-each` |
| 🖥️ **Local web UI** | A browser form over `sample.py` with live logs and history, bound to localhost, no new dependencies. | `scripts.tools ui` |
| 🧹 **Leaner codebase** | ~970 to ~330 files, 131 test files to 25, Python 3.11+, CI now builds and tests the Rust crate. | |

[Full v14 notes](docs/CHANGELOG.md#sdx-v14--natural-look-100-techniques-rust-kernels--training-readiness) ·
[Breaking changes](docs/CHANGELOG.md#migration-v13--v14)

---

## Natural-looking images

The AI look is mostly a handful of **measurable** tells. `--natural` checks every saved image for them, fixes them one at
a time and verifies each fix with the same detectors:

| Tell | What gives it away | Fix it tries |
|---|---|---|
| Decoder grid | A faint repeating 8/16 px pattern from the VAE | Subtract the folded periodic fingerprint |
| Noise floor | Too clean, or flat grain where cameras have signal-dependent noise | Poisson-Gaussian sensor noise in linear light |
| Detail spectrum | Power spectrum far from natural 1/f² | Spectrum match, wavelet detail |
| Over-smoothing | Plastic surfaces with no texture | Band-limited texture, wavelet detail |
| Oversaturation | Gamut-hugging, glossy colour | Chroma knee, OKLab chroma scale |
| Halos | Bright or dark rims at edges | Halo reduction |
| Plastic skin / waxy highlights | Airbrushed skin, specular hot spots | Texture re-injection, specular taming, highlight roll-off |

A fix is kept only when **that tell improves, overall naturalness improves, and the image content does not move**
(blurred OKLab lightness), so nothing can "win" by repainting. If the full frame scores worse afterwards, the pass is
undone. It ends with a blue-noise dither to 8 bit (no banding) and prints what it did:

```
Natural pass: naturalness 0.41 -> 0.98; fixes: grid via degrid+sensor_noise 1
```

Illustration, anime and 3-D prompts skip the camera tells, so they never get sensor noise. `--natural` removes
measurable tells; wrong hands, garbled text and impossible physics come from the checkpoint and the prompt. More levers:
[docs/QUALITY.md § 13](docs/QUALITY.md#13-making-images-look-less-ai-generated).

---

## 100 opt-in techniques

Each technique is one small function at one hook of the sampler or the final image. Strength `1` is the suggested amount
and `0` is an exact no-op; the baseline stays bit-identical when none are enabled.

| Hook | # | Examples |
|---|:-:|---|
| **Initial noise** | 15 | Pyramid / phase-whitened noise, orthogonal or Sobol-QMC batches, shared-layout variations, symmetry and rule-of-thirds priors |
| **CFG scale** | 3 | Bell schedule, resolution-aware CFG, prompt-density CFG |
| **Guidance direction** | 15 | Norm clip, per-channel centring (no CFG colour cast), subject focus, sign consensus, sparse and low-rank guidance, auto-CFG |
| **Clean estimate** | 11 | Soft clip, colour and contrast anchors, composition lock, mid-step sharpening |
| **Per step** | 10 | Particle repulsion (diverse batches), late churn, momentum, variance and divergence guards |
| **Final image** | 46 | Blue-noise / Floyd-Steinberg dither, deband, halation, bloom, OKLab vibrance, white balance, dehaze, retinex, deconvolution, Kuwahara, watercolour, halftone, cel shading |

```bash
python sample.py --list-ideas
python sample.py --ckpt best.pt --prompt "..." --ideas noise_orthogonal_batch,delta_channel_center=0.7
python -m scripts.tools bench ablate --ckpt best.pt --ideas-each delta --seed-list 1,2,3
```

None of these is proven on a trained SDX checkpoint yet: `bench ablate --ideas-each` scores every one against the
baseline on your prompts, so you keep the ones that win. Catalogue: [docs/QUALITY.md](docs/QUALITY.md#ideas-100-opt-in-techniques---ideas).

---

## Train with confidence

GPU time is expensive; SDX makes a long run prove itself on a tiny one first.

| Check | What it proves |
|---|---|
| **Dataset audit** | Unreadable images, odd modes, tiny or extreme-aspect images, empty or over-long captions, exact and near duplicates, train/val overlap, rows the loader silently skips |
| **Architecture** | Image to latent to patch grid to position embeddings, text-encoder width to conditioning width |
| **Learning** | One forward/backward with finite loss and gradients reaching every parameter, then an overfit on 8-16 samples |
| **Recovery** | Checkpoint round-trip and a stop-and-resume run that must match an uninterrupted one exactly |
| **Sampling** | `sample.py` renders an image from the checkpoint it just saved |
| **Throughput** | Images per second and peak VRAM (on GPU) |

```bash
python -m scripts.tools readiness dataset --data-path my_dataset/ --image-size 256
python -m scripts.tools readiness check --data-path my_dataset/ --results-dir results/ --flow-matching-training
```

`train.py` refuses a run longer than `--readiness-min-steps` until a READY report exists for the same commit, config and
dataset (`--skip-readiness-gate` to override). Every run also writes `run_manifest.json` (command, git commit, package
versions, GPU, dataset fingerprint), saves checkpoints atomically, skips NaN/Inf steps and logs grad norm, LR, images/s
and peak VRAM. Guide: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md#before-you-train-readiness-gate).

---

## Measure, don't guess

```bash
python -m scripts.tools bench ablate --ckpt best.pt --seed-list 42,123 \
  --variant baseline= --variant hires="--hires-fix" --variant natural="--natural"
```

`bench ablate` runs the [150-prompt pack](examples/bench_pack_v1.json) (photorealism, anatomy, text, counting, spatial,
binding, multi-subject, composition, style, lighting) for every variant with the same seeds, then writes per-variant mean
score, delta vs baseline per category, paired win-rate, mean seconds, slowdown and peak VRAM, plus each run's checkpoint
hash, GPU and torch version. Interrupted runs resume. Details: [docs/QUALITY.md](docs/QUALITY.md#benchmark--ablations).

---

## Everything else in the box

<details open>
<summary><b>Image generation</b> — <code>train.py</code>, <code>sample.py</code></summary>

- DiT + VAE latent diffusion; flow matching and VP; DPO and the GRPO family; LoRA / DoRA / LyCORIS adapters
- Holy Grail adaptive CFG, APG, CFG-Zero*, FDG, PAG / SLG, guidance intervals, SAG, detail control
- Solvers and schedules: DPM++, UniPC, AYS, Karras, APEX; block / Taylor feature caches
- TCIS critic committee, pick-best metrics (`natural`, `combo_vit_realism`, ...), OCR repair, auto-refine
- Realism: shot specs (camera, ISO, focal length) and a camera / film capture finish
- Game art and pixel art: grid snap, palettes, transparent backgrounds, seamless tiles

</details>

<details>
<summary><b>Layout &amp; control</b></summary>

- Regional box layouts (JSON with per-region prompts), prompt-to-layout planning, glyph canvas for text
- Scene graphs, ControlNet-style conditioning, style references, Style Genome (new styles without artist names)
- Invention stack: prompt passes for counting, attribute binding, negation and anatomy

</details>

<details>
<summary><b>Video studio</b> — <code>pipelines/video/</code></summary>

One **scene JSON**: retrieve, edit keyframes, add motion, polish, stitch.

- **VIDEOMAX** (`video_quality: max`): strong keyframes, per-medium motion grammar, segment gates and retry
- Engine router (realistic, anime, voxel, ...), director mode, character and world bibles
- Elements, motion brush, first/last-frame interpolation (FLF2V), storyboard cuts, continuity checks
- 25 frontier filmmaker modules (tension curve, causal ripples, witness lens, ...)

```bash
python -m scripts.tools video_generate --list-frontier
python -m scripts.tools video_generate --scene examples/scene_studio.example.json --preflight
```

</details>

<details>
<summary><b>Self-improvement</b> — RSI</summary>

`sample.py --log-feedback`, then `python -m scripts.tools feedback like|dislike|pair`, then `rsi_loop`: the model paints,
judges its art, trains on its own best-vs-worst pairs plus your ratings, and keeps a checkpoint only if held-out scores
rise. Guide: [docs/QUALITY.md](docs/QUALITY.md#rsi-your-model-teaches-itself-to-make-better-art).

</details>

<details>
<summary><b>Safety</b></summary>

An unconditional guard in [`utils/safety/`](utils/safety/minor_guard.py) blocks sexual content involving minors in
prompts, conditioning images and outputs. It cannot be turned off.

</details>

---

## How it works

<table>
<tr><th>Training — <code>train.py</code></th><th>Sampling — <code>sample.py</code></th></tr>
<tr valign="top"><td>

| # | Step |
|:-:|------|
| 1 | Load images + captions (folder or JSONL) |
| 2 | VAE-encode to latents |
| 3 | Noise at a sampled timestep (flow or VP) |
| 4 | DiT predicts noise / velocity |
| 5 | Loss + backward (optional DPO / GRPO) |
| 6 | NaN guard, optimizer step, EMA |
| 7 | Atomic checkpoint + run manifest |

</td><td>

| # | Step |
|:-:|------|
| 1 | PromptStack cleans and expands the prompt |
| 2 | Optional box layout to regional masks |
| 3 | T5 / CLIP encode the conditioning |
| 4 | Denoise: adaptive CFG, `--ideas` hooks, tiling |
| 5 | VAE decode, pick the best candidate |
| 6 | Capture finish, `--natural`, dither |
| 7 | Safety guard, save, report |

</td></tr>
</table>

---

## How SDX compares

SDX is a **framework you train**, not a hosted model. It is compared on capability classes (mid-2026 snapshot), not on
aesthetics: names and versions move weekly, the structural gaps usually do not.

**✓** strong / built in · **◐** partial / LoRA-only / plugin · **ext** community extensions · **✗** not available · **n/a** wrong product shape

| Capability | GPT Image | Midjourney | Ideogram | FLUX.2 | SD3.5 | Qwen-Image | Comfy | **SDX** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Best-in-class hosted aesthetics | ✓ | **✓** | ◐ | ✓ | ◐ | ◐ | n/a | n/a¹ |
| Text-in-image | **✓** | ◐ | **✓** | ✓ | ◐ | **✓** | ext | ◐² |
| Download / run weights | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| LoRA / adapter fine-tune | ✗ | ✗ | ✓ | ✓ | **✓** | ◐ | **✓** | **✓** |
| Full train loop on *your* folder | ✗ | ✗ | ✗ | ◐ | ◐ | ◐ | ext | **✓** |
| DPO / GRPO preference training | ✗ | ✗ | ✗ | ✗ | ext | ✗ | ext | **✓** |
| Regional box layout (native) | ◐ | ◐ | ✓ | ext | ext | ext | ext | **✓** |
| Self-critique + retry (TCIS / pick-best) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ | **✓** |
| Verified natural-look finishing | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ext | **✓** |
| Ablation harness for every option | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Open scene-graph video director | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ext | **✓** |
| Air-gap + reproducible metadata | ✗ | ✗ | ◐ | ◐ | ◐ | ◐ | ◐ | **✓** |

¹ SDX is not a published aesthetics checkpoint: you train (or bring) weights.
² Glyph canvas + OCR pick-best help; typography specialists still lead raw text rendering.

<details>
<summary><b>The full landscape</b>: closed APIs, open weights, video (click to expand)</summary>

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

| Capability (video) | Veo | Kling | Runway | Luma | Wan / LTX | **SDX** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Hosted clip quality / audio | **✓** | **✓** | ✓ | ✓ | ◐ | n/a¹ |
| Multi-shot storyboard | ✓ | **✓** | ✓ | ◐ | ◐ | **✓** |
| Open scene JSON → plan → stitch | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Identity / continuity validators | ◐ | ◐ | ✓ | ◐ | ✗ | **✓** |
| Trainable / forkable pipeline | ✗ | ✗ | ✗ | ✗ | ◐ | **✓** |

¹ The default SDX video path is an open **compositional studio** (retrieve → keyframe → motion → polish); an optional
VideoDiT exists. Frontier clip quality still belongs to hosted Veo / Kling / Runway unless you train.

</details>

**Where they still win.** Zero-setup pretty defaults (Midjourney, GPT Image, Veo), raw text-in-image demos (Ideogram,
GPT Image, Qwen-Image), the deepest node ecosystem (ComfyUI with SD / FLUX weights), hosted video with native audio (Veo,
Kling, Runway). Use those when you need pixels tomorrow; use SDX when you need to **own the loop**.

**What we don't claim.** No "beats Midjourney on aesthetics" (no bake-off is claimed), no drop-in Flux / Veo replacement
(different product shape), no single demo checkpoint that does everything. Strategy and the weakness→fix map:
[docs/RESEARCH.md](docs/RESEARCH.md#competitive-analysis--solution-roadmap).

---

## Performance & native code

Hot CPU paths have optional native builds that fall back to NumPy / PyTorch when absent, with parity tests that CI runs.

| Kernel (1024 x 1024 RGB, 4 threads) | Rust | NumPy | Speed-up |
|---|---:|---:|---:|
| Kuwahara, radius 3 | 20 ms | 704 ms | **36x** |
| Bilateral, radius 3 | 113 ms | 1319 ms | **12x** |
| 3x3 median | 41 ms | 365 ms | **9x** |
| Floyd-Steinberg to 8 bit | 16 ms | ~7.1 s (pure Python) | **~460x** |

Also native: VP schedule math, token Jaccard, blob counting (Rust); latent geometry, image metrics, mask→patch weights
(C++); latent blur and luma stats (CUDA); manifest merge (Go); fingerprints (Zig). Build: [native/README.md](native/README.md).
Every sampling run prints per-stage seconds and peak VRAM (`timing_s` / `peak_vram_gb` in `--pick-report-json`).

---

## Project structure

```
sdx/
├── train.py · sample.py · demo.py · inference.py    entry points
├── config/        presets, prompt / style registries, train config
├── data/          datasets, captions, dataset audit
├── diffusion/     flow + VP objectives, schedules, solvers, sampling loop
├── models/        DiT variants, attention, conditioning, LoRA, video DiT
├── training/      train CLI, loop, readiness gate
├── utils/
│   ├── generation/   sampler, guidance, layout, tiling, editing
│   ├── ideas/        the 100 opt-in techniques + kernels
│   ├── prompt/       PromptStack, realism, game, scene
│   ├── quality/      critics, natural pass, finishing, realism, pixel art
│   └── safety/       minor guard (always on)
├── frontier/      research: layout, multiview / 3-D, imagination
├── pipelines/     book / comic, image_gen, video studio
├── native/        Rust · C++ / CUDA · Go · Zig accelerators
├── scripts/tools/ python -m scripts.tools <command> (bench, readiness, ui, ...)
└── tests/         1750+ tests
```

Per-file map: [docs/FILES.md](docs/FILES.md) · generated tree: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

## System requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | 3.13 |
| PyTorch | 2.x | 2.11 + CUDA 12.8 |
| GPU, sampling | 24 GB (default T5-XXL encoder is ~19 GB in fp32) | 40 GB+ |
| GPU, training | 40 GB | 80 GB class (A100 / H100), cloud is fine (see `runpod/`) |
| Rust | — | stable toolchain, for the image kernels |

GPU figures are estimates for the default encoders; measure yours with the readiness check and `bench ablate`. CPU is
enough for the test suite, the readiness check and the smoke dataset.

---

## Documentation

| Topic | Link |
|---|---|
| Getting started, hardware, readiness gate | [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| **v14 release notes** | [docs/CHANGELOG.md](docs/CHANGELOG.md#sdx-v14--natural-look-100-techniques-rust-kernels--training-readiness) |
| Quality: natural look, 100 techniques, ablations, RSI | [docs/QUALITY.md](docs/QUALITY.md) |
| How generation works (pipeline, CFG, solvers, detail control) | [docs/HOW_GENERATION_WORKS.md](docs/HOW_GENERATION_WORKS.md) |
| Prompts and models | [docs/PROMPT_STACK.md](docs/PROMPT_STACK.md) · [docs/MODEL_STACK.md](docs/MODEL_STACK.md) |
| Guides: captions, datasets, book / comic, VIDEOMAX | [docs/GUIDES.md](docs/GUIDES.md) |
| Codebase map · file index | [docs/CODEBASE.md](docs/CODEBASE.md) · [docs/FILES.md](docs/FILES.md) |
| Native accelerators | [native/README.md](native/README.md) |
| Video pipeline · frontier | [pipelines/video/README.md](pipelines/video/README.md) · [frontier/README.md](frontier/README.md) |
| Model lines and release gates | [docs/MODEL_LINES_AND_RELEASES.md](docs/MODEL_LINES_AND_RELEASES.md) |
| Research and competitive notes | [docs/RESEARCH.md](docs/RESEARCH.md) |
| **Jargon and acronyms** | [docs/GLOSSARY.md](docs/GLOSSARY.md) |

---

## Contributing

```bash
ruff check . --fix && ruff format .
pytest tests/ -q
cp scripts/tools/dev/commit-msg .git/hooks/commit-msg    # keeps AI tools off the contributor list
```

Full checklist (CI mirror, type check, docs): [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Version history

| Version | Focus | Tag |
|---|---|---|
| **[v14](docs/CHANGELOG.md#sdx-v14--natural-look-100-techniques-rust-kernels--training-readiness)** | Natural-look pass, 100 techniques, Rust kernels, readiness gate, resumable training, leaner codebase | **Current** · `v14.0.0` |
| [v13](docs/CHANGELOG.md#sdx-v13--videomax-rsi-feedback--invention-waves) | VIDEOMAX, RSI feedback, Invention Lab, 1700+ tests | `v13.0.0` |
| [v12](docs/CHANGELOG.md#sdx-v12--ai-film-studio-video-pipeline--frontier-horizon) | AI film studio video, frontier horizon | `v12.0.0` |
| [v11](docs/CHANGELOG.md#sdx-v11--layout-control-frontier-research--package-restructure) | Regional box layout, frontier research, package restructure | `v11.0.0` |
| [v10](docs/CHANGELOG.md#sdx-v10-advanced-quality--explainability-release) | ELIQ, artifacts, explainable quality | `v10.0.0` |
| [v9](docs/CHANGELOG.md#sdx-v9-source-release) | GRPO family, Superior Stack, agentic training | `v9.0.0` |
| [v8](docs/CHANGELOG.md#sdx-v8--style-genome-and-unified-prompt-pipeline) | Style Genome, unified PromptStack | `v8.0.0` |
| [v7](docs/CHANGELOG.md#sdx-v7--production-quality-and-research-infrastructure) | CI, reproducibility, security, benchmarks | `v7.0.0` |
| [v6](docs/CHANGELOG.md#sdx-v6--native-acceleration-and-book-generation-pipeline) | Native acceleration, book/comic pipeline | `v6.0.0` |
| [v5](docs/CHANGELOG.md#sdx-v5--inference-scaling-and-quality-optimization) | Inference scaling, beam search, data curation | `v5.0.0` |
| [v4](docs/CHANGELOG.md#sdx-v4--smart-quality-filtering-and-advanced-models) | Smart quality filtering, adaptive iteration | `v4` |
| [v3](docs/CHANGELOG.md#sdx-v3--automated-quality-improvement-and-benchmarking) | Hard-case detection, benchmark training loops | `v3` |
| [v0.2](docs/CHANGELOG.md#sdx-v020--advanced-training-methods-flow-matching-dpo-distillation) | Flow matching, DPO, knowledge distillation | `v0.2.0` |
| [v0.1](docs/CHANGELOG.md#sdx-v010--baseline-text-to-image-training-framework) | Foundation train + sample framework | `v0.1.0` |

<details>
<summary><b>Version history in plain English</b> (click to expand)</summary>

| Version | What it actually means |
|---|---|
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
| **v13** | VIDEOMAX consistency, RSI thumbs to DPO, Invention Lab waves, 1700+ tests. |
| **v14** | Images that measure as natural; 100 techniques you can switch on and benchmark; Rust speed-ups; training that proves itself before it costs money. |

</details>

---

## Glossary

Short definitions for terms used above. [Full glossary →](docs/GLOSSARY.md)

| Term | Plain English |
|---|---|
| **DiT** | The transformer network that generates images by removing noise step by step. |
| **VAE / latent** | Compresses images to a smaller grid for fast training; expands back to pixels at the end. |
| **CFG** | How tightly the image follows your prompt (higher = more literal). |
| **Flow matching** | A modern training method, often faster than classic diffusion. |
| **DPO / GRPO** | Train from "A is better than B" pairs / from ranked groups of outputs (RL-style). |
| **Holy Grail** | SDX's adaptive CFG: looser early, stronger where it helps. |
| **TCIS** | Generate several images; scorers vote for the best match to your prompt. |
| **AI-look tell** | A measurable trace that gives a generated image away (decoder grid, missing sensor noise, plastic skin). |
| **Ablation** | Run the same prompts and seeds with and without one change, to see what that change actually does. |
| **Readiness gate** | Small checks a training setup must pass before a long, expensive run is allowed. |
| **Dither** | Tiny noise added when converting to 8 bit so smooth gradients don't show bands. |
| **Regional box layout** | Put prompts in rectangles on the canvas (character here, sky there). |
| **Frontier** | Experimental folder: new ideas live here before going to production. |
| **FLF2V** | You set the first and last frame; SDX fills in the motion between. |

---

## Citation

```bibtex
@software{sdx_2026,
  title   = {SDX: Stable Diffusion Transformer eXtended},
  author  = {Llunarstack},
  year    = {2026},
  version = {14.0.0},
  url     = {https://github.com/Llunarstack/sdx}
}
```

---

<p align="center">
  <sub>Apache 2.0 · <a href="https://github.com/Llunarstack/sdx/issues">Issues</a> · <a href="https://github.com/Llunarstack/sdx/releases">Releases</a> · <a href="CONTRIBUTING.md">Contributing</a></sub>
</p>
