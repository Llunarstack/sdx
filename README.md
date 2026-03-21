<!-- markdownlint-disable MD033 MD041 -->
<div align="center">

<br/>

# SDX

### Diffusion Transformer · Text-to-image · Your data, your captions

<p align="center">
  <strong>DiT</strong> + <strong>xformers</strong> + optional <strong>block AR</strong> · T5 (optional <strong>triple</strong>: T5 + CLIP-L + CLIP-bigG)<br/>
  <sub>ViT-Gen · REPA · MoE · MDM · RAE bridge · test-time pick · and more</sub>
</p>

No reference image required — **quality follows the dataset and captions you train on.**

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.8+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-diffusion-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License"/></a>
  <a href="docs/IMPROVEMENTS.md"><img src="https://img.shields.io/badge/docs-IMPROVEMENTS-238636?style=for-the-badge" alt="Docs"/></a>
</p>

**[Quick start](#quick-start)** · **[Train](#training)** · **[Sample](#sampling)** · **[Pipeline](#architecture-and-pipeline)** · **[Docs](#documentation-hub)** · **[Layout](#project-layout)**

<br/>

</div>

---

> **👋 New here?** Run [`scripts/tools/quick_test.py`](scripts/tools/quick_test.py), skim the **pipeline** below, then jump to [Quick start](#quick-start).

---

<details>
<summary><strong>📑 Table of contents</strong></summary>

| | |
|:---|:---|
| **Start** | [Quick start](#quick-start) · [Setup](#setup) · [Data format](#data-format) |
| **Workflow** | [Architecture and pipeline](#architecture-and-pipeline) · [Training](#training) · [Sampling](#sampling) · [JSONL fields](#data-jsonl-fields) |
| **Reference** | [Train CLI](#train-cli-quick-reference) · [SD/XL-style](#sd--sdxl-style-features) · [Extra features](#extra-features) |
| **Deep dives** | [Documentation hub](#documentation-hub) · [Project layout](#project-layout) · [References](#references) |

</details>

---

## At a glance

| Pillar | What you get |
|:-------|:-------------|
| **Model** | Text-conditioned **DiT** + cross-attention (**T5**; optional **triple** fusion), optional **AR** blocks, up to **Supreme** / **Predecessor** variants |
| **Training** | Pass-based schedule, **EMA**, **best** checkpoint, val + early stopping, bf16, compile, **DDP** |
| **Sampling** | CFG, schedulers, img2img, inpainting, LoRA, Control-style conditioning, refinement, **pick-best** |
| **Data** | Folders + `.txt` / `.caption` or **JSONL**; caption emphasis & domain boosts |

---

## Architecture and pipeline

**One line:** `data/` + manifest → **`train.py`** (`config/` · **`diffusion/`** · **`models/`** · **`utils/`**) → **checkpoint** → **`sample.py`** → **images**. Weights live in **`model/`** (gitignored), resolved by **`utils/model_paths.py`**.

<pre align="center">
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 📂 <b>Data</b>    │───▶│ ⚙️ <b>train.py</b> │───▶│ 💾 <b>Checkpoint</b>│───▶│ ✨ <b>sample.py</b> │───▶│ 🖼️ <b>Images</b>   │
│ JSONL · QA  │    │ DiT · loss  │    │ EMA · cfg   │    │ CFG · VAE   │    │ grids · pick│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
</pre>

### Repository map

| Area | Role | Consumed by |
|:-----|:-----|:------------|
| **`config/`** | `TrainConfig`, `get_dit_build_kwargs`, presets | `train.py`, `sample.py`, checkpoints |
| **`data/`** | `Text2ImageDataset`, captions | `train.py` |
| **`diffusion/`** | `GaussianDiffusion`, schedules, loss weights, respacing | `train.py`, `sample.py` |
| **`models/`** | DiT, ControlNet, MoE, RAE bridge, optional cascaded / multimodal **scaffolds** | `train.py`, `sample.py`, tests |
| **`utils/`** | Checkpoint load, text-encoder bundle, REPA helpers, QC, **pick-best**, metrics | `train.py`, `sample.py`, scripts |
| **`ViT/`** | Standalone scoring / prompt tools (**not** the DiT generator) | CLI, optional dataset QA |
| **`scripts/`** | Downloads, tools, Cascade stub | Ops & CI |
| **`native/`** | Fast JSONL helpers (Rust, Go, …) | Optional; not imported by training by default |
| **`model/`** | Downloaded HF weights | Paths via `utils/model_paths.py` |

Full index → **[docs/FILES.md](docs/FILES.md)**

### Local weights (`model/`)

| Role | Typical folder / fallback |
|:-----|:--------------------------|
| T5-XXL | `model/T5-XXL` or `google/t5-v1_1-xxl` |
| CLIP (L + bigG) | `model/CLIP-ViT-L-14`, `model/CLIP-ViT-bigG-14` or HF ids |
| DINOv2 (REPA) | `model/DINOv2-Large` or `facebook/dinov2-large` |
| Qwen LLM | `model/Qwen2.5-14B-Instruct` (optional prompt expansion) |
| Stable Cascade | `model/StableCascade-Prior`, `model/StableCascade-Decoder` (optional; **not** DiT) |

**Download:** [docs/MODEL_STACK.md](docs/MODEL_STACK.md) · `scripts/download/download_models.py` · `scripts/download/download_revolutionary_stack.py`

### Who’s who (easy to confuse)

| Name | What it is | Where |
|:-----|:-----------|:------|
| **DiT** | **Diffusion Transformer** — **generator**: patch latents, predict ε/v, **cross-attend text** | `models/dit_text.py`, `train.py`, `sample.py` |
| **ViT-style blocks inside DiT** | **ViT-Gen** (registers, RoPE, KV-merge, SSM…) — still **one DiT** | `--num-register-tokens`, `--use-rope`, … |
| **REPA vision encoder** | **Frozen** DINOv2 / CLIP **image** encoder — aux alignment loss | `--repa-weight`, `--repa-encoder-model` |
| **`ViT/` package** | **Separate** timm ViT for **QA**, ranking, embeddings — **not** DiT | `ViT/train.py`, `ViT/infer.py` |
| **Text triple (T5 + CLIP)** | **Text** towers fused → conditions **DiT** | `--text-encoder-mode triple` |

---

### Diagram 1 — Full stack (data → train → sample)

```mermaid
flowchart TB
  classDef weight fill:#f6f8fa,stroke:#24292f,stroke-width:2px,color:#24292f
  classDef data fill:#ddf4ff,stroke:#0969da,stroke-width:2px,color:#042033
  classDef train fill:#fff8c5,stroke:#bf8700,stroke-width:2px,color:#3b2600
  classDef sample fill:#dafbe1,stroke:#1a7f37,stroke-width:2px,color:#0d1f12
  classDef side fill:#fbefff,stroke:#8250df,stroke-width:1px,color:#2e1067

  subgraph WG["Weights · model/ + utils/model_paths.py"]
    W[T5 · VAE · CLIP · DINOv2 · Qwen · Cascade]:::weight
  end

  subgraph DL["Data and optional QA"]
    DS[(JSONL / folders)]:::data
    PL[prompt_lint]:::data
    NT[native/ tools]:::data
    V1[ViT/ infer → scores]:::data
    V2[ViT/ rank → filter]:::data
    DS2[Cleaner / weighted data]:::data
    DS --> PL
    DS --> NT
    DS --> V1
    V1 --> V2
    V2 --> DS2
    DS --> DS2
  end

  subgraph TR["train.py — DiT training"]
    TE[T5 or T5+CLIP fusion]:::train
    VA[VAE or RAE latents]:::train
    REP[Optional REPA vision]:::train
    DIT[DiT-Text + diffusion loss]:::train
    CKPT[(Checkpoint · EMA · config · fusion · RAE bridge)]:::weight
    TE --> DIT
    VA --> DIT
    REP -.->|aux| DIT
    DIT --> CKPT
  end

  subgraph SM["sample.py — generation"]
    TE2[Text stack from ckpt]:::sample
    DN[DiT denoise]:::sample
    DC[VAE decode]:::sample
    PB[pick-best CLIP/edge/OCR]:::sample
    IMG[(Images)]:::sample
    TE2 --> DN --> DC --> PB --> IMG
  end

  subgraph OP["Optional side paths"]
    QW[Qwen expand prompts]:::side
    CAS[Stable Cascade · scripts/cascade_generate.py]:::side
  end

  W --> TE
  W --> VA
  W --> TE2
  W -.-> REP
  W -.-> QW
  W -.-> CAS
  DS2 --> TR
  CKPT --> SM
  QW -.-> TE2
```

### Diagram 2 — Inside training (text + latent + DiT)

```mermaid
flowchart LR
  classDef t fill:#ddf4ff,stroke:#0969da,color:#042033
  classDef l fill:#fff8c5,stroke:#bf8700,color:#3b2600
  classDef d fill:#dafbe1,stroke:#1a7f37,color:#0d1f12

  subgraph TX["Text conditioning"]
    direction TB
    T5[T5-XXL]:::t
    C1[CLIP-L text]:::t
    C2[CLIP-bigG text]:::t
    FU[Fusion +2 tokens]:::t
    TXT[encoder_hidden_states]:::t
    T5 --> FU
    C1 --> FU
    C2 --> FU
    FU --> TXT
  end

  subgraph LT["Latent"]
    direction TB
    VAE[VAE 4ch]:::l
    RAE[RAE → bridge → 4ch]:::l
    LAT[Latent x]:::l
    VAE --> LAT
    RAE --> LAT
  end

  subgraph CORE["DiT-Text"]
    direction TB
    PE[patch + pos]:::d
    SA[self-attn · ViT-Gen]:::d
    CA[cross-attn → text]:::d
    OUT[ε or v]:::d
    LAT --> PE --> SA --> CA --> OUT
    TXT --> CA
  end

  subgraph RP["Optional REPA"]
    IM[Real images]:::l
    VS[Frozen DINOv2 / CLIP vision]:::l
    IM --> VS -.->|align| OUT
  end
```

### Diagram 3 — `ViT/` vs DiT (different jobs)

```mermaid
flowchart TB
  classDef gen fill:#dafbe1,stroke:#1a7f37,color:#0d1f12
  classDef tool fill:#ddf4ff,stroke:#0969da,color:#042033

  subgraph G["Generation: DiT · train.py / sample.py"]
    A[Latent diffusion objective]:::gen
    B[Image generation]:::gen
  end

  subgraph T["Tooling: ViT/ folder"]
    C[Quality + adherence scores]:::tool
    D[Manifest rank / filter]:::tool
    E[prompt_system]:::tool
    F[Embeddings]:::tool
  end

  CKPT2[(DiT checkpoint)]:::gen
  DS3[(Dataset)]:::tool
  N[best-of-N rerank]:::tool

  G -.-> CKPT2
  T -.-> DS3
  T -.-> N
```

### Stage cheat sheet

| Stage | What runs |
|:------|:----------|
| **Config** | **`config/train_config.py`**: `TrainConfig` + **`get_dit_build_kwargs`** → DiT build args |
| **Data** | **`data/t2i_dataset.py`**: folders or JSONL, emphasis, latent cache |
| **Diffusion** | **`diffusion/`**: `GaussianDiffusion`, schedules, DDIM, CFG rescale, **`respace`**, loss weights |
| **Text → DiT** | Default **T5**. **Triple**: T5 + CLIP-L + CLIP-bigG → **`text_encoder_fusion`** in ckpt |
| **Image → latent** | **VAE** or **RAE** + **`RAELatentBridge`** when channels ≠ 4 |
| **DiT core** | **`models/dit_text.py`**: patch, self/cross-attn, ViT-Gen, **MoE**, **MDM** |
| **REPA** | Frozen **vision** encoder — auxiliary alignment |
| **`ViT/` tools** | Scoring / rank / prompts — **not** the generator |
| **Sample** | **`sample.py`**: CFG, decode, **`utils/test_time_pick`** |
| **API** | **`inference.py`**: programmatic sampling |
| **Other** | **Qwen** (`utils/llm_client.py`); **Cascade** (`scripts/cascade_generate.py`) — separate from DiT forward |

**Optional scaffolds** (not default `train.py`): `diffusion/cascaded_multimodal_pipeline.py`, **`models/cascaded_multimodal_diffusion.py`**, **`models/native_multimodal_transformer.py`** — see **[docs/FILES.md](docs/FILES.md)**

| See also | |
|:---------|:--|
| **Weights** | **[docs/MODEL_STACK.md](docs/MODEL_STACK.md)** |
| **Every file** | **[docs/FILES.md](docs/FILES.md)** |
| **Config ↔ ckpt ↔ sample** | **[docs/CONNECTIONS.md](docs/CONNECTIONS.md)** |

---
## Highlights

<details open>
<summary><strong>Core training & data</strong></summary>

| Feature | Notes |
|:--------|:------|
| **Passes, not blind epochs** | `--passes N` = N full sweeps over the dataset; optional `--max-steps` cap |
| **Quality of training** | Cosine LR, **EMA**, **save best**, optional **val split + early stopping** |
| **Captions** | `(tag)` / `((tag))` emphasis, `[tag]` de-emphasis, subject-first order |
| **Negative prompts** | Trained as *positive − w×negative* so the model avoids unwanted concepts |
| **JSONL** | `caption`, `negative_*`, `style`, `control_*`, `init_image`, weights, etc. |

</details>

<details open>
<summary><strong>Architecture & speed</strong></summary>

| Feature | Flag / entry |
|:--------|:-------------|
| **Block-wise AR** | `--num-ar-blocks 2` or `4` — see [docs/AR.md](docs/AR.md) |
| **xformers + compile** | Memory-efficient attention; `torch.compile`; bf16; grad checkpointing |
| **Register tokens, RoPE, KV merge** | `--num-register-tokens`, `--use-rope`, `--kv-merge-factor` |
| **SSM mixer** | `--ssm-every-n` replaces every Nth self-attn block |
| **MoE FFN** | `--moe-num-experts`, `--moe-top-k` |
| **REPA** | `--repa-weight` + vision encoder alignment |
| **Size conditioning** | `--size-embed-dim` (PixArt-style H,W → timestep) |
| **Patch SE (zero-init)** | `--patch-se` — starts as identity, learns channel gating |
| **MDM masking** | `--mdm-mask-ratio`, schedules, inpaint-friendly training |

</details>

<details open>
<summary><strong>Sampling & polish</strong></summary>

| Feature | Notes |
|:--------|:------|
| **CFG + rescale / dynamic threshold** | High-CFG friendly; see quality docs |
| **Img2img / inpaint / from-z** | `--init-image`, `--mask`, `--init-latent`, `--inpaint-mode` |
| **LoRA & control** | `--lora`, `--control-image`, style via `--style` or `--auto-style-from-prompt` |
| **Test-time pick** | `--pick-best clip\|edge\|ocr\|combo` with `--num` |
| **RAE bridge** | Checkpoints can carry `rae_latent_bridge` for non-4ch RAE latents |

</details>

> **Model presets** live in `config/model_presets.py`; domain prompts in `config/prompt_domains.py`; caption pipeline in `data/t2i_dataset.py`.

---

## Quick start

| Step | Command |
|:-----|:--------|
| **1 · Env** | `pip install -r requirements.txt` |
| **2 · Smoke** | `python scripts/tools/quick_test.py` |
| **3 · Train** | `python train.py --data-path ... --results-dir results` |
| **4 · Sample** | `python sample.py --ckpt .../best.pt --prompt "..." --out out.png` |

```bash
cd sdx
pip install -r requirements.txt
python scripts/tools/quick_test.py    # env smoke test (no dataset)
```

**Train** (single GPU):

```bash
python train.py --data-path /path/to/image_folders --results-dir results
```

**Sample**:

```bash
python sample.py --ckpt results/.../best.pt --prompt "your prompt" --steps 50 --width 256 --height 256 --out out.png
```

**Multi-GPU**:

```bash
torchrun --nproc_per_node=4 train.py --data-path /path/to/data --global-batch-size 256
```

---

## Setup

Run commands from the **repo root** (`sdx/`) so `config`, `data`, `diffusion`, `models`, and `utils` import correctly.

| Topic | Where |
|:------|:------|
| **Code layout & conventions** | [docs/CODEBASE.md](docs/CODEBASE.md) · [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Format / lint** | `pip install ruff` → `ruff format .` · `ruff check .` (see `pyproject.toml`) |
| **Hardware & storage** (VRAM tiers, huge booru-scale data) | [docs/HARDWARE.md](docs/HARDWARE.md) |
| **HF gated models** | Copy `.env.example` → `.env`, set `HF_TOKEN` |
| **Download weights** (T5, VAE, optional CLIP/LLM) | `python scripts/download/download_models.py --all` → `model/` |
| **Curated stack** (T5 + CLIP + DINOv2 + Qwen + Cascade, optional) | `python scripts/download/download_revolutionary_stack.py` — see [docs/MODEL_STACK.md](docs/MODEL_STACK.md) |
| **Optional native tools** (Rust/Zig/C++/Go/Node) | [native/README.md](native/README.md) |

**Clone reference repos** (optional, for reading upstream code):

```bash
# Windows (PowerShell)
.\scripts\setup\clone_repos.ps1

# Linux / macOS
./scripts/setup/clone_repos.sh
```

Pulls **DiT**, **ControlNet**, **flux**, **Stability-AI/generative-models** into `external/`. Runtime is **pip-only**; clones are for reference.

---

## Documentation hub

| | Doc | Purpose |
|:-:|----|:--------|
| 📚 | [docs/README.md](docs/README.md) | Index of all docs |
| 🗺️ | [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) | Roadmap, quality ideas, what’s implemented |
| ⚙️ | [docs/HOW_GENERATION_WORKS.md](docs/HOW_GENERATION_WORKS.md) | Prompt → T5 → DiT → VAE → image |
| 🔗 | [docs/CONNECTIONS.md](docs/CONNECTIONS.md) | Config ↔ data ↔ checkpoint ↔ sample |
| ✨ | [docs/CIVITAI_QUALITY_TIPS.md](docs/CIVITAI_QUALITY_TIPS.md) | CFG, hands, resolution, oversaturation |
| 🧩 | [docs/AR.md](docs/AR.md) | Block AR: 0 vs 2 vs 4 |
| 🎨 | [docs/STYLE_ARTIST_TAGS.md](docs/STYLE_ARTIST_TAGS.md) | Style / artist extraction |
| 💡 | [docs/INSPIRATION.md](docs/INSPIRATION.md) | Upstream repos & ideas |
| 📂 | [docs/FILES.md](docs/FILES.md) | File map |
| 📝 | [docs/REGION_CAPTIONS.md](docs/REGION_CAPTIONS.md) | JSONL `parts` / `region_captions` for layout-aware training text |
| 🧱 | [docs/MODEL_STACK.md](docs/MODEL_STACK.md) | Local `model/` paths, triple encoders, Qwen, Stable Cascade |
| 🎲 | [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Seeds & determinism |

---

## Data format

### Option A — Folder layout

- `data_path` = directory with subdirs; images + sidecar captions.
- For `img.png`, add `img.txt` or `img.caption` (one caption per file).

### Option B — JSONL manifest

One JSON object per line, e.g.  
`{"image_path": "/path/to/img.png", "caption": "your caption"}`

Use `--manifest-jsonl /path/to/manifest.jsonl` (and you can leave `--data-path` empty per your workflow).

**Regional / layout labels** (optional): add **`parts`** (dict) and/or **`region_captions`** (list) so T5 sees *who/what/where* per region merged after the global `caption` (default `[layout] …`). Helps composition and part-level grounding without changing DiT. See **[docs/REGION_CAPTIONS.md](docs/REGION_CAPTIONS.md)** and `--region-caption-mode append|prefix|off`.

### Caption tips

- **Tags**: `1girl, long hair, outdoors` — `()` / `(())` emphasis, `[]` de-emphasis; subject moved forward when possible.
- **Quality tags**: `masterpiece`, `best quality`, `highres` are boosted in the dataset pipeline.
- **Negative**: JSONL `negative_caption` / `negative_prompt`, or **second line** in `.txt`.
- **Domains** (3D, photoreal, interior, etc.): see [docs/DOMAINS.md](docs/DOMAINS.md) and [docs/MODEL_WEAKNESSES.md](docs/MODEL_WEAKNESSES.md).

**Utilities**: `python scripts/tools/ckpt_info.py results/.../best.pt` — config + step info.

---

## Training

**Block AR** (see [docs/AR.md](docs/AR.md)):

```bash
python train.py --data-path /path/to/data --num-ar-blocks 2
```

**Passes (recommended)**:

```bash
python train.py --data-path /path/to/data --passes 3
# Optional cap: --max-steps 100000
```

**Stronger variants**

- **`--model DiT-P/2-Text`** — QK-norm, SwiGLU, AdaLN-Zero, larger default width/depth.
- **`--model DiT-Supreme/2-Text`** — RMSNorm, QK-norm in self+cross, SwiGLU, optional `--size-embed-dim`.

**Triple text encoders** (T5 + CLIP-L + CLIP-bigG with trainable fusion — matches a full `model/` download):

```bash
python train.py --data-path /path/to/data --text-encoder-mode triple
```

Use **`--text-encoder`**, **`--clip-text-encoder-l`**, **`--clip-text-encoder-bigg`** to override paths; empty defaults use `utils/model_paths.py` (local folders first).

**Avoid overtraining** — use val loss as the quality signal:

```bash
python train.py --data-path /path/to/data --passes 5 \
  --val-split 0.05 --val-every 2000 --early-stopping-patience 3
```

Use **`best.pt`** for inference.

**Refinement training**: `--refinement-prob` (default 0.25), `--refinement-max-t`.  
**Inference refinement**: `inference.py` / sample refinements; **`--allow-imperfect`** skips refinement where applicable.

**No xformers**:

```bash
python train.py --data-path /path/to/data --no-xformers
```

---

## Sampling

```bash
python sample.py --ckpt .../best.pt --prompt "..." --negative-prompt "..." \
  --steps 50 --width 256 --height 256 --out out.png
```

**Often-used flags**: `--cfg-scale`, `--cfg-rescale`, `--scheduler ddim|euler`, `--num N`, `--grid`, `--vae-tiling`, `--deterministic`, `--style`, `--auto-style-from-prompt`, `--control-image`, `--lora`, `--init-image`, `--mask`, `--inpaint-mode legacy|mdm`, `--sharpen`, `--contrast`, `--preset`, `--op-mode`, `--pick-best`, `--no-refine`.

**Prompt tricks**: `(word)` / `[word]` emphasis in `sample.py`; `--tags` / `--tags-file`; `--gender-swap`, anatomy/object/scene scales, `--character-sheet` JSON.

**OCR repair**: e.g. `--expected-text "OPEN" --ocr-fix` (pytesseract + masked inpaint).

**Programmatic load**: `python inference.py --ckpt .../best.pt` (use **`--allow-imperfect`** for raw output).

**Book / manga**: [scripts/book/generate_book.py](scripts/book/generate_book.py) — multi-page, optional face/speech-bubble anchoring, OCR.

```powershell
python scripts/book/generate_book.py --ckpt "C:\path\best.pt" `
  --output-dir out_book --book-type manga --model-preset anime `
  --prompts-file pages.txt --expected-text "OPEN" --ocr-fix --ocr-iters 2 `
  --anchor-face --edge-anchor --anchor-speech-bubbles
```

`pages.txt` per-line optional OCR override: `prompt text here|||OPEN`

---

## Data (JSONL) fields

| Field | Description |
|:------|:------------|
| `caption` | Positive prompt (required) |
| `negative_caption` / `negative_prompt` | Concepts to avoid |
| `parts` | Dict of regional descriptions (`subject`, `clothing`, `background`, …) merged into training caption |
| `region_captions` / `segments` | List of strings or `{label, text}` — merged with `parts` when both set |
| `style` | Style text; blended with `style_strength` |
| `control_image` / `control_path` | Control image path |
| `init_image` / `init_image_path` / `source_image` | Img2img source (with `--img2img-prob` in training) |

For `.txt`: line 1 = positive, line 2 = negative.

---

## Train CLI (quick reference)

<details>
<summary><strong>Click to expand full option table</strong></summary>

| Flag | Default | Description |
|:-----|:--------|:------------|
| `--data-path` | (required*) | Image/caption root |
| `--manifest-jsonl` | None | JSONL manifest |
| `--negative-prompt-weight` | 0.5 | Negative conditioning scale |
| `--model` | DiT-XL/2-Text | DiT-B/L/XL, DiT-P*, DiT-Supreme* |
| `--image-size` | 256 | Train resolution (latent = ÷8) |
| `--global-batch-size` | 128 | Total batch (all GPUs) |
| `--passes` | 0 | Full-dataset passes |
| `--max-steps` | 0 | Step cap |
| `--epochs` | 100 | If passes and max-steps are 0 |
| `--lr` | 1e-4 | Learning rate |
| `--num-workers` | 8 | DataLoader workers |
| `--no-bf16` | False | Disable bf16 |
| `--no-compile` | False | Disable torch.compile |
| `--no-grad-checkpoint` | False | Disable checkpointing |
| `--num-ar-blocks` | 0 | AR: 0, 2, or 4 |
| `--no-xformers` | False | PyTorch SDPA fallback |
| `--min-lr` | 1e-6 | Cosine floor |
| `--refinement-prob` | 0.25 | Refinement training probability |
| `--refinement-max-t` | 150 | Refinement t cap |
| `--no-save-best` | False | Disable best-by-train-loss ckpt |
| `--beta-schedule` | linear | `linear` or `cosine` |
| `--prediction-type` | epsilon | `epsilon` or `v` |
| `--noise-offset` | 0 | SD-style noise offset |
| `--min-snr-gamma` | 5 | Min-SNR weighting (0=off) |
| `--resume` | None | Resume path |
| `--val-split` | 0 | Val fraction |
| `--val-every` | 2000 | Val frequency |
| `--early-stopping-patience` | 0 | Early stop (0=off) |
| `--val-max-batches` | None | Cap val batches |
| `--deterministic` | False | Reproducible mode |
| `--latent-cache-dir` | None | Precomputed latents |
| `--img2img-prob` | 0 | Img2img training prob |
| `--mdm-mask-ratio` | 0 | MDM patch mask ratio |
| `--mdm-mask-schedule` | None | `t,r,...` schedule |
| `--mdm-patch-size` | 2 | MDM patch size |
| `--mdm-min-mask-patches` | 1 | Min masked patches |
| `--no-mdm-loss-only-masked` | False | Loss on full latent |
| `--moe-num-experts` | 0 | MoE experts |
| `--moe-top-k` | 2 | MoE top-k |
| `--moe-balance-loss-weight` | 0 | Router balance loss |
| `--text-encoder` | auto | T5 path or HF id; empty → `model/T5-XXL` if present |
| `--text-encoder-mode` | t5 | `t5` or `triple` (T5 + CLIP-L + CLIP-bigG fusion) |
| `--clip-text-encoder-l` | "" | CLIP-L path (triple mode) |
| `--clip-text-encoder-bigg` | "" | CLIP-bigG path (triple mode) |

</details>

---

## SD / SDXL-style features

| Feature | Flag | Description |
|:--------|:-----|:------------|
| Offset noise | `--noise-offset` | Light/dark balance |
| Min-SNR | `--min-snr-gamma` | Timestep loss balance |
| V-pred | `--prediction-type v` | Velocity parameterization |
| Cosine β | `--beta-schedule cosine` | Alternative noise schedule |

Sampling: DDIM-style loop with cond/uncond; use **`--cfg-rescale`**, **`--num`**, **`--vae-tiling`** as needed.

---

## Extra features

<details>
<summary><strong>Expand: resume, logging, export, ViT-Gen, REPA, RAE, book workflow, …</strong></summary>

| Feature | How |
|:--------|:----|
| Resume | `--resume path/to/best.pt` |
| Sample weights | JSONL `weight` / `aesthetic_score` |
| Crops | `--crop-mode center\|random\|largest_center` |
| Layout / regional text | JSONL `parts` / `region_captions`; `--region-caption-mode append\|prefix\|off`, `--region-layout-tag` — [docs/REGION_CAPTIONS.md](docs/REGION_CAPTIONS.md) |
| Caption dropout schedule | `--caption-dropout-schedule 0,0.2,10000,0.05` |
| Polyak average | `--save-polyak N` |
| WandB / TensorBoard | `--wandb-project`, `--tensorboard-dir` |
| Dry run | `--dry-run` |
| Log samples | `--log-images-every`, `--log-images-prompt` |
| Data quality script | `scripts/tools/data_quality.py` |
| Export ONNX | `scripts/tools/export_onnx.py` |
| Latent cache | `scripts/training/precompute_latents.py` + `--latent-cache-dir` |
| AdaGen / PBFM | `sample.py` `--ada-early-exit`, `--pbfm-edge-boost`, … |
| Test-time pick | `--num 4 --pick-best clip\|edge\|ocr\|combo` |
| RAE bridge | Train with RAE + bridge; ckpt stores `rae_latent_bridge` |
| Safetensors export | `scripts/tools/export_safetensors.py` |

</details>

---

## Styles, ControlNet, LoRA

- **Style**: Train with `--style-embed-dim 4096` (match T5 dim) + `style` in JSONL. Sample: `--style "..." --style-strength 0.7` or `--auto-style-from-prompt`.
- **Control**: Train `--control-cond-dim 1` + control paths in JSONL. Sample: `--control-image ... --control-scale 0.85`.
- **LoRA**: Sample-only: `--lora path.safetensors` or `path.pt:0.6`, optional `--lora-trigger`.

Keep strengths moderate (e.g. style 0.6–0.8, control 0.7–1.0) to avoid muddy blending.

---

## Img2img · inpainting · from-z

| Mode | Usage |
|:-----|:------|
| Img2img | `--init-image path.png --strength 0.75` |
| Inpaint | `--init-image ref.png --mask mask.png` — try `--inpaint-mode mdm` |
| From latent | `--init-latent z.pt --strength 0.8` |
| Output size | `--width` / `--height` (decode/resize) |
| Train img2img | JSONL `init_image` + `--img2img-prob 0.2` |

---

## Project layout

<details>
<summary><strong>Directory tree</strong></summary>

```
sdx/
├── config/           # TrainConfig, model_presets, domains, style artists
├── data/             # Text2ImageDataset, caption_utils
├── diffusion/        # gaussian_diffusion, schedules
├── docs/             # All markdown docs
├── model/            # Downloaded weights (gitignored)
├── models/           # dit_text, attention, controlnet, moe, cascaded_multimodal_diffusion, …
├── ViT/              # Quality scoring, prompt breakdown, EMA/ranking tools
├── native/           # Optional Rust/Zig/C++/Go helpers
├── scripts/
│   ├── setup/        # clone external refs
│   ├── download/     # HF download scripts
│   ├── training/     # precompute_latents, self_improve, …
│   └── tools/        # ckpt_info, export_*, quick_test, …
├── utils/
├── train.py
├── sample.py
├── inference.py
├── requirements.txt
└── README.md
```

</details>

---

## References

| Project | Link | Role |
|:--------|:-----|:-----|
| **DiT** | [facebookresearch/DiT](https://github.com/facebookresearch/DiT) | Transformer diffusion baseline |
| **ControlNet** | [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) | Structural conditioning |
| **FLUX** | [black-forest-labs/flux](https://github.com/black-forest-labs/flux) | Modern diffusion reference |
| **generative-models** | [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models) | SD3-era reference |

Local clones: `external/` after running the clone scripts.

---

## License

Licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE).

---

<div align="center">

---

### SDX

**Train on your data · sample with the stack you choose**

[↑ Back to top](#sdx)

</div>
