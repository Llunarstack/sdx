# SDX documentation

Quick links to all project docs, grouped by purpose.

---

## Project map & architecture

| Doc | Description |
|-----|--------------|
| [FILES.md](FILES.md) | File map: every SDX file and key external references. |
| [CONNECTIONS.md](CONNECTIONS.md) | How config, data, and models connect (train → checkpoint → sample). |
| [HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md) | Generation pipeline: prompt → T5 → diffusion loop → DiT → VAE → image; AR and ported code. |

---

## Operations & features

| Doc | Description |
|-----|--------------|
| [HARDWARE.md](HARDWARE.md) | PC specs, VRAM, storage, latent cache. |
| [AR.md](AR.md) | Block-wise autoregressive (AR): 0 vs 2 vs 4 blocks, when to use. |
| [STYLE_ARTIST_TAGS.md](STYLE_ARTIST_TAGS.md) | Style/artist tags (PixAI, Danbooru): extraction, training, `--auto-style-from-prompt`. |
| [DOMAINS.md](DOMAINS.md) | 3D, realistic, interior/exterior domains. |
| [MODEL_WEAKNESSES.md](MODEL_WEAKNESSES.md) | Hands, faces, text, composition: causes and fixes. |
| [COMMON_ISSUES.md](COMMON_ISSUES.md) | Community issues (SDXL, Flux, Illustrious, NoobAI, Z-Image): concept bleed, plastic skin, repetitive face, artifacts, watermarks, CFG burn, etc. — mitigations and sample.py flags. |
| [CIVITAI_QUALITY_TIPS.md](CIVITAI_QUALITY_TIPS.md) | Civitai-style tips: oversaturation, blur, hands, conflict resolution, text-in-image, hard styles, naturalize. |

---

## Roadmap & inspiration

| Doc | Description |
|-----|--------------|
| [IMPROVEMENTS.md](IMPROVEMENTS.md) | Roadmap: quality, fixes, novel ideas, “ideas to add next”. |
| [INSPIRATION.md](INSPIRATION.md) | What we take from PixAI, ComfyUI, and cloned repos. |
| [PROMPT_COOKBOOK.md](PROMPT_COOKBOOK.md) | Copy‑paste prompt recipes using presets, op‑modes, hard styles, and all the quality flags. |

---

## Ecosystem, packages, and frameworks we lean on

SDX is designed to feel familiar if you’ve used other diffusion ecosystems. We explicitly take inspiration from:

- **Python / ML stack**
  - `torch` — core training and inference.
  - `transformers` — T5 text encoder.
  - `diffusers` — reference for schedulers, CFG tricks, SDXL‑style options.
  - `safetensors` — safe, faster checkpoint / LoRA loading.
  - `xformers` — memory‑efficient attention (self + cross).
  - `numpy`, `scipy`, `Pillow` — image & post‑processing utilities.

- **UI / workflow ecosystems (inspiration)**
  - **AUTOMATIC1111** — prompt patterns, Hi‑Res Fix, face restoration workflows.
  - **ComfyUI / Forge** — graph‑style pipelines, CFG rescale ideas, ControlNet practices.
  - **Civitai** — community feedback on oversaturation, artifacts, LoRA quality, orange/green tints.
  - **PixAI / Flux / Z‑Image** — style control, hard‑style domains, “realism standard”, creative vs rigid behavior.

These are **not all hard dependencies** of SDX, but they inform our defaults, presets, and docs so the model plays nicely with existing tools and best practices.

---

## Maintenance / sanity checks

- `scripts/tools/smoke_imports.py` — Import smoke-test for internal modules (catches broken imports early).
- `scripts/tools/tag_coverage.py` — Scan a JSONL manifest for hard-style/person/anatomy/concept-bleed tag coverage.
- `scripts/tools/spatial_coverage.py` — Scan a JSONL manifest for spatial-wording coverage (`behind`, `next to`, `under`, `left of`, ...).
- `scripts/tools/op_preflight.py` — One-shot “coverage + thresholds” gate (PASS/FAIL) before training.
- `scripts/tools/complex_prompt_coverage.py` — Check coverage for clothes/weapons/food/text/foreground/background/weird/NSFW categories.
- `scripts/tools/prompt_gap_scout.py` — Analyze a single prompt and suggest missing tricky category keywords.

Run from repo root so `config`, `data`, `diffusion`, `models`, `utils` are on `sys.path`.
