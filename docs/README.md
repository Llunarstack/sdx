# SDX documentation

Quick links to all project docs, grouped by purpose.

---

## Project map & architecture

| Doc | Description |
|-----|--------------|
| [../pipelines/README.md](../pipelines/README.md) | **Two product lines:** `image_gen/` vs `book_comic/` (shared `train.py`; split docs + book script). |
| [SMOKE_TRAINING.md](SMOKE_TRAINING.md) | Minimal `train.py` run: synthetic data + small DiT + `--dry-run`. |
| [DANBOORU_HF.md](DANBOORU_HF.md) | Hugging Face Danbooru-style data → JSONL + `train.py`; one-shot `hf_download_and_train.py`. |
| [CODEBASE.md](CODEBASE.md) | **Start here for code**: layers, conventions, ruff/pytest, where to edit. |
| [CODEBASE_ORGANIZATION.md](CODEBASE_ORGANIZATION.md) | **Repo structure rules:** layers, where to add code, what not to move without a migration. |
| [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) | Recent diffusion / flow ideas vs what SDX implements (timestep sampling, roadmap). |
| [ARCHITECTURE_SHIFT_2026.md](ARCHITECTURE_SHIFT_2026.md) | **Post-diffusion era:** flow matching, bridges, hybrid AR+DiT, Mamba, DMD, RAE — mapped to SDX ([`utils/architecture_map.py`](../utils/architecture_map.py)). |
| [WORKFLOW_INTEGRATION_2026.md](WORKFLOW_INTEGRATION_2026.md) | **Workflow + efficiency narratives:** coherency/4K, LLaDA-class ideas, test-time compute, live grounding, Mamba — **disclaimers** + SDX mapping ([`utils/architecture_map.py`](../utils/architecture_map.py)). |
| [FILES.md](FILES.md) | File map: every SDX file and key external references. |
| [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) | **Start here to navigate:** top-level tree, `scripts/` layout, where to add code. |
| [../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) | **Auto-generated** full tree (`scripts/tools/update_project_structure.py`). |
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
| [LANDSCAPE_2026.md](LANDSCAPE_2026.md) | **2026 industry context:** production-grade gen, authenticity, system-of-experts pipelines, 4K/aspect, text-in-image, grounding — mapped to SDX ([utils/orchestration.py](../utils/orchestration.py)). |
| [ARCHITECTURE_SHIFT_2026.md](ARCHITECTURE_SHIFT_2026.md) | **Research / architecture:** flow matching, diffusion bridges, hybrid AR+DiT, Mamba, distillation, semantic latents — vs SDX ([`utils/architecture_map.py`](../utils/architecture_map.py)). |
| [WORKFLOW_INTEGRATION_2026.md](WORKFLOW_INTEGRATION_2026.md) | **Industry workflow commentary** (efficiency, grounding, test-time compute) — **disclaimers** + SDX hooks ([`utils/architecture_map.py`](../utils/architecture_map.py)). |
| [BOOK_COMIC_TECH.md](BOOK_COMIC_TECH.md) | Sequential art: consistency, GLIGEN/Control-style ideas, lettering, **prompt_lexicon** + `generate_book` flags. |
| [BOOK_MODEL_EXCELLENCE.md](BOOK_MODEL_EXCELLENCE.md) | “Best book output” checklist: data, training, `--book-accuracy production`, pick-best, OCR/anchoring. |
| [../ViT/EXCELLENCE_VS_DIT.md](../ViT/EXCELLENCE_VS_DIT.md) | **ViT/ vs DiT:** scoring stack vs generator; Swin-DiT, FiT, reward/IQA papers; timm backbone presets. |
| [IMPROVEMENTS.md](IMPROVEMENTS.md) | Roadmap: quality, fixes, novel ideas — includes **§11 Next-tier / insane quality** and **§12 Industry alignment (2026)**. |
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
- `scripts/tools/training_timestep_preview.py` — Preview histograms for `--timestep-sample-mode` (uniform / logit_normal / high_noise) before long training runs.
- `scripts/tools/dit_variant_compare.py` — Parameter counts and FP32/BF16 GiB estimates for DiT / EnhancedDiT registry names.
- `scripts/tools/vit_inspect.py` — Inspect ViT quality checkpoints (config + optional module tree via `utils/nn_inspect.py`).
- `scripts/tools/op_preflight.py` — One-shot “coverage + thresholds” gate (PASS/FAIL) before training.
- `scripts/tools/complex_prompt_coverage.py` — Check coverage for clothes/weapons/food/text/foreground/background/weird/NSFW categories.
- `scripts/tools/prompt_gap_scout.py` — Analyze a single prompt and suggest missing tricky category keywords.

Run from repo root so `config`, `data`, `diffusion`, `models`, `utils` are on `sys.path`.
