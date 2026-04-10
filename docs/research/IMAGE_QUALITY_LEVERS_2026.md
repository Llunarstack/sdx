# Image quality: research levers (2025–2026) + SDX map

There is no single trick for a “perfect” image model. Quality is a **stack**: data → training objective → capacity → sampler → guidance → evaluation → iteration. This note compresses **recent research directions** (with primary links) and maps them to **what SDX already exposes** vs **what would be new work**.

---

## A. Inference / sampling (often the biggest ROI)

| Idea | Why it matters | Pointers | SDX today |
|------|----------------|----------|-----------|
| **CFG rescale / dynamic CFG** | High CFG improves prompt adherence but causes saturation, contrast blow-ups, and texture collapse. | Oversaturation analysis [arXiv:2410.02416](https://arxiv.org/abs/2410.02416); dynamic CFG with feedback [arXiv:2509.16131](https://arxiv.org/abs/2509.16131) | `--cfg-rescale` on `sample.py`; Holy Grail schedules per-step CFG in `diffusion/holy_grail` / `sampling_extras`. |
| **Geometry-aware CFG for flows** | Standard CFG on rectified-flow velocities can push states **off the data manifold** → artifacts. | Rectified-CFG++ [project site](https://rectified-cfgpp.github.io/) | If you train with flow matching, pair **`flow_matching_sample`** paths in `diffusion/gaussian_diffusion.py` with conservative CFG + rescale; **native Rectified-CFG++** would be new implementation work. |
| **Fast schedules that preserve geometry** | Step count interacts with trajectory curvature; “obvious” schedules waste budget early/late. | TORS / fast sampling analysis [arXiv:2603.00763](https://arxiv.org/abs/2603.00763) | Experiment with `--steps`, solver choice (`ddim` / `heun`), and flow solver (`euler` / `heun`). Automating **learned or TORS-like** schedules → research feature. |
| **Frequency-decoupled guidance (FDG)** | Apply different guidance strength to low vs high frequency components to reduce CFG side effects. | [arXiv:2506.19713](https://arxiv.org/abs/2506.19713) | Not bundled as FDG; would need spectral decomposition in the denoise update. |
| **Adaptive projected guidance (APG)** | Decompose CFG into parallel vs orthogonal components; down-weight the part that drives oversaturation. | [arXiv:2410.02416](https://arxiv.org/abs/2410.02416) related line (APG cited in community summaries) | Would be **new** sampler math; Holy Grail is the right hook point for per-step blending. |

**Practical inference checklist (no new code):** sweep `--cfg-scale` **down** before chasing architecture; add **`--cfg-rescale 0.6–0.85`**; increase steps if mid-frequency detail is mushy; use `docs/QUALITY_AND_ISSUES.md` for Civitai-style failure modes.

---

## B. Training objective & optimization

| Idea | Why it matters | Pointers | SDX today |
|------|----------------|----------|-----------|
| **Rectified flow / flow matching training** | Straighter paths → fewer NFE at similar quality if the field is well trained. | Microsoft “Improving the Training of Rectified Flows”; MeanFlow [arXiv:2511.23342](https://arxiv.org/abs/2511.23342) | Flow training + sampling hooks exist in `diffusion/flow_matching.py` and `gaussian_diffusion.py`. |
| **Perceptual / semantic auxiliaries** | Pure MSE on latents underweights structure humans care about. | LPIPS-style diffusion losses [arXiv:2401.00110](https://arxiv.org/abs/2401.00110); PixelGen (LPIPS + DINO) [arXiv:2602.02493](https://arxiv.org/abs/2602.02493) | **REPA** and related alignment hooks in training (`--repa-weight`, see `train.py` / config). Full pixel-space LPIPS in latent DiT training → optional extension. |
| **Preference / DPO-style alignment** | Moves the model toward **human-preferred** samples beyond likelihood. | Diffusion-SDPO [arXiv:2511.03317](https://arxiv.org/abs/2511.03317); Curriculum DPO (CVPR 2025); Rethinking DPO [arXiv:2505.18736](https://arxiv.org/abs/2505.18736) | `gaussian_diffusion` contains **Diffusion-DPO** hooks; treat as **advanced**—mis-tuned preference training can **hurt** base quality (see SDPO motivation). |

---

## C. Data & captions (the silent multiplier)

| Idea | Why it matters | Pointers | SDX today |
|------|----------------|----------|-----------|
| **Autocuration / online selection** | Better examples per step can beat bigger but noisier corpora. | Autoguided curation [arXiv:2509.15267](https://arxiv.org/abs/2509.15267) | Use tooling for **hard-case mining / manifests** (`scripts/tools`, book pipeline) as lightweight curation; full online JEST-style selection is not default. |
| **Quality as conditioning (not only filtering)** | Throwing away weak data loses coverage; conditioning on quality scores can preserve diversity. | LACON [arXiv:2603.26866](https://arxiv.org/abs/2603.26866) | Would need dataset fields + model conditioning; **greenfield** relative to stock DiT text conditioning. |
| **Caption hygiene** | Wrong or lazy captions teach the wrong associations. | Industry curation writeups (e.g. Datology multimodal curation blog) | Caption utilities, prompt domains, tag reordering in `sample.py` / `data/`—improve **your** captions before chasing new arch. |

---

## D. Evaluation (if you cannot measure it, you cannot optimize it)

- **Human / pairwise** remains gold for “looks good”; automate with **fixed protocols** (same seeds, same prompts).
- **Model-based metrics** (CLIP alignment, aesthetic classifiers, OCR for text-in-image) are **gameable**—use as **diagnostics**, not the only objective.
- SDX: **`utils/quality/test_time_pick.py`**, Holy Grail pick metrics, **TCIS** (`docs/TCIS_OVERVIEW.md`) for multi-candidate ranking ideas.

---

## E. “Insane tier” backlog (high effort, high upside)

Ordered roughly by **impact if executed well** vs **engineering cost**:

1. **Manifold-safe CFG for flow models** (Rectified-CFG++-class) in the flow sampler loop.  
2. **APG / FDG** as optional guidance modes in `sampling_extras` / Holy Grail.  
3. **Learned or optimized timestep schedules** (TORS-class) exposed as presets.  
4. **Safeguarded preference training** (SDPO-class) on top of your base DiT.  
5. **Quality-conditioned training** (LACON-class) with explicit metadata channels.  
6. **Joint pixel+latent perceptual losses** where VRAM allows (PixelGen-class intuition, adapted to your stack).

---

## F. What SDX will not magically fix

- A **bad or tiny dataset** will cap any architecture.  
- **Extreme CFG** without rescale / schedule will **always** fight you.  
- **No compute** means fewer ablations; prioritize **inference sweeps** and **small controlled train runs** over rewriting the world.

---

## See also

- [`docs/QUALITY_AND_ISSUES.md`](../QUALITY_AND_ISSUES.md) — operational fixes (blur, saturation, hands, text).  
- [`docs/HOLY_GRAIL_OVERVIEW.md`](../HOLY_GRAIL_OVERVIEW.md) — adaptive sampling.  
- [`docs/TCIS_OVERVIEW.md`](../TCIS_OVERVIEW.md) — multi-candidate + ViT critique loop.  
- [`docs/recipes/quick_eval_holy_grail.md`](../recipes/quick_eval_holy_grail.md) — minimal eval recipe.

---

*Disclaimer: links point to papers/projects for follow-up reading; implementation in SDX is not implied unless listed in the “SDX today” column.*