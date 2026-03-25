# Next-gen image generation: four design pillars (and SDX mapping)

**Purpose:** Capture a **reasoning-first** view of image generation (not only denoising) and map each pillar to **what exists in SDX today**, **partial hooks**, and **real gaps**. This is a **design document**, not a promise of a trained “super-model.”

**Core tension:** *Creative flexibility* (diverse, plausible samples) vs *logical precision* (prompt geometry, counts, colors, text). Pure latent denoisers behave like strong **statistical imitators**; moving “beyond guessing” usually means **extra structure** (planning, critique, frequency/objective changes, or preference learning).

---

## 1. Semantic–geometric dual backbone (e.g. global planner + DiT)

**Idea:** A **long-context, subquadratic** module (industry narrative: **Mamba / SSM**) consumes the prompt and produces a **global layout prior** (where objects, light, relations live). A **DiT** then **renders** detail conditioned on that prior.

**Why it’s compelling:** Transformers scale **O(n²)** in sequence length; very long prompts + global spatial reasoning stress vanilla DiT cross-attn. A **linear-sequence** planner is a natural research direction for “red ball left ↔ mirror right” style constraints.

**SDX today (partial, not a separate Mamba stack):**

| Mechanism | Role |
| :--- | :--- |
| **`--num-ar-blocks`** + block masks | Raster / block **causal** bias in the DiT (hybrid AR + diffusion angle). |
| **`--dual-stage-layout`** | **Coarse latent** pass → upscale → **detail** pass (layout-first *workflow*, same DiT). |
| **`ssm_every_n` / ViT-Gen SSM swap** | Lightweight **SSM-like token mixer** in some DiT blocks — **not** a full Mamba “architect” network. |
| **`size_embed_dim`**, **control**, **scene blueprint** text | Conditioning and prompt structure — not a learned global latent map. |

**Gap for “full” pillar:** A **dedicated** Mamba (or other SSM) **planner** that outputs a **tensor blueprint** fused into every DiT layer (or into a separate low-res latent) is **not** implemented.

---

## 2. In-loop VLM / discriminative correction

**Idea:** Every *k* steps, a **vision–language model** compares **current latents / decodes** to the prompt and applies a **gradient or latent correction** so mistakes (wrong color, extra fingers) don’t get “polished in place.”

**Why it’s compelling:** Diffusion is **path-dependent**; early mistakes are expensive to undo. A **critic** is the “step back and check the reference” loop.

**SDX today (partial — no full VLM, no autograd critic in the loop):**

| Mechanism | Role |
| :--- | :--- |
| **`--clip-guard-threshold`** + **`utils/generation/clip_alignment.py`** | **CLIP** image–text cosine; optional **short second `sample_loop`** if score is low (coarse alignment, not segmentation). |
| **`--volatile-cfg-boost`** | Heuristic **CFG bump** when latent updates are **spiky** vs recent steps (cheap “unstable step” signal). |
| **`--pick-best`** (`utils/quality/test_time_pick.py`) | **Post-hoc** multi-sample scoring (CLIP / edge / OCR / combos). |
| **Refine / hires / SAG / PBFM** | Quality and structure nudges without a VLM. |

**Gap for “full” pillar:** A **mini VLM** (or detector) on a **decode preview** or **auxiliary head**, with **differentiable** or **latent-space** steering **inside** the denoising loop, is **not** wired. That requires model API hooks, memory budget, and training or frozen critic calibration.

---

## 3. Fourier / operator view (“infinite” resolution narrative)

**Idea:** Operate in **frequency** or **function** space (neural operators) so **global** low frequencies and **local** high frequencies are explicit; **resolution** can be reframed as **bandwidth** / sampling of the same underlying field.

**Why it’s compelling:** Fixed-grid CNN/ViT biases can **tile** or lose coherence when extrapolating resolution; frequency-aware objectives can emphasize **layout at high noise** and **texture at low noise**.

**SDX today (partial — training-side, not full neural operator diffusion):**

| Mechanism | Role |
| :--- | :--- |
| **`--spectral-sfp-loss`** + `diffusion/spectral_sfp.py` | **FFT-weighted** training loss on pred−target in **latent** space; timestep-dependent radial weights. |
| **`--dual-stage-layout` / `--hires-fix`** | **Multi-scale** inference paths (not the same as continuous resolution operators). |

**Gap for “full” pillar:** **NOD-style** denoising **directly** on Fourier coefficients with **zero-shot** arbitrary output grids is **not** implemented. SFP is a **loss shaping** tool on the **existing** VP latent grid, not a replacement forward process.

---

## 4. Direct preference alignment (DPO-style on images / latents)

**Idea:** Train on **pairs** *(A, B)* with human or teacher labels (“A follows prompt better”, “B is more aesthetic”) so the model **optimizes preferences**, not only likelihood under a static dataset.

**Why it’s compelling:** “Quality” is **subjective** and **multi-objective**; pairwise feedback matches how humans judge images.

**SDX today:** Standard **supervised** diffusion in **`train.py`**. Stage-2 **Diffusion-DPO** runner: **`scripts/tools/training/train_diffusion_dpo.py`** (frozen ref DiT, shared `t` + noise, **`dpo_preference_loss`** on per-sample VP losses). Helpers: **`utils/training/diffusion_dpo_loss.py`**, **`preference_jsonl.py`**, **`preference_image_dataset.py`**. **`GaussianDiffusion.per_sample_training_losses`** supports the DPO term.

**Gap:** Optional **triple-encoder** / CLIP-bundle path in the DPO script; **EMA-specific** reference checkpoints; full paper-faithful variants (e.g. multiple noise draws).

---

## Comparison table (qualitative — not verified benchmarks)

| Pillar | Main failure mode addressed | SDX status |
| :--- | :--- | :--- |
| Planner + DiT | Weak long-range layout / composition | **Partial** (AR blocks, dual-stage, SSM swap, conditioning) |
| VLM critic | Wrong semantics locked in early | **Partial** (CLIP guard, volatile CFG, pick-best) |
| Fourier / operator | Resolution / global coherence | **Partial** (spectral SFP loss; not full NOD) |
| DPO / preferences | “Average” or bland dataset mimicry | **Partial** (loss + JSONL loader; no integrated trainer) |

---

## Suggested build order (if you implement for real)

1. **Stronger layout priors** — dual-stage + AR + data with captions that stress relations; optional **auxiliary layout** head.  
2. **Cheap critic** — CLIP guard + pick-best + orchestration; graduate to **small VLM** only if metrics justify cost.  
3. **Spectral / multi-scale losses** — ablate SFP vs baseline on your domain.  
4. **Preference fine-tune** — once base model is stable, add pairwise data and DPO-class objective.

---

## See also

- [`ARCHITECTURE_SHIFT_2026.md`](ARCHITECTURE_SHIFT_2026.md) — broader 2025–2026 themes.  
- [`PROMPT_ACCURACY_BLUEPRINT.md`](PROMPT_ACCURACY_BLUEPRINT.md) — GLS blueprint, in-loop discriminative loops, frequency/FNO narrative (prompt-accuracy framing).  
- [`MODERN_DIFFUSION.md`](MODERN_DIFFUSION.md) — VP vs flow, SFP, timestep sampling.  
- [`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py) — machine-readable theme IDs (includes **nextgen_*** entries).
