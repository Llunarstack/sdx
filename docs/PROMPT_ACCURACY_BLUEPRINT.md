# Prompt-accurate generation: three structural ideas (and SDX mapping)

**Purpose:** Capture **technical** directions for beating typical DiT-style “layout and texture in one forward” behavior—without treating any row in the comparison table as a **measured** SDX result. DiTs **jointly** refine semantics and appearance; **decoupling**, **in-loop verification**, and **frequency-aware** objectives are levers to reduce **semantic bleeding**, **wrong placement**, and **global inconsistency**.

**Related:** [NEXTGEN_SUPERMODEL_ARCHITECTURE.md](NEXTGEN_SUPERMODEL_ARCHITECTURE.md) (overlapping pillars), [CONSISTENCY_FLOW_SPEED_BLUEPRINT.md](CONSISTENCY_FLOW_SPEED_BLUEPRINT.md), [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) (spectral SFP).

---

## 1. Structural blueprint — geometric–latent split (GLS)

**Problem:** When **composition** (what is where, which color belongs to which object) and **appearance** (texture, lighting) are entangled in the **same** deep blocks, errors propagate: e.g. “red cube on blue table” can **mix** hues because there is no hard boundary between “geometry solved” and “materials applied.”

**Idea — Geometric–Latent Split (GLS):** A **small, fast, spatially aware** module (often a shallow transformer or conv stack) produces a **low-resolution geometric prior** early in the process: e.g. **depth**, **surface normals**, **edges**, or a learned **layout tensor**. Later denoising is **conditioned** on this prior as a **constraint** (frozen or slowly updated) so that **object boundaries and relations** stabilize before **high-frequency** detail is filled in.

**Engineering variants:** Auxiliary heads on the DiT; a **two-phase** latent pass; explicit **edge/depth** supervision from synthetic or labeled data; or a **separate** network whose output is **concatenated** or **cross-attended** by the main denoiser.

**SDX today (partial):**

| Mechanism | Role |
| :--- | :--- |
| **`--dual-stage-layout`** | Coarse latent pass → upscale → detail pass (**workflow** layout-first, same DiT). |
| **`highfreq_layout_prior` / `--domain-prior-latent`** | Injects a **high-frequency / layout** prior in research hooks — not a full depth–normal stack. |
| **Control / scene blueprint text** | User- or data-side **structure**; not a learned immutable geometric tensor. |

**Gap:** No dedicated **Spatially-Aware Transformer** that outputs **depth + normals + edges** in the **first ~10% of steps** and **locks** the rest of sampling to that tensor as a **hard** constraint.

---

## 2. Recursive self-correction — discriminative denoising loops

**Problem:** Sampling is **causal in time**: a mistake at mid-noise is **expensive** to undo; the model often **refines** the wrong structure into a sharp but **wrong** image.

**Idea:** Periodically run a **frozen** strong **vision–language** model (or a specialized verifier) on a **decode of the current state** (or on auxiliary features). Compare **visual content** to the **prompt**. If there is a **detectable mismatch** (missing object, wrong attribute), apply a **targeted correction**: e.g. **re-noise** a **spatial mask**, **raise CFG** or **prompt strength** locally, or **rewind** a few steps and branch—**“local gradient re-roll”** in the narrative means **steered** updates in **latent space**, which in practice may be **heuristic** (mask + extra steps) unless the VLM is fully differentiable through the decoder.

**SDX today (partial):**

| Mechanism | Role |
| :--- | :--- |
| **`--clip-guard-threshold`** + [`utils/generation/clip_alignment.py`](../utils/generation/clip_alignment.py) | **CLIP** image–text similarity; optional **short** extra `sample_loop` — coarse **global** alignment, not structured VLM QA. |
| **`--clip-monitor-every`** + same CLIP model | **Mid-loop**: decode `x_0` pred every *N* steps; if cosine **<** `--clip-monitor-threshold`, multiply CFG by `(1 + --clip-monitor-cfg-boost)`. **Expensive**; not a full VLM or localized rewind. |
| **`--volatile-cfg-*`** | **Heuristic** instability signal on latent deltas — not semantic verification. |
| **`--pick-best`**, orchestration | **Post-hoc** multi-sample selection, not in-loop per-step VLM. |
| **`ViT/`** scorer | **Offline** or **post-generation** ranking — not inside the denoising trajectory. |

**Gap:** No **every-k-steps** frozen **VLM** with **spatially localized** rewind + re-denoise tied to **named** discrepancies; **“nearly 100% prompt adherence”** is an **aspirational** claim, not something to expect without heavy infrastructure and eval.

---

## 3. Frequency-domain diffusion — neural operators (FNO narrative)

**Problem:** Fixed **grid** latents bias models toward **local** mixing; **global** lighting and long-range consistency across a very large canvas are harder when each step only “sees” neighbors through a finite receptive field (even with attention, at extreme resolutions cost bites).

**Idea:** Treat the field as a **function** and operate partly in the **frequency domain** (e.g. **Fourier Neural Operators** or spectral blocks): **low** frequencies encode **global** layout and illumination; **high** frequencies encode **detail**. Denoising or velocity prediction in **spectral** space can emphasize **whole-image** coherence before filling **fine** structure.

**Caveats:** “**Infinite resolution**” and “**perfect** prompt accuracy at any size” are **not** guaranteed by FNOs alone—you still need **training scale**, **conditioning**, and **sampling** matched to the objective.

**SDX today (partial):**

| Mechanism | Role |
| :--- | :--- |
| **`--spectral-sfp-loss`** + [`diffusion/spectral_sfp.py`](../diffusion/spectral_sfp.py) | **Training** loss: FFT-weighted error on pred−target in **latent** space — **not** full FNO **forward** denoising. |
| **`--spectral-coherence-latent`** + [`spectral_latent_lowfreq_blend`](../utils/generation/inference_research_hooks.py) | **Inference** FFT low-frequency blend on the final latent (global coherence heuristic). |
| **Multi-scale inference** (hires, dual-stage) | **Process** scaling, not continuous neural-operator sampling. |

**Gap:** No **sampler** that denoises **primarily** in a learned **spectral / operator** basis with **resolution-agnostic** forward pass as in research stacks.

---

## Comparison (qualitative — not benchmarked in this repo)

| Axis | Typical joint DiT | GLS blueprint | In-loop discriminative loop | Frequency / operator view |
| :--- | :--- | :--- | :--- | :--- |
| **Prompt / attribute binding** | Good; color–object leaks possible | **Target:** stronger separation of layout vs texture | **Target:** catch errors before final polish | **Target:** global coherence |
| **Composition** | Variable | **Target:** locked low-res structure | **Target:** localized fixes | **Target:** low-freq global light/shape |
| **Implementation maturity in SDX** | Default path | **Partial** hooks | **Partial** (CLIP guard, etc.) | **Partial** (SFP loss only) |
| **Inference cost** | Baseline | +small planner or stage | +VLM forwards (high) | Depends on architecture |

---

## Suggested research order

1. **Cheap structure** — dual-stage + stronger captions + optional **auxiliary edge/depth** head (trainable).  
2. **Cheap critic** — CLIP guard + masks from **simple** saliency before a full VLM.  
3. **Spectral** — ablate SFP on your data; then consider **true** spectral blocks in the denoiser if loss-only gains plateau.

---

## Machine-readable themes

[`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py): `geometric_latent_split_blueprint`, `discriminative_denoise_vlm_loop`, `frequency_domain_global_coherence`.
