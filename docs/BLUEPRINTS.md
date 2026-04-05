# Research blueprints: fast generation & prompt accuracy

Design notes that map **literature-style ideas** to **SDX today** (not benchmark numbers). For ecosystem context see [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md), [LANDSCAPE_2026.md](LANDSCAPE_2026.md), and [utils/architecture/architecture_map.py](../utils/architecture/architecture_map.py).

---

## Part 1: Flow, solvers, distillation, and rectified paths

**Purpose:** Organize **mathematics- and speed-first** directions (few-step / one-step quality) and map them to **SDX today**. This is a **design and literature** guide, not a benchmark report: **step counts, FID, and millisecond figures** in popular write-ups are **checkpoint- and hardware-dependent**; treat them as **directional**, not guarantees for this repo.

**Related:** [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) (VP vs flow, SFP, timestep sampling), [LANDSCAPE_2026.md — architecture section](LANDSCAPE_2026.md#post-diffusion-and-structural-shifts-late-20252026), [DIFFUSION_LEVERAGE_ROADMAP.md](DIFFUSION_LEVERAGE_ROADMAP.md).

---

### 1. Consistency-style objectives on flow fields (Consistency-FM narrative)

**Idea (high level):** Standard **flow matching** learns a time-dependent velocity field \(v_\theta(x_t, t)\) that transports noise toward data. **Consistency** variants add constraints so that **endpoints agree** when integrating from **different** starting times along the *same* underlying field—reducing path dependency and enabling **larger integration steps** without drifting off the intended trajectory.

**Sketch:** One imagines enforcing that flows from \(t_1\) and \(t_2\) to a terminal time land on the **same** predicted clean state (exact forms differ by paper: distillation, consistency models, trajectory matching, etc.).

**Why people care:** Fewer solver steps if the field is **self-consistent** and the integrator can take **large jumps** without error accumulation from a highly curved effective trajectory.

**SDX today:** Default training is **VP DDPM** + `GaussianDiffusion.training_losses` (ε, v, or x₀ targets). **Optional** rectified-flow-style velocity training exists (`--flow-matching-training`, `diffusion/flow_matching.py`) with matching sampling (`sample_loop(..., flow_matching_sample=True)` / `sample.py` flags)—this is **not** a multi-time **consistency-FM** objective. Default inference is **DDIM-style** strided stepping on a **discrete** VP schedule. **Status:** dedicated **consistency** constraints on the flow field are **not implemented**; see [IMPROVEMENTS.md](IMPROVEMENTS.md) §2.4 (distillation / consistency roadmap).

---

### 2. “Dual-solver” / mixed prediction domains (ε, v, x₀) and time warping

**Idea:** Diffusion “work” is often **non-uniform** in a linear time index: early steps establish coarse structure; late steps refine texture. A **dual** or **multi-regime** solver might:

- Use different **parameterizations** (noise ε vs velocity-like v vs data x₀) in different bands of \(t\), or  
- Warp the integration grid (e.g. **log-like** emphasis on low vs high noise) to reduce **NFE** for a fixed perceptual quality.

Some industry stacks also speak of **learned** or **adaptive** schedules (meta-parameters akin to \(\tau\))—that is **not** the same as a fixed strided DDIM schedule.

**SDX today:**

| Knob | Role |
| :--- | :--- |
| `--prediction-type` `epsilon` \| `v` \| `x0` | **Single** parameterization per checkpoint (train and sample must match). |
| `--timestep-sample-mode` (e.g. `logit_normal`, `high_noise`) | **Training-time** emphasis on which noise levels are sampled—related to “where the work is,” but **not** a learned dual integrator. |
| Strided `sample_loop` | Fixed schedule over discrete VP steps. |

**Status:** **Partial** analogy only (parameter choice + non-uniform training times). **No** per-step dynamic switching between ε/v/x₀ at inference and **no** learned log/linear domain mixer \(\tau\).

---

### 3. Adversarial Diffusion Distillation (ADD-class)

**Idea:** Distill a large **teacher** generator into a smaller **student** with **adversarial** critics so the student matches both **pixel-level** sharpness and **semantic / feature-level** agreement with the teacher—often targeting **1–2 step** or very few-step generation.

**Why people care:** Pure regression-to-teacher can blur high frequencies; a **discriminator** (possibly multi-head: RGB vs frozen encoder features) pressures the student to stay sharp and on-manifold.

**SDX today:** No ADD trainer, no dual-head discriminator distillation path in-tree. **DMD / consistency** themes are likewise **not in repo** (see `dmd_distillation` in [`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py)). **Status:** **not implemented**.

---

### 4. “Straight” paths: rectified flow + optimal transport pairing

**Idea:** If the learned transport is **approximately linear** in a suitable coordinate system, an ODE solver can approach **one-step** integration:  
\(x_{t+\Delta t} \approx x_t + v_\theta(x_t, t)\,\Delta t\).

**OT coupling (Hungarian / Sinkhorn):** During training, **pair** noise samples to data samples so that **transport cost** (e.g. squared distance) is minimized batch-wise. Intuition: **shorter average paths** than i.i.d. pairing can yield **easier** fields to integrate and better few-step behavior—when combined with the right architecture and objective.

**SDX today:** Default VP training still uses **independent** Gaussian noise per sample. Optional **mini-batch Sinkhorn / Hungarian** coupling is available via **`train.py --ot-noise-pair-reg`** (see `utils/training/ot_noise_pairing.py`) — an **experimental** analogue, **not** a full rectified-flow ODE trainer.

---

### Qualitative comparison (illustrative only)

| Approach | Paradigm (rough) | Typical step regime (literature / products) | SDX |
| :--- | :--- | :--- | :--- |
| Classic VP DiT | SDE / discrete VP + DDIM family | 25–50+ | **Default path** |
| Flow matching / RF | Continuous-time velocity ODE | 10–20 (varies) | **Not** drop-in trained |
| Consistency / trajectory matching | Self-consistent field or distilled student | 2–8 | **Not** in-repo trainer |
| ADD / GAN–diffusion hybrid | Adversarial distillation | 1–2 | **Not** in-repo |
| OT-coupled RF | Straight paths + optimal pairing | 1–4 (aspirational) | **Partial** (train OT noise coupling only) |

---

### Practical build order (if you implement)

1. **Baseline** — stable VP training + v or ε + min-SNR + `timestep_sample_mode` ablations ([MODERN_DIFFUSION.md](MODERN_DIFFUSION.md)).  
2. **Separate branch** — experimental **flow** trainer + matched sampler (do not silently mix with VP checkpoints).  
3. **Distillation** — teacher → student (consistency or ADD-class) as a **second stage** with its own loss and eval.  
4. **OT / pairing** — only after (2) is stable; measure wall-clock and FID/GenEval vs cost.

---

### Machine-readable map

Theme IDs in [`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py):  
`consistency_flow_matching_velocity`, `dual_solver_time_warping`, `add_adversarial_distillation`, `rectified_flow_ot_coupling`.

---

## Part 2: Prompt-accurate generation (GLS, critics, frequency)

**Purpose:** Capture **technical** directions for beating typical DiT-style “layout and texture in one forward” behavior—without treating any row in the comparison table as a **measured** SDX result. DiTs **jointly** refine semantics and appearance; **decoupling**, **in-loop verification**, and **frequency-aware** objectives are levers to reduce **semantic bleeding**, **wrong placement**, and **global inconsistency**.

**Related:** [NEXTGEN_SUPERMODEL_ARCHITECTURE.md](NEXTGEN_SUPERMODEL_ARCHITECTURE.md) (overlapping pillars), [Part 1: Flow, solvers, distillation](#part-1-flow-solvers-distillation-and-rectified-paths), [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) (spectral SFP).

---

### 1. Structural blueprint — geometric–latent split (GLS)

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

### 2. Recursive self-correction — discriminative denoising loops

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

### 3. Frequency-domain diffusion — neural operators (FNO narrative)

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

### Comparison (qualitative — not benchmarked in this repo)

| Axis | Typical joint DiT | GLS blueprint | In-loop discriminative loop | Frequency / operator view |
| :--- | :--- | :--- | :--- | :--- |
| **Prompt / attribute binding** | Good; color–object leaks possible | **Target:** stronger separation of layout vs texture | **Target:** catch errors before final polish | **Target:** global coherence |
| **Composition** | Variable | **Target:** locked low-res structure | **Target:** localized fixes | **Target:** low-freq global light/shape |
| **Implementation maturity in SDX** | Default path | **Partial** hooks | **Partial** (CLIP guard, etc.) | **Partial** (SFP loss only) |
| **Inference cost** | Baseline | +small planner or stage | +VLM forwards (high) | Depends on architecture |

---

### Suggested research order

1. **Cheap structure** — dual-stage + stronger captions + optional **auxiliary edge/depth** head (trainable).  
2. **Cheap critic** — CLIP guard + masks from **simple** saliency before a full VLM.  
3. **Spectral** — ablate SFP on your data; then consider **true** spectral blocks in the denoiser if loss-only gains plateau.

---

### Machine-readable themes

[`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py): `geometric_latent_split_blueprint`, `discriminative_denoise_vlm_loop`, `frequency_domain_global_coherence`.
