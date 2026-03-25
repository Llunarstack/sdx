# Fast, precise generation: consistency flow, solvers, distillation (blueprint)

**Purpose:** Organize **mathematics- and speed-first** directions (few-step / one-step quality) and map them to **SDX today**. This is a **design and literature** guide, not a benchmark report: **step counts, FID, and millisecond figures** in popular write-ups are **checkpoint- and hardware-dependent**; treat them as **directional**, not guarantees for this repo.

**Related:** [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) (VP vs flow, SFP, timestep sampling), [ARCHITECTURE_SHIFT_2026.md](ARCHITECTURE_SHIFT_2026.md), [DIFFUSION_LEVERAGE_ROADMAP.md](DIFFUSION_LEVERAGE_ROADMAP.md).

---

## 1. Consistency-style objectives on flow fields (Consistency-FM narrative)

**Idea (high level):** Standard **flow matching** learns a time-dependent velocity field \(v_\theta(x_t, t)\) that transports noise toward data. **Consistency** variants add constraints so that **endpoints agree** when integrating from **different** starting times along the *same* underlying field—reducing path dependency and enabling **larger integration steps** without drifting off the intended trajectory.

**Sketch:** One imagines enforcing that flows from \(t_1\) and \(t_2\) to a terminal time land on the **same** predicted clean state (exact forms differ by paper: distillation, consistency models, trajectory matching, etc.).

**Why people care:** Fewer solver steps if the field is **self-consistent** and the integrator can take **large jumps** without error accumulation from a highly curved effective trajectory.

**SDX today:** Training is **VP DDPM** + `GaussianDiffusion.training_losses` (ε, v, or x₀ targets), **not** a dedicated consistency-FM trainer. Inference is **DDIM-style** strided stepping on a **discrete** VP schedule. **Status:** **not implemented** as a separate objective; see [IMPROVEMENTS.md](IMPROVEMENTS.md) §2.4 (distillation / consistency roadmap).

---

## 2. “Dual-solver” / mixed prediction domains (ε, v, x₀) and time warping

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

## 3. Adversarial Diffusion Distillation (ADD-class)

**Idea:** Distill a large **teacher** generator into a smaller **student** with **adversarial** critics so the student matches both **pixel-level** sharpness and **semantic / feature-level** agreement with the teacher—often targeting **1–2 step** or very few-step generation.

**Why people care:** Pure regression-to-teacher can blur high frequencies; a **discriminator** (possibly multi-head: RGB vs frozen encoder features) pressures the student to stay sharp and on-manifold.

**SDX today:** No ADD trainer, no dual-head discriminator distillation path in-tree. **DMD / consistency** themes are likewise **not in repo** (see `dmd_distillation` in [`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py)). **Status:** **not implemented**.

---

## 4. “Straight” paths: rectified flow + optimal transport pairing

**Idea:** If the learned transport is **approximately linear** in a suitable coordinate system, an ODE solver can approach **one-step** integration:  
\(x_{t+\Delta t} \approx x_t + v_\theta(x_t, t)\,\Delta t\).

**OT coupling (Hungarian / Sinkhorn):** During training, **pair** noise samples to data samples so that **transport cost** (e.g. squared distance) is minimized batch-wise. Intuition: **shorter average paths** than i.i.d. pairing can yield **easier** fields to integrate and better few-step behavior—when combined with the right architecture and objective.

**SDX today:** Default VP training still uses **independent** Gaussian noise per sample. Optional **mini-batch Sinkhorn / Hungarian** coupling is available via **`train.py --ot-noise-pair-reg`** (see `utils/training/ot_noise_pairing.py`) — an **experimental** analogue, **not** a full rectified-flow ODE trainer.

---

## Qualitative comparison (illustrative only)

| Approach | Paradigm (rough) | Typical step regime (literature / products) | SDX |
| :--- | :--- | :--- | :--- |
| Classic VP DiT | SDE / discrete VP + DDIM family | 25–50+ | **Default path** |
| Flow matching / RF | Continuous-time velocity ODE | 10–20 (varies) | **Not** drop-in trained |
| Consistency / trajectory matching | Self-consistent field or distilled student | 2–8 | **Not** in-repo trainer |
| ADD / GAN–diffusion hybrid | Adversarial distillation | 1–2 | **Not** in-repo |
| OT-coupled RF | Straight paths + optimal pairing | 1–4 (aspirational) | **Partial** (train OT noise coupling only) |

---

## Practical build order (if you implement)

1. **Baseline** — stable VP training + v or ε + min-SNR + `timestep_sample_mode` ablations ([MODERN_DIFFUSION.md](MODERN_DIFFUSION.md)).  
2. **Separate branch** — experimental **flow** trainer + matched sampler (do not silently mix with VP checkpoints).  
3. **Distillation** — teacher → student (consistency or ADD-class) as a **second stage** with its own loss and eval.  
4. **OT / pairing** — only after (2) is stable; measure wall-clock and FID/GenEval vs cost.

---

## Machine-readable map

Theme IDs in [`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py):  
`consistency_flow_matching_velocity`, `dual_solver_time_warping`, `add_adversarial_distillation`, `rectified_flow_ot_coupling`.
