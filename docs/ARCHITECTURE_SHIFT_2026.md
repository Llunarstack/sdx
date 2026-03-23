# Post-diffusion & structural shifts (~late 2025–2026)

**Context:** The field has moved past the first wave of “DiT replaces U-Net” hype into **deeper mathematical and systems refinement**. Pure **noise → image** diffusion is still dominant in production, but **flow matching**, **bridges**, **hybrid AR + diffusion**, **distillation**, and **semantic latents** are the active research and product frontier.

This doc **orients** you to those themes and points to **what SDX already has** vs **future work** — see [`utils/architecture_map.py`](../utils/architecture_map.py) for a machine-readable map.

**Disclaimer:** Names (GLM-Image, FLUX.2, etc.) are **illustrative** of categories; release details change weekly.

---

## 1. Beyond pure diffusion: Flow Matching & diffusion bridges

| Concept | Idea | Why it matters |
| :--- | :--- | :--- |
| **Flow Matching (FM)** | Learn **straight-line** (or near-straight) transport from noise to data instead of highly **curved stochastic** VP paths. | Often **more stable training** and **fewer sampling steps** than classic DDPM schedules. |
| **Rectified flow / OT** | Couples noise and data with **optimal-transport**-style couplings; velocity field along geodesic-like paths. | Common in **latent** generators (e.g. FLUX-class stacks in industry talks). |
| **Diffusion bridges** | Transition between **two** distributions (not only **noise → image**): sketch→render, day→night, edit paths that **preserve structure**. | Powers **unified edit + gen** without throwing away the source layout. |

**SDX today:** VP DDPM + `GaussianDiffusion` + DiT; **v-prediction** (`--prediction-type v`), **non-uniform timestep sampling** (`diffusion/timestep_sampling.py`), and [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) discuss flow-adjacent ideas — **full rectified-flow training is not a drop-in** (different objective/sampler). **Status:** partial / research alignment.

---

## 2. Hybrid AR + Diffusion (LLM-style planning + denoiser)

| Concept | Idea | Why it matters |
| :--- | :--- | :--- |
| **AR “planner” + diffusion decoder** | **Autoregressive** module outputs **coarse semantic tokens** (layout, relations); **DiT / diffusion** refines **texture and detail**. | Addresses **spatial logic** (“X under Y”), **long text**, and **instruction-heavy** prompts. |
| **Block-causal / raster AR in DiT** | Causal attention over **patches or blocks** so generation has a **sequence** bias without a separate tokenizer stack. | Lighter-weight hybrid than a full 9B AR image tokenizer — **controllable** in open-source. |

**SDX today:** **`num_ar_blocks`** (0 / 2 / 4 / …), block masks in `models/attention.py`, [docs/AR.md](AR.md), **`utils/ar_dit_vit.py`** for ViT alignment. This is **one** credible open-source angle on “AR + diffusion” — not a full GLM-Image-style discrete token LM. **Status:** partial (DiT-native AR, not separate AR tokenizer).

---

## 3. State space models (Mamba) vs Vision Transformers

| Concept | Idea | Why it matters |
| :--- | :--- | :--- |
| **SSM / Mamba** | **Subquadratic** sequence models — **linear** scaling in length vs **quadratic** self-attention. | Tempting for **8K+** or **long video** with **VRAM** limits. |
| **Vision Mamba, U-Mamba** | Replace or augment **ViT** backbones in vision encoders or decoders. | Still **research-heavy**; fewer mature T2I checkpoints than DiT. |

**SDX today:** DiT uses **transformer** attention (optional **xformers**, **RoPE**, **token routing**, etc.). **No Mamba backbone** in the repo. **Status:** not implemented; listed in [IMPROVEMENTS.md](IMPROVEMENTS.md) as long-term architecture exploration.

---

## 4. One-step & few-step distillation (DMD, consistency, turbo)

| Concept | Idea | Why it matters |
| :--- | :--- | :--- |
| **Distribution Matching Distillation (DMD)** | Distill a **teacher** diffusion into a **fast** student (sometimes **1 step**), combining ideas from **GANs** + **diffusion**. | **Real-time** UI, “generate as you type,” consumer GPUs. |
| **Consistency / CM** | Enforce **trajectory consistency** for few-step sampling. | Alternative path to turbo inference. |

**SDX today:** **DDIM-style** multi-step sampling; **no** built-in DMD/consistency trainer. **Test-time:** `--num K --pick-best` ([`utils/test_time_pick.py`](../utils/test_time_pick.py)) as a **quality** gate, not a distilled student. **Status:** not in-repo training; roadmap in [IMPROVEMENTS.md](IMPROVEMENTS.md) §2.4 / §11.

---

## 5. Semantic & scientific grounding

| Concept | Idea | Why it matters |
| :--- | :--- | :--- |
| **Physics-informed / physical consistency** | Bias generations toward **plausible** lighting, gravity, contact — **not only** “looks plausible.” | Strong in **video / 3D**; emerging in **still** image stacks via **VLM** or **auxiliary losses**. |
| **Scientific imaging** | Diffusion in **non-photographic** spaces (e.g. molecular, material) — **noise** = **uncertainty**, not pixels. | Different product line than **SDX**; included for **ecosystem** awareness. |
| **RAG / facts in prompt** | **User-supplied** facts merged into the prompt before encoding — [`utils/rag_prompt.py`](../utils/rag_prompt.py). | **Grounding** without retraining. |

**SDX today:** **T5 (+ optional CLIP fusion)**; **REPA** aligns features to **frozen vision** encoders; **RAE** path + `RAELatentBridge` for non–4ch latents; **no** built-in physics simulator. **Status:** partial (semantic alignment / RAE / REPA), not physics engine.

---

## 6. Representation autoencoders + DiT (NYU-style line)

| Concept | Idea | Why it matters |
| :--- | :--- | :--- |
| **RAE** | **Semantic** latents from **frozen** encoders + learned decoder; **DiT** denoises in that space. | **Richer** than “compress pixels only” VAEs. |

**SDX today:** **`--autoencoder-type rae`**, `RAELatentBridge`, training/inference paths — see [MODEL_STACK.md](MODEL_STACK.md), README. **Status:** implemented (RAE path). **REPA** ([`--repa-weight`](../train.py)) aligns with **external** vision reps — **related** to “semantic-first latents.”

---

## Summary: 2023–2024 vs 2026 (illustrative)

| Feature | Old way (2023–2024) | New way (2026 discussions) |
| :--- | :--- | :--- |
| **Backbone** | U-Net | **DiT** / **Mamba** (experimental) |
| **Logic** | Pure diffusion | **Hybrid** (AR planning + diffusion decoder), **VLM** conditioning |
| **Inference** | 20–50 steps | **1–8** steps (distilled / flow-matched) |
| **Pathing** | Stochastic / curved VP | **Flow matching**, **bridges**, bridges for **edit** |
| **Resolution** | 1024² + upscaler | **Native** multi-megapixel / **4K-class** in product stacks |
| **Latent** | “Pixel compression” VAE | **Semantic** / **RAE** / **alignment** (REPA) |

---

## Product-scale stacks (examples only)

| Line | Themes (illustrative) | SDX analogy |
| :--- | :--- | :--- |
| **Hybrid AR + DiT** (e.g. industry “AR planner + DiT decoder”) | Discrete tokens → refine | **Block AR** + DiT; **no** separate 9B AR image LM in-repo |
| **Latent flow + VLM** (e.g. FLUX-class) | Flow + VLM + new VAE | **Flow** not native; **triple** text encoders + **T5**; **VAE** from diffusers |
| **RAE + DiT** (research) | Semantic latents | **`autoencoder_type=rae`** + bridge |

---

## See also

- [WORKFLOW_INTEGRATION_2026.md](WORKFLOW_INTEGRATION_2026.md) — workflow / efficiency industry narratives + disclaimers  
- [LANDSCAPE_2026.md](LANDSCAPE_2026.md) — product trends, authenticity, pipelines  
- [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) — what SDX implements vs research  
- [IMPROVEMENTS.md](IMPROVEMENTS.md) — §11–§12 next-tier ideas  
- [AR.md](AR.md) — block-wise AR in DiT  
- [`utils/architecture_map.py`](../utils/architecture_map.py) — programmatic theme → repo mapping  
