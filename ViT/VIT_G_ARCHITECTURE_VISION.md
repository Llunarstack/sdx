# ViT-G — Vision-Intelligence Transformer for **Generation** (research vision)

**Read this first:** The name **ViT-G** here means a **hypothetical next-generation image generator** (a “DiT successor” sketch). It is **not** the same thing as the **`ViT/` package** in this repository, which trains a **discriminative** Vision Transformer to **score** quality and prompt adherence on finished images. See [README.md](README.md) and [EXCELLENCE_VS_DIT.md](EXCELLENCE_VS_DIT.md).

**Status:** **Aspirational architecture narrative** (external / exploratory). None of the claims below (e.g. “1000×”, “perfect spatial logic”, “native 16K”, “one-step convergence”) are **SDX benchmarks** or **implemented** as a single model in this repo. Treat complexity statements \(O(N^2)\) vs \(O(N \log N)\) as **design targets** from the narrative, not proven for your hardware until you measure.

---

## Why move beyond “flat” DiT?

Standard **Diffusion Transformers** often use a **uniform** grid of patch tokens: every region gets similar attention budget. A **hierarchical, self-correcting** design tries to fix three recurring pain points:

1. **Spatial inconsistency** — long-range relations (“left vs right”, reflections) are easy to smear.  
2. **Cost at high resolution** — global self-attention scales poorly with token count.  
3. **Semantic drift** — deep denoising can dilute text conditioning so details of the prompt are lost.

**ViT-G** (in this document) is a **label** for that direction: not a shipped product name in SDX.

---

## 1. Architecture sketch: hierarchical + dual-stream

### A. Multi-scale “foveated” attention

**DiT pain (narrative):** Sky and grass consume tokens similarly to faces and hands, so compute may be **misallocated** relative to perceptual importance.

**ViT-G direction:** A **foveated tokenizer** — allocate **more** tokens to high-complexity regions and **merge** tokens in smooth regions. Log-polar or pyramid layouts are one way to bias resolution toward the center or toward salient areas.

**Complexity narrative:** Proponents claim sequence length can drop from **\(O(N^2)\)** tokens (dense \(H \times W\) patches) toward **\(O(N \log N)\)** or similar under adaptive pooling — **implementation-dependent**; must be validated per backbone.

**SDX today (partial analogues, not foveation):**

- **FiT / flexible tokens** — see [EXCELLENCE_VS_DIT.md](EXCELLENCE_VS_DIT.md) (literature); not a foveated tokenizer in-tree.  
- **Block AR in DiT** (`--num-ar-blocks`) — sequence bias, not saliency-based token merging.  
- **This repo’s `ViT/`** — fixed `timm` backbones on crops; no dynamic token merge for generation.

### B. Cross-modal backbone (dual-stream)

**DiT pain (narrative):** Text is fused early and can **attenuate** over depth.

**ViT-G direction:** Two streams in parallel:

- **Symbolic stream** — prompt logic, constraints, possibly discrete structure.  
- **Latent stream** — image tokens / spatial state.

**Coupling:** **Asymmetric cross-attention** — the symbolic stream **queries** or **gates** the latent stream each layer so conditioning stays “in charge.”

**SDX today (partial):** DiT **cross-attention** to T5 (and optional CLIP) is already a dual-stream **pattern**, but it is **not** the full asymmetric “manager” design above; depth and fusion details differ by checkpoint.

---

## 2. Speed narrative: latent-shift, NOF, speculative denoising

### A. Neural operator flow (NOF)

**Idea:** Learn a **continuous velocity field** (flow viewpoint) instead of relying only on many discrete Gaussian denoise steps. **Fourier Neural Operators (FNOs)** are one way to act on **frequency-domain** representations so **global** structure might be resolved in **few** evaluations.

**Caveat:** “**Perceptual convergence in 1 step**” is a **marketing-grade** claim; real systems need matched training objectives, integrators, and eval.

**SDX today:** VP **DDIM-style** diffusion + optional research hooks ([`utils/generation/inference_research_hooks.py`](../utils/generation/inference_research_hooks.py)); **no** FNO-based generator. See [docs/BLUEPRINTS.md](../docs/BLUEPRINTS.md) and [docs/MODERN_DIFFUSION.md](../docs/MODERN_DIFFUSION.md).

### B. Speculative denoising

**Idea (LLM analogy):** A **small draft** model proposes several denoise steps; a **large validator** checks consistency and accepts or rolls back — trading extra small-model compute for fewer large-model steps.

**SDX today:** **Not implemented.** Closest **inference** ideas: multi-sample **`--pick-best`**, CLIP-guard refine — **post-hoc** or **branching**, not speculative trajectory acceptance inside one denoise chain.

---

## 3. Comparison table (aspirational vs typical DiT — not measured here)

| Feature | Typical DiT (high level) | ViT-G vision (targets) |
| :--- | :--- | :--- |
| **Logic** | Denoise / score-based sampling | Constraint- and flow-style **reasoning** (narrative) |
| **Spatial awareness** | Global attention helps but errors persist | Hierarchical + foveation + strong pos encoding **targets** |
| **Text / glyphs** | Long text and spelling are hard | Dedicated glyph / byte-level paths **in some roadmaps** |
| **Resolution** | Fixed grids; upscalers common | **Resolution-aware** or operator-style **targets** |
| **Compute** | Quadratic in token count (attention) | Adaptive tokens **aim** at sub-quadratic **behavior** |
| **Steps** | Often 20–50 for strong VP checkpoints | Few-step **if** flow + distillation + matched training |

**SDX partial hooks:** RoPE and related pos encodings live in DiT stack where configured; **glyph** experiments are touched in research hooks, not a full ByT5 text renderer in DiT. **Spectral SFP** ([`diffusion/spectral_sfp.py`](../diffusion/spectral_sfp.py)) shapes **training** loss in frequency space — not the same as NOF generation.

---

## 4. Mathematical sketch: manifold-style regularization

One can **imagine** augmenting flow-matching style losses with a **critic** that penalizes departure from prompt-consistent semantics:

\[
\mathcal{L} = \left\lVert v_\theta(x_t, t) - u_t(x_t) \right\rVert^2 + \lambda \cdot \mathrm{VLM\_Feedback}(x_0)
\]

- First term: **velocity / flow** matching (standard in flow literature, form varies).  
- Second term: a **semantic critic** (VLM, CLIP, or learned reward) — in **full** training would require careful differentiability, stopping gradients, or distillation; **not** wired as end-to-end `VLM_Feedback` in SDX core training.

**SDX analogues:** [`utils/generation/clip_alignment.py`](../utils/generation/clip_alignment.py) (CLIP alignment, optional refine), **`ViT/`** adherence head ( **offline** scoring / ranking ), [docs/NEXTGEN_SUPERMODEL_ARCHITECTURE.md](../docs/NEXTGEN_SUPERMODEL_ARCHITECTURE.md) § in-loop critic.

---

## 5. If you implement pieces in PyTorch (pointers)

- **Foveated / adaptive tokens:** start with a **saliency map** (gradient magnitude, edge detector, or small aux net) → **superpixel / quadtree** merge or **variable patch sizes**; validate that your DiT block supports **irregular** token counts or pad to block buckets.  
- **VLM / critic loop:** decode low-res preview → frozen CLIP/VLM score → **stop-grad** scalar loss or **latent refine** second pass (see existing CLIP guard pattern) before attempting full **differentiable** VLM through the decoder.

---

## See also

- [README.md](README.md) — this folder’s **scoring** ViT  
- [EXCELLENCE_VS_DIT.md](EXCELLENCE_VS_DIT.md) — papers + SDX stacking  
- [docs/NEXTGEN_SUPERMODEL_ARCHITECTURE.md](../docs/NEXTGEN_SUPERMODEL_ARCHITECTURE.md) — four-pillar “super-model” doc  
- [`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py) — theme IDs for tooling  
