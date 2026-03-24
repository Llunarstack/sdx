# Diffusion stack: high-leverage upgrades (“10× class” levers)

**What this doc is:** A **prioritized** list of **model / training / inference** directions that can compound into **large** quality or efficiency gains. “10×” here means **orders-of-magnitude-style improvement across the full stack** (data + representation + objective + sampler + alignment)—not a single architectural swap.

**Companion reads:** [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) (what is already wired), [ARCHITECTURE_SHIFT_2026.md](ARCHITECTURE_SHIFT_2026.md) (macro themes), [`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py) (theme → path map).

---

## 1. Lever hierarchy (where gains actually come from)

| Tier | Lever | Why it dominates |
| :--- | :--- | :--- |
| **A** | **Data & captions** | Better pairs beat wider nets; panel/text/comic data quality caps lettering and layout. |
| **A** | **Latent representation** (VAE vs RAE, bit depth, decode artifacts) | Ceiling on fine detail, text edges, and color stability. |
| **B** | **Training objective & time sampling** | Wrong noise levels or loss weights waste capacity (you already have logit-normal, min-SNR-soft, v-pred). |
| **B** | **Conditioning depth** (text encoders, fusion, length, control signals) | Prompt following and style lock scale with **how much** structured signal reaches DiT blocks. |
| **C** | **Backbone capacity** (depth, width, MoE, attention tricks) | Marginal if A/B are weak; essential once data/objectives are sane. |
| **C** | **Inference** (steps, distillation, guidance fixes) | Latency and “snap” to prompt without retraining the whole base. |
| **D** | **Post-train alignment** (DPO, reward fine-tuning) | Steers an **already good** base; poor substitute for broken data or latents. |

**Practical rule:** Before chasing exotic architectures, **measure** A/B with fixed compute (same steps, same resolution buckets)—see `scripts/tools/training_timestep_preview.py` and your bucket manifests.

---

## 2. Representation & autoencoder (latent stack)

| Idea | Mechanism | SDX touchpoints | Effort |
| :--- | :--- | :--- | :--- |
| **RAE / semantic latents** | Denoise in a space aligned to **vision semantics**, not only pixel compression. | `--autoencoder-type rae`, `models/rae_latent_bridge.py`, [MODEL_STACK.md](MODEL_STACK.md) | Medium (train stack) |
| **REPA-style alignment** | Auxiliary loss matches internal DiT states to **frozen ViT/DINO** features. | `--repa-weight`, `train.py`, DiT flags in `models/dit_text.py` | Low–medium (toggle + tune) |
| **Stronger VAE for text/edges** | Less smear in speech balloons, sharper line art—often **ae** upgrades beat **dit** width. | External VAE swap + compatibility tests in `train.py` / `sample.py` | High if retraining AE |
| **Multi-scale latent / MAR hybrids** | Coarse tokens + diffusion refine (masked AR in latent) | Not in-repo; overlaps “hybrid AR + diffusion” in [ARCHITECTURE_SHIFT_2026.md](ARCHITECTURE_SHIFT_2026.md) | Research |
| **Pixel-space DiT (PixelDiT-class)** | Reduce single-VAE bottleneck; heavy compute. | Listed in [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) | Long-term branch |

---

## 3. Conditioning & multimodal (prompt → denoiser)

| Idea | Mechanism | SDX touchpoints | Effort |
| :--- | :--- | :--- | :--- |
| **Multi-encoder text** | T5 + CLIP-L + CLIP-bigG fusion improves adherence and style. | `train.py --text-encoder-mode triple`, fusion weights in checkpoint | Medium |
| **Longer / structured captions** | JSONL `parts`, region captions, layout-aware training data. | `data/t2i_dataset.py`, [REGION_CAPTIONS.md](REGION_CAPTIONS.md) | Data pipeline |
| **Native multimodal transformer** | Single stack for extra modalities (sketch, layout tensor) before DiT. | `models/native_multimodal_transformer.py`, `models/cascaded_multimodal_diffusion.py` | Medium–high |
| **Reference-image conditioning** | IP-Adapter / cross-attn image tokens for **character lock** without full finetune. | Would be new adapter path on `DiT_Text` or parallel cond channel | High |
| **Regional / compositional CFG** | Different guidance per region (subject vs background). | Not in core sampler; inference research | Research |
| **Control paths** | Depth, edge, segmentation as extra channels (PixArt-style `control_cond_dim` hooks exist in `dit_text.py`). | `models/dit_text.py` (`control_cond_dim`), training glue | Medium |

---

## 4. Objectives, schedules, and auxiliary tasks

| Idea | Mechanism | SDX touchpoints | Effort |
| :--- | :--- | :--- | :--- |
| **Non-uniform timestep sampling** | Emphasize hard noise levels (high-noise, logit-normal). | `diffusion/timestep_sampling.py`, `train.py` | Low |
| **Loss weighting** | Min-SNR, soft min-SNR, SNR-aware curriculum. | `diffusion/timestep_loss_weight.py`, [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) | Low |
| **v-prediction / epsilon** | Match parameterization to schedule and CFG behavior. | `--prediction-type` | Low |
| **Auxiliary structure heads** | Side losses on **edges, depth, segmentation** for comics/layout. | New heads on DiT final layer or intermediate — **not wired** | High |
| **Self-forcing / bootstrapping** | Train on model’s own high-quality generations (filtered). | New data loop + gating | Medium |
| **Flow matching / rectified flow** | Replace VP objective with velocity field on simpler paths. | **Separate trainer**; see [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) | Very high |

---

## 5. Architecture inside the denoiser (DiT)

| Idea | Mechanism | SDX touchpoints | Effort |
| :--- | :--- | :--- | :--- |
| **MoE FFN / routed experts** | Scale params without dense FLOPs per token. | `models/dit.py`, `models/dit_text.py` (`moe_num_experts`, aux loss) | Medium |
| **Block-causal / AR hybrid** | Raster or block order bias for layout coherence. | `--num-ar-blocks`, `models/attention.py`, [AR.md](AR.md) | Medium |
| **RoPE / register tokens / token routing** | Better long-sequence and capacity use. | Flags on `DiT_Text` (`use_rope`, `num_register_tokens`, routing) | Low–medium |
| **SSM / Mamba backbone** | Subquadratic sequence models for huge latents. | Not in-repo ([ARCHITECTURE_SHIFT_2026.md](ARCHITECTURE_SHIFT_2026.md)) | Research |
| **Matryoshka / nested depth** | Train shallow→deep (depth growth) for stability. | Trainer schedule change (SANA-style ideas in [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md)) | High |

---

## 6. Inference, sampling, and distillation

| Idea | Mechanism | SDX touchpoints | Effort |
| :--- | :--- | :--- | :--- |
| **Better ODE/SDE solvers** | DPM-Solver++, UniPC, adaptive step where supported. | `sample.py` / diffusion scheduler modules | Medium |
| **CFG rescale / dynamic threshold** | Reduce oversaturation at high guidance. | `sample.py` | Low |
| **Test-time compute** | N candidates + CLIP/edge/OCR pick. | `--num`, `--pick-best`, `utils/quality/test_time_pick.py` | Low |
| **Consistency / DMD / turbo distillation** | Few-step student matching teacher distribution. | **No trainer**; roadmap [IMPROVEMENTS.md](IMPROVEMENTS.md) | Very high |
| **RF-style test-time refinement** | Extra gradient steps on scores (heavy). | Research / optional script | Research |

---

## 7. Alignment after base training

| Idea | Mechanism | SDX touchpoints | Effort |
| :--- | :--- | :--- | :--- |
| **Diffusion-DPO / preference pairs** | Steer with win/lose image pairs. | Discussed in [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md); **no default trainer** | High |
| **Automated pair mining** | Use `test_time_pick` scores to build preference datasets. | `utils/quality/test_time_pick.py` | Medium |
| **Human / model critique loop** | `orchestrate_pipeline.py` style multi-stage review. | `scripts/tools/ops/orchestrate_pipeline.py` | Workflow |

---

## 8. Suggested sequencing (example roadmap)

1. **Lock metrics:** FID/CLIPScore on a **fixed** eval prompt set + optional OCR on text panels.  
2. **Exploit existing flags:** resolution buckets, logit-normal / high-noise sampling, min-SNR-soft, triple text encoders, REPA weight sweep.  
3. **Latent audit:** Compare standard VAE vs RAE path on **text-in-image** and **line art** subsets.  
4. **Add control or reference conditioning** if character consistency is the main product gap.  
5. **Only then** widen DiT or add MoE depth—otherwise you may be **noise-fitting**.  
6. **Long fork:** flow-matching trainer or distillation student as a **separate** experiment branch, not mixed into the main `GaussianDiffusion` loop without tests.

---

## 9. Paper / keyword index (starting points)

- **Flow / OT:** rectified flow, flow matching, stochastic interpolation.  
- **Schedules / training:** EDM preconditioning, VP vs VE, cosine/sigmoid betas (you have multiple in `diffusion/schedules.py`).  
- **Efficiency:** FasterDiT (SNR-aware training), SANA (efficient DiT), DiT-Air.  
- **Alignment:** Diffusion-DPO, Pick-a-Pic, reward-model distillation.  
- **Structure:** ControlNet-class adapters, T2I-CompBench / GenEval for compositional testing.  
- **Latent:** RAE, REPA, tokenizer+flow (RecTok-class).

Keep **arXiv numbers** and product names out of the critical path—verify citations when you implement.

---

## 10. What *not* to do first

- Swapping to a **10× wider DiT** without fixing **caption noise** and **latent smear**.  
- Adding **flow matching** on top of an **unstable VP baseline**—stabilize schedules first.  
- **DPO** on a model that cannot yet render **readable text** in-domain—alignment cannot invent capability you did not train.

This roadmap should stay **living**: when you land a feature, move it from “idea” to **`architecture_map.py`** + [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) “Implemented” tables.
