# Modern diffusion & flow ideas (2024–2026)

Short survey of **recent directions** and how they relate to **SDX** (VP DDPM + DiT + `GaussianDiffusion`).

---

## Implemented in SDX (this repo)

| Idea | Where | Notes |
|:-----|:------|:------|
| **Logit-normal training timesteps** | `diffusion/timestep_sampling.py`, `train.py --timestep-sample-mode logit_normal` | Discrete-time analogue of SD3-style sampling on a normalized axis; emphasizes some noise levels vs uniform `randint`. |
| **High-noise emphasis** | `--timestep-sample-mode high_noise` | `Beta(2,1)` on normalized time → more samples at **large** `t`. |
| Min-SNR loss weighting | Training | Stabilizes across timesteps. |
| V-prediction | `--prediction-type v` | Velocity parameterization (SD2-style). |
| CFG rescale / dynamic threshold | `sample.py` | High-CFG friendly. |
| Cosine **beta** schedule | `--beta-schedule cosine` | Alternative noise schedule (not the same as “cosmap” loss weighting in SD3). |
| **Sigmoid** / **squaredcos_cap_v2** schedules | `--beta-schedule sigmoid` or `squaredcos_cap_v2` | See `diffusion/schedules.py` (diffusers-style squared cosine + smooth sigmoid ramp). |
| **Soft min-SNR** loss weight | `--loss-weighting min_snr_soft` | Smooth `gamma/(snr+gamma)` weight (see `diffusion/timestep_loss_weight.py`); alternative to hard `min_snr`. |

---

## Research directions (not fully wired — why)

| Topic | What it is | Feasibility here |
|:------|:-----------|:-----------------|
| **Rectified flow / flow matching** | Train **velocity** along straight paths in data–noise space; often continuous-time. | Needs a **different training objective** and usually a new sampler. Possible as a **separate** experimental trainer, not a drop-in for current `training_losses`. |
| **Rectified-CFG++** | Predictor–corrector on **conditional vs unconditional velocity** to reduce off-manifold drift under CFG. | Targets **flow** checkpoints (e.g. SD3, Flux). Our loop is **DDIM-style VP**; porting needs careful equivalence or a flow checkpoint. |
| **RF-Sampling / test-time scaling** | Gradient steps on alignment scores during sampling (e.g. FLUX). | Inference-time only; depends on score / CLIP setup and is heavy. |
| **TPC (temporal pair consistency)** | Couple velocity at paired times to reduce gradient variance in flow training. | Flow-training specific. |
| **RecTok** | Distill semantics into tokenizer + flow. | Architecture + tokenizer change. |
| **Consistency / distillation** | 1–4 step student. | Separate training stage; listed in [IMPROVEMENTS.md](IMPROVEMENTS.md). |

---

## Scaling, efficiency & architecture (2024–2025)

| Direction | What it is | SDX today / next step |
|:----------|:-----------|:----------------------|
| **FasterDiT** (NeurIPS 2024, arXiv **2410.10356**) | Analyzes **SNR distributions** across timesteps and proposes **training acceleration** + supervision tweaks **without changing DiT architecture** (reported ~7× faster to comparable FID on ImageNet-class setups). | You already use **Min-SNR weighting** and **non-uniform `t`** (`timestep_sampling`). Quick **distribution preview:** `scripts/tools/training_timestep_preview.py`. A full “FasterDiT-style” pass would **measure SNR PDFs on your data** and tune **loss weights + timestep schedule** jointly (experiment script, not default). |
| **Efficient scaling of DiTs** (arXiv **2412.12391**) | Large-scale study of **data + model scaling** for text-to-image DiTs (incl. U-ViT-style scaling claims in that line of work). | Use for **ablation priorities**: data quality > blind width; align **bucket / data engine** work ([IMPROVEMENTS.md](IMPROVEMENTS.md) §1.1, §11.5). |
| **DiT-Air** (arXiv **2503.10618**) | Parameter-efficient DiT variants + conditioning studies; strong **GenEval / T2I-CompBench** with less bulk. | If you need **smaller deployable checkpoints**: try **shared blocks**, **narrower FFN**, or **depth vs width** trades before adding parameters. |
| **SANA 1.5** (arXiv **2501.18427**) | **Linear / efficient DiT** ideas, **depth-growth** training, pruning, and **inference-time compute scaling** (e.g. repeated sampling + judge). | **Inference:** stack with `--num K --pick-best …`. **Train:** depth-growth = staged unfreeze / progressive depth (heavy — new trainer mode). |
| **PixelDiT** (arXiv **2511.20645**) | **Pixel-space** dual-level DiT (patch semantics + pixel detail), reducing reliance on a single VAE latent. | Major pipeline change; treat as **long-term** unless you add a **separate pixel experiment** branch. |
| **Flow matching / RF** (e.g. arXiv **2403.03206** scaling rectified flow) | Continuous-time **velocity** paths; often pairs with **fewer steps** at inference. | Still **not drop-in** for current `GaussianDiffusion` training; see table above and [IMPROVEMENTS.md](IMPROVEMENTS.md) §11.1. |

---

## Human preference & alignment (post-train)

| Direction | What it is | SDX angle |
|:----------|:-----------|:----------|
| **Diffusion-DPO** (CVPR 2024; Wallace et al.) | **Direct preference optimization** on diffusion models from pairwise (win/lose) image data. | **Stage-2** after base T2I: small LR, frozen VAE/text encoder, preference pairs from **Pick-a-Pic-style** or **internal A/B** exports. |
| **RankDPO / scalable preferences** (e.g. ICCV 2025 line) | **Ranked** or **synthetic** preferences from reward models to scale beyond hand-labeled pairs. | Use **existing pick scores** (`utils/test_time_pick.py`) as a **weak teacher** to **mine pairs** from generations (automated preference dataset). |
| **Curriculum-DPO++** (arXiv **2602.13055**) | **Curriculum** over pair difficulty + progressive capacity (unfreeze / LoRA rank). | Compose with your **caption / difficulty curriculum** knobs—same philosophy: **easy alignment first**, then hard cases. |

These do **not** replace a solid base model; they **steer** aesthetics and prompt-following once the base is stable.

---

## Product / engineering playbooks (non-paper)

- **Training ablations first** — e.g. Photoroom-style posts on **isolating one change at a time** before stacking objectives (avoids “mystery soup” configs).  
- **Noise / time sampling** — industry models increasingly treat **which noise levels you train on** as a first-class hyperparameter (you’ve started this with `timestep_sampling`).  
- **facebookresearch/flow_matching** — reference code for **continuous** flow training if you ever add a parallel trainer.

---

## Suggested reading

- arXiv **2410.10356** — FasterDiT (SNR PDF, faster DiT training).  
- arXiv **2412.12391** — Efficient scaling of diffusion transformers (T2I).  
- arXiv **2503.10618** — DiT-Air (efficient DiT + benchmarks).  
- arXiv **2501.18427** — SANA 1.5 (efficient DiT, depth growth, inference scaling).  
- arXiv **2511.20645** — PixelDiT (pixel-space DiT).  
- arXiv **2512.13421** — RecTok (flow + tokenizer).  
- arXiv **2603.06165** — Reflective flow sampling (inference).  
- Hugging Face **diffusers** SD3 discussions: logit-normal timestep sampling and loss weighting (issues #9056, #8591).  

When in doubt, prefer **documented flags** in `train.py` / `sample.py` and keep experimental objectives in a **new module** rather than overloading `GaussianDiffusion` without tests.
