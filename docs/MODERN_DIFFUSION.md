# Modern diffusion & flow ideas (2024–2026)

Short survey of **recent directions** and how they relate to **SDX** (VP DDPM + DiT + `GaussianDiffusion`).

For a **prioritized, compounding** list of upgrades (latent stack, conditioning, objectives, distillation, alignment) mapped to repo paths, see **[DIFFUSION_LEVERAGE_ROADMAP.md](DIFFUSION_LEVERAGE_ROADMAP.md)**.

For **2026 industry + architecture + workflow** context (merged), see **[LANDSCAPE_2026.md](LANDSCAPE_2026.md)** and **[`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py)**. For **few-step math** and **prompt-accuracy** blueprints in one place, see **[BLUEPRINTS.md](BLUEPRINTS.md)** (Part 1: flow/solvers/distillation; Part 2: GLS, critics, frequency).

---

## ε-prediction, v-prediction, x₀, and flow paradigms

Many **new flagship** text-to-image stacks (e.g. SD3-class and FLUX-class lines, naming varies by release) advertise **flow matching** or **rectified flow**: learn a **vector field** that transports noise toward data along a path that is often **straighter** than classic VP-DDPM, then integrate with an ODE-style solver. That is **not the same** as switching `--prediction-type v` on a **VP** schedule—SD2-style **v** is still a reparameterization of the **same VP forward process** and the same DDIM-style family of solvers.

Rough map (examples in the wild are illustrative; product details differ by checkpoint):

| Paradigm | Typical idea | Step count (rule of thumb) | Often associated with | **SDX today** |
|:---------|:-------------|:----------------------------|:----------------------|:----------------|
| **ε-prediction** | Predict Gaussian noise in the VP forward | Often 20–50+ with strided samplers | SD 1.x, many SDXL ckpts | `--prediction-type epsilon` (default) |
| **v-prediction** | Velocity-style target tied to VP | Same solver class as ε | SD 2.x, many community finetunes | `--prediction-type v` |
| **x₀ / “data” prediction** | Predict clean latent \(x_0\) | Same VP solvers after conversion | Less common as sole T2I objective; restoration / strong img2img priors | `--prediction-type x0` (**VP only**, not flow) |
| **Flow matching** | Learn field along a **chosen** noise→data path; continuous-time view | Often **fewer** steps when **training + solver** match | SD3-class, FLUX-class (high level) | **Not** implemented as FM training; VP `GaussianDiffusion` + DDIM-style loop |
| **Rectified flow / reflow** | Straighten paths; often + **distillation** for few-step | Turbo / “1–4 step” students | InstaFlow, many Lightning/Turbo variants | **Not** implemented; needs new objective and usually a student stage |

**Caveats**

- **Fewer steps** for FM/RF in papers and products usually requires training **and** a **matched** integrator—not merely lowering `--steps` on a VP-ε or VP-v checkpoint.
- **x₀ prediction in VP** can be **ill-conditioned at very high noise** (the clean signal is weak); **min-SNR** and **non-uniform `t`** help but do not turn VP into flow matching.
- Treat forum “industry standard” claims as **directional**: VP + DiT + good data is still a dominant open-source path; FM/RF is a **parallel** design space worth understanding when reading SD3/FLUX-era docs.

### Spectral Flow Prediction (SFP) — prototype in this repo

**SFP** here means a **frequency-weighted training loss** on the VP-DDPM target (ε, v, or x₀): take `FFT(pred − target)` on the latent grid, weight radial frequency bins by timestep, and minimize weighted energy. At **high noise** (large diffusion index `t`), weights emphasize **low** spatial frequencies (global layout); near **clean** (`t` small), weights shift toward **high** frequencies (texture).

This is **not** continuous **Flow Matching** (no new noise→data ODE, no FLUX-style integrator). The forward process and **sampling** code are unchanged; only the **loss** may differ from spatial MSE.

| Item | Detail |
|:-----|:-------|
| **Code** | `diffusion/spectral_sfp.py`, `GaussianDiffusion.training_losses(..., use_spectral_sfp_loss=True)` |
| **Train CLI** | `--spectral-sfp-loss` · `--spectral-sfp-low-sigma` · `--spectral-sfp-high-sigma` · `--spectral-sfp-tau-power` |
| **MDM** | Masked diffusion training still uses **spatial** MSE (spectral loss is not mixed with patch masks in this prototype). |
| **Variable step speed per band** | **Not implemented** — inference step count is unchanged; the doc narrative about “2–3 steps for low freq” would need a **multi-rate / multi-branch** sampler, not loss weighting alone. |

Treat SFP as an **experimental** knob; ablate against plain MSE on your data before trusting it for quality claims.

**Combining SFP with x₀ prediction:** use `--prediction-type x0` and `--spectral-sfp-loss` together. The network still predicts **clean latent** \(x_0\); the loss is no longer uniform spatial MSE on \((\hat x_0 - x_0)\) but **frequency-weighted** energy of \(\mathcal{F}(\hat x_0 - x_0)\). At high noise timesteps, low spatial frequencies of that error dominate the objective; near clean timesteps, high frequencies dominate—on the **x₀ error**, not on ε or v.

---

## Implemented in SDX (this repo)

| Idea | Where | Notes |
|:-----|:------|:------|
| **Logit-normal training timesteps** | `diffusion/timestep_sampling.py`, `train.py --timestep-sample-mode logit_normal` | Discrete-time analogue of SD3-style sampling on a normalized axis; emphasizes some noise levels vs uniform `randint`. |
| **High-noise emphasis** | `--timestep-sample-mode high_noise` | `Beta(2,1)` on normalized time → more samples at **large** `t`. |
| Min-SNR loss weighting | Training | Stabilizes across timesteps. |
| V-prediction | `--prediction-type v` | Velocity parameterization (SD2-style). |
| **x0 prediction** | `--prediction-type x0` | Predicts clean latent `x_0` under **VP-DDPM** (not flow matching). DDIM path converts output to implied noise. **Not** compatible with ε/v checkpoints. At **very high noise** the target is hard to infer—use min-SNR + sensible `timestep_sample_mode`; consider **v** or **ε** if training is unstable. |
| CFG rescale / dynamic threshold | `sample.py` | High-CFG friendly. |
| Cosine **beta** schedule | `--beta-schedule cosine` | Alternative noise schedule (not the same as “cosmap” loss weighting in SD3). |
| **Sigmoid** / **squaredcos_cap_v2** schedules | `--beta-schedule sigmoid` or `squaredcos_cap_v2` | See `diffusion/schedules.py` (diffusers-style squared cosine + smooth sigmoid ramp). |
| **Soft min-SNR** loss weight | `--loss-weighting min_snr_soft` | Smooth `gamma/(snr+gamma)` weight (see `diffusion/timestep_loss_weight.py`); alternative to hard `min_snr`. |
| **Spectral SFP loss (prototype)** | `--spectral-sfp-loss` | FFT-weighted loss on pred−target; timestep-dependent radial weights (`diffusion/spectral_sfp.py`). See **SFP** subsection above. |

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
| **RankDPO / scalable preferences** (e.g. ICCV 2025 line) | **Ranked** or **synthetic** preferences from reward models to scale beyond hand-labeled pairs. | Use **existing pick scores** (`utils/quality/test_time_pick.py`) as a **weak teacher** to **mine pairs** from generations (automated preference dataset). |
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
