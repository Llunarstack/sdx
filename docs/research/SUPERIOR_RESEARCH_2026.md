# Superior Stack — Research Map (2025–2026)

How recent diffusion alignment and inference papers map to SDX modules. This is an **implementation index**, not a claim of parity with frontier closed models.

## Alignment (training)

| Idea | Paper / trend | SDX module |
|------|---------------|------------|
| Timestep-aware DPO weighting | Rethinking DPO in Diffusion (AAAI 2026 oral) | `utils/training/preference.py` → `--timestep-dpo-weight high_noise` |
| Safeguarded / SDPO loser scaling | Diffusion-SDPO | `safeguarded_dpo_preference_loss` → `--safeguarded-dpo 0.85` |
| Dynamic reference (EMA ref) | Online DPO variants | `ema_update_reference` → `--ref-ema-alpha 0.01` |
| Preference mining + ViT | Internal benchmark flywheel | `vit_mining.py`, `hard_negative.py` |
| Model soup promotion | Model soups (Wortsman et al.) | `model_soup.py` |

## Inference (sampling)

| Idea | Paper / trend | SDX module |
|------|---------------|------------|
| Composable guidance stack | APG, FDG, CFG rescale, CFG-Zero★, TCFG, PAG, SLG and the guidance interval compose | `guidance_stack.py` (see below) |
| Feature / step caching | SpeCa, TeaCache | `feature_cache.py` → `--feature-cache-delta 0.012` |
| Batched CFG | Standard throughput win | `cfg_batched.py` |
| Multi-metric best-of-N | Test-time scaling | `composite_ranker.py` over `test_time_pick` terms (incl. `ai_look.py`, `art_critic.py`) |
| Self-correction refine | Align-and-refine loops | `self_correct.py` |
| Block-wise DiT cache | BWCache, TeaCache | `block_cache.py` → `--block-cache-thresh` |
| Consistency / LCM distill | Latent Consistency Models | `train_consistency_distill.py` |
| Flow-GRPO online RL | NeurIPS 2025 Flow-GRPO | `train_flow_grpo.py`, `flow_grpo.py` |
| LADD adversarial distill | Flux Schnell family | `train_ladd_distill.py` |
| TaylorSeer block forecast | ICCV 2025 cache-and-forecast | `taylor_cache.py` → `--taylor-cache` |
| DenseGRPO step rewards | 2026 dense alignment | `dense_grpo.py` → `--dense-grpo` |
| Adaptive Projected Guidance (APG) | Sadat et al., ICLR 2025 | `--apg-parallel-eta 0 --apg-momentum-beta -0.5 [--apg-norm-threshold r]` |
| Flash-GRPO | Iso-temporal grouping + grad rectification | `flash_grpo.py` → `--flash-grpo` |
| BranchGRPO | Tree branching rollouts | `branch_grpo.py` → `--branch-grpo` |
| Frequency-decoupled guidance (FDG) | Sadat et al. 2025, Laplacian split | `--fdg-low 0.6 --fdg-high 1.3` |
| CFG-Zero★ | Fan et al. 2025; zero-init on flow only | `--cfg-zero-star` |
| QSilk micrograin | Latent quantile clamp + detail gate | `latent_filters.py` → `--qsilk-micrograin 0.12` |
| TurningPoint-GRPO | Step incremental + turning-point credit | `turning_point_grpo.py` → `--tp-grpo` |
| DyDiT dynamic width | Timestep-wise width scaling | `dynamic_dit.py` → `--dynamic-dit-width` |
| SLA linear attention | Softmax/linear blend (elu+1 kernel) | `models/linear_attention.py` → `--linear-attn-fraction 0.25` |
| LCM few-step student | Consistency distill inference | `--lcm-ckpt` + `train_consistency_distill` |
| Guidance interval | Kynkäänniemi et al. 2024, keyed on noise level | `--cfg-interval 0.28,5.42` |
| CFG-Rejection | Early-path filter for best-of-N | `cfg_rejection.py` |
| GRPO-Guard | Ratio-norm + timestep reweight | `grpo_guard.py` → `--grpo-guard` |
| TCFG | Kwon et al. 2025, SVD of [cond; uncond] | `--tcfg-damping 1` |
| SLG | Skip Layer Guidance (SD3.5) | `--slg-scale 2.8` |
| PAG | Perturbed-Attention Guidance (Ahn et al. 2024) | `--pag-scale 3` (batched with the CFG forward) |
| DBCache CFG split | Cache-DiT cond-only fingerprint | `--dbc-separate-cfg` + block cache |
| CFG-Rejection rerank | Early CFG gap multi-sample order | `--cfg-rejection-rerank` |

## Data flywheel

| Stage | SDX tool |
|-------|----------|
| Curate JSONL | `utils/data_quality/pipeline.py`, `superior_curate` |
| Benchmark | `benchmark_suite.py` |
| Mine prefs | `mine_preference_pairs.py`, `vit_mining.py` |
| DPO train | `train_diffusion_dpo.py` |
| Promote | `model_soup.py`, `run_flywheel.py` |

## Honest ceiling

No wrapper replaces **pretraining scale**, **proprietary data**, or **frontier VAE/text encoders**. The Superior Stack maximizes what you can extract from **your checkpoint + your corpus + your compute** via alignment loops and inference orchestration.

## Recommended defaults (`--preset superior`)

```bash
python sample.py --ckpt ... --preset superior --num 4 \
  --pick-best superior_composite \
  --apg-parallel-eta 0 --fdg-low 0.6 --fdg-high 1.3 --cfg-rescale 0.3 \
  --cfg-zero-star --qsilk-micrograin 0.1 \
  --local-rag-jsonl datasets/facts.jsonl --superior-self-correct
```

**Guidance composition** (`utils/generation/guidance_stack.py`): every enabled technique runs, in this order. TCFG and the CFG-Zero★ scale adjust the uncond branch in the model's output space; both branches map to x0; APG (momentum, norm cap, projection) runs per FDG band; PAG and SLG add `s*(cond - perturbed)`; CFG rescale (Lin et al.) runs last. Outside the `--cfg-interval` sigma band the uncond forward is skipped. CFG-Zero★ zero-init applies to flow-matching checkpoints only.

DPO stage after benchmark:

```bash
python -m scripts.tools train_diffusion_dpo \
  --ckpt results/best.pt --preference-jsonl prefs.jsonl \
  --timestep-dpo-weight high_noise --safeguarded-dpo 0.85 \
  --ref-ema-alpha 0.01 --dpo-beta 300
```
