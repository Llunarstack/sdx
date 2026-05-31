# Superior Stack — Research Map (2025–2026)

How recent diffusion alignment and inference papers map to SDX modules. This is an **implementation index**, not a claim of parity with frontier closed models.

## Alignment (training)

| Idea | Paper / trend | SDX module |
|------|---------------|------------|
| Timestep-aware DPO weighting | Rethinking DPO in Diffusion (AAAI 2026 oral) | `utils/training/dpo_advanced.py` → `--timestep-dpo-weight high_noise` |
| Safeguarded / SDPO loser scaling | Diffusion-SDPO | `safeguarded_dpo_preference_loss` → `--safeguarded-dpo 0.85` |
| Dynamic reference (EMA ref) | Online DPO variants | `ema_update_reference` → `--ref-ema-alpha 0.01` |
| Preference mining + ViT | Internal benchmark flywheel | `vit_mining.py`, `hard_negative.py` |
| Model soup promotion | Model soups (Wortsman et al.) | `model_soup.py` |

## Inference (sampling)

| Idea | Paper / trend | SDX module |
|------|---------------|------------|
| Frequency-decoupled guidance (FDG) | FDG for CFG halos | `frequency_cfg.py` → `--fdg-cfg-strength 0.65` |
| Feature / step caching | SpeCa, TeaCache | `feature_cache.py` → `--feature-cache-delta 0.012` |
| Batched CFG | Standard throughput win | `cfg_batched.py` |
| Speculative CFG draft | Draft+verify | `speculative_denoise.py` |
| Multi-metric best-of-N | Test-time scaling | `composite_ranker.py`, `online_reward.py` |
| Self-correction refine | Align-and-refine loops | `self_correct.py` |
| Block-wise DiT cache | BWCache, TeaCache | `block_cache.py` → `--block-cache-thresh` |
| Consistency / LCM distill | Latent Consistency Models | `train_consistency_distill.py` |
| Flow-GRPO online RL | NeurIPS 2025 Flow-GRPO | `train_flow_grpo.py`, `flow_grpo.py` |
| LADD adversarial distill | Flux Schnell family | `train_ladd_distill.py` |
| TaylorSeer block forecast | ICCV 2025 cache-and-forecast | `taylor_cache.py` → `--taylor-cache` |
| DenseGRPO step rewards | 2026 dense alignment | `dense_grpo.py` → `--dense-grpo` |
| Rectified-CFG++ | Flow CFG stabilization | `rectified_cfgpp.py` → `--rcfgpp-tangent` |
| Adaptive Projected Guidance (APG) | ICLR 2025 — drop parallel CFG oversaturation | `apg_guidance.py` → `--apg-parallel-eta 0` |
| Flash-GRPO | Iso-temporal grouping + grad rectification | `flash_grpo.py` → `--flash-grpo` |
| BranchGRPO | Tree branching rollouts | `branch_grpo.py` → `--branch-grpo` |
| ZeResFDG (CADE 2.5) | FDG + zero-projection + energy rescale | `zeresfdg.py` → `--zeresfdg-strength 1` |
| CFG-Zero★ | Flow CFG optimized scale + zero-init | `cfg_zero_star.py` → `--cfg-zero-star` |
| QSilk micrograin | Latent quantile clamp + detail gate | `micrograin_stabilizer.py` → `--qsilk-micrograin 0.12` |
| TurningPoint-GRPO | Step incremental + turning-point credit | `turning_point_grpo.py` → `--tp-grpo` |
| DyDiT dynamic width | Timestep-wise width scaling | `dynamic_dit.py` → `--dynamic-dit-width` |
| SLA linear attention | Marginal-block linear attn scaffold | `linear_attention.py` → `--linear-attn-fraction 0.25` |
| APG momentum | Cross-step reverse momentum | `--apg-momentum-beta 0.2` + `--apg-parallel-eta 0` |
| LCM few-step student | Consistency distill inference | `--lcm-ckpt` + `train_consistency_distill` |
| CFG++ | Manifold-constrained CFG (ICLR 2025) | `cfg_pp.py` → `--cfg-pp-lambda 0.55` |
| Interval CFG | Skip CFG early/late denoise | `cfg_interval.py` → `--cfg-skip-early-frac 0.15` |
| CFG-Rejection | Early-path filter for best-of-N | `cfg_rejection.py` |
| GRPO-Guard | Ratio-norm + timestep reweight | `grpo_guard.py` → `--grpo-guard` |
| TCFG | Tangential damping CFG (CVPR 2025) | `tcfg.py` → `--tcfg-damping 0.9` |
| SLG | Skip Layer Guidance (SD3.5) | `slg_guidance.py` → `--slg-scale 2.8` |
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
  --zeresfdg-strength 1 --cfg-rescale 0.7 \
  --cfg-zero-star --qsilk-micrograin 0.1 \
  --local-rag-jsonl datasets/facts.jsonl --superior-self-correct
```

**Guidance priority** (first match wins): interval skip → `--cfg-zero-star` → `--zeresfdg-strength` → `--cfg-pp-lambda` → `--tcfg-damping` → `--fdg-cfg-strength` → `--apg-parallel-eta` → `--rcfgpp-tangent` → standard CFG. **SLG** (`--slg-scale`) replaces the stack with CFG+skip-layer shift when active in `[slg_start, slg_stop]` progress.

For flow-matching checkpoints, add `--cfg-zero-star` (and ensure `flow_matching_sample` is on). ZeResFDG replaces separate FDG+APG for most SD/SDXL-style runs.

DPO stage after benchmark:

```bash
python -m scripts.tools train_diffusion_dpo \
  --ckpt results/best.pt --preference-jsonl prefs.jsonl \
  --timestep-dpo-weight high_noise --safeguarded-dpo 0.85 \
  --ref-ema-alpha 0.01 --dpo-beta 300
```
