# Superior Stack

Composable modules that raise **inference and data-prep quality** on top of your trained DiT checkpoint. This is not a separate model — it is orchestration, retrieval, ranking, and correction around `sample.py`.

## Modules (`utils/superior/` → `utils/_archive/superior/`)

Implementations live in the archive; import via `utils.superior` shims.

| Module | Role |
|--------|------|
| `retrieval.py` | TF-IDF over JSONL facts (local RAG, no API) |
| `auto_stack.py` | Merge retrieved facts + light quality hints into the prompt |
| `composite_ranker.py` | Multi-metric `--pick-best superior_composite` |
| `self_correct.py` | CLIP alignment gate + short refine pass |
| `glyph_encoder.py` | Byte-hash sidecar for text-in-image experiments |
| `distill.py` | LADD distillation config helpers |
| `inference_pipeline.py` | Build `sample.py` argv for high-quality defaults |
| *(generation)* `apg_guidance.py` | Adaptive Projected Guidance (`--apg-parallel-eta`) |

## Quick start

**CLI wrapper** (4 candidates + composite ranking + optional RAG):

```bash
python -m scripts.tools superior_generate \
  --ckpt results/your_run/best.pt \
  --prompt "neon alley at night, cinematic" \
  --local-rag-jsonl datasets/style_notes.jsonl \
  --num 4 --out out.png
```

**Direct `sample.py` flags:**

```bash
python sample.py --ckpt results/.../best.pt \
  --prompt "..." --num 4 --pick-best superior_composite \
  --local-rag-jsonl datasets/captions.jsonl --local-rag-top-k 8 \
  --superior-self-correct --compile-inference --out out.png
```

## Training-side

- **`python -m scripts.tools superior_dpo_loop`** — mine benchmark pairs → train Diffusion-DPO
- **`python -m scripts.tools superior_curate`** — dedup/filter JSONL via `utils/data_quality/`
- **`python -m scripts.tools model_soup`** — average multiple checkpoints
- Use **`utils/training/ladd_distillation.py`** for teacher→student latent distillation (custom loop).
- See **`scripts/tools/training/train_kd_distill.py`** for KD entry points.

## Full alignment loop (wave 3)

One command runs benchmark → ViT-weighted preference mining → DPO → re-benchmark → optional model soup:

```bash
python -m scripts.tools superior_auto_loop \
  --base-ckpt results/run/best.pt \
  --vit-ckpt vit_quality/results/best.pt \
  --local-rag-jsonl datasets/style_notes.jsonl \
  --model-soup --iterations 2
```

Lower-level control: `python -m scripts.tools auto_improve_loop` with `--vit-ckpt`, `--local-rag-jsonl`, `--model-soup`.

## Full flywheel (wave 4)

Curate data → align (benchmark/DPO/soup) → promote best checkpoint:

```bash
python -m scripts.tools run_flywheel \
  --base-ckpt results/run/best.pt \
  --manifest-in datasets/train.jsonl \
  --manifest-out datasets/train_clean.jsonl \
  --local-rag-jsonl datasets/style_facts.jsonl \
  --vit-ckpt vit_quality/results/best.pt \
  --work-dir flywheel_run
```

**Ensemble inference** (multiple checkpoints, one winner):

```bash
python -m scripts.tools superior_ensemble \
  --checkpoints results/a/best.pt results/b/dpo_policy.pt soup_policy.pt \
  --prompt "portrait of an engineer" --num-per-ckpt 2 --out best.png
```

**Eval report** from benchmark output:

```bash
python -m scripts.tools superior_eval_report bench_after/
```

Use **`--preset superior`** in ``sample.py`` for Superior Stack sampler defaults.

## Libraries added (wave 4)

| Module | Role |
|--------|------|
| `config/defaults/superior_stack.py` | Central defaults + `FlywheelPlan` |
| `utils/superior/flywheel.py` | Curate + align orchestration |
| `utils/superior/ensemble.py` | Multi-checkpoint generation + global rank |
| `utils/superior/hard_negative.py` | Mine negatives from failed cases |
| `utils/superior/eval_report.py` | Markdown benchmark reports |

## Libraries added (wave 3)

| Module | Role |
|--------|------|
| `utils/superior/vit_mining.py` | ViT-weighted DPO pair mining |
| `utils/superior/reward_scorer.py` | Unified ViT + heuristic reward |
| `utils/superior/prompt_expand.py` | LLM or heuristic prompt expansion |
| `utils/superior/auto_loop.py` | Python API for auto-improve loop |

| Module | Role |
|--------|------|
| `utils/data_quality/` | Programmatic manifest curation API |
| `utils/superior/dpo_pipeline.py` | Benchmark → DPO training orchestration |
| `utils/superior/model_soup.py` | Checkpoint averaging |
| `utils/superior/quality_gates.py` | Fast pre/post generation gates |
| `models/taca.py` | Fused SDPA cross-attention path |

## Wave 5 (research-backed alignment + inference)

| Module | Role |
|--------|------|
| `utils/training/dpo_advanced.py` | Timestep DPO weights, SDPO safeguard, EMA ref |
| `utils/superior/frequency_cfg.py` | FDG latent CFG (`--fdg-cfg-strength`) |
| `utils/superior/feature_cache.py` | SpeCa-lite step reuse (`--feature-cache-delta`) |
| `utils/superior/online_reward.py` | Online RL / best-of-N reward scaffold |

**DPO trainer flags:** `--timestep-dpo-weight high_noise`, `--safeguarded-dpo 0.85`, `--ref-ema-alpha 0.01`

**Windows flywheel:** `scripts/tools/training/run_superior_flywheel.ps1`

See [research/SUPERIOR_RESEARCH_2026.md](research/SUPERIOR_RESEARCH_2026.md) for paper mapping.

## Wave 6 (speed + online RL + distill flywheel)

| Module | Role |
|--------|------|
| `utils/superior/block_cache.py` | BWCache/TeaCache-lite per-block DiT reuse |
| `utils/training/flow_grpo.py` | Flow-GRPO scaffold (sample → reward → weighted update) |
| `train_consistency_distill.py` | LCM-style consistency distillation (few-step student) |
| `train_flow_grpo.py` | Online GRPO alignment trainer |
| `train_ladd_distill.py` | LADD teacher MSE + latent discriminator |
| `hard_negative.py` | Flywheel closure via `--use-hard-negatives` in auto_improve_loop |

**Block cache inference:** `--block-cache-thresh 0.18 --block-cache-recompute-every 4`

**Few-step student:** `python -m scripts.tools train_consistency_distill --teacher-ckpt ... --data ...`

**Online RL:** `python -m scripts.tools train_flow_grpo --ckpt ... --prompts prompts.txt --vit-ckpt ...`

## Wave 7 (TaylorSeer + DenseGRPO + LCM fast path)

| Module | Role |
|--------|------|
| `utils/superior/taylor_cache.py` | TaylorSeer block forecast (`--taylor-cache`) |
| `utils/training/dense_grpo.py` | Step-wise dense GRPO rewards |
| `utils/generation/rectified_cfgpp.py` | Rectified-CFG++ for flow (`--rcfgpp-tangent`) |
| `sample.py --lcm-ckpt` | Few-step consistency student (4-step flow) |

**Taylor + block cache:** `--block-cache-thresh 0.18 --taylor-cache --taylor-cache-order 1`

**DenseGRPO training:** `train_flow_grpo --dense-grpo --dense-ode-steps 4`

**LCM inference:** `sample.py --ckpt base.pt --lcm-ckpt lcm_student.pt --lcm-steps 4`

## Honest scope

Frontier labs have proprietary data scale, eval harnesses, and foundation pretraining. SDX Superior Stack gives you **testable, local** levers: grounded prompts, better candidate selection, and alignment-based refine — wired to your checkpoint and prompt stack. Quality still depends on **training data and your DiT weights**.

## See also

- [IMPROVEMENTS.md](IMPROVEMENTS.md) — training/inference roadmap
- [LANDSCAPE_2026.md](LANDSCAPE_2026.md) — industry patterns (Designer / Verifier / Reasoner)
- [recipes/fast_training.md](recipes/fast_training.md) — throughput defaults
