# Sampling experiments backlog

Concrete **grids** to run when you change samplers, CFG, or flow settings. Pair with
[`examples/eval_prompts_baseline.json`(../../examples/eval_prompts_baseline.json) and
[`utils/generation/eval_prompt_pack.py`](../../utils/generation/eval_prompt_pack.py).

## Grid A — CFG vs rescale (VP / DDIM-style)

Hold: seed, steps (e.g. 30), resolution, checkpoint.

| Run | `--cfg-scale` | `--cfg-rescale` |
|-----|---------------|-----------------|
| A1 | 5.0 | 0 |
| A2 | 7.5 | 0 |
| A3 | 7.5 | 0.7 |
| A4 | 9.0 | 0.7 |
| A5 | 9.0 | 0.85 |

**Pass if:** A3–A5 reduce saturation vs A2 without losing subject (eyeball or pick-best).

## Grid B — Steps vs solver

| Run | `--steps` | notes |
|-----|-----------|--------|
| B1 | 20 | baseline |
| B2 | 30 | |
| B3 | 50 | default-ish |
| B4 | 30 | change solver / flow `heun` vs `euler` if applicable |

## Grid C — Flow matching sample path

When `--flow-matching` (or equivalent) sampling is on:

- Sweep **CFG down** first (flows punish naive CFG).
- Compare **Heun vs Euler** at fixed NFE budget.
- Track **manifold drift** symptoms: plastic textures, color locking (see [IMAGE_QUALITY_LEVERS_2026.md](IMAGE_QUALITY_LEVERS_2026.md)).

## Grid D — Holy Grail presets

For each preset you care about:

1. Fixed 5 prompts from the baseline pack.
2. Same seed list `[0,1,2,4,8]`.
3. Export filenames that include **preset id + seed + prompt id**.

Use directory diff or `test_time_pick` metrics if you have ViT checkpoints wired.

## What not to do

- Change **three** knobs at once (CFG + steps + scheduler) — you will not know what helped.
- Tune on **one** prompt; use the **baseline pack** minimum.

---

See also [eval_baseline_prompts.md](../recipes/eval_baseline_prompts.md).