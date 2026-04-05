# Holy Grail Diffusion Lab

This folder is a concentrated playground for high-upside inference/training ideas:

- **Adaptive CFG from attention entropy** (`guidance_fusion.py`)
- **Prompt coverage scoring + penalties** (`prompt_coverage.py`)
- **Latent detail refinement + dynamic clamp** (`latent_refiner.py`)
- **Unified per-step plan for CFG / ControlNet / adapters** (`blueprint.py`)
- **CADS-style condition annealing** (`condition_annealing.py`)
- **Style/detail progression router** (`style_router.py`)
- **Preset bundles + auto recommender** (`presets.py`, `recommender.py`)
- **Runtime safety sanitizer** (`runtime_guard.py`)
    
## Why this exists

Fixed guidance and static condition scaling often leave quality on the table. This package gives a
simple way to make sampling behave more like a policy:
- stronger structure control early,
- better style/detail expression late,
- attention-driven CFG modulation per sample.

## Minimal integration sketch

```python
from diffusion.holy_grail import HolyGrailRecipe, build_holy_grail_step_plan

recipe = HolyGrailRecipe(base_cfg=7.5, control_base_scale=1.2, adapter_base_scale=1.0)
for i in range(total_steps):
    plan = build_holy_grail_step_plan(recipe=recipe, step_index=i, total_steps=total_steps)
    # use: plan.cfg_scale, plan.control_scale, plan.adapter_scale, plan.refine_strength
```

## Notes

- Everything here is tensor-safe, standalone, and testable.
- Intended to be wired behind feature flags first, then benchmarked.
- Runtime wiring now exists in `diffusion/gaussian_diffusion.py` and `sample.py`.

## CLI starter

```bash
python sample.py --ckpt path/to.pt --prompt "..." --holy-grail \
  --holy-grail-preset auto \
  --holy-grail-cads-strength 0.03 \
  --holy-grail-cfg-early-ratio 0.72 \
  --holy-grail-cfg-late-ratio 1.0 \
  --holy-grail-control-mult 1.1 \
  --holy-grail-unsharp-sigma 0.6 \
  --holy-grail-unsharp-amount 0.18 \
  --holy-grail-clamp-quantile 0.995
```

