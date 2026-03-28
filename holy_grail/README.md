# Holy Grail (sampling stack)

This folder is a **landing page** on GitHub. All implementation code lives under:

**[`diffusion/holy_grail/`](../diffusion/holy_grail/)**

Python import path: `diffusion.holy_grail` (e.g. `from diffusion.holy_grail import build_holy_grail_step_plan`).

## What it does

- Per-step **CFG**, **ControlNet**, and **adapter** scaling inside `GaussianDiffusion` (VP + flow sampling).
- Optional **CADS-style** noise on text embeddings during sampling.
- **Presets** (`balanced`, `photoreal`, `anime`, `illustration`, `aggressive`) and `--holy-grail-preset auto`.
- Latent **unsharp** + **dynamic clamp** hooks at the end of sampling when enabled.

## CLI (`sample.py`)

Enable with `--holy-grail` or pick a preset:

```bash
python sample.py --ckpt …/best.pt --prompt "…" --holy-grail-preset auto --out out.png
```

Full flag list: [diffusion/holy_grail/README.md](../diffusion/holy_grail/README.md) and the main [README.md](../README.md) (section *Latest model updates → §6*).

## Tests

Unit tests: `tests/unit/test_holy_grail_diffusion.py`, `tests/unit/test_holy_grail_presets.py`.

The repo keeps a **`tests/`** tree for CI and regression checks; it is not redundant with this feature.
