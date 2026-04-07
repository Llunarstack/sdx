# SDX toolkit — training QoL, quality checks, small libs

Optional helpers that live **outside** the heavy `train.py` / `utils/` import graph. Use from repo root:

```bash
python -m toolkit.training.env_health
python -m toolkit.quality.manifest_digest data/manifest.jsonl
```

---

## Layout

| Path | Purpose |
|------|---------|
| [`training/env_health.py`](training/env_health.py) | GPU/CUDA/torch/cuDNN + optional deps (`timm`, `xformers`, `sdx_native`) — paste into bug reports |
| [`training/seed_utils.py`](training/seed_utils.py) | `seed_everything()` for reproducible runs |
| [`quality/manifest_digest.py`](quality/manifest_digest.py) | Fast JSONL line count, key histogram, optional Rust `sdx-jsonl-tools stats` |
| [`qol/timing.py`](qol/timing.py) | `StepTimer`, `@timed` for step/sec logging |
| [`libs/optional_imports.py`](libs/optional_imports.py) | `describe_optional_libs()` for install hints |
| [`extras/requirements-suggested.txt`](extras/requirements-suggested.txt) | **Commented** optional pip packages (xxhash, rich, humanize, …) |

---

## Related repo tools

- **Caption / manifest QA:** `python -m scripts.tools caption_hygiene`, `python -m scripts.tools data_quality`, `native/` Rust tools  
- **ViT:** `python -m vit_quality.train`, `python -m vit_quality.infer`
- **Docs:** [CODEBASE.md](../docs/CODEBASE.md), [NATIVE_AND_SYSTEM_LIBS.md](../docs/NATIVE_AND_SYSTEM_LIBS.md)

---

## Design rules

- **Stdlib-first** in `toolkit/` core; no new hard dependencies for imports to succeed.
- **Safe imports:** optional packages are probed, never required at import time.
