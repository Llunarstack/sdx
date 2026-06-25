# Reproducibility (IMPROVEMENTS 5.3)

How to get **same command → same results** when possible.

---

## Sampling (`sample.py`)

- **`--seed N`** — Fixes PyTorch RNG so the same noise is used for the initial latent.
- **`--deterministic`** — Enables:
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  - `torch.use_deterministic_algorithms(True, warn_only=True)` when supported

Use both for reproducible images:

```bash
python sample.py --ckpt .../best.pt --prompt "a cat" --seed 42 --deterministic --out out.png
```

**Limitations:** Some CUDA ops (e.g. scatter, certain reductions) can remain non-deterministic. If you see a runtime warning about a non-deterministic op, the same seed may still give slightly different results. For full bit-identical runs, use a single GPU and the same CUDA/cuDNN version.

---

## Training (`train.py`)

- **`--seed N`** — Sets `global_seed`; used for dataset split, dataloader, and initial model init.
- **`--deterministic`** — Enables:
  - Same as above (cudnn, CUBLAS, `use_deterministic_algorithms`)
  - Worker seeds in the DataLoader when `worker_init_fn` is set

```bash
python train.py --data-path ... --passes 1 --deterministic --seed 42
```

**Limitations:** Multi-GPU (DDP) can introduce non-determinism (order of gradients). For closest reproducibility, train with a single GPU and `--deterministic`.

---

## Checklist

| Goal | Sampling | Training |
|------|-----------|----------|
| Same image per run | `--seed N --deterministic` | — |
| Same training curve | — | `--seed N --deterministic`, single GPU |

For full bit-identical reproducibility, document your env: Python version, PyTorch/CUDA/cuDNN versions, and any `PYTHONHASHSEED`.
