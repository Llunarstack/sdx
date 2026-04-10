# Quick evaluation recipe (Holy Grail + manifests)

Copy-paste oriented. Paths assume repo root.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## 2. One-command sanity (no custom checkpoint)

Uses DiT-XL/2 ImageNet weights from Hugging Face (see repo `demo.py`).

```bash
python demo.py
```

## 3. Sample with a text checkpoint + Holy Grail preset

Replace the checkpoint path with your `results/.../best.pt` (or equivalent).

```bash
python sample.py --ckpt results/your_run/best.pt \
  --prompt "your prompt here" \
  --preset auto \
  --out out.png
```

Adjust steps, resolution, and CFG flags per `README.md` and `docs/QUALITY_AND_ISSUES.md`.

## 4. Training run artifacts

When training with `train.py`, the experiment directory receives reproducibility files (see `utils.generation.run_artifacts`):

- `config.train.json` — frozen training config snapshot
- `run_manifest.json` — command, git info, torch/CUDA versions, seed, etc.

Disable with `--no-save-run-manifest` if you truly do not want these files.

## 5. Deeper reading

- [HOLY_GRAIL_OVERVIEW.md](../HOLY_GRAIL_OVERVIEW.md)
- [TCIS_OVERVIEW.md](../TCIS_OVERVIEW.md)
- [QUALITY_AND_ISSUES.md](../QUALITY_AND_ISSUES.md)