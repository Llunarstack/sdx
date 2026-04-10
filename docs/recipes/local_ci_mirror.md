# Mirror GitHub CI locally

Same checks as [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) (minus `actions/checkout`).

## Setup

```bash
cd /path/to/sdx
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install ruff pytest basedpyright
pip install -r requirements.txt
```

## Commands

```bash
python -m ruff check .

python -m basedpyright --level error \
  native/python/sdx_native/diffusion_sigma_fast.py \
  utils/generation/run_artifacts.py \
  diffusion/snr_utils.py \
  utils/generation/inference_stages.py

python -m scripts.tools verify_doc_links

python -m pytest tests/ -q --tb=short
```

## Optional: pre-commit

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Uses [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml) (Ruff check + format).