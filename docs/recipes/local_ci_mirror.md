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

OUT=/tmp/basedpyright.json
python -m basedpyright --outputjson \
  native/python/sdx_native/diffusion_sigma_fast.py \
  utils/generation/run_artifacts.py \
  diffusion/snr_utils.py \
  utils/generation/inference_stages.py \
  utils/generation/eval_prompt_pack.py \
  examples/run_baseline_eval.py \
  > "$OUT" || true
python -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8-sig')); s=d.get('summary') or {}; ec=int(s.get('errorCount',0)); wc=int(s.get('warningCount',0)); print(f'basedpyright: {ec} errors, {wc} warnings'); sys.exit(1 if ec else 0)" "$OUT"

python -m scripts.tools verify_doc_links

python -m pytest tests/ -q --tb=short
```

CI uses the same logic so **warnings do not fail the job**; only `errorCount > 0` fails. The `|| true` after the redirect is required: **basedpyright** can exit non-zero when only warnings exist, and **bash `-e`** (as in GitHub Actions) would otherwise stop before the JSON gate runs.

## Optional: pre-commit

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Uses [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml) (Ruff check + format).
