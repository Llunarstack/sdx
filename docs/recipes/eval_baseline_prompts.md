# Baseline prompt pack for quick eval

The repo ships [`examples/eval_prompts_baseline.json`](../../examples/eval_prompts_baseline.json): short prompts that stress **object**, **hands**, **text**, **counting**, **style**, etc.

## Driver script (recommended)

[`examples/run_baseline_eval.py`](../../examples/run_baseline_eval.py) prints `sample.py` commands by default; add `--execute` to run them.

```bash
python examples/run_baseline_eval.py --ckpt results/your_run/best.pt
python examples/run_baseline_eval.py --ckpt results/your_run/best.pt --execute -- --steps 30 --preset auto
```

Arguments after `--` are forwarded to `sample.py`.

## Load in Python

```python
from pathlib import Path
from utils.generation.eval_prompt_pack import load_eval_prompt_records

for r in load_eval_prompt_records(Path("examples/eval_prompts_baseline.json")):
    print(r.id, "->", r.prompt[:60], "...")
```

## Docs

- Experiment grids: [SAMPLING_EXPERIMENTS_BACKLOG.md](../research/SAMPLING_EXPERIMENTS_BACKLOG.md)
- Quality research map: [IMAGE_QUALITY_LEVERS_2026.md](../research/IMAGE_QUALITY_LEVERS_2026.md)