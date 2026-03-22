# `config/` layout

| Path | Role |
|------|------|
| **`train_config.py`** | `TrainConfig` dataclass + `get_dit_build_kwargs()` — main training CLI mapping. |
| **`__init__.py`** | Public exports: `TrainConfig`, `get_dit_build_kwargs`, `DEFAULT_NEGATIVE_PROMPT`. |
| **`reference/`** | Prompt catalogs, domain lists, `sample.py` presets, PixAI labels — **not** train hyperparameters. |
| **Shim files** (`prompt_domains.py`, …) | Re-export from `reference/` so `from config.prompt_domains import …` keeps working. |

Add new **training flags** in `train_config.py`. Add new **prompt / preset lists** under `reference/`.
