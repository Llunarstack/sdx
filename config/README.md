# `config/` layout

| Path | Role |
|------|------|
| **`train_config.py`** | `TrainConfig` dataclass + `get_dit_build_kwargs()` — main training CLI mapping. |
| **`__init__.py`** | Public exports: `TrainConfig`, `get_dit_build_kwargs`, `DEFAULT_NEGATIVE_PROMPT`. |
| **`defaults/`** | Prompt catalogs, domain lists, `sample.py` presets, PixAI labels — **not** train hyperparameters. Import explicitly, e.g. `from config.defaults.prompt_domains import ...`. |

Add new **training flags** in `train_config.py`. Add new **prompt / preset lists** under `defaults/`.
