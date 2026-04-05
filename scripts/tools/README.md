# `scripts/tools/` — utility index

Run from repo root.

## One dispatcher (no duplicate flat shims)

Grouped tools are invoked as **subcommands** (underscores or hyphens):

```bash
python -m scripts.tools ckpt_info results/run/best.pt
python -m scripts.tools data-quality --help
python -m scripts.tools update-project-structure --max-depth 4
python -m scripts.tools help
```

| Command | Canonical script |
|---------|------------------|
| `ckpt_info` | `dev/ckpt_info.py` |
| `smoke_imports`, `quick_test`, `ar_mask_inspect` | `dev/` |
| `noise_schedule_export`, `mine_preference_pairs` | `training/` (`mine_preference_pairs` converts benchmark results into DPO-ready JSONL) |
| `data_quality`, `manifest_paths`, `jsonl_merge` | `data/` |
| `prompt_lint`, `tag_coverage` | `prompt/` |
| `export_onnx`, `export_safetensors` | `export/` |
| `op_preflight`, `orchestrate_pipeline`, `auto_improve_loop`, `startup_readiness` | `ops/` (`auto_improve_loop`: benchmark -> mine prefs -> DPO -> re-benchmark -> optional promote; supports `--iterations N` and hardcase-aware preference remine. `startup_readiness`: no-train environment and launch readiness report with JSON/Markdown output.) |
| `update_project_structure`, `verify_doc_links`, `clean_repo_artifacts` | `repo/` |

Equivalent module runs (also valid):

| Group | Example |
|------|--------|
| **dev** | `python -m scripts.tools.dev.quick_test` |
| **data** | `python -m scripts.tools.data.data_quality` |
| **prompt** | `python -m scripts.tools.prompt.prompt_lint` |
| **ops** | `python -m scripts.tools.ops.orchestrate_pipeline` |
| **export** | `python -m scripts.tools.export.export_onnx` |
| **repo** | `python -m scripts.tools.repo.update_project_structure` |

## Other scripts (single file in `tools/`)

| Tool | Purpose |
|------|--------|
| [preview_generation_prompt.py](preview_generation_prompt.py) | Print effective pos/neg after `content_controls` (no GPU / no checkpoint). |
| [benchmark_suite.py](benchmark_suite.py) | Compare one or more checkpoints on a prompt suite with composite quality scores (text/count-aware + saturation), built-in packs (`top_contender_proxy_v1`, `text_heavy_v1`, `count_stress_v1`, `biz_visual_content_v1`), optional seed-robustness ranking (`--seed-list`, `--robustness-penalty`), and hard-case mining (`--export-hardcases-jsonl`). |
| [training/mine_preference_pairs.py](training/mine_preference_pairs.py) | Convert benchmark `results.json` into DPO preference JSONL (win/lose + shared prompt), with optional hard-case boosting (`--hardcases-jsonl`, `--hardcase-extra-pairs`, `--hardcase-min-margin-scale`). |
| [data/caption_hygiene.py](data/caption_hygiene.py) | JSONL caption QA: NFKC samples, duplicate fingerprints, pos/neg overlap (`sdx_native.text_hygiene`). |
| [data/ar_tag_manifest.py](data/ar_tag_manifest.py) | Tag JSONL with `num_ar_blocks` / `ar_regime` from SDX DiT checkpoint (ViT alignment); see [docs/AR.md](../../docs/AR.md). |
| [../../toolkit/README.md](../../toolkit/README.md) | **Toolkit** (repo root): `env_health`, `manifest_digest`, seeds, timers — not under `scripts/tools/`. |

## See also

- [scripts/README.md](../README.md)
- [docs/CODEBASE.md](../../docs/CODEBASE.md)
