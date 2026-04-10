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
| `noise_schedule_export`, `mine_preference_pairs` | `tr/` (`mine_preference_pairs` converts benchmark results into DPO-ready JSONL) |
| `data_quality`, `manifest_paths`, `jsonl_merge` | `data/` |
| `prompt_lint`, `tag_coverage`, `suggest_style_packs` | `prompt/` (`suggest_style_packs`: free-text style query -> ranked preset suggestions for `--lexicon-style` / `--art-medium-pack` / `--artist-pack` / `--color-render-pack`). |
| `export_onnx`, `export_safetensors` | `export/` |
| `op_preflight`, `orchestrate_pipeline`, `auto_improve_loop`, `gen_searcher_bridge`, `pretrained_status`, `startup_readiness`, `hybrid_dit_vit_generate` | `ops/` (`auto_improve_loop`: benchmark -> mine prefs -> DPO -> re-benchmark -> optional promote; supports `--iterations N` and hardcase-aware preference remine. `gen_searcher_bridge`: convert Gen-Searcher outputs into SDX fact JSONL for grounding + optional local shard verification. `pretrained_status`: report local-vs-HF model resolution + local size summary. `startup_readiness`: no-train environment and launch readiness report with JSON/Markdown output. `hybrid_dit_vit_generate`: DiT(+AR) candidate generation with optional ViT reranking and TCIS iterative consensus mode via `--iterations` + disagreement-aware scoring + optional shape-first scaffold synthesis + Pareto elite selection + OCR/count/saturation-aware consensus + adaptive candidate budget.) |
| `normalize_captions`, `preview_generation_prompt`, `vit_inspect`, `seed_explorer`, `eval_prompts`, `training_timestep_preview`, `image_quality_qc`, `dit_variant_compare`, `make_smoke_dataset`, `spatial_coverage`, `complex_prompt_coverage`, `prompt_gap_scout`, `prompt_i18n`, `book_scene_split` | top-level `scripts/tools/` single-file utilities (now available through dispatcher too). |
| `dump_prompt_tag_csvs`, `fetch_danbooru_tags`, `download_all_danbooru_categorized_tags`, `split_danbooru_general_tags`, `merge_danbooru_categorized_tags`, `extract_civitai_snippets_for_content_controls`, `curate_civitai_triggers`, `fetch_civitai_nsfw_concepts` | top-level data/style curation utilities (dispatcher-enabled). |
| `train_diffusion_dpo`, `train_kd_distill` | `tr/` (advanced optimization helpers). |
| `make_gallery`, `generate_sdx_architecture_diagram`, `architecture_themes`, `validate_config_json` | `dev/` auxiliary tooling (dispatcher-enabled). |
| `update_project_structure`, `verify_doc_links`, `clean_repo_artifacts` | `repo/` |

Equivalent module runs (also valid):

| Group | Example |
|------|--------|
| **dev** | `python -m scripts.tools quick_test` |
| **data** | `python -m scripts.tools data_quality` |
| **prompt** | `python -m scripts.tools prompt_lint` |
| **ops** | `python -m scripts.tools orchestrate_pipeline` |
| **export** | `python -m scripts.tools export_onnx` |
| **repo** | `python -m scripts.tools update_project_structure` |

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
