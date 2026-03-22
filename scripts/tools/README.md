# `scripts/tools/` — utility index

Run from **repository root** (`python scripts/tools/<name>.py` or `python -m scripts.tools.<name>`).

Tools stay **flat** under this folder so `python -m scripts.tools.quick_test` and imports like `from scripts.tools.book_scene_split import …` remain stable. Use the tables below to find what you need by **purpose**.

---

## Smoke test & development

| Script | Purpose |
|--------|---------|
| **`quick_test.py`** | One DiT forward pass — verify torch, models, env |
| **`smoke_imports.py`** | Import-check internal packages (`python -m scripts.tools.smoke_imports`) |
| **`dit_variant_compare.py`** | Parameter counts / size estimates for DiT / EnhancedDiT registry names |
| **`ckpt_info.py`** | Print checkpoint config and keys without full model load |
| **`seed_explorer.py`** | Grid explore seeds / presets for a checkpoint |
| **`training_timestep_preview.py`** | ASCII histograms for `--timestep-sample-mode` distributions |
| **`vit_inspect.py`** | ViT QA checkpoint: config, param count, optional module tree |

---

## Data & manifests

| Script | Purpose |
|--------|---------|
| **`data_quality.py`** | Filter / dedup JSONL or folders (phash, md5, caption length, bad words) |
| **`normalize_captions.py`** | Normalize caption fields in a manifest |
| **`make_smoke_dataset.py`** | Synthetic PNGs + captions for smoke `train.py` |
| **`book_scene_split.py`** | Split `## Page N` / `---PAGE---` scripts → one line per page for `generate_book.py` |
| **`image_quality_qc.py`** | Per-image QC metrics on manifest (sharpness, contrast thresholds) |

---

## Coverage & prompt analysis

| Script | Purpose |
|--------|---------|
| **`tag_coverage.py`** | Scan JSONL for hard-style / person / anatomy / concept-bleed tags |
| **`spatial_coverage.py`** | Spatial wording coverage (`behind`, `next to`, …) |
| **`complex_prompt_coverage.py`** | Tricky categories (clothes, weapons, food, text, NSFW, …) |
| **`prompt_gap_scout.py`** | Single-prompt gap analysis + i18n suggestions |
| **`prompt_lint.py`** | CLI wrapper for JSONL prompt lint (see `utils/prompt_lint.py`) |
| **`prompt_i18n.py`** | Shared i18n helpers (imported by other tools) |
| **`eval_prompts.py`** | Evaluate prompts against a checkpoint (used by `op_pipeline.ps1`) |

---

## Export & ops

| Script | Purpose |
|--------|---------|
| **`export_onnx.py`** | Export DiT checkpoint → ONNX |
| **`export_safetensors.py`** | Export DiT weights → `.safetensors` (ComfyUI / A1111) |
| **`op_preflight.py`** | Pre-train gate: coverage + thresholds (PASS/FAIL) |
| **`op_pipeline.ps1`** | Windows: chain preflight → normalize → optional eval |

---

## Repo layout generator

| Script | Purpose |
|--------|---------|
| **`update_project_structure.py`** | Write **`PROJECT_STRUCTURE.md`** at repo root (ASCII tree; run after moves). `python scripts/tools/update_project_structure.py --help` |
| **`verify_doc_links.py`** | Check relative `[]()` links in README, `docs/`, `pipelines/`, `ViT/`, `scripts/**/*.md` — `python scripts/tools/verify_doc_links.py` (exit 1 if broken). |

## See also

- [scripts/README.md](../README.md) — all of `scripts/`
- [docs/REPOSITORY_STRUCTURE.md](../../docs/REPOSITORY_STRUCTURE.md) — full repo map
- [docs/FILES.md](../../docs/FILES.md) — exhaustive file list
