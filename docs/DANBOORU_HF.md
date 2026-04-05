# Training on Danbooru-style data from Hugging Face

**Full Danbooru tag vocabulary + category/bucket files:** see [`data/danbooru/README.md`](../data/danbooru/README.md) and `scripts/tools/fetch_danbooru_tags.py` / `split_danbooru_general_tags.py` (official API categories + heuristic splits for clothes/objects/style).

SDX trains from either **folders + sidecar captions** or a **JSONL manifest** (`image_path` + `caption`). Hugging Face hosts many **Danbooru-related** datasets; they are **not** all the same shape:

| Type | What you get | SDX export |
|------|----------------|------------|
| **Images + tags in HF** | Parquet/arrow with an `image` column and tag string / tag list | Use [`scripts/training/hf_export_to_sdx_manifest.py`](../scripts/training/hf_export_to_sdx_manifest.py) |
| **Tags / metadata only** | Post IDs, tag strings, no image bytes | You must **download images** separately (e.g. from Danbooru by ID) and build JSONL yourself, or use another tool |
| **Huge full dumps** | TAR/WebDataset, multi-GB | Prefer **`--streaming`** + **`--max-samples`** for a first test; scale storage before full exports |

## 1. Install

```bash
pip install datasets
```

(`datasets` is the Hugging Face `datasets` library.)

## Verified datasets (pixels + captions)

These were checked with `load_dataset(..., streaming=True)`: they expose an **`image`** column and a text caption, so [`hf_export_to_sdx_manifest.py`](../scripts/training/hf_export_to_sdx_manifest.py) can decode and write PNGs without extra download logic.

| Dataset | Columns (typical) | Notes |
|--------|-------------------|--------|
| [`YaYaB/onepiece-blip-captions`](https://huggingface.co/datasets/YaYaB/onepiece-blip-captions) | `image`, `text` | Small (~856 rows), BLIP-style captions; good for a **first HF smoke** |
| [`KorAI/onepiece-captioned`](https://huggingface.co/datasets/KorAI/onepiece-captioned) | `image`, `text` | Same pattern, different card |

**Example (export + train, caption field `text`):**

```bash
python scripts/training/hf_download_and_train.py \
  --dataset YaYaB/onepiece-blip-captions \
  --max-samples 200 \
  --streaming \
  --image-field image \
  --caption-field text \
  --out-dir data/hf_onepiece \
  --results-dir results/hf_onepiece_basic
```

Many Hub datasets named “Danbooru” only ship **tags + URLs or IDs** (e.g. `tag_string`, `file_url`, `md5`) and **no `image` bytes**—those need a separate image pipeline before SDX export.

## One command: export + train (basic DiT-B)

After you know `--image-field` and `--caption-field` for your dataset (see §2 and the table above):

```bash
python scripts/training/hf_download_and_train.py \
  --dataset YOUR_ORG/your-dataset \
  --max-samples 500 \
  --image-field image \
  --caption-field tag_string \
  --out-dir data/hf_danbooru \
  --results-dir results/danbooru_basic
```

Use **`--caption-field text`** when the dataset uses BLIP/sentence captions instead of `tag_string`.

Add `--dry-run` for a single training step, or pass extra `train.py` flags after `--`:

```bash
python scripts/training/hf_download_and_train.py --dataset YOUR/DATASET --max-samples 200 -- --max-steps 20
```

**Demo without Hugging Face** (synthetic images only):

```bash
python scripts/training/hf_download_and_train.py --demo --dry-run
```

## 2. Discover column names

On the dataset page, open the **Dataset Viewer** or run:

```bash
python scripts/training/hf_export_to_sdx_manifest.py \
  --dataset YOUR_ORG/your-dataset \
  --split train \
  --streaming \
  --list-columns
```

Adjust **`--image-field`** and **`--caption-field`** to match the dataset. Common names:

- Images: `image`, `img`
- Tags: `tag_string`, `tags`, `tag`, or a list column (use `--caption-tag-join ", "`)

## 3. Export a subset (recommended first)

Always start with a **small** `--max-samples` to verify disk space and caption quality.

```bash
python scripts/training/hf_export_to_sdx_manifest.py \
  --dataset YOUR_ORG/your-dataset \
  --split train \
  --image-field image \
  --caption-field tag_string \
  --out-dir data/hf_danbooru_subset \
  --max-samples 2000 \
  --streaming
```

This writes:

- `data/hf_danbooru_subset/images/*.png`
- `data/hf_danbooru_subset/manifest.jsonl`

## 4. Train SDX

```bash
python train.py \
  --manifest-jsonl data/hf_danbooru_subset/manifest.jsonl \
  --data-path data/hf_danbooru_subset \
  --model DiT-B/2-Text \
  --image-size 256 \
  --global-batch-size 4 \
  --results-dir results/danbooru_run
```

Use **`--data-path`** pointing at the same folder as the manifest so relative paths resolve consistently (SDX resolves `image_path` from the manifest; absolute paths in JSONL also work).

## 5. Licensing, ratings, and safety

- Danbooru-derived data may include **NSFW** content and **copyrighted** characters. You are responsible for **compliance** with the dataset license, Danbooru/Hugging Face terms, and local law.
- Many HF datasets are gated: you may need `huggingface-cli login` and to accept the dataset agreement on the Hub.
- For SFW-only pipelines, filter at export time (custom script) or choose a dataset that is explicitly filtered.

## 6. No `image` column?

If the Hub only provides **tags + metadata**, you cannot use `hf_export_to_sdx_manifest.py` as-is. Options:

1. Find a dataset that **includes decoded images** (check the dataset card).
2. Build JSONL with `image_path` pointing to files you downloaded via another pipeline (e.g. `gallery-dl`, official exports).
3. Use **WebDataset** / large-scale tooling (see [IMPROVEMENTS.md](IMPROVEMENTS.md) §4.1 roadmap) for TB-scale training.

## 7. Memory and disk

- Full Danbooru-scale exports are **terabytes**; always use **`--max-samples`** or shard exports across machines.
- For 16 GB VRAM training tips, see [HARDWARE.md](HARDWARE.md) and [SMOKE_TRAINING.md](SMOKE_TRAINING.md).
