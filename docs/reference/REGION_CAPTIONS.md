# Regional & layout captions (JSONL)

Train the text encoder + DiT on **where things are**, not only a single global tag string — **without** changing model architecture.

## Idea

1. From each training image you (or a VLM/segmentation pipeline) produce **short labels per region**: subject, clothing, hands, background, etc.
2. Store them in the manifest as **`parts`** (dict) and/or **`region_captions`** (list).
3. At load time, `Text2ImageDataset` merges them into one T5 string using a fixed **`[layout]`** prefix so the model learns a consistent “parts + global scene” pattern.

Optional: combine with **`--crop-mode random`** so crops + layout text encourage local detail.

## JSONL fields

| Field | Type | Description |
|:------|:-----|:------------|
| `caption` | string | Global prompt (tags or sentence) — **required** as today |
| `parts` | object | e.g. `{"subject": "1girl, long hair", "clothing": "red dress", "background": "forest"}` |
| `region_captions` | list | Strings and/or `{"label": "...", "text": "..."}` |
| `segments` | alias | Same as `region_captions` |

You may send **both** `parts` and `region_captions`; they are merged (`parts` first, then list).

### Example

```json
{
  "image_path": "/data/img001.png",
  "caption": "1girl, masterpiece, best quality, full body",
  "negative_caption": "lowres, bad anatomy",
  "parts": {
    "subject": "young woman, long black hair, looking at viewer",
    "clothing": "white blouse, pleated skirt",
    "hands": "visible hands, five fingers",
    "background": "city street, sunset, bokeh"
  },
  "region_captions": [
    {"label": "foreground", "text": "character centered"},
    {"label": "lighting", "text": "warm rim light"}
  ]
}
```

Merged into training (default **`--region-caption-mode append`**):

`… global caption after boosts … . [layout] subject: … | clothing: … | … | foreground: … | lighting: …`

## CLI (`train.py`)

| Flag | Default | Meaning |
|:-----|:--------|:--------|
| `--region-caption-mode` | `append` | `append` — after global caption; `prefix` — layout before global; `off` — ignore regions |
| `--region-layout-tag` | `[layout]` | Token(s) before the regional block; use `""` to disable |

## Inference

At sample time, write prompts in the **same style** your data used, e.g. end with  
`. [layout] subject: … | background: …`  
or rely on a strong global caption only — the model is not required to see `[layout]` at inference if you never trained with it (use `--region-caption-mode off` to A/B).

## See also

- [docs/FILES.md](FILES.md) — data module entry points  
- [README § Data format](../../README.md#data-format) — manifest overview  
