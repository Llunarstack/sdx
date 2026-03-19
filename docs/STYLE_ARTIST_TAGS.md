# Style and artist tags (PixAI, Danbooru, Gelbooru)

Make the model **really good at styles** by using artist and style tags from tag-based image boards. The pipeline extracts style from captions and prompts and feeds it into the style conditioning head.

---

## How it works

1. **Training**  
   - JSONL can include a `"style"` field (e.g. `"style": "oil painting, miyazaki"`).  
   - If `style` is **empty**, the dataset **auto-fills** it from the caption using patterns like "by X", "style of X", `artist:name`, and a list of known artist/style tags (`config/style_artists.py`).  
   - Style is encoded with T5 and blended via `style_embed_dim` / `style_strength`, so the model learns to follow it without overpowering the main prompt.

2. **Inference**  
   - Use `--style "oil painting, vivid"` as usual.  
   - Or put the style in the prompt and use **`--auto-style-from-prompt`**: the script extracts a style string from the prompt (same patterns as in training) and uses it as the style conditioning.

---

## Supported patterns (extraction)

- **Phrases:** `by artist_name`, `art by X`, `drawn by X`, `in the style of X`, `style of X`, `like X`, `similar to X`  
- **Tag-board:** `artist:name`, `style:name` (Danbooru/Gelbooru style)  
- **Known tags:** If the caption contains a tag from `ARTIST_STYLE_TAGS` in `config/style_artists.py` (e.g. `ghibli`, `oil_painting`, `makoto_shinkai`), that tag is used as the style when no explicit style is set.

---

## Training with artist/style tags

1. **Explicit style in JSONL**  
   ```json
   {"image_path": "img.png", "caption": "1girl, forest", "style": "studio ghibli, miyazaki"}
   ```

2. **Style only in caption (auto-extraction)**  
   - Use captions like: `1girl, forest, drawn by makoto shinkai` or `1girl, artist:miyazaki`.  
   - Leave `"style": ""` or omit it; the dataset will fill `style` from the caption when `extract_style_from_caption=True` (default).

3. **Add your own tags**  
   - Edit `config/style_artists.py` and extend `ARTIST_STYLE_TAGS` with artist names or style tags from your dataset (e.g. from Danbooru/PixAI exports).

---

## Inference

```bash
# Explicit style
python sample.py --ckpt .../best.pt --prompt "1girl in a forest" --style "studio ghibli, miyazaki" --style-strength 0.7 --out out.png

# Style from prompt (no --style needed)
python sample.py --ckpt .../best.pt --prompt "1girl in a forest, drawn by makoto shinkai" --auto-style-from-prompt --out out.png
```

Requires a model trained with `--style-embed-dim 4096` (or your text_dim).

---

## Files

| File | Role |
|------|------|
| `config/style_artists.py` | Patterns and `ARTIST_STYLE_TAGS`; `extract_style_from_text()`. |
| `data/caption_utils.py` | `DOMAIN_TAGS["style_artist"]`: tags boosted for style learning. |
| `data/t2i_dataset.py` | Auto-fill `style` from caption when empty (`extract_style_from_caption=True`). |
| `sample.py` | `--auto-style-from-prompt`: extract style from prompt and use as style conditioning. |

---

## Tips

- Use **consistent** artist/style tags in training data (same spelling and format as on your tag board).  
- For strong style, keep **style_strength** in the 0.6–0.8 range so style doesn’t overwhelm the subject.  
- Add artist names from your dataset to `ARTIST_STYLE_TAGS` so they are recognized even without "by" or "artist:" in every caption.
