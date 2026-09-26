# Board-scale training + corpus retrieval, multi-character, Ideogram & edit skills

## Training corpora (your plan)

Target boards/sites for captions + images (JSONL rows with `image` + `text`/`tags`):

| Source | Notes |
|--------|--------|
| Danbooru | Tag-rich; best for structure + artist styles |
| Gelbooru / Safebooru | Similar tag schema |
| e621 | Furry / anthro; keep separate rating filters |
| Rule34 / Rule34xyz | NSFW-heavy; consent + rating gates at sample |
| DeviantArt / ArtStation | Style/moodboard diversity; noisier captions |
| Pixiv / others | Add via same JSONL schema |

**Caption tip:** mirror multi-character structure from `utils/prompt/multi_subject.py` (`TRAINING_CAPTION_GUIDE`) so retrieval and T5 see the same ownership phrases you use at sample time.

## Corpus retrieval at sample time

Train on boards → at inference, **retrieve nearest rows** and inject:

1. Caption facts into the prompt (RAG)
2. Existing image files as InstantStyle / moodboard refs

```bash
# Build a slim index (optional)
python -m scripts.tools corpus_retrieve build \
  --jsonl data/danbooru_train.jsonl --out data/corpus_index.jsonl

# Sample with live retrieval
python sample.py --ckpt ... --prompt "1girl, catgirl, black lingerie, eipril style" \
  --corpus-jsonl data/corpus_index.jsonl --corpus-top-k 4 \
  --scheduler apex --solver apex
```

Code: `utils/generation/corpus_retrieve.py`.

## Character consistency + multi-character anti-blend

```bash
python -m scripts.tools scene_compile \
  --cast examples/multi_character_scene.example.json \
  --print-argv

python sample.py --ckpt ... --cast-json examples/multi_character_scene.example.json \
  --anti-bleed --label-multi-character-sheets
```

Compiles:

- Per-character **outfit / pose / action** ownership phrases
- Anti-blend negatives (no outfit swap, face swap, merged bodies)
- Box layout regions for regional CFG / layout attention
- Optional per-character sheets + face refs

Code: `utils/generation/multi_char_cast.py` (+ existing sheets / InstantStyle / `models/multi_character.py` for future trained isolation).

## Ideogram-like text & layout

```bash
python -m scripts.tools scene_compile \
  --prompt 'poster with "SUMMER SALE" and "50% OFF"' \
  --ideogram --glyph-text "SUMMER SALE" --palette "#FF3366,#111111" \
  --print-argv
```

Uses design brief + glyph canvas + optional palette lock (`utils/generation/edit_skills.py`).

## Photoshop-like edit skills

```bash
python -m scripts.tools scene_compile \
  --edit-image out.png --fix-region face --fix-region hands \
  --change-outfit "red jacket" --print-argv
```

Heuristic region masks → MDM inpaint argv (face/hands/clothing/background). Stack with `--reference-image` for identity lock.

## Perfect-gen combo

Corpus refs + cast + APEX:

```bash
python sample.py --ckpt ... \
  --cast-json my_cast.json \
  --corpus-jsonl data/corpus_index.jsonl \
  --scheduler apex --solver apex_adaptive \
  --steps 28
```
