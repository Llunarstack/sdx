# Your training images and captions

Put **your** datasets here (this folder is for local use; image files are ignored by `.gitignore` so they are not committed).

## Quick layout (folder mode)

`train.py` expects `--data-path` to point at a directory that contains **one subdirectory per sample group** (subject, style, or a single bucket). Inside each subdirectory:

- **Image:** `.png`, `.jpg`, `.jpeg`, or `.webp`
- **Caption:** same base name as `.txt` or `.caption` (e.g. `photo.png` + `photo.txt`)

Example:

```text
user_data/
  train/                    ← pass this to --data-path
    my_photos/
      img001.png
      img001.txt
      img002.jpg
      img002.txt
    another_subject/
      a.png
      a.txt
```

Train:

```bash
python train.py --data-path user_data/train --results-dir results
```

## JSONL mode

Alternatively use a manifest (one JSON object per line) anywhere under `user_data/` or elsewhere:

```bash
python train.py --manifest-jsonl user_data/my_manifest.jsonl --results-dir results
```

See **[Data format](../README.md#data-format)** in the main README and **[docs/DANBOORU_HF.md](../docs/DANBOORU_HF.md)** for larger-scale prep.

## Notes

- Keep backups of your originals outside the repo if needed.
- For smoke tests, see **`scripts/tools/make_smoke_dataset.py`**.
