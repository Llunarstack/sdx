# Danbooru tag data (local)

Large tag dumps from the Danbooru API are **gitignored** so the repo stays small. Regenerate them with:

- `scripts/tools/fetch_danbooru_tags.py` — fetch raw category files  
- `scripts/tools/split_danbooru_general_tags.py` — heuristic splits for clothes/objects/style  
- `scripts/tools/download_all_danbooru_categorized_tags.py` / `merge_danbooru_categorized_tags.py` — optional categorized workflows  

Expected layout (see root `.gitignore`):

- `tags/raw/*.txt` — raw API category lists  
- `tags/buckets/*.txt` — bucketed splits  
- `tags/all_tags_categorized.txt` — merged categorized file (when used)  

Training from Hugging Face Danbooru-style datasets is described in [docs/DANBOORU_HF.md](../../docs/DANBOORU_HF.md).
