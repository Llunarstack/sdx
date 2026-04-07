# Civitai site searches → API approximation

The [public REST API](https://developer.civitai.com/docs/api/public-rest) does **not** support the same knobs as the web UI (`category=clothing|concept|character|style`, multiple `baseModel=`, `sortBy=models_v9`).

This repo approximates your saved **site** searches like:

- [clothing + Illustrious + NoobAI + nsfw](https://civitai.com/search/models?category=clothing&baseModel=Illustrious&baseModel=NoobAI&sortBy=models_v9&query=nsfw)
- [clothing](https://civitai.com/search/models?category=clothing&sortBy=models_v9), [clothing + sex](https://civitai.com/search/models?category=clothing&sortBy=models_v9&query=sex)
- [concept + sex](https://civitai.com/search/models?category=concept&sortBy=models_v9&query=sex)
- [character + sex / nsfw / hentai / …](https://civitai.com/search/models?category=character&sortBy=models_v9&query=sex)
- [style + nsfw / anime / hentai / …](https://civitai.com/search/models?category=style&sortBy=models_v9&query=nsfw)
- Global searches: [waifu](https://civitai.com/search/models?sortBy=models_v9&query=waifu), [pov](https://civitai.com/search/models?sortBy=models_v9&query=pov), [paizuri](https://civitai.com/search/models?sortBy=models_v9&query=paizuri), etc.

## What we actually do

1. **`GET /api/v1/models`** with `query=<term>`, `nsfw=true` (unless `--no-nsfw-flag`), cursor pagination.
2. One pass with **no** `query=` (empty slot in the list) to approximate “browse” / category-only pages.
3. **Post-filter**: keep only models that have at least one `modelVersions[].baseModel` in **`Illustrious`** or **`NoobAI`** (same as your Illustrious+NoobAI links).
4. **Merge** all passes by `id` and **union** `bases` + `triggers`.

The ordered query list lives in `scripts/tools/fetch_civitai_nsfw_concepts.py` as `EXTENDED_SEARCH_QUERIES`.

## Refresh the bank

```bash
python -m scripts.tools fetch_civitai_nsfw_concepts --preset extended --max-batches-per-query 8 --sleep 0.2 --out data/civitai/nsfw_illustrious_noobai_models.csv
python -m scripts.tools curate_civitai_triggers --names-out data/civitai/model_names.txt
```

Increase `--max-batches-per-query` for deeper coverage (more API calls).

If a single query returns HTTP 500/502/etc. after retries, that query is **skipped** and the merge continues (see stderr `SKIP query`).
