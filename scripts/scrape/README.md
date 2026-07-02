# Booru dataset scrapers

Scrape **danbooru**, **e621**, **rule34.xxx**, and **rule34.xyz** into SDX training manifests.
Each downloaded post becomes an image file plus a JSONL row that `train.py`
consumes directly.

## Safety (mandatory)

`safety.py` hard-blocks CSAM-adjacent tags (loli/shota/cub/underage/etc.) on
**every** post before download. It cannot be disabled from the CLI. This is a
legal requirement when scraping these sites — do not remove it. The run summary
reports how many posts were blocked and by which tags.

## Credentials

Read from a secrets file, never committed. Default path
`D:\Development\secret.txt`; override with `--secrets PATH` or `$SDX_SECRETS_FILE`.
Keep this file outside the repo.

Supported logins (auto-parsed): danbooru (`login`+`api_key`), e621
(`login`+`api_key`, HTTP basic), rule34.xxx (`api_key`+`user_id`), rule34.xyz
(email/password → JWT via `/api/v2/auth/signin`, or paste a Bearer token as `api:`).

## Usage

```bash
# Dry run: fetch + filter, download nothing (verify wiring / tag query)
python -m scripts.scrape.scrape_cli --site danbooru --tags "scenery" --max-posts 20 --dry-run --out datasets/dan_scenery

# Real download, SFW only (general+sensitive on danbooru, safe on e621/rule34)
python -m scripts.scrape.scrape_cli --site e621 --tags "forest" --ratings s --max-posts 5000 --out /workspace/data/e621_forest

# Everything except the enforced blocklist
python -m scripts.scrape.scrape_cli --site rule34xxx --tags "landscape" --ratings all --max-posts 10000 --out /workspace/data/r34_landscape
```

Ratings: `s`/`safe` (SFW), `q`/`questionable`, `e`/`explicit`, or `all`.
The blocklist is always enforced regardless of `--ratings`.

## Output → training

```
<out>/images/<md5>.<ext>
<out>/manifest.jsonl          # {"image_path","caption","rating","md5","source",...}
```

```bash
python train.py --manifest-jsonl <out>/manifest.jsonl --data-path <out> \
    --flow-matching-training --epochs 20 --results-dir /workspace/results
```

## Behavior notes

- **Resumable**: reruns skip posts already in `manifest.jsonl` (by md5), so you
  can stop/restart or grow a dataset incrementally.
- **Polite rate limits**: per-site defaults (e621 capped at 1.5/s — they enforce
  ≤2/s hard). Override with `--rate`, but don't raise e621.
- **Captions** are the post's tags, comma-separated, underscores→spaces.
- **Artist tags** from danbooru/e621 are stored in each manifest row and indexed
  into ``artist_index.json`` for ``@AnyArtist`` prompt resolution.
- **GIFs & videos** (mp4/webm/gif) are downloaded and **frame-split into JPEGs**
  for training (one manifest row per frame). Production crawl:
  `SDX_MAX_POSTS=0` (unlimited), `SDX_MAX_FRAMES_PER_POST=0` (all frames),
  `SDX_FRAME_FPS=2` (or `0` for native video fps). Set `SDX_KEEP_RAW_MEDIA=1`
  to keep the original gif/mp4 on disk after splitting.
- Deduplicate across sites afterward by the `md5` field (rule34 mirrors a lot of
  danbooru).
