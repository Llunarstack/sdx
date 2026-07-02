# Image captioning strategy (SDX)



WD Tagger and JoyTag are fast but **not accurate enough** for character identity,

series/source, or OC vs canon. They predict tag co-occurrence, not grounded identity.



## Recommended stack



| Layer | Source | Trust | What it gives you |

|-------|--------|-------|-------------------|

| 1 | **Booru API** (danbooru/e621 scrape) | Highest | `character_tags`, `copyright_tags`, `artist_tags` |

| 2 | **RAG + uncensored VLM** (`prompt_research.py`) | High | Best diffusion prompt from your corpus + image |

| 2b | **SauceNAO / TinEye** (web upload) | Medium–High | Same as browser — no API keys; booru creds fetch tags after match |

| 3 | **Creative RAG** (moondream + Qwen) | High | Synthesized prompt from image + retrieved facts |

| 4 | **Fusion** (`image_profiler.py`) | — | Booru metadata path for scraped rows (no GPU) |



**Do not** use WD/JoyTag as primary identity tags.



## Why RAG beats reverse search for your use case



You are scraping **millions of danbooru/e621 posts** into a local corpus. That corpus

already contains the tag vocabulary, character names, artist styles, and scene patterns

you care about — including NSFW.



**RAG + uncensored VLM** (`scripts/research_image_prompt.py`):



1. VLM describes the image without censorship (explicit training-data prompt)

2. TF-IDF retrieves top-k similar entries from `rag_corpus.jsonl`

3. Creative RAG (moondream2 + Qwen2.5) merges image + facts into a diffusion prompt



SauceNAO/TinEye are optional fallbacks for images **not** represented in your corpus

(reposts, rare pixiv-only art, etc.).



## For your scraped dataset

Production preprocess (`runpod/download.sh`, `scripts/run_pipeline.py`) **defaults to VLM + RAG + LLM** caption research (`SDX_PROMPT_RESEARCH=1`):

1. Build seed `rag_corpus.jsonl` from combined booru tags
2. Run `enrich_manifest_captions.py` with `--prompt-research` (moondream2 describes image → TF-IDF retrieval → Qwen synthesizes diffusion prompt)
3. Rebuild RAG from enriched captions for inference

Booru `character_tags` / `copyright_tags` / `artist_tags` are merged into the final `caption` so identity is preserved.

Fast booru-only path (no GPU): set `SDX_PROMPT_RESEARCH=0` or pass `--booru-only` to the enrich script.

Integration smoke tests use `--booru-only` for speed; unit tests cover `prompt_research.py`.



## For unknown / new images (recommended)



```bash

# Build corpus once (after scrape + enrich)

python setup/build_rag_corpus.py --data-root /workspace/data



# Research best prompt for one image

python scripts/research_image_prompt.py path/to/image.png \

  --rag-corpus /workspace/data/rag_corpus.jsonl



# At generation time (same RAG corpus)

python sample.py --prompt "..." \

  --local-rag-jsonl /workspace/data/rag_corpus.jsonl \

  --creative-rag --creative-rag-image path/to/ref.png \

  --uncensored-mode

```



### Reverse search (web upload — no API keys)

SauceNAO and TinEye work like the browser: upload the image file. No API signup required.

After a danbooru/e621 match, SDX fetches official tags using your **secret.txt** credentials
(danbooru login+api_key, e621 basic auth) — always, not anonymously.

```bash
python scripts/profile_image_cli.py path/to/image.png
```

Optional `saucenao:` / `tineye:` entries in secret.txt only if you have paid API keys for higher rate limits.



## OC detection logic



Flag `original_character` when:



- No `character_tags` or `copyright_tags` from booru **and**

- RAG corpus has no similar character/series entries **and**

- VLM prose mentions original/unknown character



## Models (local, uncensored by default)



| Role | Model | Path |

|------|-------|------|

| Image describe | moondream2 | `pretrained/moondream2` |

| Prompt synthesis | Qwen2.5-14B-Instruct | `pretrained/Qwen2.5-14B-Instruct` |

| VLM fallback chain | moondream → Qwen-VL → Florence | `caption_image_chain` |



SDX generation runs `--uncensored-mode` by default. VLM caption prompts are explicit

so training data is not sanitized.



## RAG integration



`scene_summary` and researched `caption` are indexed by `build_rag_corpus.py`.

Use `--local-rag-jsonl` at inference for factual scene grounding.



## Training



Train on `caption` from the enriched manifest — with prompt research this is a full diffusion prompt (VLM + RAG + LLM), not raw booru tags only. `scene_summary` and `booru_caption` are kept for RAG / debugging.



See also: [IMAGE_GEN_PIPELINE.md](../runpod/IMAGE_GEN_PIPELINE.md)


