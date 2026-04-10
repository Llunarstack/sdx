# Hugging Face dataset shortlist (from provided links)

This file is the curated training shortlist from your exact dataset URLs, focused on:

- scale (enough coverage for broad prompt adherence),
- tag/caption utility (good conditioning signal),
- quality/consistency (fewer broken/low-value samples),
- practical ingestion for SDX (`image` + tags/text preferred).

## Primary datasets (use these first)

1. [`vikhyatoolkit/danbooru2023`](https://huggingface.co/datasets/vikhyatoolkit/danbooru2023)
   - Best overall baseline from your list for anime-style text-to-image coverage.
   - Large scale + broadly useful tag distribution.

2. [`ShinoharaHare/Danbooru-2024-Filtered-1M`](https://huggingface.co/datasets/ShinoharaHare/Danbooru-2024-Filtered-1M)
   - Strong quality-focused subset for cleaner signal.
   - Good anchor dataset when training for prompt adherence and fewer artifacts.

3. [`KBlueLeaf/danbooru2023-webp-4Mpixel`](https://huggingface.co/datasets/KBlueLeaf/danbooru2023-webp-4Mpixel)
   - Practical high-resolution cap (`4Mpixel`) can help detail/composition.
   - Good companion for stronger high-detail generations.

4. [`ppbrown/danbooru-cleaned`](https://huggingface.co/datasets/ppbrown/danbooru-cleaned/tree/main)
   - "Cleaned" variant helps stabilize training quality.
   - Useful as a quality-biased mixing component.

5. [`ma-xu/fine-t2i`](https://huggingface.co/datasets/ma-xu/fine-t2i)
   - Very strong general-purpose T2I corpus: large-scale (~6M+) and high-quality filtering.
   - Excellent for prompt-following and composition robustness; good as a major non-booru complement.
   - Notes: WebDataset shards, very large (~2 TB). Start with streaming and controlled subset exports.

## Secondary datasets (blend after core is stable)

6. [`ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions`](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions)
   - Strong caption quality and broad concept diversity; useful for one-shot prompt adherence.
   - Good for caption-conditioned instruction quality, but still synthetic-model output data.

7. [`p1atdev/danbooru-2024`](https://huggingface.co/datasets/p1atdev/danbooru-2024/tree/main/data)
   - Good freshness/recency add-on.
   - Use after validating schema and duplicate overlap with core sets.

8. [`zenless-archive/danbooru2023`](https://huggingface.co/datasets/zenless-archive/danbooru2023)
   - Potentially useful mirror/variant; verify overlap and caption fields before heavy use.

9. [`hipete12/danbooru-2025-10-05`](https://huggingface.co/datasets/hipete12/danbooru-2025-10-05)
   - Potentially valuable newer slice.
   - Requires schema and quality validation before promotion to primary.

## Domain-specific optional sets

10. [`deepghs/rule34_full`](https://huggingface.co/datasets/deepghs/rule34_full)
   - NSFW-heavy coverage; useful if your target model explicitly includes this domain.
   - Keep as optional or separate curriculum stage.

11. [`NebulaeWis/e621-2024-webp-4Mpixel`](https://huggingface.co/datasets/NebulaeWis/e621-2024-webp-4Mpixel)
12. [`NebulaeWis/e621_20250524_selected`](https://huggingface.co/datasets/NebulaeWis/e621_20250524_selected)
13. [`hearmeneigh/e621-rising-v3-small`](https://huggingface.co/datasets/hearmeneigh/e621-rising-v3-small)
14. [`hearmeneigh/e621-rising-v3-micro`](https://huggingface.co/datasets/hearmeneigh/e621-rising-v3-micro)
   - Furry-domain specialization; include only if that domain is a target.

15. [`moonworks/lunara-aesthetic`](https://huggingface.co/datasets/moonworks/lunara-aesthetic)
   - Small, high-quality controlled set (~2k pairs) with structured labels.
   - Best use: validation set, style/aesthetic finetune stage, or prompt-grounding evaluation.
   - Not suitable as a core pretraining corpus due to size.

## Lower-priority / duplicate-risk from your list

- `nyuuzyou/*` rule34 mirrors and split parts are likely useful for volume, but often have high overlap/duplication and mixed consistency. Keep for backfill, not as foundation.
- `adbrasi/rule34_dataset` and other broad rule34 mirrors should be deduped aggressively before any large mix-in.
- `xingjianleng/danbooru_images`, `picollect/danbooru`, `Wenaka/danbooru_variant` may be useful, but promote only after schema and quality checks.
- `nbeerbower/fixbody-dpo-danbooru` is better treated as preference/alignment data, not as core pretraining corpus.

## Recommended initial mix

Start with this weighted mix for first major training runs:

- 40% `vikhyatoolkit/danbooru2023`
- 20% `ma-xu/fine-t2i`
- 15% `ShinoharaHare/Danbooru-2024-Filtered-1M`
- 15% `KBlueLeaf/danbooru2023-webp-4Mpixel`
- 10% `ppbrown/danbooru-cleaned`

Then add optional domain packs (rule34/e621) only if desired product behavior requires it.

If you want to maximize prompt instruction adherence specifically, add:

- +5% to +15% `ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions`
- reduce `vikhyatoolkit/danbooru2023` by the same amount to keep total at 100%

## Operational note

Before full-scale ingestion, run a schema + sample-quality pass on each selected dataset:

1. confirm image field and caption/tag field names,
2. export 1k-10k samples with `scripts/training/hf_export_to_sdx_manifest.py`,
3. run `scripts/tools/data/data_quality.py`,
4. dedupe by hash/pHash before final mixture.

## Category expansion packs (researched online)

Use these as focused add-ons depending on what behavior you want to strengthen.

### Manga / anime

- [`CaptionEmporium/anime-caption-danbooru-2021-sfw-5m-hq`](https://huggingface.co/datasets/CaptionEmporium/anime-caption-danbooru-2021-sfw-5m-hq)
  - Large SFW anime-caption corpus (1.4M+ images, multi-caption fields).
  - Strong for manga/anime prompt language and safer baseline behavior.
- [`hal-utokyo/Manga109-s`](https://huggingface.co/datasets/hal-utokyo/Manga109-s)
  - Useful for manga panels and dialogue/layout patterns.
  - Gated and license-constrained; treat as specialized, not default bulk source.

Additional 5 mixed-source options:
- [Manga109 official](http://www.manga109.org/en/download_s.html)
- [Manga109 on Kaggle](https://www.kaggle.com/datasets/guansuo/manga109/code)
- [Danbooru2021 (Gwern overview)](https://www.gwern.net/danbooru2021)
- [manga109api (GitHub)](https://github.com/manga109/manga109api)
- [manga109 demos (GitHub)](https://github.com/manga109/manga109-demos)

### Text rendering / OCR grounding

- [`PosterCraft/Text-Render-2M`](https://huggingface.co/datasets/PosterCraft/Text-Render-2M)
  - Purpose-built for text rendering in generated images (`image` + `text`).
  - Gated access; excellent targeted finetune stage for typography quality.
- [`lmms-lab/TextCaps`](https://huggingface.co/datasets/lmms-lab/TextCaps)
  - Better for reading-understanding supervision (text in scenes) than pure generation.
  - Use as auxiliary alignment/eval set for text correctness.

Additional 5 mixed-source options:
- [VGG scene text datasets (Oxford)](https://robots.ox.ac.uk/~vgg/data/text)
- [SynthText generator (GitHub)](https://github.com/ankush-me/SynthText)
- [SynthText dataset readme](https://thor.robots.ox.ac.uk/datasets/SynthText/readme.txt)
- [MMOCR dataset zoo](https://mmocr.readthedocs.io/en/latest/user_guides/data_prepare/datasetzoo.html)
- [MMOCR text recognition datasets](https://mmocr.readthedocs.io/en/v0.6.3/datasets/recog.html)

### SFW/NSFW control

- SFW anchor: `CaptionEmporium/anime-caption-danbooru-2021-sfw-5m-hq`
- NSFW domain add-on: [`zxbsmk/NSFW-T2I`](https://huggingface.co/datasets/zxbsmk/NSFW-T2I), `deepghs/rule34_full`
  - Keep NSFW in an explicit stage with separate filtering and content policy checks.

Additional 5 mixed-source options:
- [Yahoo OpenNSFW (GitHub)](https://github.com/yahoo/open_nsfw)
- [Yahoo OpenNSFW mirror (GitHub)](https://github.com/ModelDepot/Yahoo-Open-NSFW)
- [SIEGuardian dataset repo (includes NPDI/Pornography-2k request info)](https://github.com/fffaded/SIEGuardian-Dataset)
- [NudeNet overview (Medium)](https://praneethbedapudi.medium.com/nudenet-an-ensemble-of-neural-nets-for-nudity-detection-and-censoring-c8fcefa6cc92)
- [NudeNet mirror article](https://office.qz.com/nudenet-an-ensemble-of-neural-nets-for-nudity-detection-and-censoring-d9f3da721e3)

### Icons / symbols / logos

- [`nyuuzyou/svgrepo`](https://huggingface.co/datasets/nyuuzyou/svgrepo)
  - Very large icon corpus (SVG + tags + title metadata).
  - Requires rasterization pipeline before SDX image training.
  - License varies per item (MIT/CC/GPL/logo/etc); enforce per-license allowlist.

Additional 5 mixed-source options:
- [OpenMoji project](https://openmoji.org/)
- [OpenMoji source (GitHub)](https://github.com/hfg-gmuend/openmoji)
- [OpenLogo benchmark](https://arxiv.org/abs/1807.01964)
- [LogoDet-3K benchmark entry](https://paperswithcode.com/dataset/logodet-3k)
- [FlickrLogos-32 dataset overview](https://mldta.com/dataset/flickrlogos-32/)

### Vehicles

- [`DamianBoborzi/car_images`](https://huggingface.co/datasets/DamianBoborzi/car_images)
  - Solid vehicle-specific image-text corpus (63k+ rows, `image` + `text`).
  - Good for improving car model fidelity and descriptive prompt matching.
- [`Multimodal-Fatima/StanfordCars_train`](https://huggingface.co/datasets/Multimodal-Fatima/StanfordCars_train)
  - Classification-oriented; useful for class coverage, weaker for caption depth.

Additional 5 mixed-source options:
- [BDD100K download docs](https://doc.bdd100k.com/download.html)
- [KITTI benchmark suite](https://www.cvlibs.net/datasets/kitti/)
- [nuScenes official](http://www.nuscenes.com/)
- [CompCars official](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)
- [VeRi benchmark project page](https://vehiclereid.github.io/VeRi/)

### Objects / relationships / composition

- [`HuggingFaceM4/COCO`](https://huggingface.co/datasets/HuggingFaceM4/COCO)
  - Strong general object/context baseline with human captions.
- [`332F/visual_genome`](https://huggingface.co/datasets/332F/visual_genome)
  - High-value object attributes + relationships; useful for spatial reasoning.
- [Open Images V7](https://storage.googleapis.com/openimages/web/index.html)
  - Very large object/relationship annotations (boxes, masks, relationships, localized narratives).
  - Best used for auxiliary grounding/object-coverage stages, not as direct caption training.

Additional 5 mixed-source options:
- [Objects365 overview](https://www.objects365.org/overview.html)
- [Objects365 download](https://www.objects365.org/download.html)
- [LVIS dataset](https://www.lvisdataset.org/dataset)
- [ADE20K dataset](https://incidentsdataset.csail.mit.edu/)
- [Open Images V7 portal](https://storage.googleapis.com/openimages/web/index.html)

### Web-scale general corpora

- [`pixparse/cc3m-wds`](https://huggingface.co/datasets/pixparse/cc3m-wds)
  - Strong broad-domain image-text corpus (~2.9M image-caption pairs, WebDataset format).
  - Good for improving general caption-language understanding and concept diversity.
- [`kakaobrain/coyo-labeled-300m`](https://huggingface.co/datasets/kakaobrain/coyo-labeled-300m)
  - Massive weakly supervised image-label corpus (machine labels from ImageNet-21K classes).
  - Better for representation pretraining / classifier heads than direct T2I caption supervision.
  - Contains URLs + labels/probabilities, so ingestion differs from standard image+caption sets.

Additional 5 mixed-source options:
- [LAION-5B paper](http://arxiv.org/abs/2210.08402v1)
- [DataComp benchmark announcement](https://laion.ai/blog/datacomp/)
- [CC12M repo](https://github.com/google-research-datasets/conceptual-12m)
- [RedCaps official download](https://redcaps.xyz/download)
- [YFCC100M paper](https://arxiv.org/abs/1503.01817)

### Styles

- [`showlab/OmniConsistency`](https://huggingface.co/datasets/showlab/OmniConsistency)
  - Paired style transfer data with 22 styles (`src`, `tar`, `prompt`).
  - Best for style-control finetunes and consistency-focused stages.
- [`moonworks/lunara-aesthetic`](https://huggingface.co/datasets/moonworks/lunara-aesthetic)
  - Small but clean; strong for validation and aesthetic tuning, not base pretraining.

Additional 5 mixed-source options:
- [BAM! (Behance Artistic Media) paper](https://arxiv.org/abs/1704.08614)
- [Painter by Numbers (Kaggle competition)](https://www.kaggle.com/competitions/painter-by-numbers)
- [Painter by Numbers data page](https://www.kaggle.com/competitions/painter-by-numbers/data)
- [WikiArt refined dataset (Kaggle)](https://www.kaggle.com/datasets/trungit/wikiart25k)
- [WikiArt API access](https://www.wikiart.org/en/App/GetApi)

## Extra sources requested: booru-like sites + style diversity

### More booru-like image sources (similar to Danbooru/Rule34)

- [`deepghs/gelbooru-webp-4Mpixel`](https://huggingface.co/datasets/deepghs/gelbooru-webp-4Mpixel)
- [`NebulaeWis/gelbooru_images`](https://huggingface.co/datasets/NebulaeWis/gelbooru_images)
- [`deepghs/safebooru_full`](https://huggingface.co/datasets/deepghs/safebooru_full)
- [`deepghs/safebooru-webp-4Mpixel`](https://huggingface.co/datasets/deepghs/safebooru-webp-4Mpixel)
- [`ssonpull519/safebooru-prompts-2023-upscore8`](https://huggingface.co/datasets/ssonpull519/safebooru-prompts-2023-upscore8)
- [`deepghs/site_tags`](https://huggingface.co/datasets/deepghs/site_tags) (multi-booru tag metadata)
- [`nyanko7/danbooru2023`](https://huggingface.co/datasets/nyanko7/danbooru2023)
- [Safebooru metadata (Kaggle)](https://www.kaggle.com/datasets/alamson/safebooru/data)
- [Tagged anime illustrations (Kaggle)](https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations/data)
- [Danbooru2021 data docs](https://www.gwern.net/danbooru2021)

Practical site-level sources to scrape/ingest carefully:
- [Gelbooru](https://gelbooru.com/)
- [Safebooru](https://safebooru.org/)
- [Konachan](https://konachan.com/)
- [yande.re](https://yande.re/)
- [e621](https://e621.net/) / [e926](https://e926.net/)

### More style-diverse datasets/sources

- [`huggan/wikiart`](https://huggingface.co/datasets/huggan/wikiart)
- [`Artificio/WikiArt`](https://huggingface.co/datasets/Artificio/WikiArt)
- [`ninar12/aesthetics-wiki`](https://huggingface.co/datasets/ninar12/aesthetics-wiki)
- [BAM! paper](https://arxiv.org/abs/1704.08614)
- [Painter by Numbers (Kaggle)](https://www.kaggle.com/competitions/painter-by-numbers)
- [WikiArt dataset (Kaggle)](https://www.kaggle.com/datasets/steubk/wikiart/code)
- [WikiArt 25k refined (Kaggle)](https://www.kaggle.com/datasets/trungit/wikiart25k)
- [OpenMoji (style/icon graphics)](https://openmoji.org/)
- [`showlab/OmniConsistency`](https://huggingface.co/datasets/showlab/OmniConsistency) (paired style transfer)
- [`moonworks/lunara-aesthetic`](https://huggingface.co/datasets/moonworks/lunara-aesthetic) (aesthetic alignment)

## Recommended staged training plan

1. Base generalization: Danbooru + `ma-xu/fine-t2i` core mix (+ optional 5-15% `pixparse/cc3m-wds`).
2. Capability boosts: add category packs (vehicles, text rendering, style) at 5-20% each.
3. Grounding stage: use Open Images / Visual Genome style supervision for object-relationship fidelity.
4. Domain stage: optional NSFW or manga-specialized stage.
5. Final polish: small high-quality sets (`moonworks/lunara-aesthetic`, filtered subsets) for alignment.
