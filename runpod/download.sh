#!/usr/bin/env bash
# Download HF models + booru datasets + training prep (enrich, RAG, control maps).
#
#   bash runpod/download.sh                  # models + scrape + preprocess
#   bash runpod/download.sh --models-only    # HF weights only (~100+ GB)
#   bash runpod/download.sh --data-only      # scrape + merge (no models, no preprocess)
#   bash runpod/download.sh --skip-preprocess
#   bash runpod/download.sh --skip-wd-tag     # skip WD tagger pass
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"
cd "$ROOT"

MODELS=1
SCRAPE=1
DATA_PREP=1
PREPROCESS=1
WD_TAG=1
for arg in "$@"; do
  case "$arg" in
    --models-only) SCRAPE=0; DATA_PREP=0; PREPROCESS=0; WD_TAG=0 ;;
    --data-only) MODELS=0 ;;
    --skip-preprocess) PREPROCESS=0 ;;
    --skip-scrape) SCRAPE=0 ;;
    --skip-wd-tag) WD_TAG=0 ;;
    *) echo "Unknown flag: $arg (use --models-only | --data-only | --skip-preprocess | --skip-scrape | --skip-wd-tag)" >&2; exit 2 ;;
  esac
done

if [ "$MODELS" = 1 ]; then
  echo "==> Pretrained models -> $SDX_PRETRAINED"
  python setup/download_pretrained.py --dest "$SDX_PRETRAINED" --workers "${SDX_DL_WORKERS:-16}" --profile "${SDX_MODEL_PROFILE:-full}"
fi

if [ "$SCRAPE" = 1 ] || [ "$DATA_PREP" = 1 ]; then
  SCRAPE_SITES=(danbooru rule34xxx)
  if [ -n "${SDX_SCRAPE_SITES:-}" ]; then
    read -r -a SCRAPE_SITES <<<"${SDX_SCRAPE_SITES//,/ }"
  fi
fi

if [ "$SCRAPE" = 1 ]; then
  SCRAPE_LOCK="${SDX_SCRAPE_LOCK:-$SDX_DATA/.scrape.lock}"
  mkdir -p "$(dirname "$SCRAPE_LOCK")"
  if ! pgrep -af "[d]ownload_datasets" >/dev/null 2>&1; then
    rm -f "$SCRAPE_LOCK"
  fi
  exec 9>"$SCRAPE_LOCK"
  if ! flock -n 9; then
    if pgrep -af "[d]ownload_datasets" >/dev/null 2>&1; then
      echo "ERROR: scrape already running (lock: $SCRAPE_LOCK)" >&2
      echo "  pgrep -af download_datasets" >&2
      exit 1
    fi
    echo "WARN: stale scrape lock removed ($SCRAPE_LOCK)" >&2
    rm -f "$SCRAPE_LOCK"
    exec 9>"$SCRAPE_LOCK"
    flock -n 9 || { echo "ERROR: could not acquire scrape lock" >&2; exit 1; }
  fi

  echo "==> Booru datasets -> $SDX_DATA (${SCRAPE_SITES[*]})"
  SCRAPE_ARGS=(
    --out "$SDX_DATA"
    --sites "${SCRAPE_SITES[@]}"
    --ratings all
    --workers "${SDX_SCRAPE_WORKERS:-20}"
    --max-posts "${SDX_MAX_POSTS:-0}"
    --secrets "$SDX_SECRETS_FILE"
    --frame-fps "${SDX_FRAME_FPS:-2}"
    --max-frames-per-post "${SDX_MAX_FRAMES_PER_POST:-0}"
  )
  if [ "${SDX_SPLIT_FRAMES:-1}" = "1" ]; then
    SCRAPE_ARGS+=(--split-frames)
  else
    SCRAPE_ARGS+=(--no-split-frames)
  fi
  if [ "${SDX_KEEP_RAW_MEDIA:-0}" = "1" ]; then
    SCRAPE_ARGS+=(--keep-raw-media)
  fi
  python setup/download_datasets.py "${SCRAPE_ARGS[@]}"
fi

if [ "$DATA_PREP" = 1 ]; then
  python setup/merge_manifests.py \
    --data-root "$SDX_DATA" \
    --out "$SDX_DATA/combined/manifest.jsonl" \
    --sites "${SCRAPE_SITES[@]}"

  python setup/cleanup_scrape_media.py \
    --data-root "$SDX_DATA" \
    --sites "${SCRAPE_SITES[@]}" \
    --rewrite-manifests \
    --backup \
    --drop-raw-media || true

  python setup/sanitize_manifest.py \
    --manifest "$SDX_DATA/combined/manifest.jsonl" \
    --data-root "$SDX_DATA" \
    --backup \
    --verify-images

  python setup/build_artist_index.py \
    --data-root "$SDX_DATA" \
    --out "$SDX_DATA/artist_index.json"
fi

TAG_MANIFEST="${SDX_MANIFEST:-$SDX_DATA/combined/manifest.jsonl}"
if [ "$WD_TAG" = 1 ] && [ "${SDX_USE_WD_TAGGER:-1}" = "1" ]; then
  if [ -f "$TAG_MANIFEST" ]; then
    echo "==> WD EVA02 tagger enrichment (supplementary tags; identity stays from booru API)"
    TAGGED="${SDX_TAGGED_MANIFEST:-$SDX_DATA/tagged/manifest.jsonl}"
    python setup/tag_manifest_wd.py \
      --manifest "$TAG_MANIFEST" \
      --data-root "$SDX_DATA" \
      --out "$TAGGED" \
      --threshold "${SDX_WD_TAG_THRESHOLD:-0.35}" \
      || echo "WARN: WD tagging failed (install onnxruntime-gpu; non-fatal)."
    if [ -f "$TAGGED" ] && [ -s "$TAGGED" ]; then
      TAG_MANIFEST="$TAGGED"
      export SDX_MANIFEST="$TAGGED"
    fi
  fi
fi

if [ "$PREPROCESS" = 1 ]; then
  MANIFEST="${SDX_MANIFEST:-$TAG_MANIFEST}"
  if [ ! -f "$MANIFEST" ]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
  fi

  echo "==> Seed RAG corpus (booru tags for retrieval)"
  RAG="${SDX_RAG_CORPUS:-$SDX_DATA/rag_corpus.jsonl}"
  python setup/build_rag_corpus.py --manifest "$MANIFEST" --out "$RAG"

  echo "==> Caption enrichment (VLM + RAG + LLM when SDX_PROMPT_RESEARCH=1)"
  ENRICH_OUT="${SDX_ENRICHED_MANIFEST:-$SDX_DATA/enriched/manifest.jsonl}"
  ENRICH_ARGS=(--manifest "$MANIFEST" --data-root "$SDX_DATA" --out "$ENRICH_OUT" --workers "${SDX_ENRICH_WORKERS:-1}")
  if [ "${SDX_PROMPT_RESEARCH:-1}" = "1" ]; then
    ENRICH_ARGS+=(--prompt-research --rag-corpus "$RAG")
  else
    if [ "${SDX_ENRICH_VLM:-0}" != "1" ]; then ENRICH_ARGS+=(--no-vlm); fi
    if [ "${SDX_ENRICH_REVERSE:-0}" != "1" ]; then ENRICH_ARGS+=(--no-reverse-search); fi
    ENRICH_ARGS+=(--booru-only)
  fi
  python setup/enrich_manifest_captions.py "${ENRICH_ARGS[@]}" || echo "WARN: enrichment failed (non-fatal)."

  echo "==> Final RAG corpus (researched captions)"
  RAG_MANIFEST="$MANIFEST"
  if [ -f "$ENRICH_OUT" ] && [ -s "$ENRICH_OUT" ]; then RAG_MANIFEST="$ENRICH_OUT"; fi
  python setup/build_rag_corpus.py --manifest "$RAG_MANIFEST" --out "$RAG"

  echo "==> Control maps"
  CTRL_MANIFEST="$MANIFEST"
  if [ -f "$ENRICH_OUT" ] && [ -s "$ENRICH_OUT" ]; then CTRL_MANIFEST="$ENRICH_OUT"; fi
  python setup/preprocess_control_maps.py \
    --manifest "$CTRL_MANIFEST" \
    --data-root "$SDX_DATA" \
    --out "${SDX_CONTROL_MANIFEST:-$SDX_DATA/control/manifest.jsonl}" \
    --control-type "${SDX_CONTROL_TYPE:-canny}" \
    --workers "${SDX_CONTROL_WORKERS:-12}"
fi

echo
echo "Download done."
echo "  data:     $SDX_DATA"
echo "  manifest: ${SDX_MANIFEST:-$SDX_DATA/combined/manifest.jsonl}"
echo "  train:    bash runpod/train.sh"
