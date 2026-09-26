#!/usr/bin/env bash
# Download pretrained models + Hugging Face training datasets + prep (WD tags, enrich).
#
#   bash runpod/download.sh                  # models + HF datasets + preprocess
#   bash runpod/download.sh --models-only    # HF weights only
#   bash runpod/download.sh --data-only      # HF datasets + merge + tag + enrich
#   bash runpod/download.sh --skip-preprocess
#   bash runpod/download.sh --skip-datasets    # skip HF export (merge/tag only)
#   bash runpod/download.sh --skip-wd-tag
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"
# shellcheck source=/dev/null
source "$ROOT/runpod/lib/hf_sites.sh"
# shellcheck source=/dev/null
source "$ROOT/runpod/lib/turbo_hf.sh"
cd "$ROOT"

MODELS=1
DATASETS=1
DATA_PREP=1
PREPROCESS=1
WD_TAG=1
for arg in "$@"; do
  case "$arg" in
    --models-only) DATASETS=0; DATA_PREP=0; PREPROCESS=0; WD_TAG=0 ;;
    --data-only) MODELS=0 ;;
    --skip-preprocess) PREPROCESS=0 ;;
    --skip-datasets|--skip-scrape) DATASETS=0 ;;
    --skip-wd-tag) WD_TAG=0 ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

sdx_export_hf_sites
sdx_apply_turbo_hf
read -r -a HF_SITES <<<"$SDX_HF_SITES"

if [ "$MODELS" = 1 ]; then
  echo "==> Pretrained models -> $SDX_PRETRAINED"
  python setup/download_pretrained.py --dest "$SDX_PRETRAINED" --workers "${SDX_DL_WORKERS:-16}" --profile "${SDX_MODEL_PROFILE:-full}"
fi

if [ "$DATASETS" = 1 ]; then
  DATA_LOCK="${SDX_DATA_LOCK:-$SDX_DATA/.data_download.lock}"
  mkdir -p "$(dirname "$DATA_LOCK")"
  if ! pgrep -af "[d]ownload_hf_datasets" >/dev/null 2>&1; then
    rm -f "$DATA_LOCK" "$SDX_DATA/.scrape.lock"
  fi
  exec 9>"$DATA_LOCK"
  if ! flock -n 9; then
    if pgrep -af "[d]ownload_hf_datasets" >/dev/null 2>&1; then
      echo "ERROR: dataset download already running (lock: $DATA_LOCK)" >&2
      exit 1
    fi
    rm -f "$DATA_LOCK"
    exec 9>"$DATA_LOCK"
    flock -n 9 || { echo "ERROR: could not acquire data lock" >&2; exit 1; }
  fi

  echo "==> Hugging Face datasets -> $SDX_DATA"
  echo "    packs (${#HF_SITES[@]}): ${HF_SITES[*]}"
    HF_ARGS=(
      --dest "$SDX_DATA"
      --max-samples "${SDX_HF_MAX_SAMPLES:-0}"
      --image-format "${SDX_HF_IMAGE_FORMAT:-jpg}"
    )
    [ -n "${SDX_HF_FORCE:-}" ] && HF_ARGS+=(--force)
    python setup/download_hf_datasets.py "${HF_ARGS[@]}"
fi

if [ "$DATA_PREP" = 1 ]; then
  python setup/merge_manifests.py \
    --data-root "$SDX_DATA" \
    --out "$SDX_DATA/combined/manifest.jsonl" \
    --sites "${HF_SITES[@]}"

  python setup/cleanup_scrape_media.py \
    --data-root "$SDX_DATA" \
    --sites "${HF_SITES[@]}" \
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
    echo "==> WD EVA02 tagger enrichment"
    TAGGED="${SDX_TAGGED_MANIFEST:-$SDX_DATA/tagged/manifest.jsonl}"
    python setup/tag_manifest_wd.py \
      --manifest "$TAG_MANIFEST" \
      --data-root "$SDX_DATA" \
      --out "$TAGGED" \
      --threshold "${SDX_WD_TAG_THRESHOLD:-0.35}" \
      || echo "WARN: WD tagging failed (non-fatal)."
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

  echo "==> Seed RAG corpus"
  RAG="${SDX_RAG_CORPUS:-$SDX_DATA/rag_corpus.jsonl}"
  python setup/build_rag_corpus.py --manifest "$MANIFEST" --out "$RAG"

  echo "==> Caption enrichment (VLM + RAG)"
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

  echo "==> Final RAG corpus"
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
