#!/usr/bin/env bash
# Download HF models + booru datasets + training prep (enrich, RAG, control maps).
#
#   bash runpod/download.sh                  # models + scrape + preprocess
#   bash runpod/download.sh --models-only    # HF weights only (~100+ GB)
#   bash runpod/download.sh --data-only      # scrape + merge (no models, no preprocess)
#   bash runpod/download.sh --skip-preprocess
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"
cd "$ROOT"

MODELS=1
DATA=1
PREPROCESS=1
for arg in "$@"; do
  case "$arg" in
    --models-only) DATA=0; PREPROCESS=0 ;;
    --data-only) MODELS=0 ;;
    --skip-preprocess) PREPROCESS=0 ;;
    *) echo "Unknown flag: $arg (use --models-only | --data-only | --skip-preprocess)" >&2; exit 2 ;;
  esac
done

if [ "$MODELS" = 1 ]; then
  echo "==> Pretrained models -> $SDX_PRETRAINED"
  python setup/download_pretrained.py --dest "$SDX_PRETRAINED" --workers "${SDX_DL_WORKERS:-16}"
fi

if [ "$DATA" = 1 ]; then
  echo "==> Booru datasets -> $SDX_DATA"
  python setup/download_datasets.py \
    --out "$SDX_DATA" \
    --ratings all \
    --workers "${SDX_SCRAPE_WORKERS:-20}" \
    --max-posts "${SDX_MAX_POSTS:-10000000}" \
    --secrets "$SDX_SECRETS_FILE"

  python setup/merge_manifests.py \
    --data-root "$SDX_DATA" \
    --out "$SDX_DATA/combined/manifest.jsonl"

  python setup/build_artist_index.py \
    --data-root "$SDX_DATA" \
    --out "$SDX_DATA/artist_index.json"
fi

if [ "$PREPROCESS" = 1 ] && [ "$DATA" = 1 ]; then
  MANIFEST="${SDX_MANIFEST:-$SDX_DATA/combined/manifest.jsonl}"
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
