#!/usr/bin/env bash
# Ultimate SDX pipeline: pretrained → scrape → WD tags → enrich → train base → LoRA bank.
#
#   bash runpod/ultimate.sh                    # full pipeline (days on H100)
#   bash runpod/ultimate.sh --data-only        # models + scrape + tag + enrich
#   bash runpod/ultimate.sh --train-only       # base + LoRA bank (data must exist)
#   bash runpod/ultimate.sh --skip-train       # stop after enrich
#
# Modular inference after training:
#   SDX_PROMPT="@wlop @style:anime 1girl, cherry blossoms" bash runpod/sample.sh
#   --artist-strength 1.2 scales the wlop LoRA; @style:anime loads anime adapter.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"
cd "$ROOT"

PHASE_SETUP=1
PHASE_MODELS=1
PHASE_SCRAPE=1
PHASE_TAG=1
PHASE_ENRICH=1
PHASE_TRAIN_BASE=1
PHASE_TRAIN_LORAS=1
PHASE_SAMPLE=0
EXTRA=()

for arg in "$@"; do
  case "$arg" in
    --data-only) PHASE_TRAIN_BASE=0; PHASE_TRAIN_LORAS=0; PHASE_SAMPLE=0 ;;
    --train-only) PHASE_SETUP=0; PHASE_MODELS=0; PHASE_SCRAPE=0; PHASE_TAG=0; PHASE_ENRICH=0 ;;
    --skip-setup) PHASE_SETUP=0 ;;
    --skip-models) PHASE_MODELS=0 ;;
    --skip-scrape) PHASE_SCRAPE=0 ;;
    --skip-tag) PHASE_TAG=0 ;;
    --skip-enrich) PHASE_ENRICH=0 ;;
    --skip-train) PHASE_TRAIN_BASE=0; PHASE_TRAIN_LORAS=0 ;;
    --skip-lora-bank) PHASE_TRAIN_LORAS=0 ;;
    --with-sample) PHASE_SAMPLE=1 ;;
    --help|-h)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    --*) echo "Unknown flag: $arg" >&2; exit 2 ;;
    *) EXTRA+=("$arg") ;;
  esac
done

export SDX_MODEL_PROFILE="${SDX_MODEL_PROFILE:-ultimate}"
export SDX_SCRAPE_SITES="${SDX_SCRAPE_SITES:-danbooru rule34xxx}"
export SDX_USE_WD_TAGGER="${SDX_USE_WD_TAGGER:-1}"
export SDX_PROMPT_RESEARCH="${SDX_PROMPT_RESEARCH:-1}"
export SDX_TRAIN_LORA_BANK="${SDX_TRAIN_LORA_BANK:-1}"

echo "=============================================="
echo " SDX Ultimate Pipeline"
echo " sites=${SDX_SCRAPE_SITES}"
echo " wd_tagger=${SDX_USE_WD_TAGGER} enrich=${SDX_PROMPT_RESEARCH}"
echo " train_base=${PHASE_TRAIN_BASE} lora_bank=${PHASE_TRAIN_LORAS}"
echo "=============================================="

if [ "$PHASE_SETUP" = 1 ]; then
  echo "==> [1/8] Setup"
  bash "$ROOT/runpod/setup.sh"
fi

if [ "$PHASE_MODELS" = 1 ]; then
  echo "==> [2/8] Pretrained models (profile=$SDX_MODEL_PROFILE)"
  python setup/download_pretrained.py \
    --dest "$SDX_PRETRAINED" \
    --profile "$SDX_MODEL_PROFILE" \
    --workers "${SDX_DL_WORKERS:-16}"
fi

if [ "$PHASE_SCRAPE" = 1 ] || [ "$PHASE_TAG" = 1 ] || [ "$PHASE_ENRICH" = 1 ]; then
  DL_ARGS=()
  [ "$PHASE_MODELS" = 0 ] && DL_ARGS+=(--data-only)
  [ "$PHASE_SCRAPE" = 0 ] && DL_ARGS+=(--skip-scrape)
  [ "$PHASE_TAG" = 0 ] && DL_ARGS+=(--skip-wd-tag)
  [ "$PHASE_ENRICH" = 0 ] && DL_ARGS+=(--skip-preprocess)
  echo "==> [3-5/8] Scrape + WD tag + enrich"
  bash "$ROOT/runpod/download.sh" "${DL_ARGS[@]}"
fi

if [ "$PHASE_TRAIN_BASE" = 1 ]; then
  echo "==> [6/8] Train base DiT checkpoint"
  export SDX_TRAIN_MODE="${SDX_TRAIN_MODE:-full}"
  export SDX_MANIFEST="${SDX_TAGGED_MANIFEST:-$SDX_DATA/tagged/manifest.jsonl}"
  if [ ! -s "$SDX_MANIFEST" ]; then
    SDX_MANIFEST="${SDX_ENRICHED_MANIFEST:-$SDX_DATA/enriched/manifest.jsonl}"
  fi
  if [ ! -s "$SDX_MANIFEST" ]; then
    SDX_MANIFEST="${SDX_DATA}/combined/manifest.jsonl"
  fi
  export SDX_MANIFEST
  bash "$ROOT/runpod/train_h100.sh" "${EXTRA[@]}"
  # Auto-pick best base for LoRA bank unless user set SDX_INIT_CKPT
  if [ -z "${SDX_INIT_CKPT:-}" ]; then
    BEST=$(find "$SDX_RESULTS" -name 'best.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$BEST" ]; then
      export SDX_INIT_CKPT="$BEST"
      echo "Base checkpoint for LoRA bank: $SDX_INIT_CKPT"
    fi
  fi
fi

if [ "$PHASE_TRAIN_LORAS" = 1 ] && [ "${SDX_TRAIN_LORA_BANK:-1}" = "1" ]; then
  echo "==> [7/8] Train modular LoRA bank (artist + style adapters)"
  bash "$ROOT/runpod/train_lora_bank.sh" "${EXTRA[@]}"
fi

if [ "$PHASE_SAMPLE" = 1 ]; then
  echo "==> [8/8] Sample smoke"
  bash "$ROOT/runpod/sample.sh"
fi

echo
echo "Ultimate pipeline done."
echo "  data:      $SDX_DATA"
echo "  enriched:  ${SDX_ENRICHED_MANIFEST:-$SDX_DATA/enriched/manifest.jsonl}"
echo "  lora bank: ${SDX_LORA_BANK:-$SDX_DATA/lora_bank}"
echo "  sample:    SDX_PROMPT='@wlop @style:anime 1girl' bash runpod/sample.sh"
