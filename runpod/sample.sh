#!/usr/bin/env bash
# Generate an image from a trained checkpoint (RAG, @artist, box layout).
#
#   bash runpod/sample.sh
#   SDX_PROMPT="@wlop 1girl" bash runpod/sample.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"
cd "$ROOT"

CKPT="${SDX_SAMPLE_CKPT:-$SDX_RESULTS/best.pt}"
PROMPT="${SDX_PROMPT:-@wlop +character: 1girl, silver hair +scene: cherry blossoms}"
RAG="${SDX_RAG_CORPUS:-$SDX_DATA/rag_corpus.jsonl}"
BOX="${SDX_BOX_LAYOUT:-}"
LORA="${SDX_LORA:-}"

ARGS=(--ckpt "$CKPT" --prompt "$PROMPT")

if [ -f "$RAG" ]; then
  ARGS+=(--local-rag-jsonl "$RAG" --local-rag-top-k "${SDX_RAG_TOP_K:-8}")
fi
if [ -n "$BOX" ] && [ -f "$BOX" ]; then
  ARGS+=(--box-layout "$BOX" --box-layout-mode regional_cfg)
fi
if [ -n "$LORA" ] && [ -f "$LORA" ]; then
  ARGS+=(--lora "$LORA:1.0")
fi

python sample.py "${ARGS[@]}" "$@"
