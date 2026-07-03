#!/usr/bin/env bash
# Train modular LoRA bank: per-artist + per-style adapters on a frozen base checkpoint.
#
# Requires SDX_INIT_CKPT (base DiT). Builds subsets from enriched manifest, trains each adapter.
#
#   SDX_INIT_CKPT=/workspace/results/.../best.pt bash runpod/train_lora_bank.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"
cd "$ROOT"

BASE="${SDX_INIT_CKPT:-}"
BANK="${SDX_LORA_BANK:-$SDX_DATA/lora_bank}"
SUBSETS="${SDX_LORA_SUBSETS:-$SDX_DATA/lora_subsets}"
MANIFEST="${SDX_ENRICHED_MANIFEST:-$SDX_DATA/enriched/manifest.jsonl}"
if [ ! -s "$MANIFEST" ]; then
  MANIFEST="${SDX_TAGGED_MANIFEST:-$SDX_DATA/tagged/manifest.jsonl}"
fi
if [ ! -s "$MANIFEST" ]; then
  MANIFEST="${SDX_DATA}/combined/manifest.jsonl"
fi

MIN_ARTIST="${SDX_LORA_MIN_SAMPLES:-150}"
MAX_ARTISTS="${SDX_LORA_BANK_MAX_ARTISTS:-48}"
MIN_STYLE="${SDX_LORA_MIN_STYLE_SAMPLES:-200}"
LORA_EPOCHS="${SDX_LORA_EPOCHS:-4}"
LORA_STEPS="${SDX_LORA_MAX_STEPS:-800}"
LORA_RANK="${SDX_LORA_RANK:-32}"
LORA_ALPHA="${SDX_LORA_ALPHA:-32}"

if [ -z "$BASE" ] || [ ! -f "$BASE" ]; then
  echo "ERROR: set SDX_INIT_CKPT to a trained base DiT checkpoint (best.pt)." >&2
  echo "  Example: SDX_INIT_CKPT=\$(find $SDX_RESULTS -name best.pt | head -1) bash runpod/train_lora_bank.sh" >&2
  exit 1
fi

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: manifest not found. Run bash runpod/download.sh first." >&2
  exit 1
fi

echo "==> Build LoRA subset manifests from $MANIFEST"
python setup/build_lora_subsets.py \
  --manifest "$MANIFEST" \
  --data-root "$SDX_DATA" \
  --out "$SUBSETS" \
  --min-artist-samples "$MIN_ARTIST" \
  --max-artists "$MAX_ARTISTS" \
  --min-style-samples "$MIN_STYLE"

train_one() {
  local kind="$1"
  local slug="$2"
  local subset_manifest="$3"
  local out_dir="$BANK/$kind/$slug"
  local count
  count=$(wc -l <"$subset_manifest")
  echo "==> LoRA [$kind/$slug] rows=$count -> $out_dir"
  mkdir -p "$out_dir"
  SDX_TRAIN_MODE=lora \
  SDX_INIT_CKPT="$BASE" \
  SDX_MANIFEST="$subset_manifest" \
  SDX_RESULTS="$out_dir" \
  SDX_EPOCHS="$LORA_EPOCHS" \
  SDX_MAX_STEPS="$LORA_STEPS" \
  SDX_LORA_RANK="$LORA_RANK" \
  SDX_LORA_ALPHA="$LORA_ALPHA" \
  SDX_GLOBAL_BATCH_SIZE="${SDX_LORA_BATCH:-8}" \
  SDX_FULL_TRAIN_FEATURES="${SDX_LORA_FULL_FEATURES:-0}" \
  bash "$ROOT/runpod/train.sh"
}

for mpath in "$SUBSETS"/artist/*/manifest.jsonl; do
  [ -f "$mpath" ] || continue
  slug=$(basename "$(dirname "$mpath")")
  train_one artist "$slug" "$mpath"
done

for mpath in "$SUBSETS"/style/*/manifest.jsonl; do
  [ -f "$mpath" ] || continue
  slug=$(basename "$(dirname "$mpath")")
  train_one style "$slug" "$mpath"
done

echo "==> Build LoRA bank index"
python setup/build_lora_bank_index.py --bank-root "$BANK" --out "$BANK/index.json"

echo "LoRA bank ready: $BANK/index.json"
echo "Sample: SDX_PROMPT='@wlop @style:anime 1girl' SDX_SAMPLE_CKPT=$BASE bash runpod/sample.sh"
