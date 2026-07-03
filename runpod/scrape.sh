#!/usr/bin/env bash
# Turbo HF dataset export (default). Use --slow for live booru API scrape.
#
#   bash runpod/scrape.sh          # HF bulk datasets (fast)
#   bash runpod/scrape.sh --slow   # live booru API (needs secret.txt)
#   bash runpod/scrape.sh --fg     # foreground
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"

SCRAPE_LOCK="${SDX_DATA_LOCK:-$SDX_DATA/.data_download.lock}"
LOG="${SDX_SCRAPE_LOG:-/workspace/scrape.log}"
FG=0
for arg in "$@"; do
  case "$arg" in
    --fg) FG=1 ;;
    --slow) export SDX_DATA_SOURCE=booru ;;
    --turbo) export SDX_DATA_SOURCE=hf ;;
    *) echo "Unknown flag: $arg (use --fg | --slow | --turbo)" >&2; exit 2 ;;
  esac
done

export SDX_DATA_SOURCE="${SDX_DATA_SOURCE:-hf}"

mkdir -p "$(dirname "$SCRAPE_LOCK")" "$(dirname "$LOG")"
exec 9>"$SCRAPE_LOCK"
if ! flock -n 9; then
  echo "Data download already running. Lock: $SCRAPE_LOCK" >&2
  exit 1
fi

echo "Data source: $SDX_DATA_SOURCE (log: $LOG)"

if [ "$FG" = 1 ]; then
  bash "$ROOT/runpod/download.sh" --data-only --skip-preprocess 2>&1 | tee -a "$LOG"
else
  nohup bash "$ROOT/runpod/download.sh" --data-only --skip-preprocess >>"$LOG" 2>&1 &
  echo "PID: $!"
  echo "  tail -f $LOG"
fi
