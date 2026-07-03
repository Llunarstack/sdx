#!/usr/bin/env bash
# Start booru scrape in background (one instance only).
#
#   bash runpod/scrape.sh          # turbo scrape (default)
#   bash runpod/scrape.sh --slow   # frame-split on, fewer threads
#   bash runpod/scrape.sh --fg     # foreground (for debugging)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"
# shellcheck source=/dev/null
source "$ROOT/runpod/lib/turbo_scrape.sh"

SCRAPE_LOCK="${SDX_SCRAPE_LOCK:-$SDX_DATA/.scrape.lock}"
LOG="${SDX_SCRAPE_LOG:-/workspace/scrape.log}"
FG=0
for arg in "$@"; do
  case "$arg" in
    --fg) FG=1 ;;
    --slow) export SDX_SCRAPE_TURBO=0; export SDX_SPLIT_FRAMES=1; export SDX_SCRAPE_WORKERS=64 ;;
    --turbo) export SDX_SCRAPE_TURBO=1 ;;
    *) echo "Unknown flag: $arg (use --fg | --slow | --turbo)" >&2; exit 2 ;;
  esac
done

sdx_apply_turbo_scrape

mkdir -p "$(dirname "$SCRAPE_LOCK")" "$(dirname "$LOG")"
exec 9>"$SCRAPE_LOCK"
if ! flock -n 9; then
  echo "Scrape already running. Lock: $SCRAPE_LOCK" >&2
  echo "  pgrep -af download_datasets" >&2
  exit 1
fi

echo "Scrape turbo: workers=$SDX_SCRAPE_WORKERS split_frames=$SDX_SPLIT_FRAMES api=${SDX_API_RATE_DANBOORU:-?}/s"

if [ "$FG" = 1 ]; then
  echo "Scrape (foreground) -> log also at $LOG"
  bash "$ROOT/runpod/download.sh" --data-only --skip-preprocess 2>&1 | tee -a "$LOG"
else
  echo "Starting scrape in background -> $LOG"
  nohup bash "$ROOT/runpod/download.sh" --data-only --skip-preprocess >>"$LOG" 2>&1 &
  echo "PID: $!"
  echo "  tail -f $LOG"
  echo "  bash runpod/scrape_stats.sh"
fi
