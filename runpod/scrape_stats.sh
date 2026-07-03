#!/usr/bin/env bash
# Quick scrape progress snapshot.
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "$0")/.." && pwd)/runpod/env.defaults"

echo "=== scrape process ==="
if pgrep -af download_datasets >/dev/null 2>&1; then
  pgrep -af download_datasets | grep -v grep || true
else
  echo "  (not running)"
fi

echo
echo "=== manifest rows ==="
total=0
for f in "$SDX_DATA"/*/manifest.jsonl; do
  [ -f "$f" ] || continue
  site=$(basename "$(dirname "$f")")
  n=$(wc -l <"$f")
  total=$((total + n))
  printf "  %-12s %d\n" "$site:" "$n"
done
printf "  %-12s %d\n" "TOTAL:" "$total"

echo
echo "=== disk ==="
du -sh "$SDX_DATA" 2>/dev/null || echo "  (no data dir)"

echo
LOG="${SDX_SCRAPE_LOG:-/workspace/scrape.log}"
if [ -f "$LOG" ]; then
  echo "=== log tail ($LOG) ==="
  tail -10 "$LOG"
fi
