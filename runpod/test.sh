#!/usr/bin/env bash
# Integration smoke test — scrape all sites + pipeline checks before spending on GPU time.
#
#   bash runpod/test.sh
#   bash runpod/test.sh --skip-train
#   bash runpod/test.sh --skip-scrape
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"

exec python3 "$ROOT/scripts/integration_smoke.py" \
  --data-root "${SDX_DATA:-/workspace/data}/integration_smoke" \
  --secrets "$SDX_SECRETS_FILE" \
  "$@"
