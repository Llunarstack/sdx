#!/usr/bin/env bash
# Full pipeline: setup → download → test → train (see scripts/run_pipeline.py --help).
#
#   bash runpod/run.sh
#   bash runpod/run.sh --skip-setup --skip-train
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"

exec python "$ROOT/scripts/run_pipeline.py" "$@"
