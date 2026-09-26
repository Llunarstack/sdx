#!/usr/bin/env bash
# Legacy entry point — delegates to runpod/setup.sh (full install).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "$ROOT/runpod/setup.sh" "$@"
