#!/usr/bin/env bash
# **Run this from /workspace** — clones SDX if needed, then runs ultimate.sh.
#
#   bash /workspace/sdx/runpod/start.sh              # after clone
#   curl -fsSL ... | bash                            # (future)
#
# Fresh pod (repo not cloned yet):
#   export SDX_REPO_URL=https://github.com/Llunarstack/sdx.git
#   bash -c 'git clone --depth 1 -b feat/runpod-readiness-scraper-lora $SDX_REPO_URL /workspace/sdx && bash /workspace/sdx/runpod/start.sh'
#
# Budget first run:
#   SDX_PROMPT_RESEARCH=0 SDX_EPOCHS=8 SDX_TRAIN_LORA_BANK=0 bash runpod/start.sh --train-only
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/lib/ensure_repo.sh"
sdx_ensure_repo || {
  echo >&2
  echo "Fix: clone manually then re-run:" >&2
  echo "  git clone --depth 1 -b feat/runpod-readiness-scraper-lora https://github.com/Llunarstack/sdx.git /workspace/sdx" >&2
  echo "  bash /workspace/sdx/runpod/start.sh" >&2
  exit 1
}

# shellcheck source=/dev/null
source "$SDX_ROOT/runpod/env.defaults"
exec bash "$SDX_ROOT/runpod/ultimate.sh" "$@"
