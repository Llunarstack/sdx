#!/usr/bin/env bash
# Install /workspace/secret.txt from runpod/secret.txt (upload via RunPod file UI).
set -euo pipefail

sdx_install_scrape_secrets() {
  export SDX_ROOT="${SDX_ROOT:-/workspace/sdx}"
  local src="$SDX_ROOT/runpod/secret.txt"
  local dest="${SDX_SECRETS_FILE:-/workspace/secret.txt}"

  if [ ! -f "$src" ]; then
    return 1
  fi

  mkdir -p "$(dirname "$dest")"
  cp "$src" "$dest"
  chmod 600 "$dest" 2>/dev/null || true
  echo "Installed scrape secrets: $src -> $dest"
  return 0
}

sdx_scrape_secrets_ok() {
  export SDX_ROOT="${SDX_ROOT:-/workspace/sdx}"
  python3 - <<'PY'
import os, sys
sys.path.insert(0, os.environ.get("SDX_ROOT", "/workspace/sdx"))
from scripts.scrape.secrets_config import get_secrets_path, parse_secrets_file
path = get_secrets_path(os.environ.get("SDX_SECRETS_FILE"))
need = {"danbooru", "rule34xxx"}
if not path.is_file():
    sys.exit(1)
have = set(parse_secrets_file(path).keys())
sys.exit(0 if need.issubset(have) else 1)
PY
}

sdx_ensure_scrape_secrets() {
  if sdx_scrape_secrets_ok 2>/dev/null; then
    return 0
  fi
  sdx_install_scrape_secrets || true
  if sdx_scrape_secrets_ok 2>/dev/null; then
    return 0
  fi
  return 1
}
