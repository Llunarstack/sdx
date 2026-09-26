#!/usr/bin/env bash
# Load HF_TOKEN from SDX_SECRETS_FILE into the environment (if not already set).
sdx_load_hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then
    case "$HF_TOKEN" in
      *YOUR_TOKEN*|*your_token*|*_HERE) unset HF_TOKEN HUGGING_FACE_HUB_TOKEN ;;
      *) return 0 ;;
    esac
  fi
  export SDX_ROOT="${SDX_ROOT:-/workspace/sdx}"
  local f="${SDX_SECRETS_FILE:-/workspace/secret.txt}"
  if [ ! -f "$f" ]; then
    return 1
  fi
  local tok
  tok=$(python3 - <<'PY' 2>/dev/null || true
import os, sys
sys.path.insert(0, os.environ.get("SDX_ROOT", "/workspace/sdx"))
try:
    from utils.hf_secrets import get_hf_token, hf_auth_source
    t = get_hf_token()
    if t:
        print(f"{hf_auth_source()}\t{t}")
except Exception:
    pass
PY
)
  if [ -n "$tok" ]; then
    local src="${tok%%$'\t'*}"
    tok="${tok#*$'\t'}"
    export HF_TOKEN="$tok"
    export HUGGING_FACE_HUB_TOKEN="$tok"
    case "$src" in
      env) echo "HF auth: token from environment" ;;
      secret) echo "HF auth: token from $f" ;;
      cli) echo "HF auth: huggingface-cli login (cached)" ;;
      *) echo "HF auth: OK" ;;
    esac
    return 0
  fi
  echo "WARN: no HF auth — run: huggingface-cli login  (or add token to $f)" >&2
  return 1
}
