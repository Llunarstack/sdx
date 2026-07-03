#!/usr/bin/env bash
# Load HF_TOKEN from SDX_SECRETS_FILE into the environment (if not already set).
sdx_load_hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then
    return 0
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
    from utils.hf_secrets import get_hf_token
    t = get_hf_token()
    if t:
        print(t)
except Exception:
    pass
PY
)
  if [ -n "$tok" ]; then
    case "$tok" in
      *YOUR_TOKEN*|*your_token*|*_HERE) tok="" ;;
    esac
  fi
  if [ -n "$tok" ]; then
    export HF_TOKEN="$tok"
    export HUGGING_FACE_HUB_TOKEN="$tok"
    echo "HF token: loaded from $f"
    return 0
  fi
  echo "WARN: no valid HF token in $f — add: huggingface / token: hf_..." >&2
  return 1
}
