#!/usr/bin/env bash
# Shared RunPod helpers. Source from runpod/*.sh — defines functions only, runs nothing.
#
#   source "$(dirname "$0")/lib.sh"
#   sdx_ensure_repo || exit 1

# --- CRLF fix -----------------------------------------------------------------------------------------
# Strip Windows CRLF from shell scripts (RunPod file browser / git autocrlf).
sdx_fix_shell_scripts() {
  local root="${1:-${SDX_ROOT:-/workspace/sdx}}"
  local f
  [ -d "$root/runpod" ] || return 0
  while IFS= read -r -d '' f; do
    sed -i 's/\r$//' "$f" 2>/dev/null || true
  done < <(find "$root/runpod" -name '*.sh' -print0 2>/dev/null)
}

# --- Repo checkout --------------------------------------------------------------------------------------
# Ensure SDX repo exists at SDX_ROOT (clone or fast-forward).
sdx_ensure_repo() {
  export SDX_ROOT="${SDX_ROOT:-/workspace/sdx}"
  export SDX_REPO_URL="${SDX_REPO_URL:-https://github.com/Llunarstack/sdx.git}"
  export SDX_REPO_REF="${SDX_REPO_REF:-feat/runpod-readiness-scraper-lora}"

  if [ -f "$SDX_ROOT/runpod/sdx.sh" ]; then
    cd "$SDX_ROOT" || return 1
    git fetch origin 2>/dev/null || true
    git checkout "$SDX_REPO_REF" 2>/dev/null || true
    if ! git pull --ff-only 2>/dev/null; then
      echo "WARN: git pull failed — resetting to origin/$SDX_REPO_REF" >&2
      git reset --hard "origin/$SDX_REPO_REF" 2>/dev/null || true
    fi
    sdx_fix_shell_scripts "$SDX_ROOT"
    return 0
  fi

  # Legacy checkout without sdx.sh — force refresh
  if [ -d "$SDX_ROOT/.git" ]; then
    echo "WARN: old SDX checkout (no sdx.sh) — run: bash $SDX_ROOT/runpod/bootstrap.sh" >&2
    cd "$SDX_ROOT" || return 1
    git pull --ff-only 2>/dev/null || true
    [ -f "$SDX_ROOT/runpod/sdx.sh" ] && return 0
  fi

  echo "SDX not found at $SDX_ROOT" >&2
  if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: install git first: apt-get update && apt-get install -y git" >&2
    return 1
  fi

  if [ -e "$SDX_ROOT" ] && [ ! -d "$SDX_ROOT/.git" ]; then
    echo "ERROR: $SDX_ROOT exists but is not an SDX git checkout." >&2
    echo "  mv $SDX_ROOT ${SDX_ROOT}.bak   # or set SDX_ROOT to another path" >&2
    return 1
  fi

  echo "==> Cloning $SDX_REPO_URL (branch $SDX_REPO_REF) -> $SDX_ROOT"
  mkdir -p "$(dirname "$SDX_ROOT")"
  if ! git clone --depth 1 --branch "$SDX_REPO_REF" "$SDX_REPO_URL" "$SDX_ROOT" 2>/dev/null; then
    echo "WARN: branch $SDX_REPO_REF unavailable — cloning default branch" >&2
    git clone --depth 1 "$SDX_REPO_URL" "$SDX_ROOT"
    cd "$SDX_ROOT" || return 1
    git checkout "$SDX_REPO_REF" 2>/dev/null || true
  fi
  cd "$SDX_ROOT" || return 1
  sdx_fix_shell_scripts "$SDX_ROOT"
  echo "==> SDX ready at $SDX_ROOT"
  return 0
}

# --- Hugging Face datasets ------------------------------------------------------------------------------
# Canonical Hugging Face booru-style training packs (no live API scraping).
# Folder names under $SDX_DATA match pack "name" in setup/hf_dataset_packs.json.
sdx_hf_sites() {
  echo "danbooru rule34xxx"
}

sdx_export_hf_sites() {
  local sites
  sites="$(sdx_hf_sites)"
  export SDX_HF_SITES="$sites"
  export SDX_DATA_SITES="$sites"
}

# Max-speed Hugging Face dataset export (parallel packs + threaded image writes).
sdx_apply_turbo_hf() {
  [ "${SDX_HF_TURBO:-1}" = "0" ] && return 0
  export SDX_HF_TURBO=1
  export SDX_HF_PARALLEL_PACKS="${SDX_HF_PARALLEL_PACKS:-2}"
  export SDX_HF_EXPORT_WORKERS="${SDX_HF_EXPORT_WORKERS:-48}"
  export SDX_HF_JPEG_QUALITY="${SDX_HF_JPEG_QUALITY:-78}"
  export SDX_HF_SKIP_REENCODE="${SDX_HF_SKIP_REENCODE:-1}"
  export SDX_HF_PREFETCH="${SDX_HF_PREFETCH:-512}"
  export SDX_HF_MANIFEST_FLUSH_EVERY="${SDX_HF_MANIFEST_FLUSH_EVERY:-512}"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export HF_XET_HIGH_PERFORMANCE=1
  export HF_HUB_DOWNLOAD_NUM_THREADS="${HF_HUB_DOWNLOAD_NUM_THREADS:-64}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
  export HF_HUB_ENABLE_PROGRESS_BARS="${HF_HUB_ENABLE_PROGRESS_BARS:-0}"
}

# --- Secrets --------------------------------------------------------------------------------------------
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
    from utils.modeling.model_paths import get_hf_token, hf_auth_source
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

# Install /workspace/secret.txt from runpod/secret.txt (upload via RunPod file UI).
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
from scripts.scrape import get_secrets_path, parse_secrets_file

sites = os.environ.get("SDX_SCRAPE_SITES", "danbooru rule34xxx e621 rule34xyz").replace(",", " ").split()
sites = [s.strip() for s in sites if s.strip()]
path = get_secrets_path(os.environ.get("SDX_SECRETS_FILE"))
if not path.is_file():
    sys.exit(1)
have = set(parse_secrets_file(path).keys())
missing = [s for s in sites if s not in have]
sys.exit(0 if not missing else 1)
PY
}

# --- Training feature flags -----------------------------------------------------------------------------
# Shared DiT image-training feature flags (everything except video/frontier), used by runpod/train.sh.
# Set SDX_FULL_TRAIN_FEATURES=0 to disable the advanced stack (baseline DiT only).
#
# WARNING: architecture flags (REPA, SSM, RoPE, register tokens, creativity, qk-norm)
# change checkpoint shape — do not --resume a checkpoint trained without them.
sdx_build_train_feature_args() {
  local -n _out=$1
  if [ "${SDX_FULL_TRAIN_FEATURES:-1}" = "0" ]; then
    return 0
  fi

  # --- Text conditioning (triple = T5 + CLIP-L + bigG fusion trained with DiT) ---
  local enc_mode="${SDX_TEXT_ENCODER_MODE:-triple}"
  _out+=(--text-encoder-mode "$enc_mode")
  if [ -n "${SDX_TEXT_ENCODER:-}" ]; then
    _out+=(--text-encoder "$SDX_TEXT_ENCODER")
  fi
  if [ -n "${SDX_CLIP_TEXT_ENCODER_L:-}" ]; then
    _out+=(--clip-text-encoder-l "$SDX_CLIP_TEXT_ENCODER_L")
  fi
  if [ -n "${SDX_CLIP_TEXT_ENCODER_BIGG:-}" ]; then
    _out+=(--clip-text-encoder-bigg "$SDX_CLIP_TEXT_ENCODER_BIGG")
  fi
  if [ "$enc_mode" = "penta" ]; then
    [ -n "${SDX_CLIP_TEXT_ENCODER_H:-}" ] && _out+=(--clip-text-encoder-h "$SDX_CLIP_TEXT_ENCODER_H")
    [ -n "${SDX_CLIP_TEXT_ENCODER_LONG:-}" ] && _out+=(--clip-text-encoder-long "$SDX_CLIP_TEXT_ENCODER_LONG")
  fi

  # --- DiT architecture / loss auxiliaries ---
  _out+=(
    --repa-weight "${SDX_REPA_WEIGHT:-0.5}"
    --repa-encoder-model "${SDX_REPA_ENCODER:-facebook/dinov2-base}"
    --repa-out-dim "${SDX_REPA_OUT_DIM:-768}"
    --ssm-every-n "${SDX_SSM_EVERY_N:-4}"
    --ssm-kernel-size "${SDX_SSM_KERNEL_SIZE:-7}"
    --num-register-tokens "${SDX_NUM_REGISTER_TOKENS:-4}"
    --use-rope
    --rope-base "${SDX_ROPE_BASE:-10000}"
    --layerscale-init "${SDX_LAYERSCALE_INIT:-1e-5}"
    --qk-norm
    --drop-path-rate "${SDX_DROP_PATH_RATE:-0.05}"
    --patch-se
    --prompt-reinject-every-n "${SDX_PROMPT_REINJECT_EVERY_N:-4}"
    --prompt-reinject-alpha "${SDX_PROMPT_REINJECT_ALPHA:-0.1}"
    --prompt-timestep-schedule-enabled
    --creativity-embed-dim "${SDX_CREATIVITY_EMBED_DIM:-64}"
    --creativity-jitter-std "${SDX_CREATIVITY_JITTER_STD:-0.1}"
    --token-keep-ratio "${SDX_TOKEN_KEEP_RATIO:-0.95}"
  )

  if [ "${SDX_MOE_NUM_EXPERTS:-0}" -gt 0 ] 2>/dev/null; then
    _out+=(--moe-num-experts "${SDX_MOE_NUM_EXPERTS}" --moe-top-k "${SDX_MOE_TOP_K:-2}")
  fi

  # --- Caption / dataset training packs (not separate VLMs — prompt shaping) ---
  _out+=(
    --train-prompt-emphasis
    --train-style-guidance-mode "${SDX_TRAIN_STYLE_GUIDANCE:-auto}"
    --region-caption-mode "${SDX_REGION_CAPTION_MODE:-append}"
    --train-art-guidance-mode "${SDX_TRAIN_ART_GUIDANCE:-auto}"
    --train-anatomy-guidance "${SDX_TRAIN_ANATOMY_GUIDANCE:-auto}"
    --boost-adherence-caption
    --train-shortcomings-mitigation "${SDX_TRAIN_SHORTCOMINGS:-auto}"
    --caption-unicode-normalize
    --foveated-train-prob "${SDX_FOVEATED_TRAIN_PROB:-0.15}"
    --train-originality-prob "${SDX_TRAIN_ORIGINALITY_PROB:-0.1}"
  )
  if [ "${SDX_TRAIN_SHORTCOMINGS_2D:-1}" = "1" ]; then
    _out+=(--train-shortcomings-2d)
  fi

  if [ "${SDX_USE_HIERARCHICAL_CAPTIONS:-1}" = "1" ]; then
    _out+=(--use-hierarchical-captions)
  fi

  # --- ControlNet (image) when control maps were preprocessed (optional per batch) ---
  if [ "${SDX_ENABLE_CONTROL_TRAIN:-1}" = "1" ]; then
    local ctrl="${SDX_CONTROL_MANIFEST:-}"
    if [ -n "$ctrl" ] && [ -s "$ctrl" ]; then
      _out+=(
        --control-cond-dim 1
        --control-num-types "${SDX_CONTROL_NUM_TYPES:-9}"
        --control-scale "${SDX_CONTROL_SCALE:-0.85}"
      )
    fi
  fi
}
