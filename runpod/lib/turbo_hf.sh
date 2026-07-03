#!/usr/bin/env bash
# Max-speed Hugging Face dataset export (parallel packs + threaded image writes).
sdx_apply_turbo_hf() {
  [ "${SDX_HF_TURBO:-1}" = "0" ] && return 0
  export SDX_HF_TURBO=1
  export SDX_HF_PARALLEL_PACKS="${SDX_HF_PARALLEL_PACKS:-2}"
  export SDX_HF_EXPORT_WORKERS="${SDX_HF_EXPORT_WORKERS:-24}"
  export SDX_HF_JPEG_QUALITY="${SDX_HF_JPEG_QUALITY:-82}"
  export SDX_HF_MANIFEST_FLUSH_EVERY="${SDX_HF_MANIFEST_FLUSH_EVERY:-256}"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export HF_XET_HIGH_PERFORMANCE=1
  export HF_HUB_DOWNLOAD_NUM_THREADS="${HF_HUB_DOWNLOAD_NUM_THREADS:-32}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
}
