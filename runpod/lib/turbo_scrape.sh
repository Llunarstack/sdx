#!/usr/bin/env bash
# Aggressive booru scrape defaults (CDN parallelism; API capped ~9.5 req/s/site).
sdx_apply_turbo_scrape() {
  [ "${SDX_SCRAPE_TURBO:-1}" = "0" ] && return 0
  export SDX_SCRAPE_WORKERS=256
  export SDX_API_RATE_DANBOORU=9.5
  export SDX_API_RATE_RULE34XXX=9.5
  export SDX_SPLIT_FRAMES=0
  export SDX_FRAME_FPS=1
  export SDX_MAX_FRAMES_PER_POST=48
  export SDX_KEEP_RAW_MEDIA=0
  export SDX_DL_CHUNK_BYTES=1048576
  export SDX_MANIFEST_FLUSH_EVERY=128
}
