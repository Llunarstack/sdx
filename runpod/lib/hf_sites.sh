#!/usr/bin/env bash
# Canonical Hugging Face booru-style training packs (no live API scraping).
# Folder names under $SDX_DATA match pack "name" in setup/hf_dataset_packs.json.
sdx_hf_sites() {
  echo "danbooru rule34xxx e621 rule34xyz"
}

sdx_export_hf_sites() {
  local sites
  sites="$(sdx_hf_sites)"
  export SDX_HF_SITES="$sites"
  export SDX_DATA_SITES="$sites"
}
