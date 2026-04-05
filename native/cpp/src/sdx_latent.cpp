#include "sdx/latent.h"

#include <cstdint>

static int check_div(int a, int b) {
  if (a <= 0 || b <= 0) return 0;
  if (a % b != 0) return 0;
  return 1;
}

extern "C" {

SDX_LATENT_API int sdx_latent_spatial_size(int image_hw, int vae_scale) {
  if (!check_div(image_hw, vae_scale)) return 0;
  return image_hw / vae_scale;
}

SDX_LATENT_API int sdx_patch_grid_dim(int latent_hw, int patch_size) {
  if (!check_div(latent_hw, patch_size)) return 0;
  return latent_hw / patch_size;
}

SDX_LATENT_API int sdx_latent_hw(int image_hw, int vae_scale) {
  return sdx_latent_spatial_size(image_hw, vae_scale);
}

SDX_LATENT_API int sdx_num_patch_tokens(int image_hw, int vae_scale, int patch_size) {
  const int lh = sdx_latent_spatial_size(image_hw, vae_scale);
  if (lh <= 0) return 0;
  const int g = sdx_patch_grid_dim(lh, patch_size);
  if (g <= 0) return 0;
  return g * g;
}

SDX_LATENT_API int sdx_latent_numel(int channels, int latent_h, int latent_w) {
  if (channels <= 0 || latent_h <= 0 || latent_w <= 0) return 0;
  const int64_t n = static_cast<int64_t>(channels) * latent_h * latent_w;
  if (n > static_cast<int64_t>(2147483647)) return 0;
  return static_cast<int>(n);
}

}
