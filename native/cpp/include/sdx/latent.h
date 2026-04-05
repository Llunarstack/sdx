/**
 * SDX latent grid helpers (SD-style VAE scale + DiT patch embedding).
 * C ABI for Python ctypes / Zig / Rust bindgen / other languages.
 */
#ifndef SDX_LATENT_H
#define SDX_LATENT_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_LATENT_BUILD)
#define SDX_LATENT_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_LATENT_API __declspec(dllimport)
#else
#define SDX_LATENT_API
#endif

/** Spatial size of latent grid per side: image_hw / vae_scale (integer division). */
SDX_LATENT_API int sdx_latent_spatial_size(int image_hw, int vae_scale);

/** Number of patches per side: latent_hw / patch_size. */
SDX_LATENT_API int sdx_patch_grid_dim(int latent_hw, int patch_size);

/**
 * Total DiT patch tokens for square images (no register tokens).
 * latent_hw = image_hw / vae_scale; tokens = (latent_hw / patch)^2
 */
SDX_LATENT_API int sdx_num_patch_tokens(int image_hw, int vae_scale, int patch_size);

/** Combined helper; returns 0 if inputs invalid (non-positive or not divisible). */
SDX_LATENT_API int sdx_latent_hw(int image_hw, int vae_scale);

/**
 * Element count for a latent tensor (channels * H * W). Any non-positive dimension → 0.
 * Useful for logging / buffer sizing next to DiT/VAE latents.
 */
SDX_LATENT_API int sdx_latent_numel(int channels, int latent_h, int latent_w);

#ifdef __cplusplus
}
#endif

#endif /* SDX_LATENT_H */
