cmake_minimum_required(VERSION 3.16)
project(sdx_latent VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(sdx_latent SHARED
  src/sdx_latent.cpp
)

target_include_directories(sdx_latent
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(sdx_latent PRIVATE SDX_LATENT_BUILD)

if(MSVC)
  target_compile_definitions(sdx_latent PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Optional: install
install(TARGETS sdx_latent
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

# Optional: fast path for diffusion inference timestep index finalization (ctypes from Python).
add_library(sdx_inference_timesteps SHARED
  src/sdx_inference_timesteps.cpp
)
target_include_directories(sdx_inference_timesteps
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_definitions(sdx_inference_timesteps PRIVATE SDX_IT_BUILD)
if(MSVC)
  target_compile_definitions(sdx_inference_timesteps PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
install(TARGETS sdx_inference_timesteps
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# Optional: squared-cosine v2 beta schedule (diffusion/schedules.py).
add_library(sdx_beta_schedules SHARED
  src/sdx_beta_schedules.cpp
)
target_include_directories(sdx_beta_schedules
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_definitions(sdx_beta_schedules PRIVATE SDX_BS_BUILD)
if(MSVC)
  target_compile_definitions(sdx_beta_schedules PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
install(TARGETS sdx_beta_schedules
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# Fast JSONL / text newline + byte count (no JSON parse).
add_library(sdx_line_stats SHARED
  src/sdx_line_stats.cpp
)
target_include_directories(sdx_line_stats
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_definitions(sdx_line_stats PRIVATE SDX_LINE_STATS_BUILD)
if(MSVC)
  target_compile_definitions(sdx_line_stats PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
install(TARGETS sdx_line_stats
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# Streaming FNV-1a 64 (manifest fingerprint; matches Python/Zig file mode).
add_library(sdx_fnv64_file SHARED
  src/sdx_fnv64_file.cpp
)
target_include_directories(sdx_fnv64_file
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_definitions(sdx_fnv64_file PRIVATE SDX_FNV64_FILE_BUILD)
if(MSVC)
  target_compile_definitions(sdx_fnv64_file PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
install(TARGETS sdx_fnv64_file
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# CPU RMSNorm rows helper (float32, in-place).
add_library(sdx_rmsnorm_rows_cpu SHARED
  src/sdx_rmsnorm_rows_cpu.cpp
)
target_include_directories(sdx_rmsnorm_rows_cpu
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(sdx_rmsnorm_rows_cpu PRIVATE SDX_RMSNORM_ROWS_CPU_BUILD)
if(MSVC)
  target_compile_definitions(sdx_rmsnorm_rows_cpu PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
install(TARGETS sdx_rmsnorm_rows_cpu
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# CPU mask → patch-weight helper for part-aware training.
add_library(sdx_mask_ops SHARED
  src/sdx_mask_ops.cpp
)
target_include_directories(sdx_mask_ops
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(sdx_mask_ops PRIVATE SDX_MASK_OPS_BUILD)
if(MSVC)
  target_compile_definitions(sdx_mask_ops PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
install(TARGETS sdx_mask_ops
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# CPU image quality metrics (luma stats, laplacian variance, component counting).
add_library(sdx_image_metrics SHARED
  src/sdx_image_metrics.cpp
)
target_include_directories(sdx_image_metrics
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(sdx_image_metrics PRIVATE SDX_IMAGE_METRICS_BUILD)
if(MSVC)
  target_compile_definitions(sdx_image_metrics PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
install(TARGETS sdx_image_metrics
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# CPU score ops (min-max normalization + weighted sum fusion for pick-best).
add_library(sdx_score_ops SHARED
  src/sdx_score_ops.cpp
)
target_include_directories(sdx_score_ops
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(sdx_score_ops PRIVATE SDX_SCORE_OPS_BUILD)
if(MSVC)
  target_compile_definitions(sdx_score_ops PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
install(TARGETS sdx_score_ops
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# Optional CUDA: image layout + small ML-style ops (requires nvcc).
option(SDX_BUILD_CUDA "Build CUDA targets (hwc_to_chw, l2_normalize_rows, flow_matching_velocity, nf4, sdpa_online, rmsnorm, rope, silu_gate)" OFF)
if(SDX_BUILD_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  add_library(sdx_cuda_hwc_to_chw SHARED
    cuda/hwc_to_chw.cu
  )
  set_property(TARGET sdx_cuda_hwc_to_chw PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(sdx_cuda_hwc_to_chw PRIVATE SDX_CUDA_HWC_BUILD)
  target_include_directories(sdx_cuda_hwc_to_chw
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  set_target_properties(sdx_cuda_hwc_to_chw PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(MSVC)
    target_compile_options(sdx_cuda_hwc_to_chw PRIVATE "-Xcompiler=/utf-8")
  endif()
  install(TARGETS sdx_cuda_hwc_to_chw
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  add_library(sdx_cuda_ml SHARED
    cuda/l2_normalize_rows.cu
  )
  set_property(TARGET sdx_cuda_ml PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(sdx_cuda_ml PRIVATE SDX_CUDA_L2_BUILD)
  target_include_directories(sdx_cuda_ml
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  set_target_properties(sdx_cuda_ml PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(MSVC)
    target_compile_options(sdx_cuda_ml PRIVATE "-Xcompiler=/utf-8")
  endif()
  install(TARGETS sdx_cuda_ml
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  add_library(sdx_cuda_flow_matching SHARED
    cuda/flow_matching_velocity.cu
  )
  set_property(TARGET sdx_cuda_flow_matching PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(sdx_cuda_flow_matching PRIVATE SDX_CUDA_FM_BUILD)
  target_include_directories(sdx_cuda_flow_matching
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  set_target_properties(sdx_cuda_flow_matching PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(MSVC)
    target_compile_options(sdx_cuda_flow_matching PRIVATE "-Xcompiler=/utf-8")
  endif()
  install(TARGETS sdx_cuda_flow_matching
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  add_library(sdx_cuda_nf4 SHARED
    cuda/nf4_dequant.cu
  )
  set_property(TARGET sdx_cuda_nf4 PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(sdx_cuda_nf4 PRIVATE SDX_CUDA_NF4_BUILD)
  target_include_directories(sdx_cuda_nf4
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  set_target_properties(sdx_cuda_nf4 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(MSVC)
    target_compile_options(sdx_cuda_nf4 PRIVATE "-Xcompiler=/utf-8")
  endif()
  install(TARGETS sdx_cuda_nf4
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  add_library(sdx_cuda_sdpa_online SHARED
    cuda/sdpa_online_softmax.cu
  )
  set_property(TARGET sdx_cuda_sdpa_online PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(sdx_cuda_sdpa_online PRIVATE SDX_CUDA_SDPA_BUILD)
  target_include_directories(sdx_cuda_sdpa_online
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  set_target_properties(sdx_cuda_sdpa_online PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(MSVC)
    target_compile_options(sdx_cuda_sdpa_online PRIVATE "-Xcompiler=/utf-8")
  endif()
  install(TARGETS sdx_cuda_sdpa_online
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  add_library(sdx_cuda_rmsnorm SHARED
    cuda/rmsnorm_rows.cu
  )
  set_property(TARGET sdx_cuda_rmsnorm PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(sdx_cuda_rmsnorm PRIVATE SDX_CUDA_RMSNORM_BUILD)
  target_include_directories(sdx_cuda_rmsnorm
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  set_target_properties(sdx_cuda_rmsnorm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(MSVC)
    target_compile_options(sdx_cuda_rmsnorm PRIVATE "-Xcompiler=/utf-8")
  endif()
  install(TARGETS sdx_cuda_rmsnorm
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  add_library(sdx_cuda_rope SHARED
    cuda/rope_apply.cu
  )
  set_property(TARGET sdx_cuda_rope PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(sdx_cuda_rope PRIVATE SDX_CUDA_ROPE_BUILD)
  target_include_directories(sdx_cuda_rope
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  set_target_properties(sdx_cuda_rope PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(MSVC)
    target_compile_options(sdx_cuda_rope PRIVATE "-Xcompiler=/utf-8")
  endif()
  install(TARGETS sdx_cuda_rope
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  add_library(sdx_cuda_silu_gate SHARED
    cuda/silu_gate.cu
  )
  set_property(TARGET sdx_cuda_silu_gate PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(sdx_cuda_silu_gate PRIVATE SDX_CUDA_SILU_GATE_BUILD)
  target_include_directories(sdx_cuda_silu_gate
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  set_target_properties(sdx_cuda_silu_gate PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(MSVC)
    target_compile_options(sdx_cuda_silu_gate PRIVATE "-Xcompiler=/utf-8")
  endif()
  install(TARGETS sdx_cuda_silu_gate
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  # Gaussian blur on latent tensors (B, C, H, W) — used by sampling_utils and holy_grail.
  add_library(sdx_cuda_gaussian_blur SHARED
    cuda/gaussian_blur_latent.cu
  )
  set_property(TARGET sdx_cuda_gaussian_blur PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(sdx_cuda_gaussian_blur PRIVATE SDX_CUDA_BLUR_BUILD)
  target_include_directories(sdx_cuda_gaussian_blur
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  set_target_properties(sdx_cuda_gaussian_blur PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(MSVC)
    target_compile_options(sdx_cuda_gaussian_blur PRIVATE "-Xcompiler=/utf-8")
  endif()
  install(TARGETS sdx_cuda_gaussian_blur
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  # Per-sample percentile clamp — used by holy_grail/latent_refiner.
  add_library(sdx_cuda_percentile_clamp SHARED
    cuda/percentile_clamp.cu
  )
  set_property(TARGET sdx_cuda_percentile_clamp PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(sdx_cuda_percentile_clamp PRIVATE SDX_CUDA_PCLAMP_BUILD)
  target_include_directories(sdx_cuda_percentile_clamp
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  set_target_properties(sdx_cuda_percentile_clamp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(MSVC)
    target_compile_options(sdx_cuda_percentile_clamp PRIVATE "-Xcompiler=/utf-8")
  endif()
  install(TARGETS sdx_cuda_percentile_clamp
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  # CUDA image luma stats + clipping ratio for uint8 HWC host buffers.
  add_library(sdx_cuda_image_metrics SHARED
    cuda/image_metrics.cu
  )
  set_property(TARGET sdx_cuda_image_metrics PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(sdx_cuda_image_metrics PRIVATE SDX_CUDA_IMAGE_METRICS_BUILD)
  target_include_directories(sdx_cuda_image_metrics
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  set_target_properties(sdx_cuda_image_metrics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(MSVC)
    target_compile_options(sdx_cuda_image_metrics PRIVATE "-Xcompiler=/utf-8")
  endif()
  install(TARGETS sdx_cuda_image_metrics
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
endif()
