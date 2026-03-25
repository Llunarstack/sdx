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

# Optional CUDA: image layout + small ML-style ops (requires nvcc).
option(SDX_BUILD_CUDA "Build CUDA targets (hwc_to_chw, l2_normalize_rows)" OFF)
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
endif()
