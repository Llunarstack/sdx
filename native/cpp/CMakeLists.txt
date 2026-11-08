cmake_minimum_required(VERSION 3.18)
project(sdx_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SDX_BUILD_CUDA "Build the sdx_cuda library (needs nvcc)" OFF)

function(sdx_library name)
  add_library(${name} SHARED ${ARGN})
  target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(${name} PRIVATE SDX_BUILD)
  install(TARGETS ${name} RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
endfunction()

sdx_library(sdx_cpu src/sdx_cpu.cpp)

if(SDX_BUILD_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  sdx_library(sdx_cuda src/sdx_cuda.cu)
  if(MSVC)
    target_compile_options(sdx_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
  endif()
endif()

install(FILES include/sdx.h DESTINATION include)
