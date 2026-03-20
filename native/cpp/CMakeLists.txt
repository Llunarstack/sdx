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
