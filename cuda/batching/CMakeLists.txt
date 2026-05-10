add_library(chessmoe_cuda_batching
  src/batch_planner.cpp
  src/cuda_stream.cpp
)

target_include_directories(chessmoe_cuda_batching
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(chessmoe_cuda_batching PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(chessmoe_cuda_batching PRIVATE /W4 /permissive-)
else()
  target_compile_options(chessmoe_cuda_batching PRIVATE -Wall -Wextra -Wpedantic)
endif()
