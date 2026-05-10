add_executable(gpu_selfplay_benchmark
  gpu_selfplay_benchmark.cpp
)

target_link_libraries(gpu_selfplay_benchmark PRIVATE chessmoe_selfplay chessmoe_eval)
target_compile_features(gpu_selfplay_benchmark PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(gpu_selfplay_benchmark PRIVATE /W4 /permissive-)
else()
  target_compile_options(gpu_selfplay_benchmark PRIVATE -Wall -Wextra -Wpedantic)
endif()
