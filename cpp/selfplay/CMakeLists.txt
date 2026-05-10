add_library(chessmoe_selfplay
  src/arena_config.cpp
  src/gpu_selfplay_pipeline.cpp
  src/replay_writer.cpp
  src/self_play_generator.cpp
)

target_include_directories(chessmoe_selfplay
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(chessmoe_selfplay PUBLIC chessmoe_chess chessmoe_eval chessmoe_search chessmoe_inference)
target_compile_features(chessmoe_selfplay PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(chessmoe_selfplay PRIVATE /W4 /permissive-)
else()
  target_compile_options(chessmoe_selfplay PRIVATE -Wall -Wextra -Wpedantic)
endif()
