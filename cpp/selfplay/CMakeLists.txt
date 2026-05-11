add_library(chessmoe_selfplay
  src/arena_config.cpp
  src/gpu_selfplay_pipeline.cpp
  src/reanalysis_config.cpp
  src/replay_writer.cpp
  src/self_play_generator.cpp
  src/selfplay_app.cpp
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

add_executable(selfplay
  src/selfplay_main.cpp
)

target_link_libraries(selfplay PRIVATE chessmoe_selfplay)
target_compile_features(selfplay PRIVATE cxx_std_20)
set_target_properties(selfplay PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(MSVC)
  target_compile_options(selfplay PRIVATE /W4 /permissive-)
else()
  target_compile_options(selfplay PRIVATE -Wall -Wextra -Wpedantic)
endif()
