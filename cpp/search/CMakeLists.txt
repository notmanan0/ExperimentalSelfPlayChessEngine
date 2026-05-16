add_library(chessmoe_search
  src/alphabeta_searcher.cpp
  src/gumbel_searcher.cpp
  src/mcts_searcher.cpp
  src/simple_searcher.cpp
)

target_include_directories(chessmoe_search
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(chessmoe_search PUBLIC chessmoe_chess chessmoe_eval)
target_compile_features(chessmoe_search PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(chessmoe_search PRIVATE /W4 /permissive-)
else()
  target_compile_options(chessmoe_search PRIVATE -Wall -Wextra -Wpedantic)
endif()
