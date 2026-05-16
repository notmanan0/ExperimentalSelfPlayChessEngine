add_library(chessmoe_eval
  src/evaluator.cpp
  src/material_evaluator.cpp
  src/material_policy_evaluator.cpp
  src/pesto_evaluator.cpp
  src/random_evaluator.cpp
)

target_include_directories(chessmoe_eval
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(chessmoe_eval PUBLIC chessmoe_chess)
target_compile_features(chessmoe_eval PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(chessmoe_eval PRIVATE /W4 /permissive-)
else()
  target_compile_options(chessmoe_eval PRIVATE -Wall -Wextra -Wpedantic)
endif()
