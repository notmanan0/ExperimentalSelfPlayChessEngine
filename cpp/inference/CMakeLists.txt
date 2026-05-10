add_library(chessmoe_inference
  src/tensor_layout.cpp
  src/tensorrt_engine.cpp
  src/tensorrt_evaluator.cpp
)

target_include_directories(chessmoe_inference
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(chessmoe_inference PUBLIC chessmoe_eval)
target_compile_features(chessmoe_inference PUBLIC cxx_std_20)

if(CHESSMOE_ENABLE_TENSORRT)
  target_compile_definitions(chessmoe_inference PUBLIC CHESSMOE_ENABLE_TENSORRT)
endif()

if(MSVC)
  target_compile_options(chessmoe_inference PRIVATE /W4 /permissive-)
else()
  target_compile_options(chessmoe_inference PRIVATE -Wall -Wextra -Wpedantic)
endif()
