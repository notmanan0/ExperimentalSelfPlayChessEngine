add_library(chessmoe_inference INTERFACE)

target_include_directories(chessmoe_inference
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(chessmoe_inference INTERFACE cxx_std_20)
