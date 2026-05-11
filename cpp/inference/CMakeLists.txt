add_library(chessmoe_inference
  src/async_batching_evaluator.cpp
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
  find_package(CUDAToolkit REQUIRED)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS "$ENV{TensorRT_ROOT}" "$ENV{TENSORRT_ROOT}"
    PATH_SUFFIXES include
  )
  find_library(TENSORRT_NVINFER_LIBRARY nvinfer
    HINTS "$ENV{TensorRT_ROOT}" "$ENV{TENSORRT_ROOT}"
    PATH_SUFFIXES lib lib64
  )
  if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY)
    message(FATAL_ERROR "CHESSMOE_ENABLE_TENSORRT=ON requires TensorRT headers and nvinfer library")
  endif()
  target_compile_definitions(chessmoe_inference PUBLIC CHESSMOE_ENABLE_TENSORRT)
  target_include_directories(chessmoe_inference PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(chessmoe_inference PRIVATE CUDA::cudart ${TENSORRT_NVINFER_LIBRARY})
endif()

if(MSVC)
  target_compile_options(chessmoe_inference PRIVATE /W4 /permissive-)
else()
  target_compile_options(chessmoe_inference PRIVATE -Wall -Wextra -Wpedantic)
endif()
