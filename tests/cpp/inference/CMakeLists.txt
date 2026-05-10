add_executable(inference_tests
  test_inference.cpp
)

target_link_libraries(inference_tests
  PRIVATE
    chessmoe_inference
    chessmoe_cuda_batching
    chessmoe_eval
)

add_test(NAME inference_tests COMMAND inference_tests)
