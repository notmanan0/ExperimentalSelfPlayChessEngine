add_executable(eval_tests
  test_eval_interface.cpp
)

target_link_libraries(eval_tests PRIVATE chessmoe_eval)

add_test(NAME eval_tests COMMAND eval_tests)
