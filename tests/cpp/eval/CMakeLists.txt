add_executable(eval_tests
  test_eval_interface.cpp
)

target_link_libraries(eval_tests PRIVATE chessmoe_eval)

add_test(NAME eval_tests COMMAND eval_tests)

add_executable(pesto_evaluator_tests
  test_pesto_evaluator.cpp
)

target_link_libraries(pesto_evaluator_tests PRIVATE chessmoe_eval)

add_test(NAME pesto_evaluator_tests COMMAND pesto_evaluator_tests)
