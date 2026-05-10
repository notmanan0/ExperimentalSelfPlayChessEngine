add_executable(mcts_tests
  test_mcts.cpp
)

target_link_libraries(mcts_tests PRIVATE chessmoe_search)

add_test(NAME mcts_tests COMMAND mcts_tests)
