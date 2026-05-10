add_executable(mcts_tests
  test_mcts.cpp
)

target_link_libraries(mcts_tests PRIVATE chessmoe_search)

add_test(NAME mcts_tests COMMAND mcts_tests)

add_executable(gumbel_search_tests
  test_gumbel_search.cpp
)

target_link_libraries(gumbel_search_tests PRIVATE chessmoe_search)

add_test(NAME gumbel_search_tests COMMAND gumbel_search_tests)
