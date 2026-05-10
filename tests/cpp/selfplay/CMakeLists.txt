add_executable(selfplay_tests
  test_selfplay.cpp
)

target_link_libraries(selfplay_tests PRIVATE chessmoe_selfplay)

add_test(NAME selfplay_tests COMMAND selfplay_tests)

add_executable(replay_writer_tests
  test_replay_writer.cpp
)

target_link_libraries(replay_writer_tests PRIVATE chessmoe_selfplay)

add_test(NAME replay_writer_tests COMMAND replay_writer_tests)

add_executable(arena_config_tests
  test_arena_config.cpp
)

target_link_libraries(arena_config_tests PRIVATE chessmoe_selfplay)

add_test(NAME arena_config_tests COMMAND arena_config_tests)
