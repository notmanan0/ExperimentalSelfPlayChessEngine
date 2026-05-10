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

add_executable(gpu_selfplay_pipeline_tests
  test_gpu_selfplay_pipeline.cpp
)

target_link_libraries(gpu_selfplay_pipeline_tests PRIVATE chessmoe_selfplay chessmoe_inference)

add_test(NAME gpu_selfplay_pipeline_tests COMMAND gpu_selfplay_pipeline_tests)

add_executable(reanalysis_config_tests
  test_reanalysis_config.cpp
)

target_link_libraries(reanalysis_config_tests PRIVATE chessmoe_selfplay)

add_test(NAME reanalysis_config_tests COMMAND reanalysis_config_tests)
