add_executable(chess_core_tests
  test_chess_core.cpp
)

target_link_libraries(chess_core_tests PRIVATE chessmoe_chess)

add_test(NAME chess_core_tests COMMAND chess_core_tests)

add_executable(bitboard_tests
  test_bitboard.cpp
)

target_link_libraries(bitboard_tests PRIVATE chessmoe_chess)

add_test(NAME bitboard_tests COMMAND bitboard_tests)
