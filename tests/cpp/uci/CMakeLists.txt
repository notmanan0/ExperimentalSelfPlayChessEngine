add_executable(uci_tests
  test_uci.cpp
)

target_link_libraries(uci_tests PRIVATE chessmoe_uci)

add_test(NAME uci_tests COMMAND uci_tests)
