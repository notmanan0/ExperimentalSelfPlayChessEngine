add_library(chessmoe_uci_lib
  src/engine_state.cpp
  src/uci_engine.cpp
  src/uci_loop.cpp
)

target_include_directories(chessmoe_uci_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(chessmoe_uci_lib PUBLIC chessmoe_chess chessmoe_search)
target_compile_features(chessmoe_uci_lib PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(chessmoe_uci_lib PRIVATE /W4 /permissive-)
else()
  target_compile_options(chessmoe_uci_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(chessmoe_uci ALIAS chessmoe_uci_lib)

add_executable(chessmoe
  main.cpp
)

target_link_libraries(chessmoe PRIVATE chessmoe_uci_lib)
