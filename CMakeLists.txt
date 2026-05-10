cmake_minimum_required(VERSION 3.20)

project(chessmoe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CHESSMOE_BUILD_TESTS "Build chessmoe tests" ON)

add_subdirectory(cpp/chess)
add_subdirectory(cpp/eval)
add_subdirectory(cpp/inference)
add_subdirectory(cpp/search)
add_subdirectory(cpp/selfplay)
add_subdirectory(cpp/uci)
add_subdirectory(tools/perft)

if(CHESSMOE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp/chess)
  add_subdirectory(tests/cpp/eval)
  add_subdirectory(tests/cpp/search)
  add_subdirectory(tests/cpp/selfplay)
  add_subdirectory(tests/cpp/uci)
endif()
