add_library(chessmoe_chess
  src/bitboard.cpp
  src/board.cpp
  src/fen.cpp
  src/move.cpp
  src/move_generator.cpp
  src/perft.cpp
  src/position.cpp
  src/types.cpp
  src/zobrist.cpp
)

target_include_directories(chessmoe_chess
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(chessmoe_chess PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(chessmoe_chess PRIVATE /W4 /permissive-)
else()
  target_compile_options(chessmoe_chess PRIVATE -Wall -Wextra -Wpedantic)
endif()
