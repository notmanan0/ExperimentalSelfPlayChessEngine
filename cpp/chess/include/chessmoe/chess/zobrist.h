#pragma once

#include <array>
#include <cstdint>

#include <chessmoe/chess/types.h>

namespace chessmoe::chess {

struct ZobristKeys {
  std::array<std::array<std::array<std::uint64_t, 64>, 6>, 2> pieces{};
  std::array<std::uint64_t, 16> castling{};
  std::array<std::uint64_t, 8> en_passant_file{};
  std::uint64_t black_to_move{};
};

const ZobristKeys& zobrist_keys();
std::uint64_t zobrist_hash(const class Position& position);

}  // namespace chessmoe::chess
