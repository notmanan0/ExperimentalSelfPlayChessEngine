#include <chessmoe/chess/zobrist.h>

#include <bit>

#include <chessmoe/chess/position.h>

namespace chessmoe::chess {

namespace {

std::uint64_t splitmix64(std::uint64_t& state) {
  std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

ZobristKeys make_keys() {
  ZobristKeys keys;
  std::uint64_t state = 0x43484553534D4F45ULL;

  for (auto& by_color : keys.pieces) {
    for (auto& by_piece : by_color) {
      for (auto& key : by_piece) {
        key = splitmix64(state);
      }
    }
  }
  for (auto& key : keys.castling) {
    key = splitmix64(state);
  }
  for (auto& key : keys.en_passant_file) {
    key = splitmix64(state);
  }
  keys.black_to_move = splitmix64(state);
  return keys;
}

}  // namespace

const ZobristKeys& zobrist_keys() {
  static const ZobristKeys keys = make_keys();
  return keys;
}

std::uint64_t zobrist_hash(const Position& position) {
  const auto& keys = zobrist_keys();
  std::uint64_t hash = 0;

  for (int color = 0; color < 2; ++color) {
    for (int piece = 0; piece < 6; ++piece) {
      Bitboard bitboard =
          position.board().pieces[static_cast<std::size_t>(color)]
                                 [static_cast<std::size_t>(piece)];
      while (bitboard != 0) {
        const int square = std::countr_zero(bitboard);
        bitboard &= bitboard - 1;
        hash ^= keys.pieces[static_cast<std::size_t>(color)]
                           [static_cast<std::size_t>(piece)]
                           [static_cast<std::size_t>(square)];
      }
    }
  }

  hash ^= keys.castling[position.castling_rights() & 0x0F];
  if (position.en_passant_square() != Square::None) {
    hash ^= keys.en_passant_file[file_of(position.en_passant_square())];
  }
  if (position.side_to_move() == Color::Black) {
    hash ^= keys.black_to_move;
  }

  return hash;
}

}  // namespace chessmoe::chess
