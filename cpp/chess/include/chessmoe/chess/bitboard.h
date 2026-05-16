#pragma once

#include <array>
#include <bit>
#include <cstdint>

#include <chessmoe/chess/types.h>

namespace chessmoe::chess {

inline int popcount(Bitboard bb) {
  return std::popcount(bb);
}

inline int lsb_index(Bitboard bb) {
  return std::countr_zero(bb);
}

inline int msb_index(Bitboard bb) {
  return 63 - std::countl_zero(bb);
}

inline Square lsb_square(Bitboard bb) {
  return static_cast<Square>(lsb_index(bb));
}

inline Square msb_square(Bitboard bb) {
  return static_cast<Square>(msb_index(bb));
}

inline Bitboard pop_lsb(Bitboard& bb) {
  const Bitboard lsb = bb & -bb;
  bb &= bb - 1;
  return lsb;
}

inline Bitboard file_mask(int file) {
  return 0x0101010101010101ULL << file;
}

inline Bitboard rank_mask(int rank) {
  return 0xFFULL << (rank * 8);
}

constexpr Bitboard kFileA = 0x0101010101010101ULL;
constexpr Bitboard kFileH = 0x8080808080808080ULL;
constexpr Bitboard kRank1 = 0x00000000000000FFULL;
constexpr Bitboard kRank2 = 0x000000000000FF00ULL;
constexpr Bitboard kRank4 = 0x00000000FF000000ULL;
constexpr Bitboard kRank5 = 0x000000FF00000000ULL;
constexpr Bitboard kRank7 = 0x00FF000000000000ULL;
constexpr Bitboard kRank8 = 0xFF00000000000000ULL;

inline Bitboard shift_n(Bitboard bb) {
  return bb << 8;
}

inline Bitboard shift_s(Bitboard bb) {
  return bb >> 8;
}

inline Bitboard shift_e(Bitboard bb) {
  return (bb & ~kFileH) << 1;
}

inline Bitboard shift_w(Bitboard bb) {
  return (bb & ~kFileA) >> 1;
}

inline Bitboard shift_ne(Bitboard bb) {
  return (bb & ~kFileH) << 9;
}

inline Bitboard shift_nw(Bitboard bb) {
  return (bb & ~kFileA) << 7;
}

inline Bitboard shift_se(Bitboard bb) {
  return (bb & ~kFileH) >> 7;
}

inline Bitboard shift_sw(Bitboard bb) {
  return (bb & ~kFileA) >> 9;
}

Bitboard knight_attacks(Square square);
Bitboard king_attacks(Square square);

std::array<Bitboard, 64> compute_knight_attack_table();
std::array<Bitboard, 64> compute_king_attack_table();

class BitboardIterator {
 public:
  explicit BitboardIterator(Bitboard bb) : bb_(bb) {}

  [[nodiscard]] bool has_next() const {
    return bb_ != 0;
  }

  Square next() {
    const Square sq = lsb_square(bb_);
    bb_ &= bb_ - 1;
    return sq;
  }

 private:
  Bitboard bb_;
};

}  // namespace chessmoe::chess
