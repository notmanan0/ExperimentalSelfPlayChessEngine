#include <chessmoe/chess/bitboard.h>

namespace chessmoe::chess {

namespace {

std::array<Bitboard, 64> init_knight_attacks() {
  std::array<Bitboard, 64> table{};
  for (int sq = 0; sq < 64; ++sq) {
    const int rank = sq >> 3;
    const int file = sq & 7;
    Bitboard attacks = 0;
    const int offsets[][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                              {1, -2},  {1, 2},  {2, -1},  {2, 1}};
    for (const auto& [dr, df] : offsets) {
      const int r = rank + dr;
      const int f = file + df;
      if (r >= 0 && r < 8 && f >= 0 && f < 8) {
        attacks |= 1ULL << (r * 8 + f);
      }
    }
    table[sq] = attacks;
  }
  return table;
}

std::array<Bitboard, 64> init_king_attacks() {
  std::array<Bitboard, 64> table{};
  for (int sq = 0; sq < 64; ++sq) {
    const int rank = sq >> 3;
    const int file = sq & 7;
    Bitboard attacks = 0;
    for (int dr = -1; dr <= 1; ++dr) {
      for (int df = -1; df <= 1; ++df) {
        if (dr == 0 && df == 0) continue;
        const int r = rank + dr;
        const int f = file + df;
        if (r >= 0 && r < 8 && f >= 0 && f < 8) {
          attacks |= 1ULL << (r * 8 + f);
        }
      }
    }
    table[sq] = attacks;
  }
  return table;
}

}  // namespace

static const auto kKnightAttacks = init_knight_attacks();
static const auto kKingAttacks = init_king_attacks();

Bitboard knight_attacks(Square square) {
  return kKnightAttacks[square_index(square)];
}

Bitboard king_attacks(Square square) {
  return kKingAttacks[square_index(square)];
}

std::array<Bitboard, 64> compute_knight_attack_table() {
  return kKnightAttacks;
}

std::array<Bitboard, 64> compute_king_attack_table() {
  return kKingAttacks;
}

}  // namespace chessmoe::chess
