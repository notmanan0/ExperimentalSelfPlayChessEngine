#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace chessmoe::chess {

using Bitboard = std::uint64_t;

enum class Color : std::uint8_t { White = 0, Black = 1 };
enum class PieceType : std::uint8_t {
  Pawn = 0,
  Knight = 1,
  Bishop = 2,
  Rook = 3,
  Queen = 4,
  King = 5,
};

enum class Square : std::uint8_t {
  A1 = 0,
  B1,
  C1,
  D1,
  E1,
  F1,
  G1,
  H1,
  A2,
  B2,
  C2,
  D2,
  E2,
  F2,
  G2,
  H2,
  A3,
  B3,
  C3,
  D3,
  E3,
  F3,
  G3,
  H3,
  A4,
  B4,
  C4,
  D4,
  E4,
  F4,
  G4,
  H4,
  A5,
  B5,
  C5,
  D5,
  E5,
  F5,
  G5,
  H5,
  A6,
  B6,
  C6,
  D6,
  E6,
  F6,
  G6,
  H6,
  A7,
  B7,
  C7,
  D7,
  E7,
  F7,
  G7,
  H7,
  A8,
  B8,
  C8,
  D8,
  E8,
  F8,
  G8,
  H8,
  None = 64,
};

struct Piece {
  Color color;
  PieceType type;
};

constexpr int color_index(Color color) {
  return static_cast<int>(color);
}

constexpr int piece_index(PieceType type) {
  return static_cast<int>(type);
}

constexpr Color opposite(Color color) {
  return color == Color::White ? Color::Black : Color::White;
}

constexpr int square_index(Square square) {
  return static_cast<int>(square);
}

constexpr Square square_from_index(int index) {
  return index >= 0 && index < 64 ? static_cast<Square>(index) : Square::None;
}

constexpr Bitboard square_bb(Square square) {
  return square == Square::None ? 0ULL : (1ULL << square_index(square));
}

constexpr int file_of(Square square) {
  return square_index(square) & 7;
}

constexpr int rank_of(Square square) {
  return square_index(square) >> 3;
}

std::string square_to_string(Square square);
Square square_from_string(const std::string& text);
char piece_to_fen(Piece piece);
std::optional<Piece> piece_from_fen(char c);

}  // namespace chessmoe::chess
