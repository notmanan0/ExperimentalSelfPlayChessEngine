#pragma once

#include <array>
#include <optional>

#include <chessmoe/chess/types.h>

namespace chessmoe::chess {

struct Board {
  std::array<std::array<Bitboard, 6>, 2> pieces{};

  void clear();
  std::optional<Piece> piece_at(Square square) const;
  Bitboard pieces_of(Color color, PieceType type) const;
  Bitboard occupancy(Color color) const;
  Bitboard occupancy() const;
  void set_piece(Square square, Piece piece);
  void remove_piece(Square square);
};

}  // namespace chessmoe::chess
