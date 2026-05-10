#include <chessmoe/chess/board.h>

namespace chessmoe::chess {

void Board::clear() {
  for (auto& by_color : pieces) {
    for (auto& bitboard : by_color) {
      bitboard = 0;
    }
  }
}

std::optional<Piece> Board::piece_at(Square square) const {
  if (square == Square::None) {
    return std::nullopt;
  }

  const Bitboard mask = square_bb(square);
  for (int color = 0; color < 2; ++color) {
    for (int piece = 0; piece < 6; ++piece) {
      if ((pieces[color][piece] & mask) != 0) {
        return Piece{static_cast<Color>(color), static_cast<PieceType>(piece)};
      }
    }
  }

  return std::nullopt;
}

Bitboard Board::pieces_of(Color color, PieceType type) const {
  return pieces[color_index(color)][piece_index(type)];
}

Bitboard Board::occupancy(Color color) const {
  Bitboard result = 0;
  for (const auto bitboard : pieces[color_index(color)]) {
    result |= bitboard;
  }
  return result;
}

Bitboard Board::occupancy() const {
  return occupancy(Color::White) | occupancy(Color::Black);
}

void Board::set_piece(Square square, Piece piece) {
  remove_piece(square);
  pieces[color_index(piece.color)][piece_index(piece.type)] |= square_bb(square);
}

void Board::remove_piece(Square square) {
  if (square == Square::None) {
    return;
  }

  const Bitboard mask = ~square_bb(square);
  for (auto& by_color : pieces) {
    for (auto& bitboard : by_color) {
      bitboard &= mask;
    }
  }
}

}  // namespace chessmoe::chess
