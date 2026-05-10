#include <chessmoe/chess/types.h>

#include <array>
#include <cctype>
#include <stdexcept>

namespace chessmoe::chess {

std::string square_to_string(Square square) {
  if (square == Square::None) {
    return "-";
  }

  const char file = static_cast<char>('a' + file_of(square));
  const char rank = static_cast<char>('1' + rank_of(square));
  return {file, rank};
}

Square square_from_string(const std::string& text) {
  if (text.size() != 2 || text[0] < 'a' || text[0] > 'h' || text[1] < '1' ||
      text[1] > '8') {
    return Square::None;
  }

  const int file = text[0] - 'a';
  const int rank = text[1] - '1';
  return square_from_index(rank * 8 + file);
}

char piece_to_fen(Piece piece) {
  static constexpr std::array<char, 6> white{'P', 'N', 'B', 'R', 'Q', 'K'};
  static constexpr std::array<char, 6> black{'p', 'n', 'b', 'r', 'q', 'k'};
  return piece.color == Color::White ? white[piece_index(piece.type)]
                                     : black[piece_index(piece.type)];
}

std::optional<Piece> piece_from_fen(char c) {
  const Color color = std::isupper(static_cast<unsigned char>(c)) ? Color::White
                                                                  : Color::Black;
  switch (std::tolower(static_cast<unsigned char>(c))) {
    case 'p':
      return Piece{color, PieceType::Pawn};
    case 'n':
      return Piece{color, PieceType::Knight};
    case 'b':
      return Piece{color, PieceType::Bishop};
    case 'r':
      return Piece{color, PieceType::Rook};
    case 'q':
      return Piece{color, PieceType::Queen};
    case 'k':
      return Piece{color, PieceType::King};
    default:
      return std::nullopt;
  }
}

}  // namespace chessmoe::chess
