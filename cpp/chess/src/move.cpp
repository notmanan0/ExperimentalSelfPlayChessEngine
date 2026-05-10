#include <chessmoe/chess/move.h>

#include <cmath>
#include <stdexcept>

#include <chessmoe/chess/position.h>

namespace chessmoe::chess {

bool Move::is_capture() const {
  return has_flag(flags, MoveFlag::Capture);
}

bool Move::is_promotion() const {
  return has_flag(flags, MoveFlag::Promotion);
}

bool Move::is_en_passant() const {
  return has_flag(flags, MoveFlag::EnPassant);
}

bool Move::is_castling() const {
  return has_flag(flags, MoveFlag::KingCastle) ||
         has_flag(flags, MoveFlag::QueenCastle);
}

std::string Move::to_uci() const {
  std::string result = square_to_string(from) + square_to_string(to);
  if (is_promotion()) {
    switch (promotion) {
      case PieceType::Knight:
        result.push_back('n');
        break;
      case PieceType::Bishop:
        result.push_back('b');
        break;
      case PieceType::Rook:
        result.push_back('r');
        break;
      case PieceType::Queen:
        result.push_back('q');
        break;
      default:
        throw std::runtime_error("invalid promotion piece");
    }
  }
  return result;
}

Move Move::from_uci(std::string_view text, const Position& position) {
  if (text.size() != 4 && text.size() != 5) {
    throw std::runtime_error("invalid UCI move length");
  }

  const auto from = square_from_string(std::string{text.substr(0, 2)});
  const auto to = square_from_string(std::string{text.substr(2, 2)});
  if (from == Square::None || to == Square::None) {
    throw std::runtime_error("invalid UCI move square");
  }

  const auto moving_piece = position.piece_at(from);
  if (!moving_piece.has_value()) {
    throw std::runtime_error("UCI move has no piece on from-square");
  }

  Move move{from, to, PieceType::Queen, MoveFlag::Quiet};

  if (position.piece_at(to).has_value()) {
    move.flags = move.flags | MoveFlag::Capture;
  }

  const int file_delta = file_of(to) - file_of(from);
  const int rank_delta = rank_of(to) - rank_of(from);

  if (moving_piece->type == PieceType::Pawn) {
    if (to == position.en_passant_square() && file_delta != 0 &&
        !position.piece_at(to).has_value()) {
      move.flags = move.flags | MoveFlag::Capture | MoveFlag::EnPassant;
    }
    if (std::abs(rank_delta) == 2) {
      move.flags = move.flags | MoveFlag::DoublePawnPush;
    }
  }

  if (moving_piece->type == PieceType::King && std::abs(file_delta) == 2) {
    move.flags = move.flags | (file_delta > 0 ? MoveFlag::KingCastle
                                              : MoveFlag::QueenCastle);
  }

  if (text.size() == 5) {
    switch (text[4]) {
      case 'n':
        move.promotion = PieceType::Knight;
        break;
      case 'b':
        move.promotion = PieceType::Bishop;
        break;
      case 'r':
        move.promotion = PieceType::Rook;
        break;
      case 'q':
        move.promotion = PieceType::Queen;
        break;
      default:
        throw std::runtime_error("invalid UCI promotion piece");
    }
    move.flags = move.flags | MoveFlag::Promotion;
  }

  return move;
}

bool operator==(const Move& lhs, const Move& rhs) {
  return lhs.from == rhs.from && lhs.to == rhs.to &&
         lhs.promotion == rhs.promotion && lhs.flags == rhs.flags;
}

}  // namespace chessmoe::chess
