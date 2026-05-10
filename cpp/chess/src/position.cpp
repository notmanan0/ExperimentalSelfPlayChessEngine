#include <chessmoe/chess/position.h>

#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>
#include <utility>

#include <chessmoe/chess/move_generator.h>
#include <chessmoe/chess/zobrist.h>

namespace chessmoe::chess {

namespace {

Square offset_square(Square square, int file_delta, int rank_delta) {
  const int file = file_of(square) + file_delta;
  const int rank = rank_of(square) + rank_delta;
  if (file < 0 || file >= 8 || rank < 0 || rank >= 8) {
    return Square::None;
  }
  return square_from_index(rank * 8 + file);
}

bool is_starting_rook_square(Square square, Color color, CastlingRight right) {
  if (color == Color::White && right == WhiteKingSide) {
    return square == Square::H1;
  }
  if (color == Color::White && right == WhiteQueenSide) {
    return square == Square::A1;
  }
  if (color == Color::Black && right == BlackKingSide) {
    return square == Square::H8;
  }
  return color == Color::Black && right == BlackQueenSide && square == Square::A8;
}

}  // namespace

Position::Position() {
  refresh_hash_and_history();
}

const Board& Position::board() const {
  return board_;
}

Board& Position::board() {
  return board_;
}

Color Position::side_to_move() const {
  return side_to_move_;
}

std::uint8_t Position::castling_rights() const {
  return castling_rights_;
}

bool Position::can_castle(CastlingRight right) const {
  return (castling_rights_ & right) != 0;
}

Square Position::en_passant_square() const {
  return en_passant_square_;
}

int Position::halfmove_clock() const {
  return halfmove_clock_;
}

int Position::fullmove_number() const {
  return fullmove_number_;
}

std::uint64_t Position::hash() const {
  return hash_;
}

std::size_t Position::repetition_count(std::uint64_t hash) const {
  return static_cast<std::size_t>(
      std::count(hash_history_.begin(), hash_history_.end(), hash));
}

void Position::set_side_to_move(Color color) {
  side_to_move_ = color;
}

void Position::set_castling_rights(std::uint8_t rights) {
  castling_rights_ = rights & 0x0F;
}

void Position::set_en_passant_square(Square square) {
  en_passant_square_ = square;
}

void Position::set_halfmove_clock(int halfmove_clock) {
  halfmove_clock_ = halfmove_clock;
}

void Position::set_fullmove_number(int fullmove_number) {
  fullmove_number_ = fullmove_number;
}

void Position::refresh_hash_and_history() {
  recompute_hash();
  hash_history_.clear();
  hash_history_.push_back(hash_);
}

std::optional<Piece> Position::piece_at(Square square) const {
  return board_.piece_at(square);
}

Square Position::king_square(Color color) const {
  const Bitboard kings = board_.pieces_of(color, PieceType::King);
  if (kings == 0) {
    throw std::runtime_error("position has no king for requested color");
  }
  return square_from_index(std::countr_zero(kings));
}

bool Position::is_square_attacked(Square square, Color by_color) const {
  static constexpr std::array<std::pair<int, int>, 8> knight_offsets{{
      {1, 2},
      {2, 1},
      {2, -1},
      {1, -2},
      {-1, -2},
      {-2, -1},
      {-2, 1},
      {-1, 2},
  }};
  static constexpr std::array<std::pair<int, int>, 8> king_offsets{{
      {1, 1},
      {1, 0},
      {1, -1},
      {0, -1},
      {-1, -1},
      {-1, 0},
      {-1, 1},
      {0, 1},
  }};
  static constexpr std::array<std::pair<int, int>, 4> rook_dirs{{
      {1, 0},
      {-1, 0},
      {0, 1},
      {0, -1},
  }};
  static constexpr std::array<std::pair<int, int>, 4> bishop_dirs{{
      {1, 1},
      {1, -1},
      {-1, 1},
      {-1, -1},
  }};

  const int pawn_source_rank_delta = by_color == Color::White ? -1 : 1;
  for (const int file_delta : {-1, 1}) {
    const auto from = offset_square(square, file_delta, pawn_source_rank_delta);
    if (from != Square::None) {
      const auto piece = piece_at(from);
      if (piece.has_value() && piece->color == by_color &&
          piece->type == PieceType::Pawn) {
        return true;
      }
    }
  }

  for (const auto [df, dr] : knight_offsets) {
    const auto from = offset_square(square, df, dr);
    if (from == Square::None) {
      continue;
    }
    const auto piece = piece_at(from);
    if (piece.has_value() && piece->color == by_color &&
        piece->type == PieceType::Knight) {
      return true;
    }
  }

  for (const auto [df, dr] : king_offsets) {
    const auto from = offset_square(square, df, dr);
    if (from == Square::None) {
      continue;
    }
    const auto piece = piece_at(from);
    if (piece.has_value() && piece->color == by_color &&
        piece->type == PieceType::King) {
      return true;
    }
  }

  for (const auto [df, dr] : rook_dirs) {
    auto from = offset_square(square, df, dr);
    while (from != Square::None) {
      const auto piece = piece_at(from);
      if (piece.has_value()) {
        if (piece->color == by_color &&
            (piece->type == PieceType::Rook || piece->type == PieceType::Queen)) {
          return true;
        }
        break;
      }
      from = offset_square(from, df, dr);
    }
  }

  for (const auto [df, dr] : bishop_dirs) {
    auto from = offset_square(square, df, dr);
    while (from != Square::None) {
      const auto piece = piece_at(from);
      if (piece.has_value()) {
        if (piece->color == by_color &&
            (piece->type == PieceType::Bishop || piece->type == PieceType::Queen)) {
          return true;
        }
        break;
      }
      from = offset_square(from, df, dr);
    }
  }

  return false;
}

bool Position::in_check(Color color) const {
  return is_square_attacked(king_square(color), opposite(color));
}

bool Position::is_checkmate() const {
  return in_check(side_to_move_) && MoveGenerator::legal_moves(*this).empty();
}

bool Position::is_stalemate() const {
  return !in_check(side_to_move_) && MoveGenerator::legal_moves(*this).empty();
}

UndoState Position::make_move(Move move) {
  const auto moved_piece = piece_at(move.from);
  if (!moved_piece.has_value()) {
    throw std::runtime_error("cannot make move without a moving piece");
  }

  UndoState undo{board_, side_to_move_, castling_rights_, en_passant_square_,
                 halfmove_clock_, fullmove_number_, hash_, hash_history_.size()};

  auto captured_piece = piece_at(move.to);
  board_.remove_piece(move.from);

  if (move.is_en_passant()) {
    const auto captured_square =
        moved_piece->color == Color::White ? square_from_index(square_index(move.to) - 8)
                                           : square_from_index(square_index(move.to) + 8);
    captured_piece = piece_at(captured_square);
    board_.remove_piece(captured_square);
  } else if (captured_piece.has_value()) {
    board_.remove_piece(move.to);
  }

  Piece placed_piece = *moved_piece;
  if (move.is_promotion()) {
    placed_piece.type = move.promotion;
  }
  board_.set_piece(move.to, placed_piece);

  if (has_flag(move.flags, MoveFlag::KingCastle)) {
    const bool white = moved_piece->color == Color::White;
    const auto rook_from = white ? Square::H1 : Square::H8;
    const auto rook_to = white ? Square::F1 : Square::F8;
    board_.remove_piece(rook_from);
    board_.set_piece(rook_to, Piece{moved_piece->color, PieceType::Rook});
  } else if (has_flag(move.flags, MoveFlag::QueenCastle)) {
    const bool white = moved_piece->color == Color::White;
    const auto rook_from = white ? Square::A1 : Square::A8;
    const auto rook_to = white ? Square::D1 : Square::D8;
    board_.remove_piece(rook_from);
    board_.set_piece(rook_to, Piece{moved_piece->color, PieceType::Rook});
  }

  update_castling_rights_after_move(move, *moved_piece, captured_piece);

  en_passant_square_ = Square::None;
  if (moved_piece->type == PieceType::Pawn &&
      has_flag(move.flags, MoveFlag::DoublePawnPush)) {
    const int midpoint = (square_index(move.from) + square_index(move.to)) / 2;
    en_passant_square_ = square_from_index(midpoint);
  }

  halfmove_clock_ =
      moved_piece->type == PieceType::Pawn || captured_piece.has_value()
          ? 0
          : halfmove_clock_ + 1;
  if (side_to_move_ == Color::Black) {
    ++fullmove_number_;
  }
  side_to_move_ = opposite(side_to_move_);
  recompute_hash();
  hash_history_.push_back(hash_);

  return undo;
}

void Position::unmake_move(Move, const UndoState& undo) {
  board_ = undo.board;
  side_to_move_ = undo.side_to_move;
  castling_rights_ = undo.castling_rights;
  en_passant_square_ = undo.en_passant_square;
  halfmove_clock_ = undo.halfmove_clock;
  fullmove_number_ = undo.fullmove_number;
  hash_ = undo.hash;
  hash_history_.resize(undo.history_size);
}

void Position::update_castling_rights_after_move(
    Move move, Piece moved_piece, std::optional<Piece> captured_piece) {
  if (moved_piece.type == PieceType::King) {
    if (moved_piece.color == Color::White) {
      castling_rights_ &= static_cast<std::uint8_t>(~(WhiteKingSide | WhiteQueenSide));
    } else {
      castling_rights_ &= static_cast<std::uint8_t>(~(BlackKingSide | BlackQueenSide));
    }
  }

  if (moved_piece.type == PieceType::Rook) {
    if (is_starting_rook_square(move.from, Color::White, WhiteKingSide)) {
      castling_rights_ &= static_cast<std::uint8_t>(~WhiteKingSide);
    } else if (is_starting_rook_square(move.from, Color::White, WhiteQueenSide)) {
      castling_rights_ &= static_cast<std::uint8_t>(~WhiteQueenSide);
    } else if (is_starting_rook_square(move.from, Color::Black, BlackKingSide)) {
      castling_rights_ &= static_cast<std::uint8_t>(~BlackKingSide);
    } else if (is_starting_rook_square(move.from, Color::Black, BlackQueenSide)) {
      castling_rights_ &= static_cast<std::uint8_t>(~BlackQueenSide);
    }
  }

  if (captured_piece.has_value() && captured_piece->type == PieceType::Rook) {
    if (move.to == Square::H1) {
      castling_rights_ &= static_cast<std::uint8_t>(~WhiteKingSide);
    } else if (move.to == Square::A1) {
      castling_rights_ &= static_cast<std::uint8_t>(~WhiteQueenSide);
    } else if (move.to == Square::H8) {
      castling_rights_ &= static_cast<std::uint8_t>(~BlackKingSide);
    } else if (move.to == Square::A8) {
      castling_rights_ &= static_cast<std::uint8_t>(~BlackQueenSide);
    }
  }
}

void Position::recompute_hash() {
  hash_ = zobrist_hash(*this);
}

}  // namespace chessmoe::chess
