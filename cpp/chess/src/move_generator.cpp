#include <chessmoe/chess/move_generator.h>

#include <array>
#include <span>
#include <utility>

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

void add_move(std::vector<Move>& moves, Square from, Square to,
              MoveFlag flags = MoveFlag::Quiet,
              PieceType promotion = PieceType::Queen) {
  moves.push_back(Move{from, to, promotion, flags});
}

void add_promotions(std::vector<Move>& moves, Square from, Square to,
                    MoveFlag flags) {
  flags = flags | MoveFlag::Promotion;
  add_move(moves, from, to, flags, PieceType::Queen);
  add_move(moves, from, to, flags, PieceType::Rook);
  add_move(moves, from, to, flags, PieceType::Bishop);
  add_move(moves, from, to, flags, PieceType::Knight);
}

void generate_pawn_moves(const Position& position, std::vector<Move>& moves,
                         Square from, Color us) {
  const Color them = opposite(us);
  const int forward = us == Color::White ? 1 : -1;
  const int start_rank = us == Color::White ? 1 : 6;
  const int promotion_from_rank = us == Color::White ? 6 : 1;

  const auto one_step = offset_square(from, 0, forward);
  if (one_step != Square::None && !position.piece_at(one_step).has_value()) {
    if (rank_of(from) == promotion_from_rank) {
      add_promotions(moves, from, one_step, MoveFlag::Quiet);
    } else {
      add_move(moves, from, one_step);
      if (rank_of(from) == start_rank) {
        const auto two_step = offset_square(from, 0, forward * 2);
        if (two_step != Square::None && !position.piece_at(two_step).has_value()) {
          add_move(moves, from, two_step, MoveFlag::DoublePawnPush);
        }
      }
    }
  }

  for (const int file_delta : {-1, 1}) {
    const auto to = offset_square(from, file_delta, forward);
    if (to == Square::None) {
      continue;
    }

    const auto captured = position.piece_at(to);
    if (captured.has_value() && captured->color == them &&
        captured->type != PieceType::King) {
      if (rank_of(from) == promotion_from_rank) {
        add_promotions(moves, from, to, MoveFlag::Capture);
      } else {
        add_move(moves, from, to, MoveFlag::Capture);
      }
    } else if (to == position.en_passant_square()) {
      add_move(moves, from, to, MoveFlag::Capture | MoveFlag::EnPassant);
    }
  }
}

void generate_knight_moves(const Position& position, std::vector<Move>& moves,
                           Square from, Color us) {
  static constexpr std::array<std::pair<int, int>, 8> offsets{{
      {1, 2},
      {2, 1},
      {2, -1},
      {1, -2},
      {-1, -2},
      {-2, -1},
      {-2, 1},
      {-1, 2},
  }};

  for (const auto [df, dr] : offsets) {
    const auto to = offset_square(from, df, dr);
    if (to == Square::None) {
      continue;
    }
    const auto target = position.piece_at(to);
    if (!target.has_value()) {
      add_move(moves, from, to);
    } else if (target->color != us && target->type != PieceType::King) {
      add_move(moves, from, to, MoveFlag::Capture);
    }
  }
}

void generate_sliding_moves(const Position& position, std::vector<Move>& moves,
                            Square from, Color us,
                            std::span<const std::pair<int, int>> dirs) {
  for (const auto [df, dr] : dirs) {
    auto to = offset_square(from, df, dr);
    while (to != Square::None) {
      const auto target = position.piece_at(to);
      if (!target.has_value()) {
        add_move(moves, from, to);
      } else {
        if (target->color != us && target->type != PieceType::King) {
          add_move(moves, from, to, MoveFlag::Capture);
        }
        break;
      }
      to = offset_square(to, df, dr);
    }
  }
}

bool has_rook_for_castle(const Position& position, Square square, Color color) {
  const auto rook = position.piece_at(square);
  return rook.has_value() && rook->color == color && rook->type == PieceType::Rook;
}

bool has_king_for_castle(const Position& position, Square square, Color color) {
  const auto king = position.piece_at(square);
  return king.has_value() && king->color == color && king->type == PieceType::King;
}

void generate_castling(const Position& position, std::vector<Move>& moves,
                       Color us) {
  const Color them = opposite(us);
  if (position.in_check(us)) {
    return;
  }

  if (us == Color::White) {
    if (position.can_castle(WhiteKingSide) &&
        has_king_for_castle(position, Square::E1, us) &&
        has_rook_for_castle(position, Square::H1, us) &&
        !position.piece_at(Square::F1).has_value() &&
        !position.piece_at(Square::G1).has_value() &&
        !position.is_square_attacked(Square::F1, them) &&
        !position.is_square_attacked(Square::G1, them)) {
      add_move(moves, Square::E1, Square::G1, MoveFlag::KingCastle);
    }
    if (position.can_castle(WhiteQueenSide) &&
        has_king_for_castle(position, Square::E1, us) &&
        has_rook_for_castle(position, Square::A1, us) &&
        !position.piece_at(Square::D1).has_value() &&
        !position.piece_at(Square::C1).has_value() &&
        !position.piece_at(Square::B1).has_value() &&
        !position.is_square_attacked(Square::D1, them) &&
        !position.is_square_attacked(Square::C1, them)) {
      add_move(moves, Square::E1, Square::C1, MoveFlag::QueenCastle);
    }
  } else {
    if (position.can_castle(BlackKingSide) &&
        has_king_for_castle(position, Square::E8, us) &&
        has_rook_for_castle(position, Square::H8, us) &&
        !position.piece_at(Square::F8).has_value() &&
        !position.piece_at(Square::G8).has_value() &&
        !position.is_square_attacked(Square::F8, them) &&
        !position.is_square_attacked(Square::G8, them)) {
      add_move(moves, Square::E8, Square::G8, MoveFlag::KingCastle);
    }
    if (position.can_castle(BlackQueenSide) &&
        has_king_for_castle(position, Square::E8, us) &&
        has_rook_for_castle(position, Square::A8, us) &&
        !position.piece_at(Square::D8).has_value() &&
        !position.piece_at(Square::C8).has_value() &&
        !position.piece_at(Square::B8).has_value() &&
        !position.is_square_attacked(Square::D8, them) &&
        !position.is_square_attacked(Square::C8, them)) {
      add_move(moves, Square::E8, Square::C8, MoveFlag::QueenCastle);
    }
  }
}

}  // namespace

std::vector<Move> MoveGenerator::pseudo_legal_moves(const Position& position) {
  static constexpr std::array<std::pair<int, int>, 4> bishop_dirs{{
      {1, 1},
      {1, -1},
      {-1, 1},
      {-1, -1},
  }};
  static constexpr std::array<std::pair<int, int>, 4> rook_dirs{{
      {1, 0},
      {-1, 0},
      {0, 1},
      {0, -1},
  }};
  static constexpr std::array<std::pair<int, int>, 8> queen_dirs{{
      {1, 1},
      {1, -1},
      {-1, 1},
      {-1, -1},
      {1, 0},
      {-1, 0},
      {0, 1},
      {0, -1},
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

  const Color us = position.side_to_move();
  std::vector<Move> moves;
  moves.reserve(128);

  for (int index = 0; index < 64; ++index) {
    const auto from = square_from_index(index);
    const auto piece = position.piece_at(from);
    if (!piece.has_value() || piece->color != us) {
      continue;
    }

    switch (piece->type) {
      case PieceType::Pawn:
        generate_pawn_moves(position, moves, from, us);
        break;
      case PieceType::Knight:
        generate_knight_moves(position, moves, from, us);
        break;
      case PieceType::Bishop:
        generate_sliding_moves(position, moves, from, us, bishop_dirs);
        break;
      case PieceType::Rook:
        generate_sliding_moves(position, moves, from, us, rook_dirs);
        break;
      case PieceType::Queen:
        generate_sliding_moves(position, moves, from, us, queen_dirs);
        break;
      case PieceType::King:
        for (const auto [df, dr] : king_offsets) {
          const auto to = offset_square(from, df, dr);
          if (to == Square::None) {
            continue;
          }
          const auto target = position.piece_at(to);
          if (!target.has_value()) {
            add_move(moves, from, to);
          } else if (target->color != us && target->type != PieceType::King) {
            add_move(moves, from, to, MoveFlag::Capture);
          }
        }
        generate_castling(position, moves, us);
        break;
    }
  }

  return moves;
}

std::vector<Move> MoveGenerator::legal_moves(const Position& position) {
  const Color us = position.side_to_move();
  std::vector<Move> legal;
  const auto pseudo = pseudo_legal_moves(position);
  legal.reserve(pseudo.size());

  for (const auto move : pseudo) {
    auto next = position;
    next.make_move(move);
    if (!next.in_check(us)) {
      legal.push_back(move);
    }
  }

  return legal;
}

bool contains_uci(const std::vector<Move>& moves, std::string_view uci) {
  for (const auto move : moves) {
    if (move.to_uci() == uci) {
      return true;
    }
  }
  return false;
}

}  // namespace chessmoe::chess
