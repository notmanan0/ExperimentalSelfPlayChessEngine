#pragma once

#include <cstdint>
#include <vector>

#include <chessmoe/chess/board.h>
#include <chessmoe/chess/move.h>

namespace chessmoe::chess {

enum CastlingRight : std::uint8_t {
  WhiteKingSide = 1 << 0,
  WhiteQueenSide = 1 << 1,
  BlackKingSide = 1 << 2,
  BlackQueenSide = 1 << 3,
};

struct UndoState {
  Board board;
  Color side_to_move;
  std::uint8_t castling_rights;
  Square en_passant_square;
  int halfmove_clock;
  int fullmove_number;
  std::uint64_t hash;
  std::size_t history_size;
};

class Position {
 public:
  Position();

  [[nodiscard]] const Board& board() const;
  [[nodiscard]] Board& board();
  [[nodiscard]] Color side_to_move() const;
  [[nodiscard]] std::uint8_t castling_rights() const;
  [[nodiscard]] bool can_castle(CastlingRight right) const;
  [[nodiscard]] Square en_passant_square() const;
  [[nodiscard]] int halfmove_clock() const;
  [[nodiscard]] int fullmove_number() const;
  [[nodiscard]] std::uint64_t hash() const;
  [[nodiscard]] std::size_t repetition_count(std::uint64_t hash) const;

  void set_side_to_move(Color color);
  void set_castling_rights(std::uint8_t rights);
  void set_en_passant_square(Square square);
  void set_halfmove_clock(int halfmove_clock);
  void set_fullmove_number(int fullmove_number);
  void refresh_hash_and_history();

  [[nodiscard]] std::optional<Piece> piece_at(Square square) const;
  [[nodiscard]] Square king_square(Color color) const;
  [[nodiscard]] bool is_square_attacked(Square square, Color by_color) const;
  [[nodiscard]] bool in_check(Color color) const;
  [[nodiscard]] bool is_checkmate() const;
  [[nodiscard]] bool is_stalemate() const;

  UndoState make_move(Move move);
  void unmake_move(Move move, const UndoState& undo);

 private:
  void update_castling_rights_after_move(Move move, Piece moved_piece,
                                         std::optional<Piece> captured_piece);
  void recompute_hash();

  Board board_{};
  Color side_to_move_{Color::White};
  std::uint8_t castling_rights_{0};
  Square en_passant_square_{Square::None};
  int halfmove_clock_{0};
  int fullmove_number_{1};
  std::uint64_t hash_{0};
  std::vector<std::uint64_t> hash_history_{};
};

}  // namespace chessmoe::chess
