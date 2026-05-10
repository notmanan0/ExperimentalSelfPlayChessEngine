#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include <chessmoe/chess/types.h>

namespace chessmoe::chess {

class Position;

enum class MoveFlag : std::uint8_t {
  Quiet = 0,
  Capture = 1 << 0,
  DoublePawnPush = 1 << 1,
  KingCastle = 1 << 2,
  QueenCastle = 1 << 3,
  EnPassant = 1 << 4,
  Promotion = 1 << 5,
};

constexpr MoveFlag operator|(MoveFlag lhs, MoveFlag rhs) {
  return static_cast<MoveFlag>(static_cast<std::uint8_t>(lhs) |
                               static_cast<std::uint8_t>(rhs));
}

constexpr bool has_flag(MoveFlag flags, MoveFlag flag) {
  return (static_cast<std::uint8_t>(flags) & static_cast<std::uint8_t>(flag)) !=
         0;
}

struct Move {
  Square from{Square::None};
  Square to{Square::None};
  PieceType promotion{PieceType::Queen};
  MoveFlag flags{MoveFlag::Quiet};

  [[nodiscard]] bool is_capture() const;
  [[nodiscard]] bool is_promotion() const;
  [[nodiscard]] bool is_en_passant() const;
  [[nodiscard]] bool is_castling() const;
  [[nodiscard]] std::string to_uci() const;

  static Move from_uci(std::string_view text, const Position& position);
};

bool operator==(const Move& lhs, const Move& rhs);

}  // namespace chessmoe::chess
