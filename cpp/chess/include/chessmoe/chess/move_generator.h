#pragma once

#include <string_view>
#include <vector>

#include <chessmoe/chess/move.h>
#include <chessmoe/chess/position.h>

namespace chessmoe::chess {

class MoveGenerator {
 public:
  static std::vector<Move> legal_moves(const Position& position);
  static std::vector<Move> pseudo_legal_moves(const Position& position);
};

bool contains_uci(const std::vector<Move>& moves, std::string_view uci);

}  // namespace chessmoe::chess
