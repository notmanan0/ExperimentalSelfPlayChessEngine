#pragma once

#include <string>
#include <string_view>

#include <chessmoe/chess/position.h>

namespace chessmoe::chess {

class Fen {
 public:
  static Position parse(std::string_view fen);
  static std::string to_string(const Position& position);
};

}  // namespace chessmoe::chess
