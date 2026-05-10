#pragma once

#include <cstdint>
#include <map>
#include <string>

#include <chessmoe/chess/position.h>

namespace chessmoe::chess {

std::uint64_t perft(const Position& position, int depth);
std::map<std::string, std::uint64_t> perft_divide(const Position& position,
                                                  int depth);

}  // namespace chessmoe::chess
