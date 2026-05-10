#include <chessmoe/chess/perft.h>

#include <stdexcept>

#include <chessmoe/chess/move_generator.h>

namespace chessmoe::chess {

std::uint64_t perft(const Position& position, int depth) {
  if (depth < 0) {
    throw std::runtime_error("perft depth must be nonnegative");
  }
  if (depth == 0) {
    return 1;
  }

  const auto moves = MoveGenerator::legal_moves(position);
  if (depth == 1) {
    return moves.size();
  }

  std::uint64_t nodes = 0;
  for (const auto move : moves) {
    auto child = position;
    child.make_move(move);
    nodes += perft(child, depth - 1);
  }
  return nodes;
}

std::map<std::string, std::uint64_t> perft_divide(const Position& position,
                                                  int depth) {
  if (depth < 1) {
    throw std::runtime_error("perft divide depth must be positive");
  }

  std::map<std::string, std::uint64_t> result;
  for (const auto move : MoveGenerator::legal_moves(position)) {
    auto child = position;
    child.make_move(move);
    result[move.to_uci()] = perft(child, depth - 1);
  }
  return result;
}

}  // namespace chessmoe::chess
