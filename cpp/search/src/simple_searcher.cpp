#include <chessmoe/search/simple_searcher.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>

#include <chessmoe/chess/move_generator.h>

namespace chessmoe::search {

namespace {

constexpr int kMateScore = 30000;
constexpr int kInfinity = 32000;

struct SearchContext {
  const eval::MaterialEvaluator& evaluator;
  chess::Color root_color;
  std::atomic_bool& stop_requested;
  std::optional<std::chrono::steady_clock::time_point> deadline;
  std::uint64_t nodes{0};
};

bool should_stop(SearchContext& context) {
  if (context.stop_requested.load(std::memory_order_relaxed)) {
    return true;
  }
  if (context.deadline.has_value() &&
      std::chrono::steady_clock::now() >= *context.deadline) {
    context.stop_requested.store(true, std::memory_order_relaxed);
    return true;
  }
  return false;
}

int negamax(chess::Position& position, int depth, int ply, int alpha, int beta,
            SearchContext& context) {
  if (should_stop(context)) {
    return context.evaluator.evaluate(position, context.root_color);
  }

  const auto legal_moves = chess::MoveGenerator::legal_moves(position);
  if (depth == 0 || legal_moves.empty()) {
    ++context.nodes;
    if (legal_moves.empty()) {
      if (position.in_check(position.side_to_move())) {
        return -kMateScore + ply;
      }
      return 0;
    }
    return context.evaluator.evaluate(position, context.root_color);
  }

  int best = -kInfinity;
  for (const auto move : legal_moves) {
    auto undo = position.make_move(move);
    const int score = -negamax(position, depth - 1, ply + 1, -beta, -alpha, context);
    position.unmake_move(move, undo);

    if (should_stop(context)) {
      return score;
    }

    best = std::max(best, score);
    alpha = std::max(alpha, score);
    if (alpha >= beta) {
      break;
    }
  }

  return best;
}

int normalized_depth(SearchLimits limits) {
  if (limits.depth > 0) {
    return limits.depth;
  }
  return limits.movetime_ms > 0 ? 64 : 1;
}

}  // namespace

SimpleSearcher::SimpleSearcher(eval::MaterialEvaluator evaluator)
    : evaluator_(evaluator) {}

SearchResult SimpleSearcher::search(const chess::Position& root,
                                    SearchLimits limits,
                                    std::atomic_bool& stop_requested) const {
  stop_requested.store(false, std::memory_order_relaxed);

  SearchContext context{
      evaluator_,
      root.side_to_move(),
      stop_requested,
      std::nullopt,
      0,
  };

  if (limits.movetime_ms > 0) {
    context.deadline = std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(limits.movetime_ms);
  }

  const auto legal_moves = chess::MoveGenerator::legal_moves(root);
  SearchResult result{};
  if (legal_moves.empty()) {
    result.nodes = 1;
    result.score_cp = root.in_check(root.side_to_move()) ? -kMateScore : 0;
    return result;
  }

  result.has_best_move = true;
  result.best_move = legal_moves.front();
  result.score_cp = -kInfinity;

  const int depth = normalized_depth(limits);
  for (const auto move : legal_moves) {
    if (should_stop(context)) {
      break;
    }

    auto child = root;
    child.make_move(move);
    const int score = -negamax(child, depth - 1, 1, -kInfinity, kInfinity, context);

    if (!stop_requested.load(std::memory_order_relaxed) && score > result.score_cp) {
      result.score_cp = score;
      result.best_move = move;
    }
  }

  result.nodes = context.nodes;
  result.stopped = stop_requested.load(std::memory_order_relaxed);
  return result;
}

}  // namespace chessmoe::search
