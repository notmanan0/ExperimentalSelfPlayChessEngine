#pragma once

#include <cstdint>

#include <chessmoe/chess/position.h>
#include <chessmoe/eval/evaluator.h>
#include <chessmoe/search/mcts_searcher.h>

namespace chessmoe::search {

struct GumbelSearchLimits {
  int simulations{16};
  int max_considered_actions{16};
  double value_scale{1.0};
  bool deterministic{false};
  std::uint32_t seed{1};
};

class GumbelSearcher {
 public:
  explicit GumbelSearcher(eval::ISinglePositionEvaluator& evaluator);

  MctsResult search(const chess::Position& root, GumbelSearchLimits limits);

 private:
  eval::ISinglePositionEvaluator& evaluator_;
};

}  // namespace chessmoe::search
