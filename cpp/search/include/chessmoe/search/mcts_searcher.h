#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <chessmoe/chess/move.h>
#include <chessmoe/chess/position.h>
#include <chessmoe/eval/evaluator.h>

namespace chessmoe::search {

struct MctsLimits {
  int visits{1};
  int max_depth{0};
  double cpuct{1.5};
  double root_dirichlet_alpha{0.0};
  double root_dirichlet_epsilon{0.0};
  std::uint32_t root_noise_seed{0};
};

struct MctsNode {
  chess::Move move{};
  double prior{1.0};
  int visit_count{0};
  double total_value{0.0};
  std::vector<std::unique_ptr<MctsNode>> children;
  bool terminal{false};
  bool expanded{false};
  std::uint64_t board_hash{0};

  [[nodiscard]] double mean_value() const {
    return visit_count == 0 ? 0.0 : total_value / visit_count;
  }
};

struct RootMoveStats {
  chess::Move move{};
  double prior{0.0};
  int visit_count{0};
  double total_value{0.0};
  double mean_value{0.0};
  std::uint64_t board_hash{0};
};

struct MctsResult {
  bool has_best_move{false};
  chess::Move best_move{};
  double root_value{0.0};
  std::uint64_t root_visits{0};
  bool terminal{false};
  std::vector<RootMoveStats> root_distribution;
};

class MctsSearcher {
 public:
  explicit MctsSearcher(eval::ISinglePositionEvaluator& evaluator);

  MctsResult search(const chess::Position& root, MctsLimits limits);

 private:
  eval::ISinglePositionEvaluator& evaluator_;
};

}  // namespace chessmoe::search
