#include <chessmoe/chess/fen.h>
#include <chessmoe/chess/move_generator.h>
#include <chessmoe/eval/evaluator.h>
#include <chessmoe/search/gumbel_searcher.h>
#include <chessmoe/search/search_mode.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void require_near(double actual, double expected, double tolerance,
                  std::string_view message) {
  if (std::fabs(actual - expected) > tolerance) {
    throw std::runtime_error(std::string(message) + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

class PreferredMoveEvaluator final : public chessmoe::eval::ISinglePositionEvaluator {
 public:
  explicit PreferredMoveEvaluator(std::string preferred) : preferred_(std::move(preferred)) {}

  chessmoe::eval::EvaluationResult evaluate(
      const chessmoe::eval::EvaluationRequest& request) override {
    chessmoe::eval::EvaluationResult result;
    result.value = 0.0;
    result.wdl = {0.0, 1.0, 0.0};
    for (const auto move : request.legal_moves) {
      result.policy.push_back(
          {move, move.to_uci() == preferred_ ? 8.0 : 0.0,
           move.to_uci() == preferred_ ? 100.0 : 1.0});
    }
    return result;
  }

 private:
  std::string preferred_;
};

double target_sum(const chessmoe::search::MctsResult& result) {
  return std::accumulate(
      result.root_distribution.begin(), result.root_distribution.end(), 0.0,
      [](double sum, const auto& stat) { return sum + stat.target_probability; });
}

void test_search_mode_enum_keeps_puct_and_gumbel_available() {
  require(chessmoe::search::SearchMode::Puct != chessmoe::search::SearchMode::Gumbel,
          "search modes are distinct");
}

void test_gumbel_root_selection_is_deterministic_in_test_mode() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  PreferredMoveEvaluator evaluator("e2e4");
  chessmoe::search::GumbelSearcher searcher(evaluator);

  const auto limits = chessmoe::search::GumbelSearchLimits{
      .simulations = 8,
      .max_considered_actions = 8,
      .deterministic = true,
      .seed = 17,
  };
  const auto a = searcher.search(position, limits);
  const auto b = searcher.search(position, limits);

  require(a.has_best_move && b.has_best_move, "Gumbel search returns best moves");
  require(a.best_move.to_uci() == b.best_move.to_uci(),
          "deterministic mode returns same move");
  require(a.best_move.to_uci() == "e2e4", "high-prior deterministic move is selected");
}

void test_gumbel_candidates_are_legal_only() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  PreferredMoveEvaluator evaluator("e2e4");
  chessmoe::search::GumbelSearcher searcher(evaluator);
  const auto result = searcher.search(
      position, chessmoe::search::GumbelSearchLimits{
                    .simulations = 6,
                    .max_considered_actions = 6,
                    .deterministic = true,
                });
  const auto legal = chessmoe::chess::MoveGenerator::legal_moves(position);

  require(result.root_distribution.size() <= 6,
          "Gumbel root only stores considered candidates");
  for (const auto& stat : result.root_distribution) {
    require(chessmoe::chess::contains_uci(legal, stat.move.to_uci()),
            "Gumbel candidate must be legal");
  }
}

void test_gumbel_policy_target_is_normalized() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  PreferredMoveEvaluator evaluator("d2d4");
  chessmoe::search::GumbelSearcher searcher(evaluator);
  const auto result = searcher.search(
      position, chessmoe::search::GumbelSearchLimits{
                    .simulations = 12,
                    .max_considered_actions = 8,
                    .deterministic = true,
                });

  require_near(target_sum(result), 1.0, 1e-12,
               "Gumbel policy improvement target sums to one");
  for (const auto& stat : result.root_distribution) {
    require(stat.target_probability >= 0.0, "target probabilities are non-negative");
    require(stat.visit_count >= 0, "visit counts are replay-compatible");
  }
}

}  // namespace

int main() {
  try {
    test_search_mode_enum_keeps_puct_and_gumbel_available();
    test_gumbel_root_selection_is_deterministic_in_test_mode();
    test_gumbel_candidates_are_legal_only();
    test_gumbel_policy_target_is_normalized();
  } catch (const std::exception& e) {
    std::cerr << "gumbel_search_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
