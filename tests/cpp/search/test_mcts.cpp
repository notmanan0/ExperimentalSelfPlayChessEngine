#include <chessmoe/chess/fen.h>
#include <chessmoe/chess/move_generator.h>
#include <chessmoe/eval/evaluator.h>
#include <chessmoe/eval/material_evaluator.h>
#include <chessmoe/eval/random_evaluator.h>
#include <chessmoe/search/mcts_searcher.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void require_eq(std::uint64_t actual, std::uint64_t expected,
                std::string_view message) {
  if (actual != expected) {
    throw std::runtime_error(std::string(message) + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

const chessmoe::search::RootMoveStats* find_stat(
    const std::vector<chessmoe::search::RootMoveStats>& stats,
    std::string_view uci) {
  const auto it = std::find_if(stats.begin(), stats.end(), [&](const auto& stat) {
    return stat.move.to_uci() == uci;
  });
  return it == stats.end() ? nullptr : &*it;
}

void test_root_expansion_masks_to_legal_moves() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  chessmoe::eval::MaterialEvaluator batch;
  chessmoe::eval::SynchronousEvaluator evaluator(batch);
  chessmoe::search::MctsSearcher searcher(evaluator);
  const auto result =
      searcher.search(position, chessmoe::search::MctsLimits{.visits = 1});

  const auto legal = chessmoe::chess::MoveGenerator::legal_moves(position);
  require_eq(result.root_visits, 1, "one visit is recorded at root");
  require_eq(result.root_distribution.size(), legal.size(),
             "root distribution contains exactly legal moves");
  for (const auto& stat : result.root_distribution) {
    require(chessmoe::chess::contains_uci(legal, stat.move.to_uci()),
            "root child move must be legal");
  }
}

void test_visit_count_conservation() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  chessmoe::eval::MaterialEvaluator batch;
  chessmoe::eval::SynchronousEvaluator evaluator(batch);
  chessmoe::search::MctsSearcher searcher(evaluator);
  const auto result =
      searcher.search(position, chessmoe::search::MctsLimits{.visits = 32});

  const auto child_visits = std::accumulate(
      result.root_distribution.begin(), result.root_distribution.end(),
      std::uint64_t{0}, [](std::uint64_t sum, const auto& stat) {
        return sum + static_cast<std::uint64_t>(stat.visit_count);
      });

  require_eq(result.root_visits, 32, "root visit count equals requested visits");
  require_eq(child_visits, 32, "child visits sum to root visits");
}

void test_terminal_position_returns_nullmove() {
  const auto mate = chessmoe::chess::Fen::parse("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
  chessmoe::eval::MaterialEvaluator batch;
  chessmoe::eval::SynchronousEvaluator evaluator(batch);
  chessmoe::search::MctsSearcher searcher(evaluator);
  const auto result =
      searcher.search(mate, chessmoe::search::MctsLimits{.visits = 8});

  require(!result.has_best_move, "terminal root has no best move");
  require(result.terminal, "terminal root is reported");
  require_eq(result.root_visits, 1, "terminal root is counted once");
}

void test_depth_budget_limits_expansion_depth() {
  class CountingEvaluator final : public chessmoe::eval::ISinglePositionEvaluator {
   public:
    int calls{0};

    chessmoe::eval::EvaluationResult evaluate(
        const chessmoe::eval::EvaluationRequest& request) override {
      ++calls;
      chessmoe::eval::EvaluationResult result;
      result.value = 0.0;
      const double prior =
          request.legal_moves.empty() ? 0.0 : 1.0 / request.legal_moves.size();
      for (const auto move : request.legal_moves) {
        result.policy.push_back({move, 0.0, prior});
      }
      result.wdl = {0.0, 1.0, 0.0};
      return result;
    }
  };

  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  CountingEvaluator evaluator;
  chessmoe::search::MctsSearcher searcher(evaluator);
  const auto result = searcher.search(
      position, chessmoe::search::MctsLimits{.visits = 5, .max_depth = 1});

  require_eq(result.root_visits, 5, "depth-limited search still spends visits");
  require_eq(static_cast<std::uint64_t>(evaluator.calls), 6,
             "max_depth 1 evaluates root plus one leaf per visit");
}

void test_deterministic_output_with_seeded_random_evaluator() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  chessmoe::eval::RandomEvaluator batch_a(1234);
  chessmoe::eval::RandomEvaluator batch_b(1234);
  chessmoe::eval::SynchronousEvaluator evaluator_a(batch_a);
  chessmoe::eval::SynchronousEvaluator evaluator_b(batch_b);
  chessmoe::search::MctsSearcher searcher_a(evaluator_a);
  chessmoe::search::MctsSearcher searcher_b(evaluator_b);

  const auto limits = chessmoe::search::MctsLimits{.visits = 24};
  const auto a = searcher_a.search(position, limits);
  const auto b = searcher_b.search(position, limits);

  require(a.has_best_move && b.has_best_move, "both searches return a move");
  require(a.best_move.to_uci() == b.best_move.to_uci(),
          "seeded evaluator produces deterministic bestmove");
  require(a.root_distribution.size() == b.root_distribution.size(),
          "seeded evaluator produces same distribution size");
  for (std::size_t i = 0; i < a.root_distribution.size(); ++i) {
    require(a.root_distribution[i].move.to_uci() ==
                b.root_distribution[i].move.to_uci(),
            "root distribution move order is deterministic");
    require(a.root_distribution[i].visit_count ==
                b.root_distribution[i].visit_count,
            "root distribution visit counts are deterministic");
  }
}

void test_policy_prior_can_drive_bestmove() {
  class PreferredMoveEvaluator final : public chessmoe::eval::ISinglePositionEvaluator {
   public:
    chessmoe::eval::EvaluationResult evaluate(
        const chessmoe::eval::EvaluationRequest& request) override {
      chessmoe::eval::EvaluationResult result;
      result.value = 0.0;
      for (const auto move : request.legal_moves) {
        result.policy.push_back(
            {move, 0.0, move.to_uci() == "e2e4" ? 100.0 : 1.0});
      }
      result.wdl = {0.0, 1.0, 0.0};
      return result;
    }
  };

  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  PreferredMoveEvaluator evaluator;
  chessmoe::search::MctsSearcher searcher(evaluator);
  const auto result =
      searcher.search(position, chessmoe::search::MctsLimits{.visits = 16});

  require(result.has_best_move, "search returns a bestmove");
  require(result.best_move.to_uci() == "e2e4", "high prior legal move wins visits");
  const auto* e2e4 = find_stat(result.root_distribution, "e2e4");
  require(e2e4 != nullptr && e2e4->visit_count > 0,
          "root distribution includes preferred move visits");
}

}  // namespace

int main() {
  try {
    test_root_expansion_masks_to_legal_moves();
    test_visit_count_conservation();
    test_terminal_position_returns_nullmove();
    test_depth_budget_limits_expansion_depth();
    test_deterministic_output_with_seeded_random_evaluator();
    test_policy_prior_can_drive_bestmove();
  } catch (const std::exception& e) {
    std::cerr << "mcts_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
