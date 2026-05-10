#include <chessmoe/chess/fen.h>
#include <chessmoe/chess/move_generator.h>
#include <chessmoe/eval/evaluator.h>
#include <chessmoe/eval/material_evaluator.h>
#include <chessmoe/eval/random_evaluator.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
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

void require_near(double actual, double expected, double tolerance,
                  std::string_view message) {
  if (std::fabs(actual - expected) > tolerance) {
    throw std::runtime_error(std::string(message) + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

double probability_sum(const chessmoe::eval::EvaluationResult& result) {
  double sum = 0.0;
  for (const auto& entry : result.policy) {
    sum += entry.probability;
  }
  return sum;
}

bool contains_policy_move(const chessmoe::eval::EvaluationResult& result,
                          std::string_view uci) {
  for (const auto& entry : result.policy) {
    if (entry.move.to_uci() == uci) {
      return true;
    }
  }
  return false;
}

void test_request_captures_position_metadata() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  const auto request = chessmoe::eval::EvaluationRequest::from_position(position);

  require(request.hash == position.hash(), "request stores board hash");
  require(request.side_to_move == position.side_to_move(),
          "request stores side to move");
  require(request.legal_moves.size() ==
              chessmoe::chess::MoveGenerator::legal_moves(position).size(),
          "request stores legal moves");
}

void test_material_evaluator_masks_and_normalizes_legal_policy() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  chessmoe::eval::MaterialEvaluator evaluator;
  const std::array requests{
      chessmoe::eval::EvaluationRequest::from_position(position)};

  const auto results = evaluator.evaluate_batch(requests);
  require(results.size() == 1, "batch material evaluator returns one result");

  const auto& result = results.front();
  require(result.policy.size() == 20, "start position has 20 legal policy entries");
  require_near(probability_sum(result), 1.0, 1e-12,
               "material policy probabilities sum to one");
  require(!contains_policy_move(result, "e2e5"), "illegal move is masked out");
  require(result.wdl.win >= 0.0 && result.wdl.draw >= 0.0 && result.wdl.loss >= 0.0,
          "WDL values are probabilities");
  require_near(result.wdl.win + result.wdl.draw + result.wdl.loss, 1.0, 1e-12,
               "WDL probabilities sum to one");
  require(result.moves_left.has_value(), "moves-left placeholder is present");
}

void test_random_evaluator_is_seed_deterministic() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  const auto request = chessmoe::eval::EvaluationRequest::from_position(position);
  const std::array requests{request};
  chessmoe::eval::RandomEvaluator left(123);
  chessmoe::eval::RandomEvaluator right(123);

  const auto a = left.evaluate_batch(requests).front();
  const auto b = right.evaluate_batch(requests).front();

  require(a.policy.size() == b.policy.size(), "deterministic random policy size");
  require_near(a.value, b.value, 1e-15, "deterministic random value");
  require_near(a.wdl.win, b.wdl.win, 1e-15, "deterministic random WDL win");
  for (std::size_t i = 0; i < a.policy.size(); ++i) {
    require(a.policy[i].move.to_uci() == b.policy[i].move.to_uci(),
            "deterministic random move mapping");
    require_near(a.policy[i].probability, b.policy[i].probability, 1e-15,
                 "deterministic random probability");
  }
  require_near(probability_sum(a), 1.0, 1e-12,
               "random policy probabilities sum to one");
}

void test_single_position_wrapper_uses_batch_evaluator() {
  class CountingBatchEvaluator final : public chessmoe::eval::IBatchEvaluator {
   public:
    int calls{0};

    std::vector<chessmoe::eval::EvaluationResult> evaluate_batch(
        std::span<const chessmoe::eval::EvaluationRequest> requests) override {
      ++calls;
      std::vector<chessmoe::eval::EvaluationResult> results;
      for (const auto& request : requests) {
        chessmoe::eval::EvaluationResult result;
        const double prior =
            request.legal_moves.empty() ? 0.0 : 1.0 / request.legal_moves.size();
        for (const auto move : request.legal_moves) {
          result.policy.push_back({move, 0.0, prior});
        }
        result.wdl = {0.25, 0.5, 0.25};
        result.value = 0.0;
        result.moves_left = 40.0;
        results.push_back(result);
      }
      return results;
    }
  };

  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  CountingBatchEvaluator batch;
  chessmoe::eval::SynchronousEvaluator single(batch);

  const auto result =
      single.evaluate(chessmoe::eval::EvaluationRequest::from_position(position));
  require(batch.calls == 1, "single wrapper calls batch evaluator once");
  require(result.policy.size() == 20, "single wrapper returns batch result");
}

}  // namespace

int main() {
  try {
    test_request_captures_position_metadata();
    test_material_evaluator_masks_and_normalizes_legal_policy();
    test_random_evaluator_is_seed_deterministic();
    test_single_position_wrapper_uses_batch_evaluator();
  } catch (const std::exception& e) {
    std::cerr << "eval_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
