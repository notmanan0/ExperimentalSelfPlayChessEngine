#include <chessmoe/chess/fen.h>
#include <chessmoe/chess/move_generator.h>
#include <chessmoe/eval/evaluator.h>
#include <chessmoe/eval/pesto_evaluator.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
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

void test_starting_position_near_zero() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  chessmoe::eval::PestoEvaluator evaluator;
  const int cp = evaluator.evaluate_tapered(position, chessmoe::chess::Color::White);
  require(std::abs(cp) < 50, "starting position eval should be near zero");
}

void test_starting_position_symmetric() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  chessmoe::eval::PestoEvaluator evaluator;
  const int white_eval = evaluator.evaluate_tapered(position, chessmoe::chess::Color::White);
  const int black_eval = evaluator.evaluate_tapered(position, chessmoe::chess::Color::Black);
  require(white_eval == black_eval, "starting position should be symmetric");
}

void test_material_advantage_positive() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppp1ppp/8/8/8/4P3/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
  chessmoe::eval::PestoEvaluator evaluator;
  const int white_eval = evaluator.evaluate_tapered(position, chessmoe::chess::Color::White);
  require(white_eval > 0, "white has extra pawn, eval should be positive from white perspective");
}

void test_black_material_advantage() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1");
  chessmoe::eval::PestoEvaluator evaluator;
  const int white_eval = evaluator.evaluate_tapered(position, chessmoe::chess::Color::White);
  require(white_eval < 0, "black has extra pawn, eval should be negative from white perspective");
}

void test_queen_vs_pawn() {
  const auto position = chessmoe::chess::Fen::parse(
      "4k3/8/8/8/8/8/8/4K2Q w - - 0 1");
  chessmoe::eval::PestoEvaluator evaluator;
  const int cp = evaluator.evaluate_tapered(position, chessmoe::chess::Color::White);
  require(cp > 500, "queen advantage should be large");
}

void test_game_phase_opening() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  chessmoe::eval::PestoEvaluator evaluator;
  const int phase = evaluator.game_phase(position);
  require(phase > 16, "opening should have high game phase");
}

void test_game_phase_endgame() {
  const auto position = chessmoe::chess::Fen::parse(
      "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
  chessmoe::eval::PestoEvaluator evaluator;
  const int phase = evaluator.game_phase(position);
  require(phase <= 1, "bare kings should have low game phase");
}

void test_batch_evaluator_returns_valid_results() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  chessmoe::eval::PestoEvaluator evaluator;
  const std::array requests{
      chessmoe::eval::EvaluationRequest::from_position(position)};
  const auto results = evaluator.evaluate_batch(requests);

  require(results.size() == 1, "returns one result");
  const auto& result = results.front();
  require(result.policy.size() == 20, "start position has 20 legal policy entries");
  require_near(probability_sum(result), 1.0, 1e-12,
               "policy probabilities sum to one");
  require(result.value >= -1.0 && result.value <= 1.0,
          "value in [-1, 1] range");
  require(result.wdl.win >= 0.0 && result.wdl.draw >= 0.0 && result.wdl.loss >= 0.0,
          "WDL values are non-negative");
  require_near(result.wdl.win + result.wdl.draw + result.wdl.loss, 1.0, 1e-12,
               "WDL probabilities sum to one");
  require(result.moves_left.has_value(), "moves-left is present");
}

void test_batch_evaluator_is_deterministic() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  chessmoe::eval::PestoEvaluator evaluator;
  const std::array requests{
      chessmoe::eval::EvaluationRequest::from_position(position)};

  const auto a = evaluator.evaluate_batch(requests).front();
  const auto b = evaluator.evaluate_batch(requests).front();

  require_near(a.value, b.value, 1e-15, "deterministic value");
  require(a.policy.size() == b.policy.size(), "deterministic policy size");
  for (std::size_t i = 0; i < a.policy.size(); ++i) {
    require(a.policy[i].move.to_uci() == b.policy[i].move.to_uci(),
            "deterministic move ordering");
    require_near(a.policy[i].probability, b.policy[i].probability, 1e-15,
                 "deterministic probability");
  }
}

void test_tactical_position_has_policy() {
  const auto position = chessmoe::chess::Fen::parse(
      "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");
  chessmoe::eval::PestoEvaluator evaluator;
  const std::array requests{
      chessmoe::eval::EvaluationRequest::from_position(position)};
  const auto results = evaluator.evaluate_batch(requests);
  const auto& result = results.front();

  require(result.policy.size() > 0, "tactical position has legal moves");
  require(contains_policy_move(result, "e1g1"), "castling available");
  require_near(probability_sum(result), 1.0, 1e-12, "policy sums to one");
}

void test_value_to_wdl_mapping() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  chessmoe::eval::PestoEvaluator evaluator;
  const std::array requests{
      chessmoe::eval::EvaluationRequest::from_position(position)};
  const auto result = evaluator.evaluate_batch(requests).front();

  require_near(result.wdl.draw, 1.0, 0.3,
               "near-equal position should have high draw probability");
}

}  // namespace

int main() {
  try {
    test_starting_position_near_zero();
    test_starting_position_symmetric();
    test_material_advantage_positive();
    test_black_material_advantage();
    test_queen_vs_pawn();
    test_game_phase_opening();
    test_game_phase_endgame();
    test_batch_evaluator_returns_valid_results();
    test_batch_evaluator_is_deterministic();
    test_tactical_position_has_policy();
    test_value_to_wdl_mapping();
    std::cout << "All PeSTO evaluator tests passed.\n";
  } catch (const std::exception& e) {
    std::cerr << "pesto_evaluator_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
