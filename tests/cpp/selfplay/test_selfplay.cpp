#include <chessmoe/chess/fen.h>
#include <chessmoe/chess/move_generator.h>
#include <chessmoe/eval/material_evaluator.h>
#include <chessmoe/selfplay/self_play_generator.h>

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

chessmoe::selfplay::SelfPlayConfig short_config() {
  chessmoe::selfplay::SelfPlayConfig config;
  config.max_plies = 4;
  config.search_visits = 8;
  config.model_version = "test-model";
  config.deterministic = true;
  config.seed = 99;
  config.temperature.initial = 0.0;
  config.temperature.final = 0.0;
  config.add_root_dirichlet_noise = true;
  config.root_dirichlet_epsilon = 0.25;
  config.root_dirichlet_alpha = 0.3;
  return config;
}

void test_generates_samples_with_legal_selected_moves() {
  chessmoe::eval::MaterialEvaluator batch;
  chessmoe::eval::SynchronousEvaluator evaluator(batch);
  chessmoe::selfplay::SelfPlayGenerator generator(evaluator);

  const auto game = generator.generate(short_config());

  require(!game.samples.empty(), "self-play emits samples");
  require(game.samples.size() <= 4, "sample count respects max plies");
  for (const auto& sample : game.samples) {
    const auto position = chessmoe::chess::Fen::parse(sample.board_fen);
    const auto legal = chessmoe::chess::MoveGenerator::legal_moves(position);
    require(chessmoe::chess::contains_uci(legal, sample.selected_move.to_uci()),
            "selected move is legal for sample position");
    require(sample.legal_moves.size() == legal.size(),
            "sample stores legal move list");
    require(!sample.visit_distribution.empty(),
            "sample stores MCTS visit distribution");
    const auto visits = std::accumulate(
        sample.visit_distribution.begin(), sample.visit_distribution.end(),
        std::uint64_t{0}, [](std::uint64_t sum, const auto& entry) {
          return sum + static_cast<std::uint64_t>(entry.visit_count);
        });
    require(visits == sample.search_budget,
            "visit distribution sums to search budget");
  }
}

void test_final_result_is_propagated_to_all_samples() {
  chessmoe::eval::MaterialEvaluator batch;
  chessmoe::eval::SynchronousEvaluator evaluator(batch);
  chessmoe::selfplay::SelfPlayGenerator generator(evaluator);
  auto config = short_config();
  config.max_plies = 2;

  const auto game = generator.generate(config);

  require(game.result == chessmoe::selfplay::GameResult::Draw,
          "short max-plies game is a draw");
  require(!game.samples.empty(), "draw game has samples");
  for (const auto& sample : game.samples) {
    require(sample.final_result == game.result,
            "sample final result is filled after game end");
  }
}

void test_generation_is_deterministic_with_seed() {
  chessmoe::eval::MaterialEvaluator batch_a;
  chessmoe::eval::MaterialEvaluator batch_b;
  chessmoe::eval::SynchronousEvaluator evaluator_a(batch_a);
  chessmoe::eval::SynchronousEvaluator evaluator_b(batch_b);
  chessmoe::selfplay::SelfPlayGenerator generator_a(evaluator_a);
  chessmoe::selfplay::SelfPlayGenerator generator_b(evaluator_b);

  const auto config = short_config();
  const auto a = generator_a.generate(config);
  const auto b = generator_b.generate(config);

  require(a.samples.size() == b.samples.size(), "deterministic sample count");
  require(a.result == b.result, "deterministic final result");
  for (std::size_t i = 0; i < a.samples.size(); ++i) {
    require(a.samples[i].board_fen == b.samples[i].board_fen,
            "deterministic board sequence");
    require(a.samples[i].selected_move.to_uci() ==
                b.samples[i].selected_move.to_uci(),
            "deterministic selected moves");
    require(a.samples[i].visit_distribution.size() ==
                b.samples[i].visit_distribution.size(),
            "deterministic distribution sizes");
    for (std::size_t j = 0; j < a.samples[i].visit_distribution.size(); ++j) {
      require(a.samples[i].visit_distribution[j].move.to_uci() ==
                  b.samples[i].visit_distribution[j].move.to_uci(),
              "deterministic distribution move order");
      require(a.samples[i].visit_distribution[j].visit_count ==
                  b.samples[i].visit_distribution[j].visit_count,
              "deterministic distribution visits");
    }
  }
}

void test_terminal_opening_fen_returns_checkmate_result_without_samples() {
  chessmoe::eval::MaterialEvaluator batch;
  chessmoe::eval::SynchronousEvaluator evaluator(batch);
  chessmoe::selfplay::SelfPlayGenerator generator(evaluator);
  auto config = short_config();
  config.opening_fen = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1";

  const auto game = generator.generate(config);

  require(game.samples.empty(), "terminal opening has no samples");
  require(game.result == chessmoe::selfplay::GameResult::WhiteWin,
          "checkmated black is a white win");
  require(game.terminal_reason == chessmoe::selfplay::TerminalReason::Checkmate,
          "terminal reason is checkmate");
}

}  // namespace

int main() {
  try {
    test_generates_samples_with_legal_selected_moves();
    test_final_result_is_propagated_to_all_samples();
    test_generation_is_deterministic_with_seed();
    test_terminal_opening_fen_returns_checkmate_result_without_samples();
  } catch (const std::exception& e) {
    std::cerr << "selfplay_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
