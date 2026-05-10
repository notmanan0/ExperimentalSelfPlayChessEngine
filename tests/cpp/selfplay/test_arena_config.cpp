#include <chessmoe/selfplay/arena_config.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void test_cpp_arena_schedule_side_swaps() {
  chessmoe::selfplay::ArenaConfig config;
  config.openings = {
      "8/8/8/8/8/8/4K3/4k3 w - - 0 1",
      "8/8/8/8/8/8/4K3/4k3 b - - 0 1",
  };
  config.games_per_opening = 2;
  config.search_budget = 32;
  config.seed = 9;

  const auto schedule = chessmoe::selfplay::build_match_schedule(config);

  require(schedule.size() == 4, "schedule contains one side swap per opening");
  require(schedule[0].candidate_color == chessmoe::selfplay::ArenaColor::White,
          "candidate starts as white");
  require(schedule[1].candidate_color == chessmoe::selfplay::ArenaColor::Black,
          "candidate side swaps");
  require(schedule[0].search_budget == 32, "search budget is copied");
  require(schedule[0].seed != schedule[1].seed, "per-game seeds differ");
}

void test_cpp_promotion_threshold() {
  chessmoe::selfplay::ArenaSummary summary;
  summary.wins = 4;
  summary.losses = 1;
  summary.draws = 1;

  chessmoe::selfplay::PromotionRule rule;
  rule.minimum_games = 6;
  rule.minimum_score_rate = 0.60;

  const auto decision = chessmoe::selfplay::evaluate_promotion(summary, rule);

  require(decision == chessmoe::selfplay::PromotionDecision::Promoted,
          "candidate meeting threshold is promoted");
}

}  // namespace

int main() {
  try {
    test_cpp_arena_schedule_side_swaps();
    test_cpp_promotion_threshold();
  } catch (const std::exception& e) {
    std::cerr << "arena_config_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
