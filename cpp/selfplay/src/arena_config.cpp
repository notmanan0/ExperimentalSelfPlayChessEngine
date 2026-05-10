#include <chessmoe/selfplay/arena_config.h>

#include <stdexcept>

namespace chessmoe::selfplay {

std::vector<ArenaGameSpec> build_match_schedule(const ArenaConfig& config) {
  if (config.openings.empty()) {
    throw std::runtime_error("arena requires at least one opening");
  }
  if (config.games_per_opening <= 0 || config.games_per_opening % 2 != 0) {
    throw std::runtime_error("games_per_opening must be positive and even");
  }
  if (config.search_budget <= 0) {
    throw std::runtime_error("search_budget must be positive");
  }

  std::vector<ArenaGameSpec> schedule;
  schedule.reserve(config.openings.size() *
                   static_cast<std::size_t>(config.games_per_opening));

  int game_id = 0;
  for (std::size_t opening_index = 0; opening_index < config.openings.size();
       ++opening_index) {
    for (int local_game = 0; local_game < config.games_per_opening;
         ++local_game) {
      ArenaGameSpec game;
      game.game_id = game_id++;
      game.opening_fen = config.openings[opening_index];
      game.candidate_color =
          local_game % 2 == 0 ? ArenaColor::White : ArenaColor::Black;
      game.search_budget = config.search_budget;
      game.seed = config.seed + static_cast<std::uint32_t>(opening_index * 1009) +
                  static_cast<std::uint32_t>(local_game * 9173);
      schedule.push_back(game);
    }
  }

  return schedule;
}

double score_rate(const ArenaSummary& summary) {
  const int games = summary.wins + summary.losses + summary.draws;
  if (games <= 0) {
    return 0.0;
  }
  return (static_cast<double>(summary.wins) +
          0.5 * static_cast<double>(summary.draws)) /
         static_cast<double>(games);
}

PromotionDecision evaluate_promotion(const ArenaSummary& summary,
                                     const PromotionRule& rule) {
  const int games = summary.wins + summary.losses + summary.draws;
  if (games < rule.minimum_games) {
    return PromotionDecision::InsufficientGames;
  }
  if (rule.sprt_enabled) {
    return PromotionDecision::SprtContinue;
  }
  return score_rate(summary) >= rule.minimum_score_rate
             ? PromotionDecision::Promoted
             : PromotionDecision::Rejected;
}

}  // namespace chessmoe::selfplay
