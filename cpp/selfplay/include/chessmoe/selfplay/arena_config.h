#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace chessmoe::selfplay {

enum class ArenaColor : std::uint8_t {
  White = 0,
  Black = 1,
};

enum class PromotionDecision : std::uint8_t {
  Promoted = 0,
  Rejected = 1,
  InsufficientGames = 2,
  SprtContinue = 3,
};

struct PromotionRule {
  int minimum_games{32};
  double minimum_score_rate{0.55};
  bool sprt_enabled{false};
};

struct ArenaConfig {
  std::string candidate_model;
  std::string best_model;
  std::vector<std::string> openings;
  int games_per_opening{2};
  int search_budget{64};
  std::uint32_t seed{1};
  PromotionRule promotion{};
};

struct ArenaGameSpec {
  int game_id{0};
  std::string opening_fen;
  ArenaColor candidate_color{ArenaColor::White};
  int search_budget{0};
  std::uint32_t seed{0};
};

struct ArenaSummary {
  int wins{0};
  int losses{0};
  int draws{0};
};

std::vector<ArenaGameSpec> build_match_schedule(const ArenaConfig& config);

double score_rate(const ArenaSummary& summary);

PromotionDecision evaluate_promotion(const ArenaSummary& summary,
                                     const PromotionRule& rule);

}  // namespace chessmoe::selfplay
