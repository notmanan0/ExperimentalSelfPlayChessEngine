#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <chessmoe/selfplay/self_play_generator.h>

namespace chessmoe::selfplay {

struct QualityProfile {
  std::string name;
  int visits{64};
  int max_plies{128};
  bool root_dirichlet_noise{true};
  double root_dirichlet_alpha{0.3};
  double root_dirichlet_epsilon{0.25};
  double temperature_initial{1.0};
  double temperature_final{0.0};
  int temperature_cutoff_ply{30};
  int games{256};
  int arena_games{64};
  double promotion_threshold{0.55};
  bool resignation_enabled{false};
  std::string description;
};

[[nodiscard]] QualityProfile resolve_quality_profile(std::string_view name);
[[nodiscard]] std::vector<QualityProfile> available_quality_profiles();
[[nodiscard]] SelfPlayConfig apply_quality_profile(const QualityProfile& quality,
                                                    std::uint32_t seed);

}  // namespace chessmoe::selfplay
