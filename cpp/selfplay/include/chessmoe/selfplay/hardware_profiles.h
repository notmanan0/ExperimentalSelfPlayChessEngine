#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <chessmoe/selfplay/selfplay_app.h>

namespace chessmoe::selfplay {

struct HardwareProfile {
  std::string name;
  EvaluatorMode evaluator{EvaluatorMode::Material};
  int concurrent_games{1};
  std::size_t fixed_batch{64};
  std::size_t max_pending_requests{4096};
  int flush_ms{2};
  int visits{64};
  int max_plies{128};
  std::string precision{"fp32"};
  int cpu_workers{0};
  int replay_chunk_games{64};
  int progress_interval{10};
  std::string description;
};

[[nodiscard]] HardwareProfile resolve_hardware_profile(std::string_view name);
[[nodiscard]] std::vector<HardwareProfile> available_hardware_profiles();
[[nodiscard]] bool is_bootstrap_profile(std::string_view name);
[[nodiscard]] std::string evaluator_mode_name_for_profile(const HardwareProfile& profile);

}  // namespace chessmoe::selfplay
