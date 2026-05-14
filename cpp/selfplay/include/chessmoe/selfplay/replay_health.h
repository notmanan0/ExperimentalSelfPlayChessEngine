#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace chessmoe::selfplay {

struct ReplayHealthConfig {
  int min_samples{100};
  double min_average_plies{20.0};
  double max_draw_rate{0.95};
  double max_max_plies_rate{0.80};
  double min_checkmate_rate{0.01};
  double max_checkmate_rate{0.90};
  std::uint64_t min_file_bytes{64};
};

struct ReplayHealthReport {
  int total_samples{0};
  int total_games{0};
  double average_plies{0.0};
  double draw_rate{0.0};
  double checkmate_rate{0.0};
  double stalemate_rate{0.0};
  double repetition_rate{0.0};
  double fifty_move_rate{0.0};
  double max_plies_rate{0.0};
  std::vector<std::string> warnings;
  std::vector<std::string> errors;
  bool passed{true};
};

[[nodiscard]] ReplayHealthReport analyze_replay_health(
    const std::vector<std::filesystem::path>& paths,
    const ReplayHealthConfig& config = {});

void write_replay_health_report(const ReplayHealthReport& report,
                                const std::filesystem::path& path);
void print_replay_health_summary(const ReplayHealthReport& report);

}  // namespace chessmoe::selfplay
