#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace chessmoe::selfplay {

enum class ReanalysisStorageMode : std::uint8_t {
  AppendTargetRecords = 0,
  SeparateChunks = 1,
};

struct ReanalysisConfig {
  std::filesystem::path replay_index;
  std::filesystem::path output_index;
  std::vector<std::uint32_t> source_model_versions;
  std::optional<std::uint64_t> older_than_timestamp_ms{};
  double minimum_sampling_priority{0.0};
  std::uint32_t current_model_version{0};
  int search_budget{64};
  int max_chunks{0};
  ReanalysisStorageMode storage_mode{ReanalysisStorageMode::AppendTargetRecords};
};

void validate_reanalysis_config(const ReanalysisConfig& config);

std::string to_string(ReanalysisStorageMode mode);

}  // namespace chessmoe::selfplay
