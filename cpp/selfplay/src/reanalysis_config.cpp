#include <chessmoe/selfplay/reanalysis_config.h>

#include <stdexcept>

namespace chessmoe::selfplay {

void validate_reanalysis_config(const ReanalysisConfig& config) {
  if (config.replay_index.empty()) {
    throw std::invalid_argument("replay index path is required");
  }
  if (config.output_index.empty()) {
    throw std::invalid_argument("output index path is required");
  }
  if (config.current_model_version == 0) {
    throw std::invalid_argument("current model version must be positive");
  }
  if (config.search_budget <= 0) {
    throw std::invalid_argument("search budget must be positive");
  }
  if (config.minimum_sampling_priority < 0.0) {
    throw std::invalid_argument("minimum sampling priority must not be negative");
  }
}

std::string to_string(ReanalysisStorageMode mode) {
  switch (mode) {
    case ReanalysisStorageMode::AppendTargetRecords:
      return "append_target_records";
    case ReanalysisStorageMode::SeparateChunks:
      return "separate_chunks";
  }
  return "append_target_records";
}

}  // namespace chessmoe::selfplay
