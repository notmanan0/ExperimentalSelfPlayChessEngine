#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <chessmoe/eval/evaluator.h>
#include <chessmoe/selfplay/gpu_selfplay_pipeline.h>
#include <chessmoe/selfplay/replay_health.h>
#include <chessmoe/selfplay/config_resolver.h>

namespace chessmoe::selfplay {

struct GenerationResult {
  std::vector<std::filesystem::path> replay_paths;
  GpuSelfPlayMetrics metrics;
  ReplayHealthReport health;
  std::filesystem::path run_dir;
  bool completed{false};
  std::vector<double> profile_timings;
  std::map<std::string, double> profile_breakdown;
};

class GenerationController {
 public:
  explicit GenerationController(eval::IBatchEvaluator& evaluator);

  GenerationResult run(const ResolvedConfig& config);

 private:
  eval::IBatchEvaluator& evaluator_;

  void setup_run_directory(const ResolvedConfig& config);
  std::set<int> detect_completed_games(const std::filesystem::path& replay_dir);
  void write_run_metadata(const ResolvedConfig& config,
                          const GenerationResult& result);
};

}  // namespace chessmoe::selfplay
