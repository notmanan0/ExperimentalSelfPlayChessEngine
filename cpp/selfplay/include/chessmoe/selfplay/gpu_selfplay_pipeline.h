#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

#include <chessmoe/eval/evaluator.h>
#include <chessmoe/inference/async_batching_evaluator.h>
#include <chessmoe/selfplay/replay_writer.h>
#include <chessmoe/selfplay/self_play_generator.h>

namespace chessmoe::selfplay {

struct GpuSelfPlayProgress {
  int completed_games{0};
  int total_games{0};
  std::uint64_t samples_written{0};
  std::uint64_t positions_evaluated{0};
  std::uint64_t batches_evaluated{0};
  std::uint64_t padded_positions{0};
  double elapsed_ms{0.0};
  double average_inference_latency_ms{0.0};
  int active_games{0};
};

struct GpuSelfPlayPipelineConfig {
  SelfPlayConfig game{};
  int total_games{1};
  int concurrent_games{1};
  std::size_t fixed_batch_size{64};
  std::size_t max_pending_requests{4096};
  std::chrono::milliseconds flush_timeout{2};
  bool write_replay{false};
  std::filesystem::path replay_output_dir{"data/replay"};
  ReplayChunkOptions replay_options{};
  std::optional<double> sampled_gpu_utilization_percent{};
  int progress_interval{0};
  std::function<void(const GpuSelfPlayProgress&)> progress_callback{};
};

struct GpuSelfPlayMetrics {
  std::uint64_t games_completed{0};
  std::uint64_t samples_written{0};
  std::uint64_t positions_evaluated{0};
  std::uint64_t batches_evaluated{0};
  std::uint64_t padded_positions{0};
  std::uint64_t max_queue_depth{0};
  double elapsed_ms{0.0};
  double positions_per_second{0.0};
  double games_per_hour{0.0};
  double gpu_utilization_percent{0.0};
  double average_inference_latency_ms{0.0};
  std::vector<std::uint64_t> batch_size_histogram;
  std::vector<std::uint64_t> valid_batch_size_histogram;
};

struct GpuSelfPlayRunResult {
  std::vector<SelfPlayGame> games;
  std::vector<std::filesystem::path> replay_paths;
  GpuSelfPlayMetrics metrics{};
};

class GpuSelfPlayPipeline {
 public:
  explicit GpuSelfPlayPipeline(eval::IBatchEvaluator& evaluator);

  GpuSelfPlayRunResult run(const GpuSelfPlayPipelineConfig& config);

 private:
  eval::IBatchEvaluator& evaluator_;
};

}  // namespace chessmoe::selfplay
