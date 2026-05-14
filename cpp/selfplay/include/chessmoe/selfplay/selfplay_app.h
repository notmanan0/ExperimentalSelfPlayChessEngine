#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include <chessmoe/eval/evaluator.h>
#include <chessmoe/selfplay/gpu_selfplay_pipeline.h>

namespace chessmoe::selfplay {

enum class EvaluatorMode {
  Material,
  TensorRT,
  Onnx,
};

struct SelfPlayAppOptions {
  EvaluatorMode evaluator_mode{EvaluatorMode::Material};
  std::optional<std::filesystem::path> engine_path{};
  GpuSelfPlayPipelineConfig pipeline{};
  int progress_interval{0};
};

struct ProgressSnapshot {
  int completed_games{0};
  int total_games{0};
  std::uint64_t samples_written{0};
  std::uint64_t positions_evaluated{0};
  std::uint64_t batches_evaluated{0};
  std::uint64_t padded_positions{0};
  double elapsed_ms{0.0};
  double average_inference_latency_ms{0.0};
  int active_games{0};
  std::filesystem::path output_dir{};
  EvaluatorMode evaluator_mode{EvaluatorMode::Material};
  std::uint32_t model_version{0};
};

[[nodiscard]] SelfPlayAppOptions parse_selfplay_options(int argc, char** argv);
[[nodiscard]] std::unique_ptr<eval::IBatchEvaluator> create_batch_evaluator(
    const SelfPlayAppOptions& options);
[[nodiscard]] std::unique_ptr<eval::IBatchEvaluator> create_batch_evaluator_from_mode(
    EvaluatorMode mode,
    const std::optional<std::filesystem::path>& engine_path,
    std::size_t fixed_batch);
[[nodiscard]] std::string evaluator_mode_name(EvaluatorMode mode);
[[nodiscard]] std::string format_progress(const ProgressSnapshot& snapshot);

}  // namespace chessmoe::selfplay
