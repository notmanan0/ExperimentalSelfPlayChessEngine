#include <chessmoe/eval/material_evaluator.h>
#include <chessmoe/selfplay/gpu_selfplay_pipeline.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

int read_int_arg(int argc, char** argv, std::string_view name, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string_view(argv[i]) == name) {
      return std::stoi(argv[i + 1]);
    }
  }
  return fallback;
}

double read_double_arg(int argc, char** argv, std::string_view name,
                       double fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string_view(argv[i]) == name) {
      return std::stod(argv[i + 1]);
    }
  }
  return fallback;
}

std::filesystem::path read_path_arg(int argc, char** argv, std::string_view name,
                                    std::filesystem::path fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string_view(argv[i]) == name) {
      return argv[i + 1];
    }
  }
  return fallback;
}

bool has_flag(int argc, char** argv, std::string_view name) {
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == name) {
      return true;
    }
  }
  return false;
}

void print_histogram(const std::vector<std::uint64_t>& histogram) {
  std::cout << "{";
  bool first = true;
  for (std::size_t i = 0; i < histogram.size(); ++i) {
    if (histogram[i] == 0) {
      continue;
    }
    if (!first) {
      std::cout << ",";
    }
    first = false;
    std::cout << "\"" << i << "\":" << histogram[i];
  }
  std::cout << "}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    chessmoe::eval::MaterialEvaluator evaluator;
    chessmoe::selfplay::GpuSelfPlayPipeline pipeline(evaluator);

    chessmoe::selfplay::GpuSelfPlayPipelineConfig config;
    config.total_games = read_int_arg(argc, argv, "--games", 8);
    config.concurrent_games =
        read_int_arg(argc, argv, "--concurrent-games", 8);
    config.fixed_batch_size =
        static_cast<std::size_t>(read_int_arg(argc, argv, "--fixed-batch", 64));
    config.max_pending_requests = static_cast<std::size_t>(
        read_int_arg(argc, argv, "--max-pending-requests", 4096));
    config.flush_timeout = std::chrono::milliseconds(
        read_int_arg(argc, argv, "--flush-ms", 2));
    config.write_replay = has_flag(argc, argv, "--write-replay");
    config.replay_output_dir =
        read_path_arg(argc, argv, "--output-dir", "data/replay/phase13");
    config.sampled_gpu_utilization_percent =
        read_double_arg(argc, argv, "--gpu-utilization-percent", 0.0);
    config.game.max_plies = read_int_arg(argc, argv, "--max-plies", 32);
    config.game.search_visits = read_int_arg(argc, argv, "--visits", 32);
    config.game.deterministic = has_flag(argc, argv, "--deterministic");
    config.game.add_root_dirichlet_noise =
        !has_flag(argc, argv, "--disable-root-noise");
    config.game.temperature.initial = 1.0;
    config.game.temperature.final = 0.0;
    config.game.temperature.cutoff_ply = 30;
    config.replay_options.model_version =
        static_cast<std::uint32_t>(read_int_arg(argc, argv, "--model-version", 0));
    config.replay_options.generator_version = 13;

    const auto result = pipeline.run(config);
    const auto& metrics = result.metrics;

    std::cout << "{";
    std::cout << "\"games_completed\":" << metrics.games_completed << ",";
    std::cout << "\"positions_evaluated\":" << metrics.positions_evaluated << ",";
    std::cout << "\"positions_per_second\":" << metrics.positions_per_second << ",";
    std::cout << "\"games_per_hour\":" << metrics.games_per_hour << ",";
    std::cout << "\"gpu_utilization_percent\":"
              << metrics.gpu_utilization_percent << ",";
    std::cout << "\"batches_evaluated\":" << metrics.batches_evaluated << ",";
    std::cout << "\"padded_positions\":" << metrics.padded_positions << ",";
    std::cout << "\"max_queue_depth\":" << metrics.max_queue_depth << ",";
    std::cout << "\"average_inference_latency_ms\":"
              << metrics.average_inference_latency_ms << ",";
    std::cout << "\"batch_size_histogram\":";
    print_histogram(metrics.batch_size_histogram);
    std::cout << ",\"valid_batch_size_histogram\":";
    print_histogram(metrics.valid_batch_size_histogram);
    std::cout << "}\n";
  } catch (const std::exception& e) {
    std::cerr << "gpu_selfplay_benchmark failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
