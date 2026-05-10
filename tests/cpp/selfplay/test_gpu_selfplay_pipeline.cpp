#include <chessmoe/chess/fen.h>
#include <chessmoe/eval/evaluator.h>
#include <chessmoe/inference/async_batching_evaluator.h>
#include <chessmoe/selfplay/gpu_selfplay_pipeline.h>
#include <chessmoe/selfplay/replay_writer.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

class RecordingBatchEvaluator final : public chessmoe::eval::IBatchEvaluator {
 public:
  std::vector<chessmoe::eval::EvaluationResult> evaluate_batch(
      std::span<const chessmoe::eval::EvaluationRequest> requests) override {
    {
      std::lock_guard lock(mutex_);
      batch_sizes_.push_back(requests.size());
    }

    std::vector<chessmoe::eval::EvaluationResult> results;
    results.reserve(requests.size());
    for (const auto& request : requests) {
      chessmoe::eval::EvaluationResult result;
      result.value = 0.0;
      result.wdl = {0.0, 1.0, 0.0};
      result.policy.reserve(request.legal_moves.size());
      for (const auto move : request.legal_moves) {
        result.policy.push_back({move, 0.0, 0.0});
      }
      results.push_back(chessmoe::eval::normalize_policy_over_legal_moves(
          std::move(result), request.legal_moves));
    }
    return results;
  }

  std::vector<std::size_t> batch_sizes() const {
    std::lock_guard lock(mutex_);
    return batch_sizes_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<std::size_t> batch_sizes_;
};

chessmoe::eval::EvaluationRequest start_request() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  return chessmoe::eval::EvaluationRequest::from_position(position);
}

chessmoe::selfplay::GpuSelfPlayPipelineConfig tiny_pipeline_config(
    const std::filesystem::path& output_dir) {
  chessmoe::selfplay::GpuSelfPlayPipelineConfig config;
  config.total_games = 2;
  config.concurrent_games = 2;
  config.fixed_batch_size = 4;
  config.max_pending_requests = 16;
  config.flush_timeout = std::chrono::milliseconds(5);
  config.write_replay = false;
  config.replay_output_dir = output_dir;
  config.game.max_plies = 1;
  config.game.search_visits = 1;
  config.game.deterministic = true;
  config.game.seed = 123;
  config.game.add_root_dirichlet_noise = false;
  config.game.temperature.initial = 0.0;
  config.game.temperature.final = 0.0;
  config.replay_options.model_version = 7;
  config.replay_options.generator_version = 13;
  config.replay_options.creation_timestamp_ms = 1'715'000'000'000ULL;
  return config;
}

std::vector<std::string> selected_moves(
    const chessmoe::selfplay::GpuSelfPlayRunResult& result) {
  std::vector<std::string> moves;
  for (const auto& game : result.games) {
    for (const auto& sample : game.samples) {
      moves.push_back(sample.selected_move.to_uci());
    }
  }
  return moves;
}

void test_async_evaluator_flushes_fixed_padded_selfplay_batch() {
  RecordingBatchEvaluator backend;
  chessmoe::inference::AsyncBatchingEvaluator evaluator(
      backend, chessmoe::inference::AsyncBatchingEvaluatorConfig{
                   4,
                   8,
                   std::chrono::milliseconds(5),
                   true,
               });

  const auto request = start_request();
  std::vector<std::future<chessmoe::eval::EvaluationResult>> futures;
  for (int i = 0; i < 3; ++i) {
    futures.push_back(std::async(std::launch::async, [&] {
      return evaluator.evaluate(request);
    }));
  }

  for (auto& future : futures) {
    require(!future.get().policy.empty(), "async evaluator returns a policy");
  }

  const auto batches = backend.batch_sizes();
  require(batches.size() == 1, "worker flushes the partial request group once");
  require(batches.front() == 4, "worker pads self-play inference to fixed batch");

  const auto metrics = evaluator.metrics_snapshot();
  require(metrics.positions_evaluated == 3,
          "metrics count only real positions");
  require(metrics.padded_positions == 1, "metrics count padded positions");
  require(metrics.batch_size_histogram.at(4) == 1,
          "batch histogram records padded batch size");
}

void test_pipeline_completes_multiple_concurrent_games() {
  RecordingBatchEvaluator backend;
  chessmoe::selfplay::GpuSelfPlayPipeline pipeline(backend);
  const auto result = pipeline.run(
      tiny_pipeline_config(std::filesystem::temp_directory_path()));

  require(result.games.size() == 2, "pipeline completes requested game count");
  require(result.metrics.games_completed == 2,
          "metrics count completed games");
  require(result.metrics.positions_evaluated >= 2,
          "pipeline evaluates at least one position per game");
  for (const auto& game : result.games) {
    require(game.result != chessmoe::selfplay::GameResult::Unknown,
            "completed game has final result");
    require(game.samples.size() == 1, "tiny self-play game records one sample");
  }
}

void test_pipeline_small_run_is_deterministic() {
  RecordingBatchEvaluator backend_a;
  RecordingBatchEvaluator backend_b;
  chessmoe::selfplay::GpuSelfPlayPipeline pipeline_a(backend_a);
  chessmoe::selfplay::GpuSelfPlayPipeline pipeline_b(backend_b);
  const auto config = tiny_pipeline_config(std::filesystem::temp_directory_path());

  const auto first = pipeline_a.run(config);
  const auto second = pipeline_b.run(config);

  require(selected_moves(first) == selected_moves(second),
          "same seed and evaluator produce same move sequence");
  require(first.metrics.games_completed == second.metrics.games_completed,
          "same seed produces same game count");
}

void test_pipeline_writes_valid_replay_chunks() {
  RecordingBatchEvaluator backend;
  chessmoe::selfplay::GpuSelfPlayPipeline pipeline(backend);

  const auto output_dir =
      std::filesystem::temp_directory_path() / "chessmoe_phase13_replay_test";
  std::filesystem::remove_all(output_dir);
  auto config = tiny_pipeline_config(output_dir);
  config.total_games = 1;
  config.concurrent_games = 1;
  config.write_replay = true;

  const auto result = pipeline.run(config);

  require(result.replay_paths.size() == 1, "pipeline reports replay path");
  require(std::filesystem::exists(result.replay_paths.front()),
          "replay chunk exists on disk");

  {
    std::ifstream input(result.replay_paths.front(), std::ios::binary);
    const std::vector<std::uint8_t> bytes{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    require(bytes.size() > chessmoe::selfplay::kReplayHeaderSize,
            "replay chunk includes payload");
    require(std::string(bytes.begin(), bytes.begin() + 8) == "CMREPLAY",
            "replay chunk magic is valid");
  }

  std::filesystem::remove_all(output_dir);
}

}  // namespace

int main() {
  try {
    test_async_evaluator_flushes_fixed_padded_selfplay_batch();
    test_pipeline_completes_multiple_concurrent_games();
    test_pipeline_small_run_is_deterministic();
    test_pipeline_writes_valid_replay_chunks();
  } catch (const std::exception& e) {
    std::cerr << "gpu_selfplay_pipeline_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
