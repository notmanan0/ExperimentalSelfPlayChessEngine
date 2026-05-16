#include <chessmoe/selfplay/gpu_selfplay_pipeline.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <chessmoe/selfplay/game_worker.h>

namespace chessmoe::selfplay {

namespace {

void validate_config(const GpuSelfPlayPipelineConfig& config) {
  if (config.total_games < 0) {
    throw std::invalid_argument("total games must not be negative");
  }
  if (config.concurrent_games <= 0) {
    throw std::invalid_argument("concurrent games must be positive");
  }
  if (config.fixed_batch_size == 0) {
    throw std::invalid_argument("fixed batch size must be positive");
  }
  if (config.max_pending_requests == 0) {
    throw std::invalid_argument("max pending requests must be positive");
  }
}

std::filesystem::path replay_path_for_game(const std::filesystem::path& output_dir,
                                           int game_id) {
  std::ostringstream name;
  name << "selfplay_game_" << std::setw(8) << std::setfill('0') << game_id
       << ".cmrep";
  return output_dir / name.str();
}

GpuSelfPlayMetrics make_metrics(
    int games_completed,
    std::uint64_t samples_written,
    double elapsed_ms,
    const inference::AsyncBatchingMetrics& inference_metrics,
    std::optional<double> sampled_gpu_utilization_percent,
    std::uint64_t mcts_legal_move_generation_calls,
    double mcts_legal_move_generation_ms) {
  GpuSelfPlayMetrics metrics;
  metrics.games_completed = static_cast<std::uint64_t>(games_completed);
  metrics.samples_written = samples_written;
  metrics.positions_evaluated = inference_metrics.positions_evaluated;
  metrics.batches_evaluated = inference_metrics.batches_evaluated;
  metrics.padded_positions = inference_metrics.padded_positions;
  metrics.max_queue_depth = inference_metrics.max_queue_depth;
  metrics.elapsed_ms = elapsed_ms;
  metrics.average_inference_latency_ms =
      inference_metrics.average_inference_latency_ms();
  metrics.batch_size_histogram = inference_metrics.batch_size_histogram;
  metrics.valid_batch_size_histogram =
      inference_metrics.valid_batch_size_histogram;
  metrics.gpu_utilization_percent =
      sampled_gpu_utilization_percent.value_or(0.0);
  metrics.mcts_legal_move_generation_calls =
      mcts_legal_move_generation_calls;
  metrics.mcts_legal_move_generation_ms = mcts_legal_move_generation_ms;

  const double elapsed_seconds = elapsed_ms / 1000.0;
  if (elapsed_seconds > 0.0) {
    metrics.positions_per_second =
        static_cast<double>(metrics.positions_evaluated) / elapsed_seconds;
    metrics.games_per_hour =
        static_cast<double>(metrics.games_completed) * 3600.0 / elapsed_seconds;
  }

  return metrics;
}

}  // namespace

GpuSelfPlayPipeline::GpuSelfPlayPipeline(eval::IBatchEvaluator& evaluator)
    : evaluator_(evaluator) {}

GpuSelfPlayRunResult GpuSelfPlayPipeline::run(
    const GpuSelfPlayPipelineConfig& config) {
  validate_config(config);

  GpuSelfPlayRunResult result;
  result.games.resize(static_cast<std::size_t>(config.total_games));
  std::vector<std::filesystem::path> replay_paths(
      static_cast<std::size_t>(config.total_games));

  if (config.write_replay) {
    std::filesystem::create_directories(config.replay_output_dir);
  }

  inference::AsyncBatchingEvaluator async_evaluator(
      evaluator_, inference::AsyncBatchingEvaluatorConfig{
                      config.fixed_batch_size,
                      config.max_pending_requests,
                      config.flush_timeout,
                      true,
                  });

  std::atomic<int> next_game{0};
  std::atomic<int> completed_games{0};
  std::atomic<std::uint64_t> samples_written{0};
  std::mutex exception_mutex;
  std::mutex profile_mutex;
  std::exception_ptr first_exception;
  std::uint64_t mcts_legal_move_generation_calls = 0;
  double mcts_legal_move_generation_ms = 0.0;
  const auto started = std::chrono::steady_clock::now();

  const int thread_count =
      std::min(config.concurrent_games, std::max(1, config.total_games));
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(thread_count));

  for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
    workers.emplace_back([&, thread_index] {
      try {
        GameWorkerConfig worker_config;
        worker_config.game = config.game;
        worker_config.worker_id = static_cast<std::uint32_t>(thread_index);
        worker_config.openings.fen_pool = config.opening_fen_pool;
        worker_config.openings.deterministic_selection = true;
        worker_config.openings.color_balancing = true;
        GameWorker worker(async_evaluator, worker_config);
        for (;;) {
          const int game_id = next_game.fetch_add(1);
          if (game_id >= config.total_games) {
            break;
          }

          auto worker_result = worker.run(game_id);
          auto game = std::move(worker_result.game);
          const auto game_samples =
              static_cast<std::uint64_t>(game.samples.size());
          {
            std::lock_guard lock(profile_mutex);
            mcts_legal_move_generation_calls +=
                worker_result.diagnostics.mcts_legal_move_generation_calls;
            mcts_legal_move_generation_ms +=
                worker_result.diagnostics.mcts_legal_move_generation_ms;
          }

          if (config.write_replay) {
            auto options = config.replay_options;
            options.game_id = static_cast<std::uint64_t>(game_id);
            options.starting_ply_index = 0;
            const auto path =
                replay_path_for_game(config.replay_output_dir, game_id);
            write_replay_chunk(path, game, options);
            replay_paths[static_cast<std::size_t>(game_id)] = path;
          }

          result.games[static_cast<std::size_t>(game_id)] = std::move(game);
          const auto now_samples =
              samples_written.fetch_add(game_samples) + game_samples;
          const int now_completed = completed_games.fetch_add(1) + 1;
          if (config.progress_callback && config.progress_interval > 0 &&
              (now_completed == config.total_games ||
               now_completed % config.progress_interval == 0)) {
            const auto current = std::chrono::steady_clock::now();
            const auto inference_metrics = async_evaluator.metrics_snapshot();
            config.progress_callback(GpuSelfPlayProgress{
                now_completed,
                config.total_games,
                now_samples,
                inference_metrics.positions_evaluated,
                inference_metrics.batches_evaluated,
                inference_metrics.padded_positions,
                std::chrono::duration<double, std::milli>(current - started)
                    .count(),
                inference_metrics.average_inference_latency_ms(),
                thread_count,
            });
          }
        }
      } catch (...) {
        std::lock_guard lock(exception_mutex);
        if (!first_exception) {
          first_exception = std::current_exception();
        }
      }
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }

  if (first_exception) {
    std::rethrow_exception(first_exception);
  }

  const auto stopped = std::chrono::steady_clock::now();
  result.metrics = make_metrics(
      completed_games.load(), samples_written.load(),
      std::chrono::duration<double, std::milli>(stopped - started).count(),
      async_evaluator.metrics_snapshot(), config.sampled_gpu_utilization_percent,
      mcts_legal_move_generation_calls, mcts_legal_move_generation_ms);

  if (config.write_replay) {
    for (auto& path : replay_paths) {
      if (!path.empty()) {
        result.replay_paths.push_back(std::move(path));
      }
    }
  }

  return result;
}

}  // namespace chessmoe::selfplay
