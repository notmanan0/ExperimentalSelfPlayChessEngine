#include <chessmoe/selfplay/selfplay_app.h>

#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    auto options = chessmoe::selfplay::parse_selfplay_options(argc, argv);
    auto evaluator = chessmoe::selfplay::create_batch_evaluator(options);

    options.pipeline.progress_interval = options.progress_interval;
    options.pipeline.progress_callback =
        [&](const chessmoe::selfplay::GpuSelfPlayProgress& progress) {
          std::cout << chessmoe::selfplay::format_progress(
                           chessmoe::selfplay::ProgressSnapshot{
                               progress.completed_games,
                               progress.total_games,
                               progress.samples_written,
                               progress.positions_evaluated,
                               progress.batches_evaluated,
                               progress.padded_positions,
                               progress.elapsed_ms,
                               progress.average_inference_latency_ms,
                               progress.active_games,
                               options.pipeline.replay_output_dir,
                               options.evaluator_mode,
                               options.pipeline.replay_options.model_version,
                           })
                    << '\n';
        };

    std::cout << "selfplay start: evaluator="
              << chessmoe::selfplay::evaluator_mode_name(options.evaluator_mode)
              << " games=" << options.pipeline.total_games
              << " concurrent_games=" << options.pipeline.concurrent_games
              << " fixed_batch=" << options.pipeline.fixed_batch_size
              << " visits=" << options.pipeline.game.search_visits
              << " max_plies=" << options.pipeline.game.max_plies
              << " output_dir=" << options.pipeline.replay_output_dir.string()
              << " model_version="
              << options.pipeline.replay_options.model_version << '\n';

    chessmoe::selfplay::GpuSelfPlayPipeline pipeline(*evaluator);
    const auto result = pipeline.run(options.pipeline);
    const auto& metrics = result.metrics;

    std::cout << "selfplay summary: games_completed="
              << metrics.games_completed
              << " samples_written=" << metrics.samples_written
              << " games/sec="
              << (metrics.elapsed_ms > 0.0
                      ? static_cast<double>(metrics.games_completed) /
                            (metrics.elapsed_ms / 1000.0)
                      : 0.0)
              << " samples/sec="
              << (metrics.elapsed_ms > 0.0
                      ? static_cast<double>(metrics.samples_written) /
                            (metrics.elapsed_ms / 1000.0)
                      : 0.0)
              << " positions/sec=" << metrics.positions_per_second
              << " batches_evaluated=" << metrics.batches_evaluated
              << " padded_positions=" << metrics.padded_positions
              << " avg_inference_latency_ms="
              << metrics.average_inference_latency_ms
              << " replay_chunks=" << result.replay_paths.size() << '\n';
  } catch (const std::exception& e) {
    std::cerr << "selfplay failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
