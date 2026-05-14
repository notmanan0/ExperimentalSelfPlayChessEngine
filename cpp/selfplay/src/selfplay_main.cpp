#include <chessmoe/selfplay/config_resolver.h>
#include <chessmoe/selfplay/generation_controller.h>
#include <chessmoe/selfplay/hardware_profiles.h>
#include <chessmoe/selfplay/replay_health.h>
#include <chessmoe/selfplay/selfplay_app.h>

#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    auto config = chessmoe::selfplay::resolve_config(argc, argv);

    if (config.probe_only) {
      chessmoe::selfplay::print_hardware_summary(config.probe);
      return EXIT_SUCCESS;
    }

    chessmoe::selfplay::print_resolved_config(config);

    if (config.calibrate) {
      std::cout << "calibration mode: running short benchmark with "
                << config.hardware_profile_name << " / "
                << config.quality_profile_name << "\n";
      config.pipeline.total_games = 4;
      config.pipeline.game.search_visits =
          std::min(config.pipeline.game.search_visits, 8);
      config.pipeline.game.max_plies =
          std::min(config.pipeline.game.max_plies, 32);
    }

    chessmoe::selfplay::enforce_generation_guards(config);

    auto evaluator =
        chessmoe::selfplay::create_batch_evaluator_from_mode(
            config.evaluator_mode, config.engine_path,
            config.pipeline.fixed_batch_size);

    config.pipeline.progress_callback =
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
                               config.pipeline.replay_output_dir,
                               config.evaluator_mode,
                               config.pipeline.replay_options.model_version,
                           })
                    << '\n';
        };

    std::cout << "selfplay start: evaluator="
              << chessmoe::selfplay::evaluator_mode_name(config.evaluator_mode)
              << " games=" << config.pipeline.total_games
              << " concurrent_games=" << config.pipeline.concurrent_games
              << " fixed_batch=" << config.pipeline.fixed_batch_size
              << " visits=" << config.pipeline.game.search_visits
              << " max_plies=" << config.pipeline.game.max_plies
              << " output_dir="
              << config.pipeline.replay_output_dir.string()
              << " model_version="
              << config.pipeline.replay_options.model_version << '\n';

    chessmoe::selfplay::GenerationController controller(*evaluator);
    auto result = controller.run(config);

    const auto& m = result.metrics;
    std::cout << "selfplay summary: games_completed=" << m.games_completed
              << " samples_written=" << m.samples_written
              << " games/sec="
              << (m.elapsed_ms > 0
                      ? m.games_completed / (m.elapsed_ms / 1000.0)
                      : 0.0)
              << " samples/sec=" << m.samples_per_second
              << " positions/sec=" << m.positions_per_second
              << " avg_plies/game=" << m.average_plies_per_game
              << " checkmate=" << m.checkmate_count
              << " stalemate=" << m.stalemate_count
              << " repetition=" << m.repetition_count
              << " fifty_move=" << m.fifty_move_count
              << " max_plies=" << m.max_plies_count
              << " batch_fill=" << m.batch_fill_ratio
              << " padding_ratio=" << m.padding_ratio
              << " avg_inference_latency_ms="
              << m.average_inference_latency_ms
              << " replay_chunks=" << result.replay_paths.size() << '\n';

    if (result.health.total_games > 0) {
      chessmoe::selfplay::print_replay_health_summary(result.health);
      chessmoe::selfplay::write_replay_health_report(
          result.health, config.run_dir / "replay_health.json");
    }

    if (config.calibrate) {
      std::cout << "calibration complete: positions/sec="
                << m.positions_per_second
                << " batch_fill=" << m.batch_fill_ratio << '\n';
    }

    if (config.profile_run) {
      std::cout << "\n=== Profile Breakdown ===\n";
      std::cout << "Total elapsed: " << m.elapsed_ms << "ms\n";
      std::cout << "Positions/sec: " << m.positions_per_second << "\n";
      std::cout << "Games/sec: " << (m.elapsed_ms > 0 ? m.games_completed / (m.elapsed_ms / 1000.0) : 0) << "\n";
      std::cout << "Batch fill ratio: " << m.batch_fill_ratio << "\n";
      std::cout << "Padding ratio: " << m.padding_ratio << "\n";
      std::cout << "Avg inference latency: " << m.average_inference_latency_ms << "ms\n";
      std::cout << "Max queue depth: " << m.max_queue_depth << "\n";
    }

  } catch (const std::exception& e) {
    std::cerr << "selfplay failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
