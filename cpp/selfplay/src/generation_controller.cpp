#include <chessmoe/selfplay/generation_controller.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <chessmoe/selfplay/game_worker.h>
#include <chessmoe/selfplay/replay_writer.h>

namespace chessmoe::selfplay {
namespace {

std::filesystem::path replay_path_for_game(const std::filesystem::path& dir,
                                           int game_id) {
  std::ostringstream name;
  name << "selfplay_game_" << std::setw(8) << std::setfill('0') << game_id
       << ".cmrep";
  return dir / name.str();
}

GpuSelfPlayMetrics build_metrics(
    int games_completed, std::uint64_t samples_written,
    double elapsed_ms,
    const inference::AsyncBatchingMetrics& inference_metrics,
    int terminal_counts[5],
    std::uint64_t mcts_legal_move_generation_calls,
    double mcts_legal_move_generation_ms) {
  GpuSelfPlayMetrics m;
  m.games_completed = static_cast<std::uint64_t>(games_completed);
  m.samples_written = samples_written;
  m.positions_evaluated = inference_metrics.positions_evaluated;
  m.batches_evaluated = inference_metrics.batches_evaluated;
  m.padded_positions = inference_metrics.padded_positions;
  m.max_queue_depth = inference_metrics.max_queue_depth;
  m.elapsed_ms = elapsed_ms;
  m.average_inference_latency_ms =
      inference_metrics.average_inference_latency_ms();
  m.batch_size_histogram = inference_metrics.batch_size_histogram;
  m.valid_batch_size_histogram = inference_metrics.valid_batch_size_histogram;

  m.checkmate_count = static_cast<std::uint64_t>(terminal_counts[0]);
  m.stalemate_count = static_cast<std::uint64_t>(terminal_counts[1]);
  m.repetition_count = static_cast<std::uint64_t>(terminal_counts[2]);
  m.fifty_move_count = static_cast<std::uint64_t>(terminal_counts[3]);
  m.max_plies_count = static_cast<std::uint64_t>(terminal_counts[4]);

  const double sec = elapsed_ms / 1000.0;
  if (sec > 0.0) {
    m.positions_per_second =
        static_cast<double>(m.positions_evaluated) / sec;
    m.games_per_hour = static_cast<double>(m.games_completed) * 3600.0 / sec;
    m.samples_per_second = static_cast<double>(samples_written) / sec;
  }
  if (m.games_completed > 0) {
    m.average_plies_per_game =
        static_cast<double>(m.samples_written) /
        static_cast<double>(m.games_completed);
  }

  const std::uint64_t total_batch_slots =
      m.positions_evaluated + m.padded_positions;
  if (total_batch_slots > 0) {
    m.batch_fill_ratio = static_cast<double>(m.positions_evaluated) /
                         static_cast<double>(total_batch_slots);
    m.padding_ratio = static_cast<double>(m.padded_positions) /
                      static_cast<double>(total_batch_slots);
  }
  m.mcts_legal_move_generation_calls = mcts_legal_move_generation_calls;
  m.mcts_legal_move_generation_ms = mcts_legal_move_generation_ms;

  return m;
}

}  // namespace

GenerationController::GenerationController(eval::IBatchEvaluator& evaluator)
    : evaluator_(evaluator) {}

GenerationResult GenerationController::run(const ResolvedConfig& config) {
  GenerationResult result;
  result.run_dir = config.run_dir;

  setup_run_directory(config);

  std::set<int> skip_games;
  if (config.resume) {
    skip_games = detect_completed_games(config.pipeline.replay_output_dir);
    if (!skip_games.empty()) {
      std::cout << "resume: found " << skip_games.size()
                << " completed games, skipping\n";
    }
  }

  if (config.pipeline.write_replay) {
    std::filesystem::create_directories(config.pipeline.replay_output_dir);
  }

  inference::AsyncBatchingEvaluator async_eval(
      evaluator_,
      inference::AsyncBatchingEvaluatorConfig{
          config.pipeline.fixed_batch_size,
          config.pipeline.max_pending_requests,
          config.pipeline.flush_timeout,
          true,
      });

  const int total = config.pipeline.total_games;
  const int thread_count =
      std::min(config.pipeline.concurrent_games, std::max(1, total));

  std::vector<std::filesystem::path> replay_paths(
      static_cast<std::size_t>(total));
  std::atomic<int> next_game{0};
  std::atomic<int> completed_games{0};
  std::atomic<std::uint64_t> samples_written{0};
  std::atomic<int> terminal_counts[5];
  for (auto& tc : terminal_counts) {
    tc.store(0);
  }
  std::mutex exception_mutex;
  std::mutex profile_mutex;
  std::exception_ptr first_exception;
  std::uint64_t mcts_legal_move_generation_calls = 0;
  double mcts_legal_move_generation_ms = 0.0;
  const auto started = std::chrono::steady_clock::now();

  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(thread_count));

  for (int ti = 0; ti < thread_count; ++ti) {
    workers.emplace_back([&, ti] {
      try {
        GameWorkerConfig gw_config;
        gw_config.game = config.pipeline.game;
        gw_config.worker_id = static_cast<std::uint32_t>(ti);
        gw_config.openings.fen_pool = config.pipeline.opening_fen_pool;
        gw_config.openings.deterministic_selection = true;
        gw_config.openings.color_balancing = true;

        GameWorker worker(async_eval, gw_config);

        for (;;) {
          const int game_id = next_game.fetch_add(1);
          if (game_id >= total) {
            break;
          }
          if (skip_games.count(game_id) > 0) {
            const int done = completed_games.fetch_add(1) + 1;
            if (config.pipeline.progress_callback &&
                config.pipeline.progress_interval > 0 &&
                (done == total || done % config.pipeline.progress_interval == 0)) {
              const auto now = std::chrono::steady_clock::now();
              const auto inf_m = async_eval.metrics_snapshot();
              config.pipeline.progress_callback(GpuSelfPlayProgress{
                  done, total, samples_written.load(),
                  inf_m.positions_evaluated, inf_m.batches_evaluated,
                  inf_m.padded_positions,
                  std::chrono::duration<double, std::milli>(now - started)
                      .count(),
                  inf_m.average_inference_latency_ms(), thread_count,
              });
            }
            continue;
          }

          auto gw_result = worker.run(game_id);
          const auto game_samples =
              static_cast<std::uint64_t>(gw_result.game.samples.size());
          {
            std::lock_guard lock(profile_mutex);
            mcts_legal_move_generation_calls +=
                gw_result.diagnostics.mcts_legal_move_generation_calls;
            mcts_legal_move_generation_ms +=
                gw_result.diagnostics.mcts_legal_move_generation_ms;
          }

          if (config.pipeline.write_replay) {
            auto options = config.pipeline.replay_options;
            options.game_id = static_cast<std::uint64_t>(game_id);
            options.starting_ply_index = 0;
            const auto path = replay_path_for_game(
                config.pipeline.replay_output_dir, game_id);
            write_replay_chunk(path, gw_result.game, options);
            replay_paths[static_cast<std::size_t>(game_id)] = path;
          }

          int reason_idx = 4;
          switch (gw_result.game.terminal_reason) {
            case TerminalReason::Checkmate:
              reason_idx = 0;
              break;
            case TerminalReason::Stalemate:
              reason_idx = 1;
              break;
            case TerminalReason::Repetition:
              reason_idx = 2;
              break;
            case TerminalReason::FiftyMoveRule:
              reason_idx = 3;
              break;
            case TerminalReason::MaxPlies:
            case TerminalReason::None:
              reason_idx = 4;
              break;
          }
          terminal_counts[reason_idx].fetch_add(1);

          const auto now_samples =
              samples_written.fetch_add(game_samples) + game_samples;
          const int now_completed = completed_games.fetch_add(1) + 1;

          if (config.pipeline.progress_callback &&
              config.pipeline.progress_interval > 0 &&
              (now_completed == total ||
               now_completed % config.pipeline.progress_interval == 0)) {
            const auto now = std::chrono::steady_clock::now();
            const auto inf_m = async_eval.metrics_snapshot();
            config.pipeline.progress_callback(GpuSelfPlayProgress{
                now_completed, total, now_samples,
                inf_m.positions_evaluated, inf_m.batches_evaluated,
                inf_m.padded_positions,
                std::chrono::duration<double, std::milli>(now - started)
                    .count(),
                inf_m.average_inference_latency_ms(), thread_count,
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

  for (auto& w : workers) {
    w.join();
  }

  if (first_exception) {
    std::rethrow_exception(first_exception);
  }

  const auto stopped = std::chrono::steady_clock::now();
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(stopped - started).count();
  int tc[5];
  for (int i = 0; i < 5; ++i) {
    tc[i] = terminal_counts[i].load();
  }
  result.metrics = build_metrics(completed_games.load(),
                                 samples_written.load(), elapsed_ms,
                                 async_eval.metrics_snapshot(), tc,
                                 mcts_legal_move_generation_calls,
                                 mcts_legal_move_generation_ms);

  if (config.pipeline.write_replay) {
    for (auto& p : replay_paths) {
      if (!p.empty()) {
        result.replay_paths.push_back(std::move(p));
      }
    }
  }

  if (!result.replay_paths.empty()) {
    result.health = analyze_replay_health(result.replay_paths);
  }

  write_run_metadata(config, result);
  result.completed = true;

  if (config.profile_run) {
    result.profile_breakdown["total_ms"] = result.metrics.elapsed_ms;
    result.profile_breakdown["games_completed"] =
        static_cast<double>(result.metrics.games_completed);
    result.profile_breakdown["positions_per_second"] =
        result.metrics.positions_per_second;
    result.profile_breakdown["batch_fill_ratio"] =
        result.metrics.batch_fill_ratio;
    result.profile_breakdown["avg_inference_latency_ms"] =
        result.metrics.average_inference_latency_ms;
    result.profile_breakdown["mcts_legal_move_generation_calls"] =
        static_cast<double>(result.metrics.mcts_legal_move_generation_calls);
    result.profile_breakdown["mcts_legal_move_generation_ms"] =
        result.metrics.mcts_legal_move_generation_ms;

    const auto profile_path = config.run_dir / "profile.json";
    std::ofstream pf(profile_path);
    if (pf) {
      pf << "{\n";
      bool first = true;
      for (const auto& [k, v] : result.profile_breakdown) {
        if (!first) pf << ",\n";
        pf << "  \"" << k << "\": " << v;
        first = false;
      }
      pf << "\n}\n";
    }
    std::cout << "profile results written to: "
              << profile_path.string() << "\n";
  }

  return result;
}

void GenerationController::setup_run_directory(const ResolvedConfig& config) {
  std::filesystem::create_directories(config.run_dir);
  write_resolved_config(config, config.run_dir / "resolved_config.json");
}

std::set<int> GenerationController::detect_completed_games(
    const std::filesystem::path& replay_dir) {
  std::set<int> completed;
  if (!std::filesystem::exists(replay_dir)) {
    return completed;
  }
  for (const auto& entry : std::filesystem::directory_iterator(replay_dir)) {
    const auto name = entry.path().filename().string();
    if (name.starts_with("selfplay_game_") && name.ends_with(".cmrep")) {
      const auto id_str = name.substr(14, 8);
      try {
        completed.insert(std::stoi(id_str));
      } catch (...) {
      }
    }
  }
  return completed;
}

void GenerationController::write_run_metadata(const ResolvedConfig& config,
                                              const GenerationResult& result) {
  const auto path = config.run_dir / "summary.json";
  std::ofstream out(path);
  if (!out) {
    return;
  }

  out << "{\n";
  out << "  \"hardware_profile\": \"" << config.hardware_profile_name
      << "\",\n";
  out << "  \"quality_profile\": \"" << config.quality_profile_name
      << "\",\n";
  out << "  \"evaluator\": \""
      << evaluator_mode_name(config.evaluator_mode) << "\",\n";
  out << "  \"games_completed\": " << result.metrics.games_completed << ",\n";
  out << "  \"samples_written\": " << result.metrics.samples_written << ",\n";
  out << "  \"elapsed_ms\": " << result.metrics.elapsed_ms << ",\n";
  out << "  \"games_per_second\": "
      << (result.metrics.elapsed_ms > 0
              ? result.metrics.games_completed /
                    (result.metrics.elapsed_ms / 1000.0)
              : 0.0)
      << ",\n";
  out << "  \"positions_per_second\": "
      << result.metrics.positions_per_second << ",\n";
  out << "  \"average_plies_per_game\": "
      << result.metrics.average_plies_per_game << ",\n";
  out << "  \"checkmate_count\": " << result.metrics.checkmate_count << ",\n";
  out << "  \"stalemate_count\": " << result.metrics.stalemate_count << ",\n";
  out << "  \"repetition_count\": " << result.metrics.repetition_count << ",\n";
  out << "  \"fifty_move_count\": " << result.metrics.fifty_move_count << ",\n";
  out << "  \"max_plies_count\": " << result.metrics.max_plies_count << ",\n";
  out << "  \"batch_fill_ratio\": " << result.metrics.batch_fill_ratio << ",\n";
  out << "  \"padding_ratio\": " << result.metrics.padding_ratio << ",\n";
  out << "  \"avg_inference_latency_ms\": "
      << result.metrics.average_inference_latency_ms << ",\n";
  out << "  \"mcts_legal_move_generation_calls\": "
      << result.metrics.mcts_legal_move_generation_calls << ",\n";
  out << "  \"mcts_legal_move_generation_ms\": "
      << result.metrics.mcts_legal_move_generation_ms << ",\n";
  out << "  \"replay_chunks\": " << result.replay_paths.size() << ",\n";
  out << "  \"health_passed\": "
      << (result.health.passed ? "true" : "false") << ",\n";
  out << "  \"health_warnings\": " << result.health.warnings.size() << ",\n";
  out << "  \"debug_build\": "
      << (config.probe.debug_build ? "true" : "false") << "\n";
  out << "}\n";
}

}  // namespace chessmoe::selfplay
