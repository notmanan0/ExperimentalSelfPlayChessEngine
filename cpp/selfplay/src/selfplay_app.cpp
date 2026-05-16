#include <chessmoe/selfplay/selfplay_app.h>

#include <chessmoe/eval/material_evaluator.h>
#include <chessmoe/eval/pesto_evaluator.h>
#include <chessmoe/inference/tensorrt_engine.h>
#include <chessmoe/inference/tensorrt_evaluator.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace chessmoe::selfplay {
namespace {

using ConfigMap = std::map<std::string, std::string>;

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open config file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string normalize_key(std::string key) {
  std::ranges::replace(key, '-', '_');
  return key;
}

ConfigMap parse_flat_json_config(const std::filesystem::path& path) {
  const auto text = read_text_file(path);
  ConfigMap values;

  const std::regex entry(
      R"re("([^"]+)"\s*:\s*("([^"]*)"|true|false|-?[0-9]+(?:\.[0-9]+)?))re");
  for (std::sregex_iterator it(text.begin(), text.end(), entry), end; it != end;
       ++it) {
    const auto key = normalize_key((*it)[1].str());
    auto value = (*it)[2].str();
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
      value = value.substr(1, value.size() - 2);
    }
    values[key] = value;
  }
  return values;
}

EvaluatorMode parse_mode(std::string_view value) {
  if (value == "material") {
    return EvaluatorMode::Material;
  }
  if (value == "pesto") {
    return EvaluatorMode::Pesto;
  }
  if (value == "tensorrt") {
    return EvaluatorMode::TensorRT;
  }
  if (value == "onnx") {
    return EvaluatorMode::Onnx;
  }
  throw std::invalid_argument("unsupported evaluator mode: " +
                               std::string(value));
}

bool parse_bool(std::string_view value) {
  if (value == "true" || value == "1") {
    return true;
  }
  if (value == "false" || value == "0") {
    return false;
  }
  throw std::invalid_argument("invalid boolean value: " + std::string(value));
}

void apply_option(SelfPlayAppOptions& options, std::string key,
                  std::string_view value) {
  key = normalize_key(std::move(key));
  auto& pipeline = options.pipeline;

  if (key == "evaluator") {
    options.evaluator_mode = parse_mode(value);
  } else if (key == "engine") {
    options.engine_path = std::filesystem::path(value);
  } else if (key == "games") {
    pipeline.total_games = std::stoi(std::string(value));
  } else if (key == "concurrent_games") {
    pipeline.concurrent_games = std::stoi(std::string(value));
  } else if (key == "fixed_batch") {
    pipeline.fixed_batch_size =
        static_cast<std::size_t>(std::stoul(std::string(value)));
  } else if (key == "max_pending_requests") {
    pipeline.max_pending_requests =
        static_cast<std::size_t>(std::stoul(std::string(value)));
  } else if (key == "flush_ms") {
    pipeline.flush_timeout =
        std::chrono::milliseconds(std::stoi(std::string(value)));
  } else if (key == "visits") {
    pipeline.game.search_visits = std::stoi(std::string(value));
  } else if (key == "max_plies") {
    pipeline.game.max_plies = std::stoi(std::string(value));
  } else if (key == "write_replay") {
    pipeline.write_replay = parse_bool(value);
  } else if (key == "output_dir") {
    pipeline.replay_output_dir = std::filesystem::path(value);
  } else if (key == "model_version") {
    pipeline.replay_options.model_version =
        static_cast<std::uint32_t>(std::stoul(std::string(value)));
  } else if (key == "generator_version") {
    pipeline.replay_options.generator_version =
        static_cast<std::uint32_t>(std::stoul(std::string(value)));
  } else if (key == "progress_interval") {
    options.progress_interval = std::stoi(std::string(value));
    pipeline.progress_interval = options.progress_interval;
  } else if (key == "deterministic") {
    pipeline.game.deterministic = parse_bool(value);
  } else if (key == "disable_root_noise") {
    pipeline.game.add_root_dirichlet_noise = !parse_bool(value);
  }
}

std::optional<std::string_view> value_after(int& index, int argc, char** argv,
                                            std::string_view flag) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string(flag) + " requires a value");
  }
  ++index;
  return argv[index];
}

std::string seconds_text(double seconds) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(1) << seconds << "s";
  return out.str();
}

}  // namespace

SelfPlayAppOptions parse_selfplay_options(int argc, char** argv) {
  SelfPlayAppOptions options;
  options.pipeline.replay_options.generator_version = 13;
  options.pipeline.game.temperature.initial = 1.0;
  options.pipeline.game.temperature.final = 0.0;
  options.pipeline.game.temperature.cutoff_ply = 30;

  std::optional<std::filesystem::path> config_path;
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--config") {
      config_path = std::filesystem::path(*value_after(i, argc, argv, "--config"));
    }
  }
  if (config_path) {
    for (const auto& [key, value] : parse_flat_json_config(*config_path)) {
      apply_option(options, key, value);
    }
  }

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--config") {
      (void)value_after(i, argc, argv, "--config");
    } else if (arg == "--evaluator") {
      apply_option(options, "evaluator", *value_after(i, argc, argv, arg));
    } else if (arg == "--engine") {
      apply_option(options, "engine", *value_after(i, argc, argv, arg));
    } else if (arg == "--games") {
      apply_option(options, "games", *value_after(i, argc, argv, arg));
    } else if (arg == "--concurrent-games") {
      apply_option(options, "concurrent_games", *value_after(i, argc, argv, arg));
    } else if (arg == "--fixed-batch") {
      apply_option(options, "fixed_batch", *value_after(i, argc, argv, arg));
    } else if (arg == "--max-pending-requests") {
      apply_option(options, "max_pending_requests",
                   *value_after(i, argc, argv, arg));
    } else if (arg == "--flush-ms") {
      apply_option(options, "flush_ms", *value_after(i, argc, argv, arg));
    } else if (arg == "--visits") {
      apply_option(options, "visits", *value_after(i, argc, argv, arg));
    } else if (arg == "--max-plies") {
      apply_option(options, "max_plies", *value_after(i, argc, argv, arg));
    } else if (arg == "--output-dir") {
      apply_option(options, "output_dir", *value_after(i, argc, argv, arg));
    } else if (arg == "--model-version") {
      apply_option(options, "model_version", *value_after(i, argc, argv, arg));
    } else if (arg == "--progress-interval") {
      apply_option(options, "progress_interval",
                   *value_after(i, argc, argv, arg));
    } else if (arg == "--write-replay") {
      apply_option(options, "write_replay", "true");
    } else if (arg == "--deterministic") {
      apply_option(options, "deterministic", "true");
    } else if (arg == "--disable-root-noise") {
      apply_option(options, "disable_root_noise", "true");
    } else {
      throw std::invalid_argument("unknown selfplay argument: " +
                                  std::string(arg));
    }
  }

  return options;
}

std::unique_ptr<eval::IBatchEvaluator> create_batch_evaluator(
    const SelfPlayAppOptions& options) {
  if (options.evaluator_mode == EvaluatorMode::Material) {
    return std::make_unique<eval::MaterialEvaluator>();
  }
  if (options.evaluator_mode == EvaluatorMode::Pesto) {
    return std::make_unique<eval::PestoEvaluator>();
  }
  if (options.evaluator_mode == EvaluatorMode::Onnx) {
    throw std::runtime_error(
        "ONNX C++ evaluator is unavailable; build a real ONNX Runtime backend "
        "before selecting --evaluator onnx");
  }
  if (options.evaluator_mode == EvaluatorMode::TensorRT) {
    if (!options.engine_path) {
      throw std::invalid_argument(
          "TensorRT evaluator requires --engine <path> and never falls back");
    }
    if (inference::tensorrt_build_status() == "not-compiled-with-tensorrt") {
      throw std::runtime_error(
          "TensorRT evaluator selected, but TensorRT support is not compiled in. "
          "Reconfigure with CHESSMOE_ENABLE_TENSORRT=ON.");
    }
    auto backend =
        std::make_shared<inference::TensorRTEngine>(inference::TensorRTEngineConfig{
            *options.engine_path,
            inference::TensorLayout::tiny_baseline(),
            options.pipeline.fixed_batch_size,
            inference::PrecisionMode::Fp32,
            0,
            0,
            false,
            false,
        });
    return std::make_unique<inference::TensorRTEvaluator>(
        std::move(backend), inference::TensorRTEvaluatorConfig{
                                inference::InferenceUseCase::SelfPlay,
                                inference::PrecisionMode::Fp32,
                                true,
                            });
  }
  throw std::runtime_error("unsupported evaluator mode");
}

std::unique_ptr<eval::IBatchEvaluator> create_batch_evaluator_from_mode(
    EvaluatorMode mode,
    const std::optional<std::filesystem::path>& engine_path,
    std::size_t fixed_batch) {
  SelfPlayAppOptions opts;
  opts.evaluator_mode = mode;
  opts.engine_path = engine_path;
  opts.pipeline.fixed_batch_size = fixed_batch;
  return create_batch_evaluator(opts);
}

std::string evaluator_mode_name(EvaluatorMode mode) {
  switch (mode) {
    case EvaluatorMode::Material:
      return "material";
    case EvaluatorMode::Pesto:
      return "pesto";
    case EvaluatorMode::TensorRT:
      return "tensorrt";
    case EvaluatorMode::Onnx:
      return "onnx";
  }
  return "unknown";
}

std::string format_progress(const ProgressSnapshot& snapshot) {
  const double elapsed_seconds = snapshot.elapsed_ms / 1000.0;
  const double total = static_cast<double>(snapshot.total_games);
  const double completed = static_cast<double>(snapshot.completed_games);
  const double percent = total > 0.0 ? completed * 100.0 / total : 0.0;
  const double games_per_second =
      elapsed_seconds > 0.0 ? completed / elapsed_seconds : 0.0;
  const double samples_per_second =
      elapsed_seconds > 0.0
          ? static_cast<double>(snapshot.samples_written) / elapsed_seconds
          : 0.0;
  const double average_plies =
      completed > 0.0 ? static_cast<double>(snapshot.samples_written) / completed
                      : 0.0;

  std::string eta = "unknown";
  if (games_per_second > 0.0 && snapshot.total_games > snapshot.completed_games) {
    eta = seconds_text((total - completed) / games_per_second);
  }

  std::ostringstream out;
  out << "progress games=" << snapshot.completed_games << "/"
      << snapshot.total_games << " (" << std::fixed << std::setprecision(1)
      << percent << "%)"
      << " samples=" << snapshot.samples_written << " games/sec="
      << std::setprecision(2) << games_per_second << " samples/sec="
      << samples_per_second << " elapsed=" << seconds_text(elapsed_seconds)
      << " ETA=" << eta << " avg_plies/game=" << std::setprecision(1)
      << average_plies << " active_games=" << snapshot.active_games
      << " output_dir=" << snapshot.output_dir.string()
      << " evaluator=" << evaluator_mode_name(snapshot.evaluator_mode)
      << " model_version=" << snapshot.model_version
      << " batches=" << snapshot.batches_evaluated
      << " padded_positions=" << snapshot.padded_positions
      << " avg_inference_latency_ms=" << std::setprecision(3)
      << snapshot.average_inference_latency_ms;
  return out.str();
}

}  // namespace chessmoe::selfplay
