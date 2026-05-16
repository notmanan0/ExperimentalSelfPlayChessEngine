#include <chessmoe/selfplay/config_resolver.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <ostream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace chessmoe::selfplay {
namespace {

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open config file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

using ConfigMap = std::map<std::string, std::string>;

std::string normalize_key(std::string key) {
  std::ranges::replace(key, '-', '_');
  return key;
}

ConfigMap parse_flat_json_config(const std::filesystem::path& path) {
  const auto text = read_text_file(path);
  ConfigMap values;

  const std::regex entry(
      R"re("([^"]+)"\s*:\s*("([^"]*)"|true|false|-?[0-9]+(?:\.[0-9]+)?))re");
  for (std::sregex_iterator it(text.begin(), text.end(), entry), end;
       it != end; ++it) {
    const auto key = normalize_key((*it)[1].str());
    auto value = (*it)[2].str();
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
      value = value.substr(1, value.size() - 2);
    }
    values[key] = value;
  }
  return values;
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

std::optional<std::string_view> value_after(int& index, int argc, char** argv,
                                            std::string_view flag) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string(flag) + " requires a value");
  }
  ++index;
  return argv[index];
}

std::string current_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf{};
#if defined(_WIN32)
  localtime_s(&tm_buf, &time_t_now);
#else
  localtime_r(&time_t_now, &tm_buf);
#endif
  std::ostringstream out;
  out << std::put_time(&tm_buf, "%Y-%m-%dT%H-%M-%S");
  return out.str();
}

void apply_legacy_config(ResolvedConfig& config,
                         const std::filesystem::path& path) {
  const auto values = parse_flat_json_config(path);

  auto get = [&](const std::string& key) -> std::optional<std::string> {
    auto it = values.find(key);
    if (it != values.end()) {
      return it->second;
    }
    return std::nullopt;
  };

  if (auto v = get("evaluator")) {
    if (*v == "material") {
      config.evaluator_mode = EvaluatorMode::Material;
    } else if (*v == "pesto") {
      config.evaluator_mode = EvaluatorMode::Pesto;
    } else if (*v == "tensorrt") {
      config.evaluator_mode = EvaluatorMode::TensorRT;
    } else if (*v == "onnx") {
      config.evaluator_mode = EvaluatorMode::Onnx;
    }
  }
  if (auto v = get("engine")) {
    config.engine_path = std::filesystem::path(*v);
  }
  if (auto v = get("games")) {
    config.pipeline.total_games = std::stoi(*v);
  }
  if (auto v = get("concurrent_games")) {
    config.pipeline.concurrent_games = std::stoi(*v);
  }
  if (auto v = get("fixed_batch")) {
    config.pipeline.fixed_batch_size =
        static_cast<std::size_t>(std::stoul(*v));
  }
  if (auto v = get("max_pending_requests")) {
    config.pipeline.max_pending_requests =
        static_cast<std::size_t>(std::stoul(*v));
  }
  if (auto v = get("flush_ms")) {
    config.pipeline.flush_timeout =
        std::chrono::milliseconds(std::stoi(*v));
  }
  if (auto v = get("visits")) {
    config.pipeline.game.search_visits = std::stoi(*v);
  }
  if (auto v = get("max_plies")) {
    config.pipeline.game.max_plies = std::stoi(*v);
  }
  if (auto v = get("write_replay")) {
    config.pipeline.write_replay = parse_bool(*v);
  }
  if (auto v = get("output_dir")) {
    config.pipeline.replay_output_dir = std::filesystem::path(*v);
  }
  if (auto v = get("model_version")) {
    config.model_version =
        static_cast<std::uint32_t>(std::stoul(*v));
    config.pipeline.replay_options.model_version = config.model_version;
  }
  if (auto v = get("generator_version")) {
    config.pipeline.replay_options.generator_version =
        static_cast<std::uint32_t>(std::stoul(*v));
  }
  if (auto v = get("progress_interval")) {
    config.pipeline.progress_interval = std::stoi(*v);
  }
  if (auto v = get("deterministic")) {
    config.pipeline.game.deterministic = parse_bool(*v);
  }
  if (auto v = get("disable_root_noise")) {
    config.pipeline.game.add_root_dirichlet_noise = !parse_bool(*v);
  }
}

void apply_cli_override(ResolvedConfig& config, std::string_view arg,
                        std::string_view value) {
  const auto key = normalize_key(std::string(arg));

  if (key == "evaluator") {
    if (value == "material") {
      config.evaluator_mode = EvaluatorMode::Material;
    } else if (value == "pesto") {
      config.evaluator_mode = EvaluatorMode::Pesto;
    } else if (value == "tensorrt") {
      config.evaluator_mode = EvaluatorMode::TensorRT;
    } else if (value == "onnx") {
      config.evaluator_mode = EvaluatorMode::Onnx;
    }
  } else if (key == "engine") {
    config.engine_path = std::filesystem::path(value);
  } else if (key == "games") {
    config.pipeline.total_games = std::stoi(std::string(value));
  } else if (key == "concurrent_games") {
    config.pipeline.concurrent_games = std::stoi(std::string(value));
  } else if (key == "fixed_batch") {
    config.pipeline.fixed_batch_size =
        static_cast<std::size_t>(std::stoul(std::string(value)));
  } else if (key == "visits") {
    config.pipeline.game.search_visits = std::stoi(std::string(value));
  } else if (key == "max_plies") {
    config.pipeline.game.max_plies = std::stoi(std::string(value));
  } else if (key == "output_dir") {
    config.pipeline.replay_output_dir = std::filesystem::path(value);
  } else if (key == "model_version") {
    config.model_version = static_cast<std::uint32_t>(std::stoul(std::string(value)));
    config.pipeline.replay_options.model_version = config.model_version;
  } else if (key == "progress_interval") {
    config.pipeline.progress_interval = std::stoi(std::string(value));
  }
}

void apply_hardware_profile_to_config(ResolvedConfig& config) {
  const auto& hw = config.hardware;
  config.evaluator_mode = hw.evaluator;
  config.pipeline.concurrent_games = hw.concurrent_games;
  config.pipeline.fixed_batch_size = hw.fixed_batch;
  config.pipeline.max_pending_requests = hw.max_pending_requests;
  config.pipeline.flush_timeout = std::chrono::milliseconds(hw.flush_ms);
  config.pipeline.progress_interval = hw.progress_interval;
  config.pipeline.replay_options.generator_version = 14;
}

void apply_quality_profile_to_config(ResolvedConfig& config) {
  const auto& q = config.quality;
  config.pipeline.game.search_visits = q.visits;
  config.pipeline.game.max_plies = q.max_plies;
  config.pipeline.game.add_root_dirichlet_noise = q.root_dirichlet_noise;
  config.pipeline.game.root_dirichlet_alpha = q.root_dirichlet_alpha;
  config.pipeline.game.root_dirichlet_epsilon = q.root_dirichlet_epsilon;
  config.pipeline.game.temperature.initial = q.temperature_initial;
  config.pipeline.game.temperature.final = q.temperature_final;
  config.pipeline.game.temperature.cutoff_ply = q.temperature_cutoff_ply;
  config.pipeline.game.resignation_enabled = q.resignation_enabled;
  config.pipeline.total_games = q.games;
}

}  // namespace

ResolvedConfig resolve_config(int argc, char** argv) {
  ResolvedConfig config;
  config.pipeline.replay_options.generator_version = 14;
  config.pipeline.game.temperature.initial = 1.0;
  config.pipeline.game.temperature.final = 0.0;
  config.pipeline.game.temperature.cutoff_ply = 30;
  config.pipeline.write_replay = true;

  std::optional<std::filesystem::path> legacy_config_path;
  std::optional<std::string> hardware_profile_name;
  std::optional<std::string> quality_profile_name;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--config") {
      legacy_config_path =
          std::filesystem::path(*value_after(i, argc, argv, "--config"));
    } else if (arg == "--hardware-profile") {
      hardware_profile_name = std::string(*value_after(i, argc, argv, arg));
    } else if (arg == "--quality") {
      quality_profile_name = std::string(*value_after(i, argc, argv, arg));
    } else if (arg == "--allow-debug") {
      config.allow_debug = true;
    } else if (arg == "--resume") {
      config.resume = true;
    } else if (arg == "--fresh" || arg == "--overwrite") {
      config.fresh = true;
    } else if (arg == "--calibrate") {
      config.calibrate = true;
    } else if (arg == "--probe") {
      config.probe_only = true;
    } else if (arg == "--profile-run") {
      config.profile_run = true;
    } else if (arg == "--phase") {
      config.phase = std::stoi(std::string(*value_after(i, argc, argv, arg)));
    } else if (arg == "--run-dir") {
      config.run_dir = std::filesystem::path(*value_after(i, argc, argv, arg));
    }
  }

  config.probe = probe_hardware();

  if (hardware_profile_name) {
    config.hardware_profile_name = *hardware_profile_name;
    config.hardware = resolve_hardware_profile(*hardware_profile_name);
  } else if (!legacy_config_path) {
    config.hardware_profile_name = config.probe.recommended_profile;
    config.hardware = resolve_hardware_profile(config.probe.recommended_profile);
  }

  if (quality_profile_name) {
    config.quality_profile_name = *quality_profile_name;
    config.quality = resolve_quality_profile(*quality_profile_name);
  } else {
    config.quality_profile_name = "balanced_generation";
    config.quality = resolve_quality_profile("balanced_generation");
  }

  if (!legacy_config_path) {
    apply_hardware_profile_to_config(config);
    apply_quality_profile_to_config(config);
  }

  if (legacy_config_path) {
    apply_legacy_config(config, *legacy_config_path);
  }

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--config" || arg == "--hardware-profile" ||
        arg == "--quality" || arg == "--phase" || arg == "--run-dir") {
      (void)value_after(i, argc, argv, arg);
    } else if (arg == "--engine") {
      apply_cli_override(config, "engine",
                         *value_after(i, argc, argv, arg));
    } else if (arg == "--evaluator") {
      apply_cli_override(config, "evaluator",
                         *value_after(i, argc, argv, arg));
    } else if (arg == "--games") {
      apply_cli_override(config, "games",
                         *value_after(i, argc, argv, arg));
    } else if (arg == "--concurrent-games") {
      apply_cli_override(config, "concurrent_games",
                         *value_after(i, argc, argv, arg));
    } else if (arg == "--fixed-batch") {
      apply_cli_override(config, "fixed_batch",
                         *value_after(i, argc, argv, arg));
    } else if (arg == "--visits") {
      apply_cli_override(config, "visits",
                         *value_after(i, argc, argv, arg));
    } else if (arg == "--max-plies") {
      apply_cli_override(config, "max_plies",
                         *value_after(i, argc, argv, arg));
    } else if (arg == "--output-dir") {
      apply_cli_override(config, "output_dir",
                         *value_after(i, argc, argv, arg));
    } else if (arg == "--model-version") {
      apply_cli_override(config, "model_version",
                         *value_after(i, argc, argv, arg));
    } else if (arg == "--progress-interval") {
      apply_cli_override(config, "progress_interval",
                         *value_after(i, argc, argv, arg));
    } else if (arg == "--allow-debug" || arg == "--resume" ||
               arg == "--fresh" || arg == "--overwrite" ||
               arg == "--calibrate" || arg == "--probe" ||
               arg == "--profile-run") {
      // already handled
    } else if (arg == "--deterministic") {
      config.pipeline.game.deterministic = true;
    } else if (arg == "--disable-root-noise") {
      config.pipeline.game.add_root_dirichlet_noise = false;
    } else if (arg[0] == '-') {
      // Skip unknown flags that start with - but were already consumed
    }
  }

  if (config.run_dir.empty()) {
    const auto run_id = generate_run_id(config.phase);
    config.run_dir = std::filesystem::path("runs") / run_id;
  }

  return config;
}

void print_resolved_config(const ResolvedConfig& config) {
  std::cout << "=== Resolved Configuration ===" << '\n';
  std::cout << "Hardware profile: " << config.hardware_profile_name << '\n';
  std::cout << "Quality profile: " << config.quality_profile_name << '\n';
  std::cout << "Evaluator: " << evaluator_mode_name(config.evaluator_mode)
            << '\n';
  std::cout << "Games: " << config.pipeline.total_games << '\n';
  std::cout << "Concurrent games: " << config.pipeline.concurrent_games
            << '\n';
  std::cout << "Fixed batch: " << config.pipeline.fixed_batch_size << '\n';
  std::cout << "Visits: " << config.pipeline.game.search_visits << '\n';
  std::cout << "Max plies: " << config.pipeline.game.max_plies << '\n';
  std::cout << "Model version: " << config.model_version << '\n';
  std::cout << "Output dir: "
            << config.pipeline.replay_output_dir.string() << '\n';
  std::cout << "Run dir: " << config.run_dir.string() << '\n';
  std::cout << "Build type: " << config.probe.build_type << '\n';

  if (config.engine_path) {
    std::cout << "Engine path: " << config.engine_path->string() << '\n';
  }
  if (config.allow_debug) {
    std::cout << "Allow debug: yes" << '\n';
  }
  if (config.resume) {
    std::cout << "Resume mode: yes" << '\n';
  }
  if (config.fresh) {
    std::cout << "Fresh mode: yes (overwrite existing)" << '\n';
  }
}

void write_resolved_config(const ResolvedConfig& config,
                           const std::filesystem::path& path) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write resolved config: " +
                             path.string());
  }

  out << "{\n";
  out << "  \"hardware_profile\": \"" << config.hardware_profile_name
      << "\",\n";
  out << "  \"quality_profile\": \"" << config.quality_profile_name
      << "\",\n";
  out << "  \"evaluator\": \""
      << evaluator_mode_name(config.evaluator_mode) << "\",\n";
  out << "  \"games\": " << config.pipeline.total_games << ",\n";
  out << "  \"concurrent_games\": " << config.pipeline.concurrent_games
      << ",\n";
  out << "  \"fixed_batch\": " << config.pipeline.fixed_batch_size << ",\n";
  out << "  \"visits\": " << config.pipeline.game.search_visits << ",\n";
  out << "  \"max_plies\": " << config.pipeline.game.max_plies << ",\n";
  out << "  \"model_version\": " << config.model_version << ",\n";
  out << "  \"output_dir\": \""
      << config.pipeline.replay_output_dir.string() << "\",\n";
  out << "  \"run_dir\": \"" << config.run_dir.string() << "\",\n";
  out << "  \"build_type\": \"" << config.probe.build_type << "\",\n";
  out << "  \"debug_build\": "
      << (config.probe.debug_build ? "true" : "false") << ",\n";
  out << "  \"tensorrt_compiled\": "
      << (config.probe.tensorrt_compiled ? "true" : "false") << ",\n";
  out << "  \"gpu\": \"" << config.probe.gpu_name << "\",\n";
  out << "  \"vram_bytes\": " << config.probe.vram_bytes << ",\n";
  out << "  \"cpu_cores\": " << config.probe.cpu_logical_cores << "\n";
  out << "}\n";
}

void enforce_generation_guards(const ResolvedConfig& config) {
  if (config.evaluator_mode == EvaluatorMode::Material) {
    std::cerr << "[WARNING] Material evaluator is bootstrap/debug only.\n";
    if (!config.allow_debug) {
      throw std::runtime_error(
          "Refusing to run serious generation with material evaluator. "
          "Use --allow-debug to override.");
    }
  }

  if (config.probe.debug_build && !config.allow_debug &&
      !is_bootstrap_profile(config.hardware_profile_name)) {
    throw std::runtime_error(
        "Debug build detected. Serious self-play will be much slower. "
        "Build with CMAKE_BUILD_TYPE=Release or use --allow-debug.");
  }

  if (config.evaluator_mode == EvaluatorMode::TensorRT) {
    if (!config.probe.tensorrt_compiled) {
      throw std::runtime_error(
          "TensorRT evaluator selected, but TensorRT support is not compiled. "
          "Reconfigure with CHESSMOE_ENABLE_TENSORRT=ON.");
    }
    if (!config.engine_path) {
      throw std::runtime_error(
          "TensorRT evaluator selected, but no engine path specified. "
          "Use --engine <path> or set engine in config.");
    }
  }

  if (config.evaluator_mode == EvaluatorMode::Onnx) {
    throw std::runtime_error(
        "ONNX C++ evaluator is not implemented. "
        "Use --evaluator tensorrt or --evaluator material.");
  }
}

std::string generate_run_id(int phase) {
  const auto timestamp = current_timestamp();
  if (phase > 0) {
    return timestamp + "_phase" + std::to_string(phase);
  }
  return timestamp;
}

}  // namespace chessmoe::selfplay
