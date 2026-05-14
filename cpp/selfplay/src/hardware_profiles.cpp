#include <chessmoe/selfplay/hardware_profiles.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace chessmoe::selfplay {
namespace {

std::map<std::string, HardwareProfile> load_profiles_from_json(
    const std::filesystem::path& path) {
  std::map<std::string, HardwareProfile> profiles;

  if (!std::filesystem::exists(path)) {
    return profiles;
  }

  std::ifstream input(path);
  if (!input) {
    return profiles;
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const auto text = buffer.str();

  const std::regex entry(
      R"re("([^"]+)"\s*:\s*\{([^}]+)\})re");
  const std::regex field(
      R"re("([^"]+)"\s*:\s*("([^"]*)"|true|false|-?[0-9]+(?:\.[0-9]+)?))re");

  for (std::sregex_iterator it(text.begin(), text.end(), entry), end;
       it != end; ++it) {
    const auto profile_name = (*it)[1].str();
    const auto body = (*it)[2].str();

    HardwareProfile profile;
    profile.name = profile_name;

    for (std::sregex_iterator fit(body.begin(), body.end(), field), fend;
         fit != fend; ++fit) {
      const auto key = (*fit)[1].str();
      auto value = (*fit)[2].str();
      if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);
      }

      if (key == "evaluator") {
        if (value == "material") {
          profile.evaluator = EvaluatorMode::Material;
        } else if (value == "tensorrt") {
          profile.evaluator = EvaluatorMode::TensorRT;
        } else if (value == "onnx") {
          profile.evaluator = EvaluatorMode::Onnx;
        }
      } else if (key == "concurrent_games") {
        profile.concurrent_games = std::stoi(value);
      } else if (key == "fixed_batch") {
        profile.fixed_batch = static_cast<std::size_t>(std::stoul(value));
      } else if (key == "max_pending_requests") {
        profile.max_pending_requests =
            static_cast<std::size_t>(std::stoul(value));
      } else if (key == "flush_ms") {
        profile.flush_ms = std::stoi(value);
      } else if (key == "visits") {
        profile.visits = std::stoi(value);
      } else if (key == "max_plies") {
        profile.max_plies = std::stoi(value);
      } else if (key == "precision") {
        profile.precision = value;
      } else if (key == "cpu_workers") {
        profile.cpu_workers = std::stoi(value);
      } else if (key == "replay_chunk_games") {
        profile.replay_chunk_games = std::stoi(value);
      } else if (key == "progress_interval") {
        profile.progress_interval = std::stoi(value);
      } else if (key == "description") {
        profile.description = value;
      }
    }

    profiles[profile_name] = profile;
  }

  return profiles;
}

HardwareProfile builtin_cpu_bootstrap_debug() {
  HardwareProfile p;
  p.name = "cpu_bootstrap_debug";
  p.evaluator = EvaluatorMode::Material;
  p.concurrent_games = 4;
  p.fixed_batch = 1;
  p.max_pending_requests = 64;
  p.flush_ms = 10;
  p.visits = 8;
  p.max_plies = 32;
  p.precision = "fp32";
  p.cpu_workers = 2;
  p.replay_chunk_games = 16;
  p.progress_interval = 4;
  p.description = "Minimal CPU-only debug run with material evaluator";
  return p;
}

HardwareProfile builtin_cpu_bootstrap_fast() {
  HardwareProfile p;
  p.name = "cpu_bootstrap_fast";
  p.evaluator = EvaluatorMode::Material;
  p.concurrent_games = 16;
  p.fixed_batch = 1;
  p.max_pending_requests = 256;
  p.flush_ms = 5;
  p.visits = 32;
  p.max_plies = 128;
  p.precision = "fp32";
  p.cpu_workers = 0;
  p.replay_chunk_games = 64;
  p.progress_interval = 16;
  p.description = "Fast CPU-only bootstrap with material evaluator";
  return p;
}

HardwareProfile builtin_gpu_low_vram() {
  HardwareProfile p;
  p.name = "gpu_low_vram";
  p.evaluator = EvaluatorMode::TensorRT;
  p.concurrent_games = 32;
  p.fixed_batch = 32;
  p.max_pending_requests = 2048;
  p.flush_ms = 2;
  p.visits = 64;
  p.max_plies = 160;
  p.precision = "fp16";
  p.cpu_workers = 0;
  p.replay_chunk_games = 128;
  p.progress_interval = 16;
  p.description = "Low VRAM GPU (4 GB or less)";
  return p;
}

HardwareProfile builtin_gpu_midrange() {
  HardwareProfile p;
  p.name = "gpu_midrange";
  p.evaluator = EvaluatorMode::TensorRT;
  p.concurrent_games = 96;
  p.fixed_batch = 64;
  p.max_pending_requests = 4096;
  p.flush_ms = 2;
  p.visits = 128;
  p.max_plies = 200;
  p.precision = "fp16";
  p.cpu_workers = 0;
  p.replay_chunk_games = 256;
  p.progress_interval = 25;
  p.description = "Mid-range GPU (6-10 GB VRAM, e.g. RTX 4060)";
  return p;
}

HardwareProfile builtin_gpu_highend() {
  HardwareProfile p;
  p.name = "gpu_highend";
  p.evaluator = EvaluatorMode::TensorRT;
  p.concurrent_games = 128;
  p.fixed_batch = 128;
  p.max_pending_requests = 8192;
  p.flush_ms = 2;
  p.visits = 200;
  p.max_plies = 256;
  p.precision = "fp16";
  p.cpu_workers = 0;
  p.replay_chunk_games = 512;
  p.progress_interval = 32;
  p.description = "High-end GPU (12-24 GB VRAM, e.g. RTX 4080/4090)";
  return p;
}

HardwareProfile builtin_gpu_datacenter() {
  HardwareProfile p;
  p.name = "gpu_datacenter";
  p.evaluator = EvaluatorMode::TensorRT;
  p.concurrent_games = 256;
  p.fixed_batch = 256;
  p.max_pending_requests = 16384;
  p.flush_ms = 1;
  p.visits = 400;
  p.max_plies = 300;
  p.precision = "fp16";
  p.cpu_workers = 0;
  p.replay_chunk_games = 1024;
  p.progress_interval = 64;
  p.description = "Datacenter GPU (40+ GB VRAM, e.g. A100/H100)";
  return p;
}

std::map<std::string, HardwareProfile> builtin_profiles() {
  std::map<std::string, HardwareProfile> profiles;
  auto add = [&](HardwareProfile p) { profiles[p.name] = std::move(p); };
  add(builtin_cpu_bootstrap_debug());
  add(builtin_cpu_bootstrap_fast());
  add(builtin_gpu_low_vram());
  add(builtin_gpu_midrange());
  add(builtin_gpu_highend());
  add(builtin_gpu_datacenter());
  return profiles;
}

}  // namespace

HardwareProfile resolve_hardware_profile(std::string_view name) {
  const auto profiles = builtin_profiles();

  auto it = profiles.find(std::string(name));
  if (it != profiles.end()) {
    return it->second;
  }

  const auto config_path =
      std::filesystem::path("configs/profiles/hardware.json");
  const auto file_profiles = load_profiles_from_json(config_path);
  auto fit = file_profiles.find(std::string(name));
  if (fit != file_profiles.end()) {
    return fit->second;
  }

  std::ostringstream error;
  error << "unknown hardware profile: " << std::string(name)
        << "; available profiles:";
  for (const auto& [key, _] : profiles) {
    error << " " << key;
  }
  throw std::invalid_argument(error.str());
}

std::vector<HardwareProfile> available_hardware_profiles() {
  const auto profiles = builtin_profiles();
  std::vector<HardwareProfile> result;
  result.reserve(profiles.size());
  for (const auto& [_, profile] : profiles) {
    result.push_back(profile);
  }
  return result;
}

bool is_bootstrap_profile(std::string_view name) {
  return name == "cpu_bootstrap_debug" || name == "cpu_bootstrap_fast";
}

std::string evaluator_mode_name_for_profile(const HardwareProfile& profile) {
  return evaluator_mode_name(profile.evaluator);
}

}  // namespace chessmoe::selfplay
