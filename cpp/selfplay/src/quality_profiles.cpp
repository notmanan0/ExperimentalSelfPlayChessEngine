#include <chessmoe/selfplay/quality_profiles.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace chessmoe::selfplay {
namespace {

std::map<std::string, QualityProfile> load_profiles_from_json(
    const std::filesystem::path& path) {
  std::map<std::string, QualityProfile> profiles;

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

    QualityProfile profile;
    profile.name = profile_name;

    for (std::sregex_iterator fit(body.begin(), body.end(), field), fend;
         fit != fend; ++fit) {
      const auto key = (*fit)[1].str();
      auto value = (*fit)[2].str();
      if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);
      }

      if (key == "visits") {
        profile.visits = std::stoi(value);
      } else if (key == "max_plies") {
        profile.max_plies = std::stoi(value);
      } else if (key == "root_dirichlet_noise") {
        profile.root_dirichlet_noise = (value == "true" || value == "1");
      } else if (key == "root_dirichlet_alpha") {
        profile.root_dirichlet_alpha = std::stod(value);
      } else if (key == "root_dirichlet_epsilon") {
        profile.root_dirichlet_epsilon = std::stod(value);
      } else if (key == "temperature_initial") {
        profile.temperature_initial = std::stod(value);
      } else if (key == "temperature_final") {
        profile.temperature_final = std::stod(value);
      } else if (key == "temperature_cutoff_ply") {
        profile.temperature_cutoff_ply = std::stoi(value);
      } else if (key == "games") {
        profile.games = std::stoi(value);
      } else if (key == "arena_games") {
        profile.arena_games = std::stoi(value);
      } else if (key == "promotion_threshold") {
        profile.promotion_threshold = std::stod(value);
      } else if (key == "resignation_enabled") {
        profile.resignation_enabled = (value == "true" || value == "1");
      } else if (key == "description") {
        profile.description = value;
      }
    }

    profiles[profile_name] = profile;
  }

  return profiles;
}

QualityProfile builtin_fast_bootstrap() {
  QualityProfile p;
  p.name = "fast_bootstrap";
  p.visits = 32;
  p.max_plies = 128;
  p.root_dirichlet_noise = true;
  p.root_dirichlet_alpha = 0.3;
  p.root_dirichlet_epsilon = 0.25;
  p.temperature_initial = 1.0;
  p.temperature_final = 0.0;
  p.temperature_cutoff_ply = 20;
  p.games = 256;
  p.arena_games = 32;
  p.promotion_threshold = 0.55;
  p.resignation_enabled = false;
  p.description = "Fast bootstrap generation for initial training data";
  return p;
}

QualityProfile builtin_balanced_generation() {
  QualityProfile p;
  p.name = "balanced_generation";
  p.visits = 128;
  p.max_plies = 200;
  p.root_dirichlet_noise = true;
  p.root_dirichlet_alpha = 0.3;
  p.root_dirichlet_epsilon = 0.25;
  p.temperature_initial = 1.0;
  p.temperature_final = 0.0;
  p.temperature_cutoff_ply = 30;
  p.games = 2048;
  p.arena_games = 128;
  p.promotion_threshold = 0.55;
  p.resignation_enabled = false;
  p.description = "Balanced generation for steady improvement";
  return p;
}

QualityProfile builtin_high_quality_generation() {
  QualityProfile p;
  p.name = "high_quality_generation";
  p.visits = 400;
  p.max_plies = 300;
  p.root_dirichlet_noise = true;
  p.root_dirichlet_alpha = 0.3;
  p.root_dirichlet_epsilon = 0.25;
  p.temperature_initial = 1.0;
  p.temperature_final = 0.0;
  p.temperature_cutoff_ply = 40;
  p.games = 4096;
  p.arena_games = 256;
  p.promotion_threshold = 0.55;
  p.resignation_enabled = true;
  p.description = "High-quality generation for strong models";
  return p;
}

QualityProfile builtin_arena_eval() {
  QualityProfile p;
  p.name = "arena_eval";
  p.visits = 200;
  p.max_plies = 300;
  p.root_dirichlet_noise = false;
  p.root_dirichlet_alpha = 0.0;
  p.root_dirichlet_epsilon = 0.0;
  p.temperature_initial = 0.0;
  p.temperature_final = 0.0;
  p.temperature_cutoff_ply = 0;
  p.games = 0;
  p.arena_games = 256;
  p.promotion_threshold = 0.55;
  p.resignation_enabled = false;
  p.description = "Arena evaluation settings with deterministic play";
  return p;
}

QualityProfile builtin_debug_smoke() {
  QualityProfile p;
  p.name = "debug_smoke";
  p.visits = 4;
  p.max_plies = 20;
  p.root_dirichlet_noise = false;
  p.root_dirichlet_alpha = 0.0;
  p.root_dirichlet_epsilon = 0.0;
  p.temperature_initial = 1.0;
  p.temperature_final = 0.0;
  p.temperature_cutoff_ply = 10;
  p.games = 4;
  p.arena_games = 4;
  p.promotion_threshold = 0.55;
  p.resignation_enabled = false;
  p.description = "Minimal smoke test for pipeline validation";
  return p;
}

std::map<std::string, QualityProfile> builtin_profiles() {
  std::map<std::string, QualityProfile> profiles;
  auto add = [&](QualityProfile p) { profiles[p.name] = std::move(p); };
  add(builtin_fast_bootstrap());
  add(builtin_balanced_generation());
  add(builtin_high_quality_generation());
  add(builtin_arena_eval());
  add(builtin_debug_smoke());
  return profiles;
}

}  // namespace

QualityProfile resolve_quality_profile(std::string_view name) {
  const auto profiles = builtin_profiles();

  auto it = profiles.find(std::string(name));
  if (it != profiles.end()) {
    return it->second;
  }

  const auto config_path =
      std::filesystem::path("configs/profiles/quality.json");
  const auto file_profiles = load_profiles_from_json(config_path);
  auto fit = file_profiles.find(std::string(name));
  if (fit != file_profiles.end()) {
    return fit->second;
  }

  std::ostringstream error;
  error << "unknown quality profile: " << std::string(name)
        << "; available profiles:";
  for (const auto& [key, _] : profiles) {
    error << " " << key;
  }
  throw std::invalid_argument(error.str());
}

std::vector<QualityProfile> available_quality_profiles() {
  const auto profiles = builtin_profiles();
  std::vector<QualityProfile> result;
  result.reserve(profiles.size());
  for (const auto& [_, profile] : profiles) {
    result.push_back(profile);
  }
  return result;
}

SelfPlayConfig apply_quality_profile(const QualityProfile& quality,
                                     std::uint32_t seed) {
  SelfPlayConfig config;
  config.search_visits = quality.visits;
  config.max_plies = quality.max_plies;
  config.add_root_dirichlet_noise = quality.root_dirichlet_noise;
  config.root_dirichlet_alpha = quality.root_dirichlet_alpha;
  config.root_dirichlet_epsilon = quality.root_dirichlet_epsilon;
  config.temperature.initial = quality.temperature_initial;
  config.temperature.final = quality.temperature_final;
  config.temperature.cutoff_ply = quality.temperature_cutoff_ply;
  config.resignation_enabled = quality.resignation_enabled;
  config.seed = seed;
  return config;
}

}  // namespace chessmoe::selfplay
