#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include <chessmoe/selfplay/gpu_selfplay_pipeline.h>
#include <chessmoe/selfplay/hardware_probe.h>
#include <chessmoe/selfplay/hardware_profiles.h>
#include <chessmoe/selfplay/quality_profiles.h>
#include <chessmoe/selfplay/selfplay_app.h>

namespace chessmoe::selfplay {

struct ResolvedConfig {
  GpuSelfPlayPipelineConfig pipeline;
  HardwareProfile hardware;
  QualityProfile quality;
  HardwareProbeResult probe;
  std::string hardware_profile_name;
  std::string quality_profile_name;
  EvaluatorMode evaluator_mode{EvaluatorMode::Material};
  std::optional<std::filesystem::path> engine_path{};
  std::filesystem::path run_dir;
  bool allow_debug{false};
  bool resume{false};
  bool fresh{false};
  bool calibrate{false};
  bool probe_only{false};
  bool profile_run{false};
  bool profile_run_output{false};
  int phase{0};
  std::uint32_t model_version{0};
};

[[nodiscard]] ResolvedConfig resolve_config(int argc, char** argv);
void print_resolved_config(const ResolvedConfig& config);
void write_resolved_config(const ResolvedConfig& config,
                           const std::filesystem::path& path);
void enforce_generation_guards(const ResolvedConfig& config);
[[nodiscard]] std::string generate_run_id(int phase);

}  // namespace chessmoe::selfplay
