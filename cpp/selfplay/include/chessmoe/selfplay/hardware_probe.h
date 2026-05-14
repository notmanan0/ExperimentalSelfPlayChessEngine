#pragma once

#include <cstdint>
#include <string>

namespace chessmoe::selfplay {

struct HardwareProbeResult {
  int cpu_logical_cores{0};
  std::string cpu_name;
  std::string gpu_name;
  std::uint64_t vram_bytes{0};
  std::uint64_t ram_bytes{0};
  bool cuda_available{false};
  bool tensorrt_compiled{false};
  std::string cuda_version;
  std::string driver_version;
  std::uint64_t disk_free_bytes{0};
  bool debug_build{false};
  std::string build_type;
  int recommended_batch{64};
  int recommended_concurrent_games{32};
  std::string recommended_profile;
};

[[nodiscard]] HardwareProbeResult probe_hardware();
void print_hardware_summary(const HardwareProbeResult& result);
[[nodiscard]] std::string recommend_profile(const HardwareProbeResult& result);

}  // namespace chessmoe::selfplay
