#include <chessmoe/selfplay/hardware_probe.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(CHESSMOE_ENABLE_TENSORRT)
#include <cuda_runtime_api.h>
#endif

#include <chessmoe/inference/tensorrt_engine.h>

namespace chessmoe::selfplay {
namespace {

int detect_cpu_cores() {
  const auto cores = std::thread::hardware_concurrency();
  return cores > 0 ? static_cast<int>(cores) : 1;
}

std::string detect_build_type() {
#ifdef NDEBUG
  return "Release";
#else
  return "Debug";
#endif
}

std::uint64_t detect_disk_free(const std::filesystem::path& path) {
  std::error_code ec;
  const auto space = std::filesystem::space(path, ec);
  if (ec) {
    return 0;
  }
  return space.available;
}

std::string format_bytes(std::uint64_t bytes) {
  if (bytes >= 1024ULL * 1024 * 1024 * 1024) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1)
        << static_cast<double>(bytes) / (1024.0 * 1024 * 1024 * 1024)
        << " TB";
    return out.str();
  }
  if (bytes >= 1024ULL * 1024 * 1024) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1)
        << static_cast<double>(bytes) / (1024.0 * 1024 * 1024) << " GB";
    return out.str();
  }
  if (bytes >= 1024ULL * 1024) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(0)
        << static_cast<double>(bytes) / (1024.0 * 1024) << " MB";
    return out.str();
  }
  return std::to_string(bytes) + " bytes";
}

int estimate_batch_from_vram(std::uint64_t vram_bytes) {
  if (vram_bytes == 0) {
    return 1;
  }
  const double gb =
      static_cast<double>(vram_bytes) / (1024.0 * 1024 * 1024);
  if (gb >= 40.0) {
    return 256;
  }
  if (gb >= 12.0) {
    return 128;
  }
  if (gb >= 6.0) {
    return 64;
  }
  if (gb >= 4.0) {
    return 32;
  }
  return 16;
}

int estimate_concurrent_from_cores(int cores) {
  if (cores >= 32) {
    return 128;
  }
  if (cores >= 16) {
    return 96;
  }
  if (cores >= 8) {
    return 64;
  }
  return std::max(4, cores * 2);
}

}  // namespace

HardwareProbeResult probe_hardware() {
  HardwareProbeResult result;

  result.cpu_logical_cores = detect_cpu_cores();
  result.build_type = detect_build_type();
  result.debug_build = (result.build_type == "Debug");
  result.tensorrt_compiled =
      (inference::tensorrt_build_status() == "compiled-with-tensorrt");

  result.disk_free_bytes = detect_disk_free(std::filesystem::current_path());

#if defined(CHESSMOE_ENABLE_TENSORRT)
  result.cuda_available = true;

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
      result.gpu_name = prop.name;
      result.vram_bytes = prop.totalGlobalMem;
    }

    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
      int major = runtime_version / 1000;
      int minor = (runtime_version % 100) / 10;
      result.cuda_version =
          std::to_string(major) + "." + std::to_string(minor);
    }

    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
      int major = driver_version / 1000;
      int minor = (driver_version % 100) / 10;
      result.driver_version =
          std::to_string(major) + "." + std::to_string(minor);
    }
  }
#else
  result.cuda_available = false;
#endif

  result.recommended_batch = estimate_batch_from_vram(result.vram_bytes);
  result.recommended_concurrent_games =
      estimate_concurrent_from_cores(result.cpu_logical_cores);
  result.recommended_profile = recommend_profile(result);

  return result;
}

std::string recommend_profile(const HardwareProbeResult& result) {
  if (!result.cuda_available || result.vram_bytes == 0) {
    if (result.debug_build) {
      return "cpu_bootstrap_debug";
    }
    return "cpu_bootstrap_fast";
  }

  const double vram_gb =
      static_cast<double>(result.vram_bytes) / (1024.0 * 1024 * 1024);

  if (vram_gb >= 40.0) {
    return "gpu_datacenter";
  }
  if (vram_gb >= 12.0) {
    return "gpu_highend";
  }
  if (vram_gb >= 6.0) {
    return "gpu_midrange";
  }
  return "gpu_low_vram";
}

void print_hardware_summary(const HardwareProbeResult& result) {
  std::cout << "=== Hardware Probe ===" << '\n';

  if (!result.gpu_name.empty()) {
    std::cout << "Detected GPU: " << result.gpu_name;
    if (result.vram_bytes > 0) {
      std::cout << ", " << format_bytes(result.vram_bytes) << " VRAM";
    }
    std::cout << '\n';
  } else if (result.cuda_available) {
    std::cout << "CUDA available but no GPU detected" << '\n';
  } else {
    std::cout << "No CUDA GPU detected" << '\n';
  }

  std::cout << "Detected CPU threads: " << result.cpu_logical_cores << '\n';

  if (!result.cpu_name.empty()) {
    std::cout << "CPU: " << result.cpu_name << '\n';
  }

  std::cout << "Build type: " << result.build_type << '\n';

  if (!result.cuda_version.empty()) {
    std::cout << "CUDA runtime: " << result.cuda_version << '\n';
  }
  if (!result.driver_version.empty()) {
    std::cout << "Driver: " << result.driver_version << '\n';
  }

  std::cout << "TensorRT compiled: "
            << (result.tensorrt_compiled ? "yes" : "no") << '\n';

  std::cout << "Disk free: " << format_bytes(result.disk_free_bytes) << '\n';

  std::cout << "Recommended profile: " << result.recommended_profile << '\n';
  std::cout << "Suggested fixed_batch: " << result.recommended_batch << '\n';
  std::cout << "Suggested concurrent_games: "
            << result.recommended_concurrent_games << '\n';

  if (result.debug_build) {
    std::cout << "Warning: Debug build detected; "
              << "serious self-play will be much slower." << '\n';
  }
  if (!result.tensorrt_compiled) {
    std::cout << "Warning: TensorRT not compiled; "
              << "neural self-play requires CHESSMOE_ENABLE_TENSORRT=ON."
              << '\n';
  }
}

}  // namespace chessmoe::selfplay
