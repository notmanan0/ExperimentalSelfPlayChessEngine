#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

#include <chessmoe/inference/tensorrt_evaluator.h>

namespace chessmoe::inference {

struct TensorRTEngineConfig {
  std::filesystem::path engine_path;
  TensorLayout layout{TensorLayout::tiny_baseline()};
  std::size_t max_batch{1};
  PrecisionMode precision{PrecisionMode::Fp32};
  int device_id{0};
  int warmup_iterations{8};
  bool enable_cuda_graphs{false};
  bool enable_profiling{false};
};

class TensorRTEngine final : public IInferenceBackend {
 public:
  explicit TensorRTEngine(TensorRTEngineConfig config);
  ~TensorRTEngine() override;

  TensorRTEngine(const TensorRTEngine&) = delete;
  TensorRTEngine& operator=(const TensorRTEngine&) = delete;
  TensorRTEngine(TensorRTEngine&&) noexcept;
  TensorRTEngine& operator=(TensorRTEngine&&) noexcept;

  [[nodiscard]] TensorLayout layout() const override;
  RawNetworkOutput infer(const NetworkInputBatch& batch) override;

 private:
  struct Impl;
  TensorRTEngineConfig config_;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::string tensorrt_build_status();

}  // namespace chessmoe::inference
