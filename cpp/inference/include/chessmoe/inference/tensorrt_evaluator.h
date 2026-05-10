#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include <chessmoe/eval/evaluator.h>
#include <chessmoe/inference/tensor_layout.h>

namespace chessmoe::inference {

enum class PrecisionMode {
  Fp32,
  Fp16,
};

enum class InferenceUseCase {
  MatchPlay,
  SelfPlay,
};

struct TensorRTEvaluatorConfig {
  InferenceUseCase use_case{InferenceUseCase::MatchPlay};
  PrecisionMode precision{PrecisionMode::Fp32};
  bool apply_legal_mask{true};
};

struct NetworkInputBatch {
  std::size_t batch_size{0};
  std::vector<float> features;
};

struct RawNetworkOutput {
  std::size_t batch_size{0};
  std::vector<float> policy_logits;
  std::vector<float> wdl_logits;
  std::vector<float> moves_left;
};

class IInferenceBackend {
 public:
  virtual ~IInferenceBackend() = default;

  [[nodiscard]] virtual TensorLayout layout() const = 0;
  virtual RawNetworkOutput infer(const NetworkInputBatch& batch) = 0;
};

class TensorRTEvaluator final : public eval::IBatchEvaluator {
 public:
  TensorRTEvaluator(std::shared_ptr<IInferenceBackend> backend,
                    TensorRTEvaluatorConfig config);

  std::vector<eval::EvaluationResult> evaluate_batch(
      std::span<const eval::EvaluationRequest> requests) override;

 private:
  std::shared_ptr<IInferenceBackend> backend_;
  TensorRTEvaluatorConfig config_;
};

}  // namespace chessmoe::inference
