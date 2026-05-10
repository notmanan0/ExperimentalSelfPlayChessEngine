#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include <chessmoe/eval/evaluator.h>

namespace chessmoe::eval {

class RandomEvaluator final : public IBatchEvaluator {
 public:
  explicit RandomEvaluator(std::uint32_t seed = 1);

  std::vector<EvaluationResult> evaluate_batch(
      std::span<const EvaluationRequest> requests) override;

 private:
  std::mt19937 rng_;
};

}  // namespace chessmoe::eval
