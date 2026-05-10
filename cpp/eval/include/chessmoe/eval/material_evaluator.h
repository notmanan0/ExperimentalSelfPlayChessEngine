#pragma once

#include <span>
#include <vector>

#include <chessmoe/eval/evaluator.h>
#include <chessmoe/chess/position.h>

namespace chessmoe::eval {

class MaterialEvaluator final : public IBatchEvaluator {
 public:
  std::vector<EvaluationResult> evaluate_batch(
      std::span<const EvaluationRequest> requests) override;

  [[nodiscard]] int evaluate(const chess::Position& position,
                             chess::Color perspective) const;
};

}  // namespace chessmoe::eval
