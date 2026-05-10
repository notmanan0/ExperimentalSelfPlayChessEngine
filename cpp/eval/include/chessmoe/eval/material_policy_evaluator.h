#pragma once

#include <chessmoe/eval/evaluator.h>
#include <chessmoe/eval/material_evaluator.h>

namespace chessmoe::eval {

class MaterialPolicyEvaluator final : public ISinglePositionEvaluator {
 public:
  EvaluationResult evaluate(const EvaluationRequest& request) override;

 private:
  MaterialEvaluator material_;
};

}  // namespace chessmoe::eval
