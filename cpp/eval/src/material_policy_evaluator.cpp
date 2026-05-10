#include <chessmoe/eval/material_policy_evaluator.h>

#include <array>

namespace chessmoe::eval {

EvaluationResult MaterialPolicyEvaluator::evaluate(const EvaluationRequest& request) {
  const std::array<EvaluationRequest, 1> requests{request};
  return material_.evaluate_batch(requests).front();
}

}  // namespace chessmoe::eval
