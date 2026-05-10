#include <chessmoe/eval/random_evaluator.h>

#include <algorithm>
#include <cmath>

namespace chessmoe::eval {

RandomEvaluator::RandomEvaluator(std::uint32_t seed) : rng_(seed) {}

std::vector<EvaluationResult> RandomEvaluator::evaluate_batch(
    std::span<const EvaluationRequest> requests) {
  std::vector<EvaluationResult> results;
  results.reserve(requests.size());
  std::uniform_real_distribution<double> prior_dist(0.01, 1.0);
  std::uniform_real_distribution<double> value_dist(-1.0, 1.0);

  for (const auto& request : requests) {
    EvaluationResult result;
    result.policy.reserve(request.legal_moves.size());
    for (const auto move : request.legal_moves) {
      const double probability = prior_dist(rng_);
      result.policy.push_back({move, std::log(probability), probability});
    }

    result.value = std::clamp(value_dist(rng_), -1.0, 1.0);
    const double win = result.value > 0.0 ? std::abs(result.value) : 0.0;
    const double loss = result.value < 0.0 ? std::abs(result.value) : 0.0;
    result.wdl = {win, std::max(0.0, 1.0 - std::abs(result.value)), loss};
    result.moves_left = 40.0;
    results.push_back(
        normalize_policy_over_legal_moves(std::move(result), request.legal_moves));
  }

  return results;
}

}  // namespace chessmoe::eval
