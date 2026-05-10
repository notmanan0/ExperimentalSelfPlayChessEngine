#include <chessmoe/eval/evaluator.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>

namespace chessmoe::eval {

EvaluationRequest EvaluationRequest::from_position(
    const chess::Position& position) {
  return EvaluationRequest{
      position,
      chess::MoveGenerator::legal_moves(position),
      position.hash(),
      position.side_to_move(),
  };
}

SynchronousEvaluator::SynchronousEvaluator(IBatchEvaluator& batch_evaluator)
    : batch_evaluator_(batch_evaluator) {}

EvaluationResult SynchronousEvaluator::evaluate(const EvaluationRequest& request) {
  const std::array<EvaluationRequest, 1> requests{request};
  auto results = batch_evaluator_.evaluate_batch(requests);
  if (results.size() != 1) {
    throw std::runtime_error("batch evaluator returned wrong result count");
  }
  return std::move(results.front());
}

EvaluationResult normalize_policy_over_legal_moves(
    EvaluationResult result, std::span<const chess::Move> legal_moves) {
  std::map<std::string, PolicyEntry> by_move;
  for (const auto& entry : result.policy) {
    const auto key = entry.move.to_uci();
    auto& existing = by_move[key];
    existing.move = entry.move;
    existing.logit += entry.logit;
    existing.probability += std::max(0.0, entry.probability);
  }

  std::vector<PolicyEntry> legal_policy;
  legal_policy.reserve(legal_moves.size());

  bool has_positive_probabilities = false;
  for (const auto move : legal_moves) {
    auto entry = by_move[move.to_uci()];
    entry.move = move;
    if (entry.probability > 0.0) {
      has_positive_probabilities = true;
    }
    legal_policy.push_back(entry);
  }

  if (legal_policy.empty()) {
    result.policy.clear();
    return result;
  }

  if (!has_positive_probabilities) {
    const double max_logit = std::max_element(
                                 legal_policy.begin(), legal_policy.end(),
                                 [](const auto& a, const auto& b) {
                                   return a.logit < b.logit;
                                 })
                                 ->logit;
    double sum = 0.0;
    for (auto& entry : legal_policy) {
      entry.probability = std::exp(entry.logit - max_logit);
      sum += entry.probability;
    }
    for (auto& entry : legal_policy) {
      entry.probability /= sum;
    }
  } else {
    double sum = 0.0;
    for (const auto& entry : legal_policy) {
      sum += std::max(0.0, entry.probability);
    }
    if (sum <= 0.0) {
      const double uniform = 1.0 / legal_policy.size();
      for (auto& entry : legal_policy) {
        entry.probability = uniform;
      }
    } else {
      for (auto& entry : legal_policy) {
        entry.probability = std::max(0.0, entry.probability) / sum;
      }
    }
  }

  result.policy = std::move(legal_policy);
  result.value = std::clamp(result.value, -1.0, 1.0);
  result.wdl.win = std::max(0.0, result.wdl.win);
  result.wdl.draw = std::max(0.0, result.wdl.draw);
  result.wdl.loss = std::max(0.0, result.wdl.loss);

  const double wdl_sum = result.wdl.win + result.wdl.draw + result.wdl.loss;
  if (wdl_sum <= 0.0) {
    result.wdl = {0.0, 1.0, 0.0};
  } else {
    result.wdl.win /= wdl_sum;
    result.wdl.draw /= wdl_sum;
    result.wdl.loss /= wdl_sum;
  }

  return result;
}

}  // namespace chessmoe::eval
